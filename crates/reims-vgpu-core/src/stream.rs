//! Segments, encoders, and the order a record has inside an EXEC.
//!
//! # The boundary is the first payload type, because everything else sits in it
//!
//! [`crate::operation`] names eleven classes of resolved operation. Ten of them
//! are records; the eleventh — `EncoderBoundary` — is the thing that says which
//! encoder those records belong to and in what order they run. It has to be
//! first: a `Render` payload with no encoder is a draw with no pass, and a
//! vocabulary that resolves draws before it resolves the encoder they sit in
//! would have to discover the encoder later, from position, which is exactly
//! the "resolve twice" shape the replacement exists to remove.
//!
//! # A segment type is parsed once, and not here
//!
//! [`reims_vgpu_protocol::segment`] owns that parse: it is a wire tag becoming
//! a meaning, which is the protocol layer's job, and it returns `None` for
//! every byte with no established contract. This module takes the resulting
//! [`SegmentKind`] as given and never sees the byte again — so there is one
//! place a new segment family is admitted, and it is not the state machine.
//!
//! # Ordering, and the parallelism this deliberately does not claim
//!
//! Records inside one encoder are ordered; that is contract. Whether two
//! *encoders* may run concurrently is not established for this wire, so
//! [`StreamOrder`] gives every record a total order within its EXEC and offers
//! no way to express "these two segments are independent". A later seam that
//! proves parallel encoding adds the independence claim there; until then the
//! model reduces parallelism rather than inventing it, which is the direction
//! the plan requires when a concurrency proof is missing.
//!
//! # What an open encoder means for admission
//!
//! [`StreamCursor`] is the state machine: it refuses a record with no open
//! encoder, a record whose rail disagrees with the segment it is inside, a
//! begin while another encoder is open, and an end with nothing to end. Each of
//! those is a wire shape that cannot be executed, so each is a typed refusal
//! and none is silently tolerated — the alternative is a draw attributed to
//! whichever pass happened to be open, which is a wrong frame rather than a
//! missing one.

use reims_vgpu_protocol::closure::Rail;
pub use reims_vgpu_protocol::segment::{segment_role, SegmentKind, SegmentRole};

/// The protection-options value a guest attached to the segment that follows.
///
/// Carried, not honoured. This device provides no protection domain, so an
/// executor cannot make the guarantee the value asks for; carrying it keeps
/// "we do not act on it" separable from "we cannot read it", and
/// [`SegmentBegin::demands_protection`] is what a report asks so the difference
/// stays visible on the failure channel instead of being lost at decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProtectionOptions(pub u64);

impl ProtectionOptions {
    /// The envelope is emitted only for a non-zero value, so a zero here came
    /// from somewhere other than the guest's argument.
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
}

/// The resolved form of an encoder opening.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentBegin {
    pub kind: SegmentKind,
    /// The `BOOL` first argument of the begin call, verbatim.
    pub flag: bool,
    /// The value from a preceding protection envelope, if one armed this
    /// segment.
    pub protection: Option<ProtectionOptions>,
}

impl SegmentBegin {
    /// Whether the guest asked for a protection domain this device does not
    /// provide.
    #[must_use]
    pub fn demands_protection(&self) -> bool {
        self.protection.is_some_and(|p| !p.is_none())
    }
}

/// The `EncoderBoundary` class's payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EncoderBoundary {
    /// An encoder opened.
    Begin(SegmentBegin),
    /// An encoder ended, having recorded this many records.
    ///
    /// The count is the model's, not the wire's: the wire fills in a byte
    /// length at `-endEncoding` and says nothing about how many records it
    /// covered. A count the model derived is checkable against a re-walk;
    /// a length copied from the header is not.
    End { records: u32 },
}

/// Which segment, and which record inside it.
///
/// [`Ord`] is the execution order, and it is total: the model has no way to
/// say two positions are unordered, because nothing has proved that they are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamPosition {
    pub segment: u32,
    pub record: u32,
}

/// A total order over the records of one EXEC.
///
/// A separate type rather than a bare comparison so that the claim "records in
/// this EXEC are totally ordered" has one owner, and so a later seam that
/// proves encoder independence changes this type instead of every comparison
/// site.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamOrder;

impl StreamOrder {
    /// Whether `earlier` must complete its ordering obligations before `later`.
    ///
    /// Always true for distinct ascending positions. That is the conservative
    /// reading and it is deliberate.
    #[must_use]
    pub fn precedes(self, earlier: StreamPosition, later: StreamPosition) -> bool {
        earlier < later
    }

    /// Whether the two may be reordered with respect to each other.
    #[must_use]
    pub fn independent(self, _a: StreamPosition, _b: StreamPosition) -> bool {
        false
    }
}

/// Why a stream shape cannot be admitted.
///
/// Every variant names a wire arrangement that has no executable meaning, and
/// each one is reported rather than repaired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamRefusal {
    /// A segment type with no established contract.
    UnknownSegmentType(u8),
    /// A begin arrived while an encoder was still open.
    EncoderStillOpen(SegmentKind),
    /// A record arrived outside any encoder.
    RecordOutsideEncoder,
    /// A record decoded on one rail was found inside another rail's segment.
    RailMismatch { segment: SegmentKind, record: Rail },
    /// An end arrived with no encoder open.
    EndWithoutBegin,
    /// A protection envelope armed nothing, because another envelope or the end
    /// of the stream followed it.
    ProtectionEnvelopeUnclaimed,
    /// The stream ended with an encoder still open.
    EncoderNeverEnded(SegmentKind),
}

impl StreamRefusal {
    /// The stable reason string for the failure channel.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::UnknownSegmentType(_) => "stream_segment_type_unknown",
            Self::EncoderStillOpen(_) => "stream_encoder_begin_while_open",
            Self::RecordOutsideEncoder => "stream_record_outside_encoder",
            Self::RailMismatch { .. } => "stream_record_rail_mismatch",
            Self::EndWithoutBegin => "stream_encoder_end_without_begin",
            Self::ProtectionEnvelopeUnclaimed => "stream_protection_envelope_unclaimed",
            Self::EncoderNeverEnded(_) => "stream_encoder_never_ended",
        }
    }
}

/// Where an encoder is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Open { kind: SegmentKind, records: u32 },
    Closed,
}

/// The encoder state machine for one EXEC's command stream.
///
/// Fed in wire order; every call either returns the resolved boundary or a
/// refusal. It holds no bytes and no host state, so a test can drive it from a
/// list of events and get the same answers the ingress path would.
#[derive(Debug)]
pub struct StreamCursor {
    phase: Phase,
    pending_protection: Option<ProtectionOptions>,
    segment: u32,
    record: u32,
}

impl Default for StreamCursor {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamCursor {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            phase: Phase::Closed,
            pending_protection: None,
            segment: 0,
            record: 0,
        }
    }

    /// A protection envelope was seen. It arms the next encoder begin.
    ///
    /// Two in a row is a refusal: the second would silently replace the first,
    /// and a dropped protection request is exactly the class of loss that has
    /// to be typed.
    pub fn protection_envelope(&mut self, options: ProtectionOptions) -> Result<(), StreamRefusal> {
        if self.pending_protection.is_some() {
            return Err(StreamRefusal::ProtectionEnvelopeUnclaimed);
        }
        if let Phase::Open { kind, .. } = self.phase {
            return Err(StreamRefusal::EncoderStillOpen(kind));
        }
        self.pending_protection = Some(options);
        Ok(())
    }

    /// A segment header opened an encoder.
    pub fn begin(
        &mut self,
        wire_type: u8,
        flag: bool,
    ) -> Result<(StreamPosition, SegmentBegin), StreamRefusal> {
        let kind = match segment_role(wire_type) {
            Some(SegmentRole::Encoder(kind)) => kind,
            // The envelope has its own entry point; arriving here means the
            // caller framed it as an encoder, which it is not.
            Some(SegmentRole::ProtectionEnvelope) | None => {
                return Err(StreamRefusal::UnknownSegmentType(wire_type))
            }
        };
        if let Phase::Open { kind, .. } = self.phase {
            return Err(StreamRefusal::EncoderStillOpen(kind));
        }
        let at = StreamPosition {
            segment: self.segment,
            record: 0,
        };
        self.phase = Phase::Open { kind, records: 0 };
        self.record = 0;
        Ok((
            at,
            SegmentBegin {
                kind,
                flag,
                protection: self.pending_protection.take(),
            },
        ))
    }

    /// A record was decoded on `rail` inside the open encoder.
    ///
    /// Returns the record's position, which is what an [`crate::access`] intent
    /// and a hazard edge are keyed by.
    pub fn record(&mut self, rail: Rail) -> Result<StreamPosition, StreamRefusal> {
        let Phase::Open { kind, records } = &mut self.phase else {
            return Err(StreamRefusal::RecordOutsideEncoder);
        };
        if kind.rail() != rail {
            return Err(StreamRefusal::RailMismatch {
                segment: *kind,
                record: rail,
            });
        }
        *records += 1;
        let at = StreamPosition {
            segment: self.segment,
            record: self.record,
        };
        self.record += 1;
        Ok(at)
    }

    /// The open encoder ended.
    pub fn end(&mut self) -> Result<EncoderBoundary, StreamRefusal> {
        let Phase::Open { records, .. } = self.phase else {
            return Err(StreamRefusal::EndWithoutBegin);
        };
        self.phase = Phase::Closed;
        self.segment += 1;
        self.record = 0;
        Ok(EncoderBoundary::End { records })
    }

    /// The stream is over. Anything still open or still armed is a refusal.
    pub fn finish(self) -> Result<u32, StreamRefusal> {
        if let Phase::Open { kind, .. } = self.phase {
            return Err(StreamRefusal::EncoderNeverEnded(kind));
        }
        if self.pending_protection.is_some() {
            return Err(StreamRefusal::ProtectionEnvelopeUnclaimed);
        }
        Ok(self.segment)
    }

    /// The kind of the open encoder, if one is open.
    #[must_use]
    pub const fn open_encoder(&self) -> Option<SegmentKind> {
        match self.phase {
            Phase::Open { kind, .. } => Some(kind),
            Phase::Closed => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each kind admits its own rail's records and no other's.
    #[test]
    fn a_segment_admits_exactly_its_own_rail() {
        for &kind in SegmentKind::ALL {
            for &other in SegmentKind::ALL {
                let mut c = StreamCursor::new();
                c.begin(kind.wire_type(), false).expect("opens");
                let got = c.record(other.rail());
                if kind == other {
                    assert!(got.is_ok());
                } else {
                    assert_eq!(
                        got,
                        Err(StreamRefusal::RailMismatch {
                            segment: kind,
                            record: other.rail(),
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn records_are_positioned_in_wire_order_across_segments() {
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Render.wire_type(), true)
            .expect("open");
        let a = c.record(Rail::Render).expect("rec");
        let b = c.record(Rail::Render).expect("rec");
        assert_eq!(c.end(), Ok(EncoderBoundary::End { records: 2 }));
        c.begin(SegmentKind::Blit.wire_type(), false).expect("open");
        let d = c.record(Rail::Blit).expect("rec");
        assert_eq!(c.end(), Ok(EncoderBoundary::End { records: 1 }));
        assert_eq!(c.finish(), Ok(2));

        assert!(a < b && b < d);
        let order = StreamOrder;
        assert!(order.precedes(a, d));
        assert!(!order.precedes(d, a));
        assert!(!order.independent(a, d));
    }

    /// Every malformed arrangement is a named refusal rather than a tolerated
    /// one, because each of them would otherwise attribute work to the wrong
    /// encoder or drop it.
    #[test]
    fn malformed_arrangements_are_named_refusals() {
        let mut c = StreamCursor::new();
        assert_eq!(
            c.record(Rail::Render),
            Err(StreamRefusal::RecordOutsideEncoder)
        );
        assert_eq!(c.end(), Err(StreamRefusal::EndWithoutBegin));
        assert_eq!(c.begin(9, false), Err(StreamRefusal::UnknownSegmentType(9)));

        c.begin(SegmentKind::Render.wire_type(), false)
            .expect("open");
        assert_eq!(
            c.begin(SegmentKind::Compute.wire_type(), false),
            Err(StreamRefusal::EncoderStillOpen(SegmentKind::Render))
        );
        assert_eq!(
            c.finish(),
            Err(StreamRefusal::EncoderNeverEnded(SegmentKind::Render))
        );
    }

    /// The envelope arms the next begin, exactly once.
    #[test]
    fn a_protection_envelope_arms_the_segment_after_it() {
        let mut c = StreamCursor::new();
        c.protection_envelope(ProtectionOptions(0x44))
            .expect("armed");
        let (_, begin) = c.begin(SegmentKind::Blit.wire_type(), false).expect("open");
        assert_eq!(begin.protection, Some(ProtectionOptions(0x44)));
        assert!(begin.demands_protection());
        c.end().expect("end");

        // Not the one after that.
        let (_, next) = c.begin(SegmentKind::Blit.wire_type(), false).expect("open");
        assert_eq!(next.protection, None);
        assert!(!next.demands_protection());
    }

    /// A second envelope would silently replace the first, and a protection
    /// request the device drops without a word is the loss this refuses.
    #[test]
    fn an_envelope_that_arms_nothing_is_refused() {
        let mut c = StreamCursor::new();
        c.protection_envelope(ProtectionOptions(1)).expect("armed");
        assert_eq!(
            c.protection_envelope(ProtectionOptions(2)),
            Err(StreamRefusal::ProtectionEnvelopeUnclaimed)
        );
        assert_eq!(c.finish(), Err(StreamRefusal::ProtectionEnvelopeUnclaimed));
    }

    /// An envelope inside an open encoder is not the burst the contract
    /// describes; it precedes a segment header, it does not interrupt one.
    #[test]
    fn an_envelope_may_not_arrive_inside_an_encoder() {
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Compute.wire_type(), false)
            .expect("open");
        assert_eq!(
            c.protection_envelope(ProtectionOptions(1)),
            Err(StreamRefusal::EncoderStillOpen(SegmentKind::Compute))
        );
    }

    /// Every refusal reason is distinct, so a log line identifies which shape
    /// was seen.
    #[test]
    fn refusal_reasons_are_distinct() {
        let all = [
            StreamRefusal::UnknownSegmentType(0),
            StreamRefusal::EncoderStillOpen(SegmentKind::Render),
            StreamRefusal::RecordOutsideEncoder,
            StreamRefusal::RailMismatch {
                segment: SegmentKind::Render,
                record: Rail::Blit,
            },
            StreamRefusal::EndWithoutBegin,
            StreamRefusal::ProtectionEnvelopeUnclaimed,
            StreamRefusal::EncoderNeverEnded(SegmentKind::Render),
        ];
        let mut seen: Vec<&str> = all.iter().map(|r| r.reason()).collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), before);
    }
}
