//! From a guest's command-stream bytes to one transaction.
//!
//! # The link that was missing
//!
//! [`crate::exec::ExecBuilder`] takes `begin_segment`, `record`, `end_segment`
//! and `finish`, in wire order, and produces the only [`ExecTransaction`] this
//! crate can make. Every one of those calls had to come from somewhere, and
//! until this module the only somewhere was a test calling them by hand. So the
//! model could resolve a record and could place a record, and had no way to be
//! handed a stream.
//!
//! [`exec`] is that way. It is the whole path — bytes, segments, records,
//! operations, accesses, transaction — and it is one function because every
//! seam inside it is a place where two halves could disagree about where a
//! record was, which is exactly the class of defect the replacement exists to
//! remove.
//!
//! # It owns none of the four parses it drives
//!
//! Segments come from [`reims_vgpu_protocol::segment::SegmentStream`]. Records
//! come from [`reims_vgpu_wire::op::OpStream`], through the protocol crate's
//! re-export. Meaning comes from [`crate::resolve::operation`]. Placement,
//! ordering and access derivation come from the builder. This module contains
//! no byte arithmetic at all; it is the composition, and the reason it is worth
//! writing down separately is that the composition is where a rail can be taken
//! from the wrong place.
//!
//! The rail is the segment's. `resolve::operation` is handed a rail rather than
//! deriving one, and the only defensible source for it is the encoder class the
//! guest wrote the record into — [`SegmentKind::rail`], from the type byte in
//! the header immediately above the record. Taking it from anywhere else, such
//! as a previous segment or a per-packet default, reads one encoder's commands
//! as another's.
//!
//! # A record that does not resolve refuses the transaction
//!
//! Not the record — the transaction. Dropping one record and executing the rest
//! is a wrong frame presented as a right one: a draw without its pipeline, a
//! blit without its barrier. The closure ledger is what makes this a schedule
//! rather than a wall, and it already says so —
//! [`reims_vgpu_protocol::closure::Closure::blocks_cutover`] holds for every
//! unresolved row, so a stream refusing here while rows remain open is the
//! ledger's prediction rather than a surprise.
//!
//! # What it refuses that the contract permits
//!
//! A segment header carries two encoder-lifetime bits and the contract behind
//! them is established, not guessed: the `beginSegment:` `BOOL` lands at `+5`
//! of the header it opens, and the serializer then reaches *back* into the
//! preceding header to mark `+6`. The two are one edge recorded from both ends
//! — "this segment continues the encoder above" and "that encoder continues
//! below" — which is why one non-zero byte cannot be read for the direction.
//! The oracle drives both ends in a single case for exactly that reason.
//!
//! So a guest may ask for one encoder to span several segments, and
//! [`crate::stream::StreamCursor`] cannot represent it: an encoder opens and
//! closes inside one segment. [`WalkRefusal::EncoderSpansSegments`] is that
//! limit of the model, stated rather than ignored. Silently opening a fresh
//! encoder instead would attribute the second segment's records to a pass the
//! guest never opened, which is a wrong frame rather than a missing one — and
//! the legacy walker's own census of a driven boot found the bits clear on all
//! 94 860 segments, so refusing costs this guest nothing while the model
//! catches up.

use crate::exec::{ExecBuilder, ExecTransaction};
use crate::resolve::{self, RefResolver, ResolveRefusal};
use crate::stream::{ProtectionOptions, StreamRefusal};
use reims_vgpu_protocol::decode::OpStream;
use reims_vgpu_protocol::segment::{FramedSegment, FramingRefusal, SegmentBody, SegmentStream};

/// Where in a stream something was refused.
///
/// Both coordinates, because neither alone finds the record: the segment index
/// is what a report names an encoder by, and the byte offset is what finds the
/// bytes in a capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamSite {
    /// Which segment, counting from zero.
    pub segment: u32,
    /// Byte offset within the whole stream.
    pub offset: u32,
}

/// Why a command stream did not become a transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkRefusal {
    /// The stream did not divide into segments.
    Framing(FramingRefusal),
    /// A segment's window did not divide into records.
    ///
    /// The wire error is not carried: it names a byte length against a buffer
    /// length, and the site names which buffer. Both halves of a report exist,
    /// and neither is a `WireError` this crate would then have to give a
    /// reason string to on wire's behalf.
    RecordFraming { at: StreamSite },
    /// A record did not become an operation.
    ///
    /// Includes the ledger's own answers — an opcode with no row, an open row,
    /// a row settled as a refusal. Those are contract answers rather than
    /// defects, and the whole transaction still refuses; see the module
    /// documentation.
    Resolve {
        at: StreamSite,
        refusal: ResolveRefusal,
    },
    /// The builder refused the operation's placement, ordering or access.
    Place {
        at: StreamSite,
        refusal: StreamRefusal,
    },
    /// A segment asked for its encoder to span a segment boundary.
    ///
    /// A limit of the model, not of the contract. See the module documentation.
    EncoderSpansSegments { at: StreamSite },
    /// The stream ended with an encoder or a protection envelope unfinished.
    Unfinished { refusal: StreamRefusal },
}

impl WalkRefusal {
    /// The stable reason string for the failure channel.
    ///
    /// Each arm that wraps an owner's refusal reports the owner's own reason
    /// rather than a second name for it, so a log line says which check
    /// refused and not merely that the walk did.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Framing(inner) => inner.reason(),
            Self::RecordFraming { .. } => "walk_record_framing_refused",
            Self::Resolve { refusal, .. } => refusal.reason(),
            Self::Place { refusal, .. } | Self::Unfinished { refusal } => refusal.reason(),
            Self::EncoderSpansSegments { .. } => "walk_encoder_spans_segments",
        }
    }

    /// Where it happened, for the arms that have a place.
    ///
    /// [`Self::Framing`] carries the framing layer's own offset and
    /// [`Self::Unfinished`] is about the stream's end rather than a point
    /// inside it.
    #[must_use]
    pub const fn site(self) -> Option<StreamSite> {
        match self {
            Self::RecordFraming { at }
            | Self::Resolve { at, .. }
            | Self::Place { at, .. }
            | Self::EncoderSpansSegments { at } => Some(at),
            Self::Framing(_) | Self::Unfinished { .. } => None,
        }
    }
}

impl From<FramingRefusal> for WalkRefusal {
    fn from(inner: FramingRefusal) -> Self {
        Self::Framing(inner)
    }
}

/// Walk one EXEC's command stream into the transaction it describes.
///
/// The builder is consumed: what comes out is either the finished transaction
/// or a refusal, and never a half-written builder a caller could submit
/// anyway.
///
/// # Errors
///
/// Any [`WalkRefusal`]. The transaction is all-or-nothing — see the module
/// documentation for why a single unresolvable record refuses the whole of it.
pub fn exec(
    bytes: &[u8],
    resolver: &impl RefResolver,
    source: &mut impl crate::access::AccessSource,
    mut builder: ExecBuilder,
) -> Result<ExecTransaction, WalkRefusal> {
    for framed in SegmentStream::new(bytes)? {
        let framed = framed?;
        segment(&framed, resolver, source, &mut builder)?;
    }
    builder
        .finish()
        .map_err(|refusal| WalkRefusal::Unfinished { refusal })
}

/// One segment's worth of the walk.
fn segment(
    framed: &FramedSegment<'_>,
    resolver: &impl RefResolver,
    source: &mut impl crate::access::AccessSource,
    builder: &mut ExecBuilder,
) -> Result<(), WalkRefusal> {
    let at = StreamSite {
        segment: framed.index,
        offset: framed.offset,
    };
    let (kind, commands) = match framed.body {
        SegmentBody::ProtectionEnvelope { options } => {
            // The envelope arms the segment after it, which is the cursor's
            // rule and not restated here.
            return builder
                .protection_envelope(ProtectionOptions(options))
                .map_err(|refusal| WalkRefusal::Place { at, refusal });
        }
        SegmentBody::Encoder { kind, commands } => (kind, commands),
    };
    if framed.continues_previous || framed.continues_into_next {
        return Err(WalkRefusal::EncoderSpansSegments { at });
    }
    builder
        .begin_encoder(kind, framed.continues_previous)
        .map_err(|refusal| WalkRefusal::Place { at, refusal })?;
    let mut records = OpStream::new(commands);
    loop {
        // Taken before the step, so it is the offset the record starts at
        // whether or not the record turns out to be readable. A refused one is
        // not a value and cannot be asked where it began.
        let started = records.consumed();
        let Some(record) = records.next() else { break };
        // The window's offsets are inside the segment; a report has to name the
        // stream. The cast is exact: the framing layer established that the
        // stream's length fits a `u32`, and the window is inside it.
        let at = StreamSite {
            segment: framed.index,
            offset: framed.commands_offset + started as u32,
        };
        let Ok(view) = record else {
            return Err(WalkRefusal::RecordFraming { at });
        };
        let resolved = resolve::operation(kind.rail(), &view, resolver, builder.arenas_mut())
            .map_err(|refusal| WalkRefusal::Resolve { at, refusal })?;
        builder
            .record(resolved, source)
            .map_err(|refusal| WalkRefusal::Place { at, refusal })?;
    }
    builder
        .end_segment()
        .map(|_| ())
        .map_err(|refusal| WalkRefusal::Place { at, refusal })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::StubRegistry;
    use crate::exec::ExecArenas;
    use crate::identity::{
        ChannelId, ChannelSequence, IngressOrdinal, ObjectListRef, ResourceId, SessionGeneration,
        SlotGeneration,
    };
    use reims_vgpu_protocol::segment::{SegmentKind, SEGMENT_TYPE_PROTECTION_OPTIONS};
    use reims_vgpu_wire::ops::blit::OPCODE_GENERATE_MIPMAPS;
    use reims_vgpu_wire::ops::render::OPCODE_SET_LINE_WIDTH;
    use reims_vgpu_wire::ops::segment::SEGMENT_HEADER_LEN;

    const DOMAIN: ChannelId = ChannelId(3);

    /// A resolver that answers every ref, because resolution is not what these
    /// tests are about.
    struct Everything;

    impl RefResolver for Everything {
        fn resource(&self, object_ref: u32) -> Option<ResourceId> {
            Some(ResourceId {
                slot: ObjectListRef(object_ref),
                generation: SlotGeneration(1),
            })
        }
    }

    fn builder() -> ExecBuilder {
        ExecBuilder::new(
            SessionGeneration::FIRST,
            DOMAIN,
            ChannelSequence(1),
            IngressOrdinal(1),
        )
    }

    /// One record, framed the way the serializer frames one.
    fn record(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let length = (OP_HEADER_LEN + payload.len()) as u32;
        out.extend_from_slice(&opcode.to_le_bytes());
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn line_width(width: f32) -> Vec<u8> {
        record(OPCODE_SET_LINE_WIDTH, &width.to_le_bytes())
    }

    fn generate_mipmaps(texture: u32) -> Vec<u8> {
        record(OPCODE_GENERATE_MIPMAPS, &texture.to_le_bytes())
    }

    /// One segment, with the length `-endEncoding` fills in.
    fn segment_bytes(wire_type: u8, records: &[Vec<u8>]) -> Vec<u8> {
        let body: usize = records.iter().map(Vec::len).sum();
        let mut out = Vec::new();
        out.extend_from_slice(&((SEGMENT_HEADER_LEN + body) as u32).to_le_bytes());
        out.push(wire_type);
        out.push(0);
        out.push(0);
        // The byte the serializer never writes.
        out.push(0xaa);
        for r in records {
            out.extend_from_slice(r);
        }
        out
    }

    const OP_HEADER_LEN: usize = reims_vgpu_protocol::decode::OP_HEADER_LEN;

    /// The whole path, from bytes to a transaction whose accesses came from the
    /// records that named them.
    #[test]
    fn a_stream_becomes_the_transaction_its_segments_describe() {
        let mut bytes = segment_bytes(
            SegmentKind::Render.wire_type(),
            &[line_width(2.5), line_width(1.25)],
        );
        bytes.extend_from_slice(&segment_bytes(
            SegmentKind::Blit.wire_type(),
            &[generate_mipmaps(4242)],
        ));

        let tx = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("a well-framed stream");

        assert_eq!(tx.streams.len(), 2);
        assert_eq!(tx.record_count(), 3);
        let positions: Vec<_> = tx.records().map(|r| r.at).collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(
            positions.iter().map(|p| p.segment).collect::<Vec<_>>(),
            [0, 0, 1]
        );
        // The mipmap generation is the only record here that names a resource,
        // and the access on the transaction is the one it named.
        assert_eq!(tx.accesses.len(), 1);
        assert_eq!(
            tx.accesses[0].key,
            crate::access::AccessKey::Whole(crate::access::ResourceKey {
                backing: crate::access::BackingId(4242),
                heap: None,
            })
        );
        assert_eq!(tx.accesses[0].domain, DOMAIN);
    }

    /// The rail a record is read on is the encoder the guest wrote it into.
    ///
    /// The same four opcode bytes are a blit record and nothing at all on the
    /// render rail. A walk that carried a rail from anywhere but the segment
    /// header immediately above the record — a previous segment, a per-packet
    /// default — would read one encoder's commands as another's, and the only
    /// evidence it had done so would be the frame.
    #[test]
    fn the_rail_a_record_is_read_on_is_the_encoder_it_was_written_into() {
        let mipmaps = generate_mipmaps(4242);
        let inside_blit = segment_bytes(
            SegmentKind::Blit.wire_type(),
            std::slice::from_ref(&mipmaps),
        );
        let inside_render = segment_bytes(SegmentKind::Render.wire_type(), &[mipmaps]);

        assert_eq!(
            exec(
                &inside_blit,
                &Everything,
                &mut StubRegistry(DOMAIN),
                builder()
            )
            .expect("a blit record in a blit segment")
            .record_count(),
            1
        );

        let refused = exec(
            &inside_render,
            &Everything,
            &mut StubRegistry(DOMAIN),
            builder(),
        )
        .expect_err("a blit record is not a render record");
        assert!(matches!(refused, WalkRefusal::Resolve { .. }));
        assert_eq!(
            refused.site(),
            Some(StreamSite {
                segment: 0,
                offset: SEGMENT_HEADER_LEN as u32,
            })
        );
    }

    /// The envelope's value reaches the segment it arms, so a report can say
    /// the guest asked for a protection domain this device does not provide.
    #[test]
    fn a_protection_envelope_reaches_the_segment_it_arms() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&segment_bytes(SEGMENT_TYPE_PROTECTION_OPTIONS, &[]));
        // The envelope's window is its payload, not a record.
        let at = bytes.len() - SEGMENT_HEADER_LEN;
        bytes[at..at + 4].copy_from_slice(&((SEGMENT_HEADER_LEN + 8) as u32).to_le_bytes());
        bytes.extend_from_slice(&0x44u64.to_le_bytes());
        bytes.extend_from_slice(&segment_bytes(
            SegmentKind::Blit.wire_type(),
            &[generate_mipmaps(7)],
        ));

        let tx = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("an envelope and the segment it arms");
        assert_eq!(tx.streams.len(), 1);
        assert_eq!(
            tx.streams[0].begin.protection,
            Some(ProtectionOptions(0x44))
        );
        assert!(tx.streams[0].begin.demands_protection());
    }

    /// An envelope with nothing after it armed nothing, and a dropped
    /// protection request is loss the stream's end has to report.
    #[test]
    fn an_envelope_at_the_end_of_a_stream_refuses_the_transaction() {
        let mut bytes = segment_bytes(SEGMENT_TYPE_PROTECTION_OPTIONS, &[]);
        bytes[..4].copy_from_slice(&((SEGMENT_HEADER_LEN + 8) as u32).to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());

        assert_eq!(
            exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder()),
            Err(WalkRefusal::Unfinished {
                refusal: StreamRefusal::ProtectionEnvelopeUnclaimed,
            })
        );
    }

    /// One record the model cannot represent refuses the whole transaction.
    ///
    /// Executing the rest would be a draw without its pipeline or a blit
    /// without its barrier, presented as a finished frame.
    #[test]
    fn one_unrepresentable_record_refuses_the_whole_transaction() {
        let good = line_width(2.5);
        let bad = record(0xffff_ff00, &[]);
        let bytes = segment_bytes(
            SegmentKind::Render.wire_type(),
            &[good.clone(), bad, line_width(1.0)],
        );

        let refused = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect_err("an opcode the render rail carries no record for");
        assert_eq!(
            refused.site(),
            Some(StreamSite {
                segment: 0,
                offset: (SEGMENT_HEADER_LEN + good.len()) as u32,
            })
        );
        assert_eq!(refused.reason(), "decode_opcode_unknown");
    }

    /// A record whose framing does not fit its segment stops the walk where it
    /// started, not where the length pointed.
    #[test]
    fn a_record_that_overruns_its_segment_names_where_it_began() {
        let good = line_width(2.5);
        let mut bytes = segment_bytes(SegmentKind::Render.wire_type(), std::slice::from_ref(&good));
        // Extend the segment by four bytes that cannot be a record header.
        let length = u32::from_le_bytes(bytes[..4].try_into().expect("four bytes"));
        bytes[..4].copy_from_slice(&(length + 4).to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]);

        assert_eq!(
            exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder()),
            Err(WalkRefusal::RecordFraming {
                at: StreamSite {
                    segment: 0,
                    offset: (SEGMENT_HEADER_LEN + good.len()) as u32,
                },
            })
        );
    }

    /// A framing the stream layer refuses never reaches the model.
    #[test]
    fn a_stream_that_does_not_frame_refuses_before_any_record() {
        let bytes = [0u8; 4];
        assert_eq!(
            exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder()),
            Err(WalkRefusal::Framing(FramingRefusal::ShortHeader {
                at: 0,
                remaining: 4,
            }))
        );
    }

    /// An encoder the guest asks to span two segments is refused, because the
    /// model cannot represent one. Opening a fresh encoder instead would
    /// attribute the second segment's records to a pass the guest never opened.
    #[test]
    fn an_encoder_that_spans_segments_is_refused_rather_than_reopened() {
        for (previous, next) in [(1u8, 0u8), (0, 1), (1, 1)] {
            let mut first = segment_bytes(SegmentKind::Blit.wire_type(), &[generate_mipmaps(1)]);
            first[5] = previous;
            first[6] = next;
            assert_eq!(
                exec(&first, &Everything, &mut StubRegistry(DOMAIN), builder()),
                Err(WalkRefusal::EncoderSpansSegments {
                    at: StreamSite {
                        segment: 0,
                        offset: 0,
                    },
                }),
                "{previous}/{next}"
            );
        }
    }

    /// An empty stream is an empty transaction, not a refusal. A guest may
    /// submit a command buffer that encoded nothing.
    #[test]
    fn an_empty_stream_is_an_empty_transaction() {
        let tx = exec(&[], &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("a stream with no segments");
        assert_eq!(tx.record_count(), 0);
        assert!(tx.streams.is_empty());
        assert!(tx.accesses.is_empty());
    }

    /// The arenas a resolver fills are the builder's, so a record that files a
    /// variable-length entry names a window the finished transaction can read
    /// back.
    #[test]
    fn resolution_files_into_the_transactions_own_arenas() {
        // A default set is what a fresh builder starts from; the walk must not
        // hand a resolver anything else, or a window filed during resolution
        // would name an arena nobody keeps.
        assert_eq!(ExecArenas::default().resources.len(), 0);
        let bytes = segment_bytes(
            SegmentKind::Blit.wire_type(),
            &[generate_mipmaps(9), generate_mipmaps(10)],
        );
        let tx = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("two single-ref records");
        assert_eq!(tx.record_count(), 2);
        assert_eq!(tx.accesses.len(), 2);
    }

    /// Every refusal reason is distinct where this module owns it, and is the
    /// owner's own where it does not.
    #[test]
    fn walk_refusal_reasons_name_the_check_that_refused() {
        let framing = FramingRefusal::UnknownType {
            at: 0,
            wire_type: 9,
        };
        assert_eq!(WalkRefusal::Framing(framing).reason(), framing.reason());
        let inner = StreamRefusal::RecordOutsideEncoder;
        assert_eq!(
            WalkRefusal::Unfinished { refusal: inner }.reason(),
            inner.reason()
        );
        let site = StreamSite {
            segment: 0,
            offset: 0,
        };
        let own = [
            WalkRefusal::RecordFraming { at: site }.reason(),
            WalkRefusal::EncoderSpansSegments { at: site }.reason(),
        ];
        assert_ne!(own[0], own[1]);
    }
}
