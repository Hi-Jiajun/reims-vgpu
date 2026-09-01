//! The EXEC transaction: everything one accepted command-stream packet means,
//! frozen.
//!
//! # Built once, then read-only
//!
//! The value of an immutable transaction is not tidiness. It is that a
//! dependency compiler, a scheduler and an executor all read the same thing and
//! none of them can be the reason it changed — so "what does this packet touch"
//! has one answer for the whole life of the work, and a late discovery is a
//! type error rather than a race.
//!
//! [`ExecBuilder`] is the only way to make one, it consumes itself to produce
//! the transaction, and the transaction has no mutating method. A resolver that
//! wanted to add an access after the fact would have to rebuild.
//!
//! # Records point into arenas, and the arenas are the transaction's
//!
//! A bind record carries a counted array; a pass descriptor is 592 bytes; a
//! resource barrier names a list. Storing those inline would make every record
//! the size of the largest one, in a `Vec` that is walked per draw. So the
//! variable-length parts live in per-kind arenas on the transaction and the
//! operations name windows of them — which also means the whole packet is three
//! or four allocations rather than one per record.
//!
//! # The vocabulary here is the vocabulary `operation` counts
//!
//! [`ResolvedOperation`] has one variant per non-empty operation class.
//! `InfoQuery` and `CompletionEffect` have no variant because they have no
//! judged operations, and that is not an omission this module has to be trusted
//! about: `operation::tests::every_class_has_a_payload_or_a_reason_to_be_empty`
//! fails the moment either count moves off zero.

use crate::access::{AccessIntent, BackingId, ContentVersion};
use crate::bind::{BufferBinding, ObjectBinding};
use crate::blit::BlitOp;
use crate::compute::ComputeOp;
use crate::icb::IcbOp;
use crate::identity::{
    ChannelId, ChannelSequence, CompletionStamp, IngressOrdinal, ResourceId, SessionGeneration,
    StampWait,
};
use crate::pass::PassDescriptor;
use crate::render::{RenderOp, ScissorRect, Viewport};
use crate::resource_state::ResourceStateOp;
use crate::stream::{
    EncoderBoundary, SegmentBegin, SegmentKind, StreamCursor, StreamPosition, StreamRefusal,
};
use crate::sync::{BarrierOp, EventOp, FenceOp};

/// One resolved operation, in the class the vocabulary put it in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResolvedOperation {
    EncoderBoundary(EncoderBoundary),
    Render(RenderOp),
    Compute(ComputeOp),
    Blit(BlitOp),
    Event(EventOp),
    Fence(FenceOp),
    Barrier(BarrierOp),
    ResourceState(ResourceStateOp),
    IndirectCommand(IcbOp),
}

/// One record, and where it sits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamRecord {
    pub at: StreamPosition,
    pub op: ResolvedOperation,
}

/// One encoder's worth of resolved records.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedStream {
    pub begin: SegmentBegin,
    pub records: Vec<StreamRecord>,
}

impl ResolvedStream {
    #[must_use]
    pub const fn kind(&self) -> SegmentKind {
        self.begin.kind
    }
}

/// A content version this transaction will produce.
///
/// Reserved at planning and committed at completion, which is the whole reason
/// it is a pair: a reader planned against `to` waits for the work, and a reader
/// that only needs `from` does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VersionReservation {
    pub backing: BackingId,
    pub from: ContentVersion,
    pub to: ContentVersion,
}

/// Something this transaction must wait for that is not a hazard edge.
///
/// Hazard edges are compiled from accesses and always point backwards in
/// ingress order. These do not: a guest may wait for a stamp or an event value
/// nothing has produced yet, and that is ordinary rather than an error. The two
/// are separate types because they are separate questions, and
/// [`crate::ready`] tracks them apart for exactly that reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prerequisite {
    /// A completion-stamp point.
    Stamp(StampWait),
    /// An event value.
    Event { event: ResourceId, value: u64 },
    /// A fence another encoder updates.
    Fence { fence: ResourceId },
}

/// What this transaction makes visible when its work completes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PublicationContract {
    /// The stamp the guest polls, if this packet carries one.
    pub stamp: Option<CompletionStamp>,
    /// The content versions that become current.
    ///
    /// Published *after* the work completes and never before, which is the rule
    /// the plan states as "results visible before the completion word": the
    /// version and the stamp become visible together, and a reader that saw the
    /// stamp without the version would read stale content with a fresh flag.
    pub versions: Vec<VersionReservation>,
}

/// The variable-length parts of a packet's records.
///
/// One arena per entry shape, because the shapes have different sizes and a
/// single arena of a union would make the smallest entry as large as the
/// largest.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExecArenas {
    pub buffer_bindings: Vec<BufferBinding>,
    pub object_bindings: Vec<ObjectBinding>,
    /// Resources named by a barrier's counted list.
    pub resources: Vec<ResourceId>,
    pub pass_descriptors: Vec<PassDescriptor>,
    pub viewports: Vec<Viewport>,
    pub scissors: Vec<ScissorRect>,
}

/// One accepted EXEC packet, resolved and frozen.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecTransaction {
    pub session: SessionGeneration,
    /// The submission ordering domain. The same value as the device
    /// transaction's channel, carried rather than looked up so a conflict test
    /// never has to reach for the envelope.
    pub domain: ChannelId,
    pub domain_sequence: ChannelSequence,
    pub ingress: IngressOrdinal,
    pub streams: Vec<ResolvedStream>,
    /// Everything this transaction touches, at the precision the records
    /// supplied.
    pub accesses: Vec<AccessIntent>,
    pub pipeline_leases: Vec<ResourceId>,
    pub prerequisites: Vec<Prerequisite>,
    pub publication: PublicationContract,
    pub arenas: ExecArenas,
}

impl ExecTransaction {
    /// Every record, in execution order.
    pub fn records(&self) -> impl Iterator<Item = &StreamRecord> {
        self.streams.iter().flat_map(|s| s.records.iter())
    }

    /// How many records this packet carries.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.streams.iter().map(|s| s.records.len()).sum()
    }

    /// Whether this transaction writes anything at all.
    ///
    /// A transaction that writes nothing cannot be the producer half of a
    /// hazard, which is worth one question rather than a scan per candidate
    /// edge.
    #[must_use]
    pub fn writes_anything(&self) -> bool {
        self.accesses.iter().any(|a| a.mode.writes())
    }
}

/// Builds one [`ExecTransaction`], and refuses the shapes that cannot execute.
///
/// It owns a [`StreamCursor`], so the segment/encoder rules are enforced here
/// rather than restated: a record with no open encoder, a rail that disagrees
/// with its segment, an encoder that never ended — each is the cursor's refusal
/// and reaches the caller unchanged.
#[derive(Debug)]
pub struct ExecBuilder {
    cursor: StreamCursor,
    session: SessionGeneration,
    domain: ChannelId,
    domain_sequence: ChannelSequence,
    ingress: IngressOrdinal,
    streams: Vec<ResolvedStream>,
    open: Option<ResolvedStream>,
    accesses: Vec<AccessIntent>,
    pipeline_leases: Vec<ResourceId>,
    prerequisites: Vec<Prerequisite>,
    publication: PublicationContract,
    arenas: ExecArenas,
}

impl ExecBuilder {
    #[must_use]
    pub fn new(
        session: SessionGeneration,
        domain: ChannelId,
        domain_sequence: ChannelSequence,
        ingress: IngressOrdinal,
    ) -> Self {
        Self {
            cursor: StreamCursor::new(),
            session,
            domain,
            domain_sequence,
            ingress,
            streams: Vec::new(),
            open: None,
            accesses: Vec::new(),
            pipeline_leases: Vec::new(),
            prerequisites: Vec::new(),
            publication: PublicationContract::default(),
            arenas: ExecArenas::default(),
        }
    }

    /// The arenas, so a resolver can file variable-length entries and name the
    /// window it filed them at.
    pub fn arenas_mut(&mut self) -> &mut ExecArenas {
        &mut self.arenas
    }

    /// A protection envelope armed the next segment.
    pub fn protection_envelope(
        &mut self,
        options: crate::stream::ProtectionOptions,
    ) -> Result<(), StreamRefusal> {
        self.cursor.protection_envelope(options)
    }

    /// Open an encoder.
    pub fn begin_segment(&mut self, wire_type: u8, flag: bool) -> Result<(), StreamRefusal> {
        let (_, begin) = self.cursor.begin(wire_type, flag)?;
        self.open = Some(ResolvedStream {
            begin,
            records: Vec::new(),
        });
        Ok(())
    }

    /// Record one resolved operation inside the open encoder.
    ///
    /// The rail is taken from the operation's own class rather than passed
    /// alongside it, so a caller cannot hand a compute payload in under a
    /// render rail and have it accepted by the segment it is standing in.
    pub fn record(&mut self, op: ResolvedOperation) -> Result<StreamPosition, StreamRefusal> {
        let rail = match rail_of(&op) {
            Some(rail) => rail,
            None => {
                // A record whose class exists on more than one rail is admitted
                // by whichever encoder is open — but only by an encoder that
                // carries the class at all, which is the check the single-rail
                // records get for free from their own rail.
                let Some(open) = self.cursor.open_encoder() else {
                    return Err(StreamRefusal::RecordOutsideEncoder);
                };
                if !admissible_on(&op, open) {
                    return Err(StreamRefusal::RailMismatch {
                        segment: open,
                        record: open.rail(),
                    });
                }
                open.rail()
            }
        };
        let at = self.cursor.record(rail)?;
        self.open
            .as_mut()
            .expect("the cursor accepted a record, so an encoder is open")
            .records
            .push(StreamRecord { at, op });
        Ok(at)
    }

    /// Close the open encoder.
    pub fn end_segment(&mut self) -> Result<EncoderBoundary, StreamRefusal> {
        let boundary = self.cursor.end()?;
        self.streams
            .push(self.open.take().expect("an encoder was open"));
        Ok(boundary)
    }

    pub fn declare_access(&mut self, access: AccessIntent) {
        self.accesses.push(access);
    }

    pub fn lease_pipeline(&mut self, pipeline: ResourceId) {
        if !self.pipeline_leases.contains(&pipeline) {
            self.pipeline_leases.push(pipeline);
        }
    }

    pub fn require(&mut self, prerequisite: Prerequisite) {
        self.prerequisites.push(prerequisite);
    }

    pub fn publish_stamp(&mut self, stamp: CompletionStamp) {
        self.publication.stamp = Some(stamp);
    }

    pub fn reserve_version(&mut self, reservation: VersionReservation) {
        self.publication.versions.push(reservation);
    }

    /// Freeze the transaction.
    ///
    /// Consumes the builder, so the value that comes out cannot be the one that
    /// was still being written to.
    pub fn finish(mut self) -> Result<ExecTransaction, StreamRefusal> {
        // `finish` on the cursor is what refuses an encoder that never ended,
        // and an envelope that armed nothing.
        self.cursor.finish()?;
        debug_assert!(self.open.is_none(), "the cursor would have refused");
        self.streams.shrink_to_fit();
        self.accesses.shrink_to_fit();
        Ok(ExecTransaction {
            session: self.session,
            domain: self.domain,
            domain_sequence: self.domain_sequence,
            ingress: self.ingress,
            streams: core::mem::take(&mut self.streams),
            accesses: core::mem::take(&mut self.accesses),
            pipeline_leases: core::mem::take(&mut self.pipeline_leases),
            prerequisites: core::mem::take(&mut self.prerequisites),
            publication: core::mem::take(&mut self.publication),
            arenas: core::mem::take(&mut self.arenas),
        })
    }
}

/// Which rail a resolved operation's records are read on, when its class
/// belongs to exactly one.
///
/// `None` for the five classes that exist on more than one encoder. Those are
/// admitted by whichever encoder is open, and [`admissible_on`] is what keeps
/// that from meaning "any encoder at all".
fn rail_of(op: &ResolvedOperation) -> Option<reims_vgpu_protocol::closure::Rail> {
    use reims_vgpu_protocol::closure::Rail;
    Some(match op {
        ResolvedOperation::Render(_) => Rail::Render,
        ResolvedOperation::Compute(_) => Rail::Compute,
        ResolvedOperation::Blit(_) => Rail::Blit,
        ResolvedOperation::Event(_) => Rail::Event,
        ResolvedOperation::EncoderBoundary(_)
        | ResolvedOperation::Fence(_)
        | ResolvedOperation::Barrier(_)
        | ResolvedOperation::ResourceState(_)
        | ResolvedOperation::IndirectCommand(_) => return None,
    })
}

/// Whether a multi-rail record may appear inside a `kind` segment.
///
/// The sets are narrower than "more than one" and the narrowness is the point:
/// a fence exists on the render and blit encoders and **not** on the compute
/// one, because the compute pair is unresolved; a barrier exists on render and
/// compute and not on blit. Admitting a class on every encoder that is not its
/// own would let a compute fence through the one door the ledger closed.
///
/// Hard-coded and then checked against the ledger, which is the same
/// arrangement the payload vocabularies use: the table is what runs, and the
/// test is what says the table still describes the contract.
fn admissible_on(op: &ResolvedOperation, kind: SegmentKind) -> bool {
    match op {
        // A boundary is the segment. Every encoder has one.
        ResolvedOperation::EncoderBoundary(_) => true,
        ResolvedOperation::Fence(_) => matches!(kind, SegmentKind::Render | SegmentKind::Blit),
        ResolvedOperation::Barrier(_) => matches!(kind, SegmentKind::Render | SegmentKind::Compute),
        ResolvedOperation::ResourceState(_) => {
            matches!(kind, SegmentKind::Blit | SegmentKind::Compute)
        }
        ResolvedOperation::IndirectCommand(_) => {
            matches!(kind, SegmentKind::Render | SegmentKind::Blit)
        }
        // The single-rail classes never reach here.
        ResolvedOperation::Render(_)
        | ResolvedOperation::Compute(_)
        | ResolvedOperation::Blit(_)
        | ResolvedOperation::Event(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessKey, AccessMode, ResourceKey};
    use crate::identity::{ObjectListRef, SlotGeneration, StampSlot, StampValue};
    use crate::stream::ProtectionOptions;
    use crate::sync::{BarrierOp, BarrierTarget, ResourceSpan};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn builder() -> ExecBuilder {
        ExecBuilder::new(
            SessionGeneration::FIRST,
            ChannelId(1),
            ChannelSequence(7),
            IngressOrdinal(42),
        )
    }

    fn a_blit() -> ResolvedOperation {
        ResolvedOperation::Blit(BlitOp::GenerateMipmaps { texture: res(1) })
    }

    fn a_barrier() -> ResolvedOperation {
        ResolvedOperation::Barrier(BarrierOp {
            target: BarrierTarget::Resources(ResourceSpan { start: 0, len: 1 }),
            after_stages: None,
            before_stages: None,
        })
    }

    #[test]
    fn a_finished_transaction_carries_its_records_in_order() {
        let mut b = builder();
        b.begin_segment(SegmentKind::Blit.wire_type(), false)
            .expect("open");
        let first = b.record(a_blit()).expect("record");
        let second = b.record(a_blit()).expect("record");
        b.end_segment().expect("end");
        b.begin_segment(SegmentKind::Blit.wire_type(), false)
            .expect("open");
        let third = b.record(a_blit()).expect("record");
        b.end_segment().expect("end");

        let tx = b.finish().expect("frozen");
        assert_eq!(tx.streams.len(), 2);
        assert_eq!(tx.record_count(), 3);
        let positions: Vec<_> = tx.records().map(|r| r.at).collect();
        assert_eq!(positions, vec![first, second, third]);
        assert!(positions.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(tx.ingress, IngressOrdinal(42));
        assert_eq!(tx.domain, ChannelId(1));
    }

    /// The builder does not restate the stream rules; it is the cursor that
    /// refuses, and the refusal reaches the caller unchanged.
    #[test]
    fn the_stream_rules_are_the_cursors_and_are_not_restated() {
        let mut b = builder();
        assert_eq!(b.record(a_blit()), Err(StreamRefusal::RecordOutsideEncoder));

        let mut b = builder();
        b.begin_segment(SegmentKind::Blit.wire_type(), false)
            .expect("open");
        assert_eq!(
            b.record(ResolvedOperation::Render(RenderOp::SetPipeline {
                pipeline: res(1)
            })),
            Err(StreamRefusal::RailMismatch {
                segment: SegmentKind::Blit,
                record: reims_vgpu_protocol::closure::Rail::Render,
            })
        );

        let mut b = builder();
        b.begin_segment(SegmentKind::Compute.wire_type(), false)
            .expect("open");
        assert_eq!(
            b.finish().err(),
            Some(StreamRefusal::EncoderNeverEnded(SegmentKind::Compute))
        );
    }

    /// A record that exists on more than one rail is admitted by whichever
    /// encoder is open — and refused by an encoder its class does not appear
    /// on. A barrier on the blit encoder is the pointed case: there is no such
    /// record, and admitting it would be inventing one.
    #[test]
    fn a_multi_rail_record_is_admitted_only_by_an_encoder_that_carries_it() {
        for kind in [SegmentKind::Render, SegmentKind::Compute] {
            let mut b = builder();
            b.begin_segment(kind.wire_type(), false).expect("open");
            b.record(a_barrier()).expect("a barrier exists on both");
            b.end_segment().expect("end");
            assert_eq!(b.finish().expect("frozen").record_count(), 1);
        }
        for kind in [SegmentKind::Blit, SegmentKind::Event, SegmentKind::Info] {
            let mut b = builder();
            b.begin_segment(kind.wire_type(), false).expect("open");
            assert_eq!(
                b.record(a_barrier()),
                Err(StreamRefusal::RailMismatch {
                    segment: kind,
                    record: kind.rail(),
                }),
                "{kind:?} carries no barrier record"
            );
        }
    }

    /// The admissibility table is the ledger's, and this is what says so.
    ///
    /// For every multi-rail class, the segments it is admitted on are exactly
    /// the rails the ledger has judged an operation of that class on. The
    /// compute fence pair is the case that makes this worth checking: it is
    /// unresolved, so the compute encoder must not admit a fence even though
    /// the selector exists.
    #[test]
    fn the_admissibility_table_matches_the_ledger() {
        use crate::operation::{classify, OperationClass, OperationHome};
        use reims_vgpu_protocol::closure::LEDGER;

        let probes: [(ResolvedOperation, OperationClass); 4] = [
            (
                ResolvedOperation::Fence(FenceOp {
                    kind: crate::sync::FenceKind::Update,
                    fence: res(1),
                    stages: None,
                }),
                OperationClass::Fence,
            ),
            (a_barrier(), OperationClass::Barrier),
            (
                ResolvedOperation::ResourceState(ResourceStateOp {
                    directive: crate::resource_state::ContentDirective::Synchronize,
                    target: crate::resource_state::ResourceStateTarget::Encoder,
                }),
                OperationClass::ResourceState,
            ),
            (
                ResolvedOperation::IndirectCommand(IcbOp::ExecuteRange {
                    icb: res(1),
                    commands: crate::icb::CommandRange::default(),
                }),
                OperationClass::IndirectCommand,
            ),
        ];

        for (op, class) in probes {
            for &kind in SegmentKind::ALL {
                let ledger_has_one = LEDGER.iter().any(|o| {
                    o.rail == kind.rail()
                        && classify(o) == Some(OperationHome::Stream(class))
                        && !matches!(
                            o.closure,
                            reims_vgpu_protocol::closure::Closure::Refused { .. }
                        )
                });
                assert_eq!(
                    admissible_on(&op, kind),
                    ledger_has_one,
                    "{class:?} on {kind:?}"
                );
            }
        }
    }

    /// An envelope that armed nothing fails at freeze, not silently.
    #[test]
    fn an_unclaimed_protection_envelope_refuses_the_whole_packet() {
        let mut b = builder();
        b.protection_envelope(ProtectionOptions(0x44))
            .expect("armed");
        assert_eq!(
            b.finish().err(),
            Some(StreamRefusal::ProtectionEnvelopeUnclaimed)
        );
    }

    /// A pipeline leased twice is leased once. The guest re-binds the same
    /// pipeline on every draw, so the list would otherwise be one entry per
    /// draw and the lease check one lookup per entry.
    #[test]
    fn a_pipeline_is_leased_once_however_often_it_is_bound() {
        let mut b = builder();
        b.lease_pipeline(res(3));
        b.lease_pipeline(res(3));
        b.lease_pipeline(res(4));
        let tx = b.finish().expect("frozen");
        assert_eq!(tx.pipeline_leases, vec![res(3), res(4)]);
    }

    /// Prerequisites are kept apart from accesses, because one of them may name
    /// work that has not arrived.
    #[test]
    fn prerequisites_and_accesses_are_separate_lists() {
        let mut b = builder();
        b.require(Prerequisite::Stamp(StampWait {
            slot: StampSlot(2),
            value: StampValue(9),
        }));
        b.require(Prerequisite::Event {
            event: res(5),
            value: 3,
        });
        b.declare_access(AccessIntent {
            domain: ChannelId(1),
            key: AccessKey::Whole(ResourceKey {
                backing: BackingId(1),
                heap: None,
            }),
            mode: AccessMode::Write,
            api_stages: 0,
            input_content_version: None,
            output_content_version: Some(ContentVersion(2)),
        });
        let tx = b.finish().expect("frozen");
        assert_eq!(tx.prerequisites.len(), 2);
        assert_eq!(tx.accesses.len(), 1);
        assert!(tx.writes_anything());
    }

    /// A transaction that touches nothing says so, and is not mistaken for one
    /// whose participation was never worked out.
    #[test]
    fn a_transaction_with_no_accesses_writes_nothing() {
        let tx = builder().finish().expect("frozen");
        assert!(!tx.writes_anything());
        assert_eq!(tx.record_count(), 0);
        assert!(tx.accesses.is_empty());
    }

    /// The publication contract carries the versions and the stamp together,
    /// which is what lets them become visible together.
    #[test]
    fn publication_carries_the_versions_beside_the_stamp() {
        let mut b = builder();
        b.publish_stamp(CompletionStamp {
            slot: StampSlot(1),
            value: StampValue(5),
        });
        b.reserve_version(VersionReservation {
            backing: BackingId(9),
            from: ContentVersion(1),
            to: ContentVersion(2),
        });
        let tx = b.finish().expect("frozen");
        assert!(tx.publication.stamp.is_some());
        assert_eq!(tx.publication.versions.len(), 1);
        assert_eq!(tx.publication.versions[0].to, ContentVersion(2));
    }
}
