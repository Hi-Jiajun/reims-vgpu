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

use crate::access::{
    AccessIntent, AccessKey, AccessSource, BackingId, ContentVersion, Participation,
};
use crate::bind::{BufferBinding, ObjectBinding};
use crate::blit::BlitOp;
use crate::compute::ComputeOp;
use crate::icb::IcbOp;
use crate::identity::{
    ChannelId, ChannelSequence, IngressOrdinal, ResourceId, SessionGeneration, TransactionIdentity,
};
use crate::operation::OperationClass;
use crate::pass::PassDescriptor;
use crate::render::{RenderOp, ScissorRect, Viewport};
use crate::resource_state::ResourceStateOp;
use crate::stream::{
    EncoderBoundary, SegmentBegin, SegmentEnd, SegmentKind, SegmentLifetime, SegmentOpening,
    StreamCursor, StreamPosition, StreamRefusal,
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

impl ResolvedOperation {
    /// The vocabulary class this operation belongs to.
    ///
    /// An exhaustive match, and that is the point: rail and segment
    /// admissibility are both answered from the class, so a variant added
    /// without a class here does not compile rather than quietly inheriting
    /// whatever the last arm said.
    #[must_use]
    pub const fn class(&self) -> OperationClass {
        match self {
            Self::EncoderBoundary(_) => OperationClass::EncoderBoundary,
            Self::Render(_) => OperationClass::Render,
            Self::Compute(_) => OperationClass::Compute,
            Self::Blit(_) => OperationClass::Blit,
            Self::Event(_) => OperationClass::Event,
            Self::Fence(_) => OperationClass::Fence,
            Self::Barrier(_) => OperationClass::Barrier,
            Self::ResourceState(_) => OperationClass::ResourceState,
            Self::IndirectCommand(_) => OperationClass::IndirectCommand,
        }
    }

    /// Every participation this record declares by itself, appended to `out`.
    ///
    /// **The link between "a record resolves" and "a transaction has
    /// accesses".** Each payload module states what its own records name — a
    /// draw's index buffer, a copy's two ends, a synchronise's content, and the
    /// classes that name nothing say so — and this is the one place those
    /// answers are collected. Before it, every one of those methods was
    /// reachable only from its own tests, and no caller could ask a resolved
    /// stream what it touched without knowing the whole vocabulary itself.
    ///
    /// Exhaustive, like [`Self::class`], and for the same reason: a variant
    /// added without an arm here would silently contribute no accesses, and an
    /// operation missing from the access list is a hazard edge that does not
    /// get built — a race, not a slowdown.
    ///
    /// `arenas` is the transaction's own, and only one arm reads it: a render
    /// `WriteDescriptor` carries a [`crate::render::PassDescriptorSlot`],
    /// because the descriptor is 592 bytes and a record that carried it by
    /// value would make every eight-byte record that size. Its participations
    /// are the pass's cost *before any draw* — which is what makes a pass with
    /// no draws still a write — and this is the only scope that can reach it.
    /// A slot past the end of the arena contributes nothing rather than
    /// panicking: the arena and the slot are built together by
    /// [`ExecBuilder`], so a mismatch is a bug in this crate, and taking a
    /// stream down over it would lose the whole packet.
    ///
    /// Appended rather than returned so a caller walking a whole EXEC keeps one
    /// buffer. A `Vec` per record would be an allocation per record on the
    /// hottest path this crate has.
    pub fn participations(&self, arenas: &ExecArenas, out: &mut Vec<Participation>) {
        match self {
            Self::EncoderBoundary(op) => out.extend(op.participations()),
            Self::Render(op) => {
                out.extend(op.participations());
                if let RenderOp::WriteDescriptor { descriptor } = op {
                    if let Some(pass) = arenas.pass_descriptors.get(descriptor.0 as usize) {
                        pass.extend_participations(out);
                    }
                }
            }
            Self::Compute(op) => out.extend(op.participations()),
            Self::Blit(op) => out.extend(op.participations()),
            Self::Event(op) => out.extend(op.participations()),
            Self::Fence(op) => out.extend(op.participations()),
            Self::Barrier(op) => out.extend(op.participations()),
            Self::ResourceState(op) => out.extend(op.participations()),
            Self::IndirectCommand(op) => out.extend(op.participations()),
        }
    }
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

/// A content version this transaction makes current, and exactly the memory it
/// covers.
///
/// **Derived from the accesses, never stated beside them.** A version claim
/// *is* a write access's claim to produce the next content of the memory it
/// names, so the two cannot be separate lists without being able to disagree
/// about the region — and that disagreement is not hypothetical. A reservation
/// that named a whole backing while its access named a range let two writers of
/// disjoint ranges both claim to produce one backing's next version, with
/// nothing ordering them and no legal answer to which version the backing ended
/// at. Region coverage is the access's own key, and the same `may_alias` that
/// decides hazards decides whether two claims collide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VersionPublication {
    pub backing: BackingId,
    /// Exactly the memory the producing access named.
    pub region: AccessKey,
    pub to: ContentVersion,
}

/// The content versions an access list makes current.
///
/// One per access that declared an output version and names memory. A heap
/// declaration or a domain-only access produces none: neither names bytes, so
/// neither can claim to have produced any.
///
/// A function of the accesses and nothing else, which is why it is free rather
/// than a method on [`ExecWork`]: a lifecycle synchronise and a query reply are
/// writes a guest reads back, and the reference interpreter has to publish
/// their versions by the same rule it publishes a draw's.
pub fn published_versions(
    accesses: &[AccessIntent],
) -> impl Iterator<Item = VersionPublication> + '_ {
    accesses.iter().filter_map(|access| {
        let to = access.output_content_version?;
        let backing = match access.key {
            AccessKey::Range(r, _) | AccessKey::Subresource(r, _) | AccessKey::Whole(r) => {
                r.backing
            }
            AccessKey::Heap(_) | AccessKey::DomainOnly => return None,
        };
        Some(VersionPublication {
            backing,
            region: access.key,
            to,
        })
    })
}

/// Something this transaction's *records* must wait for that is not a hazard
/// edge.
///
/// Hazard edges are compiled from accesses and always point backwards in
/// ingress order. These do not: a guest may wait for an event value nothing has
/// produced yet, and that is ordinary rather than an error. The two are
/// separate types because they are separate questions, and [`crate::ready`]
/// tracks them apart for exactly that reason.
///
/// **Completion-stamp waits are not here.** They are the packet's, decoded from
/// its envelope at ingress before any side effect, and a control packet that
/// carries no records carries them too — see
/// [`crate::transaction::DeviceTransaction::stamp_waits`]. A `Stamp` arm here
/// would be that same wait stated a second time, on the one class of packet
/// that also has an envelope to state it in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prerequisite {
    /// An event value.
    Event { event: ResourceId, value: u64 },
    /// A fence another encoder updates.
    Fence { fence: ResourceId },
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

/// One EXEC packet's resolved contents, frozen, and not yet anywhere in
/// particular.
///
/// **Deliberately carries no identity.** Resolving a packet's records is a
/// function of the packet's bytes and the namespaces they name; it does not
/// observe arrival, so it cannot know the packet's channel sequence or ingress
/// ordinal, and being unable to *say* is what keeps a resolver from stating one
/// that disagrees with the envelope's. [`Self::stamp`] pairs this with the
/// [`TransactionIdentity`] that admission assigned, and is the only way to get
/// an [`ExecTransaction`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExecWork {
    pub streams: Vec<ResolvedStream>,
    /// Everything this transaction touches, at the precision the records
    /// supplied.
    pub accesses: Vec<AccessIntent>,
    pub pipeline_leases: Vec<ResourceId>,
    pub prerequisites: Vec<Prerequisite>,
    pub arenas: ExecArenas,
}

impl ExecWork {
    /// Every record, in execution order.
    pub fn records(&self) -> impl Iterator<Item = &StreamRecord> {
        self.streams.iter().flat_map(|s| s.records.iter())
    }

    /// The content versions this work makes current.
    pub fn published_versions(&self) -> impl Iterator<Item = VersionPublication> + '_ {
        published_versions(&self.accesses)
    }

    /// Pair this work with where the packet carrying it arrived.
    ///
    /// Borrows rather than consuming, because the work belongs to the
    /// [`crate::transaction::DeviceTransaction`] that carries it and an
    /// executor that owned a second copy would be a second answer to "what does
    /// this packet touch".
    #[must_use]
    pub const fn stamp(&self, identity: TransactionIdentity) -> ExecTransaction<'_> {
        ExecTransaction {
            identity,
            work: self,
        }
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

/// One accepted EXEC packet as an executor sees it: where the packet sits, and
/// a borrow of what it resolved to.
///
/// **A view, not a container.** The work is the envelope's — see
/// [`crate::transaction::Payload::Exec`] — and the identity is the envelope's
/// too. Owning either would make an admitted EXEC representable twice, and two
/// representations of one packet are two answers to "what does it touch". This
/// is the derivation the plan writes as
/// `submission_domain = DeviceTransaction.channel`, spelled as a borrow so
/// there is nothing to keep in step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExecTransaction<'a> {
    /// Assigned by admission, which is the only service that observes arrival.
    pub identity: TransactionIdentity,
    pub work: &'a ExecWork,
}

impl ExecTransaction<'_> {
    /// The semantic lifetime this was accepted in.
    #[must_use]
    pub const fn session(&self) -> SessionGeneration {
        self.identity.session
    }

    /// The submission ordering domain.
    #[must_use]
    pub const fn domain(&self) -> ChannelId {
        self.identity.domain
    }

    /// Position within that domain.
    #[must_use]
    pub const fn domain_sequence(&self) -> ChannelSequence {
        self.identity.domain_sequence
    }

    /// Position in the device's single arrival order.
    #[must_use]
    pub const fn ingress(&self) -> IngressOrdinal {
        self.identity.ingress
    }

    /// Everything this transaction touches.
    #[must_use]
    pub fn accesses(&self) -> &[AccessIntent] {
        &self.work.accesses
    }

    /// Waits that are not hazard edges.
    #[must_use]
    pub fn prerequisites(&self) -> &[Prerequisite] {
        &self.work.prerequisites
    }
    /// Every record, in execution order.
    pub fn records(&self) -> impl Iterator<Item = &StreamRecord> {
        self.work.records()
    }

    /// The content versions this transaction makes current.
    pub fn published_versions(&self) -> impl Iterator<Item = VersionPublication> + '_ {
        self.work.published_versions()
    }

    /// How many records this packet carries.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.work.record_count()
    }

    /// Whether this transaction writes anything at all.
    #[must_use]
    pub fn writes_anything(&self) -> bool {
        self.work.writes_anything()
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
    streams: Vec<ResolvedStream>,
    open: Option<ResolvedStream>,
    accesses: Vec<AccessIntent>,
    pipeline_leases: Vec<ResourceId>,
    prerequisites: Vec<Prerequisite>,
    arenas: ExecArenas,
    /// Scratch [`Self::record`] gathers each operation's participations into,
    /// so the walk costs no allocation after the first record.
    participation_scratch: Vec<Participation>,
}

impl Default for ExecBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecBuilder {
    /// A builder for a packet whose contents are not yet known and whose
    /// position in the arrival order is not this layer's to know. See
    /// [`ExecWork`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            cursor: StreamCursor::new(),
            streams: Vec::new(),
            open: None,
            accesses: Vec::new(),
            pipeline_leases: Vec::new(),
            prerequisites: Vec::new(),
            arenas: ExecArenas::default(),
            participation_scratch: Vec::new(),
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

    /// Open an encoder, named by the type byte in its segment header.
    pub fn begin_segment(
        &mut self,
        wire_type: u8,
        lifetime: SegmentLifetime,
    ) -> Result<(), StreamRefusal> {
        let opening = self.cursor.begin(wire_type, lifetime)?;
        self.opened(opening);
        Ok(())
    }

    /// Open an encoder whose kind is already established.
    ///
    /// A stream walker cannot cut a segment's window without knowing which
    /// family the segment is, so by the time it gets here the type byte has
    /// been parsed. Handing back the byte for this to parse again is the
    /// "resolve twice" shape the replacement exists to remove.
    pub fn begin_encoder(
        &mut self,
        kind: SegmentKind,
        lifetime: SegmentLifetime,
    ) -> Result<(), StreamRefusal> {
        let opening = self.cursor.begin_kind(kind, lifetime)?;
        self.opened(opening);
        Ok(())
    }

    /// A segment opened. Only a *new* encoder makes a new [`ResolvedStream`];
    /// a continuation keeps writing into the one already open, which is what
    /// makes an encoder spanning segments one encoder here rather than two.
    fn opened(&mut self, opening: SegmentOpening) {
        if let SegmentOpening::Opened(_, begin) = opening {
            self.open = Some(ResolvedStream {
                begin,
                records: Vec::new(),
            });
        }
    }

    /// Record one resolved operation inside the open encoder, and declare
    /// every access it names.
    ///
    /// The rail is taken from the operation's own class rather than passed
    /// alongside it, so a caller cannot hand a compute payload in under a
    /// render rail and have it accepted by the segment it is standing in.
    ///
    /// The accesses are derived here rather than declared beside the record,
    /// and that is the point. A transaction whose `accesses` disagree with its
    /// `streams` is representable the moment the two are supplied separately —
    /// and the disagreement is silent, because nothing downstream reads the
    /// records to check. Here the record *is* the declaration: the operation
    /// states its own participation, `source` places it, and neither can be
    /// supplied without the other.
    ///
    /// # Errors
    ///
    /// As before, plus [`StreamRefusal::Access`] where a participation cannot
    /// be placed. The whole transaction refuses; see that variant.
    pub fn record(
        &mut self,
        op: ResolvedOperation,
        source: &mut impl AccessSource,
    ) -> Result<StreamPosition, StreamRefusal> {
        // Taken out and put back, so the walk costs no allocation after the
        // first record of the first transaction. It also releases the borrow
        // of `self` that `participations` takes on the arenas.
        let mut parts = core::mem::take(&mut self.participation_scratch);
        parts.clear();
        op.participations(&self.arenas, &mut parts);
        // Pushed straight onto the transaction's list and rolled back on any
        // refusal, rather than gathered into a second vector: a record's
        // accesses are at most a handful and a `Vec` per record would be an
        // allocation per record on the hottest path this crate has.
        let before = self.accesses.len();
        let mut refused = None;
        for participation in &parts {
            match source.access(participation) {
                Ok(access) => self.accesses.push(access),
                Err(refusal) => {
                    refused = Some(StreamRefusal::Access(refusal));
                    break;
                }
            }
        }
        self.participation_scratch = parts;
        match refused.map_or_else(|| self.place(op), Err) {
            Ok(at) => Ok(at),
            Err(refusal) => {
                // Nothing half-applied: a record the cursor would not take
                // leaves no accesses behind claiming it ran.
                self.accesses.truncate(before);
                Err(refusal)
            }
        }
    }

    /// The rail and cursor half of [`Self::record`].
    fn place(&mut self, op: ResolvedOperation) -> Result<StreamPosition, StreamRefusal> {
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

    /// End the open segment, and the encoder inside it if it ends here.
    ///
    /// A held encoder's [`ResolvedStream`] is not filed: it is still being
    /// written to, and filing it would put a half-recorded encoder into the
    /// transaction with its remaining records landing nowhere.
    pub fn end_segment(&mut self) -> Result<SegmentEnd, StreamRefusal> {
        let end = self.cursor.end()?;
        if matches!(end, SegmentEnd::EncoderEnded(_)) {
            self.streams
                .push(self.open.take().expect("an encoder was open"));
        }
        Ok(end)
    }

    /// State an access beside the records rather than deriving it from them.
    ///
    /// **Test-only, and the gate is the invariant.** A transaction whose
    /// `accesses` disagree with its `streams` is representable the moment the
    /// two can be supplied separately, and the disagreement is silent because
    /// nothing downstream reads the records to check — which is why
    /// [`Self::record`] derives them. What is left is a test that wants an
    /// access shape the registry would not produce, and the `cfg` is what says
    /// no production path may want the same thing.
    #[cfg(test)]
    pub(crate) fn declare_access(&mut self, access: AccessIntent) {
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

    /// Freeze the transaction.
    ///
    /// Consumes the builder, so the value that comes out cannot be the one that
    /// was still being written to.
    pub fn finish(mut self) -> Result<ExecWork, StreamRefusal> {
        // `finish` on the cursor is what refuses an encoder that never ended,
        // and an envelope that armed nothing.
        self.cursor.finish()?;
        debug_assert!(self.open.is_none(), "the cursor would have refused");
        self.streams.shrink_to_fit();
        self.accesses.shrink_to_fit();
        Ok(ExecWork {
            streams: core::mem::take(&mut self.streams),
            accesses: core::mem::take(&mut self.accesses),
            pipeline_leases: core::mem::take(&mut self.pipeline_leases),
            prerequisites: core::mem::take(&mut self.prerequisites),
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
    Some(match op.class() {
        OperationClass::Render => Rail::Render,
        OperationClass::Compute => Rail::Compute,
        OperationClass::Blit => Rail::Blit,
        OperationClass::Event => Rail::Event,
        OperationClass::InfoQuery => Rail::Info,
        OperationClass::EncoderBoundary
        | OperationClass::Fence
        | OperationClass::Barrier
        | OperationClass::ResourceState
        | OperationClass::IndirectCommand
        | OperationClass::CompletionEffect => return None,
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
    class_admissible_on(op.class(), kind)
}

/// Whether a class of record may appear inside a `kind` segment.
///
/// Keyed on the class rather than on the payload variant, because that is what
/// the ledger is keyed on. A payload added to an existing class inherits its
/// class's answer instead of needing one written for it, and a class added
/// without an answer does not compile.
///
/// The sets are narrower than "more than one" and the narrowness is the point:
/// a fence exists on the render and blit encoders and **not** on the compute
/// one, because the compute pair is unresolved; a barrier exists on render and
/// compute and not on blit; residency's rows are all unresolved, so the
/// resource-state class reaches only the encoders whose *content* records are
/// judged. Admitting a class on every encoder that is not its own would let a
/// compute fence through the one door the ledger closed.
///
/// Hard-coded and then checked against the ledger, which is the same
/// arrangement the payload vocabularies use: the table is what runs, and the
/// test is what says the table still describes the contract.
const fn class_admissible_on(class: OperationClass, kind: SegmentKind) -> bool {
    match class {
        // A boundary is the segment. Every encoder has one.
        OperationClass::EncoderBoundary => true,
        OperationClass::Fence => matches!(kind, SegmentKind::Render | SegmentKind::Blit),
        OperationClass::Barrier => matches!(kind, SegmentKind::Render | SegmentKind::Compute),
        OperationClass::ResourceState => matches!(kind, SegmentKind::Blit | SegmentKind::Compute),
        OperationClass::IndirectCommand => matches!(kind, SegmentKind::Render | SegmentKind::Blit),
        // The single-rail classes never reach here, and the two classes with no
        // stream records at all reach nothing.
        OperationClass::Render
        | OperationClass::Compute
        | OperationClass::Blit
        | OperationClass::Event
        | OperationClass::InfoQuery
        | OperationClass::CompletionEffect => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessKey, AccessMode, ResourceKey, StubRegistry};
    use crate::identity::{ObjectListRef, SlotGeneration};
    use crate::stream::ProtectionOptions;
    use crate::sync::{BarrierOp, BarrierTarget, ResourceSpan};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn builder() -> ExecBuilder {
        ExecBuilder::new()
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
        b.begin_segment(
            SegmentKind::Blit.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        let first = b
            .record(a_blit(), &mut StubRegistry(ChannelId(1)))
            .expect("record");
        let second = b
            .record(a_blit(), &mut StubRegistry(ChannelId(1)))
            .expect("record");
        b.end_segment().expect("end");
        b.begin_segment(
            SegmentKind::Blit.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        let third = b
            .record(a_blit(), &mut StubRegistry(ChannelId(1)))
            .expect("record");
        b.end_segment().expect("end");

        let tx = b.finish().expect("frozen");
        assert_eq!(tx.streams.len(), 2);
        assert_eq!(tx.record_count(), 3);
        let positions: Vec<_> = tx.records().map(|r| r.at).collect();
        assert_eq!(positions, vec![first, second, third]);
        assert!(positions.windows(2).all(|w| w[0] < w[1]));
    }

    /// The builder cannot say where its packet arrived, so the only way a
    /// transaction gets an identity is by being stamped with one — and the
    /// stamped value is exactly the one it was handed.
    #[test]
    fn work_carries_no_identity_until_it_is_stamped_with_one() {
        let identity = TransactionIdentity {
            session: SessionGeneration::FIRST,
            domain: ChannelId(1),
            domain_sequence: ChannelSequence(7),
            ingress: IngressOrdinal(42),
        };
        let work = builder().finish().expect("frozen");
        let tx = work.stamp(identity);
        assert_eq!(tx.identity, identity);
        assert_eq!(tx.ingress(), IngressOrdinal(42));
        assert_eq!(tx.domain(), ChannelId(1));
        assert_eq!(tx.domain_sequence(), ChannelSequence(7));
        assert_eq!(tx.session(), SessionGeneration::FIRST);
        assert_eq!(
            *tx.work, work,
            "stamping adds a position and changes nothing about the work"
        );
    }

    /// The builder does not restate the stream rules; it is the cursor that
    /// refuses, and the refusal reaches the caller unchanged.
    #[test]
    fn the_stream_rules_are_the_cursors_and_are_not_restated() {
        let mut b = builder();
        assert_eq!(
            b.record(a_blit(), &mut StubRegistry(ChannelId(1))),
            Err(StreamRefusal::RecordOutsideEncoder)
        );

        let mut b = builder();
        b.begin_segment(
            SegmentKind::Blit.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        assert_eq!(
            b.record(
                ResolvedOperation::Render(RenderOp::SetPipeline { pipeline: res(1) }),
                &mut StubRegistry(ChannelId(1))
            ),
            Err(StreamRefusal::RailMismatch {
                segment: SegmentKind::Blit,
                record: reims_vgpu_protocol::closure::Rail::Render,
            })
        );

        let mut b = builder();
        b.begin_segment(
            SegmentKind::Compute.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
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
            b.begin_segment(kind.wire_type(), SegmentLifetime::SELF_CONTAINED)
                .expect("open");
            b.record(a_barrier(), &mut StubRegistry(ChannelId(1)))
                .expect("a barrier exists on both");
            b.end_segment().expect("end");
            assert_eq!(b.finish().expect("frozen").record_count(), 1);
        }
        for kind in [SegmentKind::Blit, SegmentKind::Event, SegmentKind::Info] {
            let mut b = builder();
            b.begin_segment(kind.wire_type(), SegmentLifetime::SELF_CONTAINED)
                .expect("open");
            assert_eq!(
                b.record(a_barrier(), &mut StubRegistry(ChannelId(1))),
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
    /// Driven over **every** class rather than over a written list of probes.
    /// For each one, the segments it is admitted on are exactly the rails the
    /// ledger has judged an operation of that class on. A class whose rows are
    /// all unresolved is admitted nowhere, which is residency's case today and
    /// the compute fence pair's — the selector exists and the door is closed.
    ///
    /// The probe list this replaced could not see a class it did not name, and
    /// a payload added under an existing class inherits that class's answer
    /// now rather than needing an entry nobody remembers to add.
    #[test]
    fn the_admissibility_table_matches_the_ledger() {
        use crate::operation::{classify, OperationHome};
        use reims_vgpu_protocol::closure::LEDGER;

        for &class in OperationClass::ALL {
            for &kind in SegmentKind::ALL {
                let ledger_has_one = LEDGER.iter().any(|o| {
                    o.rail == kind.rail()
                        && classify(o) == Some(OperationHome::Stream(class))
                        && !matches!(
                            o.closure,
                            reims_vgpu_protocol::closure::Closure::Refused { .. }
                        )
                });
                // A single-rail class is admitted by its own rail rather than
                // by this table, and the boundary is the segment itself.
                let single_rail = matches!(
                    class,
                    OperationClass::Render
                        | OperationClass::Compute
                        | OperationClass::Blit
                        | OperationClass::Event
                        | OperationClass::InfoQuery
                );
                if single_rail || matches!(class, OperationClass::EncoderBoundary) {
                    continue;
                }
                assert_eq!(
                    class_admissible_on(class, kind),
                    ledger_has_one,
                    "{class:?} on {kind:?}"
                );
            }
        }
    }

    /// Every payload variant reports the class its records are judged under,
    /// and a class this table admits somewhere has a payload that can reach it.
    #[test]
    fn every_multi_rail_class_that_is_admitted_somewhere_has_a_payload() {
        let samples = [
            ResolvedOperation::Fence(FenceOp {
                kind: crate::sync::FenceKind::Update,
                fence: res(1),
                stages: None,
            }),
            a_barrier(),
            ResolvedOperation::ResourceState(ResourceStateOp {
                directive: crate::resource_state::ContentDirective::Synchronize,
                target: crate::resource_state::ResourceStateTarget::Encoder,
            }),
            ResolvedOperation::IndirectCommand(IcbOp::ExecuteRange {
                icb: res(1),
                commands: crate::icb::CommandRange::default(),
            }),
        ];
        for &class in OperationClass::ALL {
            let admitted = SegmentKind::ALL
                .iter()
                .any(|&kind| class_admissible_on(class, kind));
            if !admitted || matches!(class, OperationClass::EncoderBoundary) {
                continue;
            }
            assert!(
                samples.iter().any(|op| op.class() == class),
                "{class:?} is admitted and has no payload to admit"
            );
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
        assert_eq!(tx.prerequisites.len(), 1);
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

    /// A version claim is the write access's claim, so it is read off the
    /// access and cannot name different memory than the write did.
    #[test]
    fn a_published_version_covers_exactly_the_region_its_access_named() {
        let region = AccessKey::Range(
            crate::access::ResourceKey {
                backing: BackingId(9),
                heap: None,
            },
            crate::access::ByteRange {
                offset: 64,
                length: 128,
            },
        );
        let mut b = builder();
        b.declare_access(AccessIntent {
            domain: ChannelId(1),
            key: region,
            mode: crate::access::AccessMode::Write,
            api_stages: 0,
            input_content_version: None,
            output_content_version: Some(ContentVersion(2)),
        });
        let tx = b.finish().expect("frozen");
        assert_eq!(
            tx.published_versions().collect::<Vec<_>>(),
            vec![VersionPublication {
                backing: BackingId(9),
                region,
                to: ContentVersion(2),
            }]
        );
    }

    /// A heap declaration and a domain-only access name no bytes, so neither
    /// can claim to have produced any.
    #[test]
    fn an_access_that_names_no_memory_publishes_no_version() {
        let mut b = builder();
        for key in [
            AccessKey::Heap(crate::access::HeapId {
                id: 1,
                membership_generation: 0,
            }),
            AccessKey::DomainOnly,
        ] {
            b.declare_access(AccessIntent {
                domain: ChannelId(1),
                key,
                mode: crate::access::AccessMode::Write,
                api_stages: 0,
                input_content_version: None,
                output_content_version: Some(ContentVersion(2)),
            });
        }
        let tx = b.finish().expect("frozen");
        assert_eq!(tx.published_versions().count(), 0);
    }

    /// Every operation class answers the participation question, and the two
    /// that route through something other than their own fields answer it
    /// correctly.
    ///
    /// The aggregation is exhaustive by construction — the match in
    /// `participations` has no wildcard — so what a test can still catch is an
    /// arm wired to the wrong source. Two are:
    ///
    /// * `WriteDescriptor` is the only arm that reads the arena, and it is the
    ///   only participation a *pass* contributes. Wiring it to the record's own
    ///   (empty) answer would lose every attachment of every pass, and a pass
    ///   with no draws would become a transaction that touches nothing.
    /// * A barrier carries a resource list and declares no participation on it.
    ///   Reading that list as accesses would order every barrier against
    ///   everything it named.
    #[test]
    fn every_class_answers_what_it_touches_and_only_the_pass_reads_the_arena() {
        let mut arenas = ExecArenas::default();
        let mut pass = crate::pass::PassDescriptor::empty();
        pass.visibility_result_buffer =
            Some(crate::pass::VisibilityResultBuffer { buffer: res(11) });
        arenas.pass_descriptors.push(pass);
        arenas.resources.push(res(1));

        let ask = |op: ResolvedOperation, arenas: &ExecArenas| -> Vec<Participation> {
            let mut out = Vec::new();
            op.participations(arenas, &mut out);
            out
        };

        // The pass's own footprint, reached only through the arena.
        let write_descriptor = ResolvedOperation::Render(RenderOp::WriteDescriptor {
            descriptor: crate::render::PassDescriptorSlot(0),
        });
        let parts = ask(write_descriptor, &arenas);
        assert_eq!(parts.len(), 1, "the visibility buffer is the pass's write");
        assert_eq!(parts[0].resource, res(11));
        assert_eq!(parts[0].mode, AccessMode::Write);
        // And it really is the arena that supplied it: the same record against
        // an arena that does not hold the slot contributes nothing rather than
        // panicking.
        assert!(ask(write_descriptor, &ExecArenas::default()).is_empty());

        // A barrier names a resource list and participates in none of it.
        assert!(ask(a_barrier(), &arenas).is_empty());
        // A fence and an event name their own object and no memory.
        assert!(ask(
            ResolvedOperation::Fence(crate::sync::FenceOp {
                kind: crate::sync::FenceKind::Update,
                fence: res(2),
                stages: None,
            }),
            &arenas
        )
        .is_empty());
        assert!(ask(
            ResolvedOperation::Event(crate::sync::EventOp {
                kind: crate::sync::EventKind::Signal,
                event: res(3),
                value: 9,
            }),
            &arenas
        )
        .is_empty());
        // A boundary names nothing.
        assert!(ask(
            ResolvedOperation::EncoderBoundary(EncoderBoundary::End { records: 0 }),
            &arenas
        )
        .is_empty());

        // A transfer names its operand.
        let blit = ask(a_blit(), &arenas);
        assert_eq!(blit.len(), 1);
        assert_eq!(blit[0].resource, res(1));

        // A synchronise reads the content it publishes; the four directives
        // with no modelled effect name nothing.
        use crate::resource_state::{ContentDirective, ResourceStateOp, ResourceStateTarget};
        let target = ResourceStateTarget::Resource {
            resource: res(4),
            subresource: None,
        };
        let sync = ask(
            ResolvedOperation::ResourceState(ResourceStateOp {
                directive: ContentDirective::Synchronize,
                target,
            }),
            &arenas,
        );
        assert_eq!(sync.len(), 1);
        assert_eq!(sync[0].resource, res(4));
        assert_eq!(sync[0].mode, AccessMode::Read);
        for directive in [
            ContentDirective::OptimizeForCpu,
            ContentDirective::OptimizeForGpu,
            ContentDirective::InvalidateCompressed,
            ContentDirective::FlushCompressedReinterpretation,
        ] {
            assert!(
                ask(
                    ResolvedOperation::ResourceState(ResourceStateOp { directive, target }),
                    &arenas
                )
                .is_empty(),
                "{directive:?}"
            );
        }
    }
}
