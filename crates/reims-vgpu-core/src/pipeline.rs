//! The pipeline lifetime, and the rule that a draw never compiles.
//!
//! # Why this is a state machine and not a cache
//!
//! A pipeline that is compiled lazily on first use is a pipeline whose
//! compilation cost lands on a draw — and a draw is the one place in this
//! architecture that may not block, wait on a host, or discover work. Making
//! the lifetime explicit moves the cost to where the guest actually asked for
//! it: the object-creation packet starts the work, and a transaction that wants
//! the pipeline holds a lease and is not ready until the lease resolves.
//!
//! So "not compiled yet" is a state a transaction can wait on rather than a
//! cache miss a draw has to handle, and "refused" is a state rather than an
//! error return that some call sites check and others do not.
//!
//! # Refused is terminal and says why
//!
//! A pipeline the device cannot build does not retry on the next draw. It stays
//! refused for the lifetime of the object, with the reason attached, so a guest
//! re-binding it every frame produces one refusal rather than one per frame —
//! and so that the reason survives to whoever reads the failure channel.
//!
//! # What this crate cannot see
//!
//! Nothing here names a shader, a module, a descriptor layout or a native
//! handle. The translation and compilation *happen* somewhere that does; this
//! owns when they may start, what a waiting transaction observes, and when the
//! result may be dropped.

use crate::access::AccessMode;
use crate::identity::{ResourceId, SessionGeneration};
use std::collections::HashMap;

/// Where a pipeline is in its life.
///
/// The order is the lifetime's order, and [`Ord`] follows it, so "has it got at
/// least as far as X" is a comparison rather than a match. `Refused` and
/// `Retired` are both terminal and are deliberately not comparable-as-progress
/// with each other; a caller asking "is this usable" asks [`Self::is_ready`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineState {
    /// The guest has created the object; no work has started.
    Declared,
    /// The guest's shader form is being turned into the host's.
    Translating,
    /// The host is building the pipeline.
    Compiling,
    /// Usable.
    Ready,
    /// The device cannot build it, and will not try again.
    Refused,
    /// The guest deleted it, or its generation closed.
    Retired,
}

impl PipelineState {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Refused | Self::Retired)
    }

    #[must_use]
    pub const fn is_ready(self) -> bool {
        matches!(self, Self::Ready)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Translating => "translating",
            Self::Compiling => "compiling",
            Self::Ready => "ready",
            Self::Refused => "refused",
            Self::Retired => "retired",
        }
    }

    /// Whether `next` is a legal step from here.
    ///
    /// Written as a table rather than as a set of guards at each mutator,
    /// because a guard is a thing one caller can be written without.
    #[must_use]
    pub const fn may_become(self, next: PipelineState) -> bool {
        matches!(
            (self, next),
            (Self::Declared, Self::Translating)
                | (Self::Translating, Self::Compiling)
                | (Self::Compiling, Self::Ready)
                // Translation and compilation can each fail, and a pipeline can
                // be refused before either starts — a descriptor the model
                // cannot represent is refused at declaration.
                | (Self::Declared | Self::Translating | Self::Compiling, Self::Refused)
                // Deletion can arrive at any point that is not already
                // terminal. A guest deleting a pipeline mid-compile is
                // ordinary, and the compile finishing afterwards must not
                // resurrect it.
                | (
                    Self::Declared | Self::Translating | Self::Compiling | Self::Ready,
                    Self::Retired
                )
        )
    }
}

/// What a compiled pipeline does with each slot it binds.
///
/// # An immutable fact, published by whoever compiled it
///
/// Nothing in this crate can read a shader. Which of an encoder's bound slots a
/// pipeline actually references, and in which direction, is discovered during
/// translation — by the executor, which is the layer that has the shader — and
/// it reaches the model as this, once, when the pipeline becomes ready. That is
/// the plan's rule about what advances semantic state: an immutable fact
/// returned by an executor, not a query the model makes.
///
/// # Why the alternative is expensive rather than wrong
///
/// Without one, [`crate::encoder`] gives every bound slot
/// [`AccessMode::Unknown`], which conflicts with everything. No edge is missed;
/// a great many are added. The point of publishing this is to buy those back,
/// and the point of `Unknown` being its own variant is that the census can say
/// how many are still being paid for.
///
/// # A slot past the end is unreferenced, not unknown
///
/// The tables are as long as the pipeline's own binding set. A bound slot
/// beyond them is one the shader does not name, so it contributes nothing —
/// falling back to `Unknown` there would make a guest with a long-tailed bind
/// table pay forever for slots no shader reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BindingUsage {
    buffers: Vec<Option<AccessMode>>,
    textures: Vec<Option<AccessMode>>,
}

impl BindingUsage {
    #[must_use]
    pub fn new(buffers: Vec<Option<AccessMode>>, textures: Vec<Option<AccessMode>>) -> Self {
        Self { buffers, textures }
    }

    /// What the pipeline does with buffer slot `slot`, or `None` if it does not
    /// reference it.
    #[must_use]
    pub fn buffer(&self, slot: u32) -> Option<AccessMode> {
        self.buffers.get(slot as usize).copied().flatten()
    }

    /// What the pipeline does with texture slot `slot`.
    #[must_use]
    pub fn texture(&self, slot: u32) -> Option<AccessMode> {
        self.textures.get(slot as usize).copied().flatten()
    }

    /// Whether the pipeline writes anything through its bindings.
    ///
    /// A pipeline that only reads cannot be the producer half of a hazard, so
    /// this is worth one question rather than a scan at every draw.
    #[must_use]
    pub fn writes_anything(&self) -> bool {
        self.buffers
            .iter()
            .chain(self.textures.iter())
            .flatten()
            .any(|m| m.writes())
    }
}

/// Why a pipeline will not be built.
///
/// A payload rather than a slug, because the reason has to survive to a failure
/// channel this crate cannot reach — and because a refusal without a reason is
/// how a guest ends up rendering nothing with a clean log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefusalReason {
    /// A descriptor field the model has no representation for.
    Undescribable(&'static str),
    /// The guest's shader form could not be translated.
    TranslationFailed(&'static str),
    /// The host refused to build it.
    CompilationFailed(&'static str),
}

impl RefusalReason {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Undescribable(_) => "pipeline_undescribable",
            Self::TranslationFailed(_) => "pipeline_translation_failed",
            Self::CompilationFailed(_) => "pipeline_compilation_failed",
        }
    }

    pub const fn detail(self) -> &'static str {
        match self {
            Self::Undescribable(d) | Self::TranslationFailed(d) | Self::CompilationFailed(d) => d,
        }
    }
}

/// One pipeline object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pipeline {
    pub id: ResourceId,
    /// The semantic lifetime it was declared in. A pipeline outliving its
    /// generation is not usable by work from a later one, however healthy the
    /// host object is.
    pub generation: SessionGeneration,
    pub state: PipelineState,
    pub refusal: Option<RefusalReason>,
}

/// What a transaction that wants a pipeline observes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lease {
    /// Usable now.
    Ready,
    /// Not yet. The transaction is not ready either, and nothing blocks.
    Pending,
    /// It will never be usable, with the reason.
    Refused(RefusalReason),
    /// There is no such pipeline in this generation.
    Absent,
}

/// Why a transaction can never use a pipeline it binds.
///
/// Both variants are terminal for the work, and they are kept apart because
/// they are different defects: a refused pipeline is this device failing to
/// build what the guest asked for, and an absent one is work naming an object
/// this generation does not have — a use-after-delete, or a packet that
/// outlived a reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseRefusal {
    Refused {
        pipeline: ResourceId,
        reason: RefusalReason,
    },
    Absent {
        pipeline: ResourceId,
    },
}

impl LeaseRefusal {
    /// The name this reaches a failure channel under. A refused pipeline
    /// reports the build's own reason, because "the draw could not run" is not
    /// the fact anyone reading the log needs.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Refused { reason, .. } => reason.slug(),
            Self::Absent { .. } => "pipeline_absent",
        }
    }
}

/// The pipeline objects of one session.
#[derive(Debug, Default)]
pub struct PipelineTable {
    pipelines: HashMap<ResourceId, Pipeline>,
    census: Census,
}

/// What the table has seen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    pub declared: usize,
    pub ready: usize,
    pub refused: usize,
    pub retired: usize,
    /// Leases taken while a pipeline was still being built. The number that
    /// says whether starting compilation at declaration is early enough.
    pub leases_pending: usize,
    /// Leases taken on a pipeline that was already ready. The steady state.
    pub leases_ready: usize,
}

impl PipelineTable {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// Declare a pipeline. Returns whether it was new.
    ///
    /// Re-declaring an id that is live is not an update: the guest's object
    /// namespace produces a new generation for a reused slot, so two live
    /// declarations of one [`ResourceId`] would mean the namespace failed to do
    /// that, and silently replacing the first would hide it.
    pub fn declare(&mut self, id: ResourceId, generation: SessionGeneration) -> bool {
        if self.pipelines.contains_key(&id) {
            return false;
        }
        self.census.declared += 1;
        self.pipelines.insert(
            id,
            Pipeline {
                id,
                generation,
                state: PipelineState::Declared,
                refusal: None,
            },
        );
        true
    }

    #[must_use]
    pub fn get(&self, id: ResourceId) -> Option<&Pipeline> {
        self.pipelines.get(&id)
    }

    /// Advance a pipeline. Returns whether the step was legal and taken.
    ///
    /// An illegal step is refused rather than applied, and a caller that has to
    /// care can ask. The common illegal step is real: a compile that finishes
    /// after the guest deleted the pipeline, which must not resurrect it.
    pub fn advance(&mut self, id: ResourceId, next: PipelineState) -> bool {
        let Some(p) = self.pipelines.get_mut(&id) else {
            return false;
        };
        if !p.state.may_become(next) {
            return false;
        }
        p.state = next;
        match next {
            PipelineState::Ready => self.census.ready += 1,
            PipelineState::Retired => self.census.retired += 1,
            _ => {}
        }
        true
    }

    /// Refuse a pipeline, with the reason.
    pub fn refuse(&mut self, id: ResourceId, reason: RefusalReason) -> bool {
        let Some(p) = self.pipelines.get_mut(&id) else {
            return false;
        };
        if !p.state.may_become(PipelineState::Refused) {
            return false;
        }
        p.state = PipelineState::Refused;
        p.refusal = Some(reason);
        self.census.refused += 1;
        true
    }

    /// What a transaction wanting this pipeline in this generation observes.
    ///
    /// The generation is a parameter rather than read from the pipeline,
    /// because the question is whether *this* work may use it: a pipeline
    /// declared in a closed generation is absent to work from a later one even
    /// though the object is intact.
    pub fn lease(&mut self, id: ResourceId, generation: SessionGeneration) -> Lease {
        let Some(p) = self.pipelines.get(&id) else {
            return Lease::Absent;
        };
        if p.generation != generation || p.state == PipelineState::Retired {
            return Lease::Absent;
        }
        match p.state {
            PipelineState::Ready => {
                self.census.leases_ready += 1;
                Lease::Ready
            }
            PipelineState::Refused => {
                Lease::Refused(p.refusal.expect("a refused pipeline carries its reason"))
            }
            PipelineState::Declared | PipelineState::Translating | PipelineState::Compiling => {
                self.census.leases_pending += 1;
                Lease::Pending
            }
            PipelineState::Retired => Lease::Absent,
        }
    }

    /// Turn a transaction's pipeline leases into the waits it is admitted with.
    ///
    /// **The join between "a transaction leases pipelines" and "a transaction
    /// is held until they are built".** [`crate::exec::ExecWork::pipeline_leases`]
    /// says which pipelines the records bind and
    /// [`crate::session::SessionModel::admit`] takes a list of waits, and
    /// nothing turned one into the other — so the only ways to call `admit`
    /// were with no waits, which runs a draw against a pipeline that is still
    /// compiling, or with a list built somewhere else, which is a second
    /// opinion about what the records bind.
    ///
    /// A lease is taken for every pipeline, in order, which is what the
    /// census counts. Only the pending ones come back: a ready pipeline is
    /// nothing to wait for, and returning it would park the transaction on a
    /// completion that has already happened.
    ///
    /// # Errors
    ///
    /// [`LeaseRefusal`] at the first pipeline this work can never use. The
    /// transaction is refused rather than admitted with a wait that will never
    /// resolve — which is the same choice `admit` makes for every other
    /// unsatisfiable packet, and for the same reason: a completion word the
    /// guest waits on forever is worse than a refusal it is told about.
    ///
    /// Leases already taken stay counted. They happened.
    pub fn waits_for(
        &mut self,
        leases: &[ResourceId],
        generation: SessionGeneration,
    ) -> Result<Vec<ResourceId>, LeaseRefusal> {
        let mut waits = Vec::new();
        for &pipeline in leases {
            match self.lease(pipeline, generation) {
                Lease::Ready => {}
                Lease::Pending => waits.push(pipeline),
                Lease::Refused(reason) => return Err(LeaseRefusal::Refused { pipeline, reason }),
                Lease::Absent => return Err(LeaseRefusal::Absent { pipeline }),
            }
        }
        Ok(waits)
    }

    /// Retire a pipeline the guest deleted.
    pub fn retire(&mut self, id: ResourceId) -> bool {
        self.advance(id, PipelineState::Retired)
    }

    /// Drop retired pipelines' bookkeeping.
    ///
    /// Separate from [`Self::retire`] for the reason retirement and compaction
    /// are separate everywhere else here: one is a lifetime fact and the other
    /// is housekeeping, and doing the second inside the first charges it to the
    /// wrong event.
    pub fn compact(&mut self) {
        self.pipelines
            .retain(|_, p| p.state != PipelineState::Retired);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};

    fn id(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration::default(),
        }
    }
    const GEN: SessionGeneration = SessionGeneration::FIRST;

    #[test]
    fn the_happy_path_is_the_only_forward_path() {
        let mut t = PipelineTable::new();
        assert!(t.declare(id(1), GEN));
        assert_eq!(t.lease(id(1), GEN), Lease::Pending);
        assert!(t.advance(id(1), PipelineState::Translating));
        assert!(t.advance(id(1), PipelineState::Compiling));
        assert!(t.advance(id(1), PipelineState::Ready));
        assert_eq!(t.lease(id(1), GEN), Lease::Ready);
        assert_eq!(t.census().leases_pending, 1);
        assert_eq!(t.census().leases_ready, 1);
    }

    /// Only the pipelines that are still being built come back as waits.
    ///
    /// A ready pipeline is nothing to wait for, and returning it would park the
    /// transaction on a completion that has already happened — a frame that
    /// never arrives with nothing to explain it.
    #[test]
    fn a_ready_pipeline_is_not_something_to_wait_for() {
        let mut t = PipelineTable::new();
        for slot in [1, 2] {
            t.declare(id(slot), GEN);
        }
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            t.advance(id(1), step);
        }
        assert_eq!(t.waits_for(&[id(1), id(2)], GEN), Ok(vec![id(2)]));
        // Every lease was taken, and the census says which kind each was.
        assert_eq!(t.census().leases_ready, 1);
        assert_eq!(t.census().leases_pending, 1);
        assert_eq!(
            t.waits_for(&[], GEN),
            Ok(Vec::new()),
            "nothing bound, nothing held"
        );
    }

    /// A pipeline that will never build refuses the work that binds it, with
    /// the build's own reason.
    ///
    /// Admitting it with a wait that cannot resolve is a completion word the
    /// guest waits on forever, which is worse than a refusal it is told about.
    /// The slug is the compilation's, because "the draw could not run" is not
    /// the fact anyone reading the failure channel needs.
    #[test]
    fn a_pipeline_that_will_never_build_refuses_the_work_that_binds_it() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        t.refuse(id(1), RefusalReason::TranslationFailed("no such stage"));
        let refusal = t
            .waits_for(&[id(1)], GEN)
            .expect_err("a refused pipeline is terminal");
        assert_eq!(
            refusal,
            LeaseRefusal::Refused {
                pipeline: id(1),
                reason: RefusalReason::TranslationFailed("no such stage"),
            }
        );
        assert_eq!(refusal.slug(), "pipeline_translation_failed");
    }

    /// Work naming a pipeline this generation does not have is refused, and is
    /// not the same failure as one that could not be built.
    ///
    /// A pipeline declared in a closed generation is intact and unusable, which
    /// is why the generation is asked about rather than read off the object.
    #[test]
    fn a_pipeline_from_another_generation_is_absent_rather_than_refused() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            t.advance(id(1), step);
        }
        assert_eq!(t.waits_for(&[id(1)], GEN), Ok(Vec::new()));
        assert_eq!(
            t.waits_for(&[id(1)], GEN.next()),
            Err(LeaseRefusal::Absent { pipeline: id(1) })
        );
        assert_eq!(
            t.waits_for(&[id(9)], GEN),
            Err(LeaseRefusal::Absent { pipeline: id(9) }),
            "and one that was never declared at all"
        );
    }

    /// The first unusable pipeline ends the answer, and the leases taken before
    /// it stay counted.
    #[test]
    fn an_unusable_pipeline_stops_the_walk_and_what_happened_stays_counted() {
        let mut t = PipelineTable::new();
        for slot in [1, 2, 3] {
            t.declare(id(slot), GEN);
        }
        t.refuse(id(2), RefusalReason::CompilationFailed("out of registers"));
        assert!(matches!(
            t.waits_for(&[id(1), id(2), id(3)], GEN),
            Err(LeaseRefusal::Refused { pipeline, .. }) if pipeline == id(2)
        ));
        assert_eq!(
            t.census().leases_pending,
            1,
            "the pipeline before it was leased; the one after it was not"
        );
    }

    /// Skipping a step is not a shortcut. A pipeline that reached `Ready`
    /// without compiling is one whose host object nobody built.
    #[test]
    fn a_pipeline_cannot_skip_to_ready() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        assert!(!t.advance(id(1), PipelineState::Ready));
        assert!(!t.advance(id(1), PipelineState::Compiling));
        assert_eq!(t.get(id(1)).unwrap().state, PipelineState::Declared);
    }

    /// The step that actually happens: a compile finishing after the guest
    /// deleted the pipeline. It must not resurrect it.
    #[test]
    fn a_compile_that_lands_after_deletion_does_not_resurrect() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        t.advance(id(1), PipelineState::Translating);
        t.advance(id(1), PipelineState::Compiling);
        assert!(t.retire(id(1)));
        assert!(
            !t.advance(id(1), PipelineState::Ready),
            "the host finished building an object the guest no longer has"
        );
        assert_eq!(t.lease(id(1), GEN), Lease::Absent);
    }

    /// A refusal is terminal and carries its reason to whoever reads it, so a
    /// guest re-binding the pipeline every frame produces one refusal rather
    /// than one per frame.
    #[test]
    fn a_refusal_is_terminal_and_keeps_its_reason() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        assert!(t.refuse(id(1), RefusalReason::TranslationFailed("no_air")));
        assert_eq!(
            t.lease(id(1), GEN),
            Lease::Refused(RefusalReason::TranslationFailed("no_air"))
        );
        assert_eq!(
            t.lease(id(1), GEN),
            Lease::Refused(RefusalReason::TranslationFailed("no_air"))
        );
        assert!(
            !t.advance(id(1), PipelineState::Translating),
            "a refused pipeline does not retry"
        );
        assert_eq!(t.census().refused, 1);
        assert_eq!(
            t.census().leases_pending,
            0,
            "a refusal is not a pending lease; counting it as one would make \
             the number that argues for earlier compilation include work that \
             will never compile"
        );
    }

    /// A descriptor the model cannot represent is refused before any work
    /// starts, which is the whole reason `Declared -> Refused` is legal.
    #[test]
    fn a_pipeline_can_be_refused_before_anything_starts() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        assert!(t.refuse(id(1), RefusalReason::Undescribable("tessellation")));
        assert_eq!(t.get(id(1)).unwrap().state, PipelineState::Refused);
    }

    /// A pipeline from a closed generation is absent to later work even though
    /// the object is intact — the host handle may be perfectly healthy, and
    /// that is not the question.
    #[test]
    fn a_pipeline_from_a_closed_generation_is_absent_to_later_work() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        t.advance(id(1), PipelineState::Translating);
        t.advance(id(1), PipelineState::Compiling);
        t.advance(id(1), PipelineState::Ready);
        assert_eq!(t.lease(id(1), GEN), Lease::Ready);
        assert_eq!(t.lease(id(1), GEN.next()), Lease::Absent);
    }

    /// Two live declarations of one id would mean the object namespace failed
    /// to produce a new generation for a reused slot. Replacing the first
    /// silently would hide that.
    #[test]
    fn redeclaring_a_live_id_is_refused_rather_than_applied() {
        let mut t = PipelineTable::new();
        assert!(t.declare(id(1), GEN));
        t.advance(id(1), PipelineState::Translating);
        assert!(!t.declare(id(1), GEN));
        assert_eq!(t.get(id(1)).unwrap().state, PipelineState::Translating);
        // A reused slot is a different id, and declares cleanly.
        let reused = ResourceId {
            slot: ObjectListRef(1),
            generation: SlotGeneration::default().next(),
        };
        assert!(t.declare(reused, GEN));
    }

    #[test]
    fn compaction_drops_only_retired_pipelines() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        t.declare(id(2), GEN);
        t.retire(id(1));
        t.compact();
        assert_eq!(t.len(), 1);
        assert!(t.get(id(1)).is_none());
        assert!(t.get(id(2)).is_some());
    }

    #[test]
    fn an_absent_pipeline_answers_absent_rather_than_pending() {
        let mut t = PipelineTable::new();
        assert_eq!(t.lease(id(9), GEN), Lease::Absent);
        assert!(!t.advance(id(9), PipelineState::Translating));
        assert!(!t.refuse(id(9), RefusalReason::CompilationFailed("x")));
    }
}
