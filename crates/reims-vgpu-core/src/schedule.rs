//! A parallel schedule the dependency graph permits, and what it has to mean.
//!
//! # The claim this module exists to fail on
//!
//! The replacement's entire risk is that running transactions concurrently
//! stops meaning what running them one at a time meant. [`crate::interpret`]
//! defines the second thing. This defines the first: it drives
//! [`crate::depend::DependencyGraph`] and [`crate::ready::Scheduler`] over a
//! batch and completes transactions in an order those two permit — an
//! arbitrary one, chosen by a seed, so a test can sweep the space rather than
//! check one lucky interleaving.
//!
//! Then [`equivalent`] says whether the two agree, and the point of writing
//! that relation down is that it is *not* "the traces are equal". Getting it
//! wrong in the strict direction makes the test fail for reorderings a device
//! is allowed to do; getting it wrong in the loose direction makes it pass for
//! ones it is not.
//!
//! # Completion order is the whole schedule
//!
//! A transaction's effects become visible at completion and never before, so a
//! model that applies effects atomically at completion loses nothing by not
//! representing "started but not finished". Every interleaving of overlapping
//! execution is observationally some completion order, and every completion
//! order this model can produce is a linear extension of the dependency
//! relation. So the space this sweeps is the space.
//!
//! # What may differ, and why
//!
//! A backing's content versions are compared **in order**. Two transactions
//! that both make a backing's content current are ordered by their hazard edge,
//! so their versions have one legal sequence and a device that produced the
//! other sequence would be showing the guest stale bytes under a fresh
//! version.
//!
//! A completion stamp and an event generation are compared by their **final
//! value**, and every publication must advance. These are monotone points that
//! independent work races to on purpose: two packets that touch no common
//! memory carry no ordering, and the guest that submitted them asked for none.
//! Requiring their publications in a fixed order would be requiring the device
//! to serialise work the guest deliberately left independent.
//!
//! A transaction's own observations must be **contiguous** in the trace, and
//! its versions must precede its stamp. Publication is per-transaction and
//! atomic: a guest that polled the stamp and then read the content must not be
//! able to see the flag without the bytes, and interleaving another
//! transaction's publication inside one is the same failure wearing a
//! different shape.
//!
//! # The batch has to be one a serial run could execute
//!
//! Ingress order is only a legal schedule when every explicit wait names a
//! producer that already ran. A batch where a packet waits for a later
//! packet's stamp has no serial meaning to compare against — the serial
//! interpreter refuses it as unmeetable, which is correct and is not a
//! divergence. [`eligible`] names that condition, and three others, rather
//! than letting a generator quietly produce batches the comparison cannot
//! speak about.

use crate::access::{BackingId, ContentVersion};
use crate::depend::{Census, DependencyGraph};
use crate::exec::{ExecTransaction, Prerequisite};
use crate::identity::{
    IngressOrdinal, ResourceId, SessionGeneration, StampSlot, StampValue, StampWait,
};
use crate::interpret::{Interpreter, Observation, Outcome, Refusal};
use crate::prereq::{WaitGraph, WaitPoint};
use crate::ready::Scheduler;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

/// Why a batch has no serial meaning to compare a parallel run against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ineligible {
    /// Transactions were not handed over in ingress order, which is the order
    /// the hazard compiler is admitted in.
    OutOfIngressOrder {
        at: IngressOrdinal,
        after: IngressOrdinal,
    },
    /// A packet waits for a point produced by a packet that arrives no earlier
    /// than it does. Legal for a guest to submit and legal for the device to
    /// run; simply not something ingress order executes, so the serial
    /// interpreter refuses it and there is nothing to compare.
    ForwardExplicitWait {
        waiter: IngressOrdinal,
        point: WaitPoint,
        producer: IngressOrdinal,
    },
    /// A packet waits for a point nothing in the batch produces.
    UnansweredWait {
        waiter: IngressOrdinal,
        point: WaitPoint,
    },
    /// A packet carries an encoder-scoped fence prerequisite. The serial
    /// interpreter answers those from whether any packet has ever updated the
    /// fence, which is order-sensitive state neither graph carries; a batch
    /// that uses one is outside what this comparison can speak about.
    FencePrerequisite { waiter: IngressOrdinal },
    /// Two packets make the same backing's content current without a hazard
    /// edge between them, so their version sequence has no single legal order.
    ///
    /// This is a real gap and it is named rather than tolerated: a version
    /// reservation names a whole backing while an access may name a range, so
    /// two disjoint-range writers can both claim to produce the backing's next
    /// version. Region-level version coverage is what closes it, and until
    /// then such a batch is not one this comparison can judge.
    UnorderedVersionRace {
        backing: BackingId,
        first: IngressOrdinal,
        second: IngressOrdinal,
    },
    /// The batch spans more than one semantic generation, so a serial run
    /// would refuse part of it for reasons that are not about scheduling.
    MixedGeneration {
        expected: SessionGeneration,
        found: SessionGeneration,
    },
}

impl Ineligible {
    #[must_use]
    pub const fn slug(&self) -> &'static str {
        match self {
            Self::OutOfIngressOrder { .. } => "schedule_out_of_ingress_order",
            Self::ForwardExplicitWait { .. } => "schedule_forward_explicit_wait",
            Self::UnansweredWait { .. } => "schedule_unanswered_wait",
            Self::FencePrerequisite { .. } => "schedule_fence_prerequisite",
            Self::UnorderedVersionRace { .. } => "schedule_unordered_version_race",
            Self::MixedGeneration { .. } => "schedule_mixed_generation",
        }
    }
}

/// Whether ingress order is a legal schedule for this batch, so that a serial
/// run and a parallel one are comparable at all.
///
/// # Errors
///
/// The first condition the batch fails, in the order the variants are
/// declared: shape before content, so a batch that is out of ingress order is
/// reported as that rather than as whatever the misordering made of its waits.
pub fn eligible(batch: &[ExecTransaction]) -> Result<(), Ineligible> {
    for pair in batch.windows(2) {
        if pair[1].ingress <= pair[0].ingress {
            return Err(Ineligible::OutOfIngressOrder {
                at: pair[1].ingress,
                after: pair[0].ingress,
            });
        }
    }
    if let Some(first) = batch.first() {
        for tx in batch {
            if tx.session != first.session {
                return Err(Ineligible::MixedGeneration {
                    expected: first.session,
                    found: tx.session,
                });
            }
        }
    }
    for tx in batch {
        if let Some(Prerequisite::Fence { .. }) = tx
            .prerequisites
            .iter()
            .find(|p| matches!(p, Prerequisite::Fence { .. }))
        {
            return Err(Ineligible::FencePrerequisite { waiter: tx.ingress });
        }
    }

    let mut waits = WaitGraph::new();
    for tx in batch {
        waits.admit(tx);
    }
    for diagnosis in waits.diagnose() {
        if let crate::prereq::Diagnosis::Unproduced { waiter, point } = diagnosis {
            return Err(Ineligible::UnansweredWait { waiter, point });
        }
    }
    for (waiter, point, producer) in earliest_producers(&waits) {
        if producer >= waiter {
            return Err(Ineligible::ForwardExplicitWait {
                waiter,
                point,
                producer,
            });
        }
    }

    version_races(batch)?;
    Ok(())
}

/// For each unmet wait, the earliest transaction that discharges it.
///
/// A wait needs *one* producer that ran before it, not every producer to have.
/// A later packet that also discharges the point is ordinary — an event is
/// signalled repeatedly and every signal past the waited-for value discharges
/// it — so the earliest producer is both what decides eligibility and the only
/// prerequisite worth carrying: depending on the rest would order work the
/// guest left independent.
fn earliest_producers(waits: &WaitGraph) -> Vec<(IngressOrdinal, WaitPoint, IngressOrdinal)> {
    let mut out: Vec<(IngressOrdinal, WaitPoint, IngressOrdinal)> = Vec::new();
    for (waiter, point, producer) in waits.edges() {
        match out.iter_mut().find(|(w, p, _)| *w == waiter && *p == point) {
            Some(slot) => slot.2 = slot.2.min(producer),
            None => out.push((waiter, point, producer)),
        }
    }
    out
}

/// The one eligibility question that needs the hazard compiler to answer it.
fn version_races(batch: &[ExecTransaction]) -> Result<(), Ineligible> {
    let mut graph = DependencyGraph::new();
    let mut publishers: HashMap<BackingId, Vec<IngressOrdinal>> = HashMap::new();
    let mut ordered: HashMap<IngressOrdinal, Vec<IngressOrdinal>> = HashMap::new();
    for tx in batch {
        ordered.insert(tx.ingress, graph.admit(tx.ingress, &tx.accesses));
        for reservation in &tx.publication.versions {
            publishers
                .entry(reservation.backing)
                .or_default()
                .push(tx.ingress);
        }
    }
    // Reachability over hazard edges, which point backwards, so one pass in
    // ingress order settles it.
    let mut reaches: HashMap<IngressOrdinal, std::collections::BTreeSet<IngressOrdinal>> =
        HashMap::new();
    for tx in batch {
        let mut set = std::collections::BTreeSet::new();
        for earlier in &ordered[&tx.ingress] {
            set.insert(*earlier);
            if let Some(theirs) = reaches.get(earlier) {
                set.extend(theirs.iter().copied());
            }
        }
        reaches.insert(tx.ingress, set);
    }
    let mut backings: Vec<_> = publishers.keys().copied().collect();
    backings.sort_unstable();
    for backing in backings {
        let list = &publishers[&backing];
        for pair in list.windows(2) {
            let (first, second) = (pair[0], pair[1]);
            if !reaches[&second].contains(&first) {
                return Err(Ineligible::UnorderedVersionRace {
                    backing,
                    first,
                    second,
                });
            }
        }
    }
    Ok(())
}

/// One execution of a batch.
#[derive(Clone, Debug, Default)]
pub struct Run {
    /// Every observation, in the order it became visible.
    pub trace: Vec<Observation>,
    /// Completion order, and where each transaction's observations sit.
    pub spans: Vec<(IngressOrdinal, Range<usize>)>,
    /// Transactions the scheduler never released. Empty for an eligible batch;
    /// a non-empty one is a defect in the readiness service, not in the batch.
    pub stalled: Vec<IngressOrdinal>,
    /// What compiling this batch's hazards cost.
    pub census: Census,
}

impl Run {
    /// Completion order.
    #[must_use]
    pub fn order(&self) -> Vec<IngressOrdinal> {
        self.spans.iter().map(|(o, _)| *o).collect()
    }
}

/// Run the batch one transaction at a time in ingress order.
///
/// The reference. No scheduler, no readiness, nothing to be concurrent about.
#[must_use]
pub fn serial(batch: &[ExecTransaction]) -> Run {
    let mut interpreter = Interpreter::new();
    let mut run = Run::default();
    for tx in batch {
        let start = interpreter.trace().len();
        let outcome = interpreter.run(tx);
        run.spans
            .push((tx.ingress, start..interpreter.trace().len()));
        debug_assert!(
            matches!(outcome, Outcome::Ran)
                || matches!(outcome, Outcome::Refused(Refusal::StaleGeneration)),
            "an eligible batch has no unmeetable wait in ingress order"
        );
    }
    run.trace = interpreter.trace().to_vec();
    run
}

/// Run the batch through the dependency graph and readiness service,
/// completing whichever ready transaction `seed` picks.
///
/// Both graphs are driven exactly as production would drive them: hazards from
/// [`DependencyGraph::admit`], explicit event waits as ordinal prerequisites,
/// stamp waits through the readiness service's own stamp machinery, and
/// retirement on completion. What the seed chooses is only which of the
/// transactions the two of them have *already declared ready* goes next.
#[must_use]
pub fn parallel(batch: &[ExecTransaction], seed: u64) -> Run {
    let mut rng = Rng::new(seed);
    parallel_with(batch, move |ready| rng.below(ready.len()))
}

/// The same, with the choice of which ready transaction goes next handed to
/// the caller.
///
/// `pick` receives the transactions the dependency graph and the readiness
/// service have *already declared ready* and returns an index into them. It
/// cannot make an illegal schedule, only choose among the legal ones — which
/// is what makes "always take the lowest ordinal" a way to ask whether ingress
/// order is still reachable.
#[must_use]
pub fn parallel_with(
    batch: &[ExecTransaction],
    mut pick: impl FnMut(&[IngressOrdinal]) -> usize,
) -> Run {
    let by_ordinal: HashMap<IngressOrdinal, &ExecTransaction> =
        batch.iter().map(|tx| (tx.ingress, tx)).collect();

    // Explicit event waits become ordinal prerequisites. Eligibility has
    // already established that every producer is earlier, so this cannot
    // introduce a forward edge.
    let mut waits = WaitGraph::new();
    for tx in batch {
        waits.admit(tx);
    }
    let mut explicit: HashMap<IngressOrdinal, Vec<IngressOrdinal>> = HashMap::new();
    for (waiter, point, producer) in earliest_producers(&waits) {
        if matches!(point, WaitPoint::Event { .. }) {
            explicit.entry(waiter).or_default().push(producer);
        }
    }

    let mut graph = DependencyGraph::new();
    let mut scheduler = Scheduler::new();
    let mut pool: Vec<IngressOrdinal> = Vec::new();
    for tx in batch {
        let mut prerequisites = graph.admit(tx.ingress, &tx.accesses);
        prerequisites.extend(explicit.remove(&tx.ingress).unwrap_or_default());
        prerequisites.sort_unstable();
        prerequisites.dedup();
        let stamp_waits: Vec<StampWait> = tx
            .prerequisites
            .iter()
            .filter_map(|p| match p {
                Prerequisite::Stamp(w) => Some(*w),
                Prerequisite::Event { .. } | Prerequisite::Fence { .. } => None,
            })
            .collect();
        scheduler.admit(
            tx.ingress,
            &prerequisites,
            &stamp_waits,
            tx.publication.stamp,
        );
    }

    let mut interpreter = Interpreter::new();
    let mut run = Run::default();
    loop {
        pool.extend(scheduler.take_ready());
        if pool.is_empty() {
            break;
        }
        pool.sort_unstable();
        let ordinal = pool.remove(pick(&pool));
        let tx = by_ordinal[&ordinal];
        let start = interpreter.trace().len();
        let outcome = interpreter.run(tx);
        debug_assert!(
            matches!(outcome, Outcome::Ran),
            "the readiness service released {ordinal:?} and the interpreter refused it"
        );
        run.spans.push((ordinal, start..interpreter.trace().len()));
        scheduler.complete(ordinal);
        graph.retire(ordinal);
    }
    run.stalled = scheduler.stalled();
    run.trace = interpreter.trace().to_vec();
    run.census = graph.census();
    run
}

/// A way the two runs meant different things.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Divergence {
    /// A transaction ran in one and not the other.
    DifferentTransactions {
        serial: Vec<IngressOrdinal>,
        parallel: Vec<IngressOrdinal>,
    },
    /// A backing's content versions were published in a different order, or a
    /// different set of them was published.
    ContentHistory {
        backing: BackingId,
        serial: Vec<ContentVersion>,
        parallel: Vec<ContentVersion>,
    },
    /// A completion word came to rest at a different value.
    StampOutcome {
        slot: StampSlot,
        serial: Option<StampValue>,
        parallel: Option<StampValue>,
    },
    /// An event generation came to rest at a different value.
    EventOutcome {
        event: ResourceId,
        serial: u64,
        parallel: u64,
    },
    /// A monotone point was published a value that did not advance it.
    NonMonotonePublication { at: usize },
    /// A fence was updated a different number of times.
    FenceUpdates {
        fence: ResourceId,
        serial: usize,
        parallel: usize,
    },
    /// A transaction was refused in one run and not the other.
    Refusals {
        serial: Vec<(IngressOrdinal, Refusal)>,
        parallel: Vec<(IngressOrdinal, Refusal)>,
    },
    /// One transaction's publications were interrupted by another's.
    SplitPublication { ordinal: IngressOrdinal },
    /// A transaction published its completion stamp before its content
    /// versions.
    StampBeforeVersions { ordinal: IngressOrdinal },
}

impl Divergence {
    #[must_use]
    pub const fn slug(&self) -> &'static str {
        match self {
            Self::DifferentTransactions { .. } => "diverge_different_transactions",
            Self::ContentHistory { .. } => "diverge_content_history",
            Self::StampOutcome { .. } => "diverge_stamp_outcome",
            Self::EventOutcome { .. } => "diverge_event_outcome",
            Self::NonMonotonePublication { .. } => "diverge_non_monotone_publication",
            Self::FenceUpdates { .. } => "diverge_fence_updates",
            Self::Refusals { .. } => "diverge_refusals",
            Self::SplitPublication { .. } => "diverge_split_publication",
            Self::StampBeforeVersions { .. } => "diverge_stamp_before_versions",
        }
    }
}

/// Whether a parallel run means the same thing the serial one did.
///
/// See the module documentation for which parts are compared in order and
/// which by outcome, and why each is which.
///
/// # Errors
///
/// The first difference found, in a fixed order so the answer does not depend
/// on hash iteration.
pub fn equivalent(serial: &Run, parallel: &Run) -> Result<(), Divergence> {
    let mut mine = serial.order();
    let mut theirs = parallel.order();
    mine.sort_unstable();
    theirs.sort_unstable();
    if mine != theirs {
        return Err(Divergence::DifferentTransactions {
            serial: mine,
            parallel: theirs,
        });
    }

    let left = Summary::of(&serial.trace);
    let right = Summary::of(&parallel.trace);
    for (backing, versions) in &left.content {
        let theirs = right.content.get(backing).cloned().unwrap_or_default();
        if *versions != theirs {
            return Err(Divergence::ContentHistory {
                backing: *backing,
                serial: versions.clone(),
                parallel: theirs,
            });
        }
    }
    for backing in right.content.keys() {
        if !left.content.contains_key(backing) {
            return Err(Divergence::ContentHistory {
                backing: *backing,
                serial: Vec::new(),
                parallel: right.content[backing].clone(),
            });
        }
    }
    for slot in left.stamps.keys().chain(right.stamps.keys()) {
        let (a, b) = (
            left.stamps.get(slot).copied(),
            right.stamps.get(slot).copied(),
        );
        if a != b {
            return Err(Divergence::StampOutcome {
                slot: *slot,
                serial: a,
                parallel: b,
            });
        }
    }
    for event in left.events.keys().chain(right.events.keys()) {
        let (a, b) = (
            left.events.get(event).copied().unwrap_or(0),
            right.events.get(event).copied().unwrap_or(0),
        );
        if a != b {
            return Err(Divergence::EventOutcome {
                event: *event,
                serial: a,
                parallel: b,
            });
        }
    }
    for fence in left.fences.keys().chain(right.fences.keys()) {
        let (a, b) = (
            left.fences.get(fence).copied().unwrap_or(0),
            right.fences.get(fence).copied().unwrap_or(0),
        );
        if a != b {
            return Err(Divergence::FenceUpdates {
                fence: *fence,
                serial: a,
                parallel: b,
            });
        }
    }
    let (mut refused_left, mut refused_right) = (left.refusals.clone(), right.refusals.clone());
    refused_left.sort_by_key(|(o, _)| *o);
    refused_right.sort_by_key(|(o, _)| *o);
    if refused_left != refused_right {
        return Err(Divergence::Refusals {
            serial: refused_left,
            parallel: refused_right,
        });
    }

    monotone(parallel)?;
    atomic_publication(parallel)
}

/// Every publication into a monotone point advances it.
fn monotone(run: &Run) -> Result<(), Divergence> {
    let mut stamps: HashMap<StampSlot, StampValue> = HashMap::new();
    let mut events: HashMap<ResourceId, u64> = HashMap::new();
    for (at, observation) in run.trace.iter().enumerate() {
        match *observation {
            Observation::StampPublished { slot, value } => {
                if stamps.get(&slot).is_some_and(|at| !value.follows(*at)) {
                    return Err(Divergence::NonMonotonePublication { at });
                }
                stamps.insert(slot, value);
            }
            Observation::EventAdvanced { event, to } => {
                if events.get(&event).is_some_and(|at| to <= *at) {
                    return Err(Divergence::NonMonotonePublication { at });
                }
                events.insert(event, to);
            }
            Observation::VersionPublished { .. }
            | Observation::FenceUpdated { .. }
            | Observation::Refused { .. } => {}
        }
    }
    Ok(())
}

/// A transaction's observations are contiguous and its versions precede its
/// stamp.
fn atomic_publication(run: &Run) -> Result<(), Divergence> {
    let mut covered = 0usize;
    let mut spans: Vec<_> = run.spans.clone();
    spans.sort_by_key(|(_, r)| r.start);
    for (ordinal, span) in spans {
        if span.start != covered {
            return Err(Divergence::SplitPublication { ordinal });
        }
        covered = span.end;
        let mut seen_stamp = false;
        for observation in &run.trace[span] {
            match observation {
                Observation::StampPublished { .. } => seen_stamp = true,
                Observation::VersionPublished { .. } if seen_stamp => {
                    return Err(Divergence::StampBeforeVersions { ordinal });
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// What a trace came to, per observable location.
#[derive(Debug, Default)]
struct Summary {
    content: BTreeMap<BackingId, Vec<ContentVersion>>,
    stamps: BTreeMap<StampSlot, StampValue>,
    events: BTreeMap<ResourceId, u64>,
    fences: BTreeMap<ResourceId, usize>,
    refusals: Vec<(IngressOrdinal, Refusal)>,
}

impl Summary {
    fn of(trace: &[Observation]) -> Self {
        let mut out = Self::default();
        for observation in trace {
            match *observation {
                Observation::VersionPublished { backing, version } => {
                    out.content.entry(backing).or_default().push(version);
                }
                // Where the point came to rest, which on a monotone location
                // is the furthest value published into it and not the last
                // one written. Taking the last would make a trace that
                // regressed a slot compare equal to one that did not, and
                // that regression is exactly what [`monotone`] exists to
                // catch.
                Observation::StampPublished { slot, value } => {
                    let at = out.stamps.entry(slot).or_insert(value);
                    *at = at.later(value);
                }
                Observation::EventAdvanced { event, to } => {
                    let at = out.events.entry(event).or_insert(to);
                    *at = (*at).max(to);
                }
                Observation::FenceUpdated { fence } => {
                    *out.fences.entry(fence).or_insert(0) += 1;
                }
                Observation::Refused { ingress, reason } => out.refusals.push((ingress, reason)),
            }
        }
        out
    }
}

/// A reproducible choice of interleaving.
///
/// Deliberately not a general random source: what a schedule sweep needs is
/// that seed *n* names the same interleaving on every machine and every run,
/// so a failure is a bug report rather than a rumour.
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        // Any non-zero state; a zero seed is a legitimate thing for a caller
        // to pass and xorshift is stuck there.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        usize::try_from(self.next() % bound as u64).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests;
