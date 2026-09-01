//! Seam 2's exit: serial and parallel schedules mean the same thing.
//!
//! The sweep below is only worth anything if the schedules it produces
//! actually differ, so that is asserted first and separately. A property test
//! whose generator happens to produce one order is a test that passes for the
//! wrong reason, and it passes just as happily after the property breaks.

use super::*;
use crate::access::{AccessIntent, AccessKey, AccessMode, ByteRange, ResourceKey, StubRegistry};
use crate::exec::{ExecBuilder, ResolvedOperation};
use crate::identity::{ChannelId, ChannelSequence, CompletionStamp, ObjectListRef, SlotGeneration};
use crate::prereq::Diagnosis;
use crate::stream::SegmentKind;
use crate::sync::{EventKind, EventOp, FenceKind, FenceOp};

fn res(slot: u32) -> ResourceId {
    ResourceId {
        slot: ObjectListRef(slot),
        generation: SlotGeneration(1),
    }
}

fn builder(domain: u32, ingress: u64) -> ExecBuilder {
    ExecBuilder::new(
        SessionGeneration::FIRST,
        ChannelId(domain),
        ChannelSequence(ingress),
        IngressOrdinal(ingress),
    )
}

/// A whole-backing access. Writes are always whole here, which is what keeps
/// two publishers of one backing hazard-ordered — see
/// [`Ineligible::UnorderedVersionRace`].
fn whole(domain: u32, backing: u64, mode: AccessMode) -> AccessIntent {
    AccessIntent {
        domain: ChannelId(domain),
        key: AccessKey::Whole(ResourceKey {
            backing: BackingId(backing),
            heap: None,
        }),
        mode,
        api_stages: 0,
        input_content_version: None,
        output_content_version: None,
    }
}

/// A whole-backing write that also claims the region's next content version.
fn produces(domain: u32, backing: u64, to: u64) -> AccessIntent {
    AccessIntent {
        output_content_version: Some(ContentVersion(to)),
        ..whole(domain, backing, AccessMode::Write)
    }
}

fn ranged(domain: u32, backing: u64, offset: u64) -> AccessIntent {
    AccessIntent {
        domain: ChannelId(domain),
        key: AccessKey::Range(
            ResourceKey {
                backing: BackingId(backing),
                heap: None,
            },
            ByteRange { offset, length: 64 },
        ),
        mode: AccessMode::Read,
        api_stages: 0,
        input_content_version: None,
        output_content_version: None,
    }
}

// ---------------------------------------------------------------- workloads

/// A batch that is a straight hazard chain: every transaction writes the same
/// backing, so the dependency graph totally orders them.
fn chain(length: u64) -> Vec<ExecTransaction> {
    (1..=length)
        .map(|n| {
            let mut b = builder(1, n);
            b.declare_access(produces(1, 1, n));
            b.publish_stamp(CompletionStamp {
                slot: StampSlot(1),
                value: StampValue(u32::try_from(n).expect("small")),
            });
            b.finish().expect("frozen")
        })
        .collect()
}

/// A batch of transactions that touch nothing in common, so the dependency
/// graph orders none of them.
fn independent(count: u64) -> Vec<ExecTransaction> {
    (1..=count)
        .map(|n| {
            let mut b = builder(1, n);
            b.declare_access(produces(1, n, 1));
            b.finish().expect("frozen")
        })
        .collect()
}

/// A mixed workload: several domains, shared and private backings, reads and
/// writes, event signals answered by later waits, fence updates, and a
/// completion stamp per domain.
///
/// Backings are partitioned by domain because [`crate::access::requires_edge`]
/// refuses to order accesses in different domains — which is correct, and
/// which means a version reservation on a backing two domains write would have
/// no legal order at all.
fn mixed(seed: u64, count: u64) -> Vec<ExecTransaction> {
    const DOMAINS: u64 = 3;
    let mut rng = Rng::new(seed ^ 0xC0FF_EE00);
    let mut batch = Vec::new();
    // Highest value signalled into each event so far, so a wait can only name
    // a point an earlier transaction already produced.
    let mut signalled: Vec<(ResourceId, u64)> = Vec::new();
    // Next content version per backing, and the next stamp value per domain.
    let mut version = [0u64; 12];
    let mut stamp = [0u32; DOMAINS as usize];

    for n in 1..=count {
        let domain = u32::try_from(n % DOMAINS).expect("small") + 1;
        let mut b = builder(domain, n);

        // Zero to two reads of backings this domain owns.
        for _ in 0..(rng.next() % 3) {
            let backing = (u64::from(domain) * 4) + (rng.next() % 4);
            b.declare_access(ranged(domain, backing, rng.next() % 4 * 64));
        }
        // One write, sometimes, claiming the region's next content version.
        if !rng.next().is_multiple_of(3) {
            let backing = (u64::from(domain) * 4) + (rng.next() % 4);
            let slot = usize::try_from(backing).expect("small") % version.len();
            version[slot] += 1;
            b.declare_access(produces(domain, backing, version[slot]));
        }
        // A wait for a point some earlier transaction has already signalled.
        if !signalled.is_empty() && rng.next().is_multiple_of(3) {
            let (event, value) = signalled[rng.below(signalled.len())];
            b.require(Prerequisite::Event { event, value });
        }
        // A wait for a stamp value some earlier transaction already owes.
        if rng.next().is_multiple_of(4) {
            let owed = stamp[usize::try_from(domain).expect("small") - 1];
            if owed > 0 {
                b.require(Prerequisite::Stamp(StampWait {
                    slot: StampSlot(domain),
                    value: StampValue(owed),
                }));
            }
        }
        // Records: an event signal, or a fence update on a blit encoder.
        match rng.next() % 3 {
            0 => {
                let event = res(u32::try_from(rng.next() % 3).expect("small") + 20);
                let at = signalled
                    .iter()
                    .filter(|(e, _)| *e == event)
                    .map(|(_, v)| *v)
                    .max()
                    .unwrap_or(0);
                b.begin_segment(SegmentKind::Event.wire_type(), false)
                    .expect("event encoder opens");
                b.record(
                    ResolvedOperation::Event(EventOp {
                        kind: EventKind::Signal,
                        event,
                        value: at + 1,
                    }),
                    &mut StubRegistry(ChannelId(domain)),
                )
                .expect("a signal records");
                b.end_segment().expect("event encoder closes");
                signalled.push((event, at + 1));
            }
            1 => {
                b.begin_segment(SegmentKind::Blit.wire_type(), false)
                    .expect("blit encoder opens");
                b.record(
                    ResolvedOperation::Fence(FenceOp {
                        kind: FenceKind::Update,
                        fence: res(u32::try_from(rng.next() % 2).expect("small") + 30),
                        stages: None,
                    }),
                    &mut StubRegistry(ChannelId(domain)),
                )
                .expect("a fence update records");
                b.end_segment().expect("blit encoder closes");
            }
            _ => {}
        }
        // A completion stamp, on the domain's own slot, advancing.
        let slot = usize::try_from(domain).expect("small") - 1;
        stamp[slot] += 1;
        b.publish_stamp(CompletionStamp {
            slot: StampSlot(domain),
            value: StampValue(stamp[slot]),
        });
        batch.push(b.finish().expect("frozen"));
    }
    batch
}

// ------------------------------------------------------------- the sweep

/// The sweep is only meaningful if the seeds actually reach different
/// schedules, so that is its own assertion rather than a hope.
#[test]
fn independent_transactions_reach_many_completion_orders() {
    let batch = independent(6);
    eligible(&batch).expect("independent work is eligible");
    let orders: std::collections::BTreeSet<Vec<IngressOrdinal>> = (0..64u64)
        .map(|seed| parallel(&batch, seed).order())
        .collect();
    assert!(
        orders.len() > 8,
        "the seeds reached {} distinct orders of six independent transactions; \
         a sweep that only finds one is not sweeping",
        orders.len()
    );
}

/// And a chain must reach exactly one, or the dependency graph is not ordering
/// what it claims to.
#[test]
fn a_hazard_chain_reaches_exactly_one_order_and_an_identical_trace() {
    let batch = chain(6);
    eligible(&batch).expect("a chain is eligible");
    let reference = serial(&batch);
    for seed in 0..64u64 {
        let run = parallel(&batch, seed);
        assert_eq!(
            run.order(),
            (1..=6).map(IngressOrdinal).collect::<Vec<_>>(),
            "seed {seed} reordered a hazard chain"
        );
        assert_eq!(
            run.trace, reference.trace,
            "a totally ordered batch has one trace, not an equivalent one"
        );
    }
}

/// Seam 2's exit.
#[test]
fn every_permitted_schedule_means_what_the_serial_one_meant() {
    let mut reordered = 0usize;
    for workload in 0..24u64 {
        let batch = mixed(workload, 14);
        eligible(&batch).unwrap_or_else(|e| panic!("workload {workload} is ineligible: {e:?}"));
        let reference = serial(&batch);
        let mut orders = std::collections::BTreeSet::new();
        for seed in 0..24u64 {
            let run = parallel(&batch, seed);
            assert!(
                run.stalled.is_empty(),
                "workload {workload} seed {seed} stalled at {:?}",
                run.stalled
            );
            assert_eq!(
                run.order().len(),
                batch.len(),
                "workload {workload} seed {seed} left work unrun"
            );
            equivalent(&reference, &run).unwrap_or_else(|d| {
                panic!("workload {workload} seed {seed} diverged: {d:?}");
            });
            orders.insert(run.order());
        }
        if orders.len() > 1 {
            reordered += 1;
        }
        assert_eq!(
            parallel_with(&batch, |_| 0).order(),
            reference.order(),
            "workload {workload}: taking the lowest ready ordinal every time \
             must reproduce ingress order, or the readiness service is \
             withholding a transaction the serial run could execute"
        );
    }
    assert!(
        reordered >= 20,
        "only {reordered} of 24 workloads were reordered at all; a sweep over \
         schedules that are all the same schedule proves nothing"
    );
}

/// The claim ordered publication adds to the exit: however the work finishes,
/// each channel tells the guest about it in channel order.
#[test]
fn a_channel_publishes_in_its_own_order_however_the_schedule_runs() {
    let mut ever_held = false;
    for workload in 0..24u64 {
        let batch = mixed(workload, 14);
        let reference = serial(&batch);
        for seed in 0..24u64 {
            let run = parallel(&batch, seed);
            for domain in reference.domains() {
                assert_eq!(
                    run.published_by(domain),
                    reference.published_by(domain),
                    "workload {workload} seed {seed} published channel {domain:?} differently"
                );
            }
            ever_held |= run.blocked.iter().any(|(_, held)| *held > 0);
        }
    }
    assert!(
        ever_held,
        "no schedule ever finished work ahead of its channel's head, so the \
         FIFO was never asked to hold anything and this proves nothing"
    );
}

/// And a schedule that finishes in channel order costs the FIFO nothing.
#[test]
fn a_hazard_chain_never_holds_a_position() {
    let run = parallel_with(&chain(6), |_| 0);
    assert!(run.blocked.is_empty());
}

#[test]
fn the_equivalence_relation_rejects_a_channel_that_published_out_of_order() {
    let reference = serial(&chain(3));
    let mut broken = reference.clone();
    broken.releases.swap(0, 2);
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::PublicationOrder { .. })
    ));
}

/// The compiler's cost is proportional to what overlaps, and the census is how
/// that is checked rather than asserted. Independent work compiles no edges.
#[test]
fn independent_work_compiles_no_hazard_edges() {
    let run = parallel(&independent(8), 0);
    assert_eq!(run.census.edges, 0);
    assert_eq!(run.census.accesses, 8);
    assert_eq!(
        run.census.domain_only_comparisons, 0,
        "every access named a backing, so none of them met the domain"
    );
}

// -------------------------------------------------------------- eligibility

#[test]
fn a_wait_for_a_later_packets_stamp_has_no_serial_meaning() {
    let mut waiter = builder(1, 1);
    waiter.require(Prerequisite::Stamp(StampWait {
        slot: StampSlot(1),
        value: StampValue(5),
    }));
    let waiter = waiter.finish().expect("frozen");
    let mut producer = builder(1, 2);
    producer.publish_stamp(CompletionStamp {
        slot: StampSlot(1),
        value: StampValue(5),
    });
    let producer = producer.finish().expect("frozen");
    assert_eq!(
        eligible(&[waiter, producer]),
        Err(Ineligible::ForwardExplicitWait {
            waiter: IngressOrdinal(1),
            point: WaitPoint::Stamp {
                slot: StampSlot(1),
                value: StampValue(5)
            },
            producer: IngressOrdinal(2),
        })
    );
}

#[test]
fn a_wait_nothing_produces_has_no_serial_meaning() {
    let mut waiter = builder(1, 1);
    waiter.require(Prerequisite::Event {
        event: res(7),
        value: 3,
    });
    let batch = [waiter.finish().expect("frozen")];
    assert_eq!(
        eligible(&batch),
        Err(Ineligible::UnansweredWait {
            waiter: IngressOrdinal(1),
            point: WaitPoint::Event {
                event: res(7),
                value: 3
            },
        })
    );
    // And the wait graph says the same thing, because there is one answer to
    // this question and two callers of it.
    let mut graph = WaitGraph::new();
    graph.admit(&batch[0]);
    assert!(matches!(
        graph.diagnose().as_slice(),
        [Diagnosis::Unproduced { .. }]
    ));
}

#[test]
fn an_encoder_scoped_fence_prerequisite_is_outside_the_comparison() {
    let mut b = builder(1, 1);
    b.require(Prerequisite::Fence { fence: res(3) });
    assert_eq!(
        eligible(&[b.finish().expect("frozen")]),
        Err(Ineligible::FencePrerequisite {
            waiter: IngressOrdinal(1)
        })
    );
}

#[test]
fn transactions_out_of_ingress_order_are_refused_before_anything_else() {
    let batch = vec![
        builder(1, 5).finish().expect("frozen"),
        builder(1, 2).finish().expect("frozen"),
    ];
    assert_eq!(
        eligible(&batch),
        Err(Ineligible::OutOfIngressOrder {
            at: IngressOrdinal(2),
            after: IngressOrdinal(5),
        })
    );
}

#[test]
fn a_second_generation_in_one_batch_is_refused() {
    let first = builder(1, 1).finish().expect("frozen");
    let mut second = ExecBuilder::new(
        SessionGeneration::FIRST.next(),
        ChannelId(1),
        ChannelSequence(2),
        IngressOrdinal(2),
    );
    second.declare_access(whole(1, 1, AccessMode::Read));
    assert_eq!(
        eligible(&[first, second.finish().expect("frozen")]),
        Err(Ineligible::MixedGeneration {
            expected: SessionGeneration::FIRST,
            found: SessionGeneration::FIRST.next(),
        })
    );
}

/// Two writers of disjoint ranges of one backing are not a race. This is the
/// case that used to be one, because a version claim named a whole backing
/// while the access naming the bytes named a range; now the claim *is* the
/// access's region and the two histories are independent.
#[test]
fn two_publishers_of_disjoint_regions_of_one_backing_are_independent() {
    let batch: Vec<_> = [(1u64, 0u64), (2, 512)]
        .into_iter()
        .map(|(n, offset)| {
            let mut b = builder(1, n);
            b.declare_access(AccessIntent {
                mode: AccessMode::Write,
                output_content_version: Some(ContentVersion(1)),
                ..ranged(1, 1, offset)
            });
            b.finish().expect("frozen")
        })
        .collect();
    eligible(&batch).expect("disjoint regions have no shared history");
    let reference = serial(&batch);
    for seed in 0..16u64 {
        equivalent(&reference, &parallel(&batch, seed))
            .unwrap_or_else(|d| panic!("seed {seed} diverged: {d:?}"));
    }
}

/// What is left of the race, and it is real: two channels writing memory they
/// share. `requires_edge` orders nothing across domains — correctly, because
/// the guest supplied no ordering — so that region's version sequence has two
/// legal answers and this comparison declines to pick one.
#[test]
fn two_channels_writing_shared_memory_have_no_legal_version_order() {
    let batch: Vec<_> = [1u64, 2]
        .into_iter()
        .map(|n| {
            let mut b = builder(u32::try_from(n).expect("small"), n);
            b.declare_access(produces(u32::try_from(n).expect("small"), 1, 1));
            b.finish().expect("frozen")
        })
        .collect();
    assert_eq!(
        eligible(&batch),
        Err(Ineligible::UnorderedVersionRace {
            backing: BackingId(1),
            first: IngressOrdinal(1),
            second: IngressOrdinal(2),
        })
    );
}

// ------------------------------------------------- the relation itself bites

/// A relation that accepts everything proves nothing, so each arm is shown to
/// reject something.
#[test]
fn the_equivalence_relation_rejects_a_reordered_content_history() {
    let reference = serial(&chain(3));
    let mut broken = reference.clone();
    broken.trace.swap(0, 2);
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::ContentHistory { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_a_stamp_that_came_to_rest_elsewhere() {
    let reference = serial(&chain(3));
    let mut broken = reference.clone();
    for observation in &mut broken.trace {
        if let Observation::StampPublished { value, .. } = observation {
            *value = StampValue(value.0 + 100);
        }
    }
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::StampOutcome { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_a_stamp_that_goes_backwards() {
    let batch = chain(3);
    let reference = serial(&batch);
    let mut broken = reference.clone();
    // Republish the first slot value after the last one, which is exactly what
    // a device that overwrote instead of advancing would show a guest.
    broken.trace.push(Observation::StampPublished {
        slot: StampSlot(1),
        value: StampValue(1),
    });
    let last = broken.spans.last_mut().expect("a span");
    last.1.end = broken.trace.len();
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::NonMonotonePublication { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_a_publication_split_by_another_transaction() {
    let reference = serial(&chain(3));
    let mut broken = reference.clone();
    // Two transactions' completion windows overlap: one made its versions
    // visible while another was still making its own visible.
    broken.spans[1].1.start = broken.spans[0].1.start;
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::SplitPublication { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_a_stamp_published_before_its_versions() {
    let batch = chain(1);
    let reference = serial(&batch);
    let mut broken = reference.clone();
    broken.trace.swap(0, 1);
    assert!(
        matches!(
            equivalent(&reference, &broken),
            Err(Divergence::StampBeforeVersions { .. })
        ),
        "a guest that polled the stamp and then read the content must not be \
         able to see the flag without the bytes"
    );
}

#[test]
fn the_equivalence_relation_rejects_a_transaction_that_did_not_run() {
    let batch = chain(3);
    let reference = serial(&batch);
    let mut broken = reference.clone();
    broken.spans.pop();
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::DifferentTransactions { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_a_missing_fence_update() {
    let mut b = builder(1, 1);
    b.begin_segment(SegmentKind::Blit.wire_type(), false)
        .expect("blit encoder opens");
    b.record(
        ResolvedOperation::Fence(FenceOp {
            kind: FenceKind::Update,
            fence: res(30),
            stages: None,
        }),
        &mut StubRegistry(ChannelId(1)),
    )
    .expect("a fence update records");
    b.end_segment().expect("blit encoder closes");
    let reference = serial(&[b.finish().expect("frozen")]);
    assert_eq!(reference.trace.len(), 1);
    let mut broken = reference.clone();
    broken.trace.clear();
    broken.spans[0].1 = 0..0;
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::FenceUpdates { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_an_event_that_came_to_rest_elsewhere() {
    let mut b = builder(1, 1);
    b.begin_segment(SegmentKind::Event.wire_type(), false)
        .expect("event encoder opens");
    b.record(
        ResolvedOperation::Event(EventOp {
            kind: EventKind::Signal,
            event: res(20),
            value: 4,
        }),
        &mut StubRegistry(ChannelId(1)),
    )
    .expect("a signal records");
    b.end_segment().expect("event encoder closes");
    let reference = serial(&[b.finish().expect("frozen")]);
    let mut broken = reference.clone();
    broken.trace[0] = Observation::EventAdvanced {
        event: res(20),
        to: 9,
    };
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::EventOutcome { .. })
    ));
}
