//! The explicit half of the dependency compiler: waits the guest wrote down,
//! and the diagnosis when nothing will ever answer one.
//!
//! # Why this cannot live in `depend`
//!
//! [`crate::depend`] compiles hazards, and its whole value is that every edge
//! it creates points backwards in ingress order — so ingress order is a
//! topological order and a hazard cycle is not a thing that can exist. That
//! property is not an accident of the implementation; it is what lets a
//! scheduler admit in arrival order and never re-derive.
//!
//! An explicit wait is the opposite. A guest may submit a packet that waits for
//! a stamp point or an event value that nothing has produced yet, and the
//! packet that will produce it may not have arrived. Such an edge points
//! *forwards*, and two packets can wait on each other. Folding these into the
//! hazard graph would destroy the acyclicity that makes the hazard graph
//! cheap, and would replace a structural guarantee with a runtime search.
//!
//! So they are two graphs. This one is allowed to be cyclic, and its job is to
//! say so.
//!
//! # A deadlock is a diagnosis, not a timeout
//!
//! Both answers this module gives are derived from what has been admitted
//! rather than from a clock. [`Diagnosis::Unproduced`] says no admitted
//! transaction produces the point a waiter names; [`Diagnosis::Cycle`] says a
//! set of transactions wait on each other and therefore none of them can be
//! first. Neither is a claim that the guest is wrong — a packet that would
//! break the cycle may still arrive, which is why nothing here cancels
//! anything and the caller decides.
//!
//! # Fences are not in this graph, and that is a contract statement
//!
//! An event and a completion stamp are device-scoped points: any packet may
//! signal them and any packet may wait for them, so "who produces this" is a
//! cross-packet question. A fence is encoder-scoped — its producer and its
//! consumer are records inside a command stream, ordered by their positions in
//! it. Putting a fence in a cross-packet wait-for graph would invent a
//! producer relationship the contract does not have, and would report a cycle
//! for the ordinary case of one encoder updating a fence a later encoder in
//! the same packet waits on.

use crate::exec::{ExecTransaction, Prerequisite, ResolvedOperation};
use crate::identity::{IngressOrdinal, ResourceId, StampSlot, StampValue};
use crate::sync::EventKind;
use std::collections::HashMap;

/// A point a transaction waits for, in the two device-scoped flavours that
/// have cross-packet producers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WaitPoint {
    /// A completion-stamp slot must have reached a value.
    Stamp { slot: StampSlot, value: StampValue },
    /// An event's monotonic generation must have reached a value.
    Event { event: ResourceId, value: u64 },
}

/// A point a transaction publishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Production {
    Stamp { slot: StampSlot, value: StampValue },
    Event { event: ResourceId, value: u64 },
}

impl Production {
    /// Whether publishing this discharges `wait`.
    #[must_use]
    pub fn discharges(self, wait: WaitPoint) -> bool {
        match (self, wait) {
            (
                Self::Stamp {
                    slot: produced,
                    value: at,
                },
                WaitPoint::Stamp {
                    slot: wanted,
                    value: needed,
                },
            ) => produced == wanted && at.reached(needed),
            (
                Self::Event {
                    event: produced,
                    value: at,
                },
                WaitPoint::Event {
                    event: wanted,
                    value: needed,
                },
            ) => produced == wanted && at >= needed,
            _ => false,
        }
    }
}

/// Why a set of explicit waits cannot be answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Diagnosis {
    /// A waiter names a point that nothing admitted, and nothing already
    /// published, produces.
    Unproduced {
        waiter: IngressOrdinal,
        point: WaitPoint,
    },
    /// A set of transactions wait on each other, so none of them can run
    /// first. Members are in ingress order and the cycle is reported once,
    /// keyed on its lowest member.
    Cycle {
        members: Vec<IngressOrdinal>,
        /// One wait that closes the cycle, so the report names a resource and
        /// not only a set of ordinals.
        closed_by: WaitPoint,
    },
}

impl Diagnosis {
    #[must_use]
    pub const fn slug(&self) -> &'static str {
        match self {
            Self::Unproduced { .. } => "prereq_unproduced",
            Self::Cycle { .. } => "prereq_cycle",
        }
    }
}

/// One admitted transaction's explicit waits and productions.
#[derive(Clone, Debug)]
struct Node {
    ordinal: IngressOrdinal,
    waits: Vec<WaitPoint>,
    produces: Vec<Production>,
}

/// The explicit wait-for graph over admitted, uncompleted transactions.
///
/// Unlike [`crate::depend::DependencyGraph`] this accepts transactions in any
/// order: an explicit wait may name a producer that has not been admitted yet,
/// so refusing an out-of-order admission would refuse the ordinary case.
#[derive(Debug, Default)]
pub struct WaitGraph {
    nodes: Vec<Node>,
    /// Points published outside this batch — by an earlier, already-retired
    /// transaction, or by the guest writing a timeline itself. A wait these
    /// discharge is not an edge and cannot be part of a cycle.
    published_stamps: HashMap<StampSlot, StampValue>,
    published_events: HashMap<ResourceId, u64>,
}

impl WaitGraph {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a point that is already published.
    pub fn satisfy(&mut self, production: Production) {
        match production {
            Production::Stamp { slot, value } => {
                let at = self.published_stamps.entry(slot).or_insert(value);
                // Later in the wrapping order, not numerically. See
                // [`StampValue::later`].
                *at = at.later(value);
            }
            Production::Event { event, value } => {
                let at = self.published_events.entry(event).or_insert(value);
                *at = (*at).max(value);
            }
        }
    }

    /// Whether what is already published discharges `wait`.
    #[must_use]
    fn already_published(&self, wait: WaitPoint) -> bool {
        match wait {
            WaitPoint::Stamp { slot, value } => self
                .published_stamps
                .get(&slot)
                .is_some_and(|at| at.reached(value)),
            WaitPoint::Event { event, value } => self
                .published_events
                .get(&event)
                .is_some_and(|at| *at >= value),
        }
    }

    /// Admit a transaction: its packet-level prerequisites become waits and
    /// its stamp and event signals become productions.
    ///
    /// A signal record is a production even though it happens partway through
    /// the packet, because the packet's completion is the earliest point at
    /// which any *other* packet may rely on it. Treating it as available
    /// earlier would be a claim about record-level publication that no
    /// cross-packet consumer can observe.
    pub fn admit(&mut self, tx: &ExecTransaction) {
        let mut waits = Vec::new();
        for prerequisite in tx.prerequisites() {
            match *prerequisite {
                Prerequisite::Stamp(w) => waits.push(WaitPoint::Stamp {
                    slot: w.slot,
                    value: w.value,
                }),
                Prerequisite::Event { event, value } => {
                    waits.push(WaitPoint::Event { event, value });
                }
                // Encoder-scoped. See the module documentation.
                Prerequisite::Fence { .. } => {}
            }
        }
        let mut produces = Vec::new();
        if let Some(stamp) = tx.work.publication.stamp {
            produces.push(Production::Stamp {
                slot: stamp.slot,
                value: stamp.value,
            });
        }
        for record in tx.records() {
            if let ResolvedOperation::Event(event) = record.op {
                if event.kind == EventKind::Signal {
                    produces.push(Production::Event {
                        event: event.event,
                        value: event.value,
                    });
                }
            }
        }
        self.nodes.push(Node {
            ordinal: tx.ingress(),
            waits,
            produces,
        });
    }

    /// Admitted transactions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Forget a transaction once it has completed, publishing what it
    /// produced.
    ///
    /// Retiring publishes rather than discards: a wait its production
    /// discharged must stay discharged, or a later diagnosis would report a
    /// point as unproduced after the thing that produced it finished.
    pub fn retire(&mut self, ordinal: IngressOrdinal) {
        let Some(at) = self.nodes.iter().position(|n| n.ordinal == ordinal) else {
            return;
        };
        let node = self.nodes.remove(at);
        for production in node.produces {
            self.satisfy(production);
        }
    }

    /// Waiter-to-producer edges, one per (waiter, wait, producer) triple.
    ///
    /// A wait already discharged by [`Self::satisfy`] produces no edge, and a
    /// transaction never depends on itself: a packet-level prerequisite is
    /// checked before the packet's own records run, so its own signal cannot
    /// answer it. A self-wait is therefore reported by [`Self::diagnose`] as a
    /// one-member cycle rather than as an edge.
    #[must_use]
    pub fn edges(&self) -> Vec<(IngressOrdinal, WaitPoint, IngressOrdinal)> {
        let mut out = Vec::new();
        for node in &self.nodes {
            for &wait in &node.waits {
                if self.already_published(wait) {
                    continue;
                }
                for other in &self.nodes {
                    if other.produces.iter().any(|p| p.discharges(wait)) {
                        out.push((node.ordinal, wait, other.ordinal));
                    }
                }
            }
        }
        out
    }

    /// Every reason an admitted wait cannot be answered, in ingress order of
    /// the reporting transaction.
    #[must_use]
    pub fn diagnose(&self) -> Vec<Diagnosis> {
        let mut order: Vec<usize> = (0..self.nodes.len()).collect();
        order.sort_by_key(|&i| self.nodes[i].ordinal);
        let index: HashMap<IngressOrdinal, usize> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.ordinal, i))
            .collect();

        let mut out = Vec::new();
        for &i in &order {
            for &wait in &self.nodes[i].waits {
                if self.already_published(wait) {
                    continue;
                }
                if !self
                    .nodes
                    .iter()
                    .any(|n| n.produces.iter().any(|p| p.discharges(wait)))
                {
                    out.push(Diagnosis::Unproduced {
                        waiter: self.nodes[i].ordinal,
                        point: wait,
                    });
                }
            }
        }

        // Adjacency, waiter -> producers, over the same edges `edges` reports.
        let mut adjacency: Vec<Vec<(WaitPoint, usize)>> = vec![Vec::new(); self.nodes.len()];
        for (waiter, wait, producer) in self.edges() {
            adjacency[index[&waiter]].push((wait, index[&producer]));
        }

        let mut colour = vec![Colour::Unseen; self.nodes.len()];
        let mut reported: Vec<Vec<IngressOrdinal>> = Vec::new();
        for &root in &order {
            self.walk(
                root,
                &adjacency,
                &mut colour,
                &mut Vec::new(),
                &mut out,
                &mut reported,
            );
        }
        out
    }

    /// Depth-first cycle search. Recursion depth is the length of a wait
    /// chain, which is bounded by the admitted transactions and is small; the
    /// alternative is an explicit stack that says the same thing less clearly.
    fn walk(
        &self,
        at: usize,
        adjacency: &[Vec<(WaitPoint, usize)>],
        colour: &mut [Colour],
        path: &mut Vec<(usize, WaitPoint)>,
        out: &mut Vec<Diagnosis>,
        reported: &mut Vec<Vec<IngressOrdinal>>,
    ) {
        match colour[at] {
            Colour::Done => return,
            Colour::Open => {
                // `at` is on the current path: everything from its position on
                // is a cycle.
                let start = path
                    .iter()
                    .position(|&(node, _)| node == at)
                    .expect("an open node is on the path");
                let mut members: Vec<IngressOrdinal> = path[start..]
                    .iter()
                    .map(|&(n, _)| self.nodes[n].ordinal)
                    .collect();
                let closed_by = path[start].1;
                members.sort_unstable();
                if !reported.contains(&members) {
                    reported.push(members.clone());
                    out.push(Diagnosis::Cycle { members, closed_by });
                }
                return;
            }
            Colour::Unseen => {}
        }
        colour[at] = Colour::Open;
        for &(wait, next) in &adjacency[at] {
            path.push((at, wait));
            self.walk(next, adjacency, colour, path, out, reported);
            path.pop();
        }
        colour[at] = Colour::Done;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Colour {
    Unseen,
    Open,
    Done,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::StubRegistry;
    use crate::identity::{ChannelId, CompletionStamp, ObjectListRef, SlotGeneration, StampWait};
    use crate::stream::{SegmentKind, SegmentLifetime};
    use crate::sync::EventOp;

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn builder(ingress: u64) -> crate::testing::At {
        crate::testing::At::new(1, ingress)
    }

    /// A packet that waits for `waits` (event, value) and signals `signals`.
    fn packet(
        ingress: u64,
        waits: &[(u32, u64)],
        signals: &[(u32, u64)],
    ) -> crate::testing::Stamped {
        let mut b = builder(ingress);
        for &(event, value) in waits {
            b.require(Prerequisite::Event {
                event: res(event),
                value,
            });
        }
        if !signals.is_empty() {
            b.begin_segment(
                SegmentKind::Event.wire_type(),
                SegmentLifetime::SELF_CONTAINED,
            )
            .expect("segment opens");
            for &(event, value) in signals {
                b.record(
                    ResolvedOperation::Event(EventOp {
                        kind: EventKind::Signal,
                        event: res(event),
                        value,
                    }),
                    &mut StubRegistry(ChannelId(1)),
                )
                .expect("a signal records");
            }
            b.end_segment().expect("segment closes");
        }
        b.finish().expect("frozen")
    }

    #[test]
    fn a_wait_answered_by_a_later_packet_is_an_edge_that_points_forwards() {
        let mut g = WaitGraph::new();
        g.admit(&packet(1, &[(7, 4)], &[]).view());
        g.admit(&packet(2, &[], &[(7, 4)]).view());
        assert_eq!(
            g.edges(),
            vec![(
                IngressOrdinal(1),
                WaitPoint::Event {
                    event: res(7),
                    value: 4
                },
                IngressOrdinal(2)
            )],
            "the producer arrived after the waiter, which is ordinary"
        );
        assert!(g.diagnose().is_empty(), "a met wait is not a diagnosis");
    }

    #[test]
    fn a_wait_no_admitted_packet_produces_is_unproduced() {
        let mut g = WaitGraph::new();
        g.admit(&packet(1, &[(7, 4)], &[]).view());
        g.admit(&packet(2, &[], &[(7, 3)]).view());
        assert_eq!(
            g.diagnose(),
            vec![Diagnosis::Unproduced {
                waiter: IngressOrdinal(1),
                point: WaitPoint::Event {
                    event: res(7),
                    value: 4
                },
            }],
            "signalling 3 does not reach 4"
        );
    }

    #[test]
    fn an_already_published_point_discharges_the_wait_without_an_edge() {
        let mut g = WaitGraph::new();
        g.satisfy(Production::Event {
            event: res(7),
            value: 9,
        });
        g.admit(&packet(1, &[(7, 4)], &[]).view());
        assert!(g.edges().is_empty());
        assert!(g.diagnose().is_empty());
    }

    /// The property the hazard graph gets for free and this one has to check.
    #[test]
    fn two_packets_waiting_on_each_other_are_a_cycle() {
        let mut g = WaitGraph::new();
        g.admit(&packet(1, &[(8, 1)], &[(7, 1)]).view());
        g.admit(&packet(2, &[(7, 1)], &[(8, 1)]).view());
        let cycles: Vec<_> = g
            .diagnose()
            .into_iter()
            .filter(|d| matches!(d, Diagnosis::Cycle { .. }))
            .collect();
        assert_eq!(cycles.len(), 1, "one cycle, reported once");
        let Diagnosis::Cycle { members, .. } = &cycles[0] else {
            unreachable!()
        };
        assert_eq!(members, &[IngressOrdinal(1), IngressOrdinal(2)]);
    }

    #[test]
    fn a_packet_waiting_for_its_own_signal_is_a_one_member_cycle() {
        let mut g = WaitGraph::new();
        g.admit(&packet(1, &[(7, 1)], &[(7, 1)]).view());
        assert_eq!(
            g.diagnose(),
            vec![Diagnosis::Cycle {
                members: vec![IngressOrdinal(1)],
                closed_by: WaitPoint::Event {
                    event: res(7),
                    value: 1
                },
            }],
            "a packet-level prerequisite is checked before the packet's own \
             records run, so its own signal cannot answer it"
        );
    }

    #[test]
    fn a_three_packet_wait_chain_is_not_a_cycle() {
        let mut g = WaitGraph::new();
        g.admit(&packet(1, &[(7, 1)], &[]).view());
        g.admit(&packet(2, &[(8, 1)], &[(7, 1)]).view());
        g.admit(&packet(3, &[], &[(8, 1)]).view());
        assert!(g.diagnose().is_empty(), "a chain terminates");
    }

    /// Retiring publishes, so the diagnosis does not regress once the producer
    /// is gone.
    #[test]
    fn retiring_a_producer_keeps_the_wait_it_answered_answered() {
        let mut g = WaitGraph::new();
        g.admit(&packet(2, &[], &[(7, 4)]).view());
        g.admit(&packet(1, &[(7, 4)], &[]).view());
        g.retire(IngressOrdinal(2));
        assert_eq!(g.len(), 1);
        assert!(
            g.diagnose().is_empty(),
            "the point was produced; the producer finishing does not unproduce it"
        );
    }

    #[test]
    fn a_stamp_wait_participates_and_a_wrapped_value_still_discharges_it() {
        let mut b = builder(1);
        b.require(Prerequisite::Stamp(StampWait {
            slot: StampSlot(2),
            value: StampValue(1),
        }));
        let waiter = b.finish().expect("frozen");
        let mut g = WaitGraph::new();
        g.satisfy(Production::Stamp {
            slot: StampSlot(2),
            value: StampValue(u32::MAX),
        });
        g.satisfy(Production::Stamp {
            slot: StampSlot(2),
            value: StampValue(1),
        });
        g.admit(&waiter.view());
        assert!(
            g.diagnose().is_empty(),
            "1 follows u32::MAX on a wrapping timeline"
        );
    }

    /// The contract statement in the module documentation, as a test.
    #[test]
    fn a_fence_wait_is_not_a_cross_packet_edge() {
        let mut b = builder(1);
        b.require(Prerequisite::Fence { fence: res(3) });
        let tx = b.finish().expect("frozen");
        let mut g = WaitGraph::new();
        g.admit(&tx.view());
        assert!(g.edges().is_empty());
        assert!(
            g.diagnose().is_empty(),
            "a fence is encoder-scoped; this graph has no opinion about it"
        );
    }

    #[test]
    fn a_completion_stamp_answers_a_stamp_wait_from_another_packet() {
        let mut waiter = builder(1);
        waiter.require(Prerequisite::Stamp(StampWait {
            slot: StampSlot(2),
            value: StampValue(5),
        }));
        let waiter = waiter.finish().expect("frozen");
        let mut producer = builder(2);
        producer.publish_stamp(CompletionStamp {
            slot: StampSlot(2),
            value: StampValue(5),
        });
        let producer = producer.finish().expect("frozen");

        let mut g = WaitGraph::new();
        g.admit(&waiter.view());
        g.admit(&producer.view());
        assert_eq!(g.edges().len(), 1);
        assert!(g.diagnose().is_empty());
    }
}
