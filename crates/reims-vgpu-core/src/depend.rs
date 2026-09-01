//! The hazard half of the dependency compiler: which earlier transactions a
//! newly admitted one must wait for, and why.
//!
//! # What this is and is not
//!
//! This compiles **hazard** edges — the ones that exist because two accesses
//! touch the same memory and at least one of them writes. It does not compile
//! explicit synchronisation. Events and fences can name a point that has not
//! been signalled yet, so they are unresolved prerequisites with their own
//! wait-for graph and their own cycle diagnostic; folding them in here would
//! destroy the one property that makes this graph trivially acyclic.
//!
//! That property is the reason transactions are admitted in ingress order and
//! never re-derived: **every edge this compiler creates points from a newer
//! ingress ordinal to an older one**, so ingress order is a topological order
//! for hazards. It is asserted rather than assumed, on every admission.
//!
//! # Imprecision costs a scan, and the census charges it
//!
//! Accesses are indexed by the thing they name — a backing, a heap — so
//! admitting one compares against the accesses that could possibly conflict
//! rather than against all of them. [`AccessKey::DomainOnly`] is the exception
//! by construction: an access whose participation is unknown could touch
//! anything, so it meets every live access in its domain and it makes every
//! later access meet it.
//!
//! That is the honest cost of rung three, and it is measured
//! ([`Census::domain_only_comparisons`]) rather than hidden, because the point
//! of measuring it is to know what narrowing an access would buy.

use crate::access::{requires_edge, AccessIntent, AccessKey, BackingId, HeapId};
use crate::identity::{ChannelId, IngressOrdinal};
use std::collections::{BTreeSet, HashMap};

/// One live access, and the transaction that declared it.
#[derive(Clone, Copy, Debug)]
struct Entry {
    ordinal: IngressOrdinal,
    intent: AccessIntent,
    /// Cleared when the transaction retires. The slot stays so the indices
    /// pointing at it stay valid; compaction is [`DependencyGraph::compact`].
    live: bool,
}

/// What admitting transactions has cost so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Accesses admitted.
    pub accesses: usize,
    /// Hazard edges created.
    pub edges: usize,
    /// Accesses admitted at each rung of the precision ladder, indexed by
    /// `rung - 1`.
    pub by_rung: [usize; 3],
    /// Comparisons that happened only because one side's participation was
    /// unknown. This is what rung three costs, and it is the number that says
    /// what narrowing an access would be worth.
    pub domain_only_comparisons: usize,
    /// Edges created against an access whose direction was not established.
    /// Ordering bought with ignorance rather than with knowledge.
    pub edges_from_unknown_mode: usize,
}

/// The live hazard state.
///
/// Holds only accesses whose transactions have not retired. Retiring is the
/// caller's obligation and is what keeps this bounded; nothing here evicts on
/// its own, because an eviction would silently drop an edge a later
/// transaction was owed.
#[derive(Debug, Default)]
pub struct DependencyGraph {
    entries: Vec<Entry>,
    by_backing: HashMap<BackingId, Vec<usize>>,
    by_heap: HashMap<HeapId, Vec<usize>>,
    by_domain: HashMap<ChannelId, Vec<usize>>,
    domain_only: HashMap<ChannelId, Vec<usize>>,
    by_ordinal: HashMap<IngressOrdinal, Vec<usize>>,
    census: Census,
}

impl DependencyGraph {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// Live accesses, for a test or a report. Not a bound anything enforces.
    #[must_use]
    pub fn live_accesses(&self) -> usize {
        self.entries.iter().filter(|e| e.live).count()
    }

    /// Admit one transaction's accesses and return the ordinals it must wait
    /// for.
    ///
    /// The returned set is deduplicated and sorted, because a caller that has
    /// to deduplicate is a caller that will forget to, and because two accesses
    /// of one transaction conflicting with one earlier transaction is one edge
    /// and not two.
    ///
    /// # Panics
    ///
    /// If `ordinal` is not greater than every ordinal already admitted. That is
    /// the ingress-order contract, and violating it would silently produce an
    /// edge pointing forward — which is the one thing this graph promises
    /// cannot happen.
    pub fn admit(
        &mut self,
        ordinal: IngressOrdinal,
        accesses: &[AccessIntent],
    ) -> Vec<IngressOrdinal> {
        assert!(
            self.entries
                .last()
                .is_none_or(|last| ordinal > last.ordinal),
            "transactions are admitted in ingress order; {ordinal:?} arrived after a later one"
        );
        let mut waits = BTreeSet::new();
        for intent in accesses {
            for candidate in self.candidates(intent) {
                let entry = self.entries[candidate];
                if !entry.live || entry.ordinal == ordinal {
                    continue;
                }
                if requires_edge(&entry.intent, intent) {
                    if waits.insert(entry.ordinal) {
                        self.census.edges += 1;
                    }
                    if entry.intent.mode == crate::access::AccessMode::Unknown
                        || intent.mode == crate::access::AccessMode::Unknown
                    {
                        self.census.edges_from_unknown_mode += 1;
                    }
                }
            }
            self.insert(ordinal, *intent);
        }
        waits.into_iter().collect()
    }

    /// The entries an access could possibly conflict with.
    ///
    /// Indices rather than entries so the borrow ends before `insert` runs, and
    /// deliberately allowed to contain duplicates: a resource that belongs to a
    /// heap is reachable two ways, and the caller already deduplicates by
    /// ordinal.
    fn candidates(&mut self, intent: &AccessIntent) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        // Every access meets the domain's unknown-participation accesses.
        if let Some(v) = self.domain_only.get(&intent.domain) {
            self.census.domain_only_comparisons += v.len();
            out.extend_from_slice(v);
        }
        match intent.key {
            AccessKey::DomainOnly => {
                // And an unknown-participation access meets everything in its
                // domain. Both halves are charged to the same counter.
                if let Some(v) = self.by_domain.get(&intent.domain) {
                    self.census.domain_only_comparisons += v.len();
                    out.extend_from_slice(v);
                }
            }
            AccessKey::Heap(h) => {
                if let Some(v) = self.by_heap.get(&h) {
                    out.extend_from_slice(v);
                }
            }
            AccessKey::Range(r, _) | AccessKey::Subresource(r, _) | AccessKey::Whole(r) => {
                if let Some(v) = self.by_backing.get(&r.backing) {
                    out.extend_from_slice(v);
                }
                if let Some(h) = r.heap {
                    if let Some(v) = self.by_heap.get(&h) {
                        out.extend_from_slice(v);
                    }
                }
            }
        }
        out
    }

    fn insert(&mut self, ordinal: IngressOrdinal, intent: AccessIntent) {
        let idx = self.entries.len();
        self.entries.push(Entry {
            ordinal,
            intent,
            live: true,
        });
        self.census.accesses += 1;
        self.census.by_rung[usize::from(intent.key.rung()) - 1] += 1;
        self.by_domain.entry(intent.domain).or_default().push(idx);
        self.by_ordinal.entry(ordinal).or_default().push(idx);
        match intent.key {
            AccessKey::DomainOnly => self.domain_only.entry(intent.domain).or_default().push(idx),
            AccessKey::Heap(h) => self.by_heap.entry(h).or_default().push(idx),
            AccessKey::Range(r, _) | AccessKey::Subresource(r, _) | AccessKey::Whole(r) => {
                self.by_backing.entry(r.backing).or_default().push(idx);
                // Also under its heap, so a heap declaration can find it: a
                // heap-use record names the heap and never its members, so
                // membership has to be reachable from both directions.
                if let Some(h) = r.heap {
                    self.by_heap.entry(h).or_default().push(idx);
                }
            }
        }
    }

    /// Mark one transaction's accesses as no longer live.
    ///
    /// Retirement is a completion fact, not a planning one: an access stops
    /// creating edges when the work that declared it has finished, and a caller
    /// that retires early publishes a hazard it still owes.
    pub fn retire(&mut self, ordinal: IngressOrdinal) {
        for &idx in self.by_ordinal.get(&ordinal).into_iter().flatten() {
            self.entries[idx].live = false;
        }
        self.by_ordinal.remove(&ordinal);
    }

    /// Drop retired entries and rebuild the indexes.
    ///
    /// Separate from [`Self::retire`] because retirement is on the completion
    /// path and this is not: an index rebuild in a completion handler is work
    /// charged to the thing that finished rather than to the thing that grew.
    pub fn compact(&mut self) {
        let live: Vec<_> = self.entries.iter().copied().filter(|e| e.live).collect();
        self.entries.clear();
        self.by_backing.clear();
        self.by_heap.clear();
        self.by_domain.clear();
        self.domain_only.clear();
        self.by_ordinal.clear();
        // The census is a running total across the graph's life and is not
        // rebuilt: compaction is bookkeeping, and it did not admit anything.
        let saved = self.census;
        for e in live {
            self.insert(e.ordinal, e.intent);
        }
        self.census = saved;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessMode, ByteRange, ResourceKey, SubresourceRange};

    fn res(backing: u64) -> ResourceKey {
        ResourceKey {
            backing: BackingId(backing),
            heap: None,
        }
    }

    fn intent(key: AccessKey, mode: AccessMode) -> AccessIntent {
        AccessIntent {
            domain: ChannelId(1),
            key,
            mode,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        }
    }

    fn ord(n: u64) -> IngressOrdinal {
        IngressOrdinal(n)
    }

    /// The property the whole graph rests on.
    #[test]
    fn every_edge_points_backwards() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        for n in 1..=8 {
            let mode = if n % 2 == 0 {
                AccessMode::Write
            } else {
                AccessMode::Read
            };
            for w in g.admit(ord(n), &[intent(k, mode)]) {
                assert!(w < ord(n), "edge {w:?} -> {:?} points forward", ord(n));
            }
        }
    }

    #[test]
    fn a_read_waits_for_the_preceding_writer_and_not_for_readers() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        assert!(g.admit(ord(1), &[intent(k, AccessMode::Write)]).is_empty());
        assert_eq!(
            g.admit(ord(2), &[intent(k, AccessMode::Read)]),
            vec![ord(1)]
        );
        assert_eq!(
            g.admit(ord(3), &[intent(k, AccessMode::Read)]),
            vec![ord(1)],
            "a second reader waits for the writer and not for the first reader"
        );
    }

    #[test]
    fn a_write_waits_for_the_preceding_writer_and_every_preceding_reader() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        g.admit(ord(1), &[intent(k, AccessMode::Write)]);
        g.admit(ord(2), &[intent(k, AccessMode::Read)]);
        g.admit(ord(3), &[intent(k, AccessMode::Read)]);
        assert_eq!(
            g.admit(ord(4), &[intent(k, AccessMode::Write)]),
            vec![ord(1), ord(2), ord(3)]
        );
    }

    #[test]
    fn disjoint_ranges_over_one_backing_do_not_meet() {
        let mut g = DependencyGraph::new();
        let lo = AccessKey::Range(
            res(1),
            ByteRange {
                offset: 0,
                length: 64,
            },
        );
        let hi = AccessKey::Range(
            res(1),
            ByteRange {
                offset: 64,
                length: 64,
            },
        );
        g.admit(ord(1), &[intent(lo, AccessMode::Write)]);
        assert!(
            g.admit(ord(2), &[intent(hi, AccessMode::Write)]).is_empty(),
            "two writes to disjoint halves of one buffer are independent"
        );
        assert_eq!(
            g.admit(
                ord(3),
                &[intent(AccessKey::Whole(res(1)), AccessMode::Read)]
            ),
            vec![ord(1), ord(2)],
            "and a whole-backing read meets both of them"
        );
    }

    #[test]
    fn a_heap_declaration_meets_its_members_in_both_directions() {
        let heap = HeapId {
            id: 3,
            membership_generation: 1,
        };
        let member = AccessKey::Whole(ResourceKey {
            backing: BackingId(9),
            heap: Some(heap),
        });
        // Member first, then the heap declaration.
        let mut g = DependencyGraph::new();
        g.admit(ord(1), &[intent(member, AccessMode::Write)]);
        assert_eq!(
            g.admit(ord(2), &[intent(AccessKey::Heap(heap), AccessMode::Read)]),
            vec![ord(1)]
        );
        // Heap declaration first, then the member.
        let mut g = DependencyGraph::new();
        g.admit(ord(1), &[intent(AccessKey::Heap(heap), AccessMode::Write)]);
        assert_eq!(
            g.admit(ord(2), &[intent(member, AccessMode::Read)]),
            vec![ord(1)]
        );
    }

    #[test]
    fn unknown_participation_meets_everything_in_its_domain_both_ways() {
        let mut g = DependencyGraph::new();
        g.admit(
            ord(1),
            &[intent(AccessKey::Whole(res(1)), AccessMode::Write)],
        );
        g.admit(
            ord(2),
            &[intent(AccessKey::Whole(res(2)), AccessMode::Write)],
        );
        assert_eq!(
            g.admit(ord(3), &[intent(AccessKey::DomainOnly, AccessMode::Read)]),
            vec![ord(1), ord(2)],
            "an access that could touch anything waits for every writer"
        );
        assert_eq!(
            g.admit(
                ord(4),
                &[intent(AccessKey::Whole(res(3)), AccessMode::Write)]
            ),
            vec![ord(3)],
            "and a later access to an unrelated backing waits for it"
        );
        assert!(g.census().domain_only_comparisons > 0);
    }

    #[test]
    fn separate_domains_produce_no_hazard_edge() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        let mut a = intent(k, AccessMode::Write);
        a.domain = ChannelId(1);
        let mut b = intent(k, AccessMode::Read);
        b.domain = ChannelId(2);
        g.admit(ord(1), &[a]);
        assert!(
            g.admit(ord(2), &[b]).is_empty(),
            "the contract leaves separate submission domains unordered"
        );
    }

    #[test]
    fn a_retired_transaction_stops_creating_edges() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        g.admit(ord(1), &[intent(k, AccessMode::Write)]);
        g.retire(ord(1));
        assert!(g.admit(ord(2), &[intent(k, AccessMode::Read)]).is_empty());
        assert_eq!(g.live_accesses(), 1);
        g.compact();
        assert_eq!(g.live_accesses(), 1, "only the live access survives");
    }

    /// Compaction is bookkeeping. It must not change an answer, and it must not
    /// rewrite the running totals — a census that reset on compaction would
    /// under-report exactly when the graph was busiest.
    #[test]
    fn compaction_changes_no_answer_and_no_total() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        g.admit(ord(1), &[intent(k, AccessMode::Write)]);
        g.admit(ord(2), &[intent(k, AccessMode::Read)]);
        g.retire(ord(1));
        let before = g.census();
        g.compact();
        assert_eq!(g.census(), before);
        assert_eq!(
            g.admit(ord(3), &[intent(k, AccessMode::Write)]),
            vec![ord(2)]
        );
    }

    /// Ordering bought with ignorance is counted apart from ordering bought
    /// with knowledge, so a census can say which one a workload is paying.
    #[test]
    fn an_unknown_direction_is_charged_to_its_own_counter() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        g.admit(ord(1), &[intent(k, AccessMode::Read)]);
        assert_eq!(
            g.admit(ord(2), &[intent(k, AccessMode::Unknown)]),
            vec![ord(1)]
        );
        assert_eq!(g.census().edges, 1);
        assert_eq!(g.census().edges_from_unknown_mode, 1);
    }

    #[test]
    fn subresource_windows_that_do_not_overlap_are_independent() {
        let win = |base_level| {
            AccessKey::Subresource(
                res(1),
                SubresourceRange {
                    base_level,
                    level_count: 1,
                    base_slice: 0,
                    slice_count: 1,
                    plane: 0,
                },
            )
        };
        let mut g = DependencyGraph::new();
        g.admit(ord(1), &[intent(win(0), AccessMode::Write)]);
        assert!(g
            .admit(ord(2), &[intent(win(1), AccessMode::Write)])
            .is_empty());
        assert_eq!(
            g.admit(ord(3), &[intent(win(0), AccessMode::Read)]),
            vec![ord(1)]
        );
    }

    #[test]
    #[should_panic(expected = "admitted in ingress order")]
    fn admitting_out_of_order_is_a_contract_violation_and_not_a_reordering() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        g.admit(ord(2), &[intent(k, AccessMode::Write)]);
        g.admit(ord(1), &[intent(k, AccessMode::Read)]);
    }

    /// Two accesses of one transaction meeting one earlier transaction is one
    /// edge. A caller that had to deduplicate would eventually not.
    #[test]
    fn one_earlier_transaction_produces_one_edge_however_many_accesses_meet_it() {
        let mut g = DependencyGraph::new();
        g.admit(
            ord(1),
            &[intent(AccessKey::Whole(res(1)), AccessMode::Write)],
        );
        let waits = g.admit(
            ord(2),
            &[
                intent(AccessKey::Whole(res(1)), AccessMode::Read),
                intent(
                    AccessKey::Range(
                        res(1),
                        ByteRange {
                            offset: 0,
                            length: 4,
                        },
                    ),
                    AccessMode::Read,
                ),
            ],
        );
        assert_eq!(waits, vec![ord(1)]);
        assert_eq!(g.census().edges, 1);
    }

    /// Two accesses of the *same* transaction never order against each other:
    /// a transaction is one unit, and an intra-transaction edge would be a
    /// transaction waiting for itself.
    #[test]
    fn a_transaction_never_waits_for_itself() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        let waits = g.admit(
            ord(1),
            &[intent(k, AccessMode::Write), intent(k, AccessMode::Read)],
        );
        assert!(waits.is_empty());
    }
}
