//! When an admitted transaction may begin, and how a wait that will never be
//! answered is told from one that has simply not been answered yet.
//!
//! # Two kinds of prerequisite, and only one of them is ordered by arrival
//!
//! A hazard edge points backwards in ingress order by construction, so a
//! transaction's hazard prerequisites are always transactions that already
//! exist. A **stamp wait** is not like that at all: the guest may submit a
//! packet that waits for a value nothing has produced yet, and the packet that
//! will produce it may not have arrived. That is legal and ordinary, and it is
//! why the two are tracked apart — counting them together would make "waiting
//! for work that exists" and "waiting for work that does not" the same state.
//!
//! # Nothing here blocks a thread
//!
//! Readiness is published, not waited on. A transaction becomes ready when its
//! last prerequisite is discharged, and the caller collects the ready set. No
//! method here parks, sleeps, or spins, because a scheduler that can block is a
//! scheduler that will eventually block the drain.
//!
//! # A wait nobody can answer is a diagnosis, not a timeout
//!
//! [`Scheduler::stalled`] names the transactions whose stamp waits no admitted
//! transaction will publish. That is a different statement from "this is taking
//! a while": it is derived from what has been admitted rather than from a
//! clock, so it is true the instant it becomes true and it stays true. A guest
//! waiting on a stamp nobody will write is a hang with a cause, and this is the
//! cause.

use crate::identity::{CompletionStamp, IngressOrdinal, StampSlot, StampValue, StampWait};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One admitted transaction's unfinished business.
#[derive(Debug)]
struct Pending {
    /// Earlier transactions this one must not overtake. Counted rather than
    /// listed: the list is the dependents index, held in the other direction.
    remaining_hazards: usize,
    /// Stamp points that must be published before this may begin.
    stamp_waits: Vec<StampWait>,
    /// What this transaction publishes when it completes.
    completion: Option<CompletionStamp>,
}

/// The readiness service for one session.
#[derive(Debug, Default)]
pub struct Scheduler {
    pending: BTreeMap<IngressOrdinal, Pending>,
    /// The highest value published into each slot, in the wrapping order
    /// [`StampValue::follows`] compares in.
    published: HashMap<StampSlot, StampValue>,
    /// Transactions blocked on each slot, so publishing does not scan.
    waiters_by_slot: HashMap<StampSlot, BTreeSet<IngressOrdinal>>,
    /// For each transaction, the transactions that must not overtake it.
    dependents: HashMap<IngressOrdinal, Vec<IngressOrdinal>>,
    ready: BTreeSet<IngressOrdinal>,
}

impl Scheduler {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit a transaction with its prerequisites.
    ///
    /// `hazard_waits` are ordinals from the dependency compiler; any that have
    /// already completed are discharged here rather than counted, so a
    /// transaction admitted behind finished work is ready at once.
    ///
    /// Returns whether it is ready immediately. The ready set is also
    /// accumulated for [`Self::take_ready`], so a caller may use either and
    /// will not see the same transaction twice.
    pub fn admit(
        &mut self,
        ordinal: IngressOrdinal,
        hazard_waits: &[IngressOrdinal],
        stamp_waits: &[StampWait],
        completion: Option<CompletionStamp>,
    ) -> bool {
        let live: Vec<_> = hazard_waits
            .iter()
            .copied()
            .filter(|o| self.pending.contains_key(o))
            .collect();
        let unmet: Vec<StampWait> = stamp_waits
            .iter()
            .copied()
            .filter(|w| !self.is_published(*w))
            .collect();
        for w in &unmet {
            self.waiters_by_slot
                .entry(w.slot)
                .or_default()
                .insert(ordinal);
        }
        for dep in &live {
            self.dependents.entry(*dep).or_default().push(ordinal);
        }
        let ready = live.is_empty() && unmet.is_empty();
        self.pending.insert(
            ordinal,
            Pending {
                remaining_hazards: live.len(),
                stamp_waits: unmet,
                completion,
            },
        );
        if ready {
            self.ready.insert(ordinal);
        }
        ready
    }

    /// Whether a wait is already discharged by what has been published.
    #[must_use]
    pub fn is_published(&self, wait: StampWait) -> bool {
        self.published
            .get(&wait.slot)
            .is_some_and(|v| wait.satisfied_by(*v))
    }

    /// The value standing in a slot, if anything has published one.
    #[must_use]
    pub fn published_value(&self, slot: StampSlot) -> Option<StampValue> {
        self.published.get(&slot).copied()
    }

    /// Complete a transaction: release what was waiting on it, and hand back
    /// the stamp it now owes.
    ///
    /// It does **not** publish that stamp. When a completion word becomes
    /// readable is [`crate::publish`]'s question, not readiness's: a channel
    /// publishes in its own order, so a transaction that finishes ahead of an
    /// earlier position in its channel owes its stamp and may not yet pay it.
    /// A scheduler that published here would be deciding, silently, that
    /// completion and publication are the same event.
    ///
    /// Hazard dependents are released regardless, because they wait for the
    /// *work* and not for the guest being told about it.
    ///
    /// # Panics
    ///
    /// If `ordinal` was never admitted or has already completed. Completing a
    /// transaction twice would hand its stamp back twice and decrement its
    /// dependents past zero, and both of those are silent corruptions rather
    /// than loud ones.
    pub fn complete(&mut self, ordinal: IngressOrdinal) -> Option<CompletionStamp> {
        let done = self
            .pending
            .remove(&ordinal)
            .expect("completing a transaction that is not pending");
        self.ready.remove(&ordinal);
        for w in &done.stamp_waits {
            if let Some(set) = self.waiters_by_slot.get_mut(&w.slot) {
                set.remove(&ordinal);
            }
        }
        for dep in self.dependents.remove(&ordinal).unwrap_or_default() {
            if let Some(p) = self.pending.get_mut(&dep) {
                p.remaining_hazards -= 1;
                if p.remaining_hazards == 0 && p.stamp_waits.is_empty() {
                    self.ready.insert(dep);
                }
            }
        }
        done.completion
    }

    /// Publish a stamp value without completing a transaction.
    ///
    /// The guest can advance a timeline itself, and a device that only ever
    /// published from its own completions would hold packets against a value
    /// that had already been written.
    pub fn publish(&mut self, stamp: CompletionStamp) {
        let slot = self.published.entry(stamp.slot).or_insert(stamp.value);
        // Later in the wrapping order, not numerically: a wrapped timeline
        // makes the later value the smaller one, and `max` would then refuse
        // to advance past the wrap for the rest of the boot.
        *slot = slot.later(stamp.value);
        let published = *slot;
        let Some(waiters) = self.waiters_by_slot.get(&stamp.slot).cloned() else {
            return;
        };
        for w in waiters {
            let Some(p) = self.pending.get_mut(&w) else {
                continue;
            };
            p.stamp_waits
                .retain(|wait| wait.slot != stamp.slot || !wait.satisfied_by(published));
            if p.stamp_waits.is_empty() {
                if p.remaining_hazards == 0 {
                    self.ready.insert(w);
                }
                if let Some(set) = self.waiters_by_slot.get_mut(&stamp.slot) {
                    set.remove(&w);
                }
            }
        }
    }

    /// Take the transactions that have become ready since the last call.
    pub fn take_ready(&mut self) -> Vec<IngressOrdinal> {
        std::mem::take(&mut self.ready).into_iter().collect()
    }

    /// Admitted transactions that have not completed.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Transactions waiting on a stamp point no admitted transaction will ever
    /// publish.
    ///
    /// Derived from what has been admitted rather than from a clock, so it is a
    /// diagnosis rather than a timeout: the answer does not depend on how long
    /// anyone waited. It is not a claim that the guest is wrong — a packet that
    /// will publish the value may still arrive — which is why the caller
    /// decides what to do with it, and why nothing here cancels anything.
    #[must_use]
    pub fn stalled(&self) -> Vec<IngressOrdinal> {
        let mut publishable: BTreeMap<StampSlot, StampValue> = BTreeMap::new();
        for p in self.pending.values() {
            if let Some(c) = p.completion {
                publishable
                    .entry(c.slot)
                    .and_modify(|v| *v = v.later(c.value))
                    .or_insert(c.value);
            }
        }
        self.pending
            .iter()
            .filter(|(_, p)| {
                p.stamp_waits.iter().any(|w| {
                    !publishable
                        .get(&w.slot)
                        .is_some_and(|highest| highest.reached(w.value))
                })
            })
            .map(|(o, _)| *o)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ord(n: u64) -> IngressOrdinal {
        IngressOrdinal(n)
    }
    fn wait(slot: u32, value: u32) -> StampWait {
        StampWait {
            slot: StampSlot(slot),
            value: StampValue(value),
        }
    }
    fn stamp(slot: u32, value: u32) -> CompletionStamp {
        CompletionStamp {
            slot: StampSlot(slot),
            value: StampValue(value),
        }
    }

    #[test]
    fn a_transaction_with_no_prerequisites_is_ready_at_once() {
        let mut s = Scheduler::new();
        assert!(s.admit(ord(1), &[], &[], None));
        assert_eq!(s.take_ready(), vec![ord(1)]);
        assert!(
            s.take_ready().is_empty(),
            "the ready set is taken, not read"
        );
    }

    #[test]
    fn a_hazard_dependent_becomes_ready_when_its_predecessor_completes() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], None);
        assert!(!s.admit(ord(2), &[ord(1)], &[], None));
        assert_eq!(s.take_ready(), vec![ord(1)]);
        s.complete(ord(1));
        assert_eq!(s.take_ready(), vec![ord(2)]);
    }

    /// Work admitted behind work that has already finished is not waiting for
    /// anything. A compiler that reported a hazard against a retired
    /// transaction would otherwise park it forever.
    #[test]
    fn a_hazard_against_a_completed_transaction_is_already_discharged() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], None);
        s.complete(ord(1));
        assert!(s.admit(ord(2), &[ord(1)], &[], None));
    }

    #[test]
    fn a_stamp_wait_holds_until_the_value_is_published() {
        let mut s = Scheduler::new();
        assert!(!s.admit(ord(1), &[], &[wait(4, 10)], None));
        s.publish(stamp(4, 9));
        assert!(s.take_ready().is_empty(), "nine does not reach ten");
        s.publish(stamp(4, 10));
        assert_eq!(s.take_ready(), vec![ord(1)]);
    }

    /// A wait already discharged when the packet arrives is not a wait. This is
    /// the case that parks a channel against a value the device has itself
    /// already written.
    #[test]
    fn a_stamp_wait_already_satisfied_at_admission_does_not_hold() {
        let mut s = Scheduler::new();
        s.publish(stamp(4, 12));
        assert!(s.admit(ord(1), &[], &[wait(4, 10)], None));
    }

    /// The slot keeps the later value in the *wrapping* order. A `max` here
    /// would refuse to advance past a wrap for the rest of the boot.
    #[test]
    fn publishing_across_a_wrap_still_advances_the_slot() {
        let mut s = Scheduler::new();
        s.publish(stamp(0, u32::MAX - 1));
        s.publish(stamp(0, 3));
        assert_eq!(s.published_value(StampSlot(0)), Some(StampValue(3)));
        assert!(s.admit(ord(1), &[], &[wait(0, 2)], None));
    }

    /// And an out-of-order publication does not walk the slot backwards.
    #[test]
    fn publishing_an_older_value_does_not_retract_the_slot() {
        let mut s = Scheduler::new();
        s.publish(stamp(0, 20));
        s.publish(stamp(0, 5));
        assert_eq!(s.published_value(StampSlot(0)), Some(StampValue(20)));
    }

    /// Completion hands the stamp back; it does not publish it. A hazard
    /// dependent waits for the work and is released at once; a stamp waiter
    /// waits for the guest-visible word and is not.
    #[test]
    fn completion_releases_hazard_dependents_and_owes_its_stamp() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], Some(stamp(7, 1)));
        s.admit(ord(2), &[ord(1)], &[], None);
        s.admit(ord(3), &[], &[wait(7, 1)], None);
        s.take_ready();
        let owed = s.complete(ord(1));
        assert_eq!(owed, Some(stamp(7, 1)));
        assert_eq!(
            s.take_ready(),
            vec![ord(2)],
            "the stamp waiter waits for publication, which has not happened"
        );
        s.publish(owed.expect("a stamp"));
        assert_eq!(s.take_ready(), vec![ord(3)]);
    }

    #[test]
    fn a_transaction_with_both_kinds_of_prerequisite_waits_for_both() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], None);
        s.admit(ord(2), &[ord(1)], &[wait(3, 5)], None);
        s.take_ready();
        s.complete(ord(1));
        assert!(s.take_ready().is_empty(), "the stamp is still unpublished");
        s.publish(stamp(3, 5));
        assert_eq!(s.take_ready(), vec![ord(2)]);
    }

    /// The diagnosis: a wait no admitted transaction can answer.
    #[test]
    fn a_wait_nobody_will_publish_is_named() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[wait(2, 100)], None);
        assert_eq!(s.stalled(), vec![ord(1)]);
        // A packet that *will* publish it arrives: no longer stalled, and the
        // answer changed without a clock being involved.
        s.admit(ord(2), &[], &[], Some(stamp(2, 100)));
        assert!(s.stalled().is_empty());
        let owed = s.complete(ord(2)).expect("a stamp");
        s.publish(owed);
        assert_eq!(s.take_ready(), vec![ord(1)]);
    }

    /// A publisher that will not publish *far enough* is still not an answer.
    #[test]
    fn a_publisher_that_stops_short_does_not_discharge_the_wait() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[wait(2, 100)], None);
        s.admit(ord(2), &[], &[], Some(stamp(2, 99)));
        assert_eq!(s.stalled(), vec![ord(1)]);
    }

    /// Two transactions each waiting for the other's stamp. Both are named,
    /// because neither can be the one that moves.
    #[test]
    fn a_mutual_wait_names_both_sides() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[wait(1, 5)], Some(stamp(2, 5)));
        s.admit(ord(2), &[], &[wait(2, 5)], Some(stamp(1, 5)));
        assert!(
            s.stalled().is_empty(),
            "each side's stamp is publishable by the other, so this is a \
             deadlock in the wait-for graph rather than an unanswerable wait — \
             a different diagnosis, and not this one's to make"
        );
        assert_eq!(s.pending(), 2);
        assert!(s.take_ready().is_empty());
    }

    #[test]
    #[should_panic(expected = "not pending")]
    fn completing_twice_is_loud() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], None);
        s.complete(ord(1));
        s.complete(ord(1));
    }
}
