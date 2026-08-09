//! Read an empty object-list slot again later, and say whether it fills.
//!
//! # The question this answers, and the one it refuses to ask
//!
//! `ListMiss::SlotEmpty` is the whole of macos-26's lost draws — the task
//! exists and is active, its object list is registered where the guest said,
//! the ref is inside the declared count, the guest read *succeeds*, and the
//! sixteen bytes are zero. Two readings fit that, with opposite fixes:
//!
//! 1. **A race this device should tolerate.** The guest referenced the object
//!    in a packet it submitted before publishing the slot. Then the answer is to
//!    defer the packet the way an unfinished AIR translation is deferred, and
//!    dropping the draw is the defect.
//! 2. **This device looked in the wrong list.** The object lives under another
//!    task and the ref was meant to resolve against it.
//!
//! The obvious discriminator — "does another live task hold a real object at
//! this slot?" — was built, and it answered *yes* to every miss of a boot. That
//! is not a verdict for reading 2. **Every task registers its object list at the
//! same `pfn = 1`** and the refs in play are small and dense, so "somebody else
//! has something at slot 3" is close to a tautology on a busy guest. Banding the
//! claimant count against the live task count showed 82 % of the answer sitting
//! in the uninformative band.
//!
//! This module asks the question that no other task's address can confound:
//! **re-read the same slot, in the same list, later.** If it becomes non-zero
//! the guest published late and reading 1 is right. If it is still zero when the
//! task dies, reading 1 is dead and reading 2 (or a fourth thing) is live.
//!
//! # Why there is no timeout
//!
//! The terminal verdict is the guest's own task teardown, not a wall clock. A
//! horizon would have to come from somewhere, and the deferral machinery this
//! feeds has none to borrow: `ChildPacketDisposition::Deferred` leaves the
//! packet at the FIFO head and it is retried every drain until the translation
//! lands. So a watch here ends when the guest ends the task, and
//! [`Verdict::fill_us`] reports the age at which a fill was actually seen —
//! which is the number that says whether deferring is affordable, rather than a
//! number chosen in advance that decides it.
//!
//! # What it costs
//!
//! One quiet probe read per *distinct* watched `(task, ref)` per drain tranche,
//! and nothing at all when nothing is watched — which is every rail except
//! macos-26, where rails 11 through 15 record zero `list_miss_slot_empty` in a
//! driven boot.

use crate::model::DeviceState;
use crate::runtime::host::HostMemory;

use super::{list_entry_or_miss, ListLookup, ListMiss};

/// One slot being watched, keyed by the `(task, ref)` the guest named.
type WatchKey = (u32, u32);

/// When the watch started, in both clocks it is read against.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Watch {
    /// For [`Verdict::fill_us`]. `crate::observe::elapsed_us`.
    recorded_us: u64,
    /// The index of the sweep that closes the tranche the miss happened in.
    ///
    /// A watch becomes due **after** that sweep, not at it, so the first re-read
    /// is genuinely a later tranche rather than the tail of the one that
    /// produced the miss. Without the distinction a slot published microseconds
    /// after the packet would read as filled at an age of zero, and a slot the
    /// guest never publishes would be indistinguishable from it on the first
    /// sample.
    closing_sweep: u64,
}

/// The watch set, with its capacity in the type that carries it.
///
/// The bound is real and so is the overflow: a ledger that silently stopped
/// admitting would report "every miss was watched" while watching a prefix, and
/// the resulting `filled`/`still_empty` ratio would be a statement about the
/// first N refs of a boot wearing the name of a statement about all of them.
/// [`Ledger::admit`] therefore returns whether the watch was taken and the
/// caller counts the refusals.
///
/// The capacity is not derived from the contract, because the contract does not
/// bound it — the guest declares an object list of 2^20 slots and may name any
/// of them. It is sized against the measured population instead: a driven
/// macos-26 boot produces ~170 misses spread over **eight** distinct refs across
/// four tasks, so a thousand distinct live watches is three orders of magnitude
/// of headroom, and `slot_recheck_dropped` is what says if that ever stops being
/// true.
struct Ledger {
    watches: std::collections::HashMap<WatchKey, Watch>,
    /// How many sweeps have run. A watch admitted now belongs to the tranche the
    /// next one closes, which is what [`Watch::closing_sweep`] records.
    sweep: u64,
}

impl Ledger {
    const CAPACITY: usize = 1024;

    fn new() -> Self {
        Self {
            watches: std::collections::HashMap::new(),
            sweep: 0,
        }
    }

    /// Start watching `key`, or report that the ledger is full.
    ///
    /// A repeat miss on a slot already watched is the *same* watch — the guest
    /// re-issuing a packet against a ref it still has not published — so it
    /// keeps the original `recorded_us` and does not consume a second entry.
    /// Overwriting it would reset the age and make every fill look instant.
    fn admit(&mut self, key: WatchKey, now_us: u64) -> bool {
        if self.watches.contains_key(&key) {
            return true;
        }
        if self.watches.len() >= Self::CAPACITY {
            return false;
        }
        self.watches.insert(
            key,
            Watch {
                recorded_us: now_us,
                closing_sweep: self.sweep.saturating_add(1),
            },
        );
        true
    }

    /// Open a sweep and hand back the watches whose own tranche has already
    /// closed.
    ///
    /// The strict `<` is the whole of "read it again **later**": a watch
    /// admitted during tranche *N* is skipped by the sweep that closes *N* and
    /// read for the first time by the one that closes *N+1*. Without it the
    /// first re-read is the same instant as the miss, so a slot the guest
    /// publishes a microsecond afterwards is indistinguishable from one it never
    /// publishes.
    fn begin_sweep(&mut self) -> Vec<(WatchKey, Watch)> {
        self.sweep = self.sweep.saturating_add(1);
        let sweep_now = self.sweep;
        self.watches
            .iter()
            .filter(|(_, w)| w.closing_sweep < sweep_now)
            .map(|(k, w)| (*k, *w))
            .collect()
    }
}

fn ledger() -> &'static std::sync::Mutex<Ledger> {
    use std::sync::{Mutex, OnceLock};
    static LEDGER: OnceLock<Mutex<Ledger>> = OnceLock::new();
    LEDGER.get_or_init(|| Mutex::new(Ledger::new()))
}

/// How one watch ended.
///
/// Every arm is terminal except [`Self::StillEmpty`], which is the only reason a
/// watch survives a sweep.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Verdict {
    /// The slot became a real entry. **Reading 1**: the guest published after
    /// naming the ref, and the draw was dropped for a race rather than for a
    /// missing object.
    Filled,
    /// Still zero, and the task is still alive to publish it.
    StillEmpty,
    /// The task went away — deleted, deactivated, or its object list reset by a
    /// `define_task` — with the slot never published. Its own band, because
    /// folding it into a "never filled" count would credit the guest's teardown
    /// as evidence against reading 1.
    TaskGone,
    /// The re-read hit one of the other five checks. Should not happen for a
    /// slot that read cleanly once; a firing here means the list moved under the
    /// watch and the sample is not comparable.
    Unreadable,
}

impl Verdict {
    fn route(self) -> &'static str {
        match self {
            Self::Filled => "slot_recheck_filled",
            Self::StillEmpty => "slot_recheck_still_empty",
            Self::TaskGone => "slot_recheck_task_gone",
            Self::Unreadable => "slot_recheck_unreadable",
        }
    }
}

/// Classify one re-read.
///
/// Split from the sweep so the mapping from the eight-way [`ListMiss`] onto the
/// four verdicts is testable without a guest — it is the only part that can be
/// wrong in a way that changes what the next session believes, and getting
/// `NoObjectList` onto the wrong side of it would turn every `define_task`
/// reissue into evidence that the guest never publishes.
fn verdict_of(read: Result<(), ListMiss>) -> Verdict {
    match read {
        Ok(()) => Verdict::Filled,
        Err(ListMiss::SlotEmpty) => Verdict::StillEmpty,
        Err(ListMiss::NoTask | ListMiss::TaskInactive | ListMiss::NoObjectList) => {
            Verdict::TaskGone
        }
        Err(
            ListMiss::RefBeyondList
            | ListMiss::AddressOverflow
            | ListMiss::Unreadable
            | ListMiss::Undecodable,
        ) => Verdict::Unreadable,
    }
}

/// Record that the guest named `ref_` against `task_id` and the slot was zero.
///
/// Called only from the `Named` arm of the lookup — a probe misses on every task
/// that does not own the ref, which is how it finds the one that does, and
/// watching those would fill the ledger with the search.
pub(super) fn note_slot_empty(task_id: u32, ref_: u32) {
    let now = crate::observe::elapsed_us();
    let admitted = ledger()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .admit((task_id, ref_), now);
    if !admitted {
        crate::runtime::drain::note_store_route("slot_recheck_dropped");
    }
}

/// Re-read every watched slot that was recorded before this sweep.
///
/// Runs at the tail of a drain tranche, with the same `state` and host the
/// lookup used. Returns early with the lock untouched when nothing is watched,
/// which is every tranche on every rail that does not produce the miss.
pub fn sweep<M: HostMemory>(state: &DeviceState, host: &M) {
    let due = ledger()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .begin_sweep();
    if due.is_empty() {
        return;
    }

    let now = crate::observe::elapsed_us();
    let mut retired: Vec<WatchKey> = Vec::new();
    for (key, watch) in due {
        let (task_id, ref_) = key;
        let read = list_entry_or_miss(state, host, task_id, ref_, ListLookup::Probe).map(|_| ());
        let verdict = verdict_of(read);
        if verdict == Verdict::StillEmpty {
            continue;
        }
        crate::runtime::drain::note_store_route(verdict.route());
        if verdict == Verdict::Filled {
            // The age at which the fill was *seen*, which is an upper bound on
            // the age at which it happened — the slot is only sampled once per
            // tranche. It is quoted against `slot_recheck_filled` from the same
            // census window, so the mean is `fill_us / filled`.
            crate::runtime::drain::note_store_route_us(
                "slot_recheck_fill_us",
                now.saturating_sub(watch.recorded_us),
            );
        }
        retired.push(key);
    }

    if retired.is_empty() {
        return;
    }
    let mut guard = ledger().lock().unwrap_or_else(|e| e.into_inner());
    for key in retired {
        guard.watches.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four-way collapse of the eight-way miss is the whole judgement this
    /// module makes; a `NoObjectList` landing on `StillEmpty` would report every
    /// `define_task` reissue as the guest declining to publish.
    #[test]
    fn a_recheck_verdict_separates_a_live_empty_slot_from_a_torn_down_task() {
        assert_eq!(verdict_of(Ok(())), Verdict::Filled);
        assert_eq!(verdict_of(Err(ListMiss::SlotEmpty)), Verdict::StillEmpty);
        for gone in [ListMiss::NoTask, ListMiss::TaskInactive, ListMiss::NoObjectList] {
            assert_eq!(verdict_of(Err(gone)), Verdict::TaskGone, "{gone:?}");
        }
        for broken in [
            ListMiss::RefBeyondList,
            ListMiss::AddressOverflow,
            ListMiss::Unreadable,
            ListMiss::Undecodable,
        ] {
            assert_eq!(verdict_of(Err(broken)), Verdict::Unreadable, "{broken:?}");
        }
    }

    /// A repeat miss on a watched slot must not restart its clock. The guest
    /// re-issues the packet every frame it still wants the object, so an
    /// overwrite would report every fill at roughly one tranche of age however
    /// long the guest actually took.
    #[test]
    fn a_repeat_miss_keeps_the_first_sighting_and_costs_no_capacity() {
        let mut ledger = Ledger::new();
        assert!(ledger.admit((7, 3), 1_000));
        assert!(ledger.admit((7, 3), 9_000));
        assert_eq!(ledger.watches.len(), 1);
        assert_eq!(ledger.watches[&(7, 3)].recorded_us, 1_000);
    }

    /// The bound refuses rather than truncating, and says so to its caller.
    #[test]
    fn a_full_ledger_refuses_a_new_watch_instead_of_evicting_one() {
        let mut ledger = Ledger::new();
        for i in 0..Ledger::CAPACITY as u32 {
            assert!(ledger.admit((1, i), 0), "admit {i}");
        }
        assert!(!ledger.admit((1, Ledger::CAPACITY as u32), 0));
        assert_eq!(ledger.watches.len(), Ledger::CAPACITY);
        // The already-watched slot still resolves: a full ledger stops taking
        // new work, it does not stop tracking what it has.
        assert!(ledger.admit((1, 0), 0));
    }

    /// A watch may not be re-read by the sweep that is running when it is
    /// recorded, or "still empty" would be asserted of an instant.
    #[test]
    fn a_watch_is_not_due_in_the_sweep_that_closes_the_tranche_it_was_recorded_in() {
        let mut ledger = Ledger::new();
        // Tranche 1: the miss is recorded, then the tranche's own sweep runs.
        ledger.admit((2, 5), 0);
        assert!(ledger.begin_sweep().is_empty());
        // Tranche 2's sweep is the first that may read it, and it hands back the
        // watch as recorded — the age it reports is measured from the miss.
        let recorded = ledger.watches[&(2, 5)];
        assert_eq!(ledger.begin_sweep(), vec![((2, 5), recorded)]);
        // And it stays due until something retires it — a slot the guest never
        // publishes must keep being asked until the task dies.
        assert_eq!(ledger.begin_sweep().len(), 1);
    }

    /// A miss recorded by a *later* tranche must not be answered by the sweep
    /// that is already running for the earlier ones.
    #[test]
    fn a_sweep_does_not_pick_up_a_watch_admitted_after_it_began() {
        let mut ledger = Ledger::new();
        ledger.admit((1, 1), 0);
        assert!(ledger.begin_sweep().is_empty());
        ledger.admit((1, 2), 0);
        // Only the older watch: `(1, 2)` was admitted during this sweep's own
        // tranche and has not yet had a later one.
        assert_eq!(
            ledger.begin_sweep().into_iter().map(|(k, _)| k).collect::<Vec<_>>(),
            vec![(1, 1)]
        );
    }
}
