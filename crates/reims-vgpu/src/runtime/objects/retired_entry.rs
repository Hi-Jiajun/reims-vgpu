//! The entry a ref resolved to, kept until the guest retires that ref.
//!
//! # Why a ref is not a stable name
//!
//! A ref is an **index into a densely packed array, and a freed index is
//! reused**. The guest allocates one by taking the first slot whose eight-byte
//! word is zero, hands that index back as the ref, and writes the object's
//! twelve-byte entry at `index * 12` in the list this device reads. Nothing
//! about that scheme makes a ref unique over time: the same small number names
//! one object, then no object, then a different one.
//!
//! [`super::list_entry`] resolves a ref by reading that array **now**, on the
//! drain thread, which is not when the packet naming it was submitted. Between
//! the two the guest may have freed the slot, and a freed slot reads as zero —
//! [`super::ListMiss::SlotEmpty`]. The packet is then dropped for naming an
//! object that was alive when it named it.
//!
//! # What this remembers, and what it deliberately does not
//!
//! On every successful named resolve the entry is kept here under
//! `(task, ref)`. It is consulted **only** when the live read comes back
//! `SlotEmpty`, and never in place of a read that succeeds. That ordering is
//! the whole safety argument: a slot the guest has already reassigned reads
//! non-empty and the *live* entry wins, so this can never answer for an object
//! the ref no longer names. It answers only in the gap between a free and a
//! reuse, which is exactly the window the packet was submitted in.
//!
//! It is therefore not a cache in front of the read. The read still happens
//! every time and still decides whenever it can.
//!
//! # Lifetime
//!
//! Entries die with the thing they describe, through the guest's own events.
//! A ref is an index into an object list, so the list *is* the namespace and
//! every event that ends one retires it whole: `CmdSetObjectList` publishes a
//! replacement, `define_task` redefines the task's page table, and
//! `delete_task` tears the task down. There is no capacity, no eviction and no
//! age: a ref the guest still holds is a ref this must still answer for, and if
//! a guest holds a million of them then a million entries is what correctness
//! costs.
//!
//! **`CmdDeleteObject` is deliberately not one of those events**, which is the
//! opposite of what it looks like. Its ref is in the *serializer's per-kind*
//! space; this map is keyed by the kernel object-list ref, and
//! `drain::apply_delete_object`'s own doc records the measurement that equal
//! integers across those spaces are unrelated — 1 988 deletes in a driven boot,
//! none naming a live object-table ref, 22 colliding with one under a different
//! task. Retiring on that number evicts whichever entry happens to share the
//! integer, which drops exactly the packets this map exists to keep.
//!
//! # What this cannot fix
//!
//! A ref freed and then **reused** before the drain reaches the packet that
//! named it resolves to the *new* object, because the live read wins and the
//! live read is the new object. That is unchanged from before this store
//! existed and is not reachable from here: the fix is a generation captured at
//! submission time, which the wire does not currently carry. What is reachable
//! is the gap between the free and the reuse, and that is the whole of what
//! this serves.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::model::DeviceState;
use crate::runtime::decode::resource::ListObjectEntry;

/// Times a freed slot was answered from what the ref last resolved to.
static SERVED: AtomicU64 = AtomicU64::new(0);
/// Times a freed slot had nothing remembered — the guest named a ref this
/// device had never resolved, which is a different story and not this one.
static UNKNOWN: AtomicU64 = AtomicU64::new(0);

/// Keep what `ref_` resolved to in `task_id`.
///
/// On the resolve success path, so it does the least it can: one lock, one
/// insert, and **no census store**. The level is counted at census time from
/// the map itself — see [`census`] — because a `store(map.len())` here would put
/// a second atomic write on every named resolve for a number nothing reads
/// until the tranche ends.
pub(super) fn remember(state: &DeviceState, task_id: u32, ref_: u32, entry: ListObjectEntry) {
    if let Ok(mut map) = state.retired_object_entries.lock() {
        map.insert((task_id, ref_), entry);
    }
}

/// What `ref_` last resolved to, for a slot now reading empty.
pub(super) fn recall(state: &DeviceState, task_id: u32, ref_: u32) -> Option<ListObjectEntry> {
    let found = state
        .retired_object_entries
        .lock()
        .ok()
        .and_then(|map| map.get(&(task_id, ref_)).copied());
    if found.is_some() {
        SERVED.fetch_add(1, Ordering::Relaxed);
    } else {
        UNKNOWN.fetch_add(1, Ordering::Relaxed);
    }
    found
}

/// Live entries for this device, for the census.
pub(super) fn live(state: &DeviceState) -> u64 {
    state
        .retired_object_entries
        .lock()
        .map_or(0, |map| map.len() as u64)
}

/// The object list these refs index has gone away, so none of them names
/// anything: a replacement list, a redefined task, or a deleted one.
pub fn retire_task(state: &DeviceState, task_id: u32) {
    if let Ok(mut map) = state.retired_object_entries.lock() {
        map.retain(|(task, _), _| *task != task_id);
    }
}

/// Live entries, and how the freed-slot answers split.
///
/// On the census because the whole point is a hit rate that is a statement
/// about guest lifetimes rather than about this device: `served` is packets
/// this device used to drop, and `unknown` is the residue that still has no
/// answer. A `served` that climbs while `live` does not is refs churning, which
/// is the shape the ref namespace is expected to have.
///
/// `live` is read from the device the caller names, not from a process-wide
/// latch. The map is per-`DeviceState` precisely because task ids and refs are
/// small dense integers every device reuses, and a level stored into one static
/// by whichever device wrote last reported neither device's — while `served`
/// and `unknown` summed across both, so the three were not a consistent set.
/// They still sum across devices; on a boot there is one, and a two-device
/// reading has to be taken per device or not at all.
pub fn census(state: &DeviceState) -> Option<String> {
    let served = SERVED.swap(0, Ordering::Relaxed);
    let unknown = UNKNOWN.swap(0, Ordering::Relaxed);
    let live = live(state);
    if live == 0 && served == 0 && unknown == 0 {
        return None;
    }
    Some(format!(
        "retired_entry live={live} served={served} unknown={unknown}"
    ))
}
