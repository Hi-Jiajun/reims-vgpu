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
//! Entries die with the thing they describe, through the guest's own events:
//! `CmdDeleteObject` retires one ref, and a new object list for a task retires
//! that task's whole namespace because the refs are indices into the list that
//! just went away. There is no capacity, no eviction and no age: a ref the
//! guest still holds is a ref this must still answer for, and if a guest holds a
//! million of them then a million entries is what correctness costs.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::model::DeviceState;
use crate::runtime::decode::resource::ListObjectEntry;

/// Times a freed slot was answered from what the ref last resolved to.
static SERVED: AtomicU64 = AtomicU64::new(0);
/// Times a freed slot had nothing remembered — the guest named a ref this
/// device had never resolved, which is a different story and not this one.
static UNKNOWN: AtomicU64 = AtomicU64::new(0);
/// Live entries at the last census, so the level survives the swap above.
static LIVE: AtomicU64 = AtomicU64::new(0);

/// Keep what `ref_` resolved to in `task_id`.
pub(super) fn remember(state: &DeviceState, task_id: u32, ref_: u32, entry: ListObjectEntry) {
    if let Ok(mut map) = state.retired_object_entries.lock() {
        map.insert((task_id, ref_), entry);
        LIVE.store(map.len() as u64, Ordering::Relaxed);
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

/// The guest deleted this object: the ref may be handed to another one next.
pub fn retire_ref(state: &DeviceState, task_id: u32, ref_: u32) {
    if let Ok(mut map) = state.retired_object_entries.lock() {
        map.remove(&(task_id, ref_));
        LIVE.store(map.len() as u64, Ordering::Relaxed);
    }
}

/// The guest published a new object list for this task, so every ref in it is
/// an index into a list this device no longer reads.
pub fn retire_task(state: &DeviceState, task_id: u32) {
    if let Ok(mut map) = state.retired_object_entries.lock() {
        map.retain(|(task, _), _| *task != task_id);
        LIVE.store(map.len() as u64, Ordering::Relaxed);
    }
}

/// Live entries, and how the freed-slot answers split.
///
/// On the census because the whole point is a hit rate that is a statement
/// about guest lifetimes rather than about this device: `served` is packets
/// this device used to drop, and `unknown` is the residue that still has no
/// answer. A `served` that climbs while `live` does not is refs churning, which
/// is the shape the ref namespace is expected to have.
pub fn census() -> Option<String> {
    let served = SERVED.swap(0, Ordering::Relaxed);
    let unknown = UNKNOWN.swap(0, Ordering::Relaxed);
    let live = LIVE.load(Ordering::Relaxed);
    if live == 0 && served == 0 && unknown == 0 {
        return None;
    }
    Some(format!(
        "retired_entry live={live} served={served} unknown={unknown}"
    ))
}
