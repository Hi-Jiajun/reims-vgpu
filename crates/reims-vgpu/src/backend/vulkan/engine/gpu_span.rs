//! How long the GPU spent executing a draw submission, from timestamps the
//! submission writes into its own command buffer.
//!
//! # The reading this closes
//!
//! [`super::draw_phase::Phase::Slot`] is the largest phase this device has: the
//! drain worker blocked in `begin_entry` because the ring slot it wants to reuse
//! has an unsignaled fence. Its own doc measured **314 491 µs/s** of it against
//! 2 525 µs/s of actual preparation, and roughly **425 µs per
//! `ring_retire_blocks`**. Every session since has read that column as "the GPU
//! is busy" and concluded the rail is GPU-bound — five CPU wins in a row bought
//! zero frames, and that is the explanation on offer.
//!
//! It was never measured. Before this module the device wrote GPU timestamps in
//! exactly one place, the composite readback copy, and nothing anywhere timed a
//! *draw* on the GPU's own clock. So `slot_us` is a wall-clock wait whose content
//! is unattributed, and it has two readings that call for opposite fixes:
//!
//! * the submission genuinely takes ~425 µs of GPU execution, in which case the
//!   lever is less GPU work per draw — the guest buffer gather at 427 000
//!   transfer regions a second, or the writeback's scatter;
//! * the submission executes in far less than that and the rest of the wait is
//!   *bubble* — queue scheduling, the fence's signal reaching the CPU, or the
//!   ring simply not being deep enough to keep work queued — in which case every
//!   byte-level saving is aimed at the wrong thing and the lever is submission
//!   shape.
//!
//! `RING_DEPTH`'s own doc already suspects the second ("It was submit/fence-
//! bubble-bound, not GPU-compute-bound") on a workload three years of changes
//! ago. Two timestamps settle it for the workload in front of us, and they settle
//! it without correlating two clocks: the delta is GPU ticks between two points
//! on the GPU's own timeline.
//!
//! # It is a tiling, not a sample
//!
//! `busy_us` and the derived leftover `slot_us - busy_us` sum to the wait, which
//! is the property that made the drain worker's CPU split answer unambiguously
//! and the property a third sampling point would not have. Read the pair; a
//! `busy_us` quoted alone says nothing, because the same 200 ms/s is "the GPU is
//! the wall" next to a 210 ms/s wait and "the wait is nearly all bubble" next to
//! one of 900.
//!
//! Two caveats belong to the reading rather than to the code:
//!
//! * **`busy_us` is per submission, and submissions overlap the wait.** The ring
//!   is [`super::pools::RING_DEPTH`] deep, so up to eight command buffers may be
//!   in flight while the worker waits on one fence. `busy_us` summed over a
//!   census second is the GPU's total occupancy from these submissions; it is
//!   compared against the *second*, not against `slot_us`, when the question is
//!   utilisation. `slot_us - busy_us` is the right comparison only for the
//!   question "was this slot's own work the wait", which is what
//!   `busy_max_us` and `ring_retire_blocks` speak to.
//! * **Timestamps have a cost.** Two per submission at ~2 000 submissions a
//!   second is ~4 000 a second, against the readback rail's existing three per
//!   composite, and both are far below the ~110 000 an inner per-draw split would
//!   need. It is small but it is not nothing, so [`crate::env::GPU_SPANS`] can
//!   take it out and an A/B that needs the absolute floor should.
//!
//! # Coverage is reported, because a zero here has three causes
//!
//! A `busy_us` of zero means the GPU did no work, or the host has no timestamp
//! support, or the arm/seal/read triple did not close — and the census must not
//! read the last two as the first. So the window carries `armed`, `sealed` and
//! `unread`:
//!
//! * `armed` counts command buffers that reset their queries and wrote the top
//!   stamp. Zero means the probe is not on the path at all.
//! * `sealed` counts those that also wrote the bottom stamp before the CB ended.
//!   `armed - sealed` is a submit path that ends a command buffer this module
//!   does not know about, which would read as a missing sample rather than as a
//!   wrong one.
//! * `unread` counts slots re-armed while a previous arming had not been read
//!   back. It must be zero by construction — `begin_entry` retires a slot before
//!   reusing it, and retiring is where the read happens — so a non-zero reading
//!   is a real defect in the ring's own ordering and not a tuning knob.

use std::sync::atomic::{AtomicU64, Ordering};

/// GPU nanoseconds accumulated across submissions in this census window.
static BUSY_NS: AtomicU64 = AtomicU64::new(0);
/// Submissions whose two stamps were both read back.
static READ: AtomicU64 = AtomicU64::new(0);
/// The largest single submission's GPU nanoseconds this window.
static MAX_NS: AtomicU64 = AtomicU64::new(0);
static ARMED: AtomicU64 = AtomicU64::new(0);
static SEALED: AtomicU64 = AtomicU64::new(0);
static UNREAD: AtomicU64 = AtomicU64::new(0);

/// One census window of GPU-side submission timing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuSpanWindow {
    /// GPU microseconds summed over every submission read back this window.
    pub busy_us: u64,
    /// The largest single submission, in GPU microseconds.
    pub busy_max_us: u64,
    /// Submissions both stamps were read from. The denominator for `busy_us`.
    pub read: u64,
    /// Command buffers that wrote the top stamp.
    pub armed: u64,
    /// Command buffers that also wrote the bottom stamp.
    pub sealed: u64,
    /// Slots re-armed before a previous arming was read. Zero by construction.
    pub unread: u64,
}

/// Take and clear the window. `None` when nothing armed, so a host without
/// timestamp support and a boot with [`crate::env::GPU_SPANS`] off cost no line
/// — and a line's presence is what says the probe ran.
pub fn take_window() -> Option<GpuSpanWindow> {
    let armed = ARMED.swap(0, Ordering::Relaxed);
    let w = GpuSpanWindow {
        busy_us: crate::observe::phase_clock::to_us(BUSY_NS.swap(0, Ordering::Relaxed)),
        busy_max_us: crate::observe::phase_clock::to_us(MAX_NS.swap(0, Ordering::Relaxed)),
        read: READ.swap(0, Ordering::Relaxed),
        armed,
        sealed: SEALED.swap(0, Ordering::Relaxed),
        unread: UNREAD.swap(0, Ordering::Relaxed),
    };
    (armed > 0).then_some(w)
}

/// A command buffer reset its query pair and wrote the top stamp.
pub(crate) fn note_armed() {
    ARMED.fetch_add(1, Ordering::Relaxed);
}

/// A command buffer wrote the bottom stamp before ending.
pub(crate) fn note_sealed() {
    SEALED.fetch_add(1, Ordering::Relaxed);
}

/// A slot was armed while a previous arming of the same slot had not been read.
pub(crate) fn note_unread() {
    UNREAD.fetch_add(1, Ordering::Relaxed);
}

/// One submission's GPU execution time, from the delta between its two stamps.
pub(crate) fn note_busy_ns(ns: u64) {
    BUSY_NS.fetch_add(ns, Ordering::Relaxed);
    READ.fetch_add(1, Ordering::Relaxed);
    MAX_NS.fetch_max(ns, Ordering::Relaxed);
}

/// Where a ring slot's arming stands, so a read cannot invent a sample out of a
/// query the GPU never wrote.
///
/// A three-state enum rather than two bools because "armed but not sealed" and
/// "sealed" are the two states a read must tell apart, and a pair of bools admits
/// a fourth combination that means nothing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum SlotSpan {
    /// No stamp written since this slot was last read.
    #[default]
    Idle,
    /// Top stamp written; the command buffer is still recording.
    Armed,
    /// Both stamps written; the delta is readable once the fence signals.
    Sealed,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window reports what was charged and clears itself, so a census second
    /// starts from zero rather than from the boot's running total.
    #[test]
    fn a_window_takes_what_was_charged_and_leaves_nothing() {
        let _ = take_window();
        note_armed();
        note_armed();
        note_sealed();
        note_busy_ns(3_000);
        note_busy_ns(5_000);
        let w = take_window().expect("two command buffers armed");
        assert_eq!(w.armed, 2);
        assert_eq!(w.sealed, 1);
        assert_eq!(w.read, 2);
        assert_eq!(w.busy_max_us, crate::observe::phase_clock::to_us(5_000));
        assert_eq!(w.unread, 0);
        assert_eq!(take_window(), None, "the window cleared itself");
    }

    /// A boot where nothing armed publishes no line, which is how the census says
    /// "this host writes no timestamps" without a second counter to disagree.
    #[test]
    fn nothing_armed_publishes_no_window() {
        let _ = take_window();
        assert_eq!(take_window(), None);
    }

    /// `busy_us` is a sum and `busy_max_us` is a high-water: two submissions of
    /// equal length and one long one next to one short one must not read the same,
    /// because only the second says a single submission is the wall.
    #[test]
    fn the_sum_and_the_high_water_are_different_readings() {
        let _ = take_window();
        note_armed();
        note_busy_ns(1_000_000);
        note_busy_ns(1_000_000);
        let even = take_window().expect("armed");
        note_armed();
        note_busy_ns(1_900_000);
        note_busy_ns(100_000);
        let skewed = take_window().expect("armed");
        assert_eq!(even.busy_us, skewed.busy_us, "the same total");
        assert!(
            skewed.busy_max_us > even.busy_max_us,
            "{skewed:?} vs {even:?}"
        );
    }
}
