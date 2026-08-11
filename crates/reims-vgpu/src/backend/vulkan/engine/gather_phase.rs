//! Where a compute-gather dispatch's CPU cost goes, split by the mechanism that
//! would remove it.
//!
//! `draw_phase`'s `record_us` is where all of it lands, and one bar cannot
//! choose between four fixes. Ten interleaved driven macos-13 boots put the
//! whole of the gather's remaining regression there and nowhere else — the
//! matched pair `on3` / `off5`, ~27 000 draws each:
//!
//! ```text
//!                on3       off5
//! slot_us      47 468   111 275     -57 %   the GPU saving, which is real
//! record_us    79 682    48 055     +66 %   what pays for it
//! descriptors_us 8 119     6 333     +28 %   the draw's own, not the gather's
//! stage_us     32 198    28 902     +11 %
//! ```
//!
//! +31.6 ms a second over ~36 700 dispatches is **0.86 µs each**, and that is
//! the number that keeps [`crate::env::COMPUTE_GATHER`] switched off. A
//! command-buffer run-table arena and a recycled descriptor set already took it
//! down from ~1.05 µs; guessing which of what is left is the next ~0.8 is how a
//! session spends a day on `vkCmdBindPipeline` and finds it was never the cost.
//!
//! So the four candidates are timed apart:
//!
//! | part | what it is | what would remove it |
//! |---|---|---|
//! | `plan` | the `ScatterRun` vector and [`super::guest_scatter::build_gather_run_tables`] | building the table in place, from the copy regions, with no intermediate allocation |
//! | `stage` | the shared run-table arena — one `acquire_staging` and one `write_staging` per draw | nothing; it is already amortised over the draw's dispatches |
//! | `dset` | `alloc_scatter_descriptor_set` (a free-list pop) and `vkUpdateDescriptorSets` | a destination arena, which makes all three bindings constant so a draw needs one set instead of one per window |
//! | `record` | `vkCmdBindPipeline`, `vkCmdBindDescriptorSets`, `vkCmdPushConstants`, `vkCmdDispatch` | hoisting the pipeline bind out of the loop, and the same destination arena, which merges a draw's dispatches into one |
//!
//! Read them against the dispatch count and not against the draw count: a draw
//! gathers ~1.4 windows, so a per-draw reading understates each part by that
//! factor and a reader comparing one to `record_us` per draw would conclude the
//! parts do not sum.
//!
//! # This measures the planning, not the copy
//!
//! Every part here is CPU time spent *arranging* a copy the GPU makes later, in
//! the draw's own command buffer. None of it moves a byte, which is the whole
//! point of the rail — [`super::stage_phase`]'s `Gather` part states the same
//! caveat one layer up. A reading of zero here on a boot with a non-zero
//! `buffer_gather_dispatches` would mean the timer is not on the path, never
//! that the path is free.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The steps of planning and recording one draw's gather dispatches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Part {
    /// Turning copy regions into run tables. Per dispatch.
    Plan = 0,
    /// The shared run-table staging arena. Per draw, not per dispatch.
    Stage = 1,
    /// Taking a descriptor set and writing its three bindings. Per dispatch.
    Dset = 2,
    /// The command-buffer calls themselves. Per dispatch.
    Record = 3,
}

const PARTS: usize = 4;

/// Nanoseconds, per [`crate::observe::phase_clock`]. Tens of thousands of spans
/// a second is exactly the population a microsecond accumulator reports as
/// free.
static NS: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static N: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];

/// One window of the split, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GatherPhaseWindow {
    pub plan_us: u64,
    pub plan_n: u64,
    pub stage_us: u64,
    pub stage_n: u64,
    pub dset_us: u64,
    pub dset_n: u64,
    pub record_us: u64,
    pub record_n: u64,
}

/// Take and clear the window. `None` when no gather dispatched, so a boot with
/// the rail switched off costs no line — and a line's *presence* is what says
/// which arm a boot ran.
pub fn take_window() -> Option<GatherPhaseWindow> {
    let us =
        |p: Part| crate::observe::phase_clock::to_us(NS[p as usize].swap(0, Ordering::Relaxed));
    let n = |p: Part| N[p as usize].swap(0, Ordering::Relaxed);
    let w = GatherPhaseWindow {
        plan_us: us(Part::Plan),
        plan_n: n(Part::Plan),
        stage_us: us(Part::Stage),
        stage_n: n(Part::Stage),
        dset_us: us(Part::Dset),
        dset_n: n(Part::Dset),
        record_us: us(Part::Record),
        record_n: n(Part::Record),
    };
    (w.plan_n + w.stage_n + w.dset_n + w.record_n > 0).then_some(w)
}

/// Charges one step to one part, from `open` to `Drop`.
pub(crate) struct Span {
    part: Part,
    started: Instant,
}

impl Span {
    pub(crate) fn open(part: Part) -> Self {
        Self {
            part,
            started: Instant::now(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let slot = self.part as usize;
        NS[slot].fetch_add(
            crate::observe::phase_clock::charge_ns(self.started.elapsed()),
            Ordering::Relaxed,
        );
        N[slot].fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A window reports every part it was given and clears itself, so the next
    /// census second starts from zero rather than from the boot's total.
    #[test]
    fn a_window_takes_what_was_charged_and_leaves_nothing() {
        let _ = take_window();
        drop(Span::open(Part::Plan));
        drop(Span::open(Part::Plan));
        drop(Span::open(Part::Dset));
        let w = take_window().expect("three spans were charged");
        assert_eq!(w.plan_n, 2);
        assert_eq!(w.dset_n, 1);
        assert_eq!(w.stage_n, 0);
        assert_eq!(w.record_n, 0);
        assert_eq!(take_window(), None, "the window cleared itself");
    }

    /// A boot that never dispatches a gather publishes no line at all, which is
    /// how the census says which arm of [`crate::env::COMPUTE_GATHER`] ran
    /// without a second counter to disagree with.
    #[test]
    fn no_gather_publishes_no_window() {
        let _ = take_window();
        assert_eq!(take_window(), None);
    }
}
