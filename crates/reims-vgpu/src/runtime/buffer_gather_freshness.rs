//! How many of the draw-time buffer gathers land on bytes the guest has not
//! said it changed.
//!
//! # The one number the cache design turns on
//!
//! `backend::vulkan::engine::pools::buffer_gather_working_set` measured ~20 800
//! gathers a second over ~1 900 distinct windows on a driven macos-13
//! sustained-animation boot, so **91 % of them re-assemble a window this device
//! already assembled**. That is a statement about *keys*. A content cache turns
//! on whether the *bytes* moved in between, and nothing has measured that.
//!
//! This crosses the two. For each bind that takes the gather rail it compares
//! the owning buffer object's [`super::buffer_write_gen`] stamp against the one
//! this window carried the last time it was gathered, and reports the split:
//!
//! | route | meaning |
//! |---|---|
//! | `bgf_quiet` / `bgf_quiet_kb` | the guest declared no write to this object since the last gather — a hit a declaration-invalidated cache would have served |
//! | `bgf_wrote` / `bgf_wrote_kb` | the guest declared a write, so the copy was owed |
//! | `bgf_first` | no previous gather of this window to compare against |
//! | `bgf_dropped` | the tracking map was full, so this bind is not in the split |
//!
//! **Read `quiet_rate` beside `buffer_write_gen_bump`, always.** A reader here
//! compares a stamp taken on a `(task, reference)` pair at a draw-time bind
//! against a generation the decoder recorded under a `(task, object)` pair from
//! a validity record. If those two turn out to be different namespaces then no
//! comparison ever moves and this reports ~100 % quiet — a false positive in the
//! direction that licenses a cache serving stale bytes. A boot reading
//! `quiet_rate=1.000` beside `buffer_write_gen_bump=0` has measured a wiring
//! fault and not a workload.
//!
//! Read `bgf_quiet` against the two together and not against the gather count:
//! `bgf_first` is a compulsory miss no cache size removes, and folding it in
//! understates the achievable rate by the working set's own turnover.
//!
//! # It measures a ceiling, not a licence
//!
//! Nothing here decides a skip, and a high reading is not on its own permission
//! to build the cache. A cache invalidated this way would be trusting that the
//! guest's `writeInvalidates` and exec-table quads are a **complete** account of
//! CPU writes to a buffer's bytes. A surface's equivalent claim is not complete
//! — which is exactly why `runtime::gather_witness` carries a hypervisor half as
//! well — and the buffer case has not been tested either way.
//!
//! What the split does settle is whether it is worth testing. No cache
//! invalidated by declarations can beat `bgf_quiet`, so a low reading closes the
//! design outright and a high one says go and establish the soundness.
//!
//! # Why the hypervisor witness is not the instrument here
//!
//! [`super::gather_witness`] answers the same question soundly for the sampled
//! rails, and its `MAX_TRACKED_WINDOWS` of 256 is a **harvest** bound rather
//! than a memory one: `reims_vgpu_dirty_harvest` walks every page of every armed
//! set on the BQL thread at each register write that hands the device work. The
//! buffer working set is ~1 900 windows of ~38 pages, so arming it there would
//! put ~72 000 pages into a walk the whole VM waits on. Measuring with it would
//! change what is being measured.

use std::collections::HashMap;

use super::buffer_write_gen::BufferWriteStamp;

/// Which window a bind names, at the granularity `bound_buffers` resolves.
///
/// The same four fields that key a held resolution — a reference bound at two
/// offsets is two windows and a cache would hold two buffers, so counting them
/// as one would report a hit rate for a cache nobody could build. See
/// [`super::bound_buffers`] on why the offset is the dominant axis rather than
/// an inert field.
type WindowKey = (u32, u32, u64, Option<u64>);

#[derive(Default)]
struct Window {
    /// The stamp each window carried when it was last gathered.
    ///
    /// Survives across census seconds, unlike the counters: a window gathered
    /// once a second must still be comparable, and clearing this each second
    /// would report every one of them as `bgf_first` forever.
    last: HashMap<WindowKey, BufferWriteStamp>,
    quiet: u64,
    quiet_kb: u64,
    wrote: u64,
    wrote_kb: u64,
    first: u64,
    dropped: u64,
}

impl Window {
    /// The most windows tracked at once.
    ///
    /// Above the ~1 900 the working-set census measured, so the reading is not
    /// censored by its own instrument, and `dropped` says if that stops holding.
    /// This map costs one stamp per window and arms nothing on the host, which
    /// is what lets it sit an order of magnitude above `gather_witness`'s cap.
    const CAPACITY: usize = 16384;

    fn note(&mut self, key: WindowKey, stamp: BufferWriteStamp, bytes: u64) {
        let kb = bytes / 1024;
        match self.last.get(&key) {
            Some(&earlier) if stamp.quiet_since(earlier) => {
                self.quiet += 1;
                self.quiet_kb = self.quiet_kb.saturating_add(kb);
            }
            Some(_) => {
                self.wrote += 1;
                self.wrote_kb = self.wrote_kb.saturating_add(kb);
            }
            None if self.last.len() >= Self::CAPACITY => {
                self.dropped += 1;
                return;
            }
            None => self.first += 1,
        }
        self.last.insert(key, stamp);
    }

    /// The line, or `None` when nothing gathered this second.
    ///
    /// Clears the counters and **keeps** `last`: the counters are a per-window
    /// rate and the stamps are the state the next second compares against.
    fn take(&mut self) -> Option<String> {
        let asked = self.quiet + self.wrote + self.first + self.dropped;
        if asked == 0 {
            return None;
        }
        let comparable = self.quiet + self.wrote;
        let rate = if comparable == 0 {
            0.0
        } else {
            self.quiet as f64 / comparable as f64
        };
        let line = format!(
            "buffer_gather_freshness quiet={} quiet_kb={} wrote={} wrote_kb={} first={} \
             dropped={} tracked={} quiet_rate={rate:.3} \
             (of the gathers with a previous gather of the same window to compare against, the \
              share the guest declared no write to; the ceiling on any cache invalidated by the \
              guest's own declarations, and not a licence — see the module doc)",
            self.quiet,
            self.quiet_kb,
            self.wrote,
            self.wrote_kb,
            self.first,
            self.dropped,
            self.last.len(),
        );
        self.quiet = 0;
        self.quiet_kb = 0;
        self.wrote = 0;
        self.wrote_kb = 0;
        self.first = 0;
        self.dropped = 0;
        Some(line)
    }
}

fn window() -> &'static std::sync::Mutex<Window> {
    use std::sync::{Mutex, OnceLock};
    static WINDOW: OnceLock<Mutex<Window>> = OnceLock::new();
    WINDOW.get_or_init(|| Mutex::new(Window::default()))
}

/// Record one draw-time buffer bind that took the zero-copy rail.
pub fn note_bind(
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    extent_cap: Option<u64>,
    stamp: BufferWriteStamp,
    bytes: u64,
) {
    window().lock().unwrap_or_else(|e| e.into_inner()).note(
        (task_id, buffer_ref, offset, extent_cap),
        stamp,
        bytes,
    );
}

/// Drain the second's split into a census line.
pub fn census() -> Option<String> {
    window().lock().unwrap_or_else(|e| e.into_inner()).take()
}

#[cfg(test)]
mod tests {
    use super::super::buffer_write_gen::BufferWriteGens;
    use super::*;

    const KEY: WindowKey = (1, 2, 0, None);

    /// The first sight of a window is a compulsory miss and must not be counted
    /// as either side of the split — folding it in understates the achievable
    /// rate by the working set's turnover.
    #[test]
    fn a_windows_first_gather_is_neither_quiet_nor_written() {
        let mut w = Window::default();
        w.note(KEY, BufferWriteStamp::default(), 4096);
        let line = w.take().expect("a bind happened");
        assert!(line.contains("first=1"), "{line}");
        assert!(line.contains("quiet=0"), "{line}");
        assert!(line.contains("wrote=0"), "{line}");
    }

    /// A second gather with no declared write in between is the hit a
    /// declaration-invalidated cache would have served.
    #[test]
    fn a_repeat_with_no_declared_write_is_quiet() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        w.note(KEY, g.stamp(1, 2), 8192);
        w.note(KEY, g.stamp(1, 2), 8192);
        let line = w.take().expect("binds happened");
        assert!(line.contains("quiet=1"), "{line}");
        assert!(line.contains("quiet_kb=8"), "{line}");
        assert!(line.contains("quiet_rate=1.000"), "{line}");
    }

    /// A declared write between two gathers is a copy that was owed, and it
    /// must land on the other side of the split.
    #[test]
    fn a_declared_write_between_two_gathers_is_not_quiet() {
        let mut g = BufferWriteGens::default();
        let mut w = Window::default();
        w.note(KEY, g.stamp(1, 2), 1024);
        g.note_write(1, 2);
        w.note(KEY, g.stamp(1, 2), 1024);
        let line = w.take().expect("binds happened");
        assert!(line.contains("wrote=1"), "{line}");
        assert!(line.contains("quiet=0"), "{line}");
        assert!(line.contains("quiet_rate=0.000"), "{line}");
    }

    /// The stamps outlive the census second. A window gathered once a second
    /// would otherwise read as `first` forever and the rate would be undefined
    /// for exactly the population a cache has to hold longest.
    #[test]
    fn the_stamps_survive_a_census_second_even_though_the_counters_do_not() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        w.note(KEY, g.stamp(1, 2), 1024);
        w.take().expect("a bind happened");
        w.note(KEY, g.stamp(1, 2), 1024);
        let line = w.take().expect("a bind happened");
        assert!(line.contains("quiet=1"), "{line}");
        assert!(line.contains("first=0"), "{line}");
    }

    /// One reference bound at two offsets is two windows: a cache would hold two
    /// buffers, so counting them as one would report a rate for a cache nobody
    /// could build.
    #[test]
    fn one_reference_at_two_offsets_is_two_windows() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        w.note((1, 2, 0, None), g.stamp(1, 2), 1024);
        w.note((1, 2, 4096, None), g.stamp(1, 2), 1024);
        let line = w.take().expect("binds happened");
        assert!(line.contains("first=2"), "{line}");
    }

    /// Past the capacity a new window is dropped rather than evicting one whose
    /// stamp is still wanted, and the line says how many.
    #[test]
    fn a_new_window_past_the_capacity_is_dropped_and_named() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        for i in 0..(Window::CAPACITY as u64) {
            w.note((1, 2, i, None), g.stamp(1, 2), 1024);
        }
        w.note((9, 9, 9, None), g.stamp(1, 2), 1024);
        let line = w.take().expect("binds happened");
        assert!(line.contains("dropped=1"), "{line}");
        assert!(
            line.contains(&format!("tracked={}", Window::CAPACITY)),
            "{line}"
        );
    }

    /// A window already tracked still counts past the capacity, so the split
    /// stays exact for the population it can see.
    #[test]
    fn a_tracked_window_still_counts_when_the_map_is_full() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        for i in 0..(Window::CAPACITY as u64) {
            w.note((1, 2, i, None), g.stamp(1, 2), 1024);
        }
        w.take().expect("binds happened");
        w.note((1, 2, 0, None), g.stamp(1, 2), 1024);
        let line = w.take().expect("a bind happened");
        assert!(line.contains("quiet=1"), "{line}");
        assert!(line.contains("dropped=0"), "{line}");
    }

    /// Nothing gathered is no line, so an idle second does not publish a zero
    /// that reads like a measured rate.
    #[test]
    fn an_idle_second_publishes_nothing() {
        let mut w = Window::default();
        assert!(w.take().is_none());
    }
}
