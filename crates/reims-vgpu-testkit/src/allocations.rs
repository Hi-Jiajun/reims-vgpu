//! Counting the trips a path makes into the heap allocator.
//!
//! # Why this is a shared instrument
//!
//! The architecture plan lists "heap allocations per steady-state draw" among
//! its structural zeros — a required value, not a target. A structural zero
//! that nothing measures is a claim, and the way it stops being true is not a
//! visible regression: a helper that returns a `Vec` on a per-access or
//! per-draw path costs one `malloc` per call and shows up as a percent or two
//! of drain duty spread evenly across a profile. No single line got slower, so
//! nobody bisects to it.
//!
//! The semantic model and the executor rail both have paths a warm frame takes
//! many times, so both need the same instrument, and a second copy of a
//! `GlobalAlloc` is a second thing to get subtly wrong.
//!
//! # Using it
//!
//! A test crate installs the allocator and measures around the path:
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOCATOR: reims_vgpu_testkit::allocations::Counting =
//!     reims_vgpu_testkit::allocations::Counting::new();
//!
//! let (answer, trips) = reims_vgpu_testkit::allocations::measure(|| warm_path());
//! ```
//!
//! It has to be an integration test rather than a unit test wherever the crate
//! under measurement forbids `unsafe`, which `reims-vgpu-core` does — a claim
//! about the semantic model worth more than the convenience of measuring from
//! inside it.
//!
//! # The counter is per thread and off by default
//!
//! `#[global_allocator]` is program-wide and libtest runs tests in parallel, so
//! a process-wide counter would count whatever else happened to be running. The
//! count lives in thread-local storage, initialised at compile time so that
//! reading it cannot itself allocate and recurse. Only [`measure`] turns it on,
//! and only for its own thread; every other test pays one relaxed thread-local
//! read per allocation.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// Trips into the allocator on this thread since counting began. `const`
    /// initialisation matters: a lazily initialised thread-local allocates on
    /// first use, from inside the allocator.
    static COUNT: Cell<usize> = const { Cell::new(0) };
    static ON: Cell<bool> = const { Cell::new(false) };
}

/// A `System` allocator that counts, for the thread and the region a
/// [`measure`] asks about.
pub struct Counting;

impl Counting {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for Counting {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: every method forwards to `System`, which is a correct allocator. The
// bookkeeping around it allocates nothing — `Cell<usize>` and `Cell<bool>` are
// const-initialised and have no destructor, so no thread-local registration
// happens on first use. `try_with` rather than `with`, because a thread tearing
// down may already have destroyed its storage, and a panic inside the allocator
// aborts the process.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A `Vec` growing is an allocation by the measure that matters here:
        // it is a trip into the allocator and the bytes may move.
        bump();
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        System.alloc_zeroed(layout)
    }
}

fn bump() {
    if ON.try_with(Cell::get).unwrap_or(false) {
        let _ = COUNT.try_with(|c| c.set(c.get() + 1));
    }
}

/// Run `body` and return how many times it entered the allocator.
///
/// Not nestable, and it does not need to be: the counter is a single depth,
/// and a measurement inside a measurement would mean the inner path is not the
/// thing being measured.
pub fn measure<T>(body: impl FnOnce() -> T) -> (T, usize) {
    COUNT.with(|c| c.set(0));
    ON.with(|c| c.set(true));
    let out = body();
    ON.with(|c| c.set(false));
    (out, COUNT.with(Cell::get))
}
