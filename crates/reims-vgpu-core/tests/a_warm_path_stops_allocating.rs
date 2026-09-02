//! The architecture plan's "heap allocations per steady-state draw: 0", turned
//! into a number this suite asserts.
//!
//! # Why a structural zero needs an instrument
//!
//! A structural zero that nothing measures is a claim. The way it stops being
//! true is not a visible regression either: a helper that returns a `Vec` on a
//! per-access path costs one trip into the allocator per access and shows up as
//! a percent or two of drain duty spread evenly across a profile, which is
//! exactly the shape that survives review. Nobody bisects to it, because no
//! single line got slower.
//!
//! So the counts are asserted. A path that has to allocate says how many times
//! it does and why, and a change that adds a trip fails here rather than in a
//! profile six weeks later.
//!
//! # This is an integration test because the library forbids `unsafe`
//!
//! `reims-vgpu-core` carries `#![forbid(unsafe_code)]`, which is a claim about
//! the semantic model worth more than the convenience of measuring from
//! inside it. A `GlobalAlloc` implementation is unavoidably `unsafe`, and an
//! integration test is its own crate — so the instrument lives out here, the
//! model keeps its forbid, and the measurement is taken through the public API
//! a caller would use anyway.
//!
//! # The counter is per thread and off by default
//!
//! `#[global_allocator]` is program-wide and libtest runs tests in parallel, so
//! a process-wide counter would count whatever else happened to be running.
//! The count lives in thread-local storage, initialised at compile time so that
//! reading it cannot itself allocate and recurse. Only [`measure`] turns it on,
//! and only for its own thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use reims_vgpu_core::access::{
    AccessIntent, AccessKey, AccessMode, BackingId, ByteRange, ResourceKey,
};
use reims_vgpu_core::depend::DependencyGraph;
use reims_vgpu_core::identity::{ChannelId, IngressOrdinal};

thread_local! {
    /// Trips into the allocator on this thread since counting began. `const`
    /// initialisation matters: a lazily initialised thread-local allocates on
    /// first use, from inside the allocator.
    static COUNT: Cell<usize> = const { Cell::new(0) };
    static ON: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

// SAFETY: every method forwards to `System`, which is a correct allocator. The
// bookkeeping around it allocates nothing — `Cell<usize>` and `Cell<bool>` are
// const-initialised and have no destructor, so no thread-local registration
// happens on first use. `try_with` rather than `with`, because a thread tearing
// down may already have destroyed its storage and a panic inside the allocator
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

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `body` and return how many times it entered the allocator.
fn measure<T>(body: impl FnOnce() -> T) -> (T, usize) {
    COUNT.with(|c| c.set(0));
    ON.with(|c| c.set(true));
    let out = body();
    ON.with(|c| c.set(false));
    (out, COUNT.with(Cell::get))
}

/// The instrument first. A measurement that cannot see a known allocation is
/// worth nothing, and one that counts allocations the body did not make would
/// make every assertion below unfalsifiable.
#[test]
fn the_counter_sees_a_trip_and_sees_only_the_body() {
    let (v, one) = measure(|| {
        let mut v: Vec<u64> = Vec::with_capacity(4);
        v.push(1);
        v
    });
    assert_eq!(v.len(), 1);
    assert_eq!(one, 1, "one reservation, no growth");

    let (_, none) = measure(|| 1 + 1);
    assert_eq!(none, 0, "arithmetic does not allocate");

    let outside: Vec<u64> = (0..8).collect();
    let (_, still_none) = measure(|| 1 + 1);
    assert_eq!(still_none, 0, "work outside a measurement is not counted");
    assert_eq!(outside.len(), 8);

    let (_, grew) = measure(|| {
        let mut v: Vec<u8> = Vec::new();
        for n in 0..64u8 {
            v.push(n);
        }
        v
    });
    assert!(grew > 1, "growth re-enters the allocator: {grew}");
}

fn range(backing: u64, offset: u64, length: u64) -> AccessKey {
    AccessKey::Range(
        ResourceKey {
            backing: BackingId(backing),
            heap: None,
        },
        ByteRange { offset, length },
    )
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

/// One draw's worth of accesses admitted into a graph that has already seen
/// the same shape many times over.
///
/// # What "steady state" means here
///
/// A guest that has been drawing for a while re-touches the resources it
/// already touched: the same vertex buffer, the same uniform block, the same
/// sampled texture, frame after frame. The graph's indexes therefore already
/// have a bucket for each backing, already grown to the size that shape needs,
/// and admitting one more of the same draws no new capacity from anywhere.
/// That is the state the plan's zero is about — not the first admission
/// against a cold graph, which legitimately builds the buckets. A drain also
/// compacts, so the warm-up does: without it the indexes grow with the whole
/// history rather than with what is live, and "steady state" would be
/// measuring a graph that is still growing.
///
/// # The claim is that the count does not scale, not that it is zero
///
/// Two trips are structural and named in the code: the wait list `admit` hands
/// back, which its signature owns, and the per-ordinal index bucket a new
/// transaction needs. Neither is per access. So rather than pin an exact
/// figure — which would be a fact about `Vec`'s growth policy as much as about
/// this crate — the test admits two shapes, one four times the size of the
/// other, and asserts the count barely moves. A helper that allocated per
/// access or per comparison would fail that immediately, which is the
/// regression this exists to catch.
fn warm_graph_admission(accesses: &[AccessIntent]) -> usize {
    let mut graph = DependencyGraph::new();
    for n in 0..512u64 {
        let _ = graph.admit(IngressOrdinal(n), accesses);
        if n >= 4 {
            graph.retire(IngressOrdinal(n - 4));
        }
        if n % 32 == 0 {
            graph.compact();
        }
    }
    let (waits, allocations) = measure(|| graph.admit(IngressOrdinal(512), accesses));
    assert!(!waits.is_empty(), "a warm graph has something to wait for");
    allocations
}

fn buffer_accesses(count: u64) -> Vec<AccessIntent> {
    (0..count)
        .map(|n| {
            intent(
                range(n, 0, 256),
                if n % 3 == 0 {
                    AccessMode::Write
                } else {
                    AccessMode::Read
                },
            )
        })
        .collect()
}

#[test]
fn admitting_a_warm_draw_does_not_allocate_per_access() {
    let small = warm_graph_admission(&buffer_accesses(8));
    let large = warm_graph_admission(&buffer_accesses(32));
    assert!(
        small <= 4,
        "{small} trips for eight warm accesses; the structural ones are the \
         returned wait list and the per-ordinal index bucket"
    );
    assert!(
        large <= small + 2,
        "{large} trips for thirty-two accesses against {small} for eight: \
         the cost is scaling with the accesses"
    );
}

/// The same claim for the wider shape: a draw touching a heap-placed resource
/// is reachable through two indexes, so its candidate list is the one most
/// likely to be rebuilt per access.
#[test]
fn a_heap_placed_draw_does_not_allocate_per_candidate_list() {
    use reims_vgpu_core::access::HeapId;

    let heap = HeapId {
        id: 9,
        membership_generation: 1,
    };
    let placed = |count: u64| -> Vec<AccessIntent> {
        (0..count)
            .map(|n| {
                intent(
                    AccessKey::Range(
                        ResourceKey {
                            backing: BackingId(100 + n),
                            heap: Some(heap),
                        },
                        ByteRange {
                            offset: n * 1024,
                            length: 512,
                        },
                    ),
                    if n % 4 == 0 {
                        AccessMode::Write
                    } else {
                        AccessMode::Read
                    },
                )
            })
            .collect()
    };

    let small = warm_graph_admission(&placed(6));
    let large = warm_graph_admission(&placed(24));
    assert!(small <= 4, "{small} trips for six heap-placed accesses");
    assert!(
        large <= small + 2,
        "{large} trips for twenty-four heap-placed accesses against {small} for six"
    );
}

/// The read a warm frame takes: a replica that already holds the bytes.
///
/// This is the overwhelmingly common shape once a frame's resources have been
/// resident for a frame or two, and it is the shape the per-byte freshness
/// representation exists to answer cheaply. Answering it by computing the owed
/// set and finding it empty built a `RangeSet` per read; asking whether the
/// bytes are covered asks the same question and builds nothing.
#[test]
fn a_read_a_replica_already_holds_allocates_nothing() {
    use reims_vgpu_core::access::{ByteRange as Bytes, ContentVersion};
    use reims_vgpu_core::content::{ContentLedger, Replica};

    let backing = BackingId(1);
    let whole = Bytes {
        offset: 0,
        length: 1 << 20,
    };
    let mut ledger = ContentLedger::new();
    ledger.declare(backing, whole, Replica::GuestPages);
    // A frame's worth of scattered device-side production, so the freshness
    // set has real members rather than one.
    for n in 0..64u64 {
        ledger.write(
            backing,
            Bytes {
                offset: n * 4096,
                length: 2048,
            },
            Replica::DeviceOwned,
        );
    }

    let read = Bytes {
        offset: 8192,
        length: 512,
    };
    let (answer, allocations) =
        measure(|| ledger.transfer_for_read(backing, read, Replica::DeviceOwned));
    assert!(answer.is_none(), "the device wrote these bytes itself");
    assert_eq!(
        allocations, 0,
        "a read of bytes the replica already holds builds nothing"
    );

    // And the version query beside it, which a planner asks for the same read.
    let (version, none) = measure(|| ledger.version_of(backing, read));
    assert!(version.is_some());
    assert_eq!(
        none, 0,
        "asking which version covers a range builds nothing"
    );
    assert_ne!(ledger.newest_version(backing), Some(ContentVersion(0)));
}

/// The read that does owe a transfer still says what it owes, and the cost of
/// saying so does not scale with how fragmented the backing is.
#[test]
fn a_read_that_owes_a_transfer_pays_for_the_answer_and_not_for_the_search() {
    use reims_vgpu_core::access::ByteRange as Bytes;
    use reims_vgpu_core::content::{ContentLedger, Replica};

    let backing = BackingId(2);
    let build = |pieces: u64| {
        let mut ledger = ContentLedger::new();
        ledger.declare(
            backing,
            Bytes {
                offset: 0,
                length: 1 << 22,
            },
            Replica::GuestPages,
        );
        for n in 0..pieces {
            ledger.write(
                backing,
                Bytes {
                    offset: n * 4096,
                    length: 2048,
                },
                Replica::DeviceOwned,
            );
        }
        ledger
    };

    let read = Bytes {
        offset: 0,
        length: 1 << 16,
    };
    let mut few = build(16);
    let mut many = build(256);
    let (owed_few, cost_few) =
        measure(|| few.transfer_for_read(backing, read, Replica::GuestPages));
    let (owed_many, cost_many) =
        measure(|| many.transfer_for_read(backing, read, Replica::GuestPages));
    assert!(owed_few.is_some() && owed_many.is_some());
    assert!(
        cost_many <= cost_few + 4,
        "{cost_many} trips over 256 pieces against {cost_few} over 16: the \
         search is allocating per member"
    );
}
