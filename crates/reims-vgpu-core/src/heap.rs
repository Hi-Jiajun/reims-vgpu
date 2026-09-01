//! Heaps: one piece of storage, several resources placed in it, and a lifetime
//! that outlives the guest's delete.
//!
//! # A placement is a coordinate, not a new piece of memory
//!
//! A heap-allocated resource does not own storage. It occupies a window of the
//! heap's storage at an offset the guest chose, and two windows the guest chose
//! to overlap *are* the same bytes — that is what a heap is for. So a placement
//! does not get its own [`BackingId`]. It gets the heap's, and a byte range
//! within it.
//!
//! That single decision is what makes partial aliasing come out right for free.
//! [`crate::access::AccessKey::Range`] already conflicts on overlapping ranges of one backing,
//! so two resources sharing the front half of a heap window conflict on the
//! front half and not on the rest, with no alias closure to compute, no second
//! storage identity to keep consistent, and nothing that has to be told which
//! resources were declared aliasable. A design that minted a backing per
//! placement would have had to reconstruct all of that, and would have had to
//! be *told* about the overlap by something that measured it.
//!
//! # The arithmetic has one checked place
//!
//! A resource's own coordinates are relative to its placement, and the graph's
//! are relative to the heap. Converting between them is an addition, and an
//! addition performed at call sites is the class of slip that reaches a
//! neighbouring resource's bytes with the GPU's write access. There is exactly
//! one conversion, [`HeapPlacement::within`], it is checked against the
//! placement's own length, and there is no public field a caller could add to
//! something itself: `region` is the heap-relative window and reading it gives
//! no license to offset into it.
//!
//! # Deletion is the namespace rule, applied to storage
//!
//! [`crate::namespace`] establishes that a delete stops resolution and does not
//! stop work. A heap adds a second holder: its own allocations. Guest code
//! releases a heap while resources placed in it are still alive, and the heap's
//! storage is those resources' storage — freeing it on the delete frees memory
//! a live resource is reading.
//!
//! So the delete is accepted, the heap stops resolving, and the storage is
//! handed back at whichever comes last: the delete, or the removal of the final
//! allocation. [`Retirement`] is that answer, and it is `#[must_use]` because a
//! caller that dropped it either frees storage under a live resource or leaks
//! the heap. The wait has no bound and no eviction: there is no lawful loss
//! here, and a heap that outlives its delete for a long time is a guest that
//! kept its allocations, not a leak to be capped.
//!
//! # Membership is recorded, and it is not the aliasing question
//!
//! Placing or removing a resource advances the heap's membership generation,
//! and [`Heaps::membership`] stamps the generation into the [`HeapId`] a
//! command records. That says which set a `useHeap` record was written
//! against. It is deliberately not what decides whether two accesses meet —
//! see [`HeapId::same_heap`], which exists because a resource that never moved
//! must keep meeting a declaration written before its neighbour arrived.

use crate::access::{BackingId, ByteRange, HeapId, ResourceKey};
use crate::identity::ResourceId;
use std::collections::{HashMap, HashSet};

/// Why a heap operation did not happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// No live heap has this number. It was never created, or it was deleted:
    /// a delete stops the number resolving, and after it there is nothing here
    /// to tell the two apart with that a later `create` would not invalidate.
    NoSuchHeap { heap: u64 },
    /// The heap number is already in use by a live heap. A guest that wants it
    /// back deletes first; silently replacing would strand the allocations
    /// placed in the previous occupant.
    HeapExists { heap: u64 },
    /// The window does not fit inside the heap, or its end overflows.
    OutOfHeap {
        heap: u64,
        offset: u64,
        length: u64,
        heap_length: u64,
    },
    /// The window does not fit inside the placement, or its end overflows.
    OutOfPlacement {
        offset: u64,
        length: u64,
        placement_length: u64,
    },
    /// This resource is already placed in this heap. A second placement would
    /// leave the first one's compiled coordinates naming bytes nothing agrees
    /// about.
    AlreadyPlaced { resource: ResourceId },
    /// This resource is not placed in the storage the placement names.
    NotPlaced { resource: ResourceId },
    /// The placement names storage no heap and no retirement holds. Either it
    /// was already removed, or it was minted by a heap whose number has since
    /// been reused for different storage.
    StaleStorage { backing: BackingId },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoSuchHeap { .. } => "heap_no_such_heap",
            Self::HeapExists { .. } => "heap_exists",
            Self::OutOfHeap { .. } => "heap_out_of_heap",
            Self::OutOfPlacement { .. } => "heap_out_of_placement",
            Self::AlreadyPlaced { .. } => "heap_already_placed",
            Self::NotPlaced { .. } => "heap_not_placed",
            Self::StaleStorage { .. } => "heap_stale_storage",
        }
    }
}

/// What a delete or a removal left of the heap's storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "storage nothing reports as free is either freed under a live resource or leaked"]
pub enum Retirement {
    /// The heap is still named, or still holds allocations. Its storage stays.
    Held { allocations: usize, named: bool },
    /// The heap's number is gone and its last allocation is gone. Nothing can
    /// reach the storage again, so it may be torn down now.
    StorageFree { backing: BackingId },
}

/// A resource's window of a heap.
///
/// Carries the heap's [`BackingId`] rather than one of its own, because that is
/// what it is: the resource occupies the heap's storage. `region` is
/// heap-relative, which is the coordinate system the dependency graph compares
/// in, and [`HeapPlacement::within`] is the only way to turn a resource-local
/// window into one.
///
/// The backing travels with it for a second reason. A heap number may be
/// deleted and reused, and a holder that presented the *number* back would
/// remove its allocation from whatever now answers to it.
/// [`Heaps::remove`] takes the placement, so a stale one refuses instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapPlacement {
    pub heap: u64,
    pub backing: BackingId,
    pub region: ByteRange,
}

impl HeapPlacement {
    /// The access key for this placement at a membership generation.
    #[must_use]
    pub const fn key(self, membership: HeapId) -> ResourceKey {
        ResourceKey {
            backing: self.backing,
            heap: Some(membership),
        }
    }

    /// A window of this resource, in the heap's coordinates.
    ///
    /// `offset` and `length` are relative to the resource, which is how every
    /// command that names part of a buffer expresses itself. The result is
    /// relative to the heap, which is what the dependency graph compares.
    ///
    /// # Errors
    ///
    /// If the window runs past the end of the placement, or its end overflows.
    /// Both are the same defect — an access naming a neighbouring resource's
    /// bytes — and neither is clamped, because a clamped access silently
    /// executes against memory the guest did not name.
    pub fn within(self, offset: u64, length: u64) -> Result<ByteRange, Refusal> {
        if offset
            .checked_add(length)
            .is_none_or(|end| end > self.region.length)
        {
            return Err(Refusal::OutOfPlacement {
                offset,
                length,
                placement_length: self.region.length,
            });
        }
        Ok(ByteRange {
            // Cannot overflow: `offset` is at most the placement's length, and
            // the placement's end was checked against the heap's when it was
            // created.
            offset: self.region.offset.saturating_add(offset),
            length,
        })
    }

    /// The whole resource, in the heap's coordinates.
    #[must_use]
    pub const fn whole(self) -> ByteRange {
        self.region
    }
}

#[derive(Debug)]
struct Heap {
    backing: BackingId,
    length: u64,
    membership_generation: u64,
    placements: HashMap<ResourceId, ByteRange>,
}

/// Storage whose heap number is gone, kept alive by the allocations still in it.
#[derive(Debug)]
struct Retiring {
    allocations: HashSet<ResourceId>,
}

/// One session generation's heaps.
///
/// `heaps` holds only live ones: a delete removes the number, and storage the
/// delete could not free moves to `retiring`, where nothing can name it and
/// only the placements already handed out can retire it. Recreating the number
/// therefore cannot reach the previous occupant's storage — the two do not
/// share a key.
#[derive(Debug, Default)]
pub struct Heaps {
    heaps: HashMap<u64, Heap>,
    retiring: HashMap<BackingId, Retiring>,
}

impl Heaps {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a heap over a piece of storage.
    ///
    /// # Errors
    ///
    /// If a live heap already has this number.
    pub fn create(&mut self, heap: u64, backing: BackingId, length: u64) -> Result<(), Refusal> {
        if self.heaps.contains_key(&heap) {
            return Err(Refusal::HeapExists { heap });
        }
        self.heaps.insert(
            heap,
            Heap {
                backing,
                length,
                // Advances on every placement and every removal, so a command
                // records which membership set it was written against.
                membership_generation: 0,
                placements: HashMap::new(),
            },
        );
        Ok(())
    }

    /// The heap's identity at this command point.
    ///
    /// # Errors
    ///
    /// If no live heap has this number.
    pub fn membership(&self, heap: u64) -> Result<HeapId, Refusal> {
        let h = self.live(heap)?;
        Ok(HeapId {
            id: heap,
            membership_generation: h.membership_generation,
        })
    }

    /// The storage a live heap is over.
    ///
    /// # Errors
    ///
    /// If no live heap has this number.
    pub fn backing_of(&self, heap: u64) -> Result<BackingId, Refusal> {
        Ok(self.live(heap)?.backing)
    }

    /// Place a resource in the heap.
    ///
    /// # Errors
    ///
    /// If no live heap has this number, the window does not fit, or this
    /// resource is already placed here.
    pub fn place(
        &mut self,
        heap: u64,
        resource: ResourceId,
        offset: u64,
        length: u64,
    ) -> Result<HeapPlacement, Refusal> {
        let h = self
            .heaps
            .get_mut(&heap)
            .ok_or(Refusal::NoSuchHeap { heap })?;
        if offset.checked_add(length).is_none_or(|end| end > h.length) {
            return Err(Refusal::OutOfHeap {
                heap,
                offset,
                length,
                heap_length: h.length,
            });
        }
        if h.placements.contains_key(&resource) {
            return Err(Refusal::AlreadyPlaced { resource });
        }
        let region = ByteRange { offset, length };
        h.placements.insert(resource, region);
        h.membership_generation += 1;
        Ok(HeapPlacement {
            heap,
            backing: h.backing,
            region,
        })
    }

    /// The placement of a resource already in the heap.
    ///
    /// # Errors
    ///
    /// If no live heap has this number, or the resource is not placed in it.
    pub fn placement(&self, heap: u64, resource: ResourceId) -> Result<HeapPlacement, Refusal> {
        let h = self.live(heap)?;
        let region = *h
            .placements
            .get(&resource)
            .ok_or(Refusal::NotPlaced { resource })?;
        Ok(HeapPlacement {
            heap,
            backing: h.backing,
            region,
        })
    }

    /// Remove a resource from the storage its placement names, and say what
    /// that left of the storage.
    ///
    /// Takes the placement rather than the heap number, so it reaches the
    /// storage the caller was actually given — whether that heap is still named
    /// or is retiring behind a reused number.
    ///
    /// # Errors
    ///
    /// If nothing holds that storage any more, or the resource is not placed in
    /// it.
    pub fn remove(
        &mut self,
        placement: HeapPlacement,
        resource: ResourceId,
    ) -> Result<Retirement, Refusal> {
        if let Some(h) = self.heaps.get_mut(&placement.heap) {
            if h.backing == placement.backing {
                if h.placements.remove(&resource).is_none() {
                    return Err(Refusal::NotPlaced { resource });
                }
                h.membership_generation += 1;
                return Ok(Retirement::Held {
                    allocations: h.placements.len(),
                    named: true,
                });
            }
        }
        let Some(r) = self.retiring.get_mut(&placement.backing) else {
            return Err(Refusal::StaleStorage {
                backing: placement.backing,
            });
        };
        if !r.allocations.remove(&resource) {
            return Err(Refusal::NotPlaced { resource });
        }
        if r.allocations.is_empty() {
            self.retiring.remove(&placement.backing);
            return Ok(Retirement::StorageFree {
                backing: placement.backing,
            });
        }
        Ok(Retirement::Held {
            allocations: r.allocations.len(),
            named: false,
        })
    }

    /// Delete the heap.
    ///
    /// The number stops resolving immediately and becomes available again. The
    /// storage is returned now only if nothing is placed in it; otherwise the
    /// last [`Heaps::remove`] returns it, with no bound on how long that takes
    /// and no eviction — a heap outliving its delete is a guest that kept its
    /// allocations, and there is no lawful loss to substitute for waiting.
    ///
    /// # Errors
    ///
    /// If no live heap has this number.
    pub fn delete(&mut self, heap: u64) -> Result<Retirement, Refusal> {
        let Some(h) = self.heaps.remove(&heap) else {
            return Err(Refusal::NoSuchHeap { heap });
        };
        if h.placements.is_empty() {
            return Ok(Retirement::StorageFree { backing: h.backing });
        }
        let allocations: HashSet<ResourceId> = h.placements.into_keys().collect();
        let held = allocations.len();
        self.retiring.insert(h.backing, Retiring { allocations });
        Ok(Retirement::Held {
            allocations: held,
            named: false,
        })
    }

    /// How many allocations are placed in a live heap.
    #[must_use]
    pub fn allocations(&self, heap: u64) -> usize {
        self.heaps.get(&heap).map_or(0, |h| h.placements.len())
    }

    /// Whether this storage is still held by a heap or a retirement.
    #[must_use]
    pub fn holds_storage(&self, backing: BackingId) -> bool {
        self.retiring.contains_key(&backing) || self.heaps.values().any(|h| h.backing == backing)
    }

    /// Every live heap number, sorted, so a teardown's order is a property of
    /// the heaps and not of a hash seed.
    #[must_use]
    pub fn live_heaps(&self) -> Vec<u64> {
        let mut out: Vec<u64> = self.heaps.keys().copied().collect();
        out.sort_unstable();
        out
    }

    /// How many pieces of storage are outliving their heap number.
    #[must_use]
    pub fn retiring_storage(&self) -> usize {
        self.retiring.len()
    }

    fn live(&self, heap: u64) -> Result<&Heap, Refusal> {
        self.heaps.get(&heap).ok_or(Refusal::NoSuchHeap { heap })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessKey, HeapId};
    use crate::identity::{ObjectListRef, SlotGeneration};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration::default().next(),
        }
    }

    fn heaps_with(length: u64) -> Heaps {
        let mut h = Heaps::new();
        h.create(7, BackingId(100), length).expect("first create");
        h
    }

    /// The reason a placement is a coordinate and not a piece of memory: two
    /// windows the guest chose to overlap are the same bytes, and the ordinary
    /// range conflict already says so. Nothing had to be told they alias.
    #[test]
    fn two_placements_alias_exactly_where_their_windows_meet() {
        let mut heaps = heaps_with(4096);
        let a = heaps.place(7, res(1), 0, 512).expect("a fits");
        let b = heaps.place(7, res(2), 256, 512).expect("b fits");
        let c = heaps.place(7, res(3), 2048, 512).expect("c fits");
        let m = heaps.membership(7).expect("live");
        assert_eq!(a.backing, b.backing, "one heap is one piece of storage");
        let key = |p: HeapPlacement, r: ByteRange| AccessKey::Range(p.key(m), r);
        assert!(
            key(a, a.whole()).may_alias(key(b, b.whole())),
            "overlapping windows are the same bytes"
        );
        assert!(
            !key(a, a.whole()).may_alias(key(c, c.whole())),
            "disjoint windows are not"
        );
        // And the overlap is exact, not whole-resource: the front of `a` and
        // the back of `b` never meet.
        let front_of_a = a.within(0, 128).expect("inside a");
        let back_of_b = b.within(384, 128).expect("inside b");
        assert!(!key(a, front_of_a).may_alias(key(b, back_of_b)));
        // While the tail of `a` and the stretch of `b` sitting under it do.
        let tail_of_a = a.within(384, 128).expect("inside a");
        let under_it = b.within(128, 128).expect("inside b");
        assert_eq!(tail_of_a, under_it, "the same heap bytes, named two ways");
        assert!(key(a, tail_of_a).may_alias(key(b, under_it)));
    }

    /// The one conversion between a resource's coordinates and the heap's, and
    /// the reason it is the only one: an addition at a call site reaches the
    /// neighbour's bytes.
    #[test]
    fn a_window_of_a_placement_cannot_leave_it() {
        let mut heaps = heaps_with(4096);
        let p = heaps.place(7, res(1), 1024, 256).expect("fits");
        assert_eq!(
            p.within(16, 32),
            Ok(ByteRange {
                offset: 1040,
                length: 32
            })
        );
        assert_eq!(
            p.within(0, 256),
            Ok(ByteRange {
                offset: 1024,
                length: 256
            })
        );
        assert_eq!(
            p.within(240, 32),
            Err(Refusal::OutOfPlacement {
                offset: 240,
                length: 32,
                placement_length: 256,
            }),
            "one byte past the end is not clamped to the end"
        );
        assert_eq!(
            p.within(8, u64::MAX),
            Err(Refusal::OutOfPlacement {
                offset: 8,
                length: u64::MAX,
                placement_length: 256,
            }),
            "and an overflowing end is the same defect, not a wrap"
        );
    }

    #[test]
    fn a_placement_cannot_leave_the_heap() {
        let mut heaps = heaps_with(1024);
        assert_eq!(
            heaps.place(7, res(1), 768, 512),
            Err(Refusal::OutOfHeap {
                heap: 7,
                offset: 768,
                length: 512,
                heap_length: 1024,
            })
        );
        assert_eq!(
            heaps.place(7, res(1), 1, u64::MAX),
            Err(Refusal::OutOfHeap {
                heap: 7,
                offset: 1,
                length: u64::MAX,
                heap_length: 1024,
            })
        );
        assert_eq!(
            heaps.allocations(7),
            0,
            "a refused placement is not a member"
        );
        assert_eq!(
            heaps.membership(7).map(|m| m.membership_generation),
            Ok(0),
            "and did not advance membership"
        );
    }

    #[test]
    fn placing_and_removing_both_advance_membership() {
        let mut heaps = heaps_with(4096);
        let generation = |h: &Heaps| h.membership(7).expect("live").membership_generation;
        assert_eq!(generation(&heaps), 0);
        let a = heaps.place(7, res(1), 0, 64).expect("fits");
        assert_eq!(generation(&heaps), 1);
        heaps.place(7, res(2), 64, 64).expect("fits");
        assert_eq!(generation(&heaps), 2);
        assert_eq!(
            heaps.remove(a, res(1)),
            Ok(Retirement::Held {
                allocations: 1,
                named: true
            })
        );
        assert_eq!(generation(&heaps), 3);
    }

    #[test]
    fn a_resource_is_placed_once_and_removed_once() {
        let mut heaps = heaps_with(4096);
        let a = heaps.place(7, res(1), 0, 64).expect("fits");
        assert_eq!(
            heaps.place(7, res(1), 128, 64),
            Err(Refusal::AlreadyPlaced { resource: res(1) }),
            "a second placement would leave the first one's coordinates unowned"
        );
        assert_eq!(heaps.placement(7, res(1)), Ok(a));
        assert_eq!(
            heaps.placement(7, res(2)),
            Err(Refusal::NotPlaced { resource: res(2) })
        );
        assert_eq!(
            heaps.remove(a, res(1)),
            Ok(Retirement::Held {
                allocations: 0,
                named: true
            })
        );
        assert_eq!(
            heaps.remove(a, res(1)),
            Err(Refusal::NotPlaced { resource: res(1) })
        );
    }

    /// The whole reason heaps have a lifetime module of their own: the guest
    /// releases the heap while resources placed in it are alive, and the heap's
    /// storage *is* their storage.
    #[test]
    fn deleting_a_heap_with_allocations_hands_the_storage_to_them() {
        let mut heaps = heaps_with(4096);
        let a = heaps.place(7, res(1), 0, 64).expect("fits");
        let b = heaps.place(7, res(2), 64, 64).expect("fits");
        assert_eq!(
            heaps.delete(7),
            Ok(Retirement::Held {
                allocations: 2,
                named: false
            })
        );
        assert_eq!(
            heaps.membership(7),
            Err(Refusal::NoSuchHeap { heap: 7 }),
            "the delete stopped the number resolving"
        );
        assert!(
            heaps.holds_storage(BackingId(100)),
            "and did not free storage"
        );
        assert_eq!(
            heaps.remove(a, res(1)),
            Ok(Retirement::Held {
                allocations: 1,
                named: false
            })
        );
        assert!(heaps.holds_storage(BackingId(100)));
        assert_eq!(
            heaps.remove(b, res(2)),
            Ok(Retirement::StorageFree {
                backing: BackingId(100)
            }),
            "the last allocation completes the handoff"
        );
        assert!(!heaps.holds_storage(BackingId(100)));
        assert_eq!(heaps.retiring_storage(), 0);
    }

    #[test]
    fn deleting_an_empty_heap_frees_its_storage_now() {
        let mut heaps = heaps_with(4096);
        assert_eq!(
            heaps.delete(7),
            Ok(Retirement::StorageFree {
                backing: BackingId(100)
            })
        );
        assert_eq!(heaps.delete(7), Err(Refusal::NoSuchHeap { heap: 7 }));
        assert!(!heaps.holds_storage(BackingId(100)));
    }

    /// A heap number is a guest object reference and gets reused. The retiring
    /// storage behind it must be unreachable from the number, or a holder of an
    /// old placement removes its allocation from the new heap and the old
    /// storage never retires.
    #[test]
    fn a_reused_number_cannot_reach_the_previous_heaps_storage() {
        let mut heaps = heaps_with(4096);
        let old = heaps.place(7, res(1), 0, 64).expect("fits");
        assert_eq!(
            heaps.delete(7),
            Ok(Retirement::Held {
                allocations: 1,
                named: false
            })
        );
        heaps
            .create(7, BackingId(200), 4096)
            .expect("the number is available again");
        let new = heaps.place(7, res(1), 0, 64).expect("fits");
        assert_ne!(old.backing, new.backing);
        assert_eq!(
            heaps.remove(old, res(1)),
            Ok(Retirement::StorageFree {
                backing: BackingId(100)
            }),
            "the old placement retires the old storage"
        );
        assert_eq!(
            heaps.allocations(7),
            1,
            "and left the new heap's membership alone"
        );
        assert_eq!(
            heaps.remove(new, res(1)),
            Ok(Retirement::Held {
                allocations: 0,
                named: true
            }),
            "the new heap is still named, so its storage is still held"
        );
        assert!(heaps.holds_storage(BackingId(200)));
    }

    #[test]
    fn a_live_number_cannot_be_created_over() {
        let mut heaps = heaps_with(4096);
        assert_eq!(
            heaps.create(7, BackingId(200), 4096),
            Err(Refusal::HeapExists { heap: 7 })
        );
    }

    #[test]
    fn a_placement_naming_storage_nothing_holds_refuses() {
        let mut heaps = heaps_with(4096);
        let stranger = HeapPlacement {
            heap: 9,
            backing: BackingId(999),
            region: ByteRange {
                offset: 0,
                length: 64,
            },
        };
        assert_eq!(
            heaps.remove(stranger, res(1)),
            Err(Refusal::StaleStorage {
                backing: BackingId(999)
            })
        );
    }

    /// Membership stamps which set a record was written against; it is not what
    /// decides whether two accesses meet. `access::HeapId::same_heap` owns that,
    /// and this is the heap side of the same claim.
    #[test]
    fn a_key_recorded_at_one_membership_still_meets_a_later_declaration() {
        let mut heaps = heaps_with(4096);
        let a = heaps.place(7, res(1), 0, 64).expect("fits");
        let early = heaps.membership(7).expect("live");
        heaps.place(7, res(2), 2048, 64).expect("fits");
        let late = heaps.membership(7).expect("live");
        assert_ne!(early.membership_generation, late.membership_generation);
        assert!(
            AccessKey::Range(a.key(early), a.whole()).may_alias(AccessKey::Heap(HeapId { ..late }))
        );
    }

    #[test]
    fn every_refusal_has_its_own_slug() {
        let all = [
            Refusal::NoSuchHeap { heap: 1 },
            Refusal::HeapExists { heap: 1 },
            Refusal::OutOfHeap {
                heap: 1,
                offset: 0,
                length: 0,
                heap_length: 0,
            },
            Refusal::OutOfPlacement {
                offset: 0,
                length: 0,
                placement_length: 0,
            },
            Refusal::AlreadyPlaced { resource: res(1) },
            Refusal::NotPlaced { resource: res(1) },
            Refusal::StaleStorage {
                backing: BackingId(1),
            },
        ];
        let mut slugs: Vec<&str> = all.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count);
    }
}
