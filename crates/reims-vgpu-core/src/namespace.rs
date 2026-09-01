//! The object namespace: what a guest name resolves to, and what keeps the
//! thing it resolved to alive.
//!
//! # A name is resolved once, and then never again
//!
//! A guest names an object by its slot in an object list. Slots are reused, so
//! a slot number alone is not an identity: work that carried a slot number and
//! looked it up again later would find whatever now occupies it. Resolution
//! happens once, at admission, and produces a [`ResourceId`] carrying the
//! slot's generation; everything downstream carries that, and a slot reused
//! under it resolves to a different generation and refuses.
//!
//! # Deletion stops resolution; it does not stop work
//!
//! This is the invariant the whole module exists for. A guest deletes an
//! object while a submission that uses it is still executing — routinely, and
//! legally. Deleting must therefore do exactly two things: make *new*
//! resolution fail, and leave everything already accepted alone. Teardown
//! happens when the last accepted use retires, not when the guest asks.
//!
//! An implementation that freed on the delete has a use-after-free that only
//! appears under load. One that refused the delete while uses are outstanding
//! turns an ordinary guest sequence into a stall. So the delete is accepted,
//! the slot stops resolving, and [`Teardown`] says whether the object may be
//! torn down now or is owed to a later release.
//!
//! # Replacing the memory under a name is the same problem
//!
//! `replace-physical` swaps the backing a live object reads from. Work already
//! accepted against the old backing must still read the old backing — it was
//! planned against those bytes and its hazard edges were compiled against that
//! [`BackingId`]. So the replacement takes effect for resolution immediately
//! and the old backing is handed back with the same [`Teardown`] answer the
//! delete gives.
//!
//! # What this does not own
//!
//! Where the bytes are ([`crate::content`]), when a native object dies
//! ([`crate::retire`]), and what a host mapping is made of. A mapping here is
//! only *whether one exists*, because "is this mapped" changes what a guest may
//! do and "what it is mapped as" does not.

use crate::access::BackingId;
use crate::identity::{ObjectListRef, ResourceId, SlotGeneration};
use std::collections::HashMap;

/// Why a namespace operation did not happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Nothing has ever been declared in this slot.
    NotDeclared {
        slot: ObjectListRef,
    },
    /// The slot holds a different generation. The name is stale: whatever the
    /// caller meant was deleted and the slot reused.
    StaleGeneration {
        slot: ObjectListRef,
        named: SlotGeneration,
        current: SlotGeneration,
    },
    /// The object is deleted. It may still be executing; it may not be named.
    Deleted {
        id: ResourceId,
    },
    /// A slot that still holds a live object cannot be redeclared. A guest that
    /// wants the slot back deletes first, and a device that silently replaced
    /// would orphan work planned against the previous occupant.
    SlotOccupied {
        slot: ObjectListRef,
    },
    /// The object has no mapping to unmap, or already has one to map.
    NotMapped {
        id: ResourceId,
    },
    AlreadyMapped {
        id: ResourceId,
    },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NotDeclared { .. } => "namespace_not_declared",
            Self::StaleGeneration { .. } => "namespace_stale_generation",
            Self::Deleted { .. } => "namespace_deleted",
            Self::SlotOccupied { .. } => "namespace_slot_occupied",
            Self::NotMapped { .. } => "namespace_not_mapped",
            Self::AlreadyMapped { .. } => "namespace_already_mapped",
        }
    }
}

/// What a delete or a physical replacement left behind.
///
/// Two variants rather than an `Option` with a comment, because the caller has
/// to do something different in each case and a caller that ignored the
/// difference would either free memory the GPU is reading or leak it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a backing that is owed teardown is one nothing else will free"]
pub enum Teardown {
    /// Nothing accepted still reads this backing. It may be torn down now.
    Now { backing: BackingId },
    /// Accepted work still reads it. [`Namespace::release`] returns it when the
    /// last use retires.
    WhenUsesRetire {
        backing: BackingId,
        outstanding: usize,
    },
}

impl Teardown {
    #[must_use]
    pub const fn backing(self) -> BackingId {
        match self {
            Self::Now { backing } | Self::WhenUsesRetire { backing, .. } => backing,
        }
    }
}

/// A resolved name, held by work that was accepted against it.
///
/// Carries the generation, so it cannot be confused with a later occupant of
/// the same slot, and the backing it resolved to, so work planned against those
/// bytes keeps reading those bytes even after the name is pointed elsewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lease {
    pub id: ResourceId,
    pub backing: BackingId,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    generation: SlotGeneration,
    backing: BackingId,
    deleted: bool,
    mapped: bool,
    /// Accepted uses of the *current* backing that have not retired.
    outstanding: usize,
}

/// A backing whose name has moved on, kept alive by the work still reading it.
#[derive(Clone, Copy, Debug)]
struct Retiring {
    backing: BackingId,
    outstanding: usize,
}

/// One session generation's object namespace.
#[derive(Debug, Default)]
pub struct Namespace {
    slots: HashMap<ObjectListRef, Slot>,
    /// Backings detached from their slot by a delete or a replacement, still
    /// held by accepted work.
    retiring: Vec<Retiring>,
    resolved: usize,
    refused: usize,
}

impl Namespace {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare an object into a slot.
    ///
    /// A slot's generation advances on every declaration, including the first,
    /// so no live identity ever carries the default generation and a zeroed
    /// structure cannot read as a valid name.
    ///
    /// # Errors
    ///
    /// If the slot still holds an undeleted object.
    pub fn declare(
        &mut self,
        slot: ObjectListRef,
        backing: BackingId,
    ) -> Result<ResourceId, Refusal> {
        let generation = match self.slots.get(&slot) {
            Some(existing) if !existing.deleted => return Err(Refusal::SlotOccupied { slot }),
            Some(existing) => existing.generation.next(),
            None => SlotGeneration::default().next(),
        };
        self.slots.insert(
            slot,
            Slot {
                generation,
                backing,
                deleted: false,
                mapped: false,
                outstanding: 0,
            },
        );
        Ok(ResourceId { slot, generation })
    }

    /// Resolve a guest name.
    ///
    /// # Errors
    ///
    /// The one check that refused: never declared, deleted, or a different
    /// generation than the caller named.
    pub fn resolve(&mut self, name: ResourceId) -> Result<Lease, Refusal> {
        let refusal = match self.slots.get(&name.slot) {
            None => Refusal::NotDeclared { slot: name.slot },
            Some(slot) if slot.generation != name.generation => Refusal::StaleGeneration {
                slot: name.slot,
                named: name.generation,
                current: slot.generation,
            },
            Some(slot) if slot.deleted => Refusal::Deleted { id: name },
            Some(slot) => {
                self.resolved += 1;
                return Ok(Lease {
                    id: name,
                    backing: slot.backing,
                });
            }
        };
        self.refused += 1;
        Err(refusal)
    }

    /// Resolve by slot alone, for a guest name that carries no generation.
    ///
    /// # Errors
    ///
    /// As [`Self::resolve`], minus the generation check there is no input for.
    pub fn resolve_slot(&mut self, slot: ObjectListRef) -> Result<Lease, Refusal> {
        let generation = self
            .slots
            .get(&slot)
            .map(|s| s.generation)
            .ok_or(Refusal::NotDeclared { slot })?;
        self.resolve(ResourceId { slot, generation })
    }

    /// Record that accepted work holds this lease.
    ///
    /// Keeps the backing alive past a delete. A caller that resolved and then
    /// did not acquire has a name it may read and no claim on the memory, which
    /// is the shape of a use-after-free.
    pub fn acquire(&mut self, lease: Lease) {
        if let Some(slot) = self.slots.get_mut(&lease.id.slot) {
            if !slot.deleted && slot.backing == lease.backing {
                slot.outstanding += 1;
                return;
            }
        }
        if let Some(entry) = self
            .retiring
            .iter_mut()
            .find(|r| r.backing == lease.backing)
        {
            entry.outstanding += 1;
        }
    }

    /// Record that work holding this lease has retired.
    ///
    /// Returns the backing if this was the last use of one whose name has
    /// already moved on — the deferred half of [`Teardown::WhenUsesRetire`].
    pub fn release(&mut self, lease: Lease) -> Option<BackingId> {
        if let Some(slot) = self.slots.get_mut(&lease.id.slot) {
            if !slot.deleted && slot.backing == lease.backing && slot.outstanding > 0 {
                slot.outstanding -= 1;
                return None;
            }
        }
        let at = self
            .retiring
            .iter()
            .position(|r| r.backing == lease.backing)?;
        self.retiring[at].outstanding -= 1;
        if self.retiring[at].outstanding == 0 {
            return Some(self.retiring.remove(at).backing);
        }
        None
    }

    /// Delete an object: stop it resolving, and leave accepted work alone.
    ///
    /// # Errors
    ///
    /// As [`Self::resolve`]. Deleting twice is a refusal rather than a silent
    /// success, because the second delete's caller believes it owns a teardown
    /// the first one already answered for.
    pub fn delete(&mut self, name: ResourceId) -> Result<Teardown, Refusal> {
        let lease = self.resolve(name)?;
        let slot = self.slots.get_mut(&name.slot).expect("just resolved");
        slot.deleted = true;
        slot.mapped = false;
        let outstanding = slot.outstanding;
        slot.outstanding = 0;
        Ok(self.detach(lease.backing, outstanding))
    }

    /// Point a live name at different memory.
    ///
    /// Resolution moves at once; work already accepted against the old backing
    /// keeps reading the old backing, because it was planned against those
    /// bytes and its hazard edges were compiled against that identity.
    ///
    /// # Errors
    ///
    /// As [`Self::resolve`].
    pub fn replace_physical(
        &mut self,
        name: ResourceId,
        backing: BackingId,
    ) -> Result<Teardown, Refusal> {
        let lease = self.resolve(name)?;
        let slot = self.slots.get_mut(&name.slot).expect("just resolved");
        let outstanding = slot.outstanding;
        slot.backing = backing;
        slot.outstanding = 0;
        Ok(self.detach(lease.backing, outstanding))
    }

    /// Detach a backing from its slot, keeping it alive if work still reads it.
    fn detach(&mut self, backing: BackingId, outstanding: usize) -> Teardown {
        if outstanding == 0 {
            return Teardown::Now { backing };
        }
        self.retiring.push(Retiring {
            backing,
            outstanding,
        });
        Teardown::WhenUsesRetire {
            backing,
            outstanding,
        }
    }

    /// Give an object a host mapping.
    ///
    /// # Errors
    ///
    /// As [`Self::resolve`], plus [`Refusal::AlreadyMapped`]: two mappings of
    /// one object are two answers to where its bytes are.
    pub fn map(&mut self, name: ResourceId) -> Result<(), Refusal> {
        self.resolve(name)?;
        let slot = self.slots.get_mut(&name.slot).expect("just resolved");
        if slot.mapped {
            return Err(Refusal::AlreadyMapped { id: name });
        }
        slot.mapped = true;
        Ok(())
    }

    /// Take a mapping away.
    ///
    /// # Errors
    ///
    /// As [`Self::resolve`], plus [`Refusal::NotMapped`].
    pub fn unmap(&mut self, name: ResourceId) -> Result<(), Refusal> {
        self.resolve(name)?;
        let slot = self.slots.get_mut(&name.slot).expect("just resolved");
        if !slot.mapped {
            return Err(Refusal::NotMapped { id: name });
        }
        slot.mapped = false;
        Ok(())
    }

    #[must_use]
    pub fn is_mapped(&self, name: ResourceId) -> bool {
        self.slots
            .get(&name.slot)
            .is_some_and(|s| s.generation == name.generation && !s.deleted && s.mapped)
    }

    /// Accepted uses of a name's current backing that have not retired.
    #[must_use]
    pub fn outstanding(&self, name: ResourceId) -> usize {
        self.slots
            .get(&name.slot)
            .filter(|s| s.generation == name.generation)
            .map_or(0, |s| s.outstanding)
    }

    /// Every live name, with its exact generation.
    ///
    /// The complete input a teardown needs, taken without releasing or
    /// republishing anything: a teardown that discovered membership by asking
    /// each value owner separately would have to trust that they agree, and a
    /// teardown that released names as it found them could not refuse partway
    /// through without having already destroyed some of them.
    ///
    /// Sorted by slot, so a caller's teardown order is a property of the
    /// namespace and not of a hash seed.
    #[must_use]
    pub fn live_names(&self) -> Vec<ResourceId> {
        let mut out: Vec<ResourceId> = self
            .slots
            .iter()
            .filter(|(_, s)| !s.deleted)
            .map(|(slot, s)| ResourceId {
                slot: *slot,
                generation: s.generation,
            })
            .collect();
        out.sort_unstable_by_key(|id| id.slot.0);
        out
    }

    /// Backings detached from their slot and still held by accepted work.
    #[must_use]
    pub fn awaiting_teardown(&self) -> usize {
        self.retiring.len()
    }

    /// Names resolved, and names refused.
    #[must_use]
    pub const fn census(&self) -> (usize, usize) {
        (self.resolved, self.refused)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(n: u32) -> ObjectListRef {
        ObjectListRef(n)
    }

    fn backing(n: u64) -> BackingId {
        BackingId(n)
    }

    /// The invariant the module exists for.
    #[test]
    fn deleting_stops_resolution_and_leaves_accepted_work_alone() {
        let mut ns = Namespace::new();
        let id = ns.declare(slot(1), backing(10)).expect("free slot");
        let lease = ns.resolve(id).expect("live");
        ns.acquire(lease);

        assert_eq!(
            ns.delete(id),
            Ok(Teardown::WhenUsesRetire {
                backing: backing(10),
                outstanding: 1
            }),
            "the delete is accepted; the memory is not freed under the work"
        );
        assert_eq!(
            ns.resolve(id),
            Err(Refusal::Deleted { id }),
            "and nothing new may name it"
        );
        assert_eq!(
            ns.release(lease),
            Some(backing(10)),
            "the last use retiring is what makes it free-able"
        );
        assert_eq!(ns.awaiting_teardown(), 0);
    }

    #[test]
    fn deleting_an_unused_object_frees_it_at_once() {
        let mut ns = Namespace::new();
        let id = ns.declare(slot(1), backing(10)).expect("free slot");
        assert_eq!(
            ns.delete(id),
            Ok(Teardown::Now {
                backing: backing(10)
            })
        );
    }

    /// Slot reuse produces a new generation, and the old name does not follow
    /// it.
    #[test]
    fn a_reused_slot_refuses_the_name_that_used_to_live_there() {
        let mut ns = Namespace::new();
        let first = ns.declare(slot(1), backing(10)).expect("free slot");
        assert_eq!(
            ns.delete(first).expect("live"),
            Teardown::Now {
                backing: backing(10)
            }
        );
        let second = ns.declare(slot(1), backing(11)).expect("deleted, so free");
        assert_ne!(first.generation, second.generation);
        assert_eq!(
            ns.resolve(first),
            Err(Refusal::StaleGeneration {
                slot: slot(1),
                named: first.generation,
                current: second.generation,
            })
        );
        assert_eq!(ns.resolve(second).expect("live").backing, backing(11));
    }

    #[test]
    fn a_live_slot_cannot_be_redeclared() {
        let mut ns = Namespace::new();
        ns.declare(slot(1), backing(10)).expect("free slot");
        assert_eq!(
            ns.declare(slot(1), backing(11)),
            Err(Refusal::SlotOccupied { slot: slot(1) }),
            "silently replacing would orphan work planned against the previous \
             occupant"
        );
    }

    /// Work planned against the old bytes keeps reading the old bytes.
    #[test]
    fn replacing_the_memory_moves_the_name_and_not_the_accepted_work() {
        let mut ns = Namespace::new();
        let id = ns.declare(slot(1), backing(10)).expect("free slot");
        let old = ns.resolve(id).expect("live");
        ns.acquire(old);

        assert_eq!(
            ns.replace_physical(id, backing(20)),
            Ok(Teardown::WhenUsesRetire {
                backing: backing(10),
                outstanding: 1
            })
        );
        assert_eq!(
            ns.resolve(id).expect("still live").backing,
            backing(20),
            "the name resolves to the new memory at once"
        );
        assert_eq!(
            old.backing,
            backing(10),
            "and the lease already taken still names the old"
        );
        assert_eq!(ns.release(old), Some(backing(10)));
    }

    /// Two uses of one backing, released one at a time.
    #[test]
    fn a_detached_backing_survives_until_its_last_use_retires() {
        let mut ns = Namespace::new();
        let id = ns.declare(slot(1), backing(10)).expect("free slot");
        let a = ns.resolve(id).expect("live");
        let b = ns.resolve(id).expect("live");
        ns.acquire(a);
        ns.acquire(b);
        assert_eq!(ns.outstanding(id), 2);
        assert_eq!(
            ns.delete(id).expect("live"),
            Teardown::WhenUsesRetire {
                backing: backing(10),
                outstanding: 2
            }
        );
        assert_eq!(ns.release(a), None, "one use left");
        assert_eq!(ns.release(b), Some(backing(10)));
    }

    /// Deleting twice would hand two callers the same teardown.
    #[test]
    fn deleting_twice_is_refused() {
        let mut ns = Namespace::new();
        let id = ns.declare(slot(1), backing(10)).expect("free slot");
        assert_eq!(
            ns.delete(id).expect("live").backing(),
            backing(10),
            "the first delete owns the teardown"
        );
        assert_eq!(ns.delete(id), Err(Refusal::Deleted { id }));
    }

    #[test]
    fn a_name_that_was_never_declared_refuses_by_its_own_reason() {
        let mut ns = Namespace::new();
        let name = ResourceId {
            slot: slot(4),
            generation: SlotGeneration(1),
        };
        assert_eq!(
            ns.resolve(name),
            Err(Refusal::NotDeclared { slot: slot(4) })
        );
        assert_eq!(
            ns.resolve_slot(slot(4)),
            Err(Refusal::NotDeclared { slot: slot(4) })
        );
        assert_eq!(
            ns.census(),
            (0, 1),
            "one refusal, and the probe was not one"
        );
    }

    #[test]
    fn mapping_is_a_state_and_not_a_counter() {
        let mut ns = Namespace::new();
        let id = ns.declare(slot(1), backing(10)).expect("free slot");
        assert!(!ns.is_mapped(id));
        assert_eq!(ns.unmap(id), Err(Refusal::NotMapped { id }));
        ns.map(id).expect("unmapped");
        assert!(ns.is_mapped(id));
        assert_eq!(
            ns.map(id),
            Err(Refusal::AlreadyMapped { id }),
            "two mappings of one object are two answers to where its bytes are"
        );
        ns.unmap(id).expect("mapped");
        assert!(!ns.is_mapped(id));
    }

    /// A delete takes the mapping with it, so a later declaration into the slot
    /// does not inherit one.
    #[test]
    fn a_deleted_object_is_not_mapped() {
        let mut ns = Namespace::new();
        let id = ns.declare(slot(1), backing(10)).expect("free slot");
        ns.map(id).expect("unmapped");
        assert_eq!(ns.delete(id).expect("live").backing(), backing(10));
        assert!(!ns.is_mapped(id));
        let next = ns.declare(slot(1), backing(11)).expect("deleted, so free");
        assert!(!ns.is_mapped(next));
    }

    /// No live identity carries the default generation, so a zeroed structure
    /// cannot read as a valid name.
    #[test]
    fn the_first_declaration_does_not_use_the_default_generation() {
        let mut ns = Namespace::new();
        let id = ns.declare(slot(1), backing(10)).expect("free slot");
        assert_ne!(id.generation, SlotGeneration::default());
    }
}
