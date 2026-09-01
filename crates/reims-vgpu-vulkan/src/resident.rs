//! Which native object a guest resource name resolves to, and what happens to
//! the previous one when the guest reuses the name.
//!
//! # A name is a slot and a generation, and the two failures are different
//!
//! The guest names resources by object-list slot. It deletes and recreates in
//! the same slot constantly, so a slot number alone is not an identity — work
//! still holding the old number would resolve to the new object, read the
//! wrong texture, and produce a frame nobody can explain. [`ResourceId`]
//! carries the generation for that reason, and this table resolves on both.
//!
//! A slot that was never filled and a slot filled by a *later* generation are
//! kept as separate refusals. They are different defects: the first is decode
//! or ordering, the second is a lifetime that let accepted work outlive its
//! resource. One refusal covering both would make the more serious one
//! invisible inside the noisier one.
//!
//! # Publishing over a live entry does not free it
//!
//! When the guest recreates a slot, the previous object may still be named by
//! a submission the GPU has not reached. So a replacement retires the old
//! entry through [`NativeRetirement`] against the point of its last use — it
//! leaves the table immediately, because nothing may resolve to it any more,
//! and it leaves the device only when the timeline says so. Destroying it at
//! replacement is the use-after-free that looks like a driver bug.
//!
//! Publishing the *same* generation twice is refused instead of retiring
//! anything: two natives for one guest object is not a lifetime event, it is a
//! defect, and the second native is handed straight back rather than leaked.
//!
//! # This table holds handles and calls nothing
//!
//! Retirement produces the objects to destroy; the caller destroys them
//! through the device that made them. So every rule above is tested with no
//! GPU.

// A refused publication returns the native it would not take, so the caller
// destroys what it just made instead of leaking it. That makes the `Err`
// variant as large as a native object by construction; boxing it would put a
// heap allocation on the failure path in exchange for nothing.
#![allow(clippy::result_large_err)]

use ash::vk;
use reims_vgpu_core::identity::{
    DeviceEpoch, ObjectListRef, ResourceId, SessionGeneration, SlotGeneration, TimelinePoint,
};
use reims_vgpu_core::retire::{Lifetime, NativeRetirement};
use reims_vgpu_core::texture_shape::Texture;
use std::collections::BTreeMap;

use crate::buffer::BufferPlan;
use crate::image::ImagePlan;

/// A guest buffer, allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub plan: BufferPlan,
}

/// A guest texture, allocated, with every view it is addressable through.
///
/// The views are here rather than in a table beside this one because their
/// lifetime is exactly the image's — see [`crate::view`]. A second table would
/// be a second thing to retire in the same order and a second thing to get
/// wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeImage {
    /// The checked declaration this image was made from.
    ///
    /// Kept rather than re-derived from [`Self::plan`]: a transfer needs the
    /// guest's own format code to convert a byte pitch into texels and to name
    /// an aspect, and a `VkFormat` cannot answer either — several guest formats
    /// map to one native one, and the mapping is not invertible.
    pub texture: Texture,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub plan: ImagePlan,
    /// The whole-texture view a sampled binding uses.
    pub sampled: vk::ImageView,
    /// One per attachable slice, in [`crate::view::attachments`] order.
    pub attachments: Vec<vk::ImageView>,
}

/// What a guest resource name resolves to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Native {
    Buffer(NativeBuffer),
    Image(NativeImage),
}

impl Native {
    #[must_use]
    pub const fn kind(&self) -> Kind {
        match self {
            Self::Buffer(_) => Kind::Buffer,
            Self::Image(_) => Kind::Image,
        }
    }
}

/// Which of the two a name is bound to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Buffer,
    Image,
}

impl Kind {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::Image => "image",
        }
    }
}

/// Why a resource name resolved to nothing usable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Miss {
    /// Nothing was ever published in this slot.
    Unknown { slot: ObjectListRef },
    /// The slot holds a different generation than the name asks for.
    ///
    /// `held` newer than `named` is accepted work outliving its resource;
    /// `held` older is a name from ahead of what this table has been told.
    /// Both are here, with both numbers, because they are opposite defects.
    Stale {
        slot: ObjectListRef,
        held: SlotGeneration,
        named: SlotGeneration,
    },
    /// The name resolves, but to the other kind of object.
    WrongKind {
        slot: ObjectListRef,
        held: Kind,
        wanted: Kind,
    },
}

impl Miss {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unknown { .. } => "vk_resident_unknown_slot",
            Self::Stale { .. } => "vk_resident_stale_generation",
            Self::WrongKind { .. } => "vk_resident_wrong_kind",
        }
    }
}

impl std::fmt::Display for Miss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { slot } => write!(f, "{} slot={}", self.slug(), slot.0),
            Self::Stale { slot, held, named } => write!(
                f,
                "{} slot={} held={} named={}",
                self.slug(),
                slot.0,
                held.0,
                named.0
            ),
            Self::WrongKind { slot, held, wanted } => write!(
                f,
                "{} slot={} held={} wanted={}",
                self.slug(),
                slot.0,
                held.name(),
                wanted.name()
            ),
        }
    }
}

/// A second native for a guest object that already has one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Occupied {
    pub slot: ObjectListRef,
    pub generation: SlotGeneration,
}

impl Occupied {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        "vk_resident_already_published"
    }
}

impl std::fmt::Display for Occupied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} slot={} generation={}",
            self.slug(),
            self.slot.0,
            self.generation.0
        )
    }
}

/// One entry, and the lifetimes its handles are under.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    generation: SlotGeneration,
    lifetime: Lifetime,
    /// The latest point a submission naming this object will signal. Zero
    /// until something uses it, which is correct: a resource nothing has
    /// submitted against is retired the moment it is deleted.
    last_use: TimelinePoint,
    native: Native,
}

/// The native objects one device epoch holds for one session generation.
///
/// Keyed by slot, not by `ResourceId`: a slot holds at most one object, and a
/// map keyed by the pair would silently accept two generations of one slot
/// being live at once — which is exactly the state that lets old work resolve.
#[derive(Debug, Default)]
pub struct Residency {
    slots: BTreeMap<u32, Entry>,
    published: usize,
    replaced: usize,
}

impl Residency {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn population(&self) -> usize {
        self.slots.len()
    }

    /// How many objects have been published, and how many of those displaced a
    /// live predecessor. A replacement count that tracks the publication count
    /// is a guest churning one slot, which is a real thing to see.
    #[must_use]
    pub const fn census(&self) -> (usize, usize) {
        (self.published, self.replaced)
    }

    /// Bind a native object to a guest name.
    ///
    /// A previous generation in the same slot is retired against its last use;
    /// see the module doc.
    ///
    /// # Errors
    ///
    /// [`Occupied`] when the same generation is already published, with the
    /// native handed back so the caller can destroy what it just made.
    pub fn publish(
        &mut self,
        id: ResourceId,
        lifetime: Lifetime,
        native: Native,
        retire: &mut NativeRetirement<Native>,
    ) -> Result<(), (Native, Occupied)> {
        if let Some(previous) = self.slots.get(&id.slot.0) {
            if previous.generation == id.generation {
                return Err((
                    native,
                    Occupied {
                        slot: id.slot,
                        generation: id.generation,
                    },
                ));
            }
        }
        let displaced = self.slots.insert(
            id.slot.0,
            Entry {
                generation: id.generation,
                lifetime,
                last_use: TimelinePoint(0),
                native,
            },
        );
        self.published += 1;
        if let Some(previous) = displaced {
            self.replaced += 1;
            retire.queue(previous.lifetime, previous.last_use, previous.native);
        }
        Ok(())
    }

    /// What a name resolves to.
    ///
    /// # Errors
    ///
    /// [`Miss`] naming which of the two failures happened.
    pub fn resolve(&self, id: ResourceId) -> Result<&Native, Miss> {
        Ok(&self.entry(id)?.native)
    }

    fn entry(&self, id: ResourceId) -> Result<&Entry, Miss> {
        let entry = self
            .slots
            .get(&id.slot.0)
            .ok_or(Miss::Unknown { slot: id.slot })?;
        if entry.generation != id.generation {
            return Err(Miss::Stale {
                slot: id.slot,
                held: entry.generation,
                named: id.generation,
            });
        }
        Ok(entry)
    }

    /// The buffer a name resolves to.
    ///
    /// # Errors
    ///
    /// [`Miss`], including [`Miss::WrongKind`] when the name is a texture. A
    /// caller that resolved the enum itself and matched on it would either
    /// have to repeat this refusal or fall through silently.
    pub fn buffer(&self, id: ResourceId) -> Result<&NativeBuffer, Miss> {
        match self.resolve(id)? {
            Native::Buffer(buffer) => Ok(buffer),
            other => Err(Miss::WrongKind {
                slot: id.slot,
                held: other.kind(),
                wanted: Kind::Buffer,
            }),
        }
    }

    /// The image a name resolves to.
    ///
    /// # Errors
    ///
    /// [`Miss`], including [`Miss::WrongKind`] when the name is a buffer.
    pub fn image(&self, id: ResourceId) -> Result<&NativeImage, Miss> {
        match self.resolve(id)? {
            Native::Image(image) => Ok(image),
            other => Err(Miss::WrongKind {
                slot: id.slot,
                held: other.kind(),
                wanted: Kind::Image,
            }),
        }
    }

    /// Record that a submission signalling `at` names this resource.
    ///
    /// Moves forward and never back: two submissions may name one resource,
    /// and the one that frees it is the later. Keeping the earlier point is
    /// how a destroy races a submission that is still running.
    ///
    /// # Errors
    ///
    /// [`Miss`] when the name does not resolve, so a use of a stale name
    /// cannot silently extend the live object's lifetime.
    pub fn used(&mut self, id: ResourceId, at: TimelinePoint) -> Result<(), Miss> {
        // Resolved first, so a stale name changes nothing.
        self.entry(id)?;
        let entry = self
            .slots
            .get_mut(&id.slot.0)
            .expect("the entry resolved a line ago");
        if at.0 > entry.last_use.0 {
            entry.last_use = at;
        }
        Ok(())
    }

    /// The guest deleted this resource.
    ///
    /// Leaves the table at once — nothing may resolve to it again — and leaves
    /// the device when the timeline passes its last use.
    ///
    /// # Errors
    ///
    /// [`Miss`] when the name does not resolve. Nothing is removed: a delete
    /// naming a stale generation must not take the live object with it, which
    /// is precisely the bug a slot-only key produces.
    pub fn delete(
        &mut self,
        id: ResourceId,
        retire: &mut NativeRetirement<Native>,
    ) -> Result<(), Miss> {
        self.entry(id)?;
        let entry = self
            .slots
            .remove(&id.slot.0)
            .expect("the entry resolved a line ago");
        retire.queue(entry.lifetime, entry.last_use, entry.native);
        Ok(())
    }

    /// The session generation closed, or the device epoch ended: everything
    /// here stops being nameable.
    ///
    /// Queues every entry rather than destroying any, because a closed
    /// generation says nothing about whether the GPU is finished — that is
    /// still the timeline's answer, and [`NativeRetirement`] is where the two
    /// lifetimes are compared.
    pub fn drain(&mut self, retire: &mut NativeRetirement<Native>) -> usize {
        let drained = self.slots.len();
        for (_, entry) in std::mem::take(&mut self.slots) {
            retire.queue(entry.lifetime, entry.last_use, entry.native);
        }
        drained
    }

    /// What is left of a name's lease against the current lifetimes.
    ///
    /// # Errors
    ///
    /// [`Miss`] when the name does not resolve at all.
    pub fn validity(
        &self,
        id: ResourceId,
        session: SessionGeneration,
        epoch: DeviceEpoch,
    ) -> Result<reims_vgpu_core::retire::Validity, Miss> {
        Ok(self.entry(id)?.lifetime.against(session, epoch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use reims_vgpu_core::retire::Validity;
    use std::collections::BTreeSet;

    fn id(slot: u32, generation: u64) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(generation),
        }
    }

    fn lifetime() -> Lifetime {
        Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST)
    }

    fn native_buffer(handle: u64) -> Native {
        Native::Buffer(NativeBuffer {
            buffer: vk::Buffer::from_raw(handle),
            memory: vk::DeviceMemory::from_raw(handle),
            plan: BufferPlan {
                size: 64,
                usage: crate::buffer::EVERY_CLASS,
                aliased: false,
            },
        })
    }

    fn native_image(handle: u64) -> Native {
        Native::Image(NativeImage {
            texture: reims_vgpu_core::texture_shape::TextureShape {
                kind: reims_vgpu_core::texture_shape::TextureKind::D2.ordinal(),
                width: 4,
                height: 4,
                depth: 1,
                mipmap_level_count: 1,
                sample_count: 1,
                array_length: 1,
                pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                usage: reims_vgpu_core::texture_shape::TextureUsage::SHADER_READ,
            }
            .checked()
            .expect("a valid declaration"),
            image: vk::Image::from_raw(handle),
            memory: vk::DeviceMemory::from_raw(handle),
            plan: ImagePlan {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: vk::Extent3D {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
            sampled: vk::ImageView::from_raw(handle),
            attachments: Vec::new(),
        })
    }

    fn handle(native: &Native) -> u64 {
        match native {
            Native::Buffer(b) => b.buffer.as_raw(),
            Native::Image(i) => i.image.as_raw(),
        }
    }

    #[test]
    fn a_published_name_resolves_to_what_was_published() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(3, 1), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        assert_eq!(residency.population(), 1);
        assert_eq!(
            residency
                .buffer(id(3, 1))
                .expect("a buffer")
                .buffer
                .as_raw(),
            0xB1
        );
        assert_eq!(retire.outstanding(), 0);
    }

    #[test]
    fn an_unfilled_slot_and_a_stale_generation_are_different_refusals() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        assert_eq!(
            residency.resolve(id(3, 1)).err(),
            Some(Miss::Unknown {
                slot: ObjectListRef(3)
            })
        );

        residency
            .publish(id(3, 2), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        // A name from before the recreation, which is accepted work outliving
        // its resource — and must not resolve to the new object.
        assert_eq!(
            residency.resolve(id(3, 1)).err(),
            Some(Miss::Stale {
                slot: ObjectListRef(3),
                held: SlotGeneration(2),
                named: SlotGeneration(1),
            })
        );
        // And a name from ahead of it, which is the opposite defect.
        assert_eq!(
            residency.resolve(id(3, 3)).err(),
            Some(Miss::Stale {
                slot: ObjectListRef(3),
                held: SlotGeneration(2),
                named: SlotGeneration(3),
            })
        );
    }

    #[test]
    fn recreating_a_slot_retires_the_previous_object_rather_than_dropping_it() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(3, 1), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        residency
            .used(id(3, 1), TimelinePoint(9))
            .expect("a live name");
        residency
            .publish(id(3, 2), lifetime(), native_buffer(0xB2), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        // Out of the table immediately: nothing may resolve to it.
        assert_eq!(residency.population(), 1);
        assert_eq!(
            residency
                .buffer(id(3, 2))
                .expect("the new one")
                .buffer
                .as_raw(),
            0xB2
        );
        // Still on the device until the GPU is past the submission naming it.
        assert_eq!(retire.outstanding(), 1);
        assert!(retire
            .reached(DeviceEpoch::FIRST, TimelinePoint(8))
            .is_empty());
        let retired = retire.reached(DeviceEpoch::FIRST, TimelinePoint(9));
        assert_eq!(retired.len(), 1);
        assert_eq!(handle(&retired[0].object), 0xB1);
        assert_eq!(residency.census(), (2, 1));
    }

    #[test]
    fn publishing_the_same_generation_twice_hands_the_second_native_back() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(3, 1), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        let (returned, occupied) = residency
            .publish(id(3, 1), lifetime(), native_buffer(0xB2), &mut retire)
            .expect_err("one guest object has one native");

        assert_eq!(
            occupied,
            Occupied {
                slot: ObjectListRef(3),
                generation: SlotGeneration(1),
            }
        );
        // Handed back rather than leaked, and nothing was retired: the caller
        // destroys what it just made.
        assert_eq!(handle(&returned), 0xB2);
        assert_eq!(retire.outstanding(), 0);
        assert_eq!(
            residency
                .buffer(id(3, 1))
                .expect("the first")
                .buffer
                .as_raw(),
            0xB1
        );
        assert_eq!(residency.census(), (1, 0));
    }

    #[test]
    fn a_use_moves_the_last_point_forward_and_never_back() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(1, 1), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        residency.used(id(1, 1), TimelinePoint(7)).expect("live");
        // An earlier submission admitted after a later one — which happens,
        // because admission order is not submission order.
        residency.used(id(1, 1), TimelinePoint(3)).expect("live");
        residency.delete(id(1, 1), &mut retire).expect("live");

        // Freed at seven and not at three: the later submission is the one
        // still reading it.
        assert!(retire
            .reached(DeviceEpoch::FIRST, TimelinePoint(3))
            .is_empty());
        assert_eq!(
            retire.reached(DeviceEpoch::FIRST, TimelinePoint(7)).len(),
            1
        );
    }

    #[test]
    fn a_use_of_a_stale_name_extends_nothing() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(1, 2), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        assert!(residency.used(id(1, 1), TimelinePoint(99)).is_err());
        residency.delete(id(1, 2), &mut retire).expect("live");
        // Not held to 99 by a name that resolves to nothing.
        assert_eq!(
            retire.reached(DeviceEpoch::FIRST, TimelinePoint(0)).len(),
            1
        );
    }

    #[test]
    fn deleting_a_stale_name_does_not_take_the_live_object_with_it() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(4, 2), lifetime(), native_buffer(0xB2), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        assert!(matches!(
            residency.delete(id(4, 1), &mut retire),
            Err(Miss::Stale { .. })
        ));
        assert_eq!(residency.population(), 1);
        assert_eq!(retire.outstanding(), 0);
        assert!(residency.buffer(id(4, 2)).is_ok());
    }

    #[test]
    fn resolving_a_texture_as_a_buffer_refuses_by_name() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(2, 1), lifetime(), native_image(0x1A), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        assert_eq!(
            residency.buffer(id(2, 1)).err(),
            Some(Miss::WrongKind {
                slot: ObjectListRef(2),
                held: Kind::Image,
                wanted: Kind::Buffer,
            })
        );
        assert!(residency.image(id(2, 1)).is_ok());
        // And the mirror, so neither accessor is the only one that checks.
        residency
            .publish(id(5, 1), lifetime(), native_buffer(0xB5), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        assert_eq!(
            residency.image(id(5, 1)).err(),
            Some(Miss::WrongKind {
                slot: ObjectListRef(5),
                held: Kind::Buffer,
                wanted: Kind::Image,
            })
        );
    }

    #[test]
    fn a_closed_generation_leaves_the_table_and_waits_for_the_timeline() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        for slot in 0..3 {
            residency
                .publish(
                    id(slot, 1),
                    lifetime(),
                    native_buffer(u64::from(slot) + 1),
                    &mut retire,
                )
                .unwrap_or_else(|(_, e)| panic!("{e}"));
            residency
                .used(id(slot, 1), TimelinePoint(u64::from(slot) + 1))
                .expect("live");
        }

        assert_eq!(residency.drain(&mut retire), 3);
        assert_eq!(residency.population(), 0);
        assert_eq!(retire.outstanding(), 3);
        // Each at its own point: a generation closing says nothing about what
        // the GPU has finished.
        assert_eq!(
            retire.reached(DeviceEpoch::FIRST, TimelinePoint(1)).len(),
            1
        );
        assert_eq!(
            retire.reached(DeviceEpoch::FIRST, TimelinePoint(3)).len(),
            2
        );
    }

    #[test]
    fn a_name_reports_which_of_its_two_lifetimes_ended() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(1, 1), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        let session = SessionGeneration::FIRST;
        let epoch = DeviceEpoch::FIRST;
        assert_eq!(
            residency.validity(id(1, 1), session, epoch),
            Ok(Validity::Live)
        );
        assert_eq!(
            residency.validity(id(1, 1), session.next(), epoch),
            Ok(Validity::SemanticallyClosed)
        );
        assert_eq!(
            residency.validity(id(1, 1), session, epoch.next()),
            Ok(Validity::HandlesUnusable)
        );
        assert_eq!(
            residency.validity(id(1, 1), session.next(), epoch.next()),
            Ok(Validity::Gone)
        );
        assert!(residency.validity(id(1, 2), session, epoch).is_err());
    }

    #[test]
    fn every_refusal_names_itself() {
        let misses = [
            Miss::Unknown {
                slot: ObjectListRef(1),
            },
            Miss::Stale {
                slot: ObjectListRef(1),
                held: SlotGeneration(2),
                named: SlotGeneration(1),
            },
            Miss::WrongKind {
                slot: ObjectListRef(1),
                held: Kind::Buffer,
                wanted: Kind::Image,
            },
        ];
        let slugs: BTreeSet<&str> = misses.iter().map(|m| m.slug()).collect();
        assert_eq!(slugs.len(), misses.len());
        for miss in misses {
            assert!(miss.to_string().starts_with(miss.slug()));
            assert!(miss.slug().starts_with("vk_resident_"));
        }
        let occupied = Occupied {
            slot: ObjectListRef(1),
            generation: SlotGeneration(1),
        };
        assert!(occupied.to_string().starts_with(occupied.slug()));
        assert!(occupied.slug().starts_with("vk_resident_"));
    }
}
