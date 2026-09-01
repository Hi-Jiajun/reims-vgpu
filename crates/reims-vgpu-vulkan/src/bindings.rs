//! What a draw has to re-emit, and — far more often — what it does not.
//!
//! # The gate this exists for
//!
//! A steady-state draw must not rebuild its bindings. A guest that binds the
//! same forty textures every draw is the ordinary case, and an emitter that
//! wrote forty descriptors each time would spend the frame writing values that
//! were already there. So the table remembers what it emitted, a bind that does
//! not change a slot marks nothing, and [`BindingTable::take_dirty`] hands back
//! only the slots whose contents actually moved.
//!
//! # Equality is the whole mechanism, so the values have to be exact
//!
//! "Same binding" is decided by comparing the semantic binding — the resource
//! identity with its generation, the offset, the length — and never by
//! comparing native handles. Two different resources may occupy one handle
//! across a retire and a recreate, and a slot compared by handle would then
//! report clean while pointing at somebody else's memory. The identities carry
//! generations for exactly this reason, so the comparison is theirs.
//!
//! # Vulkan disturbs bindings, and pretending otherwise is a validation error
//!
//! Binding a pipeline whose layout is incompatible with the previous one
//! disturbs every descriptor set from the first incompatible index onward, and
//! a fresh command buffer starts with no bindings at all. Neither is something
//! the tracker can observe, so both are told to it:
//! [`BindingTable::disturb_all`] is the command-buffer boundary and the
//! incompatible-layout case. A tracker that quietly kept believing its own
//! contents across either would emit nothing and the draw would read whatever
//! the driver had.
//!
//! # An unbound slot is a change
//!
//! A guest unbinds by naming no object. That is a different state from "still
//! holds the previous object", and a tracker that treated `None` as "no news"
//! would leave a descriptor pointing at a resource the guest has released.

use reims_vgpu_core::bind::{BufferBinding, ObjectBinding};

/// A set of slot indices, one bit each.
///
/// Words rather than a `Vec<bool>`: a draw asks "is anything dirty" and "which
/// slots" on every emission, and both are word operations here. A hundred and
/// twenty-eight texture slots are two words.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlotMask {
    words: Vec<u64>,
}

impl SlotMask {
    #[must_use]
    pub fn with_capacity(slots: usize) -> Self {
        Self {
            words: vec![0; slots.div_ceil(64)],
        }
    }

    pub fn insert(&mut self, slot: usize) {
        let word = slot / 64;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1 << (slot % 64);
    }

    #[must_use]
    pub fn contains(&self, slot: usize) -> bool {
        self.words
            .get(slot / 64)
            .is_some_and(|w| w >> (slot % 64) & 1 == 1)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn clear(&mut self) {
        self.words.iter_mut().for_each(|w| *w = 0);
    }

    fn fill(&mut self, slots: usize) {
        self.words.resize(slots.div_ceil(64), 0);
        for (i, word) in self.words.iter_mut().enumerate() {
            let remaining = slots.saturating_sub(i * 64);
            *word = if remaining >= 64 {
                u64::MAX
            } else if remaining == 0 {
                0
            } else {
                (1u64 << remaining) - 1
            };
        }
    }

    /// The slots in the mask, ascending.
    #[must_use]
    pub fn slots(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.len());
        for (i, word) in self.words.iter().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                out.push(i * 64 + bit);
                bits &= bits - 1;
            }
        }
        out
    }
}

/// The slots one emission has to write.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[must_use = "slots reported dirty and not written leave the descriptor stale"]
pub struct Dirty {
    pub buffers: SlotMask,
    pub textures: SlotMask,
    pub samplers: SlotMask,
}

impl Dirty {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty() && self.textures.is_empty() && self.samplers.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.buffers.len() + self.textures.len() + self.samplers.len()
    }
}

/// What the table has been asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Binds that changed a slot.
    pub changed: usize,
    /// Binds that named the value already there. The number that says what the
    /// tracker is saving.
    pub redundant: usize,
    /// Slots handed to an emission.
    pub emitted: usize,
    /// Times everything was invalidated by a command-buffer boundary or an
    /// incompatible pipeline layout.
    pub disturbances: usize,
}

/// One shader-visible table's contents and its dirty set.
#[derive(Clone, Debug)]
pub struct BindingTable {
    buffers: Vec<Option<BufferBinding>>,
    textures: Vec<Option<ObjectBinding>>,
    samplers: Vec<Option<ObjectBinding>>,
    dirty: Dirty,
    census: Census,
}

impl BindingTable {
    /// A table with the guest's argument-table sizes, everything unbound.
    ///
    /// Nothing is dirty at construction: an unbound slot has no descriptor to
    /// write, and marking them all would make the first draw of every command
    /// buffer emit the whole table.
    #[must_use]
    pub fn new(buffers: usize, textures: usize, samplers: usize) -> Self {
        Self {
            buffers: vec![None; buffers],
            textures: vec![None; textures],
            samplers: vec![None; samplers],
            dirty: Dirty {
                buffers: SlotMask::with_capacity(buffers),
                textures: SlotMask::with_capacity(textures),
                samplers: SlotMask::with_capacity(samplers),
            },
            census: Census::default(),
        }
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// Bind a buffer slot, or unbind it with `None`.
    ///
    /// Returns whether anything changed. A slot outside the table is ignored
    /// and reported as no change: the argument-table sizes are the guest's own
    /// capacity hints, and a record past them is a record about a slot no shader
    /// on this pipeline can read.
    pub fn bind_buffer(&mut self, slot: usize, binding: Option<BufferBinding>) -> bool {
        let Some(current) = self.buffers.get_mut(slot) else {
            return false;
        };
        if *current == binding {
            self.census.redundant += 1;
            return false;
        }
        *current = binding;
        self.dirty.buffers.insert(slot);
        self.census.changed += 1;
        true
    }

    /// Bind a texture slot, or unbind it with `None`.
    pub fn bind_texture(&mut self, slot: usize, binding: Option<ObjectBinding>) -> bool {
        let Some(current) = self.textures.get_mut(slot) else {
            return false;
        };
        if *current == binding {
            self.census.redundant += 1;
            return false;
        }
        *current = binding;
        self.dirty.textures.insert(slot);
        self.census.changed += 1;
        true
    }

    /// Bind a sampler slot, or unbind it with `None`.
    pub fn bind_sampler(&mut self, slot: usize, binding: Option<ObjectBinding>) -> bool {
        let Some(current) = self.samplers.get_mut(slot) else {
            return false;
        };
        if *current == binding {
            self.census.redundant += 1;
            return false;
        }
        *current = binding;
        self.dirty.samplers.insert(slot);
        self.census.changed += 1;
        true
    }

    #[must_use]
    pub fn buffer(&self, slot: usize) -> Option<BufferBinding> {
        self.buffers.get(slot).copied().flatten()
    }

    #[must_use]
    pub fn texture(&self, slot: usize) -> Option<ObjectBinding> {
        self.textures.get(slot).copied().flatten()
    }

    #[must_use]
    pub fn sampler(&self, slot: usize) -> Option<ObjectBinding> {
        self.samplers.get(slot).copied().flatten()
    }

    /// Whether an emission would write anything.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.dirty.is_empty()
    }

    /// Take the slots this emission has to write, clearing the dirty set.
    ///
    /// The contents are unchanged: what was bound is still bound, and the next
    /// draw with no rebinds in between takes nothing.
    pub fn take_dirty(&mut self) -> Dirty {
        let taken = std::mem::take(&mut self.dirty);
        self.dirty = Dirty {
            buffers: SlotMask::with_capacity(self.buffers.len()),
            textures: SlotMask::with_capacity(self.textures.len()),
            samplers: SlotMask::with_capacity(self.samplers.len()),
        };
        self.census.emitted += taken.len();
        taken
    }

    /// Everything the driver believed is gone: a new command buffer, or a
    /// pipeline layout incompatible with the previous one.
    ///
    /// Every *bound* slot becomes dirty. Unbound slots do not, because there is
    /// nothing to write for them — which is also why a fresh table's first draw
    /// emits nothing rather than the whole table.
    pub fn disturb_all(&mut self) {
        self.census.disturbances += 1;
        mark_bound(&self.buffers, &mut self.dirty.buffers);
        mark_bound(&self.textures, &mut self.dirty.textures);
        mark_bound(&self.samplers, &mut self.dirty.samplers);
    }

    /// Forget every binding as well as every belief about them.
    ///
    /// For a semantic reset, where the objects the slots named are gone. Not
    /// the same as [`Self::disturb_all`], which keeps the contents because the
    /// guest has not unbound anything.
    pub fn reset(&mut self) {
        self.buffers.iter_mut().for_each(|s| *s = None);
        self.textures.iter_mut().for_each(|s| *s = None);
        self.samplers.iter_mut().for_each(|s| *s = None);
        self.dirty.buffers.clear();
        self.dirty.textures.clear();
        self.dirty.samplers.clear();
    }
}

fn mark_bound<T>(slots: &[Option<T>], mask: &mut SlotMask) {
    mask.fill(slots.len());
    for (slot, entry) in slots.iter().enumerate() {
        if entry.is_none() {
            // `fill` set every bit; clearing the unbound ones is cheaper than
            // testing each one before setting it, and the result is the same
            // set.
            let word = slot / 64;
            mask.words[word] &= !(1 << (slot % 64));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};

    /// Consume an emission whose contents this test is not asserting about.
    /// `Dirty` is `#[must_use]` because slots reported and not written leave the
    /// descriptor stale, so a test drops one deliberately or not at all.
    fn emit(t: &mut BindingTable) -> usize {
        t.take_dirty().len()
    }

    fn res(slot: u32, generation: u64) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(generation),
        }
    }

    fn buffer(slot: u32, offset: u64) -> Option<BufferBinding> {
        Some(BufferBinding {
            buffer: Some(res(slot, 1)),
            offset,
            stride: None,
        })
    }

    fn texture(slot: u32) -> Option<ObjectBinding> {
        Some(ObjectBinding {
            object: Some(res(slot, 1)),
            lod_clamps: None,
        })
    }

    fn object(slot: u32, generation: u64) -> Option<ObjectBinding> {
        Some(ObjectBinding {
            object: Some(res(slot, generation)),
            lod_clamps: None,
        })
    }

    fn table() -> BindingTable {
        BindingTable::new(32, 128, 16)
    }

    /// The gate: a guest that binds the same things every draw emits nothing
    /// after the first.
    #[test]
    fn a_steady_state_draw_emits_nothing() {
        let mut t = table();
        for slot in 0..40 {
            t.bind_texture(slot, texture(slot as u32));
        }
        assert_eq!(t.take_dirty().len(), 40, "the first draw writes them all");
        for _ in 0..100 {
            for slot in 0..40 {
                assert!(
                    !t.bind_texture(slot, texture(slot as u32)),
                    "rebinding the same value is not a change"
                );
            }
            assert!(t.is_clean());
            assert!(t.take_dirty().is_empty());
        }
        assert_eq!(
            t.census().emitted,
            40,
            "forty writes across a hundred draws"
        );
        assert_eq!(t.census().redundant, 4000);
    }

    /// Only what moved.
    #[test]
    fn only_the_slots_that_changed_are_emitted() {
        let mut t = table();
        for slot in 0..8 {
            t.bind_buffer(slot, buffer(slot as u32, 0));
        }
        emit(&mut t);
        t.bind_buffer(3, buffer(3, 256));
        t.bind_buffer(5, buffer(99, 0));
        let dirty = t.take_dirty();
        assert_eq!(dirty.buffers.slots(), vec![3, 5]);
        assert!(dirty.textures.is_empty());
    }

    /// A slot's offset is part of its binding, so the same resource at a
    /// different offset is a different binding.
    #[test]
    fn a_different_window_of_one_buffer_is_a_change() {
        let mut t = table();
        t.bind_buffer(0, buffer(1, 0));
        emit(&mut t);
        assert!(t.bind_buffer(0, buffer(1, 64)));
        assert_eq!(t.take_dirty().buffers.slots(), vec![0]);
    }

    /// Two different resources may occupy one native handle across a retire and
    /// a recreate. Comparing by identity is what stops a slot reading clean
    /// while pointing at somebody else's memory.
    #[test]
    fn a_reused_slot_at_a_new_generation_is_a_change() {
        let mut t = table();
        t.bind_texture(0, object(7, 1));
        emit(&mut t);
        assert!(
            t.bind_texture(0, object(7, 2)),
            "the same object-list slot at a later generation is a different resource"
        );
    }

    /// A tracker that read `None` as "no news" would leave a descriptor
    /// pointing at a resource the guest has released.
    #[test]
    fn unbinding_a_slot_is_a_change_and_binding_nothing_twice_is_not() {
        let mut t = table();
        t.bind_texture(4, texture(1));
        emit(&mut t);
        assert!(t.bind_texture(4, None));
        assert_eq!(t.take_dirty().textures.slots(), vec![4]);
        assert!(t.texture(4).is_none());
        assert!(!t.bind_texture(4, None), "it was already unbound");
    }

    /// A fresh command buffer has no bindings, so everything bound has to be
    /// written again — and nothing unbound does.
    #[test]
    fn a_disturbance_dirties_the_bound_slots_and_only_those() {
        let mut t = table();
        t.bind_buffer(0, buffer(1, 0));
        t.bind_buffer(7, buffer(2, 0));
        t.bind_texture(31, texture(3));
        t.bind_texture(64, texture(4));
        emit(&mut t);
        assert!(t.is_clean());

        t.disturb_all();
        let dirty = t.take_dirty();
        assert_eq!(dirty.buffers.slots(), vec![0, 7]);
        assert_eq!(
            dirty.textures.slots(),
            vec![31, 64],
            "and the mask spans more than one word"
        );
        assert!(dirty.samplers.is_empty(), "nothing was bound to a sampler");
        assert_eq!(t.census().disturbances, 1);
    }

    /// A fresh table's first draw writes nothing: an unbound slot has no
    /// descriptor to write, and marking them all would make every command
    /// buffer start by emitting a full table.
    #[test]
    fn a_fresh_table_is_clean() {
        let mut t = table();
        assert!(t.is_clean());
        t.disturb_all();
        assert!(t.is_clean(), "nothing is bound, so nothing is stale");
    }

    /// A reset is the guest's objects going away, not the driver's belief.
    #[test]
    fn a_reset_forgets_the_bindings_and_a_disturbance_does_not() {
        let mut t = table();
        t.bind_buffer(1, buffer(1, 0));
        emit(&mut t);

        let mut disturbed = t.clone();
        disturbed.disturb_all();
        assert_eq!(disturbed.buffer(1), buffer(1, 0), "still bound");
        assert!(!disturbed.is_clean(), "and owed a write");

        t.reset();
        assert_eq!(t.buffer(1), None);
        assert!(t.is_clean(), "there is nothing left to write");
    }

    /// The argument-table sizes are the guest's own capacity hints. A record
    /// past them names a slot no shader on this pipeline can read.
    #[test]
    fn a_slot_past_the_table_is_not_a_change() {
        let mut t = BindingTable::new(4, 4, 4);
        assert!(!t.bind_buffer(4, buffer(1, 0)));
        assert!(t.is_clean());
        assert_eq!(t.buffer(4), None);
    }

    #[test]
    fn the_mask_reports_exactly_the_slots_it_holds() {
        let mut m = SlotMask::with_capacity(8);
        assert!(m.is_empty());
        m.insert(0);
        m.insert(63);
        m.insert(64);
        m.insert(200);
        assert_eq!(m.slots(), vec![0, 63, 64, 200]);
        assert_eq!(m.len(), 4);
        assert!(m.contains(64));
        assert!(!m.contains(65));
        m.clear();
        assert!(m.is_empty());
        assert_eq!(m.slots(), Vec::<usize>::new());
    }
}
