//! The binding tables, and how a draw or a dispatch turns them into a
//! footprint.
//!
//! # A bind record writes a slot; this is the slots
//!
//! [`crate::compute`] and [`crate::render`] both say the same thing about
//! binds: they touch no memory. This is where the memory they *will* touch
//! accumulates, and where a dispatch or a draw reads it back out.
//!
//! # A table is not capped
//!
//! Apple's serializer truncates a plural bind at the stage's argument-table
//! size — 128 textures, 31 buffers, 16 samplers, measured — and those numbers
//! are the right *capacity hint* and the wrong *bound*. `reims-vgpu` has twice
//! shipped a cap standing in for them and twice dropped whole binds, so the
//! table here grows to whatever slot the guest names. The limits reserve the
//! usual size once; nothing refuses a slot above them.
//!
//! # What a bound slot contributes is the pipeline's answer, not the model's
//!
//! A buffer bound at slot 3 might be read, written, or never referenced. The
//! record does not say, the residency declarations that would say are entirely
//! unresolved in the ledger, and the thing that actually knows is the compiled
//! shader. So the footprint takes a [`crate::pipeline::BindingUsage`] — an
//! immutable fact an executor publishes when it finishes compiling — and
//! without one every bound slot participates as
//! [`AccessMode::Unknown`] over the whole resource.
//!
//! That fallback is correct and expensive, and both halves matter. It is
//! correct because `Unknown` conflicts with everything, so no edge is missed.
//! It is expensive because it conflicts with everything, which is why
//! `AccessMode::Unknown` is a distinct variant from `ReadWrite`: the census can
//! count how much ordering is being bought by not knowing, and a number that
//! can be counted is a number that can be driven down.

use crate::access::{AccessMode, Participation, ParticipationExtent};
use crate::bind::{BufferBinding, ObjectBinding};
use crate::identity::ResourceId;
use crate::pipeline::BindingUsage;
use reims_vgpu_protocol::render::ShaderStage;

/// The argument-table sizes Apple's serializer truncates a plural bind at.
///
/// Capacity, not capacity limits. See the module documentation.
mod table_hint {
    pub use reims_vgpu_protocol::bind::{
        BUFFER_TABLE_HINT as BUFFER, SAMPLER_TABLE_HINT as SAMPLER, TEXTURE_TABLE_HINT as TEXTURE,
    };
}

/// One resource class's slots for one stage.
///
/// Indexed by slot, sparse in content and dense in storage: the guest binds
/// low slots and leaves gaps, and a `Vec<Option<T>>` answers a lookup with one
/// bounds check where a map would hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotTable<T> {
    slots: Vec<Option<T>>,
}

impl<T: Copy> SlotTable<T> {
    #[must_use]
    pub fn with_hint(hint: u32) -> Self {
        Self {
            slots: Vec::with_capacity(hint as usize),
        }
    }

    /// Bind `value` at `slot`, growing the table if the guest reached past it.
    pub fn set(&mut self, slot: u32, value: Option<T>) {
        let index = slot as usize;
        if index >= self.slots.len() {
            // Nothing between the old end and the new slot was bound, so the
            // gap is empty rather than a copy of anything.
            self.slots.resize(index + 1, None);
        }
        self.slots[index] = value;
    }

    #[must_use]
    pub fn get(&self, slot: u32) -> Option<T> {
        self.slots.get(slot as usize).copied().flatten()
    }

    /// Every bound slot, with its index.
    pub fn bound(&self) -> impl Iterator<Item = (u32, T)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|v| (i as u32, v)))
    }

    /// How far the guest has reached, which is not how many are bound.
    #[must_use]
    pub fn extent(&self) -> usize {
        self.slots.len()
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }
}

impl<T: Copy> Default for SlotTable<T> {
    fn default() -> Self {
        Self { slots: Vec::new() }
    }
}

/// One stage's three tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageTables {
    pub buffers: SlotTable<BufferBinding>,
    pub textures: SlotTable<ObjectBinding>,
    /// Samplers bind no memory, so they never appear in a footprint. They are
    /// held because the encoder's state is the encoder's state — an executor
    /// building a descriptor set needs them — and not because they order
    /// anything.
    pub samplers: SlotTable<ObjectBinding>,
}

impl Default for StageTables {
    fn default() -> Self {
        Self {
            buffers: SlotTable::with_hint(table_hint::BUFFER),
            textures: SlotTable::with_hint(table_hint::TEXTURE),
            samplers: SlotTable::with_hint(table_hint::SAMPLER),
        }
    }
}

impl StageTables {
    /// Append what this stage's bound slots contribute.
    ///
    /// Written into a caller's buffer rather than returned, because this runs
    /// once per draw and a fresh `Vec` per draw is an allocation the frame does
    /// not need.
    pub fn footprint_into(&self, usage: Option<&BindingUsage>, out: &mut Vec<Participation>) {
        for (slot, binding) in self.buffers.bound() {
            let Some(buffer) = binding.buffer else {
                continue;
            };
            let Some(mode) = slot_mode(usage, |u| u.buffer(slot)) else {
                continue;
            };
            out.push(Participation {
                resource: buffer,
                // The bind names an offset and no length: what the shader reads
                // from there is the shader's business and the buffer's size is
                // the resource's. Whole is the honest extent, and it is where
                // reflection could narrow later.
                extent: ParticipationExtent::Whole,
                mode,
                api_stages: NO_STAGES,
            });
        }
        for (slot, binding) in self.textures.bound() {
            let Some(texture) = binding.object else {
                continue;
            };
            let Some(mode) = slot_mode(usage, |u| u.texture(slot)) else {
                continue;
            };
            out.push(Participation {
                resource: texture,
                extent: ParticipationExtent::Whole,
                mode,
                api_stages: NO_STAGES,
            });
        }
    }
}

/// The mode a bound slot contributes, or `None` when the pipeline does not
/// reference it.
///
/// Without a reflection the answer is [`AccessMode::Unknown`] for every bound
/// slot: a slot the model cannot ask about might be written, and a footprint
/// that guessed read would miss the edge that matters.
fn slot_mode(
    usage: Option<&BindingUsage>,
    ask: impl FnOnce(&BindingUsage) -> Option<AccessMode>,
) -> Option<AccessMode> {
    match usage {
        Some(usage) => ask(usage),
        None => Some(AccessMode::Unknown),
    }
}

/// A bound slot declares no stage of its own; the stages are the pipeline's.
const NO_STAGES: u32 = 0;

/// The compute encoder's state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComputeEncoderState {
    pub tables: StageTables,
    pub pipeline: Option<ResourceId>,
}

impl ComputeEncoderState {
    /// What a dispatch reads through the bound slots.
    pub fn footprint_into(&self, usage: Option<&BindingUsage>, out: &mut Vec<Participation>) {
        self.tables.footprint_into(usage, out);
    }
}

/// The render encoder's state.
///
/// Two stages, and they are separate tables rather than one with a stage tag:
/// slot 3 of the vertex stage and slot 3 of the fragment stage are different
/// slots, and a single table keyed by index alone would have them overwrite each
/// other.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderEncoderState {
    pub vertex: StageTables,
    pub fragment: StageTables,
    pub pipeline: Option<ResourceId>,
    pub depth_stencil: Option<ResourceId>,
    /// The occlusion-query mode and its offset into the pass's visibility
    /// buffer. The buffer itself is the pass descriptor's.
    pub visibility: Option<VisibilityState>,
}

/// The occlusion-query state a draw inherits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VisibilityState {
    pub mode: u64,
    pub offset: u64,
}

impl RenderEncoderState {
    #[must_use]
    pub fn stage(&self, stage: ShaderStage) -> &StageTables {
        match stage {
            ShaderStage::Vertex => &self.vertex,
            ShaderStage::Fragment => &self.fragment,
        }
    }

    pub fn stage_mut(&mut self, stage: ShaderStage) -> &mut StageTables {
        match stage {
            ShaderStage::Vertex => &mut self.vertex,
            ShaderStage::Fragment => &mut self.fragment,
        }
    }

    /// What a draw reads through the bound slots of both stages.
    ///
    /// The two usages are separate because the two stages are: a buffer the
    /// vertex shader reads and the fragment shader writes is two participations
    /// on one resource, and collapsing them would lose the write.
    pub fn footprint_into(
        &self,
        vertex_usage: Option<&BindingUsage>,
        fragment_usage: Option<&BindingUsage>,
        out: &mut Vec<Participation>,
    ) {
        self.vertex.footprint_into(vertex_usage, out);
        self.fragment.footprint_into(fragment_usage, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn buffer(id: u32) -> BufferBinding {
        BufferBinding {
            buffer: Some(res(id)),
            offset: 0x40,
            stride: None,
        }
    }

    fn texture(id: u32) -> ObjectBinding {
        ObjectBinding {
            object: Some(res(id)),
            lod_clamps: None,
        }
    }

    /// The table grows past every measured argument-table size rather than
    /// refusing there. Both of this project's dropped-bind regressions were a
    /// cap standing in for a capacity.
    #[test]
    fn a_table_grows_past_the_serializers_own_truncation_points() {
        let mut table = SlotTable::<BufferBinding>::with_hint(table_hint::BUFFER);
        let far = table_hint::TEXTURE + 500;
        table.set(far, Some(buffer(1)));
        assert_eq!(table.get(far), Some(buffer(1)));
        assert_eq!(table.bound().count(), 1);
        assert_eq!(table.extent(), far as usize + 1);
    }

    /// Growing does not bind the slots it grew past.
    #[test]
    fn growing_leaves_the_gap_unbound() {
        let mut table = SlotTable::<BufferBinding>::default();
        table.set(0, Some(buffer(1)));
        table.set(10, Some(buffer(2)));
        assert_eq!(table.get(5), None);
        let bound: Vec<_> = table.bound().map(|(slot, _)| slot).collect();
        assert_eq!(bound, vec![0, 10]);
    }

    /// Unbinding a slot leaves it holding nothing, which is not a resource.
    #[test]
    fn unbinding_leaves_nothing_rather_than_a_stale_resource() {
        let mut table = SlotTable::<BufferBinding>::default();
        table.set(2, Some(buffer(1)));
        table.set(2, None);
        assert_eq!(table.get(2), None);
        assert_eq!(table.bound().count(), 0);
    }

    /// Without a reflection every bound slot is unknown, and unknown conflicts
    /// with everything.
    #[test]
    fn an_unreflected_pipeline_makes_every_bound_slot_unknown() {
        let mut state = ComputeEncoderState::default();
        state.tables.buffers.set(0, Some(buffer(1)));
        state.tables.textures.set(4, Some(texture(2)));
        state.tables.samplers.set(0, Some(texture(3)));

        let mut out = Vec::new();
        state.footprint_into(None, &mut out);
        assert_eq!(out.len(), 2, "samplers bind no memory");
        for part in &out {
            assert_eq!(part.mode, AccessMode::Unknown);
            assert_eq!(part.extent, ParticipationExtent::Whole);
            assert!(part.mode.writes(), "unknown must conflict with a reader");
        }
    }

    /// With one, a slot the pipeline does not reference contributes nothing and
    /// the rest contribute what the shader does.
    #[test]
    fn a_reflection_narrows_the_footprint_to_what_the_shader_touches() {
        let mut state = ComputeEncoderState::default();
        state.tables.buffers.set(0, Some(buffer(1)));
        state.tables.buffers.set(1, Some(buffer(2)));
        state.tables.buffers.set(2, Some(buffer(3)));

        let usage = BindingUsage::new(
            vec![Some(AccessMode::Read), None, Some(AccessMode::Write)],
            Vec::new(),
        );
        let mut out = Vec::new();
        state.footprint_into(Some(&usage), &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].resource, res(1));
        assert_eq!(out[0].mode, AccessMode::Read);
        assert_eq!(out[1].resource, res(3));
        assert_eq!(out[1].mode, AccessMode::Write);
    }

    /// A slot past the reflection's own length is not referenced by the
    /// pipeline, so it contributes nothing — rather than falling back to
    /// unknown, which would make a long-tailed bind table expensive forever.
    #[test]
    fn a_slot_beyond_the_reflection_is_unreferenced() {
        let mut state = ComputeEncoderState::default();
        state.tables.buffers.set(9, Some(buffer(1)));
        let usage = BindingUsage::new(vec![Some(AccessMode::Read)], Vec::new());
        let mut out = Vec::new();
        state.footprint_into(Some(&usage), &mut out);
        assert!(out.is_empty());
    }

    /// The two render stages are separate slots with the same index, and a
    /// resource read by one and written by the other keeps both.
    #[test]
    fn the_two_render_stages_do_not_share_slot_numbers() {
        let mut state = RenderEncoderState::default();
        state
            .stage_mut(ShaderStage::Vertex)
            .buffers
            .set(3, Some(buffer(1)));
        state
            .stage_mut(ShaderStage::Fragment)
            .buffers
            .set(3, Some(buffer(1)));
        assert_eq!(
            state.stage(ShaderStage::Vertex).buffers.get(3),
            Some(buffer(1))
        );

        let read = BindingUsage::new(vec![None, None, None, Some(AccessMode::Read)], Vec::new());
        let write = BindingUsage::new(vec![None, None, None, Some(AccessMode::Write)], Vec::new());
        let mut out = Vec::new();
        state.footprint_into(Some(&read), Some(&write), &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].mode, AccessMode::Read);
        assert_eq!(out[1].mode, AccessMode::Write);
        assert_eq!(out[0].resource, out[1].resource);
    }

    /// The footprint is written into the caller's buffer, so a draw loop
    /// allocates nothing after the first.
    #[test]
    fn a_footprint_appends_rather_than_replacing() {
        let mut state = ComputeEncoderState::default();
        state.tables.buffers.set(0, Some(buffer(1)));
        let mut out = Vec::new();
        state.footprint_into(None, &mut out);
        state.footprint_into(None, &mut out);
        assert_eq!(out.len(), 2);
    }
}
