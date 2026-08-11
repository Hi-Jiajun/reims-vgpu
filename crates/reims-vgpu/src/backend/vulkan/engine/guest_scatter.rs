//! Scatter a detiled frame into the guest's pages with one compute dispatch
//! instead of one transfer region per run.
//!
//! # Why this rail exists
//!
//! [`crate::runtime::render_writeback`]'s module doc carries the measurement and
//! the reasoning; the short form is that the guest backs a surface in 16 KiB
//! physically-contiguous granules, so one 1080p writeback is ~507 runs and the
//! `Linear` plan's scatter was one `VkBufferCopy` region each. Quadrupling the
//! regions for byte-identical output halved the frame rate while `record_us` did
//! not move and `slot_us` nearly tripled, so the cost is GPU-side per-region work
//! rather than the driver's recording of it, and batching the same regions into
//! fewer calls could not have touched it.
//!
//! One dispatch has no regions at all. It reads the same detiled scratch and
//! writes the same guest bytes — `uint`-for-`uint`, with no format, row or texel
//! semantics anywhere in the kernel — so the result is byte-identical to the
//! transfer form by construction rather than by measurement.
//!
//! # The transfer form stays, and this is why
//!
//! [`super::plan_guest_linear_copies`] is still the path for a host without the
//! guest-RAM import, for a run this module refuses, and for the A/B baseline that
//! ranks the two. Nothing here may become the only way a frame reaches the guest.
//!
//! # The shape of one dispatch
//!
//! One workgroup per run, which makes `groupCountX` the run count and is why
//! nothing outside `shaders/guest_scatter.comp` names its `local_size_x` — see
//! [`super::scatter_shader`].
//!
//! `Dst` is bound at an **offset**, never at zero. A word index into a whole
//! RAMBlock does not fit a `uint`: a 16 GiB guest is exactly 2^32 words, and
//! `vm/boot-x86.sh` runs `-m 16G`. [`build_run_table`] binds the smallest
//! alignment-respecting window covering the writeback's own destinations and
//! makes every index relative to that base, which is single-digit MiB wide.
//!
//! The run table is a host-written storage buffer rather than push constants:
//! ~200 runs of 16 bytes is past every push-constant limit. It is written into a
//! mapped staging slot and read by the shader in place, so it costs no copy
//! region either — the fourth transfer this design was first sketched with is
//! not there.

use ash::vk;

use super::context::DeviceContext;
use super::scatter_shader::GUEST_SCATTER_SPIRV;
use super::types::DrawError;
use super::vk_call::{VkCall, VkOp};
use crate::observe::Decline;

/// Bytes in the word this kernel copies in. Every offset and length a run
/// carries has to be a whole number of these or the run cannot be expressed.
pub(crate) const SCATTER_WORD: u64 = 4;

/// `uvec4` per run: source word, destination word, word count, unused.
const WORDS_PER_RUN: usize = 4;

/// Binding numbers, matching `shaders/guest_scatter.comp`'s `layout(binding =)`
/// declarations. The kernel is compiled ahead of time and embedded, so nothing
/// in the toolchain relates these to the GLSL; the source-match test in
/// [`super::scatter_shader`] is what keeps the embedded module honest about the
/// file these were read from.
const BINDING_SRC: u32 = 0;
const BINDING_DST: u32 = 1;
const BINDING_RUNS: u32 = 2;

/// The one push constant: how many runs the table holds.
///
/// Redundant with `groupCountX` by construction and kept anyway. A dispatch
/// whose grid outran its table would read past the bound range, and under
/// `robustBufferAccess` that is defined-but-arbitrary rather than a fault — so
/// it would write arbitrary words into guest RAM instead of crashing. One
/// `uint` compared per workgroup is a cheap way for that to be impossible.
const PUSH_BYTES: u32 = 4;

/// A run this device cannot express as a dispatch, so the writeback took the
/// transfer regions instead.
///
/// Every one of these is a **routing** answer and not a loss: the frame still
/// lands, byte-identically, down [`super::plan_guest_linear_copies`]. They are
/// named because the region path is the expensive one and a boot silently
/// falling back to it would read as the dispatch not paying.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScatterDecline {
    /// A run's source offset, destination offset or length is not a whole
    /// number of [`SCATTER_WORD`] bytes.
    ///
    /// Run geometry is texel-aligned, which is four bytes for the eight-bit
    /// -per-channel formats this rail serves — but not for a narrower texel, so
    /// this is a check and never an assumption.
    Unaligned { src: u64, dst: u64, len: u64 },
    /// The window the writeback lands in is wider than the driver will bind as
    /// one storage buffer.
    RangeTooWide { range: u64, max: u64 },
    /// A run reads past the end of the detiled scratch it was planned against.
    ///
    /// Two independently-derived numbers disagreeing — the scratch is sized from
    /// the window's byte count and a run's extent comes from the guest's page
    /// plan — so this is the same class as `WindowTooSmall` one layer down.
    SourceOverrun { end: u64, have: u64 },
    /// The writeback named no runs at all, so there is nothing to dispatch.
    Empty,
}

impl Decline for ScatterDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unaligned { .. } => "scatter_run_unaligned",
            Self::RangeTooWide { .. } => "scatter_range_too_wide",
            Self::SourceOverrun { .. } => "scatter_source_overrun",
            Self::Empty => "scatter_no_runs",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unaligned { src, dst, len } => vec![
                ("src", src.to_string()),
                ("dst", dst.to_string()),
                ("len", len.to_string()),
            ],
            Self::RangeTooWide { range, max } => {
                vec![("range", range.to_string()), ("max", max.to_string())]
            }
            Self::SourceOverrun { end, have } => {
                vec![("end", end.to_string()), ("have", have.to_string())]
            }
            Self::Empty => Vec::new(),
        }
    }
}

crate::observe::decline::decline_display!(ScatterDecline);

/// One run as the planner sees it, before it becomes word indices.
///
/// `dst` is absolute in the imported buffer — `bound.offset + bound.head`, the
/// same re-basing every other planner here does — because the bind offset is not
/// known until every run has been seen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScatterRun {
    pub src: u64,
    pub dst: u64,
    pub len: u64,
}

/// The word-indexed run table for one destination buffer, and the window `Dst`
/// has to be bound over for its indices to mean anything.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RunTable {
    /// Where to bind `Dst`. Aligned down to the device's storage-buffer offset
    /// alignment, which is what makes it a legal `VkDescriptorBufferInfo::offset`.
    pub bind_offset: u64,
    /// How much of the buffer to bind, from `bind_offset`. Never `WHOLE_SIZE`:
    /// a RAMBlock is routinely wider than `maxStorageBufferRange`, and asking
    /// for the whole of one is the invalid-descriptor form of this bug.
    pub bind_range: u64,
    /// `WORDS_PER_RUN` `u32`s per run, ready to be written into a staging slot.
    pub words: Vec<u32>,
    pub run_count: u32,
}

/// Turn one destination buffer's runs into a word-indexed table and the window
/// to bind it against.
///
/// Pure, so it is testable on every arm including the one with no GPU — which
/// matters more here than usual, because every failure mode of this arithmetic
/// is a wrong *byte* in guest RAM rather than a crash.
///
/// `bind_align` is the device's storage-buffer offset alignment
/// ([`DeviceContext::guest_bind_offset_align`]) and `max_range` its
/// `maxStorageBufferRange`. `src_have` is the detiled scratch's size.
pub(crate) fn build_run_table(
    runs: &[ScatterRun],
    bind_align: u64,
    max_range: u64,
    src_have: u64,
) -> Result<RunTable, ScatterDecline> {
    let Some(first) = runs.first() else {
        return Err(ScatterDecline::Empty);
    };
    // Two passes rather than one, because the bind offset is a property of the
    // whole set and every index is relative to it.
    let mut lo = first.dst;
    let mut hi = 0u64;
    for run in runs {
        if run.src % SCATTER_WORD != 0 || run.dst % SCATTER_WORD != 0 || run.len % SCATTER_WORD != 0
        {
            return Err(ScatterDecline::Unaligned {
                src: run.src,
                dst: run.dst,
                len: run.len,
            });
        }
        let end = run.src.saturating_add(run.len);
        if end > src_have {
            return Err(ScatterDecline::SourceOverrun {
                end,
                have: src_have,
            });
        }
        lo = lo.min(run.dst);
        hi = hi.max(run.dst.saturating_add(run.len));
    }
    // `bind_align` is at least 16 and always a power of two, so the rounded-down
    // base stays a whole number of words and every relative index stays exact.
    let bind_offset = lo - lo % bind_align.max(1);
    let bind_range = hi - bind_offset;
    // Checked here and not at the descriptor write, because a range the driver
    // refuses and a range whose word index does not fit a `uint` are the same
    // bound: `maxStorageBufferRange` is a `uint32_t`, so a range that passes
    // this cannot produce an index above `u32::MAX / 4`.
    if bind_range > max_range {
        return Err(ScatterDecline::RangeTooWide {
            range: bind_range,
            max: max_range,
        });
    }
    let mut words = Vec::with_capacity(runs.len() * WORDS_PER_RUN);
    for run in runs {
        // Every one of these divisions is exact and every result is bounded by a
        // check above, so the truncating casts cannot lose a bit.
        words.push((run.src / SCATTER_WORD) as u32);
        words.push(((run.dst - bind_offset) / SCATTER_WORD) as u32);
        words.push((run.len / SCATTER_WORD) as u32);
        words.push(0);
    }
    let run_count = u32::try_from(runs.len()).map_err(|_| ScatterDecline::RangeTooWide {
        range: runs.len() as u64,
        max: u64::from(u32::MAX),
    })?;
    Ok(RunTable {
        bind_offset,
        bind_range,
        words,
        run_count,
    })
}

/// The device's own scatter pipeline, created once and held for the device's
/// life.
///
/// Not in [`super::caches`], which is keyed by guest shader digests and bounded
/// against a guest that walks pipeline space. This one is a fixture of the
/// device: exactly one exists, nothing evicts it, and a cache miss on it would
/// be a `vkCreateComputePipelines` in the middle of a writeback.
///
/// `Copy` because it is four handles and the writeback needs it while it also
/// holds `&mut ResourcePools` for the descriptor allocation and the staging
/// write. Copying the handles is what keeps that from being a borrow conflict
/// resolved by threading the pipeline through five signatures — and the owner
/// is still the single `Option` in the pools, which is what `destroy` clears.
#[derive(Clone, Copy)]
pub(crate) struct ScatterPipeline {
    module: vk::ShaderModule,
    pub(super) dsl: vk::DescriptorSetLayout,
    pub(super) layout: vk::PipelineLayout,
    pub(super) pipeline: vk::Pipeline,
}

impl ScatterPipeline {
    /// # Safety
    ///
    /// `ctx`'s device must be live, and the returned pipeline must be destroyed
    /// with [`Self::destroy`] before it.
    pub(crate) unsafe fn create(ctx: &DeviceContext) -> Result<Self, DrawError> {
        let device = &ctx.device;
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&GUEST_SCATTER_SPIRV),
                None,
            )
        }
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ScatterCreateShaderModule, e)))?;
        // Every binding is a plain storage buffer, which is the one descriptor
        // type `desc_arena`'s blocks are sized for in quantity.
        let bindings = [BINDING_SRC, BINDING_DST, BINDING_RUNS].map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let dsl = match unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        } {
            Ok(dsl) => dsl,
            Err(e) => {
                unsafe { device.destroy_shader_module(module, None) };
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::ScatterCreateSetLayout,
                    e,
                )));
            }
        };
        let set_layouts = [dsl];
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_BYTES)];
        let layout = match unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&ranges),
                None,
            )
        } {
            Ok(l) => l,
            Err(e) => {
                unsafe {
                    device.destroy_descriptor_set_layout(dsl, None);
                    device.destroy_shader_module(module, None);
                }
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::ScatterCreatePipelineLayout,
                    e,
                )));
            }
        };
        let entry = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        let info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        let pipeline =
            match unsafe { device.create_compute_pipelines(ctx.pipeline_cache, &info, None) } {
                Ok(p) => p[0],
                Err((_, e)) => {
                    unsafe {
                        device.destroy_pipeline_layout(layout, None);
                        device.destroy_descriptor_set_layout(dsl, None);
                        device.destroy_shader_module(module, None);
                    }
                    return Err(DrawError::VkCall(VkCall::new(
                        VkOp::ScatterCreatePipeline,
                        e,
                    )));
                }
            };
        Ok(Self {
            module,
            dsl,
            layout,
            pipeline,
        })
    }

    /// Consumes the copy it is called on, because the owner is the pools' single
    /// `Option` and taking it out of there is the only way to reach this.
    ///
    /// # Safety
    ///
    /// No submitted command buffer may still reference this pipeline.
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.dsl, None);
            device.destroy_shader_module(self.module, None);
        }
    }

    /// Write one dispatch's three bindings into an allocated set.
    ///
    /// # Safety
    ///
    /// `set` must have been allocated from [`Self::dsl`], and every buffer must
    /// be live and cover the offset/range pair given for it.
    pub(crate) unsafe fn write_set(
        device: &ash::Device,
        set: vk::DescriptorSet,
        src: (vk::Buffer, u64),
        dst: (vk::Buffer, u64, u64),
        runs: (vk::Buffer, u64),
    ) {
        let infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(src.0)
                .offset(0)
                .range(src.1),
            vk::DescriptorBufferInfo::default()
                .buffer(dst.0)
                .offset(dst.1)
                .range(dst.2),
            vk::DescriptorBufferInfo::default()
                .buffer(runs.0)
                .offset(0)
                .range(runs.1),
        ];
        let writes: Vec<_> = [BINDING_SRC, BINDING_DST, BINDING_RUNS]
            .iter()
            .enumerate()
            .map(|(i, binding)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[i]))
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// Bind, push the run count and dispatch one workgroup per run.
    ///
    /// # Safety
    ///
    /// `cb` must be recording, and `set` must name buffers live for the whole
    /// of the submission `cb` belongs to.
    pub(crate) unsafe fn dispatch(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        set: vk::DescriptorSet,
        run_count: u32,
    ) {
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                cb,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &run_count.to_ne_bytes(),
            );
            // One workgroup per run: the kernel strides its own run by its own
            // `local_size_x`, so no size from this side enters the arithmetic.
            device.cmd_dispatch(cb, run_count, 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALIGN: u64 = 16;
    const MAX: u64 = u32::MAX as u64;

    fn run(src: u64, dst: u64, len: u64) -> ScatterRun {
        ScatterRun { src, dst, len }
    }

    /// The whole point of the offset bind: a destination at the top of a 16 GiB
    /// RAMBlock still produces small indices, where an index from zero would
    /// have overflowed a `uint`.
    ///
    /// 16 GiB is `vm/boot-x86.sh`'s own `-m`, and it is exactly 2^32 words — so
    /// a word index from buffer byte zero has *no* headroom at the top of the
    /// block and a byte index has none from 4 GiB up. This test sits one word
    /// past the first of those two walls.
    #[test]
    fn indices_are_relative_to_the_bound_window_not_to_the_buffer() {
        let high = 16 * 1024 * 1024 * 1024u64;
        let t = build_run_table(
            &[run(0, high, 16384), run(16384, high + 65536, 16384)],
            ALIGN,
            MAX,
            1 << 20,
        )
        .expect("aligned runs plan");
        assert_eq!(t.bind_offset, high, "already aligned, so bound where it is");
        assert_eq!(t.bind_range, 65536 + 16384);
        assert_eq!(t.run_count, 2);
        assert_eq!(t.words[0..4], [0, 0, 4096, 0]);
        assert_eq!(t.words[4..8], [4096, 16384, 4096, 0]);
        // An index from buffer byte zero would not have fitted at all.
        assert!(high / SCATTER_WORD > u64::from(u32::MAX));
    }

    /// The bound the driver states is a `uint32_t`, so a range that passes it
    /// can never produce a word index that does not fit a `uint` — which is why
    /// [`build_run_table`] carries one check and not two.
    #[test]
    fn a_range_the_driver_admits_always_has_a_word_index_that_fits() {
        assert!(u64::from(u32::MAX) / SCATTER_WORD <= u64::from(u32::MAX));
    }

    #[test]
    fn the_bind_offset_rounds_down_to_the_alignment_and_indices_absorb_it() {
        let t = build_run_table(&[run(0, 1000, 8)], 16, MAX, 1 << 20).expect("plan");
        assert_eq!(t.bind_offset, 992, "1000 rounded down to a multiple of 16");
        assert_eq!(t.bind_range, 1008 - 992);
        // The 8 bytes the rounding put in front become 2 words of index.
        assert_eq!(t.words[1], 2);
    }

    /// A run the kernel cannot express must refuse rather than round, because
    /// rounding here writes the wrong guest bytes and reports success.
    #[test]
    fn a_run_that_is_not_a_whole_number_of_words_is_refused() {
        for bad in [run(1, 64, 16), run(0, 65, 16), run(0, 64, 15)] {
            let err = build_run_table(&[bad], ALIGN, MAX, 1 << 20)
                .expect_err("a sub-word run must not plan");
            assert!(matches!(err, ScatterDecline::Unaligned { .. }), "{err:?}");
        }
    }

    #[test]
    fn a_window_wider_than_the_driver_binds_is_refused() {
        // Word-aligned, so the refusal is the range and not the alignment: two
        // runs 4 GiB apart in one RAMBlock, which is what a guest larger than
        // `maxStorageBufferRange` produces the moment its window straddles.
        let far = 4 * 1024 * 1024 * 1024u64;
        let err = build_run_table(&[run(0, 0, 16), run(0, far, 16)], ALIGN, MAX, 1 << 20)
            .expect_err("a range past maxStorageBufferRange must not plan");
        assert!(
            matches!(err, ScatterDecline::RangeTooWide { .. }),
            "{err:?}"
        );
    }

    /// The scratch bound is checked from this side because the descriptor's own
    /// range cannot catch it: `Src` is bound whole, so an over-long run reads
    /// defined-but-wrong words and scatters them into the guest.
    #[test]
    fn a_run_reading_past_the_scratch_is_refused() {
        let err = build_run_table(&[run(4096, 0, 4096)], ALIGN, MAX, 4096)
            .expect_err("a run past the scratch must not plan");
        assert!(
            matches!(err, ScatterDecline::SourceOverrun { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn no_runs_is_refused_rather_than_dispatched_empty() {
        assert_eq!(
            build_run_table(&[], ALIGN, MAX, 1 << 20),
            Err(ScatterDecline::Empty)
        );
    }

    /// The table's words are what the guest's pages end up holding, so the
    /// mapping from run to `uvec4` is asserted directly rather than through the
    /// two properties above.
    #[test]
    fn every_run_becomes_four_words_in_order() {
        let runs = [run(0, 4096, 64), run(64, 8192, 128), run(192, 100, 32)];
        let t = build_run_table(&runs, ALIGN, MAX, 1 << 20).expect("plan");
        assert_eq!(t.words.len(), runs.len() * WORDS_PER_RUN);
        assert_eq!(t.run_count as usize, runs.len());
        for (i, r) in runs.iter().enumerate() {
            let w = &t.words[i * WORDS_PER_RUN..][..WORDS_PER_RUN];
            assert_eq!(u64::from(w[0]) * SCATTER_WORD, r.src);
            assert_eq!(u64::from(w[1]) * SCATTER_WORD + t.bind_offset, r.dst);
            assert_eq!(u64::from(w[2]) * SCATTER_WORD, r.len);
        }
    }

    /// The bound window has to cover every run, including one whose destination
    /// is neither the lowest nor the highest seen so far — the guest's runs
    /// arrive in window order, which is not destination order.
    #[test]
    fn the_bound_window_covers_runs_arriving_out_of_destination_order() {
        let t = build_run_table(
            &[run(0, 8192, 16), run(16, 128, 16), run(32, 4096, 16)],
            ALIGN,
            MAX,
            1 << 20,
        )
        .expect("plan");
        assert_eq!(t.bind_offset, 128, "the lowest destination, aligned down");
        assert_eq!(t.bind_range, 8192 + 16 - 128, "up to the highest end");
        for i in 0..3 {
            let w = &t.words[i * WORDS_PER_RUN..][..WORDS_PER_RUN];
            let end = u64::from(w[1]) * SCATTER_WORD + u64::from(w[2]) * SCATTER_WORD;
            assert!(end <= t.bind_range, "run {i} lands inside the bound window");
        }
    }
}
