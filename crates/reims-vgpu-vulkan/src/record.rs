//! Issuing the planned commands into a command buffer.
//!
//! # This module decides nothing
//!
//! Every other module in this crate produces values: a transfer's regions, a
//! mip ladder's rungs, a layout transition's two ends. This one turns those
//! values into `vkCmd*` calls and makes no choice of its own beyond which
//! *spelling* of a barrier this host takes. That is deliberate — a recorder
//! that also decided would be a second place every one of those decisions
//! lives, reachable only with a device present, and therefore the copy that is
//! never tested.
//!
//! So the tests for what these calls contain are in the modules that produce
//! them, and what this module needs a real driver for is whether a driver
//! accepts them at all.
//!
//! # The one choice: which barrier spelling
//!
//! Vulkan 1.2 is the baseline, so `VK_KHR_synchronization2` is an extension
//! and a host may only have `vkCmdPipelineBarrier`. [`crate::barrier`] already
//! makes that a mapping of one plan rather than a second translation; this
//! picks the emitter from the census's answer, once, at construction. A
//! per-call `if` would let one command buffer mix spellings, which is legal
//! and unreadable.

use ash::vk;

use crate::barrier::BarrierPlan;
use crate::layout::{OwnershipTransfer, Transition};
use crate::mipmap::Step;
use crate::staging::{Arena, Window};
use crate::transfer::{Command, FillPlan, StagedEdge};

/// A command buffer being recorded into, and the barrier spelling this host
/// takes.
///
/// Borrows rather than owns: the buffer's lifetime is
/// [`crate::recording::Preparation`]'s, and a recorder that owned one would be
/// a second claim on it.
#[derive(Clone, Copy)]
pub struct Recorder<'a> {
    device: &'a ash::Device,
    buffer: vk::CommandBuffer,
    synchronization2: bool,
}

impl<'a> Recorder<'a> {
    /// Record into `buffer`, using `synchronization2` barriers when this host
    /// has them.
    ///
    /// The flag comes from [`crate::census::Census::synchronization2`], which
    /// is the joined "extension enumerated *and* feature enabled, or core"
    /// answer. Passing a hopeful `true` here is the one way to get a null
    /// function pointer out of this module.
    #[must_use]
    pub const fn new(
        device: &'a ash::Device,
        buffer: vk::CommandBuffer,
        synchronization2: bool,
    ) -> Self {
        Self {
            device,
            buffer,
            synchronization2,
        }
    }

    #[must_use]
    pub const fn buffer(&self) -> vk::CommandBuffer {
        self.buffer
    }

    /// Issue one planned transfer.
    ///
    /// # Safety
    ///
    /// The command buffer is in the recording state, every handle the command
    /// names belongs to this device and is still alive, and each image is in
    /// the layout the command requires — which is [`crate::layout`]'s answer
    /// and not checked here.
    pub unsafe fn transfer(&self, command: &Command) {
        // SAFETY: the caller's preconditions are exactly the ones each of
        // these calls has; nothing here reads memory it was not handed.
        unsafe {
            match command {
                Command::CopyBuffer {
                    source,
                    dest,
                    regions,
                } => self
                    .device
                    .cmd_copy_buffer(self.buffer, *source, *dest, regions),
                Command::CopyBufferToImage {
                    source,
                    dest,
                    regions,
                } => self.device.cmd_copy_buffer_to_image(
                    self.buffer,
                    *source,
                    *dest,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions,
                ),
                Command::CopyImageToBuffer {
                    source,
                    dest,
                    regions,
                } => self.device.cmd_copy_image_to_buffer(
                    self.buffer,
                    *source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    *dest,
                    regions,
                ),
                Command::CopyImage {
                    source,
                    dest,
                    regions,
                } => self.device.cmd_copy_image(
                    self.buffer,
                    *source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    *dest,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions,
                ),
            }
        }
    }

    /// Issue a planned fill: its interior, and a copy for each staged edge.
    ///
    /// The edge bytes must already be written into the arena — see
    /// [`Arena::set`] — because a copy recorded here reads them when the GPU
    /// runs, and the arena's mapping is the only thing that can put them
    /// there.
    ///
    /// # Safety
    ///
    /// As [`Self::transfer`], and the arena's chunks are alive and bound.
    pub unsafe fn fill(&self, plan: &FillPlan, arena: &Arena) {
        if let Some(range) = plan.middle {
            // SAFETY: the caller's preconditions.
            unsafe {
                self.device.cmd_fill_buffer(
                    self.buffer,
                    plan.dest,
                    range.offset,
                    range.size,
                    range.data,
                );
            }
        }
        for edge in [plan.head, plan.tail].into_iter().flatten() {
            // SAFETY: as above.
            unsafe { self.edge(plan.dest, edge, arena) };
        }
    }

    unsafe fn edge(&self, dest: vk::Buffer, edge: StagedEdge, arena: &Arena) {
        let region = vk::BufferCopy {
            src_offset: edge.window.offset,
            dst_offset: edge.dest_offset,
            size: edge.length,
        };
        // SAFETY: the caller's preconditions.
        unsafe {
            self.device
                .cmd_copy_buffer(self.buffer, arena.buffer(edge.window), dest, &[region]);
        }
    }

    /// Issue one planned layout transition.
    ///
    /// # Safety
    ///
    /// As [`Self::transfer`]. `aspect` must be the one the image's format has
    /// — [`crate::view::aspect`] is what answers it.
    pub unsafe fn transition(
        &self,
        image: vk::Image,
        transition: &Transition,
        aspect: vk::ImageAspectFlags,
    ) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: transition.subresource.level,
            level_count: 1,
            base_array_layer: transition.subresource.layer,
            layer_count: 1,
        };
        if self.synchronization2 {
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(transition.src_stages)
                .src_access_mask(transition.src_access)
                .dst_stage_mask(transition.dst_stages)
                .dst_access_mask(transition.dst_access)
                .old_layout(transition.from)
                .new_layout(transition.to)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range);
            let barriers = [barrier];
            let info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            // SAFETY: the caller's preconditions, and `synchronization2` was
            // reported by the census that enabled it.
            unsafe { self.device.cmd_pipeline_barrier2(self.buffer, &info) };
            return;
        }

        // The same plan in the older flag types. Mapped once, in `barrier`,
        // so the two paths cannot disagree about what the guest asked for.
        let legacy = BarrierPlan {
            src_stages: transition.src_stages,
            dst_stages: transition.dst_stages,
            src_access: transition.src_access,
            dst_access: transition.dst_access,
        }
        .legacy();
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(legacy.src_access)
            .dst_access_mask(legacy.dst_access)
            .old_layout(transition.from)
            .new_layout(transition.to)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        // An empty stage mask is illegal in the older form, where
        // `synchronization2` accepts `NONE`. `TOP_OF_PIPE` as a source and
        // `BOTTOM_OF_PIPE` as a destination are the identities: they order
        // nothing, which is exactly what an empty mask meant.
        let src = if legacy.src_stages.is_empty() {
            vk::PipelineStageFlags::TOP_OF_PIPE
        } else {
            legacy.src_stages
        };
        let dst = if legacy.dst_stages.is_empty() {
            vk::PipelineStageFlags::BOTTOM_OF_PIPE
        } else {
            legacy.dst_stages
        };
        // SAFETY: the caller's preconditions.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buffer,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Issue both halves of a queue-family ownership move.
    ///
    /// Recorded into two command buffers on two queues in practice; this
    /// issues the half whose family matches `family`, so a caller cannot
    /// record the release into the acquiring buffer.
    ///
    /// # Safety
    ///
    /// As [`Self::transition`].
    pub unsafe fn ownership(
        &self,
        image: vk::Image,
        transfer: &OwnershipTransfer,
        aspect: vk::ImageAspectFlags,
        stages: vk::PipelineStageFlags,
    ) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: transfer.subresource.level,
            level_count: 1,
            base_array_layer: transfer.subresource.layer,
            layer_count: 1,
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(transfer.layout)
            .new_layout(transfer.layout)
            .src_queue_family_index(transfer.from_family)
            .dst_queue_family_index(transfer.to_family)
            .image(image)
            .subresource_range(range);
        // SAFETY: the caller's preconditions.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buffer,
                stages,
                stages,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Issue a mip ladder, in the order it was planned.
    ///
    /// The order is the ladder's whole content — see [`crate::mipmap`] — so
    /// this walks the slice and does not reorder, batch or coalesce.
    ///
    /// # Safety
    ///
    /// As [`Self::transition`], and `image` is the one the ladder was planned
    /// for.
    pub unsafe fn mipmap(&self, image: vk::Image, steps: &[Step], aspect: vk::ImageAspectFlags) {
        for step in steps {
            match step {
                // SAFETY: the caller's preconditions.
                Step::Transition(transition) => unsafe {
                    self.transition(image, transition, aspect);
                },
                Step::Blit(rung) => {
                    let regions = [rung.native()];
                    // SAFETY: as above; the ladder put the levels this blit
                    // names into the layouts it needs.
                    unsafe {
                        self.device.cmd_blit_image(
                            self.buffer,
                            image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &regions,
                            vk::Filter::LINEAR,
                        );
                    }
                }
            }
        }
    }
}

impl Arena {
    /// Write `byte` into every byte of a window.
    ///
    /// # Safety
    ///
    /// The window came from this arena, its chunk is not in flight, and no
    /// submission is reading it.
    pub unsafe fn set(&self, window: Window, byte: u8) {
        // SAFETY: the caller's preconditions; `write_at` is in range for
        // `window.size` bytes by the window's own construction.
        unsafe { std::ptr::write_bytes(self.write_at(window), byte, window.size as usize) };
    }

    /// Copy `bytes` into a window.
    ///
    /// # Safety
    ///
    /// As [`Self::set`], and `bytes.len()` is at most `window.size`.
    pub unsafe fn write(&self, window: Window, bytes: &[u8]) {
        debug_assert!(bytes.len() as u64 <= window.size);
        // SAFETY: the caller's preconditions and the length check above.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.write_at(window), bytes.len());
        }
    }
}
