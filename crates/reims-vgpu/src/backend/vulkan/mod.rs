//! Self-contained Vulkan execution backend (build-time alternate to Metal).
//!
//! Ownership mirrors [`crate::backend::metal`]: all host GPU work for this rail
//! lives under `backend/vulkan/`, driven by `ash`. Product draw encode uses the
//! internal [`engine`] (persistent ash context + content-keyed caches). This
//! crate has no external graphics-executor dependency; AIR translation comes
//! from the pinned public `metal2vulkan` crate.
//!
//! The [`Backend`] trait carries only guest-lifetime reset; the live draw seam
//! is `runtime/draw::try_metal2vulkan_draw` → [`engine::execute_draw_request`].
//!
//! [`caps`] classifies the bound host GPU into the four-cell support matrix
//! (unified/discrete memory × has/has-no DMA) that every path here must keep
//! working. Capability decisions belong there, not at call sites.
//!
//! [`translate`] is the matching seam for *state*: decoded Metal formats and
//! pipeline enums become Vulkan ones there and nowhere else, so the same
//! decision cannot be made twice with two different answers.

pub mod caps;
pub mod engine;
pub mod translate;

use crate::backend::Backend;
use crate::model::DeviceState;
use crate::runtime::blit_exec::{self, BlitStatus, LinearTextureLevel, Type11Texture};
use crate::runtime::compute_exec::{self, ComputeAccum, ComputeStatus};
use crate::runtime::decode::blit::Command as BlitCommand;
use crate::runtime::decode::compute::Command as ComputeCommand;
use crate::runtime::draw::{self, DrawEncodeRequest, EncodeStatus};
use crate::runtime::host::{HostMemory, HostOps};

/// The Vulkan rail's [`Backend`] handle.
///
/// Carries no state: the device and instance live in [`engine`]'s process-global
/// context, which spins up lazily at the first real encode so off-VM protocol
/// tests can construct this shell without a Vulkan ICD. That laziness is also
/// why there is no `probe` beside [`crate::backend::metal::MetalBackend::probe`]
/// — asking whether an ICD is present would be the very instance creation the
/// engine defers, and doing it at device create would put a Vulkan loader call
/// in front of every protocol test.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanBackend;

impl VulkanBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn reset(&self) {
        engine::reset_guest_state();
    }

    fn encode_draw_chain<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        req: &mut DrawEncodeRequest,
        writeback_guest: bool,
        force_full_store: bool,
    ) -> (EncodeStatus, Option<Vec<u8>>) {
        draw::vulkan::encode_draw_chain(state, host, req, writeback_guest, force_full_store)
    }

    fn encode_icb_execute_and_writeback<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        req: &DrawEncodeRequest,
        icb_ref: u32,
        range_location: u64,
        range_length: u64,
    ) -> EncodeStatus {
        draw::vulkan::encode_icb_execute_and_writeback(
            state,
            host,
            req,
            icb_ref,
            range_location,
            range_length,
        )
    }

    fn execute_dispatch<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        acc: &ComputeAccum,
        cmd: &ComputeCommand,
    ) -> ComputeStatus {
        compute_exec::vulkan::execute_dispatch_linux(state, host, task_id, acc, cmd)
    }

    fn try_copy_whole_plane_on_gpu<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        cmd: &BlitCommand,
    ) -> Option<BlitStatus> {
        blit_exec::vulkan::try_copy_whole_plane_on_gpu(state, host, task_id, cmd)
    }

    fn try_copy_t11_plane_to_linear_on_gpu<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        destination_ref: u32,
        src: &Type11Texture,
        dst: &LinearTextureLevel,
    ) -> Option<BlitStatus> {
        blit_exec::vulkan::try_copy_t11_plane_to_linear_on_gpu(
            state,
            host,
            task_id,
            destination_ref,
            src,
            dst,
        )
    }

    fn note_blit_t11_resident(&self, state: &DeviceState, mapping_id: u32) {
        blit_exec::vulkan::note_blit_t11_resident(state, mapping_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_vulkan() {
        assert_eq!(VulkanBackend::new().name(), "vulkan");
    }
}
