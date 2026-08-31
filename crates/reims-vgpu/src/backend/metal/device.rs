//! The `Backend` trait implementation for device lifecycle.
//!
//! Probing the device is [`super::runtime`]'s job, not this module's. Two
//! wrappers here used to say otherwise: a `MetalRuntime` unit struct whose one
//! associated function forwarded `system_device`, and a `system_device_name`
//! that forwarded the identically-named function it imported. Neither was
//! constructed or called anywhere outside this file's own test, and the second
//! put one name on two functions in two modules — so a `grep` for it reported
//! two producers and the arm a reader landed on was arbitrary.

use crate::backend::metal::runtime::system_device;
use crate::backend::{Backend, CensusSite, MipmapGeneration};
use crate::contract::mipmap::MetalMipmapError;
use crate::model::{DeviceInfoLimits, DeviceState};
use crate::runtime::compute_exec::{self, ComputeAccum, ComputeStatus};
use crate::runtime::compute_session::{self, ComputeSession};
use crate::runtime::decode::compute::Command as ComputeCommand;
use crate::runtime::draw::{self, DrawEncodeRequest, EncodeStatus};
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mipmap::MipmapStatus;

/// The Metal rail's [`Backend`] handle.
///
/// Fieldless, because there is no per-device Metal state to hold: the
/// `MTLDevice` is [`system_device`]'s process-global `OnceCell` and the command
/// queues are thread-locals beside it. This carried a `ready: bool` that nothing
/// read, kept only because constructing it was what first created that
/// `MTLDevice` — a side effect hidden in a constructor, which is why the probe
/// is now [`Self::probe`] and says so in its name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalBackend;

impl MetalBackend {
    /// Bring up the process's `MTLDevice` and report whether the host has one.
    ///
    /// The probe is the structural capability a build carrying both rails
    /// selects on — "this host can execute Metal" — and it is measured, never
    /// inferred from a device name. On a Metal-only build the answer cannot
    /// change what runs, so it is recorded and the handle is returned either
    /// way; refusing here would replace "the draw found no Metal device" with a
    /// failure at device create, which names the wrong thing.
    pub fn probe() -> Self {
        if system_device().is_none() {
            crate::observe::fail(
                "backend_probe reason=metal_no_system_device \
                 (this host exposes no MTLDevice)",
            );
        }
        Self
    }

    /// Whether this host exposes an `MTLDevice` at all.
    pub fn available() -> bool {
        system_device().is_some()
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn reset(&self) {
        crate::runtime::icb::clear_icb_cache();
    }

    fn encode_draw_chain<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        req: &mut DrawEncodeRequest,
        writeback_guest: bool,
        force_full_store: bool,
    ) -> (EncodeStatus, Option<Vec<u8>>) {
        draw::metal::encode_draw_chain(state, host, req, writeback_guest, force_full_store)
    }

    fn execute_dispatch<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        acc: &ComputeAccum,
        cmd: &ComputeCommand,
    ) -> ComputeStatus {
        compute_exec::metal::execute_dispatch_metal(state, host, task_id, acc, cmd, None)
    }

    #[allow(clippy::result_large_err, reason = "see the `Backend` declaration")]
    fn open_compute_session(&self, dispatch_type: u32) -> Result<ComputeSession, ComputeStatus> {
        compute_session::metal::MetalSession::open(dispatch_type).map(ComputeSession::from_metal)
    }

    fn execute_dispatch_nested<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        acc: &ComputeAccum,
        cmd: &ComputeCommand,
        session: &mut ComputeSession,
    ) -> ComputeStatus {
        // `None` cannot happen: `backend::selected()` is latched, so every
        // session in this process was opened by this rail. Named rather than
        // unwrapped, because a panic must never cross the QEMU FFI boundary.
        let Some(rail) = session.metal_mut() else {
            return ComputeStatus::NoMetal("compute_nested_session_not_metal");
        };
        compute_exec::metal::execute_dispatch_metal(state, host, task_id, acc, cmd, Some(rail))
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
        draw::metal::encode_icb_execute_and_writeback(
            state,
            host,
            req,
            icb_ref,
            range_location,
            range_length,
        )
    }

    /// This rail serves an Apple GPU to an Apple guest, so the table's own
    /// values already describe the executing device and there is nothing to
    /// reduce. Saturating rather than reflecting is deliberate: every one of
    /// these keys bounds what the guest's *own* Metal will then ask this device
    /// to run, and the guest's Metal is the same framework version running on
    /// the same silicon, so a smaller number here would refuse work the host
    /// can execute. The rail that has to reduce is the one whose host GPU is
    /// not the guest's — see [`crate::backend::vulkan::VulkanBackend`].
    fn device_info_limits(&self) -> DeviceInfoLimits {
        DeviceInfoLimits {
            max_sample_count: u32::MAX,
            d24_stencil8: true,
            max_threads_per_threadgroup: [u32::MAX; 3],
            max_threadgroup_memory_bytes: u32::MAX,
            native_fp16: true,
        }
    }

    /// Apple GPUs report 1024 and 32 across every family the arm64 pathway
    /// targets, and by the argument in [`Self::device_info_limits`] the GPU
    /// behind this rail is one of them.
    fn compute_threadgroup_limits(&self) -> (u32, u32) {
        (1024, 32)
    }

    fn emit_census(&self, site: CensusSite) {
        // One line, at one site. The other three are engine counters, phase
        // windows and a mutex census that this rail has no counterpart for —
        // absent rather than zeroed, so a reader cannot mistake "no such engine"
        // for "an idle one".
        if site == CensusSite::Levels {
            crate::runtime::drain::census::metal::emit_object_cache_levels();
        }
    }

    fn generate_mipmap_chain(
        &self,
        texture_ref: u32,
        fmt: u16,
        width: u32,
        height: u32,
        levels: u32,
        level0: &[u8],
    ) -> MipmapGeneration {
        match super::mipmap::generate_mipmaps_filtered(fmt, width, height, levels, level0) {
            Ok(chain) => MipmapGeneration::Chain(
                chain
                    .into_iter()
                    .map(|level| (level.width, level.height, level.tight_bytes))
                    .collect(),
            ),
            // Correct but slower: let the caller run the shared box filter, and
            // make the missing device visible as a typed degradation. This is
            // the *only* error that declines rather than refuses — every other
            // one means the filtered path was available and rejected the work.
            Err(error @ MetalMipmapError::NoDevice) => {
                crate::observe::Emit::decline("mipmap_metal_fallback", &error)
                    .field("texture", texture_ref)
                    .field("format", format!("{fmt:#x}"))
                    .field("width", width)
                    .field("height", height)
                    .off();
                MipmapGeneration::Unfiltered
            }
            Err(error) => MipmapGeneration::Refused(MipmapStatus::Metal(error)),
        }
    }

    // The rest of `Backend` takes the trait's defaults, and each default is the
    // accurate statement for this rail rather than a stub:
    //
    // * The two blit fast paths, the resident census, and the two present
    //   questions — no resident registry to copy out of, to count, or to ask.
    // * The guest-memory group — this rail's Store is a host copy that has
    //   already executed when it returns, so nothing is ever outstanding, it
    //   holds no alias of guest RAM past the call, and it pins no linear
    //   resident to release.
    // * The cadence pair — nothing is batched or deferred, so there is nothing
    //   for the heartbeat or the drain tail to flush.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::runtime::system_device_name;

    /// Named for what it asserts. It was called `system_device`, which shadowed
    /// the imported function of that name inside the test module.
    #[test]
    fn the_probe_finds_a_device_and_the_backend_reports_it_ready() {
        assert!(system_device().is_some());
        assert!(system_device_name().is_some());
        assert!(MetalBackend::available());
        assert_eq!(MetalBackend::probe().name(), "metal");
    }
}
