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

#[cfg(feature = "host-window")]
use crate::backend::window;
use crate::backend::{
    Backend, CensusSite, GuestWriteReach, ObjectRetirement, PlaneDrawReader, Rail, RetainedObject,
    StampOrdering,
};
use crate::model::{ComputeStorageResidencyKey, DeviceInfoLimits, DeviceState};
use crate::runtime::blit_exec::{self, BlitStatus, LinearTextureLevel, MapperRefTexture};
use crate::runtime::compute_exec::{self, ComputeAccum, ComputeStatus, ResidentServe};
use crate::runtime::compute_session::ComputeSession;
use crate::runtime::decode::blit::Command as BlitCommand;
use crate::runtime::decode::compute::Command as ComputeCommand;
use crate::runtime::drain;
use crate::runtime::draw::{self, DrawEncodeRequest, EncodeStatus, GvaSpan};
use crate::runtime::guest_ram::ImportId;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::render_writeback::SettleSite;
use crate::runtime::scanout;

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
        Rail::Vulkan.name()
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

    #[allow(clippy::result_large_err, reason = "see the `Backend` declaration")]
    fn open_compute_session(&self, _dispatch_type: u32) -> Result<ComputeSession, ComputeStatus> {
        // One-shot per dispatch: there is no encoder to hold open across a
        // segment's records, and no `SessionRail` variant for one.
        Err(ComputeStatus::NoMetal("compute_session_no_vulkan_path"))
    }

    fn execute_dispatch_nested<M: HostMemory + HostOps>(
        &self,
        _state: &mut DeviceState,
        _host: &mut M,
        _task_id: u32,
        _acc: &ComputeAccum,
        _cmd: &ComputeCommand,
        _session: &mut ComputeSession,
    ) -> ComputeStatus {
        // Unreachable while `open_compute_session` refuses: a nested dispatch
        // needs a session, and this rail never has one. Fail-visible anyway —
        // `exec::note_compute_refusal` names the slug for every non-`Ok`
        // compute record.
        ComputeStatus::NoMetal("compute_nested_no_vulkan_path")
    }

    fn retire_task_object(
        &self,
        state: &mut DeviceState,
        task_id: u32,
        object: RetainedObject,
        object_ref: u32,
    ) -> ObjectRetirement {
        // This rail retains both kinds by `(task, ref)` — the depth-stencil
        // table for `draw::vulkan`, the pipeline table for
        // `runtime::pipeline_resolve` — so the guest's declaration is the
        // invalidation, and it is what makes the retention sound.
        let retired = match object {
            RetainedObject::DepthStencilState => {
                state.task_depth_stencil_states.delete(task_id, object_ref)
            }
            RetainedObject::RenderPipelineState => state
                .task_render_pipeline_states
                .delete(task_id, object_ref),
        };
        if retired {
            ObjectRetirement::Retired
        } else {
            ObjectRetirement::Absent
        }
    }

    #[cfg(feature = "host-window")]
    fn presents_host_window(&self) -> bool {
        // The swapchain half of that window is `engine::window_present`, so
        // this rail is the one that can fill it. Whether this *build* compiled
        // the window at all is a different question with a different owner —
        // `device::window_publish`, where the `host-window` feature is asked.
        true
    }

    #[cfg(feature = "host-window")]
    fn window_attach(&self, surface: &window::WindowSurface) -> Result<(), window::WindowDecline> {
        engine::window_present_attach(
            surface.display,
            surface.window,
            surface.width,
            surface.height,
        )
        .map_err(window_decline)
    }

    #[cfg(feature = "host-window")]
    fn window_attached(&self) -> bool {
        engine::window_present_attached()
    }

    #[cfg(feature = "host-window")]
    fn window_resident(
        &self,
        state: &DeviceState,
        mapping_id: u32,
        width: u32,
        height: u32,
    ) -> Result<window::WindowResident, &'static str> {
        let identity =
            crate::runtime::present_identity::surface_identity(state, mapping_id, width, height);
        // One engine operation keeps this resident alive across the idle sweep,
        // reclaims aged peers, and returns the direct-present decision for this
        // exact identity and geometry — so it runs whether or not the window
        // ends up taking the resident.
        engine::prepare_window_resident_present(&identity, width, height)?;
        Ok(window::WindowResident::Vulkan(
            engine::WindowPresentSource {
                width,
                height,
                identity,
            },
        ))
    }

    #[cfg(feature = "host-window")]
    fn window_present(
        &self,
        resident: Option<&window::WindowResident>,
        cpu: Option<window::WindowCpuFrame<'_>>,
    ) -> Result<window::WindowPresentOutcome, window::WindowDecline> {
        // The rail's own resident, and the only shape this enum can hold on a
        // build that compiled this rail — Metal contributes no variant, having
        // no registry to name one from.
        let source = resident.map(|window::WindowResident::Vulkan(source)| source);
        engine::window_present_frame(source, cpu).map_err(window_decline)
    }

    #[cfg(feature = "host-window")]
    fn window_resize(&self, width: u32, height: u32) {
        engine::window_present_resize(width, height);
    }

    #[cfg(feature = "host-window")]
    fn window_detach(&self) {
        engine::window_present_detach();
    }

    #[cfg(feature = "host-window")]
    fn window_reattach_budget(&self) -> u32 {
        // The presenter dies with the device that owns its swapchain, so the
        // number of rebuilds worth attempting is the number of device recreates
        // this rail will attempt — one value, in the one place that decides it.
        engine::MAX_DEVICE_RECREATES
    }

    fn guest_writes_outstanding(&self) -> bool {
        engine::guest_writes_outstanding()
    }

    fn quiesce_guest_writes(&self) {
        engine::quiesce_guest_writes();
    }

    fn guest_writes_reaching(&self, pages: &[u64]) -> GuestWriteReach {
        engine::guest_writes_reaching(pages)
    }

    fn retire_guest_import(&self, import: ImportId) -> Option<(usize, usize)> {
        engine::retire_guest_import(import)
    }

    fn take_released_host_aliases(&self) -> Vec<(usize, usize)> {
        engine::take_released_host_aliases()
    }

    fn retire_linear_residents(&self, keys: &[ComputeStorageResidencyKey]) {
        // Two releases, and dropping either one is a leak in the opposite
        // direction: an unpin alone leaves the image holding the only copy of
        // content nothing may reclaim, and retiring the content alone leaves a
        // pinned slot no reclaim path may take. Together they make the image
        // ordinarily evictable.
        for key in keys {
            engine::unpin_resident_storage(key);
            engine::retire_resident_storage_content(key);
            crate::observe::off(format!(
                "linear_resident_retired task={} ref={} gva={:#x} {}x{} fmt={:#x}",
                key.map_generation,
                key.texture_ref,
                key.surface_offset,
                key.width,
                key.height,
                key.pixel_format
            ));
        }
    }

    fn warm_guest_ram_imports(
        &self,
        imports: &[std::sync::Arc<crate::runtime::guest_ram::GuestRamImport>],
    ) -> (usize, u64) {
        engine::warm_guest_ram_imports(imports)
    }

    fn note_plane_store_published(&self, mapping_id: u32) {
        draw::vulkan::note_plane_store_published(mapping_id);
    }

    fn note_drain_thread(&self) {
        engine::mark_drain_thread();
    }

    fn install_stamp_announce(&self, announce: crate::backend::StampAnnounce) {
        engine::stamp_completion::install_announce(announce);
    }

    fn maintain(&self, now_ms: u64) {
        engine::maintain_resources(now_ms);
    }

    fn flush_deferred_submissions(&self) {
        engine::flush_batched_draws();
    }

    fn flush_batch_for_waiting_stamp(&self, stamp_index: u32) -> bool {
        engine::submit_batch_for_waiting_stamp(stamp_index)
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
        src: &MapperRefTexture,
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

    /// This rail runs on anything from a discrete part to an iGPU sitting at the
    /// Vulkan floor, so it reflects the bound device rather than asserting a
    /// table. That is the case a fixed table gets wrong, and it gets it wrong in
    /// the direction the guest cannot recover from — the reply is asked once per
    /// boot and kept for the life of it.
    fn device_info_limits(&self) -> DeviceInfoLimits {
        engine::device_info_limits()
    }

    fn compute_threadgroup_limits(&self) -> (u32, u32) {
        engine::compute_threadgroup_limits()
    }

    fn present_resident_carries(
        &self,
        state: &DeviceState,
        mapping: u32,
        width: u32,
        height: u32,
    ) -> Option<bool> {
        scanout::vulkan::present_resident_carries(state, mapping, width, height)
    }

    fn try_capture_from_resident(
        &self,
        state: &mut DeviceState,
        buf: &mut Vec<u8>,
        mapping_id: u32,
        width: u32,
        height: u32,
    ) -> bool {
        scanout::vulkan::try_capture_from_resident(state, buf, mapping_id, width, height)
    }

    fn order_completion_stamp<M: HostMemory + HostOps>(
        &self,
        state: &DeviceState,
        host: &mut M,
        index: u32,
        value: u32,
        site: SettleSite,
    ) -> StampOrdering {
        drain::vulkan::order_completion_stamp(state, host, index, value, site)
    }

    fn resident_serve(
        &self,
        key: ComputeStorageResidencyKey,
        mirror_generation: u32,
        is_storage: bool,
        pixel_format: u16,
    ) -> Option<ResidentServe> {
        compute_exec::vulkan::resident_serve(key, mirror_generation, is_storage, pixel_format)
    }

    /// This rail's targets are engine `TargetIdentity`s; a handle it did not
    /// issue names no image it can read, and the frame is lost rather than
    /// written from the wrong one.
    fn pay_surface_writeback<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        mapping_id: u32,
        target: &crate::runtime::resident_target::ResidentTarget,
        width: u32,
        height: u32,
    ) -> bool {
        let Some(identity) = target.get::<engine::TargetIdentity>() else {
            return false;
        };
        let paid = crate::runtime::render_writeback::vulkan::store_render_frame(
            state, host, mapping_id, identity, width, height,
        );
        if !paid {
            // The hold ends either way: a frame nothing will ask for again must
            // not keep its image alive to the next device reset.
            engine::note_resident_content_copied_out(identity);
        }
        paid
    }

    fn abandon_resident(&self, target: &crate::runtime::resident_target::ResidentTarget) {
        if let Some(identity) = target.get::<engine::TargetIdentity>() {
            engine::note_resident_content_copied_out(identity);
        }
    }

    fn preflight_translations<M: HostMemory + HostOps>(
        &self,
        state: &DeviceState,
        host: &M,
        task_id: u32,
        streams: &[Vec<u8>],
    ) -> bool {
        crate::runtime::exec::vulkan::preflight_translations(state, host, task_id, streams)
    }

    fn gva_load_seed_elidable<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        span: GvaSpan,
    ) -> bool {
        draw::vulkan::gva_load_seed_elidable(state, host, task_id, span)
    }

    fn read_abandoned_chain_rgba(
        &self,
        state: &DeviceState,
        req: &DrawEncodeRequest,
    ) -> Option<Vec<u8>> {
        let identity = draw::vulkan::render_chain_identity(state, req)?;
        draw::vulkan::read_resident_chain(req, &identity)
    }

    fn pipeline_raster_sample_count<M: HostMemory + HostOps>(
        &self,
        state: &DeviceState,
        host: &M,
        task_id: u32,
        pipeline_ref: u32,
    ) -> Option<u32> {
        crate::runtime::pipeline_resolve::attachment_sample_count(
            state,
            host,
            task_id,
            pipeline_ref,
        )
    }

    fn plane_draw_witness(&self, reader: PlaneDrawReader, mapping_id: u32) -> String {
        draw::vulkan::read_plane_draw_ring(reader, mapping_id).to_string()
    }

    fn emit_census(&self, site: CensusSite) {
        use drain::census::vulkan as census;
        match site {
            CensusSite::Serialization { win_ms } => census::emit_engine_lock(win_ms),
            CensusSite::WorkingSet => census::emit_working_set(),
            CensusSite::Throughput => census::emit_engine_delta(),
            CensusSite::Levels => {
                census::emit_object_cache_levels();
                census::emit_guest_import_levels();
            }
        }
    }
}

/// Which disposition one of this rail's draw refusals carries when it reaches
/// the host window.
///
/// The window acts on exactly one distinction — a presenter that is *gone* gets
/// rebuilt — and only this rail knows which of its refusals means that. A
/// `VK_ERROR_DEVICE_LOST` destroys the presenter along with everything else
/// derived from the device, and the next present finds no presenter at all;
/// every other refusal leaves one standing and is named and dropped.
#[cfg(feature = "host-window")]
fn window_decline(error: engine::DrawError) -> window::WindowDecline {
    let lost = matches!(
        error,
        engine::DrawError::Facade(engine::EngineFacadeDecline::WindowPresenterNotAttached)
    );
    let reason = window::WindowDeclineReason::Vulkan(error);
    if lost {
        window::WindowDecline::PresenterLost(reason)
    } else {
        window::WindowDecline::Refused(reason)
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
