//! Behavior: turn guest work into HostActions / backend jobs.
//!
//! Drain FIFOs, parse wire (using [`crate::contract`]), resolve memory, plan
//! ops, update [`crate::model`] state. No GPU API calls here.

/// The split of [`chain_phase`]'s largest column, `binds_us`.
pub mod bind_phase;
/// Product-path blit fill/copy execution against guest GVA.
pub mod blit_exec;
/// Draw-time buffer binds, resolved once per reference and held until the
/// guest moves the addresses under them.
#[cfg(feature = "backend-vulkan")]
pub mod bound_buffers;
/// Always-on proxies and censuses, one per measured bug class.
pub mod census;
/// Where a draw chain's wall clock goes on the runtime side of the engine
/// boundary, which is 82% of it.
pub mod chain_phase;
/// Product-path compute bind/dispatch (pipeline + buffers + direct dispatch).
// See the note on `backend::metal`: `Status` is a 264-byte `Copy` payload
// carried on failure paths, and boxing it would cost the refusal vocabulary
// that makes each one greppable.
#[allow(clippy::result_large_err, clippy::large_enum_variant)]
pub mod compute_exec;
/// Multi-record compute encoder session (control-flow SPI + ICB execute).
// See the note on `backend::metal`: `Status` is a 264-byte `Copy` payload
// carried on failure paths, and boxing it would cost the refusal vocabulary
// that makes each one greppable.
#[allow(clippy::result_large_err, clippy::large_enum_variant)]
pub mod compute_session;
pub mod decode;
pub mod drain;
/// The always-on log sink every decline and census writes to
/// (`/tmp/reims-vgpu-fail.log`); `line()` is the `REIMS_VGPU_DRAW_LOG=1`-gated tier.
/// CmdExecIndirect2 stream walk + type-11 resolve.
pub mod exec;
/// Product-path event + encoder fence sync (event/blit/compute/render domains).
pub mod fence_exec;
/// Is the hypervisor's guest-write generation a sound cache key for the
/// zero-copy sampled gathers? Measurement, not policy.
#[cfg(feature = "backend-vulkan")]
pub mod gather_witness;
/// Gated with the `GuestWriteVerdict` it reuses and the `TargetIdentity` it
/// keys on, both of which are Vulkan-side; the type-11 twin this mirrors
/// (`mapper::mapping_guest_write_verdict`) carries the same gate.
#[cfg(feature = "backend-vulkan")]
pub mod gva_store_witness;
/// Guest-physical control-plane writes via HostOps map_pages.
pub mod gpa_map;
/// The bound on every GPU reference to guest RAM — one import per RAMBlock,
/// and the only type that can name a byte inside one.
pub mod guest_ram;
/// This process's imports of guest RAM, and the one place a guest physical
/// address becomes a bindable reference.
pub mod guest_ram_map;
/// Scattered guest windows → image-copy rectangles. Pure arithmetic, ungated so
/// both backends and every test arm reach it.
/// Task GVA → guest RAM reads.
pub mod gva_mem;
/// Task-GVA HostOps views (MapMemory2 / UnmapMemory lifecycle).
pub mod gva_view;
/// CmdHeapTextureSizeAndAlign wire decode + host requirement query.
pub mod heap_query;
pub mod host;
/// Which guest pages this device has written, and when — the half of the
/// guest-write witness the hypervisor's dirty bitmap cannot supply.
pub mod host_writes;
/// Type-7 ICB (0x36) materialization, host command fills, execute writeback.
pub mod icb;

/// Metal draw encode + writeback when MTLBs resolve.
// See the note on `backend::metal`: `Status` is a 264-byte `Copy` payload
// carried on failure paths, and boxing it would cost the refusal vocabulary
// that makes each one greppable.
#[allow(clippy::result_large_err, clippy::large_enum_variant)]
pub mod draw;
pub mod input;
/// Process-global metal2vulkan SPIR-V cache (AIR content hash → SPIR-V).
pub mod m2v_cache;
/// IOSurface mapper capture + page-table resolve.
pub mod map_audit;
pub mod mapper;
/// Write host BGRA into guest mapping pages (render writeback).
pub mod mapping_write;
/// generateMipmaps for multi-mip type-2/3 linear textures.
pub mod mipmap;
pub mod node_guard;
/// Whether a range's page-table entries are in the state the guest's own next
/// edit of them requires — the direction that is ordered is the map.
pub mod range_coverage;
pub mod released_pages;
pub mod mmio;
/// MTLB container → wrapped-AIR carve for metal2vulkan.
pub mod mtlb;
/// Object-list lookup and type-11 registration.
pub mod objects;
pub mod plan;
/// The resident identity a type-11 guest surface renders into.
#[cfg(feature = "backend-vulkan")]
pub mod present_identity;
/// Land a render Store's frame in the guest's pages, at the Store.
pub mod render_writeback;
/// The guest's per-resource validity quad, from both of its producers.
pub mod resource_validity;
/// The split of [`chain_phase`]'s largest *undivided* column, `sampled_us`.
pub mod sampled_phase;
/// Guest surface → host BGRA8 for the QEMU console.
pub mod scanout;
/// SPIR-V set-0 binding relocation for metal2vulkan + internal Vulkan engine (Linux).
pub mod spirv_bind;
mod spirv_layout;
/// How wide a translated vertex shader's stage-in reads are, per `Location`.
pub mod spirv_vertex_input;
/// Host surface cache (Linux/Vulkan discrete-GPU present, kb §8.5).
pub mod surface_cache;
/// The wire task word a command payload carries → a live task slot.
pub mod task_slot;
/// Texture / type-11 geometry registration.
pub mod texture;

/// The unit-test host double, gated with its definition. An ungated re-export
/// would keep it reachable and so keep it in the staticlib.
#[cfg(test)]
pub(crate) use host::FakeHost;
pub(crate) use host::{HostAction, HostOps};

/// Whether a type-11 render Store may hold its resolved copy for a reader,
/// read once per process.
///
/// Latched for the reason [`spirv_bind::buffer_extent_disabled`] is: this sits
/// on the per-Store path, `std::env::var_os` is a lock and an allocation, and
/// the variable cannot change under a running device. The arm is named once on
/// the off channel so a boot being compared says in its own log which arm it
/// ran, rather than the reading depending on the operator's shell history.
///
/// Default is on. `REIMS_VGPU_RENDER_PARK=off` restores the eager copy, which
/// performs strictly more work — see [`crate::env::RENDER_PARK`] for why that is
/// a narrowing and not a widening.
pub fn render_park_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let (state, value) = crate::env::read(crate::env::RENDER_PARK);
        match state {
            crate::env::Switch::Off => {
                crate::observe::off("render_park reason=render_park_disabled_by_env");
                false
            }
            // Named rather than silently read as the default, which is the one
            // way an operator concludes a switch does not work. It still takes
            // the default arm: this switch may only turn a rail off, and a value
            // nobody can parse is not that.
            crate::env::Switch::Unrecognized => {
                crate::observe::fail(format!(
                    "render_park reason=render_park_env_unrecognized value={}",
                    value.unwrap_or_default()
                ));
                true
            }
            crate::env::Switch::On | crate::env::Switch::Unset => true,
        }
    })
}
