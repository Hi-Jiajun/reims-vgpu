//! Backend selection seam.
//!
//! - [`metal`] / [`vulkan`] = concrete backends (feature-selected), each
//!   **self-contained** in this crate (Metal via `metal`; Vulkan via `ash` +
//!   [`vulkan::engine`]).
//! - Draws, compute and blits do **not** come through this module. The live
//!   seams are `runtime/draw::try_metal2vulkan_draw` → [`vulkan::engine`]
//!   on the Vulkan rail and `metal::render::render_core_mrt` /
//!   `metal::compute::compute_core` on the Metal rail; the runtime drives them
//!   directly.
//!
//! Metal indices/semantics are canonical (guest wire is serialized Metal).
//! Vulkan-only binding rewrites live only in [`vulkan`].

/// Content hashes for the Metal backend's compiled-object caches.
///
/// Declared here rather than inside `metal` — not a doc link, because that
/// module does not exist on the Vulkan arm, which is the point — and ungated,
/// for one reason: it
/// names nothing from the `metal` crate, and gating it made two test functions
/// that any host can execute run on none of them. Everything under
/// `backend/metal/` is `cfg`-ed out of the arm a Linux host builds, and
/// `cargo test --target aarch64-apple-darwin --no-run` fails at the link step —
/// so while this was a `mod hash` in that tree, the only way to run its tests
/// was to copy the file to `/tmp` and invoke `rustc --test` by hand. That is not
/// a gate, and AGENTS.md recorded it as a workaround rather than as the bug it
/// was.
///
/// The cost is a Vulkan build compiling twenty lines of arithmetic it never
/// calls, which is not a reason to hide a test from the only machine that can
/// run it.
///
/// It stays out of `contract::fnv` for the reason its own doc gives: the fold
/// here is not the shared one, and a caller reaching for the wrong one would
/// produce keys in a different keyspace without anything failing.
pub mod hash;

/// The identity of a shader blob, for the caches [`hash`] used to key on its
/// digest alone.
///
/// Ungated for the same reason and by the same argument as [`hash`]: it names
/// nothing from the `metal` crate, the thing worth testing is the byte compare,
/// and a `cfg` here would put those tests on the arm a Linux host cannot run.
pub mod blob;

/// `MTLRenderPipelineState` identity: the descriptor half, the shader half, and
/// the compare that decides a cache hit.
///
/// Ungated for the third time by the same argument as [`hash`] and [`blob`], and
/// this is the case that most needed it: the module's own tests had never
/// executed on any host, and what they test is whether two different pipelines
/// can be served as one.
pub mod render_pso_key;

// There is no fourth. `AGENTS.md` asks for pure logic under `backend/metal/` to
// be moved out here so its tests run on every arm rather than on none, and the
// three above are what that yielded; a survey of the rest found the remaining
// candidates blocked rather than overlooked, and it is cheaper to say so than to
// have the survey run again.
//
// Only `abi.rs`, `error.rs` and `util.rs` name nothing from the `metal` crate —
// `abi.rs`'s apparent references are `MTL*` type names in prose, and its sole
// import is `core::mem::offset_of`. All three are chained to `abi`, and `abi`
// must stay where it is: it is a **mirror of an archived C header**, its
// provenance is the point, and `contract::dispatch` and `contract::pass_action`
// record the reasoning. The values that are genuinely shared — the ones that
// arrive on the wire and are consumed by both backends — were already lifted
// into `contract/`, with `const` assertions in the mirror pinning the two
// spellings equal. Those assertions fire on every arm that compiles the mirror,
// including the cross-compiled `--target aarch64-apple-darwin` clippy run, so
// the mirror is not an untested file; it is tested by a mechanism `#[test]` was
// the wrong tool for.
//
// `error.rs` then cannot follow on its own, because `Status::code` is defined in
// terms of that header's `REIMS_VGPU_OK` / `_ERR_ARGS` / `_ERR_EXECUTE`, and
// re-spelling three constants out here to free five tests is the duplication
// `AGENTS.md` says to derive away rather than create.
#[cfg(feature = "backend-metal")]
// `Status` is 264 bytes — six inline `(key, value)` fields, no allocation — and
// it is the `Err` of most of this module's functions, so `result_large_err` and
// `large_enum_variant` fire across it. Boxing is the lint's remedy and it is
// the wrong trade here: the payload is what makes every refusal name the check
// that refused (see `backend::metal::error::Status`), the type is `Copy` and
// compared by value at hundreds of sites, and the cost being complained about
// is stack traffic on a **failure** path. A new error type that is large for no
// such reason should still be boxed rather than added to this exemption.
#[allow(clippy::result_large_err, clippy::large_enum_variant)]
pub mod metal;
#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

use crate::model::DeviceState;
use crate::runtime::draw::{DrawEncodeRequest, EncodeStatus};
use crate::runtime::host::{HostMemory, HostOps};

/// What the device executes guest work through.
///
/// The trait names the operations the runtime cannot perform itself, and
/// nothing else. It once declared the whole Metal-semantic operation set —
/// texture create/write/read, blit, compute, render, present — and nothing
/// called any of it, because the runtime reached the two backends through
/// same-named free functions that a `cfg` chose between. What that cost is not
/// tidiness: it made the two rails *unbuildable together*, so no run could
/// answer "is this a metal2vulkan defect or a device defect" by executing the
/// same guest stream both ways.
///
/// # Why a handle is `Copy` and its methods take `&self`
///
/// Neither backend owns its GPU. Metal's `MTLDevice` is a process-global
/// `OnceCell` ([`metal::runtime::system_device`]) and Vulkan's context is a
/// process-global `Lazy<Mutex<…>>` ([`vulkan::engine`]); a handle here is a
/// *name* for one of those, not the thing itself. Saying so in the type is what
/// lets a caller hold a backend while it mutably borrows the device state the
/// backend is about to act on — which is every call site, because the runtime's
/// unit of work is `(&mut DeviceState, &mut impl HostOps)`. A `&mut self`
/// backend would have to be borrowed out of that same state and could not be.
pub trait Backend: Copy {
    /// What this backend calls itself on the fail channel and in QEMU's trace.
    ///
    /// One of the arm names `scripts/feature-matrix` builds, so a log line and
    /// a build cell can be matched by eye.
    fn name(&self) -> &'static str;

    /// Drop all state derived from the current guest lifetime.
    ///
    /// Immutable, content-keyed shader/pipeline caches may survive. Guest object
    /// identities, resident images, and aliases of guest memory must not.
    fn reset(&self);

    /// Execute one decoded draw chain, and land its colour target in guest
    /// memory when `writeback_guest`.
    ///
    /// `req` is backend-neutral: every field is a decoded guest fact, and the
    /// two rails read the same request. What differs is entirely below this
    /// call — Metal encodes an `MTLRenderCommandEncoder`, Vulkan builds an
    /// `engine::DrawRequest` — and the asymmetry that matters is *not* in the
    /// arguments but in the Store: each rail owns when the frame reaches the
    /// guest's pages, which is why the return carries the tight RGBA8 colour 0
    /// rather than a promise that the write happened.
    ///
    /// `force_full_store` is a Metal term (its Store can be scissor-local); the
    /// Vulkan rail ignores it. It is a parameter rather than a rail-specific
    /// entry point because the *caller's* reason for setting it — an abandoned
    /// chain whose partial target must not be published — is a device fact, not
    /// a Metal one.
    fn encode_draw_chain<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        req: &mut DrawEncodeRequest,
        writeback_guest: bool,
        force_full_store: bool,
    ) -> (EncodeStatus, Option<Vec<u8>>);

    /// Execute a range of an indirect command buffer the guest has filled.
    ///
    /// A backend may not have one: `executeCommandsInBuffer:` is a Metal
    /// concept, and the Vulkan rail answers
    /// [`EncodeStatus::NoMetal`] rather than pretending. That refusal is the
    /// contract — the caller turns it into a latched `render_icb` refusal — so
    /// this is not a method with a silent default.
    fn encode_icb_execute_and_writeback<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        req: &DrawEncodeRequest,
        icb_ref: u32,
        range_location: u64,
        range_length: u64,
    ) -> EncodeStatus;
}

/// The backend a device executes through, resolved once per process.
///
/// A fieldless `Copy` enum rather than a boxed `dyn Backend`: [`Backend`]'s
/// execution methods are generic over the host-memory type the whole runtime is
/// generic over, so the trait is not object safe, and erasing that type would
/// put a virtual call on the per-draw path this device is CPU-bound on. The
/// variants are exactly the arms the build compiled, so a one-backend build
/// dispatches through a match with one arm and the optimiser removes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectedBackend {
    /// Native Metal on an Apple host. See [`metal`].
    #[cfg(feature = "backend-metal")]
    Metal(metal::MetalBackend),
    /// Vulkan — a native ICD on Linux, MoltenVK on an Apple host. See
    /// [`vulkan`].
    #[cfg(feature = "backend-vulkan")]
    Vulkan(vulkan::VulkanBackend),
}

impl Backend for SelectedBackend {
    fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.name(),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.name(),
        }
    }

    fn reset(&self) {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.reset(),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.reset(),
        }
    }

    fn encode_draw_chain<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        req: &mut DrawEncodeRequest,
        writeback_guest: bool,
        force_full_store: bool,
    ) -> (EncodeStatus, Option<Vec<u8>>) {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => {
                b.encode_draw_chain(state, host, req, writeback_guest, force_full_store)
            }
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => {
                b.encode_draw_chain(state, host, req, writeback_guest, force_full_store)
            }
        }
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
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.encode_icb_execute_and_writeback(
                state,
                host,
                req,
                icb_ref,
                range_location,
                range_length,
            ),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.encode_icb_execute_and_writeback(
                state,
                host,
                req,
                icb_ref,
                range_location,
                range_length,
            ),
        }
    }
}

/// The backend this process runs on.
///
/// Latched, and read through this function everywhere. The choice is a property
/// of the *process*, not of a device: both backends reach their GPU through a
/// process-global singleton, so two devices in one QEMU could not run on
/// different rails even if the selection were per-device. Latching also means a
/// log covers one rail — a boot that changed its mind halfway would be two
/// devices in one fail log, which is the reason
/// [`crate::runtime::writeback_debt::lazy_writeback_enabled`] gives for latching
/// its own switch.
pub fn selected() -> SelectedBackend {
    static SELECTED: std::sync::OnceLock<SelectedBackend> = std::sync::OnceLock::new();
    *SELECTED.get_or_init(select)
}

/// Resolve the process's backend from what this build compiled.
///
/// With one arm compiled there is nothing to decide, and in particular a failed
/// Metal probe is **not** a reason to refuse: a Metal build with no `MTLDevice`
/// already runs today and reports the absence where a draw needs it, which
/// names the missing device instead of the device create that came first.
/// Probing is what a build carrying *both* arms will select on.
fn select() -> SelectedBackend {
    #[cfg(feature = "backend-metal")]
    {
        SelectedBackend::Metal(metal::MetalBackend::probe())
    }
    #[cfg(feature = "backend-vulkan")]
    {
        SelectedBackend::Vulkan(vulkan::VulkanBackend::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The process answers with one rail, the same one every time, and it is a
    /// rail this build actually compiled.
    #[test]
    fn the_selected_backend_is_latched_and_names_a_compiled_arm() {
        let first = selected();
        assert_eq!(first, selected());
        assert!(matches!(first.name(), "metal" | "vulkan"));
        #[cfg(feature = "backend-metal")]
        assert_eq!(first.name(), "metal");
        #[cfg(feature = "backend-vulkan")]
        assert_eq!(first.name(), "vulkan");
    }
}
