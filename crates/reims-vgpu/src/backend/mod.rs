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

use crate::model::{ComputeStorageResidencyKey, DeviceInfoLimits, DeviceState};
use crate::runtime::blit_exec::{BlitStatus, LinearTextureLevel, Type11Texture};
use crate::runtime::compute_exec::{ComputeAccum, ComputeStatus};
use crate::runtime::compute_session::ComputeSession;
use crate::runtime::decode::blit::Command as BlitCommand;
use crate::runtime::decode::compute::Command as ComputeCommand;
use crate::runtime::draw::{DrawEncodeRequest, EncodeStatus};
use crate::runtime::guest_ram::ImportId;
use crate::runtime::host::{HostMemory, HostOps};

/// What the device executes guest work through.
///
/// `pub(crate)`: this is the device's internal seam, and its vocabulary is the
/// device's own resolved state — a `DrawEncodeRequest`, a resolved blit
/// endpoint. Nothing outside this crate implements a backend or calls one, so
/// making the trait public would only mean publishing those types to keep a
/// signature legal.
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
pub(crate) trait Backend: Copy {
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

    /// Copy a whole texture plane the rail already holds, without routing the
    /// content through the source's guest pages.
    ///
    /// `None` is the normal answer and is never a loss: the caller runs its host
    /// copy loop unchanged and lands the same pixels. So the method is an
    /// *optimisation the rail may decline*, and it is shaped that way rather
    /// than as a status a caller has to interpret — a rail with no resident
    /// registry declines everything, which is exactly what "there is nothing on
    /// the GPU to copy" means.
    fn try_copy_whole_plane_on_gpu<M: HostMemory + HostOps>(
        &self,
        _state: &mut DeviceState,
        _host: &mut M,
        _task_id: u32,
        _cmd: &BlitCommand,
    ) -> Option<BlitStatus> {
        None
    }

    /// [`Self::try_copy_whole_plane_on_gpu`] for a type-11 source landing in a
    /// guest-linear destination, which resolves its endpoints differently.
    #[allow(
        clippy::too_many_arguments,
        reason = "both endpoints are already resolved backings; re-resolving them                   inside the rail is the crossing this path exists to avoid"
    )]
    fn try_copy_t11_plane_to_linear_on_gpu<M: HostMemory + HostOps>(
        &self,
        _state: &mut DeviceState,
        _host: &mut M,
        _task_id: u32,
        _destination_ref: u32,
        _src: &Type11Texture,
        _dst: &LinearTextureLevel,
    ) -> Option<BlitStatus> {
        None
    }

    /// Execute a direct or indirect compute dispatch.
    ///
    /// Like [`Self::encode_draw_chain`], the request the two rails read is the
    /// same: `acc` is the accumulated bind state the guest declared and `cmd`
    /// the decoded dispatch, both backend-neutral. Each rail stages the guest's
    /// bytes for itself, because *what* has to be staged differs — the Vulkan
    /// rail may skip a guest read entirely when the engine already holds the
    /// window, and the Metal rail has no such mirror to skip against.
    fn execute_dispatch<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        acc: &ComputeAccum,
        cmd: &ComputeCommand,
    ) -> ComputeStatus;

    /// Open a multi-record encoder for one compute segment.
    ///
    /// A rail that cannot hold anything open across records refuses here, and
    /// that refusal is the whole contract: the segment latches a
    /// `SequencingBlock` and every later control-flow or ICB record in it is
    /// declined with the same reason. It is *one* refusal at the point the
    /// capability is asked for, rather than one at each of the four entry
    /// points a session would otherwise have to decline.
    #[allow(
        clippy::result_large_err,
        reason = "ComputeStatus carries the Metal backend's 264-byte Status so a \
                  refusal names the check that refused; see this module's note \
                  on the same exemption for `backend::metal`"
    )]
    fn open_compute_session(&self, dispatch_type: u32) -> Result<ComputeSession, ComputeStatus>;

    /// Execute a dispatch onto an already-open session's encoder.
    ///
    /// Only reachable through [`Self::open_compute_session`], so the session is
    /// always this rail's own.
    fn execute_dispatch_nested<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        acc: &ComputeAccum,
        cmd: &ComputeCommand,
        session: &mut ComputeSession,
    ) -> ComputeStatus;

    // --- Guest memory the rail also touches -------------------------------
    //
    // A rail may write guest pages from the GPU, and may hold an alias of guest
    // RAM the host mapped for it. The device has to know when those writes have
    // landed before it reads the same bytes on the CPU, and has to unmap the
    // host view only after the rail has let go of it. A rail that does neither
    // takes every default below, and the callers stay one shape.

    /// Whether this rail has guest-page writes submitted but not yet executed.
    ///
    /// One relaxed load on the rail that has them, and the gate every settle
    /// site opens with — a caller with nothing outstanding pays only this.
    fn guest_writes_outstanding(&self) -> bool {
        false
    }

    /// Block until they have.
    fn quiesce_guest_writes(&self) {}

    /// Whether anything outstanding lands in `pages`.
    ///
    /// A rail with no outstanding writes cannot reach anything, so the default
    /// is [`GuestWriteReach::Disjoint`] rather than
    /// [`GuestWriteReach::Unnamed`]: "nothing to wait for" is a fact, not a
    /// failure to answer.
    fn guest_writes_reaching(&self, _pages: &[u64]) -> GuestWriteReach {
        GuestWriteReach::Disjoint
    }

    /// Give up the rail's alias of a retired guest import.
    ///
    /// Returns the `(ptr, len)` host view the rail released, if it held one.
    /// The caller must not unmap a view the rail still owns, which is why this
    /// answers with the view rather than with a bare success.
    fn retire_guest_import(&self, _import: ImportId) -> Option<(usize, usize)> {
        None
    }

    /// Host views whose fence-safe destruction has completed since the last ask.
    ///
    /// The other half of [`Self::retire_guest_import`]: a rail may hold a view
    /// past the retirement that asked for it, and this is how the view gets
    /// unmapped without waiting for another guest mapping event.
    fn take_released_host_aliases(&self) -> Vec<(usize, usize)> {
        Vec::new()
    }

    /// Release the rail's residents for linear cache entries the guest deleted.
    fn retire_linear_residents(&self, _keys: &[ComputeStorageResidencyKey]) {}

    // --- Cadence ----------------------------------------------------------

    /// Idle-time upkeep, on the device heartbeat.
    fn maintain(&self, _now_ms: u64) {}

    /// Submit anything the rail has been holding back for batching.
    ///
    /// Called at the drain tail, so a deferred batch cannot sit until the next
    /// guest packet arrives.
    fn flush_deferred_submissions(&self) {}

    /// Count whether this rail already holds a type-11 surface's content.
    ///
    /// Census only: it changes no decision, and the caller copies the same bytes
    /// either way. A rail with no resident registry records nothing rather than
    /// a "not ready" reading it would have to be told to discount.
    fn note_blit_t11_resident(&self, _state: &DeviceState, _mapping_id: u32) {}

    // --- What the guest is told the GPU can do ------------------------------
    //
    // `CmdGetDeviceInfo` and `CmdGetComputeInfo` are asked **once** per boot and
    // the guest keeps the answer for the life of that boot, so both of these
    // describe the executing host GPU or they mislead the guest permanently.
    // Neither has a default: "what can this GPU do" has no answer that is right
    // for a rail that has not been asked, and a rail that inherited a neighbour's
    // table would report a device it is not.

    /// The device-info keys that describe the GPU rather than the protocol.
    fn device_info_limits(&self) -> DeviceInfoLimits;

    /// `(maxTotalThreadsPerThreadgroup, threadExecutionWidth)`.
    ///
    /// The two compute-info keys this device answers from the host GPU. The
    /// third key it serves — static threadgroup memory — is a property of the
    /// *pipeline* the guest named and not of the GPU, so it is not asked here.
    fn compute_threadgroup_limits(&self) -> (u32, u32);

    // --- What the rail's resident registry can say about a present ----------
    //
    // Both questions are about one surface identity, and both are answered by
    // the pools a rail may not have. `None` / `false` are the honest readings
    // for a rail with no registry, not degraded ones — see each method.

    /// Would a resident carry the present this mapping names, at this geometry?
    ///
    /// `Some(true)` a presentable resident exists, `Some(false)` none does — so
    /// a present with no guest-page frame behind it shows black — and `None` on
    /// a rail with no target registry to ask, where the honest answer is that
    /// this rail cannot tell. The caller fails closed on `None`
    /// (`unbacked_present_is_a_loss`), so a rail that cannot answer never
    /// demotes a possible black frame to a census.
    fn present_resident_carries(
        &self,
        _state: &DeviceState,
        _mapping: u32,
        _width: u32,
        _height: u32,
    ) -> Option<bool> {
        None
    }

    /// Fill `buf` from the mapping's GPU resident, without any guest-page
    /// scatter.
    ///
    /// Returns whether the resident supplied the whole frame. On `true` `buf`
    /// holds tight BGRA8; on `false` `buf` is untouched and the capture fails
    /// (keep-prior) — there is no guest-page path left for the caller to take.
    ///
    /// A rail with no resident registry answers `false` for every present, so
    /// its console holds its prior retain. That is a known gap in this pathway
    /// rather than a rail-specific one, which is why it is spelled as this
    /// method's default and not as a second capture vein.
    fn try_capture_from_resident(
        &self,
        _state: &mut DeviceState,
        _buf: &mut Vec<u8>,
        _mapping_id: u32,
        _width: u32,
        _height: u32,
    ) -> bool {
        false
    }

    /// Publish a FIFO completion stamp, ordered behind the guest-memory work it
    /// completes.
    ///
    /// The guest's fence moving is what frees everything the completed work
    /// allocated, so **before this returns, everything this device owes guest
    /// RAM for that work has to be in guest RAM.** The two answers differ only
    /// in *who writes the word*, never in whether that debt was paid:
    ///
    /// * [`StampOrdering::Queued`] — the rail attached the word to a GPU
    ///   submission and will publish it, and raise the interrupt, from its own
    ///   completion thread. The caller must not write the word.
    /// * [`StampOrdering::Settled`] — the debt has landed and the caller writes
    ///   the word on the CPU.
    ///
    /// The default is `Settled` after a blocking settle at `site`, which is the
    /// correct answer for any rail that cannot attach a word to a submission:
    /// the wait is what the ordering costs when it cannot be expressed as one.
    /// A rail that *can* attach one must still return `Settled` for the stamps
    /// it declines, and must pay the same debt on that arm before it does.
    fn order_completion_stamp<M: HostMemory + HostOps>(
        &self,
        _state: &DeviceState,
        _host: &mut M,
        _index: u32,
        _value: u32,
        site: crate::runtime::render_writeback::SettleSite,
    ) -> StampOrdering {
        crate::runtime::render_writeback::settle_guest_writes(site);
        StampOrdering::Settled
    }

    /// Emit this rail's census lines for one point in the drain's census window.
    ///
    /// Census only, and the reason it is a trait method rather than a `cfg` is
    /// that a `cfg` answers "which rail was compiled" and this asks "which rail
    /// is *running*". A build carrying both would print one rail's engine
    /// counters for a device executing on the other, and — worse, because it is
    /// silent — would drop the Metal cache-levels line entirely, since its gate
    /// spelled "the Metal arm" as `not(backend-vulkan)`.
    ///
    /// A rail with nothing to say at a site says nothing. That is not the same
    /// as a zero: an absent `engine_delta` means no such engine, where
    /// `engine_delta …=0` would mean an idle one.
    fn emit_census(&self, _site: CensusSite) {}
}

/// A point in the drain's per-second census window at which a rail may
/// contribute lines — the vocabulary of [`Backend::emit_census`].
///
/// The **order** of these points is the drain's and not a rail's, which is why
/// they are named for the drain's reason rather than for any rail's tables:
/// several of the neutral lines emitted between them are only interpretable
/// read against a rail line that must come first. Ordering *within* a site is
/// the rail's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CensusSite {
    /// Beside `window_publish`, which it divides: `window_publish fresh` is
    /// what the device offered the window and `host_window_cadence presents` is
    /// what reached the screen, and when the two disagree the first candidate is
    /// the rail's own serialization. Carries the window the drain measured.
    Serialization { win_ms: u64 },
    /// Beside the eviction routes: those say which cap fired and this says how
    /// much the workload wanted, and neither is interpretable without the other.
    WorkingSet,
    /// Before the neutral `chain_phase`, which divides against it — reading them
    /// in the other order invites treating a rail's phases as the whole draw.
    Throughput,
    /// After `chain_phase`. Levels, not per-interval deltas: entry counts of the
    /// caches that hold one entry per distinct guest object.
    Levels,
}

/// Who publishes a FIFO completion stamp word — the vocabulary of
/// [`Backend::order_completion_stamp`], and neutral because the *caller's*
/// obligation is the same on any rail: write the word, or do not.
///
/// Two cases rather than the three a rail may distinguish internally. A rail
/// that separates "nothing was owed" from "the async route declined" keeps that
/// distinction where it is load-bearing — inside itself, where the settle
/// either happens or provably need not — because both reach the drain as the
/// same instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StampOrdering {
    /// The rail took the word. It publishes it and raises the interrupt; the
    /// caller advances its fence sequence and writes nothing.
    Queued,
    /// The caller writes the word. Everything owed guest RAM has landed.
    Settled,
}

/// Whether a rail's outstanding guest-page writes can reach a set of pages.
///
/// The vocabulary of [`Backend::guest_writes_reaching`], and neutral because
/// the question is: a host-side reader of guest bytes has to know whether a GPU
/// write it did not order against is about to land in them, and that is true on
/// any rail that writes guest memory. It lived in the Vulkan engine, and
/// `gather_witness` kept a second enum with the same three cases beside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuestWriteReach {
    /// Nothing outstanding lands in any of the pages asked about. The caller may
    /// read them without settling.
    Disjoint,
    /// An outstanding writeback lands in one of them.
    Overlap,
    /// The ledger cannot say, so the caller must settle. Distinguished from
    /// [`Self::Overlap`] because the two want opposite fixes: an overlap is a
    /// wait genuinely owed and this is precision the ledger failed to keep.
    Unnamed,
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

    fn try_copy_whole_plane_on_gpu<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        cmd: &BlitCommand,
    ) -> Option<BlitStatus> {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.try_copy_whole_plane_on_gpu(state, host, task_id, cmd),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.try_copy_whole_plane_on_gpu(state, host, task_id, cmd),
        }
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
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.try_copy_t11_plane_to_linear_on_gpu(
                state,
                host,
                task_id,
                destination_ref,
                src,
                dst,
            ),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.try_copy_t11_plane_to_linear_on_gpu(
                state,
                host,
                task_id,
                destination_ref,
                src,
                dst,
            ),
        }
    }

    fn execute_dispatch<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        acc: &ComputeAccum,
        cmd: &ComputeCommand,
    ) -> ComputeStatus {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.execute_dispatch(state, host, task_id, acc, cmd),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.execute_dispatch(state, host, task_id, acc, cmd),
        }
    }

    #[allow(clippy::result_large_err, reason = "see the trait declaration")]
    fn open_compute_session(&self, dispatch_type: u32) -> Result<ComputeSession, ComputeStatus> {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.open_compute_session(dispatch_type),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.open_compute_session(dispatch_type),
        }
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
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.execute_dispatch_nested(state, host, task_id, acc, cmd, session),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.execute_dispatch_nested(state, host, task_id, acc, cmd, session),
        }
    }

    fn guest_writes_outstanding(&self) -> bool {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.guest_writes_outstanding(),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.guest_writes_outstanding(),
        }
    }

    fn quiesce_guest_writes(&self) {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.quiesce_guest_writes(),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.quiesce_guest_writes(),
        }
    }

    fn guest_writes_reaching(&self, pages: &[u64]) -> GuestWriteReach {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.guest_writes_reaching(pages),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.guest_writes_reaching(pages),
        }
    }

    fn retire_guest_import(&self, import: ImportId) -> Option<(usize, usize)> {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.retire_guest_import(import),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.retire_guest_import(import),
        }
    }

    fn take_released_host_aliases(&self) -> Vec<(usize, usize)> {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.take_released_host_aliases(),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.take_released_host_aliases(),
        }
    }

    fn retire_linear_residents(&self, keys: &[ComputeStorageResidencyKey]) {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.retire_linear_residents(keys),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.retire_linear_residents(keys),
        }
    }

    fn maintain(&self, now_ms: u64) {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.maintain(now_ms),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.maintain(now_ms),
        }
    }

    fn flush_deferred_submissions(&self) {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.flush_deferred_submissions(),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.flush_deferred_submissions(),
        }
    }

    fn note_blit_t11_resident(&self, state: &DeviceState, mapping_id: u32) {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.note_blit_t11_resident(state, mapping_id),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.note_blit_t11_resident(state, mapping_id),
        }
    }

    fn device_info_limits(&self) -> DeviceInfoLimits {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.device_info_limits(),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.device_info_limits(),
        }
    }

    fn compute_threadgroup_limits(&self) -> (u32, u32) {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.compute_threadgroup_limits(),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.compute_threadgroup_limits(),
        }
    }

    fn present_resident_carries(
        &self,
        state: &DeviceState,
        mapping: u32,
        width: u32,
        height: u32,
    ) -> Option<bool> {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.present_resident_carries(state, mapping, width, height),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.present_resident_carries(state, mapping, width, height),
        }
    }

    fn try_capture_from_resident(
        &self,
        state: &mut DeviceState,
        buf: &mut Vec<u8>,
        mapping_id: u32,
        width: u32,
        height: u32,
    ) -> bool {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.try_capture_from_resident(state, buf, mapping_id, width, height),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.try_capture_from_resident(state, buf, mapping_id, width, height),
        }
    }

    fn order_completion_stamp<M: HostMemory + HostOps>(
        &self,
        state: &DeviceState,
        host: &mut M,
        index: u32,
        value: u32,
        site: crate::runtime::render_writeback::SettleSite,
    ) -> StampOrdering {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.order_completion_stamp(state, host, index, value, site),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.order_completion_stamp(state, host, index, value, site),
        }
    }

    fn emit_census(&self, site: CensusSite) {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.emit_census(site),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.emit_census(site),
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
