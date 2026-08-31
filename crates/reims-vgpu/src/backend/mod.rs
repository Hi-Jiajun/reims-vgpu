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
use crate::runtime::draw::{DrawEncodeRequest, EncodeStatus, GvaSpan};
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

    /// What this rail remembers drawing into one plane since this witness last
    /// asked, formatted as census fields.
    ///
    /// A rail that keeps no such record answers with the **empty string**, so
    /// the fields are absent from the line rather than present and zero. That is
    /// the whole reason this is not a count: `draws=0` is a reading — "the guest
    /// stopped compositing into this plane" — and a rail with no ring has not
    /// taken it.
    fn plane_draw_witness(&self, _reader: PlaneDrawReader, _mapping_id: u32) -> String {
        String::new()
    }

    /// The raster sample count the bound pipeline declares, when this rail has
    /// to make the attachment match it.
    ///
    /// Metal requires a pipeline's `rasterSampleCount` to equal the sample count
    /// of every colour attachment it renders into. The *target* side of that
    /// equation is not recoverable here — a linear texture resource's dimensions
    /// do not retain its creation descriptor's sample count, so every resolved
    /// render target carries a provisional `1` — which leaves the pipeline as the
    /// only place the real count is written down.
    ///
    /// `None` is the default and means "this rail does not consult the
    /// pipeline": its encoder derives the attachment's sample count from the
    /// attachment it is handed. It is not a failure to resolve, and the caller
    /// treats it as "keep the target's count" rather than as a refusal. It also
    /// silences `note_attachment_sample_count_override`, which has nothing to
    /// report when no count was taken from a pipeline.
    fn pipeline_raster_sample_count<M: HostMemory + HostOps>(
        &self,
        _state: &DeviceState,
        _host: &M,
        _task_id: u32,
        _pipeline_ref: u32,
    ) -> Option<u32> {
        None
    }

    // --- Pixels the rail may already hold -----------------------------------
    //
    // Two questions with one shape: this device is about to move a frame's
    // bytes through guest memory, and a rail that still holds those bytes can
    // supply them without the round trip. Both default to "no", and on a rail
    // with no resident registry that is not a degraded answer — there is
    // nothing held, so the caller does exactly the work it would have done.

    /// Whether the rail already holds, current and unmodified, the pixels a
    /// colour LOAD is about to gather out of `span`'s guest pages.
    ///
    /// `true` means the caller may skip building the seed entirely: the rail
    /// will load from what it holds. The rail is answering about its **own**
    /// state, and answering `true` binds it — the encode either honours the
    /// elision or re-seeds — which is why this is one call rather than a query
    /// the caller interprets.
    ///
    /// `&mut` on both state and host because the answer is measured, not looked
    /// up: deciding it consults the guest-write witness for those pages.
    fn gva_load_seed_elidable<M: HostMemory + HostOps>(
        &self,
        _state: &mut DeviceState,
        _host: &mut M,
        _task_id: u32,
        _span: GvaSpan,
    ) -> bool {
        false
    }

    /// Colour 0 of a chain this device is abandoning, read back from the rail.
    ///
    /// The chain broke, so no span carries the key its last good record
    /// registered and the rail has to name the target from the state it can
    /// still see. That second derivation belongs to the rail rather than to the
    /// caller: every *other* reader of a chain has the draw's own key, and a
    /// shared re-derivation would silently give them this one's answer.
    fn read_abandoned_chain_rgba(
        &self,
        _state: &DeviceState,
        _req: &DrawEncodeRequest,
    ) -> Option<Vec<u8>> {
        None
    }

    /// Filtered mip levels 0.. for one texture, generated on the GPU.
    ///
    /// `generateMipmaps` is a Metal blit-encoder operation and its result is
    /// **filtered**, which a shared CPU box filter approximates rather than
    /// reproduces. So the three answers are genuinely three, not two with a
    /// flag: a rail with no filtered path here declines and the caller box
    /// filters (same levels, slower — never a loss), while a rail that *has*
    /// one and was refused by it must not be silently box-filtered, because the
    /// guest would keep a texture whose upper levels came from a path the
    /// refusal says was not lawful.
    ///
    /// `texture_ref` is here so a rail declining for a reason worth naming can
    /// name the texture it declined for. Reporting the decline is the rail's,
    /// since only the rail knows the reason.
    #[allow(
        clippy::too_many_arguments,
        reason = "level 0's geometry, format and bytes are the operation's whole                   input; bundling them into a struct used at one call site would                   add a type without removing a field"
    )]
    fn generate_mipmap_chain(
        &self,
        _texture_ref: u32,
        _fmt: u16,
        _width: u32,
        _height: u32,
        _levels: u32,
        _level0: &[u8],
    ) -> MipmapGeneration {
        MipmapGeneration::Unfiltered
    }

    /// Whether any of this submission's streams still needs work this rail must
    /// finish before the packet may be consumed.
    ///
    /// `true` defers the whole packet: it is left unconsumed and retried, so a
    /// replay cannot duplicate the clears, fences, dispatches or guest
    /// writeback it contains. That makes this a **safety** answer rather than an
    /// optimisation — a rail that answers `false` is promising the packet can be
    /// executed to completion now.
    ///
    /// The default is `false` and is exactly that promise for a rail with
    /// nothing to prepare: it compiles no shader ahead of the record that uses
    /// one, so there is nothing a retry could find further along.
    fn preflight_translations<M: HostMemory + HostOps>(
        &self,
        _state: &DeviceState,
        _host: &M,
        _task_id: u32,
        _streams: &[Vec<u8>],
    ) -> bool {
        false
    }
}

/// What a rail made of a `generateMipmaps` — the vocabulary of
/// [`Backend::generate_mipmap_chain`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MipmapGeneration {
    /// Level 0.., each `(width, height, tight native bytes)`.
    Chain(Vec<(u32, u32, Vec<u8>)>),
    /// This rail has no filtered path for this request. The caller runs the
    /// shared CPU box filter, which lands the same levels more slowly.
    Unfiltered,
    /// This rail has a filtered path, tried it, and it refused. Distinguished
    /// from [`Self::Unfiltered`] because the two want opposite handling: this
    /// one must reach the guest as a refusal rather than as a quieter path.
    Refused(crate::runtime::mipmap::MipmapStatus),
}

/// Which witness is asking, because two of them ask about the same plane rings
/// for different questions and neither may consume the other's window.
///
/// [`crate::runtime::scanout::note_present_field_witness`] asks about the plane
/// a present names; `note_sampled_surface_field` asks about a full-screen layer
/// a draw sampled, and on a rail where the compositor's presented planes are
/// also sampled layers a single destructive drain gave whichever witness fired
/// first the whole window and the other one `draws=0` — which is exactly the
/// reading that separates "a pass produced this field" from "nothing drew into
/// this surface", so the shared drain manufactured the more alarming of the two
/// answers.
///
/// Neutral, and here rather than beside the ring, because it names the two
/// *witnesses* — both of which are runtime census sites that exist on every
/// rail — and not the ring, which only one rail keeps.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PlaneDrawReader {
    /// The plane a present named.
    PresentedPlane,
    /// A full-screen layer a draw sampled.
    SampledLayer,
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

impl SelectedBackend {
    /// Which rail this is, by name only.
    ///
    /// The inverse of [`build`], and the form a report or a comparison wants: a
    /// handle carries a live GPU context and a [`Rail`] carries the answer.
    pub fn rail(self) -> Rail {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(_) => Rail::Metal,
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(_) => Rail::Vulkan,
        }
    }
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

    fn plane_draw_witness(&self, reader: PlaneDrawReader, mapping_id: u32) -> String {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.plane_draw_witness(reader, mapping_id),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.plane_draw_witness(reader, mapping_id),
        }
    }

    fn pipeline_raster_sample_count<M: HostMemory + HostOps>(
        &self,
        state: &DeviceState,
        host: &M,
        task_id: u32,
        pipeline_ref: u32,
    ) -> Option<u32> {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.pipeline_raster_sample_count(state, host, task_id, pipeline_ref),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.pipeline_raster_sample_count(state, host, task_id, pipeline_ref),
        }
    }

    fn gva_load_seed_elidable<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        span: GvaSpan,
    ) -> bool {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.gva_load_seed_elidable(state, host, task_id, span),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.gva_load_seed_elidable(state, host, task_id, span),
        }
    }

    fn read_abandoned_chain_rgba(
        &self,
        state: &DeviceState,
        req: &DrawEncodeRequest,
    ) -> Option<Vec<u8>> {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.read_abandoned_chain_rgba(state, req),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.read_abandoned_chain_rgba(state, req),
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
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => {
                b.generate_mipmap_chain(texture_ref, fmt, width, height, levels, level0)
            }
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => {
                b.generate_mipmap_chain(texture_ref, fmt, width, height, levels, level0)
            }
        }
    }

    fn preflight_translations<M: HostMemory + HostOps>(
        &self,
        state: &DeviceState,
        host: &M,
        task_id: u32,
        streams: &[Vec<u8>],
    ) -> bool {
        match self {
            #[cfg(feature = "backend-metal")]
            Self::Metal(b) => b.preflight_translations(state, host, task_id, streams),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(b) => b.preflight_translations(state, host, task_id, streams),
        }
    }
}

/// A rail by name — what an operator may ask for and what this device reports.
///
/// Names only, because the decision that picks one is made *before* a handle
/// exists: bringing a rail up is what measures it, and a table whose cells were
/// live backends could not be evaluated without bringing both up. The spelling
/// is the single source for the env value, the boot line and every refusal —
/// [`Backend::name`] is derived from it, so a rail cannot be called one thing in
/// a log and another in a variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rail {
    /// Native Metal on an Apple host.
    Metal,
    /// Vulkan — a native ICD on Linux, MoltenVK on an Apple host.
    Vulkan,
}

impl Rail {
    /// What this rail calls itself, everywhere.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        }
    }

    /// Every rail this crate can name, which is what [`crate::env::RAIL`] is
    /// parsed against.
    ///
    /// Deliberately *not* narrowed to the rails a build compiled: an operator
    /// who names a rail this binary does not carry has to be told that, and a
    /// list that omitted it would report the ask as a misspelling instead.
    pub const NAMES: [&'static str; 2] = [Self::Metal.name(), Self::Vulkan.name()];

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "metal" => Some(Self::Metal),
            "vulkan" => Some(Self::Vulkan),
            _ => None,
        }
    }

    /// The other one. Used where a refusal has to fall back to something.
    fn other(self) -> Self {
        match self {
            Self::Metal => Self::Vulkan,
            Self::Vulkan => Self::Metal,
        }
    }
}

/// Which rails a build carries.
///
/// An enum rather than a pair of `bool`s because "neither" is not a build
/// `lib.rs` permits, and two booleans would make that fourth state
/// representable in the very table that decides what executes guest work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compiled {
    /// Only [`Rail::Metal`].
    MetalOnly,
    /// Only [`Rail::Vulkan`].
    VulkanOnly,
    /// Both, which is the configuration this whole seam exists for: one binary
    /// that can run one guest stream two ways, so a defect can be attributed to
    /// the translation or to this device.
    Both,
}

/// What the operator asked for through [`crate::env::RAIL`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RailRequest {
    /// Nothing set; the device takes its own default.
    Unset,
    /// A rail this crate can name. Not permission — see [`resolve_rail`].
    Named(Rail),
    /// Set to something that is not a rail name at all, carrying the raw text.
    Unrecognized(String),
}

/// Why an operator's rail ask was not carried out.
///
/// Each variant is a *different operator mistake* with a different fix, which is
/// why they are not one "bad value": one needs a different build, one needs a
/// different host, and one needs a corrected spelling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RailRefusal {
    /// Named a rail this build does not carry. The fix is a build, not a run.
    NotCompiled(Rail),
    /// Named a rail this host has no device for. **This is the narrowing rule
    /// doing its job**: an override may turn a rail off and may never turn one
    /// on, so an ask that would put guest work in front of an absent device is
    /// refused rather than obeyed.
    NotAvailable(Rail),
    /// Not a rail name. Carries the raw text so the report quotes what the
    /// operator typed.
    Unrecognized(String),
}

impl RailRefusal {
    /// The registered slug for the always-on failure channel.
    pub fn slug(&self) -> &'static str {
        match self {
            Self::NotCompiled(_) => "rail_not_compiled",
            Self::NotAvailable(_) => "rail_not_available",
            Self::Unrecognized(_) => "rail_unrecognized",
        }
    }
}

/// The rail this process runs on, and anything the operator has to be told.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RailChoice {
    /// What runs.
    pub rail: Rail,
    /// `Some` when the ask could not be met. The device runs `rail` anyway —
    /// refusing to start would turn a mistyped variable into a machine that
    /// does not boot — but the refusal is on the always-on channel.
    pub refusal: Option<RailRefusal>,
}

/// Decide the rail from what the build carries, what the host was measured able
/// to run, and what the operator asked for — in that order of authority.
///
/// Pure and total, and compiled on **every** arm rather than only on the one
/// that can exercise both branches. The rule being encoded is arithmetic over
/// three inputs; needing two live GPUs present to check it would put its tests
/// on no arm at all, which is exactly how the `not(backend-vulkan)` gates this
/// seam replaced went stale unnoticed.
///
/// # `metal_available` is measured, and Vulkan's absence is not measurable here
///
/// Metal's probe is one `MTLCreateSystemDefaultDevice`, so "this host can run
/// Metal" is a fact by the time this is called. Vulkan has no equivalent: its
/// context is created lazily at the first real encode, precisely so protocol
/// tests need no ICD, and asking whether an ICD exists *is* the instance
/// creation that laziness defers. So a compiled Vulkan rail is treated as
/// runnable, and a host with no ICD reports it where the first encode needs one.
/// That asymmetry is real and is the reason Vulkan is the fallback rather than
/// the preference.
///
/// # Why one compiled rail is chosen even when its device is absent
///
/// With a single rail there is nothing to fall back *to*, and a refusal here
/// would replace "the draw found no Metal device" — which names the missing
/// device at the point it was needed — with a failure at device create, which
/// names the wrong thing. So `metal_available` only ever decides between two
/// rails; it never disqualifies the only one.
pub fn resolve_rail(
    compiled: Compiled,
    metal_available: bool,
    requested: RailRequest,
) -> RailChoice {
    let carried = |rail: Rail| {
        matches!(
            (compiled, rail),
            (Compiled::Both, _)
                | (Compiled::MetalOnly, Rail::Metal)
                | (Compiled::VulkanOnly, Rail::Vulkan)
        )
    };
    // Unset on a both-rails build takes Metal when the host has one: it is the
    // native rail on the only host that can carry both, and it is the reference
    // the other is being compared against.
    let default = match compiled {
        Compiled::MetalOnly => Rail::Metal,
        Compiled::VulkanOnly => Rail::Vulkan,
        Compiled::Both if metal_available => Rail::Metal,
        Compiled::Both => Rail::Vulkan,
    };
    let asked = match requested {
        RailRequest::Unset => {
            return RailChoice {
                rail: default,
                refusal: None,
            }
        }
        RailRequest::Unrecognized(raw) => {
            return RailChoice {
                rail: default,
                refusal: Some(RailRefusal::Unrecognized(raw)),
            }
        }
        RailRequest::Named(rail) => rail,
    };
    if !carried(asked) {
        return RailChoice {
            rail: asked.other(),
            refusal: Some(RailRefusal::NotCompiled(asked)),
        };
    }
    // The narrowing rule. Only reachable on a both-rails build: with one rail
    // carried, `carried` already sent the other ask away and this one is the
    // only rail there is.
    if asked == Rail::Metal && compiled == Compiled::Both && !metal_available {
        return RailChoice {
            rail: Rail::Vulkan,
            refusal: Some(RailRefusal::NotAvailable(Rail::Metal)),
        };
    }
    RailChoice {
        rail: asked,
        refusal: None,
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

/// What [`crate::env::RAIL`] says, in this module's vocabulary.
fn requested_rail() -> RailRequest {
    match crate::env::choice(crate::env::RAIL, &Rail::NAMES) {
        crate::env::Choice::Unset => RailRequest::Unset,
        crate::env::Choice::Named(name) => match Rail::from_name(name) {
            Some(rail) => RailRequest::Named(rail),
            // Unreachable while `Rail::NAMES` is what was parsed against, and
            // spelled as a value rather than as a panic because nothing here
            // may panic across the QEMU boundary.
            None => RailRequest::Unrecognized(name.to_owned()),
        },
        crate::env::Choice::Refused(raw) => RailRequest::Unrecognized(raw),
    }
}

/// Which rails this build carries.
///
/// The one `cfg` in the selection, and the honest one: every other question
/// here is about the *host* or the *operator*, and only this one is about the
/// build. `lib.rs` rejects a build with neither rail, which is what makes these
/// three arms total.
const fn compiled() -> Compiled {
    #[cfg(all(feature = "backend-metal", feature = "backend-vulkan"))]
    {
        Compiled::Both
    }
    #[cfg(all(feature = "backend-metal", not(feature = "backend-vulkan")))]
    {
        Compiled::MetalOnly
    }
    #[cfg(all(feature = "backend-vulkan", not(feature = "backend-metal")))]
    {
        Compiled::VulkanOnly
    }
}

/// Whether this host exposes a Metal device, measured rather than assumed.
///
/// `false` on a build with no Metal rail, which is not a measurement and does
/// not have to be: [`resolve_rail`] only consults it to choose *between* rails,
/// and a build without the Metal rail has no such choice to make.
fn metal_available() -> bool {
    #[cfg(feature = "backend-metal")]
    {
        metal::MetalBackend::available()
    }
    #[cfg(not(feature = "backend-metal"))]
    {
        false
    }
}

/// Bring up the chosen rail and report what was chosen.
///
/// The report is unconditional and on the census channel, because "which rail
/// ran" is the one fact every other line in the log has to be read against —
/// and on a binary carrying both, it is no longer answerable from the build.
fn select() -> SelectedBackend {
    let choice = resolve_rail(compiled(), metal_available(), requested_rail());
    if let Some(refusal) = &choice.refusal {
        crate::observe::fail(format!(
            "rail_refused reason={} asked={} running={} compiled={:?} \
             (an override may narrow what this device does and never widen it)",
            refusal.slug(),
            match refusal {
                RailRefusal::NotCompiled(rail) | RailRefusal::NotAvailable(rail) => rail.name(),
                RailRefusal::Unrecognized(raw) => raw.as_str(),
            },
            choice.rail.name(),
            compiled(),
        ));
    }
    crate::observe::off(format!(
        "rail_selected rail={} compiled={:?} metal_available={}",
        choice.rail.name(),
        compiled(),
        metal_available() as u8
    ));
    build(choice.rail)
}

/// Construct the handle for one rail.
///
/// Separate from [`select`] so the policy above is free of `cfg` and this is the
/// only place a rail's handle is created. The arms a build did not compile
/// cannot be asked for: [`resolve_rail`] is given [`compiled`] and never returns
/// one — but a total `match` still has to say what it would do, and it says the
/// same thing the selection already decided rather than panicking.
fn build(rail: Rail) -> SelectedBackend {
    #[cfg(feature = "backend-metal")]
    let metal = SelectedBackend::Metal(metal::MetalBackend::probe());
    #[cfg(feature = "backend-vulkan")]
    let vulkan = SelectedBackend::Vulkan(vulkan::VulkanBackend::new());
    #[cfg(all(feature = "backend-metal", feature = "backend-vulkan"))]
    match rail {
        Rail::Metal => metal,
        Rail::Vulkan => vulkan,
    }
    #[cfg(all(feature = "backend-metal", not(feature = "backend-vulkan")))]
    {
        let _ = rail;
        metal
    }
    #[cfg(all(feature = "backend-vulkan", not(feature = "backend-metal")))]
    {
        let _ = rail;
        vulkan
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
        assert_eq!(first.name(), first.rail().name());
        match compiled() {
            Compiled::MetalOnly => assert_eq!(first.rail(), Rail::Metal),
            Compiled::VulkanOnly => assert_eq!(first.rail(), Rail::Vulkan),
            // Both compiled: either is legal and which one is the *host's*
            // answer, so assert the property that holds on any host — that the
            // latched rail is the one the table names for this host and this
            // environment, rather than pinning a machine-specific outcome.
            Compiled::Both => assert_eq!(
                first.rail(),
                resolve_rail(Compiled::Both, metal_available(), requested_rail()).rail
            ),
        }
    }

    /// With nothing asked for, a build takes the rail it has — and, carrying
    /// both, the host's native one when the host has it.
    ///
    /// The `Both` rows are the reason this table exists: they are the only ones
    /// where "what did the build compile" and "what is running" differ, and
    /// before this seam they could not differ at all.
    #[test]
    fn an_unset_rail_takes_the_native_one_a_build_carries() {
        let unset = |compiled, metal| resolve_rail(compiled, metal, RailRequest::Unset);
        assert_eq!(unset(Compiled::MetalOnly, true).rail, Rail::Metal);
        assert_eq!(unset(Compiled::VulkanOnly, false).rail, Rail::Vulkan);
        assert_eq!(unset(Compiled::Both, true).rail, Rail::Metal);
        assert_eq!(unset(Compiled::Both, false).rail, Rail::Vulkan);
        // A single rail runs even with no device behind it: there is nothing to
        // fall back to, and the missing device has to be named where a draw
        // needs it rather than at device create.
        assert_eq!(unset(Compiled::MetalOnly, false).rail, Rail::Metal);
        for compiled in [Compiled::MetalOnly, Compiled::VulkanOnly, Compiled::Both] {
            for metal in [true, false] {
                assert_eq!(unset(compiled, metal).refusal, None);
            }
        }
    }

    /// An ask this build or this host cannot meet is refused, never obeyed.
    ///
    /// This is `AGENTS.md`'s narrowing rule at the one place it decides which
    /// GPU API executes guest work. Obeying `rail=metal` on a host with no
    /// `MTLDevice` would put every draw in front of an absent device; obeying it
    /// on a binary that did not compile the rail could not even be spelled.
    #[test]
    fn a_rail_ask_may_narrow_what_a_build_carries_and_may_never_widen_it() {
        // Not compiled: the fix is a build, and the device says so.
        let ask_metal_on_vulkan_build =
            resolve_rail(Compiled::VulkanOnly, false, RailRequest::Named(Rail::Metal));
        assert_eq!(ask_metal_on_vulkan_build.rail, Rail::Vulkan);
        assert_eq!(
            ask_metal_on_vulkan_build.refusal,
            Some(RailRefusal::NotCompiled(Rail::Metal))
        );

        // Not available: the fix is a host. Distinguished from the above
        // because an operator who conflated them would rebuild for nothing.
        let ask_metal_without_a_device =
            resolve_rail(Compiled::Both, false, RailRequest::Named(Rail::Metal));
        assert_eq!(ask_metal_without_a_device.rail, Rail::Vulkan);
        assert_eq!(
            ask_metal_without_a_device.refusal,
            Some(RailRefusal::NotAvailable(Rail::Metal))
        );

        // Narrowing proper: both carried, both runnable, the operator picks.
        for rail in [Rail::Metal, Rail::Vulkan] {
            let chosen = resolve_rail(Compiled::Both, true, RailRequest::Named(rail));
            assert_eq!(chosen.rail, rail);
            assert_eq!(chosen.refusal, None);
        }

        // Asking for the only rail there is, is not a refusal.
        let ask_the_only_rail =
            resolve_rail(Compiled::MetalOnly, false, RailRequest::Named(Rail::Metal));
        assert_eq!(ask_the_only_rail.rail, Rail::Metal);
        assert_eq!(ask_the_only_rail.refusal, None);
    }

    /// A misspelling reports itself and runs the default, rather than reading as
    /// the default in silence.
    ///
    /// The raw text rides along because "I set it and nothing changed" is the
    /// symptom of a typo, and a refusal that does not quote what it rejected
    /// cannot end that conversation.
    #[test]
    fn an_unrecognised_rail_name_is_reported_and_the_default_runs() {
        let typo = resolve_rail(
            Compiled::Both,
            true,
            RailRequest::Unrecognized("moltenvk".to_owned()),
        );
        assert_eq!(typo.rail, Rail::Metal);
        assert_eq!(
            typo.refusal,
            Some(RailRefusal::Unrecognized("moltenvk".to_owned()))
        );
        assert_eq!(
            typo.refusal.as_ref().map(RailRefusal::slug),
            Some("rail_unrecognized")
        );
    }

    /// One spelling of a rail's name, reachable from every direction it is
    /// asked for.
    ///
    /// The env value, the boot line, the refusals and `Backend::name` all read
    /// this, so a rail renamed in one place is renamed everywhere or fails here.
    #[test]
    fn a_rail_has_one_name_and_it_round_trips_through_the_env_spelling() {
        for rail in [Rail::Metal, Rail::Vulkan] {
            assert_eq!(Rail::from_name(rail.name()), Some(rail));
            assert!(Rail::NAMES.contains(&rail.name()));
            assert_ne!(rail.other(), rail);
            assert_eq!(rail.other().other(), rail);
        }
        assert_eq!(Rail::NAMES.len(), 2);
        assert_eq!(Rail::from_name("moltenvk"), None);
        // `Rail::NAMES` is every rail this crate can name and not the subset a
        // build carries: an operator naming an uncompiled rail has to reach
        // `NotCompiled`, and a narrowed list would report it as a misspelling.
        assert_eq!(Rail::NAMES, ["metal", "vulkan"]);
    }
}
