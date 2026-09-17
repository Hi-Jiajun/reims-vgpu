//! The canonical-provider compute rail behind `feature = "provider-compute"`.
//!
//! This module is the Gate 2 production call site: a narrow, reviewed compute
//! class leaves the self-contained Vulkan engine and is submitted through the
//! canonical contract ([`metal_api_core`]) and its Vulkan implementation
//! ([`metal_api_vulkan`]). The runtime seam is
//! `runtime/compute_exec/vulkan::execute_dispatch_linux`, which routes here
//! only while the feature is on; a default build never links or calls either
//! provider crate.
//!
//! The admitted class is deliberately the smallest falsifiable shape:
//!
//! - at most one sampled image, and only the shape the canonical compute
//!   texture face executes (`research/docs/26` §21.3, C1/C1b): a D2,
//!   single-sample, single-array-element texture carried as owned bytes, with
//!   one mip level, a mapped `R32Uint`/`R32Float` format and a binding the
//!   compiled module's own contract declares. Storage images stay on the
//!   engine, and so does every shape those bounds do not name (an array slice,
//!   a descriptor array, a mip chain, a resident or multisample source, a
//!   format the contract's closed list does not carry);
//! - the sampler half of that shape is the module's own: a request sampler is
//!   admitted only when the caller's reflection names it as an AIR constexpr
//!   sampler (`StaticSamplerState`, C1b) whose state is the state the request's
//!   descriptor carries, and a module that carries a constexpr sampler must
//!   pair it with exactly one sampled texture — the pairing the canonical
//!   contract reviews. A guest-bound `[[sampler(n)]]`, a runtime sampler slot
//!   the runtime filled with its neutral default, and every state outside the
//!   reviewed `{nearest, linear} x {clamp-to-edge, repeat}` family keep the
//!   engine;
//! - the declaration the seam states is the module's own, and it is checked
//!   twice: the request's sampler state must restate the module's constexpr
//!   state ([`ModuleSampler`]), and the policy the canonical provider compiled
//!   from the same AIR must restate it too. The two readings of one module
//!   disagreeing is [`ProviderComputeDecline::ComputeTextureSamplerMismatch`],
//!   fail-closed, exactly like the payload offset below;
//! - a textured pass's trace crosses the owner→provider wire before anything is
//!   admitted: [`super::provider_wire::submit_frame`] encodes it, the
//!   provider's own decoder reads it back, and it is the *decoded* trace
//!   admission sees. C1c's compute-texture tags carry the declaration
//!   (`PIPELINE_KIND_COMPUTE_TEXTURES`, `SUBMIT_COMPUTE_TEXTURES_REQUEST`), and
//!   the class gate reads the capability answer's own texture section
//!   (`compute_texture_support`) rather than the in-process snapshot — a
//!   device that does not declare the shape executes none of it;
//! - a `ComputeDispatch::Regions` exact-thread launch — one region or the
//!   several a partial threadgroup decomposes into — whose region list is
//!   exactly the decomposition the translator derives for the launch it
//!   carries. The canonical trace spells such a launch as *one* `ThreadsExact`
//!   pass in Metal units (`grid` threads in `threads_per_threadgroup`-sized
//!   groups) and the provider re-derives the tail regions itself
//!   (`research/docs/11` §3.4), so the rail admits a request only when the
//!   translator reproduces its regions — same order, shapes, bases and
//!   per-region payload — from the recovered launch. Whole-workgroup launches
//!   keep the engine (`supports_threadgroups` is false);
//! - the offset the request states for its region payload
//!   (`ComputeDispatch::Regions::push_offset`) is the offset the canonical
//!   provider's own reflection of the same AIR derives. One exact-thread
//!   dispatch has one payload offset — every region's payload is written at
//!   the same place — and the two rails read that number from different
//!   translators, so a request whose offset disagrees is refused by name
//!   instead of being run with the payload where no shader reads it. The
//!   check needs the compiled contract and therefore runs at admission
//!   (before any lease is imported), not in the pure class gate;
//! - every buffer binding the compiled canonical contract names must be
//!   present in the reims-staged request, so `Unused`/`Absent` reflection
//!   cases stay on the reims engine;
//! - the mirror direction too: every binding the request stages must be named
//!   by the canonical contract and agree with it on writability, so a
//!   disagreement between the two pinned translators stays on the reims engine
//!   instead of completing with a silently dropped binding.
//!
//! Anything outside the class returns [`ComputeRailOutcome::NotInNarrowClass`]
//! and the caller runs the self-contained engine unchanged — the feature only
//! narrows which submissions change rail. An in-class submission the provider
//! refuses returns [`ComputeRailOutcome::ProviderDeclined`] and the caller
//! declines the dispatch: fail-closed, never silently re-run on another rail.
//!
//! # Owner leases
//!
//! Every binding this rail submits is imported as an owner-issued lease before
//! the trace is built, through [`super::provider_owner`]: a binding whose bytes
//! came from a registered guest RAM window is imported without copying
//! (`BufferSource::BorrowedNoCopy`, which is what puts the registration into
//! the provider's own `BorrowedLeaseRegistry`), and a binding with no
//! registered window behind it is imported as a staged lease
//! (`BufferSource::StagedLease`). The provider's completion token binds the
//! lease and the completion retires it before the import is released, so the
//! owner-side `LeaseLedger`/`GuestWindows` bookkeeping is the provider's own
//! state rather than a mirror kept beside it.
//!
//! # Writeback and the guest write channel
//!
//! Which guest bytes were written comes from the completion, not from this
//! rail: the provider's own `BufferWriteback` list (allocation coordinates,
//! one complete final writeback per written view) is re-based onto the staged
//! span each binding carried, with the view bounds checked
//! ([`ProviderComputeDecline::WritebackOutsideView`]). The caller writes back
//! exactly those intervals through the existing guest write channel, and only
//! after this function returns — i.e. after [`provider_owner::Plan::settle`]
//! retired the leases and reclaimed the windows (`research/docs/20` §3.4:
//! retirement first, bytes into guest memory second, guest reads last).
//!
//! # Error mapping
//!
//! Every refusal the provider boundary returns is mapped onto this rail's own
//! typed decline: one slug per normalized `ProviderErrorClass`
//! ([`ProviderRefusalClass`]), with the provider's own slug and detail text in
//! the `detail` field so the always-on fail line carries both. `DeviceLost` is
//! the one class whose mapping is not just a name: the contract's teardown
//! guarantee runs over the owner ledger
//! ([`provider_owner::teardown_device_lost`]) and the refusal reports the
//! census of what it released. A rail whose provider is not usable refuses
//! in-class work fail-closed (`provider_unavailable`) rather than re-running it
//! on the self-contained engine, and [`recover_after_device_loss`] is the
//! in-place rebuild entry (`v64`'s `rebuild_after_device_loss`) a device
//! recreate calls.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferSource, BufferView, BufferWriteback,
    CompiledComputePipeline, CompletionDisposition, ComputePass, ComputeProvider, ComputeTrace,
    Dispatch, DispatchKind, DispatchType, NoCopyLeaseImporter, OperationId, ProviderError,
    ProviderErrorClass, ProviderHealth, ResourceTableSnapshot, SamplerAddressMode, SamplerFilter,
    SamplerPolicy, SemanticDigest, TextureAccess, TextureFootprintProof, TextureFormat,
    TextureSource, TextureType, TextureView, TracePass, ViewId, MAX_COMPUTE_TEXTURES,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};

use super::provider_owner::{
    self, DeviceLossTeardown, Request as OwnerRequest, Staged as OwnerStaged, Window,
};
use super::provider_wire;
use super::vulkan::engine::types::ComputeImageResult;
use super::vulkan::engine::{
    ComputeDispatch, ComputeRequest, ComputeSampledImageResource, ComputeSampledSource,
    SamplerResource, StorageImageFormat,
};
use crate::observe::{decline_display, Decline};
use crate::runtime::spirv_bind;

/// Why one reims compute request did not leave this rail for the engine.
#[derive(Debug)]
pub enum ComputeRailOutcome {
    /// The canonical provider executed the dispatch. `writebacks` carries one
    /// provider-reported interval per writable binding, in canonical binding
    /// order; `images` is empty because the narrow class carries no storage
    /// images.
    ProviderCompleted(ProviderComputeOutput),
    /// Outside the narrow admitted class. The caller must run the
    /// self-contained engine, exactly as a build without the feature would.
    NotInNarrowClass(&'static str),
    /// In-class, but the canonical provider refused. The caller must decline
    /// the dispatch rather than fall back to another rail.
    ProviderDeclined(ProviderComputeDecline),
}

/// Reims' own name for one normalized provider refusal class
/// (`metal_api_core::provider::ProviderErrorClass`, `research/docs/13` §3.2).
///
/// One slug per class, so a fail line answers "which kind of provider refusal
/// stopped this dispatch" without parsing provider prose, while the provider's
/// own words stay in the decline's `detail` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderRefusalClass {
    Args,
    Capability,
    Resource,
    Compile,
    Execute,
    DeviceLost,
    Internal,
}

impl ProviderRefusalClass {
    /// Every class, in the order `ProviderErrorClass` declares them. A census
    /// or a test walks this to prove the mapping is total.
    pub const ALL: [Self; 7] = [
        Self::Args,
        Self::Capability,
        Self::Resource,
        Self::Compile,
        Self::Execute,
        Self::DeviceLost,
        Self::Internal,
    ];

    /// The class in the provider's own vocabulary (the `class=` log field).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Args => "args",
            Self::Capability => "capability",
            Self::Resource => "resource",
            Self::Compile => "compile",
            Self::Execute => "execute",
            Self::DeviceLost => "device_lost",
            Self::Internal => "internal",
        }
    }

    /// The reims slug this class is named by.
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Args => "provider_args",
            Self::Capability => "provider_capability",
            Self::Resource => "provider_resource",
            Self::Compile => "provider_compile",
            Self::Execute => "provider_execute",
            Self::DeviceLost => "provider_device_lost",
            Self::Internal => "provider_internal",
        }
    }

    /// The class the provider boundary reported.
    pub const fn from_provider(class: ProviderErrorClass) -> Self {
        match class {
            ProviderErrorClass::Args => Self::Args,
            ProviderErrorClass::Capability => Self::Capability,
            ProviderErrorClass::Resource => Self::Resource,
            ProviderErrorClass::Compile => Self::Compile,
            ProviderErrorClass::Execute => Self::Execute,
            ProviderErrorClass::DeviceLost => Self::DeviceLost,
            ProviderErrorClass::Internal => Self::Internal,
        }
    }
}

/// One sampler the module itself carries: an AIR constexpr sampler the
/// translated module's reflection names, with the descriptor binding it lands
/// on and the decoded state the AIR was lowered against
/// (`research/docs/26` §21.3, C1b's `StaticSampler`).
///
/// The seam needs both halves, and neither can be read off the request alone:
/// whether a sampler descriptor the request carries is the module's own
/// constexpr state rather than a guest-bound (`[[sampler(n)]]`) one, and what
/// the module states for it. The caller reads them from the same
/// `metal2vulkan` reflection it built `req.samplers` from
/// (`runtime::spirv_bind::reflected_sampler_descriptors`); this rail does not
/// run a second translation of its own to rediscover them, because a textured
/// compute dispatch would then pay for two.
#[derive(Clone, Copy, Debug)]
pub struct ModuleSampler {
    /// The descriptor binding the AIR constexpr sampler lands on, in the
    /// numbering the request's sampler descriptors use.
    pub binding: u32,
    /// The state the module's AIR carries for it.
    pub state: metal2vulkan::reflect::StaticSamplerState,
}

/// The reviewed compute sampler family, read off the module's own AIR state.
///
/// The window is exactly the one `metal-api-vulkan` reviews when it compiles a
/// declaration (`static_sampler_policy`): one filter for minification and
/// magnification, one address mode across the three axes, no mip filter,
/// normalized coordinates, no comparison, weighted-average reduction and unit
/// anisotropy. The reviewed textures carry one mip level, so the LOD clamps
/// cannot select another level and are not part of the state the sampled bytes
/// depend on; everything that can change which texels a sample returns is
/// checked, and every other shape stays on the self-contained engine.
fn policy_of_state(
    state: &metal2vulkan::reflect::StaticSamplerState,
) -> Result<SamplerPolicy, &'static str> {
    use metal2vulkan::reflect::{
        SamplerAddressMode as AirAddress, SamplerCompareFunction, SamplerCoordinates,
        SamplerFilter as AirFilter, SamplerMipFilter, SamplerReduction,
    };

    let filter = match (state.min_filter, state.mag_filter) {
        (AirFilter::Nearest, AirFilter::Nearest) => SamplerFilter::Nearest,
        (AirFilter::Linear, AirFilter::Linear) => SamplerFilter::Linear,
        _ => {
            return Err(
                "an AIR sampler whose minification and magnification filters are not the one \
                 reviewed filter stays on the self-contained engine",
            )
        }
    };
    if state.address_mode_s != state.address_mode_t || state.address_mode_s != state.address_mode_r
    {
        return Err(
            "an AIR sampler whose axes address differently stays on the self-contained engine \
             (the reviewed family states one mode for all three)",
        );
    }
    let address =
        match state.address_mode_s {
            AirAddress::ClampToEdge => SamplerAddressMode::ClampToEdge,
            AirAddress::Repeat => SamplerAddressMode::Repeat,
            _ => return Err(
                "an AIR sampler whose address mode is not clamp-to-edge or repeat stays on the \
                 self-contained engine",
            ),
        };
    if state.mip_filter != SamplerMipFilter::None {
        return Err(
            "an AIR sampler with a mip filter stays on the self-contained engine (the reviewed \
             texture carries one level)",
        );
    }
    if state.coordinates != SamplerCoordinates::Normalized {
        return Err(
            "an AIR sampler with pixel coordinates stays on the self-contained engine (the \
             reviewed family samples normalized coordinates)",
        );
    }
    if state.compare_function != SamplerCompareFunction::Never {
        return Err("an AIR sampler with a comparison function stays on the self-contained engine");
    }
    if state.reduction != SamplerReduction::WeightedAverage {
        return Err(
            "an AIR sampler whose reduction is not the weighted average stays on the \
             self-contained engine",
        );
    }
    if state.max_anisotropy != 1 {
        return Err("an AIR sampler with anisotropy stays on the self-contained engine");
    }
    Ok(SamplerPolicy { filter, address })
}

/// The state one request sampler descriptor carries, in the contract's own
/// vocabulary, or why the descriptor is not a state this rail can state.
///
/// The descriptor is the engine's own [`SamplerResource`] — the value the
/// self-contained engine creates its `VkSampler` from — so this walk is what
/// makes "the request states the module's own state" checkable field by field
/// instead of assumed. The fields are the `MTLSampler*` ordinals both routes
/// into `SamplerResource` carry (`runtime::draw::vulkan::reflected_static_sampler_resource`
/// and the guest's own descriptor), which is why the family is written here in
/// those ordinals rather than in the AIR enum's.
fn policy_of_resource(resource: &SamplerResource) -> Result<SamplerPolicy, &'static str> {
    use crate::backend::vulkan::engine::SamplerCompareFunction;
    use reims_vgpu_core::sampler as mtl;

    let filter = match (resource.min_filter, resource.mag_filter) {
        (mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST, mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST) => {
            SamplerFilter::Nearest
        }
        (mtl::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR, mtl::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR) => {
            SamplerFilter::Linear
        }
        _ => {
            return Err(
                "a sampler descriptor whose filters are not one reviewed filter stays on the \
                 self-contained engine",
            )
        }
    };
    if resource.address_mode_u != resource.address_mode_v
        || resource.address_mode_u != resource.address_mode_w
    {
        return Err(
            "a sampler descriptor whose axes address differently stays on the self-contained \
             engine",
        );
    }
    let address =
        match resource.address_mode_u {
            mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE => SamplerAddressMode::ClampToEdge,
            mtl::MTL_SAMPLER_ADDRESS_MODE_REPEAT => SamplerAddressMode::Repeat,
            _ => return Err(
                "a sampler descriptor whose address mode is not clamp-to-edge or repeat stays on \
                 the self-contained engine",
            ),
        };
    if resource.mip_filter != mtl::MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED {
        return Err(
            "a sampler descriptor with a mip filter stays on the self-contained engine (the \
             reviewed texture carries one level)",
        );
    }
    if resource.unnormalized_coordinates {
        return Err(
            "a sampler descriptor with unnormalized coordinates stays on the self-contained \
             engine",
        );
    }
    if resource.compare_function != SamplerCompareFunction::Never {
        return Err(
            "a sampler descriptor with a comparison function stays on the self-contained engine",
        );
    }
    if resource.max_anisotropy != 1 {
        return Err("a sampler descriptor with anisotropy stays on the self-contained engine");
    }
    Ok(SamplerPolicy { filter, address })
}

/// The canonical format one staged sampled image's pixel format maps onto, or
/// `None` for every format the compute texture face does not carry.
///
/// The list is the contract's own closed one (`TextureFormat`) narrowed to the
/// two entries the canonical Vulkan rail executes compute-side: a `uint` texel
/// read's `R32Uint` and, from C1b on, the `R32Float` a module's own constexpr
/// sampler samples.
fn mapped_texture_format(format: StorageImageFormat) -> Option<TextureFormat> {
    match format {
        StorageImageFormat::R32Uint => Some(TextureFormat::R32Uint),
        StorageImageFormat::R32Float => Some(TextureFormat::R32Float),
        _ => None,
    }
}

/// The filter name one [`SamplerFilter`] travels under in a refusal.
fn filter_name(filter: SamplerFilter) -> &'static str {
    match filter {
        SamplerFilter::Nearest => "Nearest",
        SamplerFilter::Linear => "Linear",
    }
}

/// The address name one [`SamplerAddressMode`] travels under in a refusal.
fn address_name(address: SamplerAddressMode) -> &'static str {
    match address {
        SamplerAddressMode::ClampToEdge => "ClampToEdge",
        SamplerAddressMode::Repeat => "Repeat",
    }
}

/// One provider-reported writeback, mapped onto the staged binding it belongs
/// to.
///
/// The completion is the only source of "which bytes were written": `offset`
/// is the provider's own `BufferWriteback.offset` (allocation coordinates)
/// re-based onto the staged span, so a provider that reports a sub-range
/// produces an interval at *that* offset instead of a write at the start of
/// the buffer. `allocation`/`allocation_offset` keep the provider's own
/// coordinates as evidence for the log and for tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderWriteback {
    pub binding: u32,
    /// Byte offset of `bytes` inside the staged span the binding carried.
    pub offset: u64,
    pub bytes: Vec<u8>,
    /// The allocation the provider's writeback named.
    pub allocation: AllocationId,
    /// The writeback's offset in the provider's own allocation coordinates.
    pub allocation_offset: u64,
}

/// What one completed narrow-class submission returns.
#[derive(Debug)]
pub struct ProviderComputeOutput {
    /// One writeback per writable binding, in canonical binding order.
    pub writebacks: Vec<ProviderWriteback>,
    /// Storage-image results. The admitted class carries no images, so this is
    /// always empty; it exists so the runtime seam's count check keeps the same
    /// shape on both rails.
    pub images: Vec<ComputeImageResult>,
}

/// A typed refusal of the canonical-provider rail, nameable in the fail log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderComputeDecline {
    /// The process-global provider could not be created, or its lifecycle
    /// reports a terminal health (`device_lost`/`exhausted`). In-class work is
    /// refused fail-closed — the caller declines the dispatch and nothing
    /// falls back to the self-contained engine (`research/docs/13` §3.2).
    /// `health` names the state the refusal was raised on (`"create"` when
    /// there is no provider to ask), and `teardown` reports the leases and
    /// windows a device-loss teardown released, when this refusal ran it.
    ProviderUnavailable {
        detail: String,
        health: &'static str,
        teardown: Option<DeviceLossTeardown>,
    },
    /// Compiling the reviewed AIR in the canonical translator failed.
    PipelineCompile { detail: String },
    /// The request's stated payload offset is not the offset the canonical
    /// provider derives from the same AIR, so the region payload would land
    /// where the kernel does not read it. The two rails read the number from
    /// two pinned translators, so a disagreement is a refusal rather than a
    /// dispatch one of them would execute as a different launch
    /// (`research/docs/26` §7, 2026-09-17).
    PushOffsetMismatch { request: u32, contract: u32 },
    /// The sampler state this seam states for a compute texture binding is not
    /// the state the canonical provider compiled from the same AIR
    /// (`research/docs/26` §21.3, C1b). Both readings claim to be the module's
    /// own `@__air_sampler_state`, and the texels a sample returns depend on
    /// which one executes, so the disagreement is fail-closed — never a
    /// dispatch executed with the other state, exactly as the payload offset
    /// above is.
    ComputeTextureSamplerMismatch {
        binding: u32,
        /// The state the module's own AIR constexpr sampler states, as this
        /// seam read it.
        request_filter: &'static str,
        request_address: &'static str,
        /// The state the compiled contract declares for the same binding.
        contract_filter: &'static str,
        contract_address: &'static str,
    },
    /// The compute texture declaration could not cross the owner→provider wire.
    /// `step` names the codec call that answered and `detail` carries the
    /// codec's own text (a shape the frame cannot carry is a fact about the
    /// wire, and the provider's own decoder is the only thing that gets to say
    /// so — the same rule R9j's stage-buffer frames keep).
    ComputeTextureWire { step: &'static str, detail: String },
    /// The canonical provider refused, named by its normalized class: `step`
    /// names the provider call that answered, and `detail` carries the
    /// provider's own slug and detail text.
    ProviderRefused {
        class: ProviderRefusalClass,
        step: &'static str,
        detail: String,
    },
    /// The canonical provider answered `device_lost`. The contract's teardown
    /// guarantee has already run over the owner ledger: every lease was
    /// released regardless of outstanding tokens and every window was retired
    /// and reclaimed (`teardown` is the census of that pass).
    ProviderDeviceLost {
        step: &'static str,
        detail: String,
        teardown: DeviceLossTeardown,
    },
    /// The canonical admission refused the translated trace.
    TraceAdmission { detail: String },
    /// The completion was not host-visible, so there are no bytes to write
    /// back.
    CompletionNotVisible,
    /// A writable binding's readback did not come back through the provider
    /// completion channel.
    WritebackMissing { binding: u32 },
    /// The provider's writeback for one binding lies outside the view the trace
    /// carried, so its bytes cannot be placed in the staged span.
    WritebackOutsideView {
        binding: u32,
        offset: u64,
        length: u64,
        view_offset: u64,
        view_length: u64,
    },
    /// The owner rail refused a registration, an import, a retirement or a
    /// release. Delegated, so the slug names the owner check that refused.
    Owner(provider_owner::Decline),
}

impl Decline for ProviderComputeDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ProviderUnavailable { .. } => "provider_unavailable",
            Self::PipelineCompile { .. } => "pipeline_compile",
            Self::PushOffsetMismatch { .. } => "push_offset_mismatch",
            Self::ComputeTextureSamplerMismatch { .. } => "compute_texture_sampler_mismatch",
            Self::ComputeTextureWire { .. } => "compute_texture_wire",
            Self::ProviderRefused { class, .. } => class.slug(),
            Self::ProviderDeviceLost { .. } => ProviderRefusalClass::DeviceLost.slug(),
            Self::TraceAdmission { .. } => "trace_admission",
            Self::CompletionNotVisible => "completion_not_visible",
            Self::WritebackMissing { .. } => "writeback_missing",
            Self::WritebackOutsideView { .. } => "writeback_outside_view",
            Self::Owner(inner) => inner.slug(),
        }
    }

    /// Delegated with `slug`; the owner decline names its own checks, so
    /// claiming a slug for this wrapper would report a collision that is not
    /// one.
    fn owner(&self) -> &'static str {
        match self {
            Self::Owner(inner) => inner.owner(),
            _ => core::any::type_name::<Self>(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ProviderUnavailable {
                detail,
                health,
                teardown,
            } => {
                let mut fields = vec![("health", health.to_string()), ("detail", detail.clone())];
                if let Some(teardown) = teardown {
                    fields.push(("teardown_leases", teardown.leases.to_string()));
                    fields.push(("teardown_windows", teardown.windows.to_string()));
                }
                fields
            }
            Self::PipelineCompile { detail } | Self::TraceAdmission { detail } => {
                vec![("detail", detail.clone())]
            }
            Self::PushOffsetMismatch { request, contract } => vec![
                ("request", request.to_string()),
                ("contract", contract.to_string()),
            ],
            Self::ComputeTextureSamplerMismatch {
                binding,
                request_filter,
                request_address,
                contract_filter,
                contract_address,
            } => vec![
                ("binding", binding.to_string()),
                ("request_filter", (*request_filter).to_string()),
                ("request_address", (*request_address).to_string()),
                ("contract_filter", (*contract_filter).to_string()),
                ("contract_address", (*contract_address).to_string()),
            ],
            Self::ComputeTextureWire { step, detail } => {
                vec![("step", (*step).to_string()), ("detail", detail.clone())]
            }
            Self::ProviderRefused {
                class,
                step,
                detail,
            } => vec![
                ("class", class.name().to_string()),
                ("step", step.to_string()),
                ("detail", detail.clone()),
            ],
            Self::ProviderDeviceLost {
                step,
                detail,
                teardown,
            } => vec![
                ("class", ProviderRefusalClass::DeviceLost.name().to_string()),
                ("step", step.to_string()),
                ("detail", detail.clone()),
                ("teardown_leases", teardown.leases.to_string()),
                ("teardown_windows", teardown.windows.to_string()),
            ],
            Self::CompletionNotVisible => Vec::new(),
            Self::WritebackMissing { binding } => vec![("binding", binding.to_string())],
            Self::WritebackOutsideView {
                binding,
                offset,
                length,
                view_offset,
                view_length,
            } => vec![
                ("binding", binding.to_string()),
                ("offset", offset.to_string()),
                ("length", length.to_string()),
                ("view_offset", view_offset.to_string()),
                ("view_length", view_length.to_string()),
            ],
            Self::Owner(inner) => inner.fields(),
        }
    }
}

decline_display!(ProviderComputeDecline);

/// Render a [`ProviderError`] without a `Display` impl: its slug plus the
/// optional detail, which is where the canonical provider names the refusing
/// check. Class and phase are recoverable from the slug family in the log.
pub(crate) fn provider_error_detail(error: &ProviderError) -> String {
    match &error.detail {
        Some(detail) => format!("{}: {detail}", error.slug),
        None => error.slug.clone(),
    }
}

/// One process-global canonical rail: the provider, the neutral `Device` used
/// to compile AIR, and a per-AIR pipeline cache. The provider keeps its own
/// Vulkan instance/device, so it never shares the self-contained engine's.
pub(crate) struct ProviderRail {
    pub(crate) provider: VulkanComputeProvider,
    pub(crate) device: Device,
    /// The executor the provider was built from. Kept so the test-only driver
    /// loss injection can arm the very provider this rail submits through.
    executor: Arc<VulkanExecutor>,
    /// `(AIR, entry)` → compiled pipeline. Compiled pipelines are metadata
    /// handles into the provider's own registry, so the cache bounds pipeline
    /// registrations by distinct kernel AIR rather than by dispatch count. The
    /// entry point is part of the key: one AIR can carry more than one entry,
    /// and a compiled pipeline belongs to exactly one of them, so an AIR-only
    /// key would hand a caller the other entry's pipeline on a hit.
    pipelines: Mutex<HashMap<PipelineKey, CompiledComputePipeline>>,
}

/// Cache key of one compiled canonical pipeline: the kernel AIR bytes and the
/// entry point compiled in them. `apv_cs` and any second entry of the same AIR
/// are two keys, so a hit can never answer with another entry's pipeline
/// (`S3`, `evidence/reviews/reims-gate2-increment23-review-2026-09-17.md` §3).
type PipelineKey = (Vec<u8>, String);

/// The cache key for one `(air, entry)` pair.
fn pipeline_cache_key(air: &[u8], entry: &str) -> PipelineKey {
    (air.to_vec(), entry.to_owned())
}

/// The canonical view identity of the one texture this rail stages.
///
/// The buffer views beside it are numbered `binding + 1`, so a fixed base above
/// that namespace keeps the two kinds apart in a trace whose pass carries both
/// — and the identity is a function of the binding, so the same staged texture
/// keeps the same view across submissions exactly as the buffer views do.
fn texture_view_id(binding: u32) -> u64 {
    const TEXTURE_VIEW_BASE: u64 = 0x1000;
    TEXTURE_VIEW_BASE + u64::from(binding)
}

/// The allocation identity of that view.
///
/// The source is [`TextureSource::OwnedBytes`], so no allocation record is
/// needed for it (the canonical rail refuses a lease source compute-side); the
/// identity only has to be nonzero and distinct from the buffers' lease
/// allocations, which come from the provider's own namespace.
fn texture_allocation_id(binding: u32) -> u64 {
    const TEXTURE_ALLOCATION_BASE: u64 = 0x636f_6d70_7465_7874; // "comptext"
    TEXTURE_ALLOCATION_BASE + u64::from(binding)
}

static NEXT_OPERATION_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn rail() -> Result<&'static ProviderRail, ProviderComputeDecline> {
    static RAIL: OnceLock<Result<ProviderRail, String>> = OnceLock::new();
    RAIL.get_or_init(|| {
        let executor = VulkanExecutor::new().map_err(|error| error.to_string())?;
        let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor))
            .map_err(|error| provider_error_detail(&error))?;
        let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
        Ok(ProviderRail {
            provider,
            device,
            executor,
            pipelines: Mutex::new(HashMap::new()),
        })
    })
    .as_ref()
    .map_err(|detail| ProviderComputeDecline::ProviderUnavailable {
        detail: detail.clone(),
        health: "create",
        teardown: None,
    })
}

/// Route one reims compute request. `entry` is the AIR entry-point name the
/// canonical translator reports for this kernel (`apv_cs` for the reviewed
/// fixture), not the reims SPIR-V entry (`main`) — the canonical provider
/// requires the two to agree. `module_samplers` is the module's own sampler
/// half, as the caller's reflection read it ([`ModuleSampler`]): the AIR
/// constexpr samplers a request's own sampler descriptors must restate.
/// `windows` names, per canonical binding, the registered guest RAM window its
/// staged bytes came from; a binding with no entry is imported as an
/// owner-issued staged lease instead. See the module docs for the class rules
/// and the owner rail's own docs for the lease lifecycle.
pub fn submit_compute(
    air: &[u8],
    entry: &str,
    req: &ComputeRequest,
    module_samplers: &[ModuleSampler],
    windows: &[Window],
) -> ComputeRailOutcome {
    // The class gate is pure and runs first: an out-of-class shape never
    // touches the rail (no device, no compile, no lease).
    match narrow_class(req, module_samplers) {
        Err(reason) => ComputeRailOutcome::NotInNarrowClass(reason),
        Ok(class) => match submit_narrow(air, entry, req, class, windows) {
            Ok(NarrowOutcome::Completed(output)) => ComputeRailOutcome::ProviderCompleted(output),
            Ok(NarrowOutcome::Outside(reason)) => ComputeRailOutcome::NotInNarrowClass(reason),
            Err(decline) => ComputeRailOutcome::ProviderDeclined(decline),
        },
    }
}

/// The device's host-pointer import alignment, or `0` when this device has no
/// `VK_EXT_external_memory_host`.
///
/// Exposed because the alignment is the one number a registration must be
/// built against, and a caller that wants to prove the owner rail (a test, or
/// an owner that registers before its first submission) needs it without
/// reaching into the rail's own provider.
pub fn host_import_alignment() -> Result<u64, ProviderComputeDecline> {
    Ok(rail()?.provider.no_copy_alignment())
}

/// The epoch of the provider this rail holds — the epoch every lease this
/// process imports must carry.
pub fn device_epoch() -> Result<u64, ProviderComputeDecline> {
    Ok(rail()?.provider.device_epoch().get())
}

/// Rebuild the rail's provider after a confirmed device loss and drop the
/// pipeline handles that belonged to the dead device.
///
/// This is the in-process recovery entry for the terminal `DeviceLost` health
/// the refusal mapping already reports: `v64`'s `rebuild_after_device_loss`
/// refuses every other state fail-closed, installs a fresh Vulkan device and
/// advances the `DeviceEpoch`, and the compiled pipelines the rail cached are
/// device children of the dead incarnation. The owner rail was already torn
/// down when the loss was observed ([`provider_owner::teardown_device_lost`]),
/// so callers re-register guest RAM and resubmit afterwards: old-epoch leases
/// and tokens keep being refused by the provider, and nothing is silently
/// re-admitted. No production caller exists in this increment — the
/// device-recreate path is where the call belongs.
pub fn recover_after_device_loss() -> Result<(), ProviderComputeDecline> {
    let rail = rail()?;
    rail.provider
        .rebuild_after_device_loss()
        .map_err(|error| refusal_decline(&error, "recovery"))?;
    rail.pipelines
        .lock()
        .map_err(|_| ProviderComputeDecline::PipelineCompile {
            detail: "pipeline cache poisoned".into(),
        })?
        .clear();
    // The render rail's registrations are children of the same dead device
    // (`provider_render`'s module docs): its declaring kernel and every
    // registered pipeline pair are dropped here, under the same rebuild, so
    // no handle survives the incarnation it was minted under.
    #[cfg(feature = "provider-render")]
    super::provider_render::on_device_rebuilt();
    Ok(())
}

/// Arm the rail's own provider so its next queue-boundary answer is
/// `VK_ERROR_DEVICE_LOST`.
///
/// This is the emulator's own test injection (`v64`), pointed at the very
/// provider this rail submits through, so a test observes the rail's reaction —
/// typed refusal, owner teardown, `provider_unavailable` on later work — rather
/// than its own setup. The lifecycle is deliberately untouched: the provider
/// reaches `DeviceLost` only through the path a real driver answer takes.
/// Tests only; the runtime never calls it.
#[doc(hidden)]
pub fn inject_driver_device_loss_for_test(
    point: metal_api_vulkan::DeviceLossPoint,
) -> Result<(), ProviderComputeDecline> {
    rail()?.executor.inject_driver_device_loss_for_test(point);
    Ok(())
}

/// The lifecycle states in the provider's own vocabulary.
fn health_name(health: ProviderHealth) -> &'static str {
    match health {
        ProviderHealth::Usable => "usable",
        ProviderHealth::DeviceLost => "device_lost",
        ProviderHealth::Exhausted => "exhausted",
    }
}

/// The explicit policy for a rail whose provider is not usable
/// (`research/docs/13` §3.2): refuse in-class work fail-closed with
/// `provider_unavailable` — never re-run it on the self-contained engine — and
/// run the contract's device-loss teardown over the owner ledger first when the
/// terminal state is `DeviceLost`, so no lease stays imported against the dead
/// incarnation. Returns `None` while the provider is `Usable`.
pub(crate) fn refuse_unhealthy(
    provider: &VulkanComputeProvider,
    step: &'static str,
) -> Option<ProviderComputeDecline> {
    let health = provider.health();
    if health.is_usable() {
        return None;
    }
    let teardown = match health {
        ProviderHealth::DeviceLost => Some(provider_owner::teardown_device_lost()),
        ProviderHealth::Usable | ProviderHealth::Exhausted => None,
    };
    Some(ProviderComputeDecline::ProviderUnavailable {
        detail: format!("health={} step={step}", health_name(health)),
        health: health_name(health),
        teardown,
    })
}

/// Map one canonical provider refusal onto this rail's typed decline.
///
/// Every [`ProviderErrorClass`] is named by its own slug
/// ([`ProviderRefusalClass`]), and `detail` keeps the provider's own slug and
/// detail text. `DeviceLost` is the one class whose mapping is more than a
/// name: the contract treats it as a teardown guarantee, so the owner ledger is
/// released here and the refusal carries the census of what that released.
pub(crate) fn refusal_decline(error: &ProviderError, step: &'static str) -> ProviderComputeDecline {
    let class = ProviderRefusalClass::from_provider(error.class);
    let detail = provider_error_detail(error);
    match class {
        ProviderRefusalClass::DeviceLost => {
            let teardown = provider_owner::teardown_device_lost();
            ProviderComputeDecline::ProviderDeviceLost {
                step,
                detail,
                teardown,
            }
        }
        class => ProviderComputeDecline::ProviderRefused {
            class,
            step,
            detail,
        },
    }
}

/// A narrow submission's success answers: completed by the provider, or
/// discovered to be outside the class once the canonical contract is known.
/// Provider refusals travel through the `Err` arm instead.
enum NarrowOutcome {
    Completed(ProviderComputeOutput),
    Outside(&'static str),
}

/// The Metal-level launch one admitted `Regions` request is the device work of.
///
/// The canonical trace carries a compute pass in *Metal* units — a thread grid
/// and a threadgroup size — and the provider derives the device-side tail
/// regions itself (`research/docs/11` §3.4: the canonical trace 只携带 Metal
/// grid/local). A reims request is the other half of the same fact: it carries
/// the tiled decomposition. This is the launch that decomposition belongs to,
/// recovered from the tiles themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NarrowLaunch {
    /// Exact threads of the whole launch, the canonical `ThreadsExact` grid.
    grid: [u64; 3],
    /// The launch's threadgroup shape, the nominal local size the regions were
    /// tiled at.
    threads_per_threadgroup: [u64; 3],
    /// The offset the request states for every region's payload, read out of
    /// the dispatch (`ComputeDispatch::Regions::push_offset`). The class gate
    /// is pure and has no contract to compare this against, so the agreement
    /// with the canonical provider's own reflected offset is checked at
    /// admission, before the first lease
    /// ([`ProviderComputeDecline::PushOffsetMismatch`]).
    push_offset: u32,
}

/// The one sampled texture binding a request stages, in the canonical
/// vocabulary the pass view is built in.
///
/// Every field is a fact of the request's own [`ComputeSampledImageResource`]:
/// the binding the view names, the canonical format its pixel format maps
/// onto, the extent, and the tightly packed texels one mip level carries.
#[derive(Clone, Copy, Debug)]
struct StagedTexture<'a> {
    binding: u32,
    format: TextureFormat,
    width: u64,
    height: u64,
    bytes: &'a [u8],
}

/// One request's texture half: the launch, the sampled image it stages (when it
/// stages one) and the sampler descriptor that image is sampled through.
///
/// The sampler is carried as two readings of one fact — the request's own
/// descriptor and the module's own AIR state the caller's reflection named —
/// because the seam's rule is that they agree, and admission then requires the
/// compiled contract to agree with both.
#[derive(Debug)]
struct NarrowClass<'a> {
    launch: NarrowLaunch,
    texture: Option<StagedTexture<'a>>,
    /// The sampler descriptor the request carries, when it carries one.
    sampler: Option<&'a SamplerResource>,
    /// The module's own stated state for that descriptor.
    module_sampler: Option<ModuleSampler>,
}

/// One staged sampled image as the canonical view this rail would state, or why
/// the shape it carries keeps the engine.
///
/// The closed list is C1/C1b's reviewed window, and every entry is the *view*
/// side of the canonical `TextureView`: single-layer, single-array-element,
/// single-descriptor, one mip level, owned bytes, a format the compute texture
/// face carries. The shapes that fall out — an array slice, a descriptor array,
/// a mip chain, a resident storage image, a retained multisample target — have
/// no canonical spelling, so they are named here rather than discovered by a
/// provider that would refuse them later.
fn staged_texture(image: &ComputeSampledImageResource) -> Result<StagedTexture<'_>, &'static str> {
    if image.array_element != 0 {
        return Err(
            "a sampled image bound as one element of an array stays on the self-contained engine",
        );
    }
    if image.descriptor_count != 1 {
        return Err(
            "a sampled image bound as an array of descriptors stays on the self-contained engine",
        );
    }
    if image.mip_levels != 1 {
        return Err(
            "a sampled image carrying a mip chain stays on the self-contained engine (the \
             canonical view is one level)",
        );
    }
    let bytes =
        match &image.source {
            ComputeSampledSource::Bytes(bytes) => bytes.as_slice(),
            ComputeSampledSource::ResidentCopy(_) => return Err(
                "a sampled image served from a resident storage image stays on the self-contained \
                 engine (the canonical texture source is the caller's own bytes)",
            ),
            ComputeSampledSource::MultisampleTarget(_) => {
                return Err(
                    "a sampled image served from a retained multisample target stays on the \
                 self-contained engine (the canonical compute texture face samples single-sample \
                 D2 images)",
                )
            }
        };
    let Some(format) = mapped_texture_format(image.format) else {
        return Err(
            "a sampled image whose format the canonical compute texture face does not carry stays \
             on the self-contained engine",
        );
    };
    // Two numbering systems meet here, and this is the one place both are
    // known: the engine's request carries the translator's texture band
    // (`TEXTURE_BINDING_BASE + Metal index`, `runtime::spirv_bind`), while the
    // canonical contract names the Metal `[[texture(n)]]` index itself. The
    // scalar shape above (`array_element == 0`, `descriptor_count == 1`) is
    // exactly the case where the difference is the band base.
    let Some(metal_binding) = image.binding.checked_sub(spirv_bind::TEXTURE_BINDING_BASE) else {
        return Err(
            "a sampled image whose descriptor binding is not in the translator's texture band \
             stays on the self-contained engine",
        );
    };
    if image.width == 0 || image.height == 0 {
        return Err("a sampled image with no texels stays on the self-contained engine");
    }
    let expected = u64::from(image.width)
        .saturating_mul(u64::from(image.height))
        .saturating_mul(format.bytes_per_texel());
    if expected != bytes.len() as u64 {
        return Err(
            "a sampled image whose bytes are not the tightly packed extent of its view stays on \
             the self-contained engine",
        );
    }
    Ok(StagedTexture {
        binding: metal_binding,
        format,
        width: u64::from(image.width),
        height: u64::from(image.height),
        bytes,
    })
}

/// The launch `req`'s dispatch decomposes, or why the request stays on the
/// self-contained engine.
///
/// The recovered launch is *verified*, not assumed: the translator must
/// reproduce exactly this region list — same order, shapes, bases, and
/// per-region payload — for the recovered `(grid, local)`. That verification is
/// what makes one canonical pass a faithful spelling of the whole launch: the
/// provider re-derives the device work from `(grid, local)` with the same
/// planner, so a region list it would not derive (a hand-built list, a list
/// whose payloads disagree with the plan, a shape whose tiles cannot be
/// re-derived) keeps the engine rather than silently dispatching other threads.
/// The whole check is pure — it runs before the rail is even created, so a
/// refused shape costs the provider nothing.
fn narrow_class<'a>(
    req: &'a ComputeRequest,
    module_samplers: &[ModuleSampler],
) -> Result<NarrowClass<'a>, &'static str> {
    if !req.storage_images.is_empty() {
        return Err("compute storage images stay on the self-contained engine");
    }

    // The texture half (`research/docs/26` §21.3). Every condition is a fact of
    // the request or of the module's own reflection, so they all run before the
    // rail exists: a shape the canonical view cannot carry keeps the engine
    // without costing the provider a compile. A pass with no texture at all is
    // the population this rail always had and skips the whole walk.
    let texture = match req.sampled_images.as_slice() {
        [] => None,
        [image] => Some(staged_texture(image)?),
        _ => {
            return Err(
                "a pass that stages more than one sampled image stays on the self-contained \
                 engine (the canonical compute texture face reviews one binding, and the wire \
                 bounds the declaration list at that same one)",
            )
        }
    };

    // The sampler half: the request's own descriptors have to be the module's
    // own AIR constexpr samplers, state and all. A guest `[[sampler(n)]]`
    // binding, a runtime sampler slot the runtime filled with its neutral
    // default, and a state the reviewed family does not name all keep the
    // engine, because the canonical contract has no separate sampler object to
    // state them with (`research/docs/26` §21.3, C1b).
    if module_samplers.len() > 1 {
        return Err(
            "a module that carries more than one AIR constexpr sampler stays on the \
             self-contained engine (the canonical compute sampler face reviews one)",
        );
    }
    let mut sampler = None;
    let mut module_sampler = None;
    match req.samplers.as_slice() {
        [] => {}
        [resource] => {
            let Some(declared) = module_samplers
                .iter()
                .find(|declared| declared.binding == resource.binding)
            else {
                return Err(
                    "a sampler the module does not carry as its own AIR constexpr state stays on \
                     the self-contained engine",
                );
            };
            let carried = policy_of_resource(resource)?;
            let stated = policy_of_state(&declared.state)?;
            if carried != stated {
                return Err(
                    "a sampler descriptor whose state is not the state the module's own AIR \
                     constexpr sampler carries stays on the self-contained engine",
                );
            }
            sampler = Some(resource);
            module_sampler = Some(*declared);
        }
        _ => {
            return Err(
                "a pass that carries more than one sampler stays on the self-contained engine \
                 (the canonical compute sampler face reviews one)",
            )
        }
    }
    match (module_sampler, texture) {
        (Some(_), Some(_)) => {}
        (Some(_), None) => {
            return Err(
                "a module whose AIR constexpr sampler the request does not restate stays on the \
                 self-contained engine (the reviewed shape pairs one constexpr sampler with one \
                 staged texture)",
            )
        }
        // The mirror direction: a module that carries a constexpr sampler the
        // request does not name at all has the same gap as the pair above — the
        // declaration would travel without the descriptor that restates it.
        (None, _) if module_samplers.is_empty() => {}
        (None, _) => {
            return Err(
                "a module whose AIR constexpr sampler the request does not restate stays on the \
                 self-contained engine (the request carries no sampler descriptor for it)",
            )
        }
    }
    let ComputeDispatch::Regions {
        push_offset,
        threadgroups_per_grid,
        regions,
    } = &req.dispatch
    else {
        return Err(
            "whole-workgroup launches stay on the self-contained engine \
             (the canonical Vulkan provider admits exact threads only)",
        );
    };
    if regions.is_empty() {
        return Err("a launch with no regions stays on the self-contained engine");
    }

    // The launch's extent is the tiling's own cover: a region's threads start
    // at the origin its payload pushes (words 3..6 — the base the kernel adds
    // to its invocation ids) and reach `base + local_size * group_count`, so
    // the widest region of each axis reaches the whole launch's thread count.
    // Its threadgroup shape is the widest local size the plan specialized a
    // pipeline to — the nominal size for axes the plan tiles, and the effective
    // size for an axis whose whole grid is one partial threadgroup. The base is
    // read out of the payload rather than a struct field because the payload is
    // what the device pushes; a base the planner would not derive fails below,
    // where the plan is compared word for word against it.
    let mut grid = [0u64; 3];
    let mut local = [0u64; 3];
    for region in regions {
        let base = [
            region.push_constants[3],
            region.push_constants[4],
            region.push_constants[5],
        ];
        for axis in 0..3 {
            let local_size = region.local_size[axis];
            let group_count = region.group_count[axis];
            if local_size == 0 || group_count == 0 {
                return Err("a region that launches no threads stays on the self-contained engine");
            }
            let span = u64::from(base[axis])
                .saturating_add(u64::from(local_size) * u64::from(group_count));
            grid[axis] = grid[axis].max(span);
            local[axis] = local[axis].max(u64::from(local_size));
        }
    }

    // The planner speaks the translator's `u32` ABI; a launch that does not fit
    // it cannot be re-derived here, and the engine keeps that shape.
    let mut planned_grid = [0u32; 3];
    let mut planned_local = [0u32; 3];
    for axis in 0..3 {
        planned_grid[axis] = u32::try_from(grid[axis])
            .map_err(|_| "a thread grid wider than the translator's u32 stays on the engine")?;
        planned_local[axis] = u32::try_from(local[axis])
            .map_err(|_| "a local size wider than the translator's u32 stays on the engine")?;
    }
    // The dynamic variant is the one whose plan depends only on `(grid, local)`
    // — which is what the canonical pass carries. A kernel whose AIR bakes a
    // fixed grid is checked by the provider it is compiled against: it either
    // agrees (same grid, same plan) or the provider refuses the pass, which is
    // a typed decline rather than a fallback to the engine.
    let plan = metal2vulkan::reflect::KernelDispatch::ThreadsDynamic { offset: 0 }
        .plan(planned_local, Some(planned_grid))
        .map_err(|_| {
            "a launch the translator refuses to plan stays on the self-contained engine"
        })?;
    let reproduced = plan.threadgroups_per_grid == *threadgroups_per_grid
        && plan.regions.len() == regions.len()
        && plan.regions.iter().zip(regions).all(|(planned, carried)| {
            planned.local_size == carried.local_size
                && planned.group_count == carried.group_count
                && plan.push_constants(*planned) == carried.push_constants
        });
    if !reproduced {
        return Err(
            "a region list the translator would not derive from one exact-thread launch \
             stays on the self-contained engine",
        );
    }
    Ok(NarrowClass {
        launch: NarrowLaunch {
            grid,
            threads_per_threadgroup: local,
            push_offset: *push_offset,
        },
        texture,
        sampler,
        module_sampler,
    })
}

fn submit_narrow(
    air: &[u8],
    entry: &str,
    req: &ComputeRequest,
    class: NarrowClass<'_>,
    windows: &[Window],
) -> Result<NarrowOutcome, ProviderComputeDecline> {
    let rail = rail()?;
    let provider = &rail.provider;
    // The health gate runs before anything is compiled or imported: a provider
    // the lifecycle already reports terminal cannot take this dispatch, and the
    // answer is a refusal, never a silent switch to the other rail.
    if let Some(decline) = refuse_unhealthy(provider, "admission") {
        return Err(decline);
    }

    // The one device answer the texture half needs (C1c's capability section):
    // a provider that does not declare compute texture sampling executes none
    // of these declarations, so the pass keeps the engine under the capability
    // frame's own reading rather than being met later by the provider's
    // `compute_texture_input_unsupported` refusal. The frame is read exactly as
    // the stage-buffer half reads its own (`provider_wire`), and it is read
    // before the compile because the answer is a property of the device and not
    // of the module.
    let NarrowClass {
        launch,
        texture,
        sampler,
        module_sampler,
    } = class;
    if let Some(staged) = &texture {
        let support = provider_wire::compute_texture_support(
            provider.device_epoch(),
            &provider.capabilities(),
        )
        .map_err(|decline| ProviderComputeDecline::ComputeTextureWire {
            step: decline.step,
            detail: decline.detail,
        })?;
        if !support.supported
            || support.maximum < MAX_COMPUTE_TEXTURES as u32
            || !support.formats.contains(&staged.format)
        {
            return Ok(NarrowOutcome::Outside(
                "a pass that stages a sampled image stays on the self-contained engine when the \
                 provider's own capability answer does not carry the compute texture shape (no \
                 compute texture sampling, a binding cap below one, or a format outside the \
                 admitted list — each of which admission refuses by name)",
            ));
        }
    }

    // One canonical pass carries the whole launch in Metal units, and the
    // provider re-derives the device-side regions from exactly these two
    // numbers — the tiling `narrow_class` verified the request's own.
    let NarrowLaunch {
        grid,
        threads_per_threadgroup,
        push_offset: request_push_offset,
    } = launch;

    let pipeline = {
        let mut cache =
            rail.pipelines
                .lock()
                .map_err(|_| ProviderComputeDecline::TraceAdmission {
                    detail: "pipeline cache poisoned".into(),
                })?;
        let key = pipeline_cache_key(air, entry);
        if let Some(pipeline) = cache.get(&key) {
            pipeline.clone()
        } else {
            let digest = SemanticDigest::new("reims-provider-compute-v1", air.to_vec()).map_err(
                |error| ProviderComputeDecline::PipelineCompile {
                    detail: error.to_string(),
                },
            )?;
            let function = rail
                .device
                .new_library_with_binary_air(air.to_vec())
                .map_err(|error| ProviderComputeDecline::PipelineCompile {
                    detail: error.to_string(),
                })?
                .function(entry)
                .map_err(|error| ProviderComputeDecline::PipelineCompile {
                    detail: error.to_string(),
                })?;
            let compiled = provider
                .compile_pipeline(&function, digest)
                .map_err(|error| refusal_decline(&error, "pipeline_compile"))?;
            cache.insert(key, compiled.clone());
            compiled
        }
    };

    // The payload offset is the one number the class gate cannot verify: one
    // exact-thread dispatch has one payload offset — every region's payload is
    // written at it — and the request states it from the reims-pinned
    // translator while the provider re-derives its own from the same AIR. A
    // disagreement means the payload would land where the kernel does not read
    // it on whichever rail acted on the other number, so the dispatch is
    // refused by name instead (`research/docs/26` §7). This runs before the
    // first lease is imported, so a mismatched request costs the device
    // nothing.
    if pipeline.contract.push_constant_offset != request_push_offset {
        return Err(ProviderComputeDecline::PushOffsetMismatch {
            request: request_push_offset,
            contract: pipeline.contract.push_constant_offset,
        });
    }

    // Reims stages only used bindings; a canonical binding it did not stage (an
    // `Unused` / `Absent` reflection case) keeps the shape on the reims engine
    // rather than inventing bytes the kernel does not touch. This is checked
    // before any lease is imported, so a shape that turns out to be out of
    // class costs the provider nothing.
    let mut bindings = Vec::with_capacity(pipeline.contract.buffer_bindings.len());
    for binding in &pipeline.contract.buffer_bindings {
        let Some(staged) = req
            .storage_buffers
            .iter()
            .find(|staged| staged.binding == binding.metal_binding)
        else {
            return Ok(NarrowOutcome::Outside(
                "the canonical contract names a buffer binding reims did not stage",
            ));
        };
        if staged.bytes.is_empty() {
            return Ok(NarrowOutcome::Outside(
                "a zero-length staged buffer has no canonical allocation extent",
            ));
        }
        bindings.push((
            binding.metal_binding,
            binding.access,
            staged.bytes.as_slice(),
        ));
    }

    // The mirror direction of the class rule: every binding the request staged
    // must be named by the canonical contract, and the two must agree on
    // whether it is writable. A staged binding the contract does not name is
    // the two pinned translators (`9e0e99a` on this side, `43c46ac` in the
    // provider) describing different kernels — the binding's bytes would never
    // reach the trace, and a writable one would come back as a readback-count
    // mismatch only after the provider had already run. The shape therefore
    // stays on the reims engine, exactly like the forward rule above; the
    // seam's own count check remains the fail-closed backstop for anything
    // this walk does not name. Checked before any lease is imported.
    for staged in &req.storage_buffers {
        let Some(named) = pipeline
            .contract
            .buffer_bindings
            .iter()
            .find(|binding| binding.metal_binding == staged.binding)
        else {
            return Ok(NarrowOutcome::Outside(
                "the reims reflection staged a binding the canonical contract does not name",
            ));
        };
        if named.access.is_writable() != staged.writable {
            return Ok(NarrowOutcome::Outside(
                "the two reflections disagree on whether a staged binding is writable",
            ));
        }
    }

    // The texture half's own pair rule (`research/docs/26` §21.3, steps 1-3).
    // The trace carries the module's compiled contract, so the request's staged
    // view has to be the view that declaration names, field by field; a
    // disagreement is the two readings of one module rather than a shape the
    // provider refuses, and the engine keeps it exactly as the buffer face's
    // mirror rule above does. The sampler half is the one place the two
    // readings must agree or the dispatch is fail-closed: both claim to be the
    // module's own `@__air_sampler_state`, and the texels a sample returns
    // depend on which one executes.
    let mut textures = Vec::new();
    if let Some(staged) = &texture {
        let declared =
            match pipeline.contract.texture_bindings.as_slice() {
                [declared] => declared,
                [] => return Ok(NarrowOutcome::Outside(
                    "the request staged a sampled image the canonical contract does not declare",
                )),
                _ => return Ok(NarrowOutcome::Outside(
                    "a contract that declares more than one texture binding keeps the dispatch on \
                     the self-contained engine (the reviewed compute texture face pairs one)",
                )),
            };
        if declared.metal_binding != staged.binding {
            return Ok(NarrowOutcome::Outside(
                "the contract's texture binding is not the binding the request staged",
            ));
        }
        if declared.access != TextureAccess::Sampled
            || declared.texture_type != TextureType::D2
            || declared.footprint != TextureFootprintProof::WholeView
        {
            return Ok(NarrowOutcome::Outside(
                "a texture binding the canonical contract declares outside the reviewed D2 \
                 sampled whole-view shape keeps the dispatch on the self-contained engine",
            ));
        }
        if declared.format != staged.format {
            return Ok(NarrowOutcome::Outside(
                "the two readings of the same module disagree on the sampled texture's format",
            ));
        }
        if let Some(module_sampler) = module_sampler {
            let stated = policy_of_state(&module_sampler.state)
                .expect("the class gate admitted the module's own state, so it is a nameable one");
            let Some(contract_policy) = declared.sampler else {
                return Ok(NarrowOutcome::Outside(
                    "a sampled texture declaration that carries no sampler state keeps the \
                     dispatch on the self-contained engine (the two rails cannot pair a state \
                     nobody stated)",
                ));
            };
            if contract_policy != stated {
                return Err(ProviderComputeDecline::ComputeTextureSamplerMismatch {
                    binding: declared.metal_binding,
                    request_filter: filter_name(stated.filter),
                    request_address: address_name(stated.address),
                    contract_filter: filter_name(contract_policy.filter),
                    contract_address: address_name(contract_policy.address),
                });
            }
        }
        textures.push(TextureView {
            view_id: ViewId::new(texture_view_id(staged.binding)),
            metal_binding: staged.binding,
            allocation_id: AllocationId::new(texture_allocation_id(staged.binding)),
            texture_type: TextureType::D2,
            format: staged.format,
            width: staged.width,
            height: staged.height,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access: TextureAccess::Sampled,
            source: TextureSource::OwnedBytes(staged.bytes.to_vec()),
        });
    }
    // The sampler descriptor the request carried is consumed here: the
    // canonical contract states its state on the texture binding instead, so
    // nothing about it travels beside the view. Reading it keeps the walk's
    // two sources visible in one place rather than leaving `sampler` an unused
    // binding.
    debug_assert_eq!(
        sampler.is_some(),
        module_sampler.is_some(),
        "the class gate pairs the request's sampler with the module's own state"
    );

    // Every binding leaves as an owner-issued lease: a binding whose bytes came
    // from a registered guest RAM window is imported without copying, and one
    // with no window behind it is imported as a staged lease. Both are imported
    // through the provider's own lease API before the trace exists, so a
    // refused import never reaches admission.
    let requests: Vec<OwnerRequest<'_>> = bindings
        .iter()
        .map(
            |(binding, _, bytes)| match windows.iter().find(|w| w.binding == *binding) {
                Some(window) => OwnerRequest::Window(*window),
                None => OwnerRequest::Staged(OwnerStaged {
                    binding: *binding,
                    bytes,
                }),
            },
        )
        .collect();
    let leases =
        provider_owner::plan(provider, &requests).map_err(ProviderComputeDecline::Owner)?;

    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size, reservation) in leases.leases() {
        if let Err(error) = resources.insert_allocation(AllocationRecord {
            allocation_id: allocation,
            owner_epoch: provider.device_epoch(),
            size,
        }) {
            leases.abort(provider);
            return Err(ProviderComputeDecline::TraceAdmission {
                detail: error.to_string(),
            });
        }
        if let Err(error) = resources.insert_lease(reservation) {
            leases.abort(provider);
            return Err(ProviderComputeDecline::TraceAdmission {
                detail: error.to_string(),
            });
        }
    }
    let mut buffers = Vec::with_capacity(bindings.len());
    for (binding, access, _) in &bindings {
        let view = leases
            .view(*binding)
            .expect("the owner plan covers every staged binding");
        buffers.push(BufferView {
            view_id: ViewId::new(u64::from(*binding) + 1),
            metal_binding: *binding,
            allocation_id: view.allocation,
            offset: view.view_offset,
            length: view.view_length,
            access: *access,
            attribute_stride: None,
            source: match view.channel {
                provider_owner::Channel::Borrowed => BufferSource::BorrowedNoCopy(view.lease),
                provider_owner::Channel::Staged => BufferSource::StagedLease(view.lease),
            },
        });
    }

    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(NEXT_OPERATION_ID.fetch_add(1, Ordering::Relaxed)),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![TracePass::Compute(ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers,
            textures: textures.clone(),
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid,
                threads_per_threadgroup,
            },
        })],
        completion_policy: metal_api_core::provider::CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    // C1c: a pass whose table carries compute texture declarations crosses the
    // owner→provider wire before anything is admitted. The frame is the payload
    // — the declaration the provider compiled beside the view this rail staged,
    // source bytes and all — it is decoded again with the provider's own
    // decoder, and it is what the admission below sees, so a field the frame
    // cannot carry is a typed decline here at the seam rather than a difference
    // discovered when the owner and the provider are two processes. A trace
    // whose table declares no texture keeps the exact path (and bytes) it had
    // before the texture half existed.
    let (trace, resources) = if textures.is_empty() {
        (trace, resources)
    } else {
        let frame = match provider_wire::submit_frame(&trace, &resources) {
            Ok(frame) => frame,
            Err(decline) => {
                leases.abort(provider);
                return Err(ProviderComputeDecline::ComputeTextureWire {
                    step: decline.step,
                    detail: decline.detail,
                });
            }
        };
        provider_wire::note_submit_frame();
        match provider_wire::carried_submission(&frame) {
            Ok(pair) => pair,
            Err(decline) => {
                leases.abort(provider);
                return Err(ProviderComputeDecline::ComputeTextureWire {
                    step: decline.step,
                    detail: decline.detail,
                });
            }
        }
    };
    let validated = match provider.capabilities().validate_trace(trace, resources) {
        Ok(validated) => validated,
        Err(error) => {
            leases.abort(provider);
            return Err(refusal_decline(&error, "trace_admission"));
        }
    };
    let result = match provider.submit(validated) {
        Ok(result) => result,
        Err(error) => {
            // The refusal is mapped first: a `DeviceLost` refusal runs the
            // contract's teardown over the owner ledger, and the census it
            // reports is the state the teardown found — before this plan's own
            // unwind below releases what it imported.
            let decline = refusal_decline(&error, "submission");
            leases.abort(provider);
            return Err(decline);
        }
    };
    if !matches!(
        result.completion,
        CompletionDisposition::CompletedVisible { .. }
    ) {
        leases.abort(provider);
        return Err(ProviderComputeDecline::CompletionNotVisible);
    }

    // Which bytes the provider wrote comes from the completion itself
    // (`BufferWriteback`, allocation coordinates). They are re-based onto the
    // staged spans here, with the view bounds checked, so the caller's guest
    // write is exactly the interval the provider named.
    let writable: Vec<u32> = bindings
        .iter()
        .filter(|(_, access, _)| access.is_writable())
        .map(|(binding, _, _)| *binding)
        .collect();
    let writebacks = match staged_writebacks(&writable, leases.views(), &result.writebacks) {
        Ok(writebacks) => writebacks,
        Err(decline) => {
            leases.abort(provider);
            return Err(decline);
        }
    };
    // The retirement chain: the completion token binds every lease, the
    // completion retires it, the window is reclaimed and the provider's import
    // is released — `research/docs/20` §3.4 in order. It runs before the caller
    // writes anything back, so no provider-visible byte reaches guest memory
    // before the leases that cover it are known to be retired. A refusal here
    // declines the dispatch rather than letting the import outlive its
    // evidence.
    leases
        .settle(provider, result.completion)
        .map_err(ProviderComputeDecline::Owner)?;
    Ok(NarrowOutcome::Completed(ProviderComputeOutput {
        writebacks,
        images: Vec::new(),
    }))
}

/// Re-base one submission's provider writebacks onto the staged spans.
///
/// Writable bindings only, in the caller's binding order. The writeback's
/// view/allocation identity is the same `binding + 1` mapping the view carried
/// in, so one table is enough; a missing writeback is
/// [`ProviderComputeDecline::WritebackMissing`], and one that does not lie
/// inside the view the trace carried is
/// [`ProviderComputeDecline::WritebackOutsideView`] — a provider-reported range
/// the staged span cannot hold is never truncated into place.
fn staged_writebacks(
    writable: &[u32],
    views: &[provider_owner::View],
    writebacks: &[BufferWriteback],
) -> Result<Vec<ProviderWriteback>, ProviderComputeDecline> {
    let mut out = Vec::with_capacity(writable.len());
    for binding in writable {
        let view = views
            .iter()
            .find(|view| view.binding == *binding)
            .expect("the owner plan covers every staged binding");
        let view_id = ViewId::new(u64::from(*binding) + 1);
        let writeback = match writebacks.iter().find(|writeback| {
            writeback.view_id == view_id && writeback.allocation_id == view.allocation
        }) {
            Some(writeback) => writeback,
            None => {
                return Err(ProviderComputeDecline::WritebackMissing { binding: *binding });
            }
        };
        let length = u64::try_from(writeback.bytes.len()).unwrap_or(u64::MAX);
        let view_end = view.view_offset.checked_add(view.view_length).ok_or(
            ProviderComputeDecline::WritebackOutsideView {
                binding: *binding,
                offset: writeback.offset,
                length,
                view_offset: view.view_offset,
                view_length: view.view_length,
            },
        )?;
        let writeback_end = writeback.offset.checked_add(length).ok_or(
            ProviderComputeDecline::WritebackOutsideView {
                binding: *binding,
                offset: writeback.offset,
                length,
                view_offset: view.view_offset,
                view_length: view.view_length,
            },
        )?;
        if writeback.offset < view.view_offset || writeback_end > view_end {
            return Err(ProviderComputeDecline::WritebackOutsideView {
                binding: *binding,
                offset: writeback.offset,
                length,
                view_offset: view.view_offset,
                view_length: view.view_length,
            });
        }
        out.push(ProviderWriteback {
            binding: *binding,
            offset: writeback.offset - view.view_offset,
            bytes: writeback.bytes.clone(),
            allocation: writeback.allocation_id,
            allocation_offset: writeback.offset,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::engine::ComputeDispatchRegion;
    use metal_api_core::provider::{
        AllocationId, BufferLease, DeviceEpoch, LeaseId, LeaseReservation, ProviderPhase,
    };

    /// One view whose geometry the trace would carry: a window inside a larger
    /// allocation, exactly the owner rail's borrowed shape.
    fn window_view(binding: u32, view_offset: u64, view_length: u64) -> provider_owner::View {
        let lease = LeaseId::new(u64::from(binding) + 1);
        let allocation = AllocationId::new(u64::from(binding) + 11);
        provider_owner::View {
            binding,
            channel: provider_owner::Channel::Borrowed,
            lease,
            allocation,
            allocation_size: 2 * 4096,
            reservation: LeaseReservation {
                lease: BufferLease {
                    lease_id: lease,
                    allocation_id: allocation,
                    owner_epoch: DeviceEpoch::new(1),
                },
                offset: view_offset,
                length: view_length,
            },
            view_offset,
            view_length,
        }
    }

    fn writeback(allocation: u64, offset: u64, bytes: Vec<u8>) -> BufferWriteback {
        BufferWriteback {
            view_id: ViewId::new(1),
            allocation_id: AllocationId::new(allocation),
            offset,
            bytes,
        }
    }

    /// One `ComputeRequest` whose dispatch is exactly the translator's own
    /// decomposition of `grid` threads in `local`-sized threadgroups.
    fn request_of_launch(grid: [u32; 3], local: [u32; 3]) -> ComputeRequest {
        let plan = metal2vulkan::reflect::KernelDispatch::ThreadsDynamic { offset: 0 }
            .plan(local, Some(grid))
            .expect("the launch plans");
        ComputeRequest {
            spirv: Vec::new(),
            entry: "main".into(),
            dispatch: ComputeDispatch::Regions {
                push_offset: 0,
                threadgroups_per_grid: plan.threadgroups_per_grid,
                regions: plan
                    .regions
                    .iter()
                    .map(|region| ComputeDispatchRegion {
                        local_size: region.local_size,
                        group_count: region.group_count,
                        push_constants: plan.push_constants(*region),
                    })
                    .collect(),
            },
            storage_buffers: vec![crate::backend::vulkan::engine::ComputeBufferResource {
                binding: 0,
                bytes: vec![0u8; 64],
                writable: true,
            }],
            sampled_images: Vec::new(),
            samplers: Vec::new(),
            storage_images: Vec::new(),
        }
    }

    /// The class admits exactly the translator's own decomposition of one
    /// exact-thread launch — the single-region shape, and the several regions a
    /// partial threadgroup tiles into — and recovers that launch's Metal-level
    /// `(grid, local)` from the tiles.
    ///
    /// The recovered launch is what the canonical pass carries, so it is the
    /// thing to assert: for a tail launch the widest region is the interior
    /// (whose local size is the nominal one) and the far edge is the tail's
    /// `thread_base + local_size` — not the tail region's own renumbered grid.
    #[test]
    fn the_class_recovers_the_launch_a_region_list_decomposes() {
        // The single-region control: the old mapping and the recovered launch
        // are the same numbers.
        let single_request = request_of_launch([4, 1, 1], [4, 1, 1]);
        let single = narrow_class(&single_request, &[]).expect("one exact region is the class");
        assert_eq!(single.launch.grid, [4, 1, 1]);
        assert_eq!(single.launch.threads_per_threadgroup, [4, 1, 1]);

        // A tail launch: 6 threads in 4-wide groups are two regions (the
        // interior 0..3 and the slab 4..5), and the launch they decompose is
        // still 6 threads in 4-wide groups.
        let tail_request = request_of_launch([6, 1, 1], [4, 1, 1]);
        let ComputeDispatch::Regions { regions, .. } = &tail_request.dispatch else {
            panic!("a tail launch is a regions dispatch");
        };
        assert_eq!(regions.len(), 2, "a partial threadgroup is two regions");
        assert_eq!(
            regions[1].push_constants[3..6],
            [4, 0, 0],
            "the slab's payload starts its threads at 4"
        );
        let tail =
            narrow_class(&tail_request, &[]).expect("the translator's own tiling is the class");
        assert_eq!(
            tail.launch.grid,
            [6, 1, 1],
            "the launch is the tiling's cover"
        );
        assert_eq!(tail.launch.threads_per_threadgroup, [4, 1, 1]);

        // Eight regions (a boundary on every axis): the cover is the whole
        // grid, not any one region's span.
        let all_axes_request = request_of_launch([130, 5, 3], [64, 2, 2]);
        let all_axes =
            narrow_class(&all_axes_request, &[]).expect("an eight-region tiling is the class");
        assert_eq!(all_axes.launch.grid, [130, 5, 3]);
        assert_eq!(all_axes.launch.threads_per_threadgroup, [64, 2, 2]);

        // A grid smaller than one threadgroup is a single slab: the recovered
        // local size is the effective one, and re-planning it reproduces the
        // same single region.
        let sub_group_request = request_of_launch([3, 1, 1], [8, 1, 1]);
        let sub_group =
            narrow_class(&sub_group_request, &[]).expect("a sub-threadgroup grid is the class");
        assert_eq!(sub_group.launch.grid, [3, 1, 1]);
        assert_eq!(sub_group.launch.threads_per_threadgroup, [3, 1, 1]);
    }

    /// The class *carries* the request's stated payload offset rather than
    /// judging it: the agreement with the canonical contract needs the
    /// compiled reflection, so it is an admission check, and the pure gate
    /// admits a launch whatever offset the request names
    /// (`ProviderComputeDecline::PushOffsetMismatch`). This is the boundary the
    /// module docs draw, and the integration rail test is what proves the
    /// refusal itself.
    #[test]
    fn the_class_carries_the_requests_stated_payload_offset() {
        assert_eq!(
            {
                let request = request_of_launch([4, 1, 1], [4, 1, 1]);
                narrow_class(&request, &[])
                    .expect("the derived tiling is the class")
                    .launch
                    .push_offset
            },
            0,
            "the translator's own payload sits at the reflected offset"
        );
        let mut hacked = request_of_launch([4, 1, 1], [4, 1, 1]);
        let ComputeDispatch::Regions { push_offset, .. } = &mut hacked.dispatch else {
            panic!("a launch is a regions dispatch");
        };
        *push_offset = 16;
        assert_eq!(
            narrow_class(&hacked, &[])
                .expect("the class gate has no contract to compare an offset against")
                .launch
                .push_offset,
            16,
            "the class hands the request's claim to admission untouched"
        );
    }

    /// Every way a region list can fail to be the translator's decomposition is
    /// named as out of class — the engine keeps the shape instead of the
    /// provider dispatching other threads.
    #[test]
    fn a_region_list_the_translator_would_not_derive_stays_on_the_engine() {
        const NOT_A_DECOMPOSITION: &str =
            "a region list the translator would not derive from one exact-thread launch \
             stays on the self-contained engine";
        let reason_of = |request: &ComputeRequest| {
            narrow_class(request, &[]).expect_err("this list is not a decomposition")
        };

        // A second copy of the interior region: no single launch tiles into the
        // same threads twice.
        let mut duplicated = request_of_launch([6, 1, 1], [4, 1, 1]);
        let ComputeDispatch::Regions { regions, .. } = &mut duplicated.dispatch else {
            unreachable!()
        };
        regions.push(regions[0]);
        assert_eq!(reason_of(&duplicated), NOT_A_DECOMPOSITION);

        // The tail slab's base moved to zero: the same shapes, but the threads
        // they cover are not this launch's.
        let mut rebased = request_of_launch([6, 1, 1], [4, 1, 1]);
        let ComputeDispatch::Regions { regions, .. } = &mut rebased.dispatch else {
            unreachable!()
        };
        regions[1].push_constants[3] = 0;
        assert_eq!(reason_of(&rebased), NOT_A_DECOMPOSITION);

        // A payload the plan would not produce: the engine would push these
        // words where the provider pushes its own.
        let mut doctored = request_of_launch([6, 1, 1], [4, 1, 1]);
        let ComputeDispatch::Regions { regions, .. } = &mut doctored.dispatch else {
            unreachable!()
        };
        regions[1].push_constants[0] = 5;
        assert_eq!(reason_of(&doctored), NOT_A_DECOMPOSITION);

        // A dropped region: the launch would lose its slab.
        let mut truncated = request_of_launch([6, 1, 1], [4, 1, 1]);
        let ComputeDispatch::Regions { regions, .. } = &mut truncated.dispatch else {
            unreachable!()
        };
        regions.truncate(1);
        assert_eq!(reason_of(&truncated), NOT_A_DECOMPOSITION);

        // A region that launches nothing.
        let mut empty = request_of_launch([6, 1, 1], [4, 1, 1]);
        let ComputeDispatch::Regions { regions, .. } = &mut empty.dispatch else {
            unreachable!()
        };
        regions[0].group_count = [0, 1, 1];
        assert_eq!(
            reason_of(&empty),
            "a region that launches no threads stays on the self-contained engine"
        );

        // A dispatch with no regions at all.
        let mut none = request_of_launch([6, 1, 1], [4, 1, 1]);
        let ComputeDispatch::Regions { regions, .. } = &mut none.dispatch else {
            unreachable!()
        };
        regions.clear();
        assert_eq!(
            reason_of(&none),
            "a launch with no regions stays on the self-contained engine"
        );

        // A threadgroup census that disagrees with the tiling.
        let mut wrong_census = request_of_launch([6, 1, 1], [4, 1, 1]);
        let ComputeDispatch::Regions {
            threadgroups_per_grid,
            ..
        } = &mut wrong_census.dispatch
        else {
            unreachable!()
        };
        *threadgroups_per_grid = [9, 1, 1];
        assert_eq!(reason_of(&wrong_census), NOT_A_DECOMPOSITION);

        // Whole-workgroup launches keep the engine: the canonical Vulkan
        // provider admits exact threads only.
        let mut workgroups = request_of_launch([6, 1, 1], [4, 1, 1]);
        workgroups.dispatch = ComputeDispatch::Workgroups([1, 1, 1]);
        assert!(reason_of(&workgroups).contains("whole-workgroup"));
    }

    /// Every normalized provider class is named by its own reims slug, and the
    /// non-loss classes carry the provider's own slug/detail text through the
    /// refusal builder.
    ///
    /// The `DeviceLost` arm is asserted here only through its slug and its
    /// census-bearing decline: mapping it runs the owner-ledger teardown, which
    /// belongs to the integration test that owns that ledger, not to a unit
    /// test sharing this process's global rail.
    #[test]
    fn every_provider_refusal_class_has_its_own_reims_slug() {
        let table = [
            (
                ProviderErrorClass::Args,
                ProviderRefusalClass::Args,
                "provider_args",
            ),
            (
                ProviderErrorClass::Capability,
                ProviderRefusalClass::Capability,
                "provider_capability",
            ),
            (
                ProviderErrorClass::Resource,
                ProviderRefusalClass::Resource,
                "provider_resource",
            ),
            (
                ProviderErrorClass::Compile,
                ProviderRefusalClass::Compile,
                "provider_compile",
            ),
            (
                ProviderErrorClass::Execute,
                ProviderRefusalClass::Execute,
                "provider_execute",
            ),
            (
                ProviderErrorClass::DeviceLost,
                ProviderRefusalClass::DeviceLost,
                "provider_device_lost",
            ),
            (
                ProviderErrorClass::Internal,
                ProviderRefusalClass::Internal,
                "provider_internal",
            ),
        ];
        assert_eq!(
            table.len(),
            ProviderRefusalClass::ALL.len(),
            "the mapping covers every normalized class"
        );
        for (provider_class, class, slug) in table {
            assert_eq!(ProviderRefusalClass::from_provider(provider_class), class);
            assert_eq!(class.slug(), slug);
            assert!(ProviderRefusalClass::ALL.contains(&class));
            if class == ProviderRefusalClass::DeviceLost {
                continue;
            }
            let error =
                ProviderError::new(ProviderPhase::Submit, provider_class, "boundary_refusal")
                    .expect("a static non-empty slug");
            let decline = refusal_decline(&error, "submission");
            assert_eq!(decline.slug(), slug);
            let fields = decline.fields();
            assert!(fields
                .iter()
                .any(|(key, value)| *key == "class" && value == class.name()));
            assert!(fields
                .iter()
                .any(|(key, value)| *key == "step" && value == "submission"));
            assert!(
                fields
                    .iter()
                    .any(|(key, value)| *key == "detail" && value.contains("boundary_refusal")),
                "the provider's own text must ride along: {fields:?}"
            );
        }

        // The device-loss refusal is its own typed variant, and its census is
        // rendered on the same line as the provider's text.
        let lost = ProviderComputeDecline::ProviderDeviceLost {
            step: "submission",
            detail: "vulkan-queue-submit: VK_ERROR_DEVICE_LOST".into(),
            teardown: DeviceLossTeardown {
                leases: 2,
                windows: 1,
            },
        };
        assert_eq!(lost.slug(), "provider_device_lost");
        let fields = lost.fields();
        assert!(fields
            .iter()
            .any(|(key, value)| *key == "teardown_leases" && value == "2"));
        assert!(fields
            .iter()
            .any(|(key, value)| *key == "teardown_windows" && value == "1"));
        assert!(fields
            .iter()
            .any(|(key, value)| *key == "detail" && value.contains("VK_ERROR_DEVICE_LOST")));
    }

    /// The provider's writeback keeps its own offset: an interval in the middle
    /// of a view maps to that offset inside the staged span, and the whole-view
    /// writeback maps to zero. Nothing here assumes "the whole buffer was
    /// written".
    #[test]
    fn a_provider_writeback_keeps_its_offset_inside_the_view() {
        let views = [window_view(0, 4096, 16)];

        let partial = [writeback(11, 4104, vec![0xAB; 8])];
        let mapped =
            staged_writebacks(&[0], &views, &partial).expect("a writeback inside its view");
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].binding, 0);
        assert_eq!(
            mapped[0].offset, 8,
            "the interval keeps its offset instead of being moved to the start \
             of the staged span"
        );
        assert_eq!(mapped[0].bytes, vec![0xAB; 8]);
        assert_eq!(
            mapped[0].allocation_offset, 4104,
            "the provider's own coordinates stay on the writeback as evidence"
        );
        assert_eq!(mapped[0].allocation, AllocationId::new(11));

        let whole = [writeback(11, 4096, vec![0xCD; 16])];
        let mapped = staged_writebacks(&[0], &views, &whole).expect("the whole view");
        assert_eq!(mapped[0].offset, 0);
        assert_eq!(mapped[0].bytes, vec![0xCD; 16]);

        // A writeback for a binding the trace wrote nothing to is the provider
        // saying more than the trace asked for: read-only bindings are skipped,
        // so it is simply not collected.
        let mapped = staged_writebacks(&[], &views, &[writeback(11, 4096, vec![0xCD; 16])])
            .expect("no writable binding, no interval");
        assert!(mapped.is_empty());
    }

    /// A writeback the staged span cannot hold is refused by name — past the
    /// view's end, straddling it, or before its start — and a missing writeback
    /// keeps its own slug. Neither is truncated into place.
    #[test]
    fn a_writeback_outside_its_view_is_refused_by_name() {
        let views = [window_view(0, 4096, 16)];

        let past_end = [writeback(11, 4112, vec![0u8; 8])];
        assert_eq!(
            staged_writebacks(&[0], &views, &past_end)
                .expect_err("past the view end")
                .slug(),
            "writeback_outside_view"
        );
        let straddling = [writeback(11, 4104, vec![0u8; 16])];
        let decline = staged_writebacks(&[0], &views, &straddling).expect_err("straddling the end");
        assert_eq!(decline.slug(), "writeback_outside_view");
        assert_eq!(
            decline.fields(),
            vec![
                ("binding", "0".to_string()),
                ("offset", "4104".to_string()),
                ("length", "16".to_string()),
                ("view_offset", "4096".to_string()),
                ("view_length", "16".to_string()),
            ],
            "the refusal names the interval and the view it missed"
        );
        let before_start = [writeback(11, 4090, vec![0u8; 8])];
        assert_eq!(
            staged_writebacks(&[0], &views, &before_start)
                .expect_err("before the view start")
                .slug(),
            "writeback_outside_view"
        );
        assert_eq!(
            staged_writebacks(&[0], &views, &[])
                .expect_err("no writeback")
                .slug(),
            "writeback_missing"
        );
        let foreign_allocation = [writeback(99, 4096, vec![0u8; 16])];
        assert_eq!(
            staged_writebacks(&[0], &views, &foreign_allocation)
                .expect_err("another allocation's writeback")
                .slug(),
            "writeback_missing"
        );
    }

    /// S3: the pipeline cache key carries the entry point. One AIR with two
    /// entry names is two cache keys, so a lookup for the second entry can
    /// never be answered by the first entry's compiled pipeline — which is
    /// what an AIR-only key would do.
    #[test]
    fn the_pipeline_cache_key_carries_the_entry_point() {
        let air = b"one AIR, two entries".as_slice();
        assert_eq!(
            pipeline_cache_key(air, "apv_cs"),
            pipeline_cache_key(air, "apv_cs"),
            "one (AIR, entry) pair is one key"
        );
        assert_ne!(
            pipeline_cache_key(air, "apv_cs"),
            pipeline_cache_key(air, "apv_cs_second"),
            "the same AIR under another entry is another key"
        );

        // The key is the one the rail's own map uses, so two entries of one AIR
        // occupy two slots instead of colliding.
        let mut cache: HashMap<PipelineKey, &'static str> = HashMap::new();
        cache.insert(pipeline_cache_key(air, "apv_cs"), "first");
        cache.insert(pipeline_cache_key(air, "apv_cs_second"), "second");
        assert_eq!(cache.len(), 2);
        assert_eq!(
            cache.get(&pipeline_cache_key(air, "apv_cs_second")),
            Some(&"second"),
            "the second entry hits its own compiled pipeline"
        );
    }
}
