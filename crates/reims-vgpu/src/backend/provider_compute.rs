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
//! - no sampled images, samplers, or storage images (pure storage buffers);
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
    ProviderErrorClass, ProviderHealth, ResourceTableSnapshot, SemanticDigest, TracePass, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};

use super::provider_owner::{
    self, DeviceLossTeardown, Request as OwnerRequest, Staged as OwnerStaged, Window,
};
use super::vulkan::engine::types::ComputeImageResult;
use super::vulkan::engine::{ComputeDispatch, ComputeRequest};
use crate::observe::{decline_display, Decline};

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
/// requires the two to agree. `windows` names, per canonical binding, the
/// registered guest RAM window its staged bytes came from; a binding with no
/// entry is imported as an owner-issued staged lease instead. See the module
/// docs for the class rules and the owner rail's own docs for the lease
/// lifecycle.
pub fn submit_compute(
    air: &[u8],
    entry: &str,
    req: &ComputeRequest,
    windows: &[Window],
) -> ComputeRailOutcome {
    // The class gate is pure and runs first: an out-of-class shape never
    // touches the rail (no device, no compile, no lease).
    match narrow_class(req) {
        Err(reason) => ComputeRailOutcome::NotInNarrowClass(reason),
        Ok(launch) => match submit_narrow(air, entry, req, launch, windows) {
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
fn narrow_class(req: &ComputeRequest) -> Result<NarrowLaunch, &'static str> {
    if !req.sampled_images.is_empty() || !req.samplers.is_empty() || !req.storage_images.is_empty()
    {
        return Err("compute images/samplers stay on the self-contained engine");
    }
    let ComputeDispatch::Regions {
        threadgroups_per_grid,
        regions,
        ..
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
    Ok(NarrowLaunch {
        grid,
        threads_per_threadgroup: local,
    })
}

fn submit_narrow(
    air: &[u8],
    entry: &str,
    req: &ComputeRequest,
    launch: NarrowLaunch,
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

    // One canonical pass carries the whole launch in Metal units, and the
    // provider re-derives the device-side regions from exactly these two
    // numbers — the tiling `narrow_class` verified the request's own.
    let NarrowLaunch {
        grid,
        threads_per_threadgroup,
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
            textures: Vec::new(),
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
        let single = narrow_class(&request_of_launch([4, 1, 1], [4, 1, 1]))
            .expect("one exact region is the class");
        assert_eq!(single.grid, [4, 1, 1]);
        assert_eq!(single.threads_per_threadgroup, [4, 1, 1]);

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
        let tail = narrow_class(&tail_request).expect("the translator's own tiling is the class");
        assert_eq!(tail.grid, [6, 1, 1], "the launch is the tiling's cover");
        assert_eq!(tail.threads_per_threadgroup, [4, 1, 1]);

        // Eight regions (a boundary on every axis): the cover is the whole
        // grid, not any one region's span.
        let all_axes = narrow_class(&request_of_launch([130, 5, 3], [64, 2, 2]))
            .expect("an eight-region tiling is the class");
        assert_eq!(all_axes.grid, [130, 5, 3]);
        assert_eq!(all_axes.threads_per_threadgroup, [64, 2, 2]);

        // A grid smaller than one threadgroup is a single slab: the recovered
        // local size is the effective one, and re-planning it reproduces the
        // same single region.
        let sub_group = narrow_class(&request_of_launch([3, 1, 1], [8, 1, 1]))
            .expect("a sub-threadgroup grid is the class");
        assert_eq!(sub_group.grid, [3, 1, 1]);
        assert_eq!(sub_group.threads_per_threadgroup, [3, 1, 1]);
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
            narrow_class(request).expect_err("this list is not a decomposition")
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
