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
//! - a `ComputeDispatch::Regions` launch with exactly one region — the shape
//!   the canonical Vulkan provider can express as one `ThreadsExact` dispatch
//!   (it refuses whole-workgroup launches today, `supports_threadgroups` is
//!   false);
//! - every buffer binding the compiled canonical contract names must be
//!   present in the reims-staged request, so `Unused`/`Absent` reflection
//!   cases stay on the reims engine.
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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use metal_api_core::provider::{
    AllocationRecord, BufferSource, BufferView, CompiledComputePipeline, CompletionDisposition,
    ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType,
    NoCopyLeaseImporter, OperationId, ProviderError, ResourceTableSnapshot, SemanticDigest,
    TracePass, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};

use super::provider_owner::{self, Request as OwnerRequest, Staged as OwnerStaged, Window};
use super::vulkan::engine::types::ComputeBufferOutput;
use super::vulkan::engine::{ComputeDispatch, ComputeOutput, ComputeRequest};
use crate::observe::{decline_display, Decline};

/// Why one reims compute request did not leave this rail for the engine.
#[derive(Debug)]
pub enum ComputeRailOutcome {
    /// The canonical provider executed the dispatch. `buffers` carries one
    /// readback per writable binding, in canonical binding order; `images` is
    /// empty because the narrow class carries no storage images.
    ProviderCompleted(ComputeOutput),
    /// Outside the narrow admitted class. The caller must run the
    /// self-contained engine, exactly as a build without the feature would.
    NotInNarrowClass(&'static str),
    /// In-class, but the canonical provider refused. The caller must decline
    /// the dispatch rather than fall back to another rail.
    ProviderDeclined(ProviderComputeDecline),
}

/// A typed refusal of the canonical-provider rail, nameable in the fail log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderComputeDecline {
    /// The process-global provider could not be created: no usable Vulkan
    /// device for the canonical rail, or a capability snapshot refused.
    ProviderUnavailable { detail: String },
    /// Compiling the reviewed AIR in the canonical translator failed.
    PipelineCompile { detail: String },
    /// The canonical admission refused the translated trace.
    TraceAdmission { detail: String },
    /// The provider refused or lost the submission.
    Submission { detail: String },
    /// The completion was not host-visible, so there are no bytes to write
    /// back.
    CompletionNotVisible,
    /// A writable binding's readback did not come back through the provider
    /// completion channel.
    WritebackMissing { binding: u32 },
    /// The owner rail refused a registration, an import, a retirement or a
    /// release. Delegated, so the slug names the owner check that refused.
    Owner(provider_owner::Decline),
}

impl Decline for ProviderComputeDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ProviderUnavailable { .. } => "provider_unavailable",
            Self::PipelineCompile { .. } => "pipeline_compile",
            Self::TraceAdmission { .. } => "trace_admission",
            Self::Submission { .. } => "submission",
            Self::CompletionNotVisible => "completion_not_visible",
            Self::WritebackMissing { .. } => "writeback_missing",
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
            Self::ProviderUnavailable { detail }
            | Self::PipelineCompile { detail }
            | Self::TraceAdmission { detail }
            | Self::Submission { detail } => vec![("detail", detail.clone())],
            Self::CompletionNotVisible => Vec::new(),
            Self::WritebackMissing { binding } => vec![("binding", binding.to_string())],
            Self::Owner(inner) => inner.fields(),
        }
    }
}

decline_display!(ProviderComputeDecline);

/// Render a [`ProviderError`] without a `Display` impl: its slug plus the
/// optional detail, which is where the canonical provider names the refusing
/// check. Class and phase are recoverable from the slug family in the log.
fn provider_error_detail(error: &ProviderError) -> String {
    match &error.detail {
        Some(detail) => format!("{}: {detail}", error.slug),
        None => error.slug.clone(),
    }
}

/// One process-global canonical rail: the provider, the neutral `Device` used
/// to compile AIR, and a per-AIR pipeline cache. The provider keeps its own
/// Vulkan instance/device, so it never shares the self-contained engine's.
struct ProviderRail {
    provider: VulkanComputeProvider,
    device: Device,
    /// AIR identity → compiled pipeline. Compiled pipelines are metadata
    /// handles into the provider's own registry, so the cache bounds pipeline
    /// registrations by distinct kernel AIR rather than by dispatch count.
    pipelines: Mutex<HashMap<Vec<u8>, CompiledComputePipeline>>,
}

static NEXT_OPERATION_ID: AtomicU64 = AtomicU64::new(1);

fn rail() -> Result<&'static ProviderRail, ProviderComputeDecline> {
    static RAIL: OnceLock<Result<ProviderRail, String>> = OnceLock::new();
    RAIL.get_or_init(|| {
        let executor = VulkanExecutor::new().map_err(|error| error.to_string())?;
        let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor))
            .map_err(|error| provider_error_detail(&error))?;
        let device = Device::new(executor as Arc<dyn ComputeExecutor>);
        Ok(ProviderRail {
            provider,
            device,
            pipelines: Mutex::new(HashMap::new()),
        })
    })
    .as_ref()
    .map_err(|detail| ProviderComputeDecline::ProviderUnavailable {
        detail: detail.clone(),
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
    if let Some(reason) = narrow_class_reason(req) {
        return ComputeRailOutcome::NotInNarrowClass(reason);
    }
    match submit_narrow(air, entry, req, windows) {
        Ok(NarrowOutcome::Completed(output)) => ComputeRailOutcome::ProviderCompleted(output),
        Ok(NarrowOutcome::Outside(reason)) => ComputeRailOutcome::NotInNarrowClass(reason),
        Err(decline) => ComputeRailOutcome::ProviderDeclined(decline),
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

/// A narrow submission's success answers: completed by the provider, or
/// discovered to be outside the class once the canonical contract is known.
/// Provider refusals travel through the `Err` arm instead.
enum NarrowOutcome {
    Completed(ComputeOutput),
    Outside(&'static str),
}

fn narrow_class_reason(req: &ComputeRequest) -> Option<&'static str> {
    if !req.sampled_images.is_empty() || !req.samplers.is_empty() || !req.storage_images.is_empty()
    {
        return Some("compute images/samplers stay on the self-contained engine");
    }
    let ComputeDispatch::Regions { regions, .. } = &req.dispatch else {
        return Some(
            "whole-workgroup launches stay on the self-contained engine \
             (the canonical Vulkan provider admits exact threads only)",
        );
    };
    if regions.len() != 1 {
        return Some("multi-region exact-thread launches stay on the self-contained engine");
    }
    None
}

fn submit_narrow(
    air: &[u8],
    entry: &str,
    req: &ComputeRequest,
    windows: &[Window],
) -> Result<NarrowOutcome, ProviderComputeDecline> {
    let rail = rail()?;
    let provider = &rail.provider;

    let ComputeDispatch::Regions { regions, .. } = &req.dispatch else {
        unreachable!("narrow_class_reason admitted the class");
    };
    let region = &regions[0];

    // The single region covers the whole logical grid, so the canonical
    // `ThreadsExact` grid is the region's group count times its local size,
    // and its threadgroup shape is the region's local size.
    let mut grid = [0u64; 3];
    for axis in 0..3 {
        grid[axis] = u64::from(region.group_count[axis])
            .checked_mul(u64::from(region.local_size[axis]))
            .ok_or_else(|| ProviderComputeDecline::TraceAdmission {
                detail: format!("axis {axis} thread grid overflows u64"),
            })?;
    }
    let threads_per_threadgroup = region.local_size.map(u64::from);

    let pipeline = {
        let mut cache =
            rail.pipelines
                .lock()
                .map_err(|_| ProviderComputeDecline::TraceAdmission {
                    detail: "pipeline cache poisoned".into(),
                })?;
        if let Some(pipeline) = cache.get(air) {
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
                .map_err(|error| ProviderComputeDecline::PipelineCompile {
                    detail: provider_error_detail(&error),
                })?;
            cache.insert(air.to_vec(), compiled.clone());
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
            return Err(ProviderComputeDecline::TraceAdmission {
                detail: provider_error_detail(&error),
            });
        }
    };
    let result = match provider.submit(validated) {
        Ok(result) => result,
        Err(error) => {
            leases.abort(provider);
            return Err(ProviderComputeDecline::Submission {
                detail: provider_error_detail(&error),
            });
        }
    };
    if !matches!(
        result.completion,
        CompletionDisposition::CompletedVisible { .. }
    ) {
        leases.abort(provider);
        return Err(ProviderComputeDecline::CompletionNotVisible);
    }

    // Writable bindings only, in canonical binding order. The writeback's
    // view/allocation identity is the same `binding + 1` mapping the view
    // carried in, so it maps back to the reims binding without a second table.
    let mut output_buffers = Vec::new();
    for (binding, access, _) in &bindings {
        if !access.is_writable() {
            continue;
        }
        let view = leases
            .view(*binding)
            .expect("the owner plan covers every staged binding");
        let view_id = ViewId::new(u64::from(*binding) + 1);
        let writeback = match result
            .writebacks
            .iter()
            .find(|writeback| writeback.view_id == view_id)
        {
            Some(writeback) if writeback.allocation_id == view.allocation => writeback,
            _ => {
                leases.abort(provider);
                return Err(ProviderComputeDecline::WritebackMissing { binding: *binding });
            }
        };
        output_buffers.push(ComputeBufferOutput {
            binding: *binding,
            bytes: writeback.bytes.clone(),
        });
    }
    // The retirement chain: the completion token binds every lease, the
    // completion retires it, the window is reclaimed and the provider's import
    // is released — `research/docs/20` §3.4 in order. A refusal here declines
    // the dispatch rather than letting the import outlive its evidence.
    leases
        .settle(provider, result.completion)
        .map_err(ProviderComputeDecline::Owner)?;
    Ok(NarrowOutcome::Completed(ComputeOutput {
        buffers: output_buffers,
        images: Vec::new(),
    }))
}
