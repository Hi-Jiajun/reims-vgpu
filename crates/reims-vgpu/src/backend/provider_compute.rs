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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferSource, BufferView, CompiledComputePipeline,
    CompletionDisposition, ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchKind,
    DispatchType, OperationId, ProviderError, ResourceTableSnapshot, SemanticDigest, TracePass,
    ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};

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
/// requires the two to agree. See the module docs for the class rules.
pub fn submit_compute(air: &[u8], entry: &str, req: &ComputeRequest) -> ComputeRailOutcome {
    if let Some(reason) = narrow_class_reason(req) {
        return ComputeRailOutcome::NotInNarrowClass(reason);
    }
    match submit_narrow(air, entry, req) {
        Ok(NarrowOutcome::Completed(output)) => ComputeRailOutcome::ProviderCompleted(output),
        Ok(NarrowOutcome::Outside(reason)) => ComputeRailOutcome::NotInNarrowClass(reason),
        Err(decline) => ComputeRailOutcome::ProviderDeclined(decline),
    }
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

    // One allocation and one view per canonical binding. Reims stages only
    // used bindings; a canonical binding it did not stage (an `Unused` /
    // `Absent` reflection case) keeps the shape on the reims engine rather
    // than inventing bytes the kernel does not touch.
    let mut buffers = Vec::with_capacity(pipeline.contract.buffer_bindings.len());
    let mut resources = ResourceTableSnapshot::new();
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
        // Nonzero identities only: contract admission refuses identity 0.
        let allocation_id = AllocationId::new(u64::from(binding.metal_binding) + 1);
        let view_id = ViewId::new(u64::from(binding.metal_binding) + 1);
        resources
            .insert_allocation(AllocationRecord {
                allocation_id,
                owner_epoch: provider.device_epoch(),
                size: staged.bytes.len() as u64,
            })
            .map_err(|error| ProviderComputeDecline::TraceAdmission {
                detail: error.to_string(),
            })?;
        buffers.push(BufferView {
            view_id,
            metal_binding: binding.metal_binding,
            allocation_id,
            offset: 0,
            length: staged.bytes.len() as u64,
            access: binding.access,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(staged.bytes.clone()),
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
    let validated = provider
        .capabilities()
        .validate_trace(trace, resources)
        .map_err(|error| ProviderComputeDecline::TraceAdmission {
            detail: provider_error_detail(&error),
        })?;
    let result =
        provider
            .submit(validated)
            .map_err(|error| ProviderComputeDecline::Submission {
                detail: provider_error_detail(&error),
            })?;
    if !matches!(
        result.completion,
        CompletionDisposition::CompletedVisible { .. }
    ) {
        return Err(ProviderComputeDecline::CompletionNotVisible);
    }

    // Writable bindings only, in canonical binding order. The writeback's
    // view/allocation identity is the same `binding + 1` mapping the view
    // carried in, so it maps back to the reims binding without a second table.
    let mut output_buffers = Vec::new();
    for binding in &pipeline.contract.buffer_bindings {
        if !binding.access.is_writable() {
            continue;
        }
        let view_id = ViewId::new(u64::from(binding.metal_binding) + 1);
        let writeback = result
            .writebacks
            .iter()
            .find(|writeback| writeback.view_id == view_id)
            .ok_or(ProviderComputeDecline::WritebackMissing {
                binding: binding.metal_binding,
            })?;
        if writeback.allocation_id != AllocationId::new(u64::from(binding.metal_binding) + 1) {
            return Err(ProviderComputeDecline::WritebackMissing {
                binding: binding.metal_binding,
            });
        }
        output_buffers.push(ComputeBufferOutput {
            binding: binding.metal_binding,
            bytes: writeback.bytes.clone(),
        });
    }
    Ok(NarrowOutcome::Completed(ComputeOutput {
        buffers: output_buffers,
        images: Vec::new(),
    }))
}
