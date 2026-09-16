//! Gate 2 entry spike: drive the canonical compute provider from this crate.
//!
//! One reviewed compute trace — the same `copy_word` fixture the
//! `metal-api-emulator` provider-smoke suite uses — is compiled through the
//! contract's [`metal_api_core::provider::ComputeProvider`] entry points and
//! submitted to [`metal_api_vulkan::VulkanComputeProvider`] on a native
//! Lavapipe ICD. The falsifiable claim is byte equality between the input
//! `OwnedBytes` view and the writeback the provider's completion returns.
//!
//! This is the dependency edge, not the production rail: the shipped
//! staticlib never reaches this file, and the guest submit path does not call
//! the provider yet. It proves the fork's build graph can link and drive the
//! canonical provider with no Docker or VM.

#![cfg(feature = "backend-vulkan")]

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferAccess, BufferSource, BufferView,
    CompiledComputePipeline, CompletionDisposition, ComputePass, ComputeProvider, ComputeTrace,
    Dispatch, DispatchKind, DispatchType, OperationId, ResourceTableSnapshot, SemanticDigest,
    TracePass, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

/// The reviewed single-word copy kernel from `metal-api-emulator`'s
/// `examples/metal-smoke/shaders/kernel_copy_word.ll`. It reads one `uint` at
/// binding 0 and stores it to binding 1.
const COPY_WORD_AIR: &str = include_str!(
    "../../../../../metal-api-emulator/examples/metal-smoke/shaders/kernel_copy_word.ll"
);

const INPUT_VIEW: ViewId = ViewId::new(1001);
const INPUT_ALLOCATION: AllocationId = AllocationId::new(1010);
const OUTPUT_VIEW: ViewId = ViewId::new(1002);
const OUTPUT_ALLOCATION: AllocationId = AllocationId::new(1011);

/// The input word the trace writes through, chosen to make the writeback
/// unmistakable in the captured output.
const INPUT_WORD: [u8; 4] = 0x6745_2301u32.to_le_bytes();

#[test]
fn reims_side_code_drives_the_canonical_compute_provider_to_readback() {
    let executor = match VulkanExecutor::new() {
        Ok(executor) => executor,
        Err(error) => {
            eprintln!(
                "SKIP: no Vulkan device for the canonical provider \
                 (pin VK_ICD_FILENAMES to a Lavapipe ICD): {error}"
            );
            return;
        }
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor))
        .expect("canonical provider starts");
    let device = Device::new(executor as Arc<dyn ComputeExecutor>);

    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("fixture library loads")
        .function("copy_word")
        .expect("fixture entry exists");
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new(
                "reims-gate2-entry-v1",
                b"canonical_provider_compute".to_vec(),
            )
            .expect("digest"),
        )
        .expect("compute pipeline registers");

    let trace = trace(&provider, pipeline);
    let admitted = provider
        .capabilities()
        .validate_trace(trace, resources(&provider))
        .expect("the reviewed compute trace is admitted");
    let result = provider
        .submit(admitted)
        .expect("the canonical dispatch completes");

    assert!(
        matches!(
            result.completion,
            CompletionDisposition::CompletedVisible { .. }
        ),
        "completion must be host-visible: {:?}",
        result.completion
    );
    let writeback = result
        .writebacks
        .iter()
        .find(|writeback| writeback.view_id == OUTPUT_VIEW)
        .unwrap_or_else(|| panic!("view {OUTPUT_VIEW:?} has no writeback"));
    eprintln!(
        "canonical provider readback (view {:?}, allocation {:?}): {:02x?}",
        writeback.view_id, writeback.allocation_id, writeback.bytes
    );
    assert_eq!(writeback.bytes, INPUT_WORD);
}

/// One compute-only canonical trace: a single `ThreadsExact` dispatch of the
/// copy kernel with one input view and one output view, both owned bytes, and
/// host readback as the completion policy.
fn trace(provider: &VulkanComputeProvider, pipeline: CompiledComputePipeline) -> ComputeTrace {
    let pipeline_id = pipeline.pipeline_id;
    ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(41),
        pipelines: vec![pipeline],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![TracePass::Compute(ComputePass {
            pipeline: pipeline_id,
            buffers: vec![
                BufferView {
                    view_id: INPUT_VIEW,
                    metal_binding: 0,
                    allocation_id: INPUT_ALLOCATION,
                    offset: 0,
                    length: 4,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(INPUT_WORD.to_vec()),
                },
                BufferView {
                    view_id: OUTPUT_VIEW,
                    metal_binding: 1,
                    allocation_id: OUTPUT_ALLOCATION,
                    offset: 0,
                    length: 4,
                    access: BufferAccess::Write,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0xab; 4]),
                },
            ],
            textures: Vec::new(),
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
        })],
        completion_policy: metal_api_core::provider::CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    }
}

/// The allocation table the trace's views name: two four-byte allocations in
/// the provider's device epoch.
fn resources(provider: &VulkanComputeProvider) -> ResourceTableSnapshot {
    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [(INPUT_ALLOCATION, 4), (OUTPUT_ALLOCATION, 4)] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("gate 2 entry fixture allocation");
    }
    resources
}
