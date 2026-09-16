//! Gate 2 production-rail integration test: drive the shipped
//! `backend::provider_compute` seam directly with the reviewed `mul3add1`
//! fixture and assert the canonical provider's byte-exact readback.
//!
//! This is the same seam `runtime/compute_exec::vulkan` calls for an in-class
//! guest dispatch under `feature = "provider-compute"`; the seam is public so
//! this off-VM test can prove the conversion (reims `ComputeRequest` + wrapped
//! AIR → canonical `ComputeTrace`) without a VM. Requires a Vulkan ICD; the
//! pinned Lavapipe path is the acceptance environment.

#![cfg(feature = "provider-compute")]

use reims_vgpu::backend::provider_compute::{submit_compute, ComputeRailOutcome};
use reims_vgpu::backend::vulkan::engine::{
    ComputeBufferResource, ComputeDispatch, ComputeDispatchRegion, ComputeRequest,
};
use std::path::PathBuf;

fn mul3add1_request() -> ComputeRequest {
    let input: Vec<u8> = [1u32, 2, 3, 4]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    ComputeRequest {
        // The canonical rail re-compiles AIR itself; the reims-translated
        // SPIR-V is deliberately left out of this test's stimulus.
        spirv: Vec::new(),
        entry: "main".into(),
        dispatch: ComputeDispatch::Regions {
            push_offset: 0,
            threadgroups_per_grid: [1, 1, 1],
            regions: vec![ComputeDispatchRegion {
                local_size: [4, 1, 1],
                group_count: [1, 1, 1],
                push_constants: [0; 12],
            }],
        },
        storage_buffers: vec![ComputeBufferResource {
            binding: 0,
            bytes: input,
            writable: true,
        }],
        sampled_images: Vec::new(),
        samplers: Vec::new(),
        storage_images: Vec::new(),
    }
}

fn fixture_air() -> Vec<u8> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compute_mul3add1.mtlb");
    let mtlb = std::fs::read(&path).expect("compute_mul3add1.mtlb fixture");
    reims_vgpu::runtime::mtlb::extract_air(&mtlb)
        .expect("wrapped AIR bitcode")
        .to_vec()
}

#[test]
fn the_production_seam_submits_the_reviewed_fixture_through_the_canonical_provider() {
    let air = fixture_air();
    let request = mul3add1_request();
    // `apv_cs` is the AIR entry point this fixture's kernel metadata names;
    // the runtime seam passes it through from its own reflection.
    match submit_compute(&air, "apv_cs", &request) {
        ComputeRailOutcome::ProviderCompleted(out) => {
            assert_eq!(out.images.len(), 0, "the narrow class carries no images");
            assert_eq!(out.buffers.len(), 1, "one writable binding readback");
            assert_eq!(out.buffers[0].binding, 0);
            let readback: Vec<u32> = out.buffers[0]
                .bytes
                .chunks(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("4-byte chunk")))
                .collect();
            assert_eq!(readback, vec![4, 7, 10, 13]);
        }
        ComputeRailOutcome::NotInNarrowClass(reason) => {
            panic!("the reviewed fixture is in the narrow class; refused: {reason}")
        }
        ComputeRailOutcome::ProviderDeclined(decline) => {
            panic!("the canonical provider declined the reviewed fixture: {decline}")
        }
    }
}

#[test]
fn out_of_class_shapes_stay_on_the_self_contained_engine() {
    let air = fixture_air();

    let mut whole_workgroups = mul3add1_request();
    whole_workgroups.dispatch = ComputeDispatch::Workgroups([1, 1, 1]);
    assert!(matches!(
        submit_compute(&air, "apv_cs", &whole_workgroups),
        ComputeRailOutcome::NotInNarrowClass(_)
    ));

    let mut multi_region = mul3add1_request();
    if let ComputeDispatch::Regions { regions, .. } = &mut multi_region.dispatch {
        regions.push(regions[0]);
    }
    assert!(matches!(
        submit_compute(&air, "apv_cs", &multi_region),
        ComputeRailOutcome::NotInNarrowClass(_)
    ));
}
