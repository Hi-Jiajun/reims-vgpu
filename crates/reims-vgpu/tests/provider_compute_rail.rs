//! Gate 2 production-rail integration test: drive the shipped
//! `backend::provider_compute` seam directly with the reviewed `mul3add1`
//! fixture and assert the canonical provider's byte-exact readback.
//!
//! This is the same seam `runtime/compute_exec::vulkan` calls for an in-class
//! guest dispatch under `feature = "provider-compute"`; the seam is public so
//! this off-VM test can prove the conversion (reims `ComputeRequest` + wrapped
//! AIR → canonical `ComputeTrace`) without a VM. Requires a Vulkan ICD; the
//! pinned Lavapipe path is the acceptance environment.
//!
//! The owner-rail tests at the bottom drive the same seam with a *registered*
//! guest RAM window: the registration and the lease lifecycle are the shipped
//! `backend::provider_owner` ones, and the host range is a test-owned aligned
//! allocation because only a booted VM has a QEMU RAMBlock mapping.

#![cfg(feature = "provider-compute")]

use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};
use reims_vgpu::backend::provider_compute::{
    device_epoch, host_import_alignment, submit_compute, ComputeRailOutcome, ProviderComputeDecline,
};
use reims_vgpu::backend::provider_owner::{self, Region, Request, Staged, Window};
use reims_vgpu::backend::vulkan::engine::{
    ComputeBufferResource, ComputeDispatch, ComputeDispatchRegion, ComputeRequest,
};
use reims_vgpu::observe::Decline as _;
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
    match submit_compute(&air, "apv_cs", &request, &[]) {
        ComputeRailOutcome::ProviderCompleted(out) => {
            assert_eq!(out.images.len(), 0, "the narrow class carries no images");
            assert_eq!(out.writebacks.len(), 1, "one writable binding readback");
            assert_eq!(out.writebacks[0].binding, 0);
            // This submission had no registered window, so the binding left as
            // a staged lease whose allocation *is* the staged span: the
            // provider's writeback covers it from offset zero.
            assert_eq!(out.writebacks[0].offset, 0);
            assert_eq!(out.writebacks[0].allocation_offset, 0);
            let readback: Vec<u32> = out.writebacks[0]
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
        submit_compute(&air, "apv_cs", &whole_workgroups, &[]),
        ComputeRailOutcome::NotInNarrowClass(_)
    ));

    let mut multi_region = mul3add1_request();
    if let ComputeDispatch::Regions { regions, .. } = &mut multi_region.dispatch {
        regions.push(regions[0]);
    }
    assert!(matches!(
        submit_compute(&air, "apv_cs", &multi_region, &[]),
        ComputeRailOutcome::NotInNarrowClass(_)
    ));
}

/// Page-aligned owner memory for a `VK_EXT_external_memory_host` import: the
/// stand-in for the QEMU RAMBlock mapping the production rail registers, with
/// the same shape — a stable host pointer and a page-aligned extent.
struct AlignedBuffer {
    pointer: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

impl AlignedBuffer {
    fn new(len: usize, alignment: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, alignment).expect("layout");
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
        Self {
            pointer: std::ptr::NonNull::new(pointer).expect("aligned allocation"),
            layout,
        }
    }

    fn as_ptr(&self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.layout.size()) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) }
    }
}

fn input_words(words: [u32; 4]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn readback_words(out: &[u8]) -> Vec<u32> {
    out.chunks(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("4-byte chunk")))
        .collect()
}

/// The owner channel end to end: a registered guest RAM window is imported
/// without copying, the device writes into the owner's own memory, and the
/// lease is retired and released before the dispatch returns.
///
/// This is the same call the production seam makes when a staged compute
/// buffer's bytes resolve to one registered window
/// (`runtime/compute_exec/vulkan.rs::owner_window_for_staged`). The
/// registration here comes from a test-owned allocation because only a booted
/// VM has a QEMU RAMBlock; the `HostRegion` type, the provider's checks and the
/// whole lease lifecycle are the shipped ones.
#[test]
fn a_registered_guest_window_is_imported_without_copying_and_released_after_completion() {
    let alignment = host_import_alignment().expect("the provider rail");
    assert!(
        alignment > 0,
        "this device must advertise VK_EXT_external_memory_host for the owner channel"
    );
    let page = usize::try_from(alignment).expect("alignment fits usize");
    let mut owner = AlignedBuffer::new(2 * page, page);
    // The kernel's 16 bytes are the first bytes of the registration's second
    // page, so the window is an import granule that the kernel only partly
    // touches — the shape a staged bind has, because the guest slice is widened
    // to the import granularity before any provider sees it. The window starts
    // on that granule because a no-copy import names the view's own pointer,
    // which the provider requires to be import-aligned.
    let window_offset = page;
    let head = 0usize;
    let input = input_words([1, 2, 3, 4]);
    // Sentinel every byte of the registration first. What the provider and the
    // rail must both leave alone is the rest of the window — the part of the
    // import granule the kernel's 16 bytes do not cover — so a writeback that
    // moved "the whole buffer" would be visible here.
    owner.as_mut_slice().fill(0xA5);
    owner.as_mut_slice()[window_offset + head..window_offset + head + input.len()]
        .copy_from_slice(&input);

    let import = 0x5eed_u64;
    provider_owner::register(Region {
        import,
        epoch: 1,
        host_pointer: owner.as_ptr() as usize,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x10_0000),
    })
    .expect("a page-aligned, page-sized registration is a legal provider region");

    // The registration is the provider's own type, field by field: nothing here
    // is a reims-side lookalike.
    let registered = provider_owner::registered(import).expect("the registration is held");
    assert_eq!(registered.host_pointer, owner.as_ptr() as usize);
    assert_eq!(registered.length, 2 * page as u64);
    assert_eq!(registered.page_size, alignment);
    assert!(
        !registered.lease_id.is_zero() && !registered.owner_epoch.is_zero(),
        "the provider refuses zero identities"
    );

    let air = fixture_air();
    let request = mul3add1_request();
    let window = Window {
        binding: 0,
        import,
        host_va: owner.as_ptr() as u64 + window_offset as u64,
        length: page as u64,
        head: head as u64,
        bytes_len: input.len() as u64,
    };

    for expected in [[4u32, 7, 10, 13], [13, 22, 31, 40]] {
        match submit_compute(&air, "apv_cs", &request, &[window]) {
            ComputeRailOutcome::ProviderCompleted(out) => {
                assert_eq!(out.writebacks.len(), 1);
                let writeback = &out.writebacks[0];
                assert_eq!(readback_words(&writeback.bytes), expected.to_vec());
                // The interval is the provider's own, in both coordinate
                // systems: allocation offset = where the window starts inside
                // the registration (+ the view head), staged offset = the head
                // itself. Nothing here assumes the writeback starts at zero.
                assert_eq!(
                    writeback.allocation_offset,
                    window_offset as u64 + head as u64
                );
                assert_eq!(writeback.offset, head as u64);
                assert_eq!(writeback.bytes.len(), input.len());
                assert_eq!(
                    writeback.allocation.is_zero(),
                    false,
                    "the writeback names the allocation the trace carried"
                );
            }
            ComputeRailOutcome::NotInNarrowClass(reason) => {
                panic!("the reviewed fixture is in the narrow class; refused: {reason}")
            }
            ComputeRailOutcome::ProviderDeclined(decline) => {
                panic!("the owner channel declined the reviewed fixture: {decline}")
            }
        }
        // The device wrote through the imported window in place: the owner's own
        // bytes carry the result, which a staged copy could not produce.
        assert_eq!(
            &owner.as_slice()[window_offset + head..window_offset + head + 16],
            expected
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<u8>>(),
            "the writeback must land in the owner mapping itself"
        );
        // Byte-exact partial write, at the provider's own granularity: the
        // kernel's 16 bytes moved, and nothing else inside the registered
        // window did — neither the first granule (outside the window) nor the
        // remaining 4064 bytes of the window's own granule.
        assert!(
            owner.as_slice()[..window_offset]
                .iter()
                .all(|byte| *byte == 0xA5),
            "the registration before the window must be untouched"
        );
        assert!(
            owner.as_slice()[window_offset + head + input.len()..]
                .iter()
                .all(|byte| *byte == 0xA5),
            "the rest of the window's granule must be untouched"
        );
    }
    // The loop above submitted twice over one registration. The second import
    // only succeeds because the first lease was retired and released at the end
    // of its own submission; the provider would have refused a still-held lease
    // (`lease_already_imported`).
    provider_owner::reset();
}

/// A lease from a previous device incarnation is refused by the provider, and
/// the refusal is a typed decline on reims' own vocabulary — never a silent
/// re-run on another rail.
#[test]
fn a_lease_from_a_previous_incarnation_is_refused_by_the_provider() {
    let executor = VulkanExecutor::new().expect("a Vulkan executor");
    let provider = VulkanComputeProvider::with_executor(executor).expect("provider");
    let current = provider.device_epoch().get();
    assert!(current > 0);
    // The previous incarnation's epoch when there is one, otherwise the next
    // one: either way it is not this device's, which is the whole question the
    // provider answers.
    let stale = if current > 1 {
        current - 1
    } else {
        current + 1
    };
    assert_ne!(stale, current, "the stale epoch must not be this device's");
    assert!(
        device_epoch().expect("the rail's provider epoch") > 0,
        "the shipped rail's own provider must exist"
    );

    let bytes = vec![0u8; 64];
    let request = Request::Staged(Staged {
        binding: 0,
        bytes: &bytes,
    });
    let decline = provider_owner::plan_with_epoch(&provider, stale, &[request])
        .expect_err("the provider refuses a lease that names another incarnation");
    assert_eq!(decline.slug(), "owner_lease_import");
    assert!(
        decline
            .fields()
            .iter()
            .any(|(key, value)| *key == "detail" && value.contains("lease_epoch_mismatch")),
        "the provider's own slug must be visible in the decline: {decline:?}"
    );

    // The compute rail maps the owner refusal onto its own typed decline, which
    // is what `submit_narrow` returns to the dispatch path: the caller declines
    // the submission and nothing falls back to the self-contained engine.
    let mapped = ProviderComputeDecline::Owner(decline);
    assert_eq!(mapped.slug(), "owner_lease_import");
    assert!(
        mapped.owner().contains("provider_owner"),
        "the wrapper must delegate the slug's owner to the owner rail: {}",
        mapped.owner()
    );
}
