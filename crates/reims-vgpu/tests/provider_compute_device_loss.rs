//! Gate 2 error path: the rail's typed refusal mapping and the device-loss
//! teardown, driven by the emulator's own driver-loss test injection.
//!
//! One registered guest RAM window goes through the shipped
//! `backend::provider_compute` seam twice: once to prove the rail completes,
//! and once with `VK_ERROR_DEVICE_LOST` substituted at the provider's own queue
//! boundary. What the second call returns and what it leaves behind are both
//! asserted: the refusal is `provider_device_lost` carrying the provider's
//! original slug text and the census of the owner-ledger teardown, later work is
//! refused fail-closed with `provider_unavailable`, and the same rail rebuilds
//! in place (`v64`'s `rebuild_after_device_loss`) so a re-registered window can
//! submit again.
//!
//! This test lives in its own binary because it takes the process-global rail
//! through a device loss; the other rail tests need that provider usable.

#![cfg(feature = "provider-compute")]

use metal_api_vulkan::DeviceLossPoint;
use reims_vgpu::backend::provider_compute::{
    device_epoch, host_import_alignment, inject_driver_device_loss_for_test,
    recover_after_device_loss, submit_compute, ComputeRailOutcome, ProviderComputeDecline,
};
use reims_vgpu::backend::provider_owner::{self, Region, Window};
use reims_vgpu::backend::vulkan::engine::{
    ComputeBufferResource, ComputeDispatch, ComputeDispatchRegion, ComputeRequest,
};
use reims_vgpu::observe::Decline as _;
use reims_vgpu_observe::Emit;
use std::path::PathBuf;

fn fixture_air() -> Vec<u8> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compute_mul3add1.mtlb");
    let mtlb = std::fs::read(&path).expect("compute_mul3add1.mtlb fixture");
    reims_vgpu::runtime::mtlb::extract_air(&mtlb)
        .expect("wrapped AIR bitcode")
        .to_vec()
}

fn mul3add1_request() -> ComputeRequest {
    let input: Vec<u8> = [1u32, 2, 3, 4]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    ComputeRequest {
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

/// Page-aligned owner memory for a `VK_EXT_external_memory_host` import — the
/// same test-owned stand-in the owner-rail test uses, because only a booted VM
/// has a QEMU RAMBlock mapping.
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

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) }
    }
}

fn readback_words(out: &[u8]) -> Vec<u32> {
    out.chunks(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("4-byte chunk")))
        .collect()
}

fn decline_for(air: &[u8], request: &ComputeRequest, windows: &[Window]) -> ProviderComputeDecline {
    match submit_compute(air, "apv_cs", request, windows) {
        ComputeRailOutcome::ProviderDeclined(decline) => decline,
        ComputeRailOutcome::ProviderCompleted(_) => {
            panic!("the rail completed a submission the test expected to be refused")
        }
        ComputeRailOutcome::NotInNarrowClass(reason) => {
            panic!("the reviewed fixture is in the narrow class; refused: {reason}")
        }
    }
}

#[test]
fn a_driver_reported_device_loss_is_a_typed_refusal_with_an_executed_teardown() {
    let alignment = host_import_alignment().expect("the provider rail");
    assert!(
        alignment > 0,
        "this device must advertise VK_EXT_external_memory_host for the owner channel"
    );
    let page = usize::try_from(alignment).expect("alignment fits usize");
    let mut owner = AlignedBuffer::new(2 * page, page);
    let input: Vec<u8> = [1u32, 2, 3, 4]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    owner.as_mut_slice()[..input.len()].copy_from_slice(&input);

    let import = 0x9e11_u64;
    let epoch = device_epoch().expect("the rail's provider epoch");
    provider_owner::register(Region {
        import,
        epoch,
        host_pointer: owner.as_ptr() as usize,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x20_0000),
    })
    .expect("a page-aligned registration is a legal provider region");
    let window = Window {
        binding: 0,
        import,
        host_va: owner.as_ptr() as u64,
        length: page as u64,
        head: 0,
        bytes_len: input.len() as u64,
    };

    let air = fixture_air();
    let request = mul3add1_request();

    // 1. The rail carries the reviewed fixture before the injection, so the
    //    loss below is the only difference between the two submissions.
    match submit_compute(&air, "apv_cs", &request, &[window]) {
        ComputeRailOutcome::ProviderCompleted(out) => {
            assert_eq!(readback_words(&out.writebacks[0].bytes), vec![4, 7, 10, 13]);
        }
        other => panic!("the owner channel must complete before the injection: {other:?}"),
    }

    // 2. Substitute the provider's own next driver answer with
    //    VK_ERROR_DEVICE_LOST and submit the same trace.
    inject_driver_device_loss_for_test(DeviceLossPoint::Submit)
        .expect("the rail's own provider takes the injection");
    let decline = decline_for(&air, &request, &[window]);
    assert_eq!(
        decline.slug(),
        "provider_device_lost",
        "the normalized class is the rail's own slug: {decline:?}"
    );
    let fields = decline.fields();
    let field = |key: &str| {
        fields
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| value.as_str())
            .unwrap_or_default()
    };
    assert_eq!(field("class"), "device_lost");
    assert_eq!(field("step"), "submission");
    assert!(
        field("detail").contains("vulkan-queue-submit"),
        "the provider's own boundary slug must ride along: {fields:?}"
    );
    assert_eq!(
        field("teardown_leases"),
        "1",
        "the plan's borrowed lease was released by the teardown: {fields:?}"
    );
    assert_eq!(field("teardown_windows"), "1");

    // 3. The teardown is executed state, not a mapping table: the ledger ran
    //    once, and no registration survives the dead incarnation.
    assert_eq!(provider_owner::device_loss_teardowns(), 1);
    assert_eq!(
        provider_owner::last_device_loss_teardown().leases,
        1,
        "the recorded census is the lease the teardown found"
    );
    assert_eq!(provider_owner::registered_regions(), 0);
    assert!(provider_owner::registered(import).is_none());

    // 4. Later in-class work is refused fail-closed — `provider_unavailable`,
    //    never a silent re-run on the self-contained engine — and the refusal
    //    names the terminal health in the provider's own vocabulary.
    let declined = decline_for(&air, &request, &[window]);
    assert_eq!(declined.slug(), "provider_unavailable");
    let fields = declined.fields();
    assert!(fields
        .iter()
        .any(|(key, value)| *key == "health" && value == "device_lost"));
    assert!(fields
        .iter()
        .any(|(key, value)| *key == "teardown_leases" && value == "0"));
    assert_eq!(
        provider_owner::device_loss_teardowns(),
        2,
        "the gate runs the idempotent teardown again on an already-clean ledger"
    );

    // 5. The always-on line is the one the seam emits, and it carries both the
    //    reims slug and the provider's original text.
    Emit::decline("compute_provider", &declined)
        .field("pipe", 26u32)
        .fail_once(0x26_01);
    let log = std::fs::read_to_string(reims_vgpu_observe::fail_log_path()).expect("fail log");
    assert!(
        log.lines().any(|line| {
            line.contains("compute_provider reason=provider_unavailable")
                && line.contains("health=device_lost")
        }),
        "the refusal line must name the reims slug and the terminal health"
    );
    Emit::decline("compute_provider", &decline)
        .field("pipe", 26u32)
        .fail_once(0x26_02);
    let log = std::fs::read_to_string(reims_vgpu_observe::fail_log_path()).expect("fail log");
    assert!(
        log.lines().any(|line| {
            line.contains("compute_provider reason=provider_device_lost")
                && line.contains("detail=vulkan-queue-submit")
        }),
        "the refusal line must carry the provider's own boundary slug"
    );

    // 6. Recovery in the same process: the lost provider rebuilds in place
    //    (`v64`), the epoch advances, and a re-registered window submits the
    //    same trace on the fresh device. The owner bytes were written in place
    //    by the first submission, so this pass computes over them.
    recover_after_device_loss().expect("a lost provider rebuilds in place");
    let recovered_epoch = device_epoch().expect("the rebuilt provider's epoch");
    assert!(
        recovered_epoch > epoch,
        "the rebuild must advance the device epoch ({epoch} -> {recovered_epoch})"
    );
    let second_import = import + 1;
    provider_owner::register(Region {
        import: second_import,
        epoch: recovered_epoch,
        host_pointer: owner.as_ptr() as usize,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x20_0000),
    })
    .expect("the recovered rail re-registers guest RAM");
    let window = Window {
        binding: 0,
        import: second_import,
        host_va: owner.as_ptr() as u64,
        length: page as u64,
        head: 0,
        bytes_len: input.len() as u64,
    };
    match submit_compute(&air, "apv_cs", &request, &[window]) {
        ComputeRailOutcome::ProviderCompleted(out) => {
            assert_eq!(
                readback_words(&out.writebacks[0].bytes),
                vec![13, 22, 31, 40],
                "the same trace resubmits on the rebuilt device"
            );
        }
        other => panic!("the rebuilt rail must carry the same trace: {other:?}"),
    }
    provider_owner::reset();
}
