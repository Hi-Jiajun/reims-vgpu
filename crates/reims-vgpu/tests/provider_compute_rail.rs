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

use metal_api_core::provider::{CompletionDisposition, CompletionToken, SubmissionId};
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
                // The payload the translator derives for this launch: four
                // threads in 4-wide groups, no region base, one threadgroup.
                // The rail verifies it against its own plan, so a placeholder
                // would keep the shape on the engine.
                push_constants: [4, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
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

/// The owner rail's ledger and registrations are process-global, so a test
/// that resets them (`provider_owner::reset()`, at the end of the tests that
/// register) would otherwise refuse a lease another test had in flight — the
/// two would be sharing one ledger by accident. Every test here that submits
/// through the rail or plans an owner lease takes this lock, which is what
/// makes them independent without also making them serial by construction.
static OWNER_RAIL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn owner_rail_guard() -> std::sync::MutexGuard<'static, ()> {
    OWNER_RAIL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn the_production_seam_submits_the_reviewed_fixture_through_the_canonical_provider() {
    let _guard = owner_rail_guard();
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
    let _guard = owner_rail_guard();
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

/// The request's stated payload offset must be the offset the canonical
/// provider's own reflection of the same AIR derives (`research/docs/26` §7):
/// one exact-thread dispatch writes every region's payload at one offset, and
/// the two rails read that number from two pinned translators. A request that
/// states another offset is refused by name — fail-closed, never dispatched on
/// either rail with the payload where the kernel does not read it.
#[test]
fn a_payload_offset_the_contract_does_not_state_is_refused_by_name() {
    let _guard = owner_rail_guard();
    let air = fixture_air();

    // Positive control on the same fixture: the offset the translator derives
    // completes and writes the kernel's result back, so the refusal below is
    // about the hacked offset rather than about the fixture.
    let valid = mul3add1_request();
    match submit_compute(&air, "apv_cs", &valid, &[]) {
        ComputeRailOutcome::ProviderCompleted(out) => {
            assert_eq!(
                readback_words(&out.writebacks[0].bytes),
                vec![4u32, 7, 10, 13]
            );
        }
        other => panic!("the reviewed fixture must complete: {other:?}"),
    }

    // The hack: the same payload is claimed to land at offset 16, where the
    // kernel does not read it. The provider's contract states 0, so the two
    // claims disagree and the dispatch is declined instead of executed.
    let mut hacked = mul3add1_request();
    let ComputeDispatch::Regions { push_offset, .. } = &mut hacked.dispatch else {
        panic!("the fixture is a regions dispatch");
    };
    *push_offset = 16;
    match submit_compute(&air, "apv_cs", &hacked, &[]) {
        ComputeRailOutcome::ProviderDeclined(decline) => {
            assert_eq!(
                decline,
                ProviderComputeDecline::PushOffsetMismatch {
                    request: 16,
                    contract: 0,
                },
                "the refusal names both the request's claim and the contract's offset"
            );
            assert_eq!(decline.slug(), "push_offset_mismatch");
        }
        ComputeRailOutcome::ProviderCompleted(_) => {
            panic!("a payload offset the contract does not state must not execute")
        }
        ComputeRailOutcome::NotInNarrowClass(reason) => {
            panic!("the offset agreement is an admission question, not a class one: {reason}")
        }
    }
}

/// S2, mirror direction: a request that stages a binding the canonical contract
/// does not name is the two pinned translators describing different kernels.
/// The shape keeps the reims engine — the same answer the forward rule gives —
/// instead of completing with the binding's bytes silently dropped (which the
/// seam could only notice after the provider had already run).
#[test]
fn a_staged_binding_the_canonical_contract_does_not_name_stays_on_the_engine() {
    let _guard = owner_rail_guard();
    let air = fixture_air();

    let mut extra = mul3add1_request();
    extra.storage_buffers.push(ComputeBufferResource {
        binding: 7,
        bytes: vec![0u8; 16],
        writable: true,
    });
    assert!(
        matches!(
            submit_compute(&air, "apv_cs", &extra, &[]),
            ComputeRailOutcome::NotInNarrowClass(_)
        ),
        "a staged binding the canonical contract does not name must keep the reims engine"
    );

    // The access half of the same rule: the two reflections must agree on
    // writability, or the readback set the seam counts would disagree with the
    // one the provider returned.
    let mut access = mul3add1_request();
    access.storage_buffers[0].writable = false;
    assert!(
        matches!(
            submit_compute(&air, "apv_cs", &access, &[]),
            ComputeRailOutcome::NotInNarrowClass(_)
        ),
        "reflections that disagree on writability must keep the reims engine"
    );
}

/// S2, fail-closed: an in-class shape the canonical provider itself refuses —
/// here a staged span shorter than the kernel's reachable footprint — is a
/// typed decline on reims' own vocabulary, never a silent re-run on the
/// self-contained engine.
#[test]
fn an_in_class_shape_the_provider_refuses_is_a_typed_decline() {
    let _guard = owner_rail_guard();
    let air = fixture_air();
    let mut short = mul3add1_request();
    // The kernel's four threads reach 4 × 4 bytes; staging one word makes the
    // canonical footprint proof refuse the trace before anything runs.
    short.storage_buffers[0].bytes.truncate(4);
    match submit_compute(&air, "apv_cs", &short, &[]) {
        ComputeRailOutcome::ProviderDeclined(decline) => {
            assert_eq!(decline.slug(), "provider_capability");
            let fields = decline.fields();
            assert!(
                fields
                    .iter()
                    .any(|(key, value)| *key == "step" && value == "trace_admission"),
                "the refusal names the provider step that refused: {fields:?}"
            );
            assert!(
                fields.iter().any(|(key, value)| *key == "detail"
                    && value.contains("buffer_footprint_exceeds_view")),
                "the provider's own slug rides along: {fields:?}"
            );
        }
        ComputeRailOutcome::ProviderCompleted(_) => panic!(
            "the provider completed a dispatch whose staged span is shorter than its footprint"
        ),
        ComputeRailOutcome::NotInNarrowClass(reason) => {
            panic!("an in-class refusal must not fall back to the engine: {reason}")
        }
    }
}

/// S3 end to end: the pipeline cache is keyed by `(AIR, entry)`. The same AIR
/// asked for a second entry must recompile (and here refuse), never be answered
/// by the first entry's cached pipeline.
#[test]
fn a_second_entry_over_the_same_air_does_not_hit_the_cached_pipeline() {
    let _guard = owner_rail_guard();
    let air = fixture_air();
    let request = mul3add1_request();
    assert!(
        matches!(
            submit_compute(&air, "apv_cs", &request, &[]),
            ComputeRailOutcome::ProviderCompleted(_)
        ),
        "the reviewed entry compiles and completes, and is now cached"
    );
    match submit_compute(&air, "apv_cs_second", &request, &[]) {
        ComputeRailOutcome::ProviderDeclined(decline) => {
            assert_eq!(decline.slug(), "provider_compile");
            let fields = decline.fields();
            assert!(
                fields
                    .iter()
                    .any(|(key, value)| *key == "step" && value == "pipeline_compile"),
                "the refusal names the compile step: {fields:?}"
            );
            assert!(
                fields
                    .iter()
                    .any(|(key, value)| *key == "detail" && value.contains("apv_cs_second")),
                "the provider's own entry-mismatch text rides along: {fields:?}"
            );
        }
        ComputeRailOutcome::ProviderCompleted(_) => {
            panic!("an AIR-only cache would have answered with the first entry's pipeline")
        }
        ComputeRailOutcome::NotInNarrowClass(reason) => {
            panic!("a compile refusal must be a typed decline, not an engine fallback: {reason}")
        }
    }
}

/// S1 end to end: an aborted settle cannot release a provider import the owner
/// ledger still holds. The completion names a token but is not retirement
/// evidence, so `settle` binds the lease and refuses; the abort that follows
/// must leave the import in the provider's own registry instead of releasing it
/// behind the ledger's back.
#[test]
fn an_aborted_settle_keeps_a_lease_the_owner_ledger_still_holds() {
    let _guard = owner_rail_guard();
    let executor = VulkanExecutor::new().expect("a Vulkan executor");
    let provider = VulkanComputeProvider::with_executor(executor).expect("provider");
    let bytes = vec![0xA5u8; 64];
    let plan = provider_owner::plan(
        &provider,
        &[Request::Staged(Staged {
            binding: 0,
            bytes: &bytes,
        })],
    )
    .expect("a staged binding imports as an owner lease");
    assert_eq!(
        provider.lease_registry().len(),
        1,
        "the import landed in the provider's own staged registry"
    );

    // `TimedOut` carries a token but is not retirement evidence: `settle` binds
    // the token to the lease, refuses the completion, and aborts.
    let token = CompletionToken {
        submission_id: SubmissionId::new(1),
        device_epoch: provider.device_epoch(),
    };
    let decline = plan
        .settle(&provider, CompletionDisposition::TimedOut { token })
        .expect_err("a timeout is not retirement evidence");
    assert_eq!(decline.slug(), "owner_completion_not_retiring");
    assert_eq!(
        provider.lease_registry().len(),
        1,
        "the lease is still bound in the owner ledger, so its import must still be imported"
    );

    // The paired direction on the same provider: a plan whose lease the ledger
    // gives up releases its import in the same step.
    let second = vec![0x5Au8; 64];
    let plan = provider_owner::plan(
        &provider,
        &[Request::Staged(Staged {
            binding: 0,
            bytes: &second,
        })],
    )
    .expect("a second staged import");
    assert_eq!(provider.lease_registry().len(), 2);
    plan.abort(&provider);
    assert_eq!(
        provider.lease_registry().len(),
        1,
        "an unbound lease's import is released with the lease itself"
    );
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

/// One request whose dispatch is exactly the translator's own decomposition of
/// `grid` threads in `local`-sized threadgroups: the shape the production seam
/// builds in `runtime/compute_exec/vulkan.rs::kernel_dispatch_launch`, plan and
/// payloads included. This is the stimulus the widened class admits — the rail
/// verifies the region list against the same planner, so a hand-written
/// decomposition would not do.
fn request_of_launch(grid: [u32; 3], local: [u32; 3], words: &[u32]) -> (ComputeRequest, usize) {
    let plan = metal2vulkan::reflect::KernelDispatch::ThreadsDynamic { offset: 0 }
        .plan(local, Some(grid))
        .expect("an exact-thread launch plans its regions");
    let input: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let request = ComputeRequest {
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
        storage_buffers: vec![ComputeBufferResource {
            binding: 0,
            bytes: input,
            writable: true,
        }],
        sampled_images: Vec::new(),
        samplers: Vec::new(),
        storage_images: Vec::new(),
    };
    (request, plan.regions.len())
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
    let _guard = owner_rail_guard();
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

/// A launch whose thread grid is not a multiple of its threadgroup is *one*
/// Metal `dispatchThreads` that the device tiles into several regions, and the
/// canonical trace spells it as one `ThreadsExact` pass carrying the launch's
/// own `(grid, local)`. The provider re-derives the regions, and the readback
/// proves the derivation is the request's: the slab's threads run at their
/// launch offsets (words 5 and 6), not renumbered from zero — which is exactly
/// what spelling the slab as its own pass would have produced.
#[test]
fn a_partial_threadgroup_launch_reaches_the_provider_as_one_threads_exact_pass() {
    let _guard = owner_rail_guard();
    let air = fixture_air();
    let (request, regions) = request_of_launch([6, 1, 1], [4, 1, 1], &[1, 2, 3, 4, 5, 6]);
    assert_eq!(
        regions, 2,
        "6 threads in 4-wide groups tile into two regions"
    );

    match submit_compute(&air, "apv_cs", &request, &[]) {
        ComputeRailOutcome::ProviderCompleted(out) => {
            assert_eq!(out.images.len(), 0, "the narrow class carries no images");
            assert_eq!(out.writebacks.len(), 1, "one writable binding readback");
            let writeback = &out.writebacks[0];
            assert_eq!(writeback.binding, 0);
            assert_eq!(writeback.offset, 0);
            assert_eq!(
                readback_words(&writeback.bytes),
                vec![4, 7, 10, 13, 16, 19],
                "the interior's four threads and the slab's two each ran at their \
                 launch offsets: the slab is words 5 and 6 (16 and 19), not a \
                 renumbered 4 and 7"
            );
        }
        ComputeRailOutcome::NotInNarrowClass(reason) => {
            panic!("a partial-threadgroup launch is in the widened class; refused: {reason}")
        }
        ComputeRailOutcome::ProviderDeclined(decline) => {
            panic!("the canonical provider declined the tiled launch: {decline}")
        }
    }
}

/// Submission order is the ordering the narrow class has — and the ordering it
/// can promise. The regions of one tiled launch are *one* Metal
/// `dispatchThreads`, and both rails say so: the engine's own compute path
/// documents that no barrier separates the regions and that its "threads have
/// no ordering among themselves", and the provider's executor loops the same
/// way inside a pass. So the falsifiable claim here is the one the guest can
/// actually depend on: a *later* submission reads the bytes an earlier one
/// wrote, tiled launch after tiled launch.
#[test]
fn a_second_tiled_submission_reads_the_first_one_s_writeback() {
    let _guard = owner_rail_guard();
    let air = fixture_air();
    let mut words = vec![1u32, 2, 3, 4, 5, 6];
    for expected in [vec![4u32, 7, 10, 13, 16, 19], vec![13, 22, 31, 40, 49, 58]] {
        let (request, regions) = request_of_launch([6, 1, 1], [4, 1, 1], &words);
        assert_eq!(regions, 2, "each submission tiles the same way");
        match submit_compute(&air, "apv_cs", &request, &[]) {
            ComputeRailOutcome::ProviderCompleted(out) => {
                words = readback_words(&out.writebacks[0].bytes);
                assert_eq!(
                    words, expected,
                    "the second submission's input is the first one's writeback"
                );
            }
            ComputeRailOutcome::NotInNarrowClass(reason) => {
                panic!("a partial-threadgroup launch is in the widened class; refused: {reason}")
            }
            ComputeRailOutcome::ProviderDeclined(decline) => {
                panic!("the canonical provider declined the tiled launch: {decline}")
            }
        }
    }
}

/// The counterexample half of the widening: a region list the translator would
/// not derive from one exact-thread launch keeps the engine — the whole
/// launch, not just the offending region — and the provider is never handed
/// the shape.
///
/// The window is the evidence that "never handed" is causal rather than
/// bookkeeping: the same registration and the same request submit a valid
/// tiled launch that the device writes *in place* (the borrowed channel), while
/// the corrupted request leaves every byte of that owner memory alone even
/// though a provider that had run would have written into it.
#[test]
fn a_region_list_the_translator_would_not_derive_never_reaches_the_provider() {
    let _guard = owner_rail_guard();
    let alignment = host_import_alignment().expect("the provider rail");
    assert!(
        alignment > 0,
        "this device must advertise VK_EXT_external_memory_host for the owner channel"
    );
    let page = usize::try_from(alignment).expect("alignment fits usize");
    let mut owner = AlignedBuffer::new(page, page);
    let input = [1u32, 2, 3, 4, 5, 6]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<u8>>();
    owner.as_mut_slice().fill(0xA5);
    owner.as_mut_slice()[..input.len()].copy_from_slice(&input);

    let import = 0x5eed_6a11_u64;
    provider_owner::register(Region {
        import,
        epoch: 1,
        host_pointer: owner.as_ptr() as usize,
        length: page as u64,
        page_size: alignment,
        gpa_base: Some(0x20_0000),
    })
    .expect("a page-aligned, page-sized registration is a legal provider region");
    let window = Window {
        binding: 0,
        import,
        host_va: owner.as_ptr() as u64,
        length: page as u64,
        head: 0,
        bytes_len: input.len() as u64,
    };
    let air = fixture_air();

    // Positive control on this very registration: the valid tiled launch does
    // write through the window, so "the bytes are still the input" below is
    // evidence about the rail and not about an inert setup.
    let (valid, _) = request_of_launch([6, 1, 1], [4, 1, 1], &[1, 2, 3, 4, 5, 6]);
    match submit_compute(&air, "apv_cs", &valid, &[window]) {
        ComputeRailOutcome::ProviderCompleted(out) => assert_eq!(
            readback_words(&out.writebacks[0].bytes),
            vec![4u32, 7, 10, 13, 16, 19]
        ),
        ComputeRailOutcome::NotInNarrowClass(reason) => {
            panic!("a partial-threadgroup launch is in the widened class; refused: {reason}")
        }
        ComputeRailOutcome::ProviderDeclined(decline) => {
            panic!("the canonical provider declined the tiled launch: {decline}")
        }
    }
    assert_eq!(
        readback_words(&owner.as_slice()[..input.len()]),
        vec![4u32, 7, 10, 13, 16, 19],
        "the owner mapping itself carries the result before the corruption"
    );

    // Restore the input, then corrupt the decomposition: the slab's payload
    // claims its threads start at zero, which is a list the planner would not
    // derive for this launch.
    owner.as_mut_slice()[..input.len()].copy_from_slice(&input);
    let (mut corrupted, _) = request_of_launch([6, 1, 1], [4, 1, 1], &[1, 2, 3, 4, 5, 6]);
    let ComputeDispatch::Regions { regions, .. } = &mut corrupted.dispatch else {
        panic!("the tiled launch is a regions dispatch");
    };
    regions[1].push_constants[3] = 0;
    match submit_compute(&air, "apv_cs", &corrupted, &[window]) {
        ComputeRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason,
            "a region list the translator would not derive from one exact-thread launch \
             stays on the self-contained engine",
            "the whole launch keeps the engine"
        ),
        ComputeRailOutcome::ProviderCompleted(_) => {
            panic!("a region list outside the class must not reach the provider")
        }
        ComputeRailOutcome::ProviderDeclined(decline) => {
            panic!("out-of-class shapes fall back to the engine, not to a decline: {decline}")
        }
    }
    assert_eq!(
        &owner.as_slice()[..input.len()],
        input.as_slice(),
        "the provider never ran over the window: every byte is the input again"
    );
    assert!(
        owner.as_slice()[input.len()..]
            .iter()
            .all(|byte| *byte == 0xA5),
        "and nothing else in the registration moved either"
    );
    provider_owner::reset();
}

/// A lease from a previous device incarnation is refused by the provider, and
/// the refusal is a typed decline on reims' own vocabulary — never a silent
/// re-run on another rail.
#[test]
fn a_lease_from_a_previous_incarnation_is_refused_by_the_provider() {
    let _guard = owner_rail_guard();
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
