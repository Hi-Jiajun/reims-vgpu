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
    device_epoch, host_import_alignment, submit_compute, ComputeRailOutcome, ModuleSampler,
    ProviderComputeDecline,
};
use reims_vgpu::backend::provider_owner::{self, Region, Request, Staged, StagedBytes, Window};
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
    match submit_compute(&air, "apv_cs", &request, &[], &[]) {
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
        submit_compute(&air, "apv_cs", &whole_workgroups, &[], &[]),
        ComputeRailOutcome::NotInNarrowClass(_)
    ));

    let mut multi_region = mul3add1_request();
    if let ComputeDispatch::Regions { regions, .. } = &mut multi_region.dispatch {
        regions.push(regions[0]);
    }
    assert!(matches!(
        submit_compute(&air, "apv_cs", &multi_region, &[], &[]),
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
    match submit_compute(&air, "apv_cs", &valid, &[], &[]) {
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
    match submit_compute(&air, "apv_cs", &hacked, &[], &[]) {
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
            submit_compute(&air, "apv_cs", &extra, &[], &[]),
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
            submit_compute(&air, "apv_cs", &access, &[], &[]),
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
    match submit_compute(&air, "apv_cs", &short, &[], &[]) {
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
            submit_compute(&air, "apv_cs", &request, &[], &[]),
            ComputeRailOutcome::ProviderCompleted(_)
        ),
        "the reviewed entry compiles and completes, and is now cached"
    );
    match submit_compute(&air, "apv_cs_second", &request, &[], &[]) {
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
        &mut [Request::Staged(Staged {
            binding: 0,
            bytes: StagedBytes::Borrowed(&bytes),
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
        &mut [Request::Staged(Staged {
            binding: 0,
            bytes: StagedBytes::Borrowed(&second),
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
        match submit_compute(&air, "apv_cs", &request, &[], &[window]) {
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

    match submit_compute(&air, "apv_cs", &request, &[], &[]) {
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
        match submit_compute(&air, "apv_cs", &request, &[], &[]) {
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
    match submit_compute(&air, "apv_cs", &valid, &[], &[window]) {
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
    match submit_compute(&air, "apv_cs", &corrupted, &[], &[window]) {
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
        bytes: StagedBytes::Borrowed(&bytes),
    });
    let decline = provider_owner::plan_with_epoch(&provider, stale, &mut [request])
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

// ---------------------------------------------------------------------------
// R9l: the compute texture half (`research/docs/26` §21.3, C1/C1b/C1c)
// ---------------------------------------------------------------------------

/// The process-global engine, the second rail these tests execute the same
/// request on. Its resets serialize beside the owner ledger's for the same
/// reason: both are process-global, and a parity reading is only meaningful
/// when each call starts from a fresh context.
///
/// The lock order is **owner → engine**. The engine's device creation runs the
/// production recreate step, which resets the runtime's guest-RAM map and — with
/// this feature on — the provider-owner rail's ledger
/// (`backend/vulkan/engine/context.rs`): a lease another test had in flight
/// would be dropped out of the ledger underneath it. Holding the owner guard
/// across the engine calls is what makes "no lease is in flight while the device
/// comes up" true for this binary.
static ENGINE_RAIL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The device every engine call in this file is made against, with the engine
/// reset so a warm cache from a previous test is never part of a reading.
fn engine_rail_guard() -> std::sync::MutexGuard<'static, ()> {
    let guard = ENGINE_RAIL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    reims_vgpu::backend::vulkan::engine::test_reset_engine(engine_device());
    guard
}

fn engine_device() -> &'static reims_vgpu::model::DeviceState {
    static DEVICE: std::sync::OnceLock<reims_vgpu::model::DeviceState> = std::sync::OnceLock::new();
    DEVICE.get_or_init(|| {
        reims_vgpu::model::DeviceState::new(
            reims_vgpu::model::DeviceId(1),
            reims_vgpu_protocol::gva::PAGE_SHIFT_X86,
        )
    })
}

/// The reviewed compute texture fixture's AIR: the C1b kernel body assembled
/// from the `.ll` beside it (`tests/fixtures/air/README.md`), carved out of its
/// bitcode wrapper the way the production seam carves a guest's MTLB
/// (`runtime::mtlb::extract_air`) — so what both translators read here is the
/// same bytes a guest's module would hand them.
fn texture_air() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/air/sample_texture_2d_nearest_clamp.air");
    let raw = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    reims_vgpu::runtime::mtlb::extract_air(&raw)
        .unwrap_or_else(|decline| panic!("{}: {decline}", path.display()))
        .to_vec()
}

/// The translated module beside that AIR, through the same
/// `translate_cached_kernel_reflected` the production seam's caller runs: the
/// SPIR-V the engine executes and the reflection both sampler lists are built
/// from are one translation rather than two spellings of it.
fn texture_module(air: &[u8]) -> std::sync::Arc<reims_vgpu::runtime::m2v_cache::CachedShader> {
    reims_vgpu::runtime::m2v_cache::translate_cached_kernel_reflected(air, [1, 1, 1], 0x9e_1c)
        .expect("the compute texture fixture translates")
}

/// One 4x4 `R32Float` texture whose every row is `values`, tightly packed — the
/// rows the fixture's two sample points read.
fn sampled_rows(values: [f32; 4]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 * 4 * 4);
    for _row in 0..4 {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// One dispatch of the fixture as the production seam's caller builds it: the
/// module's own sampler descriptors (its AIR constexpr sampler, through
/// `runtime::draw::vulkan::reflected_static_sampler_resource`) beside the one
/// sampled image the kernel reads, with the module's own statement of that
/// sampler beside them.
struct TextureStimulus {
    request: ComputeRequest,
    module_samplers: Vec<ModuleSampler>,
}

fn texture_stimulus(
    module: &reims_vgpu::runtime::m2v_cache::CachedShader,
    rows: [f32; 4],
) -> TextureStimulus {
    use reims_vgpu::backend::vulkan::engine::{
        ComputeSampledImageResource, ComputeSampledSource, SamplerResource, StorageImageFormat,
    };
    use reims_vgpu::runtime::draw::vulkan::reflected_static_sampler_resource;
    use reims_vgpu::runtime::spirv_bind::{reflected_sampler_descriptors, TEXTURE_BINDING_BASE};

    let plan = metal2vulkan::reflect::KernelDispatch::ThreadsDynamic { offset: 0 }
        .plan([1, 1, 1], Some([1, 1, 1]))
        .expect("one thread in one threadgroup plans");
    let descriptors = reflected_sampler_descriptors(&module.reflection, false);
    let module_samplers: Vec<ModuleSampler> = descriptors
        .iter()
        .filter_map(|descriptor| {
            descriptor.static_state.map(|state| ModuleSampler {
                binding: descriptor.binding,
                state,
            })
        })
        .collect();
    let samplers: Vec<SamplerResource> = descriptors
        .iter()
        .filter_map(|descriptor| {
            descriptor.static_state.map(|state| {
                reflected_static_sampler_resource("kernel", descriptor.binding, state)
                    .expect("the reviewed AIR state builds a sampler resource")
            })
        })
        .collect();
    TextureStimulus {
        request: ComputeRequest {
            spirv: module.words.as_ref().clone(),
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
                bytes: vec![0u8; 8],
                writable: true,
            }],
            sampled_images: vec![ComputeSampledImageResource {
                // The engine's own numbering: the translator's texture band is
                // `TEXTURE_BINDING_BASE + Metal index`.
                binding: TEXTURE_BINDING_BASE,
                array_element: 0,
                descriptor_count: 1,
                format: StorageImageFormat::R32Float,
                width: 4,
                height: 4,
                mip_levels: 1,
                source: ComputeSampledSource::Bytes(sampled_rows(rows)),
            }],
            samplers,
            storage_images: Vec::new(),
        },
        module_samplers,
    }
}

/// The two `f32` readings the fixture's store lands, from one provider output.
fn provider_readings(
    out: &reims_vgpu::backend::provider_compute::ProviderComputeOutput,
) -> [f32; 2] {
    let bytes = &out.writebacks[0].bytes;
    [
        f32::from_le_bytes(bytes[0..4].try_into().expect("first reading")),
        f32::from_le_bytes(bytes[4..8].try_into().expect("second reading")),
    ]
}

/// The two `f32` readings the engine's own output carries.
fn engine_readings(out: &reims_vgpu::backend::vulkan::engine::ComputeOutput) -> [f32; 2] {
    let bytes = &out.buffers[0].bytes;
    [
        f32::from_le_bytes(bytes[0..4].try_into().expect("first reading")),
        f32::from_le_bytes(bytes[4..8].try_into().expect("second reading")),
    ]
}

/// R9l end to end: the one compute texture shape this class admits — a single
/// `R32Float` texture sampled through the module's own AIR constexpr sampler —
/// leaves for the provider, and the engine executes the very same request to
/// the same bytes.
///
/// Two texture payloads are read, not one: the second row set moves both
/// readings, so the agreement is about the sampled bytes rather than about two
/// rails that both return a constant.
#[test]
fn a_compute_texture_pass_reads_the_same_bytes_on_both_rails() {
    // Owner first, engine second (see `ENGINE_RAIL`): the engine's device
    // creation resets the owner rail's process-global ledger, and no other test
    // may hold a lease across it.
    let _owner = owner_rail_guard();
    let _engine = engine_rail_guard();
    let air = texture_air();
    let module = texture_module(&air);

    // Row [0, 4, 8, 12]: u = 1.375 clamps to texel 3 (12) and the second sample
    // point lands in texel 1 (4). Row [12, 8, 4, 0]: the same two points read
    // 0 and 8.
    let cases = [
        ([0.0_f32, 4.0, 8.0, 12.0], [12.0_f32, 4.0]),
        ([12.0_f32, 8.0, 4.0, 0.0], [0.0_f32, 8.0]),
    ];
    for (rows, expected) in cases {
        let stimulus = texture_stimulus(&module, rows);
        let provider = match submit_compute(
            &air,
            "sample_texture_2d",
            &stimulus.request,
            &stimulus.module_samplers,
            &[],
        ) {
            ComputeRailOutcome::ProviderCompleted(out) => provider_readings(&out),
            ComputeRailOutcome::NotInNarrowClass(reason) => {
                panic!("the reviewed compute texture shape is in class; refused: {reason}")
            }
            ComputeRailOutcome::ProviderDeclined(decline) => {
                panic!("the canonical provider declined the reviewed texture pass: {decline}")
            }
        };
        let engine = reims_vgpu::backend::vulkan::engine::execute_compute_request(
            engine_device(),
            &stimulus.request,
        )
        .expect("the engine executes the same request");
        let engine = engine_readings(&engine);
        eprintln!("R9l texture rows {rows:?}: provider {provider:?} engine {engine:?}");
        assert_eq!(
            provider, expected,
            "the provider samples the module's own state against these rows"
        );
        assert_eq!(
            engine, provider,
            "the two rails must land byte-identical readings for one request"
        );
    }
}

/// The shapes beside the admitted one, every one of them named and none of them
/// widened (`research/docs/26` §21.3's boundary list): a second sampled image, a
/// mip chain, an array slice, a descriptor array, a resident source, a format
/// the compute texture face does not carry, bytes that are not the view's
/// extent, a descriptor binding outside the translator's texture band, a
/// sampler the module does not carry itself, a sampler descriptor whose state
/// is not the module's own, a module whose constexpr sampler the request does
/// not restate, and a module that carries two.
#[test]
fn the_shapes_beside_the_reviewed_texture_stay_on_the_engine_by_name() {
    let _guard = owner_rail_guard();
    let air = texture_air();
    let module = texture_module(&air);
    let reviewed = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    assert_eq!(
        reviewed.module_samplers.len(),
        1,
        "the fixture carries exactly one AIR constexpr sampler"
    );

    // Positive control: the reviewed shape itself is in class, so every refusal
    // below is about the one fact that was moved.
    match submit_compute(
        &air,
        "sample_texture_2d",
        &reviewed.request,
        &reviewed.module_samplers,
        &[],
    ) {
        ComputeRailOutcome::ProviderCompleted(out) => {
            assert_eq!(provider_readings(&out), [12.0, 4.0]);
        }
        other => panic!("the reviewed texture shape must complete: {other:?}"),
    }

    let reason_of =
        |request: &ComputeRequest, module_samplers: &[ModuleSampler]| match submit_compute(
            &air,
            "sample_texture_2d",
            request,
            module_samplers,
            &[],
        ) {
            ComputeRailOutcome::NotInNarrowClass(reason) => {
                eprintln!("R9l out-of-class: {reason}");
                reason
            }
            ComputeRailOutcome::ProviderCompleted(_) => {
                panic!("a shape outside the class must not reach the provider")
            }
            ComputeRailOutcome::ProviderDeclined(decline) => {
                panic!("out-of-class shapes fall back to the engine, not to a decline: {decline}")
            }
        };

    let mut two_textures = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    let mut second = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0])
        .request
        .sampled_images
        .pop()
        .expect("the stimulus stages one texture");
    second.binding += 1;
    two_textures.request.sampled_images.push(second);
    assert!(
        reason_of(&two_textures.request, &two_textures.module_samplers)
            .contains("more than one sampled image")
    );

    let mut mip_chain = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    mip_chain.request.sampled_images[0].mip_levels = 2;
    assert!(reason_of(&mip_chain.request, &mip_chain.module_samplers).contains("mip chain"));

    let mut array_slice = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    array_slice.request.sampled_images[0].array_element = 1;
    assert!(
        reason_of(&array_slice.request, &array_slice.module_samplers)
            .contains("one element of an array")
    );

    let mut descriptor_array = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    descriptor_array.request.sampled_images[0].descriptor_count = 2;
    assert!(
        reason_of(&descriptor_array.request, &descriptor_array.module_samplers)
            .contains("array of descriptors")
    );

    let mut resident = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    resident.request.sampled_images[0].source =
        reims_vgpu::backend::vulkan::engine::ComputeSampledSource::ResidentCopy(
            reims_vgpu::backend::vulkan::engine::ComputeResidentSampleBind {
                identity: reims_vgpu::model::ComputeStorageResidencyKey {
                    mapping_id: 3,
                    map_generation: 1,
                    surface_offset: 0,
                    surface_bpr: 16,
                    span_end: 64,
                    width: 4,
                    height: 4,
                    pixel_format: 0,
                    texture_ref: 1,
                },
                generation: 1,
            },
        );
    assert!(reason_of(&resident.request, &resident.module_samplers).contains("resident"));

    let mut foreign_format = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    foreign_format.request.sampled_images[0].format =
        reims_vgpu::backend::vulkan::engine::StorageImageFormat::Rgba8Unorm;
    assert!(
        reason_of(&foreign_format.request, &foreign_format.module_samplers)
            .contains("format the canonical compute texture face does not carry")
    );

    let mut short_bytes = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    if let reims_vgpu::backend::vulkan::engine::ComputeSampledSource::Bytes(bytes) =
        &mut short_bytes.request.sampled_images[0].source
    {
        bytes.truncate(32);
    }
    assert!(
        reason_of(&short_bytes.request, &short_bytes.module_samplers)
            .contains("tightly packed extent")
    );

    let mut outside_band = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    outside_band.request.sampled_images[0].binding = 3;
    assert!(
        reason_of(&outside_band.request, &outside_band.module_samplers)
            .contains("not in the translator's texture band")
    );

    // A sampler the module does not carry as its own constexpr state: the
    // guest-bound (`[[sampler(n)]]`) shape, in the same descriptor band.
    let mut foreign_sampler = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    let binding = foreign_sampler.module_samplers[0].binding;
    foreign_sampler.request.samplers =
        vec![reims_vgpu::backend::vulkan::engine::SamplerResource::normalized_default(binding + 1)];
    assert!(
        reason_of(&foreign_sampler.request, &foreign_sampler.module_samplers)
            .contains("does not carry as its own AIR constexpr state")
    );

    // A runtime sampler slot the runtime filled with its neutral default: the
    // state is in the reviewed family, but the module carries no constexpr
    // sampler at all, so there is nothing for the canonical contract to state.
    let mut runtime_sampler = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    runtime_sampler.request.samplers =
        vec![reims_vgpu::backend::vulkan::engine::SamplerResource::normalized_default(binding)];
    assert!(reason_of(&runtime_sampler.request, &[]).contains("does not carry as its own"));

    // A sampler descriptor whose state is not the module's own: the reviewed
    // binding, a filter the AIR does not carry.
    let mut drifted = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    drifted.request.samplers[0].min_filter =
        reims_vgpu_core::sampler::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR;
    drifted.request.samplers[0].mag_filter =
        reims_vgpu_core::sampler::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR;
    assert!(reason_of(&drifted.request, &drifted.module_samplers)
        .contains("not the state the module's own AIR constexpr sampler carries"));

    // A module whose constexpr sampler the request does not restate.
    let mut withheld = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    withheld.request.samplers.clear();
    assert!(reason_of(&withheld.request, &withheld.module_samplers)
        .contains("the request does not restate"));

    // A module that carries two constexpr samplers.
    let mut doubled = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);
    let mut second_sampler = doubled.module_samplers[0];
    second_sampler.binding += 1;
    doubled.module_samplers.push(second_sampler);
    assert!(reason_of(&doubled.request, &doubled.module_samplers)
        .contains("more than one AIR constexpr sampler"));
}

/// R9l, the fail-closed arm: a declaration that moves away from the module the
/// provider compiled is refused by name, never executed with the other state.
///
/// The two readings are injected the way R9j injected a capability answer: the
/// caller states the module's sampler as linear + clamp — and builds its request
/// descriptor that way too, which is what a drifted translator pin would
/// produce — while the AIR the provider compiles still carries nearest + clamp.
/// The seam's own comparison is what refuses, and its fields name both halves.
#[test]
fn a_declaration_that_moves_away_from_the_modules_sampler_is_refused_by_name() {
    let _guard = owner_rail_guard();
    let air = texture_air();
    let module = texture_module(&air);

    // The fixture's own AIR state is nearest filtering; the foreign reading
    // moves that one word, in both places the caller states it.
    let mut foreign = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]).module_samplers[0];
    assert_eq!(
        reims_vgpu::runtime::draw::vulkan::reflected_static_sampler_resource(
            "kernel",
            foreign.binding,
            foreign.state
        )
        .expect("the module's own state builds")
        .min_filter,
        reims_vgpu_core::sampler::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
        "the fixture's own AIR state is nearest filtering"
    );
    foreign.state.min_filter = metal2vulkan::reflect::SamplerFilter::Linear;
    foreign.state.mag_filter = metal2vulkan::reflect::SamplerFilter::Linear;
    let mut request = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]).request;
    request.samplers[0].min_filter = reims_vgpu_core::sampler::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR;
    request.samplers[0].mag_filter = reims_vgpu_core::sampler::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR;

    match submit_compute(&air, "sample_texture_2d", &request, &[foreign], &[]) {
        ComputeRailOutcome::ProviderDeclined(decline) => {
            eprintln!("R9l sampler mismatch decline: {decline}");
            assert_eq!(decline.slug(), "compute_texture_sampler_mismatch");
            let fields = decline.fields();
            let field = |name: &str| {
                fields
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| value.clone())
                    .unwrap_or_else(|| panic!("the refusal carries {name}: {fields:?}"))
            };
            assert_eq!(field("binding"), "0");
            assert_eq!(field("request_filter"), "Linear");
            assert_eq!(field("request_address"), "ClampToEdge");
            assert_eq!(field("contract_filter"), "Nearest");
            assert_eq!(field("contract_address"), "ClampToEdge");
        }
        ComputeRailOutcome::ProviderCompleted(_) => {
            panic!("a declaration that does not restate the module must not execute")
        }
        ComputeRailOutcome::NotInNarrowClass(reason) => panic!(
            "the two readings disagreeing is an admission question, not a class one: {reason}"
        ),
    }
}

/// R9l, the wire half: the declaration this rail states is carried by an `MCC1`
/// frame, the provider's own decoder reads it back, and the provider's own
/// admission answers for what it read.
#[test]
fn the_texture_declaration_crosses_the_wire_and_the_provider_reads_it_back() {
    let _guard = owner_rail_guard();
    use metal_api_core::provider::ComputeProvider as _;
    use reims_vgpu::backend::provider_wire;

    let air = texture_air();
    let module = texture_module(&air);
    let stimulus = texture_stimulus(&module, [0.0, 4.0, 8.0, 12.0]);

    provider_wire::capture_submission_frames(true);
    let frames_before = provider_wire::wire_counts();
    match submit_compute(
        &air,
        "sample_texture_2d",
        &stimulus.request,
        &stimulus.module_samplers,
        &[],
    ) {
        ComputeRailOutcome::ProviderCompleted(_) => (),
        other => panic!("the reviewed texture shape is in class: {other:?}"),
    }
    let frames = provider_wire::captured_submission_frames();
    let counts = provider_wire::wire_counts();
    provider_wire::capture_submission_frames(false);
    assert_eq!(
        counts.submit_frames,
        frames_before.submit_frames + 1,
        "the seam produced exactly one submission frame"
    );
    assert_eq!(frames.len(), 1, "and the capture holds it");

    let frame = &frames[0];
    eprintln!(
        "wire submission frame: {} bytes, head {:02x} {:02x} {:02x} {:02x}",
        frame.len(),
        frame[0],
        frame[1],
        frame[2],
        frame[3],
    );
    assert_eq!(
        &frame[..4],
        b"MCC1",
        "the frame is the command channel's own, magic and all"
    );
    // The payload tag follows the frame header: a compute-only trace whose
    // table carries texture declarations takes C1c's own tag
    // (`SUBMIT_COMPUTE_TEXTURES_REQUEST` = 0x11).
    eprintln!("wire submission frame body head: {:02x?}", &frame[9..24]);
    assert_eq!(
        frame[9], 0x11,
        "a compute trace with a texture declaration takes the C1c tag"
    );

    let (trace, resources) = provider_wire::carried_submission(frame)
        .expect("the provider's own decoder reads the frame back");
    let reencoded = provider_wire::submit_frame(&trace, &resources).expect("re-encode");
    eprintln!(
        "wire re-encode: bytes_in={} bytes_out={} identical={}",
        frame.len(),
        reencoded.len(),
        reencoded == *frame,
    );
    assert_eq!(
        reencoded, *frame,
        "the frame is a fixed point of the owner's own encoder and the provider's decoder"
    );

    let pipeline = &trace.pipelines[0];
    let declarations = &pipeline.contract.texture_bindings;
    let pass = match &trace.passes[0] {
        metal_api_core::provider::TracePass::Compute(pass) => pass,
        other => panic!("the frame carries a compute pass: {other:?}"),
    };
    assert_eq!(declarations.len(), 1, "one declaration crossed the wire");
    assert_eq!(pass.textures.len(), 1, "and one view beside it");
    let declaration = &declarations[0];
    let view = &pass.textures[0];
    eprintln!(
        "wire decoded declaration: binding={} access={:?} type={:?} format={:?} sampler={:?} footprint={:?}",
        declaration.metal_binding,
        declaration.access,
        declaration.texture_type,
        declaration.format,
        declaration.sampler,
        declaration.footprint,
    );
    eprintln!(
        "wire decoded view: metal_binding={} access={:?} {}x{}",
        view.metal_binding, view.access, view.width, view.height,
    );
    assert_eq!(declaration.metal_binding, 0);
    assert_eq!(
        declaration.format,
        metal_api_core::provider::TextureFormat::R32Float
    );
    assert_eq!(
        declaration.sampler,
        Some(metal_api_core::provider::SamplerPolicy {
            filter: metal_api_core::provider::SamplerFilter::Nearest,
            address: metal_api_core::provider::SamplerAddressMode::ClampToEdge,
        }),
        "the declaration restates the module's own AIR sampler state"
    );
    assert_eq!(view.metal_binding, declaration.metal_binding);
    assert_eq!(view.format, declaration.format);
    match &view.source {
        // The three spellings the wire has for these texels since the statement
        // payload table (E-SW3): the shipped one carries them, the table's
        // declaration arm carries them *and* files them under a slot, and its
        // reference arm names what this same frame's earlier submission filed —
        // which is the same 64 bytes. What this test is about is that the texels
        // are stated by the wire view, so it holds in all three.
        metal_api_core::provider::TextureSource::OwnedBytes(bytes)
        | metal_api_core::provider::TextureSource::OwnedInSlot { bytes, .. } => assert_eq!(
            bytes.len(),
            64,
            "the staged texels crossed the wire in the view"
        ),
        metal_api_core::provider::TextureSource::SlottedBytes { length, .. } => assert_eq!(
            *length, 64,
            "the view names the staged texels this frame already carried"
        ),
        other => panic!("the reviewed texture source is the staged texels: {other:?}"),
    }

    // The capability answer, read the way the class gate reads it: out of the
    // response frame the provider would send.
    let probe = VulkanComputeProvider::with_executor(
        VulkanExecutor::new().expect("the acceptance environment has a Vulkan device"),
    )
    .expect("the canonical provider builds");
    let support =
        provider_wire::compute_texture_support(probe.device_epoch(), &probe.capabilities())
            .expect("the capability answer encodes and decodes");
    eprintln!(
        "wire capability answer: supports_compute_texture_sampling={} max_compute_textures={} formats={:?}",
        support.supported, support.maximum, support.formats,
    );
    assert!(support.supported, "this device declares the shape");
    assert_eq!(
        support.maximum as usize,
        metal_api_core::provider::MAX_COMPUTE_TEXTURES,
        "and the same cap the contract states"
    );
    assert!(support.formats.contains(&declaration.format));

    // The fail-closed arm: the same frame, admitted by a snapshot that does not
    // declare the shape. This is the provider's own answer, not this rail's —
    // the rail keeps the pass on the engine before a frame is ever sent.
    let mut unsupporting = probe.capabilities();
    unsupporting.supports_compute_texture_sampling = false;
    let refusal = unsupporting
        .validate_trace(trace.clone(), resources.clone())
        .expect_err("a snapshot without the bit refuses the pass");
    eprintln!(
        "provider answer without the capability: class={:?} slug={} fields={:?}",
        refusal.class, refusal.slug, refusal.fields
    );
    assert_eq!(refusal.slug, "compute_texture_input_unsupported");
    assert_eq!(
        refusal.class,
        metal_api_core::provider::ProviderErrorClass::Capability
    );
    // And the same frame under the device's own snapshot is admitted: the
    // capability bit is the only difference between the two answers.
    probe
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the provider that declares the shape admits the same frame");

    // The population this increment does not touch: no declaration, no frame —
    // so those shapes keep the bytes (and the path) they had.
    let buffer_only = mul3add1_request();
    let before = provider_wire::wire_counts().submit_frames;
    match submit_compute(&fixture_air(), "apv_cs", &buffer_only, &[], &[]) {
        ComputeRailOutcome::ProviderCompleted(_) => (),
        other => panic!("the reviewed buffer fixture is in class: {other:?}"),
    }
    assert_eq!(
        provider_wire::wire_counts().submit_frames,
        before,
        "a trace whose table declares no texture keeps the path it had"
    );
    assert!(
        provider_wire::captured_submission_frames().is_empty(),
        "and produces no frame at all"
    );
    provider_wire::capture_submission_frames(false);

    // The frame re-encoded out of the pair the provider's decoder produced,
    // twice: once from copies of that pair (the path this rail ran before
    // sp13) and once from the pair itself, moved into the request. The copy is
    // the only difference between the two encoders.
    let copied = provider_wire::submit_frame(&trace, &resources)
        .expect("the copying encoder re-encodes the frame it produced");
    let moved = provider_wire::submit_frame_owned(trace, resources)
        .expect("the moving encoder encodes the same request");
    assert_eq!(
        copied, *frame,
        "the frame is a fixed point of the owner's encoder and the provider's decoder"
    );
    assert_eq!(
        moved, copied,
        "the frame is the same bytes whether its inputs were copied or moved: \
         REIMS_VGPU_SUBMIT_FRAME_OWNED cannot change what the guest sees"
    );
}

/// The provider's own gate for the declaration this rail states
/// (`metal-api-vulkan`'s `refuse_foreign_texture_samplers`), driven through the
/// reims fixture: a trace whose declaration moves the sampler away from the
/// module the provider registered is refused by name, with the binding and both
/// state halves — and the module's own declaration is admitted and executes,
/// which keeps the refusal from being a blanket rejection of the shape.
#[test]
fn the_provider_refuses_a_declaration_that_moves_away_from_its_registered_module() {
    use metal_api_core::provider::{
        AllocationId, AllocationRecord, BufferAccess, BufferSource, BufferView, CompletionPolicy,
        ComputePass, ComputeProvider as _, ComputeTrace, Dispatch, DispatchKind, DispatchType,
        OperationId, ResourceTableSnapshot, SamplerAddressMode, SamplerFilter, SamplerPolicy,
        SemanticDigest, TextureAccess, TextureSource, TextureType, TextureView, TracePass, ViewId,
        PROVIDER_SCHEMA_VERSION,
    };
    use metal_api_core::{ComputeExecutor, Device};
    use std::sync::Arc;

    let air = texture_air();
    let executor = VulkanExecutor::new().expect("a Vulkan executor");
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("a canonical provider");
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let function = device
        .new_library_with_binary_air(air.clone())
        .expect("the fixture library loads")
        .function("sample_texture_2d")
        .expect("the fixture entry exists");
    let registered = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("reims-r9l-fixture", air).expect("digest"),
        )
        .expect("the fixture registers");

    let texture = TextureView {
        view_id: ViewId::new(901),
        metal_binding: 0,
        allocation_id: AllocationId::new(902),
        texture_type: TextureType::D2,
        format: registered.contract.texture_bindings[0].format,
        width: 4,
        height: 4,
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access: TextureAccess::Sampled,
        source: TextureSource::OwnedBytes(sampled_rows([0.0, 4.0, 8.0, 12.0])),
    };
    let trace_of = |pipeline: &metal_api_core::provider::CompiledComputePipeline| ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(9),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![TracePass::Compute(ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers: vec![BufferView {
                view_id: ViewId::new(903),
                metal_binding: 0,
                allocation_id: AllocationId::new(904),
                offset: 0,
                length: 8,
                access: BufferAccess::Write,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0u8; 8]),
            }],
            textures: vec![texture.clone()],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
        })],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let resources_of = || {
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(904),
                owner_epoch: provider.device_epoch(),
                size: 8,
            })
            .expect("the output allocation");
        resources
    };

    // The positive control: the module's own declaration executes, so the
    // refusal below is about the state that moved.
    let admitted = provider
        .capabilities()
        .validate_trace(trace_of(&registered), resources_of())
        .expect("the module's own declaration is admitted");
    let submission = provider
        .submit(admitted)
        .expect("the registered module executes");
    let writeback = submission
        .writebacks
        .iter()
        .find(|writeback| writeback.view_id == ViewId::new(903))
        .expect("the fixture writes its output buffer");
    let reading = [
        f32::from_le_bytes(writeback.bytes[0..4].try_into().expect("first reading")),
        f32::from_le_bytes(writeback.bytes[4..8].try_into().expect("second reading")),
    ];
    eprintln!("provider-path reading, module's own sampler: {reading:?}");
    assert_eq!(reading, [12.0, 4.0]);

    // The foreign declaration: the same trace, the sampler moved to linear.
    let mut tampered = registered.clone();
    tampered.contract.texture_bindings[0].sampler = Some(SamplerPolicy {
        filter: SamplerFilter::Linear,
        address: SamplerAddressMode::ClampToEdge,
    });
    let admitted = provider
        .capabilities()
        .validate_trace(trace_of(&tampered), resources_of())
        .expect("the tampered declaration is structurally admissible");
    let refusal = provider
        .submit(admitted)
        .expect_err("a foreign sampler declaration is refused");
    eprintln!(
        "provider refusal for a moved declaration: class={:?} slug={} fields={:?}",
        refusal.class, refusal.slug, refusal.fields
    );
    assert_eq!(refusal.slug, "compute_texture_sampler_unsupported");
    assert_eq!(
        refusal.class,
        metal_api_core::provider::ProviderErrorClass::Capability
    );
    assert_eq!(
        refusal.fields.get("binding"),
        Some(&metal_api_core::provider::FieldValue::Unsigned(0))
    );
    assert_eq!(
        refusal.fields.get("module_filter"),
        Some(&metal_api_core::provider::FieldValue::Text(
            "Nearest".into()
        ))
    );
}
