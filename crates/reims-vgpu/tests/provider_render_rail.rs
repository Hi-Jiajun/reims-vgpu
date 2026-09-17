//! Gate 2 production-rail integration test: drive the shipped
//! `backend::provider_render` seam with the rail's own fixtures and assert the
//! canonical provider's byte-exact attachment, the self-contained engine's
//! agreement with it, the class boundary, and the fail-closed decline.
//!
//! This is the render sibling of `provider_compute_rail.rs`: the seam is the
//! function `runtime/draw/vulkan.rs::try_metal2vulkan_draw` calls for an
//! in-class draw under `feature = "provider-render"`, and it is public so this
//! off-VM test can prove the conversion — one reims `DrawRequest` + the two
//! translated stages' AIR → canonical `ComputeTrace` with a declaring compute
//! pass and one render pass — without a VM. Requires a Vulkan ICD; the pinned
//! Lavapipe path is the acceptance environment.
//!
//! The engine arm runs `engine::execute_draw_request` with the *same* AIR
//! translated by this crate's own translator, which is what makes the
//! byte-for-byte comparison a statement about the two rails rather than about
//! one rail twice: the provider translates the AIR with the canonical
//! translator and executes it, the engine translates the same AIR with reims'
//! pinned translator and executes it, and the attachment has to come out the
//! same.

#![cfg(feature = "provider-render")]

use metal_api_core::provider::{AttachmentFormat, ComputeProvider, VertexFormat};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, SpirvFeaturePolicy, TranslatedRenderStage, VulkanComputeProvider, VulkanExecutor,
};
use reims_vgpu::backend::provider_render::{
    self, ProviderRenderDecline, RenderChainRole, RenderRailInputs, RenderRailOutcome,
};
use reims_vgpu::backend::vulkan::engine::{
    self, BlendStateResource, BufferContent, DepthState, DrawRequest, IndexType,
    IndexedDrawResource, PrimitiveTopology, ReadbackSkipReason, SamplerResource, ScissorResource,
    VertexAttributeFormat, VertexAttributeResource, VertexStepFunction, ViewportResource,
};
use reims_vgpu::observe::Decline as _;
use reims_vgpu::protocol::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA8_UNORM};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// The engine is process-global, and so is the canonical rail's provider; the
/// engine suite's own lock is private to that test binary, so this file takes
/// its own and resets the engine under it, exactly as `vk_engine_parity` does.
///
/// The engine's own per-device state lives in a `DeviceState` the caller now
/// owns, so this file keeps one per process exactly as `vk_engine_batch` does.
fn engine_device() -> &'static reims_vgpu::model::DeviceState {
    static DEVICE: OnceLock<reims_vgpu::model::DeviceState> = OnceLock::new();
    DEVICE.get_or_init(|| {
        reims_vgpu::model::DeviceState::new(
            reims_vgpu::model::DeviceId(1),
            reims_vgpu::protocol::gva::PAGE_SHIFT_X86,
        )
    })
}

fn engine_test_session() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = LOCK
        .get_or_init(|| {
            reims_vgpu::observe::redirect_logs_for_tests();
            Mutex::new(())
        })
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    engine::test_reset_engine(engine_device());
    guard
}

/// One fixture AIR module, carved out of its bitcode wrapper.
///
/// The carve is what the production seam hands the rail as well: `load_mtlb` +
/// `extract_air` is how a guest's MTLB becomes the AIR both translators read,
/// and the canonical gate refuses a wrapped module that is not offset-zero.
fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/air")
        .join(name);
    let raw = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    reims_vgpu::runtime::mtlb::extract_air(&raw)
        .unwrap_or_else(|decline| panic!("{name}: {decline}"))
        .to_vec()
}

/// One translated stage pair, beside the reflection facts the class gate reads
/// from the vertex half.
///
/// The AIR travels to both arms of every test — the canonical provider
/// translates it itself, the engine goes through this crate's translator — so
/// the two must be the same bytes; the entry names and the reflected attribute
/// locations are what the seam would take from `CachedShader::reflection` and
/// hands the rail as inputs.
struct Stages {
    air: (Vec<u8>, Vec<u8>),
    vertex_entry: &'static str,
    fragment_entry: &'static str,
    vertex_attribute_locations: Vec<u32>,
}

/// The reviewer's solid-colour fragment, shared by every vertex fixture here.
const FRAGMENT_ENTRY: &str = "fmain";

fn stages(vertex_fixture: &str, vertex_entry: &'static str, locations: &[u32]) -> Stages {
    Stages {
        air: (fixture(vertex_fixture), fixture("render_frag.air")),
        vertex_entry,
        fragment_entry: FRAGMENT_ENTRY,
        vertex_attribute_locations: locations.to_vec(),
    }
}

/// The reviewed one-stream shape: `float2` per vertex at location 0, the
/// fixture the rail shipped with.
fn reviewed_stages() -> Stages {
    stages("reims_indexed_tri.air", "reims_indexed_vertex", &[0])
}

/// The two-stream shape: position at location 0 and an offset at location 1,
/// each its own stream and each read by the vertex stage.
fn two_stream_stages() -> Stages {
    stages(
        "reims_indexed_tri_two_stream.air",
        "reims_two_stream_vertex",
        &[0, 1],
    )
}

/// The four-stream shape — the widest interface the canonical contract can
/// state (`max_vertex_buffers`): position at location 0 and three offsets after
/// it, all read by the vertex stage.
fn four_stream_stages() -> Stages {
    stages(
        "reims_indexed_tri_four_stream.air",
        "reims_four_stream_vertex",
        &[0, 1, 2, 3],
    )
}

/// `float2` records in the order a stream carries them: little-endian pairs.
fn f32x2(records: &[(f32, f32)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * 8);
    for (x, y) in records {
        out.extend_from_slice(&x.to_ne_bytes());
        out.extend_from_slice(&y.to_ne_bytes());
    }
    out
}

/// The reviewed position stream: `(-1, -3)`, `(-1, 1)`, `(3, 1)` — the
/// full-screen triangle the reviewed milestone uses, read from a caller-held
/// stream so the vertex-input half of the class is exercised.
fn position_records() -> Vec<u8> {
    f32x2(&[(-1.0, -3.0), (-1.0, 1.0), (3.0, 1.0)])
}

/// `[0, 1, 2]` as `u32` little-endian: the reviewed indexed shape, whose index
/// values select the three vertices in order.
const INDEX_BYTES: [u8; 12] = [
    0x00, 0x00, 0x00, 0x00, //
    0x01, 0x00, 0x00, 0x00, //
    0x02, 0x00, 0x00, 0x00, //
];

/// The attachment window the canonical rail declares on this device.
///
/// Read from a live provider's capability snapshot rather than stated here:
/// R1b made the number `min(reviewed ceiling, device framebuffer limit)` per
/// axis, so a copy in this file would go stale on a device that declares less
/// than the ceiling — exactly the drift the class gate no longer carries. The
/// one provider built here is a second, independent snapshot: the rail's gate
/// has to answer with the same numbers.
fn declared_window() -> [u64; 2] {
    static WINDOW: OnceLock<[u64; 2]> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        let executor =
            VulkanExecutor::new().expect("the acceptance environment has a Vulkan device");
        let provider =
            VulkanComputeProvider::with_executor(executor).expect("the canonical provider builds");
        let window = provider.capabilities().max_attachment_dimension;
        assert!(
            window[0] > 0 && window[1] > 0,
            "the acceptance environment's provider declares a render window: {window:?}"
        );
        window
    })
}

/// The attachment extent the reviewed shape draws: the declared window's own
/// size, so the fixture sits on the class gate's inclusive boundary.
fn extent() -> (u32, u32) {
    let window = declared_window();
    (
        u32::try_from(window[0]).expect("the declared width fits a u32"),
        u32::try_from(window[1]).expect("the declared height fits a u32"),
    )
}

/// `float4(0.25, 0.5, 0.75, 1)` in the 8-bit encoding both rails round to.
const FRAGMENT_TEXEL: [u8; 4] = [64, 128, 191, 255];

/// A clear every component of which is exactly representable in 8 bits, so the
/// engine's `float32` clear and the canonical pass's `ClearColor` bytes are the
/// same colour.
const CLEAR: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

fn attachment(format: u16) -> reims_vgpu::backend::vulkan::engine::ColorAttachmentState {
    reims_vgpu::backend::vulkan::translate::pixel::color_attachment(format)
        .expect("the fixture's attachment format is renderable")
        .0
        .with_clear(CLEAR)
}

/// One vertex stream of a test's draw, in the terms the production seam builds
/// it: a `float2` attribute at its own location, one record per vertex, and the
/// bytes of the whole window.
///
/// Separate from [`VertexAttributeResource`] because the two arms of a test —
/// the rail and the engine — each need their own request, and the resource type
/// owns its bytes.
struct StreamSpec {
    location: u32,
    offset: u32,
    stride: u32,
    bytes: Vec<u8>,
}

fn streams(specs: &[StreamSpec]) -> Vec<VertexAttributeResource> {
    specs
        .iter()
        .map(|spec| VertexAttributeResource {
            location: spec.location,
            // The engine numbers one Vulkan binding per attribute *location*
            // (`runtime/draw/vulkan.rs` builds `binding: a.location`), which is
            // why the seam's own requests spell it that way too.
            binding: spec.location,
            format: VertexAttributeFormat::parse(MTL_FORMAT_VERTEX_FLOAT2)
                .expect("Float2 is a vertex format"),
            offset: spec.offset,
            stride: spec.stride,
            step_function: VertexStepFunction::PerVertex,
            step_rate: 1,
            content: BufferContent::Bytes(std::sync::Arc::new(spec.bytes.clone())),
        })
        .collect()
}

/// The reviewed position stream: the full-screen triangle at location 0.
fn position_stream() -> StreamSpec {
    StreamSpec {
        location: 0,
        offset: 0,
        stride: 8,
        bytes: position_records(),
    }
}

fn position_streams() -> [StreamSpec; 1] {
    [position_stream()]
}

/// One stream of the multi-stream shapes: `float2` records at `location`, stride
/// eight.
///
/// The fixtures behind these streams add the offset streams into the position
/// with `fadd fast` — the flag run a Metal module compiled with the default math
/// mode carries, and the reason the fast twins keep translating on every device.
/// A module whose add *withholds* the permission is admitted exactly when the
/// device answers for `FloatControls2` (R8 in the emulator, adopted on this seam
/// by R8b, `a_float_op_that_withholds_a_permission_lands_the_fast_twins_bytes`).
/// The offsets the tests hand these streams are exact binary fractions, so the
/// two rails' own arithmetic rounds them the same way and parity stays an
/// assertion about stream plumbing.
fn stream(location: u32, records: &[(f32, f32)]) -> StreamSpec {
    StreamSpec {
        location,
        offset: 0,
        stride: 8,
        bytes: f32x2(records),
    }
}

/// The admitted request, in the terms the production seam builds it: one colour
/// attachment, one vertex stream (one `float2` attribute at location 0), one
/// index stream, and the pooled offscreen target whose whole frame is read back.
fn narrow_request(format: u16) -> DrawRequest {
    request_with_streams(format, &position_streams())
}

/// The same shape with the streams a test hands it, so the multi-stream half of
/// the class is exercised through the same request builder the rail sees.
fn request_with_streams(format: u16, specs: &[StreamSpec]) -> DrawRequest {
    let (width, height) = extent();
    DrawRequest {
        width,
        height,
        vertex_count: 3,
        instance_count: Some(1),
        primitive_topology: PrimitiveTopology(reims_vgpu_core::topology::PrimitiveType::Triangle),
        raster_sample_count: 1,
        color_sample_count: 1,
        indexed: Some(IndexedDrawResource {
            index_type: IndexType::U32,
            index_count: 3,
            vertex_offset: 0,
            content: BufferContent::Bytes(std::sync::Arc::new(INDEX_BYTES.to_vec())),
        }),
        vertex_attributes: streams(specs),
        color_attachment: Some(attachment(format)),
        color0_declared: Some(reims_vgpu::protocol::pass_action::LoadAction::Clear),
        ..Default::default()
    }
}

/// `MTLVertexFormat::Float2`.
const MTL_FORMAT_VERTEX_FLOAT2: u32 = 29;

fn inputs<'a>(stages: &'a Stages, role: RenderChainRole) -> RenderRailInputs<'a> {
    RenderRailInputs {
        vertex_air: &stages.air.0,
        fragment_air: &stages.air.1,
        vertex_entry: Some(stages.vertex_entry),
        fragment_entry: Some(stages.fragment_entry),
        role,
        vertex_attribute_locations: &stages.vertex_attribute_locations,
    }
}

/// The engine arm: the same AIR, translated by this crate's own translator.
fn engine_request(stages: &Stages, format: u16, specs: &[StreamSpec]) -> DrawRequest {
    translated(stages, request_with_streams(format, specs))
}

/// One request with this crate's own translation of the same AIR in it: what
/// the self-contained engine needs, and the only difference between the two
/// arms of a test.
fn translated(stages: &Stages, mut req: DrawRequest) -> DrawRequest {
    let words = |stage| -> Vec<u32> {
        let shader = reims_vgpu::runtime::m2v_cache::translate_cached_reflected(
            match stage {
                metal2vulkan::passes::Stage::Vertex => stages.air.0.as_slice(),
                metal2vulkan::passes::Stage::Fragment => stages.air.1.as_slice(),
                metal2vulkan::passes::Stage::Kernel => unreachable!("render stages only"),
            },
            stage,
            0,
        )
        .expect("the fixture translates");
        shader.words.as_ref().clone()
    };
    req.vert_spirv = std::sync::Arc::new(words(metal2vulkan::passes::Stage::Vertex));
    req.frag_spirv = std::sync::Arc::new(words(metal2vulkan::passes::Stage::Fragment));
    req
}

/// The canonical rail's own frame for one request, in semantic RGBA8.
fn provider_pixels(label: &str, stages: &Stages, req: &DrawRequest) -> Vec<u8> {
    match provider_render::submit_render(&inputs(stages, RenderChainRole::SoleOrTail), req) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("{label}: the canonical provider has to execute this shape: {other:?}"),
    }
}

/// The self-contained engine's frame for the same request, in semantic RGBA8.
///
/// `None` when the acceptance environment has no Vulkan device at all, which is
/// the one condition the reviewed cases treat as a skip.
fn engine_pixels(label: &str, stages: &Stages, req: DrawRequest) -> Option<Vec<u8>> {
    match engine::execute_draw_request(engine_device(), &translated(stages, req)) {
        Ok(out) => Some(semantic_rgba(out.pixels, out.pixels_bgra)),
        Err(error) => {
            let text = error.to_string();
            if skip_if_no_gpu(&text) {
                eprintln!("SKIP {label} engine arm: no GPU ({text})");
                return None;
            }
            panic!("{label} engine arm: {text}")
        }
    }
}

fn skip_if_no_gpu(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    lower.contains("no vulkan")
        || lower.contains("load vulkan")
        || lower.contains("create_instance")
        || lower.contains("no graphics")
        || lower.contains("vk_engine_init")
}

/// Read a draw's pixels in semantic RGBA8, whatever physical order the
/// attachment read back in, so the two rails are compared as colours and
/// layouts separately.
fn semantic_rgba(mut pixels: Vec<u8>, bgra: bool) -> Vec<u8> {
    if bgra {
        for texel in pixels.chunks_exact_mut(4) {
            texel.swap(0, 2);
        }
    }
    pixels
}

fn assert_solid(label: &str, pixels: &[u8]) {
    assert_texel_count(label, pixels);
    for (index, texel) in pixels.chunks_exact(4).enumerate() {
        assert_texel_near(
            &format!("{label}: texel {index}"),
            [texel[0], texel[1], texel[2], texel[3]],
            FRAGMENT_TEXEL,
        );
    }
}

/// The attachment's whole extent has to come back, whatever the frame is.
fn assert_texel_count(label: &str, pixels: &[u8]) {
    let (width, height) = extent();
    assert_eq!(
        pixels.len(),
        (width * height * 4) as usize,
        "{label}: the attachment's whole extent has to come back"
    );
}

/// One texel of a readback, in the attachment's own physical order.
fn texel_at(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let (width, _) = extent();
    let offset = ((y * width + x) * 4) as usize;
    [
        pixels[offset],
        pixels[offset + 1],
        pixels[offset + 2],
        pixels[offset + 3],
    ]
}

/// A texel the fragment stage covered: the fragment's own colour, within a
/// rounding step, because the attachment rounds the `float32` output to eight
/// bits on both rails.
fn assert_texel_near(label: &str, got: [u8; 4], want: [u8; 4]) {
    for channel in 0..4 {
        assert!(
            (i32::from(got[channel]) - i32::from(want[channel])).abs() <= 1,
            "{label}: channel {channel} is {}, expected ~{} (whole texel {got:?}, want {want:?})",
            got[channel],
            want[channel],
        );
    }
}

/// A texel the fragment stage did *not* cover: the clear, byte-exact.
fn assert_clear_texel(label: &str, got: [u8; 4]) {
    assert_eq!(
        got,
        [0, 0, 0, 255],
        "{label}: a texel outside the scissor keeps the clear's own bytes"
    );
}

#[test]
fn the_production_seam_completes_the_reviewed_shape_and_agrees_with_the_engine() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let req = narrow_request(MTL_FORMAT_RGBA8_UNORM);

    let provider =
        match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
            RenderRailOutcome::ProviderCompleted(out) => out,
            // The pooled shape names no resident, so the resident arm cannot
            // answer for it.
            RenderRailOutcome::ProviderCompletedResident(frame) => {
                panic!("the pooled class published no frame and kept one instead: {frame:?}")
            }
            RenderRailOutcome::NotInNarrowClass(reason) => {
                panic!("the reviewed shape is in the narrow class; refused: {reason}")
            }
            RenderRailOutcome::ProviderDeclined(decline) => {
                panic!("the canonical provider declined the reviewed shape: {decline}")
            }
        };
    assert!(
        !provider.bgra,
        "an Rgba8Unorm attachment reads back in RGBA order"
    );
    let provider_pixels = semantic_rgba(provider.bytes, provider.bgra);
    assert_solid("provider", &provider_pixels);

    let engine_req = engine_request(&stages, MTL_FORMAT_RGBA8_UNORM, &position_streams());
    let engine_out = match engine::execute_draw_request(engine_device(), &engine_req) {
        Ok(out) => out,
        Err(error) => {
            let text = error.to_string();
            if skip_if_no_gpu(&text) {
                eprintln!("SKIP engine arm: no GPU ({text})");
                return;
            }
            panic!("engine arm: {text}");
        }
    };
    let engine_pixels = semantic_rgba(engine_out.pixels, engine_out.pixels_bgra);
    assert_solid("engine", &engine_pixels);
    assert_eq!(
        provider_pixels, engine_pixels,
        "the same AIR, drawn by the canonical provider and by the self-contained engine, has to \
         land the same bytes in the attachment"
    );
}

/// The chain half of the class: a packet's *first* record hands its frame on.
///
/// The exec loop's store plan grants the guest writeback to a packet's last
/// record only (`runtime::exec::multi_draw_store_plan`), so every record before
/// it produces a frame that belongs to the chain rather than to guest memory —
/// and the record that *opens* the pass is the one that needs no frame from
/// anywhere. That is the position this class executes: the pass's own beginning
/// (`Clear`, which the class already admits) plus the frame's way back to the
/// caller, which the encode side already has (`runtime/draw/vulkan.rs` returns
/// the pixels for every `writeback_guest == false` encode instead of storing
/// them, and `runtime::exec` hands them to the next record as its seed).
///
/// The handback is driven here the way the exec loop drives it: the head's
/// frame, in the seed's own order (`SeedOrder::Rgba8`, the order
/// `encode_draw_chain` reorders a readback into before returning it), becomes
/// the second record's `Load` seed, and the packet's final attachment has to be
/// the bytes the engine's own chain produces from the engine's own head frame.
/// The control is what keeps that claim falsifiable: the same second record
/// begun from its own `Clear` — the head's frame dropped, which is what a chain
/// whose first record never landed looks like — has to land different bytes, so
/// a rail that lost the frame cannot pass by drawing the same thing either way.
#[test]
fn a_chain_head_hands_its_frame_to_the_next_record() {
    use reims_vgpu::protocol::pass_action::LoadAction;

    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();
    let half = width / 2;

    // The packet's first record: it opens the pass, a record follows it, and it
    // owns no guest writeback.
    let mut head = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    head.render_pass_continues = true;

    let head_provider =
        match provider_render::submit_render(&inputs(&stages, RenderChainRole::Head), &head) {
            RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
            RenderRailOutcome::ProviderCompletedResident(frame) => {
                panic!("the chain head reads no resident and keeps none: {frame:?}")
            }
            RenderRailOutcome::NotInNarrowClass(reason) => {
                panic!(
                    "the record that opens the pass is in the class: it needs no frame from \
                     anywhere, and its frame goes back to the caller; refused: {reason}"
                )
            }
            RenderRailOutcome::ProviderDeclined(decline) => {
                panic!("the canonical provider declined the chain head: {decline}")
            }
        };
    assert_solid("chain head (provider)", &head_provider);

    // The same record on the engine rail: the frame a build without the
    // provider would carry through the chain.
    let Some(head_engine) = engine_pixels("chain head", &stages, head) else {
        return;
    };
    assert_eq!(
        head_provider, head_engine,
        "the head's frame is the frame the chain carries, so both rails have to produce it"
    );
    assert_eq!(
        head_provider.len(),
        (width * height * 4) as usize,
        "the frame handed back is the whole attachment"
    );

    // The record after it: a record with a predecessor LOADs the chain's frame
    // and draws the left half of it.
    let second_record = |seed: Option<Vec<u8>>| {
        let mut req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, &position_streams());
        req.scissors.push(ScissorResource {
            x: 0,
            y: 0,
            width: half,
            height,
        });
        if seed.is_some() {
            req.color0_declared = Some(LoadAction::Load);
        }
        req.target_rgba8 = seed.map(std::sync::Arc::new);
        engine_pixels("the record after the head", &stages, req)
    };

    let Some(chained) = second_record(Some(head_provider)) else {
        return;
    };
    let Some(from_engine_head) = second_record(Some(head_engine)) else {
        return;
    };
    assert_eq!(
        chained, from_engine_head,
        "the packet's final attachment is the same whether its first record ran on the canonical \
         rail or on the engine: the handback is the only thing W1 moved"
    );
    // The half the second record did not draw carries the head's frame rather
    // than the clear, which is the whole reading this test exists for.
    assert_texel_near(
        "the second record: the half it drew",
        texel_at(&chained, half / 2, height / 2),
        FRAGMENT_TEXEL,
    );
    assert_texel_near(
        "the second record: the half the head filled",
        texel_at(&chained, width - 1, height / 2),
        FRAGMENT_TEXEL,
    );

    // The control: the same second record with the head's frame dropped, where
    // the pass begins from the clear the guest declared.
    let Some(dropped) = second_record(None) else {
        return;
    };
    assert_ne!(
        chained, dropped,
        "the head's frame has to reach the record after it: with the frame dropped, the half the \
         second record does not draw is the clear"
    );
    assert_texel_near(
        "the dropped chain: the half the second record drew",
        texel_at(&dropped, half / 2, height / 2),
        FRAGMENT_TEXEL,
    );
    assert_clear_texel(
        "the dropped chain: the half the head would have filled",
        texel_at(&dropped, width - 1, height / 2),
    );

    // What the class does *not* read: whether the guest has a record after this
    // one. The frame comes back to the caller either way, so the successor fact
    // is the caller's business rather than a class condition — the exec walk
    // never produces a record without one (every `!do_writeback` record is
    // followed by another), and this states the answer if one arrived.
    let uncontinued = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::Head), &uncontinued) {
        RenderRailOutcome::ProviderCompleted(out) => assert_solid(
            "a head with no record after it",
            &semantic_rgba(out.bytes, out.bgra),
        ),
        other => {
            panic!("the successor fact belongs to the caller rather than to the class: {other:?}")
        }
    }
}

/// The chain in the order the census's shapes mostly carry: BGRA8.
///
/// The rail reports a readback's physical order instead of converting it, and
/// `encode_draw_chain` is what turns a `!writeback_guest` readback into the
/// `SeedOrder::Rgba8` image the next record loads. This test drives that
/// conversion the way the encode side does — the head's frame is read in guest
/// scanout order and reordered — and asserts the packet still lands the head's
/// own colour: the fragment's channels are not symmetric under an R/B exchange
/// (`64` against `191`), so a chain that seeded the *raw* readback would put
/// `191` in the red slot, and the half the second record does not draw would
/// say so. The assertion is therefore about the order the chain carries, not
/// only about the two rails agreeing.
#[test]
fn a_scanout_order_chain_head_seeds_the_next_record() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();
    let half = width / 2;

    // The head, on the format the census's shapes mostly state.
    let mut head = narrow_request(MTL_FORMAT_BGRA8_UNORM);
    head.render_pass_continues = true;
    let head_provider =
        match provider_render::submit_render(&inputs(&stages, RenderChainRole::Head), &head) {
            RenderRailOutcome::ProviderCompleted(out) => {
                assert!(out.bgra, "a Bgra8Unorm attachment reads back in BGRA order");
                // The seed's own order, which is what `encode_draw_chain` builds
                // for a chain image before it hands it back.
                semantic_rgba(out.bytes, out.bgra)
            }
            other => panic!("a Bgra8Unorm chain head is in the class: {other:?}"),
        };
    let Some(head_engine) = engine_pixels("scanout-order chain head", &stages, head) else {
        return;
    };
    assert_eq!(
        head_provider, head_engine,
        "the head's frame is the same colour on both rails, whatever order it was read in"
    );

    // The record after it, on the same format: the packet's records share one
    // attachment template, so the seed crosses the order boundary in the one
    // place the engine already handles it (`DrawRequest::target_seed_order`).
    let mut second = request_with_streams(MTL_FORMAT_BGRA8_UNORM, &position_streams());
    second.scissors.push(ScissorResource {
        x: 0,
        y: 0,
        width: half,
        height,
    });
    second.color0_declared = Some(reims_vgpu::protocol::pass_action::LoadAction::Load);
    second.target_rgba8 = Some(std::sync::Arc::new(head_provider));
    let Some(chained) = engine_pixels("scanout-order chain", &stages, second) else {
        return;
    };
    assert_texel_near(
        "scanout-order chain: the half the second record drew",
        texel_at(&chained, half / 2, height / 2),
        FRAGMENT_TEXEL,
    );
    assert_texel_near(
        "scanout-order chain: the half the head filled",
        texel_at(&chained, width - 1, height / 2),
        FRAGMENT_TEXEL,
    );
}

/// The guest-visible write order is the other half of the pairing: a BGRA
/// attachment is what a mapper-ref-texture target reads back in, and the rail
/// reports the order rather than converting, exactly as the engine does.
#[test]
fn a_bgra_attachment_reports_guest_scanout_order() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let req = narrow_request(MTL_FORMAT_BGRA8_UNORM);
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::ProviderCompleted(out) => {
            assert!(out.bgra, "a Bgra8Unorm attachment reads back in BGRA order");
            // The same colour, in the other physical order: the fragment
            // stage's stored red lands third.
            let pixels = semantic_rgba(out.bytes, out.bgra);
            assert_solid("provider_bgra", &pixels);
        }
        other => panic!("a Bgra8Unorm attachment is in the class: {other:?}"),
    }
}

/// The class boundary, one shape at a time: every mutation below leaves the
/// self-contained engine's behaviour exactly as a build without the feature
/// would run it.
#[test]
fn out_of_class_shapes_stay_on_the_self_contained_engine() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let class = |req: &DrawRequest| match provider_render::submit_render(
        &inputs(&stages, RenderChainRole::SoleOrTail),
        req,
    ) {
        RenderRailOutcome::NotInNarrowClass(_) => (),
        other => panic!("expected an out-of-class answer, got {other:?}"),
    };

    // The chain positions whose frame this class cannot name stay on the
    // engine: the packet's middle (a predecessor and a successor) when it
    // carries no resident source, and the packet's last record when it begins
    // from guest bytes rather than the provider's image. The positions R7b and
    // W1 admit are exercised in `a_resident_middle_record_is_admitted_and_a_
    // source_less_one_is_not` and `a_chain_head_hands_its_frame_to_the_next_
    // record`.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.continues_render_pass = true;
    req.render_pass_continues = true;
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::Middle), &req) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_chain_middle",
            "a record with a predecessor and a successor stays on the engine: {reason}"
        ),
        other => panic!("a chain middle is out of class: {other:?}"),
    }
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.continues_render_pass = true;
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_encoder",
            "the record that continues an encoder stays on the engine: {reason}"
        ),
        other => panic!("a continued record is out of class: {other:?}"),
    }

    // A non-indexed draw.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.indexed = None;
    class(&req);

    // A second stream the vertex stage does not read: several streams are in
    // class now, but only the ones the shader's own reflection names — the
    // canonical registration gate compares the two and refuses a stream nothing
    // reads, which is a decline rather than a fallback.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    let format = req.vertex_attributes[0].format;
    req.vertex_attributes.push(VertexAttributeResource {
        location: 1,
        binding: 1,
        format,
        offset: 8,
        stride: 16,
        step_function: VertexStepFunction::PerVertex,
        step_rate: 1,
        content: BufferContent::Bytes(std::sync::Arc::new(position_records())),
    });
    class(&req);

    // A vertex format outside the canonical set.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.vertex_attributes[0].format =
        VertexAttributeFormat::parse(12).expect("UChar4Normalized is a vertex format");
    class(&req);

    // The zero-copy gather rail: an index stream with no CPU-staged bytes.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.indexed.as_mut().expect("indexed").content =
        BufferContent::from(vec![0u8; 12].into_iter().collect::<Vec<u8>>());
    // (`BufferContent::from(Vec<u8>)` is the CPU form; the gather form is built
    // through `GuestRunSource`, which this test cannot mint without a guest —
    // the empty-bytes case below is what pins the "no bytes" arm.)
    let mut no_bytes = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    no_bytes.indexed.as_mut().expect("indexed").content =
        BufferContent::Bytes(std::sync::Arc::new(Vec::new()));
    class(&no_bytes);
    let _ = req;

    // A declared Load whose previous contents this rail cannot name keeps the
    // engine: the class states the provider's own image or a clear, and the
    // guest-bytes arm is the increment after this one.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.color0_declared = Some(reims_vgpu::protocol::pass_action::LoadAction::Load);
    class(&req);

    // A clear that the 8-bit encoding cannot represent exactly.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.color_attachment = Some(
        reims_vgpu::backend::vulkan::translate::pixel::color_attachment(MTL_FORMAT_RGBA8_UNORM)
            .expect("renderable")
            .0
            .with_clear([0.5, 0.0, 0.0, 1.0]),
    );
    class(&req);

    // 8-bit formats only.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    // An sRGB view of the same 8-bit layout: renderable, and not the

    // linear format the class names.
    req.color_attachment = Some(attachment(
        reims_vgpu::protocol::pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB,
    ));
    class(&req);

    // Blend state, a write mask, an explicit viewport and a target identity no
    // record here loads from or keeps all leave their rails to the engine.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.blend = Some(BlendStateResource {
        src_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_SOURCE_ALPHA,
        dst_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
        op_rgb: reims_vgpu_core::blend::MTL_BLEND_OPERATION_ADD,
        src_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ONE,
        dst_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
        op_alpha: reims_vgpu_core::blend::MTL_BLEND_OPERATION_ADD,
    });
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.color_write_mask = reims_vgpu::protocol::blend::ColorWriteMask::NONE;
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    let (width, height) = extent();
    req.viewports.push(ViewportResource {
        x: 0.0,
        y: 0.0,
        width: width as f32,
        height: height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    });
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.skip_readback = true;
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.target_identity = Some(engine::TargetIdentity::Surface {
        id: 1,
        width,
        height,
        generation: 1,
        format: reims_vgpu::backend::vulkan::translate::pixel::SCANOUT_FORMAT,
    });
    class(&req);

    // A storage buffer, a sampled image and an occlusion query.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.storage_buffers.push(engine::StorageBufferResource {
        binding: 0,
        content: BufferContent::Bytes(std::sync::Arc::new(vec![0u8; 16])),
    });
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.samplers.push(SamplerResource::normalized_default(0));
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.occlusion_query = Some(engine::VisibilityResultMode::Boolean);
    class(&req);

    // Multisample and depth.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.raster_sample_count = 4;
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.depth = Some(DepthState {
        identity: None,
        test_enable: false,
        write_enable: false,
        compare: Default::default(),
        clear_value: 1.0,
        load: false,
        stencil: None,
    });
    class(&req);

    // A non-triangle-list topology and an instanced draw.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.primitive_topology = PrimitiveTopology(reims_vgpu_core::topology::PrimitiveType::Line);
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.instance_count = Some(2);
    class(&req);

    // An attachment outside the window this device's provider declares (R1b):
    // the provider would refuse it at admission, and a shape the provider
    // always refuses is not one this class executes.
    let (window_width, window_height) = extent();
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.width = window_width + 1;
    req.height = window_height;
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.width = window_width * 8;
    req.height = window_height * 8;
    class(&req);
}

/// The census half of the class boundary: every out-of-class answer is counted
/// under its own stable bucket, so a boot's `store_routes` says *how much* of a
/// draw stream each condition kept on the engine and not only that the
/// condition exists.
///
/// Read from the rail's own live window
/// (`runtime::drain::store_route_count_for_test`) rather than from the emitted
/// line: `store_routes` is drained on the drain worker's census tick, which an
/// off-VM rail test never runs, and a counter nobody reads back is one that can
/// be deleted, renamed, or wired to the wrong `return` with nothing noticing.
///
/// The buckets are the 2026-09-17 render profile's eighth observability gap:
/// the profile had to re-derive the out-of-class population from a dozen
/// counters in a dozen places ("how much of a boot's draw stream was instanced,
/// was multisampled, was Load rather than Clear"), because the seam's own
/// `linux_render_provider out_of_class` line is latched per pipeline and sized
/// by neither time nor draws.
#[test]
fn each_out_of_class_condition_is_counted_under_its_own_bucket() {
    use reims_vgpu::protocol::pass_action::LoadAction;

    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let count = |route: &str| reims_vgpu::runtime::drain::store_route_count_for_test(route);
    let out_of_class = |req: &DrawRequest| match provider_render::submit_render(
        &inputs(&stages, RenderChainRole::SoleOrTail),
        req,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            assert!(
                !reason.detail().is_empty(),
                "an out-of-class answer still carries its sentence"
            );
        }
        other => panic!("expected an out-of-class answer, got {other:?}"),
    };

    let instanced = count("render_provider_out_of_class_instanced");
    let load_action = count("render_provider_out_of_class_load_action");
    let depth = count("render_provider_out_of_class_depth");
    let window = count("render_provider_out_of_class_attachment_window");
    // A condition none of the shapes below exercises: its bucket must not move
    // when its neighbours do, or the buckets are one counter wearing five
    // names.
    let untouched = count("render_provider_out_of_class_mrt");

    // An in-class shape is not an out-of-class answer and must not charge any
    // bucket: the counters are charged where the gate answers, not where a
    // render request arrives.
    let req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("the reviewed shape is in class: {other:?}"),
    }

    // One shape per condition, plus one repeat so the counter is a count and
    // not a latch.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.instance_count = Some(4);
    out_of_class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.instance_count = Some(2);
    out_of_class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.color0_declared = Some(LoadAction::Load);
    out_of_class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.depth = Some(DepthState {
        identity: None,
        test_enable: false,
        write_enable: false,
        compare: Default::default(),
        clear_value: 1.0,
        load: false,
        stencil: None,
    });
    out_of_class(&req);
    let (window_width, window_height) = extent();
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.width = window_width + 1;
    req.height = window_height;
    out_of_class(&req);

    assert_eq!(
        count("render_provider_out_of_class_instanced"),
        instanced + 2,
        "two instanced shapes answer the same condition twice, so the bucket counts rather than \
         latches"
    );
    assert_eq!(
        count("render_provider_out_of_class_load_action"),
        load_action + 1,
        "the Load shape is charged to the load-action bucket"
    );
    assert_eq!(
        count("render_provider_out_of_class_depth"),
        depth + 1,
        "the depth shape is charged to the depth bucket"
    );
    assert_eq!(
        count("render_provider_out_of_class_attachment_window"),
        window + 1,
        "the over-window shape is charged to the window bucket"
    );
    assert_eq!(
        count("render_provider_out_of_class_mrt"),
        untouched,
        "a bucket nobody answered with does not move"
    );
}

/// The class window is the number the canonical rail declares on this device,
/// not a copy inside this crate: a shape at the declared window still reaches
/// the provider, and one texel beyond it stays on the engine *before* the
/// provider is asked.
///
/// The boundary is read from a live provider snapshot here, so a device whose
/// framebuffer limit is below the reviewed ceiling moves the check with it: a
/// gate that lagged that declaration fails the one-texel case instead of
/// passing on this machine's declared window.
#[test]
fn the_class_window_is_the_providers_declaration() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (window_width, window_height) = extent();
    eprintln!(
        "declared attachment window: {window_width}x{window_height} (read from a live provider \
         snapshot, not stated in this file)"
    );

    // At the window: in class, and the shape really reaches the provider (so
    // the untouched-counter assertion below is not vacuous).
    let delivered = provider_render::provider_submissions();
    let req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::ProviderCompleted(out) => assert_solid(
            "at the declared window",
            &semantic_rgba(out.bytes, out.bgra),
        ),
        other => panic!("an attachment at the declared window is in class: {other:?}"),
    }
    assert_eq!(
        provider_render::provider_submissions(),
        delivered + 1,
        "the at-window shape reached the provider"
    );

    // One texel beyond it on either axis: out of class, the engine's to draw,
    // and the provider is not called for it.
    let delivered = provider_render::provider_submissions();
    for (width, height) in [
        (window_width + 1, window_height),
        (window_width, window_height + 1),
    ] {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.width = width;
        req.height = height;
        let request_extent = format!("{width}x{height}");
        let declared = format!("{window_width}x{window_height}");
        match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                assert!(
                    reason.detail().contains(request_extent.as_str()),
                    "the reason names the request's own extent: {reason}"
                );
                assert!(
                    reason.detail().contains(declared.as_str()),
                    "the reason names the window the provider declared: {reason}"
                );
            }
            other => panic!("{request_extent} is outside the declared window: {other:?}"),
        }
    }
    assert_eq!(
        provider_render::provider_submissions(),
        delivered,
        "a shape outside the declared window stays on the engine without the provider seeing it"
    );
}

/// Fail-closed: an in-class shape the canonical provider itself refuses — here
/// an index view shorter than the draw's index count — is a typed decline on
/// reims' own vocabulary, never a silent re-run on the self-contained engine.
#[test]
fn an_in_class_shape_the_provider_refuses_is_a_typed_decline() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.indexed.as_mut().expect("indexed").content =
        BufferContent::Bytes(std::sync::Arc::new(INDEX_BYTES[..4].to_vec()));
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::ProviderDeclined(decline) => {
            assert_eq!(
                decline.slug(),
                "provider_capability",
                "the refusal is the provider's own class, under this rail's name"
            );
            let fields = decline.fields();
            assert!(
                fields
                    .iter()
                    .any(|(key, value)| *key == "step" && value == "submission"),
                "the refusal names the provider step that refused: {fields:?}"
            );
            assert!(
                fields.iter().any(|(key, value)| *key == "detail"
                    && value.contains("render_index_buffer_footprint_unsupported")),
                "the provider's own slug rides along: {fields:?}"
            );
        }
        RenderRailOutcome::ProviderCompleted(_) => {
            panic!("the provider completed a draw whose index view is too short")
        }
        RenderRailOutcome::ProviderCompletedResident(_) => {
            panic!("a draw with no target identity cannot keep a resident frame")
        }
        RenderRailOutcome::NotInNarrowClass(reason) => {
            panic!("an in-class refusal must not fall back to the engine: {reason}")
        }
    }
}

/// Fail-closed, chain half: an admitted chain head the provider itself refuses
/// ends the draw as a typed decline, exactly as the reviewed record does — the
/// engine does not quietly draw the head the class took.
///
/// The position matters rather than the shape: the head is the record whose
/// frame the chain's later records build on, so a fallback here would hand the
/// packet a frame from a rail the class did not choose.
#[test]
fn an_in_class_chain_head_the_provider_refuses_is_a_typed_decline() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.render_pass_continues = true;
    req.indexed.as_mut().expect("indexed").content =
        BufferContent::Bytes(std::sync::Arc::new(INDEX_BYTES[..4].to_vec()));
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::Head), &req) {
        RenderRailOutcome::ProviderDeclined(decline) => assert_eq!(
            decline.slug(),
            "provider_capability",
            "the head's refusal is the provider's own class, under this rail's name: {decline}"
        ),
        RenderRailOutcome::ProviderCompleted(_) => {
            panic!("the provider completed a head whose index view is too short")
        }
        RenderRailOutcome::ProviderCompletedResident(_) => {
            panic!("a head with no target identity cannot keep a resident frame")
        }
        RenderRailOutcome::NotInNarrowClass(reason) => {
            panic!("an in-class chain head's refusal must not fall back to the engine: {reason}")
        }
    }
}

/// The class gate is pure: a request whose *pipeline* has no translated stages
/// is out of class before anything provider-side is touched, and the translated
/// registration gate is what a reused submission hits next.
#[test]
fn a_second_submission_reuses_the_registration() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    for round in 0..2 {
        match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
            RenderRailOutcome::ProviderCompleted(out) => {
                assert_solid(
                    &format!("round {round}"),
                    &semantic_rgba(out.bytes, out.bgra),
                );
            }
            other => panic!("round {round}: {other:?}"),
        }
    }
}

/// The mapping the provider's own classes get on this rail: one slug per
/// normalized class, so a fail line names the check without parsing prose.
#[test]
fn the_decline_slugs_name_the_provider_class() {
    let decline = ProviderRenderDecline::CompletionNotVisible;
    assert_eq!(decline.slug(), "completion_not_visible");
    let decline = ProviderRenderDecline::AttachmentWritebackMissing;
    assert_eq!(decline.slug(), "attachment_writeback_missing");
    assert_eq!(
        ProviderRenderDecline::TraceAdmission {
            detail: String::new()
        }
        .slug(),
        "trace_admission"
    );
    // The attachment format mapping is the class's own statement about which
    // canonical format each admitted attachment is, and both are in the
    // canonical admitted set.
    for format in [AttachmentFormat::Rgba8Unorm, AttachmentFormat::Bgra8Unorm] {
        assert!(
            AttachmentFormat::ADMITTED.contains(&format),
            "{format:?} is one of the admitted colour formats"
        );
    }
    assert!(VertexFormat::ADMITTED.contains(&VertexFormat::Float32x2));
}

/// The multi-stream half of the class, end to end.
///
/// Two streams, one attribute each: the vertex stage adds the second to the
/// first, so a rail that dropped the second stream — or bound the first one
/// twice — would draw the *reviewed* full-screen frame instead. The test
/// separates those two outcomes rather than only asserting that the rails agree,
/// because "both rails agree" is also what a rail that ignores its input says.
///
/// The offset is x-only on purpose. It leaves the shape symmetric under the NDC
/// y convention the two rails have never agreed about (`metal-api-vulkan`'s own
/// render fixture records the same split: Vulkan's y points down, Metal's points
/// up), so the parity asserted here is about the stream plumbing and not about
/// an axis this increment does not touch.
#[test]
fn a_two_stream_shape_draws_the_same_bytes_on_both_rails() {
    let _guard = engine_test_session();
    let stages = two_stream_stages();
    let (width, height) = extent();

    let reviewed_req = request_with_streams(
        MTL_FORMAT_RGBA8_UNORM,
        &[position_stream(), stream(1, &[(0.0, 0.0); 3])],
    );
    let reviewed = provider_pixels("two streams, zero offset", &stages, &reviewed_req);
    assert_solid("two streams, zero offset (provider)", &reviewed);
    let Some(engine_reviewed) = engine_pixels("two streams, zero offset", &stages, reviewed_req)
    else {
        return;
    };
    assert_eq!(
        reviewed, engine_reviewed,
        "the identity offset has to land the reviewed frame on both rails"
    );

    let shifted_req = request_with_streams(
        MTL_FORMAT_RGBA8_UNORM,
        &[position_stream(), stream(1, &[(0.25, 0.0); 3])],
    );
    let shifted = provider_pixels("two streams, shifted", &stages, &shifted_req);
    assert_texel_count("two streams, shifted (provider)", &shifted);
    assert_ne!(
        shifted, reviewed,
        "the second stream's bytes have to reach the vertex stage: with them zeroed the frame is \
         the reviewed one, and with an offset on them it is not"
    );
    for row in [0, height / 2, height - 1] {
        assert_clear_texel(
            &format!("shifted: texel (0, {row})"),
            texel_at(&shifted, 0, row),
        );
    }
    assert_texel_near(
        "shifted: texel (width - 1, 0)",
        texel_at(&shifted, width - 1, 0),
        FRAGMENT_TEXEL,
    );
    let Some(engine_shifted) = engine_pixels("two streams, shifted", &stages, shifted_req) else {
        return;
    };
    assert_eq!(
        shifted, engine_shifted,
        "the same two streams, drawn by the canonical provider and by the self-contained engine, \
         have to land the same bytes"
    );
}

/// The widest interface the canonical contract can state, and the first one it
/// cannot.
///
/// Three offsets add into one x-shift, so no stream is decorative: zeroing any
/// one of them moves the boundary, which is what makes "four streams were
/// fetched" falsifiable rather than asserted. The fifth stream is the contract's
/// own ceiling (`max_vertex_buffers`), and the gate has to answer it by name
/// *before* the provider is asked.
#[test]
fn four_vertex_streams_are_admitted_and_a_fifth_stays_on_the_engine() {
    let _guard = engine_test_session();
    let stages = four_stream_stages();
    let (width, _) = extent();
    let shifted = |a: f32, b: f32, c: f32| {
        [
            position_stream(),
            stream(1, &[(a, 0.0); 3]),
            stream(2, &[(b, 0.0); 3]),
            stream(3, &[(c, 0.0); 3]),
        ]
    };

    let reviewed_req =
        request_with_streams(MTL_FORMAT_RGBA8_UNORM, &shifted(0.125, 0.0625, 0.0625));
    let reviewed = provider_pixels("four streams", &stages, &reviewed_req);
    assert_texel_count("four streams (provider)", &reviewed);
    assert_clear_texel("four streams: texel (0, 0)", texel_at(&reviewed, 0, 0));
    assert_texel_near(
        "four streams: texel (width - 1, 0)",
        texel_at(&reviewed, width - 1, 0),
        FRAGMENT_TEXEL,
    );
    let Some(engine) = engine_pixels("four streams", &stages, reviewed_req) else {
        return;
    };
    assert_eq!(
        reviewed, engine,
        "four streams, drawn by the canonical provider and by the self-contained engine"
    );

    // One move per offset stream: each of them has to be fetched, because each
    // one changes where the triangle's edge lands.
    for (label, streams) in [
        ("the first offset zeroed", shifted(0.0, 0.0625, 0.0625)),
        ("the second offset zeroed", shifted(0.125, 0.0, 0.0625)),
        ("the third offset zeroed", shifted(0.125, 0.0625, 0.0)),
    ] {
        let moved = provider_pixels(
            label,
            &stages,
            &request_with_streams(MTL_FORMAT_RGBA8_UNORM, &streams),
        );
        assert_ne!(
            moved, reviewed,
            "{label}: the stream's own bytes have to reach the vertex stage"
        );
    }

    let delivered = provider_render::provider_submissions();
    let five = request_with_streams(
        MTL_FORMAT_RGBA8_UNORM,
        &[
            position_stream(),
            stream(1, &[(0.125, 0.0); 3]),
            stream(2, &[(0.0625, 0.0); 3]),
            stream(3, &[(0.0625, 0.0); 3]),
            stream(4, &[(0.0, 0.0); 3]),
        ],
    );
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &five) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_vertex_stream_limit",
                "a fifth stream is refused by name: {reason}"
            );
            assert!(
                reason.detail().contains("4"),
                "the sentence names the ceiling the draw crossed: {reason}"
            );
        }
        other => panic!("the contract states four streams and no more: {other:?}"),
    }
    assert_eq!(
        provider_render::provider_submissions(),
        delivered,
        "a layout the contract cannot state stays on the engine without the provider seeing it"
    );
}

/// The scissor half of the class, end to end.
///
/// A rectangle over half the attachment has to clip the canonical pass exactly
/// where it clips the engine's, and the texels outside it keep the load op's
/// bytes — the clear. An unscissored submission of the same shape lands a
/// different frame, which is what separates "the scissor was carried" from "the
/// scissor was ignored".
#[test]
fn a_scissor_clips_the_pass_the_same_way_on_both_rails() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();
    let half = width / 2;

    let mut scissored = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    scissored.scissors.push(ScissorResource {
        x: 0,
        y: 0,
        width: half,
        height,
    });
    let clipped = provider_pixels("scissored", &stages, &scissored);
    assert_texel_count("scissored (provider)", &clipped);
    assert_texel_near(
        "scissored: the last texel inside the rectangle",
        texel_at(&clipped, half - 1, height / 2),
        FRAGMENT_TEXEL,
    );
    assert_clear_texel(
        "scissored: the first texel outside the rectangle",
        texel_at(&clipped, half, height / 2),
    );
    assert_clear_texel(
        "scissored: the far corner",
        texel_at(&clipped, width - 1, height - 1),
    );

    let unscissored = provider_pixels(
        "unscissored",
        &stages,
        &narrow_request(MTL_FORMAT_RGBA8_UNORM),
    );
    assert_ne!(
        clipped, unscissored,
        "a rail that dropped the rectangle would draw the whole attachment"
    );

    let Some(engine) = engine_pixels("scissored", &stages, scissored) else {
        return;
    };
    assert_eq!(
        clipped, engine,
        "one rectangle, clipped by the canonical provider and by the self-contained engine"
    );
}

/// The scissor shapes the canonical pass cannot state stay on the engine, by
/// name, without the provider being asked.
///
/// Every refusal here is a shape the *engine* still answers — it clamps a
/// rectangle that reaches past the attachment and draws the intersection — so a
/// class that admitted them would be handing the guest a different frame rather
/// than a refusal.
#[test]
fn a_scissor_the_canonical_pass_cannot_state_stays_on_the_engine() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();
    let delivered = provider_render::provider_submissions();
    let refused = |req: &DrawRequest, slug: &str| match provider_render::submit_render(
        &inputs(&stages, RenderChainRole::SoleOrTail),
        req,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            assert_eq!(
                reason.slug(),
                slug,
                "the refusal names the condition: {reason}"
            );
            assert!(
                !reason.detail().is_empty(),
                "an out-of-class answer still carries its sentence"
            );
        }
        other => panic!("expected {slug} for this shape, got {other:?}"),
    };

    // One texel past the right edge: inside neither the contract's rule nor the
    // class's, while the engine draws the clamped intersection.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.scissors.push(ScissorResource {
        x: 0,
        y: 0,
        width: width + 1,
        height,
    });
    refused(&req, "render_provider_out_of_class_scissor");

    // An origin at the right edge: the rectangle's own numbers are inside the
    // attachment, and it still covers nothing.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.scissors.push(ScissorResource {
        x: width,
        y: 0,
        width: 1,
        height,
    });
    refused(&req, "render_provider_out_of_class_scissor");

    // A zero-area rectangle: "nothing landed" is a shape the contract refuses
    // and the engine executes as a draw that writes nothing.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.scissors.push(ScissorResource {
        x: 0,
        y: 0,
        width: 0,
        height,
    });
    refused(&req, "render_provider_out_of_class_scissor");

    // Two rectangles: the canonical pass carries one, and one cannot state the
    // other, so the engine draws this one.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.scissors.push(ScissorResource {
        x: 0,
        y: 0,
        width: half_of(width),
        height,
    });
    req.scissors.push(ScissorResource {
        x: half_of(width),
        y: 0,
        width: width - half_of(width),
        height,
    });
    refused(&req, "render_provider_out_of_class_scissor_count");

    assert_eq!(
        provider_render::provider_submissions(),
        delivered,
        "a rectangle the canonical pass cannot state stays on the engine without the provider \
         seeing it"
    );
}

fn half_of(width: u32) -> u32 {
    width / 2
}

/// The identity a resident test's target names, in the namespace the
/// mapper-ref-texture rail mints: a mapping id, the attachment's own geometry,
/// and the format the attachment renders in.
///
/// `TargetIdentity` is the *engine's* key, and it is also what the provider-side
/// pair is minted from (`provider_render::resident_attachment`) — so a test that
/// drives both rails through their own identities is driving the same guest
/// target through both.
fn surface_identity(id: u32) -> engine::TargetIdentity {
    let (width, height) = extent();
    engine::TargetIdentity::Surface {
        id,
        width,
        height,
        generation: 1,
        format: ash::vk::Format::R8G8B8A8_UNORM,
    }
}

/// The frame a resident seed leaves behind: a colour that is neither the
/// fragment stage's texel nor the *second* record's own clear value
/// ([`CLEAR`]), so "the resident load was skipped and the pass cleared instead"
/// cannot pass as "the frame stayed where it was".
const RESIDENT_SEED_CLEAR: [f64; 4] = [0.0, 1.0, 0.0, 1.0];
const RESIDENT_SEED_TEXEL: [u8; 4] = [0, 255, 0, 255];

/// The degenerate stream a resident seed draws with: three coincident vertices,
/// so the pass's `Clear` is what stays in the image.
///
/// The seed has to *not* draw over its own clear, or the chain's comparison
/// would be between two drawn frames: the reviewed full-screen triangle paints
/// the fragment's texel across the whole attachment, and the bytes the second
/// record fails to cover would then be that texel rather than the frame the
/// seed left. A zero-area triangle rasterizes nothing on both rails.
fn degenerate_stream() -> StreamSpec {
    stream(0, &[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
}

/// The seeding record of a resident chain, in the terms the GVA and
/// mapper-ref-texture rails build it: a byte-exact `Clear` whose draw covers
/// nothing, the request's own target identity, and the resident-store pair
/// (`skip_readback` with the reason recorded where the flag is set).
fn resident_seed_request(identity: &engine::TargetIdentity) -> DrawRequest {
    let mut req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, &[degenerate_stream()]);
    req.target_identity = Some(identity.clone());
    req.skip_readback = true;
    req.readback_skip_reason = ReadbackSkipReason::ResidentStore;
    req.color_attachment = Some(
        reims_vgpu::backend::vulkan::translate::pixel::color_attachment(MTL_FORMAT_RGBA8_UNORM)
            .expect("renderable")
            .0
            .with_clear(RESIDENT_SEED_CLEAR),
    );
    req
}

/// The record that composites onto a resident frame: `load_from_target` is the
/// engine's own spelling of "the previous contents are the live GPU image", and
/// the scissor covers half the attachment. `publishes` chooses between the
/// record that keeps its own frame (a chain intermediate) and the record whose
/// frame comes back through the completion (the shape a chain's last record
/// takes when its target is not one of the two deferring rails).
fn resident_load_request(identity: &engine::TargetIdentity, publishes: bool) -> DrawRequest {
    let (width, height) = extent();
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.target_identity = Some(identity.clone());
    req.load_from_target = true;
    req.color0_declared = Some(reims_vgpu::protocol::pass_action::LoadAction::Load);
    req.scissors.push(ScissorResource {
        x: 0,
        y: 0,
        width: half_of(width),
        height,
    });
    if !publishes {
        req.skip_readback = true;
        req.readback_skip_reason = ReadbackSkipReason::ResidentStore;
    }
    req
}

fn route_count(route: &str) -> u64 {
    reims_vgpu::runtime::drain::store_route_count_for_test(route)
}

/// The rail's answer for one resident-shaped request the class admitted.
fn declined(label: &str, stages: &Stages, req: &DrawRequest) -> ProviderRenderDecline {
    match provider_render::submit_render(&inputs(stages, RenderChainRole::SoleOrTail), req) {
        RenderRailOutcome::ProviderDeclined(decline) => decline,
        other => panic!("{label}: expected a typed decline, got {other:?}"),
    }
}

/// The provider's own slug and detail for a refusal, as the rail reports them.
fn refused_slug(label: &str, stages: &Stages, req: &DrawRequest) -> (String, String) {
    let decline = declined(label, stages, req);
    let fields = reims_vgpu::observe::Decline::fields(&decline);
    let field = |key: &str| {
        fields
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| value.clone())
            .unwrap_or_default()
    };
    (decline.slug().to_owned(), field("detail"))
}

/// R7b: the frame a resident store keeps, and the record that composites onto
/// it.
///
/// The seeding record declares `StoreOp::Resident` under the request's own
/// target identity, so it publishes nothing — the outcome carries no bytes at
/// all, and a provider that published a writeback for the attachment anyway
/// would be declined by name (`resident_writeback_published`). The second
/// record loads that image (`LoadOp::Resident`), covers half the attachment
/// with its scissor, and publishes the mixed frame. The comparison is the
/// engine's own two-record chain driven by the same requests: same AIR, same
/// geometry, the engine's resident in place of the provider's image.
#[test]
fn a_resident_store_keeps_the_frame_a_later_record_loads() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();
    let identity = surface_identity(0x7b_00_01);
    let attachment = provider_render::resident_attachment(&identity);
    let stores_before = route_count("render_provider_resident_store");
    let loads_before = route_count("render_provider_resident_load");

    // 1. The seeding record. `ProviderCompletedResident` *is* the assertion
    //    that nothing was published: there is no byte-carrying arm of it.
    let seed = resident_seed_request(&identity);
    let frame = match provider_render::submit_render(
        &inputs(&stages, RenderChainRole::SoleOrTail),
        &seed,
    ) {
        RenderRailOutcome::ProviderCompletedResident(frame) => frame,
        other => panic!("the resident seed is in class: {other:?}"),
    };
    assert_eq!(
        frame.attachment, attachment,
        "the frame stayed under the request's own identity"
    );
    assert!(
        !frame.loaded,
        "a record that clears cannot have loaded the image it keeps"
    );

    // 2. The record that composites onto it: the half outside the rectangle
    //    keeps the resident's own bytes, the half inside is the fragment's.
    let chained = resident_load_request(&identity, true);
    let provider = match provider_render::submit_render(
        &inputs(&stages, RenderChainRole::SoleOrTail),
        &chained,
    ) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("a resident load with a published store is in class: {other:?}"),
    };
    assert_texel_count("resident chain (provider)", &provider);
    assert_texel_near(
        "resident chain: the last texel inside the rectangle",
        texel_at(&provider, half_of(width) - 1, height / 2),
        FRAGMENT_TEXEL,
    );
    for x in half_of(width)..width {
        assert_eq!(
            texel_at(&provider, x, height / 2),
            RESIDENT_SEED_TEXEL,
            "texel ({x}, {}) keeps the resident's own bytes: a rail that cleared instead of \
             loading, or that loaded a stale image, lands another colour here",
            height / 2,
        );
    }

    // 3. The engine's chain, from the same two requests. Its second record
    //    loads the engine's own resident, which is what makes the comparison a
    //    statement about the two rails and not about one rail twice.
    let Some(engine_seed_pixels) =
        engine_pixels("resident seed", &stages, resident_seed_request(&identity))
    else {
        return;
    };
    assert!(
        engine_seed_pixels.is_empty(),
        "a resident store's readback is withheld on the engine too"
    );
    let Some(engine_chained) = engine_pixels(
        "resident chain",
        &stages,
        resident_load_request(&identity, true),
    ) else {
        return;
    };
    assert_eq!(
        provider, engine_chained,
        "the provider's resident chain and the engine's own resident chain land the same frame"
    );

    assert_eq!(
        route_count("render_provider_resident_store") - stores_before,
        1,
        "one submission kept its frame in the provider's image"
    );
    assert_eq!(
        route_count("render_provider_resident_load") - loads_before,
        1,
        "one submission began from the provider's image"
    );
}

/// R7b's chain position: the packet's *middle* record — a predecessor and a
/// successor — is now executed by the class when it both begins from and keeps
/// the provider's image, and the same record with no resident source keeps the
/// engine under the same name as before.
#[test]
fn a_resident_middle_record_is_admitted_and_a_source_less_one_is_not() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let identity = surface_identity(0x7b_00_02);

    // The seed first, so the image the middle loads exists.
    let seed = resident_seed_request(&identity);
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &seed) {
        RenderRailOutcome::ProviderCompletedResident(_) => (),
        other => panic!("the resident seed is in class: {other:?}"),
    }

    // A middle record that takes the frame from the provider's image and hands
    // its own frame on: nothing about it touches guest memory.
    let mut middle = resident_load_request(&identity, false);
    middle.continues_render_pass = true;
    middle.render_pass_continues = true;
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::Middle), &middle) {
        RenderRailOutcome::ProviderCompletedResident(frame) => assert!(
            frame.loaded,
            "the middle record both loaded the image and kept its own frame"
        ),
        other => panic!("a resident-sourced middle record is in class: {other:?}"),
    }

    // The same position without a resident source: refused by the same slug it
    // was refused by before R7b, because the class still cannot name the frame
    // the predecessor produced.
    let mut source_less = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    source_less.continues_render_pass = true;
    source_less.render_pass_continues = true;
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::Middle), &source_less) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_chain_middle",
            "a middle record with no resident source keeps the engine: {reason}"
        ),
        other => panic!("a source-less chain middle is out of class: {other:?}"),
    }
}

/// R7b's lifecycle answers are the *provider's* own, carried through the rail's
/// typed decline: an identity nothing stored, the least recently used identity
/// once the registry is over budget, and an identity asked for at another
/// geometry.
///
/// The two retirements this test cannot drive from reims — a lease release and a
/// device-epoch advance — are the provider's own suite's
/// (`metal-api-vulkan`'s `render_e2e.rs`), because the render rail's attachment
/// allocation is trace-owned rather than leased, and an epoch advance needs a
/// rebuilt provider. What this test pins is the wiring: reims does not re-name
/// the provider's answer.
#[test]
fn resident_lifecycle_failures_are_the_providers_own_names() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();

    // (a) A load for an identity nothing ever stored. The class admits the
    //     shape — it is the resident-load arm — and the *provider* answers.
    let never_stored = surface_identity(0x7b_00_03);
    let load = resident_load_request(&never_stored, true);
    let (slug, detail) = refused_slug("never stored", &stages, &load);
    assert_eq!(
        slug, "provider_capability",
        "an admitted resident load is refused at the provider boundary"
    );
    assert!(
        detail.contains("resident_target_unavailable"),
        "the provider's own name for an identity that holds no image rides along: {detail}"
    );

    // (b) The registry's budget: `budget + 2` identities leave the registry
    //     holding `budget`, and the identity it evicted is refused by name
    //     while the newest one still resolves.
    let budget = metal_api_vulkan::RESIDENT_TARGET_BUDGET;
    let identities: Vec<_> = (0..budget + 2)
        .map(|index| surface_identity(0x7b_01_00 + u32::try_from(index).expect("index fits")))
        .collect();
    for identity in &identities {
        let seed = resident_seed_request(identity);
        match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &seed) {
            RenderRailOutcome::ProviderCompletedResident(_) => (),
            other => panic!("every seeding store is in class: {other:?}"),
        }
    }
    let (slug, detail) = refused_slug(
        "evicted",
        &stages,
        &resident_load_request(&identities[0], true),
    );
    assert!(
        detail.contains("resident_target_evicted"),
        "the least recently used identity is refused as evicted: {slug} / {detail}"
    );
    let newest = resident_load_request(identities.last().expect("identities"), true);
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &newest) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("the most recent identity is still resident: {other:?}"),
    }

    // (c) The same identity at another geometry: the provider holds an image of
    //     one shape and is asked for another.
    let shaped = surface_identity(0x7b_02_00);
    let seed = resident_seed_request(&shaped);
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &seed) {
        RenderRailOutcome::ProviderCompletedResident(_) => (),
        other => panic!("the shaped seed is in class: {other:?}"),
    }
    let (width, height) = extent();
    let mut smaller = resident_load_request(&shaped, true);
    // The scissor is the load request's own half-attachment rectangle; a
    // smaller attachment makes it reach outside, which is a *class* answer and
    // would hide the shape change behind the scissor's own name.
    smaller.scissors.clear();
    smaller.width = width / 2;
    smaller.height = height / 2;
    let (slug, detail) = refused_slug("shape changed", &stages, &smaller);
    assert!(
        detail.contains("resident_target_shape_changed"),
        "a load at another geometry is refused as a changed shape: {slug} / {detail}"
    );
}

/// R7b's boundary, by name: the shapes *beside* the two resident arms keep the
/// engine, each under its own slug.
#[test]
fn the_shapes_beside_the_resident_arms_stay_on_the_engine_by_name() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let identity = surface_identity(0x7b_00_04);
    let out_of_class =
        |label: &str, req: &DrawRequest, slug: &str| match provider_render::submit_render(
            &inputs(&stages, RenderChainRole::SoleOrTail),
            req,
        ) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                assert_eq!(reason.slug(), slug, "{label}: {reason}")
            }
            other => panic!("{label}: expected an out-of-class answer, got {other:?}"),
        };

    // A store that publishes nothing: no resident, no reader, and no writeback
    // the caller could land. This is the arm R7b deliberately did *not* open.
    let mut unpublished = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    unpublished.skip_readback = true;
    unpublished.readback_skip_reason = ReadbackSkipReason::UnpublishedStore;
    out_of_class(
        "unpublished store",
        &unpublished,
        "render_provider_out_of_class_unpublished_store",
    );

    // A resident store with no identity to key the image on.
    let mut identity_less = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    identity_less.skip_readback = true;
    identity_less.readback_skip_reason = ReadbackSkipReason::ResidentStore;
    out_of_class(
        "identity-less resident store",
        &identity_less,
        "render_provider_out_of_class_resident_identity",
    );

    // Guest bytes as the previous contents, with and without a resident source
    // beside them: one load op per attachment, so the two cannot both be stated.
    let mut seed_only = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    seed_only.target_rgba8 = Some(std::sync::Arc::new(vec![0u8; 16]));
    seed_only.color0_declared = Some(reims_vgpu::protocol::pass_action::LoadAction::Load);
    out_of_class(
        "guest seed",
        &seed_only,
        "render_provider_out_of_class_load_seed",
    );
    let mut both_sources = resident_load_request(&identity, true);
    both_sources.target_rgba8 = Some(std::sync::Arc::new(vec![0u8; 16]));
    out_of_class(
        "both sources",
        &both_sources,
        "render_provider_out_of_class_load_source",
    );

    // A target identity this record neither loads from nor keeps.
    let mut name_only = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    name_only.target_identity = Some(identity);
    out_of_class(
        "identity without an image",
        &name_only,
        "render_provider_out_of_class_target",
    );
}
/// The vertex-interface half of the class: one stream per attribute the vertex
/// stage *reads*, and nothing else.
///
/// Both directions are refused, because both are shapes the canonical
/// registration gate refuses — and a refusal there is a decline, not a fallback,
/// so the class has to answer them before the provider is asked.
#[test]
fn a_stream_the_vertex_stage_does_not_read_stays_on_the_engine() {
    let _guard = engine_test_session();
    let delivered = provider_render::provider_submissions();
    let refused = |stages: &Stages, specs: &[StreamSpec]| {
        let req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, specs);
        match provider_render::submit_render(&inputs(stages, RenderChainRole::SoleOrTail), &req) {
            RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_vertex_interface",
                "the refusal names the interface and not the count: {reason}"
            ),
            other => panic!("a stream the stage does not read is out of class: {other:?}"),
        }
    };

    // Declared, not read: the reviewed one-attribute stage with a second stream,
    // which a real pipeline descriptor is free to state (the descriptor may
    // declare more than the shader consumes).
    refused(
        &reviewed_stages(),
        &[position_stream(), stream(1, &[(0.0, 0.0); 3])],
    );
    // Read, not declared: the two-stream stage with one stream bound, which
    // would leave the location the shader reads undefined.
    refused(&two_stream_stages(), &[position_stream()]);
    assert_eq!(
        provider_render::provider_submissions(),
        delivered,
        "a request whose streams disagree with its own reflection stays on the engine"
    );
}

/// The census names this increment's splits, and charges the band the vertex
/// widening is sized on.
#[test]
fn the_widening_splits_are_counted_under_their_own_names() {
    use reims_vgpu::backend::provider_render::attribute_count_route;

    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let count = |route: &str| reims_vgpu::runtime::drain::store_route_count_for_test(route);
    let submit = |inputs: &RenderRailInputs<'_>, req: &DrawRequest| {
        let _ = provider_render::submit_render(inputs, req);
    };

    // The band: charged by the gate it sizes, before any condition answers, so a
    // request the interface check refuses is still in its arm.
    let band_1 = count("draw_vertex_attrs_1");
    let band_2_4 = count("draw_vertex_attrs_2_4");
    let one = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    submit(&inputs(&stages, RenderChainRole::SoleOrTail), &one);
    assert_eq!(
        count("draw_vertex_attrs_1"),
        band_1 + 1,
        "the reviewed shape declares one stream"
    );
    let two = request_with_streams(
        MTL_FORMAT_RGBA8_UNORM,
        &[position_stream(), stream(1, &[(0.0, 0.0); 3])],
    );
    submit(&inputs(&stages, RenderChainRole::SoleOrTail), &two);
    assert_eq!(
        count("draw_vertex_attrs_2_4"),
        band_2_4 + 1,
        "a two-stream request is banded even when the vertex interface keeps it on the engine"
    );
    assert_eq!(attribute_count_route(0), "draw_vertex_attrs_0");
    assert_eq!(attribute_count_route(1), "draw_vertex_attrs_1");
    assert_eq!(attribute_count_route(4), "draw_vertex_attrs_2_4");
    assert_eq!(attribute_count_route(5), "draw_vertex_attrs_gt4");
    assert_eq!(
        metal_api_core::provider::MAX_VERTEX_BUFFERS,
        4,
        "the band's `_gt4` arm and the class's ceiling are the contract's own number: a contract \
         that moved has to rename both"
    );

    // The chain split: the record that opens a packet is in the class (W1), so
    // the bucket it used to answer under stops charging for it while a record
    // with a predecessor keeps its own name. The bucket the split replaced no
    // longer moves either.
    let head = count("render_provider_out_of_class_chain_head");
    let middle = count("render_provider_out_of_class_chain_middle");
    let retired = count("render_provider_out_of_class_writeback");
    let delivered = provider_render::provider_submissions();
    let mut first = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    first.render_pass_continues = true;
    submit(&inputs(&stages, RenderChainRole::Head), &first);
    assert_eq!(
        count("render_provider_out_of_class_chain_head"),
        head,
        "the packet's first record is in the class, so the bucket it used to answer under does \
         not move for it"
    );
    assert_eq!(
        provider_render::provider_submissions(),
        delivered + 1,
        "the packet's first record reached the provider rather than the engine"
    );
    let mut later = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    later.render_pass_continues = true;
    later.continues_render_pass = true;
    submit(&inputs(&stages, RenderChainRole::Middle), &later);
    assert_eq!(
        count("render_provider_out_of_class_chain_middle"),
        middle + 1,
        "a record after the first is a chain middle"
    );
    assert_eq!(
        count("render_provider_out_of_class_writeback"),
        retired,
        "the split retired the old bucket rather than charging it beside the two"
    );

    // The readback split, after R7b: a store that publishes nothing keeps its
    // own name, a skip with no recorded reason keeps the old one, and the
    // *resident* store — the population that used to answer
    // `render_provider_out_of_class_resident_store` — is now the rail's own
    // work. Its bucket is retired rather than re-charged, and the positive
    // counter the seam reads is charged once per admitted resident store.
    let unpublished = count("render_provider_out_of_class_unpublished_store");
    let retired_resident = count("render_provider_out_of_class_resident_store");
    let identity_less = count("render_provider_out_of_class_resident_identity");
    let unnamed = count("render_provider_out_of_class_skip_readback");
    let kept = count("render_provider_resident_store");
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.skip_readback = true;
    req.readback_skip_reason = ReadbackSkipReason::UnpublishedStore;
    submit(&inputs(&stages, RenderChainRole::SoleOrTail), &req);
    assert_eq!(
        count("render_provider_out_of_class_unpublished_store"),
        unpublished + 1,
        "a store that publishes nothing is its own population"
    );
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.skip_readback = true;
    req.readback_skip_reason = ReadbackSkipReason::ResidentStore;
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_resident_identity",
            "a resident store with no identity to key its image on is refused by that name: \
             {reason}"
        ),
        other => panic!("a resident store without an identity is out of class: {other:?}"),
    }
    assert_eq!(
        count("render_provider_out_of_class_resident_identity"),
        identity_less + 1,
        "the identity-less resident store answers under its own condition"
    );
    assert_eq!(
        count("render_provider_out_of_class_resident_store"),
        retired_resident,
        "the bucket the resident arms replaced does not move for the shapes they now execute"
    );
    let resident = resident_seed_request(&surface_identity(0x7b_00_05));
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &resident) {
        RenderRailOutcome::ProviderCompletedResident(_) => (),
        other => panic!("a resident store with an identity is in class: {other:?}"),
    }
    assert_eq!(
        count("render_provider_resident_store"),
        kept + 1,
        "the admitted resident store is counted under the rail's own positive name"
    );
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.skip_readback = true;
    submit(&inputs(&stages, RenderChainRole::SoleOrTail), &req);
    assert_eq!(
        count("render_provider_out_of_class_skip_readback"),
        unnamed + 1,
        "a skip with no recorded reason is reported as such rather than guessed into one of the two"
    );
}

/// The boundary R6 measured, now closed on this seam by adopting the device's
/// own capability answer (R8b).
///
/// R6 pinned where the rail *found* the boundary rather than pinning the
/// limitation: a vertex stage whose floating-point operation *withholds* a
/// fast-math permission is decorated `FPFastMathMode` by the translator
/// revision the canonical provider pins, which demands `FloatControls2` +
/// `SPV_KHR_float_controls2`. The seam translated the stages through the
/// translation entry point's Phase-1 default, so the module was refused with a
/// typed decline even though the condition is a property of the translation and
/// not of the request. The emulator answered the capability from the selected
/// device (R8: the extension *and* the feature bit, nothing else), and this seam
/// now translates with that answer (`provider.spirv_feature_policy()`), so the
/// same fixture has to *execute*.
///
/// What it has to land is the `fast` twin's own bytes: the withheld permission
/// must not change the drawn frame, only the module's capability set. The
/// request is the one R6 used — the second stream's offset is `(0.25, 0)`, so
/// the triangle's left edge sits at `-0.25` and the attachment's left column
/// keeps the clear sentinel while the right column carries the fragment's own
/// texel. A rail that dropped the withheld-permission module cannot pass by
/// drawing the reviewed frame either, because the twin's frame and the reviewed
/// frame differ (the offset shifts the covered columns).
///
/// The fail-closed arm R6 wrote is not deleted, it moved to its own test beside
/// this one: `a_device_without_the_feature_still_refuses_the_withheld_permission_module`
/// hands the translator the Phase-1 policy a device without the feature
/// derives, whatever this machine's device answers.
#[test]
fn a_float_op_that_withholds_a_permission_lands_the_fast_twins_bytes() {
    let _guard = engine_test_session();
    let fast = two_stream_stages();
    let precise = stages(
        "reims_indexed_tri_two_stream_precise.air",
        "reims_two_stream_vertex",
        &[0, 1],
    );
    let (width, height) = extent();
    let req = request_with_streams(
        MTL_FORMAT_RGBA8_UNORM,
        &[position_stream(), stream(1, &[(0.25, 0.0); 3])],
    );
    let reviewed_req = request_with_streams(
        MTL_FORMAT_RGBA8_UNORM,
        &[position_stream(), stream(1, &[(0.0, 0.0); 3])],
    );

    let fast_pixels = provider_pixels("fast twin", &fast, &req);
    let precise_pixels = provider_pixels("withheld float permission", &precise, &req);
    let reviewed_pixels = provider_pixels(
        "withheld float permission, zero offset",
        &precise,
        &reviewed_req,
    );
    // The reading, beside the assertions: the frame this module lands is what a
    // reader compares against the day the translator's decoration moves again.
    eprintln!(
        "withheld-permission module landed [{}] texels, left column {:02x?}, right column {:02x?}",
        precise_pixels.len() / 4,
        texel_at(&precise_pixels, 0, 0),
        texel_at(&precise_pixels, width - 1, 0)
    );
    assert_texel_count("withheld float permission", &precise_pixels);
    for row in [0, height / 2, height - 1] {
        assert_clear_texel(
            &format!("withheld permission: texel (0, {row})"),
            texel_at(&precise_pixels, 0, row),
        );
    }
    assert_texel_near(
        "withheld permission: texel (width - 1, 0)",
        texel_at(&precise_pixels, width - 1, 0),
        FRAGMENT_TEXEL,
    );
    assert_ne!(
        precise_pixels, reviewed_pixels,
        "the second stream's offset has to reach the drawn frame: without it the frame is the \
         reviewed one, which is what a rail that dropped the offset stream would land"
    );
    assert_eq!(
        precise_pixels, fast_pixels,
        "the withheld permission must not change the drawn bytes: the module has to land exactly \
         what its `fast` twin lands"
    );
    let Some(engine_frame) = engine_pixels("withheld float permission", &precise, req) else {
        return;
    };
    assert_eq!(
        precise_pixels, engine_frame,
        "the withheld-permission module, drawn by the canonical provider and by the \
         self-contained engine, has to land the same bytes"
    );
}

/// R8b's fail-closed arm, simulated rather than deleted: a device that reports
/// no `shaderFloatControls2` still gets the sentence R6 recorded.
///
/// The acceptance environment's device (Lavapipe) answers for the capability,
/// so this test does not wait for a no-feature device: it hands the translator
/// exactly the policy such a device derives — `SpirvFeaturePolicy::PHASE1`,
/// which is what `FloatControls2Support::policy()` returns while the extension
/// name or the feature bit is missing — and asserts the refusal is still the
/// *translation*'s own, capability number and subset sentence included. The
/// same module under this device's own answer has to translate instead, so the
/// test separates the two arms the way the rail does rather than only checking
/// that a refusal exists.
#[test]
fn a_device_without_the_feature_still_refuses_the_withheld_permission_module() {
    let executor = VulkanExecutor::new().expect("the acceptance environment has a Vulkan device");
    let device =
        Device::new(std::sync::Arc::clone(&executor) as std::sync::Arc<dyn ComputeExecutor>);
    let function = device
        .new_library_with_binary_air(fixture("reims_indexed_tri_two_stream_precise.air"))
        .expect("the precise fixture is a binary AIR module")
        .function("reims_two_stream_vertex")
        .expect("the fixture's entry exists");
    let error = match TranslatedRenderStage::translate_with_policy(
        RenderStage::Vertex,
        &function,
        SpirvFeaturePolicy::PHASE1,
    ) {
        Ok(_) => panic!("a device that does not answer for FloatControls2 refuses the module"),
        Err(error) => error,
    };
    let text = error.to_string();
    eprintln!("no-feature device refusal: {text}");
    assert!(
        text.contains("capability 6029"),
        "the refusal keeps the capability number R6 recorded: {text}"
    );
    assert!(
        text.contains("Phase 1 subset"),
        "the refusal keeps the translation's own sentence R6 recorded: {text}"
    );

    let provider = VulkanComputeProvider::with_executor(std::sync::Arc::clone(&executor))
        .expect("the canonical provider builds");
    let policy = provider.spirv_feature_policy();
    match TranslatedRenderStage::translate_with_policy(RenderStage::Vertex, &function, policy) {
        Ok(_) => assert!(
            policy.float_controls2(),
            "the device did not answer for FloatControls2 and the module translated anyway"
        ),
        Err(error) => assert!(
            !policy.float_controls2(),
            "the device answered for FloatControls2 and the module still failed: {error}"
        ),
    }
}
