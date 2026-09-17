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
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};
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
/// The fixtures behind these streams assemble their clip position by extraction
/// and insertion rather than by arithmetic — the canonical provider's translator
/// revision asks for `FloatControls2` on every floating-point operation and its
/// capability subset refuses it, so a shader with an `fadd` in it never reaches
/// this rail at all (the increment's report records that finding).
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

    // The two chain positions the class leaves to the engine, because both of
    // them begin from a frame this class cannot name: the packet's middle (a
    // predecessor and a successor) and the packet's last record, which begins
    // from the frame before it as well even though it owns the guest
    // writeback. The record that *opens* the pass is the third position and the
    // one W1 admits — see `a_chain_head_hands_its_frame_to_the_next_record`.
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

    // A declared Load keeps the engine: the class clears.
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

    // Blend state, a write mask, an explicit viewport and a resident target all
    // leave their rails to the engine.
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

    // The readback split: the two rails that skip a readback carry their own
    // names, and a skip with no recorded reason keeps the old one.
    let unpublished = count("render_provider_out_of_class_unpublished_store");
    let resident = count("render_provider_out_of_class_resident_store");
    let unnamed = count("render_provider_out_of_class_skip_readback");
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
    submit(&inputs(&stages, RenderChainRole::SoleOrTail), &req);
    assert_eq!(
        count("render_provider_out_of_class_resident_store"),
        resident + 1,
        "a frame in a resident the guest reads back through is the other one"
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

/// A boundary this widening measured, pinned where it was found.
///
/// The rail's promise is that everything it admits is a shape the provider
/// *executes*, and that a shape the provider refuses ends the draw rather than
/// silently falling back. There is one class of shape the gate cannot answer
/// before the provider does, because the condition is a property of the
/// translation rather than of the request: a vertex stage whose floating-point
/// operation *withholds* a fast-math permission is decorated `FPFastMathMode` by
/// the translator revision the canonical provider pins, which demands
/// `FloatControls2` + `SPV_KHR_float_controls2` — neither of which the provider's
/// Phase-1 capability subset admits. The two-stream fixture beside this test
/// carries the `fast` flag run a Metal module compiled with the default math mode
/// carries and executes; this one, the same module without it, is refused.
///
/// The assertions below pin the *boundary's shape* and not the limitation as a
/// requirement: `ProviderDeclined` is the fail-closed answer (the draw ends, the
/// engine is not run behind the guest's back), the decline names the provider
/// step that refused, and its detail carries the provider's own words. A
/// translator pin or capability-gate change that admits this module will fail
/// this test — which is the point of writing it down: the increment's report
/// names the follow-up rather than leaving the boundary to be rediscovered.
#[test]
fn a_float_op_that_withholds_a_permission_is_a_typed_decline() {
    let _guard = engine_test_session();
    let stages = stages(
        "reims_indexed_tri_two_stream_precise.air",
        "reims_two_stream_vertex",
        &[0, 1],
    );
    let req = request_with_streams(
        MTL_FORMAT_RGBA8_UNORM,
        &[position_stream(), stream(1, &[(0.25, 0.0); 3])],
    );
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::ProviderDeclined(decline) => {
            // The reading, beside the assertions: the provider's own words are
            // what a reader compares against the day this boundary moves.
            eprintln!(
                "withheld float permission: slug={} fields={:?}",
                decline.slug(),
                decline.fields()
            );
            assert_eq!(
                decline.slug(),
                "pipeline_compile",
                "the refusal is the translation step's own class: {decline}"
            );
            let fields = decline.fields();
            assert!(
                fields
                    .iter()
                    .any(|(key, value)| *key == "step" && value == "vertex_stage"),
                "the refusal names which stage refused: {fields:?}"
            );
            assert!(
                fields
                    .iter()
                    .any(|(key, value)| *key == "detail" && value.contains("Phase 1 subset")),
                "the provider's own words ride along: {fields:?}"
            );
        }
        RenderRailOutcome::ProviderCompleted(_) => panic!(
            "the canonical provider executed a module its capability subset refuses: the boundary \
             this test pins has moved, and the fixtures' own comment says so"
        ),
        RenderRailOutcome::NotInNarrowClass(reason) => panic!(
            "a withheld float permission is the translation's condition and not the gate's; a \
             class answer here means the gate learned something this test does not know: {reason}"
        ),
    }
}
