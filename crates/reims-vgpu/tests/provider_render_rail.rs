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

use metal_api_core::provider::{
    AttachmentFormat, ComputeProvider, FieldValue, RenderPipelineContract, RenderPipelineStage,
    SemanticDigest, VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, VertexStep,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, SpirvFeaturePolicy, TranslatedRenderPipelineRequest, TranslatedRenderStage,
    VulkanComputeProvider, VulkanExecutor,
};
use reims_vgpu::backend::provider_owner::{self, Region};
use reims_vgpu::backend::provider_render::{
    self, PresentSurfaceKey, ProviderRenderDecline, RenderChainRole, RenderPresentRequest,
    RenderRailInputs, RenderRailOutcome, StageBufferAccess, StageBufferBind,
    StageBufferDeclaration, StageBufferFootprint, StageBufferLanding, StageBufferWindow,
    StageWriteback,
};
use reims_vgpu::backend::vulkan::engine::{
    self, BlendStateResource, BufferContent, DepthState, DrawRequest, IndexType,
    IndexedDrawResource, PrimitiveTopology, ReadbackSkipReason, SamplerResource, ScissorResource,
    VertexAttributeFormat, VertexAttributeResource, VertexStepFunction, ViewportResource,
};
use reims_vgpu::observe::Decline as _;
use reims_vgpu::protocol::pixel_format::{
    MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA16_FLOAT, MTL_FORMAT_RGBA8_UNORM,
};
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
#[derive(Clone)]
struct Stages {
    air: (Vec<u8>, Vec<u8>),
    vertex_entry: &'static str,
    fragment_entry: &'static str,
    vertex_attribute_locations: Vec<u32>,
    /// The `[[buffer(N)]]` arguments each stage's reflection declares (R9b),
    /// the other reflected fact the class gate reads. Empty for every fixture
    /// whose stages are the reviewed `[[stage_in]]` shapes; the buffer fixtures
    /// below fill them from the translation itself, never by hand.
    vertex_stage_buffer_declarations: Vec<StageBufferDeclaration>,
    fragment_stage_buffer_declarations: Vec<StageBufferDeclaration>,
}

/// The reviewer's solid-colour fragment, shared by every vertex fixture here.
const FRAGMENT_ENTRY: &str = "fmain";

fn stages(vertex_fixture: &str, vertex_entry: &'static str, locations: &[u32]) -> Stages {
    Stages {
        air: (fixture(vertex_fixture), fixture("render_frag.air")),
        vertex_entry,
        fragment_entry: FRAGMENT_ENTRY,
        vertex_attribute_locations: locations.to_vec(),
        vertex_stage_buffer_declarations: Vec::new(),
        fragment_stage_buffer_declarations: Vec::new(),
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
    attachment_with_clear(format, CLEAR)
}

/// [`attachment`] with a test's own clear, because the wide arm's whole point is
/// that the clear is stated in the attachment's *own* format: 0.5 is a half and
/// is not a multiple of `1/255`, so the two arms admit different clears.
fn attachment_with_clear(
    format: u16,
    clear: [f64; 4],
) -> reims_vgpu::backend::vulkan::engine::ColorAttachmentState {
    reims_vgpu::backend::vulkan::translate::pixel::color_attachment(format)
        .expect("the fixture's attachment format is renderable")
        .0
        .with_clear(clear)
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
        vertex_stage_buffer_declarations: &stages.vertex_stage_buffer_declarations,
        fragment_stage_buffer_declarations: &stages.fragment_stage_buffer_declarations,
        // The R9d half: no stage buffer of this draw is stated, which is the
        // shape every pre-R9d test in this file is about — a request whose
        // stages declare nothing answers exactly as it did before.
        stage_buffer_binds: &[],
        present: None,
    }
}

/// The same inputs with the draw's own `[[buffer(N)]]` binds stated (R9d).
///
/// The seam builds this list from `vtx_storage` / `frag_storage`, one entry per
/// bound buffer at the Metal index of the stage that bound it; a test states the
/// same shape from the bytes it hands the request, because the two rails have to
/// read one set of bytes for a byte-for-byte comparison to mean anything.
fn inputs_with_binds<'a>(
    stages: &'a Stages,
    role: RenderChainRole,
    binds: &'a [StageBufferBind<'a>],
) -> RenderRailInputs<'a> {
    RenderRailInputs {
        stage_buffer_binds: binds,
        ..inputs(stages, role)
    }
}

/// One stage buffer as the seam states it: the bytes the owner holds, and no
/// registered window behind them.
fn staged_bind<'a>(
    stage: RenderPipelineStage,
    index: u32,
    content: &'a BufferContent,
) -> StageBufferBind<'a> {
    StageBufferBind {
        stage,
        index,
        content,
        window: None,
        // The read-only population R9d/R9e admitted: a read has no
        // destination to state (R9j), and the gate only asks for one when the
        // stage's own declaration is writable.
        landing: None,
    }
}

/// One *writable* stage buffer as the seam states it (R9j): the bytes the owner
/// holds, no window behind them, and the guest destination the completion's
/// writeback lands in.
fn writable_bind<'a>(
    stage: RenderPipelineStage,
    index: u32,
    content: &'a BufferContent,
    landing: StageBufferLanding<'a>,
) -> StageBufferBind<'a> {
    StageBufferBind {
        stage,
        index,
        content,
        window: None,
        landing: Some(landing),
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
    let out = provider_frame(label, stages, req);
    semantic_rgba(out.bytes, out.bgra)
}

/// The canonical rail's frame for one request **as the provider published it**:
/// the attachment's whole packed extent at the attachment's own texel width.
///
/// Separate from [`provider_pixels`] because a wide attachment's frame is not
/// eight-bit colour until the seam narrows it, and that narrowing is exactly
/// what the wide-texel test has to be able to see past.
fn provider_frame(
    label: &str,
    stages: &Stages,
    req: &DrawRequest,
) -> provider_render::RenderRailOutput {
    match provider_render::submit_render(&inputs(stages, RenderChainRole::SoleOrTail), req) {
        RenderRailOutcome::ProviderCompleted(out) => out,
        other => panic!("{label}: the canonical provider has to execute this shape: {other:?}"),
    }
}

/// One eight-byte attachment frame narrowed to the eight-bit colour the span's
/// consumers speak, by the protocol rule the seam itself applies
/// (`runtime/draw/vulkan`'s `provider_span_pixels`, counted as
/// `target_read_narrowed`) — so the provider's raw bytes can be compared with
/// the engine's own readback without the test re-deriving the quantization.
fn narrow_wide_frame(label: &str, frame: &[u8], texels: u32) -> Vec<u8> {
    let mut narrowed = vec![0u8; texels as usize * 4];
    assert!(
        reims_vgpu::protocol::pixel_format::narrow_texel_to_rgba8(
            reims_vgpu::protocol::pixel_format::TexelLayout::Rgba16Float,
            frame,
            texels,
            &mut narrowed,
        ),
        "{label}: a 16F frame narrows to the span's colour"
    );
    narrowed
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

/// One texel of a readback at whatever width the attachment published it, so a
/// wide frame's eight bytes can be read as directly as a narrow frame's four.
fn texel_of(pixels: &[u8], width: u32, x: u32, y: u32, bytes_per_texel: usize) -> Vec<u8> {
    let offset = (usize::try_from(y * width + x).expect("the extent fits usize")) * bytes_per_texel;
    pixels[offset..offset + bytes_per_texel].to_vec()
}

/// Two frames compared byte for byte, reporting the *first* byte that differs
/// rather than printing two whole attachments (which is 16 KiB of hex at the
/// declared window and buries the reading).
fn assert_frames_equal(label: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{label}: the frames' extents");
    if let Some((index, (got, want))) = got
        .iter()
        .zip(want)
        .enumerate()
        .find(|(_, (got, want))| got != want)
    {
        let texel = index / 4;
        panic!(
            "{label}: byte {index} (texel {texel}, channel {}) is {got:#04x}, expected \
             {want:#04x}",
            index % 4,
        );
    }
}

/// Two frames asserted to be *different* frames, at the first byte that differs
/// (a `assert_ne!` on two attachments would print both whole, which is the same
/// 16 KiB of hex the equality helper avoids).
fn assert_frames_differ(label: &str, got: &[u8], want: &[u8]) {
    if got.len() == want.len() && got == want {
        panic!(
            "{label}: the frames are the same frame ({} bytes)",
            got.len()
        );
    }
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

    // A storage buffer, a sampled image and an occlusion query. The buffer arm
    // moved twice: R9b admitted a *bind* no stage declares (its bytes cannot
    // reach either rail's frame — see
    // `stage_buffers_no_stage_declares_leave_for_the_provider_and_agree_with_the_engine`),
    // and R9d admits a stage's own read-only declaration once the request
    // states the bind (`a_read_only_stage_buffer_leaves_for_the_provider_and_
    // agrees_with_the_engine`). What stays on the engine is the declaration the
    // request leaves unbound, which is the arm pinned here — so the pair cannot
    // drift back into "buffers are out of class".
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.storage_buffers.push(engine::StorageBufferResource {
        binding: 0,
        content: BufferContent::Bytes(std::sync::Arc::new(vec![0u8; 16])),
    });
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("a bind no stage declares is in class: {other:?}"),
    }
    let declared = buffer_declaring_stages("render_frag_buffer.air", "reims_buffer_frag");
    match provider_render::submit_render(&inputs(&declared, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_stage_buffer_unbound",
            "a declared buffer argument no bind fills stays on the engine: {reason}"
        ),
        other => panic!("a declared stage buffer without a bind is out of class: {other:?}"),
    }
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

/// The wide-texel half of the class, end to end.
///
/// An `Rgba16Float` attachment is one **eight-byte** texel — four little-endian
/// halves, `research/docs/23` §78 — and the canonical provider publishes it as
/// such. That is what makes the half rounding falsifiable rather than asserted:
/// the bytes pinned below are the driver's own rounding of the fragment stage's
/// `float32` output, and of a clear the guest stated as a float, into the
/// attachment's format — so a rail that read the attachment as four bytes, or
/// spelled its clear as four, cannot produce them.
///
/// The shape is the resident chain of
/// `a_resident_store_keeps_the_frame_a_later_record_loads`, at 16F: a record
/// that clears a green frame into the provider's own image and keeps it there,
/// then a record that loads that image, covers half the attachment with the
/// fragment's texel and *publishes* the mixed frame. The second record is the
/// wide shape this class reads back at its own width, and it is also one the
/// engine's own wide rail draws — a *resident* image is created at the
/// attachment's format, unlike the pooled one
/// (`a_pooled_wide_attachment_stays_on_the_engine_by_name`) — which is what
/// makes the byte comparison a statement about the two rails and not about one
/// rail twice.
#[test]
fn a_rgba16float_attachment_travels_at_its_own_width_and_agrees_with_the_engine() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();
    let half = half_of(width);
    let texels = width * height;
    let identity = wide_surface_identity(0x7a_16_00_01);

    // The seed: a clear every component of which a half holds exactly, kept in
    // the provider's own image under the request's own identity. The outcome
    // carries no bytes — that *is* the resident-store arm.
    let seed = wide_resident_seed_request(&identity);
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &seed) {
        RenderRailOutcome::ProviderCompletedResident(frame) => assert!(
            !frame.loaded,
            "a record that clears cannot have loaded the image it keeps"
        ),
        other => panic!("a 16F resident seed is in class: {other:?}"),
    }

    // The record that composites onto it and publishes what it drew.
    let delivered = provider_render::provider_submissions();
    let chained = wide_resident_load_request(&identity, true);
    let frame = provider_frame("rgba16float", &stages, &chained);
    assert_eq!(
        provider_render::provider_submissions(),
        delivered + 1,
        "the wide shape reached the canonical provider rather than staying on the engine"
    );
    assert!(
        !frame.bgra,
        "the wide arm is RGBA in its own halves; only the 8-bit orders carry a scanout order"
    );
    assert_eq!(
        frame.bytes.len(),
        texels as usize * 8,
        "the frame is the attachment's whole packed extent at eight bytes per texel"
    );

    // A texel the second record never covered: the *seed's* own halves, green as
    // `0x0000, 0x3c00, 0x0000, 0x3c00`. A rail that read the resident's bytes as
    // four-byte texels cannot produce this, and neither can one that wrote a
    // four-byte clear into it.
    assert_eq!(
        texel_of(&frame.bytes, width, width - 1, height / 2, 8),
        [0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x3c],
        "a texel outside the rectangle keeps the resident's own halves"
    );
    // A texel the second record covered: the half rounding of the fragment's own
    // output. The fixture writes `float4(0.25, 0.5, 0.75, 1)` — four values a
    // half holds exactly, so these bytes are that colour with no step between,
    // and they are *halves* rather than the four bytes of an 8-bit order.
    assert_eq!(
        texel_of(&frame.bytes, width, half - 1, height / 2, 8),
        [0x00, 0x34, 0x00, 0x38, 0x00, 0x3a, 0x00, 0x3c],
        "a covered texel is the fragment's float4(0.25, 0.5, 0.75, 1) as four little-endian \
         halves (0x3400, 0x3800, 0x3a00, 0x3c00)"
    );

    // Narrowed to the eight-bit colour the span's consumers read, the frame is
    // exactly the engine's own readback of the same two-record chain: the
    // engine's resident is created at the attachment's own format, and its
    // readback quantizes a wide resident with the same protocol rule.
    let narrowed = narrow_wide_frame("rgba16float", &frame.bytes, texels);
    assert_texel_count("rgba16float narrowed", &narrowed);
    for x in half..width {
        assert_eq!(
            texel_at(&narrowed, x, height / 2),
            RESIDENT_SEED_TEXEL,
            "texel ({x}, {}) is the seed's green narrowed: a rail that cleared instead of \
             loading, or that loaded a stale image, lands another colour here",
            height / 2,
        );
    }
    assert_texel_near(
        "rgba16float narrowed: the last texel inside the rectangle",
        texel_at(&narrowed, half - 1, height / 2),
        FRAGMENT_TEXEL,
    );
    let Some(engine_seed) = engine_pixels(
        "wide resident seed",
        &stages,
        wide_resident_seed_request(&identity),
    ) else {
        return;
    };
    assert!(
        engine_seed.is_empty(),
        "a resident store's readback is withheld on the engine too"
    );
    let Some(engine) = engine_pixels(
        "wide resident chain",
        &stages,
        wide_resident_load_request(&identity, true),
    ) else {
        return;
    };
    assert_frames_equal(
        "rgba16float narrowed against the engine",
        &narrowed,
        &engine,
    );

    // The narrow order is untouched by all of this: the same shape at eight bits
    // still publishes four-byte texels, and those bytes are still the engine's.
    let narrow_req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    let narrow = provider_frame("rgba8", &stages, &narrow_req);
    assert!(
        !narrow.bgra,
        "the reviewed attachment is the RGBA order of the two 8-bit ones"
    );
    assert_eq!(
        narrow.bytes.len(),
        texels as usize * 4,
        "the 8-bit arm's frame is still four bytes per texel"
    );
    let Some(narrow_engine) = engine_pixels("rgba8", &stages, narrow_req) else {
        return;
    };
    assert_frames_equal("rgba8 against the engine", &narrow.bytes, &narrow_engine);

    // Neither arm of the clear rule is the other's relaxation, and both answer
    // before the provider is asked.
    let delivered = provider_render::provider_submissions();
    let mut half_only_clear_on_eight_bits = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    half_only_clear_on_eight_bits.color_attachment = Some(attachment_with_clear(
        MTL_FORMAT_RGBA8_UNORM,
        [0.0, 0.0, 0.0, 0.5],
    ));
    match provider_render::submit_render(
        &inputs(&stages, RenderChainRole::SoleOrTail),
        &half_only_clear_on_eight_bits,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_clear_bytes",
            "0.5 is exact in half and not a multiple of 1/255: {reason}"
        ),
        other => panic!("a half-only clear on a narrow attachment is out of class: {other:?}"),
    }
    let mut eight_bit_only_clear_on_wide =
        request_with_streams(MTL_FORMAT_RGBA16_FLOAT, &position_streams());
    eight_bit_only_clear_on_wide.target_identity = Some(identity.clone());
    eight_bit_only_clear_on_wide.skip_readback = true;
    eight_bit_only_clear_on_wide.readback_skip_reason = ReadbackSkipReason::ResidentStore;
    eight_bit_only_clear_on_wide.color_attachment = Some(attachment_with_clear(
        MTL_FORMAT_RGBA16_FLOAT,
        [64.0 / 255.0, 128.0 / 255.0, 191.0 / 255.0, 1.0],
    ));
    match provider_render::submit_render(
        &inputs(&stages, RenderChainRole::SoleOrTail),
        &eight_bit_only_clear_on_wide,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_clear_bytes",
            "an eight-bit-exact texel is not a half-exact one: {reason}"
        ),
        other => panic!("a clear the half cannot hold is out of class: {other:?}"),
    }
    assert_eq!(
        provider_render::provider_submissions(),
        delivered,
        "both clear-rule refusals stay on the engine without the provider seeing them"
    );
}

/// The pooled wide attachment: a shape the *engine* cannot draw.
///
/// The engine's pooled offscreen target is one image at
/// `translate::pixel::RESIDENT_RGBA_FORMAT` — its `TargetKey` carries no format
/// — while the render pass is keyed on the attachment's own. At eight bytes per
/// texel the pass therefore lands nothing, which this test measures rather than
/// assumes. A class that admitted the shape would hand the guest a frame the
/// engine never drew, so it stays on the engine by name; the day the pooled arm
/// follows the attachment's format, the last assertion here fails and the guard
/// can be deleted.
#[test]
fn a_pooled_wide_attachment_stays_on_the_engine_by_name() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();

    let delivered = provider_render::provider_submissions();
    let pooled = request_with_streams(MTL_FORMAT_RGBA16_FLOAT, &position_streams());
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &pooled) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_wide_pooled",
            "a pooled wide attachment is a shape the engine does not draw: {reason}"
        ),
        other => panic!("a pooled wide attachment stays on the engine: {other:?}"),
    }
    assert_eq!(
        provider_render::provider_submissions(),
        delivered,
        "the refusal answers before the provider is asked"
    );

    // The frame the *resident* arm lands for the same draw, on the provider's
    // side of this file: the previous test pins that arm's byte parity with the
    // engine. What the engine answers for the *pooled* shape is a different
    // frame — the measurement the guard rests on.
    let identity = wide_surface_identity(0x7a_16_00_02);
    let seed = wide_resident_seed_request(&identity);
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &seed) {
        RenderRailOutcome::ProviderCompletedResident(_) => (),
        other => panic!("the 16F seed is in class: {other:?}"),
    }
    let resident = provider_frame(
        "wide resident",
        &stages,
        &wide_resident_load_request(&identity, true),
    );
    let resident_narrowed =
        narrow_wide_frame("wide resident", &resident.bytes, extent().0 * extent().1);
    let Some(pooled_engine) = engine_pixels("pooled wide", &stages, pooled) else {
        return;
    };
    assert_frames_differ(
        "the pooled wide frame against the resident one",
        &pooled_engine,
        &resident_narrowed,
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

/// The same identity at the **wide** attachment's format.
///
/// The engine's resident image is created from the request's attachment, while
/// the identity is what both rails key that image on — so a 16F resident has to
/// name 16F here, exactly as a guest's own `TargetIdentity` would.
fn wide_surface_identity(id: u32) -> engine::TargetIdentity {
    let (width, height) = extent();
    engine::TargetIdentity::Surface {
        id,
        width,
        height,
        generation: 1,
        format: ash::vk::Format::R16G16B16A16_SFLOAT,
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

/// [`resident_seed_request`] at the wide attachment's own format: the same
/// degenerate stream, the same resident-store pair, and a clear every component
/// of which a half holds exactly ([`RESIDENT_SEED_CLEAR`]).
fn wide_resident_seed_request(identity: &engine::TargetIdentity) -> DrawRequest {
    let mut req = request_with_streams(MTL_FORMAT_RGBA16_FLOAT, &[degenerate_stream()]);
    req.target_identity = Some(identity.clone());
    req.skip_readback = true;
    req.readback_skip_reason = ReadbackSkipReason::ResidentStore;
    req.color_attachment = Some(attachment_with_clear(
        MTL_FORMAT_RGBA16_FLOAT,
        RESIDENT_SEED_CLEAR,
    ));
    req
}

/// [`resident_load_request`] at the wide attachment's own format.
fn wide_resident_load_request(identity: &engine::TargetIdentity, publishes: bool) -> DrawRequest {
    let (width, height) = extent();
    let mut req = request_with_streams(MTL_FORMAT_RGBA16_FLOAT, &position_streams());
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

/// The y-asymmetric fixture the convention pair draws: a triangle whose own top
/// half is covered — `(-1, 0)`, `(1, 0)`, `(-1, 1)` in the guest's clip space —
/// into an 8x4 attachment.
///
/// Every edge misses the pixel centres: the horizontal edge sits at `y = 0`
/// between two centre rows, the vertical one at `x = -1` outside the first
/// centre column, and the diagonal `y = (1 - x) / 2` lands on eighths and
/// sixteenths no centre (a quarter in `y`, an eighth in `x`) shares. So the
/// coverage of both rails is the rasterizer's own answer with no edge-tie rule
/// in it — which is what makes the expectation below a statement about the y
/// convention and about nothing else.
const ASYMMETRIC_VERTICES: [(f32, f32); 3] = [(-1.0, 0.0), (1.0, 0.0), (-1.0, 1.0)];

/// The fixture's attachment, small enough to read a frame row by row in a log.
const ASYMMETRIC_WIDTH: u32 = 8;
const ASYMMETRIC_HEIGHT: u32 = 4;

/// Whether the pixel centre `point` is inside the triangle `vertices`, by the
/// three signed edge functions a rasterizer evaluates for a centre.
///
/// The assertion in the middle is the fixture's own precondition: a centre *on*
/// an edge would be decided by the rasterizer's tie rule rather than by this
/// function, and the pair would be pinning a tie rule instead of a convention.
fn point_in_triangle(vertices: [(f32, f32); 3], point: (f64, f64)) -> bool {
    let edge = |a: (f32, f32), b: (f32, f32)| -> f64 {
        let (ax, ay) = (f64::from(a.0), f64::from(a.1));
        let (bx, by) = (f64::from(b.0), f64::from(b.1));
        (bx - ax) * (point.1 - ay) - (by - ay) * (point.0 - ax)
    };
    let edges = [
        edge(vertices[0], vertices[1]),
        edge(vertices[1], vertices[2]),
        edge(vertices[2], vertices[0]),
    ];
    for value in edges {
        assert!(
            value.abs() > 1.0e-6,
            "the fixture's centre {point:?} sits on an edge ({value}): the tie rule, not the y \
             convention, would decide this texel"
        );
    }
    edges.iter().all(|value| *value > 0.0) || edges.iter().all(|value| *value < 0.0)
}

/// The same frame with its rows reversed: what the identical NDC vertices
/// produce when the attachment is rasterized with `+Y` down (Vulkan's own clip
/// space) instead of Metal's `+Y` up.
fn flip_rows(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let row = (width * 4) as usize;
    let mut out = Vec::with_capacity(pixels.len());
    for index in (0..height).rev() {
        let start = index as usize * row;
        out.extend_from_slice(&pixels[start..start + row]);
    }
    out
}

/// The y-asymmetric draw: the reviewed request shape — one `float2` stream, one
/// index stream, `Clear` plus writeback — at the fixture's own small extent,
/// with the fixture's vertices in the stream the pass-through vertex stage
/// forwards.
fn asymmetric_request() -> DrawRequest {
    let specs = [StreamSpec {
        location: 0,
        offset: 0,
        stride: 8,
        bytes: f32x2(&ASYMMETRIC_VERTICES),
    }];
    let mut request = request_with_streams(MTL_FORMAT_RGBA8_UNORM, &specs);
    request.width = ASYMMETRIC_WIDTH;
    request.height = ASYMMETRIC_HEIGHT;
    request
}

/// One frame against **Metal's own mapping**, derived from the guest's clip-space
/// vertices instead of from either rail: `+Y` up (a vertex at `y = +1` is in row
/// 0), fragment colour on a covered centre, the clear's own bytes elsewhere, and
/// both populations present.
///
/// This is the adjudicating half of the pair: it reads neither the engine's
/// viewport nor the provider's, so a frame that matches it is the Metal answer
/// whatever the other rail does.
fn assert_frame_is_the_metal_mapping(label: &str, frame: &[u8], vertices: [(f32, f32); 3]) {
    assert_eq!(
        frame.len(),
        (ASYMMETRIC_WIDTH * ASYMMETRIC_HEIGHT * 4) as usize,
        "{label}: the attachment's whole extent has to come back"
    );
    let mut covered = 0;
    let mut cleared = 0;
    for row in 0..ASYMMETRIC_HEIGHT {
        // Metal's row 0 is the top of the attachment, where `y = +1` maps.
        let y = 1.0 - (2.0 * f64::from(row) + 1.0) / f64::from(ASYMMETRIC_HEIGHT);
        for column in 0..ASYMMETRIC_WIDTH {
            let x = -1.0 + (2.0 * f64::from(column) + 1.0) / f64::from(ASYMMETRIC_WIDTH);
            let texel = texel_of(frame, ASYMMETRIC_WIDTH, column, row, 4);
            let texel = [texel[0], texel[1], texel[2], texel[3]];
            let label = format!("{label}: texel ({column}, {row})");
            if point_in_triangle(vertices, (x, y)) {
                covered += 1;
                assert_texel_near(&label, texel, FRAGMENT_TEXEL);
            } else {
                cleared += 1;
                assert_clear_texel(&label, texel);
            }
        }
    }
    assert!(
        covered > 0 && cleared > 0,
        "{label}: the shape has to cover part of the attachment and leave part of it ({covered} \
         covered, {cleared} cleared), or the frame says nothing about a y-asymmetric draw"
    );
}

/// The self-contained engine draws the y-asymmetric fixture the way Metal's clip
/// space describes it.
///
/// This half is the adjudication and never mentions the canonical rail: the
/// expectation is Metal's mapping, so a frame that matches it is the Metal answer
/// whatever produced it. The engine states that mapping in its viewport — a
/// negative height with the origin on the bottom edge
/// (`reims-vgpu-vulkan/src/raster.rs`) — which is the one place `raster.rs`'s
/// module doc allows the flip to live.
#[test]
fn the_engines_asymmetric_frame_is_the_metal_ndc_mapping() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let Some(engine) = engine_pixels("ndc-y fixture", &stages, asymmetric_request()) else {
        return;
    };
    assert_frame_is_the_metal_mapping("ndc-y fixture (engine)", &engine, ASYMMETRIC_VERTICES);
    assert_ne!(
        engine,
        flip_rows(&engine, ASYMMETRIC_WIDTH, ASYMMETRIC_HEIGHT),
        "the fixture's own frame has to differ from its rows reversed: a shape whose two \
         readings agree would make this pair's expectation vacuous"
    );
}

// ---------------------------------------------------------------------------
// R4b: the present tail, and the frame it hands the display rail
// ---------------------------------------------------------------------------

/// The inputs of one presenting submission: [`inputs`] plus the surface the
/// record says its frame lands in.
fn present_inputs<'a>(
    stages: &'a Stages,
    role: RenderChainRole,
    surface: PresentSurfaceKey,
) -> RenderRailInputs<'a> {
    RenderRailInputs {
        present: Some(RenderPresentRequest { surface }),
        ..inputs(stages, role)
    }
}

/// One presenting submission of the reviewed shape, returned as the canonical
/// rail's own completion.
fn submitted_present(
    label: &str,
    stages: &Stages,
    surface: PresentSurfaceKey,
) -> provider_render::RenderRailOutput {
    let req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    match provider_render::submit_render(
        &present_inputs(stages, RenderChainRole::SoleOrTail, surface),
        &req,
    ) {
        RenderRailOutcome::ProviderCompleted(out) => out,
        other => panic!("{label}: the canonical provider has to present this shape: {other:?}"),
    }
}

/// R4b: a presenting record's frame is the provider's *present target* readback,
/// and it is the same picture the pooled arm lands.
///
/// The whole claim of the present rail is that nothing about the frame changes:
/// the pass renders the same shape into the provider's own target, the target
/// is acquired and presented once, and its bytes come back through the same
/// completion channel. So the readings are a pair — the presenting frame against
/// the pooled frame, byte for byte — plus the provider's own counters, which are
/// the only thing that can say the present action ran at all. The second
/// reversal is the control: the first present of a surface creates one target,
/// presenting it again reuses that target, and a *different* mapping is a second
/// target — the registry is keyed by the surface's own identity, not by the
/// submission.
#[test]
fn a_presenting_record_lands_the_present_targets_own_frame() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();

    // The frame the same shape lands without a present tail.
    let pooled = provider_pixels(
        "pooled frame",
        &stages,
        &narrow_request(MTL_FORMAT_RGBA8_UNORM),
    );

    let surface = PresentSurfaceKey {
        mapping_id: 0x4b_00_01,
        map_generation: 1,
    };
    let targets_before = provider_render::present_target_count();
    let counts_before = provider_render::present_counts();
    let out = submitted_present("first present", &stages, surface);
    let present = out
        .present
        .expect("a present-bearing completion reports the tail it executed");
    assert_eq!(
        (present.acquires, present.presents),
        (1, 1),
        "the provider acquired the target once and presented it once"
    );
    let expected_target = provider_render::present_attachment(
        &surface,
        AttachmentFormat::Rgba8Unorm,
        u64::from(width),
        u64::from(height),
    );
    assert_eq!(
        present.attachment, expected_target,
        "the target the completion names is the surface's own mint"
    );
    let presented = semantic_rgba(out.bytes.clone(), out.bgra);
    assert_solid("presented frame", &presented);
    assert_eq!(
        presented, pooled,
        "the present tail changes which image the frame comes out of and nothing about the frame"
    );
    let Some(engine) = engine_pixels(
        "presented frame",
        &stages,
        narrow_request(MTL_FORMAT_RGBA8_UNORM),
    ) else {
        return;
    };
    assert_eq!(
        presented, engine,
        "the presented frame is the frame the self-contained engine draws for the same shape"
    );
    let counts_after = provider_render::present_counts();
    assert_eq!(
        (
            counts_after.0.saturating_sub(counts_before.0),
            counts_after.1.saturating_sub(counts_before.1),
        ),
        (1, 1),
        "the submission drove exactly one acquire and one present"
    );
    assert_eq!(
        provider_render::present_target_count(),
        targets_before + 1,
        "the first present of a surface creates exactly one provider target"
    );

    // The same surface again: one more round trip on the *same* target.
    let again = submitted_present("second present", &stages, surface);
    assert_eq!(
        again
            .present
            .expect("the second present reports its tail")
            .attachment,
        expected_target,
        "a second present of one surface incarnation names the same target"
    );
    assert_eq!(
        provider_render::present_target_count(),
        targets_before + 1,
        "and the provider reused the image instead of minting a second one"
    );
    assert_eq!(
        semantic_rgba(again.bytes, again.bgra),
        pooled,
        "the reused target lands the same frame"
    );

    // Another mapping at the same geometry: another target. This is the
    // rubber-band residue class `present_identity.rs` records — two guest
    // surfaces presenting out of one provider image would fuse their damage
    // histories.
    let neighbour = PresentSurfaceKey {
        mapping_id: 0x4b_00_02,
        map_generation: 1,
    };
    let neighbour_out = submitted_present("neighbour surface", &stages, neighbour);
    assert_ne!(
        neighbour_out
            .present
            .expect("the neighbour present reports its tail")
            .attachment,
        expected_target,
        "two guest surfaces never present out of one target"
    );
    assert_eq!(
        provider_render::present_target_count(),
        targets_before + 2,
        "the neighbour's first present is a second target"
    );
}

/// R4b's identity rule, as a property of the mint itself.
///
/// One target per **surface incarnation**: the same `(mapping, generation)` at
/// the same shape resolves to the same `(allocation, view)` every time, and each
/// of the four ways a surface can be *another* surface — another mapping,
/// another generation, another extent, another format — resolves to a different
/// one. The last assertion is the namespace half: a present target and the
/// resident a render record keeps for the same guest target must be two
/// identities, because they are two provider registries
/// (`present_targets` / `resident_targets`) and one allocation naming both would
/// make one registry's eviction retire the other's image.
#[test]
fn the_present_identity_follows_the_surface_incarnation_and_its_shape() {
    let surface = PresentSurfaceKey {
        mapping_id: 0x4b_01_00,
        map_generation: 3,
    };
    let base = provider_render::present_attachment(&surface, AttachmentFormat::Rgba8Unorm, 64, 32);
    assert_eq!(
        base,
        provider_render::present_attachment(&surface, AttachmentFormat::Rgba8Unorm, 64, 32),
        "the mint is a function of the surface and its shape"
    );
    let other_mapping = provider_render::present_attachment(
        &PresentSurfaceKey {
            mapping_id: 0x4b_01_01,
            map_generation: 3,
        },
        AttachmentFormat::Rgba8Unorm,
        64,
        32,
    );
    assert_ne!(base, other_mapping, "two mappings are two surfaces");
    let other_generation = provider_render::present_attachment(
        &PresentSurfaceKey {
            mapping_id: 0x4b_01_00,
            map_generation: 4,
        },
        AttachmentFormat::Rgba8Unorm,
        64,
        32,
    );
    assert_ne!(
        base, other_generation,
        "a re-mapped surface is another buffer, not another present of the old image"
    );
    let other_extent =
        provider_render::present_attachment(&surface, AttachmentFormat::Rgba8Unorm, 64, 33);
    assert_ne!(base, other_extent, "the target's extent is part of its key");
    let other_format =
        provider_render::present_attachment(&surface, AttachmentFormat::Bgra8Unorm, 64, 32);
    assert_ne!(base, other_format, "the target's format is part of its key");

    // The resident mint is the other registry's namespace: same guest target,
    // different provider image.
    let resident = provider_render::resident_attachment(&engine::TargetIdentity::Surface {
        id: surface.mapping_id,
        width: 64,
        height: 32,
        generation: u64::from(surface.map_generation),
        format: ash::vk::Format::R8G8B8A8_UNORM,
    });
    assert_ne!(
        base.allocation, resident.allocation,
        "a present target is not the resident a render record keeps for the same surface"
    );
}

/// R4b: the frame the provider presented is the frame the display rail captures.
///
/// The display rail's only frame-fetch entry point is
/// `runtime::scanout::capture_present_frame`, and its first source is the host
/// surface cache — the map the Store route publishes a mapper-ref-texture
/// frame into (`runtime::mapping_write::write_bgra8`'s cache half). So the test
/// lands the provider's *present target* bytes through that same publish and
/// asserts the capture API returns them byte for byte; a rail that had
/// presented a different image would fail here even with identical counters.
///
/// The control is the other half of the two-vein rule: a present whose frame
/// nothing published has no capture at all. The capture refuses and keeps the
/// prior retain rather than opening a guest-page third source — which is what
/// "the display rail consumes provider frames" has to mean if the frame is
/// real.
#[test]
fn the_display_rail_fetches_the_present_targets_own_frame() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();
    let mapping_id = 0x4b_02_00;
    let map_generation = 11;
    let out = submitted_present(
        "display rail present",
        &stages,
        PresentSurfaceKey {
            mapping_id,
            map_generation,
        },
    );
    let frame = out.bytes.clone();
    assert_texel_count("presented frame", &frame);

    let mut state = reims_vgpu::model::DeviceState::new(
        reims_vgpu::model::DeviceId(1),
        reims_vgpu::protocol::gva::PAGE_SHIFT_X86,
    );
    reims_vgpu::runtime::surface_cache::store_rows(
        &mut state,
        mapping_id,
        width,
        height,
        &frame,
        width * 4,
    );
    assert!(
        reims_vgpu::runtime::scanout::capture_present_frame(
            &mut state,
            mapping_id,
            width,
            height,
            map_generation,
        ),
        "the display rail captures the frame the store route published"
    );
    assert_frames_equal("display capture", &state.present.frame_bgra, &frame);
    assert_eq!(
        state.present.frame_mapping, mapping_id,
        "the capture names the mapping the provider presented"
    );

    // The control: the same capture API, the same mapping, nothing published.
    let mut unpublished = reims_vgpu::model::DeviceState::new(
        reims_vgpu::model::DeviceId(1),
        reims_vgpu::protocol::gva::PAGE_SHIFT_X86,
    );
    assert!(
        !reims_vgpu::runtime::scanout::capture_present_frame(
            &mut unpublished,
            mapping_id,
            width,
            height,
            map_generation,
        ),
        "a present whose frame reached no source fails visibly instead of being invented"
    );
    assert!(
        unpublished.present.frame_bgra.is_empty(),
        "and the failed capture keeps its (empty) prior retain"
    );
}

/// R4b's class boundary: a present tail on any other shape keeps the engine by
/// name, and the provider never sees the request.
#[test]
fn a_presenting_shape_beside_the_class_stays_on_the_engine_by_name() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let surface = PresentSurfaceKey {
        mapping_id: 0x4b_03_00,
        map_generation: 1,
    };
    let delivered_before = provider_render::provider_submissions();
    let refused = |label: &str, role: RenderChainRole, req: &DrawRequest, want: &str| {
        match provider_render::submit_render(&present_inputs(&stages, role, surface), req) {
            RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
                reason.slug(),
                want,
                "{label}: the class names the condition that kept the shape on the engine: {reason}"
            ),
            other => panic!(
                "{label}: a presenting shape beside the class stays on the engine: {other:?}"
            ),
        }
    };

    // The chain head's frame belongs to the record after it.
    let head = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    refused(
        "a presenting chain head",
        RenderChainRole::Head,
        &head,
        "render_provider_out_of_class_present_position",
    );
    // And a record that continues an encoder is not the packet's sole record
    // either, whatever role the caller states.
    let mut continued = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    continued.continues_render_pass = true;
    refused(
        "a presenting continuation",
        RenderChainRole::SoleOrTail,
        &continued,
        "render_provider_out_of_class_present_position",
    );
    // A withheld readback would present a target whose bytes never come back.
    let mut withheld = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    withheld.skip_readback = true;
    withheld.readback_skip_reason = ReadbackSkipReason::ResidentStore;
    refused(
        "a presenting record that withholds its readback",
        RenderChainRole::SoleOrTail,
        &withheld,
        "render_provider_out_of_class_present_unpublished",
    );
    // A record naming a resident target is the emulator's own
    // `resident_target_present_unsupported` shape, answered here so the draw
    // falls back instead of being declined.
    let mut resident = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    resident.target_identity = Some(surface_identity(0x4b_03_01));
    refused(
        "a presenting record that names a resident target",
        RenderChainRole::SoleOrTail,
        &resident,
        "render_provider_out_of_class_present_target",
    );

    assert_eq!(
        provider_render::provider_submissions(),
        delivered_before,
        "a present tail the class refuses never reaches the provider"
    );
}

/// R4b's fail-closed boundary for a packet whose chain is admitted only in
/// part.
///
/// Routing is per record: a resident store hands the *next* record a frame that
/// now lives in the provider, and a later record that stays on the engine for
/// any other reason asks the engine's own registry for that image. The engine
/// has never held it, so the read is refused by name —
/// `read_target_unknown_identity` — and that refusal, not a wrong frame, is what
/// a split chain produces. This test drives the exact pair: the provider keeps
/// the frame under the resident identity R7b mints, and the engine is asked to
/// read that identity.
#[test]
fn a_split_chain_fails_closed_on_the_engines_own_name() {
    use reims_vgpu::observe::Decline as _;

    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let identity = surface_identity(0x4b_04_00);

    // The provider executes the record whose frame stays in its own image.
    let seed = resident_seed_request(&identity);
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &seed) {
        RenderRailOutcome::ProviderCompletedResident(_) => (),
        other => panic!("the resident seed is in class: {other:?}"),
    }

    // The engine's own registry never saw that identity — the frame is in the
    // provider's image — so the read a chain-splitting record would perform
    // refuses by name rather than returning bytes from nothing.
    let error = match engine::read_target(&identity) {
        Ok(_) => panic!("the engine holds no resident the provider wrote"),
        Err(error) => error,
    };
    let engine::DrawError::TargetRead(reason) = &error else {
        panic!("the refusal is the readback rail's own vocabulary: {error:?}");
    };
    assert_eq!(
        reason.slug(),
        "read_target_unknown_identity",
        "a frame the provider kept is a named refusal on the engine, not a wrong frame"
    );
}

/// The canonical rail's frame for an asymmetric draw is the engine's own frame,
/// byte for byte.
///
/// This is the disagreement every positive case in this file had to avoid with
/// y-symmetric geometry since R6 — pinned as an expectation instead of a remark.
/// The provider now states Metal's convention where it translates a guest vertex
/// stage: `y` is negated on every `BuiltIn Position` output on the way into
/// SPIR-V (`metal-api-vulkan/src/lib.rs::negate_position_y`, merged as
/// `ed60380`), so the same shape lands on the same rows under the provider's
/// positive-height viewport as under the engine's negative-height one.
///
/// `research/docs/26` §18 recorded the pair of readings *before* the provider's
/// change (provider = engine's rows reversed); `research/docs/23` §80 has the
/// provider-side reading. This test is the same fixture after it: the first
/// assertion derives the Metal mapping from the guest's own vertices, and the
/// last one compares the two rails directly.
#[test]
fn the_canonical_rails_asymmetric_frame_is_the_engines_own_frame() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let provider = provider_pixels("ndc-y fixture", &stages, &asymmetric_request());
    assert_frame_is_the_metal_mapping("ndc-y fixture (provider)", &provider, ASYMMETRIC_VERTICES);
    let Some(engine) = engine_pixels("ndc-y fixture", &stages, asymmetric_request()) else {
        return;
    };
    assert_frames_equal("ndc-y fixture", &provider, &engine);
    assert_ne!(
        engine,
        flip_rows(&engine, ASYMMETRIC_WIDTH, ASYMMETRIC_HEIGHT),
        "the fixture's own frame has to differ from its rows reversed: a shape whose two \
         readings agree would make this pair's expectation vacuous"
    );
}
/// A stage pair whose fragment half declares a `[[buffer(0)]]` argument, with
/// the class gate's declaration list taken from the fixture's *own* translation
/// (R9b).
///
/// `provider_render_rail`'s other fixtures are the `[[stage_in]]` shapes, whose
/// reflections declare no buffer at all; this one is the fixture that does, so
/// the gate's input here is the fact the runtime would hand it rather than a
/// second spelling of it. The vertex half stays the reviewed one — the draw is
/// otherwise an admitted shape, which is what makes the answer about the
/// fragment declaration alone.
fn buffer_declaring_stages(fragment_fixture: &str, fragment_entry: &'static str) -> Stages {
    let mut stages = Stages {
        air: (fixture("reims_indexed_tri.air"), fixture(fragment_fixture)),
        vertex_entry: "reims_indexed_vertex",
        fragment_entry,
        vertex_attribute_locations: vec![0],
        vertex_stage_buffer_declarations: Vec::new(),
        fragment_stage_buffer_declarations: Vec::new(),
    };
    stages.vertex_stage_buffer_declarations =
        declared_stage_buffers(&stages.air.0, RenderStage::Vertex, stages.vertex_entry);
    stages.fragment_stage_buffer_declarations =
        declared_stage_buffers(&stages.air.1, RenderStage::Fragment, stages.fragment_entry);
    stages
}

/// The `[[buffer(N)]]` arguments one fixture's stage really declares, read back
/// from the canonical translation the provider will run.
///
/// The mapping from `metal2vulkan`'s access answer to the seam's class
/// vocabulary is the one `backend/vulkan/pipeline_resolve.rs::
/// stage_buffer_declarations` applies in production; it is re-stated here only
/// so a test can hand the gate the *measured* fact of a fixture, and the
/// expectation it is compared against in each test below is written by hand —
/// so a fixture whose reflection moves fails the assertion instead of quietly
/// changing what the seam is asked about.
fn declared_stage_buffers(
    air: &[u8],
    stage: RenderStage,
    entry: &str,
) -> Vec<StageBufferDeclaration> {
    let executor = VulkanExecutor::new().expect("the acceptance environment has a Vulkan device");
    let device =
        Device::new(std::sync::Arc::clone(&executor) as std::sync::Arc<dyn ComputeExecutor>);
    let function = device
        .new_library_with_binary_air(air.to_vec())
        .expect("the fixture is a binary AIR module")
        .function(entry)
        .expect("the fixture's entry exists");
    let translated =
        TranslatedRenderStage::translate(stage, &function).expect("the fixture translates");
    translated
        .reflection()
        .bindings
        .iter()
        .filter(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::Buffer)
        .map(|binding| StageBufferDeclaration {
            index: binding.metal_index,
            access: match binding.access {
                Some(metal2vulkan::reflect::ResourceAccess::Unused) => StageBufferAccess::Unused,
                Some(metal2vulkan::reflect::ResourceAccess::ReadOnly) => StageBufferAccess::Read,
                Some(metal2vulkan::reflect::ResourceAccess::WriteOnly) => StageBufferAccess::Write,
                Some(metal2vulkan::reflect::ResourceAccess::ReadWrite) => {
                    StageBufferAccess::ReadWrite
                }
                Some(
                    metal2vulkan::reflect::ResourceAccess::Sampled
                    | metal2vulkan::reflect::ResourceAccess::Storage,
                )
                | None => StageBufferAccess::Unknown,
            },
            // The reach, read the way `pipeline_resolve::stage_buffer_declarations`
            // reads it and the way the canonical registration computes its own
            // (R9d/R9j): the largest exclusive byte offset over the static
            // ranges when every reach is static, the reflected affine access
            // set when a stride is involved, and `Unstated` for an unbounded or
            // range-less reach.
            footprint: match binding.footprint.as_ref() {
                Some(footprint)
                    if !footprint.has_unbounded_access && footprint.strided_accesses.is_empty() =>
                {
                    match footprint
                        .static_ranges
                        .iter()
                        .map(|range| range.offset.saturating_add(range.size))
                        .max()
                    {
                        Some(max_bytes) => StageBufferFootprint::Static { max_bytes },
                        None => StageBufferFootprint::Unstated,
                    }
                }
                Some(footprint) if !footprint.has_unbounded_access => {
                    match reflected_affine_accesses(footprint) {
                        Some(accesses) => StageBufferFootprint::Affine { accesses },
                        None => StageBufferFootprint::Unstated,
                    }
                }
                _ => StageBufferFootprint::Unstated,
            },
        })
        .collect()
}

/// The reflected affine access set in the contract's own terms, restated from
/// `pipeline_resolve::stage_buffer_declarations` (R9j) so a test can hand the
/// gate the *measured* footprint of a fixture.
///
/// The restatement is the same one [`declared_stage_buffers`] makes for the
/// access half: the production mapping is the authority, and the expectation
/// each test writes by hand is what would fail if a fixture's reflection moved.
fn reflected_affine_accesses(
    footprint: &metal2vulkan::reflect::BufferFootprint,
) -> Option<Vec<metal_api_core::provider::AffineAccess>> {
    use metal2vulkan::reflect::BufferIndexSource;
    if footprint.has_unbounded_access {
        return None;
    }
    let mut accesses =
        Vec::with_capacity(footprint.static_ranges.len() + footprint.strided_accesses.len());
    for range in &footprint.static_ranges {
        accesses.push(metal_api_core::provider::AffineAccess {
            base_offset: range.offset,
            access_size: range.size,
            terms: Vec::new(),
        });
    }
    for access in &footprint.strided_accesses {
        let mut terms = Vec::with_capacity(access.terms.len());
        for term in &access.terms {
            let axis = match term.source {
                BufferIndexSource::VertexIndex => 0,
                BufferIndexSource::InstanceIndex => 1,
                _ => return None,
            };
            terms.push(metal_api_core::provider::AffineTerm {
                axis,
                stride: term.stride,
            });
        }
        accesses.push(metal_api_core::provider::AffineAccess {
            base_offset: access.base_offset,
            access_size: access.access_size,
            terms,
        });
    }
    Some(accesses)
}

/// R9b, first half: a draw whose stages bind buffers that neither stage's
/// translation declares leaves for the provider — and the two rails land the
/// same bytes, because the bytes that differ are the bytes no stage reads.
///
/// This is the population the v2 census's 99.3% door held. The door was
/// `!req.storage_buffers.is_empty()` with the sentence "the canonical render
/// contract has no buffer bindings for a pipeline"; the contract states them now
/// (v83), but a *translated* stage that names a buffer is still refused by the
/// provider's registration, and a request whose binds fill indices neither stage
/// names needs no declaration at all — the pipeline-level buffer face is about
/// the shader's interface, and these binds are not in it. The engine's own bind
/// path already serves exactly this population without reading it: an index the
/// reflection does not mention is `ReflectedBufferAccess::Absent`, and the
/// engine either stages those bytes or hands the bind the neutral page, while
/// nothing in the module dereferences it.
///
/// Falsifiability, both directions, so the equality cannot be the equality of
/// two frames nobody drew:
///
/// - the bound bytes move and the frame does *not* — on both rails, which is
///   the statement that these binds carry no semantics for either one;
/// - the position stream beside them moves and the frame *does*.
#[test]
fn stage_buffers_no_stage_declares_leave_for_the_provider_and_agree_with_the_engine() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let binds = |word: u8| {
        vec![
            // A vertex bind at 5 and a fragment bind at 7: indices no
            // `[[buffer(N)]]` argument of this pair names. The two stages'
            // index spaces are independent in Metal, so the pair is stated with
            // one bind in each — the shape the census's `buffers` door was
            // counting without being able to split it.
            engine::StorageBufferResource {
                binding: 5,
                content: BufferContent::Bytes(std::sync::Arc::new(vec![word; 64])),
            },
            engine::StorageBufferResource {
                binding: 7,
                content: BufferContent::Bytes(std::sync::Arc::new(vec![!word; 64])),
            },
        ]
    };
    let request = |word: u8, specs: &[StreamSpec]| {
        let mut req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, specs);
        req.storage_buffers = binds(word);
        req
    };

    let before = route_count("draw_stage_buffers_2_4");
    let provider = provider_pixels(
        "binds no stage declares",
        &stages,
        &request(0x00, &position_streams()),
    );
    assert_solid("binds no stage declares", &provider);
    assert_eq!(
        route_count("draw_stage_buffers_2_4"),
        before + 1,
        "the bound-buffer band is charged once per request the gate is handed"
    );
    let Some(engine) = engine_pixels(
        "binds no stage declares",
        &stages,
        request(0x00, &position_streams()),
    ) else {
        return;
    };
    // The engine's own frame for the same shape: the binds are the only
    // difference from the reviewed request this suite already compares.
    assert_solid("binds no stage declares (engine)", &engine);
    assert_frames_equal(
        "canonical provider vs engine, two undeclared stage buffers",
        &provider,
        &engine,
    );

    // The same shape with other bytes behind those binds: neither rail may
    // move. This is also the negative control for the two binds' *indices*
    // mattering: a rail that bound them into the descriptors its module does
    // declare would land a different frame here.
    let provider_other = provider_pixels(
        "binds no stage declares, other bytes",
        &stages,
        &request(0xa5, &position_streams()),
    );
    assert_frames_equal(
        "the undeclared binds' bytes do not reach the frame (provider)",
        &provider_other,
        &provider,
    );
    let Some(engine_other) = engine_pixels(
        "binds no stage declares, other bytes",
        &stages,
        request(0xa5, &position_streams()),
    ) else {
        return;
    };
    assert_frames_equal(
        "the undeclared binds' bytes do not reach the frame (engine)",
        &engine_other,
        &engine,
    );

    // The positive control: the *declared* interface beside the binds still
    // decides the frame, so the equality above is a statement about these binds
    // and not about a draw whose inputs stopped arriving.
    let moved = provider_pixels(
        "declared interface beside the binds",
        &stages,
        &request(0x00, &[stream(0, &[(0.25, 0.0); 3])]),
    );
    assert_frames_differ(
        "the declared stream still reaches the frame",
        &moved,
        &provider,
    );
}
/// R9d: a read-only `[[buffer(N)]]` argument leaves for the provider — through
/// the owner rail — and the two rails land the same bytes.
///
/// R9b kept this shape on the engine because a translated stage that names a
/// buffer had no way through the canonical rail at all: the registration
/// refused the stage outright, and the execution gate served the reviewed
/// stage-buffer pair alone. R9c made the pair executable from a translated
/// stage; R9d states it from *this* rail, one declaration per read-only
/// argument (stage, index, access, footprint) paired with one view per
/// declaration, whose bytes the owner rail imports.
///
/// The provider-side half stays measured rather than asserted: the same pair
/// handed to the canonical registration under a contract that declares *no*
/// stage buffer is still refused by name (`render_stage_unsupported_interface`,
/// field `bindings`) — which is what makes the *declaration* the thing this
/// increment had to state, and the translation alone not enough.
///
/// Falsifiability: the fixture's fragment stores its buffer's first `float32`
/// in red, so both rails are read on a zeroed bind (`00 00 00 ff`) and on a
/// `1.0f` one (`ff 00 00 ff`), and the two frames are required to differ.
#[test]
fn a_read_only_stage_buffer_leaves_for_the_provider_and_agrees_with_the_engine() {
    let _guard = engine_test_session();
    let stages = buffer_declaring_stages("render_frag_buffer.air", "reims_buffer_frag");
    assert!(
        stages.vertex_stage_buffer_declarations.is_empty(),
        "the reviewed vertex fixture declares no buffer: {:#?}",
        stages.vertex_stage_buffer_declarations
    );
    assert_eq!(
        stages.fragment_stage_buffer_declarations,
        vec![StageBufferDeclaration {
            index: 0,
            access: StageBufferAccess::Read,
            // The fixture reads one `float32`, and the extent it is declared
            // with is that reach — the number the canonical registration
            // compares its own translation's footprint against.
            footprint: StageBufferFootprint::Static { max_bytes: 4 },
        }],
        "the buffer fixture's own translation declares one read-only buffer at index 0"
    );

    // The provider-side fact the declaration exists for, measured on the same
    // pair through the canonical provider's public registration entry point: a
    // contract that declares no stage buffer leaves the translated stage's own
    // `[[buffer(0)]]` argument unbacked, and the registration says so by name.
    let executor = VulkanExecutor::new().expect("the acceptance environment has a Vulkan device");
    let provider = VulkanComputeProvider::with_executor(std::sync::Arc::clone(&executor))
        .expect("the canonical provider builds");
    let device =
        Device::new(std::sync::Arc::clone(&executor) as std::sync::Arc<dyn ComputeExecutor>);
    let policy = provider.spirv_feature_policy();
    let translate = |air: &[u8], stage: RenderStage, entry: &str| {
        let function = device
            .new_library_with_binary_air(air.to_vec())
            .expect("the fixture is a binary AIR module")
            .function(entry)
            .expect("the fixture's entry exists");
        TranslatedRenderStage::translate_with_policy(stage, &function, policy)
            .expect("the fixture translates under this device's policy")
    };
    let refused = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: stages.vertex_entry.to_owned(),
                fragment_entry: stages.fragment_entry.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::Buffers(vec![VertexBufferLayout {
                    stride: 8,
                    step: VertexStep::PerVertex,
                    attributes: vec![VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32x2,
                    }],
                }]),
                // The declaration-less contract: the slot the stage reads is
                // the one the pair below has no view for.
                stage_buffers: Vec::new(),
            },
            vertex: translate(&stages.air.0, RenderStage::Vertex, stages.vertex_entry),
            fragment: translate(&stages.air.1, RenderStage::Fragment, stages.fragment_entry),
            logical_digest: SemanticDigest::new(
                "reims-provider-render-rail-v1",
                b"declared-stage-buffer-interface".to_vec(),
            )
            .expect("the digest names a case"),
        })
        .expect_err("a reflected buffer binding the contract does not declare has no view");
    eprintln!("provider answer for the declared-buffer pair: {refused:?}");
    assert_eq!(refused.slug, "render_stage_unsupported_interface");
    assert_eq!(
        refused.fields.get("stage"),
        Some(&FieldValue::Text("fragment".to_owned())),
        "the refusal names the stage whose interface it is about: {refused:?}"
    );
    assert_eq!(
        refused.fields.get("field"),
        Some(&FieldValue::Text("bindings".to_owned())),
        "and the reflected field the contract failed to declare: {refused:?}"
    );

    // The admitted shape, on both rails. One request builder feeds the two —
    // the request's own `storage_buffers` is what the engine's bind rail reads,
    // and the `stage_buffer_binds` list beside it is what the seam states to
    // the canonical rail — so the two arms read one set of bytes. The fixture's
    // fragment stores the buffer's first `float32` in red: a zeroed bind lands
    // `00 00 00 ff`, `1.0f` lands `ff 00 00 ff`.
    let framed = |word: [u8; 4]| -> (DrawRequest, BufferContent) {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        let mut bytes = vec![0u8; 16];
        bytes[..4].copy_from_slice(&word);
        let content = BufferContent::Bytes(std::sync::Arc::new(bytes));
        req.storage_buffers.push(engine::StorageBufferResource {
            binding: 0,
            content: content.clone(),
        });
        (req, content)
    };
    let (zeroed_req, zeroed_content) = framed([0, 0, 0, 0]);
    let (one_req, one_content) = framed([0, 0, 0x80, 0x3f]);
    let zeroed_binds = [staged_bind(
        RenderPipelineStage::Fragment,
        0,
        &zeroed_content,
    )];
    let one_binds = [staged_bind(RenderPipelineStage::Fragment, 0, &one_content)];
    let provider_frame = |label: &str, req: &DrawRequest, binds: &[StageBufferBind<'_>]| {
        match provider_render::submit_render(
            &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, binds),
            req,
        ) {
            RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
            other => panic!(
                "{label}: a declared read-only stage buffer has to leave for the provider: \
                 {other:?}"
            ),
        }
    };
    let band = route_count("draw_stage_buffers_1");
    let zeroed_provider =
        provider_frame("declared stage buffer, zeroed", &zeroed_req, &zeroed_binds);
    eprintln!(
        "declared stage buffer: fragment [[buffer(0)]] class={:?} footprint={:?}",
        stages.fragment_stage_buffer_declarations[0].access,
        stages.fragment_stage_buffer_declarations[0].footprint,
    );
    eprintln!(
        "provider texel for a zeroed bind: {:?}; draw_stage_buffers_1 = {}",
        texel_at(&zeroed_provider, 0, 0),
        route_count("draw_stage_buffers_1"),
    );
    assert_eq!(
        texel_at(&zeroed_provider, 0, 0),
        [0, 0, 0, 255],
        "the provider reads the stage's own [[buffer(0)]] bytes through the owner's lease"
    );
    assert_eq!(
        route_count("draw_stage_buffers_1"),
        band + 1,
        "the bound-buffer band is charged once per request the gate is handed"
    );
    let one_provider = provider_frame("declared stage buffer, 1.0f", &one_req, &one_binds);
    eprintln!(
        "provider texel for a 1.0f bind: {:?}; the two frames differ: {}",
        texel_at(&one_provider, 0, 0),
        one_provider != zeroed_provider,
    );
    assert_eq!(
        texel_at(&one_provider, 0, 0),
        [255, 0, 0, 255],
        "and the bytes behind that declaration are the provider frame's own source"
    );
    assert_frames_differ(
        "the declared buffer's bytes reach the provider's frame",
        &one_provider,
        &zeroed_provider,
    );

    // The engine arm, for the same two draws: the comparison this rail promises
    // is byte-for-byte, and it is the only thing that makes "the two rails
    // execute one shape" a statement about the shape.
    let Some(zeroed_engine) = engine_pixels(
        "declared stage buffer, zeroed",
        &stages,
        framed([0, 0, 0, 0]).0,
    ) else {
        return;
    };
    let Some(one_engine) = engine_pixels(
        "declared stage buffer, 1.0f",
        &stages,
        framed([0, 0, 0x80, 0x3f]).0,
    ) else {
        return;
    };
    assert_eq!(
        texel_at(&zeroed_engine, 0, 0),
        [0, 0, 0, 255],
        "the engine draws the fragment the declared binding names"
    );
    assert_frames_equal(
        "canonical provider vs engine, a declared read-only stage buffer, zeroed",
        &zeroed_provider,
        &zeroed_engine,
    );
    assert_frames_equal(
        "canonical provider vs engine, a declared read-only stage buffer, 1.0f",
        &one_provider,
        &one_engine,
    );
    eprintln!(
        "provider vs engine, declared read-only stage buffer: {} bytes equal (zeroed) and {} \
         bytes equal (1.0f)",
        zeroed_provider.len(),
        one_provider.len(),
    );
    assert_frames_differ(
        "the declared buffer's bytes reach the engine's frame",
        &one_engine,
        &zeroed_engine,
    );
}

/// The door's buckets after R9d, each its own counter: the three classes the
/// canonical contract cannot state keep the draw on the engine under the name
/// their own access gives them, and the read-only class — the one this
/// increment lifts — is answered by the *fact* that stands between the
/// declaration and a stated pair (`research/docs/26` §R9b, §R9d).
///
/// Two arms are driven by a real fixture whose `[[buffer(0)]]` metadata states
/// that access, so the split is measured against modules rather than against
/// the seam's own vocabulary: an unread declaration must not be counted as a
/// read one, and a writable one — the arm the canonical contract refuses
/// before admission — must not hide inside the read population.
///
/// The `Unknown` arm is the one no fixture here can state: a shader body that
/// reads its buffer makes `metal2vulkan` reflect `ReadOnly` from the emitted
/// module's own decoration whatever the metadata omitted, so the arm is handed
/// to the gate directly. It is still a separate bucket, which is the property
/// this test is about — an unclassified declaration must not be counted as a
/// read one.
#[test]
fn the_stage_buffer_door_names_what_keeps_a_declared_draw_on_the_engine() {
    let _guard = engine_test_session();
    let bases = reviewed_stages();
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.storage_buffers.push(engine::StorageBufferResource {
        binding: 0,
        content: BufferContent::Bytes(std::sync::Arc::new(vec![0u8; 16])),
    });
    // The refusals: one bucket per arm the canonical contract cannot state —
    // unchanged by R9d — plus the writable arm R9j states and can only admit
    // when the request names a destination for it.
    let slug = |access| match access {
        StageBufferAccess::Unused => (
            "render_provider_out_of_class_stage_buffer_unused",
            Some(("render_frag_buffer_unused.air", "reims_unused_buffer_frag")),
        ),
        // A writable declaration with no landing: this request's bind states
        // no guest destination for it, which is the one fact R9j could not
        // land.
        StageBufferAccess::Write | StageBufferAccess::ReadWrite => (
            "render_provider_out_of_class_stage_buffer_write",
            Some(("render_frag_buffer_write.air", "reims_write_buffer_frag")),
        ),
        // The read-only arm is admitted, so it has no refusal bucket of its
        // own any more: it is asserted below, against the request that states
        // the bind and the one that does not.
        StageBufferAccess::Read => (
            "render_provider_out_of_class_stage_buffer_unbound",
            Some(("render_frag_buffer.air", "reims_buffer_frag")),
        ),
        StageBufferAccess::Unknown => ("render_provider_out_of_class_stage_buffer_unknown", None),
    };

    for access in [
        StageBufferAccess::Unused,
        StageBufferAccess::Read,
        // The writable fixture stores the word it loaded, so its reflection
        // states both uses — `ReadWrite`, the arm R9f landed — while the
        // request states no destination for it: the pair the gate has to
        // answer as a write it cannot land.
        StageBufferAccess::ReadWrite,
        StageBufferAccess::Unknown,
    ] {
        let (bucket, module) = slug(access);
        let mut stages = bases.clone();
        match module {
            Some((fixture_name, entry)) => {
                stages.air.1 = fixture(fixture_name);
                stages.fragment_entry = entry;
                stages.fragment_stage_buffer_declarations =
                    declared_stage_buffers(&stages.air.1, RenderStage::Fragment, entry);
                // The access and the index are the two facts this test is
                // about; the reach beside them is whatever the fixture's own
                // translation reports (R9d), so it is read and not restated.
                let declarations = &stages.fragment_stage_buffer_declarations;
                assert_eq!(declarations.len(), 1, "{fixture_name}: one declaration");
                assert_eq!(
                    (declarations[0].index, declarations[0].access),
                    (0, access),
                    "{fixture_name} declares one buffer of its own access at index 0"
                );
            }
            None => {
                stages.fragment_stage_buffer_declarations = vec![StageBufferDeclaration {
                    index: 0,
                    access,
                    footprint: StageBufferFootprint::Static { max_bytes: 4 },
                }];
            }
        }
        let before = route_count(bucket);
        // The writable arm needs a bind to reach its own fact: with the slot
        // empty the gate answers `_unbound` first, which is the *other* arm's
        // bucket. Its bind states no destination, so the write it cannot land
        // is what the gate answers.
        let content = BufferContent::Bytes(std::sync::Arc::new(vec![0u8; 16]));
        let writable_binds = [staged_bind(RenderPipelineStage::Fragment, 0, &content)];
        let inputs = match access {
            StageBufferAccess::Write | StageBufferAccess::ReadWrite => {
                inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &writable_binds)
            }
            _ => inputs(&stages, RenderChainRole::SoleOrTail),
        };
        match provider_render::submit_render(&inputs, &req) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                assert_eq!(
                    reason.slug(),
                    bucket,
                    "{access:?} is its own bucket: {reason}"
                );
                assert!(
                    reason.detail().contains("[[buffer(0)]]"),
                    "the sentence names the declaration: {reason}"
                );
            }
            other => panic!("{access:?} is out of class: {other:?}"),
        }
        assert_eq!(
            route_count(bucket),
            before + 1,
            "{access:?} charges its own bucket exactly once"
        );
    }

    // The read-only class, both ways: with no bind stated the pair cannot be
    // completed and the draw stays on the engine under the fact that is
    // missing; with the bind stated the same shape leaves for the provider —
    // which is R9d's whole content.
    let read_stages = buffer_declaring_stages("render_frag_buffer.air", "reims_buffer_frag");
    let unbound = route_count("render_provider_out_of_class_stage_buffer_unbound");
    match provider_render::submit_render(&inputs(&read_stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_stage_buffer_unbound",
            "a read-only declaration no bind fills stays on the engine by name: {reason}"
        ),
        other => panic!("a declared slot nothing binds is out of class: {other:?}"),
    }
    assert_eq!(
        route_count("render_provider_out_of_class_stage_buffer_unbound"),
        unbound + 1,
        "and that fact is its own counter"
    );
    let content = BufferContent::Bytes(std::sync::Arc::new(vec![0u8; 16]));
    let binds = [staged_bind(RenderPipelineStage::Fragment, 0, &content)];
    assert_eq!(
        route_count("render_provider_out_of_class_stage_buffer_unbound"),
        unbound + 1,
        "the admitted arm is asserted after this one, so the counter is read first"
    );
    match provider_render::submit_render(
        &inputs_with_binds(&read_stages, RenderChainRole::SoleOrTail, &binds),
        &req,
    ) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("a declared read-only stage buffer leaves for the provider: {other:?}"),
    }

    // The vertex stage's declarations answer first: one request, both stages
    // declaring a class the contract cannot state, and the vertex bucket is the
    // one that moves.
    let mut stages = bases.clone();
    stages.vertex_stage_buffer_declarations = vec![StageBufferDeclaration {
        index: 2,
        access: StageBufferAccess::Unused,
        footprint: StageBufferFootprint::Static { max_bytes: 16 },
    }];
    stages.fragment_stage_buffer_declarations = vec![StageBufferDeclaration {
        index: 3,
        access: StageBufferAccess::Write,
        footprint: StageBufferFootprint::Static { max_bytes: 16 },
    }];
    let vertex_bucket = route_count("render_provider_out_of_class_stage_buffer_unused");
    let fragment_bucket = route_count("render_provider_out_of_class_stage_buffer_write");
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_stage_buffer_unused",
            "the vertex half answers first: {reason}"
        ),
        other => panic!("a declared vertex buffer keeps the draw on the engine: {other:?}"),
    }
    assert_eq!(
        route_count("render_provider_out_of_class_stage_buffer_unused"),
        vertex_bucket + 1
    );
    assert_eq!(
        route_count("render_provider_out_of_class_stage_buffer_write"),
        fragment_bucket,
        "the stage that did not answer does not charge a bucket"
    );

    // And the admitted arm charges none of them: the buckets are the door's
    // refusals, not the population it lets through.
    let clean = reviewed_stages();
    let before: Vec<u64> = [
        StageBufferAccess::Unused,
        StageBufferAccess::Write,
        StageBufferAccess::Unknown,
    ]
    .iter()
    .map(|access| route_count(slug(*access).0))
    .collect();
    match provider_render::submit_render(&inputs(&clean, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("the undeclared binds are in class: {other:?}"),
    }
    let after: Vec<u64> = [
        StageBufferAccess::Unused,
        StageBufferAccess::Write,
        StageBufferAccess::Unknown,
    ]
    .iter()
    .map(|access| route_count(slug(*access).0))
    .collect();
    assert_eq!(before, after, "an admitted draw charges no door bucket");
}

/// Every fact between a declaration and a stated pair is its own bucket
/// (`research/docs/26` §R9d).
///
/// R9d admits a read-only declaration only when the *whole* pair can be
/// stated, and each way of falling short is a different next increment: a
/// contract that can state an affine or unbounded footprint, a request that
/// binds the slot its stage reads, a bind whose bytes cover the declared
/// extent, a window derivation for the draw path's own gathers, and the
/// canonical list's own shape rules. They are counted apart rather than folded
/// into one door, so the next census can say which one holds the population.
///
/// The sentences are asserted beside the slugs, because the slug is a
/// vocabulary and the sentence is what a reader compares against the shape.
#[test]
fn the_stage_buffer_door_names_every_fact_between_a_declaration_and_the_provider() {
    let _guard = engine_test_session();
    let bases = reviewed_stages();
    let declaration = |footprint: StageBufferFootprint| StageBufferDeclaration {
        index: 0,
        access: StageBufferAccess::Read,
        footprint,
    };
    let with_fragment = |declarations: Vec<StageBufferDeclaration>| {
        let mut stages = bases.clone();
        stages.fragment_stage_buffer_declarations = declarations;
        stages
    };
    let content = BufferContent::Bytes(std::sync::Arc::new(vec![0u8; 16]));
    let bind = staged_bind(RenderPipelineStage::Fragment, 0, &content);
    let answer =
        |label: &str, stages: &Stages, binds: &[StageBufferBind<'_>]| -> (String, String) {
            match provider_render::submit_render(
                &inputs_with_binds(stages, RenderChainRole::SoleOrTail, binds),
                &narrow_request(MTL_FORMAT_RGBA8_UNORM),
            ) {
                RenderRailOutcome::NotInNarrowClass(reason) => {
                    (reason.slug().to_owned(), reason.detail().to_owned())
                }
                other => panic!("{label}: the shape is out of class: {other:?}"),
            }
        };

    // The reach is not a proof the contract can state: unbounded, or stating
    // no byte range at all.
    let stages = with_fragment(vec![declaration(StageBufferFootprint::Unstated)]);
    let (slug, detail) = answer("unbounded reach", &stages, &[bind]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_footprint");
    assert!(
        detail.contains("[[buffer(0)]]") && detail.contains("unbounded"),
        "the sentence names the declaration and the fact: {detail}"
    );

    // The bind's bytes do not cover the extent the stage reaches.
    let stages = with_fragment(vec![declaration(StageBufferFootprint::Static {
        max_bytes: 32,
    })]);
    let (slug, detail) = answer("short bind", &stages, &[bind]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_short");
    assert!(
        detail.contains("reaches 32 byte(s)") && detail.contains("binds 16 byte(s)"),
        "the sentence names both extents: {detail}"
    );

    // A stated window that is not the bind's own bytes.
    let stages = with_fragment(vec![declaration(StageBufferFootprint::Static {
        max_bytes: 4,
    })]);
    let wrong_window = StageBufferBind {
        stage: RenderPipelineStage::Fragment,
        index: 0,
        content: &content,
        window: Some(StageBufferWindow {
            import: 0x9e1e_u64,
            host_va: 0x1000,
            length: 4096,
            head: 8,
            bytes_len: 8,
        }),
        landing: None,
    };
    let (slug, detail) = answer("window wider than the view", &stages, &[wrong_window]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_window");
    assert!(
        detail.contains("covering 8 byte(s)") && detail.contains("16 byte bind"),
        "the sentence names the window and the bind: {detail}"
    );

    // A gather the draw path's own bind rail performs, with no window stated:
    // the source this rail has no arm for.
    let gather = BufferContent::GuestRuns(engine::GuestRunSource {
        runs: std::sync::Arc::new(Vec::new()),
        source_offset: 0,
        total_len: 16,
        row_length_texels: 0,
        pages: None,
        direct_image: None,
    });
    let gather_bind = staged_bind(RenderPipelineStage::Fragment, 0, &gather);
    let (slug, detail) = answer("guest gather", &stages, &[gather_bind]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_gather");
    assert!(
        detail.contains("gathers from guest RAM") && detail.contains("owner rail"),
        "the sentence names the source and the rail that would carry it: {detail}"
    );

    // The canonical list's own shape: at most four declarations, one per slot.
    let five = (0..5)
        .map(|index| StageBufferDeclaration {
            index,
            access: StageBufferAccess::Read,
            footprint: StageBufferFootprint::Static { max_bytes: 4 },
        })
        .collect::<Vec<_>>();
    let stages = with_fragment(five);
    let (slug, detail) = answer("five declarations", &stages, &[bind]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_shape");
    assert!(
        detail.contains("declare 5 stage buffers"),
        "the sentence names the count: {detail}"
    );
    let stages = with_fragment(vec![
        declaration(StageBufferFootprint::Static { max_bytes: 4 }),
        declaration(StageBufferFootprint::Static { max_bytes: 4 }),
    ]);
    let (slug, detail) = answer("one slot twice", &stages, &[bind]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_shape");
    assert!(
        detail.contains("[[buffer(0)]] twice"),
        "the sentence names the slot: {detail}"
    );
    // A vertex declaration inside the canonical layout: this class states the
    // request's own stream as binding 0, and the contract refuses one Metal
    // binding described twice.
    let mut stages = bases.clone();
    stages.vertex_stage_buffer_declarations = vec![StageBufferDeclaration {
        index: 0,
        access: StageBufferAccess::Read,
        footprint: StageBufferFootprint::Static { max_bytes: 4 },
    }];
    let vertex_bind = staged_bind(RenderPipelineStage::Vertex, 0, &content);
    let (slug, detail) = answer("vertex declaration in the layout", &stages, &[vertex_bind]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_shape");
    assert!(
        detail.contains("1 vertex stream(s)") && detail.contains("canonical layout"),
        "the sentence names the layout it collides with: {detail}"
    );

    // Every bucket above is a counter, not a latch.
    let buckets = [
        "render_provider_out_of_class_stage_buffer_footprint",
        "render_provider_out_of_class_stage_buffer_short",
        "render_provider_out_of_class_stage_buffer_window",
        "render_provider_out_of_class_stage_buffer_gather",
        "render_provider_out_of_class_stage_buffer_shape",
    ];
    for bucket in buckets {
        assert!(
            route_count(bucket) > 0,
            "{bucket} was charged by the shape it names"
        );
    }
}

/// R9d's borrowed arm: a stage buffer whose bytes live in a registered guest
/// RAM window leaves through the canonical rail's *no-copy* arm.
///
/// The owner rail was already carrying this arm for compute
/// (`research/docs/20` §3.4, `provider_compute_device_loss`), and R3c taught
/// the canonical render rail to resolve it per input
/// (`BufferSource::BorrowedNoCopy`). What this test pins is that the render
/// seam reaches it: the bind is stated with a window over a real registration
/// and with staged bytes that are a *different* picture, so the frame says
/// which arm carried the bytes — and moving the owner's own mapping moves the
/// frame, with nothing else touched. A rail that had copied the bind would draw
/// the staged picture twice.
#[test]
fn a_stage_buffer_in_a_registered_window_leaves_without_a_copy() {
    use reims_vgpu::backend::provider_compute::{device_epoch, host_import_alignment};

    let _guard = engine_test_session();
    let alignment = host_import_alignment().expect("the owner rail's provider answers");
    assert!(
        alignment > 0,
        "this device must advertise VK_EXT_external_memory_host for the no-copy arm"
    );
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let mut owner = AlignedHost::new(2 * page, page);
    // The mapping holds the fragment's red; the staged copy beside it holds
    // black. Only one of the two can be the frame's source.
    owner.as_mut_slice()[..4].copy_from_slice(&[0, 0, 0x80, 0x3f]);
    let import = 0x9e1d_u64;
    provider_owner::register(Region {
        import,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: owner.pointer as usize,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x40_0000),
    })
    .expect("a page-aligned registration is a legal provider region");
    let window = StageBufferWindow {
        import,
        host_va: owner.pointer as u64,
        length: page as u64,
        head: 0,
        bytes_len: 16,
    };
    let staged = BufferContent::Bytes(std::sync::Arc::new(vec![0u8; 16]));
    let stages = buffer_declaring_stages("render_frag_buffer.air", "reims_buffer_frag");
    let request = || {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.storage_buffers.push(engine::StorageBufferResource {
            binding: 0,
            content: staged.clone(),
        });
        req
    };
    let binds = [StageBufferBind {
        stage: RenderPipelineStage::Fragment,
        index: 0,
        content: &staged,
        window: Some(window),
        landing: None,
    }];
    let frame = |label: &str| -> Vec<u8> {
        match provider_render::submit_render(
            &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
            &request(),
        ) {
            RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
            other => panic!("{label}: a window-backed stage buffer is in class: {other:?}"),
        }
    };

    let red = frame("registered window, red in the mapping");
    eprintln!(
        "no-copy stage buffer: device host-import alignment = {alignment}; window covers \
         {page} byte(s) of registration {import:#x}; staged copy beside it is 16 zero bytes",
    );
    eprintln!(
        "provider texel with the mapping red: {:?} (the staged copy would have landed \
         [0, 0, 0, 255])",
        texel_at(&red, 0, 0),
    );
    assert_eq!(
        texel_at(&red, 0, 0),
        [255, 0, 0, 255],
        "the provider read the owner's own mapping: the staged copy beside it is black, so a \
         copied bind would have landed 00 00 00 ff"
    );

    // The falsifiable half: move the owner's bytes and the frame follows. A
    // rail that read a copy would be unmoved by this.
    owner.as_mut_slice()[..4].copy_from_slice(&[0, 0, 0, 0]);
    let black = frame("registered window, black in the mapping");
    eprintln!(
        "provider texel after moving the mapping's first float: {:?}",
        texel_at(&black, 0, 0),
    );
    assert_eq!(
        texel_at(&black, 0, 0),
        [0, 0, 0, 255],
        "the same bind now reads the moved bytes"
    );
    assert_frames_differ(
        "the owner's own bytes reach the provider's frame",
        &red,
        &black,
    );
}

/// R9e: the window a borrowed stage buffer travels under is one the *seam*
/// derives from the bind's own zero-copy gather.
///
/// R9d needed the caller to state the window beside the bytes
/// (`a_stage_buffer_in_a_registered_window_leaves_without_a_copy` above), which
/// a driven draw path could not do: its binds are resolved by the guest RAM
/// rail, and the window the registration ledger derived for them rides on the
/// gather's page runs (`GuestWindowRun::window`). This test hands the rail
/// exactly that shape — `BufferContent::GuestRuns`, no stated window — and
/// requires the frame to follow the owner's own mapping, with the lease row
/// naming the no-copy arm. A rail that still refused every gather would answer
/// `..._stage_buffer_gather`; one that copied the bind would be unmoved by
/// moving the mapping.
#[test]
fn a_stage_buffer_the_seam_derives_from_its_gather_leaves_without_a_copy() {
    use reims_vgpu::backend::provider_compute::{device_epoch, host_import_alignment};
    use reims_vgpu::runtime::guest_ram::{GuestRamImport, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::{GuestWindowRun, RegisteredWindow};

    let _guard = engine_test_session();
    let alignment = host_import_alignment().expect("the owner rail's provider answers");
    assert!(
        alignment > 0,
        "this device must advertise VK_EXT_external_memory_host for the no-copy arm"
    );
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let mut owner = AlignedHost::new(2 * page, page);
    // The mapping holds the fragment's red; the gather's own host run reads the
    // same bytes, so the engine's answer for this bind is red as well.
    owner.as_mut_slice()[..4].copy_from_slice(&[0, 0, 0x80, 0x3f]);
    let import = std::sync::Arc::new(
        GuestRamImport::new_host_allocation(owner.pointer as usize, 2 * page as u64, alignment)
            .expect("a page-aligned synthetic host allocation"),
    );
    let anchor = import
        .slice(0, page as u64)
        .expect("the first granule is inside the import");
    let guest = GuestRef::new(std::sync::Arc::clone(&import), anchor)
        .expect("the slice came from this import");
    let import_id = import.id().get();
    provider_owner::register(Region {
        import: import_id,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: owner.pointer as usize,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x40_0000),
    })
    .expect("a page-aligned registration is a legal provider region");
    // The provider-shaped window the ledger would have derived for that run.
    let registered = RegisteredWindow {
        import: import.id(),
        base: owner.pointer as u64,
        length: page as u64,
        epoch: 1,
    };
    // The bind as the draw path builds it: one page run over the owner's
    // mapping, the bind's bytes at its start, and the window on the run.
    let content = BufferContent::GuestRuns(engine::GuestRunSource {
        runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
            owner.pointer as usize,
            2 * page as u64,
            0,
            16,
        )
        .expect("the bind's own bytes are inside the mapping")]),
        source_offset: 0,
        total_len: 16,
        row_length_texels: 0,
        pages: Some(std::sync::Arc::new(vec![GuestWindowRun {
            window_offset: 0,
            guest,
            window: Some(registered),
        }])),
        direct_image: None,
    });
    let stages = buffer_declaring_stages("render_frag_buffer.air", "reims_buffer_frag");
    let request = || {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.storage_buffers.push(engine::StorageBufferResource {
            binding: 0,
            content: content.clone(),
        });
        req
    };
    let binds = [StageBufferBind {
        stage: RenderPipelineStage::Fragment,
        index: 0,
        content: &content,
        // The seam states no window: everything the lease needs is derived.
        window: None,
        landing: None,
    }];
    let log_before = std::fs::read_to_string(reims_vgpu_observe::fail_log_path())
        .unwrap_or_default()
        .len();
    let frame = |label: &str| -> Vec<u8> {
        match provider_render::submit_render(
            &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
            &request(),
        ) {
            RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
            other => panic!("{label}: a derived-window stage buffer is in class: {other:?}"),
        }
    };

    let red = frame("derived window, red in the mapping");
    eprintln!(
        "derived stage-buffer window: device host-import alignment = {alignment}; the gather's \
         one run carries a {page} byte window over registration {import_id}, the bind's bytes \
         start at its 0th",
    );
    eprintln!(
        "provider texel with the mapping red: {:?}",
        texel_at(&red, 0, 0),
    );
    assert_eq!(
        texel_at(&red, 0, 0),
        [255, 0, 0, 255],
        "the provider read the owner's own mapping through the window the seam derived"
    );
    // The lease row, verbatim: the arm this bind left through.
    let log = std::fs::read_to_string(reims_vgpu_observe::fail_log_path()).expect("fail log");
    let fresh = &log[log_before.min(log.len())..];
    let lease = fresh
        .lines()
        .find(|line| line.contains("provider_owner_lease") && line.contains("no_copy=1"))
        .unwrap_or_else(|| panic!("the borrowed lease row was emitted: {fresh}"));
    eprintln!("lease row: {lease}");
    assert!(
        lease.contains("channel=borrowed") && lease.contains(&format!("import={import_id}")),
        "the row names the no-copy arm and this import: {lease}"
    );

    // The falsifiable half: move the owner's bytes and the frame follows. A
    // rail that copied the bind would be unmoved by this.
    owner.as_mut_slice()[..4].copy_from_slice(&[0, 0, 0, 0]);
    let black = frame("derived window, black in the mapping");
    eprintln!(
        "provider texel after moving the mapping's first float: {:?}",
        texel_at(&black, 0, 0),
    );
    assert_eq!(
        texel_at(&black, 0, 0),
        [0, 0, 0, 255],
        "the same bind now reads the moved bytes"
    );
    assert_frames_differ(
        "the owner's own bytes reach the provider's frame",
        &red,
        &black,
    );
}

/// R9e's refusal half: a gather the seam cannot cut one window from stays on
/// the engine, each under the bucket its own fact names.
///
/// Three shapes, none of them a looser door: bytes scattered over more than one
/// run, a run whose import the registration ledger never registered, and a
/// packed bind whose `source_offset` leaves the view's own host pointer off the
/// device's import granule. The first two answer `..._stage_buffer_gather` (the
/// rail mints no copy for them either), and the third answers
/// `..._stage_buffer_alignment` — a class answer, because the canonical rail
/// refuses an unaligned import by name and a declined draw is not a fallback.
#[test]
fn a_gather_the_seam_cannot_cut_a_window_from_stays_on_the_engine() {
    use reims_vgpu::backend::provider_compute::host_import_alignment;
    use reims_vgpu::runtime::guest_ram::{GuestRamImport, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::{GuestWindowRun, RegisteredWindow};

    let _guard = engine_test_session();
    let alignment = host_import_alignment().expect("the owner rail's provider answers");
    assert!(
        alignment > 0,
        "the refusal half needs the no-copy device too"
    );
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let owner = AlignedHost::new(2 * page, page);
    let import = std::sync::Arc::new(
        GuestRamImport::new_host_allocation(owner.pointer as usize, 2 * page as u64, alignment)
            .expect("a page-aligned synthetic host allocation"),
    );
    let slice = import
        .slice(0, page as u64)
        .expect("the first granule is inside the import");
    let guest = || {
        GuestRef::new(std::sync::Arc::clone(&import), slice)
            .expect("the slice came from this import")
    };
    let registered = RegisteredWindow {
        import: import.id(),
        base: owner.pointer as u64,
        length: page as u64,
        epoch: 1,
    };
    // The gather a test hands the gate: `runs` are not read on the refusal
    // arms, so one host run covers every shape below.
    let gather = |source_offset: u64, pages: Vec<GuestWindowRun>| -> BufferContent {
        BufferContent::GuestRuns(engine::GuestRunSource {
            runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
                owner.pointer as usize,
                2 * page as u64,
                0,
                16,
            )
            .expect("the bind's own bytes are inside the mapping")]),
            source_offset,
            total_len: 16,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(pages)),
            direct_image: None,
        })
    };
    let stages = buffer_declaring_stages("render_frag_buffer.air", "reims_buffer_frag");
    let answer = |label: &str, content: &BufferContent| -> (String, String) {
        let binds = [StageBufferBind {
            stage: RenderPipelineStage::Fragment,
            index: 0,
            content,
            window: None,
            landing: None,
        }];
        match provider_render::submit_render(
            &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
            &narrow_request(MTL_FORMAT_RGBA8_UNORM),
        ) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                (reason.slug().to_owned(), reason.detail().to_owned())
            }
            other => {
                panic!("{label}: a gather with no one-stretch window is out of class: {other:?}")
            }
        }
    };

    // Scattered: two runs tile the bind, so no single host range is its bytes.
    let scattered = gather(
        0,
        vec![
            GuestWindowRun {
                window_offset: 0,
                guest: guest(),
                window: Some(registered),
            },
            GuestWindowRun {
                window_offset: 8,
                guest: guest(),
                window: Some(registered),
            },
        ],
    );
    let (slug, detail) = answer("scattered gather", &scattered);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_gather");
    assert!(
        detail.contains("gathers from guest RAM"),
        "the sentence names the gather: {detail}"
    );

    // Unregistered: the ledger derived no window for this run.
    let unregistered = gather(
        0,
        vec![GuestWindowRun {
            window_offset: 0,
            guest: guest(),
            window: None,
        }],
    );
    let (slug, detail) = answer("unregistered gather", &unregistered);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_gather");

    // Off-granule: the bind starts inside the window, so the view's own host
    // pointer is not a whole number of the device's import granules.
    let off_granule = gather(
        4,
        vec![GuestWindowRun {
            window_offset: 0,
            guest: guest(),
            window: Some(registered),
        }],
    );
    let (slug, detail) = answer("off-granule gather", &off_granule);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_alignment");
    assert!(
        detail.contains(&format!("{alignment} byte alignment")),
        "the sentence names the alignment it crossed: {detail}"
    );

    // Both buckets are counters, not latches.
    assert!(route_count("render_provider_out_of_class_stage_buffer_gather") >= 2);
    assert!(route_count("render_provider_out_of_class_stage_buffer_alignment") >= 1);
}

/// A host allocation whose first byte is aligned to `alignment`, so the owner
/// rail's registration can name it (`research/docs/20` §3.2).
struct AlignedHost {
    pointer: *mut u8,
    bytes: usize,
    alignment: usize,
}

impl AlignedHost {
    fn new(bytes: usize, alignment: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(bytes, alignment).expect("a legal layout");
        // SAFETY: the layout is non-zero-sized and this struct owns the
        // allocation until it is dropped.
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!pointer.is_null(), "the host allocation");
        Self {
            pointer,
            bytes,
            alignment,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `pointer` owns `bytes` writable bytes for this struct's
        // lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.bytes) }
    }
}

impl Drop for AlignedHost {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `alloc_zeroed` with this layout.
        unsafe {
            std::alloc::dealloc(
                self.pointer,
                std::alloc::Layout::from_size_align(self.bytes, self.alignment)
                    .expect("the layout it was allocated with"),
            )
        }
    }
}

/// R9j, write half: a *writable* `[[buffer(N)]]` argument leaves for the
/// provider, and the bytes its stage wrote come back through the completion's
/// own `BufferWriteback` channel — the channel a stored attachment lands
/// through (R9f), not a second one.
///
/// The declaration is the stage's own reflection (access `Write`/`ReadWrite`
/// and the reach it states), the pass's view carries the same access
/// (`StageBufferAccessMismatch` is core's pairing of the two), the landing
/// names the guest destination the writeback goes back to, and the trace
/// declares the view in its own pool — a landing whose view the trace does not
/// hold has nowhere to land and is refused by name
/// (`render_stage_buffer_writeback_unknown`).
///
/// Falsifiability, both directions: the fixture stores the word it read back
/// into its own slot, so the frame *and* the writeback are functions of the
/// bind's payload. A rail that bound the slot read-only, dropped the writeback
/// or landed fixed bytes would leave one of the two readings behind.
#[test]
fn a_writable_stage_buffer_lands_its_bytes_through_the_writeback_channel() {
    let _guard = engine_test_session();
    let stages = buffer_declaring_stages("render_frag_buffer_write.air", "reims_write_buffer_frag");
    eprintln!(
        "writable fixture declaration: {:?}",
        stages.fragment_stage_buffer_declarations
    );
    assert_eq!(
        stages.fragment_stage_buffer_declarations.len(),
        1,
        "the writable fixture declares one buffer"
    );
    assert!(
        stages.fragment_stage_buffer_declarations[0]
            .access
            .is_writable(),
        "and the declaration says so: {:?}",
        stages.fragment_stage_buffer_declarations[0]
    );

    // The bind's own bytes, and the destination the writeback lands at. The
    // landing is stated but not walked here: this test is the rail's half, and
    // the guest write it feeds is `runtime/draw/vulkan`'s.
    let payload = |word: [u8; 4]| -> BufferContent {
        let mut bytes = vec![0u8; 16];
        bytes[..4].copy_from_slice(&word);
        BufferContent::Bytes(std::sync::Arc::new(bytes))
    };
    let frame = |label: &str, content: &BufferContent| -> (Vec<u8>, Vec<StageWriteback>) {
        let landing = StageBufferLanding {
            gva: 0x4000_0000,
            pages: None,
        };
        let binds = [writable_bind(
            RenderPipelineStage::Fragment,
            0,
            content,
            landing,
        )];
        match provider_render::submit_render(
            &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
            &narrow_request(MTL_FORMAT_RGBA8_UNORM),
        ) {
            RenderRailOutcome::ProviderCompleted(out) => (out.bytes, out.stage_writebacks),
            other => panic!("{label}: a writable stage buffer leaves for the provider: {other:?}"),
        }
    };

    let one = frame(
        "writable stage buffer, 1.0f",
        &payload([0x00, 0x00, 0x80, 0x3f]),
    );
    eprintln!(
        "writable stage buffer: frame texel {:?}, writebacks {:?}",
        texel_at(&one.0, 0, 0),
        one.1
            .iter()
            .map(|writeback| (
                writeback.stage,
                writeback.index,
                writeback.offset,
                writeback.bytes.clone()
            ))
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        one.1.len(),
        1,
        "one writeback per writable stage buffer: {:?}",
        one.1
    );
    assert_eq!(
        (one.1[0].stage, one.1[0].index, one.1[0].offset),
        (RenderPipelineStage::Fragment, 0, 0),
        "the writeback names the slot the declaration named"
    );
    let mut want = vec![0u8; 16];
    want[..4].copy_from_slice(&[0x00, 0x00, 0x80, 0x3f]);
    assert_eq!(
        one.1[0].bytes, want,
        "the writeback is the bind's own bytes with the stage's store in them"
    );
    assert_eq!(
        texel_at(&one.0, 0, 0),
        [255, 0, 0, 255],
        "and the frame is the value the stage read through the same slot"
    );

    let zero = frame("writable stage buffer, 0.0f", &payload([0, 0, 0, 0]));
    eprintln!(
        "writable stage buffer, zeroed payload: frame texel {:?}, writeback head {:?}",
        texel_at(&zero.0, 0, 0),
        &zero.1[0].bytes[..4],
    );
    assert_eq!(
        zero.1[0].bytes[..4],
        [0, 0, 0, 0],
        "the writeback follows the payload rather than a constant"
    );
    assert_frames_differ(
        "the writable slot's bytes reach both the frame and the writeback",
        &one.0,
        &zero.0,
    );
    assert_ne!(
        one.1[0].bytes, zero.1[0].bytes,
        "and so does the writeback, which is what makes it the stage's own write"
    );
}

/// R9j, affine half: a `[[buffer(N)]]` argument whose reach is
/// `constant + stride * index` is stated as the contract's own affine proof
/// (R9f), bounded over the draw's own vertex index — and a proof this rail
/// cannot evaluate keeps the draw on the engine by name.
///
/// The fixture reads `positions[vertex_id]`, one `float2` per vertex, so the
/// declaration is the translator's measured access set and the frame is a
/// function of the buffer's bytes: moving the triangle moves the covered
/// texel. The refusals beside it are the two shapes the proof cannot be
/// evaluated over — an unbounded reach, and an access set naming an axis a
/// draw does not have.
#[test]
fn an_affine_stage_buffer_footprint_is_bounded_by_the_draw() {
    let _guard = engine_test_session();
    let mut stages = Stages {
        air: (
            fixture("render_vtx_buffer_positions.air"),
            fixture("render_frag.air"),
        ),
        vertex_entry: "reims_buffer_positions_vertex",
        fragment_entry: FRAGMENT_ENTRY,
        // The stage reads its vertices through the buffer, not through
        // `[[stage_in]]`: no attribute locations, and the request declares no
        // streams beside the index one.
        vertex_attribute_locations: Vec::new(),
        vertex_stage_buffer_declarations: Vec::new(),
        fragment_stage_buffer_declarations: Vec::new(),
    };
    stages.vertex_stage_buffer_declarations =
        declared_stage_buffers(&stages.air.0, RenderStage::Vertex, stages.vertex_entry);
    stages.fragment_stage_buffer_declarations =
        declared_stage_buffers(&stages.air.1, RenderStage::Fragment, stages.fragment_entry);
    eprintln!(
        "affine fixture declarations: vertex {:?}, fragment {:?}",
        stages.vertex_stage_buffer_declarations, stages.fragment_stage_buffer_declarations
    );
    assert_eq!(
        stages.vertex_stage_buffer_declarations,
        vec![StageBufferDeclaration {
            index: 0,
            access: StageBufferAccess::Read,
            footprint: StageBufferFootprint::Affine {
                // The `float2` the fixture loads is two four-byte reaches at
                // one stride — the same pair R9f's own fixture reports, read
                // from the same translator pin: `0/4 + vertex_id * 8`.
                accesses: vec![
                    metal_api_core::provider::AffineAccess {
                        base_offset: 0,
                        access_size: 4,
                        terms: vec![metal_api_core::provider::AffineTerm { axis: 0, stride: 8 }],
                    },
                    metal_api_core::provider::AffineAccess {
                        base_offset: 4,
                        access_size: 4,
                        terms: vec![metal_api_core::provider::AffineTerm { axis: 0, stride: 8 }],
                    },
                ],
            },
        }],
        "the affine fixture's own translation states two strided accesses over vertex_id"
    );
    assert!(
        stages.fragment_stage_buffer_declarations.is_empty(),
        "and the fragment half declares no buffer: {:#?}",
        stages.fragment_stage_buffer_declarations
    );

    let positions = |records: &[(f32, f32)]| -> BufferContent {
        BufferContent::Bytes(std::sync::Arc::new(f32x2(records)))
    };
    let request = || {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.vertex_attributes.clear();
        req
    };
    let frame = |label: &str, content: &BufferContent| -> Vec<u8> {
        let binds = [staged_bind(RenderPipelineStage::Vertex, 0, content)];
        match provider_render::submit_render(
            &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
            &request(),
        ) {
            RenderRailOutcome::ProviderCompleted(out) => out.bytes,
            other => panic!("{label}: an affine stage buffer leaves for the provider: {other:?}"),
        }
    };

    // The reviewed full-screen triangle, out of the buffer this time.
    let screen = frame(
        "affine stage buffer, full-screen triangle",
        &positions(&[(-1.0, -3.0), (-1.0, 1.0), (3.0, 1.0)]),
    );
    assert_texel_count("affine stage buffer, full-screen triangle", &screen);
    eprintln!(
        "affine stage buffer, full-screen triangle: texel {:?}",
        texel_at(&screen, 0, 0),
    );
    // The same draw with a degenerate vertex: the covered texel is a function
    // of the bytes the stage read, which is what makes the affine proof's bound
    // the thing that decided the view's length.
    let degenerate = frame(
        "affine stage buffer, degenerate",
        &positions(&[(-1.0, -3.0), (-1.0, -3.0), (-1.0, -3.0)]),
    );
    eprintln!(
        "affine stage buffer, degenerate triangle: texel {:?}",
        texel_at(&degenerate, 0, 0),
    );
    assert!(
        degenerate.iter().all(|byte| *byte == degenerate[0])
            || texel_at(&degenerate, 0, 0) != texel_at(&screen, 0, 0),
        "a degenerate triangle does not draw the screen triangle's texels"
    );

    // The refusals: an unbounded reach, and an access set over an axis a draw
    // does not have. Both are proofs the contract refuses by name, so the draw
    // stays on the engine under the footprint bucket rather than being declined
    // by the provider.
    let content = positions(&[(-1.0, -3.0), (-1.0, 1.0), (3.0, 1.0)]);
    let refuses = |label: &str, footprint: StageBufferFootprint| -> (String, String) {
        let mut stages = stages.clone();
        stages.vertex_stage_buffer_declarations = vec![StageBufferDeclaration {
            index: 0,
            access: StageBufferAccess::Read,
            footprint,
        }];
        let binds = [staged_bind(RenderPipelineStage::Vertex, 0, &content)];
        match provider_render::submit_render(
            &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
            &request(),
        ) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                (reason.slug().to_owned(), reason.detail().to_owned())
            }
            other => panic!("{label}: an unevaluable affine proof stays on the engine: {other:?}"),
        }
    };
    let (slug, detail) = refuses("unbounded reach", StageBufferFootprint::Unstated);
    eprintln!("affine refusal, unbounded: slug={slug} detail={detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_footprint");
    assert!(
        detail.contains("unbounded") && detail.contains("[[buffer(0)]]"),
        "the sentence names the reach and the slot: {detail}"
    );
    let (slug, detail) = refuses(
        "an axis a draw does not have",
        StageBufferFootprint::Affine {
            accesses: vec![metal_api_core::provider::AffineAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![metal_api_core::provider::AffineTerm { axis: 3, stride: 4 }],
            }],
        },
    );
    eprintln!("affine refusal, no such axis: slug={slug} detail={detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_footprint");
    assert!(
        detail.contains("cannot") && detail.contains("axis"),
        "the sentence names what could not be evaluated: {detail}"
    );
}

/// R9j, wire half: the declaration this rail states is carried by an `MCC1`
/// frame, the provider's own decoder reads it back, and the provider's own
/// admission answers for what it read.
///
/// The frame is the one the *production seam* produced — captured, not
/// rebuilt — because "the wire carries the declaration" is a statement about
/// bytes. Three readings come out of it:
///
/// 1. the decoded declaration and view fields, printed from the decoded trace;
/// 2. the capability answer the class gate reads, and the provider's answer to
///    the same frame under a snapshot that does not declare the shape
///    (`render_stage_buffer_unsupported`, with its fields);
/// 3. the shapes this increment does not touch: a trace whose stages declare no
///    buffer is encoded by nothing at all, so its bytes are exactly the ones it
///    had (`wire_counts`), and the frame bytes of the shapes that *do* exist are
///    R9h's regression.
#[test]
fn the_declaration_crosses_the_wire_and_the_provider_reads_it_back() {
    let _guard = engine_test_session();
    use reims_vgpu::backend::provider_wire;

    // The read-only shape R9d shipped: it declares one stage buffer, so the
    // rail crosses the wire for it.
    let stages = buffer_declaring_stages("render_frag_buffer.air", "reims_buffer_frag");
    let content = BufferContent::Bytes(std::sync::Arc::new(vec![0u8; 16]));
    let binds = [staged_bind(RenderPipelineStage::Fragment, 0, &content)];

    provider_wire::capture_submission_frames(true);
    let frames_before = provider_wire::wire_counts();
    match provider_render::submit_render(
        &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
        &narrow_request(MTL_FORMAT_RGBA8_UNORM),
    ) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("a declared read-only stage buffer is in class: {other:?}"),
    }
    let frames = provider_wire::captured_submission_frames();
    let counts = provider_wire::wire_counts();
    // Disarm before anything else is encoded: from here on the capture would be
    // holding this test's own re-encodes, and the reading below is about the
    // frames the *rail* produced.
    provider_wire::capture_submission_frames(false);
    assert_eq!(
        counts.submit_frames,
        frames_before.submit_frames + 1,
        "the seam produced exactly one submission frame"
    );
    assert_eq!(frames.len(), 1, "and the capture holds it");

    let frame = &frames[0];
    eprintln!(
        "wire submission frame: {} bytes, head {:02x} {:02x} {:02x} {:02x} {:02x}",
        frame.len(),
        frame[0],
        frame[1],
        frame[2],
        frame[3],
        frame[4],
    );
    assert_eq!(
        &frame[..4],
        b"MCC1",
        "the frame is the command channel's own, magic and all"
    );
    // The pass tag follows the frame header (tag byte, then the frame's own
    // layout): a stage-buffer pass is `0x13`, a sampled-plus-stage one `0x14`,
    // and a pass with neither keeps the bytes it always had.
    eprintln!("wire submission frame body head: {:02x?}", &frame[9..20]);

    let (trace, resources) = provider_wire::carried_submission(frame)
        .expect("the provider's own decoder reads the frame back");
    // The frame is a fixed point of the codec: decoding it and encoding the
    // decoded request again gives the same bytes, exactly as R9h's own
    // regression asserts for the shapes that predate the tag (`encode →
    // decode → encode` byte for byte). A field that crossed the wire and came
    // back differently would show up here rather than at the owner/provider
    // split.
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
    let declarations = trace
        .pipelines
        .iter()
        .find_map(|pipeline| pipeline.render.as_ref())
        .map(|render| render.stage_buffers.clone())
        .unwrap_or_default();
    let views = trace
        .passes
        .iter()
        .find_map(|pass| pass.as_render())
        .map(|pass| pass.stage_buffers.clone())
        .unwrap_or_default();
    assert_eq!(declarations.len(), 1, "one declaration crossed the wire");
    assert_eq!(views.len(), 1, "and one view beside it");
    let declaration = &declarations[0];
    let view = &views[0];
    let source = |view: &metal_api_core::provider::BufferView| match &view.source {
        metal_api_core::provider::BufferSource::OwnedBytes(bytes) => {
            format!("owned_bytes={}", bytes.len())
        }
        metal_api_core::provider::BufferSource::StagedLease(lease) => {
            format!("staged_lease={}", lease.get())
        }
        metal_api_core::provider::BufferSource::BorrowedNoCopy(lease) => {
            format!("borrowed_lease={}", lease.get())
        }
    };
    eprintln!(
        "wire decoded declaration: stage={:?} index={} access={:?} footprint={:?}",
        declaration.stage, declaration.index, declaration.access, declaration.footprint
    );
    eprintln!(
        "wire decoded view: stage={:?} metal_binding={} access={:?} offset={} length={} source={}",
        view.stage,
        view.view.metal_binding,
        view.view.access,
        view.view.offset,
        view.view.length,
        source(&view.view),
    );
    assert_eq!(declaration.stage, RenderPipelineStage::Fragment);
    assert_eq!(declaration.index, 0);
    assert_eq!(
        declaration.access,
        metal_api_core::provider::BufferAccess::Read
    );
    assert_eq!(
        declaration.footprint,
        metal_api_core::provider::FootprintProof::Static { max_bytes: 4 },
        "the declaration's own extent crossed the wire unchanged"
    );
    assert_eq!(
        view.view.access, declaration.access,
        "the view carries the access the declaration states"
    );
    assert_eq!(
        view.view.length, 16,
        "and so did the view it is paired with"
    );

    // The capability answer, read the way the class gate reads it: out of the
    // response frame the provider would send.
    let probe = VulkanComputeProvider::with_executor(
        VulkanExecutor::new().expect("the acceptance environment has a Vulkan device"),
    )
    .expect("the canonical provider builds");
    let support = provider_wire::stage_buffer_support(probe.device_epoch(), &probe.capabilities())
        .expect("the capability answer encodes and decodes");
    eprintln!(
        "wire capability answer: supports_render_stage_buffers={} max_render_stage_buffers={}",
        support.supported, support.maximum
    );
    assert!(support.supported, "this device declares the shape");
    assert_eq!(
        support.maximum as usize,
        metal_api_core::provider::MAX_RENDER_STAGE_BUFFERS,
        "and the same cap the contract states"
    );

    // The fail-closed arm: the same frame, admitted by a snapshot that does not
    // declare the shape. This is the provider's own answer, not this rail's —
    // the rail keeps the draw on the engine before a frame is ever sent.
    let mut unsupporting = probe.capabilities();
    unsupporting.supports_render_stage_buffers = false;
    unsupporting.max_render_stage_buffers = 0;
    let unsupported =
        provider_wire::stage_buffer_support(probe.device_epoch(), &unsupporting).expect("decode");
    eprintln!(
        "wire capability answer, undeclared snapshot: supports_render_stage_buffers={} \
         max_render_stage_buffers={}",
        unsupported.supported, unsupported.maximum
    );
    assert!(!unsupported.supported);
    let refusal = unsupporting
        .validate_trace(trace.clone(), resources.clone())
        .expect_err("a snapshot without the bit refuses the pass");
    eprintln!(
        "provider answer without the capability: class={:?} slug={} fields={:?}",
        refusal.class, refusal.slug, refusal.fields
    );
    assert_eq!(refusal.slug, "render_stage_buffer_unsupported");
    assert_eq!(
        refusal.class,
        metal_api_core::provider::ProviderErrorClass::Capability
    );
    assert_eq!(
        refusal.fields.get("bindings"),
        Some(&FieldValue::Unsigned(1)),
        "the refusal counts the bindings it refused"
    );
    // And the same frame under the device's own snapshot is admitted: the
    // capability bit is the only difference between the two answers.
    probe
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the provider that declares the shape admits the same frame");

    // The population this increment does not touch: no declaration, no frame —
    // so those shapes keep the bytes (and the path) they had.
    let clean = reviewed_stages();
    let before = provider_wire::wire_counts().submit_frames;
    match provider_render::submit_render(
        &inputs(&clean, RenderChainRole::SoleOrTail),
        &narrow_request(MTL_FORMAT_RGBA8_UNORM),
    ) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("the reviewed shape is in class: {other:?}"),
    }
    assert_eq!(
        provider_wire::wire_counts().submit_frames,
        before,
        "a trace whose stages declare no buffer keeps the path it had"
    );
    assert!(
        provider_wire::captured_submission_frames().is_empty(),
        "and produces no frame at all"
    );
    provider_wire::capture_submission_frames(false);
}
