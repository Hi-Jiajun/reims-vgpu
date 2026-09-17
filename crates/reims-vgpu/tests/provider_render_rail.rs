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

use metal_api_core::provider::{AttachmentFormat, VertexFormat};
use reims_vgpu::backend::provider_render::{
    self, ProviderRenderDecline, RenderRailInputs, RenderRailOutcome, MAX_ATTACHMENT_DIMENSION,
};
use reims_vgpu::backend::vulkan::engine::{
    self, BlendStateResource, BufferContent, DepthState, DrawRequest, IndexType,
    IndexedDrawResource, PrimitiveTopology, SamplerResource, VertexAttributeFormat,
    VertexAttributeResource, VertexStepFunction, ViewportResource,
};
use reims_vgpu::observe::Decline as _;
use reims_vgpu::protocol::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA8_UNORM};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// The engine is process-global, and so is the canonical rail's provider; the
/// engine suite's own lock is private to that test binary, so this file takes
/// its own and resets the engine under it, exactly as `vk_engine_parity` does.
fn engine_test_session() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = LOCK
        .get_or_init(|| {
            reims_vgpu::observe::redirect_logs_for_tests();
            Mutex::new(())
        })
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    engine::test_reset_engine();
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

/// The vertex module and the fragment module of the admitted shape: the rail's
/// own `float2`-per-vertex fixture and the reviewer's solid-colour fragment.
fn stage_air() -> (Vec<u8>, Vec<u8>) {
    (fixture("reims_indexed_tri.air"), fixture("render_frag.air"))
}

/// The entry names those two modules declare (`llvm-dis` shows them; the
/// canonical gate checks the contract against the reflection, and the
/// reflection carries the AIR function's name).
const VERTEX_ENTRY: &str = "reims_indexed_vertex";
const FRAGMENT_ENTRY: &str = "fmain";

/// Three vertices of `float2` clip positions covering the viewport — the same
/// full-screen triangle the reviewed milestone uses, but read from a
/// caller-held stream so the vertex-input half of the class is exercised.
///
/// `(-1, -3)`, `(-1, 1)`, `(3, 1)` as little-endian `f32` pairs.
const VERTEX_BYTES: [u8; 24] = [
    0x00, 0x00, 0x80, 0xbf, // -1.0
    0x00, 0x00, 0x40, 0xc0, // -3.0
    0x00, 0x00, 0x80, 0xbf, // -1.0
    0x00, 0x00, 0x80, 0x3f, //  1.0
    0x00, 0x00, 0x40, 0x40, //  3.0
    0x00, 0x00, 0x80, 0x3f, //  1.0
];

/// `[0, 1, 2]` as `u32` little-endian: the reviewed indexed shape, whose index
/// values select the three vertices in order.
const INDEX_BYTES: [u8; 12] = [
    0x00, 0x00, 0x00, 0x00, //
    0x01, 0x00, 0x00, 0x00, //
    0x02, 0x00, 0x00, 0x00, //
];

/// The canonical render rail's own window (`MAX_ATTACHMENT_DIMENSION`), which
/// is what the class gate states.
const WIDTH: u32 = MAX_ATTACHMENT_DIMENSION;
const HEIGHT: u32 = MAX_ATTACHMENT_DIMENSION;

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

/// The admitted request, in the terms the production seam builds it: one colour
/// attachment, one vertex stream (one `float2` attribute at location 0), one
/// index stream, and the pooled offscreen target whose whole frame is read back.
fn narrow_request(format: u16) -> DrawRequest {
    DrawRequest {
        width: WIDTH,
        height: HEIGHT,
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
        vertex_attributes: vec![VertexAttributeResource {
            location: 0,
            binding: 0,
            format: VertexAttributeFormat::parse(MTL_FORMAT_VERTEX_FLOAT2)
                .expect("Float2 is a vertex format"),
            offset: 0,
            stride: 8,
            step_function: VertexStepFunction::PerVertex,
            step_rate: 1,
            content: BufferContent::Bytes(std::sync::Arc::new(VERTEX_BYTES.to_vec())),
        }],
        color_attachment: Some(attachment(format)),
        color0_declared: Some(reims_vgpu::protocol::pass_action::LoadAction::Clear),
        ..Default::default()
    }
}

/// `MTLVertexFormat::Float2`.
const MTL_FORMAT_VERTEX_FLOAT2: u32 = 29;

fn inputs<'a>(air: &'a (Vec<u8>, Vec<u8>), writeback_guest: bool) -> RenderRailInputs<'a> {
    RenderRailInputs {
        vertex_air: &air.0,
        fragment_air: &air.1,
        vertex_entry: Some(VERTEX_ENTRY),
        fragment_entry: Some(FRAGMENT_ENTRY),
        writeback_guest,
    }
}

/// The engine arm: the same AIR, translated by this crate's own translator.
fn engine_request(air: &(Vec<u8>, Vec<u8>), format: u16) -> DrawRequest {
    let mut req = narrow_request(format);
    let words = |stage| -> Vec<u32> {
        let shader = reims_vgpu::runtime::m2v_cache::translate_cached_reflected(
            match stage {
                metal2vulkan::passes::Stage::Vertex => air.0.as_slice(),
                metal2vulkan::passes::Stage::Fragment => air.1.as_slice(),
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
    assert_eq!(
        pixels.len(),
        (WIDTH * HEIGHT * 4) as usize,
        "{label}: the attachment's whole extent has to come back"
    );
    for (index, texel) in pixels.chunks_exact(4).enumerate() {
        for channel in 0..4 {
            let got = texel[channel];
            let want = FRAGMENT_TEXEL[channel];
            assert!(
                (i32::from(got) - i32::from(want)).abs() <= 1,
                "{label}: texel {index} channel {channel} is {got}, expected ~{want}; the \
                 full-screen triangle has to cover every texel"
            );
        }
    }
}

#[test]
fn the_production_seam_completes_the_reviewed_shape_and_agrees_with_the_engine() {
    let _guard = engine_test_session();
    let air = stage_air();
    let req = narrow_request(MTL_FORMAT_RGBA8_UNORM);

    let provider = match provider_render::submit_render(&inputs(&air, true), &req) {
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

    let engine_req = engine_request(&air, MTL_FORMAT_RGBA8_UNORM);
    let engine_out = match engine::execute_draw_request(&engine_req) {
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

/// The guest-visible write order is the other half of the pairing: a BGRA
/// attachment is what a mapper-ref-texture target reads back in, and the rail
/// reports the order rather than converting, exactly as the engine does.
#[test]
fn a_bgra_attachment_reports_guest_scanout_order() {
    let _guard = engine_test_session();
    let air = stage_air();
    let req = narrow_request(MTL_FORMAT_BGRA8_UNORM);
    match provider_render::submit_render(&inputs(&air, true), &req) {
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
    let air = stage_air();
    let class = |req: &DrawRequest| match provider_render::submit_render(&inputs(&air, true), req) {
        RenderRailOutcome::NotInNarrowClass(_) => (),
        other => panic!("expected an out-of-class answer, got {other:?}"),
    };

    // A record that does not own the guest writeback is a chain intermediate.
    let req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    match provider_render::submit_render(&inputs(&air, false), &req) {
        RenderRailOutcome::NotInNarrowClass(_) => (),
        other => panic!("a chain intermediate is out of class: {other:?}"),
    }

    // A non-indexed draw.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.indexed = None;
    class(&req);

    // Two vertex attributes: the admitted class is one location-0 attribute.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    let first = &req.vertex_attributes[0];
    req.vertex_attributes.push(VertexAttributeResource {
        location: 1,
        binding: 1,
        format: first.format,
        offset: 8,
        stride: 16,
        step_function: VertexStepFunction::PerVertex,
        step_rate: 1,
        content: BufferContent::Bytes(std::sync::Arc::clone(match &first.content {
            BufferContent::Bytes(bytes) => bytes,
            BufferContent::GuestRuns(_) => panic!("the fixture is CPU-staged"),
        })),
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
    req.viewports.push(ViewportResource {
        x: 0.0,
        y: 0.0,
        width: WIDTH as f32,
        height: HEIGHT as f32,
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
        width: WIDTH,
        height: HEIGHT,
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

    // An attachment outside the canonical rail's declared window: the provider
    // would refuse it at admission, and a shape the provider always refuses is
    // not one this class executes.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.width = MAX_ATTACHMENT_DIMENSION + 1;
    req.height = MAX_ATTACHMENT_DIMENSION;
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.width = 512;
    req.height = 512;
    class(&req);
}

/// Fail-closed: an in-class shape the canonical provider itself refuses — here
/// an index view shorter than the draw's index count — is a typed decline on
/// reims' own vocabulary, never a silent re-run on the self-contained engine.
#[test]
fn an_in_class_shape_the_provider_refuses_is_a_typed_decline() {
    let _guard = engine_test_session();
    let air = stage_air();
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.indexed.as_mut().expect("indexed").content =
        BufferContent::Bytes(std::sync::Arc::new(INDEX_BYTES[..4].to_vec()));
    match provider_render::submit_render(&inputs(&air, true), &req) {
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

/// The class gate is pure: a request whose *pipeline* has no translated stages
/// is out of class before anything provider-side is touched, and the translated
/// registration gate is what a reused submission hits next.
#[test]
fn a_second_submission_reuses_the_registration() {
    let _guard = engine_test_session();
    let air = stage_air();
    let req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    for round in 0..2 {
        match provider_render::submit_render(&inputs(&air, true), &req) {
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
