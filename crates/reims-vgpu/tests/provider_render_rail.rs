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
    AttachmentFormat, BufferAccess, BufferSource, ComputeProvider, FieldValue, FootprintProof,
    RenderPipelineContract, RenderPipelineStage, SemanticDigest, StageBufferBinding,
    TextureBindingContract, TextureSource, TextureView, TracePass, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexLayout, VertexStep, MAX_RENDER_SAMPLERS,
    MAX_RENDER_TEXTURES,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, SpirvFeaturePolicy, TranslatedRenderPipelineRequest, TranslatedRenderStage,
    VulkanComputeProvider, VulkanExecutor,
};
use reims_vgpu::backend::provider_owner::{self, Region};
use reims_vgpu::backend::provider_render::{
    self, PresentSurfaceKey, ProviderRenderDecline, RenderChainRole, RenderInterfaceRefusal,
    RenderPresentRequest, RenderRailInputs, RenderRailOutcome, RenderRuntimeSampler,
    RenderSamplerFamily, RenderSamplerRefusal, RenderSamplerState, RenderTextureDeclaration,
    RenderTextureShape, RenderTextureShapeRefusal, StageBufferAccess, StageBufferBind,
    StageBufferDeclaration, StageBufferFootprint, StageBufferLanding, StageBufferWindow,
    StageWriteback,
};
use reims_vgpu::backend::vulkan::engine::{
    self, BlendStateResource, BufferContent, DepthState, DrawRequest, IndexType,
    IndexedDrawResource, PrimitiveTopology, ReadbackSkipReason, SampledImageResource,
    SampledSource, SamplerResource, ScissorResource, VertexAttributeFormat,
    VertexAttributeResource, VertexStepFunction, ViewportResource,
};
use reims_vgpu::observe::Decline as _;
use reims_vgpu::protocol::pixel_format::{
    MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA16_FLOAT, MTL_FORMAT_RGBA8_UNORM,
};
// The R28 helpers below are file-scope, so the two runtime types they name are
// too: the guest reference a window's run carries, and the window the
// registration ledger derived for it.
use reims_vgpu::backend::provider_wire;
use reims_vgpu::runtime::guest_ram::GuestRef;
use reims_vgpu::runtime::guest_ram_map::{GuestWindowRun, RegisteredWindow};
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
    // The engine's driver evidence — the breadcrumb a crash leaves and the
    // quarantine list that refuses the call next time — is named per process in
    // a test run and after the product in a boot, because these paths are
    // global to the machine and this file's runs sit beside live boots.
    // Asserted rather than assumed: the split depends on how this binary was
    // built (`cfg(test)` does not reach the lib), and a build that lost it
    // would otherwise only say so by refusing a boot's call later — see
    // `driver_breadcrumb::prefix_for`.
    assert!(
        reims_vgpu::observe::test_scoped(),
        "a rail run must be test-scoped: its driver evidence may not be the product's files"
    );
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
    /// The `[[texture(i)]]` arguments the fragment stage's reflection
    /// declares, with the AIR sampler state each was lowered against and the
    /// device bindings this crate's runtime resolves the request's own binds
    /// at (R10). Empty for every fixture whose fragment stage samples nothing.
    fragment_texture_declarations: Vec<RenderTextureDeclaration>,
    /// The sampler family the fragment stage's reflection declares (R12): the
    /// runtime `[[sampler(n)]]` arguments it binds and the AIR static samplers
    /// it carries. Empty for every fixture whose fragment stage samples through
    /// neither form, and filled from the translation itself where it does.
    sampler_family: RenderSamplerFamily,
    /// The Metal arguments outside the family the canonical translated render
    /// rail executes, across both stages (R10).
    texture_interface_refusals: Vec<RenderInterfaceRefusal>,
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
        fragment_texture_declarations: Vec::new(),
        sampler_family: RenderSamplerFamily::default(),
        texture_interface_refusals: Vec::new(),
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

/// The four-component twin of the reviewed shape (R14): one `float4` attribute
/// at location 0 whose `x`/`y` become the clip position and whose `w` divides
/// it — the member shape the canonical contract's `unorm8x4` / `unorm16x4`
/// storages pair with (`research/docs/23` §103).
fn vec4_stages() -> Stages {
    stages("reims_indexed_tri_vec4.air", "reims_vec4_vertex", &[0])
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

/// The sampled shape (R10): the reviewed vertex stage beside a fragment stage
/// that reads one `[[texture(0)]]` through the AIR `constexpr sampler` its own
/// module carries.
///
/// The declarations are the *production* walk over the fixture's own
/// translation (`provider_render::texture_declarations`), which is the fact the
/// runtime hands the rail; the expectations the sampling tests compare those
/// declarations with are written by hand in `sampled_stages_are_what_the_module_says`,
/// so a fixture whose reflection moves fails an assertion instead of quietly
/// changing what the seam is asked about.
fn sampled_stages() -> Stages {
    sampled_fragment_stages("render_frag_sampled_2d.air", "reims_sampled_frag")
}

/// The runtime-sampled shape (R12): the reviewed vertex stage beside a fragment
/// stage that reads one `[[texture(0)]]` through a runtime `[[sampler(0)]]`
/// argument — the module carries no sampler state, so the *request* states it
/// (`research/docs/23` §102).
///
/// The declarations and the sampler family are the *production* walks over the
/// fixture's own translation, which is the fact the runtime hands the rail; the
/// expectations the R12 tests compare them with are written by hand in
/// `the_runtime_sampler_declarations_are_what_the_module_says`, so a fixture or
/// reflection that moves fails an assertion instead of quietly changing what
/// the seam is asked about.
fn runtime_sampled_stages() -> Stages {
    sampled_fragment_stages(
        "render_frag_runtime_sampler.air",
        "reims_runtime_sampled_frag",
    )
}

/// The runtime-sampled shape's negative-side sibling (R21,
/// `research/docs/26` §44): the R12 module with its one fixed sample point
/// moved from `(1.375, 0.875)` to `(-0.375, 0.875)`, on the side of the
/// surface where the two mirroring address modes part from the two modes they
/// sit beside.
///
/// One sample point cannot state the whole address-mode rule: at `u = 1.375`
/// mirror-clamp-to-edge clamps, so it lands what clamp-to-edge lands; at
/// `u = -0.375` it mirrors, so it lands what mirror-repeat lands there. The
/// pair is what makes each added mode's frame falsifiable.
fn mirror_runtime_sampled_stages() -> Stages {
    sampled_fragment_stages(
        "render_frag_runtime_sampler_mirror.air",
        "reims_runtime_sampled_mirror_frag",
    )
}

/// The texel-fetched shape (R15): the reviewed vertex stage beside a fragment
/// stage that reads one `[[texture(0)]]` with `texture.read()` — Metal's
/// `access::read` qualifier.
///
/// The module carries no sampler at all: no AIR `constexpr sampler`, no runtime
/// `[[sampler(n)]]` argument, because an `access::read` argument is fetched by
/// integer coordinate at an explicit level of detail. The declarations are the
/// *production* walk over the fixture's own translation
/// (`provider_render::texture_declarations`), which is the fact the runtime
/// hands the rail; the expectation the fetch tests compare them with is written
/// by hand in `the_fetch_only_declarations_are_what_the_module_says`, so a
/// fixture or reflection that moves fails an assertion instead of quietly
/// changing what the seam is asked about.
fn fetch_stages() -> Stages {
    sampled_fragment_stages("render_frag_fetch_texture_2d.air", "reims_fetch_frag")
}

/// The fetched shape's sibling fixture: the same stage, binding and declaration
/// with the two texels it reads moved, so a test can watch the frame follow the
/// *coordinates* rather than anything the declaration states (R15).
fn fetch_far_stages() -> Stages {
    sampled_fragment_stages(
        "render_frag_fetch_texture_2d_far.air",
        "reims_fetch_far_frag",
    )
}

/// The mixed shape (R15): one fragment stage with a fetched `[[texture(0)]]`
/// and a sampled `[[texture(1)]]` — the sampled half reading through the
/// runtime `[[sampler(0)]]` argument R12 states.
///
/// One module, one draw, both access arms, which is the shape the guest's own
/// captures carry (one fetch-only texture beside runtime-sampled textures).
fn fetch_and_sample_stages() -> Stages {
    sampled_fragment_stages(
        "render_frag_fetch_and_sample.air",
        "reims_fetch_and_sample_frag",
    )
}

/// The sparse shape (R16): the runtime-sampler fixture's own body with its one
/// texture argument moved to `[[texture(3)]]` — Metal index 3 with nothing in
/// the argument table below it (`research/docs/23` §104).
///
/// This is the census v12 shape: all 35269 positional refusals of that boot read
/// literally "sampled texture 0 is `[[texture(3)]]`", and the captured modules
/// behind them carry their `img_tex_0A` texture at index 3 beside the runtime
/// `img_samp_0` `[[sampler(0)]]` argument — the family this fixture states.
///
/// The declarations are the *production* walk over the fixture's own
/// translation, and the expectation the sparse tests compare them with is
/// written by hand in `the_sparse_declarations_are_what_the_module_says`, so a
/// fixture whose reflection moves fails an assertion instead of quietly
/// changing what the seam is asked about.
fn sparse_sampled_stages() -> Stages {
    sampled_fragment_stages(
        "render_frag_runtime_sampler_index3.air",
        "reims_runtime_sampled_index3_frag",
    )
}

/// One fragment fixture's class-gate facts, taken from its own translation:
/// the texture declarations (`texture_declarations`, the walk that reads the
/// module's own sample sites for the runtime sampler family) and the sampler
/// family (`sampler_family`).
fn sampled_fragment_stages(fragment_fixture: &str, fragment_entry: &'static str) -> Stages {
    let mut stages = Stages {
        air: (fixture("reims_indexed_tri.air"), fixture(fragment_fixture)),
        vertex_entry: "reims_indexed_vertex",
        fragment_entry,
        vertex_attribute_locations: vec![0],
        vertex_stage_buffer_declarations: Vec::new(),
        fragment_stage_buffer_declarations: Vec::new(),
        fragment_texture_declarations: Vec::new(),
        sampler_family: RenderSamplerFamily::default(),
        texture_interface_refusals: Vec::new(),
    };
    let executor = VulkanExecutor::new().expect("the acceptance environment has a Vulkan device");
    let device =
        Device::new(std::sync::Arc::clone(&executor) as std::sync::Arc<dyn ComputeExecutor>);
    let function = device
        .new_library_with_binary_air(stages.air.1.clone())
        .expect("the fixture is a binary AIR module")
        .function(stages.fragment_entry)
        .expect("the fixture's entry exists");
    let translated = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fixture translates");
    // The same words both rails execute, in the numbering the declarations are
    // stated in: the translator's own, before any draw adds the fragment
    // sampled-band relocation.
    let words = translated
        .spirv()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect::<Vec<u32>>();
    stages.fragment_texture_declarations =
        reims_vgpu::backend::provider_render::texture_declarations(translated.reflection(), &words)
            .to_vec();
    stages.sampler_family =
        reims_vgpu::backend::provider_render::sampler_family(translated.reflection());
    // The same production read for the `[[buffer(N)]]` half, so a sampling
    // fixture that also declares a buffer hands the gate the declaration its
    // own translation reports rather than an empty list.
    stages.vertex_stage_buffer_declarations =
        declared_stage_buffers(&stages.air.0, RenderStage::Vertex, stages.vertex_entry);
    stages.fragment_stage_buffer_declarations =
        declared_stage_buffers(&stages.air.1, RenderStage::Fragment, stages.fragment_entry);
    stages
}

/// The one texel the sampled fixture reads: texel `(6, 3)` of the 8x4 surface
/// the tests draw at, whose centre is the fixed coordinate the fragment half
/// samples.
const SAMPLED_TEXEL: (usize, usize) = (6, 3);

/// The sampler resource the fixture's own AIR state derives, spelled from the
/// `MTL*` ordinals the runtime's `reflected_static_sampler_resource` maps that
/// state onto: nearest filtering, clamped addressing, one mip level, normalized
/// coordinates, no comparison and no anisotropy.
fn sampled_sampler_resource(binding: u32) -> SamplerResource {
    use reims_vgpu::protocol::sampler as mtl;
    SamplerResource {
        binding,
        min_filter: mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
        mag_filter: mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
        mip_filter: mtl::MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
        address_mode_u: mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        address_mode_v: mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        address_mode_w: mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        border_color: mtl::MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
        compare_function: reims_vgpu::backend::vulkan::engine::SamplerCompareFunction::Never,
        lod_min: 0.0f32.to_bits(),
        lod_max: f32::MAX.to_bits(),
        max_anisotropy: 1,
        unnormalized_coordinates: false,
    }
}

/// The attachment-covering draw with the sampled pair bound (R10): the
/// reviewed position stream and index stream beside one 2x2 `rgba8_unorm`
/// texture at the device binding the fragment stage's declaration names, and
/// the AIR static sampler resource that state derives.
fn sampled_request(stages: &Stages, texels: Vec<Vec<u8>>, extent: (u32, u32)) -> DrawRequest {
    sampled_request_in(stages, texels, extent, ash::vk::Format::R8G8B8A8_UNORM)
}

/// [`sampled_request`] with the *bind's own view format* stated (R19/E-TX1,
/// `research/docs/23` §107): the same sampled pair whose texture view is the
/// second four-byte 8-bit order, so its bytes carry the channels in the order
/// the guest's own `B8G8R8A8_UNORM` view names them.
fn sampled_request_in(
    stages: &Stages,
    texels: Vec<Vec<u8>>,
    extent: (u32, u32),
    format: ash::vk::Format,
) -> DrawRequest {
    let mut req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, &position_streams());
    req.width = extent.0;
    req.height = extent.1;
    let declaration = stages.fragment_texture_declarations[0];
    req.sampled_images.push(image_resource_in(
        declaration.binding,
        texels,
        extent,
        format,
    ));
    req.samplers
        .push(sampled_sampler_resource(declaration.sampler_binding));
    req
}

/// One texture the draw binds at `binding`: the request's own tightly packed
/// `rgba8_unorm` copy of `texels`, whose extent is the attachment's own (the
/// class gate's extent rule).
///
/// One spelling for every texture arm — a sampled texture, a runtime-sampled
/// one and a texel-fetched one reach the pass as the same view, because what
/// tells them apart is the declaration's sampler half and nothing in the view.
fn image_resource(binding: u32, texels: Vec<Vec<u8>>, extent: (u32, u32)) -> SampledImageResource {
    image_resource_in(binding, texels, extent, ash::vk::Format::R8G8B8A8_UNORM)
}

/// [`image_resource`] whose view format is the test's own (R19/E-TX1): the
/// same tightly packed bytes, under the byte order the caller states — which is
/// the fact the class gate resolves and the canonical view is created with.
fn image_resource_in(
    binding: u32,
    texels: Vec<Vec<u8>>,
    extent: (u32, u32),
    format: ash::vk::Format,
) -> SampledImageResource {
    let mut bytes = Vec::with_capacity(texels.len() * 4);
    for texel in texels {
        bytes.extend_from_slice(&texel);
    }
    SampledImageResource {
        binding,
        array_element: 0,
        descriptor_count: 1,
        width: extent.0,
        height: extent.1,
        layers: 1,
        kind: reims_vgpu_core::texture_shape::TextureKind::D2,
        multisampled: false,
        source: SampledSource::Bytes(std::sync::Arc::new(bytes)),
        byte_origin: Default::default(),
        format,
        identity: None,
        swizzle: Default::default(),
    }
}

/// The attachment-covering draw with one texel-fetched texture bound (R15):
/// the reviewed position stream beside the request's own copy of `texels`, at
/// the device binding the fragment stage's declaration names — and **no**
/// sampler resource at all, because the module names no sampler and the
/// canonical contract's fetched arm states none.
fn fetched_request(stages: &Stages, texels: Vec<Vec<u8>>, extent: (u32, u32)) -> DrawRequest {
    let mut req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, &position_streams());
    req.width = extent.0;
    req.height = extent.1;
    let declaration = stages.fragment_texture_declarations[0];
    req.sampled_images
        .push(image_resource(declaration.binding, texels, extent));
    req
}

/// The attachment-covering draw with one fetched and one sampled texture bound
/// (R15): both at the device bindings their declarations name, one view each,
/// beside the runtime `[[sampler(0)]]` state the sampled half reads through —
/// nearest filtering with clamped addressing, which is the state the fixture's
/// fixed coordinate is read against (R12's family).
fn fetch_and_sample_request(
    stages: &Stages,
    texels: Vec<Vec<u8>>,
    extent: (u32, u32),
) -> DrawRequest {
    use reims_vgpu::protocol::sampler as mtl;
    let mut req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, &position_streams());
    req.width = extent.0;
    req.height = extent.1;
    let fetched = stages.fragment_texture_declarations[0];
    let sampled = stages.fragment_texture_declarations[1];
    req.sampled_images
        .push(image_resource(fetched.binding, texels.clone(), extent));
    req.sampled_images
        .push(image_resource(sampled.binding, texels, extent));
    req.samplers.push(family_sampler_resource(
        sampled.sampler_binding,
        mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
        mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    ));
    req
}

/// One sampler resource in the canonical policy family's own four states
/// (R12): the `MTL*` ordinals the runtime's `request_sampler_policy` maps onto
/// `SamplerPolicy`, with every other field at the neutral value that family
/// admits.
fn family_sampler_resource(
    binding: u32,
    min_mag_filter: u32,
    address_mode: u32,
) -> SamplerResource {
    use reims_vgpu::protocol::sampler as mtl;
    SamplerResource {
        binding,
        min_filter: min_mag_filter,
        mag_filter: min_mag_filter,
        mip_filter: mtl::MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        address_mode_w: address_mode,
        border_color: mtl::MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
        compare_function: engine::SamplerCompareFunction::Never,
        lod_min: 0.0f32.to_bits(),
        lod_max: f32::MAX.to_bits(),
        max_anisotropy: 1,
        unnormalized_coordinates: false,
    }
}

/// [`family_sampler_resource`] with the mip filter stated as its own fact
/// (R21, `research/docs/26` §44): the widened family's third enumeration, which
/// every pre-R21 arm of these tests leaves at `not_mipmapped` through the
/// helper above.
fn widened_sampler_resource(
    binding: u32,
    min_mag_filter: u32,
    mip_filter: u32,
    address_mode: u32,
) -> SamplerResource {
    SamplerResource {
        mip_filter,
        ..family_sampler_resource(binding, min_mag_filter, address_mode)
    }
}

/// The attachment-covering draw with the runtime-sampled pair bound (R12): one
/// 8x4 `rgba8_unorm` texture at the declaration's device binding, and one
/// sampler state at the runtime `[[sampler(0)]]` argument's own device binding.
///
/// The state travels as the `MTL*` ordinals the runtime resolves the guest's
/// `MTLSamplerState` into, which is what `request_sampler_policy` maps onto the
/// canonical policy — the two arms of every reading below are therefore the two
/// states the *request* can state, and no state lives in the module.
fn runtime_sampled_request(
    stages: &Stages,
    texels: Vec<Vec<u8>>,
    extent: (u32, u32),
    min_mag_filter: u32,
    address_mode: u32,
) -> DrawRequest {
    widened_runtime_sampled_request(
        stages,
        texels,
        extent,
        min_mag_filter,
        reims_vgpu::protocol::sampler::MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
        address_mode,
    )
}

/// [`runtime_sampled_request`] with the *widened* state family stated (R21,
/// `research/docs/26` §44): min/mag filter, mip filter and address mode are
/// three independent facts of the request's own bind, exactly as the guest's
/// `MTLSamplerState` carries them.
fn widened_runtime_sampled_request(
    stages: &Stages,
    texels: Vec<Vec<u8>>,
    extent: (u32, u32),
    min_mag_filter: u32,
    mip_filter: u32,
    address_mode: u32,
) -> DrawRequest {
    let mut req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, &position_streams());
    req.width = extent.0;
    req.height = extent.1;
    let declaration = stages.fragment_texture_declarations[0];
    let runtime = stages.sampler_family.runtime[0];
    req.sampled_images
        .push(image_resource(declaration.binding, texels, extent));
    req.samplers.push(widened_sampler_resource(
        runtime.binding,
        min_mag_filter,
        mip_filter,
        address_mode,
    ));
    req
}

/// A half-step linear blend of two texels: what the fixture's wrapped
/// coordinate lands. Exact in eight bits because every channel of the tests'
/// texture is a multiple of sixteen.
fn half_blend(first: [u8; 4], second: [u8; 4]) -> [u8; 4] {
    let mut out = [0u8; 4];
    for channel in 0..4 {
        out[channel] = ((u16::from(first[channel]) + u16::from(second[channel])) / 2) as u8;
    }
    out
}

/// One 8x4 texture whose texels are all distinct, and the colour of the texel
/// the fragment fixture reads. The value is a function of the position, so
/// "changed the read texel" and "changed another texel" name different bytes.
fn sampled_texels(width: u32, height: u32) -> Vec<Vec<u8>> {
    (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                vec![
                    (x * 16).min(255) as u8,
                    (y * 64).min(255) as u8,
                    ((x + y) * 8).min(255) as u8,
                    255,
                ]
            })
        })
        .collect()
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

/// `unorm8x2` records: one byte per component, in the order the storage names.
fn unorm8x2(records: &[(u8, u8)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * 2);
    for (x, y) in records {
        out.extend_from_slice(&[*x, *y]);
    }
    out
}

/// `unorm8x4` records: four bytes per vertex, in the storage's own order.
fn unorm8x4(records: &[(u8, u8, u8, u8)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * 4);
    for (x, y, z, w) in records {
        out.extend_from_slice(&[*x, *y, *z, *w]);
    }
    out
}

/// The 16-bit twins of [`unorm8x2`] / [`unorm8x4`], written as `k * 257`.
///
/// `k * 257 / 65535` is exactly `k / 255`, so the same quotients the 8-bit
/// storages name are stated in the 16-bit ones and the frames of the two
/// storages are compared byte for byte rather than within a rounding step
/// (`research/docs/23` §103 states the same equality for the contract).
fn unorm16x2(records: &[(u8, u8)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * 4);
    for (x, y) in records {
        for component in [x, y] {
            out.extend_from_slice(&(u16::from(*component) * 257).to_le_bytes());
        }
    }
    out
}

fn unorm16x4(records: &[(u8, u8, u8, u8)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * 8);
    for (x, y, z, w) in records {
        for component in [x, y, z, w] {
            out.extend_from_slice(&(u16::from(*component) * 257).to_le_bytes());
        }
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

/// The reviewed request shape with its one attribute declared in `storage`
/// (R14): the four normalized storages are not the reviewed `float2` shape —
/// each is its own declaration, a different byte footprint and (for the
/// four-component pair) a different component shape — so the tests state the
/// attribute rather than reinterpreting [`request_with_streams`]' specs.
fn request_with_vertex_storage(
    format: u16,
    storage: u32,
    stride: u32,
    bytes: Vec<u8>,
) -> DrawRequest {
    let mut request = request_with_streams(format, &[]);
    request.vertex_attributes = vec![VertexAttributeResource {
        location: 0,
        binding: 0,
        format: VertexAttributeFormat::parse(storage).expect("a protocol vertex format"),
        offset: 0,
        stride,
        step_function: VertexStepFunction::PerVertex,
        step_rate: 1,
        content: BufferContent::Bytes(std::sync::Arc::new(bytes)),
    }];
    request
}

/// [`request_with_vertex_storage`] at the 8x4 attachment the y-convention
/// helpers read, so a normalized draw's coverage can be asserted texel by texel
/// instead of counted in a declared-window frame.
fn small_request_with_vertex_storage(storage: u32, stride: u32, bytes: Vec<u8>) -> DrawRequest {
    let mut request = request_with_vertex_storage(MTL_FORMAT_RGBA8_UNORM, storage, stride, bytes);
    request.width = ASYMMETRIC_WIDTH;
    request.height = ASYMMETRIC_HEIGHT;
    request
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
        // `true`: the capability the resident arms are elected by, and the one
        // this file's R7b battery drives. The *production* seam states `false`
        // until R4b's byte channel lands — every caller-side reader of a render
        // target reads the engine's registry, and census v15 measured what a
        // frame kept here costs them (6 824 `chain_resident_land_fail`,
        // 6 833 `load_target_content_not_ready`, one driven boot). A test that
        // wants the seam's own answer states it through [`inputs_held`].
        resident_frames_fetchable: true,
        // R23: no previous contents are handed over unless a test reads them
        // out of the engine's own registry and states them here — the shape
        // [`inputs_held_with_source`] drives.
        resident_source_bytes: None,
        // R26: the same read for the surface the *LOAD elision* named, stated
        // only by the shape [`inputs_held_with_surface_source`] drives.
        surface_resident_source_bytes: None,
        // R25: no predecessor's frame is handed over unless a test states the
        // bytes the walk carries — the shape
        // [`inputs_held_with_chain_value`] drives.
        chain_middle_source_bytes: None,
        // R24: no sampled GPU target's frame is handed over unless a test reads
        // one out of the engine's registry and states it — the shape
        // [`inputs_held_with_sampled_frames`] drives.
        sampled_target_frames: &[],
        vertex_attribute_locations: &stages.vertex_attribute_locations,
        vertex_stage_buffer_declarations: &stages.vertex_stage_buffer_declarations,
        fragment_stage_buffer_declarations: &stages.fragment_stage_buffer_declarations,
        // The R10 half: the fragment stage's own declarations, which are empty
        // for every fixture whose fragment samples nothing — the shape every
        // pre-R10 test in this file is about, and one whose requests answer
        // exactly as they did before the sampled-texture face existed.
        fragment_texture_declarations: &stages.fragment_texture_declarations,
        // The R12 half beside it: every fixture whose fragment stage declares
        // no sampler of either form carries the empty family, which is the
        // shape every pre-R12 test in this file is about.
        sampler_family: &stages.sampler_family,
        texture_interface_refusals: &stages.texture_interface_refusals,
        // The R9d half: no stage buffer of this draw is stated, which is the
        // shape every pre-R9d test in this file is about — a request whose
        // stages declare nothing answers exactly as it did before.
        stage_buffer_binds: &[],
        present: None,
    }
}

/// The same inputs with the caller stating that it **cannot** read a frame this
/// rail keeps — the production seam's own answer until R4b's byte channel
/// lands.
///
/// The class answers a record that withheld its readback by publishing the
/// frame instead of keeping it, and a record whose previous contents are this
/// rail's own image stays on the engine by name. See
/// [`RenderRailInputs::resident_frames_fetchable`].
fn inputs_held<'a>(stages: &'a Stages, role: RenderChainRole) -> RenderRailInputs<'a> {
    RenderRailInputs {
        resident_frames_fetchable: false,
        ..inputs(stages, role)
    }
}

/// [`inputs_held`] with the chain's own frame handed over as bytes (R23).
///
/// This is the production seam's other half: a caller that cannot fetch a frame
/// this rail keeps can still read the frame the *engine* holds — the record's
/// previous contents, named by the request's own `target_identity` — and hand
/// it over. The class then states the contract's trace-owned `Load` arm
/// (`LoadOp::Load`) instead of leaving the record on the engine, and publishes
/// the record's own frame rather than keeping it.
fn inputs_held_with_source<'a>(
    stages: &'a Stages,
    role: RenderChainRole,
    source: &'a [u8],
) -> RenderRailInputs<'a> {
    RenderRailInputs {
        resident_source_bytes: Some(source),
        ..inputs_held(stages, role)
    }
}

/// [`inputs_held`] with the frame the record *before* this one produced handed
/// over as bytes (R25).
///
/// The other half of the production seam's answer: the exec walk carries each
/// record's frame to the record after it, and a middle's previous contents are
/// that frame. The class states the contract's trace-owned `Load` arm for it
/// (`LoadOp::Load`), exactly as it does for the frame R23's caller reads out of
/// the engine's registry.
fn inputs_held_with_chain_value<'a>(
    stages: &'a Stages,
    role: RenderChainRole,
    value: &'a [u8],
) -> RenderRailInputs<'a> {
    RenderRailInputs {
        chain_middle_source_bytes: Some(value),
        ..inputs_held(stages, role)
    }
}

/// [`inputs_held`] with the frame of the surface the *LOAD elision* named handed
/// over as bytes (R26).
///
/// The seam's own statement of this door: the mapper-ref-texture composite's
/// LOAD was elided because the engine's registry already holds the surface's
/// contents under the identity the record itself names
/// (`mapper_ref_texture_load_currency_query` returns that identity), and the
/// caller that read the elision out owns the registry it read. The class states
/// the same trace-owned load R23's arm states, counts the population under its
/// own name (`render_provider_surface_resident_source_bytes`), and the caller's
/// own obligation — the landing that consumes this frame must advance the
/// mapping's content epoch — is stated where the seam hands the bytes over, not
/// here.
fn inputs_held_with_surface_source<'a>(
    stages: &'a Stages,
    role: RenderChainRole,
    source: &'a [u8],
) -> RenderRailInputs<'a> {
    RenderRailInputs {
        surface_resident_source_bytes: Some(source),
        ..inputs_held(stages, role)
    }
}

/// [`inputs_held`] with the frames the caller read out of the engine's registry
/// for sampled GPU targets the rail has no production to restate (R24).
///
/// This is the production seam's third hand-over (after R23's chain frame): the
/// same registry read, for the other kind of resident the census's Target arm
/// refuses — the one a *sampled* bind names rather than the record's own
/// attachment.
fn inputs_held_with_sampled_frames<'a>(
    stages: &'a Stages,
    role: RenderChainRole,
    frames: &'a [provider_render::SampledTargetFrame<'a>],
) -> RenderRailInputs<'a> {
    RenderRailInputs {
        sampled_target_frames: frames,
        ..inputs_held(stages, role)
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

/// One texel as each rail's frame carried it, as the comparison helpers report
/// a disagreement.
type TexelPair = ([u8; 4], [u8; 4]);

/// Two rails' *drawn* texels, compared within the one rounding step the eight
/// bit conversion carries (`assert_texel_near`'s tolerance, with a message that
/// names the first offending texel rather than printing two whole buffers).
///
/// The step is not slack: a fragment output of `0.5` is a tie in the eight-bit
/// conversion and Vulkan leaves that last bit to the implementation, so the two
/// rails' pipelines may round it either way. Every byte a record did *not* draw
/// is asserted byte-exact instead.
fn assert_frames_within_a_step(label: &str, provider: &[u8], engine: &[u8]) {
    assert_eq!(
        provider.len(),
        engine.len(),
        "{label}: the two frames are the attachment's whole extent"
    );
    let (width, _) = extent();
    assert_eq!(provider.len() % 4, 0);
    let mut differing = 0usize;
    let mut first: Option<((u32, u32), TexelPair)> = None;
    for (index, (out, kept)) in provider
        .chunks_exact(4)
        .zip(engine.chunks_exact(4))
        .enumerate()
    {
        let within_a_step =
            (0..4).all(|channel| (i32::from(out[channel]) - i32::from(kept[channel])).abs() <= 1);
        if within_a_step {
            continue;
        }
        differing += 1;
        if first.is_none() {
            first = Some((
                (index as u32 % width, index as u32 / width),
                (
                    [out[0], out[1], out[2], out[3]],
                    [kept[0], kept[1], kept[2], kept[3]],
                ),
            ));
        }
    }
    if let Some((texel, (out, kept))) = first {
        panic!(
            "{label}: {differing} of {} texels differ by more than a rounding step; the first \
             is at {texel:?}: provider {out:?} against the engine's {kept:?}",
            provider.len() / 4,
        );
    }
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
    // The R10 doors below print the way the dedicated refusal tests do, so the
    // viewport face's own names are evidence rather than only assertions.
    let door = |label: &str, expected: &str, req: &DrawRequest| match provider_render::submit_render(
        &inputs(&stages, RenderChainRole::SoleOrTail),
        req,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            eprintln!("door ({label}): {}\n  {}", reason.slug(), reason.detail());
            assert_eq!(reason.slug(), expected, "{label}: the refusal's own name");
        }
        other => panic!("{label}: expected an out-of-class answer, got {other:?}"),
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
    // record here loads from or keeps all leave their rails to the engine —
    // except the two faces R10 moved (the viewport above, and the blend entry
    // below): a blend the canonical pass can state *and* the v40 wire section
    // carries is in class, while a write mask, an alpha operation of its own
    // and the three factor families the provider refuses by name stay on the
    // engine under their own buckets.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.blend = Some(BlendStateResource {
        src_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_SOURCE_ALPHA,
        dst_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
        op_rgb: reims_vgpu_core::blend::MTL_BLEND_OPERATION_ADD,
        src_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ONE,
        dst_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ZERO,
        op_alpha: reims_vgpu_core::blend::MTL_BLEND_OPERATION_ADD,
    });
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("a v40-shaped blend is in the class: {other:?}"),
    }
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.blend = Some(BlendStateResource {
        src_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_SOURCE_ALPHA,
        dst_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
        op_rgb: reims_vgpu_core::blend::MTL_BLEND_OPERATION_ADD,
        src_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ONE,
        dst_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ZERO,
        op_alpha: reims_vgpu_core::blend::MTL_BLEND_OPERATION_MAX,
    });
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.blend = Some(BlendStateResource {
        src_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_BLEND_COLOR,
        dst_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ZERO,
        op_rgb: reims_vgpu_core::blend::MTL_BLEND_OPERATION_ADD,
        src_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ONE,
        dst_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ZERO,
        op_alpha: reims_vgpu_core::blend::MTL_BLEND_OPERATION_ADD,
    });
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.blend = Some(BlendStateResource {
        src_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_SOURCE_1_ALPHA,
        dst_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ZERO,
        op_rgb: reims_vgpu_core::blend::MTL_BLEND_OPERATION_ADD,
        src_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ONE,
        dst_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ZERO,
        op_alpha: reims_vgpu_core::blend::MTL_BLEND_OPERATION_ADD,
    });
    class(&req);
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.blend = Some(BlendStateResource {
        src_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_SOURCE_ALPHA_SATURATED,
        dst_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_SOURCE_ALPHA_SATURATED,
        op_rgb: reims_vgpu_core::blend::MTL_BLEND_OPERATION_ADD,
        src_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ONE,
        dst_alpha: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ZERO,
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
    // R10: a viewport the contract can name left this door. The covering
    // rect is byte for byte the default the class stated before the face
    // existed, so it is admitted; the shapes the contract has no spelling
    // for — a fractional or negative origin, an empty or out-of-bounds
    // rect, a depth range of its own, and more than one rect — keep the
    // engine under their own names below.
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("a covering viewport is in the class: {other:?}"),
    }
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.viewports.push(ViewportResource {
        x: 0.5,
        y: 0.0,
        width: 2.0,
        height: 2.0,
        min_depth: 0.0,
        max_depth: 1.0,
    });
    door(
        "fractional origin",
        "render_provider_out_of_class_viewport_spelling",
        &req,
    );
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.viewports.push(ViewportResource {
        x: -1.0,
        y: 0.0,
        width: 2.0,
        height: 2.0,
        min_depth: 0.0,
        max_depth: 1.0,
    });
    door(
        "negative origin",
        "render_provider_out_of_class_viewport_spelling",
        &req,
    );
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.viewports.push(ViewportResource {
        x: 0.0,
        y: 0.0,
        width: width as f32 + 1.0,
        height: height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    });
    door(
        "out-of-bounds rect",
        "render_provider_out_of_class_viewport_extent",
        &req,
    );
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.viewports.push(ViewportResource {
        x: 0.0,
        y: 0.0,
        width: 0.0,
        height: height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    });
    door(
        "empty rect",
        "render_provider_out_of_class_viewport_empty",
        &req,
    );
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.viewports.push(ViewportResource {
        x: 0.0,
        y: 0.0,
        width: 2.0,
        height: 2.0,
        min_depth: 0.25,
        max_depth: 0.75,
    });
    door(
        "depth range",
        "render_provider_out_of_class_viewport_depth",
        &req,
    );
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.viewports.push(ViewportResource {
        x: 0.0,
        y: 0.0,
        width: 2.0,
        height: 2.0,
        min_depth: 0.0,
        max_depth: 1.0,
    });
    req.viewports.push(ViewportResource {
        x: 1.0,
        y: 1.0,
        width: 2.0,
        height: 2.0,
        min_depth: 0.0,
        max_depth: 1.0,
    });
    door(
        "two rects",
        "render_provider_out_of_class_viewport_count",
        &req,
    );
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
    // R10: a sampler bind the fragment stage never declares left this door.
    // The module reads nothing at that slot, so neither rail's frame can move
    // — the same rule the stage-buffer door states for a bind no stage declares
    // (`stage_buffers_no_stage_declares_leave_for_the_provider_and_agree_with_the_engine`),
    // and the pass states nothing for it.
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("a sampler bind no stage declares is in class: {other:?}"),
    }
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

/// [`surface_identity`] at the format census v18's `chain_middle` shapes state:
/// every one of the 657 records is `fmt=0x50`, the guest's scanout order.
fn scanout_surface_identity(id: u32) -> engine::TargetIdentity {
    let (width, height) = extent();
    engine::TargetIdentity::Surface {
        id,
        width,
        height,
        generation: 1,
        format: ash::vk::Format::B8G8R8A8_UNORM,
    }
}

/// The walk's chain value in the attachment's own order — what the seam's
/// `chain_middle_source_frame` hands the class for a scanout-order attachment
/// (R25).
///
/// `encode_draw_chain` hands the walk every chain value in `SeedOrder::Rgba8`,
/// and the canonical attachment uploads the caller's bytes verbatim into the
/// view its pass declares, so the frame crosses that order boundary in the one
/// place the engine crosses it too (`write_staging_swap_rb`).
fn scanout_order(seed: &[u8]) -> Vec<u8> {
    let mut bytes = seed.to_vec();
    for texel in bytes.chunks_exact_mut(4) {
        texel.swap(0, 2);
    }
    bytes
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

/// R20: the split census v15 measured, reproduced on the rail, and the answer
/// the production seam states today.
///
/// A record that withholds its readback leaves a frame somebody else has to
/// read — the deferred GVA debt, the mapper-ref-texture store, and the next
/// record of the packet all read the *engine's* registry. A caller that cannot
/// fetch a kept frame therefore gets the frame **published** instead
/// (`render_provider_publish_held_resident`), and a record whose previous
/// contents are this rail's own image stays on the engine by name
/// (`render_provider_out_of_class_resident_source`).
///
/// Census v15 (`evidence/gate3-census-v15-2026-09-18`, driven macos-13,
/// `REIMS_VGPU_DRAW_LOG=1`) is the pair driven below: at `t=45522` the provider
/// answered `ok resident` for `gva=0x3cd2000`, and at `t=45525` the next record
/// of that packet was refused by the engine's LOAD gate
/// (`vk_draw_exec_load_target_content_not_ready`) and then failed its landing
/// (`chain_resident_land_fail ... read_target_no_ready_content`): 6 824 land
/// failures and a garbled desktop in one boot, every one of them on a GVA the
/// provider had just answered for. Step 0 reproduces that pair through the
/// rails; steps 1-3 are what the seam's own state does instead, and the last
/// assertion is that the frame lands.
#[test]
fn a_frame_the_caller_cannot_fetch_is_published_and_never_kept() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();
    let identity = surface_identity(0x7b_00_04);

    // 0. The failure, reproduced. A caller that *can* fetch a kept frame is
    //    answered with one, and the engine — asked for that identity's content
    //    the way the next record of a packet asks — refuses by name, because
    //    the frame is in the provider's image and no record ever stored it in
    //    the engine.
    match provider_render::submit_render(
        &inputs(&stages, RenderChainRole::SoleOrTail),
        &resident_seed_request(&identity),
    ) {
        RenderRailOutcome::ProviderCompletedResident(_) => (),
        other => panic!("the kept arm is what a fetching caller is answered with: {other:?}"),
    }
    let split = match engine::read_target(&identity) {
        Ok(_) => panic!("the engine holds no resident the provider wrote"),
        Err(error) => error,
    };
    let engine::DrawError::TargetRead(reason) = &split else {
        panic!("the refusal is the readback rail's own vocabulary: {split:?}");
    };
    assert_eq!(
        reason.slug(),
        "read_target_unknown_identity",
        "the split's own refusal, which `a_split_chain_fails_closed_on_the_engines_own_name` \
         pins, and the one the seam's state has to stop producing"
    );

    // 1. The same record under the seam's own state: the frame comes back as
    //    bytes — byte for byte the frame the engine's own record lands — and
    //    nothing stays in an image this caller cannot read.
    let published_before = route_count("render_provider_publish_held_resident");
    let stores_before = route_count("render_provider_resident_store");
    let seed = match provider_render::submit_render(
        &inputs_held(&stages, RenderChainRole::SoleOrTail),
        &resident_seed_request(&identity),
    ) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("a withheld readback is answered with the frame: {other:?}"),
    };
    assert_texel_count("published held frame", &seed);
    assert_eq!(
        route_count("render_provider_publish_held_resident") - published_before,
        1,
        "the held population is counted where the answer happens"
    );
    assert_eq!(
        route_count("render_provider_resident_store") - stores_before,
        0,
        "no frame stays in an image this caller cannot read"
    );
    // The engine's own answer for the same shape, with its readback *kept*: a
    // resident-store record withholds it there too, so the control states the
    // one difference this comparison is about.
    let mut engine_control = resident_seed_request(&identity);
    engine_control.skip_readback = false;
    engine_control.readback_skip_reason = ReadbackSkipReason::None;
    let Some(engine_seed) = engine_pixels("published seed", &stages, engine_control) else {
        return;
    };
    assert!(
        !engine_seed.is_empty() && seed == engine_seed,
        "the published frame is the frame the engine's own record lands, not any frame: \
         provider {} bytes against the engine's {}",
        seed.len(),
        engine_seed.len()
    );

    // 2. A record whose previous contents are this rail's own image: while the
    //    arms are held no record here ever stored that image, so the record
    //    stays on the engine — which owns the resident the caller's chain
    //    names — instead of being answered with an image nothing wrote.
    let chained = resident_load_request(&identity, true);
    let source_bucket_before = route_count("render_provider_out_of_class_resident_source");
    match provider_render::submit_render(
        &inputs_held(&stages, RenderChainRole::SoleOrTail),
        &chained,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_resident_source",
            "a resident-sourced record keeps the engine while the arms are held: {reason}"
        ),
        other => panic!("a resident-sourced record is not the held class's: {other:?}"),
    }
    assert_eq!(
        route_count("render_provider_out_of_class_resident_source") - source_bucket_before,
        1,
        "the refused shape is counted under its own name in the census's own bucket vocabulary"
    );

    // 3. The route the walk takes instead: the publish answer is handed on as a
    //    CPU seed (`multi_draw_chain_source`'s `Cpu` arm), and the frame lands.
    //    The engine's own two-record resident chain is the control.
    let mut seeded = resident_load_request(&identity, true);
    seeded.load_from_target = false;
    seeded.target_rgba8 = Some(std::sync::Arc::new(seed));
    match provider_render::submit_render(
        &inputs_held(&stages, RenderChainRole::SoleOrTail),
        &seeded,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_load_seed",
            "a CPU-seeded record stays on the engine: {reason}"
        ),
        other => panic!("a seeded record is not the provider's: {other:?}"),
    }
    let Some(landed) = engine_pixels("published chain", &stages, seeded) else {
        return;
    };
    let Some(own_chain) = engine_pixels(
        "resident chain",
        &stages,
        resident_load_request(&identity, true),
    ) else {
        return;
    };
    assert!(
        !landed.is_empty() && landed == own_chain,
        "the seed the caller carried forward lands the same composite as the engine's own \
         resident chain ({} bytes against {}): the split's frame is not lost, it is handed on",
        landed.len(),
        own_chain.len()
    );
    assert_texel_near(
        "published chain: the last texel inside the rectangle",
        texel_at(&landed, half_of(width) - 1, height / 2),
        FRAGMENT_TEXEL,
    );
    for x in half_of(width)..width {
        assert_eq!(
            texel_at(&landed, x, height / 2),
            RESIDENT_SEED_TEXEL,
            "texel ({x}, {}) keeps the published frame's own bytes: a route that cleared \
             instead of seeding lands another colour here",
            height / 2,
        );
    }
}

/// The frame the engine's own chain resident holds, in the order the attachment
/// declares — what the production seam hands the canonical rail as
/// `resident_source_bytes` (R23).
///
/// This is the seam's own read (`runtime::draw::vulkan`'s
/// `resident_chain_source_frame`): `TargetIdentity::is_bgra` is the attachment's
/// order, and four-byte colour is the only width that readback narrows.
fn engine_chain_source(identity: &engine::TargetIdentity) -> Vec<u8> {
    let frame = engine::read_target(identity).expect("the chain's frame is readable");
    if identity.is_bgra() {
        frame.into_bgra8().expect("four-byte colour")
    } else {
        frame.into_rgba8().expect("four-byte colour")
    }
}

/// [`engine_chain_source`] in semantic RGBA8, so a provider frame and an engine
/// frame can be compared as colours rather than as two runs of one rail.
fn engine_chain_semantic(identity: &engine::TargetIdentity) -> Vec<u8> {
    semantic_rgba(engine_chain_source(identity), identity.is_bgra())
}

/// R23: the frame the *engine* holds, handed over as bytes.
///
/// Census v17's first refusal is this shape — 3 271 records (73.2 %) with
/// `skip=resident store=1 seed=chain load=Load` — and every one of them belongs
/// to a packet whose head the class refused for another reason: the engine drew
/// that head, kept the chain's frame in its own registry, and the records after
/// it name contents this rail never stored. `LoadOp::Resident` would be an image
/// no provider pass wrote, so the class carries the frame the other way: the
/// caller reads it out of the registry the chain names and hands it over, the
/// pass states the contract's trace-owned load (`LoadOp::Load`), and the rail
/// uploads exactly those bytes into the pass's own image before it opens.
///
/// The counterweight is R20's, unchanged: this caller cannot fetch a frame the
/// rail keeps, so the record's *own* frame is published rather than kept. The
/// comparison is the engine's own answer for the same record from the same
/// previous contents — the engine's registry still holds them, because the
/// provider never wrote there.
#[test]
fn the_chains_frame_the_engine_holds_carries_the_record_into_the_provider() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();

    // 1. The packet's head: refused by the class (a resident store with a
    //    withheld readback is the shape every other reason sends here), drawn
    //    by the engine, and its frame kept in the engine's registry.
    let head_identity = surface_identity(0x7b_00_08);
    let Some(head) = engine_pixels("chain head", &stages, resident_seed_request(&head_identity))
    else {
        return;
    };
    assert!(
        head.is_empty(),
        "a resident store publishes nothing on the engine either"
    );

    // 2. The record that composites onto it, without the bytes: exactly what
    //    census v17 counts under `resident_source`.
    let chained = {
        let mut chained = resident_load_request(&head_identity, false);
        chained.continues_render_pass = true;
        chained
    };
    let source_bucket_before = route_count("render_provider_out_of_class_resident_source");
    match provider_render::submit_render(&inputs_held(&stages, RenderChainRole::Middle), &chained) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_resident_source",
            "a chain whose frame is the engine's stays on the engine while nothing is handed \
             over: {reason}"
        ),
        other => panic!("a source-less resident load is not the held class's: {other:?}"),
    }
    assert_eq!(
        route_count("render_provider_out_of_class_resident_source") - source_bucket_before,
        1,
        "the refusal is counted under the census's own bucket vocabulary"
    );
    // The sentence a census reader joins that bucket to, verbatim.
    let refusal = match provider_render::submit_render(
        &inputs_held(&stages, RenderChainRole::Middle),
        &chained,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => reason.detail().to_owned(),
        other => panic!("the source-less shape is refused twice the same way: {other:?}"),
    };
    eprintln!("R23 refusal, verbatim: {}", refusal);
    assert!(
        refusal.contains(
            "a record whose previous contents are the live GPU image stays on the engine \
             while the caller can neither read a frame this rail keeps nor hand the frame over"
        ),
        "the sentence names both halves the caller lacks: {refusal}"
    );

    // 3. The frame the chain names, read out of the engine by the caller.
    let source = engine_chain_source(&head_identity);
    assert_eq!(
        source.len(),
        (width as usize) * (height as usize) * 4,
        "the caller hands over the attachment's whole packed extent"
    );

    // 4. The packet's *middle* record — the census's dominant shape (`wb=0`,
    //    `skip=resident`, `continues=1`) — with the bytes handed over: in
    //    class, answered by the provider, and **published**, because this
    //    caller cannot fetch a frame the rail keeps (R20's arm, unchanged).
    let published_before = route_count("render_provider_publish_held_resident");
    let carried_before = route_count("render_provider_resident_source_bytes");
    let stores_before = route_count("render_provider_resident_store");
    let loads_before = route_count("render_provider_resident_load");
    let provider_middle = match provider_render::submit_render(
        &inputs_held_with_source(&stages, RenderChainRole::Middle, &source),
        &chained,
    ) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("a resident load whose frame the caller hands over is in class: {other:?}"),
    };
    assert_texel_count("published middle frame", &provider_middle);
    assert_texel_near(
        "published middle: the last texel inside the rectangle",
        texel_at(&provider_middle, half_of(width) - 1, height / 2),
        FRAGMENT_TEXEL,
    );
    for x in half_of(width)..width {
        assert_eq!(
            texel_at(&provider_middle, x, height / 2),
            RESIDENT_SEED_TEXEL,
            "texel ({x}, {}) keeps the frame that was handed over: a channel that uploaded the \
             wrong bytes, or cleared instead of loading, lands another colour here",
            height / 2,
        );
    }

    // 5. The engine's own answer for the same middle record, from the same
    //    frame (the provider wrote nothing into the engine's registry, so it is
    //    still the head's). A middle record keeps its frame there, so the
    //    engine's answer is read back out of the registry to compare.
    let mut middle_control = resident_load_request(&head_identity, false);
    middle_control.continues_render_pass = true;
    let Some(engine_middle) = engine_pixels("engine middle", &stages, middle_control) else {
        return;
    };
    assert!(
        engine_middle.is_empty(),
        "a resident middle publishes nothing on the engine either"
    );
    let engine_kept = engine_chain_semantic(&head_identity);
    assert_texel_count("engine middle frame", &engine_kept);
    eprintln!(
        "R23 carried frame: identity={:?}; handed over {} bytes; provider published {} bytes; \
         engine kept {} bytes; texel at the scissor edge [{},{},{}], outside [{},{},{}]; \
         resident_source_bytes +{}; publish_held_resident +{}; resident_store +{}; \
         resident_load +{}",
        head_identity,
        source.len(),
        provider_middle.len(),
        engine_kept.len(),
        texel_at(&provider_middle, half_of(width) - 1, height / 2)[0],
        texel_at(&provider_middle, half_of(width) - 1, height / 2)[1],
        texel_at(&provider_middle, half_of(width) - 1, height / 2)[2],
        texel_at(&provider_middle, width - 1, height / 2)[0],
        texel_at(&provider_middle, width - 1, height / 2)[1],
        texel_at(&provider_middle, width - 1, height / 2)[2],
        route_count("render_provider_resident_source_bytes") - carried_before,
        route_count("render_provider_publish_held_resident") - published_before,
        route_count("render_provider_resident_store") - stores_before,
        route_count("render_provider_resident_load") - loads_before,
    );
    assert_eq!(
        provider_middle,
        engine_kept,
        "the frame the provider published from the handed-over bytes and the frame the engine \
         kept after the same record are not two runs of one rail: provider {} bytes against the \
         engine's {}",
        provider_middle.len(),
        engine_kept.len(),
    );

    assert_eq!(
        route_count("render_provider_resident_source_bytes") - carried_before,
        1,
        "the carried population is counted where the answer happens"
    );
    assert_eq!(
        route_count("render_provider_publish_held_resident") - published_before,
        1,
        "the record's own frame is published, never kept"
    );
    assert_eq!(
        route_count("render_provider_resident_store") - stores_before,
        0,
        "no frame stays in an image this caller cannot read"
    );
    assert_eq!(
        route_count("render_provider_resident_load") - loads_before,
        0,
        "the pass began from the caller's bytes, not from a resident load"
    );

    // 6. The packet's *last* record (`wb=1`, `continues=1`), starting from the
    //    frame the middle record left in the engine's registry. Its store
    //    publishes a writeback on both rails, so the comparison is again the
    //    engine's own answer from the same previous contents.
    let tail_source = engine_chain_source(&head_identity);
    let tail = {
        let mut tail = resident_load_request(&head_identity, true);
        tail.continues_render_pass = true;
        tail
    };
    let carried_before = route_count("render_provider_resident_source_bytes");
    let published_before = route_count("render_provider_publish_held_resident");
    let provider_tail = match provider_render::submit_render(
        &inputs_held_with_source(&stages, RenderChainRole::SoleOrTail, &tail_source),
        &tail,
    ) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!(
            "the packet's last resident record is in class with its chain's frame: {other:?}"
        ),
    };
    let Some(engine_tail) = engine_pixels("engine tail", &stages, {
        let mut tail = resident_load_request(&head_identity, true);
        tail.continues_render_pass = true;
        tail
    }) else {
        return;
    };
    assert_texel_count("engine tail frame", &engine_tail);
    assert_eq!(
        provider_tail,
        engine_tail,
        "the packet's last record lands the same frame on both rails: provider {} bytes against \
         the engine's {}",
        provider_tail.len(),
        engine_tail.len(),
    );
    assert_eq!(
        route_count("render_provider_resident_source_bytes") - carried_before,
        1,
        "the last record's chain frame is carried too"
    );
    assert_eq!(
        route_count("render_provider_publish_held_resident") - published_before,
        0,
        "a record whose store publishes its own readback is not the held arm"
    );

    // 7. A caller that hands over bytes of the wrong extent is a wiring bug
    //    named as one, not a provider decline: the class keeps the record on
    //    the engine under its own slug.
    let short = &source[..source.len() - 4];
    let short_identity = surface_identity(0x7b_00_09);
    let _ = engine_pixels(
        "short chain head",
        &stages,
        resident_seed_request(&short_identity),
    );
    let mut short_chained = resident_load_request(&short_identity, true);
    short_chained.continues_render_pass = true;
    match provider_render::submit_render(
        &inputs_held_with_source(&stages, RenderChainRole::SoleOrTail, short),
        &short_chained,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_resident_source_shape",
            "bytes that are not the attachment's extent keep the record on the engine: {reason}"
        ),
        other => panic!("a short carried frame is not the held class's: {other:?}"),
    }
}

/// R25's own seed colour: an eight-bit colour whose first and third channels
/// differ, so a rail that handed the pass the walk's bytes in the wrong order
/// lands a *different* texel. The fragment's own texel and a colour like green
/// are symmetric under that exchange and would read the same either way.
const WALK_SEED_TEXEL: [u8; 4] = [17, 34, 51, 255];

/// R25: the frame the *walk* carries, handed over as bytes.
///
/// Census v18's fourth bucket is this shape — `chain_middle`, 657 records
/// (8.0 %), every one of them `fmt=0x50 load=Load wb=0 skip=resident store=1
/// seed=bytes continues=1`. The record is the packet's *middle*: it begins from
/// the frame the record before it produced, and that frame is the exec walk's
/// own chain value — `encode_draw_chain` normalizes both rails' chain values to
/// `SeedOrder::Rgba8` and the walk hands them on as the request's
/// `target_rgba8`. R7b's arm cannot name it (no provider pass stored an image
/// under the attachment's identity), so the class carries it the same way R23
/// carries the engine's: the caller — the walk — hands the frame over, and the
/// pass states the contract's trace-owned `Load`.
///
/// The comparison is the engine's own answer for the same record from the same
/// previous contents: a middle keeps its frame in its resident on the engine,
/// so the engine's frame is read back out of the identity the record names.
#[test]
fn a_guest_backed_chain_middle_leaves_for_the_provider_and_its_tail_does_not() {
    // Census v21's `guest_backing`
    // (`evidence/gate3-census-v21-2026-09-18`): 22 latched shapes, all
    // `door=mapping` — the attachment's bytes are the guest's own pages — of
    // which 21 are the *middle* of a multi-record packet (`wb=0
    // continues=1 pass_cont=1`) and one is the packet's tail (`wb=1`).
    //
    // The two halves are one question: whose frame has to reach the guest's
    // pages? The middle's frame is the chain value the exec walk hands the
    // record after it, so the class answers it with the published frame — the
    // same answer R25 gives the non-guest-backed middle. The tail's store *is*
    // the guest writeback, and that landing is the arm this rail still does
    // not carry, so its refusal and its sentence stay by name.
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();
    let half = half_of(width);

    let identity = scanout_surface_identity(0x7d_00_01);
    let mut seed = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for _ in 0..(width * height) {
        seed.extend_from_slice(&WALK_SEED_TEXEL);
    }
    // The record's own shape: a guest-backed attachment (the mapper-ref-texture
    // surface's pages), the load the guest declared, the walk's chain value,
    // the withheld readback a middle carries, and the partial scissor the
    // census's shapes show.
    let guest_backed = || {
        let mut middle = request_with_streams(MTL_FORMAT_BGRA8_UNORM, &position_streams());
        middle.target_identity = Some(identity.clone());
        middle.color0_declared = Some(reims_vgpu::protocol::pass_action::LoadAction::Load);
        middle.target_rgba8 = Some(std::sync::Arc::new(seed.clone()));
        middle.skip_readback = true;
        middle.readback_skip_reason = ReadbackSkipReason::ResidentStore;
        middle.continues_render_pass = true;
        middle.render_pass_continues = true;
        middle.scissors.push(ScissorResource {
            x: 0,
            y: 0,
            width: half,
            height,
        });
        middle.guest_target_memory = Some(guest_target_memory());
        middle
    };

    // 1. The tail (`wb=1`, census group B): its frame is the one the guest's
    //    pages are owed, and the refusal keeps its own slug and sentence. Its
    //    own shape is the census's one non-`Load` row — a `Clear` that opens
    //    the raster and hands it on — because a tail that stated the walk's
    //    seed instead would be refused one gate earlier, under `load_seed`,
    //    exactly as census v21 measures those shapes.
    let tail = || {
        let mut tail = request_with_streams(MTL_FORMAT_BGRA8_UNORM, &position_streams());
        tail.target_identity = Some(identity.clone());
        tail.color0_declared = Some(reims_vgpu::protocol::pass_action::LoadAction::Clear);
        tail.skip_readback = true;
        tail.readback_skip_reason = ReadbackSkipReason::ResidentStore;
        tail.guest_target_memory = Some(guest_target_memory());
        tail
    };
    let bucket_before = route_count("render_provider_out_of_class_guest_backing");
    let refusal = match provider_render::submit_render(
        &inputs_held(&stages, RenderChainRole::SoleOrTail),
        &tail(),
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_guest_backing",
                "the tail keeps the census's own bucket: {reason}"
            );
            reason.detail().to_owned()
        }
        other => panic!("a guest-backed tail is not the class's answer: {other:?}"),
    };
    assert_eq!(
        route_count("render_provider_out_of_class_guest_backing") - bucket_before,
        1,
        "the refusal is counted under the census's own bucket vocabulary"
    );
    eprintln!("E-TX8 tail refusal, verbatim: {refusal}");
    assert!(
        refusal.contains(
            "a record whose attachment is backed by the guest's own pages stays on the engine \
             while its frame is the one the guest's pages are owed"
        ),
        "the sentence names the half the rail lacks: {refusal}"
    );

    // 2. The middle (`wb=0`, census group A) with the walk's frame handed over:
    //    in class, answered by the provider, and published — the walk consumes
    //    that frame as the next record's seed, and the packet's last record is
    //    the one that lands the complete frame in the guest's pages.
    let bucket_before = route_count("render_provider_out_of_class_guest_backing");
    let handed = scanout_order(&seed);
    match provider_render::submit_render(
        &inputs_held_with_chain_value(&stages, RenderChainRole::Middle, &handed),
        &guest_backed(),
    ) {
        RenderRailOutcome::ProviderCompleted(out) => {
            assert!(out.bgra, "a Bgra8Unorm attachment reads back in BGRA order");
            let frame = semantic_rgba(out.bytes, out.bgra);
            assert_texel_count("published guest-backed middle frame", &frame);
            assert_texel_near(
                "published guest-backed middle: the last texel inside the rectangle",
                texel_at(&frame, half_of(half) - 1, height / 2),
                FRAGMENT_TEXEL,
            );
            for x in half..width {
                assert_eq!(
                    texel_at(&frame, x, height / 2),
                    WALK_SEED_TEXEL,
                    "texel ({x}, {}) keeps the frame the walk handed over",
                    height / 2,
                );
            }
        }
        other => {
            panic!("a guest-backed middle whose frame the walk carries is in class: {other:?}")
        }
    }
    assert_eq!(
        route_count("render_provider_out_of_class_guest_backing") - bucket_before,
        0,
        "the admitted middle charges no refusal: the bucket moved with the class"
    );
}

/// The guest's own allocation behind a mapper-ref-texture surface: one host
/// mapping and the page footprint that describes it, built the way the
/// engine's own tests build it (`images_and_registry.rs`).
fn guest_target_memory() -> reims_vgpu::backend::vulkan::engine::GuestTargetMemory {
    reims_vgpu::backend::vulkan::engine::GuestTargetMemory {
        backing: reims_vgpu::backend::vulkan::engine::GuestTargetBacking {
            allocation_host_ptr: 0x1000,
            allocation_len: 0x4000,
            plane_offset: 0,
            row_pitch: 64,
        },
        import: std::sync::Arc::new(
            reims_vgpu::runtime::guest_ram::GuestRamImport::new_host_allocation(
                0x1000, 0x4000, 0x1000,
            )
            .expect("the host allocation is a valid import"),
        ),
        footprint: reims_vgpu::runtime::guest_ram::GuestPageFootprint::new(
            std::sync::Arc::from([0x1000_u64]),
            0x1000,
        )
        .expect("one page"),
    }
}

#[test]
fn the_chain_value_the_walk_carries_reaches_the_provider() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();
    let half = half_of(width);

    // The census's own shape: the scanout-order attachment (`fmt=0x50`), the
    // identity the record names, and the resident-store pair a record whose
    // readback is withheld carries.
    let identity = scanout_surface_identity(0x7c_00_01);
    // The frame the record before this one produced, as the walk holds it:
    // semantic RGBA8, in one colour the fragment stage never draws and whose
    // channels are not symmetric under the order exchange, so neither "the pass
    // cleared instead" nor "the bytes went in un-exchanged" can pass as "the
    // walk's bytes were carried".
    let mut seed = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for _ in 0..(width * height) {
        seed.extend_from_slice(&WALK_SEED_TEXEL);
    }
    let middle_request = || {
        let mut middle = request_with_streams(MTL_FORMAT_BGRA8_UNORM, &position_streams());
        middle.target_identity = Some(identity.clone());
        middle.color0_declared = Some(reims_vgpu::protocol::pass_action::LoadAction::Load);
        middle.target_rgba8 = Some(std::sync::Arc::new(seed.clone()));
        middle.skip_readback = true;
        middle.readback_skip_reason = ReadbackSkipReason::ResidentStore;
        middle.continues_render_pass = true;
        middle.render_pass_continues = true;
        middle.scissors.push(ScissorResource {
            x: 0,
            y: 0,
            width: half,
            height,
        });
        middle
    };

    // 1. Without the hand-over: exactly what census v18 counts under
    //    `chain_middle`.
    let bucket_before = route_count("render_provider_out_of_class_chain_middle");
    match provider_render::submit_render(
        &inputs_held(&stages, RenderChainRole::Middle),
        &middle_request(),
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_chain_middle",
            "a middle the caller hands nothing to stays on the engine: {reason}"
        ),
        other => panic!("a source-less chain middle is not the class's: {other:?}"),
    }
    assert_eq!(
        route_count("render_provider_out_of_class_chain_middle") - bucket_before,
        1,
        "the refusal is counted under the census's own bucket vocabulary"
    );
    // The sentence a census reader joins that bucket to, verbatim.
    let refusal = match provider_render::submit_render(
        &inputs_held(&stages, RenderChainRole::Middle),
        &middle_request(),
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => reason.detail().to_owned(),
        other => panic!("the source-less middle is refused twice the same way: {other:?}"),
    };
    eprintln!("R25 refusal, verbatim: {refusal}");
    assert!(
        refusal.contains(
            "a record in the middle of a multi-record packet stays on the engine while the \
             caller does not hand it the frame the record before it produced"
        ),
        "the sentence names the half the caller lacks: {refusal}"
    );

    // 2. The bytes the seam hands over: the walk's chain value in the
    //    attachment's own order (`chain_middle_source_frame` folds the
    //    `SeedOrder::Rgba8` seed into the view the pass declares).
    let handed = scanout_order(&seed);
    assert_eq!(handed.len(), seed.len());

    // 3. The middle, with the frame handed over: in class, answered by the
    //    provider, and published — this caller cannot fetch a frame the rail
    //    keeps (R20's arm, unchanged) — while the pass begins from the
    //    caller's bytes. The four names R20/R23 must leave at zero are read
    //    off the fail log's own new tail, the way the census reads them.
    let log_before = std::fs::read_to_string(reims_vgpu_observe::fail_log_path())
        .unwrap_or_default()
        .len();
    let published_before = route_count("render_provider_publish_held_resident");
    let carried_before = route_count("render_provider_chain_middle_source_bytes");
    let source_before = route_count("render_provider_resident_source_bytes");
    let stores_before = route_count("render_provider_resident_store");
    let loads_before = route_count("render_provider_resident_load");
    let provider_middle = match provider_render::submit_render(
        &inputs_held_with_chain_value(&stages, RenderChainRole::Middle, &handed),
        &middle_request(),
    ) {
        RenderRailOutcome::ProviderCompleted(out) => {
            assert!(out.bgra, "a Bgra8Unorm attachment reads back in BGRA order");
            semantic_rgba(out.bytes, out.bgra)
        }
        other => panic!("a middle whose frame the walk carries is in class: {other:?}"),
    };
    assert_texel_count("published middle frame", &provider_middle);
    assert_texel_near(
        "published middle: the last texel inside the rectangle",
        texel_at(&provider_middle, half_of(half) - 1, height / 2),
        FRAGMENT_TEXEL,
    );
    for x in half..width {
        assert_eq!(
            texel_at(&provider_middle, x, height / 2),
            WALK_SEED_TEXEL,
            "texel ({x}, {}) keeps the frame the walk handed over: a channel that uploaded the \
             wrong bytes, cleared, or seeded from the free-running order would land another \
             colour here",
            height / 2,
        );
    }

    // 4. The engine's own answer for the same middle record from the same
    //    previous contents. A middle keeps its frame in the resident the record
    //    names, so the engine's frame is read back out of that resident; the
    //    provider wrote nothing into it (the class's byte arms ride the pooled
    //    pair), so the two frames are two rails' answers to one record.
    let Some(engine_middle) = engine_pixels("engine middle", &stages, middle_request()) else {
        return;
    };
    assert!(
        engine_middle.is_empty(),
        "a resident middle publishes nothing on the engine either"
    );
    let engine_kept = engine_chain_semantic(&identity);
    assert_texel_count("engine middle frame", &engine_kept);
    eprintln!(
        "R25 carried chain value: identity={:?}; handed over {} bytes; provider published {} \
         bytes; engine kept {} bytes; texel at the scissor edge [{},{},{}], outside [{},{},{}]; \
         chain_middle_source_bytes +{}; publish_held_resident +{}; resident_source_bytes +{}; \
         resident_store +{}; resident_load +{}",
        identity,
        handed.len(),
        provider_middle.len(),
        engine_kept.len(),
        texel_at(&provider_middle, half_of(half) - 1, height / 2)[0],
        texel_at(&provider_middle, half_of(half) - 1, height / 2)[1],
        texel_at(&provider_middle, half_of(half) - 1, height / 2)[2],
        texel_at(&provider_middle, width - 1, height / 2)[0],
        texel_at(&provider_middle, width - 1, height / 2)[1],
        texel_at(&provider_middle, width - 1, height / 2)[2],
        route_count("render_provider_chain_middle_source_bytes") - carried_before,
        route_count("render_provider_publish_held_resident") - published_before,
        route_count("render_provider_resident_source_bytes") - source_before,
        route_count("render_provider_resident_store") - stores_before,
        route_count("render_provider_resident_load") - loads_before,
    );
    // The half the record did not draw is byte-exact on both rails: the pass
    // began from the walk's bytes and neither rail converts them. The drawn
    // half is compared within the eight-bit rounding step instead — the
    // fragment's `0.5` is a tie in that conversion and Vulkan leaves its last
    // bit to the implementation, which the two rails' pipelines state
    // differently on this attachment. The whole-frame byte-exact reading is the
    // no-draw control below, where the carried frame *is* the whole frame.
    let (row_bytes, drawn_bytes) = ((width * 4) as usize, (half * 4) as usize);
    for y in 0..height {
        let row = row_bytes * y as usize;
        assert_frames_equal(
            &format!("the walk's frame in row {y}"),
            &provider_middle[row + drawn_bytes..row + row_bytes],
            &engine_kept[row + drawn_bytes..row + row_bytes],
        );
    }
    assert_frames_within_a_step(
        "the middle's drawn texels beside the frame it kept",
        &provider_middle,
        &engine_kept,
    );
    assert_eq!(
        route_count("render_provider_chain_middle_source_bytes") - carried_before,
        1,
        "the middle's carried frame is counted under its own arm"
    );
    assert_eq!(
        route_count("render_provider_publish_held_resident") - published_before,
        1,
        "the record's own frame is published, never kept"
    );
    assert_eq!(
        route_count("render_provider_resident_source_bytes") - source_before,
        0,
        "R23's counter names the frame the engine's registry holds, and this record's frame is \
         the walk's — the two arms are two numbers"
    );
    assert_eq!(
        route_count("render_provider_resident_store") - stores_before,
        0,
        "no frame stays in an image this caller cannot read"
    );
    assert_eq!(
        route_count("render_provider_resident_load") - loads_before,
        0,
        "the pass began from the caller's bytes, not from a resident load"
    );
    // The four names census v18 read as zero on both sinks stay quiet here.
    let log = std::fs::read_to_string(reims_vgpu_observe::fail_log_path()).expect("fail log");
    let fresh = &log[log_before.min(log.len())..];
    for name in [
        "chain_resident_land_fail",
        "load_target_content_not_ready",
        "draws_skipped_after_engine_refusal",
        "vk_engine_target_read",
    ] {
        assert!(
            !fresh.contains(name),
            "the middle's answer must leave `{name}` at zero — the frame travels as the \
             caller's bytes and nothing is kept: {fresh}"
        );
    }

    // 5. A middle whose stream rasterizes nothing: the whole frame *is* the
    //    frame the walk carried, so the two rails' frames are byte-exact over
    //    the whole attachment — the arm's own claim with no rendering step in
    //    between, and the literal reading of "both rails' frames agree byte for
    //    byte".
    let still_identity = scanout_surface_identity(0x7c_00_02);
    let still_request = || {
        let mut still = request_with_streams(MTL_FORMAT_BGRA8_UNORM, &[degenerate_stream()]);
        still.target_identity = Some(still_identity.clone());
        still.color0_declared = Some(reims_vgpu::protocol::pass_action::LoadAction::Load);
        still.target_rgba8 = Some(std::sync::Arc::new(seed.clone()));
        still.skip_readback = true;
        still.readback_skip_reason = ReadbackSkipReason::ResidentStore;
        still.continues_render_pass = true;
        still
    };
    let provider_still = match provider_render::submit_render(
        &inputs_held_with_chain_value(&stages, RenderChainRole::Middle, &handed),
        &still_request(),
    ) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("a middle that draws nothing is in class: {other:?}"),
    };
    let Some(engine_still) = engine_pixels("engine still middle", &stages, still_request()) else {
        return;
    };
    assert!(
        engine_still.is_empty(),
        "a resident middle publishes nothing on the engine either"
    );
    let engine_still_kept = engine_chain_semantic(&still_identity);
    eprintln!(
        "R25 carried chain value, whole frame: provider {} bytes, engine {} bytes, the walk's \
         frame {} bytes",
        provider_still.len(),
        engine_still_kept.len(),
        seed.len(),
    );
    assert_frames_equal(
        "the frame the provider published from the walk's bytes and the frame the engine kept \
         after the same record",
        &provider_still,
        &engine_still_kept,
    );
    assert_frames_equal(
        "the whole frame a record that draws nothing publishes is the frame the walk carried",
        &provider_still,
        &seed,
    );

    // 6. A caller that hands over bytes of the wrong extent is a wiring bug
    //    named as one, not a provider decline: the class keeps the record on
    //    the engine under its own slug.
    let short = &handed[..handed.len() - 4];
    match provider_render::submit_render(
        &inputs_held_with_chain_value(&stages, RenderChainRole::Middle, short),
        &middle_request(),
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            eprintln!("R25 short hand-over, verbatim: {}", reason.detail());
            assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_chain_middle_shape",
                "bytes that are not the attachment's extent keep the record on the engine: \
                 {reason}"
            );
        }
        other => panic!("a short carried chain value is not the class's: {other:?}"),
    }
}

/// R26: the frame of the *surface* the LOAD elision names — the
/// mapper-ref-texture composite's own spelling of the fact R23's chain states.
///
/// Census v19 (`evidence/gate3-census-v19-2026-09-18`) leaves `resident_source`
/// at 919 records (26.9 %), and 47.8 % of them are this shape:
/// `fmt=0x50 load=Load door=mapping skip=resident store=1 seed=none wb=1
/// continues=0 pass_cont=0` — the composite whose LOAD the engine elided
/// (`mapper_ref_texture_load_currency_query`) because its registry already
/// holds the surface's contents under the identity the record itself names.
/// The caller that read the elision out owns that registry, so it can hand the
/// same frame over on `RenderRailInputs::surface_resident_source_bytes`, and
/// the class answers under its own counter
/// (`render_provider_surface_resident_source_bytes`) — the load it states is
/// R23's, the obligation the caller states is not (the landing that consumes
/// this record's frame has to be the one that moves the mapping's content
/// epoch).
///
/// The comparison is the engine's own answer for the same record from the same
/// previous contents: the engine's registry still holds them, because the
/// provider never wrote there — which is also what keeps the elision's
/// currency test honest.
#[test]
fn the_surfaces_frame_the_elision_names_carries_the_composite_into_the_provider() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = extent();

    // 1. The frame the elision will name: an engine record that lands its frame
    //    in the engine's registry under a *surface* identity, with nothing
    //    published — the mapper-ref-texture rail's own Store.
    let identity = surface_identity(0x7b_26_01);
    let Some(seed) = engine_pixels("surface seed", &stages, resident_seed_request(&identity))
    else {
        return;
    };
    assert!(
        seed.is_empty(),
        "a resident store publishes nothing on the engine either"
    );

    // 2. The widened door: the composite's own fields (`load_from_target`, the
    //    half-attachment scissor, the withheld readback with its recorded
    //    reason), and the caller's copy of the registry's frame on R26's input.
    let resident = engine_chain_source(&identity);
    let request = resident_load_request(&identity, false);
    let answers_before = route_count("render_provider_surface_resident_source_bytes");
    let published_before = route_count("render_provider_publish_held_resident");
    let chain_bytes_before = route_count("render_provider_resident_source_bytes");
    let store_before = route_count("render_provider_resident_store");
    let load_before = route_count("render_provider_resident_load");
    let provider = match provider_render::submit_render(
        &inputs_held_with_surface_source(&stages, RenderChainRole::SoleOrTail, &resident),
        &request,
    ) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("the surface's frame, handed over, is the held class's: {other:?}"),
    };
    eprintln!(
        "R26 handed over {} bytes; provider published {} bytes; counters: \
         surface_resident_source_bytes +{}, publish_held_resident +{}, chain_source_bytes +{}, \
         resident_store +{}, resident_load +{}",
        resident.len(),
        provider.len(),
        route_count("render_provider_surface_resident_source_bytes") - answers_before,
        route_count("render_provider_publish_held_resident") - published_before,
        route_count("render_provider_resident_source_bytes") - chain_bytes_before,
        route_count("render_provider_resident_store") - store_before,
        route_count("render_provider_resident_load") - load_before,
    );
    assert_texel_count("the composite's published frame", &provider);
    assert_eq!(
        route_count("render_provider_surface_resident_source_bytes") - answers_before,
        1,
        "the door's own population is counted where the answer happens"
    );
    assert_eq!(
        route_count("render_provider_publish_held_resident") - published_before,
        1,
        "the held population is counted where the answer happens"
    );
    assert_eq!(
        route_count("render_provider_resident_source_bytes") - chain_bytes_before,
        0,
        "R23's arm is its own population: the chain's frame and the surface's are two numbers"
    );
    assert_eq!(
        route_count("render_provider_resident_store") - store_before,
        0,
        "no frame stays in an image this caller cannot read"
    );
    assert_eq!(
        route_count("render_provider_resident_load") - load_before,
        0,
        "this arm loads from the caller's bytes, not from a resident of this rail's own"
    );

    // 3. The engine's own answer for the same record from the same previous
    //    contents, with its readback kept: the two rails' frames agree, and the
    //    half the composite did not draw keeps the surface's own bytes.
    let mut engine_control = resident_load_request(&identity, false);
    engine_control.skip_readback = false;
    engine_control.readback_skip_reason = ReadbackSkipReason::None;
    let Some(engine_frame) = engine_pixels("surface composite", &stages, engine_control) else {
        return;
    };
    assert!(
        !engine_frame.is_empty(),
        "the engine's control keeps its readback, so it has a frame to compare"
    );
    // The two rails translate the fixture's vertex stage separately, so the
    // half the composite *drew* may differ by one step a channel
    // (`assert_frames_within_a_step`, the same tolerance every two-rail
    // comparison in this file states).
    assert_frames_within_a_step(
        "the composite the two rails draw from one surface frame",
        &provider,
        &engine_frame,
    );
    // The half it did *not* draw is the surface frame both rails began from,
    // and that half is byte-for-byte the same: the frame travelled as bytes and
    // neither rail reinterpreted it.
    assert_texel_near(
        "the composite's own half, at the scissor edge",
        texel_at(&provider, half_of(width) - 1, height / 2),
        FRAGMENT_TEXEL,
    );
    for y in 0..height {
        for x in half_of(width)..width {
            assert_eq!(
                texel_at(&provider, x, y),
                texel_at(&engine_frame, x, y),
                "the undrawn texel ({x}, {y}) is the same on both rails: this half is the \
                 surface frame they both began from, handed over as bytes"
            );
        }
    }
    for x in half_of(width)..width {
        assert_eq!(
            texel_at(&provider, x, height / 2),
            RESIDENT_SEED_TEXEL,
            "texel ({x}, {}) keeps the surface frame's own bytes: a route that cleared instead \
             of loading lands another colour here",
            height / 2,
        );
    }

    // 4. The same request with nothing handed over: exactly what census v19
    //    counts under `resident_source`, and what every record of the GVA
    //    elision and every mapper-ref-texture record without a guest writeback
    //    still gets.
    let bucket_before = route_count("render_provider_out_of_class_resident_source");
    match provider_render::submit_render(
        &inputs_held(&stages, RenderChainRole::SoleOrTail),
        &request,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            eprintln!("R26 un-widened door, verbatim: {}", reason.detail());
            assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_resident_source",
                "a record whose frame the caller does not hand over keeps the engine: {reason}"
            );
        }
        other => panic!("a source-less resident load is not the held class's: {other:?}"),
    }
    assert_eq!(
        route_count("render_provider_out_of_class_resident_source") - bucket_before,
        1,
        "the refusal is counted under the census's own bucket vocabulary"
    );

    // 5. A caller that hands over bytes of the wrong extent is a wiring bug
    //    named as one, not a provider decline.
    let short = &resident[..resident.len() - 4];
    match provider_render::submit_render(
        &inputs_held_with_surface_source(&stages, RenderChainRole::SoleOrTail, short),
        &request,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            eprintln!("R26 short hand-over, verbatim: {}", reason.detail());
            assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_surface_resident_shape",
                "bytes that are not the attachment's extent keep the record on the engine: \
                 {reason}"
            );
        }
        other => panic!("a short surface hand-over is not the class's: {other:?}"),
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

/// The census reads *which storage* a draw declared, by name (R-VF1).
///
/// The gate's own sentence names the canonical set and not the value that met
/// it, and the v10 failure log carries no other field that would: 26233 of
/// 93259 seam rows are one sentence, one slug and one `(slug, shape)` row. So
/// the reading is a route per declared format, and this test pins the three
/// halves that make it a reading rather than a counter somebody wired:
///
/// * every format the protocol names has its own route, spelled
///   `vertex_format_<name>` — two storages sharing a key would collapse the
///   distribution the widening order is sized on;
/// * a storage the format gate refuses (`Char4Normalized`, the signed half of
///   the family the normalized widening did *not* carry) is charged for its
///   request, because the counters are the *declaration* population: charged
///   before any condition answers, at the same place the attribute-count band
///   is;
/// * a storage the widening moved into the class (`UChar4Normalized`, since
///   R14 maps it to the contract's `Unorm8x4`) is charged in the same
///   population, so the refused and the widened halves line up;
/// * an admitted storage (`float2`, the reviewed stream) is charged for the same
///   reason, so the admitted and refused halves of the distribution line up.
#[test]
fn the_declared_vertex_formats_are_counted_under_their_own_names() {
    use reims_vgpu::backend::provider_render::vertex_format_route;

    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let count = |route: &str| reims_vgpu::runtime::drain::store_route_count_for_test(route);
    let submit = |inputs: &RenderRailInputs<'_>, req: &DrawRequest| {
        let _ = provider_render::submit_render(inputs, req);
    };

    let mut routes: Vec<&'static str> = Vec::new();
    for format in VertexAttributeFormat::ALL {
        let route = vertex_format_route(format);
        assert_eq!(
            route,
            format!("vertex_format_{}", format.name()),
            "{} is named after the protocol's own spelling",
            format.ordinal()
        );
        assert!(
            !routes.contains(&route),
            "two formats share the route {route}"
        );
        routes.push(route);
    }
    assert_eq!(
        routes.len(),
        VertexAttributeFormat::ALL.len(),
        "every format the protocol names gets exactly one route"
    );
    // The 32-bit storages the class has always admitted, the normalized pair
    // R14 moved in and the signed half beside it name their own routes.
    assert_eq!(
        vertex_format_route(VertexAttributeFormat::parse(29).expect("Float2 is a vertex format")),
        "vertex_format_float2"
    );
    assert_eq!(
        vertex_format_route(VertexAttributeFormat::parse(27).expect("Half4 is a vertex format")),
        "vertex_format_half4"
    );
    assert_eq!(
        vertex_format_route(
            VertexAttributeFormat::parse(9).expect("UChar4Normalized is a vertex format")
        ),
        "vertex_format_uchar4_normalized"
    );
    assert_eq!(
        vertex_format_route(
            VertexAttributeFormat::parse(12).expect("Char4Normalized is a vertex format")
        ),
        "vertex_format_char4_normalized"
    );

    // A request the format gate refuses is still in the reading: the sentence
    // says "a vertex attribute outside the canonical format set", and the route
    // is what says which one.
    let refused_before = count("vertex_format_char4_normalized");
    let mut refused = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    refused.vertex_attributes[0].format =
        VertexAttributeFormat::parse(12).expect("Char4Normalized is a vertex format");
    submit(&inputs(&stages, RenderChainRole::SoleOrTail), &refused);
    assert_eq!(
        count("vertex_format_char4_normalized"),
        refused_before + 1,
        "the refused storage is counted by name"
    );

    // A storage the widening moved in is counted in the same place: the
    // request below passes the format gate and is answered by the provider's
    // own shape rule (the reviewed fixture reads a `float2`, and `unorm8x4`
    // pairs with a `float4` one), which is exactly the point — the count is the
    // *declaration*, not the outcome.
    let widened_before = count("vertex_format_uchar4_normalized");
    let mut widened = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    widened.vertex_attributes[0].format =
        VertexAttributeFormat::parse(9).expect("UChar4Normalized is a vertex format");
    submit(&inputs(&stages, RenderChainRole::SoleOrTail), &widened);
    assert_eq!(
        count("vertex_format_uchar4_normalized"),
        widened_before + 1,
        "the widened storage is counted in the same population as the refused one"
    );

    // And the reviewed storage is counted beside them, so the halves of the
    // distribution are the same measurement.
    let admitted_before = count("vertex_format_float2");
    let admitted = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    submit(&inputs(&stages, RenderChainRole::SoleOrTail), &admitted);
    assert_eq!(
        count("vertex_format_float2"),
        admitted_before + 1,
        "the admitted storage is counted in the same population"
    );
}

/// The two-component normalized storages, drawn for real (R14).
///
/// `unorm8x2` and `unorm16x2` are the pair of the four E-VF1 storages
/// (`research/docs/23` §103) whose component shape is the reviewed `float2`:
/// the same `reims_indexed_tri.air` the class shipped with, declared in a
/// storage two (or four) bytes wide instead of eight. The claims, in the order
/// they are read:
///
/// * the draw is *in* the class and the provider executes it — the format gate
///   maps the storage instead of keeping the engine;
/// * the frame is the *Metal* mapping of the triangle the bytes state, read
///   from neither rail's own viewport;
/// * the canonical provider and the self-contained engine land the same bytes;
/// * the 16-bit twin's frame is byte-identical to the 8-bit one's, because
///   `k * 257 / 65535` is `k / 255` — the equality the contract itself states;
/// * moving the vertex bytes moves the frame, so a rail that dropped the
///   normalized fetch cannot pass by drawing the reviewed triangle.
#[test]
fn the_two_component_normalized_storages_land_the_same_bytes_on_both_rails() {
    use reims_vgpu_core::vertex_format as mtl;

    let _guard = engine_test_session();
    let stages = reviewed_stages();

    // The clip-space triangle the bytes state: `(0, 0)`, `(1, 0)` and `(0, 1)`
    // — the top-right half of the guest's clip square, at the 8x4 attachment
    // whose edges miss every pixel centre (`assert_frame_is_the_metal_mapping`
    // carries the tie-rule reasoning). The second triangle moves the third
    // vertex's corner onto `(1, 1)`.
    let triangle = [(0u8, 0u8), (255, 0), (0, 255)];
    let corner = [(0u8, 0u8), (255, 0), (255, 255)];

    let eight = small_request_with_vertex_storage(
        mtl::MTL_VERTEX_FORMAT_U_CHAR2_NORMALIZED,
        2,
        unorm8x2(&triangle),
    );
    let eight_frame = provider_pixels("unorm8x2", &stages, &eight);
    assert_frame_is_the_metal_mapping(
        "unorm8x2",
        &eight_frame,
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    );
    let Some(engine_eight) = engine_pixels("unorm8x2", &stages, eight) else {
        return;
    };
    assert_frames_equal(
        "unorm8x2 (engine against provider)",
        &engine_eight,
        &eight_frame,
    );

    let sixteen = small_request_with_vertex_storage(
        mtl::MTL_VERTEX_FORMAT_U_SHORT2_NORMALIZED,
        4,
        unorm16x2(&triangle),
    );
    let sixteen_frame = provider_pixels("unorm16x2", &stages, &sixteen);
    assert_frames_equal(
        "unorm16x2 against unorm8x2: k * 257 / 65535 is k / 255",
        &sixteen_frame,
        &eight_frame,
    );
    let Some(engine_sixteen) = engine_pixels("unorm16x2", &stages, sixteen) else {
        return;
    };
    assert_frames_equal(
        "unorm16x2 (engine against provider)",
        &engine_sixteen,
        &sixteen_frame,
    );

    let moved = small_request_with_vertex_storage(
        mtl::MTL_VERTEX_FORMAT_U_CHAR2_NORMALIZED,
        2,
        unorm8x2(&corner),
    );
    let moved_frame = provider_pixels("unorm8x2, third vertex moved", &stages, &moved);
    assert_frames_differ("unorm8x2, third vertex moved", &moved_frame, &eight_frame);
    assert_frame_is_the_metal_mapping(
        "unorm8x2, third vertex moved",
        &moved_frame,
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
    );
    let Some(engine_moved) = engine_pixels("unorm8x2, third vertex moved", &stages, moved) else {
        return;
    };
    assert_frames_equal(
        "unorm8x2, third vertex moved (engine against provider)",
        &engine_moved,
        &moved_frame,
    );
    eprintln!(
        "unorm8x2 / unorm16x2 landed the same bytes on both rails; the moved corner moved the \
         frame"
    );
}

/// The four-component normalized storages, drawn for real (R14).
///
/// `unorm8x4` and `unorm16x4` pair with a `float4` AIR member, which is what
/// `reims_indexed_tri_vec4.air` states: its `x`/`y` become the clip position
/// and its `w` divides it, so the *fourth* component reaches the frame. The
/// last arm is the one that makes the four-component fetch falsifiable rather
/// than asserted: it moves only the fourth byte of one vertex — `255` (the
/// sentinel `w = 1`) to `128` — and the frame has to change, which a rail that
/// fetched two components cannot do.
#[test]
fn the_four_component_normalized_storages_land_the_same_bytes_on_both_rails() {
    use reims_vgpu_core::vertex_format as mtl;

    let _guard = engine_test_session();
    let stages = vec4_stages();

    // `(x, y, z, w)` per vertex; the fixture forces `z = 0` and divides by `w`,
    // so `255` is the sentinel `w = 1` and every triangle below is stated where
    // the two-component pair states its own.
    let triangle = [(0u8, 0u8, 0u8, 255u8), (255, 0, 0, 255), (0, 255, 0, 255)];
    let corner = [(0u8, 0u8, 0u8, 255u8), (255, 0, 0, 255), (255, 255, 0, 255)];
    let divided = [(0u8, 0u8, 0u8, 255u8), (255, 0, 0, 128), (0, 255, 0, 255)];

    let eight = small_request_with_vertex_storage(
        mtl::MTL_VERTEX_FORMAT_U_CHAR4_NORMALIZED,
        4,
        unorm8x4(&triangle),
    );
    let eight_frame = provider_pixels("unorm8x4", &stages, &eight);
    assert_frame_is_the_metal_mapping(
        "unorm8x4",
        &eight_frame,
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    );
    let Some(engine_eight) = engine_pixels("unorm8x4", &stages, eight) else {
        return;
    };
    assert_frames_equal(
        "unorm8x4 (engine against provider)",
        &engine_eight,
        &eight_frame,
    );

    let sixteen = small_request_with_vertex_storage(
        mtl::MTL_VERTEX_FORMAT_U_SHORT4_NORMALIZED,
        8,
        unorm16x4(&triangle),
    );
    let sixteen_frame = provider_pixels("unorm16x4", &stages, &sixteen);
    assert_frames_equal(
        "unorm16x4 against unorm8x4: k * 257 / 65535 is k / 255",
        &sixteen_frame,
        &eight_frame,
    );
    let Some(engine_sixteen) = engine_pixels("unorm16x4", &stages, sixteen) else {
        return;
    };
    assert_frames_equal(
        "unorm16x4 (engine against provider)",
        &engine_sixteen,
        &sixteen_frame,
    );

    let moved = small_request_with_vertex_storage(
        mtl::MTL_VERTEX_FORMAT_U_CHAR4_NORMALIZED,
        4,
        unorm8x4(&corner),
    );
    let moved_frame = provider_pixels("unorm8x4, third vertex moved", &stages, &moved);
    assert_frames_differ("unorm8x4, third vertex moved", &moved_frame, &eight_frame);
    let Some(engine_moved) = engine_pixels("unorm8x4, third vertex moved", &stages, moved) else {
        return;
    };
    assert_frames_equal(
        "unorm8x4, third vertex moved (engine against provider)",
        &engine_moved,
        &moved_frame,
    );

    let wide = small_request_with_vertex_storage(
        mtl::MTL_VERTEX_FORMAT_U_CHAR4_NORMALIZED,
        4,
        unorm8x4(&divided),
    );
    let wide_frame = provider_pixels("unorm8x4, fourth component moved", &stages, &wide);
    assert_frames_differ(
        "unorm8x4, fourth component moved",
        &wide_frame,
        &eight_frame,
    );
    let Some(engine_wide) = engine_pixels("unorm8x4, fourth component moved", &stages, wide) else {
        return;
    };
    assert_frames_equal(
        "unorm8x4, fourth component moved (engine against provider)",
        &engine_wide,
        &wide_frame,
    );
    eprintln!(
        "unorm8x4 / unorm16x4 landed the same bytes on both rails; moving the fourth component \
         alone moved the frame"
    );
}

/// The normalized storages in the census's own reading (R14): the routes move
/// on the shapes that actually cross the seam, the storages the mapping does
/// not carry keep the engine under the format gate's own name, and the shape
/// rule stays the provider's own answer.
#[test]
fn the_normalized_storages_are_counted_and_the_rest_stay_on_the_engine_by_name() {
    use reims_vgpu_core::vertex_format as mtl;

    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let count = |route: &str| reims_vgpu::runtime::drain::store_route_count_for_test(route);

    // The draw the widening admitted: the provider executes it, and the
    // declared storage is what the route names.
    let admitted_before = count("vertex_format_uchar2_normalized");
    let admitted = small_request_with_vertex_storage(
        mtl::MTL_VERTEX_FORMAT_U_CHAR2_NORMALIZED,
        2,
        unorm8x2(&[(0, 0), (255, 0), (0, 255)]),
    );
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &admitted) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("a unorm8x2 draw is in the class: {other:?}"),
    }
    assert_eq!(
        count("vertex_format_uchar2_normalized"),
        admitted_before + 1,
        "the admitted normalized storage is counted by name"
    );

    // A storage the mapping does not carry — the three-channel 16-bit
    // normalized shape, which Vulkan leaves optional as a vertex input format
    // — keeps the engine under the format gate's own slug, and its route still
    // moves: the counters are the declaration population.
    let refused_before = count("vertex_format_ushort3_normalized");
    let mut refused = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    refused.vertex_attributes[0].format =
        VertexAttributeFormat::parse(mtl::MTL_VERTEX_FORMAT_U_SHORT3_NORMALIZED)
            .expect("UShort3Normalized is a vertex format");
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &refused) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            eprintln!(
                "unmapped storage: slug={} detail={}",
                reason.slug(),
                reason.detail()
            );
            assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_vertex_format",
                "an unmapped storage keeps the gate's own name: {reason}"
            );
        }
        other => panic!("an unmapped storage stays on the engine: {other:?}"),
    }
    assert_eq!(
        count("vertex_format_ushort3_normalized"),
        refused_before + 1,
        "the refused storage is counted by name in the same population"
    );

    // The seam's half of the shape rule: `unorm8x4` passes the format gate —
    // that is the R14 mapping — and the provider's registration gate answers
    // the component shape the reviewed fixture still declares (`float2`). A
    // typed decline, not an out-of-class fallback, so the two rules stay
    // readable apart.
    let mut mismatched = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    mismatched.vertex_attributes[0].format =
        VertexAttributeFormat::parse(mtl::MTL_VERTEX_FORMAT_U_CHAR4_NORMALIZED)
            .expect("UChar4Normalized is a vertex format");
    let (slug, detail) = refused_slug("unorm8x4 beside a float2 member", &stages, &mismatched);
    eprintln!("unorm8x4 beside a float2 member: slug={slug} detail={detail:?}");
    assert_eq!(
        slug, "provider_capability",
        "an admitted storage beside the wrong AIR shape is answered at the provider boundary"
    );
    assert!(
        detail.contains("render_stage_reflection_mismatch"),
        "the provider's own name for the shape mismatch rides along: {detail}"
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

/// One frame against the rect a pass *declares*: the fragment colour inside the
/// rect and the clear's own bytes outside it.
///
/// The rect is the expectation rather than either rail's binding — the
/// attachment-covering default is the rect that covers everything — so a frame
/// that matches it answers "which texels did NDC rasterize into" without
/// reading either rail's viewport.
fn assert_frame_is_viewport(
    label: &str,
    frame: &[u8],
    width: u32,
    height: u32,
    rect: [u32; 4],
) -> (u32, u32) {
    assert_eq!(
        frame.len(),
        (width * height * 4) as usize,
        "{label}: the attachment's whole extent has to come back"
    );
    let [rect_x, rect_y, rect_width, rect_height] = rect;
    let mut covered = 0;
    let mut cleared = 0;
    for row in 0..height {
        for column in 0..width {
            let texel = texel_of(frame, width, column, row, 4);
            let texel = [texel[0], texel[1], texel[2], texel[3]];
            let inside = column >= rect_x
                && column < rect_x + rect_width
                && row >= rect_y
                && row < rect_y + rect_height;
            let label = format!("{label}: texel ({column}, {row})");
            if inside {
                covered += 1;
                assert_texel_near(&label, texel, FRAGMENT_TEXEL);
            } else {
                cleared += 1;
                assert_clear_texel(&label, texel);
            }
        }
    }
    (covered, cleared)
}

/// R10: the guest's own viewport rect (`research/docs/23` §100, E-RV1/v100).
///
/// The canonical pass states the rect in its own body, so this face needs no
/// wire section and the draw leaves for the provider. The reading is
/// falsifiable in both directions: the *rect's* texels carry the fragment
/// colour and every other texel keeps the clear (a rail that ignored the rect
/// would fill the attachment), the covering control is the same draw with no
/// viewport of its own (a rail that lost it would land the same frame either
/// way), and the two rails have to be byte-identical — the engine rasterizes
/// through a negative-height viewport while the canonical rail states the rect
/// as-is, and the pass's own vertex module carries the matching y alignment.
#[test]
fn a_declared_viewport_lands_its_own_rect_and_agrees_with_the_engine() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = (8u32, 4u32);
    let rect = [1u32, 1, 3, 2];
    let request = |viewport: Option<[u32; 4]>| {
        let mut req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, &position_streams());
        req.width = width;
        req.height = height;
        if let Some([x, y, rect_width, rect_height]) = viewport {
            req.viewports.push(ViewportResource {
                x: x as f32,
                y: y as f32,
                width: rect_width as f32,
                height: rect_height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            });
        }
        req
    };

    let provider = provider_pixels("declared viewport", &stages, &request(Some(rect)));
    let (covered, cleared) = assert_frame_is_viewport(
        "declared viewport (provider)",
        &provider,
        width,
        height,
        rect,
    );
    assert!(
        covered > 0 && cleared > 0,
        "the declared rect has to cover part of the attachment and leave part of it ({covered} \
         covered, {cleared} cleared), or the frame says nothing about a declared viewport"
    );
    let Some(engine) = engine_pixels("declared viewport", &stages, request(Some(rect))) else {
        return;
    };
    assert_frames_equal("declared viewport", &provider, &engine);

    let covering = provider_pixels("covering viewport", &stages, &request(None));
    let (all_covered, all_cleared) = assert_frame_is_viewport(
        "covering viewport (provider)",
        &covering,
        width,
        height,
        [0, 0, width, height],
    );
    assert_eq!(
        (all_covered, all_cleared),
        (width * height, 0),
        "a request that binds no viewport keeps the attachment-covering default"
    );
    assert_frames_differ(
        "the declared rect is what moved the frame",
        &provider,
        &covering,
    );
    eprintln!(
        "R10 viewport: 8x4 attachment, declared rect {rect:?} covers {covered} texel(s) of the \
         fragment's own colour and leaves {cleared} at the clear; the attachment-covering default \
         covers all {}; provider and engine frames are byte-identical",
        width * height,
    );
}

/// One frame whose every texel is `want`, byte for byte.
///
/// The blend fixtures below cover the whole attachment (the reviewed triangle
/// covers the whole NDC square) and blend the *same* value into every texel, so
/// a uniform frame is the expectation and a per-texel walk would only repeat
/// it.
fn assert_uniform_frame(label: &str, frame: &[u8], width: u32, height: u32, want: [u8; 4]) {
    assert_eq!(
        frame.len(),
        (width * height * 4) as usize,
        "{label}: the attachment's whole extent has to come back"
    );
    for (index, texel) in frame.chunks_exact(4).enumerate() {
        assert_texel_near(
            &format!("{label}: texel {index}"),
            [texel[0], texel[1], texel[2], texel[3]],
            want,
        );
    }
}

/// R10: the attachment's own blend state (`research/docs/23` §100, E-RV1/v100).
///
/// The draw's fragment colour is `64, 128, 191, 255` and the attachment clears
/// to green (`0, 255, 0, 255`), so the three equations a v40-shaped entry can
/// state land three different frames by hand:
///
/// - no blend at all (`blendingEnabled` clear): the fragment replaces the clear;
/// - `One`/`One` with `Add`: `source + destination` per channel, clamped;
/// - `Zero`/`One` with `Add`: the destination survives untouched.
///
/// Every one of them is byte-exact in the attachment's own texel, and the two
/// rails have to land the same frame — the engine bakes the blend into its
/// pipeline from the same six ordinals, the canonical rail states the entry in
/// the pass's own blend list.
#[test]
fn a_declared_blend_state_lands_the_blended_frame_and_agrees_with_the_engine() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = (8u32, 4u32);
    // Green, every component exact in eight bits — the class's clear rule
    // requires that, and it keeps the blend arithmetic's expectations exact.
    let green = [0.0, 1.0, 0.0, 1.0];
    let request = |blend: Option<BlendStateResource>| {
        let mut req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, &position_streams());
        req.width = width;
        req.height = height;
        req.color_attachment = Some(attachment_with_clear(MTL_FORMAT_RGBA8_UNORM, green));
        req.blend = blend;
        req
    };
    use reims_vgpu_core::blend::{
        MTL_BLEND_FACTOR_ONE, MTL_BLEND_FACTOR_SOURCE_ALPHA, MTL_BLEND_FACTOR_ZERO,
        MTL_BLEND_OPERATION_ADD,
    };
    let source_plus_destination = BlendStateResource {
        src_rgb: MTL_BLEND_FACTOR_ONE,
        dst_rgb: MTL_BLEND_FACTOR_ONE,
        op_rgb: MTL_BLEND_OPERATION_ADD,
        src_alpha: MTL_BLEND_FACTOR_ONE,
        dst_alpha: MTL_BLEND_FACTOR_ONE,
        op_alpha: MTL_BLEND_OPERATION_ADD,
    };
    let destination_only = BlendStateResource {
        src_rgb: MTL_BLEND_FACTOR_ZERO,
        dst_rgb: MTL_BLEND_FACTOR_ONE,
        op_rgb: MTL_BLEND_OPERATION_ADD,
        src_alpha: MTL_BLEND_FACTOR_SOURCE_ALPHA,
        dst_alpha: MTL_BLEND_FACTOR_ZERO,
        op_alpha: MTL_BLEND_OPERATION_ADD,
    };
    let cases: [(&str, Option<BlendStateResource>, [u8; 4]); 3] = [
        ("no blend", None, [64, 128, 191, 255]),
        (
            "source plus destination",
            Some(source_plus_destination),
            [64, 255, 191, 255],
        ),
        ("destination only", Some(destination_only), [0, 255, 0, 255]),
    ];
    let mut frames = Vec::new();
    for (label, blend, want) in cases {
        let provider = provider_pixels(label, &stages, &request(blend));
        assert_uniform_frame(label, &provider, width, height, want);
        let Some(engine) = engine_pixels(label, &stages, request(blend)) else {
            return;
        };
        assert_uniform_frame(&format!("{label} (engine)"), &engine, width, height, want);
        assert_frames_equal(label, &provider, &engine);
        frames.push((label, provider));
    }
    // The three equations are three frames: a rail that dropped the blend (or
    // executed the wrong one) would repeat one of the others.
    for (index, (label, frame)) in frames.iter().enumerate() {
        for (other_label, other) in frames.iter().skip(index + 1) {
            assert_frames_differ(&format!("{label} vs {other_label}"), frame, other);
        }
    }
    eprintln!(
        "R10 blend: green clear 0,255,0,255 + fragment 64,128,191,255 -> no blend=64,128,191,255; \
         One/One Add=64,255,191,255; Zero/One Add=0,255,0,255; each provider frame is \
         8x4={} byte(s) and byte-identical to the engine's",
        width * height * 4,
    );
}

/// R10: the blend and write-mask shapes beside the admitted entry, each under
/// its own name (`research/docs/23` §100).
///
/// The write-mask half is the wire's boundary rather than the contract's: the
/// canonical pass can state a mask (E-RV1 executes it), but the command
/// channel's v40 blend section carries four factors and one operation per entry
/// with every channel written, so a masked attachment is a shape this rail's
/// wire refuses by name. The *engine's* own reading of the same shape is beside
/// it — that is the rail the draw keeps, so the reading is what makes "kept on
/// the engine" a claim about a frame rather than about a string.
#[test]
fn the_blend_shapes_beside_the_entry_stay_on_the_engine_by_name() {
    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let (width, height) = (8u32, 4u32);
    let green = [0.0, 1.0, 0.0, 1.0];
    let request = |blend: Option<BlendStateResource>, mask| {
        let mut req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, &position_streams());
        req.width = width;
        req.height = height;
        req.color_attachment = Some(attachment_with_clear(MTL_FORMAT_RGBA8_UNORM, green));
        req.blend = blend;
        req.color_write_mask = mask;
        req
    };
    let answer = |label: &str, req: &DrawRequest| -> (String, String) {
        match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), req) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                (reason.slug().to_owned(), reason.detail().to_owned())
            }
            other => panic!("{label}: the shape is out of class: {other:?}"),
        }
    };
    use reims_vgpu::protocol::blend::ColorWriteMask;
    use reims_vgpu_core::blend::{
        MTL_BLEND_FACTOR_BLEND_COLOR, MTL_BLEND_FACTOR_ONE, MTL_BLEND_FACTOR_SOURCE_1_ALPHA,
        MTL_BLEND_FACTOR_SOURCE_ALPHA, MTL_BLEND_FACTOR_SOURCE_ALPHA_SATURATED,
        MTL_BLEND_FACTOR_ZERO, MTL_BLEND_OPERATION_ADD, MTL_BLEND_OPERATION_MAX,
    };
    let entry = |src_rgb, dst_rgb, op_rgb, src_alpha, dst_alpha, op_alpha| BlendStateResource {
        src_rgb,
        dst_rgb,
        op_rgb,
        src_alpha,
        dst_alpha,
        op_alpha,
    };

    // The write mask: the engine writes red and alpha and leaves green and blue
    // at the clear, which is exactly what "kept on the engine" means here.
    let masked = request(
        None,
        ColorWriteMask::new(
            reims_vgpu::protocol::blend::MTL_COLOR_WRITE_MASK_RED
                | reims_vgpu::protocol::blend::MTL_COLOR_WRITE_MASK_ALPHA,
        )
        .expect("red and alpha are a mask"),
    );
    let (slug, detail) = answer("write mask", &masked);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_blend_mask");
    assert!(
        detail.contains("0x9") && detail.contains("v40"),
        "the sentence names the mask and the wire section: {detail}"
    );
    let Some(engine) = engine_pixels("masked attachment (engine)", &stages, masked) else {
        return;
    };
    assert_uniform_frame(
        "masked attachment (engine)",
        &engine,
        width,
        height,
        [64, 255, 0, 255],
    );
    let unmasked = engine_pixels(
        "unmasked attachment (engine)",
        &stages,
        request(None, ColorWriteMask::ALL),
    );
    if let Some(unmasked) = unmasked {
        assert_frames_differ(
            "the write mask is what moved the engine's frame",
            &engine,
            &unmasked,
        );
    }

    // An alpha operation of its own: the canonical pass states two, the v40
    // section carries one.
    let two_operations = request(
        Some(entry(
            MTL_BLEND_FACTOR_SOURCE_ALPHA,
            MTL_BLEND_FACTOR_ONE,
            MTL_BLEND_OPERATION_ADD,
            MTL_BLEND_FACTOR_ONE,
            MTL_BLEND_FACTOR_ZERO,
            MTL_BLEND_OPERATION_MAX,
        )),
        ColorWriteMask::ALL,
    );
    let (slug, detail) = answer("alpha operation", &two_operations);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_blend_alpha_operation");

    // The three factor families the provider refuses by name.
    let constant = request(
        Some(entry(
            MTL_BLEND_FACTOR_BLEND_COLOR,
            MTL_BLEND_FACTOR_ZERO,
            MTL_BLEND_OPERATION_ADD,
            MTL_BLEND_FACTOR_ONE,
            MTL_BLEND_FACTOR_ZERO,
            MTL_BLEND_OPERATION_ADD,
        )),
        ColorWriteMask::ALL,
    );
    let (slug, detail) = answer("blend constant", &constant);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_blend_constant");
    assert!(
        detail.contains("BlendColor") && detail.contains("blend_constant_unsupported"),
        "the sentence names the factor and the provider's own refusal: {detail}"
    );
    let dual_source = request(
        Some(entry(
            MTL_BLEND_FACTOR_SOURCE_1_ALPHA,
            MTL_BLEND_FACTOR_ZERO,
            MTL_BLEND_OPERATION_ADD,
            MTL_BLEND_FACTOR_ONE,
            MTL_BLEND_FACTOR_ZERO,
            MTL_BLEND_OPERATION_ADD,
        )),
        ColorWriteMask::ALL,
    );
    let (slug, detail) = answer("dual source", &dual_source);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_blend_dual_source");
    let saturated_destination = request(
        Some(entry(
            MTL_BLEND_FACTOR_SOURCE_ALPHA,
            MTL_BLEND_FACTOR_SOURCE_ALPHA_SATURATED,
            MTL_BLEND_OPERATION_ADD,
            MTL_BLEND_FACTOR_ONE,
            MTL_BLEND_FACTOR_ZERO,
            MTL_BLEND_OPERATION_ADD,
        )),
        ColorWriteMask::ALL,
    );
    let (slug, detail) = answer("saturated destination", &saturated_destination);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_blend_factor_slot");
    assert!(
        detail.contains("destination rgb") && detail.contains("SourceAlphaSaturated"),
        "the sentence names the slot and the factor: {detail}"
    );
}

/// R10: the declarations are what the module says.
///
/// The expectation is written by hand — one `[[texture(0)]]`, its view at the
/// translator's texture band base, its AIR static sampler at the device's
/// widened sampler band base, the state the module carries, and the one shape
/// the canonical render sampler executes — so a fixture or reflection that
/// moves fails here rather than silently changing what the seam is asked.
#[test]
fn the_sampled_declarations_are_what_the_module_says() {
    let stages = sampled_stages();
    assert_eq!(
        stages.fragment_texture_declarations,
        vec![RenderTextureDeclaration {
            index: 0,
            binding: 32,
            sampler_binding: 160,
            sampler: RenderSamplerState::Policy(metal_api_core::provider::SamplerPolicy {
                filter: metal_api_core::provider::SamplerFilter::Nearest,
                address: metal_api_core::provider::SamplerAddressMode::ClampToEdge,
            }),
            shape: RenderTextureShape::Sampled2D,
        }],
        "the fragment fixture declares one sampled 2D texture read through one \
         nearest/clamped AIR static sampler"
    );
    assert!(
        stages.texture_interface_refusals.is_empty(),
        "the fixture's interface is inside the translated family: {:?}",
        stages.texture_interface_refusals
    );
    eprintln!(
        "R10 fixture declarations: {:?} interface={:?}",
        stages.fragment_texture_declarations, stages.texture_interface_refusals,
    );
}

/// R10: the sampled-texture reading, and the frame it lands.
///
/// The fragment half samples one fixed coordinate — the centre of the 8x4
/// texture's texel `(6, 3)` — so the whole attachment is that texel's colour on
/// both rails. Three falsifiable halves:
///
/// - the frame *is* the texel the fixture reads, not the texture's first one
///   (every texel is distinct, so any other texel would be another frame);
/// - changing the texel the fragment reads changes the frame, while changing a
///   texel it does not read leaves it alone — the bytes reach the shader
///   through the binding the request and the declaration agree on;
/// - the engine and the canonical provider land the same frame byte for byte.
#[test]
fn a_declared_sampled_texture_lands_the_texel_it_reads_and_agrees_with_the_engine() {
    let _guard = engine_test_session();
    let stages = sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    let read_texel = |texels: &[Vec<u8>]| {
        let (x, y) = SAMPLED_TEXEL;
        let texel = &texels[y * width as usize + x];
        [texel[0], texel[1], texel[2], texel[3]]
    };
    let request = |texels: Vec<Vec<u8>>| sampled_request(&stages, texels, (width, height));

    let wanted = read_texel(&texels);
    let provider = provider_pixels("sampled texture", &stages, &request(texels.clone()));
    assert_uniform_frame(
        "sampled texture (provider)",
        &provider,
        width,
        height,
        wanted,
    );
    let Some(engine) = engine_pixels("sampled texture", &stages, request(texels.clone())) else {
        return;
    };
    assert_uniform_frame("sampled texture (engine)", &engine, width, height, wanted);
    assert_frames_equal("sampled texture", &provider, &engine);

    // The read texel's own bytes: another colour there is another frame.
    let mut other = texels.clone();
    let (read_x, read_y) = SAMPLED_TEXEL;
    other[read_y * width as usize + read_x] = vec![255, 0, 128, 255];
    let moved = provider_pixels("other sampled texel", &stages, &request(other.clone()));
    assert_uniform_frame(
        "other sampled texel (provider)",
        &moved,
        width,
        height,
        read_texel(&other),
    );
    assert_frames_differ("the read texel's bytes moved the frame", &provider, &moved);

    // A texel the fragment never reads: the same bytes land in the texture,
    // and neither rail's frame may move. This is the control that keeps the
    // reading above a statement about *this* binding rather than about "some
    // texture was uploaded".
    let mut unread = texels.clone();
    unread[0] = vec![7, 7, 7, 255];
    let untouched = provider_pixels("unread texel", &stages, &request(unread.clone()));
    assert_frames_equal(
        "a texel the fragment does not read does not reach the frame",
        &provider,
        &untouched,
    );
    if let Some(engine_other) = engine_pixels("unread texel", &stages, request(unread)) {
        assert_frames_equal("unread texel (engine)", &engine, &engine_other);
    }
    eprintln!(
        "R10 sampled texture: attachment {width}x{height} ({} texels), sampled texel {SAMPLED_TEXEL:?} \
         = {wanted:?}; provider frame = every texel is that colour, engine equal; read-texel change \
         moved the frame, unread-texel change did not",
        width * height,
    );
}

/// R19 (`research/docs/23` §107, E-TX1): the sampled texture's *second* 8-bit
/// byte order.
///
/// The census's dominant bind is the guest's own `B8G8R8A8_UNORM` view —
/// `evidence/gate3-census-v14-2026-09-17/` §4.4 reads 1718 of the boot's 1892
/// `texture_bind` lines as that one 896x1024 shape. The provider's window has
/// covered both 8-bit orders since E-TX1; this test is the class gate's own
/// half of that widening, written so the byte order itself is falsifiable
/// rather than assumed:
///
/// - one colour pattern is stated in both byte orders — `[R,G,B,A]` under
///   `rgba8_unorm`, `[B,G,R,A]` under `bgra8_unorm` — and the two binds land
///   the *same* frame, which only an upload that carries each under its own
///   name can do: the two byte strings differ by the red/blue swap, and that
///   swap is visible in the frame;
/// - the frame *is* the colour the shader's channels name for that byte order
///   (red is the third byte of a `bgra8_unorm` texel) and not the bytes read in
///   memory order, so "the name was carried" is measured rather than implied;
/// - moving the read texel's bytes moves the frame, so the reading measures the
///   upload rather than the run;
/// - the engine and the canonical provider land the frame byte for byte, which
///   is what makes this a statement about both rails.
#[test]
fn a_bgra_sampled_texture_lands_the_byte_order_the_bind_states() {
    let _guard = engine_test_session();
    let stages = sampled_stages();
    let (width, height) = (8u32, 4u32);
    // The colour pattern in the two byte orders of it: the same colours, two
    // spellings, with the red and blue halves swapped between them.
    let rgba_texels = sampled_texels(width, height);
    let bgra_texels: Vec<Vec<u8>> = rgba_texels
        .iter()
        .map(|texel| vec![texel[2], texel[1], texel[0], texel[3]])
        .collect();
    let (read_x, read_y) = SAMPLED_TEXEL;
    let read = read_y * width as usize + read_x;
    let wanted: [u8; 4] = rgba_texels[read]
        .clone()
        .try_into()
        .expect("a texel is four bytes");
    let memory_order: [u8; 4] = bgra_texels[read]
        .clone()
        .try_into()
        .expect("a texel is four bytes");
    assert_ne!(
        wanted, memory_order,
        "the fixture's colour has to separate the two byte orders"
    );
    let bgra_request = |texels: Vec<Vec<u8>>| {
        sampled_request_in(
            &stages,
            texels,
            (width, height),
            ash::vk::Format::B8G8R8A8_UNORM,
        )
    };

    // The RGBA8 sibling: the same colours under the first byte order, which is
    // the reading R10 already pins. Its frame is what "the same colours" means
    // for the second one.
    let rgba = provider_pixels(
        "rgba8 sampled texture",
        &stages,
        &sampled_request(&stages, rgba_texels.clone(), (width, height)),
    );
    let bgra = provider_pixels(
        "bgra8 sampled texture",
        &stages,
        &bgra_request(bgra_texels.clone()),
    );
    assert_uniform_frame(
        "bgra8 sampled texture (provider)",
        &bgra,
        width,
        height,
        wanted,
    );
    assert_ne!(
        &bgra[..4],
        memory_order.as_slice(),
        "the BGRA8 frame is the colour the shader's channels name, not the bytes read in \
         memory order"
    );
    assert_frames_equal("the two byte orders of one colour pattern", &rgba, &bgra);

    let Some(engine) = engine_pixels(
        "bgra8 sampled texture",
        &stages,
        bgra_request(bgra_texels.clone()),
    ) else {
        return;
    };
    assert_uniform_frame(
        "bgra8 sampled texture (engine)",
        &engine,
        width,
        height,
        wanted,
    );
    assert_frames_equal("bgra8 sampled texture", &bgra, &engine);

    // The read texel's own bytes: another colour there is another frame, and
    // the frame is that colour read through the bind's byte order — so the
    // value the fragment stage returns follows the bytes *and* the name.
    let mut moved = bgra_texels.clone();
    moved[read] = vec![255, 0, 128, 255];
    let moved_frame = provider_pixels("moved bgra8 texel", &stages, &bgra_request(moved));
    assert_uniform_frame(
        "moved bgra8 texel (provider)",
        &moved_frame,
        width,
        height,
        [128, 0, 255, 255],
    );
    assert_frames_differ(
        "the read texel's bytes moved the frame",
        &bgra,
        &moved_frame,
    );
    eprintln!(
        "R19 bgra8 sampled texture: attachment {width}x{height} ({} texels), sampled texel \
         {SAMPLED_TEXEL:?} = {wanted:?} under `bgra8_unorm` (bytes {memory_order:?}); the rgba8 \
         sibling's bytes land the same frame, the memory-order reading differs, engine equal, \
         read-texel change moved the frame to [128, 0, 255, 255]",
        width * height,
    );
}

/// R19: the formats beside the widened window keep the class's own boundary.
///
/// The window the class now states is the provider's whole `RENDER_SAMPLED`
/// list — the two four-byte 8-bit UNORM byte orders (E-TX1,
/// `research/docs/23` §107) — and every other texel stays on the engine, which
/// is the rail that can run it. The census's other `texture_bind` formats are
/// walked here (`evidence/gate3-census-v14-2026-09-17/` §4.4: `R8_UNORM` 160,
/// `R8G8_UNORM` 11, `R16G16B16A16_SFLOAT` 3), beside the two sRGB spellings of
/// the same 8-bit orders: the provider's sampled window names linear byte
/// orders, so an sRGB *view* is another name the class does not state.
///
/// Each refusal keeps the class's existing bucket, names the bind's own format
/// and the provider's own refusal (`render_texture_format_unsupported`), and
/// never reaches the provider — as it should not, because the engine is the
/// rail that executes these binds.
#[test]
fn the_formats_beside_the_two_byte_orders_stay_on_the_engine_by_name() {
    let _guard = engine_test_session();
    let stages = sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    let delivered = provider_render::provider_submissions();
    for format in [
        ash::vk::Format::R8_UNORM,
        ash::vk::Format::R8G8_UNORM,
        ash::vk::Format::R16G16B16A16_SFLOAT,
        ash::vk::Format::R8G8B8A8_SRGB,
        ash::vk::Format::B8G8R8A8_SRGB,
    ] {
        let request = sampled_request_in(&stages, texels.clone(), (width, height), format);
        match provider_render::submit_render(
            &inputs(&stages, RenderChainRole::SoleOrTail),
            &request,
        ) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                assert_eq!(
                    reason.slug(),
                    "render_provider_out_of_class_texture_bind",
                    "{format:?}: the bind's own bucket"
                );
                let detail = reason.detail();
                assert!(
                    detail.contains(&format!("{format:?}")),
                    "{format:?}: the sentence names the bind's own format: {detail}"
                );
                assert!(
                    detail.contains("render_texture_format_unsupported"),
                    "{format:?}: the sentence names the provider's own refusal: {detail}"
                );
                eprintln!("format door: {format:?} -> {}\n  {detail}", reason.slug());
            }
            other => panic!("{format:?} is outside the widened window: {other:?}"),
        }
    }
    assert_eq!(
        provider_render::provider_submissions(),
        delivered,
        "a bind outside the window stays on the engine without the provider seeing it"
    );
}

/// R10: the sampled-texture shapes beside the admitted entry, each under its
/// own name.
///
/// One door per rule the canonical render sampler's contract states: the
/// module's interface (a resource family the translated rail does not execute),
/// the module's list (a texture that is not its own position, a reflected shape
/// outside the family, a sampler state the rail cannot create), the draw's bind
/// (a declaration with no view, a view the pass cannot state, texels that are
/// not the request's own copy), and the draw's sampler state (a bind that does
/// not repeat the module's own AIR state). Every one of them keeps the draw on
/// the engine, which is the rail that can run it.
#[test]
fn the_sampled_texture_shapes_beside_the_entry_stay_on_the_engine_by_name() {
    let _guard = engine_test_session();
    let stages = sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    let answer = |label: &str, stages: &Stages, req: &DrawRequest| -> (String, String) {
        match provider_render::submit_render(&inputs(stages, RenderChainRole::SoleOrTail), req) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                (reason.slug().to_owned(), reason.detail().to_owned())
            }
            other => panic!("{label}: the shape is out of class: {other:?}"),
        }
    };
    let in_class =
        |label: &str, stages: &Stages, req: &DrawRequest| match provider_render::submit_render(
            &inputs(stages, RenderChainRole::SoleOrTail),
            req,
        ) {
            RenderRailOutcome::ProviderCompleted(_) => (),
            other => panic!("{label}: the shape is in the class: {other:?}"),
        };
    let sampled = || sampled_request(&stages, texels.clone(), (width, height));

    // The positive control: the fixture's own shape is in the class.
    in_class("the sampled fixture", &stages, &sampled());

    // 1. The module's interface: an argument family the translated rail does
    //    not execute, named with its stage, kind and index.
    let mut refused_stages = sampled_stages();
    refused_stages.texture_interface_refusals = vec![RenderInterfaceRefusal {
        stage: RenderPipelineStage::Fragment,
        kind: "runtime sampler",
        index: 0,
    }];
    let (slug, detail) = answer("runtime sampler", &refused_stages, &sampled());
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_interface");
    assert!(
        detail.contains("fragment") && detail.contains("runtime sampler") && detail.contains('0'),
        "the sentence names the stage, the kind and the index: {detail}"
    );

    // 2. The framebuffer-fetch arm: a subpass input, not a sampled texture.
    let mut color_input = sampled();
    color_input.color_input = true;
    let (slug, detail) = answer("framebuffer fetch", &stages, &color_input);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_color_input");

    // 3. The module's declaration list (R10/R16, `research/docs/23` §101, §104):
    //    a `[[texture(n)]]` that is not its own position is *in* the class since
    //    E-RS3 — the two lists pair by the index each entry states rather than
    //    by position, and the sparse fixture's own reading lives in
    //    `a_sparse_texture_declaration_lands_the_dense_frame`. What this walk
    //    still answers under `..._texture_binding` is the list's own rules — an
    //    index at the contract's bound and an index stated twice
    //    (`the_sparse_declaration_rules_stay_refusals_by_name`) — and a
    //    declaration no draw filled. Two sampled textures: the widened cap
    //    (`research/docs/23` §102, E-RS2) admits the declarations, so this shape
    //    reaches the binding-pairing gate — the pass binds one view and the
    //    second declaration has nothing to pair with, which is refused by its
    //    own name rather than by the count.
    let mut two_textures = sampled_stages();
    let first = two_textures.fragment_texture_declarations[0];
    two_textures.fragment_texture_declarations = vec![
        first,
        RenderTextureDeclaration {
            index: 1,
            binding: first.binding + 1,
            sampler_binding: first.sampler_binding + 1,
            ..first
        },
    ];
    let (slug, detail) = answer("two sampled textures", &two_textures, &sampled());
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_unbound");

    // 4. The module's reflected shape and sampler family.
    let mut arrayed = sampled_stages();
    arrayed.fragment_texture_declarations = vec![RenderTextureDeclaration {
        shape: RenderTextureShape::Unsupported(RenderTextureShapeRefusal::Arrayed),
        ..arrayed.fragment_texture_declarations[0]
    }];
    let (slug, detail) = answer("arrayed texture", &arrayed, &sampled());
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_shape");
    assert!(
        detail.contains("arrayed texture"),
        "the sentence names the reflected shape: {detail}"
    );
    let mut runtime_sampler = sampled_stages();
    runtime_sampler.fragment_texture_declarations = vec![RenderTextureDeclaration {
        sampler: RenderSamplerState::Runtime { index: 0 },
        ..runtime_sampler.fragment_texture_declarations[0]
    }];
    let (slug, detail) = answer("runtime sampler state", &runtime_sampler, &sampled());
    eprintln!("door: {slug}\n  {detail}");
    // The R10 fixture's stage binds no runtime sampler argument, so a
    // declaration that names one disagrees with the reflection: R12 answers
    // that with its own name rather than the old "runtime samplers do not
    // exist" sentence.
    assert_eq!(
        slug,
        "render_provider_out_of_class_texture_sampler_mismatch"
    );
    let mut unsupported_state = sampled_stages();
    unsupported_state.fragment_texture_declarations = vec![RenderTextureDeclaration {
        sampler: RenderSamplerState::Unsupported(RenderSamplerRefusal::AirState),
        ..unsupported_state.fragment_texture_declarations[0]
    }];
    let (slug, detail) = answer("unsupported sampler state", &unsupported_state, &sampled());
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_sampler");

    // 5. The draw's bind: a declaration with no view, a view the pass cannot
    //    state, and texels that are not the request's own copy.
    let mut unbound = sampled();
    unbound.sampled_images.clear();
    let (slug, detail) = answer("unbound texture", &stages, &unbound);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_unbound");

    let mut wrong_shape = sampled();
    wrong_shape.sampled_images[0].descriptor_count = 2;
    let (slug, detail) = answer("arrayed bind", &stages, &wrong_shape);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_bind");
    assert!(
        detail.contains("descriptors 2"),
        "the sentence names the bind's own shape: {detail}"
    );
    // The second 8-bit byte order is *in* the class since R19 (E-TX1 widened
    // the provider's window, and `a_bgra_sampled_texture_lands_the_byte_order_the_bind_states`
    // reads it), so the bind this walk states is a texel outside the window —
    // the census's narrow lane — answered under the same bucket with its own
    // format in the sentence (`the_formats_beside_the_two_byte_orders_stay_on_the_engine_by_name`
    // walks the rest).
    let mut wrong_format = sampled();
    wrong_format.sampled_images[0].format = ash::vk::Format::R8_UNORM;
    let (slug, detail) = answer("narrow bind", &stages, &wrong_format);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_bind");
    assert!(
        detail.contains("R8_UNORM"),
        "the sentence names the bind's own format: {detail}"
    );

    let mut wrong_extent = sampled();
    wrong_extent.sampled_images[0].width = width / 2;
    wrong_extent.sampled_images[0].source = SampledSource::Bytes(std::sync::Arc::new(vec![
        0u8;
        (width / 2 * height * 4) as usize
    ]));
    let (slug, detail) = answer("other extent", &stages, &wrong_extent);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_extent");
    assert!(
        detail.contains("4x4") && detail.contains("8x4"),
        "the sentence names both extents: {detail}"
    );

    let mut resident = sampled();
    resident.sampled_images[0].source = SampledSource::Target(engine::TargetIdentity::Surface {
        id: 0x10_10,
        width: 2,
        height: 2,
        generation: 1,
        format: ash::vk::Format::R8G8B8A8_UNORM,
    });
    let (slug, detail) = answer("resident texture", &stages, &resident);
    eprintln!("door: {slug}\n  {detail}");
    // R22 splits the arm this shape used to share with the guest gather: a
    // GPU target is in class once a pass of this rail has declared its
    // production, so the refusal is the *undeclared* half of it, named on its
    // own. A target nothing produced is exactly this shape.
    assert_eq!(
        slug,
        "render_provider_out_of_class_texture_source_undeclared"
    );
    assert!(
        detail.contains("TextureSource::TraceView"),
        "the sentence names the arm the rail would have to state: {detail}"
    );

    // 6. The draw's sampler state: the bind has to repeat the module's own AIR
    //    state, and a state outside the family is answered under the same name.
    let mut other_state = sampled();
    other_state.samplers[0].min_filter =
        reims_vgpu::protocol::sampler::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR;
    other_state.samplers[0].mag_filter =
        reims_vgpu::protocol::sampler::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR;
    let (slug, detail) = answer("linear bind", &stages, &other_state);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_state");
    assert!(
        detail.contains("Nearest") && detail.contains("Linear"),
        "the sentence names both states: {detail}"
    );
    let mut repeat = sampled();
    repeat.samplers[0].address_mode_u =
        reims_vgpu::protocol::sampler::MTL_SAMPLER_ADDRESS_MODE_REPEAT;
    repeat.samplers[0].address_mode_v =
        reims_vgpu::protocol::sampler::MTL_SAMPLER_ADDRESS_MODE_REPEAT;
    repeat.samplers[0].address_mode_w =
        reims_vgpu::protocol::sampler::MTL_SAMPLER_ADDRESS_MODE_REPEAT;
    let (slug, detail) = answer("repeating bind", &stages, &repeat);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_state");
    let mut anisotropic = sampled();
    anisotropic.samplers[0].max_anisotropy = 4;
    let (slug, _) = answer("anisotropic bind", &stages, &anisotropic);
    assert_eq!(slug, "render_provider_out_of_class_texture_state");
}

/// R12: the runtime-sampler declarations, read back off the fixture's own
/// translation.
///
/// The expectation is written by hand — one `[[texture(0)]]` at the
/// translator's texture band base, one runtime `[[sampler(0)]]` at the device's
/// sampler band base, the pairing the module's own sample sites name — so a
/// fixture or reflection that moves fails here rather than silently changing
/// what the seam is asked.
#[test]
fn the_runtime_sampler_declarations_are_what_the_module_says() {
    let stages = runtime_sampled_stages();
    assert_eq!(
        stages.fragment_texture_declarations,
        vec![RenderTextureDeclaration {
            index: 0,
            binding: 32,
            sampler_binding: 160,
            sampler: RenderSamplerState::Runtime { index: 0 },
            shape: RenderTextureShape::Sampled2D,
        }],
        "the fragment fixture declares one sampled 2D texture read through the \
         runtime `[[sampler(0)]]` argument its own sample site names"
    );
    assert_eq!(
        stages.sampler_family,
        RenderSamplerFamily {
            runtime: vec![RenderRuntimeSampler {
                index: 0,
                binding: 160,
            }]
            .into(),
            statics: Vec::new().into(),
        },
        "the stage binds one runtime `[[sampler(0)]]` argument and carries no AIR static sampler"
    );
    assert!(
        stages.texture_interface_refusals.is_empty(),
        "the runtime sampler family is inside the translated rail since v102: {:?}",
        stages.texture_interface_refusals
    );
    eprintln!(
        "R12 fixture declarations: {:?} family={:?} interface={:?}",
        stages.fragment_texture_declarations,
        stages.sampler_family,
        stages.texture_interface_refusals,
    );
}

/// R12: the runtime-sampler reading, and the frames the request's own states
/// land.
///
/// The fragment half samples one fixed coordinate — `(1.375, 0.875)` of the
/// 8x4 surface: texel `(7, 3)`'s centre after clamping, `x = 3.0` after
/// wrapping, row 3 under either filter — so the four states the canonical
/// policy family has land three different readings, and the module carries
/// **none** of them. Three falsifiable halves:
///
/// - the state is the request's: the address mode moves the frame under both
///   filters, the filter moves it under repeat, and clamp-to-edge lands one
///   texel under either filter because the coordinate clamps to its centre;
/// - the texture's own bytes still reach the shader through the binding the
///   declaration names: the read texel moves the frame and an unread one does
///   not;
/// - the engine and the canonical provider land the same frame byte for byte
///   under all four states.
#[test]
fn a_runtime_sampler_lands_the_texels_its_request_states_and_agrees_with_the_engine() {
    let _guard = engine_test_session();
    let stages = runtime_sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    let texel = |texels: &[Vec<u8>], x: usize, y: usize| -> [u8; 4] {
        let texel = &texels[y * width as usize + x];
        [texel[0], texel[1], texel[2], texel[3]]
    };
    use reims_vgpu::protocol::sampler as mtl;
    let nearest = mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST;
    let linear = mtl::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR;
    let clamp = mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    let repeat = mtl::MTL_SAMPLER_ADDRESS_MODE_REPEAT;
    let blend = half_blend(texel(&texels, 2, 3), texel(&texels, 3, 3));
    let arms = [
        (
            "runtime sampler nearest + clamp",
            nearest,
            clamp,
            texel(&texels, 7, 3),
        ),
        (
            "runtime sampler linear + clamp",
            linear,
            clamp,
            texel(&texels, 7, 3),
        ),
        (
            "runtime sampler nearest + repeat",
            nearest,
            repeat,
            texel(&texels, 3, 3),
        ),
        ("runtime sampler linear + repeat", linear, repeat, blend),
    ];
    let mut frames: Vec<(&str, Vec<u8>)> = Vec::new();
    for (what, filter, address, want) in arms {
        let request =
            runtime_sampled_request(&stages, texels.clone(), (width, height), filter, address);
        let provider = provider_pixels(what, &stages, &request);
        assert_uniform_frame(what, &provider, width, height, want);
        let Some(engine) = engine_pixels(what, &stages, request) else {
            return;
        };
        assert_uniform_frame(what, &engine, width, height, want);
        assert_frames_equal(what, &provider, &engine);
        frames.push((what, provider));
    }
    // The state is the only thing that moved between the arms.
    assert_frames_differ(
        "the address mode moved the frame",
        &frames[0].1,
        &frames[2].1,
    );
    assert_frames_differ(
        "the filter moved the frame under repeat",
        &frames[2].1,
        &frames[3].1,
    );
    assert_frames_equal(
        "clamp-to-edge lands one texel under either filter",
        &frames[0].1,
        &frames[1].1,
    );

    // The texture's own bytes reach the shader through the binding the
    // declaration names: the texel the wrapped coordinate reads moves the
    // frame, and a texel no sample reads does not.
    let mut moved = texels.clone();
    moved[3 * width as usize + 3] = vec![255, 0, 128, 255];
    let moved_frame = provider_pixels(
        "other sampled texel (runtime sampler)",
        &stages,
        &runtime_sampled_request(&stages, moved.clone(), (width, height), nearest, repeat),
    );
    assert_uniform_frame(
        "other sampled texel (runtime sampler)",
        &moved_frame,
        width,
        height,
        [255, 0, 128, 255],
    );
    assert_frames_differ(
        "the read texel's bytes moved the frame",
        &frames[2].1,
        &moved_frame,
    );
    let mut unread = texels.clone();
    unread[0] = vec![7, 7, 7, 255];
    let untouched = provider_pixels(
        "unread texel (runtime sampler)",
        &stages,
        &runtime_sampled_request(&stages, unread, (width, height), nearest, repeat),
    );
    assert_frames_equal(
        "a texel no sample reads does not reach the frame",
        &frames[2].1,
        &untouched,
    );
    if let Some(engine_moved) = engine_pixels(
        "other sampled texel (runtime sampler, engine)",
        &stages,
        runtime_sampled_request(&stages, moved, (width, height), nearest, repeat),
    ) {
        assert_frames_equal("other sampled texel (engine)", &moved_frame, &engine_moved);
    }
    eprintln!(
        "R12 runtime sampler: attachment {width}x{height}, one request fact per arm — \
         clamp lands texel (7, 3), repeat (3, 3) and its linear reading the half blend \
         {blend:?}; the module states none of them, provider and engine agree byte for byte \
         on all four arms, moving the read texel moved the frame and an unread one did not",
    );
}

/// R12: the runtime-sampler shapes beside the admitted entry, each under its
/// own name.
///
/// One door per rule the canonical contract states for the runtime family: the
/// stage's sampler forms (both families in one stage, more arguments than
/// Metal's table holds, an argument no texture reads through), the
/// declaration's own index and device slot against the reflection, the draw's
/// bind and its state, the module's sample sites, and the command channel's own
/// ceiling — the frame carries the pass's sampled textures and not its runtime
/// sampler list, so a pass whose declaration makes the trace cross that wire
/// keeps the engine rather than losing the states in the decode.
#[test]
fn the_runtime_sampler_shapes_beside_the_entry_stay_on_the_engine_by_name() {
    let _guard = engine_test_session();
    let stages = runtime_sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    use reims_vgpu::protocol::sampler as mtl;
    let nearest = mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST;
    let clamp = mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    let answer =
        |label: &str, stages: &Stages, req: &DrawRequest, binds: &[StageBufferBind<'_>]| {
            let inputs = inputs_with_binds(stages, RenderChainRole::SoleOrTail, binds);
            match provider_render::submit_render(&inputs, req) {
                RenderRailOutcome::NotInNarrowClass(reason) => {
                    (reason.slug().to_owned(), reason.detail().to_owned())
                }
                other => panic!("{label}: the shape is out of class: {other:?}"),
            }
        };
    let in_class =
        |label: &str, stages: &Stages, req: &DrawRequest| match provider_render::submit_render(
            &inputs(stages, RenderChainRole::SoleOrTail),
            req,
        ) {
            RenderRailOutcome::ProviderCompleted(_) => (),
            other => panic!("{label}: the shape is in the class: {other:?}"),
        };
    let request = |stages: &Stages| {
        runtime_sampled_request(stages, texels.clone(), (width, height), nearest, clamp)
    };

    // The positive control: the fixture's own shape is in the class.
    in_class("the runtime-sampled fixture", &stages, &request(&stages));

    // 1. The stage's sampler family: both forms in one stage is not a shape
    //    one canonical pairing rule can state — its static half pairs by
    //    position, its runtime half by the index a declaration names.
    let mut mixed = runtime_sampled_stages();
    mixed.sampler_family.statics = vec![0].into();
    let (slug, detail) = answer("mixed sampler families", &mixed, &request(&mixed), &[]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_sampler_family");
    assert!(
        detail.contains("static") && detail.contains("runtime"),
        "the sentence names both forms: {detail}"
    );

    // 2. More runtime arguments than Metal's own sampler table holds.
    let mut wide = runtime_sampled_stages();
    wide.sampler_family.runtime = (0..=u32::try_from(MAX_RENDER_SAMPLERS).unwrap())
        .map(|index| RenderRuntimeSampler {
            index,
            binding: 160 + index,
        })
        .collect::<Vec<_>>()
        .into();
    let (slug, detail) = answer(
        "runtime samplers past the table",
        &wide,
        &request(&wide),
        &[],
    );
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_sampler_count");
    assert!(
        detail.contains("16") && detail.contains("17"),
        "the sentence names both counts: {detail}"
    );

    // 3. A runtime argument no texture reads through: the module binds
    //    `[[sampler(1)]]` beside the one the declaration names.
    let mut unpaired = runtime_sampled_stages();
    unpaired.sampler_family.runtime = vec![
        RenderRuntimeSampler {
            index: 0,
            binding: 160,
        },
        RenderRuntimeSampler {
            index: 1,
            binding: 161,
        },
    ]
    .into();
    let mut unpaired_request = request(&unpaired);
    unpaired_request
        .samplers
        .push(family_sampler_resource(161, nearest, clamp));
    let (slug, detail) = answer(
        "unpaired runtime sampler",
        &unpaired,
        &unpaired_request,
        &[],
    );
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(
        slug,
        "render_provider_out_of_class_texture_sampler_unpaired"
    );
    assert!(
        detail.contains("[[sampler(1)]]") && detail.contains("161"),
        "the sentence names the argument and its device slot: {detail}"
    );

    // 4. The declaration against the reflection: an argument the stage's
    //    reflection does not bind, and the right argument at another device
    //    slot.
    let mut wrong_index = runtime_sampled_stages();
    wrong_index.fragment_texture_declarations = vec![RenderTextureDeclaration {
        sampler: RenderSamplerState::Runtime { index: 1 },
        ..wrong_index.fragment_texture_declarations[0]
    }];
    let (slug, detail) = answer(
        "missing runtime index",
        &wrong_index,
        &request(&wrong_index),
        &[],
    );
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(
        slug,
        "render_provider_out_of_class_texture_sampler_mismatch"
    );
    assert!(
        detail.contains("[[sampler(1)]]"),
        "the sentence names the index the reflection does not bind: {detail}"
    );
    let mut wrong_slot = runtime_sampled_stages();
    wrong_slot.fragment_texture_declarations = vec![RenderTextureDeclaration {
        sampler_binding: 161,
        ..wrong_slot.fragment_texture_declarations[0]
    }];
    let (slug, detail) = answer(
        "runtime sampler slot drift",
        &wrong_slot,
        &request(&wrong_slot),
        &[],
    );
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(
        slug,
        "render_provider_out_of_class_texture_sampler_mismatch"
    );
    assert!(
        detail.contains("160") && detail.contains("161"),
        "the sentence names both device slots: {detail}"
    );

    // 5. The draw's own half: no state at the argument's slot, and a state
    //    outside the family the canonical rail creates.
    let mut unbound = request(&stages);
    unbound.samplers.clear();
    let (slug, detail) = answer("unbound runtime sampler", &stages, &unbound, &[]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_unbound");
    assert!(
        detail.contains("[[sampler(0)]]"),
        "the sentence names the argument: {detail}"
    );
    let mut anisotropic = request(&stages);
    anisotropic.samplers[0].max_anisotropy = 4;
    let (slug, detail) = answer("anisotropic runtime state", &stages, &anisotropic, &[]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_state");
    assert!(
        detail.contains("[[sampler(0)]]"),
        "the sentence names the argument: {detail}"
    );

    // 6. The module's own sample sites: a texture whose sites name no runtime
    //    argument carries the walk's own answer, and the class keeps it on the
    //    engine under the sampler name.
    let mut unnamed = runtime_sampled_stages();
    unnamed.fragment_texture_declarations = vec![RenderTextureDeclaration {
        sampler: RenderSamplerState::Unsupported(RenderSamplerRefusal::SampleSite),
        ..unnamed.fragment_texture_declarations[0]
    }];
    let (slug, detail) = answer("unnamed runtime pairing", &unnamed, &request(&unnamed), &[]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_sampler");
    assert!(
        detail.contains("sample sites"),
        "the sentence names the fact: {detail}"
    );

    // 7. The command channel: the very same shape beside one `[[buffer(0)]]`
    //    argument — a stated stage buffer is what makes the trace cross the
    //    owner→provider wire, and the frame carries the pass's sampled textures
    //    (v70) and not its runtime sampler list.
    let wired = sampled_fragment_stages(
        "render_frag_runtime_sampler_buffer.air",
        "reims_runtime_sampled_buffer_frag",
    );
    assert_eq!(
        wired.fragment_stage_buffer_declarations.len(),
        1,
        "the fixture declares one fragment stage buffer"
    );
    let wired_request = request(&wired);
    let content = BufferContent::from(vec![0u8; 4]);
    let binds = [staged_bind(RenderPipelineStage::Fragment, 0, &content)];
    let (slug, detail) = answer(
        "runtime sampler across the wire",
        &wired,
        &wired_request,
        &binds,
    );
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_sampler_wire");
    assert!(
        detail.contains("runtime sampler list"),
        "the sentence names what the frame does not carry: {detail}"
    );
}

/// One frame's colour when every texel agrees, or how many texels it carries —
/// what the R21 readings print, so a reader can see which states landed one
/// texel and which did not.
fn frame_colour(frame: &[u8]) -> String {
    let distinct: std::collections::BTreeSet<[u8; 4]> = frame
        .chunks_exact(4)
        .map(|texel| [texel[0], texel[1], texel[2], texel[3]])
        .collect();
    if distinct.len() == 1 {
        format!("{:?}", distinct.iter().next().expect("one distinct texel"))
    } else {
        format!("{} distinct texels", distinct.len())
    }
}

/// R21: the widened runtime-sampler state family (`research/docs/26` §44) —
/// the states v109 opened on the canonical side, executed from the *request's*
/// own bind and landed by both rails.
///
/// Two fixtures carry the reading, because one sample point cannot state the
/// whole address-mode rule: the R12 fixture samples at `(1.375, 0.875)` of the
/// 8x4 surface — outside the texture in `u` — and its negative-side sibling at
/// `(-0.375, 0.875)`. Between them every added mode lands a frame the modules
/// could not carry before, and their pair states the one thing neither states
/// alone: mirror-clamp-to-edge *is* mirror-repeat inside `[-1, 1]` and clamps
/// (rather than mirrors) outside it, so it lands mirror-repeat's own `(3, 3)`
/// on the negative-side fixture and clamp-to-edge's own `(7, 3)` on the
/// positive-side one.
///
/// Four falsifiable halves, all read off the attachment's bytes:
///
/// - the three added address modes are executions and not fallbacks: each lands
///   the texel its own rule names, and each differs from what the two
///   pre-existing modes land at the same coordinate;
/// - the added min/mag and mip filter names are the *mode* a sample is selected
///   under, not a second state: every canonical view here carries one mip
///   level, so all three mip filters land the frame their own min/mag half
///   lands under the same address mode;
/// - the border mode reads the family's own transparent black and not the
///   descriptor's colour field, which the family does not name;
/// - the engine and the canonical provider agree byte for byte, so the states
///   are what the two rails executed and not what one of them assumed.
#[test]
fn the_widened_runtime_sampler_family_lands_its_states_and_agrees_with_the_engine() {
    let _guard = engine_test_session();
    let stages = runtime_sampled_stages();
    let mirror_stages = mirror_runtime_sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    let texel = |x: usize, y: usize| -> [u8; 4] {
        let texel = &texels[y * width as usize + x];
        [texel[0], texel[1], texel[2], texel[3]]
    };
    use reims_vgpu::protocol::sampler as mtl;
    let filters = [
        ("nearest", mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST),
        ("linear", mtl::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR),
    ];
    let mips = [
        ("not-mipmapped", mtl::MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED),
        ("mip-nearest", mtl::MTL_SAMPLER_MIP_FILTER_NEAREST),
        ("mip-linear", mtl::MTL_SAMPLER_MIP_FILTER_LINEAR),
    ];
    let addresses = [
        ("clamp-to-edge", mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
        (
            "mirror-clamp-to-edge",
            mtl::MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
        ),
        ("repeat", mtl::MTL_SAMPLER_ADDRESS_MODE_REPEAT),
        ("mirror-repeat", mtl::MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT),
        ("clamp-to-zero", mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO),
    ];
    let (nearest, linear) = (0usize, 1usize);
    let (clamp, mirror_clamp, repeat, mirror_repeat, clamp_zero) =
        (0usize, 1usize, 2usize, 3usize, 4usize);

    // Reading 1: the family's whole product on the positive-side fixture — two
    // min/mag filters crossed with three mip filters crossed with five address
    // modes — executed by the canonical provider and by the engine, which have
    // to land the same bytes for every one of the thirty states.
    let mut frames: std::collections::BTreeMap<(usize, usize, usize), Vec<u8>> =
        std::collections::BTreeMap::new();
    for (fi, (filter_name, filter)) in filters.iter().enumerate() {
        for (mi, (mip_name, mip)) in mips.iter().enumerate() {
            for (ai, (address_name, address)) in addresses.iter().enumerate() {
                let what = format!("runtime sampler {filter_name} + {mip_name} + {address_name}");
                let request = widened_runtime_sampled_request(
                    &stages,
                    texels.clone(),
                    (width, height),
                    *filter,
                    *mip,
                    *address,
                );
                let provider = provider_pixels(&what, &stages, &request);
                let Some(engine) = engine_pixels(&what, &stages, request) else {
                    return;
                };
                assert_frames_equal(&what, &provider, &engine);
                frames.insert((fi, mi, ai), provider);
            }
        }
    }
    let positive = |filter: usize, address: usize| -> &Vec<u8> { &frames[&(filter, 0, address)] };
    for ((fi, mi, ai), frame) in &frames {
        eprintln!(
            "R21 reading 1: {} + {} + {} = {}",
            filters[*fi].0,
            mips[*mi].0,
            addresses[*ai].0,
            frame_colour(frame),
        );
    }

    // Reading 2: the mip filter is the mode a sample is selected under. Every
    // canonical view carries one mip level, so the three mip filters land the
    // one level their min/mag half lands — the mip half is carried, and carried
    // *by name*: the same function refuses the ordinals the enumerations do not
    // have (`the_states_beside_the_widened_runtime_sampler_family_stay_on_the_engine_by_name`),
    // so a class that dropped the state would not answer this frame for it.
    for (fi, (filter_name, _)) in filters.iter().enumerate() {
        for (ai, (address_name, _)) in addresses.iter().enumerate() {
            for (mi, (mip_name, _)) in mips.iter().enumerate().skip(1) {
                assert_frames_equal(
                    &format!("{filter_name} + {mip_name} + {address_name}"),
                    &frames[&(fi, mi, ai)],
                    &frames[&(fi, 0, ai)],
                );
            }
        }
    }

    // Reading 3: the address modes, read off the texels each rule names at this
    // coordinate. `u = 1.375` is above the surface, so clamp-to-edge and
    // mirror-clamp-to-edge clamp to texel 7, repeat wraps to texel 3 and
    // mirror-repeat mirrors to texel 5 — three different frames from one
    // coordinate, which is what makes each an execution rather than a fallback.
    assert_uniform_frame(
        "nearest + clamp-to-edge",
        positive(nearest, clamp),
        width,
        height,
        texel(7, 3),
    );
    assert_uniform_frame(
        "nearest + mirror-clamp-to-edge (u above 1 clamps)",
        positive(nearest, mirror_clamp),
        width,
        height,
        texel(7, 3),
    );
    assert_uniform_frame(
        "nearest + repeat",
        positive(nearest, repeat),
        width,
        height,
        texel(3, 3),
    );
    assert_uniform_frame(
        "nearest + mirror-repeat",
        positive(nearest, mirror_repeat),
        width,
        height,
        texel(5, 3),
    );
    assert_uniform_frame(
        "linear + repeat",
        positive(linear, repeat),
        width,
        height,
        half_blend(texel(2, 3), texel(3, 3)),
    );
    assert_uniform_frame(
        "linear + mirror-repeat",
        positive(linear, mirror_repeat),
        width,
        height,
        half_blend(texel(4, 3), texel(5, 3)),
    );
    assert_frames_differ(
        "mirror-repeat moved the frame off repeat",
        positive(nearest, mirror_repeat),
        positive(nearest, repeat),
    );
    assert_frames_differ(
        "clamp-to-zero moved the frame off clamp-to-edge",
        positive(nearest, clamp_zero),
        positive(nearest, clamp),
    );
    assert_frames_differ(
        "clamp-to-zero moved the frame off mirror-repeat",
        positive(nearest, clamp_zero),
        positive(nearest, mirror_repeat),
    );

    // The border mode reads the one border the family states: transparent
    // black, whatever the descriptor's own colour field carries — the
    // clamp-to-border-colour mode whose colour *would* be read is the one the
    // family refuses by name.
    let mut opaque_border = widened_runtime_sampled_request(
        &stages,
        texels.clone(),
        (width, height),
        mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
        mtl::MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
        mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
    );
    opaque_border.samplers[0].border_color = mtl::MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE;
    let declared = provider_pixels(
        "clamp-to-zero + declared opaque white",
        &stages,
        &opaque_border,
    );
    let Some(declared_engine) = engine_pixels(
        "clamp-to-zero + declared opaque white",
        &stages,
        opaque_border,
    ) else {
        return;
    };
    assert_frames_equal(
        "clamp-to-zero + declared opaque white",
        &declared,
        &declared_engine,
    );
    assert_frames_equal(
        "the family reads its own transparent border, not the declared colour",
        &declared,
        positive(nearest, clamp_zero),
    );
    eprintln!(
        "R21 reading 3: positive-side frames — clamp-to-edge and mirror-clamp-to-edge {:?}, \
         repeat {:?}, mirror-repeat {:?}, clamp-to-zero {}; a declared opaque-white border \
         lands the same frame as the family's own transparent black",
        texel(7, 3),
        texel(3, 3),
        texel(5, 3),
        frame_colour(positive(nearest, clamp_zero)),
    );

    // Reading 4: the mirroring modes' whole rule needs both coordinates. On the
    // negative side of the surface mirror-clamp-to-edge mirrors — landing what
    // mirror-repeat lands — while clamp-to-edge clamps and repeat wraps the
    // other way; with the positive-side frames above that states the rule the
    // two fixtures cannot state alone.
    let mirror_arms: [(usize, usize, [u8; 4]); 8] = [
        (nearest, clamp, texel(0, 3)),
        (nearest, mirror_clamp, texel(3, 3)),
        (nearest, mirror_repeat, texel(3, 3)),
        (nearest, repeat, texel(5, 3)),
        (linear, clamp, texel(0, 3)),
        (linear, mirror_clamp, half_blend(texel(2, 3), texel(3, 3))),
        (linear, mirror_repeat, half_blend(texel(2, 3), texel(3, 3))),
        (linear, repeat, half_blend(texel(4, 3), texel(5, 3))),
    ];
    for (filter, address, want) in mirror_arms {
        let what = format!(
            "mirror-side runtime sampler {} + {}",
            filters[filter].0, addresses[address].0,
        );
        let request = widened_runtime_sampled_request(
            &mirror_stages,
            texels.clone(),
            (width, height),
            filters[filter].1,
            mips[0].1,
            addresses[address].1,
        );
        let provider = provider_pixels(&what, &mirror_stages, &request);
        assert_uniform_frame(&what, &provider, width, height, want);
        let Some(engine) = engine_pixels(&what, &mirror_stages, request) else {
            return;
        };
        assert_uniform_frame(&format!("{what} (engine)"), &engine, width, height, want);
        assert_frames_equal(&what, &provider, &engine);
        eprintln!("R21 reading 4: {what} = {want:?}");
    }
}

/// R21: the states beside the widened family, each still under its own name.
///
/// The family is the two min/mag filters crossed with the three mip filters
/// and the five address modes (`research/docs/26` §44); every state outside it
/// keeps the draw on the engine under
/// `render_provider_out_of_class_texture_state`, exactly as before the
/// widening. The doors below walk the refusals E-TX2's own canonical rail
/// answers for the same reasons — the address mode whose border colour the
/// family does not name, and every ordinal the three enumerations do not have —
/// beside the shapes the family never admitted.
#[test]
fn the_states_beside_the_widened_runtime_sampler_family_stay_on_the_engine_by_name() {
    let _guard = engine_test_session();
    let stages = runtime_sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    use reims_vgpu::protocol::sampler as mtl;
    let nearest = mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST;
    let linear = mtl::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR;
    let not_mipmapped = mtl::MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED;
    let mirror_repeat = mtl::MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT;
    let state = |filter: u32, mip: u32, address: u32| {
        widened_runtime_sampled_request(
            &stages,
            texels.clone(),
            (width, height),
            filter,
            mip,
            address,
        )
    };
    let in_class = |label: &str, req: &DrawRequest| match provider_render::submit_render(
        &inputs(&stages, RenderChainRole::SoleOrTail),
        req,
    ) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("{label}: the shape is in the class: {other:?}"),
    };
    let refusal = |label: &str, req: &DrawRequest| -> (String, String) {
        match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), req) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                (reason.slug().to_owned(), reason.detail().to_owned())
            }
            other => panic!("{label}: the shape is out of class: {other:?}"),
        }
    };

    // The positive control: the widest states the class now states, at the end
    // of the address table and the mip table alike.
    in_class(
        "linear + mip-linear + mirror-repeat",
        &state(linear, mtl::MTL_SAMPLER_MIP_FILTER_LINEAR, mirror_repeat),
    );
    in_class(
        "nearest + mip-nearest + clamp-to-zero",
        &state(
            nearest,
            mtl::MTL_SAMPLER_MIP_FILTER_NEAREST,
            mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
        ),
    );

    // 1. The one address mode the family refuses by name on both rails: its
    //    border colour is a state of its own, so a sampler created for it would
    //    answer with a colour the request never stated.
    let (slug, detail) = refusal(
        "clamp-to-border-colour",
        &state(
            nearest,
            not_mipmapped,
            mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR,
        ),
    );
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_state");
    assert!(
        detail.contains("mirror-clamped") && detail.contains("clamped to zero"),
        "the sentence states the widened family the bind is outside: {detail}"
    );

    // 2. The ordinals none of the three enumerations has — on the address half,
    //    the min/mag half and the mip half of the same state.
    for (label, filter, mip, address) in [
        (
            "an address ordinal past the enumeration",
            nearest,
            not_mipmapped,
            6,
        ),
        (
            "a min/mag ordinal past the enumeration",
            2,
            not_mipmapped,
            mirror_repeat,
        ),
        (
            "a mip ordinal past the enumeration",
            nearest,
            3,
            mirror_repeat,
        ),
    ] {
        let (slug, detail) = refusal(label, &state(filter, mip, address));
        eprintln!("door: {label} -> {slug}\n  {detail}");
        assert_eq!(
            slug, "render_provider_out_of_class_texture_state",
            "{label}"
        );
    }

    // 3. The shapes the family never admitted, every one of them still by name:
    //    a min/mag pair that disagrees, axes that address differently, an
    //    unnormalized declaration, a comparison function, and anisotropy.
    let mut disagreeing = state(nearest, not_mipmapped, mirror_repeat);
    disagreeing.samplers[0].mag_filter = linear;
    let (slug, detail) = refusal("a min/mag pair that disagrees", &disagreeing);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_state");

    let mut axes = state(nearest, not_mipmapped, mirror_repeat);
    axes.samplers[0].address_mode_v = mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    let (slug, detail) = refusal("axes that address differently", &axes);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_state");

    let mut unnormalized = state(nearest, not_mipmapped, mirror_repeat);
    unnormalized.samplers[0].unnormalized_coordinates = true;
    let (slug, detail) = refusal("an unnormalized declaration", &unnormalized);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_state");

    let mut comparing = state(nearest, not_mipmapped, mirror_repeat);
    comparing.samplers[0].compare_function = engine::SamplerCompareFunction::Less;
    let (slug, detail) = refusal("a comparison function", &comparing);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_state");

    let mut anisotropic = state(nearest, not_mipmapped, mirror_repeat);
    anisotropic.samplers[0].max_anisotropy = 4;
    let (slug, detail) = refusal("anisotropy", &anisotropic);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_state");
}

/// R15: the fetch-only declarations, read back off the fixture's own
/// translation (`research/docs/23` §3.3, v105).
///
/// The expectation is written by hand — one `[[texture(0)]]` at the
/// translator's texture band base, no sampler slot at all, and the sampler-free
/// arm the module's own image sites state — so a fixture or reflection that
/// moves fails here rather than silently changing what the seam is asked about.
#[test]
fn the_fetch_only_declarations_are_what_the_module_says() {
    for (label, stages) in [
        ("the fetch fixture", fetch_stages()),
        ("the far fetch fixture", fetch_far_stages()),
    ] {
        assert_eq!(
            stages.fragment_texture_declarations,
            vec![RenderTextureDeclaration {
                index: 0,
                binding: 32,
                sampler_binding: 0,
                sampler: RenderSamplerState::Fetched,
                shape: RenderTextureShape::Sampled2D,
            }],
            "{label}: one 2D texture the module's own `OpImageFetch` reads, \
             with no sampler form at all"
        );
        assert_eq!(
            stages.sampler_family,
            RenderSamplerFamily::default(),
            "{label}: the stage carries neither an AIR static sampler nor a runtime \
             `[[sampler(n)]]` argument"
        );
        assert!(
            stages.texture_interface_refusals.is_empty(),
            "{label}: a fetched texture is inside the translated family: {:?}",
            stages.texture_interface_refusals
        );
        eprintln!(
            "R15 {label} declarations: {:?} family={:?}",
            stages.fragment_texture_declarations, stages.sampler_family,
        );
    }
}

/// R15: the fetched reading, and the frames the texels land.
///
/// The fragment half fetches two fixed texels of the 4x4 surface — texel
/// (1, 0).x into red and texel (0, 1).y into green — so the whole attachment is
/// those two channels of the texture the request carries. Four falsifiable
/// halves:
///
/// - the frame *is* those texels, not the surface's first one (every texel is
///   distinct, so any other texel would be another frame);
/// - moving the texel the shader fetches moves the frame, while moving a texel
///   it never reads leaves it alone — the bytes reach the shader through the
///   binding the request and the declaration agree on, with no sampler bound;
/// - the sibling fixture that differs only in the two coordinates it reads
///   lands another frame, over the same declaration;
/// - the engine and the canonical provider land the same frame byte for byte,
///   under a declaration whose sampler half is absent on both rails.
#[test]
fn a_fetched_texture_lands_the_texels_it_reads_and_agrees_with_the_engine() {
    let _guard = engine_test_session();
    let stages = fetch_stages();
    let (width, height) = (4u32, 4u32);
    let texels = sampled_texels(width, height);
    let texel = |texels: &[Vec<u8>], x: usize, y: usize| -> [u8; 4] {
        let texel = &texels[y * width as usize + x];
        [texel[0], texel[1], texel[2], texel[3]]
    };
    let want = [texel(&texels, 1, 0)[0], texel(&texels, 0, 1)[1], 0, 255];
    assert_eq!(
        want,
        [16, 64, 0, 255],
        "the fixture reads texel (1, 0).x and texel (0, 1).y"
    );
    let provider = provider_pixels(
        "fetched texture",
        &stages,
        &fetched_request(&stages, texels.clone(), (width, height)),
    );
    assert_uniform_frame("fetched texture", &provider, width, height, want);
    let Some(engine) = engine_pixels(
        "fetched texture",
        &stages,
        fetched_request(&stages, texels.clone(), (width, height)),
    ) else {
        return;
    };
    assert_uniform_frame("fetched texture (engine)", &engine, width, height, want);
    assert_frames_equal("fetched texture", &provider, &engine);

    // The payload: the fetched texel's bytes reach the frame through the
    // sampler-free binding, and a texel no fetch reads does not.
    let mut moved = texels.clone();
    moved[1] = vec![201, 7, 7, 255];
    let moved_frame = provider_pixels(
        "other fetched texel",
        &stages,
        &fetched_request(&stages, moved.clone(), (width, height)),
    );
    assert_uniform_frame(
        "other fetched texel",
        &moved_frame,
        width,
        height,
        [201, 64, 0, 255],
    );
    assert_frames_differ("the fetched texel moved the frame", &provider, &moved_frame);
    let mut unread = texels.clone();
    unread[2 * width as usize + 2] = vec![9, 9, 9, 255];
    let untouched = provider_pixels(
        "unread texel",
        &stages,
        &fetched_request(&stages, unread, (width, height)),
    );
    assert_frames_equal(
        "a texel no fetch reads does not reach the frame",
        &provider,
        &untouched,
    );
    if let Some(engine_moved) = engine_pixels(
        "other fetched texel (engine)",
        &stages,
        fetched_request(&stages, moved, (width, height)),
    ) {
        assert_frames_equal("other fetched texel (engine)", &moved_frame, &engine_moved);
    }

    // The coordinates: the sibling fixture states the same declaration and
    // reads other texels, so the frame can only have followed them.
    let far_stages = fetch_far_stages();
    let far_want = [texel(&texels, 3, 0)[0], texel(&texels, 0, 3)[1], 0, 255];
    assert_eq!(
        far_want,
        [48, 192, 0, 255],
        "the sibling reads texel (3, 0).x and texel (0, 3).y"
    );
    let far_provider = provider_pixels(
        "far fetched texture",
        &far_stages,
        &fetched_request(&far_stages, texels.clone(), (width, height)),
    );
    assert_uniform_frame(
        "far fetched texture",
        &far_provider,
        width,
        height,
        far_want,
    );
    assert_frames_differ("the coordinates moved the frame", &provider, &far_provider);
    if let Some(far_engine) = engine_pixels(
        "far fetched texture",
        &far_stages,
        fetched_request(&far_stages, texels, (width, height)),
    ) {
        assert_uniform_frame(
            "far fetched texture (engine)",
            &far_engine,
            width,
            height,
            far_want,
        );
        assert_frames_equal("far fetched texture", &far_provider, &far_engine);
    }
    eprintln!(
        "R15 fetched texture: attachment {width}x{height}, no sampler resource bound — \
         texel (1, 0).x and (0, 1).y landed {want:?}, the sibling's (3, 0).x and (0, 3).y \
         landed {far_want:?}, moving the fetched texel moved the frame and an unread one \
         did not; provider and engine agree byte for byte",
    );
}

/// R15: one module, two textures, two access arms — and both of them land.
///
/// The declaration list carries a sampler-free entry and a runtime-sampled one
/// at once, which is the shape the guest's own captures have. Two falsifiable
/// halves beyond the declarations themselves: the two halves of the frame move
/// with the two different textures (so both bindings reached the shader), and
/// the engine and the canonical provider land the same bytes.
#[test]
fn a_fetched_and_a_sampled_texture_are_declared_apart_and_both_land() {
    let _guard = engine_test_session();
    let stages = fetch_and_sample_stages();
    assert_eq!(
        stages.fragment_texture_declarations,
        vec![
            RenderTextureDeclaration {
                index: 0,
                binding: 32,
                sampler_binding: 0,
                sampler: RenderSamplerState::Fetched,
                shape: RenderTextureShape::Sampled2D,
            },
            RenderTextureDeclaration {
                index: 1,
                binding: 33,
                sampler_binding: 160,
                sampler: RenderSamplerState::Runtime { index: 0 },
                shape: RenderTextureShape::Sampled2D,
            },
        ],
        "the fixture's `[[texture(0)]]` is fetched and its `[[texture(1)]]` is sampled \
         through the runtime `[[sampler(0)]]` argument"
    );
    assert_eq!(
        stages.sampler_family,
        RenderSamplerFamily {
            runtime: vec![RenderRuntimeSampler {
                index: 0,
                binding: 160,
            }]
            .into(),
            statics: Vec::new().into(),
        },
        "the stage binds one runtime `[[sampler(0)]]` argument and carries no AIR static sampler"
    );
    let (width, height) = (4u32, 4u32);
    let texels = sampled_texels(width, height);
    let texel = |texels: &[Vec<u8>], x: usize, y: usize| -> [u8; 4] {
        let texel = &texels[y * width as usize + x];
        [texel[0], texel[1], texel[2], texel[3]]
    };
    let want = [texel(&texels, 1, 0)[0], texel(&texels, 3, 2)[1], 0, 255];
    assert_eq!(
        want,
        [16, 128, 0, 255],
        "the fetched half reads texel (1, 0).x and the sampled half texel (3, 2).y"
    );
    let provider = provider_pixels(
        "fetched + sampled",
        &stages,
        &fetch_and_sample_request(&stages, texels.clone(), (width, height)),
    );
    assert_uniform_frame("fetched + sampled", &provider, width, height, want);
    let Some(engine) = engine_pixels(
        "fetched + sampled",
        &stages,
        fetch_and_sample_request(&stages, texels.clone(), (width, height)),
    ) else {
        return;
    };
    assert_uniform_frame("fetched + sampled (engine)", &engine, width, height, want);
    assert_frames_equal("fetched + sampled", &provider, &engine);

    // Each half of the frame follows its own texture: the fetched texel moves
    // red alone, the sampled texel moves green alone.
    let mut moved_fetch = texels.clone();
    moved_fetch[1] = vec![201, 7, 7, 255];
    let fetch_moved = provider_pixels(
        "other fetched texel (mixed)",
        &stages,
        &fetch_and_sample_request(&stages, moved_fetch.clone(), (width, height)),
    );
    assert_uniform_frame(
        "other fetched texel (mixed)",
        &fetch_moved,
        width,
        height,
        [201, 128, 0, 255],
    );
    assert_frames_differ("the fetched texel moved the frame", &provider, &fetch_moved);
    let mut moved_sample = texels.clone();
    moved_sample[2 * width as usize + 3] = vec![1, 202, 1, 255];
    let sample_moved = provider_pixels(
        "other sampled texel (mixed)",
        &stages,
        &fetch_and_sample_request(&stages, moved_sample.clone(), (width, height)),
    );
    assert_uniform_frame(
        "other sampled texel (mixed)",
        &sample_moved,
        width,
        height,
        [16, 202, 0, 255],
    );
    assert_frames_differ(
        "the sampled texel moved the frame",
        &provider,
        &sample_moved,
    );
    assert_frames_differ("the two halves moved apart", &fetch_moved, &sample_moved);
    if let Some(engine_moved) = engine_pixels(
        "other sampled texel (mixed, engine)",
        &stages,
        fetch_and_sample_request(&stages, moved_sample, (width, height)),
    ) {
        assert_frames_equal("other sampled texel (mixed)", &sample_moved, &engine_moved);
    }
    eprintln!(
        "R15 fetched + sampled: declarations [Fetched@32, Runtime{{index: 0}}@33→160], \
         frame {want:?}; the fetched texel moved red to 201 and the sampled texel moved \
         green to 202, independently; provider and engine agree byte for byte",
    );
}

/// R15: a declaration that does not repeat the module's own access.
///
/// The production walk reads the module, so the two halves cannot disagree
/// through it — which is exactly why the door has to be probed with a
/// declaration stated by hand. Two readings of the same pair:
///
/// - through this rail: the class gate admits the *declaration* (it trusts the
///   walk that produced it) and the canonical registration refuses the pair by
///   name, so the seam declines the draw instead of executing a substituted
///   descriptor;
/// - through the provider's own registration entry point: the refusal's raw
///   fields name both access arms and the binding, which is the vocabulary a
///   reader compares a census line against.
#[test]
fn a_declaration_that_does_not_repeat_the_module_is_refused_by_name() {
    let _guard = engine_test_session();
    let stages = fetch_stages();
    let (width, height) = (4u32, 4u32);
    let texels = sampled_texels(width, height);
    let sampled_state = metal_api_core::provider::SamplerPolicy {
        filter: metal_api_core::provider::SamplerFilter::Nearest,
        address: metal_api_core::provider::SamplerAddressMode::ClampToEdge,
    };

    // 1. The rail's own answer: a declaration that says the AIR static arm for
    //    a module whose sites only fetch it.
    let mut forged = stages.clone();
    forged.fragment_texture_declarations = vec![RenderTextureDeclaration {
        index: 0,
        binding: 32,
        sampler_binding: 160,
        sampler: RenderSamplerState::Policy(sampled_state),
        shape: RenderTextureShape::Sampled2D,
    }];
    let mut forged_request = fetched_request(&forged, texels.clone(), (width, height));
    forged_request.samplers.push(sampled_sampler_resource(160));
    match provider_render::submit_render(
        &inputs(&forged, RenderChainRole::SoleOrTail),
        &forged_request,
    ) {
        RenderRailOutcome::ProviderDeclined(decline) => {
            eprintln!("R15 forged sample declaration on the fetch module: {decline:?}");
            let ProviderRenderDecline::ProviderRefused { step, detail, .. } = &decline else {
                panic!("the forgery has to be a provider refusal: {decline:?}");
            };
            assert_eq!(*step, "registration");
            assert!(
                detail.starts_with("render_texture_access_unsupported: "),
                "the provider names the access mismatch: {detail}"
            );
        }
        other => panic!("a declaration the module does not repeat has to be refused: {other:?}"),
    }

    // 2. The provider's own registration, for the raw field map.
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
                stage_buffers: Vec::new(),
                // The declaration states the sampled arm the module's own
                // instructions do not: the fetched image is written through
                // `texture.read()`, so no `OpSampledImage` reaches it.
                textures: vec![metal_api_core::provider::TextureBindingContract::sampled(
                    0,
                    metal_api_core::provider::TextureFormat::Rgba8Unorm,
                    sampled_state,
                )],
            },
            vertex: translate(&stages.air.0, RenderStage::Vertex, stages.vertex_entry),
            fragment: translate(&stages.air.1, RenderStage::Fragment, stages.fragment_entry),
            logical_digest: SemanticDigest::new(
                "reims-provider-render-rail-r15",
                b"fetched-module-sampled-declaration".to_vec(),
            )
            .expect("the digest names a case"),
        })
        .expect_err("the module's own fetches do not repeat a sampled declaration");
    eprintln!("R15 provider answer for the forged declaration: {refused:?}");
    assert_eq!(refused.slug, "render_texture_access_unsupported");
    assert_eq!(
        refused.fields.get("binding"),
        Some(&FieldValue::Unsigned(0)),
        "the refusal names the Metal texture index: {refused:?}"
    );
    assert_eq!(
        refused.fields.get("declared_access"),
        Some(&FieldValue::Text("Sampled".to_owned())),
        "the refusal names the declaration's arm: {refused:?}"
    );
    assert_eq!(
        refused.fields.get("module_access"),
        Some(&FieldValue::Text("fetched".to_owned())),
        "and the module's own arm: {refused:?}"
    );
}

/// R16: the sparse declarations, read back off the fixture's own translation.
///
/// The expectation is written by hand — one `[[texture(3)]]` at the translator's
/// texture band base *plus* its own index, the runtime `[[sampler(0)]]`
/// argument's device slot beside it, and no AIR static sampler anywhere — so a
/// fixture or reflection that moves fails here rather than silently changing
/// what the seam is asked.
#[test]
fn the_sparse_declarations_are_what_the_module_says() {
    let stages = sparse_sampled_stages();
    assert_eq!(
        stages.fragment_texture_declarations,
        vec![RenderTextureDeclaration {
            index: 3,
            binding: 35,
            sampler_binding: 160,
            sampler: RenderSamplerState::Runtime { index: 0 },
            shape: RenderTextureShape::Sampled2D,
        }],
        "the fragment fixture declares one sampled 2D texture at Metal index 3 — \
         binding 32 + 3, with nothing declared below it — read through the runtime \
         `[[sampler(0)]]` argument its own sample site names"
    );
    assert_eq!(
        stages.sampler_family,
        RenderSamplerFamily {
            runtime: vec![RenderRuntimeSampler {
                index: 0,
                binding: 160,
            }]
            .into(),
            statics: Vec::new().into(),
        },
        "the stage binds one runtime `[[sampler(0)]]` argument and carries no AIR static sampler"
    );
    assert!(
        stages.texture_interface_refusals.is_empty(),
        "a texture at Metal index 3 is inside the translated family: {:?}",
        stages.texture_interface_refusals
    );
    eprintln!(
        "R16 sparse declarations: {:?} family={:?}",
        stages.fragment_texture_declarations, stages.sampler_family,
    );
}

/// R16: the sparse declaration executes, and lands the dense sibling's own frame.
///
/// The census v12 shape — `[[texture(3)]]` with nothing below it — used to be
/// the class's own answer (`render_provider_out_of_class_texture_binding`,
/// 35269 = 57.8% of that boot's first-failure lines) because the canonical
/// contract's texture list was positional. Since E-RS3 (v104) the two lists pair
/// by the index each entry states, and the two fixtures here are one body under
/// two indices — the same sample point, the same request state, the same
/// payload — so the reading is a claim with a falsifier in it:
///
/// - the sparse request is *executed by the canonical provider*, not answered by
///   the class gate, and its frame is byte for byte the `[[texture(0)]]`
///   sibling's;
/// - the frame follows the payload the request carries, so the binding really
///   reached the shader rather than a descriptor nobody filled;
/// - the engine and the canonical provider land the same frame for the sparse
///   request, byte for byte.
#[test]
fn a_sparse_texture_declaration_lands_the_dense_frame() {
    let _guard = engine_test_session();
    let sparse = sparse_sampled_stages();
    let dense = runtime_sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    use reims_vgpu::protocol::sampler as mtl;
    let nearest = mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST;
    let clamp = mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    let texel = |texels: &[Vec<u8>], x: usize, y: usize| -> [u8; 4] {
        let texel = &texels[y * width as usize + x];
        [texel[0], texel[1], texel[2], texel[3]]
    };
    // The fixture samples `(1.375, 0.875)` of the 8x4 surface, which clamps to
    // texel (7, 3) under nearest + clamp-to-edge.
    let want = texel(&texels, 7, 3);
    let request = |stages: &Stages, texels: Vec<Vec<u8>>| {
        runtime_sampled_request(stages, texels, (width, height), nearest, clamp)
    };
    let sparse_frame =
        provider_pixels("sparse texture", &sparse, &request(&sparse, texels.clone()));
    assert_uniform_frame("sparse texture", &sparse_frame, width, height, want);
    let dense_frame = provider_pixels("dense sibling", &dense, &request(&dense, texels.clone()));
    assert_uniform_frame("dense sibling", &dense_frame, width, height, want);
    assert_frames_equal(
        "the index is a binding and not a position",
        &sparse_frame,
        &dense_frame,
    );
    let Some(engine) = engine_pixels("sparse texture", &sparse, request(&sparse, texels.clone()))
    else {
        return;
    };
    assert_uniform_frame("sparse texture (engine)", &engine, width, height, want);
    assert_frames_equal("sparse texture", &sparse_frame, &engine);

    // The payload: the texel the fixture reads reaches the frame through the
    // `[[texture(3)]]` binding, and a texel it never reads does not.
    let mut moved = texels.clone();
    moved[3 * width as usize + 7] = vec![17, 200, 0, 255];
    let moved_frame = provider_pixels(
        "other sparse texel",
        &sparse,
        &request(&sparse, moved.clone()),
    );
    assert_uniform_frame(
        "other sparse texel",
        &moved_frame,
        width,
        height,
        [17, 200, 0, 255],
    );
    assert_frames_differ(
        "the read texel moved the sparse frame",
        &sparse_frame,
        &moved_frame,
    );
    let mut unread = texels.clone();
    unread[0] = vec![9, 9, 9, 255];
    let untouched = provider_pixels("unread texel", &sparse, &request(&sparse, unread));
    assert_frames_equal(
        "a texel the sparse fixture never reads does not reach the frame",
        &sparse_frame,
        &untouched,
    );
    if let Some(engine_moved) =
        engine_pixels("other sparse texel", &sparse, request(&sparse, moved))
    {
        assert_frames_equal("other sparse texel", &moved_frame, &engine_moved);
    }
    eprintln!(
        "R16 sparse texture: `[[texture(3)]]` (binding 35) declared beside runtime \
         `[[sampler(0)]]` (160), executed by both rails, frame {:?} byte for byte the \
         `[[texture(0)]]` sibling's; moving the read texel moved it and an unread one did not",
        want,
    );
}

/// R16: the list's own rules stay refusals by name.
///
/// The positional rule is gone, so what is left of the *binding* face is the set
/// of lists no canonical declaration can state — an index at the contract's own
/// bound, one index stated twice, more textures than the count cap — beside the
/// declaration the draw never bound. Each answers under its own name and its
/// sentence names the fact a census line is read against.
#[test]
fn the_sparse_declaration_rules_stay_refusals_by_name() {
    let _guard = engine_test_session();
    let stages = sparse_sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    use reims_vgpu::protocol::sampler as mtl;
    let request = |stages: &Stages, texels: Vec<Vec<u8>>| {
        runtime_sampled_request(
            stages,
            texels,
            (width, height),
            mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
            mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        )
    };
    let answer = |label: &str, stages: &Stages, req: &DrawRequest| -> (String, String) {
        match provider_render::submit_render(&inputs(stages, RenderChainRole::SoleOrTail), req) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                (reason.slug().to_owned(), reason.detail().to_owned())
            }
            other => panic!("{label}: the shape is out of class: {other:?}"),
        }
    };

    // The positive control: the fixture's own sparse shape is in the class and
    // reaches the provider, which is what makes the four refusals below
    // statements about the *rules* rather than about the shape.
    match provider_render::submit_render(
        &inputs(&stages, RenderChainRole::SoleOrTail),
        &request(&stages, texels.clone()),
    ) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("the sparse fixture is in the class: {other:?}"),
    }

    // 1. An index at the contract's own bound (`MAX_RENDER_TEXTURE_INDEX`): the
    //    count cap and the index bound are two different facts.
    let mut past_the_bound = sparse_sampled_stages();
    past_the_bound.fragment_texture_declarations = vec![RenderTextureDeclaration {
        index: 16,
        binding: 48,
        ..past_the_bound.fragment_texture_declarations[0]
    }];
    let (slug, detail) = answer(
        "texture index at the bound",
        &past_the_bound,
        &request(&past_the_bound, texels.clone()),
    );
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_binding");
    assert!(
        detail.contains("[[texture(16)]]")
            && detail.contains("16")
            && detail.contains("render_texture_index_unsupported"),
        "the sentence names the index, the bound and the contract's own refusal: {detail}"
    );

    // 2. The same index twice: the contract holds no index twice, and this walk
    //    answers the repeat rather than dropping one of the two declarations.
    let mut repeated = sparse_sampled_stages();
    let first = repeated.fragment_texture_declarations[0];
    repeated.fragment_texture_declarations = vec![first, first];
    let (slug, detail) = answer(
        "texture index twice",
        &repeated,
        &request(&repeated, texels.clone()),
    );
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_binding");
    assert!(
        detail.contains("[[texture(3)]]")
            && detail.contains("twice")
            && detail.contains("trace_contract_invalid"),
        "the sentence names the repeated index and the contract's own refusal: {detail}"
    );

    // 3. The count cap, one face over from the index bound: more textures than
    //    the canonical contract states, in canonical order so the cap is what
    //    the walk answers.
    let mut wide = sparse_sampled_stages();
    wide.fragment_texture_declarations = (0..=u32::try_from(MAX_RENDER_TEXTURES).unwrap())
        .map(|index| RenderTextureDeclaration {
            index,
            binding: 32 + index,
            ..first
        })
        .collect::<Vec<_>>();
    let (slug, detail) = answer(
        "textures past the cap",
        &wide,
        &request(&wide, texels.clone()),
    );
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_count");
    assert!(
        detail.contains("8") && detail.contains("9"),
        "the sentence names the cap and the count: {detail}"
    );

    // 4. The declaration the draw never bound: the request's own copy sits at
    //    the *dense* slot, so the `[[texture(3)]]` declaration has no view.
    let mut unbound = request(&stages, texels.clone());
    unbound.sampled_images[0].binding = 32;
    let (slug, detail) = answer("unbound sparse texture", &stages, &unbound);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_unbound");
    assert!(
        detail.contains("[[texture(3)]]"),
        "the sentence names the declaration the draw did not fill: {detail}"
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
        fragment_texture_declarations: Vec::new(),
        sampler_family: RenderSamplerFamily::default(),
        texture_interface_refusals: Vec::new(),
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
                textures: Vec::new(),
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

/// The door's buckets after R9n, each its own counter: the two facts that still
/// keep a declared draw on the engine, and the two arms the increments lifted —
/// the read-only class R9d admits once its bind is stated, and the slot whose
/// entry never dereferences it, which R9n stops answering for altogether.
///
/// The fixture-driven arms come from real modules whose `[[buffer(0)]]`
/// metadata states that access, so the split is measured against translations
/// rather than against the seam's own vocabulary: an unread declaration must
/// not be counted as a writable one, and the writable arm — the one the
/// canonical contract refuses before admission — must not hide inside the read
/// population beside it.
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
    // One fixture-driven declaration, asserted to carry the access this test
    // means: the arms come from the fixtures' own translations, and the reach
    // beside them (R9d) is read rather than restated.
    let with_fragment = |fixture_name: &str, entry: &'static str, access: StageBufferAccess| {
        let mut stages = bases.clone();
        stages.air.1 = fixture(fixture_name);
        stages.fragment_entry = entry;
        stages.fragment_stage_buffer_declarations =
            declared_stage_buffers(&stages.air.1, RenderStage::Fragment, entry);
        let declarations = &stages.fragment_stage_buffer_declarations;
        assert_eq!(declarations.len(), 1, "{fixture_name}: one declaration");
        assert_eq!(
            (declarations[0].index, declarations[0].access),
            (0, access),
            "{fixture_name} declares one buffer of its own access at index 0"
        );
        stages
    };
    let content = BufferContent::Bytes(std::sync::Arc::new(vec![0u8; 16]));
    let binds = [staged_bind(RenderPipelineStage::Fragment, 0, &content)];

    // The writable arm: the fixture stores the word it loaded, so its
    // reflection states both uses — `ReadWrite`, the arm R9f landed — while the
    // request states no destination for it: the pair the gate has to answer as
    // a write it cannot land. The bind is what reaches that fact at all; with
    // the slot empty the `_unbound` arm would answer first.
    let writable = with_fragment(
        "render_frag_buffer_write.air",
        "reims_write_buffer_frag",
        StageBufferAccess::ReadWrite,
    );
    let write_bucket = route_count("render_provider_out_of_class_stage_buffer_write");
    match provider_render::submit_render(
        &inputs_with_binds(&writable, RenderChainRole::SoleOrTail, &binds),
        &req,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_stage_buffer_write",
                "a writable declaration with no landing is its own bucket: {reason}"
            );
            assert!(
                reason.detail().contains("[[buffer(0)]]"),
                "the sentence names the declaration: {reason}"
            );
        }
        other => panic!("a writable declaration with no landing stays on the engine: {other:?}"),
    }
    assert_eq!(
        route_count("render_provider_out_of_class_stage_buffer_write"),
        write_bucket + 1,
        "the writable fact charges its own bucket exactly once"
    );

    // The unclassified arm, handed to the gate directly: no fixture here can
    // state it (the module's own decoration answers whatever the metadata
    // omitted), and it is still its own bucket.
    let mut unknown = bases.clone();
    unknown.fragment_stage_buffer_declarations = vec![StageBufferDeclaration {
        index: 0,
        access: StageBufferAccess::Unknown,
        footprint: StageBufferFootprint::Static { max_bytes: 4 },
    }];
    let unknown_bucket = route_count("render_provider_out_of_class_stage_buffer_unknown");
    match provider_render::submit_render(&inputs(&unknown, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_stage_buffer_unknown",
                "an access the translation does not classify is its own bucket: {reason}"
            );
            assert!(
                reason.detail().contains("[[buffer(0)]]"),
                "the sentence names the declaration: {reason}"
            );
        }
        other => panic!("an unclassified declaration stays on the engine: {other:?}"),
    }
    assert_eq!(
        route_count("render_provider_out_of_class_stage_buffer_unknown"),
        unknown_bucket + 1,
        "the unclassified fact charges its own bucket exactly once"
    );

    // The refusals this file still reads on the stage-buffer door, named once:
    // the two arms above and the fact that stands between a read-only
    // declaration and its stated pair.
    let refusals = [
        "render_provider_out_of_class_stage_buffer_write",
        "render_provider_out_of_class_stage_buffer_unknown",
        "render_provider_out_of_class_stage_buffer_unbound",
    ];
    let unread = with_fragment(
        "render_frag_buffer_unused.air",
        "reims_unused_buffer_frag",
        StageBufferAccess::Unused,
    );
    let before: Vec<u64> = refusals.iter().map(|slug| route_count(slug)).collect();
    match provider_render::submit_render(
        &inputs_with_binds(&unread, RenderChainRole::SoleOrTail, &binds),
        &req,
    ) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("a slot no entry dereferences is admitted since R9n: {other:?}"),
    }
    let after: Vec<u64> = refusals.iter().map(|slug| route_count(slug)).collect();
    assert_eq!(
        before, after,
        "an unread declaration charges none of the door's buckets: its own is gone (R9n)"
    );

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

    // The vertex half answers first: one request, both stages declaring a fact
    // the class answers for, and the vertex bucket is the one that moves.
    let mut stages = bases.clone();
    stages.vertex_stage_buffer_declarations = vec![StageBufferDeclaration {
        index: 2,
        access: StageBufferAccess::Unknown,
        footprint: StageBufferFootprint::Static { max_bytes: 16 },
    }];
    stages.fragment_stage_buffer_declarations = vec![StageBufferDeclaration {
        index: 3,
        access: StageBufferAccess::Write,
        footprint: StageBufferFootprint::Static { max_bytes: 16 },
    }];
    let vertex_bucket = route_count("render_provider_out_of_class_stage_buffer_unknown");
    let fragment_bucket = route_count("render_provider_out_of_class_stage_buffer_write");
    match provider_render::submit_render(&inputs(&stages, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::NotInNarrowClass(reason) => assert_eq!(
            reason.slug(),
            "render_provider_out_of_class_stage_buffer_unknown",
            "the vertex half answers first: {reason}"
        ),
        other => panic!("a declared vertex buffer keeps the draw on the engine: {other:?}"),
    }
    assert_eq!(
        route_count("render_provider_out_of_class_stage_buffer_unknown"),
        vertex_bucket + 1
    );
    assert_eq!(
        route_count("render_provider_out_of_class_stage_buffer_write"),
        fragment_bucket,
        "the stage that did not answer does not charge a bucket"
    );

    // And an admitted draw charges none of the refusals' buckets: the buckets
    // are the door's refusals, not the population it lets through.
    let clean = reviewed_stages();
    let before: Vec<u64> = refusals.iter().map(|slug| route_count(slug)).collect();
    match provider_render::submit_render(&inputs(&clean, RenderChainRole::SoleOrTail), &req) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("the undeclared binds are in class: {other:?}"),
    }
    let after: Vec<u64> = refusals.iter().map(|slug| route_count(slug)).collect();
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

    // The canonical list's own shape: at most `MAX_RENDER_STAGE_BUFFERS`
    // declarations, one per slot. The ceiling is the live contract constant —
    // E-SB1 raised it from four to eight (`research/docs/23` §108) — so the
    // shape that crosses it is built from the constant rather than written as
    // five, and the sentence it is compared with is derived the same way.
    let over_cap = metal_api_core::provider::MAX_RENDER_STAGE_BUFFERS + 1;
    let too_many = (0..over_cap as u32)
        .map(|index| StageBufferDeclaration {
            index,
            access: StageBufferAccess::Read,
            footprint: StageBufferFootprint::Static { max_bytes: 4 },
        })
        .collect::<Vec<_>>();
    let stages = with_fragment(too_many);
    let (slug, detail) = answer("over-cap declarations", &stages, &[bind]);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_stage_buffer_shape");
    assert!(
        detail.contains(&format!("declare {over_cap} stage buffers")),
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

/// The two doors' *overlap* is its own boundary, and it is the class's to
/// answer rather than the provider's.
///
/// Each door above admits one face at a time: a sampled `[[texture(i)]]`
/// declaration beside its bind, and a `[[buffer(n)]]` declaration beside its
/// bind. A pass that states both is a shape the canonical rail's *pipeline
/// layout* cannot hold — the rail's set 0 is the sampled pipeline's combined
/// image samplers (`create_render_textures`), a translated module reads its
/// `[[buffer(n)]]` arguments from the set its own reflection names (set 0 by
/// the default resource layout), and `create_stage_buffers` lays the buffers'
/// sets out positionally behind that same slot — so the provider answers it
/// when the layout is built, by name (`render_texture_layout_unsupported`).
///
/// That answer is a *decline*, and a decline on an in-class draw is
/// fail-closed: `runtime/draw/vulkan.rs` does not re-run the engine, so the
/// record's pixels are lost rather than drawn by the other rail. Census v20
/// measured the cost of leaving it outside the class — 388
/// `draws_skipped_after_engine_refusal` records, every one of them pipe 157's
/// vertex buffer beside its fragment texture (the window's own icon layers,
/// missing from the frame). The gate states the boundary instead: this test
/// pins the sentence, the bucket, and the control that the *each* face alone
/// still crosses to the provider, so the boundary is the overlap and not
/// either half.
#[test]
fn a_sampled_texture_beside_a_stage_buffer_stays_on_the_engine_by_name() {
    let _guard = engine_test_session();
    // The measured shape: the vertex stage reads `[[buffer(2)]]` while the
    // fragment stage samples `[[texture(0)]]`.
    let mut stages = sampled_stages();
    stages.vertex_stage_buffer_declarations = vec![StageBufferDeclaration {
        index: 2,
        access: StageBufferAccess::Read,
        footprint: StageBufferFootprint::Static { max_bytes: 8 },
    }];
    let content = BufferContent::Bytes(std::sync::Arc::new(vec![0u8; 16]));
    let bind = staged_bind(RenderPipelineStage::Vertex, 2, &content);
    let (width, height) = (8u32, 4u32);
    let req = sampled_request(&stages, sampled_texels(width, height), (width, height));

    let bucket = route_count("render_provider_out_of_class_texture_layout");
    match provider_render::submit_render(
        &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &[bind]),
        &req,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            eprintln!("overlap: {reason}");
            assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_texture_layout",
                "the overlap is the layout's own bucket: {reason}"
            );
            let detail = reason.detail();
            assert!(
                detail.contains("[[buffer(2)]]") && detail.contains("1 sampled texture(s)"),
                "the sentence names both faces and the bind's own slot: {detail}"
            );
            assert!(
                detail.contains("render_texture_layout_unsupported"),
                "the sentence names the provider answer it stands in front of: {detail}"
            );
        }
        other => {
            panic!("a sampled texture beside a stage buffer stays on the engine: {other:?}")
        }
    }
    assert_eq!(
        route_count("render_provider_out_of_class_texture_layout"),
        bucket + 1,
        "the overlap charges its own bucket exactly once"
    );

    // The control: the same draw with the fragment texture alone — the
    // declaration removed — still crosses to the provider, so the boundary is
    // the *pair* of faces and not the texture half. (The stage-buffer half's
    // own control is `the_statement_and_the_wire_carry_the_one_slot_the_entry_dereferences`,
    // which admits the reviewed buffer fixture with no texture beside it.)
    let texture_only = sampled_stages();
    match provider_render::submit_render(&inputs(&texture_only, RenderChainRole::SoleOrTail), &req)
    {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("the sampled texture alone is in class: {other:?}"),
    }
}

/// R9o: the shape door's three rules, each charged under its own route.
///
/// The census v5 segment's first failure is `stage_buffer_shape` — 88026 of
/// 89277 seam rows (98.6%) — and that one slug is three rules: a statement
/// longer than the canonical contract states, one `(stage, index)` declared
/// twice, and a vertex declaration inside the canonical layout's own
/// `0..vertex_streams` bindings. A census can size the population of the slug
/// and nothing about the rules inside it, so the next round's first question
/// ("which rule answers 88026 draws") has no reading until each arm is counted
/// beside the refusal it answers. This is that reading.
///
/// The refusals themselves do not move: every arm keeps the same slug
/// (`render_provider_out_of_class_stage_buffer_shape`), the same sentence, the
/// same point in the gate, and the same `NotInNarrowClass` answer — only the
/// census key beside it is new. The three routes are charged where the gate
/// answers (not where a render request arrives), so the in-class shape pays
/// none of them and each shape below moves exactly one route.
#[test]
fn each_stage_buffer_shape_rule_is_counted_under_its_own_route() {
    let _guard = engine_test_session();
    let bases = reviewed_stages();
    let declaration = |index: u32| StageBufferDeclaration {
        index,
        access: StageBufferAccess::Read,
        footprint: StageBufferFootprint::Static { max_bytes: 4 },
    };
    let with_declarations = |stages: &Stages,
                             vertex: Vec<StageBufferDeclaration>,
                             fragment: Vec<StageBufferDeclaration>| {
        let mut shaped = stages.clone();
        shaped.vertex_stage_buffer_declarations = vertex;
        shaped.fragment_stage_buffer_declarations = fragment;
        shaped
    };

    let too_many = [
        "stage_buffer_shape_gt4",
        "stage_buffer_shape_duplicate",
        "stage_buffer_shape_vertex_layout",
    ];
    let before = too_many.map(route_count);
    // The three shapes are fuelled by the reviewed fixture's own vertex stream:
    // the class states one stream as binding 0, which is what makes arm 3 a
    // collision and what keeps arm 2's earlier slot non-colliding.
    assert_eq!(
        bases.vertex_attribute_locations.len(),
        1,
        "the reviewed shape states one vertex stream, the layout arm 3 collides with"
    );

    // An in-class control: the reviewed shape declares nothing, so it reaches
    // the provider without the shape door answering — and charges no route.
    let in_class = with_declarations(&bases, Vec::new(), Vec::new());
    match provider_render::submit_render(
        &inputs_with_binds(&in_class, RenderChainRole::SoleOrTail, &[]),
        &narrow_request(MTL_FORMAT_RGBA8_UNORM),
    ) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("the reviewed shape is in class: {other:?}"),
    }
    assert_eq!(
        too_many.map(route_count),
        before,
        "an in-class shape charges no shape route"
    );

    // One shape per rule. Each is the smallest shape the rule answers, and each
    // answers with the same slug the shape door has always used.
    // Each shape answers with the same slug the shape door has always used;
    // what is new is the route beside it, and the bind each one states so the
    // gate reaches the rule rather than an earlier one (the duplicate rule sits
    // behind the unbound check, so its slot is bound here).
    let content = BufferContent::Bytes(std::sync::Arc::new(vec![0u8; 16]));
    let duplicate_binds = [staged_bind(RenderPipelineStage::Fragment, 1, &content)];
    // The over-cap shape and the sentence naming its count both come from the
    // live contract ceiling rather than from the four the shape was written
    // against: E-SB1 raised `MAX_RENDER_STAGE_BUFFERS` to eight
    // (`research/docs/23` §108), and the rule this arm reads is "longer than the
    // contract states", not a number of its own.
    let over_cap = metal_api_core::provider::MAX_RENDER_STAGE_BUFFERS + 1;
    let shapes = [
        (
            "too many",
            "render_provider_out_of_class_stage_buffer_shape",
            format!("declare {over_cap} stage buffers"),
            with_declarations(
                &bases,
                Vec::new(),
                (0..over_cap as u32).map(declaration).collect(),
            ),
            &[] as &[StageBufferBind<'_>],
        ),
        (
            "one slot twice",
            "render_provider_out_of_class_stage_buffer_shape",
            "[[buffer(1)]] twice".to_owned(),
            with_declarations(&bases, Vec::new(), vec![declaration(1), declaration(1)]),
            &duplicate_binds,
        ),
        (
            "vertex declaration inside the layout",
            "render_provider_out_of_class_stage_buffer_shape",
            "1 vertex stream(s)".to_owned(),
            with_declarations(&bases, vec![declaration(0)], Vec::new()),
            &[] as &[StageBufferBind<'_>],
        ),
    ];
    let mut refusals: Vec<(&str, String)> = Vec::new();
    for (label, expected_slug, expected_sentence, shaped, binds) in &shapes {
        let answer = match provider_render::submit_render(
            &inputs_with_binds(shaped, RenderChainRole::SoleOrTail, binds),
            &narrow_request(MTL_FORMAT_RGBA8_UNORM),
        ) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                (reason.slug().to_owned(), reason.detail().to_owned())
            }
            other => panic!("{label}: the shape is out of class: {other:?}"),
        };
        eprintln!(
            "shape route reading: {label}\n  slug={}\n  detail={}",
            answer.0, answer.1
        );
        assert_eq!(
            answer.0, *expected_slug,
            "{label}: the refusal slug is unchanged"
        );
        assert!(
            answer.1.contains(expected_sentence.as_str()),
            "{label}: the sentence is unchanged: {}",
            answer.1
        );
        refusals.push((label, answer.0));
    }
    // The one-counter-wearing-three-names check: each route moved exactly once,
    // and each shape moved the route of its own rule rather than any other.
    assert_eq!(
        too_many.map(route_count),
        [before[0] + 1, before[1] + 1, before[2] + 1],
        "each rule's route is charged once, by one shape"
    );
    for (route, (label, _)) in too_many.into_iter().zip(&refusals) {
        eprintln!(
            "shape route charged: {route}={} ({label})",
            route_count(route)
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

/// R18: a window-backed stage buffer whose view pointer misses the device's
/// import granules is *copied* into the owner's staged arm instead of keeping
/// the draw on the engine.
///
/// Census v13 read this shape as `render_provider_out_of_class_stage_buffer_alignment`
/// — 2772 of one boot's 10696 seam rows, 25.92 %,
/// `evidence/gate3-census-v13-2026-09-17` §0.4 — with the sentence "the view's
/// own host pointer is not a whole number of the device's import granules".
/// The window *does* cover the bind's bytes; the bind simply starts inside the
/// granule (a packed resource's own `source_offset`), and the canonical rail
/// refuses a no-copy import at that pointer by name
/// (`lease_alignment_unsupported`). The answer this increment states is the
/// arm the class already has for a bind with no window behind it: the bytes the
/// borrowed arm would have bound are read out of the registration
/// ([`reims_vgpu::backend::provider_owner::window_bytes`]) and imported as an
/// owner-issued staged lease.
///
/// Every fact is checkable at once: the draw reaches the provider, the wire
/// carries the view as `StagedLease`, the frame is byte-identical to the
/// engine's for the same request and the same bytes, the lease row names the
/// staged channel, and moving the owner's own mapping between submissions moves
/// the frame.
#[test]
fn an_unaligned_stage_buffer_window_is_copied_into_the_owner_staged_arm() {
    use reims_vgpu::backend::provider_compute::{device_epoch, host_import_alignment};
    use reims_vgpu::runtime::guest_ram::{GuestRamImport, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::{GuestWindowRun, RegisteredWindow};

    /// The bind's own offset inside the granule: not a whole number of the
    /// device's import granules, and a whole number of the four-byte storage the
    /// fragment stage reads.
    const HEAD: u64 = 4;
    /// Bytes the declaration's own reach covers at this bind.
    const BIND_BYTES: u64 = 16;

    let _guard = engine_test_session();
    let alignment = host_import_alignment().expect("the owner rail's provider answers");
    assert!(
        alignment > 0,
        "the staged arm is still gated on the device importing host pointers"
    );
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let mut owner = AlignedHost::new(2 * page, page);
    let head = usize::try_from(HEAD).expect("the head fits usize");
    // The fragment's red, written at the bind's own unaligned offset: the bytes
    // the copied range has to carry.
    owner.as_mut_slice()[head..head + 4].copy_from_slice(&[0, 0, 0x80, 0x3f]);
    let import = std::sync::Arc::new(
        GuestRamImport::new_host_allocation(owner.pointer as usize, 2 * page as u64, alignment)
            .expect("a page-aligned synthetic host allocation"),
    );
    let guest = || {
        let anchor = import
            .slice(0, page as u64)
            .expect("the first granule is inside the import");
        GuestRef::new(std::sync::Arc::clone(&import), anchor)
            .expect("the slice came from this import")
    };
    let import_id = import.id().get();
    // The provider-shaped window the registration ledger derives: the whole
    // granule, which does cover the bind's bytes.
    let registered = RegisteredWindow {
        import: import.id(),
        base: owner.pointer as u64,
        length: page as u64,
        epoch: 1,
    };
    // The bind as the draw path builds it: one host run over the owner's
    // mapping, the window's first byte `HEAD` into it, and the window the
    // ledger derived on the run.
    let gather = || {
        BufferContent::GuestRuns(engine::GuestRunSource {
            runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
                owner.pointer as usize,
                2 * page as u64,
                0,
                HEAD + BIND_BYTES,
            )
            .expect("the bind's own bytes are inside the mapping")]),
            source_offset: HEAD,
            total_len: BIND_BYTES,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(vec![GuestWindowRun {
                window_offset: 0,
                guest: guest(),
                window: Some(registered),
            }])),
            direct_image: None,
        })
    };
    let stages = buffer_declaring_stages("render_frag_buffer.air", "reims_buffer_frag");
    let request = |content: &BufferContent| {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.storage_buffers.push(engine::StorageBufferResource {
            binding: 0,
            content: content.clone(),
        });
        req
    };
    // The engine's own frame for the same request and the same bytes, taken
    // before anything is registered on the owner rail: the engine's device
    // context is created lazily on its first draw, and that creation resets the
    // owner rail.
    let Some(engine_frame) =
        engine_pixels("unaligned stage-buffer window", &stages, request(&gather()))
    else {
        return;
    };
    provider_owner::register(Region {
        import: import_id,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: owner.pointer as usize,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x40_0000),
    })
    .expect("a page-aligned registration is a legal provider region");
    let log_before = std::fs::read_to_string(reims_vgpu_observe::fail_log_path())
        .unwrap_or_default()
        .len();
    let content = gather();
    let binds = [StageBufferBind {
        stage: RenderPipelineStage::Fragment,
        index: 0,
        content: &content,
        // The seam states no window: everything the copy needs is derived.
        window: None,
        landing: None,
    }];

    use reims_vgpu::backend::provider_wire;

    provider_wire::capture_submission_frames(true);
    let frames_before = provider_wire::wire_counts();
    let delivered = provider_render::provider_submissions();
    let red = match provider_render::submit_render(
        &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
        &request(&content),
    ) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!(
            "an unaligned window inside the device's granules is staged, not refused: {other:?}"
        ),
    };
    let frames = provider_wire::captured_submission_frames();
    provider_wire::capture_submission_frames(false);
    eprintln!(
        "unaligned stage-buffer window: device host-import alignment = {alignment}; the bind's \
         bytes start {HEAD} byte(s) into a {page} byte granule; provider submissions {delivered} \
         -> {}, texel (0, 0) {:?}",
        provider_render::provider_submissions(),
        texel_at(&red, 0, 0),
    );
    assert!(
        provider_render::provider_submissions() > delivered,
        "the census shape reaches the canonical provider instead of the engine"
    );
    assert_eq!(
        texel_at(&red, 0, 0),
        [255, 0, 0, 255],
        "the fragment read the bind's own bytes at the unaligned offset"
    );
    assert_frames_equal(
        "unaligned stage-buffer window, both rails",
        &red,
        &engine_frame,
    );

    // The wire's own reading: the copy is what crossed as the bind's source,
    // under this binding's label, at the bind's own length.
    assert_eq!(
        provider_wire::wire_counts().submit_frames,
        frames_before.submit_frames + 1,
        "the seam produced exactly one submission frame for the staged draw"
    );
    assert_eq!(frames.len(), 1, "and the capture holds it");
    let (wire_trace, _wire_resources) = provider_wire::carried_submission(&frames[0])
        .expect("the provider's own decoder reads the frame back");
    let wire_pass = wire_trace
        .passes
        .iter()
        .find_map(|pass| pass.as_render())
        .expect("the frame carries the render pass");
    let view = wire_pass
        .stage_buffers
        .first()
        .expect("the frame carries the declared stage buffer");
    let source = match &view.view.source {
        BufferSource::OwnedBytes(bytes) => format!("owned_bytes({})", bytes.len()),
        BufferSource::StagedLease(lease) => format!("staged_lease({})", lease.get()),
        BufferSource::BorrowedNoCopy(lease) => format!("borrowed_no_copy({})", lease.get()),
        BufferSource::GuestRuns(runs) => format!("guest_runs({})", runs.len()),
    };
    eprintln!(
        "wire stage-buffer view: stage={:?} offset={} length={} source={source}",
        view.stage, view.view.offset, view.view.length,
    );
    assert!(
        matches!(view.view.source, BufferSource::StagedLease(_)),
        "the copied bind crosses the wire as the owner's staged lease"
    );
    assert_eq!(
        view.view.length, BIND_BYTES,
        "the staged view is the bind's own bytes, not the whole granule"
    );

    // The lease row, verbatim: the arm this bind left through, and the label
    // the plan found its view by (`stage_buffer_owner_binding(Fragment, 0)`).
    let log = std::fs::read_to_string(reims_vgpu_observe::fail_log_path()).expect("fail log");
    let fresh = &log[log_before.min(log.len())..];
    let lease = fresh
        .lines()
        .find(|line| line.contains("provider_owner_lease") && line.contains("no_copy=0"))
        .unwrap_or_else(|| panic!("the staged lease row was emitted: {fresh}"));
    eprintln!("lease row: {lease}");
    assert!(
        lease.contains("channel=staged") && lease.contains("binding=65536"),
        "the row names the staged arm under the fragment stage buffer's own label: {lease}"
    );
    assert!(
        !fresh
            .lines()
            .any(|line| line.contains("provider_owner_lease") && line.contains("no_copy=1")),
        "nothing about this bind was imported: the copy replaced the borrowed arm wholesale"
    );
    assert!(route_count("render_provider_unaligned_window_staged") >= 1);
    assert!(
        route_count("render_provider_unaligned_window_bytes") >= BIND_BYTES,
        "the byte census carries the bytes the copy moved"
    );

    // The falsifiable half: move the owner's own bytes and the next submission
    // follows. A rail that had cached one copy would be unmoved by this.
    owner.as_mut_slice()[head..head + 4].copy_from_slice(&[0, 0, 0, 0]);
    let black = match provider_render::submit_render(
        &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
        &request(&content),
    ) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("the same shape stays in class after the mapping moved: {other:?}"),
    };
    eprintln!(
        "provider texel after moving the mapping's unaligned float: {:?}",
        texel_at(&black, 0, 0),
    );
    assert_eq!(
        texel_at(&black, 0, 0),
        [0, 0, 0, 255],
        "the same bind now carries the moved bytes"
    );
    assert_frames_differ(
        "the owner's own bytes reach the provider's frame through the copy",
        &red,
        &black,
    );
}

/// R18's read is bounded by the registration that names it, and every way it
/// can fail is a named refusal rather than a shorter answer.
///
/// The class gate cannot reach these shapes — `gather_window` refuses a window
/// that does not cover the bind's bytes before the copy is asked for one — so
/// they are driven here, against the owner rail's own API, which is where the
/// read happens: an unregistered import, a bind whose `head` leaves the window,
/// a window that reaches past the registration, an empty bind, and an import
/// the address rail retired. A copy that read any of them would be reading
/// bytes outside the range the caller named, which is the one thing this
/// function must never do.
#[test]
fn a_window_copy_reads_only_bytes_its_own_registration_covers() {
    use reims_vgpu::backend::provider_compute::{device_epoch, host_import_alignment};
    use reims_vgpu::runtime::guest_ram::GuestRamImport;

    let _guard = engine_test_session();
    let alignment = host_import_alignment().expect("the owner rail's provider answers");
    assert!(alignment > 0, "the copy needs the import gate open");
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let mut owner = AlignedHost::new(2 * page, page);
    owner.as_mut_slice()[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let import =
        GuestRamImport::new_host_allocation(owner.pointer as usize, 2 * page as u64, alignment)
            .expect("a page-aligned synthetic host allocation");
    let import_id = import.id().get();
    let base = owner.pointer as u64;
    let window = |import: u64, host_va: u64, length: u64, head: u64, bytes_len: u64| {
        provider_owner::Window {
            binding: 65536,
            import,
            host_va,
            length,
            head,
            bytes_len,
        }
    };
    let slug = |refusal: provider_owner::Decline| refusal.slug().to_owned();

    // An import the rail was never handed: nothing to bound the read with.
    assert_eq!(
        slug(
            provider_owner::window_bytes(window(import_id, base, page as u64, 4, 8))
                .expect_err("an unregistered import has no window")
        ),
        "owner_unregistered_region",
    );

    provider_owner::register(Region {
        import: import_id,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: owner.pointer as usize,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x40_0000),
    })
    .expect("a page-aligned registration is a legal provider region");

    // The bind's own bytes, at an unaligned head: exactly the range the
    // borrowed arm would have bound.
    assert_eq!(
        provider_owner::window_bytes(window(import_id, base, page as u64, 4, 4))
            .expect("a window inside its registration reads"),
        vec![5, 6, 7, 8],
    );

    // A `head` that leaves the bind's bytes outside the window.
    assert_eq!(
        slug(
            provider_owner::window_bytes(window(
                import_id,
                base,
                page as u64,
                (page - 4) as u64,
                8,
            ))
            .expect_err("a bind reaching past its window")
        ),
        "owner_window_outside_region",
    );

    // A window that reaches past the registration, even with the bind inside
    // the window.
    assert_eq!(
        slug(
            provider_owner::window_bytes(window(
                import_id,
                base + page as u64,
                (2 * page) as u64,
                0,
                8,
            ))
            .expect_err("a window reaching past its registration")
        ),
        "owner_window_outside_region",
    );

    // A window that starts before the registration.
    assert_eq!(
        slug(
            provider_owner::window_bytes(window(import_id, base - 8, page as u64, 0, 8,))
                .expect_err("a window starting before its registration")
        ),
        "owner_window_outside_region",
    );

    // An empty bind: the staged arm it would become has nothing to hold.
    assert_eq!(
        slug(
            provider_owner::window_bytes(window(import_id, base, page as u64, 4, 0))
                .expect_err("an empty bind")
        ),
        "owner_window_empty",
    );

    // The address rail announced the mapping moved. An alias-shaped
    // registration is the one a retirement ends (a RAMBlock registration never
    // is), so the copy has to stop with it.
    let alias_id = 0x18_a11a5_u64;
    provider_owner::register(Region {
        import: alias_id,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: owner.pointer as usize,
        length: page as u64,
        page_size: alignment,
        gpa_base: None,
    })
    .expect("a page-aligned registration is a legal provider region");
    provider_owner::retire_region(alias_id);
    assert_eq!(
        slug(
            provider_owner::window_bytes(window(alias_id, base, page as u64, 4, 8))
                .expect_err("a retired registration derives no window")
        ),
        "owner_region_retired",
    );
}

/// R9e's refusal half: a gather the seam cannot cut one window from stays on
/// the engine, each under the bucket its own fact names.
///
/// Three shapes, none of them a looser door: bytes scattered over more than one
/// run, a run whose import the registration ledger never registered, and a
/// packed bind whose `source_offset` leaves the view's own host pointer off the
/// device's import granule *and* whose import the owner rail was never handed,
/// so there is no registration to copy the bytes out of. The first two answer
/// `..._stage_buffer_gather` (the rail mints no copy for them either), and the
/// third answers `..._stage_buffer_alignment` — since R18 that bucket is the
/// *unreadable* window's, not every off-granule one: a window the rail can read
/// is copied into the staged arm
/// (`an_unaligned_stage_buffer_window_is_copied_into_the_owner_staged_arm`),
/// and only a window it cannot read keeps the draw on the engine, because a
/// declined draw is not a fallback.
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
    // pointer is not a whole number of the device's import granules — and this
    // import was never handed to the owner rail, so there is no registration
    // the bytes could be copied out of (R18).
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
    assert!(
        detail.contains("owner_unregistered_region"),
        "the sentence names the fact that stopped the copy: {detail}"
    );

    // Both buckets are counters, not latches.
    assert!(route_count("render_provider_out_of_class_stage_buffer_gather") >= 2);
    assert!(route_count("render_provider_out_of_class_stage_buffer_alignment") >= 1);
}

/// R9q: a vertex stream the draw path resolved through the zero-copy rail
/// leaves for the canonical provider as the *guest RAM window its bind was cut
/// from* — not as a copy, and not as a refusal.
///
/// The census v7 shape is the one R9p's fixture already states — four
/// `[[stage_in]]` attributes over two interleaved tables beside a
/// `[[buffer(2)]]` argument — but the boot resolved those tables through the
/// zero-copy rail, so every attribute arrived as `BufferContent::GuestRuns`
/// (`buffer_guest_imports = 359273`, `buffer_guest_gathers = 0`,
/// `evidence/gate3-census-v7-2026-09-17/`). R9p's grouping key answered
/// `_ => false` for that arm, the count stayed four, and the argument at Metal
/// index 2 stayed inside the streams' block: 88968 of the 88982
/// `stage_buffer_shape_vertex_layout` rows.
///
/// This test drives that shape with the bytes in a registered host mapping, so
/// every fact is checkable at once: the two tables still merge into two
/// canonical streams, the draw reaches the provider, the provider's frame reads
/// the owner's own mapping (moving it moves the frame), the frame is
/// byte-identical to the engine's for the same request, and the lease row names
/// the no-copy arm. The three-table control beside it is the door that is still
/// the door: a request whose attributes really do read three guest binds states
/// three streams, and the same `[[buffer(2)]]` argument then occupies one of
/// them.
#[test]
fn a_vertex_stream_shared_by_two_attributes_reads_its_guest_window_without_a_copy() {
    use reims_vgpu::backend::provider_compute::{device_epoch, host_import_alignment};
    use reims_vgpu::runtime::guest_ram::{GuestRamImport, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::{GuestWindowRun, RegisteredWindow};

    let _guard = engine_test_session();
    let stages = shared_table_stages();
    assert_eq!(stages.vertex_attribute_locations, vec![0, 1, 2, 3]);
    assert_eq!(
        stages.vertex_stage_buffer_declarations.len(),
        1,
        "the fixture's vertex stage declares the [[buffer(2)]] argument: {:#?}",
        stages.vertex_stage_buffer_declarations
    );
    let alignment = host_import_alignment().expect("the owner rail's provider answers");
    assert!(
        alignment > 0,
        "this device must advertise VK_EXT_external_memory_host for the no-copy arm"
    );
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    // Three granules: one per guest vertex bind the request reads, so the
    // three-table control below has a real registration to name as well.
    let mut owner = AlignedHost::new(3 * page, page);
    let import = std::sync::Arc::new(
        GuestRamImport::new_host_allocation(owner.pointer as usize, 3 * page as u64, alignment)
            .expect("a page-aligned synthetic host allocation"),
    );
    // One granule per guest vertex bind, so the reference's own bound is the
    // range the ledger derived the window from — the shape production hands
    // both rails (`window_range` refuses a window that is not the bound it was
    // cut from).
    let guest = |granule: usize| {
        let anchor = import
            .slice((granule * page) as u64, page as u64)
            .expect("one granule is inside the import");
        GuestRef::new(std::sync::Arc::clone(&import), anchor)
            .expect("the slice came from this import")
    };
    let import_id = import.id().get();
    // The mapping's own address, copied out here so the controls below can move
    // the bytes without holding the allocation borrowed.
    let base = owner.pointer as usize;

    // The two interleaved tables the descriptor behind those four attributes
    // reads (`shared_table_stages`'s fixture), written into the owner's own
    // mapping: positions with one offset in the first granule, the other two
    // offsets in the second. The engine arm below reads the same bytes through
    // the runs' host pointers, so both rails compare one set of bytes.
    let positions = [(-1.0_f32, -3.0_f32), (-1.0, 1.0), (3.0, 1.0)];
    let offsets = |x: f32| [(x, 0.0_f32); 3];
    let first_bytes = interleaved(positions, offsets(0.125));
    let second_bytes = interleaved(offsets(0.0625), offsets(0.0625));
    let third_bytes = interleaved(offsets(0.0625), offsets(0.0625));
    owner.as_mut_slice()[..48].copy_from_slice(&first_bytes);
    owner.as_mut_slice()[page..page + 48].copy_from_slice(&second_bytes);
    owner.as_mut_slice()[2 * page..2 * page + 48].copy_from_slice(&third_bytes);

    // The window the registration ledger would derive for each bind: the
    // granule the bind's bytes live in, base-aligned and one granule long.
    let window = |granule: usize| RegisteredWindow {
        import: import.id(),
        base: base as u64 + (granule * page) as u64,
        length: page as u64,
        epoch: 1,
    };
    // One zero-copy bind as the draw path builds it: one run over the owner's
    // mapping at the bind's own offset, the bind's bytes at its start, and the
    // provider-shaped window the ledger derived on the run.
    let table = |granule: usize, bytes: &[u8]| -> engine::GuestRunSource {
        engine::GuestRunSource {
            runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
                base,
                3 * page as u64,
                (granule * page) as u64,
                bytes.len() as u64,
            )
            .expect("the bind's own bytes are inside the mapping")]),
            source_offset: 0,
            total_len: bytes.len() as u64,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(vec![GuestWindowRun {
                window_offset: 0,
                guest: guest(granule),
                window: Some(window(granule)),
            }])),
            direct_image: None,
        }
    };
    let first = table(0, &first_bytes);
    let second = table(1, &second_bytes);

    let guest_attribute =
        |location: u32, offset: u32, source: &engine::GuestRunSource| -> VertexAttributeResource {
            VertexAttributeResource {
                location,
                // The engine numbers one Vulkan binding per attribute
                // *location*, exactly as the staged fixtures do.
                binding: location,
                format: VertexAttributeFormat::parse(MTL_FORMAT_VERTEX_FLOAT2)
                    .expect("Float2 is a vertex format"),
                offset,
                stride: 16,
                step_function: VertexStepFunction::PerVertex,
                step_rate: 1,
                content: BufferContent::GuestRuns(source.clone()),
            }
        };
    let request = |first: &engine::GuestRunSource,
                   second: &engine::GuestRunSource,
                   third: Option<&engine::GuestRunSource>,
                   tail: &std::sync::Arc<Vec<u8>>| {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.vertex_attributes = vec![
            guest_attribute(0, 0, first),
            guest_attribute(1, 8, first),
            guest_attribute(2, 0, second),
            match third {
                Some(third) => guest_attribute(3, 0, third),
                None => guest_attribute(3, 8, second),
            },
        ];
        req.storage_buffers.push(engine::StorageBufferResource {
            binding: 2,
            content: BufferContent::Bytes(std::sync::Arc::clone(tail)),
        });
        req
    };
    let still = std::sync::Arc::new(f32x2(&[(0.0, 0.0)]));
    let moved = std::sync::Arc::new(f32x2(&[(-0.25, 0.0)]));
    let (width, _) = extent();

    // The engine's own frame for the same request, before anything is
    // registered on the owner rail: the engine's device context is created
    // lazily on its first draw, and that creation resets the owner rail (the
    // imports die with the device it builds), so the registration below has to
    // follow it. The shape is one both rails execute, which is what the class
    // gate promises and what this comparison reads.
    let Some(engine_frame) = engine_pixels(
        "two guest tables",
        &stages,
        request(&first, &second, None, &still),
    ) else {
        return;
    };

    provider_owner::register(Region {
        import: import_id,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: base,
        length: 3 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x40_0000),
    })
    .expect("a page-aligned registration is a legal provider region");
    let log_before = std::fs::read_to_string(reims_vgpu_observe::fail_log_path())
        .unwrap_or_default()
        .len();

    let frame = |label: &str,
                 first: &engine::GuestRunSource,
                 second: &engine::GuestRunSource,
                 tail: &std::sync::Arc<Vec<u8>>| {
        let req = request(first, second, None, tail);
        let content = BufferContent::Bytes(std::sync::Arc::clone(tail));
        let binds = [staged_bind(RenderPipelineStage::Vertex, 2, &content)];
        match provider_render::submit_render(
            &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
            &req,
        ) {
            RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
            other => panic!(
                "{label}: a draw whose vertex streams live in guest RAM windows leaves for the \
                 provider: {other:?}"
            ),
        }
    };

    // The frame the seam encodes for this shape, held so the provider's own
    // decoder can be asked what crossed the owner→provider wire (R9j's rule:
    // the wire's reading is the one a remote owner would see).
    use reims_vgpu::backend::provider_wire;

    provider_wire::capture_submission_frames(true);
    let frames_before = provider_wire::wire_counts();
    let delivered = provider_render::provider_submissions();
    let reviewed = frame("two guest tables", &first, &second, &still);
    let frames = provider_wire::captured_submission_frames();
    provider_wire::capture_submission_frames(false);
    eprintln!(
        "two-guest-table draw: provider submissions {delivered} -> {}, texel (0, 0) {:?}, \
         texel (width - 1, 0) {:?}, host-import alignment {alignment}",
        provider_render::provider_submissions(),
        texel_at(&reviewed, 0, 0),
        texel_at(&reviewed, width - 1, 0),
    );
    assert!(
        provider_render::provider_submissions() > delivered,
        "the census pair reaches the canonical provider instead of the engine"
    );
    assert_clear_texel("two guest tables: texel (0, 0)", texel_at(&reviewed, 0, 0));
    assert_texel_near(
        "two guest tables: texel (width - 1, 0)",
        texel_at(&reviewed, width - 1, 0),
        FRAGMENT_TEXEL,
    );
    // The frame crossed the owner→provider wire on the plan's own rule — a
    // vertex window with no stage-buffer declaration is a leasing submission
    // too (R9q) — and the provider's decoder reads the two streams back as the
    // borrowed arm.
    assert_eq!(
        provider_wire::wire_counts().submit_frames,
        frames_before.submit_frames + 1,
        "the seam produced exactly one submission frame for the window-backed draw"
    );
    assert_eq!(frames.len(), 1, "and the capture holds it");
    let (wire_trace, _wire_resources) = provider_wire::carried_submission(&frames[0])
        .expect("the provider's own decoder reads the frame back");
    let wire_pass = wire_trace
        .passes
        .iter()
        .find_map(|pass| pass.as_render())
        .expect("the frame carries the render pass");
    for view in &wire_pass.vertex_buffers {
        let source = match &view.source {
            BufferSource::OwnedBytes(bytes) => format!("owned_bytes({})", bytes.len()),
            BufferSource::StagedLease(lease) => format!("staged_lease({})", lease.get()),
            BufferSource::BorrowedNoCopy(lease) => format!("borrowed_no_copy({})", lease.get()),
            BufferSource::GuestRuns(runs) => format!("guest_runs({})", runs.len()),
        };
        eprintln!(
            "wire vertex view: binding={} offset={} length={} source={source}",
            view.metal_binding, view.offset, view.length,
        );
        assert!(
            matches!(view.source, BufferSource::BorrowedNoCopy(_)),
            "each guest table crosses the wire as the owner's own mapping"
        );
    }
    assert_eq!(
        wire_pass.vertex_buffers.len(),
        2,
        "two fetch tables crossed the wire, not four attributes"
    );

    // The lease row, verbatim: the arm this draw's vertex bytes left through.
    // One registration holds both windows, so the plan imports one lease and
    // the trace resolves two views inside it.
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
    assert_frames_equal("two guest tables, both rails", &reviewed, &engine_frame);

    // Every input of the shape has to reach the frame: the `[[buffer(2)]]`
    // argument's own bytes, and one attribute of either guest table. The two
    // table controls move the *owner's own mapping*, which is the falsifiable
    // half of the no-copy claim: a rail that had copied the bind would be
    // unmoved by them.
    let argument_moved = frame("argument bytes moved", &first, &second, &moved);
    assert_ne!(
        argument_moved, reviewed,
        "the [[buffer(2)]] argument's own bytes have to reach the vertex stage"
    );

    owner.as_mut_slice()[..48].copy_from_slice(&interleaved(positions, offsets(0.0)));
    let first_zeroed = frame("first table's tail zeroed", &first, &second, &still);
    assert_ne!(
        first_zeroed, reviewed,
        "the first guest table's own bytes have to reach the frame"
    );
    owner.as_mut_slice()[..48].copy_from_slice(&first_bytes);

    owner.as_mut_slice()[page..page + 48]
        .copy_from_slice(&interleaved(offsets(0.0625), offsets(0.0)));
    let second_zeroed = frame("second table's tail zeroed", &first, &second, &still);
    assert_ne!(
        second_zeroed, reviewed,
        "the second guest table's own bytes have to reach the frame"
    );
    owner.as_mut_slice()[page..page + 48].copy_from_slice(&second_bytes);

    // The rule the census read as the door is still the door: a request whose
    // attributes really do read three guest binds states three canonical
    // streams, and the same `[[buffer(2)]]` argument then occupies one of
    // them.
    let third = table(2, &third_bytes);
    let three_tables = request(&first, &second, Some(&third), &still);
    let content = BufferContent::Bytes(std::sync::Arc::clone(&still));
    let binds = [staged_bind(RenderPipelineStage::Vertex, 2, &content)];
    let band = route_count("stage_buffer_shape_vertex_layout");
    let delivered = provider_render::provider_submissions();
    match provider_render::submit_render(
        &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
        &three_tables,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            eprintln!(
                "three guest tables, [[buffer(2)]]: slug={} detail={} route {} -> {}",
                reason.slug(),
                reason.detail(),
                band,
                route_count("stage_buffer_shape_vertex_layout"),
            );
            assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_stage_buffer_shape"
            );
            assert!(
                reason.detail().contains("[[buffer(2)]]")
                    && reason.detail().contains("3 vertex stream(s)"),
                "the sentence names the slot and the streams it lands inside: {reason}"
            );
        }
        other => panic!("a declaration inside the stream block stays on the engine: {other:?}"),
    }
    assert_eq!(
        route_count("stage_buffer_shape_vertex_layout"),
        band + 1,
        "the arm the census reads is the one this refusal charged"
    );
    assert_eq!(
        provider_render::provider_submissions(),
        delivered,
        "and the draw never reaches the provider"
    );
}

/// R11: the draw's *index* stream, resolved by the draw path through the
/// zero-copy rail, leaves for the canonical provider as the guest RAM window
/// its bind was cut from — not as a copy, and not as a refusal.
///
/// Census v9 (`evidence/gate3-census-v9-2026-09-17/`) is the reading this
/// increment answers: once R10's sampling declarations opened, the
/// second-largest first failure of the boot's 90618 seam rows was
/// `render_provider_out_of_class_index_staging` — 25489 rows, 28.1% — because
/// the class's index arm was `staged_bytes` only. A real boot resolves an
/// indexed draw's buffer through the same zero-copy rail as its vertex streams
/// (`runtime/draw/vulkan.rs`'s `load_index_content_reason` →
/// `load_buffer_content_resolved`, both with the zero-copy rail allowed), so
/// those draws arrived as `BufferContent::GuestRuns` and had no source this
/// rail could state.
///
/// This test drives that shape with the index bytes in a registered host
/// mapping, so every fact is checkable at once: the draw reaches the provider,
/// the wire's index view names the borrowed arm, the provider's frame reads the
/// owner's own mapping (moving it moves the frame), and the frame is
/// byte-identical to the engine's for the same request and the same bytes.
#[test]
fn an_index_stream_in_a_registered_window_leaves_without_a_copy() {
    use reims_vgpu::backend::provider_compute::{device_epoch, host_import_alignment};
    use reims_vgpu::runtime::guest_ram::{GuestRamImport, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::{GuestWindowRun, RegisteredWindow};

    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let alignment = host_import_alignment().expect("the owner rail's provider answers");
    assert!(
        alignment > 0,
        "this device must advertise VK_EXT_external_memory_host for the no-copy arm"
    );
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    // Two granules: the three indices live in the first, so the window the
    // ledger derives is a real sub-range of an imported block rather than the
    // block itself.
    let mut owner = AlignedHost::new(2 * page, page);
    let import = std::sync::Arc::new(
        GuestRamImport::new_host_allocation(owner.pointer as usize, 2 * page as u64, alignment)
            .expect("a page-aligned synthetic host allocation"),
    );
    let guest = || {
        let anchor = import
            .slice(0, page as u64)
            .expect("the first granule is inside the import");
        GuestRef::new(std::sync::Arc::clone(&import), anchor)
            .expect("the slice came from this import")
    };
    let import_id = import.id().get();
    let base = owner.pointer as usize;
    // The index bytes the draw reads, written into the owner's own mapping:
    // the reviewed shape's own `[0, 1, 2]`, so the frame the provider lands has
    // to be the frame the engine draws from the same request and the same
    // bytes.
    owner.as_mut_slice()[..INDEX_BYTES.len()].copy_from_slice(&INDEX_BYTES);
    // The provider-shaped window the registration ledger would derive for the
    // draw's index bind: the granule its bytes were cut from.
    let registered = RegisteredWindow {
        import: import.id(),
        base: base as u64,
        length: page as u64,
        epoch: 1,
    };
    // The index bind as the draw path builds it: one host run over the owner's
    // mapping, the three `u32` indices at its start, and the window the ledger
    // derived on the run. Nothing here reads the mapping — the engine arm below
    // reads it through the run, and the provider arm through the lease.
    let indices = || engine::GuestRunSource {
        runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
            base,
            2 * page as u64,
            0,
            INDEX_BYTES.len() as u64,
        )
        .expect("the bind's own bytes are inside the mapping")]),
        source_offset: 0,
        total_len: INDEX_BYTES.len() as u64,
        row_length_texels: 0,
        pages: Some(std::sync::Arc::new(vec![GuestWindowRun {
            window_offset: 0,
            guest: guest(),
            window: Some(registered),
        }])),
        direct_image: None,
    };
    let request = |source: &engine::GuestRunSource| {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.indexed = Some(IndexedDrawResource {
            index_type: IndexType::U32,
            index_count: 3,
            vertex_offset: 0,
            content: BufferContent::GuestRuns(source.clone()),
        });
        req
    };
    // The engine's own frame for the same request, before anything is
    // registered on the owner rail: the engine's device context is created
    // lazily on its first draw, and that creation resets the owner rail (the
    // imports die with the device it builds), so the registration below has to
    // follow it.
    let Some(engine_frame) = engine_pixels("index window", &stages, request(&indices())) else {
        return;
    };
    provider_owner::register(Region {
        import: import_id,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: base,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x40_0000),
    })
    .expect("a page-aligned registration is a legal provider region");
    let log_before = std::fs::read_to_string(reims_vgpu_observe::fail_log_path())
        .unwrap_or_default()
        .len();
    let frame = |label: &str| -> Vec<u8> {
        match provider_render::submit_render(
            &inputs(&stages, RenderChainRole::SoleOrTail),
            &request(&indices()),
        ) {
            RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
            other => panic!(
                "{label}: a draw whose index stream lives in a guest RAM window leaves for the \
                 provider: {other:?}"
            ),
        }
    };

    // The frame the seam encodes for this shape, held so the provider's own
    // decoder can be asked what crossed the owner→provider wire.
    use reims_vgpu::backend::provider_wire;

    provider_wire::capture_submission_frames(true);
    let frames_before = provider_wire::wire_counts();
    let delivered = provider_render::provider_submissions();
    let reviewed = frame("guest index window");
    let frames = provider_wire::captured_submission_frames();
    provider_wire::capture_submission_frames(false);
    eprintln!(
        "guest index window draw: provider submissions {delivered} -> {}, texel (0, 0) {:?}, \
         host-import alignment {alignment}, window {page} byte(s) of registration {import_id}",
        provider_render::provider_submissions(),
        texel_at(&reviewed, 0, 0),
    );
    assert!(
        provider_render::provider_submissions() > delivered,
        "the census shape reaches the canonical provider instead of the engine"
    );
    assert_solid("guest index window", &reviewed);
    // The frame crossed the owner→provider wire on the plan's own rule — a
    // window-backed index bind is a leasing submission even with no stage
    // buffer declaration and no vertex window — and the provider's decoder
    // reads the index view back as the borrowed arm.
    assert_eq!(
        provider_wire::wire_counts().submit_frames,
        frames_before.submit_frames + 1,
        "the seam produced exactly one submission frame for the window-backed draw"
    );
    assert_eq!(frames.len(), 1, "and the capture holds it");
    let (wire_trace, _wire_resources) = provider_wire::carried_submission(&frames[0])
        .expect("the provider's own decoder reads the frame back");
    let wire_pass = wire_trace
        .passes
        .iter()
        .find_map(|pass| pass.as_render())
        .expect("the frame carries the render pass");
    let index_view = wire_pass
        .indices
        .as_ref()
        .expect("the frame carries the index binding");
    let source = match &index_view.view.source {
        BufferSource::OwnedBytes(bytes) => format!("owned_bytes({})", bytes.len()),
        BufferSource::StagedLease(lease) => format!("staged_lease({})", lease.get()),
        BufferSource::BorrowedNoCopy(lease) => format!("borrowed_no_copy({})", lease.get()),
        BufferSource::GuestRuns(runs) => format!("guest_runs({})", runs.len()),
    };
    eprintln!(
        "wire index view: binding={} offset={} length={} source={source}",
        index_view.view.metal_binding, index_view.view.offset, index_view.view.length,
    );
    assert!(
        matches!(index_view.view.source, BufferSource::BorrowedNoCopy(_)),
        "the index stream crosses the wire as the owner's own mapping"
    );
    assert_eq!(
        index_view.view.length,
        INDEX_BYTES.len() as u64,
        "the view is the bind's own three indices, not the whole granule"
    );

    // The lease row, verbatim: the arm this draw's index bytes left through.
    // The label is the index stream's own namespace (`0x30000`), which is how
    // the plan found the view it minted for this bind.
    let log = std::fs::read_to_string(reims_vgpu_observe::fail_log_path()).expect("fail log");
    let fresh = &log[log_before.min(log.len())..];
    let lease = fresh
        .lines()
        .find(|line| line.contains("provider_owner_lease") && line.contains("no_copy=1"))
        .unwrap_or_else(|| panic!("the borrowed lease row was emitted: {fresh}"));
    eprintln!("lease row: {lease}");
    assert!(
        lease.contains("channel=borrowed")
            && lease.contains(&format!("import={import_id}"))
            && lease.contains("binding=196608"),
        "the row names the no-copy arm, this import and the index stream's own label: {lease}"
    );
    assert_frames_equal("guest index window, both rails", &reviewed, &engine_frame);

    // The falsifiable half: rewrite the owner's own mapping, with nothing else
    // touched. Every index becomes zero, both triangles degenerate and the
    // attachment keeps the clear's bytes wherever the quad would have covered —
    // a rail that had copied the indices into its own trace would be unmoved by
    // this.
    owner.as_mut_slice()[..INDEX_BYTES.len()].fill(0);
    let degenerated = frame("guest index window, indices rewritten to zero");
    let (width, _) = extent();
    eprintln!(
        "provider texel after the owner's index bytes became zeros: {:?}",
        texel_at(&degenerated, 0, 0),
    );
    assert_clear_texel(
        "degenerated window indices: texel (0, 0)",
        texel_at(&degenerated, 0, 0),
    );
    assert_clear_texel(
        "degenerated window indices: texel (width - 1, 0)",
        texel_at(&degenerated, width - 1, 0),
    );
    assert_frames_differ(
        "the owner's own index bytes reach the provider's frame",
        &reviewed,
        &degenerated,
    );
}

/// R18: the index half of the same answer — a dust-sized stream, read once, and
/// the arm where the extra copy costs least.
///
/// The bind's three `u32` indices start four bytes into the granule, so the
/// view's own host pointer misses the device's import granules while the window
/// covers the bytes; the class copies them out of the registration and imports
/// them as the owner's staged lease instead of keeping the draw on the engine
/// (`render_provider_out_of_class_index_alignment` is the bucket this shape
/// answered with before this increment). The index stream is the smallest bind
/// this class carries, so the copy is also the cheapest place to read what the
/// new arm costs: twelve bytes, one lease, and a view that is the bind's own
/// length rather than the granule's.
#[test]
fn an_unaligned_index_stream_window_is_copied_into_the_owner_staged_arm() {
    use reims_vgpu::backend::provider_compute::{device_epoch, host_import_alignment};
    use reims_vgpu::runtime::guest_ram::{GuestRamImport, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::{GuestWindowRun, RegisteredWindow};

    /// The index bind's own offset inside its granule.
    const HEAD: u64 = 4;

    let _guard = engine_test_session();
    let stages = reviewed_stages();
    let alignment = host_import_alignment().expect("the owner rail's provider answers");
    assert!(
        alignment > 0,
        "the staged arm is still gated on the device importing host pointers"
    );
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let mut owner = AlignedHost::new(2 * page, page);
    let head = usize::try_from(HEAD).expect("the head fits usize");
    // The reviewed shape's own indices, at the bind's unaligned offset.
    owner.as_mut_slice()[head..head + INDEX_BYTES.len()].copy_from_slice(&INDEX_BYTES);
    let import = std::sync::Arc::new(
        GuestRamImport::new_host_allocation(owner.pointer as usize, 2 * page as u64, alignment)
            .expect("a page-aligned synthetic host allocation"),
    );
    let guest = || {
        let anchor = import
            .slice(0, page as u64)
            .expect("the first granule is inside the import");
        GuestRef::new(std::sync::Arc::clone(&import), anchor)
            .expect("the slice came from this import")
    };
    let import_id = import.id().get();
    let base = owner.pointer as usize;
    let registered = RegisteredWindow {
        import: import.id(),
        base: base as u64,
        length: page as u64,
        epoch: 1,
    };
    let indices = || engine::GuestRunSource {
        runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
            base,
            2 * page as u64,
            0,
            HEAD + INDEX_BYTES.len() as u64,
        )
        .expect("the bind's own bytes are inside the mapping")]),
        source_offset: HEAD,
        total_len: INDEX_BYTES.len() as u64,
        row_length_texels: 0,
        pages: Some(std::sync::Arc::new(vec![GuestWindowRun {
            window_offset: 0,
            guest: guest(),
            window: Some(registered),
        }])),
        direct_image: None,
    };
    let request = |source: &engine::GuestRunSource| {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.indexed = Some(IndexedDrawResource {
            index_type: IndexType::U32,
            index_count: 3,
            vertex_offset: 0,
            content: BufferContent::GuestRuns(source.clone()),
        });
        req
    };
    let Some(engine_frame) = engine_pixels("unaligned index stream", &stages, request(&indices()))
    else {
        return;
    };
    provider_owner::register(Region {
        import: import_id,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: base,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x40_0000),
    })
    .expect("a page-aligned registration is a legal provider region");
    let log_before = std::fs::read_to_string(reims_vgpu_observe::fail_log_path())
        .unwrap_or_default()
        .len();
    let frame = |label: &str| -> Vec<u8> {
        match provider_render::submit_render(
            &inputs(&stages, RenderChainRole::SoleOrTail),
            &request(&indices()),
        ) {
            RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
            other => panic!(
                "{label}: an unaligned index window inside the device's granules is staged, not \
                 refused: {other:?}"
            ),
        }
    };

    use reims_vgpu::backend::provider_wire;

    provider_wire::capture_submission_frames(true);
    let delivered = provider_render::provider_submissions();
    let reviewed = frame("unaligned index stream");
    let frames = provider_wire::captured_submission_frames();
    provider_wire::capture_submission_frames(false);
    eprintln!(
        "unaligned index stream: device host-import alignment = {alignment}; the three indices \
         start {HEAD} byte(s) into a {page} byte granule; provider submissions {delivered} -> {}, \
         texel (0, 0) {:?}",
        provider_render::provider_submissions(),
        texel_at(&reviewed, 0, 0),
    );
    assert!(
        provider_render::provider_submissions() > delivered,
        "the census shape reaches the canonical provider instead of the engine"
    );
    assert_solid("unaligned index stream", &reviewed);
    assert_frames_equal(
        "unaligned index stream, both rails",
        &reviewed,
        &engine_frame,
    );

    // The wire's own reading: the index view crosses as the copy, at the bind's
    // own length, not the whole granule.
    assert_eq!(frames.len(), 1, "the capture holds the submission frame");
    let (wire_trace, _wire_resources) = provider_wire::carried_submission(&frames[0])
        .expect("the provider's own decoder reads the frame back");
    let wire_pass = wire_trace
        .passes
        .iter()
        .find_map(|pass| pass.as_render())
        .expect("the frame carries the render pass");
    let index_view = wire_pass
        .indices
        .as_ref()
        .expect("the frame carries the index binding");
    let source = match &index_view.view.source {
        BufferSource::OwnedBytes(bytes) => format!("owned_bytes({})", bytes.len()),
        BufferSource::StagedLease(lease) => format!("staged_lease({})", lease.get()),
        BufferSource::BorrowedNoCopy(lease) => format!("borrowed_no_copy({})", lease.get()),
        BufferSource::GuestRuns(runs) => format!("guest_runs({})", runs.len()),
    };
    eprintln!(
        "wire index view: offset={} length={} source={source}",
        index_view.view.offset, index_view.view.length,
    );
    assert!(
        matches!(index_view.view.source, BufferSource::StagedLease(_)),
        "the copied index bind crosses the wire as the owner's staged lease"
    );
    assert_eq!(
        index_view.view.length,
        INDEX_BYTES.len() as u64,
        "the staged view is the bind's own three indices"
    );

    // The lease row, verbatim: the staged arm under the index stream's own
    // label (`0x30000`).
    let log = std::fs::read_to_string(reims_vgpu_observe::fail_log_path()).expect("fail log");
    let fresh = &log[log_before.min(log.len())..];
    let lease = fresh
        .lines()
        .find(|line| line.contains("provider_owner_lease") && line.contains("no_copy=0"))
        .unwrap_or_else(|| panic!("the staged lease row was emitted: {fresh}"));
    eprintln!("lease row: {lease}");
    assert!(
        lease.contains("channel=staged")
            && lease.contains(&format!("bytes={}", INDEX_BYTES.len()))
            && lease.contains("binding=196608"),
        "the row names the staged arm, the bind's own length and the index stream's label: {lease}"
    );
    assert!(route_count("render_provider_unaligned_window_staged") >= 1);

    // The falsifiable half: rewrite the owner's own index bytes, with nothing
    // else touched. Every index becomes zero, both triangles degenerate and the
    // attachment keeps the clear's bytes wherever the quad would have covered.
    owner.as_mut_slice()[head..head + INDEX_BYTES.len()].fill(0);
    let degenerated = frame("unaligned index stream, indices rewritten to zero");
    let (width, _) = extent();
    eprintln!(
        "provider texel after the owner's index bytes became zeros: {:?}",
        texel_at(&degenerated, 0, 0),
    );
    assert_clear_texel(
        "degenerated unaligned indices: texel (0, 0)",
        texel_at(&degenerated, 0, 0),
    );
    assert_clear_texel(
        "degenerated unaligned indices: texel (width - 1, 0)",
        texel_at(&degenerated, width - 1, 0),
    );
    assert_frames_differ(
        "the owner's own index bytes reach the provider's frame through the copy",
        &reviewed,
        &degenerated,
    );
}

/// R11's refusal half: an index gather the seam cannot cut one registered
/// window from stays on the engine, each under the bucket its own fact names.
///
/// The same three shapes R9q's refusal half drives for a vertex stream, on the
/// arm this increment moved: bytes scattered over more than one run, a run
/// whose import the registration ledger never registered, and a bind whose
/// `source_offset` leaves the view's own host pointer off the device's import
/// granule beside an import the owner rail was never handed. The first two
/// answer `render_provider_out_of_class_index_staging` — the bucket every index
/// gather answered with before this increment — and the third answers
/// `render_provider_out_of_class_index_alignment`, the *unreadable* window's
/// bucket since R18: an off-granule window the rail can read is copied into the
/// staged arm (`an_unaligned_index_stream_window_is_copied_into_the_owner_staged_arm`),
/// and only one with no registration to read from keeps the draw on the engine.
#[test]
fn an_index_gather_the_seam_cannot_cut_a_window_from_stays_on_the_engine() {
    use reims_vgpu::backend::provider_compute::host_import_alignment;
    use reims_vgpu::runtime::guest_ram::{GuestRamImport, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::{GuestWindowRun, RegisteredWindow};

    let _guard = engine_test_session();
    let stages = reviewed_stages();
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
    let import_id = import.id().get();
    // The import is deliberately *not* handed to the owner rail: the
    // off-granule window below is the shape whose copy needs a registration to
    // read the bytes out of, so this is what the alignment bucket still
    // answers (R18).
    let registered = RegisteredWindow {
        import: import.id(),
        base: owner.pointer as u64,
        length: page as u64,
        epoch: 1,
    };
    // The gather a test hands the gate: one host run holding the three indices,
    // and the page runs below are the only variable.
    let gather = |source_offset: u64, pages: Vec<GuestWindowRun>| -> engine::GuestRunSource {
        engine::GuestRunSource {
            runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
                owner.pointer as usize,
                2 * page as u64,
                source_offset,
                INDEX_BYTES.len() as u64,
            )
            .expect("the bind's own bytes are inside the mapping")]),
            source_offset,
            total_len: INDEX_BYTES.len() as u64,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(pages)),
            direct_image: None,
        }
    };
    let request = |source: &engine::GuestRunSource| {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.indexed = Some(IndexedDrawResource {
            index_type: IndexType::U32,
            index_count: 3,
            vertex_offset: 0,
            content: BufferContent::GuestRuns(source.clone()),
        });
        req
    };
    let answer = |label: &str, source: &engine::GuestRunSource| -> (String, String) {
        match provider_render::submit_render(
            &inputs(&stages, RenderChainRole::SoleOrTail),
            &request(source),
        ) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                (reason.slug().to_owned(), reason.detail().to_owned())
            }
            other => {
                panic!("{label}: an index gather outside one window is out of class: {other:?}")
            }
        }
    };
    let delivered = provider_render::provider_submissions();

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
                window_offset: 6,
                guest: guest(),
                window: Some(registered),
            },
        ],
    );
    let (slug, detail) = answer("scattered index gather", &scattered);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_index_staging");
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
    let (slug, detail) = answer("unregistered index gather", &unregistered);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_index_staging");

    // Off-granule: the bind starts inside the window, so the view's own host
    // pointer is not a whole number of the device's import granules — and this
    // import was never handed to the owner rail, so there is no registration
    // the bytes could be copied out of (R18).
    let off_granule = gather(
        4,
        vec![GuestWindowRun {
            window_offset: 0,
            guest: guest(),
            window: Some(registered),
        }],
    );
    let (slug, detail) = answer("off-granule index gather", &off_granule);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_index_alignment");
    assert!(
        detail.contains(&format!("{alignment} byte alignment")),
        "the sentence names the alignment it crossed: {detail}"
    );
    assert!(
        detail.contains("owner_unregistered_region"),
        "the sentence names the fact that stopped the copy: {detail}"
    );

    // Both buckets are counters, not latches, and no refused shape reaches the
    // provider.
    assert!(route_count("render_provider_out_of_class_index_staging") >= 2);
    assert!(route_count("render_provider_out_of_class_index_alignment") >= 1);
    assert_eq!(
        provider_render::provider_submissions(),
        delivered,
        "a refused index gather never reaches the provider"
    );
    assert_eq!(registered.import.get(), import_id);
}

/// R18: the same staged answer for a **vertex stream** the device cannot import
/// in place.
///
/// Census v13 read 131 rows of this shape as
/// `render_provider_out_of_class_vertex_alignment` — the vertex half of the
/// same sentence (`evidence/gate3-census-v13-2026-09-17` §0.4) — and here the
/// copy has to reach the *vertex stage*'s own fetches: the bind's bytes start
/// four bytes into the granule, the window still covers them, and the class
/// states the copied bytes as an owner-issued staged lease instead of keeping
/// the draw on the engine. The two-stream shape is the reviewed one: the
/// second table is aligned and keeps the borrowed arm in the same submission,
/// which is what makes the two arms distinguishable in one lease set.
#[test]
fn an_unaligned_vertex_stream_window_is_copied_into_the_owner_staged_arm() {
    use reims_vgpu::backend::provider_compute::{device_epoch, host_import_alignment};
    use reims_vgpu::runtime::guest_ram::{GuestRamImport, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::{GuestWindowRun, RegisteredWindow};

    /// The first stream's own offset inside its granule.
    const HEAD: u64 = 4;
    /// The stream's own bytes (`interleaved`'s three vertices).
    const TABLE_BYTES: u64 = 48;

    let _guard = engine_test_session();
    let stages = shared_table_stages();
    let alignment = host_import_alignment().expect("the owner rail's provider answers");
    assert!(
        alignment > 0,
        "the staged arm is still gated on the device importing host pointers"
    );
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let mut owner = AlignedHost::new(2 * page, page);
    let import = std::sync::Arc::new(
        GuestRamImport::new_host_allocation(owner.pointer as usize, 2 * page as u64, alignment)
            .expect("a page-aligned synthetic host allocation"),
    );
    let guest = |granule: usize| {
        let anchor = import
            .slice((granule * page) as u64, page as u64)
            .expect("one granule is inside the import");
        GuestRef::new(std::sync::Arc::clone(&import), anchor)
            .expect("the slice came from this import")
    };
    let import_id = import.id().get();
    let base = owner.pointer as usize;
    let head = usize::try_from(HEAD).expect("the head fits usize");

    let positions = [(-1.0_f32, -3.0_f32), (-1.0, 1.0), (3.0, 1.0)];
    let offsets = |x: f32| [(x, 0.0_f32); 3];
    let first_bytes = interleaved(positions, offsets(0.125));
    let second_bytes = interleaved(offsets(0.0625), offsets(0.0625));
    // The first table's bytes live at the bind's own unaligned offset; the
    // second table is the aligned control beside it.
    owner.as_mut_slice()[head..head + TABLE_BYTES as usize].copy_from_slice(&first_bytes);
    owner.as_mut_slice()[page..page + TABLE_BYTES as usize].copy_from_slice(&second_bytes);

    let window = |granule: usize| RegisteredWindow {
        import: import.id(),
        base: base as u64 + (granule * page) as u64,
        length: page as u64,
        epoch: 1,
    };
    let table = |granule: usize, source_offset: u64, bytes: &[u8]| -> engine::GuestRunSource {
        let granule_base = (granule * page) as u64;
        engine::GuestRunSource {
            runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
                base,
                2 * page as u64,
                granule_base,
                source_offset + bytes.len() as u64,
            )
            .expect("the bind's own bytes are inside the mapping")]),
            source_offset,
            total_len: bytes.len() as u64,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(vec![GuestWindowRun {
                window_offset: 0,
                guest: guest(granule),
                window: Some(window(granule)),
            }])),
            direct_image: None,
        }
    };
    let first = table(0, HEAD, &first_bytes);
    let second = table(1, 0, &second_bytes);
    let guest_attribute =
        |location: u32, offset: u32, source: &engine::GuestRunSource| -> VertexAttributeResource {
            VertexAttributeResource {
                location,
                binding: location,
                format: VertexAttributeFormat::parse(MTL_FORMAT_VERTEX_FLOAT2)
                    .expect("Float2 is a vertex format"),
                offset,
                stride: 16,
                step_function: VertexStepFunction::PerVertex,
                step_rate: 1,
                content: BufferContent::GuestRuns(source.clone()),
            }
        };
    let request = |first: &engine::GuestRunSource,
                   second: &engine::GuestRunSource,
                   tail: &std::sync::Arc<Vec<u8>>| {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.vertex_attributes = vec![
            guest_attribute(0, 0, first),
            guest_attribute(1, 8, first),
            guest_attribute(2, 0, second),
            guest_attribute(3, 8, second),
        ];
        req.storage_buffers.push(engine::StorageBufferResource {
            binding: 2,
            content: BufferContent::Bytes(std::sync::Arc::clone(tail)),
        });
        req
    };
    let still = std::sync::Arc::new(f32x2(&[(0.0, 0.0)]));
    let (width, _) = extent();

    // The engine's own frame for the same request and the same bytes, before
    // anything is registered: the engine's device creation resets the owner
    // rail.
    let Some(engine_frame) = engine_pixels(
        "unaligned vertex stream",
        &stages,
        request(&first, &second, &still),
    ) else {
        return;
    };
    provider_owner::register(Region {
        import: import_id,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: base,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x40_0000),
    })
    .expect("a page-aligned registration is a legal provider region");
    let log_before = std::fs::read_to_string(reims_vgpu_observe::fail_log_path())
        .unwrap_or_default()
        .len();
    let frame = |label: &str, tail: &std::sync::Arc<Vec<u8>>| -> Vec<u8> {
        let content = BufferContent::Bytes(std::sync::Arc::clone(tail));
        let binds = [staged_bind(RenderPipelineStage::Vertex, 2, &content)];
        match provider_render::submit_render(
            &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
            &request(&first, &second, tail),
        ) {
            RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
            other => panic!(
                "{label}: an unaligned vertex window inside the device's granules is staged, not \
                 refused: {other:?}"
            ),
        }
    };

    use reims_vgpu::backend::provider_wire;

    provider_wire::capture_submission_frames(true);
    let delivered = provider_render::provider_submissions();
    let reviewed = frame("unaligned vertex stream", &still);
    let frames = provider_wire::captured_submission_frames();
    provider_wire::capture_submission_frames(false);
    eprintln!(
        "unaligned vertex stream: device host-import alignment = {alignment}; the first table's \
         bytes start {HEAD} byte(s) into its granule; provider submissions {delivered} -> {}, \
         texel (0, 0) {:?}, texel (width - 1, 0) {:?}",
        provider_render::provider_submissions(),
        texel_at(&reviewed, 0, 0),
        texel_at(&reviewed, width - 1, 0),
    );
    assert!(
        provider_render::provider_submissions() > delivered,
        "the census shape reaches the canonical provider instead of the engine"
    );
    assert_clear_texel(
        "unaligned vertex stream: texel (0, 0)",
        texel_at(&reviewed, 0, 0),
    );
    assert_texel_near(
        "unaligned vertex stream: texel (width - 1, 0)",
        texel_at(&reviewed, width - 1, 0),
        FRAGMENT_TEXEL,
    );
    assert_frames_equal(
        "unaligned vertex stream, both rails",
        &reviewed,
        &engine_frame,
    );

    // The wire's own reading: the two streams cross under their own arms, and
    // the staged one carries the copied bytes at the bind's own length.
    assert_eq!(frames.len(), 1, "the capture holds the submission frame");
    let (wire_trace, _wire_resources) = provider_wire::carried_submission(&frames[0])
        .expect("the provider's own decoder reads the frame back");
    let wire_pass = wire_trace
        .passes
        .iter()
        .find_map(|pass| pass.as_render())
        .expect("the frame carries the render pass");
    let arms: Vec<String> = wire_pass
        .vertex_buffers
        .iter()
        .map(|view| match &view.source {
            BufferSource::OwnedBytes(bytes) => format!("owned_bytes({})", bytes.len()),
            BufferSource::StagedLease(lease) => format!("staged_lease({})", lease.get()),
            BufferSource::BorrowedNoCopy(lease) => format!("borrowed_no_copy({})", lease.get()),
            BufferSource::GuestRuns(runs) => format!("guest_runs({})", runs.len()),
        })
        .collect();
    eprintln!("wire vertex views: {arms:?}");
    assert_eq!(
        arms.len(),
        2,
        "two fetch tables crossed, not four attributes"
    );
    assert!(
        arms[0].starts_with("staged_lease"),
        "the unaligned table crosses as the owner's staged lease: {}",
        arms[0]
    );
    assert!(
        arms[1].starts_with("borrowed_no_copy"),
        "the aligned table beside it keeps the no-copy arm: {}",
        arms[1]
    );

    // The lease rows, verbatim: the copy under the first stream's own label
    // (`vertex_stream_owner_binding(0)`), the import under the second's, and
    // the request's own `[[buffer(2)]]` argument staged beside them — the arm
    // a request with no window behind it takes, which this shape's copies have
    // to be distinguishable from.
    let log = std::fs::read_to_string(reims_vgpu_observe::fail_log_path()).expect("fail log");
    let fresh = &log[log_before.min(log.len())..];
    let staged = fresh
        .lines()
        .find(|line| {
            line.contains("provider_owner_lease")
                && line.contains("no_copy=0")
                && line.contains("binding=131072")
        })
        .unwrap_or_else(|| panic!("the stream's own staged lease row was emitted: {fresh}"));
    let borrowed = fresh
        .lines()
        .find(|line| line.contains("provider_owner_lease") && line.contains("no_copy=1"))
        .unwrap_or_else(|| panic!("the borrowed lease row was emitted: {fresh}"));
    eprintln!("staged lease row: {staged}\nborrowed lease row: {borrowed}");
    assert!(
        staged.contains("channel=staged") && staged.contains(&format!("bytes={TABLE_BYTES}")),
        "the copy is stated under the first stream's own label, at the bind's own length: {staged}"
    );
    assert!(
        borrowed.contains("channel=borrowed") && borrowed.contains(&format!("import={import_id}")),
        "the aligned table's window is imported under the second stream's label: {borrowed}"
    );
    assert!(route_count("render_provider_unaligned_window_staged") >= 1);

    // The falsifiable half: move the unaligned table's own bytes and the next
    // submission follows them.
    owner.as_mut_slice()[head..head + TABLE_BYTES as usize]
        .copy_from_slice(&interleaved(offsets(0.125), offsets(0.0)));
    let moved = frame("unaligned vertex stream, first table moved", &still);
    assert_frames_differ(
        "the owner's own vertex bytes reach the provider's frame through the copy",
        &reviewed,
        &moved,
    );
}

/// R9q's refusal half: a vertex gather the seam cannot cut one registered
/// window from stays on the engine, each under the bucket its own fact names.
///
/// The same three shapes R9e's refusal half drives for a stage buffer, on the
/// arm this increment moved: bytes scattered over more than one run, a run
/// whose import the registration ledger never registered, and a bind whose
/// `source_offset` leaves the view's own host pointer off the device's import
/// granule beside an import the owner rail was never handed. The first two
/// answer `render_provider_out_of_class_vertex_staging` — the bucket every
/// gather answered with before this increment — and the third answers
/// `render_provider_out_of_class_vertex_alignment`, the *unreadable* window's
/// bucket since R18: an off-granule window the rail can read is copied into the
/// staged arm (`an_unaligned_vertex_stream_window_is_copied_into_the_owner_staged_arm`),
/// and only one with no registration to read from keeps the draw on the engine.
#[test]
fn a_vertex_gather_the_seam_cannot_cut_a_window_from_stays_on_the_engine() {
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
    let import_id = import.id().get();
    // The import is deliberately *not* handed to the owner rail: the
    // off-granule window below is the shape whose copy needs a registration to
    // read the bytes out of, so this is what the alignment bucket still
    // answers (R18).
    let registered = RegisteredWindow {
        import: import.id(),
        base: owner.pointer as u64,
        length: page as u64,
        epoch: 1,
    };
    // The gather a test hands the gate: one host run whose bytes are the bind,
    // and the page runs below are the only variable.
    let gather = |source_offset: u64, pages: Vec<GuestWindowRun>| -> engine::GuestRunSource {
        engine::GuestRunSource {
            runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
                owner.pointer as usize,
                2 * page as u64,
                source_offset,
                16,
            )
            .expect("the bind's own bytes are inside the mapping")]),
            source_offset,
            total_len: 16,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(pages)),
            direct_image: None,
        }
    };
    let stages = reviewed_stages();
    let request = |source: &engine::GuestRunSource| {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.vertex_attributes = vec![VertexAttributeResource {
            location: 0,
            binding: 0,
            format: VertexAttributeFormat::parse(MTL_FORMAT_VERTEX_FLOAT2)
                .expect("Float2 is a vertex format"),
            offset: 0,
            stride: 8,
            step_function: VertexStepFunction::PerVertex,
            step_rate: 1,
            content: BufferContent::GuestRuns(source.clone()),
        }];
        req
    };
    let answer = |label: &str, source: &engine::GuestRunSource| -> (String, String) {
        match provider_render::submit_render(
            &inputs(&stages, RenderChainRole::SoleOrTail),
            &request(source),
        ) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                (reason.slug().to_owned(), reason.detail().to_owned())
            }
            other => {
                panic!("{label}: a vertex gather outside one window is out of class: {other:?}")
            }
        }
    };
    let delivered = provider_render::provider_submissions();

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
    let (slug, detail) = answer("scattered vertex gather", &scattered);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_vertex_staging");
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
    let (slug, detail) = answer("unregistered vertex gather", &unregistered);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_vertex_staging");

    // Off-granule: the bind starts inside the window, so the view's own host
    // pointer is not a whole number of the device's import granules — and this
    // import was never handed to the owner rail, so there is no registration
    // the bytes could be copied out of (R18).
    let off_granule = gather(
        4,
        vec![GuestWindowRun {
            window_offset: 0,
            guest: guest(),
            window: Some(registered),
        }],
    );
    let (slug, detail) = answer("off-granule vertex gather", &off_granule);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_vertex_alignment");
    assert!(
        detail.contains(&format!("{alignment} byte alignment")),
        "the sentence names the alignment it crossed: {detail}"
    );
    assert!(
        detail.contains("owner_unregistered_region"),
        "the sentence names the fact that stopped the copy: {detail}"
    );

    // Both buckets are counters, not latches, and no refused shape reaches the
    // provider.
    assert!(route_count("render_provider_out_of_class_vertex_staging") >= 2);
    assert!(route_count("render_provider_out_of_class_vertex_alignment") >= 1);
    assert_eq!(
        provider_render::provider_submissions(),
        delivered,
        "a refused vertex gather never reaches the provider"
    );
    assert_eq!(registered.import.get(), import_id);
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
        fragment_texture_declarations: Vec::new(),
        sampler_family: RenderSamplerFamily::default(),
        texture_interface_refusals: Vec::new(),
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
        metal_api_core::provider::BufferSource::GuestRuns(runs) => {
            format!("guest_runs={}", runs.len())
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

/// R9m's drop and R9n's admission: a `[[buffer(N)]]` argument whose translated
/// entry point never dereferences it is not a declaration this rail states, and
/// since R9n it does not keep the draw on the engine either.
///
/// The reflection answers `ResourceAccess::Unused` for such an argument — the
/// metadata declares it and the emitted entry point never reaches it — and
/// `pipeline_resolve` maps that answer onto the class's own vocabulary
/// (`StageBufferAccess::Unused`). R9m's change is what the class does with it:
/// the statement the seam builds carries only the arguments the entries reach,
/// the dropped declaration is *counted* (`stage_buffer_skipped_unused_1`) so a
/// census can size the population, and no submission frame is produced at all —
/// a frame is what carries declarations, and this request states none.
///
/// R9n is the other half: the drop is not an answer any more. The canonical
/// registration leaves a reflected `Unused` slot out of the pairing when the
/// contract does not declare it (`research/docs/23` §95), so the request leaves
/// for the provider with the slot neither declared nor bound — the R9b path a
/// bind no stage declares already takes — and the two rails land the fixture's
/// own frame byte for byte. The door's old bucket is charged by nothing any
/// more, and the declaration that would state the slot is still refused by name
/// (`the_registration_leaves_a_slot_the_entry_never_dereferences_out_of_the_pairing`).
#[test]
fn a_stage_buffer_the_entry_never_dereferences_is_dropped_from_the_statement_and_admitted() {
    let _guard = engine_test_session();
    use reims_vgpu::backend::provider_wire;

    // The measured fact: the fixture's own translation answers `Unused` for the
    // one slot it declares, and states no byte range for it.
    let stages =
        buffer_declaring_stages("render_frag_buffer_unused.air", "reims_unused_buffer_frag");
    assert_eq!(
        stages.fragment_stage_buffer_declarations,
        vec![StageBufferDeclaration {
            index: 0,
            access: StageBufferAccess::Unused,
            footprint: StageBufferFootprint::Unstated,
        }],
        "the unused fixture declares one slot its entry never dereferences"
    );

    // The statement: the slot is not in it, and the drop is a count beside it.
    // The request states the bind the slot would have had on the engine, so the
    // shape is the production one: the guest set the buffer and the entry never
    // reached it.
    let content = BufferContent::Bytes(std::sync::Arc::new(vec![0xa5u8; 16]));
    let binds = [staged_bind(RenderPipelineStage::Fragment, 0, &content)];
    let inputs = inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds);
    let statement = inputs.stage_buffer_statement();
    eprintln!(
        "statement for the unused shape: declared={} dropped={} unstated={:?}",
        statement.declared.len(),
        statement.dropped(),
        statement
            .unstated
            .iter()
            .map(|(stage, declaration)| (
                stage.name(),
                declaration.index,
                declaration.access.name()
            ))
            .collect::<Vec<_>>(),
    );
    assert!(
        statement.declared.is_empty(),
        "no declaration is stated for a slot no entry reaches: {statement:?}"
    );
    assert_eq!(statement.dropped(), 1, "and the drop is counted");

    let skipped = route_count("stage_buffer_skipped_unused_1");
    let zeros = route_count("stage_buffer_skipped_unused_0");
    let door = route_count("render_provider_out_of_class_stage_buffer_unused");
    provider_wire::capture_submission_frames(true);
    let frames_before = provider_wire::wire_counts().submit_frames;
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.storage_buffers.push(engine::StorageBufferResource {
        binding: 0,
        content: content.clone(),
    });
    let provider = match provider_render::submit_render(&inputs, &req) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("a slot no entry reaches leaves for the provider since R9n: {other:?}"),
    };
    let frames = provider_wire::captured_submission_frames();
    provider_wire::capture_submission_frames(false);
    eprintln!(
        "gate for the dropped declaration: the request left for the provider, {} byte(s), \
         texel={:?}",
        provider.len(),
        texel_at(&provider, 0, 0),
    );
    assert_eq!(
        route_count("stage_buffer_skipped_unused_1"),
        skipped + 1,
        "the dropped declaration is counted in its own band"
    );
    assert_eq!(
        route_count("stage_buffer_skipped_unused_0"),
        zeros,
        "and the band's zero arm is not charged for it"
    );
    assert_eq!(
        route_count("render_provider_out_of_class_stage_buffer_unused"),
        door,
        "the bucket the drop used to answer under is charged by nothing any more"
    );
    eprintln!(
        "skip band for the dropped declaration: stage_buffer_skipped_unused_1={}, \
         stage_buffer_skipped_unused_0={}; the old door bucket reads {}",
        route_count("stage_buffer_skipped_unused_1"),
        route_count("stage_buffer_skipped_unused_0"),
        route_count("render_provider_out_of_class_stage_buffer_unused"),
    );
    assert!(
        frames.is_empty(),
        "no declaration reaches the wire for this request"
    );
    assert_eq!(
        provider_wire::wire_counts().submit_frames,
        frames_before,
        "and the seam produced no submission frame at all"
    );

    // The dropped slot's bytes are not the frame's source: other bytes behind
    // the same slot land the same frame, which is what "the entry never
    // dereferences it" means on this rail.
    let other_content = BufferContent::Bytes(std::sync::Arc::new(vec![0x5au8; 16]));
    let other_binds = [staged_bind(
        RenderPipelineStage::Fragment,
        0,
        &other_content,
    )];
    let other = match provider_render::submit_render(
        &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &other_binds),
        &req,
    ) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("the same shape with other bytes is in class: {other:?}"),
    };
    assert_frames_equal(
        "the dropped slot's bytes do not reach the frame",
        &other,
        &provider,
    );

    // The engine's own frame for the same draw, byte for byte: the fixture's
    // fragment never reads the slot on the engine either.
    assert_solid("R9n: the dropped slot on the provider rail", &provider);
    let Some(engine) = engine_pixels("R9n dropped stage buffer", &stages, req) else {
        return;
    };
    assert_solid("R9n: the dropped slot on the engine rail", &engine);
    assert_frames_equal("R9n: the dropped slot on both rails", &provider, &engine);
    eprintln!(
        "provider vs engine, dropped stage buffer: {} bytes equal; texel={:?}",
        provider.len(),
        texel_at(&provider, 0, 0),
    );
}

/// The same pass with the slot dereferenced: the statement carries exactly that
/// one declaration, the drop band's zero arm is charged, the declaration crosses
/// the wire with a view beside it, and the two rails land the same frame.
///
/// This is R9m's other half. The fixture is the same shape as
/// `a_stage_buffer_the_entry_never_dereferences_is_dropped_from_the_statement_and_admitted`'s
/// — one `[[buffer(0)]]` argument, one admitted draw — with the entry point
/// actually reading it, so the pair of tests is a statement about the *access
/// arm* rather than about the slot, the stage or the draw.
#[test]
fn the_statement_and_the_wire_carry_the_one_slot_the_entry_dereferences() {
    let _guard = engine_test_session();
    use reims_vgpu::backend::provider_wire;

    let stages = buffer_declaring_stages("render_frag_buffer.air", "reims_buffer_frag");
    assert_eq!(
        stages.fragment_stage_buffer_declarations,
        vec![StageBufferDeclaration {
            index: 0,
            access: StageBufferAccess::Read,
            footprint: StageBufferFootprint::Static { max_bytes: 4 },
        }],
        "the read fixture declares one slot its entry dereferences"
    );
    // The fixture stores its bind's first `float32` in red: `1.0f` lands
    // `ff 00 00 ff`, and a zeroed bind lands `00 00 00 ff`.
    let mut bytes = vec![0u8; 16];
    bytes[..4].copy_from_slice(&[0, 0, 0x80, 0x3f]);
    let content = BufferContent::Bytes(std::sync::Arc::new(bytes));
    let binds = [staged_bind(RenderPipelineStage::Fragment, 0, &content)];
    let inputs = inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds);

    let statement = inputs.stage_buffer_statement();
    eprintln!(
        "statement for the dereferenced shape: declared={:?} dropped={} unstated={}",
        statement
            .declared
            .iter()
            .map(|(stage, declaration, access)| (
                stage.name(),
                declaration.index,
                declaration.access.name(),
                *access
            ))
            .collect::<Vec<_>>(),
        statement.dropped(),
        statement.unstated.len(),
    );
    assert_eq!(
        statement.declared.len(),
        1,
        "the one slot the entry reaches is stated"
    );
    assert_eq!(statement.declared[0].1.index, 0);
    assert_eq!(
        statement.declared[0].2,
        BufferAccess::Read,
        "under the access the contract states it with"
    );
    assert!(
        statement.unstated.is_empty(),
        "and nothing is withheld: {statement:?}"
    );
    assert_eq!(statement.dropped(), 0);

    let zeros = route_count("stage_buffer_skipped_unused_0");
    provider_wire::capture_submission_frames(true);
    let frames_before = provider_wire::wire_counts().submit_frames;
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.storage_buffers.push(engine::StorageBufferResource {
        binding: 0,
        content: content.clone(),
    });
    match provider_render::submit_render(&inputs, &req) {
        RenderRailOutcome::ProviderCompleted(_) => (),
        other => panic!("a dereferenced stage buffer is in class: {other:?}"),
    }
    let frames = provider_wire::captured_submission_frames();
    provider_wire::capture_submission_frames(false);
    assert_eq!(
        route_count("stage_buffer_skipped_unused_0"),
        zeros + 1,
        "the drop band is charged for every request the gate is handed, zero arm and all"
    );
    assert_eq!(
        provider_wire::wire_counts().submit_frames,
        frames_before + 1,
        "the seam produced exactly one submission frame"
    );
    assert_eq!(frames.len(), 1, "and the capture holds it");

    let (trace, _resources) = provider_wire::carried_submission(&frames[0])
        .expect("the provider's own decoder reads the frame back");
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
    eprintln!(
        "wire frame for the dereferenced slot: {} bytes, declarations={:?}, views={}",
        frames[0].len(),
        declarations,
        views.len(),
    );
    assert_eq!(
        declarations,
        vec![StageBufferBinding {
            stage: RenderPipelineStage::Fragment,
            index: 0,
            access: BufferAccess::Read,
            footprint: FootprintProof::Static { max_bytes: 4 },
        }],
        "the frame carries exactly the slot the entry dereferences"
    );
    assert_eq!(views.len(), 1, "and one view beside it");

    // The other rail, for the same draw: the same bytes behind the same slot.
    let provider = match provider_render::submit_render(&inputs, &req) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("the dereferenced shape leaves for the provider: {other:?}"),
    };
    assert_eq!(
        texel_at(&provider, 0, 0),
        [255, 0, 0, 255],
        "the provider reads the bytes behind the declaration"
    );
    let Some(engine) = engine_pixels("R9m dereferenced stage buffer", &stages, req) else {
        return;
    };
    assert_frames_equal(
        "R9m: the dereferenced slot on both rails",
        &provider,
        &engine,
    );
}

/// R9n's mixed shape: one stage that reaches one slot and never reaches
/// another.
///
/// The population the door was counting is not draws whose *only* declaration
/// is unread — it is draws whose translations state several buffers, only some
/// of which the entries reach. Dropping the unread slot must not drop the rest:
/// the statement carries the read slot with its own view beside it, the frame
/// crosses the wire with exactly that one declaration (the drop is a partition,
/// not a silence), and the provider registers and executes the pair while
/// leaving the unread slot out of the pairing (`research/docs/23` §95). Both
/// rails land the same frame, and the unread slot's bytes reach neither.
#[test]
fn the_statement_drops_the_unread_slot_beside_the_one_the_entry_reaches() {
    let _guard = engine_test_session();
    use reims_vgpu::backend::provider_wire;

    // The measured fact: the two-slot fixture's own translation states the
    // reached slot as a read and the other as `Unused`, with no reach at all
    // behind it.
    let stages = buffer_declaring_stages(
        "render_frag_buffer_read_and_unused.air",
        "reims_read_and_unused_buffer_frag",
    );
    assert_eq!(
        stages.fragment_stage_buffer_declarations,
        vec![
            StageBufferDeclaration {
                index: 0,
                access: StageBufferAccess::Read,
                footprint: StageBufferFootprint::Static { max_bytes: 4 },
            },
            StageBufferDeclaration {
                index: 1,
                access: StageBufferAccess::Unused,
                footprint: StageBufferFootprint::Unstated,
            },
        ],
        "the two-slot fixture declares a read slot beside an unread one"
    );

    // The statement: the slot the entry reaches is stated, the other is dropped
    // and counted.
    let mut bytes = vec![0u8; 16];
    bytes[..4].copy_from_slice(&[0, 0, 0x80, 0x3f]);
    let content = BufferContent::Bytes(std::sync::Arc::new(bytes));
    let binds = [staged_bind(RenderPipelineStage::Fragment, 0, &content)];
    let inputs = inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds);
    let statement = inputs.stage_buffer_statement();
    eprintln!(
        "statement for the mixed shape: declared={:?} dropped={} unstated={:?}",
        statement
            .declared
            .iter()
            .map(|(stage, declaration, access)| (
                stage.name(),
                declaration.index,
                declaration.access.name(),
                *access
            ))
            .collect::<Vec<_>>(),
        statement.dropped(),
        statement
            .unstated
            .iter()
            .map(|(stage, declaration)| (
                stage.name(),
                declaration.index,
                declaration.access.name()
            ))
            .collect::<Vec<_>>(),
    );
    assert_eq!(statement.declared.len(), 1, "the reached slot is stated");
    assert_eq!(statement.declared[0].1.index, 0);
    assert_eq!(
        statement.declared[0].2,
        BufferAccess::Read,
        "under the access the contract states it with"
    );
    assert_eq!(statement.dropped(), 1, "the unread slot is dropped");
    assert_eq!(
        statement.unstated.len(),
        1,
        "and the unread slot is the only thing withheld"
    );

    // The frame carries exactly the stated declaration, with one view beside
    // it: the unread slot crosses neither half of the canonical pair.
    let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    req.storage_buffers.push(engine::StorageBufferResource {
        binding: 0,
        content: content.clone(),
    });
    let skipped = route_count("stage_buffer_skipped_unused_1");
    provider_wire::capture_submission_frames(true);
    let frames_before = provider_wire::wire_counts().submit_frames;
    let provider = match provider_render::submit_render(&inputs, &req) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("the mixed shape leaves for the provider: {other:?}"),
    };
    let frames = provider_wire::captured_submission_frames();
    provider_wire::capture_submission_frames(false);
    assert_eq!(
        route_count("stage_buffer_skipped_unused_1"),
        skipped + 1,
        "the drop is counted for a request that states something"
    );
    assert_eq!(
        provider_wire::wire_counts().submit_frames,
        frames_before + 1,
        "the stated half produces exactly one submission frame"
    );
    assert_eq!(frames.len(), 1, "and the capture holds it");
    let (trace, _resources) = provider_wire::carried_submission(&frames[0])
        .expect("the provider's own decoder reads the frame back");
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
    eprintln!(
        "wire frame for the mixed shape: {} bytes, declarations={:?}, views={}",
        frames[0].len(),
        declarations,
        views.len(),
    );
    assert_eq!(
        declarations,
        vec![StageBufferBinding {
            stage: RenderPipelineStage::Fragment,
            index: 0,
            access: BufferAccess::Read,
            footprint: FootprintProof::Static { max_bytes: 4 },
        }],
        "the frame carries the reached slot alone"
    );
    assert_eq!(views.len(), 1, "with its own view beside it");

    // Both rails, for the same draw: the reached slot still decides the frame,
    // and the unread slot's bytes reach neither rail.
    assert_eq!(
        texel_at(&provider, 0, 0),
        [255, 0, 0, 255],
        "the provider reads the bytes behind the stated declaration"
    );
    let Some(engine) = engine_pixels("R9n mixed stage buffers", &stages, req) else {
        return;
    };
    assert_frames_equal("R9n: the mixed shape on both rails", &provider, &engine);
    eprintln!(
        "provider vs engine, mixed stage buffers: {} bytes equal; texel={:?}",
        provider.len(),
        texel_at(&provider, 0, 0),
    );
}

/// R9n's measured boundary: the canonical registration leaves a slot a
/// translated entry point never dereferences out of the pairing, and still
/// refuses every declaration that states it by name.
///
/// This is the fact the R9m drop is answered with, measured on the provider's
/// own registration entry point rather than inferred from this rail's code: the
/// same fragment stage is handed to `register_translated_render_pipeline` under
/// the contracts this rail could state for it. Undeclared — the shape R9m's
/// statement produces — registers (`research/docs/23` §95: the translation
/// classifies the argument `Unused`, the module reads nothing there, so the
/// slot needs no declaration and takes no part in the pairing), while a
/// declaration stating `unused` is refused by the contract itself
/// (`render_pipeline_contract_invalid`) and one stating `read` is a reflection
/// mismatch (`render_stage_reflection_mismatch`) — so the drop is not a
/// silence, and the same stage with the slot dereferenced registers against the
/// very declaration the rail states for it.
#[test]
fn the_registration_leaves_a_slot_the_entry_never_dereferences_out_of_the_pairing() {
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
    let vertex_air = fixture("reims_indexed_tri.air");
    let unused_air = fixture("render_frag_buffer_unused.air");
    let read_air = fixture("render_frag_buffer.air");
    let unused_fragment = translate(
        &unused_air,
        RenderStage::Fragment,
        "reims_unused_buffer_frag",
    );
    eprintln!(
        "unused fragment reflection bindings: {:?}",
        unused_fragment
            .reflection()
            .bindings
            .iter()
            .map(|binding| (
                binding.metal_index,
                binding.kind,
                binding.access,
                binding.descriptor,
                binding.footprint.clone(),
            ))
            .collect::<Vec<_>>()
    );
    let contract = |entry: &str, stage_buffers: Vec<StageBufferBinding>| RenderPipelineContract {
        vertex_entry: "reims_indexed_vertex".to_owned(),
        fragment_entry: entry.to_owned(),
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
        stage_buffers,
        textures: Vec::new(),
    };
    let digest = |name: &str| {
        SemanticDigest::new("reims-provider-render-rail-r9m", name.as_bytes().to_vec())
            .expect("the digest names a case")
    };
    let declaration = |access: BufferAccess| StageBufferBinding {
        stage: RenderPipelineStage::Fragment,
        index: 0,
        access,
        footprint: FootprintProof::Static { max_bytes: 16 },
    };

    // Arm 1: no declaration at all — the shape R9m's statement produces and
    // R9n admits. The registration leaves the reflected `Unused` slot out of
    // the pairing, so the stage registers with no stage-buffer declaration.
    let registered = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract("reims_unused_buffer_frag", Vec::new()),
            vertex: translate(&vertex_air, RenderStage::Vertex, "reims_indexed_vertex"),
            fragment: translate(
                &unused_air,
                RenderStage::Fragment,
                "reims_unused_buffer_frag",
            ),
            logical_digest: digest("unused-undeclared"),
        })
        .expect("an unread slot the contract does not declare takes no part in the pairing");
    eprintln!(
        "unused, undeclared: registered pipeline_id={:?}",
        registered.pipeline_id
    );

    // Arm 2: declared with the access the reflection itself reports. The
    // contract has no `Unused` declaration at all, so the refusal comes from
    // the contract rather than from the pairing.
    let refused = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(
                "reims_unused_buffer_frag",
                vec![declaration(BufferAccess::Unused)],
            ),
            vertex: translate(&vertex_air, RenderStage::Vertex, "reims_indexed_vertex"),
            fragment: translate(
                &unused_air,
                RenderStage::Fragment,
                "reims_unused_buffer_frag",
            ),
            logical_digest: digest("unused-declared-unused"),
        })
        .expect_err("the contract states no Unused declaration");
    eprintln!(
        "unused, declared unused: class={:?} slug={} detail={:?}",
        refused.class, refused.slug, refused.detail
    );
    assert_eq!(refused.slug, "render_pipeline_contract_invalid");

    // Arm 3: declared as the read a rail could try to fabricate for the slot.
    let refused = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(
                "reims_unused_buffer_frag",
                vec![declaration(BufferAccess::Read)],
            ),
            vertex: translate(&vertex_air, RenderStage::Vertex, "reims_indexed_vertex"),
            fragment: translate(
                &unused_air,
                RenderStage::Fragment,
                "reims_unused_buffer_frag",
            ),
            logical_digest: digest("unused-declared-read"),
        })
        .expect_err("the reflection and the contract disagree about the access");
    eprintln!(
        "unused, declared read: class={:?} slug={} fields={:?}",
        refused.class, refused.slug, refused.fields
    );
    assert_eq!(refused.slug, "render_stage_reflection_mismatch");
    assert_eq!(
        refused.fields.get("declared_access"),
        Some(&FieldValue::Text("read".to_owned()))
    );
    assert_eq!(
        refused.fields.get("reflected_access"),
        Some(&FieldValue::Text("unused".to_owned()))
    );

    // The control: the same stage with the slot dereferenced registers against
    // the declaration this rail states for it.
    let registered =
        provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract("reims_buffer_frag", vec![declaration(BufferAccess::Read)]),
            vertex: translate(&vertex_air, RenderStage::Vertex, "reims_indexed_vertex"),
            fragment: translate(&read_air, RenderStage::Fragment, "reims_buffer_frag"),
            logical_digest: digest("read-declared-read"),
        });
    assert!(
        registered.is_ok(),
        "the dereferenced slot registers against its own declaration: {registered:?}"
    );
}

/// The fixture of the R9p shape: four `[[stage_in]]` attributes at locations
/// 0..3 beside a vertex-stage `[[buffer(2)]]` argument the entry reads.
///
/// The vertex half is the census v6 door's own shape — the pair 77094 of the
/// 77103 `stage_buffer_shape_vertex_layout` rows answered with — and the
/// fragment half is the reviewed solid colour, so a frame is a function of the
/// bytes both stages read.
fn shared_table_stages() -> Stages {
    let mut stages = Stages {
        air: (
            fixture("reims_indexed_tri_two_stream_buffer2.air"),
            fixture("render_frag.air"),
        ),
        vertex_entry: "reims_two_stream_buffer2_vertex",
        fragment_entry: FRAGMENT_ENTRY,
        vertex_attribute_locations: vec![0, 1, 2, 3],
        vertex_stage_buffer_declarations: Vec::new(),
        fragment_stage_buffer_declarations: Vec::new(),
        fragment_texture_declarations: Vec::new(),
        sampler_family: RenderSamplerFamily::default(),
        texture_interface_refusals: Vec::new(),
    };
    stages.vertex_stage_buffer_declarations =
        declared_stage_buffers(&stages.air.0, RenderStage::Vertex, stages.vertex_entry);
    stages.fragment_stage_buffer_declarations =
        declared_stage_buffers(&stages.air.1, RenderStage::Fragment, stages.fragment_entry);
    eprintln!(
        "R9p fixture declarations: attributes={:?} vertex={:?} fragment={:?}",
        stages.vertex_attribute_locations,
        stages.vertex_stage_buffer_declarations,
        stages.fragment_stage_buffer_declarations,
    );
    stages
}

/// One attribute of one already-resolved fetch table.
///
/// The staged allocation is passed in rather than built here, because that
/// identity *is* the table: two attributes that hand this helper the same
/// `Arc` read one interleaved guest stream, which is the shape the runtime
/// builds (`runtime/draw/vulkan` resolves each attribute through its own
/// buffer index's single bind, and one bind is one `BufferContent::Bytes`
/// allocation).
fn attribute_of(
    location: u32,
    offset: u32,
    stride: u32,
    table: &std::sync::Arc<Vec<u8>>,
) -> VertexAttributeResource {
    VertexAttributeResource {
        location,
        binding: location,
        format: VertexAttributeFormat::parse(MTL_FORMAT_VERTEX_FLOAT2)
            .expect("Float2 is a vertex format"),
        offset,
        stride,
        step_function: VertexStepFunction::PerVertex,
        step_rate: 1,
        content: BufferContent::Bytes(std::sync::Arc::clone(table)),
    }
}

/// One interleaved table: three records of two `float2`s, the first at offset 0
/// and the second at offset 8 of a sixteen-byte record.
fn interleaved(heads: [(f32, f32); 3], tails: [(f32, f32); 3]) -> std::sync::Arc<Vec<u8>> {
    let mut bytes = Vec::with_capacity(48);
    for index in 0..3 {
        bytes.extend_from_slice(&f32x2(&[heads[index], tails[index]]));
    }
    std::sync::Arc::new(bytes)
}

/// R9p: a vertex stage that declares a `[[buffer(N)]]` argument beside its
/// `[[stage_in]]` attributes leaves for the provider exactly when the argument's
/// Metal index is clear of the draw's own *streams* — and one canonical stream
/// is one fetch table, not one attribute.
///
/// The census v6 measured this population as the stage-buffer door's whole
/// first failure: `stage_buffer_shape_vertex_layout` = 77103 of the 77700
/// `stage_buffer_shape` rows, and 77094 of those answered with one and the same
/// pair — a `[[buffer(2)]]` argument beside four attributes. The request the
/// seam builds carries one entry per attribute *location* (the engine numbers
/// one Vulkan binding per location), so a rail that numbers one canonical
/// stream per attribute occupies bindings 0..3 and the argument at Metal index 2
/// lands inside them. The descriptor behind those four attributes reads two
/// interleaved tables — locations 0/1 out of the first, 2/3 out of the second —
/// and the request still states that: one staged allocation per guest vertex
/// buffer, shared by every attribute that reads it.
///
/// Two halves, one per side of the pair:
///
/// * the canonical registration itself refuses the statement the old numbering
///   produced (one layout entry per attribute beside the declaration:
///   `StageBufferVertexLayoutConflict`, slug `trace_contract_invalid`) and mints
///   the one this increment states, which is what makes this rail's rule the
///   contract's own rule mirrored rather than a policy of this rail;
/// * both rails draw the shape and land the same bytes, and the refusal beside
///   them is the rule that is *still* the door: a request that really does read
///   three tables while declaring a `[[buffer(2)]]` argument states three
///   canonical streams, so index 2 lands inside them and the draw keeps the
///   engine by name.
///
/// Falsifiability: the argument's `float2` is added into the clip position and so
/// is every one of the four attributes, so each move below — the argument's own
/// bytes, and one attribute of either table — has to change the frame.
#[test]
fn a_vertex_stream_shared_by_two_attributes_leaves_room_for_its_stage_buffer() {
    let _guard = engine_test_session();
    let stages = shared_table_stages();
    assert_eq!(stages.vertex_attribute_locations, vec![0, 1, 2, 3]);
    assert_eq!(
        stages.vertex_stage_buffer_declarations,
        vec![StageBufferDeclaration {
            index: 2,
            access: StageBufferAccess::Read,
            // The entry loads one `float2` at a constant address: two four-byte
            // reaches at one stride of zero, which is the contract's static
            // ceiling and the number the registration compares its own
            // translation's footprint against.
            footprint: StageBufferFootprint::Static { max_bytes: 8 },
        }],
        "the fixture's own translation declares one read-only buffer at Metal index 2"
    );
    assert!(
        stages.fragment_stage_buffer_declarations.is_empty(),
        "the reviewed fragment half declares no buffer: {:#?}",
        stages.fragment_stage_buffer_declarations
    );

    // The provider's own half, read through its public registration entry point
    // rather than inferred from this rail's code: the statement one layout entry
    // per attribute would build is refused by the contract that consumes it, and
    // the statement one entry per table builds registers.
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
    let attribute = |location: u32, offset: u64| VertexAttribute {
        location,
        offset,
        format: VertexFormat::Float32x2,
    };
    let one_entry_per_attribute = VertexLayout::Buffers(
        (0..4)
            .map(|location| VertexBufferLayout {
                stride: 16,
                step: VertexStep::PerVertex,
                attributes: vec![attribute(location, 0)],
            })
            .collect(),
    );
    let one_entry_per_table = VertexLayout::Buffers(vec![
        VertexBufferLayout {
            stride: 16,
            step: VertexStep::PerVertex,
            attributes: vec![attribute(0, 0), attribute(1, 8)],
        },
        VertexBufferLayout {
            stride: 16,
            step: VertexStep::PerVertex,
            attributes: vec![attribute(2, 0), attribute(3, 8)],
        },
    ]);
    let declaration = StageBufferBinding {
        stage: RenderPipelineStage::Vertex,
        index: 2,
        access: BufferAccess::Read,
        footprint: FootprintProof::Static { max_bytes: 8 },
    };
    let register = |label: &str, layout: VertexLayout| {
        provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: stages.vertex_entry.to_owned(),
                fragment_entry: stages.fragment_entry.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: layout,
                stage_buffers: vec![declaration.clone()],
                textures: Vec::new(),
            },
            vertex: translate(&stages.air.0, RenderStage::Vertex, stages.vertex_entry),
            fragment: translate(&stages.air.1, RenderStage::Fragment, stages.fragment_entry),
            logical_digest: SemanticDigest::new(
                "reims-provider-render-rail-r9p",
                label.as_bytes().to_vec(),
            )
            .expect("the digest names a case"),
        })
    };
    let refused = register("one-entry-per-attribute", one_entry_per_attribute)
        .expect_err("the contract refuses an argument inside its own stream block");
    eprintln!(
        "registration, one entry per attribute: class={:?} slug={} fields={:?} refusal={refused:?}",
        refused.class, refused.slug, refused.fields,
    );
    assert_eq!(refused.slug, "render_pipeline_contract_invalid");
    assert!(
        format!("{refused:?}").contains("vertex stage buffer 2 shares its binding with a vertex"),
        "the refusal is the contract's own vertex-layout conflict: {refused:?}"
    );
    let registered = register("one-entry-per-table", one_entry_per_table)
        .expect("two streams leave the argument at Metal index 2 clear of them");
    eprintln!(
        "registration, one entry per table: pipeline={:?}",
        registered.pipeline_id
    );

    // The four attributes the census pair had, as two interleaved tables: the
    // positions with the first offset in the first, the second and third offsets
    // in the second. One request builder feeds both rails — the attributes and
    // the `[[buffer(2)]]` bind are the same allocations for the engine's own
    // bind rail and for the seam's list beside it — so the byte comparison below
    // is about the rails' plumbing and nothing else.
    let positions = [(-1.0_f32, -3.0_f32), (-1.0, 1.0), (3.0, 1.0)];
    let offsets = |x: f32| [(x, 0.0_f32); 3];
    let request = |first: &std::sync::Arc<Vec<u8>>,
                   second: &std::sync::Arc<Vec<u8>>,
                   tail: &std::sync::Arc<Vec<u8>>| {
        let mut req = narrow_request(MTL_FORMAT_RGBA8_UNORM);
        req.vertex_attributes = vec![
            attribute_of(0, 0, 16, first),
            attribute_of(1, 8, 16, first),
            attribute_of(2, 0, 16, second),
            attribute_of(3, 8, 16, second),
        ];
        req.storage_buffers.push(engine::StorageBufferResource {
            binding: 2,
            content: BufferContent::Bytes(std::sync::Arc::clone(tail)),
        });
        req
    };
    let first = interleaved(positions, offsets(0.125));
    let second = interleaved(offsets(0.0625), offsets(0.0625));
    let still = std::sync::Arc::new(f32x2(&[(0.0, 0.0)]));
    let moved = std::sync::Arc::new(f32x2(&[(-0.25, 0.0)]));
    let (width, _) = extent();

    let frame = |label: &str,
                 first: &std::sync::Arc<Vec<u8>>,
                 second: &std::sync::Arc<Vec<u8>>,
                 tail: &std::sync::Arc<Vec<u8>>| {
        let req = request(first, second, tail);
        let content = BufferContent::Bytes(std::sync::Arc::clone(tail));
        let binds = [staged_bind(RenderPipelineStage::Vertex, 2, &content)];
        match provider_render::submit_render(
            &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
            &req,
        ) {
            RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
            other => panic!(
                "{label}: a draw whose argument is clear of its streams leaves for the provider: \
                 {other:?}"
            ),
        }
    };

    let delivered = provider_render::provider_submissions();
    let reviewed = frame("two tables", &first, &second, &still);
    eprintln!(
        "two-table draw: provider submissions {delivered} -> {}, texel (0, 0) {:?}, \
         texel (width - 1, 0) {:?}",
        provider_render::provider_submissions(),
        texel_at(&reviewed, 0, 0),
        texel_at(&reviewed, width - 1, 0),
    );
    assert!(
        provider_render::provider_submissions() > delivered,
        "the census pair reaches the canonical provider instead of the engine"
    );
    assert_clear_texel("two tables: texel (0, 0)", texel_at(&reviewed, 0, 0));
    assert_texel_near(
        "two tables: texel (width - 1, 0)",
        texel_at(&reviewed, width - 1, 0),
        FRAGMENT_TEXEL,
    );

    // The engine's own frame for the same request: the shape is one both rails
    // execute, which is what the class gate promises and what this comparison
    // reads.
    let Some(engine) = engine_pixels("two tables", &stages, request(&first, &second, &still))
    else {
        return;
    };
    assert_frames_equal("two tables, both rails", &reviewed, &engine);

    // Every input of the shape has to reach the frame: the argument's own bytes,
    // and one attribute out of each table.
    for (label, first, second, tail) in [
        ("the argument's own bytes moved", &first, &second, &moved),
        (
            "the first table's second attribute zeroed",
            &interleaved(positions, offsets(0.0)),
            &second,
            &still,
        ),
        (
            "the second table's second attribute zeroed",
            &first,
            &interleaved(offsets(0.0625), offsets(0.0)),
            &still,
        ),
    ] {
        let moved_frame = frame(label, first, second, tail);
        assert_ne!(
            moved_frame, reviewed,
            "{label}: the bytes behind this input have to reach the vertex stage"
        );
    }

    // The rule the census read as the door is still the door: a request whose
    // attributes really do read three tables states three canonical streams, and
    // the same `[[buffer(2)]]` argument then occupies one of them.
    let third = interleaved(offsets(0.0625), offsets(0.0625));
    let mut three_tables = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    three_tables.vertex_attributes = vec![
        attribute_of(0, 0, 16, &first),
        attribute_of(1, 8, 16, &first),
        attribute_of(2, 0, 16, &second),
        attribute_of(3, 0, 16, &third),
    ];
    three_tables
        .storage_buffers
        .push(engine::StorageBufferResource {
            binding: 2,
            content: BufferContent::Bytes(std::sync::Arc::clone(&still)),
        });
    let content = BufferContent::Bytes(std::sync::Arc::clone(&still));
    let binds = [staged_bind(RenderPipelineStage::Vertex, 2, &content)];
    let band = route_count("stage_buffer_shape_vertex_layout");
    let delivered = provider_render::provider_submissions();
    match provider_render::submit_render(
        &inputs_with_binds(&stages, RenderChainRole::SoleOrTail, &binds),
        &three_tables,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            eprintln!(
                "three tables, [[buffer(2)]]: slug={} detail={} route {} -> {}",
                reason.slug(),
                reason.detail(),
                band,
                route_count("stage_buffer_shape_vertex_layout"),
            );
            assert_eq!(
                reason.slug(),
                "render_provider_out_of_class_stage_buffer_shape"
            );
            assert!(
                reason.detail().contains("[[buffer(2)]]")
                    && reason.detail().contains("3 vertex stream(s)"),
                "the sentence names the slot and the streams it lands inside: {reason}"
            );
        }
        other => panic!("a declaration inside the stream block stays on the engine: {other:?}"),
    }
    assert_eq!(
        route_count("stage_buffer_shape_vertex_layout"),
        band + 1,
        "the arm the census reads is the one this refusal charged"
    );
    assert_eq!(
        provider_render::provider_submissions(),
        delivered,
        "and the draw never reaches the provider"
    );
}

/// The extent a trace-production test draws at: the sampled fixture's own 8x4
/// surface, whose fixed sample point ([`SAMPLED_TEXEL`]) exists only there, so
/// the consumer's frame is one texel of the production on both rails.
const PRODUCTION_EXTENT: (u32, u32) = (8, 4);

/// The guest target a trace-production test renders into: the production
/// extent's own geometry and the texel order the test's attachment declares.
fn production_identity(id: u32, format: ash::vk::Format) -> engine::TargetIdentity {
    engine::TargetIdentity::Surface {
        id,
        width: PRODUCTION_EXTENT.0,
        height: PRODUCTION_EXTENT.1,
        generation: 1,
        format,
    }
}

/// The producing record of a trace production: a byte-exact `Clear` whose draw
/// covers nothing, under the request's own target identity, with the
/// withheld-readback pair the seam states for a frame that lands in a resident.
///
/// Deliberately the *seam's* own shape rather than the test-only resident arm
/// ([`resident_seed_request`]): the production this file's tests drive is the
/// one the production seam actually states, which the R20 arm answers by
/// publishing the frame.
fn production_seed_request(
    identity: &engine::TargetIdentity,
    format: u16,
    clear: [f64; 4],
) -> DrawRequest {
    let (width, height) = PRODUCTION_EXTENT;
    let mut req = request_with_streams(format, &[degenerate_stream()]);
    req.width = width;
    req.height = height;
    req.target_identity = Some(identity.clone());
    req.skip_readback = true;
    req.readback_skip_reason = ReadbackSkipReason::ResidentStore;
    req.color_attachment = Some(attachment_with_clear(format, clear));
    req
}

/// The consuming record of a trace production (R22): the reviewed sampled
/// fixture, whose one `[[texture(0)]]` bind is the GPU target `identity` rather
/// than the request's own copy.
///
/// The bind states `format` — the *view's* own texel order, which the class gate
/// compares against the surface the production stored (E-TX3's
/// `RenderTextureSourceShapeMismatch`).
fn produced_sample_request(
    stages: &Stages,
    identity: &engine::TargetIdentity,
    format: ash::vk::Format,
) -> DrawRequest {
    let (width, height) = PRODUCTION_EXTENT;
    let mut req = request_with_streams(MTL_FORMAT_RGBA8_UNORM, &position_streams());
    req.width = width;
    req.height = height;
    let declaration = stages.fragment_texture_declarations[0];
    let mut image = image_resource_in(declaration.binding, Vec::new(), (width, height), format);
    image.source = SampledSource::Target(identity.clone());
    req.sampled_images.push(image);
    req.samplers
        .push(sampled_sampler_resource(declaration.sampler_binding));
    req
}

/// The frame the production seam hands back for a seed the class published
/// (R20's held arm), in semantic RGBA8.
fn published_seed(label: &str, stages: &Stages, req: &DrawRequest) -> Vec<u8> {
    match provider_render::submit_render(&inputs_held(stages, RenderChainRole::SoleOrTail), req) {
        RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
        other => panic!("{label}: the seam's own arm publishes the frame: {other:?}"),
    }
}

/// R22: a sampled texture whose texels are a GPU target is executed by stating
/// the trace's own production of that target (E-TX3's `TextureSource::TraceView`).
///
/// Three readings, all on one Lavapipe device:
///
/// * the arm **executes**: the producing record's pass rides the consuming
///   record's own trace, the consumer samples the bytes that pass stored, and
///   its frame is byte for byte the frame the engine's own two-record chain
///   lands;
/// * the frame **follows the production**: re-producing the same target with
///   another clear moves the consumer's frame with it;
/// * a consumer whose production the rail cannot name stays on the engine —
///   see [`a_production_the_consumer_cannot_name_stays_on_the_engine_by_name`].
#[test]
fn a_sampled_gpu_target_rides_the_consuming_traces_own_production() {
    let _guard = engine_test_session();
    let producer_stages = reviewed_stages();
    let consumer_stages = sampled_stages();
    let (width, height) = PRODUCTION_EXTENT;
    let identity = production_identity(0x7b_00_40, ash::vk::Format::R8G8B8A8_UNORM);
    let green = [0.0, 1.0, 0.0, 1.0];
    let red = [1.0, 0.0, 0.0, 1.0];

    // 1. The production: the clear the degenerate draw never covers, under the
    //    guest's own target identity. The class publishes the frame (R20) and
    //    records the pass as the target's production (R22).
    let seeded = published_seed(
        "trace production",
        &producer_stages,
        &production_seed_request(&identity, MTL_FORMAT_RGBA8_UNORM, green),
    );
    assert_uniform_frame("trace production", &seeded, width, height, [0, 255, 0, 255]);

    // 2. The consumer: the sampled texture's texels are that target, and the
    //    submission reaches the provider — with no bytes of its own to carry.
    let consumer =
        || produced_sample_request(&consumer_stages, &identity, ash::vk::Format::R8G8B8A8_UNORM);
    let delivered = provider_render::provider_submissions();
    let sampled = provider_pixels("trace-produced sample", &consumer_stages, &consumer());
    assert_eq!(
        provider_render::provider_submissions(),
        delivered + 1,
        "the trace-produced shape reached the canonical provider rather than staying on the \
         engine"
    );
    assert_uniform_frame(
        "trace-produced sample (provider)",
        &sampled,
        width,
        height,
        [0, 255, 0, 255],
    );

    // 3. The engine's own chain, from the same two requests: its first record
    //    stores the resident, its second samples it — the comparison is a
    //    statement about the two rails and not about one rail twice.
    let Some(engine_seed) = engine_pixels(
        "trace production (engine)",
        &producer_stages,
        production_seed_request(&identity, MTL_FORMAT_RGBA8_UNORM, green),
    ) else {
        return;
    };
    assert!(
        engine_seed.is_empty(),
        "a resident store's readback is withheld on the engine too"
    );
    let Some(engine_frame) = engine_pixels(
        "trace-produced sample (engine)",
        &consumer_stages,
        consumer(),
    ) else {
        return;
    };
    assert_frames_equal(
        "trace-produced sample vs the engine's own chain",
        &sampled,
        &engine_frame,
    );

    // 4. The frame follows the production: the same target re-produced with
    //    another clear, and the consumer's frame moves with it — on both rails.
    let reproduced = published_seed(
        "re-produced target",
        &producer_stages,
        &production_seed_request(&identity, MTL_FORMAT_RGBA8_UNORM, red),
    );
    assert_uniform_frame(
        "re-produced target",
        &reproduced,
        width,
        height,
        [255, 0, 0, 255],
    );
    let moved = provider_pixels("re-produced sample", &consumer_stages, &consumer());
    assert_uniform_frame(
        "re-produced sample (provider)",
        &moved,
        width,
        height,
        [255, 0, 0, 255],
    );
    assert_frames_differ("the production decides the sample", &moved, &sampled);
    let _ = engine_pixels(
        "re-produced target (engine)",
        &producer_stages,
        production_seed_request(&identity, MTL_FORMAT_RGBA8_UNORM, red),
    );
    let Some(engine_moved) =
        engine_pixels("re-produced sample (engine)", &consumer_stages, consumer())
    else {
        return;
    };
    assert_frames_equal(
        "re-produced sample vs the engine's own chain",
        &moved,
        &engine_moved,
    );
}

/// R22's three refusals, each by its own name and each under its own census
/// bucket: a target no pass of this rail declared a production for, a record
/// that samples the attachment it writes (the production would have to follow
/// the read), and a declaration that restates another shape than the
/// production stored.
#[test]
fn a_production_the_consumer_cannot_name_stays_on_the_engine_by_name() {
    let _guard = engine_test_session();
    let producer_stages = reviewed_stages();
    let consumer_stages = sampled_stages();
    let answer = |label: &str, req: &DrawRequest| -> (String, String) {
        match provider_render::submit_render(
            &inputs_held(&consumer_stages, RenderChainRole::SoleOrTail),
            req,
        ) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                (reason.slug().to_owned(), reason.detail().to_owned())
            }
            other => panic!("{label}: the shape stays on the engine: {other:?}"),
        }
    };

    // 1. Undeclared: the target exists as a GPU source (the request says so),
    //    and no pass of this rail ever stated a production for it.
    let unknown = production_identity(0x7b_00_41, ash::vk::Format::R8G8B8A8_UNORM);
    let undeclared_before = route_count("render_provider_out_of_class_texture_source_undeclared");
    let (slug, detail) = answer(
        "undeclared production",
        &produced_sample_request(&consumer_stages, &unknown, ash::vk::Format::R8G8B8A8_UNORM),
    );
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(
        slug,
        "render_provider_out_of_class_texture_source_undeclared"
    );
    assert!(
        detail.contains("TextureSource::TraceView"),
        "the sentence names the arm the rail would have to state: {detail}"
    );
    assert_eq!(
        route_count("render_provider_out_of_class_texture_source_undeclared") - undeclared_before,
        1,
        "the refused shape is counted under its own name"
    );

    // 2. Order: a record that samples the target it renders into. The target has
    //    a production, so the refusal is the order and not the declaration.
    let ordered = production_identity(0x7b_00_42, ash::vk::Format::R8G8B8A8_UNORM);
    let _ = published_seed(
        "ordered production",
        &producer_stages,
        &production_seed_request(&ordered, MTL_FORMAT_RGBA8_UNORM, [0.0, 1.0, 0.0, 1.0]),
    );
    let mut ill_ordered =
        produced_sample_request(&consumer_stages, &ordered, ash::vk::Format::R8G8B8A8_UNORM);
    ill_ordered.target_identity = Some(ordered.clone());
    let order_before = route_count("render_provider_out_of_class_texture_source_order");
    let (slug, detail) = answer("self-sampled production", &ill_ordered);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_source_order");
    assert!(
        detail.contains("RenderTextureAttachmentConflict"),
        "the sentence names the contract's own refusal: {detail}"
    );
    assert_eq!(
        route_count("render_provider_out_of_class_texture_source_order") - order_before,
        1,
        "the refused shape is counted under its own name"
    );

    // 3. Shape: the production stored another texel order than the declaration
    //    restates. The wide target's own resident arm is the one the class
    //    elects it by (`resident_frames_fetchable`), so this seed is driven
    //    through the fetching caller's inputs.
    let wide = production_identity(0x7b_00_43, ash::vk::Format::R16G16B16A16_SFLOAT);
    let wide_seed = production_seed_request(&wide, MTL_FORMAT_RGBA16_FLOAT, [0.0, 1.0, 0.0, 1.0]);
    match provider_render::submit_render(
        &inputs(&producer_stages, RenderChainRole::SoleOrTail),
        &wide_seed,
    ) {
        RenderRailOutcome::ProviderCompletedResident(_) => (),
        other => panic!("a wide production's own store is the resident arm: {other:?}"),
    }
    let shape_before = route_count("render_provider_out_of_class_texture_source_shape");
    let (slug, detail) = answer(
        "mismatched production shape",
        &produced_sample_request(&consumer_stages, &wide, ash::vk::Format::R8G8B8A8_UNORM),
    );
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_source_shape");
    assert!(
        detail.contains("Rgba16Float") && detail.contains("R8G8B8A8_UNORM"),
        "the sentence names both shapes: {detail}"
    );
    assert_eq!(
        route_count("render_provider_out_of_class_texture_source_shape") - shape_before,
        1,
        "the refused shape is counted under its own name"
    );
}

/// R22: a production's **window-backed** bind is re-imported into the consuming
/// trace.
///
/// The producing record reads a registered guest window (R9e's no-copy arm);
/// the consuming record samples that record's own production. The lease the
/// producing submission imported is retired with its completion, so the
/// production's pass has to import the same window again inside the consuming
/// trace — and the reading that makes the re-import checkable is that the
/// consumer's frame is byte for byte the frame the producing record published,
/// both before and after the mapping's own bytes move.
#[test]
fn a_productions_window_backed_bind_is_re_imported_into_the_consuming_trace() {
    use reims_vgpu::backend::provider_compute::{device_epoch, host_import_alignment};

    let _guard = engine_test_session();
    let alignment = host_import_alignment().expect("the owner rail's provider answers");
    assert!(
        alignment > 0,
        "this device must advertise VK_EXT_external_memory_host for the no-copy arm"
    );
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let mut owner = AlignedHost::new(2 * page, page);
    owner.as_mut_slice()[..4].copy_from_slice(&[0, 0, 0x80, 0x3f]);
    let import = 0x9e22_u64;
    provider_owner::register(Region {
        import,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: owner.pointer as usize,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x41_0000),
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
    let producer_stages = buffer_declaring_stages("render_frag_buffer.air", "reims_buffer_frag");
    let consumer_stages = sampled_stages();
    let (width, height) = PRODUCTION_EXTENT;
    let identity = production_identity(0x7b_00_44, ash::vk::Format::R8G8B8A8_UNORM);

    let mut seed = narrow_request(MTL_FORMAT_RGBA8_UNORM);
    seed.width = width;
    seed.height = height;
    seed.target_identity = Some(identity.clone());
    seed.skip_readback = true;
    seed.readback_skip_reason = ReadbackSkipReason::ResidentStore;
    seed.storage_buffers.push(engine::StorageBufferResource {
        binding: 0,
        content: staged.clone(),
    });
    let binds = [StageBufferBind {
        stage: RenderPipelineStage::Fragment,
        index: 0,
        content: &staged,
        window: Some(window),
        landing: None,
    }];
    // The seam's own capability answer (R20): the record's frame comes back
    // through the completion rather than staying in an image this caller cannot
    // read, which is the arm the production seam states and the one a
    // byte-for-byte comparison against the consumer's frame can be made on.
    let held = RenderRailInputs {
        resident_frames_fetchable: false,
        ..inputs_with_binds(&producer_stages, RenderChainRole::SoleOrTail, &binds)
    };
    let publish = |label: &str| -> Vec<u8> {
        match provider_render::submit_render(&held, &seed) {
            RenderRailOutcome::ProviderCompleted(out) => semantic_rgba(out.bytes, out.bgra),
            other => panic!("{label}: a window-backed production is in class: {other:?}"),
        }
    };
    let consume = || {
        provider_pixels(
            "window-backed sample",
            &consumer_stages,
            &produced_sample_request(&consumer_stages, &identity, ash::vk::Format::R8G8B8A8_UNORM),
        )
    };

    // The mapping holds the fragment's red; the sampled frame has to be that
    // colour, and the consumer's frame has to be the producer's own frame.
    let red = publish("red window");
    assert_uniform_frame(
        "window-backed production",
        &red,
        width,
        height,
        [255, 0, 0, 255],
    );
    let sampled = consume();
    assert_frames_equal(
        "the consumer samples the production's own frame",
        &sampled,
        &red,
    );

    // The falsifiable half: move the mapping's bytes, re-produce, and the
    // consumer's frame follows — the re-import read the window, not a copy the
    // producing submission left behind.
    owner.as_mut_slice()[..4].copy_from_slice(&[0, 0, 0, 0]);
    let black = publish("black window");
    assert_uniform_frame(
        "window-backed production (moved)",
        &black,
        width,
        height,
        [0, 0, 0, 255],
    );
    let moved = consume();
    assert_frames_equal("the consumer samples the re-produced frame", &moved, &black);
    assert_frames_differ(
        "the production's own window decides the sample",
        &moved,
        &sampled,
    );
}

/// R28: the sample pass whose binds the owner→provider frame has to carry now
/// leaves for the provider, and the *frame* is what says so.
///
/// R22's guard kept this shape on the engine by name, because the frame's
/// render contract carried no texture declarations then
/// (`research/docs/23` §101.5: a codec kind for the compute half's declarations
/// and none for the render half's), so the decoded pass bound views no contract
/// declared and admission refused it (`UndeclaredTextureBinding`) — a decline,
/// not a fallback. E-TX4 gave the frame both halves of that statement (the
/// render contract's texture declarations and the pass block that pairs them
/// with the views), and this increment reads the frame rather than assuming
/// either answer: the class asks the wire's own capability section
/// (`declared_render_texture_support`) and the submission's frame is decoded
/// and read back.
///
/// The falsifiable half is the decoded frame: one declaration at the Metal
/// index the module states, one view under it, the view's source arm named. A
/// frame that stripped the declarations would decode `view=absent`, which is
/// the reading that used to make this shape a decline.
#[test]
fn a_sampled_pass_whose_frame_carries_the_declarations_leaves_for_the_provider() {
    use reims_vgpu::backend::provider_compute::device_epoch;
    use reims_vgpu::runtime::guest_ram::GuestRamImport;

    let _guard = engine_test_session();
    let alignment = reims_vgpu::backend::provider_compute::host_import_alignment()
        .expect("the owner rail's provider answers");
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let mut owner = AlignedHost::new(2 * page, page);
    // The reviewed full-screen triangle's own three `float2` positions, in the
    // owner's mapping: the bind is a live guest stream, not a copy.
    owner.as_mut_slice()[..24].copy_from_slice(&f32x2(&[(-1.0, -3.0), (-1.0, 1.0), (3.0, 1.0)]));
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
    let registered = RegisteredWindow {
        import: import.id(),
        base: owner.pointer as u64,
        length: page as u64,
        epoch: 1,
    };
    let content = BufferContent::GuestRuns(engine::GuestRunSource {
        runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
            owner.pointer as usize,
            2 * page as u64,
            0,
            24,
        )
        .expect("the bind's own bytes are inside the mapping")]),
        source_offset: 0,
        total_len: 24,
        row_length_texels: 0,
        pages: Some(std::sync::Arc::new(vec![GuestWindowRun {
            window_offset: 0,
            guest,
            window: Some(registered),
        }])),
        direct_image: None,
    });
    let stages = sampled_stages();
    let request = || {
        let mut request = sampled_request(&stages, sampled_texels(8, 4), (8, 4));
        request.vertex_attributes[0].content = content.clone();
        request
    };
    // The engine arm runs first: the engine's device context is created lazily
    // on its first draw, and that creation resets the owner rail — a
    // registration made before it would be dropped before the provider asked
    // for it.
    let engine = engine_pixels("R28 wire-carrying sampled pass", &stages, request())
        .expect("the engine draws this shape");
    provider_owner::register(Region {
        import: import_id,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: owner.pointer as usize,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x42_0000),
    })
    .expect("a page-aligned registration is a legal provider region");
    let wire_before = route_count("render_provider_out_of_class_texture_wire");
    let submissions = provider_render::provider_submissions();
    // The frame this submission produces is kept, so the reading below is taken
    // from the bytes that travel rather than from the values this test built.
    let (declarations, views, provider) =
        wire_textures_of_one_submission("R28 wire-carrying sampled pass", || {
            provider_pixels("R28 wire-carrying sampled pass", &stages, &request())
        });
    assert_frames_equal(
        "the sampled pass lands the engine's frame",
        &provider,
        &engine,
    );
    assert_eq!(
        route_count("render_provider_out_of_class_texture_wire") - wire_before,
        0,
        "the shape is no longer a wire refusal: the frame carries its declarations"
    );
    assert_eq!(
        provider_render::provider_submissions(),
        submissions + 1,
        "the draw reached the provider instead of falling back"
    );
    // The frame's own reading: one texture declaration, one view under it, and
    // the source arm the class stated.
    assert_eq!(declarations.len(), 1, "the frame carries the declaration");
    assert_eq!(declarations[0].metal_binding, 0);
    assert_eq!(views.len(), 1, "the frame carries the pass's own view");
    assert_eq!(views[0].metal_binding, declarations[0].metal_binding);
    assert!(
        !matches!(views[0].source, TextureSource::TraceView),
        "the sampled pass's view is a view the frame carries, not a dropped one"
    );
    eprintln!(
        "R28 wire: {} declaration(s) beside {} view(s); view source = {:?}; provider frame == \
         engine frame",
        declarations.len(),
        views.len(),
        texture_source_arm(&views[0].source),
    );
}

/// The name one decoded texture view's source arm is reported under, with the
/// lease it names when it names one (R28).
fn texture_source_arm(source: &TextureSource) -> String {
    match source {
        TextureSource::OwnedBytes(bytes) => format!("owned_bytes={}", bytes.len()),
        TextureSource::StagedLease(lease) => format!("staged_lease={}", lease.get()),
        TextureSource::BorrowedNoCopy(lease) => format!("borrowed_lease={}", lease.get()),
        TextureSource::TraceView => "trace_view".to_owned(),
    }
}

/// One submission's *decoded* frame, read for the texture declarations the
/// pipeline entry carries and the views the pass carries under them (R28).
///
/// The reading is taken from the bytes the rail produced — the capture facility
/// in [`reims_vgpu::backend::provider_wire`] — and decoded with the provider's
/// own decoder, because "the frame carries the pass's texture declarations" is
/// a statement about bytes and every other reading would be a second spelling
/// of the values the rail built.
fn wire_textures_of_one_submission(
    label: &str,
    submit: impl FnOnce() -> Vec<u8>,
) -> (Vec<TextureBindingContract>, Vec<TextureView>, Vec<u8>) {
    provider_wire::capture_submission_frames(true);
    let frame_bytes = submit();
    let frames = provider_wire::captured_submission_frames();
    provider_wire::capture_submission_frames(false);
    let decoded = frames
        .iter()
        .filter_map(|frame| provider_wire::carried_submission(frame).ok())
        .find_map(|(trace, _)| {
            let declarations = trace
                .pipelines
                .iter()
                .find_map(|pipeline| pipeline.render.as_ref())
                .map(|render| render.textures.clone())?;
            let pass = trace.passes.iter().find_map(TracePass::as_render)?;
            Some((declarations, pass.textures.clone()))
        })
        .unwrap_or_else(|| panic!("{label}: the submission's frame decodes as a sampled pass"));
    (decoded.0, decoded.1, frame_bytes)
}

// ---------------------------------------------------------------------------
// R28: the sampled texture's own window — the guest gather, declared
// ---------------------------------------------------------------------------

/// One sampled texture the *zero-copy rail* resolved: the request carries no
/// copy of the texels, the bytes live in the owner's registered mapping, and
/// the one run names the provider-shaped window the ledger derived for it.
///
/// `head` is the byte distance from the window's first byte to the texture's
/// (the run's own in-granule offset plus the source's `source_offset`), which
/// is the one fact that decides between the two arms the class can state: a
/// window whose first byte *is* the texture's leaves as the borrowed no-copy
/// lease, and one that starts earlier leaves as the owner's staged copy of the
/// extent ([`texture_window_arm`]).
fn sampled_window_source(
    base: usize,
    mapping_len: u64,
    guest: GuestRef,
    window: RegisteredWindow,
    head: u64,
    extent: u64,
) -> SampledSource {
    SampledSource::GuestRuns(
        engine::GuestRunSource {
            runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
                base,
                mapping_len,
                0,
                head + extent,
            )
            .expect("the bind's own bytes are inside the mapping")]),
            source_offset: head,
            total_len: extent,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(vec![GuestWindowRun {
                window_offset: 0,
                guest,
                window: Some(window),
            }])),
            direct_image: None,
        },
        reims_vgpu::runtime::gather_witness::GatherVouch::Fresh,
    )
}

/// R28: the sampled texture whose texels are the guest's own pages — the arm
/// census v19 read 288 times as `texture_source` — leaves for the canonical
/// provider through the owner rail's *borrowed* window.
///
/// The shape is the vertex streams' R9q arm one binding over: the bind's one
/// page run carries the window the registration ledger derived for it, the
/// window's first byte is the texture's own (so the contract's window rule —
/// a texture's tightly packed extent at the reservation's own start — states
/// exactly these bytes), and the declaration names the lease the plan imported.
///
/// Three readings, because each one alone would pass for a rail that did
/// something else:
///
/// - the frame the provider lands is the texel the shader samples *out of the
///   owner's mapping*, not a colour the request carries: the request states no
///   bytes at all for this bind;
/// - moving the mapping's read texel moves the provider's frame with it, so the
///   bytes travel from the mapping rather than from a copy taken anywhere else;
/// - the submission's own frame decodes with the declaration beside the view
///   and the view's source named `borrowed_lease`, which is the wire's statement
///   that this is the no-copy arm and not a staged one.
///
/// The engine and the provider land the same frame byte for byte on top of
/// that.
#[test]
fn a_sampled_texture_in_a_registered_window_leaves_without_a_copy() {
    use reims_vgpu::backend::provider_compute::device_epoch;
    use reims_vgpu::runtime::guest_ram::GuestRamImport;

    let _guard = engine_test_session();
    let stages = sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    let (read_x, read_y) = SAMPLED_TEXEL;
    let read_bytes = |texels: &[Vec<u8>]| {
        let texel = &texels[read_y * width as usize + read_x];
        [texel[0], texel[1], texel[2], texel[3]]
    };
    let flat =
        |texels: &[Vec<u8>]| -> Vec<u8> { texels.iter().flat_map(|texel| texel.clone()).collect() };
    let alignment = reims_vgpu::backend::provider_compute::host_import_alignment()
        .expect("the owner rail's provider answers");
    assert!(
        alignment > 0,
        "this device must advertise VK_EXT_external_memory_host for the no-copy arm"
    );
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let mut owner = AlignedHost::new(2 * page, page);
    // The texture's whole extent at the window's own first byte: the shape the
    // borrowed arm is the answer to.
    let bytes = flat(&texels);
    owner.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
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
    let registered = RegisteredWindow {
        import: import.id(),
        base: owner.pointer as u64,
        length: page as u64,
        epoch: 1,
    };
    // The request as the zero-copy rail builds it: no bytes for the bind, one
    // run over the owner's mapping and the window the ledger derived.
    // The mapping's own address, read once: the closure below must not borrow
    // the owner itself, because the falsifiable half moves its bytes.
    let base = owner.pointer as usize;
    let request = || {
        let mut request = sampled_request(&stages, texels.clone(), (width, height));
        request.sampled_images[0].source = sampled_window_source(
            base,
            2 * page as u64,
            guest.clone(),
            registered,
            0,
            bytes.len() as u64,
        );
        request
    };
    // The engine arm first: its device context is created lazily on the first
    // draw, and that creation resets the owner rail.
    let engine = engine_pixels("R28 borrowed sampled window", &stages, request())
        .expect("the engine gathers this shape");
    assert_uniform_frame(
        "R28 borrowed sampled window (engine)",
        &engine,
        width,
        height,
        read_bytes(&texels),
    );
    provider_owner::register(Region {
        import: import_id,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: owner.pointer as usize,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x44_0000),
    })
    .expect("a page-aligned registration is a legal provider region");
    let borrowed_before = route_count("render_provider_sampled_window_borrowed");
    let staged_before = route_count("render_provider_sampled_window_staged");
    let (declarations, views, provider) =
        wire_textures_of_one_submission("R28 borrowed sampled window", || {
            provider_pixels("R28 borrowed sampled window", &stages, &request())
        });
    assert_uniform_frame(
        "R28 borrowed sampled window (provider)",
        &provider,
        width,
        height,
        read_bytes(&texels),
    );
    assert_frames_equal("the two rails agree on the window", &provider, &engine);
    assert_eq!(
        route_count("render_provider_sampled_window_borrowed") - borrowed_before,
        1,
        "the arm the device took is the borrowed window"
    );
    assert_eq!(
        route_count("render_provider_sampled_window_staged") - staged_before,
        0,
        "nothing was copied for this bind"
    );
    assert_eq!(declarations.len(), 1, "the frame carries the declaration");
    assert_eq!(views.len(), 1, "the frame carries the pass's own view");
    assert!(
        matches!(views[0].source, TextureSource::BorrowedNoCopy(_)),
        "the declaration names the owner's no-copy window: {:?}",
        texture_source_arm(&views[0].source)
    );

    // The falsifiable half: move the texel the fragment samples in the owner's
    // own mapping, re-submit, and the provider's frame follows — the bytes came
    // from the mapping, not from a copy this rail took elsewhere.
    let mut moved = texels.clone();
    moved[read_y * width as usize + read_x] = vec![255, 0, 128, 255];
    owner.as_mut_slice()[..bytes.len()].copy_from_slice(&flat(&moved));
    let after = provider_pixels("R28 borrowed sampled window (moved)", &stages, &request());
    assert_uniform_frame(
        "R28 borrowed sampled window (moved)",
        &after,
        width,
        height,
        read_bytes(&moved),
    );
    assert_frames_differ("the mapping's own bytes reach the frame", &provider, &after);
    eprintln!(
        "R28 borrowed sampled window: {width}x{height} texels in one registered window, head=0; \
         provider frame == engine frame == {:?}; the read texel's move moved it; the frame's view \
         is {}",
        read_bytes(&texels),
        texture_source_arm(&views[0].source),
    );
}

/// R28: the same gather whose window starts *before* the texture is the arm the
/// contract's window rule cannot state as a borrow, and the class copies the
/// extent out of the registration instead.
///
/// E reads a lease-backed texture at the reservation's own start (the window
/// rule is "the texture's tightly packed extent at the reservation's start"), so
/// a texture whose first byte is four bytes into the granule would be read from
/// the wrong bytes by the borrowed arm. That is the same shape R18 answered for
/// a stage buffer whose view pointer the granules turn away — this is its
/// sampled sibling, and the copy is the arm the class states.
///
/// What has to be falsifiable is *which* bytes the copy read: the pattern is
/// offset by four bytes inside the window, so a copy taken from the window's
/// start would land the neighbouring texel's colour in the frame, and the
/// assertions below would name it rather than pass.
#[test]
fn a_sampled_texture_whose_window_starts_inside_the_granule_is_copied() {
    use reims_vgpu::backend::provider_compute::device_epoch;
    use reims_vgpu::runtime::guest_ram::GuestRamImport;

    /// The texture's first byte inside its window (R18's own head).
    const HEAD: u64 = 4;

    let _guard = engine_test_session();
    let stages = sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    let (read_x, read_y) = SAMPLED_TEXEL;
    let read_bytes = |texels: &[Vec<u8>]| {
        let texel = &texels[read_y * width as usize + read_x];
        [texel[0], texel[1], texel[2], texel[3]]
    };
    let flat =
        |texels: &[Vec<u8>]| -> Vec<u8> { texels.iter().flat_map(|texel| texel.clone()).collect() };
    let alignment = reims_vgpu::backend::provider_compute::host_import_alignment()
        .expect("the owner rail's provider answers");
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let mut owner = AlignedHost::new(2 * page, page);
    let bytes = flat(&texels);
    // Four bytes the texture does not own, then the extent — so a copy that
    // started at the window's own first byte would sample another colour.
    owner.as_mut_slice()[..4].copy_from_slice(&[1, 2, 3, 255]);
    owner.as_mut_slice()[HEAD as usize..HEAD as usize + bytes.len()].copy_from_slice(&bytes);
    let shifted = {
        let mut shifted = texels.clone();
        // The texel a window-start read would land: one texel earlier, which
        // every pattern here makes a different colour.
        shifted[read_y * width as usize + read_x] =
            texels[read_y * width as usize + read_x - 1].clone();
        read_bytes(&shifted)
    };
    assert_ne!(
        shifted,
        read_bytes(&texels),
        "the fixture has to make a shifted read visible"
    );
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
    let registered = RegisteredWindow {
        import: import.id(),
        base: owner.pointer as u64,
        length: page as u64,
        epoch: 1,
    };
    let request = || {
        let mut request = sampled_request(&stages, texels.clone(), (width, height));
        request.sampled_images[0].source = sampled_window_source(
            owner.pointer as usize,
            2 * page as u64,
            guest.clone(),
            registered,
            HEAD,
            bytes.len() as u64,
        );
        request
    };
    let engine = engine_pixels("R28 copied sampled window", &stages, request())
        .expect("the engine gathers this shape");
    assert_uniform_frame(
        "R28 copied sampled window (engine)",
        &engine,
        width,
        height,
        read_bytes(&texels),
    );
    provider_owner::register(Region {
        import: import_id,
        epoch: device_epoch().expect("the rail's provider epoch"),
        host_pointer: owner.pointer as usize,
        length: 2 * page as u64,
        page_size: alignment,
        gpa_base: Some(0x45_0000),
    })
    .expect("a page-aligned registration is a legal provider region");
    let staged_before = route_count("render_provider_sampled_window_staged");
    let staged_bytes_before = route_count("render_provider_sampled_window_bytes");
    let (declarations, views, provider) =
        wire_textures_of_one_submission("R28 copied sampled window", || {
            provider_pixels("R28 copied sampled window", &stages, &request())
        });
    assert_uniform_frame(
        "R28 copied sampled window (provider)",
        &provider,
        width,
        height,
        read_bytes(&texels),
    );
    assert_frames_equal("the two rails agree on the copy", &provider, &engine);
    assert_eq!(
        route_count("render_provider_sampled_window_staged") - staged_before,
        1,
        "the arm the class took is the owner's staged copy"
    );
    assert_eq!(
        route_count("render_provider_sampled_window_bytes") - staged_bytes_before,
        bytes.len() as u64,
        "the copy is exactly the texture's extent"
    );
    assert_eq!(declarations.len(), 1, "the frame carries the declaration");
    assert!(
        matches!(views[0].source, TextureSource::StagedLease(_)),
        "the declaration names the owner's staged copy: {:?}",
        texture_source_arm(&views[0].source)
    );
    eprintln!(
        "R28 copied sampled window: the extent starts {HEAD} byte(s) into the window; provider \
         frame == engine frame == {:?} (a window-start copy would have landed {shifted:?}); the \
         frame's view is {}",
        read_bytes(&texels),
        texture_source_arm(&views[0].source),
    );
}

/// R28: the answer the class's texture-wire question reads comes out of the
/// *frame*, and it is falsifiable in both directions.
///
/// The gate asks `render_texture_support`, which encodes the provider's own
/// capability snapshot, decodes it again with the provider's decoder and reads
/// the render-sampler section back out. Two readings of one snapshot below: the
/// device's own (the section is present and holds the two 8-bit orders) and the
/// same snapshot with the section's bits cleared, which the codec keeps as the
/// shorter legacy frame and the decoder reads as the section's defaults —
/// `false`/`0`/empty, the answer that keeps a sampled pass on the engine.
#[test]
fn the_render_texture_capability_the_class_reads_comes_out_of_the_frame() {
    use metal_api_core::provider::TextureFormat;

    let _guard = engine_test_session();
    let executor = VulkanExecutor::new().expect("the acceptance environment has a Vulkan device");
    let provider =
        VulkanComputeProvider::with_executor(executor).expect("the canonical provider builds");
    let epoch = provider.device_epoch();
    let declared = provider_wire::render_texture_support(epoch, &provider.capabilities())
        .expect("the capability frame round-trips");
    assert!(
        declared.supported,
        "the acceptance environment's provider declares render texture sampling"
    );
    assert!(declared.maximum >= 1, "and a binding it admits");
    assert!(declared.formats.contains(&TextureFormat::Rgba8Unorm));
    assert!(declared.formats.contains(&TextureFormat::Bgra8Unorm));

    let mut refused = provider.capabilities();
    refused.supports_render_texture_sampling = false;
    refused.max_render_textures = 0;
    refused.supported_render_texture_formats.clear();
    let silent = provider_wire::render_texture_support(epoch, &refused)
        .expect("the shorter frame still round-trips");
    assert!(
        !silent.supported && silent.maximum == 0 && silent.formats.is_empty(),
        "a snapshot without the section reads as the refusal: {silent:?}"
    );
    eprintln!(
        "R28 wire capability: supported={} max={} formats={:?}; the same snapshot without the \
         section reads supported={} max={} formats={:?}",
        declared.supported,
        declared.maximum,
        declared.formats,
        silent.supported,
        silent.maximum,
        silent.formats,
    );
}

// ---------------------------------------------------------------------------
// R24: the frame the registry holds, carried into a sampled bind
// ---------------------------------------------------------------------------

/// The frame the production seam reads out of the engine's registry for a
/// sampled GPU target (R24): `read_target`'s own-bytes sibling, so the bytes
/// are the image's own — a resident the ordinary readback would *quantize* is
/// refused instead of being mistaken for the image — and no channel exchange
/// happens here, because the bind's own view format names the order the
/// declaration states (E-TX1, `research/docs/23` §107).
fn sampled_target_source(identity: &engine::TargetIdentity) -> Vec<u8> {
    engine::read_target_four_byte_color(identity)
        .expect("the sampled target's own bytes are readable")
        .expect("four-byte colour")
}

/// One submission carrying the caller's sampled-target frames, answered by the
/// provider (R24).
fn sampled_frame_submission(
    label: &str,
    stages: &Stages,
    frames: &[provider_render::SampledTargetFrame<'_>],
    req: &DrawRequest,
) -> Vec<u8> {
    let delivered = provider_render::provider_submissions();
    match provider_render::submit_render(
        &inputs_held_with_sampled_frames(stages, RenderChainRole::SoleOrTail, frames),
        req,
    ) {
        RenderRailOutcome::ProviderCompleted(out) => {
            assert_eq!(
                provider_render::provider_submissions(),
                delivered + 1,
                "{label}: the record reached the provider rather than the engine"
            );
            semantic_rgba(out.bytes, out.bgra)
        }
        other => {
            panic!("{label}: the caller's frame is the arm that answers this shape: {other:?}")
        }
    }
}

/// One submission carrying the caller's sampled-target frames that the class
/// answers by name (R24), returning the slug and the sentence.
fn sampled_frame_refusal(
    label: &str,
    stages: &Stages,
    frames: &[provider_render::SampledTargetFrame<'_>],
    req: &DrawRequest,
) -> (String, String) {
    let delivered = provider_render::provider_submissions();
    match provider_render::submit_render(
        &inputs_held_with_sampled_frames(stages, RenderChainRole::SoleOrTail, frames),
        req,
    ) {
        RenderRailOutcome::NotInNarrowClass(reason) => {
            assert_eq!(
                provider_render::provider_submissions(),
                delivered,
                "{label}: the refused shape never reaches the provider"
            );
            (reason.slug().to_owned(), reason.detail().to_owned())
        }
        other => panic!("{label}: the shape stays on the engine: {other:?}"),
    }
}

/// R24: a sampled GPU target the rail has no production for is answered from
/// the frame the caller read out of the registry that holds it.
///
/// Census v18's third refusal is this shape — 893 records whose sampled bind
/// resolved to a resident no pass of this rail recorded a production for
/// (`..._texture_source_undeclared`, 10.8 % of the boot's refusals) — and its
/// bytes are not a mystery: the resolution that produced the arm was a resident
/// bind, so the caller that owns that registry can read the image out and hand
/// it over, and the declaration states the request's own copy, which is the arm
/// every pre-R22 sampled texture takes.
///
/// Four readings, on one Lavapipe device, in both 8-bit byte orders:
///
/// * the arm **executes**: the record reaches the provider instead of the
///   engine, and its frame is byte for byte the engine's own answer from the
///   same registry image;
/// * the frame travels in the **image's** order and is read through the
///   **bind's** name, so the bgra8 case (whose red is stored `[0, 0, 255, 255]`)
///   would land the other colour if the frame were handed over in the other
///   order;
/// * the frame **follows the target**: re-seeding the registry image moves the
///   provider's frame with it, on both rails;
/// * the arm has its **own counter** (`render_provider_sampled_target_frames`),
///   so the population it answers is a number rather than a silence.
#[test]
fn an_undeclared_targets_frame_carries_the_sample_into_the_provider() {
    let _guard = engine_test_session();
    let producer_stages = reviewed_stages();
    let consumer_stages = sampled_stages();
    let (width, height) = PRODUCTION_EXTENT;
    let red = [1.0, 0.0, 0.0, 1.0];
    let green = [0.0, 1.0, 0.0, 1.0];

    for (label, seed_format, vk_format, id) in [
        (
            "rgba8 target",
            MTL_FORMAT_RGBA8_UNORM,
            ash::vk::Format::R8G8B8A8_UNORM,
            0x7c_00_40,
        ),
        (
            "bgra8 target",
            MTL_FORMAT_BGRA8_UNORM,
            ash::vk::Format::B8G8R8A8_UNORM,
            0x7c_00_41,
        ),
    ] {
        let identity = production_identity(id, vk_format);
        // 1. The registry image. It is drawn by the *engine*, so no pass of
        //    this rail ever stated a production for it — which is exactly the
        //    census's shape: the head of a packet the class refused, kept in
        //    the engine's own registry.
        let Some(seed) = engine_pixels(
            label,
            &producer_stages,
            production_seed_request(&identity, seed_format, red),
        ) else {
            return;
        };
        assert!(
            seed.is_empty(),
            "a resident store's readback is withheld on the engine too"
        );
        let frame = sampled_target_source(&identity);
        let frames = [provider_render::SampledTargetFrame {
            identity: identity.clone(),
            bytes: &frame,
        }];
        let consumer = || produced_sample_request(&consumer_stages, &identity, vk_format);

        // 2. The consumer reaches the provider and lands the image's colour.
        let answered = route_count("render_provider_sampled_target_frames");
        let sampled = sampled_frame_submission(label, &consumer_stages, &frames, &consumer());
        assert_uniform_frame(
            &format!("{label} (provider)"),
            &sampled,
            width,
            height,
            [255, 0, 0, 255],
        );
        assert_eq!(
            route_count("render_provider_sampled_target_frames") - answered,
            1,
            "the arm that answered is counted under its own name"
        );
        eprintln!(
            "{label}: registry frame texel (0, 0) {:?}, provider frame texel (0, 0) {:?}, \
             render_provider_sampled_target_frames {answered} -> {}, submissions reached the \
             provider",
            texel_at(&frame, 0, 0),
            texel_at(&sampled, 0, 0),
            route_count("render_provider_sampled_target_frames"),
        );

        // 3. The engine's own answer from the same registry image: the
        //    comparison is a statement about the two rails.
        let Some(engine_frame) = engine_pixels(label, &consumer_stages, consumer()) else {
            return;
        };
        assert_uniform_frame(
            &format!("{label} (engine)"),
            &engine_frame,
            width,
            height,
            [255, 0, 0, 255],
        );
        assert_frames_equal(
            &format!("{label}: the carried frame vs the engine's own answer"),
            &sampled,
            &engine_frame,
        );
        eprintln!(
            "{label}: engine frame texel (0, 0) {:?} — equal to the provider's, byte for byte",
            texel_at(&engine_frame, 0, 0)
        );

        // 4. The frame follows the image: the same target re-seeded on the
        //    engine, and both rails move with it.
        let _ = engine_pixels(
            label,
            &producer_stages,
            production_seed_request(&identity, seed_format, green),
        );
        let moved = sampled_target_source(&identity);
        let moved_frames = [provider_render::SampledTargetFrame {
            identity: identity.clone(),
            bytes: &moved,
        }];
        let sampled_moved = sampled_frame_submission(
            &format!("{label}, re-seeded"),
            &consumer_stages,
            &moved_frames,
            &consumer(),
        );
        assert_uniform_frame(
            &format!("{label}: re-seeded (provider)"),
            &sampled_moved,
            width,
            height,
            [0, 255, 0, 255],
        );
        assert_frames_differ(
            "the registry image decides the sample",
            &sampled_moved,
            &sampled,
        );
        let Some(engine_moved) = engine_pixels(label, &consumer_stages, consumer()) else {
            return;
        };
        assert_frames_equal(
            &format!("{label}: re-seeded, the engine's own answer"),
            &sampled_moved,
            &engine_moved,
        );
        eprintln!(
            "{label}: re-seeded registry frame texel (0, 0) {:?}, provider frame texel (0, 0) \
             {:?}, engine equal, and it differs from the first frame",
            texel_at(&moved, 0, 0),
            texel_at(&sampled_moved, 0, 0),
        );
    }
}

/// R24's refusals, each by its own name: the arm's absence (the record keeps
/// R22's `undeclared`), a frame that is not the declaration's own extent (its
/// own slug), and the record that samples the attachment it writes — which
/// keeps `order` even with a correct frame in hand, because that read is live
/// and no canonical arm can state it.
#[test]
fn a_targets_frame_keeps_the_class_own_refusals_by_name() {
    let _guard = engine_test_session();
    let producer_stages = reviewed_stages();
    let consumer_stages = sampled_stages();
    let identity = production_identity(0x7c_00_42, ash::vk::Format::R8G8B8A8_UNORM);
    let Some(seed) = engine_pixels(
        "refusal shapes",
        &producer_stages,
        production_seed_request(&identity, MTL_FORMAT_RGBA8_UNORM, [1.0, 0.0, 0.0, 1.0]),
    ) else {
        return;
    };
    assert!(
        seed.is_empty(),
        "the resident's readback is withheld on the engine too"
    );
    let frame = sampled_target_source(&identity);
    let frames = [provider_render::SampledTargetFrame {
        identity: identity.clone(),
        bytes: &frame,
    }];
    let consumer =
        || produced_sample_request(&consumer_stages, &identity, ash::vk::Format::R8G8B8A8_UNORM);

    // 1. No frame: the declaration this rail cannot state, unchanged.
    let before = route_count("render_provider_out_of_class_texture_source_undeclared");
    let (slug, detail) = sampled_frame_refusal("no frame", &consumer_stages, &[], &consumer());
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(
        slug,
        "render_provider_out_of_class_texture_source_undeclared"
    );
    assert!(
        detail.contains("TextureSource::TraceView") && detail.contains("hands no frame over"),
        "the sentence names both arms that could have carried the bytes: {detail}"
    );
    assert_eq!(
        route_count("render_provider_out_of_class_texture_source_undeclared") - before,
        1,
        "the refused shape is counted under its own name"
    );

    // 2. A frame that is not the declaration's extent is a caller wiring bug
    //    under its own name, not an upload the contract would refuse.
    let short = vec![0u8; 4];
    let short_frames = [provider_render::SampledTargetFrame {
        identity: identity.clone(),
        bytes: &short,
    }];
    let before = route_count("render_provider_out_of_class_texture_source_frame_shape");
    let (slug, detail) =
        sampled_frame_refusal("short frame", &consumer_stages, &short_frames, &consumer());
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(
        slug,
        "render_provider_out_of_class_texture_source_frame_shape"
    );
    assert!(
        detail.contains("caller's 4 byte(s) frame")
            && detail.contains("its 8x4 R8G8B8A8_UNORM surface"),
        "the sentence names the frame it was handed and the surface it has to fill: {detail}"
    );
    assert_eq!(
        route_count("render_provider_out_of_class_texture_source_frame_shape") - before,
        1,
        "the refused shape is counted under its own name"
    );

    // 3. The record that samples its own attachment keeps `order`: its read is
    //    the live frame the pass writes, which no canonical arm states — a
    //    frame in hand does not change that.
    let mut self_sampling = consumer();
    self_sampling.target_identity = Some(identity.clone());
    let before = route_count("render_provider_out_of_class_texture_source_order");
    let (slug, detail) =
        sampled_frame_refusal("self-sampling", &consumer_stages, &frames, &self_sampling);
    eprintln!("door: {slug}\n  {detail}");
    assert_eq!(slug, "render_provider_out_of_class_texture_source_order");
    assert!(
        detail.contains("RenderTextureAttachmentConflict")
            && detail.contains("fallback arm, not its answer"),
        "the sentence names the contract's own refusal and why a copy is not the answer: \
         {detail}"
    );
    assert_eq!(
        route_count("render_provider_out_of_class_texture_source_order") - before,
        1,
        "the refused shape is counted under its own name"
    );
}

/// R24 composes with R22: a record whose *sampled* target the caller carried in
/// is itself restatable as a production.
///
/// The reach of the two arms is what this reads. A record that samples a
/// registry image and keeps its own frame under a guest target is exactly the
/// head of the census's chains (`skip=resident store=1`), and its pass
/// descriptor carries the carried frame's bytes rather than a lease — so it is
/// as restatable as any request-carried copy, and the *next* consumer of that
/// target is answered by R22's trace-produced arm with no host round trip at
/// all.
#[test]
fn a_carried_frames_own_record_is_restatable_as_a_production() {
    let _guard = engine_test_session();
    let producer_stages = reviewed_stages();
    let consumer_stages = sampled_stages();
    let (width, height) = PRODUCTION_EXTENT;
    let source = production_identity(0x7c_00_43, ash::vk::Format::R8G8B8A8_UNORM);
    let target = production_identity(0x7c_00_44, ash::vk::Format::R8G8B8A8_UNORM);

    // 1. The engine's registry holds the sampled target's image.
    let Some(seed) = engine_pixels(
        "carried frame source",
        &producer_stages,
        production_seed_request(&source, MTL_FORMAT_RGBA8_UNORM, [1.0, 0.0, 0.0, 1.0]),
    ) else {
        return;
    };
    assert!(
        seed.is_empty(),
        "the resident's readback is withheld on the engine too"
    );

    // 2. The producer: a sampled bind on that image (carried in as its frame)
    //    whose own frame stays under `target` — the seam's withheld-readback
    //    shape, which records `target`'s production from the same submission.
    let producer_request = || {
        let mut req =
            produced_sample_request(&consumer_stages, &source, ash::vk::Format::R8G8B8A8_UNORM);
        req.target_identity = Some(target.clone());
        req.skip_readback = true;
        req.readback_skip_reason = ReadbackSkipReason::ResidentStore;
        req
    };
    let frame = sampled_target_source(&source);
    let frames = [provider_render::SampledTargetFrame {
        identity: source.clone(),
        bytes: &frame,
    }];
    let published = sampled_frame_submission(
        "carried frame production",
        &consumer_stages,
        &frames,
        &producer_request(),
    );
    assert_uniform_frame(
        "the carried-frame record's own frame",
        &published,
        width,
        height,
        [255, 0, 0, 255],
    );
    eprintln!(
        "carried frame record: its own published frame texel (0, 0) {:?} (the sampled frame's \
         colour), so the submission that recorded `target`'s production is the one that read the \
         registry image",
        texel_at(&published, 0, 0),
    );

    // 3. The next record samples `target` through R22's arm: no frame is
    //    handed over, and the producing pass rides the consumer's own trace.
    let delivered = provider_render::provider_submissions();
    let sampled = provider_pixels(
        "the carried-frame record's consumer",
        &consumer_stages,
        &produced_sample_request(&consumer_stages, &target, ash::vk::Format::R8G8B8A8_UNORM),
    );
    assert_eq!(
        provider_render::provider_submissions(),
        delivered + 1,
        "the consumer of a recorded production reaches the provider"
    );
    assert_uniform_frame(
        "the production's consumer (provider)",
        &sampled,
        width,
        height,
        [255, 0, 0, 255],
    );

    // 4. Both rails, from the same two records.
    let Some(engine_published) = engine_pixels(
        "carried frame production",
        &consumer_stages,
        producer_request(),
    ) else {
        return;
    };
    assert!(
        engine_published.is_empty(),
        "the producer's own readback is withheld on the engine too"
    );
    let Some(engine_sampled) = engine_pixels(
        "the carried-frame record's consumer",
        &consumer_stages,
        produced_sample_request(&consumer_stages, &target, ash::vk::Format::R8G8B8A8_UNORM),
    ) else {
        return;
    };
    assert_frames_equal(
        "the carried-frame chain vs the engine's own chain",
        &sampled,
        &engine_sampled,
    );
    eprintln!(
        "the production's consumer: provider frame texel (0, 0) {:?}, engine frame texel (0, 0) \
         {:?}, no frame handed to the consumer — R22's arm restated the producing pass",
        texel_at(&sampled, 0, 0),
        texel_at(&engine_sampled, 0, 0),
    );
}
/// R28: the gathers the class *cannot* state as one registered window keep the
/// engine under the same bucket, one sentence naming the fact that refused.
///
/// The window the contract states for a texture is its tightly packed extent at
/// the reservation's own start, so every gather outside that shape is an exit —
/// and each one below is a different fact, not one looser rule: no registration
/// under the current epoch, padded guest rows (a lease window carries no
/// stride), more than one stretch (no single host range is the bind's), a span
/// that is not the texture's extent, and a window that does not reach the
/// extent from its first byte. The class is pure here: no provider is asked and
/// no lease is minted for a shape that stays on the engine.
#[test]
fn the_gathers_outside_one_registered_window_stay_on_the_engine_by_name() {
    use reims_vgpu::runtime::guest_ram::GuestRamImport;

    let _guard = engine_test_session();
    let stages = sampled_stages();
    let (width, height) = (8u32, 4u32);
    let texels = sampled_texels(width, height);
    let extent = u64::from(width) * u64::from(height) * 4;
    let alignment = reims_vgpu::backend::provider_compute::host_import_alignment()
        .expect("the owner rail's provider answers");
    let page = usize::try_from(alignment).expect("the alignment fits usize");
    let mut owner = AlignedHost::new(2 * page, page);
    let import = std::sync::Arc::new(
        GuestRamImport::new_host_allocation(owner.pointer as usize, 2 * page as u64, alignment)
            .expect("a page-aligned synthetic host allocation"),
    );
    let anchor = import
        .slice(0, page as u64)
        .expect("the first granule is inside the import");
    let guest = GuestRef::new(std::sync::Arc::clone(&import), anchor)
        .expect("the slice came from this import");
    let base = owner.pointer as usize;
    let registered = RegisteredWindow {
        import: import.id(),
        base: base as u64,
        length: page as u64,
        epoch: 1,
    };
    // One run over the owner's mapping, `len` bytes long, and the page runs the
    // gather carries: the two numbers a shape below varies.
    let gather = |spec: (u64, u32, Vec<GuestWindowRun>)| {
        let (total_len, row_length_texels, pages) = spec;
        SampledSource::GuestRuns(
            engine::GuestRunSource {
                runs: std::sync::Arc::new(vec![engine::GuestRun::in_mapping(
                    base,
                    2 * page as u64,
                    0,
                    extent,
                )
                .expect("the bind's own bytes are inside the mapping")]),
                source_offset: 0,
                total_len,
                row_length_texels,
                pages: Some(std::sync::Arc::new(pages)),
                direct_image: None,
            },
            reims_vgpu::runtime::gather_witness::GatherVouch::Fresh,
        )
    };
    let run = |guest: GuestRef, window: Option<RegisteredWindow>| GuestWindowRun {
        window_offset: 0,
        guest,
        window,
    };
    let short_window = RegisteredWindow {
        length: extent / 2,
        ..registered
    };
    let answer = |label: &str, source: SampledSource| -> (String, String) {
        let mut request = sampled_request(&stages, texels.clone(), (width, height));
        request.sampled_images[0].source = source;
        match provider_render::submit_render(
            &inputs(&stages, RenderChainRole::SoleOrTail),
            &request,
        ) {
            RenderRailOutcome::NotInNarrowClass(reason) => {
                (reason.slug().to_owned(), reason.detail().to_owned())
            }
            other => panic!("{label}: the shape is out of class: {other:?}"),
        }
    };
    let deliveries = provider_render::provider_submissions();

    // No registration under the current epoch: the one run carries no window.
    let unregistered = answer(
        "unregistered gathered texture",
        gather((extent, 0, vec![run(guest.clone(), None)])),
    );
    eprintln!("door: {}\n  {}", unregistered.0, unregistered.1);
    assert_eq!(
        unregistered.0,
        "render_provider_out_of_class_texture_source"
    );
    assert!(
        unregistered.1.contains("no registered window"),
        "the sentence names the fact that met it: {}",
        unregistered.1
    );

    // Padded rows: a lease window carries a tightly packed extent, not a
    // stride, so the stride is a fact this class cannot state.
    let padded = answer(
        "padded-row gathered texture",
        gather((extent, width, vec![run(guest.clone(), Some(registered))])),
    );
    eprintln!("door: {}\n  {}", padded.0, padded.1);
    assert_eq!(padded.0, "render_provider_out_of_class_texture_source");
    assert!(
        padded.1.contains("rows are padded") && padded.1.contains("row_length_texels`=8"),
        "the sentence names the stride: {}",
        padded.1
    );

    // Two stretches: no single host range is the bind's bytes.
    let scattered = answer(
        "scattered gathered texture",
        gather((
            extent,
            0,
            vec![
                run(guest.clone(), Some(registered)),
                GuestWindowRun {
                    window_offset: extent / 2,
                    ..run(guest.clone(), Some(registered))
                },
            ],
        )),
    );
    eprintln!("door: {}\n  {}", scattered.0, scattered.1);
    assert_eq!(scattered.0, "render_provider_out_of_class_texture_source");
    assert!(
        scattered.1.contains("scattered over 2 stretch(es)"),
        "the sentence names the scatter: {}",
        scattered.1
    );

    // A span that is not the texture's extent: the lease reads exactly the
    // extent, so a short window is a shape this class does not guess at.
    let short_span = answer(
        "short-span gathered texture",
        gather((extent - 4, 0, vec![run(guest.clone(), Some(registered))])),
    );
    eprintln!("door: {}\n  {}", short_span.0, short_span.1);
    assert_eq!(short_span.0, "render_provider_out_of_class_texture_source");
    assert!(
        short_span.1.contains("span is 124 byte(s) for a 128 byte"),
        "the sentence names both numbers: {}",
        short_span.1
    );

    // A window that does not reach the extent from the texture's first byte.
    let short_window_answer = answer(
        "short-window gathered texture",
        gather((extent, 0, vec![run(guest.clone(), Some(short_window))])),
    );
    eprintln!(
        "door: {}\n  {}",
        short_window_answer.0, short_window_answer.1
    );
    assert_eq!(
        short_window_answer.0,
        "render_provider_out_of_class_texture_source"
    );
    assert!(
        short_window_answer
            .1
            .contains("starts 0 byte(s) into a 64 byte window"),
        "the sentence names the window and the extent: {}",
        short_window_answer.1
    );

    // The class is pure for every one of them: no provider, no lease.
    assert_eq!(
        provider_render::provider_submissions(),
        deliveries,
        "a gather outside one registered window never reaches the provider"
    );
}
