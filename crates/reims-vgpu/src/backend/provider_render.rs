//! The canonical-provider render rail behind `feature = "provider-render"`.
//!
//! This is the render sibling of [`super::provider_compute`]: a narrow,
//! reviewed offscreen class leaves the self-contained Vulkan engine and is
//! submitted through the canonical contract ([`metal_api_core`]) and its
//! Vulkan implementation ([`metal_api_vulkan`]). The runtime seam is
//! `runtime::draw::vulkan::try_metal2vulkan_draw`, which routes here only while
//! the feature is on — and only for a request whose *whole* shape is the class
//! below. A default build never links or calls either provider crate.
//!
//! # The admitted class
//!
//! The smallest falsifiable render shape, chosen in
//! `research/docs/26` §3 against the increments the canonical render rail
//! already executes:
//!
//! - **one colour attachment** at an admitted 8-bit format (`Rgba8Unorm` /
//!   `Bgra8Unorm`), loaded by a `Clear` whose four components are byte-exact in
//!   the attachment's own 8-bit encoding, and stored. A clear that is not
//!   byte-exact is a *different* colour on the two rails — the engine hands
//!   Vulkan a `float32` clear and the driver rounds it to UNORM, while the
//!   canonical pass states the byte value — so such a request keeps the engine
//!   instead of being compared at a tolerance;
//! - **one indexed draw**, `instance_count == 1`, `base_vertex == 0`, triangle
//!   list, single-sample;
//! - **zero or one vertex stream** and exactly one index stream, each carried
//!   into the trace as trace-owned bytes. The engine numbers one Vulkan binding
//!   per attribute *location*, so the one-stream shape is one attribute at
//!   location 0 — the shape the reviewed provider fixtures exercise;
//! - the pipeline pair is the request's own translated stages: the two SPIR-V
//!   modules reims' pipeline resolution produced, registered through the
//!   canonical *translated* registration gate
//!   (`register_translated_render_pipeline`), which checks each stage's
//!   reflection against the contract field by field;
//! - no depth, stencil, MSAA, MRT, resolve, present, blend, colour write mask,
//!   viewport/scissor override, occlusion query, sampled image, sampler or
//!   storage buffer — and every one of those is a *reason*, not a silent
//!   downgrade;
//! - no resident, deferred or chained target: the request is the pooled,
//!   offscreen, CPU-readback shape (`target_identity == None`,
//!   `skip_readback == false`, no seed and no mapper backing), which is what
//!   makes the provider's whole-attachment readback the frame the store route
//!   then lands in guest memory.
//!
//! Anything outside the class returns [`RenderRailOutcome::NotInNarrowClass`]
//! and the caller runs the self-contained engine unchanged — the feature only
//! narrows which submissions change rail. An in-class submission the provider
//! refuses returns [`RenderRailOutcome::ProviderDeclined`] and the caller
//! declines the draw: fail-closed, never silently re-run on the engine.
//!
//! # The declaring pass, and why the class carries one
//!
//! The frozen render contract resolves every colour attachment against a
//! *buffer view the trace declares*, and only a compute pass declares one
//! (`metal-api-core`'s `validate_serial_buffer_reuse` →
//! `AttachmentViewUnknown`). A render-only trace is therefore refused before
//! it can run, and the milestone's own trace carries a declaring compute pass
//! for exactly this reason (`crates/metal-api-vulkan/tests/render_e2e.rs`).
//! This rail does the same with [`RENDER_DECLARE_SOURCE`]: one thread that
//! reads one word of the attachment view. The read is what binds the
//! declaration, and a read cannot race the store the render pass then performs
//! over the same bytes (core admission refuses a compute pass that *writes* a
//! view an attachment stores into).
//!
//! # What this costs, and the bound that pays for it
//!
//! The declaring view has to be backed by bytes the trace states, so the rail
//! hands it `BufferSource::OwnedBytes` of the attachment's own packed extent —
//! zeros, since a `Clear` load reads none of it. That is a real per-submission
//! copy, and it is the one cost this rail cannot avoid while the contract only
//! lets an attachment land through a byte-carrying view. It is why the class
//! states a bound ([`MAX_DECLARED_ATTACHMENT_BYTES`]) instead of admitting any
//! extent: a class that copies a whole 1080p frame per draw to declare an
//! attachment would be a pessimization wearing a feature flag. An attachment
//! above the bound keeps the engine, and the byte-less declaration the bound
//! stands in for is the named follow-up on the emulator side.
//!
//! # The lease channel, and the gap this increment reports
//!
//! `research/docs/26` §3 planned to carry the vertex and index streams through
//! the owner rail's lease channel (the one [`super::provider_owner`] gives the
//! compute rail). The canonical render rail does **not** accept leases for
//! render inputs yet: `render.rs::resolve_vertex_streams`, `decode_indices` and
//! the encoder's `create_vertex_inputs` each admit `BufferSource::OwnedBytes`
//! only, with "the first vertex-input increment executes trace-owned bytes
//! only" as the refusal text. The streams therefore travel as trace-owned
//! bytes here too — the same bytes reims would have staged — and the
//! lease/window arm for render inputs is an emulator-side increment this
//! module cannot make from here (the rail is not allowed to change
//! `metal-api-emulator`). The report beside this increment names the exact
//! call sites, and the owner rail stays where it is: shared with the compute
//! rail, untouched by this one.
//!
//! # Error mapping
//!
//! Every refusal the provider boundary returns is mapped onto this rail's own
//! typed decline ([`ProviderRenderDecline`]) with one slug per normalized
//! `ProviderErrorClass`, the provider's own slug and detail text in `detail`,
//! and `step` naming the provider call that answered. `DeviceLost` runs the
//! contract's teardown over the owner ledger exactly as the compute rail does
//! ([`provider_owner::teardown_device_lost`]) and reports the census of what it
//! released. A rail whose provider is not usable refuses in-class work
//! fail-closed rather than re-running it on the engine.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, IndexBufferBinding,
    IndexFormat, LoadOp, OperationId, RenderAttachment, RenderPassDescriptor,
    RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp, TracePass,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, VertexStep, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::Device;
use metal_api_vulkan::{RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage};

use super::provider_compute::{
    provider_error_detail, rail, refusal_decline, refuse_unhealthy, ProviderRefusalClass,
};
use super::provider_owner::{self, DeviceLossTeardown};
use super::vulkan::engine::types::{
    BufferContent, ColorClearValue, DrawRequest, VertexStepFunction,
};
use crate::observe::{decline_display, Decline};

/// The declaring kernel this rail ships (`crates/reims-vgpu/src/backend/render_declare.ll`).
///
/// Owned, not derived from a third-party metallib; the file states what it is
/// for and why a read is the whole body.
const RENDER_DECLARE_SOURCE: &str = include_str!("render_declare.ll");

/// Entry point [`RENDER_DECLARE_SOURCE`] declares.
const RENDER_DECLARE_ENTRY: &str = "reims_declare";

/// The attachment window this rail admits, one number per axis.
///
/// Two facts set it, and both are the canonical rail's rather than this one's:
///
/// * the canonical render rail *declares* this window — `metal-api-vulkan`'s
///   `MAX_ATTACHMENT_DIMENSION`, the size its reviewed fixtures prove — and
///   refuses a larger attachment at admission
///   (`attachment_dimension_limit`). A class gate that did not know the window
///   would hand the provider a shape it always refuses, which is a declined
///   draw where the engine would have drawn it: the wrong answer, since the
///   rail never claimed to execute that size;
/// * the declaring pass has to hand the attachment view trace-owned bytes (see
///   the module docs), which are copied once per submission. At this window
///   that copy is at most 4×4 texels, so the declaration cannot be a
///   pessimization on the small offscreen targets the increment is reviewed
///   against.
///
/// Widening the window is the emulator's change to make first; this constant
/// and the rail's own case move with it, and admission's typed refusal stays
/// the backstop for the interval where the two disagree.
pub const MAX_ATTACHMENT_DIMENSION: u32 = 4;

/// The allocation the colour attachment's declaring view lives in.
///
/// Chosen far above the provider's own allocation counter so a render
/// submission cannot alias an allocation the owner rail minted for another
/// binding in the same trace; the value itself is private to this rail and
/// never leaves it.
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(0x7265_6e64_6572_0001);

/// The attachment view's identity, and the only identity the provider's
/// writeback channel can name for it.
const ATTACHMENT_VIEW: ViewId = ViewId::new(1);

/// The view identities of the render inputs, in declaration order: the vertex
/// stream (when the class carries one) then the index stream.
const FIRST_INPUT_VIEW: u64 = 2;

/// What one render submission needs to leave this rail: the two stage
/// modules' AIR, the entries the translation reports for them, and whether this
/// record is the one that owes the guest writeback.
///
/// The AIR travels, rather than the SPIR-V, because the canonical registration
/// this rail uses is its *translated* one: the emulator translates the module
/// itself and checks the reflection it produced against the contract field by
/// field (`metal-api-vulkan`'s `register_translated_render_pipeline`). That is
/// the render half of the same hand-off the compute rail performs with its
/// kernel AIR — reims' own MTLB carve, the canonical translator's compilation.
pub struct RenderRailInputs<'a> {
    pub vertex_air: &'a [u8],
    pub fragment_air: &'a [u8],
    /// The AIR function name each stage's translation reports
    /// (`CachedShader::reflection.entry_point`), which is what the contract's
    /// entry names are checked against. `None` is a translation that reported
    /// none — a shape whose entry the contract could not name, so the class
    /// keeps it on the engine.
    pub vertex_entry: Option<&'a str>,
    pub fragment_entry: Option<&'a str>,
    /// Whether the record this request belongs to owns the guest writeback.
    /// A record that does not (a chain intermediate) is outside the class:
    /// its frame is a seed for the next record, which is a rail this increment
    /// does not execute.
    pub writeback_guest: bool,
}

/// What one completed narrow-class submission returns.
#[derive(Debug)]
pub struct RenderRailOutput {
    /// The attachment's whole packed extent, in the attachment's own physical
    /// order — the order `bgra` names.
    pub bytes: Vec<u8>,
    /// Whether those bytes are guest scanout order (BGRA), which is what the
    /// store route needs to know before it can land them.
    pub bgra: bool,
}

/// Why one reims render request did not leave this rail for the engine.
#[derive(Debug)]
pub enum RenderRailOutcome {
    /// The canonical provider executed the pass.
    ProviderCompleted(RenderRailOutput),
    /// Outside the narrow admitted class. The caller must run the
    /// self-contained engine, exactly as a build without the feature would.
    NotInNarrowClass(&'static str),
    /// In-class, but the canonical provider refused. The caller must decline
    /// the draw rather than fall back to another rail.
    ProviderDeclined(ProviderRenderDecline),
}

/// A typed refusal of the canonical-provider render rail, nameable in the fail
/// log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderRenderDecline {
    /// The process-global provider could not be created, or its lifecycle
    /// reports a terminal health. In-class work is refused fail-closed, and a
    /// `device_lost` health runs the contract's teardown over the owner ledger
    /// first (`teardown` is the census of that pass).
    ProviderUnavailable {
        detail: String,
        health: &'static str,
        teardown: Option<DeviceLossTeardown>,
    },
    /// Compiling the declaring kernel, or one of the two stage modules, failed
    /// in the canonical translator. `step` names which of the three.
    PipelineCompile { step: &'static str, detail: String },
    /// The canonical provider refused, named by its normalized class: `step`
    /// names the provider call that answered, and `detail` carries the
    /// provider's own slug and detail text.
    ProviderRefused {
        class: ProviderRefusalClass,
        step: &'static str,
        detail: String,
    },
    /// The canonical provider answered `device_lost`; the teardown census is
    /// reported beside the provider's own words.
    ProviderDeviceLost {
        step: &'static str,
        detail: String,
        teardown: DeviceLossTeardown,
    },
    /// The canonical admission refused the trace this rail built.
    TraceAdmission { detail: String },
    /// The completion was not host-visible, so there are no bytes to write
    /// back.
    CompletionNotVisible,
    /// The completion carried no writeback for the colour attachment, so the
    /// pass ran without landing a frame.
    AttachmentWritebackMissing,
    /// The completion's writeback for the colour attachment does not cover the
    /// attachment's whole extent, so its bytes cannot be the frame.
    AttachmentWritebackShape {
        offset: u64,
        length: u64,
        expected: u64,
    },
    /// The owner rail refused a registration, an import, a retirement or a
    /// release. Not raised by this rail's own class (its inputs are
    /// trace-owned), but shared with the compute rail so one device-loss
    /// teardown has one name.
    Owner(provider_owner::Decline),
}

impl Decline for ProviderRenderDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ProviderUnavailable { .. } => "provider_unavailable",
            Self::PipelineCompile { .. } => "pipeline_compile",
            Self::ProviderRefused { class, .. } => class.slug(),
            Self::ProviderDeviceLost { .. } => ProviderRefusalClass::DeviceLost.slug(),
            Self::TraceAdmission { .. } => "trace_admission",
            Self::CompletionNotVisible => "completion_not_visible",
            Self::AttachmentWritebackMissing => "attachment_writeback_missing",
            Self::AttachmentWritebackShape { .. } => "attachment_writeback_shape",
            Self::Owner(inner) => inner.slug(),
        }
    }

    fn owner(&self) -> &'static str {
        match self {
            Self::Owner(inner) => inner.owner(),
            _ => core::any::type_name::<Self>(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ProviderUnavailable {
                detail,
                health,
                teardown,
            } => {
                let mut fields = vec![("health", health.to_string()), ("detail", detail.clone())];
                if let Some(teardown) = teardown {
                    fields.push(("teardown_leases", teardown.leases.to_string()));
                    fields.push(("teardown_windows", teardown.windows.to_string()));
                }
                fields
            }
            Self::PipelineCompile { step, detail } => {
                vec![("step", step.to_string()), ("detail", detail.clone())]
            }
            Self::ProviderRefused {
                class,
                step,
                detail,
            } => vec![
                ("class", class.name().to_string()),
                ("step", step.to_string()),
                ("detail", detail.clone()),
            ],
            Self::ProviderDeviceLost {
                step,
                detail,
                teardown,
            } => vec![
                ("class", ProviderRefusalClass::DeviceLost.name().to_string()),
                ("step", step.to_string()),
                ("detail", detail.clone()),
                ("teardown_leases", teardown.leases.to_string()),
                ("teardown_windows", teardown.windows.to_string()),
            ],
            Self::TraceAdmission { detail } => vec![("detail", detail.clone())],
            Self::CompletionNotVisible => Vec::new(),
            Self::AttachmentWritebackMissing => Vec::new(),
            Self::AttachmentWritebackShape {
                offset,
                length,
                expected,
            } => vec![
                ("offset", offset.to_string()),
                ("length", length.to_string()),
                ("expected", expected.to_string()),
            ],
            Self::Owner(inner) => inner.fields(),
        }
    }
}

decline_display!(ProviderRenderDecline);

/// The render half of the process-global canonical rail.
///
/// The provider and the neutral device are the compute rail's own
/// ([`provider_compute::rail`]): one provider context, one device epoch and one
/// pipeline registry behind both rails, so a device loss tears both down
/// together and the registration gate answers for one device incarnation.
/// What lives here is only what a render registration adds to that: the
/// declaring kernel registered once, and the one table entry per distinct
/// `(stages, contract)` pair the trace table names.
#[derive(Default)]
struct RenderRail {
    declaring: Mutex<Option<CompiledComputePipeline>>,
    pipelines: Mutex<HashMap<RenderPipelineKey, CompiledComputePipeline>>,
}

/// Cache key of one registered render pipeline: everything the registration
/// gate reads, so a hit can never answer with another shape's entry. The two
/// modules and the entries identify the stages; [`Self::contract`] is the
/// canonical rendering of the attachment format list and the vertex layout,
/// which is the other half of what `validate_stage_pair` compares against.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RenderPipelineKey {
    vertex_air: Vec<u8>,
    fragment_air: Vec<u8>,
    vertex_entry: String,
    fragment_entry: String,
    contract: String,
}

static RENDER_RAIL: OnceLock<RenderRail> = OnceLock::new();
static NEXT_OPERATION_ID: AtomicU64 = AtomicU64::new(1);

fn render_rail() -> &'static RenderRail {
    RENDER_RAIL.get_or_init(RenderRail::default)
}

/// Render a [`RenderPipelineContract`] canonically, for the cache key and the
/// registration digest.
///
/// Spelled rather than `Debug`-formatted: the key and the digest are read back
/// by the provider registry across a process, so they must be a function of the
/// contract's *values* and not of a derive's field order.
fn contract_fingerprint(contract: &RenderPipelineContract) -> String {
    let mut out = String::new();
    for format in &contract.color_formats {
        out.push_str(&format!("f{:02x};", format.code()));
    }
    match &contract.vertex_layout {
        VertexLayout::None => out.push_str("none;"),
        VertexLayout::Buffers(buffers) => {
            for buffer in buffers {
                out.push_str(&format!(
                    "b{:x}:{:?}:{:x};",
                    buffer.stride,
                    buffer.step.code(),
                    buffer.attributes.len()
                ));
                for attribute in &buffer.attributes {
                    out.push_str(&format!(
                        "a{}:{}:{:02x};",
                        attribute.location,
                        attribute.offset,
                        attribute.format.code()
                    ));
                }
            }
        }
    }
    out
}

/// Route one reims draw request: the class gate first, and the canonical
/// provider only if the whole shape is in the class.
pub fn submit_render(inputs: &RenderRailInputs<'_>, req: &DrawRequest) -> RenderRailOutcome {
    // The class gate is pure and runs first: an out-of-class shape never
    // touches the rail (no provider, no compile, no registration).
    let pass = match narrow_class(inputs, req) {
        Err(reason) => return RenderRailOutcome::NotInNarrowClass(reason),
        Ok(pass) => pass,
    };
    match submit_narrow(inputs, &pass) {
        Ok(output) => RenderRailOutcome::ProviderCompleted(output),
        Err(decline) => RenderRailOutcome::ProviderDeclined(decline),
    }
}

/// Drop every render registration that belonged to a device incarnation.
///
/// Called by [`super::provider_compute::recover_after_device_loss`]: the
/// registered handles are children of the dead `VkDevice`, exactly as the
/// compute rail's compiled pipelines are.
pub(crate) fn on_device_rebuilt() {
    let rail = render_rail();
    if let Ok(mut declaring) = rail.declaring.lock() {
        *declaring = None;
    }
    if let Ok(mut pipelines) = rail.pipelines.lock() {
        pipelines.clear();
    }
}

/// One admitted vertex stream: the single attribute its binding carries.
struct NarrowVertexStream<'a> {
    location: u32,
    offset: u64,
    stride: u64,
    format: VertexFormat,
    bytes: &'a [u8],
}

/// The admitted index stream.
struct NarrowIndexStream<'a> {
    format: IndexFormat,
    bytes: &'a [u8],
}

/// The whole admitted shape, in the terms the trace needs.
struct NarrowPass<'a> {
    /// The AIR entry each stage's translation reported, which is what the
    /// contract names and what the canonical gate checks the reflection
    /// against.
    vertex_entry: String,
    fragment_entry: String,
    format: AttachmentFormat,
    clear: [u8; 4],
    bgra: bool,
    width: u64,
    height: u64,
    extent: u64,
    index_count: u32,
    vertex_stream: Option<NarrowVertexStream<'a>>,
    index_stream: NarrowIndexStream<'a>,
}

impl NarrowPass<'_> {
    /// The contract's vertex layout for the admitted streams.
    ///
    /// One entry per stream, one attribute per entry: the engine numbers one
    /// Vulkan binding per attribute *location*, so the class's one-stream shape
    /// is one attribute at location 0 and the contract says exactly that.
    /// Derived in one place because the registration gate and the pass
    /// descriptor have to agree about it — two spellings of one layout is how
    /// a registration ends up describing a pass that is never submitted.
    fn vertex_layout(&self) -> VertexLayout {
        match &self.vertex_stream {
            Some(stream) => VertexLayout::Buffers(vec![VertexBufferLayout {
                stride: stream.stride,
                step: VertexStep::PerVertex,
                attributes: vec![VertexAttribute {
                    location: stream.location,
                    offset: stream.offset,
                    format: stream.format,
                }],
            }]),
            None => VertexLayout::None,
        }
    }
}

/// Whether one request is the narrow class, and the facts the trace is built
/// from when it is.
///
/// Pure, and ordered cheapest-first so a refused shape costs nothing: no
/// provider call, no translation, no registration. Every refusal names the
/// condition that kept the shape on the engine, because that string is what the
/// observer reports when a class boundary moves.
fn narrow_class<'a>(
    inputs: &RenderRailInputs<'_>,
    req: &'a DrawRequest,
) -> Result<NarrowPass<'a>, &'static str> {
    if !inputs.writeback_guest {
        return Err("a record that does not own the guest writeback stays on the engine");
    }
    if inputs.vertex_air.is_empty() || inputs.fragment_air.is_empty() {
        return Err("a request without both translated stages stays on the engine");
    }
    let (Some(vertex_entry), Some(fragment_entry)) = (inputs.vertex_entry, inputs.fragment_entry)
    else {
        return Err(
            "a stage whose translation reports no AIR entry point stays on the engine: the \
             canonical contract names that entry, and a name this rail invented would describe \
             another module",
        );
    };
    let Some(attachment) = req.color_attachment else {
        return Err("a request that never stated its attachment state stays on the engine");
    };
    let format = match attachment.format() {
        ash::vk::Format::R8G8B8A8_UNORM => AttachmentFormat::Rgba8Unorm,
        ash::vk::Format::B8G8R8A8_UNORM => AttachmentFormat::Bgra8Unorm,
        _ => {
            return Err(
                "only the admitted 8-bit colour formats leave for the canonical rail \
                 (Rgba8Unorm, Bgra8Unorm)",
            )
        }
    };
    let ColorClearValue::Float(clear) = attachment.clear() else {
        return Err("a non-float clear stays on the engine");
    };
    let Some(clear) = clear_bytes(clear) else {
        return Err(
            "a clear that is not byte-exact in the attachment's 8-bit encoding stays on the \
             engine: the canonical pass states bytes, the engine states floats, and the two \
             rounds must be the same value for the rails to agree",
        );
    };
    if req.width == 0 || req.height == 0 {
        return Err("a zero-sized attachment stays on the engine");
    }
    let extent = u64::from(req.width)
        .checked_mul(u64::from(req.height))
        .and_then(|texels| texels.checked_mul(4))
        .ok_or("an attachment extent that overflows u64 stays on the engine")?;
    if req.width > MAX_ATTACHMENT_DIMENSION || req.height > MAX_ATTACHMENT_DIMENSION {
        return Err(
            "an attachment outside the canonical rail's declared window stays on the engine: the \
             canonical render rail refuses a larger attachment at admission, and a shape the \
             provider always refuses is not one this class executes",
        );
    }
    if req.color0_declared != Some(crate::protocol::pass_action::LoadAction::Clear) {
        return Err(
            "the canonical class loads by `Clear` only; a Load or DontCare record stays on the \
             engine",
        );
    }
    if req.skip_readback {
        return Err("a record that skips its readback stays on the engine");
    }
    if req.target_identity.is_some()
        || req.seed_from_target.is_some()
        || req.load_from_target
        || req.target_rgba8.is_some()
        || req.target_guest_seed.is_some()
        || req.guest_target_memory.is_some()
        || req.load_guest_target_backing
        || req.record_guest_store
    {
        return Err(
            "a resident, seeded, chained or guest-backed target stays on the engine: the class \
             is the pooled offscreen target whose whole frame comes back through the completion",
        );
    }
    if req.continues_render_pass || req.render_pass_continues {
        return Err("a record inside a multi-record encoder stays on the engine");
    }
    if !req.secondary_targets.is_empty() {
        return Err("MRT stays on the engine");
    }
    if req.raster_sample_count > 1 || req.color_sample_count > 1 || req.multisample_resolve {
        return Err("a multisample or resolving pass stays on the engine");
    }
    if req.depth.is_some() {
        return Err("a depth or stencil state stays on the engine");
    }
    if !req.storage_buffers.is_empty() {
        return Err(
            "a pass whose stages bind buffers stays on the engine: the canonical render contract \
             has no buffer bindings for a pipeline",
        );
    }
    if !req.sampled_images.is_empty() || !req.samplers.is_empty() || req.color_input {
        return Err(
            "a pass that samples stays on the engine: the canonical render pass carries no \
             texture or sampler binding",
        );
    }
    if req.occlusion_query.is_some() {
        return Err("a visibility-armed draw stays on the engine");
    }
    if req.blend.is_some() || req.color_write_mask != crate::protocol::blend::ColorWriteMask::ALL {
        return Err("a blending or write-masked attachment stays on the engine");
    }
    if req.blend_color != [0.0; 4] {
        return Err("a draw with a blend constant stays on the engine");
    }
    if !req.viewports.is_empty() || !req.scissors.is_empty() {
        return Err(
            "an explicit viewport or scissor stays on the engine: the canonical pass states the \
             attachment-covering viewport and no scissor",
        );
    }
    if req.raster.cull_mode != reims_vgpu_vulkan::raster::GuestRasterState::DEFAULT.cull_mode
        || req.raster.winding != reims_vgpu_vulkan::raster::GuestRasterState::DEFAULT.winding
        || req.raster.depth_clip_mode
            != reims_vgpu_vulkan::raster::GuestRasterState::DEFAULT.depth_clip_mode
        || req.raster.fill_mode != reims_vgpu_vulkan::raster::GuestRasterState::DEFAULT.fill_mode
        || req.line_width.is_some()
    {
        return Err("a non-default raster state stays on the engine");
    }
    if req.base_instance != 0 {
        return Err("baseInstance stays on the engine");
    }
    if req.instance_count != Some(1) {
        return Err("an instanced draw stays on the engine");
    }
    if req.first_vertex != 0 {
        return Err("a non-indexed first-vertex offset stays on the engine");
    }
    if req.primitive_topology.0 != crate::protocol::topology::PrimitiveType::Triangle {
        return Err("only a triangle-list draw leaves for the canonical rail");
    }
    let Some(index) = req.indexed.as_ref() else {
        return Err(
            "the admitted shape is one indexed draw; a non-indexed draw stays on the engine",
        );
    };
    if index.index_count == 0 {
        return Err("a zero-length draw stays on the engine");
    }
    if index.vertex_offset != 0 {
        return Err("a baseVertex offset stays on the engine");
    }
    let index_format = match index.index_type {
        crate::backend::vulkan::engine::IndexType::U16 => IndexFormat::Uint16,
        crate::backend::vulkan::engine::IndexType::U32 => IndexFormat::Uint32,
    };
    let index_bytes = staged_bytes(&index.content)
        .ok_or("an index stream the GPU gathers from guest RAM stays on the engine")?;
    if index_bytes.is_empty() {
        return Err("an empty index stream stays on the engine");
    }

    // The engine numbers one Vulkan binding per attribute location, so the
    // one-stream class is one attribute at location 0. The same shape reaches
    // the contract as one stream with one attribute.
    let vertex_stream = match req.vertex_attributes.as_slice() {
        [] => None,
        [attribute] => {
            if attribute.binding != 0 || attribute.location != 0 {
                return Err(
                    "a vertex attribute that is not location 0 of binding 0 stays on the engine",
                );
            }
            if attribute.step_function != VertexStepFunction::PerVertex {
                return Err("a per-instance vertex stream stays on the engine");
            }
            let Some(format) = vertex_format(attribute.format) else {
                return Err(
                    "a vertex attribute outside the canonical format set stays on the engine \
                     (Float32x2, Float32x3, Float32x4, Uint32)",
                );
            };
            let stride = u64::from(attribute.stride);
            if stride < u64::from(attribute.offset) + format.bytes() {
                return Err(
                    "a vertex layout whose attribute does not fit its stride stays on the engine",
                );
            }
            let bytes = staged_bytes(&attribute.content)
                .ok_or("a vertex stream the GPU gathers from guest RAM stays on the engine")?;
            if bytes.len() < usize::try_from(stride).unwrap_or(usize::MAX) {
                return Err("a vertex stream shorter than one record stays on the engine");
            }
            Some(NarrowVertexStream {
                location: attribute.location,
                offset: u64::from(attribute.offset),
                stride,
                format,
                bytes,
            })
        }
        _ => {
            return Err(
                "a pass with more than one vertex attribute stays on the engine: the admitted \
                 class is one attribute at location 0",
            )
        }
    };

    Ok(NarrowPass {
        vertex_entry: vertex_entry.to_owned(),
        fragment_entry: fragment_entry.to_owned(),
        format,
        clear,
        bgra: format == AttachmentFormat::Bgra8Unorm,
        width: u64::from(req.width),
        height: u64::from(req.height),
        extent,
        index_count: index.index_count,
        vertex_stream,
        index_stream: NarrowIndexStream {
            format: index_format,
            bytes: index_bytes,
        },
    })
}

/// A clear value that survives the round trip through the 8-bit encoding.
///
/// The engine hands Vulkan `float32` components and the driver rounds them to
/// UNORM; the canonical pass states bytes. `Some` only when the rounding is the
/// identity, so "the two rails drew the same colour" is a byte comparison
/// rather than a tolerance.
fn clear_bytes(clear: [f32; 4]) -> Option<[u8; 4]> {
    let mut out = [0u8; 4];
    for (index, component) in clear.iter().enumerate() {
        if !component.is_finite() || *component < 0.0 || *component > 1.0 {
            return None;
        }
        let byte = (component * 255.0).round();
        if !byte.is_finite() || byte < 0.0 || byte > 255.0 {
            return None;
        }
        if (byte / 255.0 - component).abs() > 1e-6 {
            return None;
        }
        out[index] = byte as u8;
    }
    Some(out)
}

/// The CPU-staged bytes of one render input, or `None` when the content is the
/// zero-copy gather form.
fn staged_bytes(content: &BufferContent) -> Option<&[u8]> {
    match content {
        BufferContent::Bytes(bytes) => Some(bytes.as_slice()),
        BufferContent::GuestRuns(_) => None,
    }
}

/// The contract's vertex format for one reims attribute format.
fn vertex_format(
    format: crate::backend::vulkan::engine::VertexAttributeFormat,
) -> Option<VertexFormat> {
    use crate::protocol::vertex_format as raw;
    match format.ordinal() {
        raw::MTL_VERTEX_FORMAT_FLOAT2 => Some(VertexFormat::Float32x2),
        raw::MTL_VERTEX_FORMAT_FLOAT3 => Some(VertexFormat::Float32x3),
        raw::MTL_VERTEX_FORMAT_FLOAT4 => Some(VertexFormat::Float32x4),
        raw::MTL_VERTEX_FORMAT_U_INT => Some(VertexFormat::Uint32),
        _ => None,
    }
}

fn submit_narrow(
    inputs: &RenderRailInputs<'_>,
    pass: &NarrowPass<'_>,
) -> Result<RenderRailOutput, ProviderRenderDecline> {
    let rail = rail().map_err(IntoRender::into_render)?;
    let provider = &rail.provider;
    // The health gate runs before anything is compiled or registered: a
    // provider the lifecycle already reports terminal cannot take this draw,
    // and the answer is a refusal, never a silent switch to the other rail.
    if let Some(decline) = refuse_unhealthy(provider, "admission") {
        return Err(decline.into_render());
    }

    let declaring = declaring_pipeline(&rail.provider, &rail.device)?;
    let render_pipeline = register_render_pipeline(&rail.provider, &rail.device, inputs, pass)?;

    // The trace's own declaration of the attachment view. `OwnedBytes` of the
    // packed extent is what the frozen contract asks a storing attachment to
    // land through; the load is a clear, so none of these bytes are read.
    let declaration = BufferView {
        view_id: ATTACHMENT_VIEW,
        metal_binding: 0,
        allocation_id: ATTACHMENT_ALLOCATION,
        offset: 0,
        length: pass.extent,
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(vec![0u8; usize::try_from(pass.extent).unwrap_or(0)]),
    };
    let mut resources = ResourceTableSnapshot::new();
    for allocation in input_allocations(pass) {
        let (allocation_id, size) = allocation;
        resources
            .insert_allocation(AllocationRecord {
                allocation_id,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .map_err(|error| ProviderRenderDecline::TraceAdmission {
                detail: error.to_string(),
            })?;
    }
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: ATTACHMENT_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: pass.extent,
        })
        .map_err(|error| ProviderRenderDecline::TraceAdmission {
            detail: error.to_string(),
        })?;

    let mut vertex_buffers = Vec::new();
    let mut next_view = FIRST_INPUT_VIEW;
    if let Some(stream) = &pass.vertex_stream {
        vertex_buffers.push(BufferView {
            view_id: ViewId::new(next_view),
            metal_binding: stream.location,
            allocation_id: input_allocation(next_view),
            offset: 0,
            length: u64::try_from(stream.bytes.len()).unwrap_or(u64::MAX),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(stream.bytes.to_vec()),
        });
        next_view += 1;
    }
    let index_view = ViewId::new(next_view);
    let indices = IndexBufferBinding {
        view: BufferView {
            view_id: index_view,
            metal_binding: 0,
            allocation_id: input_allocation(next_view),
            offset: 0,
            length: u64::try_from(pass.index_stream.bytes.len()).unwrap_or(u64::MAX),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(pass.index_stream.bytes.to_vec()),
        },
        format: pass.index_stream.format,
    };

    let pass_descriptor = RenderPassDescriptor {
        pipeline: render_pipeline.pipeline_id,
        color_attachments: vec![RenderAttachment {
            view_id: ATTACHMENT_VIEW,
            allocation_id: ATTACHMENT_ALLOCATION,
            format: pass.format,
            width: pass.width,
            height: pass.height,
            load: LoadOp::Clear(ClearColor::new(pass.clear)),
            store: StoreOp::Store,
        }],
        viewport: [
            0,
            0,
            u32::try_from(pass.width).unwrap_or(u32::MAX),
            u32::try_from(pass.height).unwrap_or(u32::MAX),
        ],
        scissor: None,
        vertices: pass.index_count,
        vertex_buffers,
        indices: Some(indices),
        base_vertex: 0,
        cull: None,
        blend: None,
        multisample: None,
        depth_resolve: None,
        depth: None,
        depth_test: None,
        stencil: None,
        stencil_resolve: None,
        stencil_test: None,
        instance_count: 1,
        present: None,
    };
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(NEXT_OPERATION_ID.fetch_add(1, Ordering::Relaxed)),
        pipelines: vec![declaring.clone(), render_pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![
            TracePass::Compute(ComputePass {
                pipeline: declaring.pipeline_id,
                buffers: vec![declaration],
                textures: Vec::new(),
                dispatch: Dispatch {
                    kind: DispatchKind::ThreadsExact,
                    grid: [1, 1, 1],
                    threads_per_threadgroup: [1, 1, 1],
                },
            }),
            TracePass::Render(pass_descriptor),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let validated = provider
        .capabilities()
        .validate_trace(trace, resources)
        .map_err(|error| ProviderRenderDecline::TraceAdmission {
            detail: provider_error_detail(&error),
        })?;
    let result = provider
        .submit(validated)
        .map_err(|error| refusal_decline(&error, "submission").into_render())?;
    if !matches!(
        result.completion,
        CompletionDisposition::CompletedVisible { .. }
    ) {
        return Err(ProviderRenderDecline::CompletionNotVisible);
    }
    let Some(writeback) = result.writebacks.iter().find(|writeback| {
        writeback.view_id == ATTACHMENT_VIEW && writeback.allocation_id == ATTACHMENT_ALLOCATION
    }) else {
        return Err(ProviderRenderDecline::AttachmentWritebackMissing);
    };
    let length = u64::try_from(writeback.bytes.len()).unwrap_or(u64::MAX);
    if writeback.offset != 0 || length != pass.extent {
        return Err(ProviderRenderDecline::AttachmentWritebackShape {
            offset: writeback.offset,
            length,
            expected: pass.extent,
        });
    }
    Ok(RenderRailOutput {
        bytes: writeback.bytes.clone(),
        bgra: pass.bgra,
    })
}

/// Every input allocation the trace's render half carries: one per vertex
/// stream and one for the index stream, with the view's own length as the
/// extent.
fn input_allocations(pass: &NarrowPass<'_>) -> Vec<(AllocationId, u64)> {
    let mut out = Vec::new();
    let mut next_view = FIRST_INPUT_VIEW;
    if let Some(stream) = &pass.vertex_stream {
        out.push((
            input_allocation(next_view),
            u64::try_from(stream.bytes.len()).unwrap_or(u64::MAX),
        ));
        next_view += 1;
    }
    out.push((
        input_allocation(next_view),
        u64::try_from(pass.index_stream.bytes.len()).unwrap_or(u64::MAX),
    ));
    out
}

/// The allocation identity of one render input view: the view identity offset
/// into this rail's own namespace, so a stream can never alias the attachment.
fn input_allocation(view: u64) -> AllocationId {
    AllocationId::new(0x7265_6e64_6572_0100 + view)
}

/// Register (or look up) the one-thread kernel that declares the attachment
/// view.
fn declaring_pipeline(
    provider: &metal_api_vulkan::VulkanComputeProvider,
    device: &Device,
) -> Result<CompiledComputePipeline, ProviderRenderDecline> {
    let rail = render_rail();
    let mut declaring =
        rail.declaring
            .lock()
            .map_err(|_| ProviderRenderDecline::PipelineCompile {
                step: "declare_kernel",
                detail: "the declaring-kernel cache is poisoned".to_owned(),
            })?;
    if let Some(pipeline) = declaring.as_ref() {
        return Ok(pipeline.clone());
    }
    let digest = SemanticDigest::new(
        "reims-provider-render-declare-v1",
        RENDER_DECLARE_SOURCE.as_bytes().to_vec(),
    )
    .map_err(|error| ProviderRenderDecline::PipelineCompile {
        step: "declare_kernel",
        detail: error.to_string(),
    })?;
    let function = device
        .new_library_with_air(RENDER_DECLARE_SOURCE)
        .map_err(|error| ProviderRenderDecline::PipelineCompile {
            step: "declare_kernel",
            detail: error.to_string(),
        })?
        .function(RENDER_DECLARE_ENTRY)
        .map_err(|error| ProviderRenderDecline::PipelineCompile {
            step: "declare_kernel",
            detail: error.to_string(),
        })?;
    let compiled = provider
        .compile_pipeline(&function, digest)
        .map_err(|error| refusal_decline(&error, "declare_kernel").into_render())?;
    *declaring = Some(compiled.clone());
    Ok(compiled)
}

/// Register (or look up) the request's own translated pipeline pair.
///
/// The contract is built from the *request*, not from the AIR: the attachment
/// format list and the vertex layout are what reims' pipeline resolution
/// produced for this draw, and the canonical gate's answer is whether the two
/// translated stages agree with them field by field.
fn register_render_pipeline(
    provider: &metal_api_vulkan::VulkanComputeProvider,
    device: &Device,
    inputs: &RenderRailInputs<'_>,
    pass: &NarrowPass<'_>,
) -> Result<CompiledComputePipeline, ProviderRenderDecline> {
    let contract = RenderPipelineContract {
        vertex_entry: pass.vertex_entry.clone(),
        fragment_entry: pass.fragment_entry.clone(),
        color_formats: vec![pass.format],
        vertex_layout: pass.vertex_layout(),
    };
    let fingerprint = contract_fingerprint(&contract);
    let key = RenderPipelineKey {
        vertex_air: inputs.vertex_air.to_vec(),
        fragment_air: inputs.fragment_air.to_vec(),
        vertex_entry: pass.vertex_entry.clone(),
        fragment_entry: pass.fragment_entry.clone(),
        contract: fingerprint.clone(),
    };
    let rail = render_rail();
    let mut pipelines =
        rail.pipelines
            .lock()
            .map_err(|_| ProviderRenderDecline::PipelineCompile {
                step: "render_pipeline",
                detail: "the render pipeline cache is poisoned".to_owned(),
            })?;
    if let Some(pipeline) = pipelines.get(&key) {
        return Ok(pipeline.clone());
    }
    let stage = |air: &[u8], entry: &str, which: &'static str, stage| {
        let function = device
            .new_library_with_binary_air(air.to_vec())
            .map_err(|error| ProviderRenderDecline::PipelineCompile {
                step: which,
                detail: error.to_string(),
            })?
            .function(entry)
            .map_err(|error| ProviderRenderDecline::PipelineCompile {
                step: which,
                detail: error.to_string(),
            })?;
        TranslatedRenderStage::translate(stage, &function).map_err(|error| {
            ProviderRenderDecline::PipelineCompile {
                step: which,
                detail: error.to_string(),
            }
        })
    };
    let vertex = stage(
        inputs.vertex_air,
        &pass.vertex_entry,
        "vertex_stage",
        RenderStage::Vertex,
    )?;
    let fragment = stage(
        inputs.fragment_air,
        &pass.fragment_entry,
        "fragment_stage",
        RenderStage::Fragment,
    )?;
    let digest = SemanticDigest::new(
        "reims-provider-render-v1",
        format!(
            "{}|{}|{}",
            pass.vertex_entry, pass.fragment_entry, fingerprint
        )
        .into_bytes(),
    )
    .map_err(|error| ProviderRenderDecline::PipelineCompile {
        step: "render_pipeline",
        detail: error.to_string(),
    })?;
    let compiled = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract,
            vertex,
            fragment,
            logical_digest: digest,
        })
        .map_err(|error| refusal_decline(&error, "registration").into_render())?;
    pipelines.insert(key, compiled.clone());
    Ok(compiled)
}

/// Re-shape one compute-rail refusal onto this rail's decline.
///
/// The two rails share one provider, one owner ledger and one device-loss
/// guarantee, so the mapping is the same function; only the enum that carries
/// it differs. Spelled as a conversion rather than a second mapping so the two
/// cannot drift into reporting the same provider answer under two names.
trait IntoRender {
    fn into_render(self) -> ProviderRenderDecline;
}

impl IntoRender for super::provider_compute::ProviderComputeDecline {
    fn into_render(self) -> ProviderRenderDecline {
        use super::provider_compute::ProviderComputeDecline as Compute;
        match self {
            Compute::ProviderUnavailable {
                detail,
                health,
                teardown,
            } => ProviderRenderDecline::ProviderUnavailable {
                detail,
                health,
                teardown,
            },
            Compute::PipelineCompile { detail } => ProviderRenderDecline::PipelineCompile {
                step: "provider",
                detail,
            },
            Compute::ProviderRefused {
                class,
                step,
                detail,
            } => ProviderRenderDecline::ProviderRefused {
                class,
                step,
                detail,
            },
            Compute::ProviderDeviceLost {
                step,
                detail,
                teardown,
            } => ProviderRenderDecline::ProviderDeviceLost {
                step,
                detail,
                teardown,
            },
            Compute::TraceAdmission { detail } => ProviderRenderDecline::TraceAdmission { detail },
            Compute::CompletionNotVisible => ProviderRenderDecline::CompletionNotVisible,
            // The compute rail's own writeback shapes cannot arise on the
            // render half: it has no staged bindings to re-base. Reaching one
            // means the shared mapping grew a case this rail has not learned,
            // which is reported as a compile-step refusal rather than silently
            // folded into an attachment shape.
            other => ProviderRenderDecline::PipelineCompile {
                step: "provider",
                detail: format!("unmapped compute-rail refusal: {other:?}"),
            },
        }
    }
}
