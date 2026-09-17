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
//! - **up to four vertex streams** and exactly one index stream, each carried
//!   into the trace as trace-owned bytes, one canonical binding per stream. Four
//!   is the canonical contract's `MAX_VERTEX_BUFFERS`, and it is the stream axis
//!   the 2026-09-17 census measured as the one a real draw stream lives on
//!   (`8327 / 8526` draws bind two to four streams, `199` bind more); a fifth
//!   stays on the engine by name. The engine numbers one Vulkan binding per
//!   attribute location, so a request's attribute list is a list of streams —
//!   one attribute each — and the canonical binding index is the stream's
//!   position in that list, which is why the class renumbers rather than
//!   carrying the guest's binding numbers across;
//! - **at most one scissor rectangle inside the attachment**, and no viewport
//!   override: the canonical pass states the attachment-covering viewport and
//!   carries the guest's own rectangle through the pass scissor
//!   ([`RenderPassDescriptor::scissor`], the v29 channel both rails execute), so
//!   the texels outside the rectangle keep the load op's bytes on both. A
//!   rectangle that is empty, that reaches outside the attachment, or that is one
//!   of several stays on the engine: the contract refuses such a rect by name
//!   (`ScissorOutOfBounds`) while the engine *clamps* one that reaches past the
//!   attachment, so a class that repaired it here would answer with a frame the
//!   engine never drew;
//! - the pipeline pair is the request's own translated stages: the two SPIR-V
//!   modules reims' pipeline resolution produced, registered through the
//!   canonical *translated* registration gate
//!   (`register_translated_render_pipeline`), which checks each stage's
//!   reflection against the contract field by field;
//! - no depth, stencil, MSAA, MRT, resolve, present, blend, colour write mask,
//!   occlusion query, sampled image, sampler or storage buffer — and every one
//!   of those is a *reason*, not a silent downgrade;
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
//! # The window, and where its number comes from
//!
//! The class admits an attachment only inside the window the canonical rail
//! *declares* on this device. R1b made that declaration a device fact:
//! `ProviderCapabilities::max_attachment_dimension` is the smaller, per axis,
//! of the rail's reviewed ceiling and the selected device's own framebuffer
//! limit, and core admission refuses a wider attachment by name
//! (`attachment_dimension_limit`, carrying the maximum it crossed). This rail
//! reads that number out of the provider's own snapshot
//! ([`declared_attachment_window`]) instead of stating a copy: the copy it
//! shipped first (4×4) was one increment's reviewing window, and a gate that
//! outran the declaration would hand the provider a shape it always refuses —
//! a declined draw where the engine would have drawn it, which is the wrong
//! answer for a rail that only narrows which submissions change rail.
//!
//! The declaring view has to be backed by bytes the trace states, so the rail
//! hands it `BufferSource::OwnedBytes` of the attachment's own packed extent —
//! zeros, since a `Clear` load reads none of it. That is a real per-submission
//! copy, and it is the one cost this rail cannot avoid while the contract only
//! lets an attachment land through a byte-carrying view. At the declared
//! window the copy is at most 64×64 texels, the size the canonical rail's own
//! reviewed fixtures execute, so the declaration cannot be a pessimization
//! wearing a feature flag; the byte-less declaration that would lift the bound
//! entirely is the named follow-up on the emulator side.
//!
//! # The lease channel, and the gap this increment reports
//!
//! `research/docs/26` §3 planned to carry the vertex and index streams through
//! the owner rail's lease channel (the one [`super::provider_owner`] gives the
//! compute rail). The canonical side has since opened that channel per binding:
//! R3c's `resolve_render_input` resolves each stream's own
//! `BufferSource::OwnedBytes` / `StagedLease` / `BorrowedNoCopy` arm under the
//! input's binding index (`render.rs::resolve_vertex_streams` zips the pass's
//! views with the pipeline layout, so N streams are N independent
//! resolutions — nothing in it assumes one stream). What is still missing is on
//! *this* side: this rail holds no lease for a guest-gathered stream, so every
//! stream here travels as trace-owned bytes (the same bytes reims would have
//! staged) and a stream the GPU would gather from guest RAM is refused by name
//! (`render_provider_out_of_class_vertex_staging`) instead of being read
//! through a lease this module cannot mint from the owner rail without the
//! emulator's own lease channel. Multi-stream admission therefore changes the
//! number of streams and not their source; the lease arm stays the named
//! follow-up the report beside this increment records.
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

use std::borrow::Cow;
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
    MAX_VERTEX_BUFFERS, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::Device;
use metal_api_vulkan::{RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage};

use super::provider_compute::{
    provider_error_detail, rail, refusal_decline, refuse_unhealthy, ProviderRefusalClass,
};
use super::provider_owner::{self, DeviceLossTeardown};
use super::vulkan::engine::types::{
    BufferContent, ColorClearValue, DrawRequest, ReadbackSkipReason, VertexAttributeResource,
    VertexStepFunction,
};
use crate::observe::{decline_display, Decline};

/// The declaring kernel this rail ships (`crates/reims-vgpu/src/backend/render_declare.ll`).
///
/// Owned, not derived from a third-party metallib; the file states what it is
/// for and why a read is the whole body.
const RENDER_DECLARE_SOURCE: &str = include_str!("render_declare.ll");

/// Entry point [`RENDER_DECLARE_SOURCE`] declares.
const RENDER_DECLARE_ENTRY: &str = "reims_declare";

/// The attachment window the canonical rail declares on this device, one
/// number per axis.
///
/// Read from the provider's own capability snapshot rather than stated here
/// (see the module docs): `max_attachment_dimension` is `min(reviewed ceiling,
/// device framebuffer limit)` per axis when the snapshot is built, so the
/// number this gate compares against is the number admission itself enforces
/// (`attachment_dimension_limit`). A second copy in this crate would drift the
/// moment either half moves — and this rail shipped exactly such a copy once
/// (4×4, the reviewing window of the increment that opened the rail), which is
/// why the class gate asks the provider instead.
///
/// Reading it is what puts the rail's provider in place the first time a shape
/// passes every other class condition; a shape the pure gate already refused
/// never gets this far. The read itself is a snapshot read, not provider work:
/// nothing is translated, registered, admitted or submitted for a request this
/// check keeps on the engine.
fn declared_attachment_window() -> Result<[u64; 2], ProviderRenderDecline> {
    let rail = rail().map_err(IntoRender::into_render)?;
    Ok(rail.provider.capabilities().max_attachment_dimension)
}

/// Whether one attachment extent is inside the declared window, and the class
/// reason that keeps it on the engine when it is not.
///
/// Pure over its inputs, so the boundary is the provider's declaration and
/// nothing else — the reason names both numbers, because that string is what
/// the observer reports when the boundary moves.
fn window_admits(window: [u64; 2], width: u64, height: u64) -> Result<(), OutOfClass> {
    if width <= window[0] && height <= window[1] {
        return Ok(());
    }
    // The window is one condition however many shapes meet it, so one bucket:
    // the numbers that moved are in the sentence, and what a reader wants from
    // the counter is how much of a boot's stream the window refuses.
    Err(OutOfClass::owned(
        "render_provider_out_of_class_attachment_window",
        format!(
            "an attachment of {width}x{height} is outside the window this device's provider \
             declares ({}x{}, `max_attachment_dimension`): the canonical rail states the window, \
             admission refuses a wider attachment by name (`attachment_dimension_limit`), and a \
             shape the provider always refuses is not one this class executes",
            window[0], window[1],
        ),
    ))
}

/// Whether one scissor rectangle is a shape the canonical pass can state.
///
/// The contract carries at most one rectangle per pass and refuses one that is
/// empty or reaches outside the viewport (`ScissorOutOfBounds`), while the
/// engine *clamps* a rectangle that reaches past the attachment and draws the
/// intersection. The two rails therefore answer an out-of-bounds rect
/// differently — one refuses, one draws a smaller rectangle — so such a request
/// is a class exit and not a decline: the engine's own answer is the one the
/// class has to leave it to.
fn scissor_admits(
    scissor: crate::backend::vulkan::engine::ScissorResource,
    width: u32,
    height: u32,
) -> Result<[u32; 4], OutOfClass> {
    let inside = u64::from(scissor.x)
        .checked_add(u64::from(scissor.width))
        .is_some_and(|end| end <= u64::from(width))
        && u64::from(scissor.y)
            .checked_add(u64::from(scissor.height))
            .is_some_and(|end| end <= u64::from(height));
    if scissor.width != 0 && scissor.height != 0 && inside {
        return Ok([scissor.x, scissor.y, scissor.width, scissor.height]);
    }
    Err(OutOfClass::owned(
        "render_provider_out_of_class_scissor",
        format!(
            "a scissor rectangle of {}x{} at ({}, {}) stays on the engine: the canonical pass \
             refuses a rectangle that is empty or reaches outside the {width}x{height} \
             attachment (`ScissorOutOfBounds`), while the engine clamps such a rectangle and \
             draws the part inside it — so this is the engine's own answer to give",
            scissor.width, scissor.height, scissor.x, scissor.y,
        ),
    ))
}

/// Whether a request's declared attributes name exactly the locations the
/// vertex stage's own translation reports.
///
/// Both halves are small — the stream count is bounded by the contract's
/// `max_vertex_buffers` and one stream carries one attribute — so the comparison
/// is a linear walk of a list no longer than four rather than a set. A request
/// with two attributes at one location cannot pass it: the reflected locations
/// are distinct, so a duplicate leaves one of them uncovered.
fn attribute_locations_match(attributes: &[VertexAttributeResource], reflected: &[u32]) -> bool {
    attributes.len() == reflected.len()
        && attributes
            .iter()
            .all(|attribute| reflected.contains(&attribute.location))
}

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
    /// The *vertex stage's own* attribute locations, as the translation that
    /// produced `vertex_air` reflected them
    /// (`CachedShader::reflection.vertex_attributes`), in the reflection's
    /// order.
    ///
    /// The canonical registration gate compares the contract's vertex layout
    /// with the reflection attribute by attribute and refuses a layout that
    /// names an attribute the shader does not read, or one attribute fewer than
    /// the shader reads. A request whose own declared attributes disagree with
    /// that set is therefore a shape the provider *always* refuses — and a
    /// refusal is a decline, not a fallback, so the class has to answer it
    /// before the provider is asked. Carried as locations rather than as the
    /// reflection type itself so this module keeps its provider-facing imports
    /// and compares the one field the gate reads.
    pub vertex_attribute_locations: &'a [u32],
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
    NotInNarrowClass(OutOfClass),
    /// In-class, but the canonical provider refused. The caller must decline
    /// the draw rather than fall back to another rail.
    ProviderDeclined(ProviderRenderDecline),
}

/// Why a request is outside the admitted class, in the two spellings it needs.
///
/// The class gate answers with a *sentence* — the seam prints it beside
/// `linux_render_provider out_of_class`, and it is what a reader compares
/// against when the class moves — and with a *bucket*, counted into the
/// `store_routes` window as `render_provider_out_of_class_<slug>`.
///
/// The bucket is the half the 2026-09-17 render profile asked for and the
/// reason it is a type rather than a lookup: the profile had to *re-derive* the
/// out-of-class population from a dozen counters in a dozen windows ("how much
/// of a boot's draw stream was instanced, was multisampled, was Load rather
/// than Clear"), because the only per-reason answer the seam printed was the
/// first-appearance line — deduplicated per process and therefore sized by
/// neither time nor draws. The slug is a stable, spelled-out name beside the
/// sentence in the same `return`, so the two cannot drift into disagreeing
/// about which condition fired.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutOfClass {
    /// The census bucket, as the `store_routes` name it is counted under:
    /// `render_provider_out_of_class_<slug>`.
    route: &'static str,
    /// The sentence the seam reports. Borrowed for the conditions a request
    /// answers by itself; owned for the one that is a property of the device —
    /// the declared attachment window — because that answer names the numbers
    /// it compared.
    detail: Cow<'static, str>,
}

impl OutOfClass {
    /// One class condition a request answers about itself.
    const fn new(slug: &'static str, detail: &'static str) -> Self {
        Self {
            route: slug,
            detail: Cow::Borrowed(detail),
        }
    }

    /// One class condition whose sentence names a number the gate *read* — the
    /// device's declared window, the contract's stream ceiling — instead of one
    /// this module states.
    ///
    /// Owned for the reason the window reason is: the number is part of the
    /// answer, and a reader comparing two boots needs it beside the condition
    /// rather than in a copy that can drift from the declaration it came from.
    fn owned(slug: &'static str, detail: String) -> Self {
        Self {
            route: slug,
            detail: Cow::Owned(detail),
        }
    }

    /// The sentence, for the caller to print.
    pub fn detail(&self) -> &str {
        &self.detail
    }

    /// The census bucket, for a caller that has to name the answer *and* the
    /// shape it was asked about: the seam's latched `slug × shape` row prints
    /// this beside the request's own fields, and the profile's probe could only
    /// *re-derive* that join from a dozen counters while the slug was private.
    /// Public for the same reason [`Self::detail`] is: the answer's two halves
    /// belong to whoever prints the boundary, and neither may be re-spelled
    /// here and there.
    pub fn slug(&self) -> &'static str {
        self.route
    }

    /// Count this answer into the census window.
    ///
    /// Called at the exit of [`submit_render`] rather than by the seam, so the
    /// bucket fires for every caller of the rail — including the off-VM tests
    /// that drive `submit_render` directly — and so a future caller cannot
    /// forget it.
    fn note(&self) {
        crate::runtime::drain::note_store_route(self.route);
    }
}

impl std::fmt::Display for OutOfClass {
    /// The sentence, so a caller that only wants to print the boundary does not
    /// have to reach for the field — the same shape `Cow<'static, str>` gave
    /// this answer before it grew a bucket.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
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

/// How many traces this rail has handed to the canonical provider, one per
/// submission that reached it.
static PROVIDER_SUBMISSIONS: AtomicU64 = AtomicU64::new(0);

/// The number of render submissions this rail has handed to the canonical
/// provider.
///
/// Test observation; production callers do not read it. The class gate decides
/// before this number moves, so a shape the gate keeps on the engine — the
/// attachment-window boundary included — leaves it untouched, which is how the
/// rail's own test asserts a fallback never reached the provider rather than
/// only that the answer looked like a fallback.
pub fn provider_submissions() -> u64 {
    PROVIDER_SUBMISSIONS.load(Ordering::Relaxed)
}

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
///
/// Two stages, in this order: the pure gate, which answers everything the
/// request states about itself, and the declared-window check, which is the one
/// condition that belongs to the device. A shape that fails either is out of
/// class and the caller runs the self-contained engine; only a shape that
/// passes both is offered to the provider, and a refusal from there is a
/// decline (never a fallback).
pub fn submit_render(inputs: &RenderRailInputs<'_>, req: &DrawRequest) -> RenderRailOutcome {
    // The band a widening order sizes the vertex axis on, charged for every
    // request the gate is handed and before any condition answers — so a shape
    // the gate refuses is still in the denominator, and the four arms sum to the
    // band's population by construction. The list it counts is the one the
    // vertex half of the class reads (`req.vertex_attributes`), which nothing
    // else in this device measured: the 2026-09-17 census had the *bound* stream
    // count (`draw_vertex_streams_*`) and no reading of the declared attributes
    // the gate actually compares.
    crate::runtime::drain::note_store_route(attribute_count_route(req.vertex_attributes.len()));
    // The class gate is pure and runs first: an out-of-class shape never
    // touches the rail (no provider, no compile, no registration).
    let pass = match narrow_class(inputs, req) {
        Err(reason) => {
            reason.note();
            return RenderRailOutcome::NotInNarrowClass(reason);
        }
        Ok(pass) => pass,
    };
    // The window is the one class condition a request cannot answer by itself:
    // it is the canonical rail's declaration on this device (R1b), read from
    // the provider's capability snapshot — before this submission is
    // translated, registered, admitted or submitted, so a request wider than
    // the window the provider declares is a *class* answer and not a decline.
    // The read is what puts the rail's provider in place the first time a
    // candidate shape arrives; the pure gate above has already answered for
    // every shape that is out of class for any other reason.
    let window = match declared_attachment_window() {
        Ok(window) => window,
        // A provider that cannot be reached cannot answer the one question the
        // gate still has to ask, and an unclassifiable candidate is an
        // in-class candidate: fail closed, exactly as `submit_narrow` does for
        // the shapes it refuses, rather than silently re-running the draw
        // somewhere the class never named.
        Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
    };
    if let Err(reason) = window_admits(window, pass.width, pass.height) {
        reason.note();
        return RenderRailOutcome::NotInNarrowClass(reason);
    }
    match submit_narrow(inputs, &pass) {
        Ok(output) => RenderRailOutcome::ProviderCompleted(output),
        Err(decline) => RenderRailOutcome::ProviderDeclined(decline),
    }
}

/// The census band of one request's declared vertex streams.
///
/// `MAX_VERTEX_BUFFERS` is the arm boundary as well as the class's ceiling: a
/// build where the contract moved to five streams would have to rename the
/// `_gt4` arm rather than silently count five as `2_4`, which is what the
/// assertion in this module's tests pins.
pub fn attribute_count_route(declared: usize) -> &'static str {
    match declared {
        0 => "draw_vertex_attrs_0",
        1 => "draw_vertex_attrs_1",
        2..=4 => "draw_vertex_attrs_2_4",
        _ => "draw_vertex_attrs_gt4",
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
    /// The admitted streams, in the request's own attribute order: entry `i`
    /// becomes canonical binding `i`. One attribute per stream, because the
    /// engine numbers one Vulkan binding per attribute location — an interleaved
    /// stream is several attributes in the *guest's* pipeline, and each one
    /// arrives here as its own attribute with its own bytes and stride.
    vertex_streams: Vec<NarrowVertexStream<'a>>,
    index_stream: NarrowIndexStream<'a>,
    /// The scissor rectangle the pass states, or `None` for the whole
    /// attachment — the canonical pass's own default
    /// ([`RenderPassDescriptor::scissor`]).
    scissor: Option<[u32; 4]>,
}

impl NarrowPass<'_> {
    /// The contract's vertex layout for the admitted streams.
    ///
    /// One entry per stream, one attribute per entry: the engine numbers one
    /// Vulkan binding per attribute *location*, so a stream is one attribute
    /// and the contract says exactly that — at the location the guest declared,
    /// which is the location the shader reads and the one the canonical
    /// pipeline's vertex input state is keyed on.
    /// Derived in one place because the registration gate and the pass
    /// descriptor have to agree about it — two spellings of one layout is how
    /// a registration ends up describing a pass that is never submitted.
    fn vertex_layout(&self) -> VertexLayout {
        if self.vertex_streams.is_empty() {
            return VertexLayout::None;
        }
        VertexLayout::Buffers(
            self.vertex_streams
                .iter()
                .map(|stream| VertexBufferLayout {
                    stride: stream.stride,
                    step: VertexStep::PerVertex,
                    attributes: vec![VertexAttribute {
                        location: stream.location,
                        offset: stream.offset,
                        format: stream.format,
                    }],
                })
                .collect(),
        )
    }
}

/// Whether one request is the narrow class, and the facts the trace is built
/// from when it is.
///
/// Pure, and ordered cheapest-first so a refused shape costs nothing: no
/// provider call, no translation, no registration. Every refusal names the
/// condition that kept the shape on the engine, because that string is what the
/// observer reports when a class boundary moves. The one condition that is not
/// a property of the request — the attachment window, which belongs to the
/// device's provider — is [`declared_attachment_window`]'s, asked by
/// [`submit_render`] after this gate has accepted the shape.
fn narrow_class<'a>(
    inputs: &RenderRailInputs<'_>,
    req: &'a DrawRequest,
) -> Result<NarrowPass<'a>, OutOfClass> {
    if !inputs.writeback_guest {
        // Two records answer this and they are different shapes: the *head* of a
        // packet (the first record; `continues_render_pass == false`) hands its
        // frame to the next record, and every record after it hands the chain
        // on. Splitting the answer by name costs one branch and is what makes
        // the next census read the 2653 chain heads and the 2065 middles apart
        // instead of as one 4718-wide bucket; the class still refuses both,
        // because a frame that has to reach the next record is a rail this
        // increment does not execute.
        return Err(if req.continues_render_pass {
            OutOfClass::new(
                "render_provider_out_of_class_chain_middle",
                "a record in the middle of a multi-record packet stays on the engine: its frame \
                 is a seed for the record after it, which is a rail this class does not execute",
            )
        } else {
            OutOfClass::new(
                "render_provider_out_of_class_chain_head",
                "the first record of a multi-record packet stays on the engine: it hands its \
                 frame to the next record rather than to guest memory",
            )
        });
    }
    if inputs.vertex_air.is_empty() || inputs.fragment_air.is_empty() {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_stages",
            "a request without both translated stages stays on the engine",
        ));
    }
    let (Some(vertex_entry), Some(fragment_entry)) = (inputs.vertex_entry, inputs.fragment_entry)
    else {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_stage_entry",
            "a stage whose translation reports no AIR entry point stays on the engine: the \
             canonical contract names that entry, and a name this rail invented would describe \
             another module",
        ));
    };
    let Some(attachment) = req.color_attachment else {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_attachment_state",
            "a request that never stated its attachment state stays on the engine",
        ));
    };
    let format = match attachment.format() {
        ash::vk::Format::R8G8B8A8_UNORM => AttachmentFormat::Rgba8Unorm,
        ash::vk::Format::B8G8R8A8_UNORM => AttachmentFormat::Bgra8Unorm,
        _ => {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_format",
                "only the admitted 8-bit colour formats leave for the canonical rail \
                 (Rgba8Unorm, Bgra8Unorm)",
            ))
        }
    };
    let ColorClearValue::Float(clear) = attachment.clear() else {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_clear_kind",
            "a non-float clear stays on the engine",
        ));
    };
    let Some(clear) = clear_bytes(clear) else {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_clear_bytes",
            "a clear that is not byte-exact in the attachment's 8-bit encoding stays on the \
             engine: the canonical pass states bytes, the engine states floats, and the two \
             rounds must be the same value for the rails to agree",
        ));
    };
    if req.width == 0 || req.height == 0 {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_zero_extent",
            "a zero-sized attachment stays on the engine",
        ));
    }
    let extent = u64::from(req.width)
        .checked_mul(u64::from(req.height))
        .and_then(|texels| texels.checked_mul(4))
        .ok_or(OutOfClass::new(
            "render_provider_out_of_class_extent_overflow",
            "an attachment extent that overflows u64 stays on the engine",
        ))?;
    if req.color0_declared != Some(crate::protocol::pass_action::LoadAction::Clear) {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_load_action",
            "the canonical class loads by `Clear` only; a Load or DontCare record stays on the \
             engine",
        ));
    }
    if req.skip_readback {
        // Which of the two rails set the flag decides the name, because they are
        // not the same shape: a store that publishes nothing has no resident and
        // no reader, while a resident store is a frame the guest reads back
        // through a mapping or a GVA (`draw_partial_load_from_target`). The
        // class refuses both here — the second until the provider can own a
        // named resident, the first until its writeback route is reviewed — and
        // the split is what lets the next census size them separately.
        return Err(match req.readback_skip_reason {
            ReadbackSkipReason::ResidentStore => OutOfClass::new(
                "render_provider_out_of_class_resident_store",
                "a record whose frame lands in an engine-resident target stays on the engine: the \
                 resident is what the guest reads back through, and the canonical rail cannot \
                 name it yet",
            ),
            ReadbackSkipReason::UnpublishedStore => OutOfClass::new(
                "render_provider_out_of_class_unpublished_store",
                "a record whose store action publishes nothing stays on the engine: the class \
                 reads its frame back from the completion, and a store that lands nowhere is a \
                 route this increment has not reviewed",
            ),
            // A skip with no recorded reason is a fact about the caller rather
            // than a third rail, and guessing which of the two it is would name
            // the wrong population.
            ReadbackSkipReason::None => OutOfClass::new(
                "render_provider_out_of_class_skip_readback",
                "a record that skips its readback for a reason this rail cannot name stays on the \
                 engine",
            ),
        });
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
        return Err(OutOfClass::new(
            "render_provider_out_of_class_target",
            "a resident, seeded, chained or guest-backed target stays on the engine: the class \
             is the pooled offscreen target whose whole frame comes back through the completion",
        ));
    }
    if req.continues_render_pass || req.render_pass_continues {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_encoder",
            "a record inside a multi-record encoder stays on the engine",
        ));
    }
    if !req.secondary_targets.is_empty() {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_mrt",
            "MRT stays on the engine",
        ));
    }
    if req.raster_sample_count > 1 || req.color_sample_count > 1 || req.multisample_resolve {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_multisample",
            "a multisample or resolving pass stays on the engine",
        ));
    }
    if req.depth.is_some() {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_depth",
            "a depth or stencil state stays on the engine",
        ));
    }
    if !req.storage_buffers.is_empty() {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_buffers",
            "a pass whose stages bind buffers stays on the engine: the canonical render contract \
             has no buffer bindings for a pipeline",
        ));
    }
    if !req.sampled_images.is_empty() || !req.samplers.is_empty() || req.color_input {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_sampling",
            "a pass that samples stays on the engine: the canonical render pass carries no \
             texture or sampler binding",
        ));
    }
    if req.occlusion_query.is_some() {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_visibility",
            "a visibility-armed draw stays on the engine",
        ));
    }
    if req.blend.is_some() || req.color_write_mask != crate::protocol::blend::ColorWriteMask::ALL {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_blend",
            "a blending or write-masked attachment stays on the engine",
        ));
    }
    if req.blend_color != [0.0; 4] {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_blend_color",
            "a draw with a blend constant stays on the engine",
        ));
    }
    // The viewport the canonical pass states is the attachment-covering one and
    // nothing else, so a request that binds a viewport of its own stays on the
    // engine. The scissor is the other half of that pair and the contract does
    // carry it: `RenderPassDescriptor::scissor` is the v29 channel both rails
    // execute, and the pass states it verbatim — so a request that binds exactly
    // one rectangle inside the attachment is in class, and its texels outside
    // the rectangle keep the load op's bytes on both rails.
    if !req.viewports.is_empty() {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_viewport",
            "a request that binds a viewport of its own stays on the engine: the canonical pass \
             states the attachment-covering viewport, and another size or origin is not state the \
             frozen pass descriptor can carry",
        ));
    }
    let scissor = match req.scissors.as_slice() {
        [] => None,
        [scissor] => Some(scissor_admits(*scissor, req.width, req.height)?),
        _ => {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_scissor_count",
                "a draw that binds more than one scissor rectangle stays on the engine: the \
                 canonical pass carries at most one, and one rectangle cannot state the others",
            ))
        }
    };
    if req.raster.cull_mode != reims_vgpu_vulkan::raster::GuestRasterState::DEFAULT.cull_mode
        || req.raster.winding != reims_vgpu_vulkan::raster::GuestRasterState::DEFAULT.winding
        || req.raster.depth_clip_mode
            != reims_vgpu_vulkan::raster::GuestRasterState::DEFAULT.depth_clip_mode
        || req.raster.fill_mode != reims_vgpu_vulkan::raster::GuestRasterState::DEFAULT.fill_mode
        || req.line_width.is_some()
    {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_raster",
            "a non-default raster state stays on the engine",
        ));
    }
    if req.base_instance != 0 {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_base_instance",
            "baseInstance stays on the engine",
        ));
    }
    if req.instance_count != Some(1) {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_instanced",
            "an instanced draw stays on the engine",
        ));
    }
    if req.first_vertex != 0 {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_first_vertex",
            "a non-indexed first-vertex offset stays on the engine",
        ));
    }
    if req.primitive_topology.0 != crate::protocol::topology::PrimitiveType::Triangle {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_topology",
            "only a triangle-list draw leaves for the canonical rail",
        ));
    }
    let Some(index) = req.indexed.as_ref() else {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_nonindexed",
            "the admitted shape is one indexed draw; a non-indexed draw stays on the engine",
        ));
    };
    if index.index_count == 0 {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_index_count_zero",
            "a zero-length draw stays on the engine",
        ));
    }
    if index.vertex_offset != 0 {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_base_vertex",
            "a baseVertex offset stays on the engine",
        ));
    }
    let index_format = match index.index_type {
        crate::backend::vulkan::engine::IndexType::U16 => IndexFormat::Uint16,
        crate::backend::vulkan::engine::IndexType::U32 => IndexFormat::Uint32,
    };
    let index_bytes = staged_bytes(&index.content).ok_or(OutOfClass::new(
        "render_provider_out_of_class_index_staging",
        "an index stream the GPU gathers from guest RAM stays on the engine",
    ))?;
    if index_bytes.is_empty() {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_index_empty",
            "an empty index stream stays on the engine",
        ));
    }

    // One stream per declared attribute, because that is the shape both rails
    // execute: the engine numbers one Vulkan binding per attribute *location*
    // (`runtime/draw/vulkan.rs` builds `binding: a.location`), and the canonical
    // layout this class states describes binding `i` with the request's `i`-th
    // attribute and nothing else. Four is the canonical contract's
    // `max_vertex_buffers` — the axis 97.7 % of the measured draw stream lives on
    // — and a fifth is a layout the contract cannot state, so it stays on the
    // engine rather than in a trace admission would refuse.
    if req.vertex_attributes.len() > MAX_VERTEX_BUFFERS {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_vertex_stream_limit",
            format!(
                "a draw that binds more than {MAX_VERTEX_BUFFERS} vertex streams stays on the \
                 engine: {} is the canonical contract's `max_vertex_buffers`, and a wider layout \
                 is a shape admission refuses by name",
                MAX_VERTEX_BUFFERS,
            ),
        ));
    }
    // The declared attributes and the vertex stage's *own* reflection have to
    // name the same locations. The registration gate compares the two field by
    // field and refuses a mismatch — a stream the shader never reads, or a
    // location it reads that no stream covers — so a request that disagrees with
    // its own translation is a shape the provider always refuses, and a refusal
    // is a decline rather than a fallback. Answering it here is what keeps the
    // class's one promise: everything it admits is a shape the provider executes.
    if !attribute_locations_match(&req.vertex_attributes, inputs.vertex_attribute_locations) {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_vertex_interface",
            "a request whose declared vertex attributes do not name exactly the locations the \
             vertex stage reads stays on the engine: the canonical registration gate compares \
             the layout with the reflection field by field, so this is a shape the provider \
             always refuses",
        ));
    }
    let mut vertex_streams = Vec::with_capacity(req.vertex_attributes.len());
    for attribute in &req.vertex_attributes {
        if attribute.step_function != VertexStepFunction::PerVertex {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_vertex_step",
                "a per-instance vertex stream stays on the engine: the canonical layout states \
                 one step per stream and the class admits the per-vertex one",
            ));
        }
        let Some(format) = vertex_format(attribute.format) else {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_vertex_format",
                "a vertex attribute outside the canonical format set stays on the engine \
                 (Float32x2, Float32x3, Float32x4, Uint32)",
            ));
        };
        let stride = u64::from(attribute.stride);
        if stride < u64::from(attribute.offset) + format.bytes() {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_vertex_stride",
                "a vertex layout whose attribute does not fit its stride stays on the engine",
            ));
        }
        let bytes = staged_bytes(&attribute.content).ok_or(OutOfClass::new(
            "render_provider_out_of_class_vertex_staging",
            "a vertex stream the GPU gathers from guest RAM stays on the engine",
        ))?;
        if bytes.len() < usize::try_from(stride).unwrap_or(usize::MAX) {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_vertex_short",
                "a vertex stream shorter than one record stays on the engine",
            ));
        }
        vertex_streams.push(NarrowVertexStream {
            location: attribute.location,
            offset: u64::from(attribute.offset),
            stride,
            format,
            bytes,
        });
    }

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
        vertex_streams,
        index_stream: NarrowIndexStream {
            format: index_format,
            bytes: index_bytes,
        },
        scissor,
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
    for (binding, stream) in pass.vertex_streams.iter().enumerate() {
        // The canonical binding index is the stream's position in the request,
        // not the guest's own binding number: the contract requires entry `i` to
        // carry `metal_binding == i` (`VertexBufferBindingMismatch` otherwise),
        // and the pipeline's vertex input state is keyed on attribute
        // *locations*, which are the guest's and are carried unchanged. Two
        // rails that agree on every location, format, offset and stride fetch
        // the same bytes whatever the binding numbers are called.
        vertex_buffers.push(BufferView {
            view_id: ViewId::new(next_view),
            metal_binding: u32::try_from(binding).unwrap_or(u32::MAX),
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
        scissor: pass.scissor,
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
        // The v70 sampler channel: this class refuses sampled images, samplers
        // and color input before it ever gets here, so the pass binds none.
        textures: Vec::new(),
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
    // Counted here, at the boundary: this is the point past which the draw is
    // the provider's work, so a request that stays on the engine must leave
    // the counter where it was.
    PROVIDER_SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
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
    let mut out = Vec::with_capacity(pass.vertex_streams.len() + 1);
    let mut next_view = FIRST_INPUT_VIEW;
    for stream in &pass.vertex_streams {
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
