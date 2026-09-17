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
//! - **one colour attachment** at an admitted format — the two 8-bit colour
//!   orders (`Rgba8Unorm` / `Bgra8Unorm`) and the wide `Rgba16Float` (eight
//!   bytes per texel, `research/docs/23` §78) — loaded by a `Clear` whose
//!   components are byte-exact in that format's own texel, and stored. A clear
//!   that is not byte-exact is a *different* colour on the two rails — the
//!   engine hands Vulkan a `float32` clear and the driver rounds it to the
//!   attachment's texel, while the canonical pass states the texel bytes — so
//!   such a request keeps the engine instead of being compared at a tolerance;
//!   a **wide** attachment is in class only where the frame is the provider's
//!   own image (the resident arms below), because the engine's pooled offscreen
//!   target is its own four-byte image: a pooled wide attachment is a shape the
//!   engine does not draw at all, and this class cannot answer for one
//!   (`render_provider_out_of_class_wide_pooled`);
//! - **one indexed draw**, `instance_count == 1`, `base_vertex == 0`, triangle
//!   list, single-sample;
//! - **the record that opens the packet** when the draw is one record of a
//!   multi-record `exec` packet (W1). The packet's store plan grants the guest
//!   writeback to its last record alone
//!   (`runtime::exec::multi_draw_store_plan`), so a record before it produces a
//!   frame that belongs to the chain — and the *first* one needs no frame from
//!   anywhere, because the class's own `Clear` is the pass's beginning. That
//!   record is in class ([`RenderChainRole::Head`]) and its frame goes back to
//!   the caller, which is the route the encode side already takes for every
//!   `writeback_guest == false` record (`runtime/draw/vulkan.rs` returns those
//!   pixels and `runtime::exec` hands them to the next record as its seed). A
//!   record *with* a predecessor — the packet's middle, and its last record —
//!   begins from a frame the class cannot name yet and keeps the engine;
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
//! - no depth, stencil, MSAA, MRT, resolve, blend, colour write mask,
//!   occlusion query, sampled image or sampler — and every one of those is a
//!   *reason*, not a silent downgrade;
//! - **stage buffers by the stage's own interface** (R9b): a request whose
//!   stages bind buffers directly is the v2 census's 99.3% door, and the door
//!   now answers by what each stage's *translation* declares rather than by
//!   the presence of the binds. Neither stage declaring a `[[buffer(N)]]`
//!   argument means the binds fill indices no descriptor can be needed for, so
//!   the canonical pass declares and binds nothing for them and the draw is
//!   admitted; a stage that declares one keeps the draw on the engine under
//!   the bucket its own access names (`StageBufferDeclaration`,
//!   `stage_buffer_gate`), because the canonical rail executes its
//!   pipeline-level buffer face through the reviewed fixture pair alone;
//! - **a present tail, when the caller states one** (R4b): the record owns the
//!   packet's frame and hands it to the display rail, so the canonical pass
//!   *presents* the provider's own target for the guest surface the frame lands
//!   in ([`RenderPresentRequest`]). The target is keyed by the surface's own
//!   incarnation — mapping and mapping generation, plus the attachment's shape
//!   ([`present_attachment`]) — and its bytes come back through the same
//!   completion channel the pooled arm uses, so the frame the store route lands
//!   (and therefore the frame the display rail's `capture_present_frame` reads)
//!   *is* the present target's readback. A present tail on any other position —
//!   a chain record, a withheld readback, a record naming a resident target —
//!   keeps the engine under its own name, because a present action hands on the
//!   frame it is attached to and only the packet's sole record owns the frame
//!   the guest displays;
//! - **the pooled offscreen target, or one of the resident arms** (R7b): the
//!   request is either the pooled, offscreen, CPU-readback shape
//!   (`target_identity == None`, `skip_readback == false`, no seed and no
//!   mapper backing) — which is what makes the provider's whole-attachment
//!   readback the frame the store route then lands in guest memory, or, for a
//!   chain head, the frame `exec` hands to the record after it — or a record
//!   whose attachment is the *provider-owned* image named by the request's own
//!   [`TargetIdentity`] (`research/docs/23` §76, R7):
//!   - the frame **stays** there (`StoreOp::Resident`) when the guest's own
//!     store action is published but this record's readback was withheld
//!     because the frame landed in a resident (`ReadbackSkipReason::
//!     ResidentStore`): the two rails that set that pair are the GVA render
//!     Store and the mapper-ref-texture composite Store
//!     (`runtime/draw/vulkan.rs`), and the identity is the attachment's own
//!     `(allocation, view)` pair — one pair per guest target, stable across the
//!     records of one packet, so a later record of the same chain finds the
//!     image the earlier one wrote;
//!   - the pass **begins** from it (`LoadOp::Resident`) when the request's own
//!     load action names the live GPU image rather than guest bytes
//!     (`DrawRequest::load_from_target`, the `chain_load_from_target` /
//!     mapper-currency arm the production profile measured as
//!     `draw_partial_load_from_target`).
//!
//!   The two arms are chosen by the request's own facts and never both: a
//!   request that carries guest bytes as its previous contents (a CPU seed, a
//!   guest target backing) stays on the engine under its own name, because the
//!   canonical attachment is one load op and the class refuses to state two.
//!   Nothing about the frame's *guest* landing changes here: the provider now
//!   owns the bytes a later provider pass loads, and the rails that fetch them
//!   out of the provider for the guest (GVA flush, window publish, scanout) are
//!   the R4b increment — every shape that reaches this rail is counted under
//!   `render_provider_resident_store` / `render_provider_resident_load` so that
//!   population is a number and not a silence.
//!
//!   # Routing is per record, and a chain has to be admitted whole
//!
//!   The class answers for one record, and a resident store hands the *next*
//!   record of its packet a frame that now lives in the provider. So a packet
//!   whose chain is admitted only in part — a resident store that leaves for
//!   the provider, followed by a record that stays on the engine for any other
//!   reason — ends with that later record failing by name (`read_target_unknown_
//!   identity`: the engine resident the caller's own chain would read was never
//!   written). That is a refusal, not a wrong frame, and it is the price of a
//!   per-record gate; the packet-level admission ("hold the chain only when the
//!   whole packet is in class", which the exec walk can answer and this pure
//!   gate cannot) is the named follow-up, beside R4b's byte channel.
//!
//! Anything outside the class returns [`RenderRailOutcome::NotInNarrowClass`]
//! and the caller runs the self-contained engine unchanged — the feature only
//! narrows which submissions change rail. An in-class submission the provider
//! refuses returns [`RenderRailOutcome::ProviderDeclined`] and the caller
//! declines the draw: fail-closed, never silently re-run on the engine.
//!
//! # The y convention, which both rails now state
//!
//! Metal's clip space is `+Y` up and its framebuffer rows count from the top, so
//! a guest vertex at `y = +1` belongs in row 0. The engine states that mapping in
//! the viewport it hands Vulkan — a negative height with the origin moved to the
//! bottom edge (`reims-vgpu-vulkan/src/raster.rs`) — and the comparison this rail
//! makes is in *that* vocabulary, because the frame belongs to the guest.
//!
//! The canonical rail states the same convention where it translates a guest
//! vertex stage: the provider negates the `y` of every `BuiltIn Position` output
//! on the way into SPIR-V (`metal-api-vulkan/src/lib.rs::negate_position_y`,
//! merged as `ed60380`), so the guest's `+Y` lands in row 0 under the provider's
//! positive-height viewport. The hand-written reviewed modules keep their own
//! `OpFNegate` and are not translated, so the emulator's own fixtures are
//! untouched.
//!
//! The y axis itself is pinned by the pair
//! `the_engines_asymmetric_frame_is_the_metal_ndc_mapping` (the Metal mapping,
//! derived from the guest's own vertices) and
//! `the_canonical_rails_asymmetric_frame_is_the_engines_own_frame` (the two rails
//! hand back byte-identical frames for the same asymmetric draw). Every other
//! positive case in `tests/provider_render_rail.rs` still draws a shape whose
//! *frame* is the same under the mirror (a full-screen triangle, an x-only
//! offset, an x-only scissor), which keeps those cases about their own subject
//! rather than about this axis. `research/docs/26` §18 and `research/docs/23` §80
//! record the readings on both sides of the seam.
//!
//! # The frame comes back at the attachment's own width
//!
//! A completion publishes the attachment's whole packed extent **at its own
//! texel width**: four bytes per texel for the two 8-bit orders, eight for
//! `Rgba16Float`, whose four little-endian halves are the format's own bytes and
//! not a rounding of four bytes (`research/docs/23` §78). `bgra` below names the
//! physical order of the 8-bit orders alone.
//!
//! The span the production seam returns (`M2vDrawSpan::Pixels`,
//! `runtime::draw::vulkan`) speaks eight-bit colour, so *that* boundary is where
//! a wider frame is narrowed — through the protocol's own rule, under the
//! engine's own census name for the loss (`target_read_narrowed`), exactly where
//! the engine's draw tail narrows its own wide readback
//! (`engine::narrow_readback_to_rgba8`). This rail states the bytes the canonical
//! provider published: a quantization performed here as well would be a second
//! answer to a question the span already answers, and the raw halves are what
//! make the format's own rounding observable instead of asserted.
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
    half_to_f32, AcquirePolicy, AllocationId, AllocationRecord, AttachmentFormat, BufferAccess,
    BufferSource, BufferView, ClearColor, CompiledComputePipeline, CompletionDisposition,
    CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchKind,
    DispatchType, IndexBufferBinding, IndexFormat, InitialState, LoadOp, OperationId,
    PresentDescriptor, PresentMode, PresentTarget, RenderAttachment, RenderPassDescriptor,
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
    BufferContent, ColorClearValue, DrawRequest, ReadbackSkipReason, TargetIdentity,
    VertexAttributeResource, VertexStepFunction,
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

/// The stage-buffer gate: the v2 census's 99.3% door, answered by what each
/// stage's own translation declares (`research/docs/23` §3.3, v83 /
/// `research/docs/26` §R9b).
///
/// The door used to be one condition — `!req.storage_buffers.is_empty()` — and
/// it kept every draw whose stages bind buffers on the engine, with the
/// sentence "the canonical render contract has no buffer bindings for a
/// pipeline". The contract has that face now (the E side's
/// `RenderPipelineContract::stage_buffers` / `RenderPassDescriptor::
/// stage_buffers`), but this rail registers through the *translated* gate, and
/// two refusals stand between a translated stage and that face. Both are
/// properties of the *stage*, not of the request:
///
/// 1. **registration** refuses a translated stage whose reflection names a
///    buffer at all — `render_stage_unsupported_interface`, field `bindings`,
///    whatever the access and whether or not the entry point dereferences it
///    (`metal-api-vulkan`'s `unsupported_interface_field`). The
///    translated-stage-buffer interface is the increment that lifts this;
/// 2. **execution** accepts a pass's stage buffers only when its vertex stage
///    *is* the reviewed `stage_buffer_positions` module and its fragment stage
///    the reviewed tint module — `render_stage_buffer_stage_unsupported`
///    otherwise — so a pass whose stages are translated could not be executed
///    even if the registration let it through.
///
/// So the fact this gate answers on is whether either stage declares a
/// `[[buffer(N)]]` argument, and the split is by the declaring stage's own
/// access class, in the vocabulary the engine's bind census already counts
/// (`access_unused` / `access_dereferenced` / `access_undeclared`) and the
/// canonical contract already states (`BufferAccess`):
///
/// - **neither stage declares one**: every bound stage buffer fills an index no
///   descriptor can be needed for, so the canonical pass declares and binds
///   none of them and the draw leaves for the provider. Both rails land the
///   same bytes for it, because the bytes that differ are the bytes no stage
///   reads — which is the falsifiable half of this population (`provider_
///   render_rail.rs`: the sentinel bind moves nothing, the stream bytes beside
///   it move the frame);
/// - **a stage declares one**: the draw stays on the engine under the bucket
///   its declaration's access names, so the census says which increment lifts
///   it instead of counting one 99.3% door.
///
/// The count of binds rides in the sentence rather than in the slug: the slug
/// is the census bucket and has to stay a property of the *shape*, while the
/// count is a property of this request.
fn stage_buffer_gate(inputs: &RenderRailInputs<'_>, binds: usize) -> Result<(), OutOfClass> {
    for (stage, declarations) in [
        ("vertex", inputs.vertex_stage_buffer_declarations),
        ("fragment", inputs.fragment_stage_buffer_declarations),
    ] {
        // One declaration anywhere in either stage is enough to keep the draw
        // on the engine, and the vertex half is asked first: that is the order
        // the contract states its two buffers in, and the order a reader of the
        // sentence expects to see the stage named in.
        let Some(declaration) = declarations.first() else {
            continue;
        };
        let (slug, access) = match declaration.class {
            StageBufferDeclarationClass::Unused => {
                ("render_provider_out_of_class_stage_buffer_unused", "unused")
            }
            StageBufferDeclarationClass::ReadOnly => (
                "render_provider_out_of_class_stage_buffer_read",
                "read-only",
            ),
            StageBufferDeclarationClass::Writable => (
                "render_provider_out_of_class_stage_buffer_write",
                "writable",
            ),
            StageBufferDeclarationClass::Unknown => (
                "render_provider_out_of_class_stage_buffer_unknown",
                "of an unclassified access",
            ),
        };
        return Err(OutOfClass::owned(
            slug,
            format!(
                "a draw whose {stage} stage declares a [[buffer({})]] argument ({access}) stays \
                 on the engine: the canonical contract states pipeline-level buffers (v83) but \
                 its Vulkan rail executes that face through the reviewed stage-buffer pair \
                 alone, and a translated stage that names a buffer is refused by name at \
                 registration (`render_stage_unsupported_interface`, field `bindings`) before \
                 any descriptor exists. The request binds {binds} stage buffer(s); the \
                 translated-stage-buffer interface is the increment that lifts this population",
                declaration.index,
            ),
        ));
    }
    Ok(())
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

/// The allocation namespace resident identities are minted from.
///
/// One allocation per guest target, far above both the pooled attachment's own
/// constant and the input-view namespace, so a resident image can never alias a
/// view this rail minted for one submission's streams. The registry behind it is
/// what makes the pair *the attachment's own*: the same guest target always
/// resolves to the same `(allocation, view)` — across the records of one packet
/// and across packets — and two distinct targets never share one.
const RESIDENT_ALLOCATION_BASE: u64 = 0x7265_7369_0000_0000;

/// The view identity of a resident attachment. One view per allocation: the
/// pair the canonical contract keys a resident target on is `(allocation,
/// view)`, and the allocation is the half that varies per guest target.
const RESIDENT_VIEW: ViewId = ViewId::new(1);

/// The allocation namespace present target identities are minted from (R4b).
///
/// One present target per guest *surface incarnation*, in its own namespace
/// rather than the resident one: a present target is the image the provider
/// acquires and presents for a display surface, and the resident mint is the
/// image a render record keeps. They are different provider registries (and, on
/// the emulator side, different lifetimes — the present registry is LRU-bounded
/// on its own), so one namespace's collision would be invisible in the other's.
const PRESENT_ALLOCATION_BASE: u64 = 0x7265_7072_0000_0000;

/// The view identity of a present target. One view per allocation for the same
/// reason [`RESIDENT_VIEW`] is one: the pair the provider keys the target on is
/// `(allocation, view)`, and the allocation is the half that varies per surface.
const PRESENT_VIEW: ViewId = ViewId::new(1);

/// The provider-owned image one record names, in the two fields the canonical
/// contract keys a resident target on.
///
/// This is not a second naming channel beside `(allocation, view)`: it *is* the
/// pair the trace declares for the attachment, carried out of the rail so a
/// caller (and a test) can ask the provider whether that identity is resident
/// (`VulkanComputeProvider::resident_target_is_live`) without re-deriving it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentAttachment {
    pub allocation: AllocationId,
    pub view: ViewId,
}

/// The process-global mint for resident identities.
///
/// A hash of the `TargetIdentity` would have been simpler and is the reason
/// this is a map: two guest targets that collide would silently share one
/// provider image, and the failure mode of a collision is a frame read from the
/// wrong surface — the exact class of defect the resident rail's own documents
/// call out. A counter cannot collide, and the map is what keeps the mint
/// *stable*: the same target has to resolve to the same image on every record
/// of its chain, or the load would be refused by name
/// (`resident_target_undeclared`) instead of finding the frame.
struct ResidentIdentities {
    next: u64,
    by_target: HashMap<TargetIdentity, ResidentAttachment>,
}

static RESIDENT_IDENTITIES: OnceLock<Mutex<ResidentIdentities>> = OnceLock::new();

/// The provider image the given guest target renders into.
///
/// Memoised, so the answer is stable for the process lifetime; the rail's
/// provider is itself a process singleton (`render_rail`), so the two lifetimes
/// agree.
pub fn resident_attachment(target: &TargetIdentity) -> ResidentAttachment {
    let registry = RESIDENT_IDENTITIES.get_or_init(|| {
        Mutex::new(ResidentIdentities {
            next: 0,
            by_target: HashMap::new(),
        })
    });
    let mut registry = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(attachment) = registry.by_target.get(target) {
        return *attachment;
    }
    let allocation = AllocationId::new(RESIDENT_ALLOCATION_BASE + registry.next);
    registry.next += 1;
    let attachment = ResidentAttachment {
        allocation,
        view: RESIDENT_VIEW,
    };
    registry.by_target.insert(target.clone(), attachment);
    attachment
}

/// The guest surface one presenting record hands to the display rail (R4b).
///
/// The two fields are the surface's *incarnation*: the guest mapping the frame
/// lands in (`runtime::draw::vulkan`'s `c0.mapping_id`, the same mid
/// `mapping_write` stores into) and that mapping's generation at the draw. The
/// generation is what makes a re-mapped surface a different provider target
/// instead of a reused image: `present_identity.rs` records why two guest
/// surfaces sharing one resident fuses their damage histories into the
/// rubber-band residue class, and a generation is exactly a surface that came
/// back as another buffer.
///
/// Deliberately *not* carrying geometry or format: both are the pass's own
/// facts, and the mint below reads them from the attachment the same
/// `narrow_class` pass states — one source, so a caller cannot describe a
/// surface whose shape disagrees with the pass it is attached to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentSurfaceKey {
    pub mapping_id: u32,
    pub map_generation: u32,
}

/// The provider-owned present target one presenting record hands on, in the two
/// fields the canonical contract keys a present target on
/// (`PresentTarget::allocation_id` / `view_id`).
///
/// The pair is also the attachment's own identity for that pass: the contract
/// requires the present target's view and allocation to *be* the pass's colour
/// attachment's (`PresentDescriptor::validate_against`), so the completion's
/// writeback, the provider's present registry and the trace's attachment all
/// name one resource. There is no second key to keep in step, which is what
/// this type is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentAttachment {
    pub allocation: AllocationId,
    pub view: ViewId,
}

/// The present tail one record states for the frame it owns (R4b).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderPresentRequest {
    /// The guest surface the frame lands in. See [`PresentSurfaceKey`] for why
    /// this is a mapping and a generation rather than a provider identity.
    pub surface: PresentSurfaceKey,
}

/// What one present-bearing submission reports back.
///
/// The counters are the provider's own acquire/present deltas as the rail read
/// them around the submission; both are exactly one by the time this value
/// exists, because the rail refuses the submission rather than reporting a
/// tail that did not run ([`ProviderRenderDecline::PresentNotExecuted`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentCompletion {
    /// The target the frame was presented from — the same pair the completion's
    /// writeback landed under.
    pub attachment: PresentAttachment,
    pub acquires: usize,
    pub presents: usize,
}

/// The mint key of one present target: the surface's incarnation *and* the
/// attachment's shape.
///
/// Shape is part of the key because the provider reuses a present target by
/// `(allocation, view)` alone (`metal-api-vulkan`'s `present_target`): an
/// identity reused at another extent or format would hand the pass an image of
/// the old shape. Keying the mint on the shape the class states makes that
/// unrepresentable rather than something a later record could notice.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PresentMintKey {
    surface: PresentSurfaceKey,
    /// [`AttachmentFormat::code`], because the provider's format enum is not
    /// `Hash` — and the code is the value the wire format and the census use.
    format: u8,
    width: u64,
    height: u64,
}

/// The process-global mint for present target identities.
///
/// A counter behind a map, for the reason [`ResidentIdentities`] is one: two
/// surfaces that collided would silently present out of one provider image, and
/// the failure mode of that collision is a frame from the wrong surface. The
/// map is what keeps the mint *stable*: every record that presents the same
/// surface incarnation resolves to the same target, so the provider's registry
/// reuses one image (and one layout lock) instead of accumulating one per
/// submission.
struct PresentIdentities {
    next: u64,
    by_surface: HashMap<PresentMintKey, PresentAttachment>,
}

static PRESENT_IDENTITIES: OnceLock<Mutex<PresentIdentities>> = OnceLock::new();

/// The provider-owned present target the given surface incarnation presents
/// from at the given shape.
///
/// Memoised for the process lifetime, exactly as [`resident_attachment`] is: the
/// provider is a process singleton ([`render_rail`]), so the two lifetimes
/// agree. A surface that changes geometry or format under one generation is a
/// different mint key and therefore a different target — see [`PresentMintKey`].
pub fn present_attachment(
    surface: &PresentSurfaceKey,
    format: AttachmentFormat,
    width: u64,
    height: u64,
) -> PresentAttachment {
    let key = PresentMintKey {
        surface: *surface,
        format: format.code(),
        width,
        height,
    };
    let registry = PRESENT_IDENTITIES.get_or_init(|| {
        Mutex::new(PresentIdentities {
            next: 0,
            by_surface: HashMap::new(),
        })
    });
    let mut registry = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(attachment) = registry.by_surface.get(&key) {
        return *attachment;
    }
    let allocation = AllocationId::new(PRESENT_ALLOCATION_BASE + registry.next);
    registry.next += 1;
    let attachment = PresentAttachment {
        allocation,
        view: PRESENT_VIEW,
    };
    registry.by_surface.insert(key, attachment);
    attachment
}

/// How one admitted pass establishes the attachment's previous contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NarrowLoad {
    /// Fill every texel with the guest's byte-exact clear, carried as **one
    /// texel of the attachment's own format** ([`ClearColor`]): four bytes for
    /// the two 8-bit orders, four little-endian halves for `Rgba16Float`.
    /// Widening the payload with the attachment is what keeps the contract's
    /// `clear.len() == format.bytes_per_texel()` rule satisfied rather than
    /// re-derived here.
    Clear(ClearColor),
    /// Keep the bytes the provider already holds under the attachment's own
    /// identity (`LoadOp::Resident`).
    Resident(ResidentAttachment),
}

/// Where one admitted pass's frame goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NarrowStore {
    /// The completion publishes the whole extent, and the caller's store route
    /// lands it.
    Writeback,
    /// The frame stays in the provider's image under the attachment's own
    /// identity, and no writeback is published (`StoreOp::Resident`).
    Resident(ResidentAttachment),
}

/// Where one record sits in the packet the exec loop walks.
///
/// The 2026-09-17 probe (`evidence/reviews/writeback-class-probe-2026-09-17.md`)
/// read the class's old `writeback` condition as a *position* rather than a
/// shape: `writeback_guest` is `multi_draw_store_plan`'s `do_writeback`, true
/// for a packet's last record alone, so the records the class refused under
/// that one name were the packet's heads (2653) and its middles (2065). Naming
/// the position once, here, is what lets the gate answer about a record's place
/// in its chain rather than about a boolean whose meaning has to be looked up —
/// and it is what retires the census's `chain_head` bucket (those records are
/// admitted now) while leaving `chain_middle` in place beside it.
///
/// The successor fact (`render_pass_continues`) is deliberately not part of the
/// role: the class's question is where a record's *frame* goes — to guest
/// memory ([`Self::SoleOrTail`]) or back to the caller ([`Self::Head`] and
/// [`Self::Middle`]) — and whether the guest has a record after this one is the
/// caller's business, because the frame comes back either way. The census row
/// still prints that fact beside the role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderChainRole {
    /// The record owns the guest writeback: the packet's only record, or its
    /// last one. Whether it may begin from a frame produced before it is the
    /// encoder gate's question and not the role's.
    SoleOrTail,
    /// The record opens the packet: no frame has to be seeded into it, and its
    /// own frame goes to the record after it.
    Head,
    /// A record with a predecessor: it begins from a frame the record before it
    /// produced.
    Middle,
}

impl RenderChainRole {
    /// One record's role, from the two facts the exec packet walk sets for it:
    /// `do_writeback` (`runtime::exec::multi_draw_store_plan`) and
    /// `continues_render_pass` (`runtime::exec::render_pass_chain_position`).
    pub fn of(writeback_guest: bool, continues_render_pass: bool) -> Self {
        match (writeback_guest, continues_render_pass) {
            (true, _) => Self::SoleOrTail,
            (false, false) => Self::Head,
            (false, true) => Self::Middle,
        }
    }
}

/// What one stage's own translation says about a `[[buffer(N)]]` argument it
/// declares (`research/docs/23` §3.3, v83 / `research/docs/26` §R9b).
///
/// The classes are the render bind census's own
/// (`runtime::bind_phase`'s `access_unused` / `access_dereferenced` /
/// `access_undeclared` over the engine's `ReflectedBufferAccess`) minus the one
/// class that is no declaration at all: a bind reflection does not mention is
/// *absent*, and that is exactly the population this class admits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageBufferDeclarationClass {
    /// Declared, and the specialized entry point never dereferences it. The
    /// canonical registration still refuses the stage on the declaration's own
    /// presence (`render_stage_unsupported_interface`, field `bindings`), so
    /// this class names a population rather than a cheaper admission.
    Unused,
    /// Declared and read — `BufferAccess::Read`'s Metal spelling.
    ReadOnly,
    /// Declared and writable: the writeback landing neither rail carries for a
    /// stage buffer yet.
    Writable,
    /// Declared without a usable access answer: fail closed, exactly as the
    /// engine's own bind path does.
    Unknown,
}

/// One `[[buffer(N)]]` argument one stage's translation declares.
///
/// The index is the stage's *own* Metal buffer index space — the one
/// `setVertexBuffer(_:offset:index:)` and `setFragmentBuffer(_:offset:index:)`
/// name, and the one the canonical contract's `StageBufferBinding::index`
/// speaks — so a vertex declaration and a fragment declaration at one index are
/// two different arguments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageBufferDeclaration {
    pub index: u32,
    pub class: StageBufferDeclarationClass,
}

/// What one render submission needs to leave this rail: the two stage
/// modules' AIR, the entries the translation reports for them, and this
/// record's place in the chain it belongs to.
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
    /// This record's place in the packet its encode belongs to
    /// ([`RenderChainRole`]). The position decides whether the class executes
    /// it, because what the class has to be able to name is where the frame
    /// goes: to guest memory, or back to the caller that owns the chain.
    pub role: RenderChainRole,
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
    /// The present tail this record carries, when the caller states one (R4b).
    ///
    /// `None` is the pre-R4b device: the record's frame comes back through the
    /// pooled or resident arms and no present action is stated. `Some` asks this
    /// class for the shape the display rail reads — see [`RenderPresentRequest`]
    /// — and every position or store this class cannot present keeps the engine
    /// by name.
    pub present: Option<RenderPresentRequest>,
    /// The `[[buffer(N)]]` arguments each stage's own translation declares
    /// (`CachedShader::reflection.bindings`, filtered to Metal's buffer kind),
    /// with the access that reflection reports for each.
    ///
    /// Carried for the reason [`Self::vertex_attribute_locations`] is: the
    /// canonical provider refuses a *translated* stage whose reflection names a
    /// buffer (`render_stage_unsupported_interface`, field `bindings`, before
    /// any descriptor exists) and executes stage buffers only through its
    /// reviewed fixture pair — so a request whose stage declares one is a shape
    /// this class does not execute, while a request whose stages declare none
    /// leaves for the provider with its binds carried by nothing at all.
    pub vertex_stage_buffer_declarations: &'a [StageBufferDeclaration],
    pub fragment_stage_buffer_declarations: &'a [StageBufferDeclaration],
}

/// What one completed narrow-class submission returns.
#[derive(Debug)]
pub struct RenderRailOutput {
    /// The attachment's whole packed extent, at the attachment's **own texel
    /// width**: four bytes per texel in the order `bgra` names for the two
    /// 8-bit orders, eight bytes of four little-endian halves in RGBA for
    /// `Rgba16Float` (`research/docs/23` §78).
    ///
    /// The span the seam returns speaks eight-bit colour, so a wider frame is
    /// narrowed there and not here — the module docs say why, and
    /// [`ProviderRenderDecline::AttachmentFrameNotNarrowable`] is the name for
    /// a frame that cannot make that trip.
    pub bytes: Vec<u8>,
    /// Whether those bytes are guest scanout order (BGRA), which is what the
    /// store route needs to know before it can land them. The wide arm is RGBA,
    /// so this is `false` for it by construction.
    pub bgra: bool,
    /// The present action this submission executed, when it stated one (R4b).
    ///
    /// The bytes above are the presented target's own readback whenever this is
    /// `Some`, which is what lets the caller say *which* frame the store route
    /// is about to land.
    pub present: Option<PresentCompletion>,
}

/// What one resident-class submission leaves behind.
///
/// There are no bytes: the whole point of `StoreOp::Resident` is that the frame
/// stays in the provider's image and no writeback is published for it. What the
/// caller gets instead is the identity the frame stayed under, so it can name
/// the same image the engine rail would have named — and whether this pass also
/// began from that image, which is what distinguishes a seeding store from a
/// record that composited onto the frame before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentFrame {
    pub attachment: ResidentAttachment,
    /// `true` when the pass declared `LoadOp::Resident` as well, i.e. it read
    /// the image it then kept.
    pub loaded: bool,
}

/// Why one reims render request did not leave this rail for the engine.
#[derive(Debug)]
pub enum RenderRailOutcome {
    /// The canonical provider executed the pass.
    ProviderCompleted(RenderRailOutput),
    /// The canonical provider executed the pass into the provider-owned image
    /// the attachment names, and published no writeback for it.
    ProviderCompletedResident(ResidentFrame),
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
    /// A record stated a present tail and the provider's acquire/present
    /// counters did not advance by exactly one each (R4b).
    ///
    /// The frame comes back through the same completion either way, so this is
    /// the one failure that would otherwise be silent: bytes that look like the
    /// frame while the action that makes it the *presented* frame never ran.
    /// The observed deltas travel with the refusal because they are the whole
    /// answer — a zero pair says the action did not run, and a pair above one
    /// says the counters moved for a submission this one did not drive.
    PresentNotExecuted { acquires: usize, presents: usize },
    /// The completion's frame carries a texel the seam's span cannot speak: the
    /// attachment's format is wider than four bytes per texel and the engine's
    /// own widening rule has no four-byte meaning for it.
    ///
    /// Raised at the seam's span conversion (`runtime::draw::vulkan`), not inside
    /// this rail, which publishes the provider's bytes exactly as it received
    /// them. Named rather than guessed: a frame read as the wrong texel width is
    /// a wrong picture, not a quantized one. The format is carried because it is
    /// the whole answer — the frame above it was well formed.
    AttachmentFrameNotNarrowable { format: ash::vk::Format },
    /// A resident store's completion published a writeback for the attachment.
    ///
    /// `StoreOp::Resident` is defined by *not* publishing one — the frame stays
    /// in the provider's image and the pass that later loads it is what makes
    /// the bytes observable. A provider that published one anyway would leave
    /// this rail unable to say where the frame is, so the shape is declined by
    /// name rather than accepted under either reading.
    ResidentWritebackPublished {
        allocation: AllocationId,
        view: ViewId,
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
            Self::AttachmentFrameNotNarrowable { .. } => "attachment_frame_not_narrowable",
            Self::PresentNotExecuted { .. } => "render_present_not_executed",
            Self::ResidentWritebackPublished { .. } => "resident_writeback_published",
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
            Self::AttachmentFrameNotNarrowable { format } => {
                vec![("format", format!("{format:?}"))]
            }
            Self::PresentNotExecuted { acquires, presents } => vec![
                ("acquires", acquires.to_string()),
                ("presents", presents.to_string()),
            ],
            Self::ResidentWritebackPublished { allocation, view } => vec![
                ("allocation", format!("{:#x}", allocation.get())),
                ("view", view.get().to_string()),
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

/// The canonical provider's own presentation counters, as the rail's tests read
/// them (R4b).
///
/// Test observation, like [`provider_submissions`]: production callers do not
/// read it. `(0, 0)` when this rail's provider has never been built, which is
/// the honest answer for a process that presented nothing rather than a second
/// counter this module would have to keep in step with the emulator's.
pub fn present_counts() -> (usize, usize) {
    rail()
        .map(|rail| rail.provider.present_counts())
        .unwrap_or((0, 0))
}

/// How many provider-owned present targets the rail's provider currently holds.
///
/// Test observation: the rail test that pins target *reuse* reads it, so a
/// second present of one surface incarnation is a statement about the
/// provider's registry rather than an inference from the counters.
pub fn present_target_count() -> usize {
    rail()
        .map(|rail| rail.provider.present_target_count())
        .unwrap_or(0)
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
    // The bound stage-buffer axis (R9b), charged for the same reason and at the
    // same place: the door's four buckets answer *why* a draw stayed on the
    // engine, and this answers *how many* buffers were behind it — the split
    // the v2 census could not make (its §7.3).
    crate::runtime::drain::note_store_route(stage_buffer_count_route(req.storage_buffers.len()));
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
        Ok(RenderCompletion::Writeback(output)) => RenderRailOutcome::ProviderCompleted(output),
        Ok(RenderCompletion::Resident(frame)) => {
            RenderRailOutcome::ProviderCompletedResident(frame)
        }
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

/// The census band of one request's *bound* stage buffers (`research/docs/26`
/// §R9b).
///
/// The census could say that 99.3% of a boot's draws hit the buffer door and
/// nothing about the binds behind it — "`buffers` 不拆分 stage", its own §7.3 —
/// because the seam had one boolean for the whole population. This is the
/// reading that replaces the boolean: charged for every request the gate is
/// handed, exactly as [`attribute_count_route`] is, so the band's population
/// and the door's four buckets are the same draws. The bands break at the
/// canonical contract's own `MAX_RENDER_STAGE_BUFFERS` (4), which is what makes
/// "two to four" one arm rather than an arbitrary split.
pub fn stage_buffer_count_route(binds: usize) -> &'static str {
    match binds {
        0 => "draw_stage_buffers_0",
        1 => "draw_stage_buffers_1",
        2..=4 => "draw_stage_buffers_2_4",
        _ => "draw_stage_buffers_gt4",
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
    /// The previous-contents arm this pass declares ([`NarrowLoad`]).
    load: NarrowLoad,
    /// Where this pass's frame goes ([`NarrowStore`]).
    store: NarrowStore,
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
    /// The present target this pass hands on, or `None` for the pooled/resident
    /// arms (R4b). Present whenever the caller stated a present tail; the class
    /// conditions above are what make that the *only* shape a presenting record
    /// can reach this point with.
    present: Option<PresentAttachment>,
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
    match inputs.role {
        // W1. The packet's first record owns the pass's own beginning — the
        // class's `Clear` — so no frame has to be seeded into it, and its frame
        // goes back to the caller that owns the chain, which is the route the
        // encode side already takes for every `writeback_guest == false`
        // encode. Admitting it is also what turns the probe's 4718-wide
        // "unknown" into a distribution: a head that fails any other condition
        // of this class now answers with *that* condition's name.
        RenderChainRole::Head => {}
        // A record with a predecessor begins from a frame this class cannot
        // name. The middle is the one position that both takes and hands on a
        // frame, so it has no route here at all; the packet's last record takes
        // one too, and answers for that at the encoder gate below.
        //
        // R7b: the middle's frame now *is* nameable when the request says it
        // begins from the live GPU image (`load_from_target`) under a target
        // identity — `LoadOp::Resident` names the source and `StoreOp::Resident`
        // the sink, so the record neither reads nor writes guest bytes. The
        // admission here is deliberately the request's own pair and not the
        // computed arms: the deeper conditions below still answer for a middle
        // that names a resident and then fails on its streams, scissor or
        // format, and the census keeps those reasons.
        RenderChainRole::Middle if req.load_from_target && req.target_identity.is_some() => {}
        RenderChainRole::Middle => {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_chain_middle",
                "a record in the middle of a multi-record packet stays on the engine: it begins \
                 from the frame the record before it produced, which is a rail this class does \
                 not execute",
            ))
        }
        RenderChainRole::SoleOrTail => {}
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
        ash::vk::Format::R16G16B16A16_SFLOAT => AttachmentFormat::Rgba16Float,
        _ => {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_format",
                "only the admitted colour formats leave for the canonical rail \
                 (Rgba8Unorm, Bgra8Unorm, Rgba16Float): the format is the attachment's own \
                 fact, and a pass the canonical contract cannot state is one this class \
                 hands back to the engine",
            ))
        }
    };
    let ColorClearValue::Float(clear) = attachment.clear() else {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_clear_kind",
            "a non-float clear stays on the engine",
        ));
    };
    let Some(clear) = clear_bytes(format, clear) else {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_clear_bytes",
            "a clear that is not byte-exact in the attachment's own texel stays on the engine: \
             the canonical pass states one texel of bytes, the engine states floats, and the \
             two rounds must be the same value for the rails to agree (eight bits per channel \
             for the 8-bit orders, one half per channel for Rgba16Float)",
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
        .and_then(|texels| texels.checked_mul(format.bytes_per_texel()))
        .ok_or(OutOfClass::new(
            "render_provider_out_of_class_extent_overflow",
            "an attachment extent that overflows u64 stays on the engine",
        ))?;
    // The present tail (R4b): the record hands the frame it owns to the display
    // rail through a provider-owned target. The three conditions below are the
    // whole shape rule — everything else about the record is answered by the
    // conditions above and below this block, unchanged.
    //
    // 1. Only the packet's *sole* record owns the frame the guest displays. A
    //    chain head's frame belongs to the record after it, and a middle's to
    //    the one after that, so presenting either would hand the display a
    //    picture the guest never finished. That is a class answer rather than a
    //    decline: the record is a shape this class does not execute at all
    //    (the encoder gate below refuses it for the frame's own reason, and
    //    this names the position the caller asked to present).
    if inputs.present.is_some() {
        if inputs.role != RenderChainRole::SoleOrTail || req.continues_render_pass {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_present_position",
                "a present tail on a record that is not the packet's sole record stays on the \
                 engine: the presented frame has to be the one the guest displays, and every \
                 other record of a chain hands its frame to the record after it",
            ));
        }
        // 2. The frame has to come back to this rail, or the display rail has
        //    nothing to read: a present action whose record withholds its
        //    readback would keep the target's bytes in the provider with no
        //    consumer — the one thing this increment is for.
        if req.skip_readback {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_present_unpublished",
                "a present tail on a record whose readback is withheld stays on the engine: the \
                 provider would present a target whose bytes never come back, and the display \
                 rail would have nothing to capture",
            ));
        }
        // 3. A record that names a resident target is the emulator's own
        //    `resident_target_present_unsupported` shape: the present action
        //    hands on the provider's own target, and a pass that also declares
        //    the render rail's resident would ask two registries to own one
        //    identity. Answered here so the draw falls back to the engine
        //    instead of being declined by the provider.
        if req.target_identity.is_some() {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_present_target",
                "a present tail on a record that names a resident target stays on the engine: \
                 the present action hands on the provider's own target, and the canonical \
                 provider refuses a pass that declares the resident beside it \
                 (`resident_target_present_unsupported`)",
            ));
        }
    }
    // The provider target a presenting record hands on, keyed by the surface
    // incarnation the caller named and the attachment's own shape. Derived here
    // rather than in `submit_narrow` so the present conditions above and the
    // pass the trace states cannot disagree about which surface was presented.
    let present = inputs.present.map(|present| {
        present_attachment(
            &present.surface,
            format,
            u64::from(req.width),
            u64::from(req.height),
        )
    });
    // The provider image this record's own attachment names, when the request
    // carries a guest target at all. Derived once, from the request, so the load
    // arm and the store arm below cannot name two images for one attachment.
    let resident = req.target_identity.as_ref().map(resident_attachment);
    // Where this record's previous contents come from — the request's own load
    // action, read in the order `DrawRequest` documents: `load_from_target`
    // wins, else the declared action decides.
    //
    // The two arms are the contract's two attachment loads: the *provider's*
    // image under the attachment's own identity (`LoadOp::Resident`), or the
    // guest's byte-exact clear. A record whose previous contents are guest bytes
    // is a third shape the canonical attachment cannot state beside either (one
    // load op per attachment), so it keeps the engine by name.
    let load = if req.load_from_target {
        let Some(resident) = resident else {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_resident_identity",
                "a record that loads the live GPU image stays on the engine when it names no \
                 target identity: the canonical attachment's resident load is keyed on the \
                 attachment's own (allocation, view), and a pair this rail invented would be an \
                 image no record ever stored",
            ));
        };
        if req.target_rgba8.is_some() || req.target_guest_seed.is_some() {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_load_source",
                "a record that carries guest bytes *and* names the live GPU image stays on the \
                 engine: the canonical attachment states one load op, and the two sources \
                 disagree about which bytes the pass begins from",
            ));
        }
        NarrowLoad::Resident(resident)
    } else {
        if req.target_rgba8.is_some()
            || req.target_guest_seed.is_some()
            || req.load_guest_target_backing
        {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_load_seed",
                "a record whose previous contents are guest bytes stays on the engine: stating \
                 them would be the canonical attachment's trace-owned `Load` arm, which this \
                 class does not carry (its streams travel as bytes, its previous contents do \
                 not)",
            ));
        }
        if req.color0_declared != Some(crate::protocol::pass_action::LoadAction::Clear) {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_load_action",
                "the canonical class loads by `Clear` or from the provider's own image; a Load \
                 or DontCare record whose previous contents this rail cannot name stays on the \
                 engine",
            ));
        }
        NarrowLoad::Clear(clear)
    };
    // Where this record's frame goes. A record that skipped its readback is one
    // of two rails, and the flag's recorded reason is what tells them apart —
    // re-deriving the split from the assembler around it is what the 2026-09-17
    // probe had to do:
    //
    // - `ResidentStore` is a frame the record left in an image, and the
    //   canonical attachment can name that image by the attachment's own
    //   `(allocation, view)`: `StoreOp::Resident`, no writeback published. The
    //   request has to carry the identity the image is keyed on; a resident
    //   store without one is a wiring bug named as such rather than a
    //   differently-shaped pass.
    // - `UnpublishedStore` is a store action that publishes nothing: no
    //   resident, no reader, and no writeback the caller could land. It keeps
    //   the engine until its route is reviewed, exactly as before.
    let store =
        if req.skip_readback {
            match req.readback_skip_reason {
                ReadbackSkipReason::ResidentStore => {
                    let Some(resident) = resident else {
                        return Err(OutOfClass::new(
                            "render_provider_out_of_class_resident_identity",
                            "a record whose frame stays in a resident stays on the engine when it \
                         names no target identity: the canonical attachment keys its resident on \
                         the attachment's own (allocation, view), and a pair this rail invented \
                         would be an image no record ever stored",
                        ));
                    };
                    NarrowStore::Resident(resident)
                }
                ReadbackSkipReason::UnpublishedStore => return Err(OutOfClass::new(
                    "render_provider_out_of_class_unpublished_store",
                    "a record whose store action publishes nothing stays on the engine: the class \
                     reads its frame back from the completion, and a store that lands nowhere is \
                     a route this increment has not reviewed",
                )),
                // A skip with no recorded reason is a fact about the caller rather
                // than a third rail, and guessing which of the two it is would name
                // the wrong population.
                ReadbackSkipReason::None => return Err(OutOfClass::new(
                    "render_provider_out_of_class_skip_readback",
                    "a record that skips its readback for a reason this rail cannot name stays on \
                     the engine",
                )),
            }
        } else {
            NarrowStore::Writeback
        };
    // A guest target identity this pass neither loads from nor keeps is a name
    // without an image: the seams that answer that identity's later readers
    // would be looking for an image the provider never made, and a class that
    // drew the frame anyway would be handing the guest a picture its own rails
    // cannot find. The two resident arms above are the only shapes here that
    // name an image, and they name it by construction — either one is enough.
    if resident.is_some()
        && !matches!(load, NarrowLoad::Resident(_))
        && !matches!(store, NarrowStore::Resident(_))
    {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_target",
            "a record that names a target identity without loading from it or keeping its frame \
             stays on the engine: the class cannot say which image that name refers to",
        ));
    }
    // A seed copied from *another* resident is a channel of its own: the bytes
    // travel between two images, which is a copy the class would have to state
    // as a pass rather than as an attachment load.
    if req.seed_from_target.is_some() {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_seed",
            "a record that seeds itself from another resident stays on the engine: the copy \
             between two images is not an attachment load the canonical pass can state",
        ));
    }
    // A texel wider than four bytes is a shape the *frame's* two rails answer
    // for differently unless the frame stays in an image the provider owns.
    // `Rgba16Float` is eight bytes per texel, and the engine's pooled offscreen
    // target is its own four-byte image (`translate::pixel::RESIDENT_RGBA_FORMAT`;
    // `TargetKey` carries no format at all), so a pooled wide attachment is one
    // the engine cannot draw — measured, not assumed: the pass lands nothing and
    // the readback comes back empty. This class only narrows *which submissions
    // change rail*; it may not answer for a shape the engine has no answer to,
    // so such a record keeps the engine under its own name. The two resident
    // arms are the other side of that line: there the frame is the provider's
    // own image, created at the attachment's own format, and whether it is kept
    // (`StoreOp::Resident`) or read back (a published store on a loaded
    // resident) it is the frame both rails would land.
    if format.bytes_per_texel() > ClearColor::BYTES as u64
        && !matches!(load, NarrowLoad::Resident(_))
        && !matches!(store, NarrowStore::Resident(_))
    {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_wide_pooled",
            "a colour attachment wider than four bytes per texel whose frame comes back through \
             this class's pooled readback stays on the engine: the pooled offscreen target is \
             the engine's own four-byte image, so this is a shape the engine does not draw, and \
             the class may not answer for it. A wide attachment is in class where the frame \
             comes from the provider's own image — the resident load and store arms",
        ));
    }
    // A guest-backed attachment's home is the guest's own pages. The class
    // renders into the provider's image; a record whose *load* names that
    // backing already took the resident arm above, and a record that would
    // render straight into the guest's memory is a landing this rail cannot
    // perform.
    if req.guest_target_memory.is_some() && !req.load_from_target {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_guest_backing",
            "a record whose attachment is backed by the guest's own pages stays on the engine: \
             the class renders into a provider image, and writing the guest's pages from it is a \
             landing this rail does not carry",
        ));
    }
    // W1 named the frame's *destination* for the record that opens a packet;
    // this is its source. A record that continues an encoder begins from the
    // frame the record before it produced, and the class can execute it exactly
    // when that frame is the provider's own image — `LoadOp::Resident`, chosen
    // above from the request's `load_from_target`. A continuing record that
    // would begin from guest bytes, or from nothing at all, keeps the engine.
    if req.continues_render_pass && !matches!(load, NarrowLoad::Resident(_)) {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_encoder",
            "a record that continues a multi-record encoder stays on the engine: it begins from \
             the frame the record before it produced, which this class can only name when that \
             frame is the provider's own image",
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
    // R9b: whether the request *binds* buffers is no longer the condition. What
    // decides which rail executes the draw is whether either stage's own
    // translation *declares* a `[[buffer(N)]]` argument, and both answers live
    // in `stage_buffer_gate`.
    stage_buffer_gate(inputs, req.storage_buffers.len())?;
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
        load,
        store,
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
        present,
    })
}

/// A clear that survives the round trip through the attachment's own texel.
///
/// The engine hands Vulkan `float32` components and the driver rounds them *to
/// the attachment's format*; the canonical pass states the texel's bytes. `Some`
/// only when that rounding is the identity — an exact multiple of `1/255` for
/// the two 8-bit orders, a value a half holds exactly for `Rgba16Float` — so
/// "the two rails drew the same colour" is a byte comparison rather than a
/// tolerance.
///
/// The payload is one texel wide, which is what the contract checks against the
/// carrying attachment (`RenderAttachment::validate_shape` →
/// `AttachmentClearLengthMismatch`); the width here is the format's own
/// [`AttachmentFormat::bytes_per_texel`] and never a constant of this module.
fn clear_bytes(format: AttachmentFormat, clear: [f32; 4]) -> Option<ClearColor> {
    match format {
        AttachmentFormat::Rgba8Unorm | AttachmentFormat::Bgra8Unorm => {
            let mut out = [0u8; ClearColor::BYTES];
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
            Some(ClearColor::new(out))
        }
        AttachmentFormat::Rgba16Float => {
            // The contract's eight bytes are four little-endian halves in
            // memory order (`research/docs/23` §78), which is the order both
            // rails decode them in.
            let mut out = [0u8; 8];
            for (index, component) in clear.iter().enumerate() {
                let half = f32_to_half_bits(*component);
                // The widening back is the whole rule: an encoding that does not
                // reproduce the component bit for bit is one the driver's own
                // rounding could have chosen differently, and a clear the two
                // rails round differently is a different colour on each. NaN
                // fails here for free (`NaN != NaN`), which is right — a NaN
                // clear's payload is not a value either rail could agree on.
                if half_to_f32(half) != *component {
                    return None;
                }
                out[index * 2..index * 2 + 2].copy_from_slice(&half.to_le_bytes());
            }
            ClearColor::from_bytes(&out)
        }
        // Unreachable through `narrow_class`, which answers
        // `render_provider_out_of_class_format` for every format this class does
        // not admit; spelled out so a format added to that gate has to answer
        // here as well rather than inherit an arm by accident.
        AttachmentFormat::R32Float | AttachmentFormat::R32Uint => None,
    }
}

/// The IEEE-754 binary16 encoding of one `f32`: round to nearest, ties to even.
///
/// This is the rounding a driver performs when it writes a `float32` clear or a
/// fragment output into a half attachment, and it is what makes the canonical
/// pass's bytes and the engine's `float32` clear the same colour. Only values
/// this function encodes *exactly* are admitted ([`clear_bytes`] widens the
/// result back), so its rounding never decides a value — it decides which values
/// are values at all, and a truncating encoder would admit a different set
/// (0.5 + 2⁻¹² encodes to `0x3801` here and to `0x3800` under truncation).
fn f32_to_half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let biased = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    // Infinities and every NaN keep the exponent's meaning; a NaN's payload has
    // no half spelling of its own, so it becomes quiet.
    if biased == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let exponent = biased - 127 + 15;
    // Larger than the widest half: rounds to an infinity. The tie at 65520 is
    // settled below with the rest of the mantissa rounding.
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if exponent <= 0 {
        // Zero, and everything below the smallest normal half, shares one
        // encoding: the mantissa's implicit one is shifted down to the
        // subnormal scale of `2^-24`.
        if exponent < -10 {
            return sign;
        }
        let shift = (14 - exponent) as u32;
        let widened = mantissa | 0x0080_0000;
        let mut half = (widened >> shift) as u16;
        let remainder = widened & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if remainder > halfway || (remainder == halfway && half & 1 == 1) {
            half += 1;
        }
        return sign | half;
    }
    let mut half = (((exponent as u32) << 10) | (mantissa >> 13)) as u16;
    let remainder = mantissa & 0x1fff;
    // A carry out of the mantissa lands in the exponent field, which is what
    // rounding up to the next binade (and, at the top, to an infinity) means.
    if remainder > 0x1000 || (remainder == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    sign | half
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

/// What one narrow-class submission returned, in the two shapes an attachment
/// can end in.
enum RenderCompletion {
    /// The completion published the attachment's whole extent.
    Writeback(RenderRailOutput),
    /// The pass kept the frame in the provider's image and published nothing for
    /// it (`StoreOp::Resident`).
    Resident(ResidentFrame),
}

/// The `(allocation, view)` pair one admitted pass declares for its attachment,
/// in the one shape every arm below reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AttachmentIdentity {
    allocation: AllocationId,
    view: ViewId,
}

impl From<ResidentAttachment> for AttachmentIdentity {
    fn from(resident: ResidentAttachment) -> Self {
        Self {
            allocation: resident.allocation,
            view: resident.view,
        }
    }
}

impl From<PresentAttachment> for AttachmentIdentity {
    fn from(present: PresentAttachment) -> Self {
        Self {
            allocation: present.allocation,
            view: present.view,
        }
    }
}

/// The `(allocation, view)` pair one admitted pass declares for its attachment.
///
/// Three arms, one per route the class states: the pooled class names its own
/// two constants, either resident arm names the request's own identity (both
/// read the same `resident` local out of `narrow_class`, so a record that loads
/// the image and keeps it cannot name two), and a **presenting** record names
/// the present target's pair (R4b) — which is not a fourth identity beside the
/// attachment's but *is* it: the canonical contract requires the present
/// target's view and allocation to be the pass colour attachment's own, so the
/// writeback the completion publishes under this pair is the presented target's
/// readback. The present arm is checked first because the class conditions make
/// it exclusive: a presenting record is refused if it names a resident target,
/// so no pass can state both routes' identities at once.
fn attachment_identity(pass: &NarrowPass<'_>) -> AttachmentIdentity {
    if let Some(present) = pass.present {
        return present.into();
    }
    let resident = match (pass.load, pass.store) {
        (NarrowLoad::Resident(resident), _) => resident,
        (_, NarrowStore::Resident(resident)) => resident,
        (NarrowLoad::Clear(_), NarrowStore::Writeback) => ResidentAttachment {
            allocation: ATTACHMENT_ALLOCATION,
            view: ATTACHMENT_VIEW,
        },
    };
    resident.into()
}

fn submit_narrow(
    inputs: &RenderRailInputs<'_>,
    pass: &NarrowPass<'_>,
) -> Result<RenderCompletion, ProviderRenderDecline> {
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
    let attachment = attachment_identity(pass);
    let loads_resident = matches!(pass.load, NarrowLoad::Resident(_));

    // The trace's own declaration of the attachment view. `OwnedBytes` of the
    // packed extent is what the frozen contract asks a storing attachment to
    // land through; neither the clear nor either resident arm reads them, and
    // the byte-less declaration that would lift the copy is the named follow-up
    // on the emulator side.
    let declaration = BufferView {
        view_id: attachment.view,
        metal_binding: 0,
        allocation_id: attachment.allocation,
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
            allocation_id: attachment.allocation,
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
            view_id: attachment.view,
            allocation_id: attachment.allocation,
            format: pass.format,
            width: pass.width,
            height: pass.height,
            load: match pass.load {
                NarrowLoad::Clear(clear) => LoadOp::Clear(clear),
                NarrowLoad::Resident(_) => LoadOp::Resident,
            },
            store: match pass.store {
                NarrowStore::Writeback => StoreOp::Store,
                NarrowStore::Resident(_) => StoreOp::Resident,
            },
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
        // R4b: a presenting record states the provider-owned target it hands
        // on. The descriptor restates the attachment's own identity, format and
        // extent — the four agreements `PresentDescriptor::validate_against`
        // checks — because those *are* the present target's; `mode` and
        // `acquire` are the only values the first increment admits, and
        // `initial` is `Undefined`: the pass loads by `Clear`, so the target's
        // previous contents are not part of anything either rail states.
        present: pass.present.map(|present| PresentDescriptor {
            target: PresentTarget {
                allocation_id: present.allocation,
                view_id: present.view,
                format: pass.format,
                width: pass.width,
                height: pass.height,
                image_count: 1,
                initial: InitialState::Undefined,
            },
            source: attachment.view,
            mode: PresentMode::Fifo,
            acquire: AcquirePolicy::Blocking,
        }),
        // The v70 sampler channel: this class refuses sampled images, samplers
        // and color input before it ever gets here, so the pass binds none.
        textures: Vec::new(),
        // The v83 stage-buffer half, empty by construction: the class admits a
        // request whose binds fill indices neither stage declares
        // (`stage_buffer_gate`), which is exactly the shape no stage buffer
        // binding has to be stated for. A pass that *did* bind one would be
        // refused here rather than executed: the canonical rail's execution
        // gate serves the reviewed stage-buffer pair alone
        // (`render_stage_buffer_stage_unsupported`), and this pass's stages are
        // the request's translated ones.
        stage_buffers: Vec::new(),
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
    // R4b: the provider's presentation counters are the only channel that can
    // say whether a stated present tail actually acquired and presented. Read
    // around the submission, exactly as the emulator's own present tests do —
    // the rail's callers submit from one worker, so the delta names this
    // submission's action rather than a neighbour's.
    let present_before = pass.present.map(|_| provider.present_counts());
    let result = provider
        .submit(validated)
        .map_err(|error| refusal_decline(&error, "submission").into_render())?;
    if !matches!(
        result.completion,
        CompletionDisposition::CompletedVisible { .. }
    ) {
        return Err(ProviderRenderDecline::CompletionNotVisible);
    }
    // A stated present tail that did not run is a decline, never a frame that
    // looks right: the display rail is about to read these bytes as the
    // presented frame, so the one fact that makes that claim checkable is the
    // provider's own acquire/present pair.
    let present = match (pass.present, present_before) {
        (Some(target), Some(before)) => {
            let after = provider.present_counts();
            let acquires = after.0.saturating_sub(before.0);
            let presents = after.1.saturating_sub(before.1);
            if acquires != 1 || presents != 1 {
                return Err(ProviderRenderDecline::PresentNotExecuted { acquires, presents });
            }
            Some(PresentCompletion {
                attachment: target,
                acquires,
                presents,
            })
        }
        _ => None,
    };
    // The resident arm's whole claim is that the frame stayed in the provider's
    // image. The one fact that would make that claim unreadable is a published
    // writeback for the same attachment, so it is checked rather than assumed —
    // and it is a typed decline, because an in-class shape is never re-run
    // somewhere else.
    let resident_writeback = result.writebacks.iter().any(|writeback| {
        writeback.view_id == attachment.view && writeback.allocation_id == attachment.allocation
    });
    if let NarrowStore::Resident(_) = pass.store {
        if resident_writeback {
            return Err(ProviderRenderDecline::ResidentWritebackPublished {
                allocation: attachment.allocation,
                view: attachment.view,
            });
        }
        // Two populations, counted where they happen: a store that kept the
        // frame, and a *load* that began from one. The second is counted for
        // both store arms — a record that loads the image and publishes its own
        // frame is the shape the next record's chain depends on, and it is the
        // one the `draw_partial_load_from_target` readers are.
        crate::runtime::drain::note_store_route("render_provider_resident_store");
        crate::runtime::drain::note_store_route(if loads_resident {
            "render_provider_resident_store_chained"
        } else {
            "render_provider_resident_store_seed"
        });
        if loads_resident {
            crate::runtime::drain::note_store_route("render_provider_resident_load");
        }
        return Ok(RenderCompletion::Resident(ResidentFrame {
            attachment: ResidentAttachment {
                allocation: attachment.allocation,
                view: attachment.view,
            },
            loaded: loads_resident,
        }));
    }
    if loads_resident {
        crate::runtime::drain::note_store_route("render_provider_resident_load");
    }
    let Some(writeback) = result.writebacks.iter().find(|writeback| {
        writeback.view_id == attachment.view && writeback.allocation_id == attachment.allocation
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
    // One population of its own, counted where it happens: a submission whose
    // frame is the provider's own present target rather than a pooled scratch
    // image. The seam prints the counters beside the same line, so a boot can
    // read the ratio of present-bearing submissions to the class's whole
    // population.
    if present.is_some() {
        crate::runtime::drain::note_store_route("render_provider_present");
    }
    Ok(RenderCompletion::Writeback(RenderRailOutput {
        bytes: writeback.bytes.clone(),
        bgra: pass.bgra,
        present,
    }))
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
        // The v83 half this class does not state yet (R9b): a stage buffer
        // declaration is a *shader interface*, and the class admits only the
        // requests whose stages declare none, so an empty list is the whole
        // truth about every contract it registers. Stating a declaration here
        // would be inventing an interface: core pairs it with the pass's own
        // list (`validate_against`), and the provider's translated
        // registration refuses the stage that would have to read it
        // (`render_stage_unsupported_interface`).
        stage_buffers: Vec::new(),
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
    // The device's own capability answer, not the translation entry point's
    // Phase-1 default: both stages are registered against the very device this
    // provider answers for, so a module the device stated it can execute has to
    // survive translation here too. R8 made the capability device-answered
    // (`FloatControls2` + `SPV_KHR_float_controls2`, admitted only by the
    // extension-and-feature conjunction) and R8b adopts that answer on this
    // seam: before it, a vertex stage whose floating-point operation withholds
    // a fast-math permission was refused by the Phase-1 default even on a
    // device that answered for the capability. A device without the feature
    // derives the Phase-1 policy, so its refusal text stays byte for byte what
    // it was.
    let policy = provider.spirv_feature_policy();
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
        TranslatedRenderStage::translate_with_policy(stage, &function, policy).map_err(|error| {
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

#[cfg(test)]
mod clear_payload_tests {
    use super::*;

    /// The two 8-bit orders keep the four-byte payload every pre-v78 trace
    /// carries, byte for byte — widening the *rule* did not widen these bytes.
    #[test]
    fn the_narrow_orders_keep_their_four_byte_payload() {
        // The reviewed texel: the fixture fragment's own `64/255, 128/255,
        // 191/255, 1`, every component an exact multiple of `1/255`.
        let reviewed = [64.0 / 255.0, 128.0 / 255.0, 191.0 / 255.0, 1.0];
        for format in [AttachmentFormat::Rgba8Unorm, AttachmentFormat::Bgra8Unorm] {
            let clear = clear_bytes(format, reviewed).expect("a multiple of 1/255");
            assert_eq!(clear.as_bytes(), [64, 128, 191, 255]);
            assert_eq!(clear.len(), ClearColor::BYTES);
        }
        // 0.5 is *not* one of them: it is 127.5/255, so the driver's rounding
        // and the byte would be two different colours — the whole reason this
        // arm compares instead of quantizing.
        assert!(clear_bytes(AttachmentFormat::Rgba8Unorm, [0.0, 0.5, 0.0, 1.0]).is_none());
    }

    /// The wide format's clear is **its own texel**: four little-endian halves
    /// in memory order, and a payload that is not four bytes.
    #[test]
    fn the_wide_format_carries_four_little_endian_halves() {
        let clear = clear_bytes(AttachmentFormat::Rgba16Float, [0.0, 0.5, -2.0, 1.0])
            .expect("0.5 and -2 are halves");
        assert_eq!(
            clear.as_bytes(),
            [0x00, 0x00, 0x00, 0x38, 0x00, 0xc0, 0x00, 0x3c]
        );
        assert_eq!(clear.len(), 8);
        // The very same colour is *not* a clear on an 8-bit attachment: 0.5 is
        // 127.5/255, which no byte holds. The rule is the carrying format's, so
        // widening the payload could not have relaxed the 8-bit half of it.
        assert!(clear_bytes(AttachmentFormat::Rgba8Unorm, [0.0, 0.5, 0.0, 1.0]).is_none());
        // And it holds in the other direction too: the reviewed `64/255` texel
        // the narrow arm admits is *not* a half (`0x3404` widens to
        // 0.2509765625), so the wide arm refuses it. Neither arm is a relaxation
        // of the other; both are the format's own round trip.
        assert_eq!(
            clear_bytes(AttachmentFormat::Rgba16Float, [0.25, 0.5, 0.75, 1.0])
                .expect("quarters are halves")
                .as_bytes(),
            [0x00, 0x34, 0x00, 0x38, 0x00, 0x3a, 0x00, 0x3c]
        );
        assert!(clear_bytes(
            AttachmentFormat::Rgba16Float,
            [64.0 / 255.0, 128.0 / 255.0, 191.0 / 255.0, 1.0]
        )
        .is_none());
    }

    /// The rule is a *round trip*, so what the half cannot hold is not a clear:
    /// a component the driver would round elsewhere is a different colour on the
    /// two rails, and one no equality can pin (NaN) is refused for the same
    /// reason.
    #[test]
    fn a_component_the_half_cannot_hold_is_not_a_clear() {
        for component in [0.1, 1e-9, 1.0e30, 65_520.0, f32::NAN] {
            assert!(
                clear_bytes(AttachmentFormat::Rgba16Float, [component, 0.0, 0.0, 1.0]).is_none(),
                "{component} must not be a half-exact clear"
            );
        }
        // The edges a half *does* hold, subnormal through infinity, plus the
        // signed zero whose sign the payload keeps.
        for component in [0.0, -0.0, 2f32.powi(-24), 65_504.0, f32::INFINITY] {
            assert!(
                clear_bytes(AttachmentFormat::Rgba16Float, [component, 0.0, 0.0, 1.0]).is_some(),
                "{component} is a half"
            );
        }
        assert_eq!(
            clear_bytes(AttachmentFormat::Rgba16Float, [-0.0, 0.0, 0.0, 1.0])
                .expect("a signed zero is a half")
                .as_bytes()[..2],
            [0x00, 0x80]
        );
    }

    /// The encoder's own rounding, pinned where it is observable: the classic
    /// `half(0.1)` bit pattern, the tie that lands on an infinity, both edges of
    /// the subnormal range, and a zero that keeps its sign.
    #[test]
    fn the_half_encoder_rounds_to_nearest_even() {
        assert_eq!(f32_to_half_bits(0.1), 0x2e66);
        assert_eq!(f32_to_half_bits(65_520.0), 0x7c00);
        assert_eq!(f32_to_half_bits(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_half_bits(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_half_bits(-0.0), 0x8000);
        assert_eq!(f32_to_half_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_half_bits(f32::NAN), 0x7e00);
    }

    /// Every half that is a *value* survives the round trip this module's rule
    /// is stated in, across the whole format: 65536 patterns minus the two
    /// signs' 1023 NaN payloads each.
    #[test]
    fn every_half_value_widens_and_encodes_back_to_itself() {
        let mut checked = 0_u32;
        for bits in 0..=u16::MAX {
            if bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0 {
                continue;
            }
            assert_eq!(f32_to_half_bits(half_to_f32(bits)), bits, "{bits:#06x}");
            checked += 1;
        }
        assert_eq!(checked, 65_536 - 2 * 1023);
    }
}
