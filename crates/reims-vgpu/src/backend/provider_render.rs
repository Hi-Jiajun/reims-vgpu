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
//! - **up to four vertex streams** and exactly one index stream, one canonical
//!   binding per stream. A stream the request holds as staged bytes is carried
//!   into the trace as trace-owned bytes; a stream the draw path resolved
//!   through the zero-copy rail is carried as the registered window its bind
//!   was cut from, through the owner rail's borrowed arm (`R9q` for the vertex
//!   streams and `R11` for the index stream, the lease channel below). Four is
//!   the canonical contract's `MAX_VERTEX_BUFFERS`, and
//!   it is the stream axis the 2026-09-17 census measured as the one a real
//!   draw stream lives on (`8327 / 8526` draws bind two to four streams, `199`
//!   bind more); a fifth stays on the engine by name. The engine numbers one
//!   Vulkan binding per attribute location, so a request's attribute list is
//!   one entry per location; the canonical layout states one entry per *fetch
//!   table* with the attributes that read it (`research/docs/26` §31), and the
//!   canonical binding index is the table's position in that list, which is why
//!   the class renumbers rather than carrying the guest's binding numbers
//!   across;
//! - **at most one scissor rectangle inside the attachment, and at most one
//!   viewport of its own** (R10, `research/docs/23` §100): the canonical pass
//!   carries the guest's own rectangle through the pass scissor
//!   ([`RenderPassDescriptor::scissor`], the v29 channel both rails execute) and
//!   *states the guest's viewport rect* in its own body
//!   ([`RenderPassDescriptor::viewport`], the v100 channel), so the texels
//!   outside either rectangle keep the load op's bytes on both. A scissor that
//!   is empty, that reaches outside the attachment, or that is one of several
//!   stays on the engine (the contract refuses such a rect by name,
//!   `ScissorOutOfBounds`, while the engine *clamps* one that reaches past the
//!   attachment), and so does a viewport the contract has no spelling for — a
//!   negative or fractional origin, an empty or out-of-bounds rect, a depth
//!   range of its own, or more than one rect ([`viewport_admits`], one bucket
//!   per shape);
//! - **one sampled texture per fragment-stage `[[texture(n)]]`, declared beside
//!   its bind** (R10, `research/docs/23` §101): the canonical pass's texture
//!   list *is* the fragment stage's texture argument space, so the class states
//!   one declaration per reflected `[[texture(n)]]` — the module's own AIR
//!   sampler state, the Metal index the entry states (the two lists pair by
//!   that index rather than by position since E-RS3, §104), and the shapes the
//!   render sampler uploads (a single-sample, non-arrayed, read-only 2D
//!   surface whose texels are one of the two four-byte 8-bit UNORM byte orders
//!   — `rgba8_unorm`/`bgra8_unorm`, the provider's whole `RENDER_SAMPLED`
//!   window since E-TX1/§107 — at the render area's own extent) — beside
//!   the draw's own bind, and requires the bind's sampler state to repeat the
//!   module's. Every other shape is a named exit ([`sampled_textures`]): a
//!   resource family the translated rail does not execute, a texture index at
//!   or above the contract's own bound or a list that repeats an index or walks
//!   backwards, a reflected shape or state outside the family, an unbound
//!   declaration, a bind whose texels are a guest gather or a resident image, a
//!   texture of another extent, and a draw whose sampler says another state;
//! - **a sampled texture whose texels are the trace's own production** (R22,
//!   E-TX3/`research/docs/23` §110): a request whose bind resolved to a GPU
//!   target (`SampledSource::Target`) leaves for the provider once a pass of
//!   this rail has declared that guest target's production — the producing
//!   record's own pass rides the consuming record's trace ahead of it, the
//!   consuming declaration samples it through `TextureSource::TraceView`, and
//!   the frame the producing record landed is the frame the consumer reads. A
//!   target no pass here produced (`..._texture_source_undeclared`), a record
//!   that samples the attachment it writes (`..._texture_source_order`), a
//!   declaration that restates another shape than the production stored
//!   (`..._texture_source_shape`) and a sampled pass whose binds the
//!   owner→provider frame would have to carry (`..._texture_wire`, whose
//!   contract still carries no texture declarations) all stay on the engine.
//!   The zero-copy guest gather keeps its own exit
//!   (`..._texture_source`) — that arm is the increment after this one;
//! - **the attachment's own blend state** (R10, `research/docs/23` §100): a
//!   blend the canonical pass can state *and* the command channel's v40 section
//!   carries (blending enabled, one operation for both channel pairs, every
//!   channel written) leaves for the provider as the pass's blend entry, while
//!   a write mask, an alpha operation of its own and the three factor families
//!   the provider refuses by name keep the engine under their own buckets
//!   ([`declared_blend`]);
//! - the pipeline pair is the request's own translated stages: the two SPIR-V
//!   modules reims' pipeline resolution produced, registered through the
//!   canonical *translated* registration gate
//!   (`register_translated_render_pipeline`), which checks each stage's
//!   reflection against the contract field by field;
//! - no depth, stencil, MSAA, MRT, resolve, colour write mask a wire section
//!   cannot carry, occlusion query or framebuffer fetch — and every one of those
//!   is a *reason*, not a silent downgrade;
//! - **stage buffers by the stage's own interface** (R9b/R9d/R9j): a request
//!   whose stages bind buffers directly is the v2 census's 99.3% door, and the
//!   door answers by what each stage's *translation* declares rather than by
//!   the presence of the binds. Neither stage declaring a `[[buffer(N)]]`
//!   argument means the binds fill indices no descriptor can be needed for, so
//!   the canonical pass declares and binds nothing for them and the draw is
//!   admitted. A **declared** argument leaves for the provider once the whole
//!   canonical pair can be stated — the declaration from the stage's own
//!   reflection (stage, index, the access it reports, and the reach it states:
//!   a static ceiling or the affine proof R9f landed), the view from the
//!   request's own bind, whose bytes the owner rail imports
//!   ([`StageBufferBind`], `plan_owner_leases`) — and a **writable** one is a
//!   landing: its bytes come back through the completion's own
//!   `BufferWriteback` and the caller places them where the bind's bytes came
//!   from ([`StageBufferLanding`]). The declaration rides the canonical command
//!   channel for the shapes that carry one ([`provider_wire`]); of the two arms
//!   the contract has no slot for, the one whose access the translation does
//!   not classify (`unknown`) keeps the draw on the engine under its own
//!   bucket, and the one whose entry never dereferences the slot (`unused`) is
//!   not declared at all (R9m,
//!   [`RenderRailInputs::stage_buffer_statement`]) **and no longer keeps the
//!   draw on the engine** (R9n): the statement carries only the arguments the
//!   entries reach, the canonical registration leaves a reflected `Unused`
//!   slot out of the pairing when the contract does not declare it
//!   (`research/docs/23` §95), and the draw proceeds with that slot neither
//!   declared nor bound — while a contract that states the slot is refused by
//!   name (measured in `provider_render_rail.rs`);
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
//!   # The resident arms are the caller's capability, and it is held today
//!
//!   That refusal is not hypothetical, and census v15 measured its price:
//!   11 892 `ok resident` answers, 6 824 `chain_resident_land_fail`
//!   (6 821 `read_target_no_ready_content` + 3 `read_target_unknown_identity`)
//!   and 6 833 `load_target_content_not_ready` on one driven macos-13 boot —
//!   every failure on a GVA whose frame the provider had answered for moments
//!   earlier, and a guest desktop with its Dock unpainted. The engine's
//!   refusal is not a race to be waited out: the frame is not *pending*, it is
//!   in an image the engine's registry never held, and no caller-side reader
//!   (the deferred GVA debt, the mapper-ref-texture store, the next record of
//!   the packet) can reach it.
//!
//!   So the arms are elected by what the caller can consume
//!   ([`RenderRailInputs::resident_frames_fetchable`]): a caller that cannot
//!   fetch a kept frame gets the frame **published** instead (the pooled arm's
//!   own answer, `render_provider_publish_held_resident`), and a record whose
//!   previous contents are an image this rail holds stays on the engine by
//!   name (`render_provider_out_of_class_resident_source`). The production
//!   seam states `false` until R4b's byte channel lands; this rail's own tests
//!   state `true`, because the capability is what they drive. The packet-level
//!   admission above remains the follow-up for the state where the arms are
//!   live.
//!
//!   # The reverse direction: the frame the *engine* holds, carried in (R23)
//!
//!   Census v17 (`evidence/gate3-census-v17-2026-09-18`, same rig and log)
//!   moved the boundary onto the other side of the same seam: its first
//!   refusal is `resident_source` at 3 271 records (73.2 %), every one of them
//!   a record whose previous contents are the live GPU image — `skip=resident`,
//!   `store=1`, `seed=chain`, `load=Load` — in a packet whose *head* the class
//!   refused for another reason (a guest seed, a guest-backed attachment), so
//!   the engine owns the chain's frame from the first record on. `LoadOp::
//!   Resident` cannot name it (this rail never stored an image under that
//!   identity), and R20's answer for a frame the caller cannot fetch is not to
//!   keep one — but the frame itself is readable *by the caller*: it lives in
//!   the engine's own registry, whose reader is `read_target`. So the class
//!   carries it the other way: the caller hands the frame over
//!   ([`RenderRailInputs::resident_source_bytes`]) and the pass states the
//!   canonical attachment's trace-owned load (`NarrowLoad::Bytes` →
//!   `LoadOp::Load`), which uploads exactly those bytes into the pass's own
//!   image before it opens (`research/docs/23` §3.3/§74, R5b). Its own frame is
//!   then published (this rail still keeps nothing its caller cannot fetch),
//!   and the exec walk hands those bytes to the record after it — so a chain
//!   whose head stayed on the engine continues in the provider from the second
//!   record on, with the frame's bytes the one channel it travels through.
//!   Bytes that are not the attachment's own extent, and a record whose caller
//!   hands nothing over, keep the engine under the names above.
//!
//!   # The packet's own chain value, carried into the middle (R25)
//!
//!   Census v18 (`evidence/gate3-census-v18-2026-09-18`, same rig) moved the
//!   boundary one record further into the same packet. With R23 in place the
//!   class answers the chain's *head* from the frame's bytes, and the record
//!   after it begins from the frame the head published — the exec walk's own
//!   chain value, which is what `encode_draw_chain` normalizes a
//!   `!writeback_guest` readback to and hands on as the request's
//!   `target_rgba8`. That population is the census's fourth bucket,
//!   `chain_middle` at 657 records (8.0 %), every one of them `fmt=0x50`,
//!   `load=Load`, `wb=0`, `skip=resident`, `store=1`, `seed=bytes`,
//!   `continues=1`. `LoadOp::Resident` cannot name that frame either (no
//!   provider pass stored one under the attachment's identity), so the class
//!   carries it the same way R23 carries the engine's: the caller hands the
//!   frame over ([`RenderRailInputs::chain_middle_source_bytes`]) and the pass
//!   states the contract's trace-owned load (`NarrowLoad::Bytes` →
//!   `LoadOp::Load`). The record's own frame is published for the walk to hand
//!   to the record after it, exactly as R23's arm publishes it. A middle whose
//!   caller hands nothing over, one whose frame cannot travel as four-byte
//!   colour, and a hand-over that is not the attachment's own extent keep the
//!   engine under the names above.
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
//! # The lease channel, and where each input's arm stands
//!
//! `research/docs/26` §3 planned to carry the vertex and index streams through
//! the owner rail's lease channel (the one [`super::provider_owner`] gives the
//! compute rail). The canonical side opened that channel per binding in R3c:
//! `resolve_render_input` resolves each input's own `BufferSource::OwnedBytes` /
//! `StagedLease` / `BorrowedNoCopy` arm under the input's binding index
//! (`render.rs::resolve_vertex_streams` zips the pass's views with the pipeline
//! layout, so N streams are N independent resolutions — nothing in it assumes
//! one stream).
//!
//! R9d wires the first half of that channel on *this* side. A stage buffer whose
//! bytes the owner holds goes out as an owner-issued **staged lease**
//! (`plan_owner_leases` → `provider_owner::plan` → `BufferSource::StagedLease`),
//! and one whose bytes the seam cut out of a registered guest RAM window goes
//! out as the **borrowed no-copy** arm (`BufferSource::BorrowedNoCopy`), gated
//! on the device's own host-pointer import:
//!
//! - the staged arm is what the production seam states for a bind whose bytes
//!   travel as the CPU staging origin (`BufferContent::Bytes`): the owner holds
//!   those bytes, and the window they were cut from is not part of what the
//!   staging read carried;
//! - the borrowed arm is stated two ways and both are driven. The seam may
//!   state a window itself
//!   (`a_stage_buffer_in_a_registered_window_leaves_without_a_copy`, R9d), and
//!   since R9e it may also leave the window to this rail: a bind the draw path
//!   resolved through the zero-copy rail arrives as a gather
//!   (`BufferContent::GuestRuns`), whose page runs already carry the
//!   provider-shaped window the registration ledger derived for them, so
//!   [`gather_window`] derives it from the source's one stretch instead
//!   (`a_stage_buffer_the_seam_derives_from_its_gather_leaves_without_a_copy`).
//!   Either way the frame follows the owner's own mapping and nothing is
//!   copied;
//! - a gather this rail cannot cut a window from — scattered across stretches,
//!   unregistered, or with a `source_offset` reaching past its stretch's window
//!   — keeps the engine by name
//!   (`render_provider_out_of_class_stage_buffer_gather`);
//! - R18 gives the window-backed arm its third answer, and it is the one the
//!   device decides: a window that *does* cover the bind's bytes still cannot be
//!   imported in place when the view's own host pointer is not a whole number of
//!   the device's import granules (the registration aligns the window's base,
//!   but a bind at a packed resource's own offset moves the view off the
//!   granule). The bytes the borrowed arm would have bound are copied out of the
//!   registration's own mapping ([`provider_owner::window_bytes`]) and travel as
//!   an owner-issued **staged lease**, exactly as a bind with no window does —
//!   so the draw reaches the provider instead of keeping the engine
//!   (`render_provider_out_of_class_stage_buffer_alignment` was that population
//!   before this increment). Only a window this rail cannot read — its import
//!   unregistered, its address rail retired, or its coordinates outside the
//!   registration — still keeps the draw on the engine under that same name.
//!
//! R9q wires the *vertex* half of the same channel. A vertex stream is the same
//! kind of bind as a stage buffer — the guest's own buffer, resolved by the
//! draw path's zero-copy rail — so the two arms are the same two arms:
//!
//! - staged bytes stay the third arm, trace-owned (`BufferSource::OwnedBytes`):
//!   they are the bytes the runtime already read, not a guest bind;
//! - a stream whose one registered window covers the bind's own bytes travels as
//!   **borrowed no-copy** ([`StreamSource::Window`], derived by the same
//!   [`gather_window`], imported by the same plan): the frame follows the
//!   owner's own mapping, and moving that mapping moves the vertex bytes;
//! - R18's third answer applies here unchanged: a stream whose view pointer
//!   misses the device's granules is copied into an owner-issued staged lease
//!   rather than keeping the engine, and the view this rail states for it names
//!   that lease (`BufferSource::StagedLease`) exactly as a declared stage
//!   buffer's staged arm does;
//! - a gather the seam cannot cut one window from still keeps the engine under
//!   `render_provider_out_of_class_vertex_staging` — the bucket every gather
//!   answered with before this increment — and one whose granules miss it *and*
//!   whose window cannot be read keeps it under
//!   `render_provider_out_of_class_vertex_alignment`; a device without
//!   host-pointer import answers `render_provider_out_of_class_vertex_import`.
//!
//! R11 wires the *index* half, and it is the same two arms a third time: an
//! indexed draw binds exactly one index buffer, and the draw path resolves it
//! through the same zero-copy rail as a vertex stream (`load_index_content_reason`
//! → `load_buffer_content_resolved`), so the seam derives its window with the
//! same [`gather_window`] and imports it through the same plan:
//!
//! - staged index bytes stay the third arm, trace-owned
//!   (`BufferSource::OwnedBytes`): they are the bytes the runtime already read;
//! - an index stream whose one registered window covers the bind's own bytes
//!   travels as **borrowed no-copy** ([`StreamSource::Window`]): the frame
//!   follows the owner's own mapping, and moving that mapping moves the index
//!   bytes — which is what the census read as `index_staging` on every draw
//!   whose index bind the draw path had already imported;
//! - R18's staged copy is the third arm here too, and an index bind is where it
//!   costs least: the stream is small, read once, and the canonical pass states
//!   it as one lease-backed view like any other staged bind;
//! - a gather the seam cannot cut one window from still keeps the engine under
//!   `render_provider_out_of_class_index_staging`, one whose granules miss the
//!   bind *and* whose window cannot be read keeps it under
//!   `render_provider_out_of_class_index_alignment`, and a device without
//!   host-pointer import answers `render_provider_out_of_class_index_import`.
//!
//! The attachment declaration keeps the third arm: trace-owned bytes
//! (`BufferSource::OwnedBytes`), since an attachment's load seed is not a bind
//! of the guest's buffer.
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
use std::sync::{Arc, Mutex, OnceLock};

use metal_api_core::provider::{
    half_to_f32, AcquirePolicy, AffineAccess, AllocationId, AllocationRecord, AttachmentFormat,
    BlendAttachment, BlendFactor, BlendFactorSlot, BlendOperation, BufferAccess, BufferSource,
    BufferView, BufferWriteback, ClearColor, ColorWriteMask, CompiledComputePipeline,
    CompletionDisposition, CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace, Dispatch,
    DispatchKind, DispatchType, FootprintProof, IndexBufferBinding, IndexFormat, InitialState,
    LoadOp, NoCopyLeaseImporter, OperationId, PresentDescriptor, PresentMode, PresentTarget,
    RenderAttachment, RenderPassBlend, RenderPassDescriptor, RenderPipelineContract,
    RenderPipelineStage, RenderSamplerBinding, ResourceTableSnapshot, SamplerPolicy,
    SemanticDigest, StageBufferBinding, StageBufferView, StoreOp, TextureAccess,
    TextureBindingContract, TextureFootprintProof, TextureFormat, TextureSource, TextureType,
    TextureView, TracePass, VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout,
    VertexStep, ViewId, MAX_RENDER_SAMPLERS, MAX_RENDER_STAGE_BUFFERS, MAX_RENDER_TEXTURES,
    MAX_RENDER_TEXTURE_INDEX, MAX_SERIAL_RESOURCES, MAX_VERTEX_BUFFERS, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::Device;
use metal_api_vulkan::{RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage};

use super::provider_compute::{
    provider_error_detail, rail, refusal_decline, refuse_unhealthy, ProviderRefusalClass,
};
use super::provider_owner::{self, DeviceLossTeardown};
use super::provider_wire;
use super::vulkan::engine::types::{
    BufferContent, ColorClearValue, DrawRequest, GuestRunSource, ReadbackSkipReason,
    TargetIdentity, VertexAttributeResource, VertexStepFunction,
};
use crate::observe::{decline_display, Decline};

/// The declaring kernel this rail ships (`crates/reims-vgpu/src/backend/render_declare.ll`).
///
/// Owned, not derived from a third-party metallib; the file states what it is
/// for and why a read is the whole body.
const RENDER_DECLARE_SOURCE: &str = include_str!("render_declare.ll");

/// Entry point [`RENDER_DECLARE_SOURCE`] declares.
const RENDER_DECLARE_ENTRY: &str = "reims_declare";

/// The byte reach the declaring kernel's own reflection states for the buffer it
/// reads (`render_declare.ll`'s `air.buffer_size`).
///
/// A writable stage buffer rides into the trace's pool as a view a declare pass
/// binds (see `submit_narrow`), and that pass's view is proven against the
/// declaring contract's own footprint — a shorter view is a shape admission
/// refuses by name (`buffer_footprint_exceeds_view`), so the class gate answers
/// it first.
const RENDER_DECLARE_REACH: u64 = 4;

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

/// Whether this device can import the owner's host pointers at all, and at
/// which alignment.
///
/// The second device fact this rail's class gate reads, for the same reason as
/// the window above: a bind whose bytes the seam cut out of a registered guest
/// RAM range leaves the rail only through the canonical provider's *no-copy*
/// arm (`BufferSource::BorrowedNoCopy`), and the owner rail refuses that arm by
/// name on a device that reports no `VK_EXT_external_memory_host`
/// ([`provider_owner::channel`]'s `HostImportUnavailable`). Asking here keeps
/// the class's one promise — everything it admits is a shape the provider
/// executes — instead of turning a device answer into a declined draw the
/// engine would have drawn.
fn declared_host_import() -> Result<u64, ProviderRenderDecline> {
    let rail = rail().map_err(IntoRender::into_render)?;
    Ok(rail.provider.no_copy_alignment())
}

/// The names one window-backed shape answers under (`R9q`).
///
/// A binding's spelling in the refusal's sentence, and the two slugs the census
/// counts it under — one for a device that cannot import host pointers, one for
/// a view pointer the device's granules do not align. A struct rather than a
/// pair of arguments because the three travel together at every call site and
/// the slugs have to stay literal: they are read back out of the static
/// library's strings.
#[derive(Clone, Copy)]
struct WindowShape {
    name: &'static str,
    import_slug: &'static str,
    alignment_slug: &'static str,
}

/// The stage-buffer half of the window-backed shapes (`R9e`).
const STAGE_BUFFER_WINDOW: WindowShape = WindowShape {
    name: "stage buffer",
    import_slug: "render_provider_out_of_class_stage_buffer_import",
    alignment_slug: "render_provider_out_of_class_stage_buffer_alignment",
};

/// The vertex-stream half of the window-backed shapes (`R9q`).
const VERTEX_STREAM_WINDOW: WindowShape = WindowShape {
    name: "vertex stream",
    import_slug: "render_provider_out_of_class_vertex_import",
    alignment_slug: "render_provider_out_of_class_vertex_alignment",
};

/// The index-stream half of the window-backed shapes (`R11`).
///
/// The third namespace of the same two rules: an index buffer is a bind of the
/// guest's own buffer like a vertex stream, and its view pointer is its own
/// (the window's base plus the bind's own head), so the two device answers are
/// asked for it separately. The slugs are new names because the census reads
/// *which* shape crossed the device, not only which rule it met.
const INDEX_STREAM_WINDOW: WindowShape = WindowShape {
    name: "index stream",
    import_slug: "render_provider_out_of_class_index_import",
    alignment_slug: "render_provider_out_of_class_index_alignment",
};

/// The view's own host pointer of one window-backed binding: the window's base
/// plus the bind's own head, when the two are addressable together.
///
/// `None` is a coordinate pair whose sum leaves the address space — a malformed
/// window rather than a slow one, answered by [`window_binding_admits`]'s own
/// refusal.
fn window_view_pointer(window: StageBufferWindow) -> Option<u64> {
    window.host_va.checked_add(window.head)
}

/// The device answer one window-backed binding has to pass before this class
/// states it at all (`R9e`, `R9q`, `R11`).
///
/// One question, about the device rather than the request: a device that cannot
/// import host pointers has no window arm at all — the owner rail refuses a
/// window-backed bind on it rather than turning it into a copy
/// ([`provider_owner::channel`]'s `HostImportUnavailable`), and a declined draw
/// where the engine would have gathered is the wrong answer for a class that
/// only narrows which submissions change rail. The *second* device answer — the
/// view pointer's alignment — is not a refusal any more: since R18 it decides
/// between the borrowed arm and the staged copy, and [`window_arm`] asks it.
///
/// `shape` carries the binding's own spelling in the sentence and the two slugs
/// it answers under, because the bucket is the census's own reading of *which*
/// shape crossed the device: `..._stage_buffer_import` /
/// `..._stage_buffer_alignment` and `..._vertex_import` /
/// `..._vertex_alignment` are four names for two rules.
fn window_binding_admits(
    shape: WindowShape,
    window: StageBufferWindow,
    alignment: u64,
) -> Result<(), OutOfClass> {
    let WindowShape {
        name,
        import_slug,
        alignment_slug,
    } = shape;
    if alignment == 0 {
        return Err(OutOfClass::owned(
            import_slug,
            format!(
                "a draw whose {name} is covered by a registered guest RAM window stays on the \
                 engine on a device that cannot import host pointers: the canonical rail's \
                 no-copy arm needs VK_EXT_external_memory_host, and the owner rail refuses a \
                 window-backed bind rather than silently copying it",
            ),
        ));
    }
    if window_view_pointer(window).is_none() {
        return Err(OutOfClass::owned(
            alignment_slug,
            format!(
                "a draw whose {name} is covered by a registered guest RAM window stays on the \
                 engine when the view's own host pointer is not addressable: the window's base \
                 plus the bind's own head overflows the address space",
            ),
        ));
    }
    Ok(())
}

/// The arm one window-backed binding takes on this device (`R9e`, `R9q`, `R11`,
/// `R18`).
///
/// `Ok(None)` is the borrowed arm: the view's own host pointer is a whole
/// number of the device's import granules, so the plan imports the window and
/// nothing is copied. `Ok(Some(bytes))` is the staged arm: the registration
/// aligns the window's base, but a bind at a packed resource's own
/// `source_offset` moves the view off the granule, the canonical rail refuses an
/// import at that pointer by name (`lease_alignment_unsupported`), and a
/// declined draw where the engine would have gathered is not an answer this
/// class may give — so the bind's own bytes are copied out of the registration
/// ([`provider_owner::window_bytes`]) and travel as the owner's staged lease,
/// which is the arm a bind with no window behind it already takes. What the copy
/// costs is the device's own constraint; what it preserves is *which bytes* the
/// declaration reads, because the copied range is exactly the range the
/// borrowed arm would have bound.
///
/// A window this rail cannot read at all — an import the owner rail does not
/// hold, one its address rail retired, coordinates outside the registration, or
/// an empty bind — keeps the draw on the engine under the shape's own alignment
/// bucket, because there is no copy to state and a decline is not a fallback.
fn window_arm(
    shape: WindowShape,
    binding: u32,
    window: StageBufferWindow,
    alignment: u64,
) -> Result<Option<Vec<u8>>, OutOfClass> {
    window_binding_admits(shape, window, alignment)?;
    let WindowShape {
        name,
        alignment_slug,
        ..
    } = shape;
    // The refusal above answers the unaddressable pair, so the sum exists here.
    let pointer =
        window_view_pointer(window).expect("an admitted window's view pointer is addressable");
    if pointer.is_multiple_of(alignment) {
        return Ok(None);
    }
    match provider_owner::window_bytes(owner_window(binding, window)) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(decline) => Err(OutOfClass::owned(
            alignment_slug,
            format!(
                "a draw whose {name} is covered by a registered guest RAM window stays on the \
                 engine when the view's own host pointer is not a whole number of the device's \
                 import granules and the window's bytes cannot be copied out of the \
                 registration that names it: the pointer is {pointer}, the device imports host \
                 pointers at {alignment} byte alignment, and the copy this class states in \
                 place of the no-copy import was refused by the owner rail (`{}`)",
                decline.slug(),
            ),
        )),
    }
}

/// The bytes one pass copies out of its window-backed binds (`R18`), under the
/// owner binding label the plan looks each view up by.
///
/// The copy itself is made by the class gate, where the device's import
/// alignment is read — before the pass is submitted and before any lease exists
/// — and this is what carries it to [`plan_owner_leases`], which states those
/// binds through the owner's staged arm instead of the borrowed one. Keyed by
/// the owner label rather than by position because the three shapes' namespaces
/// overlap (`stage_buffer_owner_binding`'s two stages, a stream's index, the
/// index stream's constant), and a copy read back under another namespace's
/// label would be a wrong frame rather than a refusal.
#[derive(Default)]
struct WindowCopies {
    entries: Vec<(u32, Vec<u8>)>,
}

impl WindowCopies {
    fn insert(&mut self, binding: u32, bytes: Vec<u8>) {
        self.entries.push((binding, bytes));
    }

    /// The copied bytes for one window-backed bind, when this pass stages it.
    fn bytes(&self, binding: u32) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|(label, _)| *label == binding)
            .map(|(_, bytes)| bytes.as_slice())
    }
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

/// The shape one fragment-stage `[[texture(i)]]` argument was reflected with,
/// reduced to the question the render sampler's class gate asks (R10,
/// `research/docs/23` §101).
///
/// The canonical translated arm executes exactly one texture family: a
/// single-sample, non-arrayed, read-only `D2` surface whose component is a
/// float, one descriptor in set 0. The reflection is asked about the shape
/// *before* the request is, because a module that declares another family is a
/// shape the provider refuses by name (`render_texture_shape_unsupported` /
/// `render_texture_format_unsupported` / `render_texture_layout_unsupported`)
/// whatever the draw bound — and a refusal is a decline, not a fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderTextureShape {
    /// The one family both sampler arms upload and sample.
    Sampled2D,
    /// Everything else, named for the class gate's bucket.
    Unsupported(RenderTextureShapeRefusal),
}

/// Which reflected fact put a texture outside the executable family.
///
/// A value rather than a string for the reason every other route in this module
/// is one: two arms that answer with different census names are two facts, and
/// a typo'd literal would file both under one name with nothing failing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderTextureShapeRefusal {
    /// A dimension that is not `D2` (1D, 3D, cube, …).
    Dimension,
    /// An arrayed (or array-reference) texture.
    Arrayed,
    /// A multisampled texture.
    Multisampled,
    /// A writable (storage) texture.
    Writable,
    /// A component that is not the float family (uint/int samples).
    Component,
    /// A descriptor outside set 0 / count 1, or one the reflection does not
    /// carry at all.
    Descriptor,
}

impl RenderTextureShapeRefusal {
    /// The shape, as the refusal's sentence names it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dimension => "a texture whose dimension is not D2",
            Self::Arrayed => "an arrayed texture",
            Self::Multisampled => "a multisampled texture",
            Self::Writable => "a writable (storage) texture",
            Self::Component => "a texture whose component is not a float",
            Self::Descriptor => "a texture whose descriptor is not one binding in set 0",
        }
    }
}

/// The sampler form one fragment-stage `[[texture(i)]]` argument's samples read
/// through, as the module's own translation states it (R10/R12/R15,
/// `research/docs/23` §101, §102, §3.3).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderSamplerState {
    /// An AIR `constexpr sampler` the module carries: the state the canonical
    /// rail creates its `VkSampler` from (R10).
    Policy(SamplerPolicy),
    /// A runtime `[[sampler(n)]]` argument the module's own sample sites name
    /// (R12): the module carries no state, so the *request* states it and this
    /// is the Metal index it belongs to.
    Runtime {
        /// The Metal `[[sampler(n)]]` argument index.
        index: u32,
    },
    /// The module's own image sites only texel-fetch this texture (R15): it
    /// carries no AIR sampler state and names no runtime `[[sampler(n)]]`
    /// argument, because `texture.read()` is an `OpImageFetch` of the image
    /// alone. The declaration states no sampler form at all and the canonical
    /// rail binds the image descriptor by itself
    /// ([`TextureBindingContract::fetched`]).
    Fetched,
    /// A sampler form this class cannot name, with the fact that stopped it.
    Unsupported(RenderSamplerRefusal),
}

/// Why one sampled texture's sampler form is not one this class states.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderSamplerRefusal {
    /// The AIR `constexpr sampler`'s state is outside the family the canonical
    /// rail creates (R10, widened to the six-filter/five-mode family by R21):
    /// a differing min/mag filter, a bicubic filter, axes that address
    /// differently, a `clampToBorderColor`, non-normalized coordinates, a
    /// compare function, anisotropy, or a reduction.
    AirState,
    /// The module's own sample sites do not name a runtime `[[sampler(n)]]`
    /// argument the reflection binds for this texture (R12): no sample site
    /// reaches it, two samplers do, or the site's operands are values the
    /// module's own walk cannot follow to a descriptor.
    SampleSite,
}

impl RenderSamplerRefusal {
    /// The fact, as the refusal's sentence names it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::AirState => {
                "the AIR state is outside the family the canonical rail creates — a min/mag \
                 filter of nearest or linear crossed with the not-mipmapped, nearest or linear \
                 mip filter; one address mode for all three axes, of clamp-to-edge, \
                 mirror-clamp-to-edge, repeat, mirror-repeat or clamp-to-zero; normalized \
                 coordinates, no comparison, no anisotropy and weighted-average reduction"
            }
            Self::SampleSite => {
                "the module's own sample sites name no runtime `[[sampler(n)]]` argument the \
                 reflection binds for it"
            }
        }
    }
}

/// One runtime `[[sampler(n)]]` argument the fragment stage's reflection binds
/// (R12, `research/docs/23` §102).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderRuntimeSampler {
    /// The Metal `[[sampler(n)]]` argument index.
    pub index: u32,
    /// The device binding the runtime resolves this draw's own sampler bind at,
    /// before the fragment sampled-band relocation the draw adds on top.
    pub binding: u32,
}

/// The sampler family one fragment stage's translation declares (R12).
///
/// Two lists rather than one because the class asks two different questions of
/// them: which runtime `[[sampler(n)]]` arguments the stage binds (each of which
/// the canonical contract has to pair with a texture, or the provider refuses
/// the registration by name), and whether the stage carries AIR static samplers
/// at all — the two families do not mix in one stage the canonical rail can
/// pair, because its static half pairs positionally and its runtime half by the
/// declaration's own index.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RenderSamplerFamily {
    pub runtime: Arc<[RenderRuntimeSampler]>,
    /// The Metal index of every `ResourceKind::StaticSampler` binding, in the
    /// reflection's own order — the order the canonical rail's positional
    /// pairing counts against.
    pub statics: Arc<[u32]>,
}

/// One `[[texture(i)]]` argument the fragment stage's own translation declares
/// (R10, `research/docs/23` §101).
///
/// The three facts the class gate needs, and the one place each comes from:
/// the Metal index and the sampler state are the *module's* (the declaration
/// has to repeat the state the module was lowered against, or the provider
/// refuses the pair by name), the two device bindings are the slots the
/// *request's* binds were resolved at — the runtime applies the same fragment
/// sampled-band relocation to both lists, so declaration `i` and the draw's
/// own bind pair by number here exactly as they do inside the engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderTextureDeclaration {
    /// The Metal `[[texture(n)]]` argument index. The canonical contract pairs
    /// its two texture lists by the index each entry *states* (E-RS3,
    /// `research/docs/23` §104), so this — and not the entry's position — is
    /// what the declaration's own `metal_binding` carries and what the two
    /// lists are held to each other by.
    pub index: u32,
    /// The device binding the request resolves this texture's view at.
    pub binding: u32,
    /// The device binding of the sampler this texture's samples go through:
    /// the AIR static sampler [`Self::sampler`] states the policy of, or — for
    /// the runtime family (R12) — the `[[sampler(n)]]` argument
    /// [`Self::sampler`] names the Metal index of. A texel-fetched declaration
    /// ([`RenderSamplerState::Fetched`], R15) states none, and this is the
    /// contract's own zero — the arm reads no sampler slot.
    pub sampler_binding: u32,
    /// The module's own state for that sampler, or the sampler-free arm the
    /// module's own image sites state.
    pub sampler: RenderSamplerState,
    /// The reflected texture shape.
    pub shape: RenderTextureShape,
}

/// One Metal resource argument outside the family the canonical *translated*
/// render rail executes, with the stage and kind that answer for it (R10).
///
/// The translated rail binds `[[buffer(n)]]` arguments, one sampled
/// `[[texture(i)]]` per AIR static sampler, and nothing else: a runtime
/// `[[sampler(n)]]`, a storage image, a texture array, a framebuffer-fetch
/// `[[color(n)]]` or a vertex-stage image is refused by the provider's own name
/// (`render_stage_unsupported_interface`), so a draw whose module declares one
/// is a shape this class has to answer for before the provider is asked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderInterfaceRefusal {
    /// The stage whose reflection declares it.
    pub stage: RenderPipelineStage,
    /// The Metal resource kind, as `metal2vulkan`'s reflection names it.
    pub kind: &'static str,
    /// The argument's Metal index.
    pub index: u32,
}

/// The `[[texture(i)]]` arguments one fragment stage's reflection declares, with
/// the AIR sampler state its samples were lowered against (R10,
/// `research/docs/23` §101) — or the sampler-free answer a texture only
/// `texture.read()` reaches states (R15, §3.3).
///
/// The canonical render sampler pairs the *i*-th reflected sampled texture with
/// the *i*-th AIR static sampler (`metal-api-vulkan`'s
/// `translated_texture_slots`), so this walk makes the same pairing over the
/// same reflection — one list of textures, one list of static samplers, paired
/// by position — instead of a second rule that could disagree about which state
/// belongs to which texture.
///
/// The state is mapped into the canonical policy family here, once per resolved
/// pipeline, because that is the question the class gate asks on every draw:
/// `Some` exactly for the states the canonical rail creates a `VkSampler` from
/// — nearest or linear min/mag filtering crossed with the three `MTLSamplerMipFilter`
/// values, one address mode on all three axes (clamped, mirror-clamped,
/// repeated, mirror-repeated or clamped to zero), normalized coordinates, no
/// comparison, no anisotropy, weighted-average reduction — and the refusals
/// the gate answers with otherwise. The refusals' *names* live in the rail
/// (`RenderSamplerState`), the facts live here.
///
/// A texture with no AIR static sampler beside it reads through the *other*
/// Metal sampler family: a runtime `[[sampler(n)]]` argument, whose state the
/// pass states rather than the module (R12, `research/docs/23` §102). Which
/// argument a texture reads through is not in the argument metadata — the
/// translation states it in the module body — so this walk asks `words`, the
/// module this resolution translated, for the pairing its own sample sites
/// name (`runtime::spirv_bind::sampled_image_pairs`). A texture whose sites
/// name no such argument, or more than one, is `Unsupported` with that fact,
/// and the gate keeps it on the engine by name.
#[cfg(feature = "provider-render")]
pub fn texture_declarations(
    reflection: &metal2vulkan::reflect::ShaderReflection,
    words: &[u32],
) -> Arc<[RenderTextureDeclaration]> {
    use metal2vulkan::meta::{TextureComponent, TextureDimension, TextureShape};
    use metal2vulkan::reflect::{
        ResourceKind, SamplerAddressMode as AirAddress, SamplerCompareFunction, SamplerCoordinates,
        SamplerFilter as AirFilter, SamplerMipFilter, SamplerReduction,
    };
    use metal_api_core::provider::{SamplerAddressMode, SamplerFilter};

    /// The module's AIR state in the canonical policy's two fields, or `None`
    /// when the state is outside the family the canonical rail creates.
    ///
    /// This is the same rule `metal-api-vulkan`'s `static_sampler_policy`
    /// applies to the very same reflection, at the widened family E-TX2 landed
    /// there (R21, `research/docs/26` §44): the six filters the two
    /// `MTLSamplerMinMagFilter` values and three `MTLSamplerMipFilter` values
    /// cross into, and the four address modes the translator's own vocabulary
    /// can name. The two rails translate one AIR with one translator, so this
    /// is one measurement restated, not a second opinion — including its two
    /// by-name refusals (`bicubic`, whose four taps the family cannot state,
    /// and `clampToBorderColor`, whose border colour is a state of its own).
    fn air_sampler_policy(
        state: &metal2vulkan::reflect::StaticSamplerState,
    ) -> Option<SamplerPolicy> {
        if state.min_filter != state.mag_filter
            || state.address_mode_s != state.address_mode_t
            || state.address_mode_s != state.address_mode_r
            || state.coordinates != SamplerCoordinates::Normalized
            || state.compare_function != SamplerCompareFunction::Never
            || state.reduction != SamplerReduction::WeightedAverage
            || state.max_anisotropy != 1
        {
            return None;
        }
        let filter = match (state.min_filter, state.mip_filter) {
            (AirFilter::Nearest, SamplerMipFilter::None) => SamplerFilter::Nearest,
            (AirFilter::Linear, SamplerMipFilter::None) => SamplerFilter::Linear,
            (AirFilter::Nearest, SamplerMipFilter::Nearest) => SamplerFilter::NearestMipNearest,
            (AirFilter::Nearest, SamplerMipFilter::Linear) => SamplerFilter::NearestMipLinear,
            (AirFilter::Linear, SamplerMipFilter::Nearest) => SamplerFilter::LinearMipNearest,
            (AirFilter::Linear, SamplerMipFilter::Linear) => SamplerFilter::LinearMipLinear,
            (AirFilter::Bicubic, _) => return None,
        };
        let address = match state.address_mode_s {
            AirAddress::ClampToEdge => SamplerAddressMode::ClampToEdge,
            AirAddress::Repeat => SamplerAddressMode::Repeat,
            AirAddress::MirroredRepeat => SamplerAddressMode::MirrorRepeat,
            AirAddress::ClampToZero => SamplerAddressMode::ClampToZero,
            AirAddress::ClampToBorder => return None,
        };
        Some(SamplerPolicy { filter, address })
    }

    /// The reflected shape, reduced to the one family the canonical render
    /// sampler executes and the named refusal otherwise.
    fn sampled_shape(binding: &metal2vulkan::reflect::ResourceBinding) -> RenderTextureShape {
        let Some(shape): Option<&TextureShape> = binding.texture_shape.as_ref() else {
            return RenderTextureShape::Unsupported(RenderTextureShapeRefusal::Descriptor);
        };
        if shape.dimension != TextureDimension::D2 {
            return RenderTextureShape::Unsupported(RenderTextureShapeRefusal::Dimension);
        }
        if shape.arrayed || shape.array_ref || shape.array_length.is_some() {
            return RenderTextureShape::Unsupported(RenderTextureShapeRefusal::Arrayed);
        }
        if shape.multisampled {
            return RenderTextureShape::Unsupported(RenderTextureShapeRefusal::Multisampled);
        }
        if shape.writable {
            return RenderTextureShape::Unsupported(RenderTextureShapeRefusal::Writable);
        }
        if shape.component != TextureComponent::Float {
            return RenderTextureShape::Unsupported(RenderTextureShapeRefusal::Component);
        }
        let Some(descriptor) = binding.descriptor else {
            return RenderTextureShape::Unsupported(RenderTextureShapeRefusal::Descriptor);
        };
        if descriptor.set != 0 || descriptor.count != 1 {
            return RenderTextureShape::Unsupported(RenderTextureShapeRefusal::Descriptor);
        }
        RenderTextureShape::Sampled2D
    }

    // The AIR static samplers first: the canonical rail pairs them with the
    // *sampled* textures by position — a texel-fetched texture states no
    // sampler, so it takes no position in that pairing (R15) — and so does the
    // loop below.
    let samplers = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::StaticSampler)
        .collect::<Vec<_>>();
    // The runtime `[[sampler(n)]]` arguments beside the pairing the module's
    // own sample sites state (R12): the walk answers with *descriptor
    // bindings*, and this list is what turns a binding back into the Metal
    // index the contract names.
    let pairing = crate::runtime::spirv_bind::sampled_image_pairs(words);
    let runtime_samplers = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::Sampler)
        .filter_map(|binding| {
            crate::runtime::spirv_bind::reflected_sampler_binding(binding, false)
                .map(|slot| (binding.metal_index, slot))
        })
        .collect::<Vec<_>>();
    // The position among the *sampled* textures, which is the one the
    // positional static pairing counts against: the canonical rail skips a
    // fetched binding when it advances its own static sampler, so this walk
    // has to skip it too (R15).
    let mut static_read = 0usize;
    reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::Texture)
        .map(|texture| {
            let binding = texture
                .descriptor
                .map_or(0, |descriptor| descriptor.binding);
            // Which arm the module's own image sites state for this slot
            // (R15): a texture whose sites are all `OpImageFetch` and no
            // `OpSampledImage` is fetched, and its declaration states no
            // sampler at all. Every other answer — sampled, directly read,
            // never read, or unreadable — keeps the R10/R12 walk below, which
            // either names the sampler the module samples through or refuses
            // the shape by name.
            let fetched = crate::runtime::spirv_bind::descriptor_image_use(words, binding)
                == crate::runtime::spirv_bind::DescriptorImageUse::Fetched;
            let paired = if fetched {
                None
            } else {
                let paired = samplers.get(static_read);
                static_read += 1;
                paired
            };
            // The device binding the runtime resolves this draw's own sampler
            // bind at: the translator's sampler band is widened into the
            // device's before any shader is cached
            // (`runtime::spirv_bind::widen_sampled_bands`), and the fragment
            // sampled-band relocation is added per draw on top of that (see
            // [`RenderTextureDeclaration::sampler_binding`]). The one helper
            // that states the pair is the runtime's own, and it is the same
            // helper [`sampler_family`] reads the stage's runtime list with —
            // so the declaration and the gate name one device slot.
            let static_slot = paired
                .and_then(|sampler| {
                    crate::runtime::spirv_bind::reflected_sampler_binding(sampler, false)
                })
                .unwrap_or(0);
            // The runtime `[[sampler(n)]]` argument the module's own sample
            // sites pair this texture with, as `(metal index, device slot)`.
            let runtime_pair = pairing.sampler_of(binding).and_then(|slot| {
                runtime_samplers
                    .iter()
                    .find(|(_, runtime_slot)| *runtime_slot == slot)
                    .copied()
            });
            // The sampler-free declaration states no slot: the canonical
            // contract's fetched arm carries no sampler, so a number here
            // would be a binding nothing reads.
            let (sampler_binding, sampler) = if fetched {
                (0, RenderSamplerState::Fetched)
            } else {
                match paired.and_then(|sampler| sampler.static_sampler.as_ref()) {
                    Some(state) => (
                        static_slot,
                        match air_sampler_policy(state) {
                            Some(policy) => RenderSamplerState::Policy(policy),
                            None => RenderSamplerState::Unsupported(RenderSamplerRefusal::AirState),
                        },
                    ),
                    // No decoded AIR state beside this texture: either the
                    // AIR static sampler carries none the translator could
                    // read (the provider refuses that by name), or the texture
                    // reads through a runtime `[[sampler(n)]]` argument, whose
                    // slot and index the module's own sample sites name.
                    None => match runtime_pair {
                        Some((index, slot)) => (slot, RenderSamplerState::Runtime { index }),
                        None if paired.is_some() => (
                            static_slot,
                            RenderSamplerState::Unsupported(RenderSamplerRefusal::AirState),
                        ),
                        None => (
                            0,
                            RenderSamplerState::Unsupported(RenderSamplerRefusal::SampleSite),
                        ),
                    },
                }
            };
            RenderTextureDeclaration {
                index: texture.metal_index,
                binding,
                sampler_binding,
                sampler,
                shape: sampled_shape(texture),
            }
        })
        .collect::<Vec<_>>()
        .into()
}

/// The Metal arguments outside the family the canonical *translated* render rail
/// executes, across both stages (R10).
///
/// The translated rail binds `[[buffer(n)]]` arguments, one sampled
/// `[[texture(i)]]` per AIR static sampler, and a fragment stage's runtime
/// `[[sampler(n)]]` arguments since v102 (R12 states their pairing and state in
/// the class gate): every other kind (a storage image, a texture array, a
/// framebuffer-fetch `[[color(n)]]`, and a *vertex* stage's runtime sampler,
/// which the translated rail's texture walk never reaches) is refused by the
/// provider by name (`render_stage_unsupported_interface`), so the class gate
/// answers them here rather than letting the provider decline a draw the engine
/// could run.
#[cfg(feature = "provider-render")]
pub fn texture_interface_refusals(
    vertex: &metal2vulkan::reflect::ShaderReflection,
    fragment: &metal2vulkan::reflect::ShaderReflection,
) -> Arc<[RenderInterfaceRefusal]> {
    use metal2vulkan::reflect::ResourceKind;

    /// One reflected kind, as the refusal's sentence names it. The three kinds
    /// the translated rail executes have arms too so the match is total — a
    /// kind the translator gains later is a compile error here rather than a
    /// silent reclassification of a refusal the census already counts.
    fn kind_name(kind: ResourceKind) -> &'static str {
        match kind {
            ResourceKind::Buffer => "buffer",
            ResourceKind::Texture => "sampled texture",
            ResourceKind::StaticSampler => "AIR static sampler",
            ResourceKind::ThreadgroupBuffer => "threadgroup buffer",
            ResourceKind::KernelStageInput => "kernel stage input",
            ResourceKind::TextureArray => "texture array",
            ResourceKind::StorageImage => "storage image",
            ResourceKind::Sampler => "runtime sampler",
            ResourceKind::ColorInput => "framebuffer-fetch colour input",
            ResourceKind::AccelerationStructureShadow => "acceleration-structure shadow",
            ResourceKind::PrimitiveAccelerationStructure => "acceleration structure",
            ResourceKind::VisibleFunctionTable => "visible-function table",
            ResourceKind::IntersectionFunctionTable => "intersection-function table",
            ResourceKind::EmbeddedArgBufferTexture => "argument-buffer texture",
            ResourceKind::EmbeddedArgBufferBuffer => "argument-buffer buffer",
            ResourceKind::BufferAddressTable => "buffer address table",
            ResourceKind::SynthesizedNullTexture => "synthesized null texture",
            ResourceKind::SynthesizedReadSampler => "synthesized read sampler",
        }
    }

    fn refused(
        stage: metal_api_core::provider::RenderPipelineStage,
        reflection: &metal2vulkan::reflect::ShaderReflection,
    ) -> Vec<RenderInterfaceRefusal> {
        reflection
            .bindings
            .iter()
            .filter(|binding| {
                if matches!(
                    binding.kind,
                    ResourceKind::Buffer | ResourceKind::Texture | ResourceKind::StaticSampler
                ) {
                    return false;
                }
                // A fragment stage's runtime `[[sampler(n)]]` argument is the
                // translated rail's own family (v102, `research/docs/23`
                // §102): the class gate states the pairing the module's sample
                // sites name and the state the request binds, and answers the
                // stage-level shapes it cannot state under their own names
                // rather than under this one. A *vertex* stage's stays here:
                // the translated rail walks the fragment stage's textures.
                !(stage == metal_api_core::provider::RenderPipelineStage::Fragment
                    && binding.kind == ResourceKind::Sampler)
            })
            .map(|binding| RenderInterfaceRefusal {
                stage,
                kind: kind_name(binding.kind),
                index: binding.metal_index,
            })
            .collect()
    }
    let mut out = refused(
        metal_api_core::provider::RenderPipelineStage::Vertex,
        vertex,
    );
    out.extend(refused(
        metal_api_core::provider::RenderPipelineStage::Fragment,
        fragment,
    ));
    out.into()
}

/// The sampler family one fragment stage's translation declares (R12,
/// `research/docs/23` §102).
///
/// A *stage-level* fact like the texture declarations beside it, collected once
/// per resolved pipeline from the same reflection they are: which runtime
/// `[[sampler(n)]]` arguments the stage binds — each of which the canonical
/// contract has to pair with a sampled texture, or the registration is refused
/// by name — and whether the stage carries AIR static samplers at all, because
/// the canonical rail pairs those positionally and one stage's two sampler
/// forms are not a shape the class states.
#[cfg(feature = "provider-render")]
pub fn sampler_family(fragment: &metal2vulkan::reflect::ShaderReflection) -> RenderSamplerFamily {
    use metal2vulkan::reflect::ResourceKind;

    let runtime = fragment
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::Sampler)
        .map(|binding| RenderRuntimeSampler {
            index: binding.metal_index,
            // The same helper the declarations above use, so the class gate's
            // two halves name one device slot.
            binding: crate::runtime::spirv_bind::reflected_sampler_binding(binding, false)
                .unwrap_or(0),
        })
        .collect::<Vec<_>>();
    let statics = fragment
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::StaticSampler)
        .map(|binding| binding.metal_index)
        .collect::<Vec<_>>();
    RenderSamplerFamily {
        runtime: runtime.into(),
        statics: statics.into(),
    }
}
/// One `MTLBlendFactor` ordinal in the canonical contract's own vocabulary
/// (`research/docs/23` §100, E-RV1/v100).
///
/// The two enumerations are both Metal's list, but their *codes* are not the
/// same: the contract keeps `2`/`3` for `SourceAlpha`/`OneMinusSourceAlpha`
/// (the values v40 published) and gives the rest Metal's ordinals, so a
/// translation by ordinal number would put the source colour and source alpha
/// families on each other's factors. The mapping is therefore by *name*, one
/// arm per `MTLRenderPipeline.h` value, and an ordinal outside the list is
/// `None` — the runtime's pipeline gate parses the same list, so a value here
/// that is not a factor is a shape the engine would have refused at build time
/// rather than one this class has to execute.
fn blend_factor(ordinal: u32) -> Option<BlendFactor> {
    Some(match ordinal {
        0 => BlendFactor::Zero,
        1 => BlendFactor::One,
        2 => BlendFactor::SourceColor,
        3 => BlendFactor::OneMinusSourceColor,
        4 => BlendFactor::SourceAlpha,
        5 => BlendFactor::OneMinusSourceAlpha,
        6 => BlendFactor::DestinationColor,
        7 => BlendFactor::OneMinusDestinationColor,
        8 => BlendFactor::DestinationAlpha,
        9 => BlendFactor::OneMinusDestinationAlpha,
        10 => BlendFactor::SourceAlphaSaturated,
        11 => BlendFactor::BlendColor,
        12 => BlendFactor::OneMinusBlendColor,
        13 => BlendFactor::BlendAlpha,
        14 => BlendFactor::OneMinusBlendAlpha,
        15 => BlendFactor::Source1Color,
        16 => BlendFactor::OneMinusSource1Color,
        17 => BlendFactor::Source1Alpha,
        18 => BlendFactor::OneMinusSource1Alpha,
        _ => return None,
    })
}

/// One `MTLBlendOperation` ordinal, in the contract's own vocabulary.
///
/// Metal's order and the contract's are the same five values, and the arm is
/// written out rather than cast so a sixth value is a `None` here instead of
/// becoming one by accident.
fn blend_operation(ordinal: u32) -> Option<BlendOperation> {
    Some(match ordinal {
        0 => BlendOperation::Add,
        1 => BlendOperation::Subtract,
        2 => BlendOperation::ReverseSubtract,
        3 => BlendOperation::Min,
        4 => BlendOperation::Max,
        _ => return None,
    })
}

/// The sampled textures one request's fragment stage declares, weighed against
/// the module's own declarations (R10, `research/docs/23` §101).
///
/// Every rule below is one of the three questions the canonical render
/// sampler's contract answers, and each keeps the draw on the engine under its
/// own name rather than narrowing anything:
///
/// 1. **What the module declares.** A fragment stage whose reflection names a
///    resource family the translated rail does not execute — a storage image,
///    a texture array, a framebuffer-fetch colour input, a vertex-stage image
///    — is refused by the provider by name, so the class answers it here
///    (`render_provider_out_of_class_texture_interface`). The two texture
///    lists pair by the index each entry *states* rather than by position (the
///    rule E-RS3 landed on the canonical side, `research/docs/23` §104), so a
///    list that skips an index is one the class states — while an index the
///    contract's own bound refuses, or a list that would repeat an index or
///    walk backwards out of the module's argument order, stays on the engine
///    under `..._texture_binding`. A reflected shape outside the executable
///    family is `..._texture_shape`, and a missing or unsupported AIR static
///    sampler is `..._texture_sampler` — the state comes from the *module*, not
///    from the draw, which is exactly the rule E-RS1 landed on the canonical
///    side.
///
///    The module's `[[texture(i)]]` arguments may also read through the *other*
///    Metal sampler family, a runtime `[[sampler(n)]]` argument whose state the
///    request states (R12, `research/docs/23` §102). That family is admitted
///    through the pairing the module's own sample sites name, and the
///    stage-level shapes it cannot be admitted through are answered under their
///    own names: a stage that carries both sampler forms
///    (`..._texture_sampler_family`), more runtime arguments than the Metal
///    sampler table holds (`..._texture_sampler_count`), a runtime argument no
///    texture reads through (`..._texture_sampler_unpaired`), and a declaration
///    whose index, device slot or module the reflection does not back
///    (`..._texture_sampler_mismatch`).
/// 2. **What the draw bound.** Declaration `i` pairs with the request's own
///    bind at the device binding the runtime resolved it at; a declaration
///    without one is `..._texture_unbound`, a bind whose shape the pass cannot
///    state (dimensionality, layers, descriptor count, a texel outside the two
///    8-bit byte orders `rgba8_unorm`/`bgra8_unorm`, a view swizzle) is
///    `..._texture_bind`, and a bind whose texels are a guest
///    gather or a resident image rather than the request's own copy is
///    `..._texture_source` — this increment carries text-owned bytes the way
///    the vertex streams' staged arm does.
/// 3. **What the sampler state is.** The draw's bound sampler at the
///    declaration's slot has to repeat the state the module's AIR carries
///    (`..._texture_state`), field by field in the two fields the policy has:
///    a bind that says another filtering or address mode is the drift the
///    declaration exists to stop — the engine would sample through the bind
///    while the canonical rail creates its sampler from the declaration, and
///    the two frames would differ with nothing named.
///
///    A runtime sampler's state is the *request's* fact, so the same rule reads
///    the other way round: the draw's own bind at the argument's device slot is
///    what the pass states, and a slot the draw leaves unbound
///    (`..._texture_unbound`) or a bind outside the canonical policy family
///    (`..._texture_state`) keeps the draw on the engine.
///
/// Binds the module does not declare (a vertex stage's texture, or a fragment
/// texture no reflected slot names) are **not** an error: neither rail's module
/// reads them, so the pass states nothing for them and both frames are the
/// same — the same rule the stage-buffer door states for a bind no stage
/// declares.
fn sampled_textures<'a>(
    inputs: &RenderRailInputs<'_>,
    req: &'a DrawRequest,
) -> Result<NarrowSampling<'a>, OutOfClass> {
    if req.color_input {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_color_input",
            "a fragment stage that fetches the attachment it renders into stays on the engine: \
             the fetch is a subpass input, not a sampled texture, and the canonical pass carries \
             no input attachment",
        ));
    }
    // The canonical contract states one render texture (`MAX_RENDER_TEXTURES`),
    // and both the registration and the pass refuse a longer list by name
    // (`RenderTextureLimitExceeded`), so a wider statement is a shape this
    // class answers before the provider is asked.
    if inputs.fragment_texture_declarations.len() > MAX_RENDER_TEXTURES {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_texture_count",
            format!(
                "a fragment stage that declares {} sampled textures stays on the engine: the \
                 canonical contract states {MAX_RENDER_TEXTURES} (`MAX_RENDER_TEXTURES`), and a \
                 longer list is refused by name (`render_texture_limit`) rather than executed \
                 with the rest dropped",
                inputs.fragment_texture_declarations.len(),
            ),
        ));
    }
    if let Some(refused) = inputs.texture_interface_refusals.first() {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_texture_interface",
            format!(
                "a draw whose {} stage declares a {} at Metal index {} stays on the engine: the \
                 canonical translated rail executes `[[buffer(n)]]` arguments, one sampled \
                 `[[texture(i)]]` per AIR static sampler beside a fragment stage's runtime \
                 `[[sampler(n)]]` arguments, and every other resource kind is refused by the \
                 provider's own name \
                 (`render_stage_unsupported_interface`) rather than executed with the binding \
                 dropped",
                match refused.stage {
                    RenderPipelineStage::Vertex => "vertex",
                    RenderPipelineStage::Fragment => "fragment",
                },
                refused.kind,
                refused.index,
            ),
        ));
    }
    // The stage's sampler family (R12, `research/docs/23` §102), before any
    // texture is weighed against its bind: a stage that carries both sampler
    // forms, or more runtime `[[sampler(n)]]` arguments than Metal's own
    // sampler table holds, is a shape the canonical rail's registration rules
    // refuse by name, so the class answers it here instead of letting the
    // provider decline a draw the engine could run.
    let family = inputs.sampler_family;
    if !family.runtime.is_empty() && !family.statics.is_empty() {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_texture_sampler_family",
            format!(
                "a fragment stage that carries both AIR static samplers and runtime \
                 `[[sampler(n)]]` arguments stays on the engine: {} static sampler(s) and {} \
                 runtime sampler(s) are two forms the canonical rail pairs by different rules — \
                 the static half by position, the runtime half by the index a declaration names — \
                 and one stage's textures do not state both",
                family.statics.len(),
                family.runtime.len(),
            ),
        ));
    }
    if family.runtime.len() > MAX_RENDER_SAMPLERS {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_texture_sampler_count",
            format!(
                "a fragment stage that binds {} runtime `[[sampler(n)]]` arguments stays on the \
                 engine: Metal's sampler argument table holds {MAX_RENDER_SAMPLERS} \
                 (`MAX_RENDER_SAMPLERS`), and a longer list is refused by name \
                 (`render_sampler_limit`) rather than executed with the rest dropped",
                family.runtime.len(),
            ),
        ));
    }
    let mut textures = Vec::with_capacity(inputs.fragment_texture_declarations.len());
    // The runtime sampler states the pass states, one per Metal index the
    // admitted declarations pair with, kept in the order the declarations name
    // them and canonicalised below (the contract's list is ascending and
    // unique).
    let mut runtime_samplers: Vec<NarrowRuntimeSampler> = Vec::new();
    // The index the previous declaration stated (E-RS3, `research/docs/23`
    // §104): the canonical contract's two texture lists pair by the index each
    // entry *states* — entry `i` is whatever `[[texture(n)]]` argument it names,
    // so a list may skip an index — and each list is canonical (ascending by
    // that index, no index twice). The walk keeps the module's own argument
    // order and answers a list that would repeat an index or walk backwards by
    // name instead of reordering or dropping one of its arguments.
    let mut previous: Option<u32> = None;
    for declaration in inputs.fragment_texture_declarations.iter() {
        if let Some(previous) = previous {
            if previous == declaration.index {
                return Err(OutOfClass::owned(
                    "render_provider_out_of_class_texture_binding",
                    format!(
                        "a fragment stage that declares `[[texture({})]]` twice stays on the \
                         engine: the canonical contract's two texture lists pair by the index \
                         each entry states and hold no index twice, and the contract refuses the \
                         repeat by name (`trace_contract_invalid`, \"duplicate Metal binding {}\") \
                         rather than executing one declaration of the two",
                        declaration.index, declaration.index,
                    ),
                ));
            }
            if previous > declaration.index {
                return Err(OutOfClass::owned(
                    "render_provider_out_of_class_texture_binding",
                    format!(
                        "a fragment stage whose `[[texture({})]]` follows `[[texture({previous})]]` \
                         in its own declaration walk stays on the engine: the canonical \
                         contract's texture list is canonical — ascending by the index each entry \
                         states — so a list that would have to walk backwards is not one this \
                         class states, and the contract refuses it by name \
                         (`trace_contract_invalid`) rather than sorting a module's own arguments",
                        declaration.index,
                    ),
                ));
            }
        }
        previous = Some(declaration.index);
        // The *index* bound, one face over from the list's own order (E-RS3):
        // the count cap and the index bound are two different facts, and the
        // contract publishes this one for the fragment texture argument table
        // every admitted device's sampled-image band covers
        // (`MAX_RENDER_TEXTURE_INDEX`). An index at or above it is refused by
        // name (`render_texture_index_unsupported`) rather than bound into a
        // descriptor band the review did not cover.
        if declaration.index >= MAX_RENDER_TEXTURE_INDEX {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_texture_binding",
                format!(
                    "a fragment stage whose sampled texture is `[[texture({})]]` stays on the \
                     engine: the canonical contract's index bound for a render texture is {} \
                     (`MAX_RENDER_TEXTURE_INDEX`) and the contract refuses an index at or above it \
                     by name (`render_texture_index_unsupported`) rather than binding a \
                     descriptor band the review did not cover",
                    declaration.index, MAX_RENDER_TEXTURE_INDEX,
                ),
            ));
        }
        let sampler = match declaration.sampler {
            RenderSamplerState::Policy(policy) => NarrowSampler::Static(policy),
            // The sampler-free arm (R15): the module's own sites texel-fetch
            // this texture, so the declaration states no sampler and the class
            // has none to resolve — neither from the module nor from the
            // request. The bind the draw states at some sampler slot is not
            // this texture's fact, exactly as the canonical contract's fetched
            // arm states none.
            RenderSamplerState::Fetched => NarrowSampler::Fetched,
            // The runtime half (R12): the declaration's Metal index has to be
            // one the stage's own reflection binds, at the device slot the
            // declaration states — the two are one measurement of one
            // translation — and the state is the *request's*, read off the
            // draw's own bind at that slot.
            RenderSamplerState::Runtime { index } => {
                let Some(runtime) = family.runtime.iter().find(|runtime| runtime.index == index)
                else {
                    return Err(OutOfClass::owned(
                        "render_provider_out_of_class_texture_sampler_mismatch",
                        format!(
                            "a fragment stage whose `[[texture({})]]` is declared against runtime \
                             `[[sampler({index})]]` stays on the engine: the stage's own \
                             reflection binds no runtime sampler argument at that index, so the \
                             declaration and the module disagree, and the canonical rail refuses \
                             such a pairing by name (`render_runtime_sampler_unpaired`) rather \
                             than filling a descriptor nothing samples through",
                            declaration.index,
                        ),
                    ));
                };
                if runtime.binding != declaration.sampler_binding {
                    return Err(OutOfClass::owned(
                        "render_provider_out_of_class_texture_sampler_mismatch",
                        format!(
                            "a fragment stage whose `[[texture({})]]` reads through \
                             `[[sampler({index})]]` stays on the engine: the declaration resolves \
                             that argument at device binding {}, while the stage's own reflection \
                             resolves it at {} — the canonical rail fills the descriptor the \
                             module samples through, so a declaration naming another slot would \
                             state a pairing the two halves do not share",
                            declaration.index, declaration.sampler_binding, runtime.binding,
                        ),
                    ));
                }
                let Some(bound) = req
                    .samplers
                    .iter()
                    .find(|sampler| sampler.binding == runtime.binding)
                else {
                    return Err(OutOfClass::owned(
                        "render_provider_out_of_class_texture_unbound",
                        format!(
                            "a draw that samples `[[texture({})]]` through runtime \
                             `[[sampler({index})]]` without binding a sampler state at device \
                             binding {} stays on the engine: the module carries no state for that \
                             argument — the request states it — and the canonical rail refuses a \
                             declaration whose sampler the pass leaves unstated by name \
                             (`render_runtime_sampler_missing`) rather than filling the descriptor \
                             with a sampler nobody stated",
                            declaration.index, runtime.binding,
                        ),
                    ));
                };
                let policy = request_sampler_policy(bound).ok_or_else(|| {
                    OutOfClass::owned(
                        "render_provider_out_of_class_texture_state",
                        format!(
                            "a draw whose runtime sampler at `[[sampler({index})]]` is outside \
                             the family the canonical rail creates stays on the engine: the \
                             canonical `VkSampler` is created from nearest or linear filtering \
                             under one of the three mip filters, with one address mode on all \
                             three axes (clamped, mirror-clamped, repeated, mirror-repeated or \
                             clamped to zero), normalized coordinates, no comparison and no \
                             anisotropy, and the bind at device binding {} states another \
                             filter, another mip filter, another address mode, unnormalized \
                             coordinates, a comparison or anisotropy",
                            runtime.binding,
                        ),
                    )
                })?;
                // Canonical: the contract's sampler list is ascending by Metal
                // index and unique, and two textures may read through one
                // argument — the state is the same bind either way.
                match runtime_samplers
                    .iter_mut()
                    .find(|sampler| sampler.index == index)
                {
                    Some(existing) => existing.policy = policy,
                    None => runtime_samplers.push(NarrowRuntimeSampler { index, policy }),
                }
                NarrowSampler::Runtime { index }
            }
            RenderSamplerState::Unsupported(reason) => {
                return Err(OutOfClass::owned(
                    "render_provider_out_of_class_texture_sampler",
                    format!(
                        "a fragment stage whose `[[texture({})]]` samples through a sampler form \
                         this class cannot name stays on the engine: {}. The canonical render \
                         sampler executes the module's own AIR static state — nearest or linear \
                         filtering under one of the three mip filters, one address mode on all \
                         three axes (clamped, mirror-clamped, repeated, mirror-repeated or \
                         clamped to zero), normalized coordinates, no comparison, no anisotropy \
                         — or the runtime `[[sampler(n)]]` argument the module's own sample sites \
                         name, one per sampled texture",
                        declaration.index,
                        reason.name(),
                    ),
                ))
            }
        };
        if let RenderTextureShape::Unsupported(reason) = declaration.shape {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_texture_shape",
                format!(
                    "a fragment stage whose `[[texture({})]]` is {} stays on the engine: the \
                     canonical render sampler uploads and samples one single-sample, non-arrayed, \
                     read-only 2D surface with a float component, and the provider refuses every \
                     other reflected shape by name",
                    declaration.index,
                    reason.name(),
                ),
            ));
        }
        let Some(image) = req
            .sampled_images
            .iter()
            .find(|image| image.binding == declaration.binding)
        else {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_texture_unbound",
                format!(
                    "a draw that samples `[[texture({})]]` without binding a texture there stays \
                     on the engine: the canonical pass pairs the declaration that states \
                     `metal_binding` {} with the pass's own view at that index (E-RS3), and a \
                     declaration without a view would leave the descriptor the module reads \
                     undefined",
                    declaration.index, declaration.index,
                ),
            ));
        };
        // The texel the canonical render sampler uploads, named by the bind's
        // own Vulkan view format (E-TX1, `research/docs/23` §107): the
        // provider's whole `RENDER_SAMPLED` window is the two four-byte 8-bit
        // UNORM byte orders, and the bytes travel verbatim into the image the
        // *name* selects, so the fragment stage reads the channels the guest's
        // own view states. Any other format — the narrow single- and
        // dual-channel lanes and the wide half-float the census counts — is
        // refused by the provider under `render_texture_format_unsupported`, so
        // this class answers it here rather than handing the provider a draw
        // the engine would have run.
        let format = match image.format {
            ash::vk::Format::R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
            ash::vk::Format::B8G8R8A8_UNORM => Some(TextureFormat::Bgra8Unorm),
            _ => None,
        };
        let bindable = image.array_element == 0
            && image.descriptor_count == 1
            && image.layers == 1
            && image.kind == reims_vgpu_core::texture_shape::TextureKind::D2
            && !image.multisampled
            && image.width != 0
            && image.height != 0
            && crate::protocol::pixel_format::swizzle_is_identity(&image.swizzle);
        let Some(format) = format.filter(|_| bindable) else {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_texture_bind",
                format!(
                    "a draw that binds a texture of its own shape at `[[texture({})]]` stays on \
                     the engine: the canonical pass states one single-sample, non-arrayed 2D view \
                     with one descriptor, `rgba8_unorm` or `bgra8_unorm` texels (the provider's \
                     whole `RENDER_SAMPLED` window) and an identity channel mapping, and the bind \
                     is {}x{} {:?} (kind {:?}, layers {}, descriptors {}, element {}, multisampled \
                     {}) — a texel outside that window is refused by the provider by name \
                     (`render_texture_format_unsupported`) rather than uploaded under another \
                     format",
                    declaration.index,
                    image.width,
                    image.height,
                    image.format,
                    image.kind,
                    image.layers,
                    image.descriptor_count,
                    image.array_element,
                    image.multisampled,
                ),
            ));
        };
        // Where the texels come from (E-TX3/R22): the request's own copy, or —
        // the arm this increment opens — the trace's own production of a guest
        // target, which the trace carries ahead of this pass and samples
        // through `TextureSource::TraceView`.
        let source = match &image.source {
            crate::backend::vulkan::engine::SampledSource::Bytes(bytes) => {
                // Both window formats are four-byte texels, and the count comes
                // from the format the bind states rather than from a constant
                // here, so a later widening of the provider's window has one
                // place to answer for its own texel width.
                let expected = u64::from(image.width)
                    .checked_mul(u64::from(image.height))
                    .and_then(|texels| texels.checked_mul(format.bytes_per_texel()));
                if expected != u64::try_from(bytes.len()).ok() {
                    return Err(OutOfClass::owned(
                        "render_provider_out_of_class_texture_bind",
                        format!(
                            "a draw whose `[[texture({})]]` view carries {} byte(s) for a {}x{} \
                             {:?} surface stays on the engine: the canonical view's byte source \
                             has to be the whole tightly packed extent ({} byte(s))",
                            declaration.index,
                            bytes.len(),
                            image.width,
                            image.height,
                            image.format,
                            u64::from(image.width)
                                * u64::from(image.height)
                                * format.bytes_per_texel(),
                        ),
                    ));
                }
                NarrowTextureSource::Bytes(bytes)
            }
            crate::backend::vulkan::engine::SampledSource::Target(identity) => {
                // A record that samples the attachment it writes would need the
                // production *after* the read: the canonical contract refuses a
                // pass that samples an identity it stores as an attachment
                // (`RenderTextureAttachmentConflict`), and the trace order this
                // class states has no second reading for it either, so the
                // shape is answered by name rather than reordered.
                if req.writes_attachment(identity) {
                    return Err(OutOfClass::owned(
                        "render_provider_out_of_class_texture_source_order",
                        format!(
                            "a draw whose `[[texture({})]]` samples the very attachment it \
                             renders into stays on the engine: the texels this class serves are \
                             the trace's own earlier production of that identity, and a record \
                             that writes the view it reads would put its own store after the \
                             read — the contract refuses that shape by name \
                             (`RenderTextureAttachmentConflict`)",
                            declaration.index,
                        ),
                    ));
                }
                let Some(production) = recorded_production(identity) else {
                    return Err(OutOfClass::owned(
                        "render_provider_out_of_class_texture_source_undeclared",
                        format!(
                            "a draw whose `[[texture({})]]` texels come from a GPU target stays on \
                             the engine when no pass of this rail has declared that target's \
                             production: the canonical rail samples a trace-produced view \
                             (`TextureSource::TraceView`) by restating the pass that stored it, \
                             and the resident and guest-gather arms are the increments after \
                             this one — a production no record here stated is one this class \
                             cannot restate either",
                            declaration.index,
                        ),
                    ));
                };
                // The sampled declaration has to restate the stored surface's
                // format and extent, exactly as the contract holds it
                // (`RenderTextureSourceShapeMismatch`): the bytes the trace
                // produces are the *stored* surface's, so a declaration that
                // names another texel order or another extent would sample
                // texels the production never wrote.
                if production.format.as_texture_format() != format
                    || [production.width, production.height]
                        != [u64::from(image.width), u64::from(image.height)]
                {
                    return Err(OutOfClass::owned(
                        "render_provider_out_of_class_texture_source_shape",
                        format!(
                            "a draw whose `[[texture({})]]` is {}x{} {:?} stays on the engine \
                             when the target's own production stored {:?} at {}x{}: the \
                             canonical rail pairs a trace-produced declaration with the store \
                             that defines its bytes field by field and refuses a disagreement by \
                             name (`render_texture_source_shape_mismatch`), and a declaration \
                             that restates another shape is not one this class executes",
                            declaration.index,
                            image.width,
                            image.height,
                            image.format,
                            production.format,
                            production.width,
                            production.height,
                        ),
                    ));
                }
                NarrowTextureSource::Produced {
                    production: Arc::clone(&production),
                }
            }
            // The zero-copy guest gather: the bytes are the guest's own pages,
            // read inside the draw's command buffer, which is the increment
            // after this one (`render_provider_out_of_class_texture_source`
            // is how much of the census's stream is still behind it).
            crate::backend::vulkan::engine::SampledSource::GuestRuns(..) => {
                return Err(OutOfClass::owned(
                    "render_provider_out_of_class_texture_source",
                    format!(
                        "a draw whose `[[texture({})]]` texels are gathered from guest memory \
                         rather than from a copy the request carries stays on the engine: this \
                         class carries a sampled texture as trace-owned bytes or as the trace's \
                         own production, the way the vertex streams' two arms do, and the \
                         guest-gather arm is the increment after them",
                        declaration.index,
                    ),
                ));
            }
        };
        // The canonical render sampler executes one texture *extent*: the
        // render area's own, so every fragment's sample stands on a texel
        // centre of the surface it reads (the rail's
        // `render_texture_extent_unsupported`). A texture of another extent is
        // a shape the provider always refuses, so the class answers it here.
        if image.width != req.width || image.height != req.height {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_texture_extent",
                format!(
                    "a draw whose `[[texture({})]]` is {}x{} in a {}x{} pass stays on the engine: \
                     the canonical render sampler samples a texture of the render area's own \
                     extent (the provider refuses another extent by name, \
                     `render_texture_extent_unsupported`), and a shape the provider always \
                     refuses is not one this class executes",
                    declaration.index, image.width, image.height, req.width, req.height,
                ),
            ));
        }
        // The static half's own rule (R10): the draw's bound sampler at the
        // declaration's slot has to repeat the state the module was lowered
        // against. The runtime half's state was resolved above — there it is
        // the *request's* fact, so there is nothing to repeat.
        if let NarrowSampler::Static(sampler) = sampler {
            let Some(bound) = req
                .samplers
                .iter()
                .find(|sampler| sampler.binding == declaration.sampler_binding)
            else {
                return Err(OutOfClass::owned(
                    "render_provider_out_of_class_texture_unbound",
                    format!(
                        "a draw that samples `[[texture({})]]` without binding a sampler for the \
                         module's own AIR state stays on the engine: the declaration states the \
                         state the module was lowered against, and the engine samples through the \
                         bind — a missing bind is a state the two rails would resolve differently",
                        declaration.index,
                    ),
                ));
            };
            let bound = request_sampler_policy(bound).ok_or_else(|| {
                OutOfClass::owned(
                    "render_provider_out_of_class_texture_state",
                    format!(
                        "a draw whose sampler at `[[texture({})]]`'s slot is outside the family \
                         the canonical rail creates stays on the engine: the bind states a state \
                         the declaration could not repeat (nearest or linear filtering under one \
                         of the three mip filters, one address mode on all three axes, normalized \
                         coordinates, no comparison, no anisotropy)",
                        declaration.index,
                    ),
                )
            })?;
            if bound != sampler {
                return Err(OutOfClass::owned(
                    "render_provider_out_of_class_texture_state",
                    format!(
                        "a draw whose sampler at `[[texture({})]]`'s slot does not repeat the \
                         module's own AIR state stays on the engine: the module's samples were \
                         lowered against {sampler:?} and the engine would execute the bind's \
                         {bound:?} while the canonical rail creates its sampler from the \
                         declaration — the two rails would sample differently with nothing named",
                        declaration.index,
                    ),
                ));
            }
        }
        textures.push(NarrowTexture {
            index: declaration.index,
            width: u64::from(image.width),
            height: u64::from(image.height),
            format,
            sampler,
            source,
        });
    }
    // Every runtime `[[sampler(n)]]` argument the stage binds has to be one a
    // texture declaration pairs with: a state nothing samples through would
    // fill a descriptor slot the registration never said a texture reads
    // through, and the canonical rail refuses exactly that by name
    // (`render_runtime_sampler_undeclared`) — a refusal, not a fallback, so the
    // class answers it here rather than letting the provider decline a draw the
    // engine could run.
    if let Some(unpaired) = family
        .runtime
        .iter()
        .find(|runtime| !runtime_samplers.iter().any(|s| s.index == runtime.index))
    {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_texture_sampler_unpaired",
            format!(
                "a fragment stage that binds runtime `[[sampler({})]]` at device binding {} \
                 without a sampled texture reading through it stays on the engine: the canonical \
                 contract pairs every runtime `[[sampler(n)]]` argument with a texture \
                 declaration, and the registration refuses one nothing pairs with by name \
                 (`render_runtime_sampler_undeclared`) rather than binding a state nothing \
                 samples through",
                unpaired.index, unpaired.binding,
            ),
        ));
    }
    // The contract's sampler list is canonical: ascending Metal index, unique —
    // two textures may read through one argument, and the state is the bind's
    // either way.
    runtime_samplers.sort_unstable_by_key(|sampler| sampler.index);
    runtime_samplers.dedup_by_key(|sampler| sampler.index);
    // The productions this record's trace has to carry ahead of its own pass,
    // in the order the textures state them. Deduplicated by identity, because
    // two textures of one pass may well sample the same target and the trace
    // states its production once.
    let mut productions: Vec<Arc<RecordedProduction>> = Vec::new();
    for texture in &textures {
        let NarrowTextureSource::Produced { production, .. } = &texture.source else {
            continue;
        };
        if !productions
            .iter()
            .any(|known| known.identity == production.identity)
        {
            productions.push(Arc::clone(production));
        }
    }
    Ok(NarrowSampling {
        textures,
        runtime_samplers,
        productions,
    })
}

/// One bound sampler resource in the canonical policy's own two fields, or
/// `None` when the state is outside the family the canonical rail creates.
///
/// This is the request-side half of the rule E-RS1 landed on the canonical
/// side (`static_sampler_policy`), at the widened family E-TX2 landed there
/// (R21, `research/docs/26` §44): the two enumerations are the same family —
/// `MTLSamplerMinMagFilter`, `MTLSamplerMipFilter` and `MTLSamplerAddressMode`
/// as the runtime resolved them — and a state the canonical rail cannot create
/// is answered here rather than executed under another state.
///
/// The family is the two min/mag filters crossed with the three mip filters —
/// six names, because the mip filter is the mode the filter is selected under
/// and not a filter of its own — beside the five address modes
/// `clampToEdge`, `mirrorClampToEdge`, `repeat`, `mirrorRepeat` and
/// `clampToZero`. `clampToBorderColor` stays outside by name on both rails:
/// its border colour is a state of its own (`MTLSamplerBorderColor`) that the
/// family does not name, so a sampler created for it would answer with a
/// colour the request never stated.
fn request_sampler_policy(
    sampler: &crate::backend::vulkan::engine::SamplerResource,
) -> Option<SamplerPolicy> {
    use crate::protocol::sampler::{
        MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
        MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE, MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT,
        MTL_SAMPLER_ADDRESS_MODE_REPEAT, MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
        MTL_SAMPLER_MIN_MAG_FILTER_NEAREST, MTL_SAMPLER_MIP_FILTER_LINEAR,
        MTL_SAMPLER_MIP_FILTER_NEAREST, MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
    };
    use metal_api_core::provider::{SamplerAddressMode, SamplerFilter};
    if sampler.min_filter != sampler.mag_filter {
        return None;
    }
    if sampler.address_mode_u != sampler.address_mode_v
        || sampler.address_mode_u != sampler.address_mode_w
    {
        return None;
    }
    if sampler.unnormalized_coordinates
        || sampler.compare_function != crate::backend::vulkan::engine::SamplerCompareFunction::Never
        || sampler.max_anisotropy != 1
    {
        return None;
    }
    let filter = match (sampler.min_filter, sampler.mip_filter) {
        (MTL_SAMPLER_MIN_MAG_FILTER_NEAREST, MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED) => {
            SamplerFilter::Nearest
        }
        (MTL_SAMPLER_MIN_MAG_FILTER_LINEAR, MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED) => {
            SamplerFilter::Linear
        }
        (MTL_SAMPLER_MIN_MAG_FILTER_NEAREST, MTL_SAMPLER_MIP_FILTER_NEAREST) => {
            SamplerFilter::NearestMipNearest
        }
        (MTL_SAMPLER_MIN_MAG_FILTER_NEAREST, MTL_SAMPLER_MIP_FILTER_LINEAR) => {
            SamplerFilter::NearestMipLinear
        }
        (MTL_SAMPLER_MIN_MAG_FILTER_LINEAR, MTL_SAMPLER_MIP_FILTER_NEAREST) => {
            SamplerFilter::LinearMipNearest
        }
        (MTL_SAMPLER_MIN_MAG_FILTER_LINEAR, MTL_SAMPLER_MIP_FILTER_LINEAR) => {
            SamplerFilter::LinearMipLinear
        }
        _ => return None,
    };
    let address = match sampler.address_mode_u {
        MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE => SamplerAddressMode::ClampToEdge,
        MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE => SamplerAddressMode::MirrorClampToEdge,
        MTL_SAMPLER_ADDRESS_MODE_REPEAT => SamplerAddressMode::Repeat,
        MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT => SamplerAddressMode::MirrorRepeat,
        MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO => SamplerAddressMode::ClampToZero,
        _ => return None,
    };
    Some(SamplerPolicy { filter, address })
}

/// The blend state one request states as the canonical pass's own
/// (`research/docs/23` §100, E-RV1/v100), or the name of the shape that keeps
/// it on the engine.
///
/// `None` from this function is "the request declares no blend state at all",
/// which the canonical pass spells by leaving its `blend` list absent — the
/// no-blend, all-writes entry both rails ran before v40.
///
/// Everything else is stated **or refused by name**, never narrowed:
///
/// - a write mask other than `ALL` is a shape the wire's v40 blend section
///   cannot carry (five bytes per entry: four factors and one operation, every
///   channel written), so it keeps the engine here rather than being framed as
///   the all-writes shape it is not;
/// - an alpha operation of its own is the same wire question, one field over;
/// - the three factor families the canonical contract refuses by name — the
///   blend constant (`blend_constant_unsupported`: the pass carries no blend
///   constant), the second colour output (`blend_dual_source_unsupported`), and
///   `SourceAlphaSaturated` in a destination slot
///   (`blend_factor_slot_unsupported`) — are shapes the provider *always*
///   refuses, so the class answers them before the provider is asked, exactly
///   as it answers the attachment window.
///
/// The two halves above are the reason this gate exists: the wire question is
/// this rail's (`provider_wire` encodes the trace this function helped build),
/// and the execution question is the contract's. Neither is answered by
/// rounding one state into another.
fn declared_blend(req: &DrawRequest) -> Result<Option<RenderPassBlend>, OutOfClass> {
    if req.color_write_mask != crate::protocol::blend::ColorWriteMask::ALL {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_blend_mask",
            format!(
                "an attachment whose colour write mask is {:#x} stays on the engine: the \
                 canonical pass can state a mask, but the command channel's v40 blend section \
                 carries four factors and one operation per entry and every channel written, so \
                 a masked entry is a shape this rail's wire refuses by name \
                 (`RenderBlendStateUnsupported`) rather than framing as the all-writes shape it \
                 is not",
                req.color_write_mask.bits(),
            ),
        ));
    }
    let Some(state) = req.blend else {
        return Ok(None);
    };
    let factor = |field: &'static str, ordinal: u32| {
        blend_factor(ordinal).ok_or_else(|| {
            OutOfClass::owned(
                "render_provider_out_of_class_blend_ordinal",
                format!(
                    "a blend whose {field} factor is the ordinal {ordinal} stays on the engine: \
                     {ordinal} is not an `MTLBlendFactor`, so the canonical pass has no factor to \
                     state there",
                ),
            )
        })
    };
    let source_rgb = factor("source rgb", state.src_rgb)?;
    let destination_rgb = factor("destination rgb", state.dst_rgb)?;
    let source_alpha = factor("source alpha", state.src_alpha)?;
    let destination_alpha = factor("destination alpha", state.dst_alpha)?;
    let operation = blend_operation(state.op_rgb).ok_or_else(|| {
        OutOfClass::owned(
            "render_provider_out_of_class_blend_ordinal",
            format!(
                "a blend whose rgb operation is the ordinal {} stays on the engine: it is not an \
                 `MTLBlendOperation`, so the canonical pass has no operation to state there",
                state.op_rgb,
            ),
        )
    })?;
    let alpha_operation = blend_operation(state.op_alpha).ok_or_else(|| {
        OutOfClass::owned(
            "render_provider_out_of_class_blend_ordinal",
            format!(
                "a blend whose alpha operation is the ordinal {} stays on the engine: it is not \
                 an `MTLBlendOperation`, so the canonical pass has no operation to state there",
                state.op_alpha,
            ),
        )
    })?;
    // The v40 section is one operation per entry, shared by the colour and
    // alpha equations. Both APIs state two, and this class states both — the
    // wire is the half that cannot carry the second one.
    if alpha_operation != operation {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_blend_alpha_operation",
            format!(
                "a blend whose alpha operation ({}) differs from its colour operation ({}) stays \
                 on the engine: the canonical pass states both, but the command channel's v40 \
                 section carries one operation per entry, so the difference is a shape this \
                 rail's wire refuses by name (`RenderBlendStateUnsupported`) rather than \
                 dropping",
                state.op_alpha, state.op_rgb,
            ),
        ));
    }
    for (slot, factor_value) in [
        (BlendFactorSlot::SourceRgb, source_rgb),
        (BlendFactorSlot::DestinationRgb, destination_rgb),
        (BlendFactorSlot::SourceAlpha, source_alpha),
        (BlendFactorSlot::DestinationAlpha, destination_alpha),
    ] {
        if factor_value.needs_blend_constant() {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_blend_constant",
                format!(
                    "a blend whose {} factor is {factor_value:?} stays on the engine: it reads \
                     the encoder's blend colour, the canonical pass carries no blend constant, \
                     and a shape the provider always refuses (`blend_constant_unsupported`) is \
                     not one this class executes",
                    slot.name(),
                ),
            ));
        }
        if factor_value.is_dual_source() {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_blend_dual_source",
                format!(
                    "a blend whose {} factor is {factor_value:?} stays on the engine: it \
                     reads the fragment shader's second colour output, which the reviewed stages \
                     do not declare, and a shape the provider always refuses \
                     (`blend_dual_source_unsupported`) is not one this class executes",
                    slot.name(),
                ),
            ));
        }
        if slot.is_destination() && factor_value == BlendFactor::SourceAlphaSaturated {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_blend_factor_slot",
                format!(
                    "a blend whose {} factor is SourceAlphaSaturated stays on the engine: both \
                     APIs define that factor for a source slot only, and a shape the provider \
                     always refuses (`blend_factor_slot_unsupported`) is not one this class \
                     executes",
                    slot.name(),
                ),
            ));
        }
    }
    Ok(Some(RenderPassBlend {
        attachments: vec![BlendAttachment {
            enabled: true,
            source_rgb,
            destination_rgb,
            source_alpha,
            destination_alpha,
            operation,
            alpha_operation,
            write_mask: ColorWriteMask::ALL,
        }],
    }))
}

/// Whether one viewport rectangle is a shape the canonical pass can state
/// (`research/docs/23` §100, E-RV1/v100).
///
/// The contract's `viewport` is a *rect* in framebuffer coordinates —
/// `[origin_x, origin_y, width, height]`, the rectangle NDC maps onto — and it
/// is stated in the pass body, so this face needs no wire section. What it
/// cannot state is a rect outside the attachment (the contract refuses it by
/// name, `viewport_extent_unsupported`, rather than clipping), a rect whose
/// origin or extent has no `u32` spelling (Metal states the four numbers as
/// `f64`; a negative or fractional one is a shape the engine draws and the
/// contract cannot name), a rect whose depth range is not `0..1`, and more than
/// one rect (the canonical pass carries one). Every one of those is the
/// engine's own answer to give, so each keeps the draw on the engine under its
/// own name instead of being clamped or truncated here.
fn viewport_admits(
    viewport: crate::backend::vulkan::engine::ViewportResource,
    width: u32,
    height: u32,
) -> Result<[u32; 4], OutOfClass> {
    // The engine states the four numbers as `f32`; the contract as `u32`. The
    // conversions below are total in the order they are written: `f64` widens
    // the `f32` exactly, so `fract` and the comparison read the value the
    // engine will rasterize with, and the cast is only reached for an integral
    // value inside the `u32` range.
    let spelling = |value: f32| -> Option<u32> {
        let wide = f64::from(value);
        if !wide.is_finite() || wide < 0.0 || wide.fract() != 0.0 || wide > f64::from(u32::MAX) {
            return None;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(wide as u32)
    };
    let (Some(x), Some(y), Some(width_u32), Some(height_u32)) = (
        spelling(viewport.x),
        spelling(viewport.y),
        spelling(viewport.width),
        spelling(viewport.height),
    ) else {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_viewport_spelling",
            format!(
                "a viewport of {}x{} at ({}, {}) stays on the engine: the canonical pass states \
                 the rect as four integers in framebuffer coordinates ([origin_x, origin_y, \
                 width, height]), and a negative, fractional or over-wide number is a rect the \
                 frozen pass descriptor has no spelling for — truncating it here would draw a \
                 rect the engine never drew",
                viewport.width, viewport.height, viewport.x, viewport.y,
            ),
        ));
    };
    if width_u32 == 0 || height_u32 == 0 {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_viewport_empty",
            format!(
                "a {width_u32}x{height_u32} viewport stays on the engine: the canonical pass \
                 refuses a rect with an empty extent by name, while the engine rasterizes through \
                 it — a draw that covers no texel is the engine's own answer to give, not a shape \
                 this class repairs",
            ),
        ));
    }
    if viewport.min_depth != 0.0 || viewport.max_depth != 1.0 {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_viewport_depth",
            format!(
                "a viewport with depth range {}..{} stays on the engine: the canonical pass \
                 states the rect alone and both rails rasterize with the 0..1 range, so a \
                 different range is state the frozen pass descriptor cannot carry",
                viewport.min_depth, viewport.max_depth,
            ),
        ));
    }
    let inside = u64::from(x)
        .checked_add(u64::from(width_u32))
        .is_some_and(|end| end <= u64::from(width))
        && u64::from(y)
            .checked_add(u64::from(height_u32))
            .is_some_and(|end| end <= u64::from(height));
    if !inside {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_viewport_extent",
            format!(
                "a viewport of {width_u32}x{height_u32} at ({x}, {y}) stays on the engine: it \
                 reaches outside the {width}x{height} attachment, the canonical pass refuses a \
                 rect outside the raster by name (`viewport_extent_unsupported`) rather than \
                 clipping it, and the engine clamps such a rect — so this is the engine's own \
                 answer to give",
            ),
        ));
    }
    Ok([x, y, width_u32, height_u32])
}

/// Whether a request's declared attributes name exactly the locations the
/// vertex stage's own translation reports.
///
/// The request's attribute list is the engine's own shape — one entry per
/// attribute *location* — and the reflection lists one input per location, so
/// the comparison is a linear walk of one list against the other rather than a
/// set. A request with two attributes at one location cannot pass it: the
/// reflected locations are distinct, so a duplicate leaves one of them
/// uncovered.
fn attribute_locations_match(attributes: &[VertexAttributeResource], reflected: &[u32]) -> bool {
    attributes.len() == reflected.len()
        && attributes
            .iter()
            .all(|attribute| reflected.contains(&attribute.location))
}

/// Whether two of a request's attributes read one vertex stream.
///
/// Three facts decide it, and all three are the request's own. The staged bytes
/// the two were resolved from: the runtime resolves every attribute that names
/// one guest vertex buffer through that buffer's single bind, and
/// [`BufferContent`]'s own contract is that the allocation behind a bind is
/// *shared* — "several attributes on one interleaved stream, or a stage-in
/// buffer doubling as a storage bind, reference the same allocation instead of
/// cloning it" — so two attributes off one interleaved guest stream carry the
/// same `Arc`, whether the bind's bytes are the owner's staged copy
/// ([`BufferContent::Bytes`]) or a zero-copy bind's `runs` allocation
/// ([`BufferContent::GuestRuns`], the same sharing contract's second origin;
/// R9q). The stride the descriptor declared for that buffer, and the step
/// beside it: two attributes of one table advance by one stride at one rate.
///
/// Two attributes that agree on all three read one table at two offsets, which
/// is one canonical vertex stream. Nothing else in the request can join them:
/// bytes that merely *compare* equal are two allocations and stay two streams,
/// which is the conservative reading — the layout then states more streams than
/// the guest's own descriptor had, and the gate's vertex-layout rule answers
/// the same draws it would have answered had this grouping never landed.
///
/// A zero-copy bind is grouped by the *bind*, not by comparing bytes: the bytes
/// belong to guest RAM this process maps, and reading them here would be a host
/// read the zero-copy origin exists to avoid. Two `GuestRuns` name one table
/// when they share the `runs` allocation — one per resolved bind — at the same
/// window inside it (`source_offset` and `total_len` are what a packed
/// resource's per-offset bind varies, `research/docs/26` §14.2), which is
/// exactly the shape a request that resolved one guest buffer twice hands this
/// rail. A gather that is neither a staged copy nor one registered window is
/// answered by name before any layout is stated from it ([`narrow_class`]'s
/// `vertex_staging` arm), so it never reaches this walk.
fn one_vertex_stream(a: &VertexAttributeResource, b: &VertexAttributeResource) -> bool {
    a.stride == b.stride
        && a.step_function == b.step_function
        && a.step_rate == b.step_rate
        && match (&a.content, &b.content) {
            (BufferContent::Bytes(left), BufferContent::Bytes(right)) => {
                std::sync::Arc::ptr_eq(left, right)
            }
            // R9q: the zero-copy origin of the same sharing contract. The
            // `runs` allocation's identity is the bind's, and the window
            // beside it keeps two binds of one packed resource — same runs,
            // different offsets — two tables rather than one.
            (BufferContent::GuestRuns(left), BufferContent::GuestRuns(right)) => {
                std::sync::Arc::ptr_eq(&left.runs, &right.runs)
                    && left.source_offset == right.source_offset
                    && left.total_len == right.total_len
            }
            _ => false,
        }
}

/// How many canonical vertex streams a request's attributes form.
///
/// The count [`stage_buffer_gate`]'s vertex-layout rule reads, and the one the
/// canonical layout is built with: both come from this walk, so the number a
/// refusal names and the number a registration states are one measurement
/// rather than two spellings of it.
fn canonical_vertex_stream_count(attributes: &[VertexAttributeResource]) -> usize {
    let mut heads: Vec<&VertexAttributeResource> = Vec::new();
    for attribute in attributes {
        if !heads.iter().any(|head| one_vertex_stream(head, attribute)) {
            heads.push(attribute);
        }
    }
    heads.len()
}

/// The registered window one zero-copy gather was cut from (`R9e`).
///
/// The gather's page runs are the guest RAM rail's own bounded references into
/// the import this process holds, one per maximal stretch, ascending and tiling
/// the requested window exactly; each carries the provider-shaped window the
/// registration ledger derived for its own bound
/// (`crate::runtime::guest_ram_map::GuestWindowRun::window`). The two facts this
/// function reads are therefore both already derived — where the stretch's
/// window is, and where inside the stretch the bind's window begins
/// (`WindowStretch::skip`, the `source_offset` a packed resource binds at).
///
/// `None` for every shape a borrowed lease cannot name, and the caller keeps
/// the draw on the engine for each of them rather than inventing coordinates:
///
/// - **more than one stretch** (`single_stretch`): the bytes are scattered, so
///   no single host range is the bind's bytes — the GPU copy per stretch is an
///   engine rail (or a named `_gather` bucket here), not a lease;
/// - **no registered window on the stretch** (`window: None`): the import was
///   never registered under the current epoch, or its registration was refused,
///   so there is no provider region to cut a window from;
/// - **a window that does not cover the bind's own bytes**: a `source_offset`
///   reaching past the stretch's window is a malformed source rather than a
///   slow one, and stating it would name bytes the stage never reads.
///
/// What the returned window still has to satisfy is a device fact — the view's
/// own host pointer has to be a whole number of the provider's import
/// alignment, or the canonical rail refuses the import by name
/// (`resolve_render_input`'s lease alignment refusal) — so that check lives
/// with the other device answer in [`submit_render`], not here.
fn gather_window(source: &GuestRunSource) -> Option<StageBufferWindow> {
    let stretch = source.single_stretch()?;
    let window = stretch.window?;
    if stretch.len == 0 || window.length == 0 {
        return None;
    }
    // The window has to be the bind's own bytes: the view is what the stage
    // reads, so a head that reaches past the window, or a bind longer than what
    // is left of it, is a source this rail cannot state as a lease.
    if stretch.skip.checked_add(stretch.len)? > window.length {
        return None;
    }
    Some(StageBufferWindow {
        import: window.import.get(),
        host_va: window.base,
        length: window.length,
        head: stretch.skip,
        bytes_len: stretch.len,
    })
}

/// Where one admitted stream's bytes come from (`R9q`, `R11`).
///
/// The same two arms one stage buffer has ([`NarrowStageBuffer`]), on the same
/// rule: a stream the request already holds as staged bytes travels as
/// trace-owned bytes, and a stream the draw path resolved through the zero-copy
/// rail travels as the registered window its bind was cut from — the
/// `BufferSource::BorrowedNoCopy` arm of the lease channel, imported by
/// [`plan_owner_leases`] before the trace exists, exactly as a stage buffer's
/// window is. Nothing is copied on either arm: the staged arm's bytes are the
/// allocation the request already holds, and the window arm's are the owner's
/// own mapping.
///
/// One enum and not two, because the draw path hands both a vertex stream and
/// an index buffer through the same zero-copy resolution
/// (`crate::runtime::bound_buffers::BoundBuffer`), so the two are the same kind
/// of bind and differ only in which shape's sentence answers a gather this rail
/// cannot state.
enum StreamSource<'a> {
    /// The owner's staged copy: `BufferContent::Bytes`.
    Staged(&'a [u8]),
    /// The registered guest RAM window the bind's bytes were cut from.
    Window(StageBufferWindow),
}

impl StreamSource<'_> {
    /// The bytes this stream's view covers.
    ///
    /// The staged arm's allocation length, or the bind's own bytes inside its
    /// window — the number the record-length rule below reads and the length
    /// the trace states for the view, so the two cannot drift apart.
    fn len(&self) -> u64 {
        match self {
            Self::Staged(bytes) => u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            Self::Window(window) => window.bytes_len,
        }
    }
}

/// The bytes one stream may be stated from, arm by arm (`R9q`, `R11`).
///
/// The decision itself, so the two shapes that reach it cannot disagree about
/// which arms exist: staged bytes are trace-owned, and a zero-copy bind is the
/// window the seam derives from the source's own gather ([`gather_window`]).
/// `None` is every gather the seam cannot cut one window from — scattered over
/// stretches, never registered under the current epoch, or with a bind reaching
/// past its stretch's window — and each caller turns it into its own shape's
/// named refusal, because the census reads the shape that crossed the device.
fn stream_source(content: &BufferContent) -> Option<StreamSource<'_>> {
    match content {
        BufferContent::Bytes(bytes) => Some(StreamSource::Staged(bytes)),
        BufferContent::GuestRuns(source) => gather_window(source).map(StreamSource::Window),
    }
}

/// The source one request attribute's bytes may be stated from (`R9q`).
///
/// Staged bytes are the pre-R9q arm unchanged. A zero-copy bind leaves through
/// the window the seam derives from its own gather ([`gather_window`]); a
/// gather the seam cannot cut one window from — scattered over stretches,
/// never registered under the current epoch, or with a bind reaching past its
/// stretch's window — keeps the draw on the engine under
/// `render_provider_out_of_class_vertex_staging`, which is the bucket every
/// gather answered with before this increment. The narrowing is the arm, not
/// the rule: the class only claims a gather it can state as one contiguous
/// registered window, and everything else stays where it was.
fn vertex_stream_source(content: &BufferContent) -> Result<StreamSource<'_>, OutOfClass> {
    stream_source(content).ok_or_else(|| {
        OutOfClass::new(
            "render_provider_out_of_class_vertex_staging",
            "a vertex stream the GPU gathers from guest RAM stays on the engine when the \
             gather is not one registered window: this class mints a stream's bytes through \
             the owner rail — the staged copy the request holds, or the registered window a \
             zero-copy bind was cut from — and a gather that is neither has no source this \
             rail can state",
        )
    })
}

/// The source the draw's one index stream may be stated from (`R11`).
///
/// The index sibling of [`vertex_stream_source`]: the draw path resolves an
/// indexed draw's buffer through the same zero-copy rail
/// (`load_index_content_reason` → `load_buffer_content_resolved`), so an index
/// bind can arrive as a gather for exactly the same reason a vertex bind does,
/// and the window it can be stated from is derived by the same
/// [`gather_window`]. Staged bytes are the pre-R11 arm unchanged; a gather the
/// seam cannot cut one window from keeps the draw on the engine under
/// `render_provider_out_of_class_index_staging`, the bucket every index gather
/// answered with before this increment.
fn index_stream_source(content: &BufferContent) -> Result<StreamSource<'_>, OutOfClass> {
    stream_source(content).ok_or_else(|| {
        OutOfClass::new(
            "render_provider_out_of_class_index_staging",
            "an index stream the GPU gathers from guest RAM stays on the engine when the \
             gather is not one registered window: this class mints a stream's bytes through \
             the owner rail — the staged copy the request holds, or the registered window a \
             zero-copy bind was cut from — and a gather that is neither has no source this \
             rail can state",
        )
    })
}

/// The stage-buffer gate: the v2 census's 99.3% door, answered by what each
/// stage's own translation declares (`research/docs/23` §3.3, v83 /
/// `research/docs/26` §R9b and §R9d).
///
/// The door used to be one condition — `!req.storage_buffers.is_empty()` — and
/// it kept every draw whose stages bind buffers on the engine, with the
/// sentence "the canonical render contract has no buffer bindings for a
/// pipeline". R9b replaced that with the fact the canonical pair is keyed on —
/// whether either stage's *translation* declares a `[[buffer(N)]]` argument —
/// and R9d answers the declarations themselves: the pair the canonical rail
/// executes is the pipeline's declaration (stage, index, access, footprint —
/// v83) beside the pass's own view of the bytes, and a translated stage that
/// names a buffer is paired against that contract field by field
/// (`metal-api-vulkan`'s `validate_translated_stage_buffers`, `9aa64d5`).
///
/// The split is by the declaring stage's own access class, in the vocabulary
/// the engine's bind census already counts (`access_unused` /
/// `access_dereferenced` / `access_undeclared`) and the canonical contract
/// already states (`BufferAccess`):
///
/// - **neither stage declares one**: every bound stage buffer fills an index no
///   descriptor can be needed for, so the canonical pass declares and binds
///   none of them and the draw leaves for the provider. Both rails land the
///   same bytes for it, because the bytes that differ are the bytes no stage
///   reads — which is the falsifiable half of this population (`provider_
///   render_rail.rs`: the sentinel bind moves nothing, the stream bytes beside
///   it move the frame);
/// - **a stage declares one**: the three classes the canonical contract cannot
///   state — `unused` (the entry never reaches it), `write` (the contract
///   admits `BufferAccess::Read` alone) and `unknown` (fail closed) — keep the
///   draw on the engine under the bucket its own access names, exactly as R9b
///   split them;
/// - **a read-only declaration** is the class R9d lifts: the draw leaves for
///   the provider when the *whole* pair can be stated — every declaration in
///   both stages is read-only, the request binds the stage's own index, the
///   reflection's reach is a static byte extent the bind's bytes cover, and
///   those bytes come from the owner rail ([`StageBufferBind::content`] or the
///   registered window beside it). Every fact between the request and that
///   pair is its own bucket (`_unbound`, `_footprint`, `_short`, `_window`,
///   `_gather`, `_shape`, `_import`) rather than a looser declaration.
///
/// The count of binds rides in the sentence rather than in the slug: the slug
/// is the census bucket and has to stay a property of the *shape*, while the
/// count is a property of this request.
///
/// # Where the derived window comes from (R9e)
///
/// A bind the draw path resolved through the zero-copy rail arrives as a
/// gather ([`BufferContent::GuestRuns`]) whose page runs are the *bounded
/// references into this process's import* the guest RAM rail cut when it built
/// the resolution (`crate::runtime::guest_ram_map::references_for_runs`), and
/// each run carries the provider-shaped window the registration ledger derived
/// for it. So the window a borrowed stage buffer travels under is a fact of the
/// gather itself — [`gather_window`] reads it off [`GuestRunSource::single_stretch`]
/// rather than asking the seam to restate coordinates the ledger already
/// produced. One stretch whose window covers exactly the bind's own bytes is
/// the whole condition; every other shape keeps the draw on the engine under
/// the bucket its own fact names, which is what makes this a narrowing and not
/// a second bind policy.
fn declared_stage_buffer_support(
) -> Result<provider_wire::StageBufferSupport, ProviderRenderDecline> {
    let rail = rail().map_err(IntoRender::into_render)?;
    provider_wire::stage_buffer_support(rail.provider.device_epoch(), &rail.provider.capabilities())
        .map_err(|decline| ProviderRenderDecline::StageBufferWire {
            step: decline.step,
            detail: decline.detail,
        })
}

/// The two invocation counts one draw's affine stage-buffer footprint is
/// bounded by (`research/docs/23` §3.3, v86).
///
/// The same two numbers the canonical contract reads out of the trace's own
/// index bytes (`metal-api-core`'s `render_affine_axis_counts`): axis 0 counts
/// the vertices the draw names — for this class's indexed draws
/// `base_vertex + highest index + 1`, over the same bytes the pass binds — and
/// axis 1 its instances, which the class fixes at one. `None` is a draw whose
/// index bytes do not travel with the trace (a lease-backed window, `R11`),
/// which is a proof this rail cannot evaluate here: the stage-buffer gate keeps
/// such a draw on the engine by name rather than inventing a bound over bytes
/// this derivation cannot read.
fn stage_buffer_affine_counts(req: &DrawRequest) -> Option<[u64; 2]> {
    use crate::backend::vulkan::engine::IndexType;
    let index = req.indexed.as_ref()?;
    let bytes = staged_bytes(&index.content)?;
    let width = match index.index_type {
        IndexType::U16 => 2,
        IndexType::U32 => 4,
    };
    let count = usize::try_from(index.index_count).ok()?;
    let readable = count.checked_mul(width)?;
    if bytes.len() < readable {
        return None;
    }
    let mut highest = 0u64;
    for chunk in bytes.chunks_exact(width).take(count) {
        let value = match index.index_type {
            IndexType::U16 => u64::from(u16::from_ne_bytes([chunk[0], chunk[1]])),
            IndexType::U32 => {
                u64::from(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            }
        };
        highest = highest.max(value);
    }
    let vertices = u64::try_from(index.vertex_offset)
        .ok()?
        .checked_add(highest)?
        .checked_add(1)?;
    Some([vertices, u64::from(req.instance_count.unwrap_or(1))])
}

/// The byte extent one affine proof reaches over a draw's own invocation counts
/// (`metal-api-core`'s `render_affine_required_bytes`): each access reaches
/// `base + size + (count - 1) * stride` per term, and the widest access is what
/// the pass's view has to cover. `None` is an expression that overflows or
/// names an axis the draw does not have — a proof the contract refuses by name
/// rather than one this rail evaluates against a bound nothing states.
fn stage_buffer_affine_required_bytes(accesses: &[AffineAccess], counts: [u64; 2]) -> Option<u64> {
    let mut required = 0u64;
    for access in accesses {
        let mut end = access.base_offset.checked_add(access.access_size)?;
        for term in &access.terms {
            let count = counts.get(usize::from(term.axis))?;
            end = end.checked_add(count.saturating_sub(1).checked_mul(term.stride)?)?;
        }
        required = required.max(end);
    }
    Some(required)
}

/// What one frame carries, printed from the *decoded* trace
/// (`research/docs/26` §R9j).
///
/// One line per declared binding: the pipeline's own declaration (stage,
/// index, access, footprint) beside the pass's view (access, offset, length,
/// source arm) — read back out of the decoded value rather than out of the
/// values this rail built, because it is the frame's reading that the
/// provider's registration and admission pair. A declaration that decoded
/// differently from the one stated is exactly what this line makes visible.
fn note_wire_stage_buffers(trace: &ComputeTrace) {
    let declarations = trace
        .pipelines
        .iter()
        .find_map(|pipeline| pipeline.render.as_ref())
        .map(|render| render.stage_buffers.as_slice())
        .unwrap_or(&[]);
    let views = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .map(|pass| pass.stage_buffers.as_slice())
        .unwrap_or(&[]);
    for declaration in declarations {
        let view = views.iter().find(|view| {
            view.stage == declaration.stage && view.view.metal_binding == declaration.index
        });
        let source = match view.map(|view| &view.view.source) {
            Some(BufferSource::OwnedBytes(bytes)) => format!("owned_bytes={}", bytes.len()),
            Some(BufferSource::StagedLease(lease)) => format!("staged_lease={}", lease.get()),
            Some(BufferSource::BorrowedNoCopy(lease)) => {
                format!("borrowed_lease={}", lease.get())
            }
            None => "view=absent".to_owned(),
        };
        crate::observe::line(format!(
            "render_provider_wire stage_buffer stage={} index={} access={} footprint={} {} \
             view_access={} offset={} length={}",
            declaration.stage.name(),
            declaration.index,
            buffer_access_name(declaration.access),
            footprint_name(&declaration.footprint),
            source,
            view.map(|view| buffer_access_name(view.view.access))
                .unwrap_or("none"),
            view.map(|view| view.view.offset).unwrap_or(0),
            view.map(|view| view.view.length).unwrap_or(0),
        ));
    }
}

/// The name one contract access is reported under, in the spelling the
/// refusal's fields use (`metal-api-vulkan`'s own `buffer_access_name`).
fn buffer_access_name(access: BufferAccess) -> &'static str {
    match access {
        BufferAccess::Read => "read",
        BufferAccess::Write => "write",
        BufferAccess::ReadWrite => "read_write",
        BufferAccess::Unused => "unused",
    }
}

/// The name one footprint proof is reported under, with the number that makes
/// it checkable: a static ceiling in bytes, or an affine access set.
fn footprint_name(proof: &FootprintProof) -> String {
    match proof {
        FootprintProof::Static { max_bytes } => format!("static bytes={max_bytes}"),
        FootprintProof::Affine { accesses } => format!("affine accesses={}", accesses.len()),
        FootprintProof::Unbounded => "unbounded".to_owned(),
    }
}

/// The request's two stages' `[[buffer(N)]]` statement, every rule the class
/// answers it under, in the order the census reads them.
///
/// `vertex_streams` is the number of canonical vertex streams the request's
/// attributes form — its fetch tables, not its attributes
/// ([`canonical_vertex_stream_count`]) — because that is the number of
/// bindings the canonical layout will occupy and the number the contract's own
/// vertex-layout rule is written against (`research/docs/26` §31).
fn stage_buffer_gate<'a>(
    inputs: &'a RenderRailInputs<'a>,
    req: &DrawRequest,
    binds: usize,
    vertex_streams: usize,
) -> Result<Vec<NarrowStageBuffer<'a>>, OutOfClass> {
    // The one statement this request's two stages make (R9m): `declared` is
    // what the contract, the pass's own views and the wire frame are built
    // from, and `unstated` is the class's own account of what the statement
    // left out — the slots it now lets through (R9n) and the one arm that still
    // answers.
    let statement = inputs.stage_buffer_statement();
    // The arm whose entry never dereferences the slot is *ignored* (R9n): the
    // statement drops it, and the canonical registration leaves a reflected
    // `Unused` slot out of the pairing when the contract does not declare it
    // (`research/docs/23` §95) — nothing reads the slot, so it needs no
    // declaration and carries no view. The arm the translation does not
    // classify is still fail-closed, and still answers in the statement's own
    // order: the vertex half first — the order the contract states its two
    // buffers in — and inside a stage the reflection's own order, which is what
    // `pipeline_resolve` collected.
    if let Some((stage, declaration)) = statement
        .unstated
        .iter()
        .find(|(_, declaration)| declaration.access != StageBufferAccess::Unused)
    {
        let stage_name = stage.name();
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_stage_buffer_unknown",
            format!(
                "a draw whose {stage_name} stage declares a [[buffer({})]] argument stays on the \
                 engine: the canonical contract states pipeline-level buffers (v83) and pairs a \
                 declaration with a translation that reaches it, while this declaration's access \
                 ({access}) is one the translation does not classify — fail-closed, exactly as \
                 the engine's own bind path answers it. The request binds {binds} stage \
                 buffer(s)",
                declaration.index,
                access = declaration.access.name(),
            ),
        ));
    }
    // Nothing to state: a request whose stages declare no `[[buffer(N)]]`
    // argument at all keeps the path it always had (R9b), and since R9n the
    // same holds for a request whose declarations are all slots the entries
    // never dereference — the dropped slot crosses into the provider neither
    // declared nor bound, exactly as the binds of an argument no stage declares
    // already did.
    if statement.declared.is_empty() {
        return Ok(Vec::new());
    }
    // The stated half is built in the contract's own canonical order — vertex
    // bindings first by index, then fragment bindings — rather than the
    // reflection's order, because that order is a rule of the canonical list
    // (`NonCanonicalBindingOrder`) and not a preference.
    let mut ordered = statement.declared;
    ordered.sort_by_key(|(stage, declaration, _)| (stage.code(), declaration.index));
    if ordered.len() > MAX_RENDER_STAGE_BUFFERS {
        // The first of the three rules this slug names, counted under its own
        // route so the next census can size it (R9o): the refusal itself is
        // unchanged, still the bare shape slug.
        note_stage_buffer_shape(StageBufferShapeRoute::TooMany);
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_stage_buffer_shape",
            format!(
                "a draw whose stages declare {} stage buffers stays on the engine: the canonical \
                 contract states at most {MAX_RENDER_STAGE_BUFFERS} pipeline-level buffers, and a \
                 longer list is a shape admission refuses by name",
                ordered.len(),
            ),
        ));
    }
    let mut out: Vec<NarrowStageBuffer<'a>> = Vec::with_capacity(ordered.len());
    for (stage, declaration, access) in ordered {
        // One slot, one declaration: a reflection that names the same stage
        // buffer twice describes two interfaces for one index, which the
        // canonical list refuses as a duplicate rather than reconciling.
        if out
            .iter()
            .any(|stated| stated.stage == stage && stated.index == declaration.index)
        {
            // The duplicate rule's own route (R9o), beside the unchanged slug.
            note_stage_buffer_shape(StageBufferShapeRoute::Duplicate);
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_stage_buffer_shape",
                format!(
                    "a draw whose {} stage declares [[buffer({})]] twice stays on the engine: one \
                     stage buffer is one declaration, and the canonical list refuses a slot named \
                     twice",
                    stage.name(),
                    declaration.index,
                ),
            ));
        }
        // A vertex stage buffer may not occupy an index the pipeline's vertex
        // layout already describes: this class states the request's streams as
        // canonical bindings `0..vertex_streams`, and one binding described
        // twice is what the contract refuses by name
        // (`StageBufferVertexLayoutConflict`).
        //
        // `vertex_streams` is the number of the request's own *fetch tables*
        // (`canonical_vertex_stream_count`), not its attribute count, and the
        // canonical layout is built from that same walk — so the rule reads the
        // same number the registration will (`research/docs/26` §31). It is the
        // contract's own rule mirrored here rather than a second opinion: a
        // shape whose declaration lands inside the block is one the provider's
        // admission refuses, and a refusal is a decline, not a fallback.
        if stage == RenderPipelineStage::Vertex && (declaration.index as usize) < vertex_streams {
            // The layout-collision rule's own route (R9o), beside the
            // unchanged slug.
            note_stage_buffer_shape(StageBufferShapeRoute::VertexLayout);
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_stage_buffer_shape",
                format!(
                    "a draw whose vertex stage declares a [[buffer({})]] argument stays on the \
                     engine: this class states the request's {vertex_streams} vertex stream(s) as \
                     the canonical layout's own bindings, and the contract refuses a stage buffer \
                     that occupies one of those indices",
                    declaration.index,
                ),
            ));
        }
        // The declaration's own proof, stated as the contract states it
        // (`FootprintProof`), beside the number this rail proves the bind's
        // bytes against. A static declaration is a ceiling; an affine one is
        // the reflected access set itself, evaluated here over the draw's own
        // invocation counts — the same arithmetic (`stage_buffer_affine_counts`
        // → `stage_buffer_affine_required_bytes`) the canonical admission runs
        // over the same bytes, because a proof this rail cannot evaluate is one
        // the provider would refuse by name.
        let (proof, max_bytes) = match &declaration.footprint {
            StageBufferFootprint::Static { max_bytes } => (
                FootprintProof::Static {
                    max_bytes: *max_bytes,
                },
                *max_bytes,
            ),
            StageBufferFootprint::Affine { accesses } => {
                let Some(counts) = stage_buffer_affine_counts(req) else {
                    return Err(OutOfClass::owned(
                        "render_provider_out_of_class_stage_buffer_footprint",
                        format!(
                            "a draw whose {} stage declares a [[buffer({})]] argument with an \
                             affine footprint stays on the engine when the draw's own invocation \
                             counts are not readable from the bytes the trace carries: the \
                             canonical contract bounds the proof over the trace's own index bytes \
                             and refuses a count nothing states (`buffer_footprint_axis_invalid` \
                             / `StageBufferFootprintProofUnsupported`), so a lease-backed or \
                             gathered index view is a shape this rail cannot state",
                            stage.name(),
                            declaration.index,
                        ),
                    ));
                };
                let Some(required) = stage_buffer_affine_required_bytes(accesses, counts) else {
                    return Err(OutOfClass::owned(
                        "render_provider_out_of_class_stage_buffer_footprint",
                        format!(
                            "a draw whose {} stage declares a [[buffer({})]] argument with an \
                             affine footprint stays on the engine when the proof's bound cannot be \
                             evaluated over this draw: {} access(es) over the draw's own \
                             invocation counts either overflow the byte extent or name an axis a \
                             draw does not have, and the contract refuses such a proof by name",
                            stage.name(),
                            declaration.index,
                            accesses.len(),
                        ),
                    ));
                };
                (
                    FootprintProof::Affine {
                        accesses: accesses.clone(),
                    },
                    required,
                )
            }
            StageBufferFootprint::Unstated => {
                return Err(OutOfClass::owned(
                    "render_provider_out_of_class_stage_buffer_footprint",
                    format!(
                        "a draw whose {} stage declares a [[buffer({})]] argument stays on the \
                         engine when the translation's reach is unbounded or states no byte range \
                         at all: the canonical contract states stage buffer footprints as a \
                         static ceiling or a bounded affine proof, and both registration and \
                         admission refuse anything else by name rather than executing it against \
                         a bound nothing states",
                        stage.name(),
                        declaration.index,
                    ),
                ))
            }
        };
        let Some(bind) = inputs
            .stage_buffer_binds
            .iter()
            .find(|bind| bind.stage == stage && bind.index == declaration.index)
        else {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_stage_buffer_unbound",
                format!(
                    "a draw whose {} stage declares a [[buffer({})]] argument stays on the engine \
                     when the request binds no buffer at that index: the canonical pass fills one \
                     view per declaration, and a slot nothing filled would be read as undefined \
                     bytes. The request binds {binds} stage buffer(s)",
                    stage.name(),
                    declaration.index,
                ),
            ));
        };
        let bind_bytes = u64::try_from(bind.content.len()).unwrap_or(u64::MAX);
        if bind_bytes < max_bytes {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_stage_buffer_short",
                format!(
                    "a draw whose {} stage reaches {max_bytes} byte(s) of its [[buffer({})]] \
                     argument stays on the engine: the request binds {bind_bytes} byte(s) there, \
                     and the canonical contract proves the pass's view against the declared \
                     extent rather than the other way round",
                    stage.name(),
                    declaration.index,
                ),
            ));
        }
        // A writable declaration is a landing (R9f): the provider publishes one
        // writeback per writable view, and this rail has to be able to place it
        // where the guest reads. The bind's own guest address and the pages it
        // resolved to when it was staged are that place; a bind with neither —
        // the neutral bytes an index no stage reads is served, a bind whose
        // destination the seam never resolved — would have its writeback
        // dropped, so the draw keeps the engine by name.
        if declaration.access.is_writable() && bind.landing.is_none() {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_stage_buffer_write",
                format!(
                    "a draw whose {} stage declares a writable [[buffer({})]] argument stays on \
                     the engine when the request names no guest destination for it: the canonical \
                     rail publishes one writeback per writable stage buffer, and a bind this rail \
                     cannot land those bytes back into is a write the guest would never see. The \
                     request binds {binds} stage buffer(s)",
                    stage.name(),
                    declaration.index,
                ),
            ));
        }
        // The landing's bytes also ride as a *pool* view: the trace's pool is
        // the set of views a compute pass declares, and a writeback is keyed by
        // the view the pool holds, so a writable stage buffer gets one declare
        // pass of its own (`submit_narrow`). That pass binds the same bytes and
        // is proven against the declaring kernel's own reach, so a shorter bind
        // is a shape the provider always refuses — answered here rather than
        // declined.
        if declaration.access.is_writable() && bind_bytes < RENDER_DECLARE_REACH {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_stage_buffer_write",
                format!(
                    "a draw whose {} stage declares a writable [[buffer({})]] argument stays on \
                     the engine when the request binds {bind_bytes} byte(s) there: a landing has \
                     to name a view the trace's own pool declares, and the declaring kernel that \
                     holds those entries reads {RENDER_DECLARE_REACH} byte(s) of each, so a \
                     shorter bind is a view admission refuses by name. The request binds {binds} \
                     stage buffer(s)",
                    stage.name(),
                    declaration.index,
                ),
            ));
        }
        let bytes = match bind.content {
            BufferContent::Bytes(bytes) => Some(bytes.as_slice()),
            // A gather the GPU would perform from guest RAM is not staged
            // bytes: the two arms this rail mints are the owner's staged copy
            // and the owner's registered window, and R9e derives the second
            // from the gather's own page runs below.
            BufferContent::GuestRuns(_) => None,
        };
        let window = match bind.window {
            Some(window) => {
                // The window has to be the bind's own bytes: the view is what
                // the stage reads, so a window that starts elsewhere or stops
                // short would bind bytes the declaration never described.
                if window.bytes_len != bind_bytes
                    || window.head.checked_add(window.bytes_len) > Some(window.length)
                {
                    return Err(OutOfClass::owned(
                        "render_provider_out_of_class_stage_buffer_window",
                        format!(
                            "a draw whose {} stage declares a [[buffer({})]] argument covered by a \
                             guest RAM window stays on the engine when that window is not the \
                             bind's own bytes: window {} byte(s), head {}, covering {} byte(s) of \
                             a {bind_bytes} byte bind",
                            stage.name(),
                            declaration.index,
                            window.length,
                            window.head,
                            window.bytes_len,
                        ),
                    ));
                }
                Some(window)
            }
            // R9e: the seam states no window, but a bind whose bytes the GPU
            // would gather from one contiguous registered stretch *has* a
            // window, and the page runs the source was built from name it — so
            // this rail derives it here rather than asking the seam for a
            // second copy of the same coordinates. A gather that is scattered,
            // unregistered, or whose one stretch does not hold the bind's own
            // bytes derives nothing and keeps the engine by name below.
            None => match bind.content {
                BufferContent::GuestRuns(source) => gather_window(source),
                BufferContent::Bytes(_) => None,
            },
        };
        if bytes.is_none() && window.is_none() {
            return Err(OutOfClass::owned(
                "render_provider_out_of_class_stage_buffer_gather",
                format!(
                    "a draw whose {} stage reads a [[buffer({})]] argument the GPU gathers from \
                     guest RAM stays on the engine: this class mints a stage buffer's bytes \
                     through the owner rail — the staged copy it holds, or the registered window \
                     a zero-copy bind was cut from — and a gather that is neither has no source \
                     this rail can state",
                    stage.name(),
                    declaration.index,
                ),
            ));
        }
        out.push(NarrowStageBuffer {
            stage,
            index: declaration.index,
            access,
            proof,
            bytes,
            window,
        });
    }
    Ok(out)
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

/// The view-number namespace one recorded production's own views are minted
/// from (R22, `research/docs/26` §46).
///
/// Far above the consuming pass's own numbering (which starts at
/// [`FIRST_INPUT_VIEW`] and stays in the low thousands at worst), so a
/// production's streams, index and textures can never alias a view the
/// consuming pass states in the same trace. The value itself is private to
/// this rail: what leaves the rail is the `(allocation, view)` pair the trace
/// declares, and the two rails compare those, not this counter.
///
/// The contract's view-id namespace is one per trace — its conflict and
/// declaration walks are keyed by [`ViewId`] alone — so two productions of one
/// trace, and a production beside the consuming pass's own views, have to be
/// distinct ids and not merely distinct pairs. [`PRODUCTION_VIEW_STRIDE`] is
/// what keeps the productions apart.
const PRODUCTION_VIEW_BASE: u64 = 0x0000_0001_0000_0000;

/// The view-number block one production owns inside that namespace: the
/// production at position `i` mints `PRODUCTION_VIEW_BASE + i * STRIDE + k`,
/// and `k` is the recorded view's own number, which is a handful for every
/// shape this rail admits (`FIRST_INPUT_VIEW`'s streams, the index stream, the
/// stage buffers and the textures).
const PRODUCTION_VIEW_STRIDE: u64 = 0x0000_0000_0001_0000;

/// How many productions this rail keeps, one per guest target that a later
/// record may sample.
///
/// A bound rather than a promise: the map is keyed by [`TargetIdentity`], every
/// entry holds the pass that produced that target (its own stream bytes
/// included), and a guest that renders into thousands of distinct surfaces in
/// one session would otherwise keep a copy of every one of them alive for the
/// process lifetime. The oldest entry is evicted past this many — the shape a
/// re-production of the same identity takes is a replacement in place, so the
/// eviction only ever costs a consumer *its* production, and a consumer with no
/// production stays on the engine by name.
const PRODUCTION_LIMIT: usize = 64;

/// One pass of this rail a later record can sample (R22, E-TX3's
/// `TextureSource::TraceView`).
///
/// The census's head gate is `render_provider_out_of_class_texture_source`: a
/// draw whose sampled texture's texels come from the GPU rather than from a
/// copy the request carries. The canonical contract's answer is
/// `TextureSource::TraceView` — the sampled declaration names a view an
/// *earlier render pass of the same trace* stored — so the fork's answer is to
/// carry the pass that produced the target into the consuming record's own
/// trace, where the provider can produce and sample it without a CPU round
/// trip.
///
/// The fields are the producing pass's own facts, kept in the terms the trace
/// states them:
///
/// * `descriptor` is the pass descriptor the producing submission built,
///   already carrying the trace-owned copies of its streams and textures. The
///   consuming trace shifts its view numbers into the production namespace and
///   restates its attachment as the identity's own view with
///   [`StoreOp::Store`], which is what makes the frame land in the trace's
///   writeback channel;
/// * `identity` / `attachment` are the guest target and the provider
///   `(allocation, view)` pair both passes name — one identity, not a second
///   name a trace could get wrong;
/// * `format`, `width`, `height` and `extent` are the attachment's, which is
///   what the contract compares the consuming declaration against
///   (`RenderTextureSourceShapeMismatch`) and what this rail compares in its
///   own gate;
/// * `pipeline` is the registered render pipeline, which the provider already
///   holds: the consuming trace lists it beside the consumer's, so no
///   translation or registration is paid on the sampling side;
/// * `allocations` are the staged views' allocation records the consuming
///   trace has to declare, and `windows` the binds that are re-imported.
struct RecordedProduction {
    identity: TargetIdentity,
    /// The provider allocation the guest target's images live under, minted by
    /// [`resident_attachment`] and therefore stable for the process: the
    /// *view* is minted per trace ([`PRODUCTION_VIEW_BASE`]), because the
    /// contract's view-id namespace is one per trace.
    allocation: AllocationId,
    format: AttachmentFormat,
    width: u64,
    height: u64,
    extent: u64,
    /// The pass descriptor the producing submission built, with every bind
    /// restated as the trace's own bytes: see [`production_bytes`] for why the
    /// bytes are read at record time rather than re-imported at consume time.
    descriptor: RenderPassDescriptor,
    pipeline: CompiledComputePipeline,
    /// How many views this pass costs the trace's serial pool: its attachment,
    /// its streams, its index stream, its stage buffers and its textures — the
    /// same walk [`serial_resource_budget`] makes over a consuming pass, so the
    /// budget below is one sum of one count.
    views: usize,
    /// Monotone mint order, for the eviction above. One number per recording,
    /// so "the oldest" is a fact this rail wrote rather than a property of
    /// [`HashMap`]'s iteration order.
    sequence: u64,
}

/// The process-global registry of recorded productions, one per guest target.
struct ProductionRegistry {
    next: u64,
    by_identity: HashMap<TargetIdentity, Arc<RecordedProduction>>,
}

static PRODUCTIONS: OnceLock<Mutex<ProductionRegistry>> = OnceLock::new();

fn production_registry() -> &'static Mutex<ProductionRegistry> {
    PRODUCTIONS.get_or_init(|| {
        Mutex::new(ProductionRegistry {
            next: 0,
            by_identity: HashMap::new(),
        })
    })
}

/// The production a later record may sample, or `None` when no pass of this
/// rail ever produced that identity.
fn recorded_production(identity: &TargetIdentity) -> Option<Arc<RecordedProduction>> {
    let registry = production_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.by_identity.get(identity).cloned()
}

/// Restate one producing pass's binds as the trace's own bytes, in place.
///
/// The reason is the one thing a recorded pass cannot carry across
/// submissions: a *lease*. Every window-backed bind of the producing pass — a
/// vertex stream (R9q), the index stream (R11) or a stage buffer (R9e/R18) —
/// travels into its trace as a lease, and [`provider_owner::Plan::settle`]
/// retires that lease with the producing submission's completion. A stage
/// buffer is a lease in **both** of its arms, so the staged copy beside a
/// window is retired the same way.
///
/// The two alternatives were re-importing the window inside the consuming
/// submission, or restating the bytes. The bytes are restated because a
/// consuming trace that states no lease needs no owner plan at all: it stays
/// the in-process trace every sampled pass before R22 was. That matters because
/// the owner→provider wire's render contract still drops a pass's texture
/// declarations (`research/docs/23` §101.5) — a sampled pass that crosses the
/// wire is a typed decline today — so a production carrying a lease would push
/// every trace-produced consumer onto that decline.
///
/// The bytes are the ones the producing record *read*: a staged stream's own
/// copy, or [`provider_owner::window_bytes`]' range for a window, which is the
/// range the borrowed arm would have bound. `None` is a window this rail
/// cannot read — the same refusal the class answers for a live window bind, at
/// the one moment the production is recorded rather than the moment it is
/// consumed, so a production whose bytes this rail cannot state is simply not
/// recorded and its consumer keeps the engine by name.
fn production_bytes(pass: &NarrowPass<'_>, descriptor: &mut RenderPassDescriptor) -> Option<()> {
    fn own(view: &mut BufferView, bytes: Vec<u8>) {
        view.offset = 0;
        view.length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        view.source = BufferSource::OwnedBytes(bytes);
    }
    for (binding, stream) in pass.vertex_streams.iter().enumerate() {
        let view = &mut descriptor.vertex_buffers[binding];
        match &stream.source {
            // A staged stream already travels as the trace's own bytes.
            StreamSource::Staged(_) => {}
            StreamSource::Window(window) => {
                let bytes = provider_owner::window_bytes(owner_window(
                    vertex_stream_owner_binding(binding),
                    *window,
                ))
                .ok()?;
                own(view, bytes);
            }
        }
    }
    if let Some(indices) = descriptor.indices.as_mut() {
        if let StreamSource::Window(window) = &pass.index_stream.source {
            let bytes =
                provider_owner::window_bytes(owner_window(index_stream_owner_binding(), *window))
                    .ok()?;
            own(&mut indices.view, bytes);
        }
    }
    for (index, buffer) in pass.stage_buffers.iter().enumerate() {
        let view = &mut descriptor.stage_buffers[index].view;
        let label = stage_buffer_owner_binding(buffer.stage, buffer.index);
        let bytes = match (buffer.window, buffer.bytes) {
            (Some(window), _) => provider_owner::window_bytes(owner_window(label, window)).ok()?,
            (None, Some(bytes)) => bytes.to_vec(),
            // A bind with neither a window nor staged bytes is an arm the
            // class admits only under its own name; a pass that reached a
            // completion carries one of the two.
            (None, None) => continue,
        };
        own(view, bytes);
    }
    Some(())
}

/// Record one successful submission's pass as the production of a guest
/// target, or leave the registry untouched when the pass is not one a later
/// trace can restate.
///
/// `None` is deliberately not one condition: a pass that loads from the
/// provider's own image (no such image exists in the consuming trace's
/// registry), one that *writes* a stage buffer (its landing would be paid
/// twice), one that presents, and one that itself samples a trace-produced
/// texture are all shapes whose re-run would state a different trace than the
/// one that ran. Each keeps the consuming record on the engine by name —
/// `render_provider_out_of_class_texture_source_undeclared` — rather than
/// letting this rail invent the production's bytes.
fn production_recordable(req: &DrawRequest, pass: &NarrowPass<'_>) -> bool {
    if req.target_identity.is_none() {
        return false;
    }
    if !matches!(pass.load, NarrowLoad::Clear(_)) {
        return false;
    }
    if pass.present.is_some() {
        return false;
    }
    if pass
        .stage_buffers
        .iter()
        .any(|buffer| buffer.access.is_writable())
    {
        return false;
    }
    if pass
        .textures
        .iter()
        .any(|texture| !matches!(texture.source, NarrowTextureSource::Bytes(_)))
    {
        return false;
    }
    true
}

fn record_production(
    req: &DrawRequest,
    pass: &NarrowPass<'_>,
    pipeline: &CompiledComputePipeline,
    descriptor: &RenderPassDescriptor,
) -> Option<Arc<RecordedProduction>> {
    if !production_recordable(req, pass) {
        return None;
    }
    let identity = req.target_identity.clone()?;
    // One identity, minted the one way this rail mints resident pairs: the same
    // call the pass's own attachment went through, so the production and the
    // consumer's sampled view cannot name two images.
    let attachment = resident_attachment(&identity);
    match pass.store {
        // The frame went back to the caller and the caller's store route landed
        // it under this identity (the R20 published arm is the shape the
        // production seam states for a withheld readback).
        NarrowStore::Writeback => {}
        // The frame stayed in the provider's own image under the identity.
        NarrowStore::Resident(resident) if resident == attachment => {}
        NarrowStore::Resident(_) => return None,
    }
    // The descriptor is kept as the producing submission built it, with one
    // rewrite: every bind that traveled as a lease becomes the trace's own
    // bytes, because a lease is retired with the submission that imported it.
    // The views themselves are minted into the production's own namespace when
    // this record is *consumed*, because the contract's view-id namespace is
    // one per trace and the production's position in that trace is not known
    // here.
    let mut descriptor = descriptor.clone();
    production_bytes(pass, &mut descriptor)?;
    let registry = production_registry();
    let mut registry = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.next += 1;
    let production = Arc::new(RecordedProduction {
        identity: identity.clone(),
        allocation: attachment.allocation,
        format: pass.format,
        width: pass.width,
        height: pass.height,
        extent: pass.extent,
        descriptor,
        pipeline: pipeline.clone(),
        views: pass.views(),
        sequence: registry.next,
    });
    registry
        .by_identity
        .insert(identity, Arc::clone(&production));
    while registry.by_identity.len() > PRODUCTION_LIMIT {
        let Some(oldest) = registry
            .by_identity
            .iter()
            .min_by_key(|(_, production)| production.sequence)
            .map(|(identity, _)| identity.clone())
        else {
            break;
        };
        registry.by_identity.remove(&oldest);
    }
    Some(production)
}

/// Drop every recorded production, for the one event that retires the traces
/// they were recorded in: a rebuilt device.
fn clear_productions() {
    let registry = production_registry();
    let mut registry = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.by_identity.clear();
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
enum NarrowLoad<'a> {
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
    /// Begin from the caller's own bytes, declared for the attachment's view
    /// (`LoadOp::Load`, `research/docs/23` §3.3/§74, R5b).
    ///
    /// The contract's trace-owned load arm, and the only one that can state
    /// contents the provider never stored: the bytes travel with the trace
    /// exactly as a stream's do, and the rail uploads them into the pass's own
    /// image before it opens. The class elects it for two shapes, and both are
    /// one statement: the request names where its previous contents come from,
    /// and the caller owns the bytes that are those contents.
    ///
    /// * R23: a record whose previous contents are the live GPU image the
    ///   caller's chain names, on a caller that cannot hand this rail an image
    ///   under that identity
    ///   ([`RenderRailInputs::resident_frames_fetchable`]) but *can* hand over
    ///   the frame itself — the engine's own registry holds it, and the caller
    ///   is the one that reads it out
    ///   ([`RenderRailInputs::resident_source_bytes`]).
    /// * R25: the packet's middle record, whose previous contents are the frame
    ///   the record before it produced. That frame is the exec walk's own chain
    ///   value, and the walk — the caller — hands it over
    ///   ([`RenderRailInputs::chain_middle_source_bytes`]).
    ///
    /// Every other byte-bearing shape is still refused by name: a guest seed is
    /// a source the class cannot confirm is the live image, and an assertion
    /// this rail cannot check is not a load arm it may state.
    Bytes(&'a [u8]),
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
/// declares (`research/docs/23` §3.3, v83/v86 / `research/docs/26` §R9b, §R9j,
/// §R9m).
///
/// The three stated arms are the canonical contract's own access vocabulary
/// ([`metal_api_core::provider::BufferAccess`]), so a declaration states the
/// interface the provider pairs against its reflection rather than a class this
/// rail would have to translate; the two remaining arms are the render bind
/// census's (`runtime::bind_phase`'s `access_unused` / `access_undeclared` over
/// the engine's `ReflectedBufferAccess`), minus the one class that is no
/// declaration at all: a bind reflection does not mention is *absent*, and that
/// is exactly the population this class admits. Of those two, `Unused` is not
/// stated to the provider at all (R9m): the statement this rail builds carries
/// only the arguments the translated entries reach
/// ([`RenderRailInputs::stage_buffer_statement`]) and the class no longer
/// answers for the dropped slot beside it (R9n): it is counted and ignored,
/// and the draw proceeds with the slot neither declared nor bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageBufferAccess {
    /// Declared, and the specialized entry point never dereferences it.
    ///
    /// This rail states **no declaration** for such a slot (R9m): an argument
    /// no entry reaches is not interface a canonical pair — a declaration
    /// beside the view that fills it — could describe, so the statement drops
    /// it and the count of dropped declarations is a reading of its own. The
    /// drop does not keep the draw on the engine either (R9n): the canonical
    /// registration leaves a reflected `Unused` slot out of the pairing when
    /// the contract does not declare it (`research/docs/23` §95), so the
    /// request proceeds with that slot declared nowhere and bound nowhere.
    /// What is still refused by name is the opposite declaration, a contract
    /// that states the slot: `render_pipeline_contract_invalid` for one
    /// stating `unused`, `render_stage_reflection_mismatch` for one stating
    /// `read`. Measured in `provider_render_rail.rs`.
    Unused,
    /// Declared and read: `BufferAccess::Read`'s Metal spelling.
    Read,
    /// Declared and written: the `BufferAccess::Write` arm R9f landed, whose
    /// bytes leave through the completion's `BufferWriteback` channel.
    Write,
    /// Declared, read and written: `BufferAccess::ReadWrite`, stated the same
    /// way and landed the same way.
    ReadWrite,
    /// Declared without a usable access answer: fail closed, exactly as the
    /// engine's own bind path does.
    Unknown,
}

impl StageBufferAccess {
    /// The access the canonical contract states for this declaration, or
    /// `None` for the two arms the contract has no slot for: the arm this
    /// rail drops from its statement ([`Self::Unused`]) and the arm whose
    /// access the translation does not classify ([`Self::Unknown`]).
    pub fn contract_access(self) -> Option<metal_api_core::provider::BufferAccess> {
        use metal_api_core::provider::BufferAccess;
        match self {
            Self::Read => Some(BufferAccess::Read),
            Self::Write => Some(BufferAccess::Write),
            Self::ReadWrite => Some(BufferAccess::ReadWrite),
            Self::Unused | Self::Unknown => None,
        }
    }

    /// Whether this declaration is a landing (R9f): a writeback this rail has
    /// to see placed, not just bytes to read.
    pub fn is_writable(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }

    /// The spelling the class census and the refusal sentences use.
    pub fn name(self) -> &'static str {
        match self {
            Self::Unused => "unused",
            Self::Read => "read",
            Self::Write => "write",
            Self::ReadWrite => "read_write",
            Self::Unknown => "of an unclassified access",
        }
    }
}

/// The byte reach one `[[buffer(N)]]` declaration accounts for
/// (`research/docs/26` §R9d).
///
/// A declaration is stated to the canonical contract beside the extent the
/// stage reaches, because that extent is what the pass's own view is proven
/// against (`FootprintProof::Static { max_bytes }`) and what the translated
/// registration compares the module's reflection with. The three arms are the
/// registration's own split of the translator's footprint:
///
/// - a static extent: the largest exclusive byte offset any of the
///   declaration's static ranges names — the same number, computed the same
///   way, as the canonical rail's own `reflected_bytes`, from the same
///   translator pin over the same AIR;
/// - an affine reach (`FootprintProof::Affine`, R9f/v86): the reflected
///   `base + stride * index` accesses restated over the draw's own invocation
///   axes. The canonical registration pairs an affine declaration with the
///   reflection as *two measurements of one module* — the same access set,
///   order- and duplicate-insensitively — so these are the translator's own
///   numbers rather than a ceiling this rail chose;
/// - an unbounded reach, or one that states no range at all: a proof the
///   contract refuses by name, which keeps the draw on the engine under its own
///   bucket.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StageBufferFootprint {
    Static { max_bytes: u64 },
    Affine { accesses: Vec<AffineAccess> },
    Unstated,
}

/// One `[[buffer(N)]]` argument one stage's translation declares.
///
/// The index is the stage's *own* Metal buffer index space — the one
/// `setVertexBuffer(_:offset:index:)` and `setFragmentBuffer(_:offset:index:)`
/// name, and the one the canonical contract's `StageBufferBinding::index`
/// speaks — so a vertex declaration and a fragment declaration at one index are
/// two different arguments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageBufferDeclaration {
    pub index: u32,
    /// The access the contract states beside this slot, or the arm the
    /// contract has no slot for.
    pub access: StageBufferAccess,
    /// The byte extent the declaration's own reflection reaches, or the
    /// reason it cannot be stated as one.
    pub footprint: StageBufferFootprint,
}

/// Where a writable bind's bytes go back to the guest (`research/docs/26`
/// §R9j).
///
/// A writable stage buffer is a landing (R9f): the provider publishes one
/// `BufferWriteback` for it, and those bytes have to reach the address the
/// guest reads. This is the bind's own guest address — the one this rail read
/// the bytes from — beside the pages that address resolved to **when the bind
/// was staged**. The pages are the write's authority rather than a second walk
/// taken at completion time, for the reason
/// `runtime::gva_view::write_span_within` states: between the two walks the
/// guest can re-point the range, and the address a later walk answers is then
/// not the one these bytes belong to.
#[derive(Clone, Copy, Debug)]
pub struct StageBufferLanding<'a> {
    /// Guest address of the bind's first byte.
    pub gva: u64,
    /// Pages that address resolved to when the bind was staged.
    pub pages: crate::runtime::gva_view::WindowPages<'a>,
}

/// One writable stage buffer's bytes, as the provider's completion published
/// them (`BufferWriteback`), re-based onto the bind the pass stated.
///
/// `offset` is the writeback's first byte inside the bind's own bytes, so a
/// caller holding the bind's guest destination writes exactly the interval the
/// provider named — never "the whole buffer".
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageWriteback {
    pub stage: RenderPipelineStage,
    pub index: u32,
    pub offset: u64,
    pub bytes: Vec<u8>,
}

/// The registered guest RAM window that covers one stage buffer's bytes
/// (`research/docs/20` §3.2, `research/docs/26` §R9d).
///
/// This is the owner rail's *borrowed* arm as the runtime seam states it: the
/// page-aligned host range a bind's bytes live in, inside the registration
/// named by `import`. The rail fills the binding label itself — the two stages'
/// index spaces overlap, so the identity a window travels under has to come
/// from the pair the rail walks, not from the seam — which is why this type
/// carries the window's facts and not [`provider_owner::Window`] itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageBufferWindow {
    /// Registration the window was cut from ([`provider_owner::Region::import`]).
    pub import: u64,
    /// First host byte of the window.
    pub host_va: u64,
    /// Window length in bytes, a whole number of the registration's granules.
    pub length: u64,
    /// Bytes from the window's first byte to the bind's own first byte.
    pub head: u64,
    /// Bytes the bind covers from `head` onwards: the view's own length.
    pub bytes_len: u64,
}

/// One stage buffer the request's draw binds, as the runtime seam resolved it
/// (`research/docs/26` §R9d).
///
/// The declaration is a property of the *pipeline* (what the stage's own
/// translation names); this is the property of the *request* (which bytes the
/// draw put behind that name), and the canonical pair needs both. The two
/// owner arms the canonical rail resolves (`metal-api-vulkan`'s
/// `resolve_render_input`, R3c) are stated here: `content` is the owner's
/// staged copy of the bytes, and `window` is the registered guest RAM range
/// that covers them when the seam could cut one.
#[derive(Clone, Copy, Debug)]
pub struct StageBufferBind<'a> {
    /// Which stage's index space `index` is in.
    pub stage: RenderPipelineStage,
    /// The binding inside that stage's own Metal buffer index space.
    pub index: u32,
    /// The bytes the owner holds for this bind.
    pub content: &'a BufferContent,
    /// The registered window the bind's bytes came from, when the seam states
    /// one. `None` is the staged arm: the owner copies, and no registration is
    /// named.
    pub window: Option<StageBufferWindow>,
    /// Where this bind's bytes go back to, for the declarations whose access
    /// is writable (`research/docs/26` §R9j). `None` is a bind this rail has no
    /// guest destination for — a shape a writable declaration cannot land in,
    /// which the gate answers by name rather than dropping a writeback.
    pub landing: Option<StageBufferLanding<'a>>,
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
    /// Whether the caller can read a frame this rail keeps in its own image.
    ///
    /// The two resident arms (`LoadOp::Resident` / `StoreOp::Resident`, R7b)
    /// leave a frame somewhere only this rail can see it. That is an answer
    /// only for a caller that can fetch it back: every reader of a render
    /// target on the reims side — the deferred GVA debt, the mapper-ref-texture
    /// store, and the next record of a packet when it lands back on the engine
    /// — reads the *engine's* registry, and a frame kept here is a frame none
    /// of them can reach.
    ///
    /// Measured, not assumed. Census v15 (`evidence/gate3-census-v15-2026-09-18`,
    /// `REIMS_VGPU_DRAW_LOG=1`, driven macos-13): **11 892** `ok resident`
    /// answers, **6 824** `chain_resident_land_fail` (6 821
    /// `read_target_no_ready_content`, 3 `read_target_unknown_identity`) and
    /// 6 833 `load_target_content_not_ready` — a garbled guest desktop with the
    /// Dock unpainted. Every one of the 6 824 land failures has an `ok resident`
    /// answer on the same GVA earlier in the same packet, and the engine's
    /// refusal is not a race: at `t=45522` the provider answered `resident` for
    /// `gva=0x3cd2000` and at `t=45525` the next record of that packet asked
    /// the engine's LOAD gate for the same identity and was refused, because
    /// the engine had never held it.
    ///
    /// While this is `false` the class answers a record that withheld its
    /// readback by **publishing** its frame instead of keeping it (the route
    /// the pooled arm has always carried, and the one every caller-side reader
    /// already consumes), and a record whose previous contents are this rail's
    /// own image stays on the engine by name — so no answer this rail gives can
    /// leave a frame only the provider can see. `true` is a caller that can
    /// fetch a kept frame; R4b's byte channel is what makes the production seam
    /// state it, and until then the seam states `false`.
    pub resident_frames_fetchable: bool,
    /// The frame the record's own chain names, when the caller can read it out
    /// of the rail that holds it (R23).
    ///
    /// A record with `load_from_target` says its previous contents *are* the
    /// live GPU image. While [`Self::resident_frames_fetchable`] is `false`
    /// that image is not this rail's — it is the engine's registry resident,
    /// written by the record before this one — so `LoadOp::Resident` would name
    /// an image no provider pass ever stored. The caller that owns that
    /// registry can hand the frame over instead, in the attachment's own texel
    /// order and at the attachment's own tightly packed extent: the class then
    /// states the contract's trace-owned `Load` arm
    /// ([`NarrowLoad::Bytes`]), which is the one load the canonical attachment
    /// can carry without the provider having stored anything.
    ///
    /// `None` is every caller that cannot or does not read the frame back —
    /// and every record whose previous contents are not the live image — and
    /// those records keep the class's own refusal by name. Bytes that are not
    /// exactly the attachment's extent are a caller wiring bug and are refused
    /// under their own slug rather than being uploaded and rejected by the
    /// contract, because a decline is never a fallback.
    pub resident_source_bytes: Option<&'a [u8]>,
    /// The frame a middle record's previous contents are, when the caller
    /// hands it over (R25).
    ///
    /// A packet's records share one attachment, so every record after the first
    /// begins from the frame its predecessor produced — and the exec walk is
    /// what carries that frame between records: a record whose predecessor
    /// returned bytes arrives with them as the request's own `target_rgba8`,
    /// in `SeedOrder::Rgba8` (`encode_draw_chain` normalizes both rails' chain
    /// values to it before the walk takes them). Those bytes are the caller's,
    /// exactly as R23's are: the walk owns the frame it hands on. The class
    /// states them for the attachment's view through the contract's trace-owned
    /// load ([`NarrowLoad::Bytes`]) instead of leaving the middle on the engine
    /// — the position that both takes and hands on a frame, and the one census
    /// v18 named `chain_middle` at 657 records.
    ///
    /// The bytes follow the same two rules R23's do, and for the same reason —
    /// the canonical attachment uploads them verbatim into the pass's own
    /// image: they are the attachment's own texel at the attachment's own
    /// tightly packed extent, in the attachment's own order (`pass.format` is
    /// the view they are declared for, and the seam folds the walk's
    /// `SeedOrder::Rgba8` into it). A hand-over that is not exactly the
    /// attachment's extent is a caller wiring bug and is refused under
    /// `render_provider_out_of_class_chain_middle_shape` rather than being
    /// uploaded and rejected by the contract.
    ///
    /// `None` is every caller that does not hand the frame over — and every
    /// record that is not the packet's middle — and those records keep the
    /// class's refusal by name.
    pub chain_middle_source_bytes: Option<&'a [u8]>,
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
    /// canonical contract states a stage buffer as a *declaration beside the
    /// view that fills it*, so the access a translation reports is the half the
    /// registration pairs against its own reflection (R9d/R9j) — and the two
    /// arms the contract has no slot for are answered by the class rather than
    /// stated: a slot the entry never dereferences is not declared at all (R9m,
    /// [`Self::stage_buffer_statement`]) and an access the translation does not
    /// classify keeps the draw on the engine fail-closed. A request whose
    /// stages declare no `[[buffer(N)]]` argument at all leaves for the
    /// provider with its binds carried by nothing, which is the population this
    /// class has always admitted.
    pub vertex_stage_buffer_declarations: &'a [StageBufferDeclaration],
    pub fragment_stage_buffer_declarations: &'a [StageBufferDeclaration],
    /// The `[[texture(i)]]` arguments the fragment stage's own translation
    /// declares, with the module's AIR sampler state and the device bindings
    /// this draw's binds were resolved at (R10, `research/docs/23` §101).
    ///
    /// Carried for the reason the two stage-buffer lists are: the canonical
    /// render sampler's declaration has to repeat the state the module was
    /// lowered against, and the class gate has to answer the shapes the
    /// provider always refuses (`render_texture_sampler_unsupported`,
    /// `render_texture_shape_unsupported`, …) before the provider is asked.
    /// The runtime applies the fragment sampled-band relocation to both this
    /// list and the request's own binds, so entry `i` and the draw's bind at
    /// the same number are one binding on both sides of this seam.
    pub fragment_texture_declarations: &'a [RenderTextureDeclaration],
    /// The sampler family the fragment stage's own translation declares (R12,
    /// `research/docs/23` §102): every runtime `[[sampler(n)]]` argument the
    /// reflection binds beside the AIR static samplers it carries.
    ///
    /// Carried for the reason the declaration list above is: the canonical
    /// contract pairs each runtime argument with a texture declaration
    /// [`RenderTextureDeclaration::sampler`] names, and a stage whose two
    /// sampler forms or whose unpaired arguments the registration would refuse
    /// by name is a shape the class has to answer before the provider is asked
    /// ([`sampled_textures`]).
    pub sampler_family: &'a RenderSamplerFamily,
    /// The Metal arguments outside the family the canonical translated rail
    /// executes, across both stages (R10). A draw whose module declares one
    /// keeps the engine under the class's own name; empty is every shape whose
    /// interface the provider accounts for.
    pub texture_interface_refusals: &'a [RenderInterfaceRefusal],
    /// The stage's own `[[buffer(N)]]` binds this request's draw carries, at
    /// the Metal indices of the stage that named them (R9d).
    ///
    /// The declarations above say which slots the *pipeline* reads; this says
    /// which bytes the *draw* put behind them, which is the other half of the
    /// canonical pair. The seam builds it from the same two lists the engine's
    /// own bind rail reads (`vtx_storage` / `frag_storage`), so the Metal index
    /// a fragment declaration names is not lost to the engine's own
    /// single-namespace relocation, and each entry's bytes are the ones this
    /// draw already resolved.
    pub stage_buffer_binds: &'a [StageBufferBind<'a>],
}

/// The statement one request's two stages make, and the declarations it leaves
/// out (`research/docs/26` §R9m, §R9n).
///
/// Built by [`RenderRailInputs::stage_buffer_statement`], which is the one
/// place the two stages' reflection lists become a canonical statement: the
/// contract the pipeline is registered with, the pass's own view list and the
/// frame that crosses the owner→provider wire are all built from
/// [`Self::declared`], so "the wire carries the declaration" and "the pass
/// states one view per declaration" stay one statement instead of three
/// spellings of it.
#[derive(Debug)]
pub struct StageBufferStatement<'a> {
    /// Every declaration this class states, in the order the two stages state
    /// it (the vertex stage first, the reflection's own order inside a stage),
    /// each with the access the contract states it under.
    pub declared: Vec<(
        RenderPipelineStage,
        &'a StageBufferDeclaration,
        BufferAccess,
    )>,
    /// Every declaration this class does **not** state, in the same order.
    ///
    /// Two arms end up here, and R9n splits what happens to each: a slot the
    /// entry never dereferences (`Unused`) is *ignored* — dropped from the
    /// statement, counted ([`Self::dropped`]), and passed through to the
    /// canonical pairing, which leaves a reflected `Unused` slot out of it when
    /// the contract does not declare it (`research/docs/23` §95) — and an
    /// access the translation does not classify (`Unknown`) is fail-closed,
    /// exactly as it always was.
    pub unstated: Vec<(RenderPipelineStage, &'a StageBufferDeclaration)>,
}

impl StageBufferStatement<'_> {
    /// How many declarations were dropped because the slot's entry never
    /// dereferences it — the `Unused` arm of [`Self::unstated`], which is what
    /// the census band counts and what the class answered for before R9n
    /// stopped answering for it.
    pub fn dropped(&self) -> usize {
        self.unstated
            .iter()
            .filter(|(_, declaration)| declaration.access == StageBufferAccess::Unused)
            .count()
    }
}

impl<'a> RenderRailInputs<'a> {
    /// The one statement this request's two stages make (R9m).
    ///
    /// A `[[buffer(N)]]` argument the translated entry point never reaches is
    /// not part of it: there is no view the canonical pair could fill for a
    /// slot no use covers, so stating one would be a declaration this rail
    /// cannot bind. The slot is reported in
    /// [`StageBufferStatement::unstated`] instead, where the census counts it
    /// and the class ignores it (R9n): the canonical pairing admits exactly the
    /// shape the statement then carries (see [`StageBufferAccess::Unused`]).
    pub fn stage_buffer_statement(&self) -> StageBufferStatement<'a> {
        let mut statement = StageBufferStatement {
            declared: Vec::new(),
            unstated: Vec::new(),
        };
        let stages = [
            (
                RenderPipelineStage::Vertex,
                self.vertex_stage_buffer_declarations,
            ),
            (
                RenderPipelineStage::Fragment,
                self.fragment_stage_buffer_declarations,
            ),
        ];
        for (stage, declarations) in stages {
            for declaration in declarations {
                match declaration.access.contract_access() {
                    Some(access) => statement.declared.push((stage, declaration, access)),
                    None => statement.unstated.push((stage, declaration)),
                }
            }
        }
        statement
    }
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
    /// The bytes of every writable stage buffer the pass declared, in the
    /// contract's canonical order (`research/docs/26` §R9j).
    ///
    /// One entry per writable declaration, re-based onto the bind's own bytes
    /// and range-checked against the view the trace carried. Empty for every
    /// pass whose stage buffers are read-only — which is every shape R9d/R9e
    /// admitted — so a caller that lands these bytes has nothing to do for the
    /// population that came before this increment.
    pub stage_writebacks: Vec<StageWriteback>,
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
    /// The completion published no writeback for a writable stage buffer, so
    /// the pass ran without landing the bytes the guest reads (R9f/R9j).
    StageBufferWritebackMissing {
        stage: RenderPipelineStage,
        index: u32,
    },
    /// The completion's writeback for a writable stage buffer does not lie
    /// inside the view the trace carried, so its bytes are not that bind's.
    StageBufferWritebackShape {
        stage: RenderPipelineStage,
        index: u32,
        offset: u64,
        length: u64,
        view_offset: u64,
        view_length: u64,
    },
    /// The frame this rail states could not be produced or consumed by the
    /// canonical command channel, so the shape is one the wire does not carry
    /// (`research/docs/26` §R9j). `step` names the codec call that answered.
    StageBufferWire { step: &'static str, detail: String },
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
            Self::StageBufferWritebackMissing { .. } => "stage_buffer_writeback_missing",
            Self::StageBufferWritebackShape { .. } => "stage_buffer_writeback_shape",
            Self::StageBufferWire { .. } => "stage_buffer_wire",
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
            Self::StageBufferWritebackMissing { stage, index } => vec![
                ("stage", stage.name().to_string()),
                ("index", index.to_string()),
            ],
            Self::StageBufferWritebackShape {
                stage,
                index,
                offset,
                length,
                view_offset,
                view_length,
            } => vec![
                ("stage", stage.name().to_string()),
                ("index", index.to_string()),
                ("offset", offset.to_string()),
                ("length", length.to_string()),
                ("view_offset", view_offset.to_string()),
                ("view_length", view_length.to_string()),
            ],
            Self::StageBufferWire { step, detail } => {
                vec![("step", step.to_string()), ("detail", detail.clone())]
            }
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
    // The pipeline-level buffer face (R9d). It is part of the contract the
    // canonical rail registers, so it is part of what the cache keys on: two
    // requests whose stages declare different buffers — or the same buffer with
    // a different reach — are two registrations, and a fingerprint that omitted
    // them would hand one request the pipeline the other registered.
    for binding in &contract.stage_buffers {
        out.push_str(&format!("s{}:{}:", binding.stage.name(), binding.index,));
        match &binding.footprint {
            FootprintProof::Static { max_bytes } => out.push_str(&format!("{max_bytes:x};")),
            FootprintProof::Affine { .. } => out.push_str("affine;"),
            FootprintProof::Unbounded => out.push_str("unbounded;"),
        }
    }
    // The sampled-texture face (R10/R19), for the reason above one face over.
    // The declarations are part of the contract the canonical rail registers,
    // so they are part of what the cache keys on — and since R19 the *format*
    // is a fact two requests can differ in while agreeing on everything else:
    // the provider's window states two byte orders, and a fingerprint that
    // omitted the format would hand the second one the pipeline the first
    // registered. That collision is answered by name rather than with a wrong
    // frame (the trace's own view naming `Bgra8Unorm` beside a declaration that
    // says `Rgba8Unorm` is the provider's `trace_contract_invalid`), but it is
    // a decline of a shape this class admits, which is what makes it a defect
    // in the key rather than a boundary.
    for texture in &contract.textures {
        // The format travels as the contract's own name rather than as a wire
        // code: this key is an in-process string, and the name is the spelling
        // the two rails' own refusals use (`trace_contract_invalid` names it
        // too), so a reader of a cache miss sees the fact that moved.
        out.push_str(&format!(
            "t{}:{:?}:{:?}:{:?}:{:?}:",
            texture.metal_binding,
            texture.format,
            texture.access,
            texture.sampler,
            texture.runtime_sampler,
        ));
        out.push_str(match texture.footprint {
            TextureFootprintProof::WholeView => "whole;",
            TextureFootprintProof::Unbounded => "unbounded;",
        });
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
    // R-VF1: the same population, keyed by the *storage* each declared attribute
    // was declared in, because the census's `vertex_format` gate (`26233` of one
    // boot's `93259` seam rows, `evidence/gate3-census-v10-2026-09-17` §4) refuses
    // 28.1% of the class exits with one sentence that names no format at all —
    // the sentence lists the canonical set, not the declared value. One count per
    // attribute, named from the protocol's own `VertexFormat::name`, is the whole
    // reading: which storages the surviving draws declare, and in what mix, is
    // what a widening order has to be sized on. Charged at the same place and for
    // the same reason as the band above, so a request the gate refuses for some
    // *other* face is still in the denominator and the two readings line up.
    for attribute in &req.vertex_attributes {
        crate::runtime::drain::note_store_route(vertex_format_route(attribute.format));
    }
    // The bound stage-buffer axis (R9b), charged for the same reason and at the
    // same place: the door's four buckets answer *why* a draw stayed on the
    // engine, and this answers *how many* buffers were behind it — the split
    // the v2 census could not make (its §7.3).
    crate::runtime::drain::note_store_route(stage_buffer_count_route(req.storage_buffers.len()));
    // R9m/R9n: the declarations the same request's translations state and its
    // statement does not carry, charged for the same population and for the
    // same reason — this says how many declarations a request left behind on
    // the way in. R9n is the increment the band now sizes: the drop no longer
    // keeps the draw on the engine (the door has no bucket for it any more), so
    // the band reads the population the class hands to the canonical pairing
    // with a slot it never states. The statement is the seam's one spelling of
    // the partition ([`RenderRailInputs::stage_buffer_statement`]), so the band
    // and the gate cannot disagree about what was dropped.
    crate::runtime::drain::note_store_route(stage_buffer_unused_skip_route(
        inputs.stage_buffer_statement().dropped(),
    ));
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
    // The second device answer the class needs (R9d): a stage buffer the seam
    // cut out of a registered guest RAM window leaves through the owner rail's
    // no-copy arm, which a device without host-pointer import refuses by name.
    // Read here rather than inside the pure gate, exactly as the window above,
    // so a shape the request alone already answers never reaches the provider.
    //
    // The capability answer comes first (R9j), and it is the *wire's* reading
    // of it: a provider that does not declare the stage-buffer shape executes
    // none of these declarations, so the draw keeps the engine under the
    // capability's own bucket rather than being met by the provider's
    // `render_stage_buffer_unsupported` refusal.
    if !pass.stage_buffers.is_empty() {
        let support = match declared_stage_buffer_support() {
            Ok(support) => support,
            Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
        };
        let declared = pass.stage_buffers.len();
        if !support.supported || declared > support.maximum as usize {
            let reason = OutOfClass::owned(
                "render_provider_out_of_class_stage_buffer_capability",
                format!(
                    "a draw whose stages declare {declared} stage buffer(s) stays on the engine \
                     when the provider's own capability answer does not carry the shape: the \
                     frame declares supports_render_stage_buffers={} and \
                     max_render_stage_buffers={}, and a pass above either bound is one admission \
                     refuses by name (`render_stage_buffer_unsupported` / \
                     `render_stage_buffer_limit`) rather than a slot this rail could bind",
                    support.supported, support.maximum,
                ),
            );
            reason.note();
            return RenderRailOutcome::NotInNarrowClass(reason);
        }
    }
    // The third device answer (R9e/R9q/R11/R18): every window-backed binding
    // this pass states — a stage buffer's window, a vertex stream's since R9q,
    // and the index stream's since R11 — is imported by the owner rail unless
    // this device cannot take the view's own pointer, in which case the gate
    // copies the window's bytes and the plan states the staged arm instead
    // (`window_arm`); a device that cannot import host pointers at all refuses
    // that arm by name rather than turning it into a copy. The answer is asked
    // once for the pass, in the order the binds are stated (the stage buffers,
    // then the vertex streams, then the index stream), so the bucket a refusal
    // lands in is the one its own shape owns — and so the copies the plan reads
    // back are keyed the same way the plan states them.
    let window_backed = || {
        pass.stage_buffers
            .iter()
            .filter_map(|buffer| {
                buffer.window.map(|window| {
                    (
                        STAGE_BUFFER_WINDOW,
                        stage_buffer_owner_binding(buffer.stage, buffer.index),
                        window,
                    )
                })
            })
            .chain(pass.vertex_windows().map(|(binding, window)| {
                (
                    VERTEX_STREAM_WINDOW,
                    vertex_stream_owner_binding(binding),
                    window,
                )
            }))
            .chain(
                pass.index_window()
                    .map(|window| (INDEX_STREAM_WINDOW, index_stream_owner_binding(), window)),
            )
    };
    // R18: the binds the device's granules turn away from the borrowed arm. The
    // copy is made here rather than inside the plan because this is where the
    // device answer lives, and because a window this rail cannot read has to
    // keep the draw on the engine *by name* — an answer only the class gate can
    // give (`plan`'s refusals are declines, and a decline is not a fallback).
    let mut copies = WindowCopies::default();
    if window_backed().next().is_some() {
        let alignment = match declared_host_import() {
            Ok(alignment) => alignment,
            Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
        };
        for (shape, binding, window) in window_backed() {
            match window_arm(shape, binding, window, alignment) {
                Ok(Some(bytes)) => {
                    // The extra copy this class pays, counted where it happens:
                    // one route per staged window and one byte total beside it,
                    // so a boot can say how much of the class's traffic the
                    // device's granules moved onto the staging arm.
                    crate::runtime::drain::note_store_route(
                        "render_provider_unaligned_window_staged",
                    );
                    crate::runtime::drain::note_store_route_n(
                        "render_provider_unaligned_window_bytes",
                        u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                    );
                    copies.insert(binding, bytes);
                }
                Ok(None) => {}
                Err(reason) => {
                    reason.note();
                    return RenderRailOutcome::NotInNarrowClass(reason);
                }
            }
        }
    }
    match submit_narrow(inputs, req, &pass, &copies) {
        Ok(RenderCompletion::Writeback(output)) => RenderRailOutcome::ProviderCompleted(output),
        Ok(RenderCompletion::Resident(frame)) => {
            RenderRailOutcome::ProviderCompletedResident(frame)
        }
        Err(decline) => RenderRailOutcome::ProviderDeclined(decline),
    }
}

/// The census band of one request's declared vertex *attributes* — the engine's
/// own list, one entry per location.
///
/// The band is charged from `req.vertex_attributes.len()` and keeps the
/// contract's own `MAX_VERTEX_BUFFERS` breakpoints, because a build where the
/// contract moved to five would have to rename the `_gt4` arm rather than
/// silently count five as `2_4` (the assertion in this module's tests pins the
/// four names). The class's ceiling is on the canonical *streams*, which are the
/// request's fetch tables and can be fewer than its attributes
/// (`research/docs/26` §31), so an admitted draw may be banded here past the
/// number of streams the layout it registered with states.
pub fn attribute_count_route(declared: usize) -> &'static str {
    match declared {
        0 => "draw_vertex_attrs_0",
        1 => "draw_vertex_attrs_1",
        2..=4 => "draw_vertex_attrs_2_4",
        _ => "draw_vertex_attrs_gt4",
    }
}

/// The census route of one *declared* vertex format, named after the Metal
/// storage it was declared in (R-VF1).
///
/// Why a route per format rather than a latched line. The gate's own sentence
/// (`render_provider_out_of_class_vertex_format`) states the canonical set and
/// not the value that met it, and the failure log has no other field that
/// carries one: a draw refused at that gate is visible as a count and as a
/// `(slug, shape)` row, and the question a widening order has to answer — *which
/// storage do these draws declare, and in what mix* — is exactly the field the
/// log loses. One counter per attribute, keyed by the protocol's own
/// [`VertexAttributeFormat::name`], makes the reading a subtraction over a
/// window's `store_routes` line.
///
/// The spelling is `vertex_format_<name>` rather than the bare name so the keys
/// cannot be confused with another route family that happens to use the same
/// word (`float`, `half`, … are ordinary English). The match is exhaustive over
/// the protocol's closed list on purpose: a format added there has to name its
/// route here, and the compiler is what says so.
///
/// Charged for every attribute of every request the class gate is handed, at the
/// same place [`attribute_count_route`] is charged: the counters are the
/// *declaration* population, so the formats of draws that stayed on the engine
/// for some other face are in the reading too, and a widening order reads the
/// admitted and refused halves as one distribution rather than as a remainder.
pub fn vertex_format_route(
    format: crate::backend::vulkan::engine::VertexAttributeFormat,
) -> &'static str {
    use crate::backend::vulkan::engine::VertexAttributeFormat as Format;
    match format {
        Format::UChar2 => "vertex_format_uchar2",
        Format::UChar3 => "vertex_format_uchar3",
        Format::UChar4 => "vertex_format_uchar4",
        Format::Char2 => "vertex_format_char2",
        Format::Char3 => "vertex_format_char3",
        Format::Char4 => "vertex_format_char4",
        Format::UChar2Normalized => "vertex_format_uchar2_normalized",
        Format::UChar3Normalized => "vertex_format_uchar3_normalized",
        Format::UChar4Normalized => "vertex_format_uchar4_normalized",
        Format::Char2Normalized => "vertex_format_char2_normalized",
        Format::Char3Normalized => "vertex_format_char3_normalized",
        Format::Char4Normalized => "vertex_format_char4_normalized",
        Format::UShort2 => "vertex_format_ushort2",
        Format::UShort3 => "vertex_format_ushort3",
        Format::UShort4 => "vertex_format_ushort4",
        Format::Short2 => "vertex_format_short2",
        Format::Short3 => "vertex_format_short3",
        Format::Short4 => "vertex_format_short4",
        Format::UShort2Normalized => "vertex_format_ushort2_normalized",
        Format::UShort3Normalized => "vertex_format_ushort3_normalized",
        Format::UShort4Normalized => "vertex_format_ushort4_normalized",
        Format::Short2Normalized => "vertex_format_short2_normalized",
        Format::Short3Normalized => "vertex_format_short3_normalized",
        Format::Short4Normalized => "vertex_format_short4_normalized",
        Format::Half2 => "vertex_format_half2",
        Format::Half3 => "vertex_format_half3",
        Format::Half4 => "vertex_format_half4",
        Format::Float => "vertex_format_float",
        Format::Float2 => "vertex_format_float2",
        Format::Float3 => "vertex_format_float3",
        Format::Float4 => "vertex_format_float4",
        Format::Int => "vertex_format_int",
        Format::Int2 => "vertex_format_int2",
        Format::Int3 => "vertex_format_int3",
        Format::Int4 => "vertex_format_int4",
        Format::UInt => "vertex_format_uint",
        Format::UInt2 => "vertex_format_uint2",
        Format::UInt3 => "vertex_format_uint3",
        Format::UInt4 => "vertex_format_uint4",
        Format::Int1010102Normalized => "vertex_format_int1010102_normalized",
        Format::UInt1010102Normalized => "vertex_format_uint1010102_normalized",
        Format::UChar4NormalizedBgra => "vertex_format_uchar4_normalized_bgra",
        Format::UChar => "vertex_format_uchar",
        Format::Char => "vertex_format_char",
        Format::UCharNormalized => "vertex_format_uchar_normalized",
        Format::CharNormalized => "vertex_format_char_normalized",
        Format::UShort => "vertex_format_ushort",
        Format::Short => "vertex_format_short",
        Format::UShortNormalized => "vertex_format_ushort_normalized",
        Format::ShortNormalized => "vertex_format_short_normalized",
        Format::Half => "vertex_format_half",
        Format::FloatRg11B10 => "vertex_format_float_rg11b10",
        Format::FloatRgb9E5 => "vertex_format_float_rgb9e5",
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

/// The census band of the stage-buffer declarations a request's translations
/// state and the statement does not carry (`research/docs/26` §R9m, §R9n).
///
/// The reflection reports a `[[buffer(N)]]` argument whose specialized entry
/// point never dereferences it as `ResourceAccess::Unused`, and this rail
/// states no declaration for such a slot: there is no view the canonical pair
/// could fill, and since R9n the canonical pairing admits exactly that shape —
/// a reflected `Unused` slot the contract does not declare takes no part in the
/// pairing (`research/docs/23` §95, [`StageBufferAccess::Unused`]) — so the
/// draw proceeds with the slot neither declared nor bound. The band sizes that
/// population. It is charged for every request the gate is handed, beside
/// [`stage_buffer_count_route`] and with the same breakpoints, so one census
/// can read both how many binds sat behind a draw and how many declarations
/// were dropped — and so the arms sum to the same denominator the bound band's
/// arms share.
pub fn stage_buffer_unused_skip_route(skipped: usize) -> &'static str {
    match skipped {
        0 => "stage_buffer_skipped_unused_0",
        1 => "stage_buffer_skipped_unused_1",
        2..=4 => "stage_buffer_skipped_unused_2_4",
        _ => "stage_buffer_skipped_unused_gt4",
    }
}

/// The census route of one `stage_buffer_shape` refusal (R9o,
/// `research/docs/26` §30).
///
/// The refusal slug is one name three rules answer under — more declarations
/// than the canonical contract states ([`MAX_RENDER_STAGE_BUFFERS`]), one
/// `(stage, index)` declared twice, and a vertex declaration inside the
/// canonical layout's own `0..vertex_streams` bindings — and until this
/// increment every one of them was counted as the bare slug: the
/// 2026-09-17 census v5 read 88026 / 89277 first failures under
/// `render_provider_out_of_class_stage_buffer_shape` and could not say which
/// rule answered them (`research/docs/26` §0.4). Each arm now charges its own
/// route beside the refusal, `note_store_route(name)` does the counting, and
/// the sentence every arm answers with is unchanged.
pub fn stage_buffer_shape_route(stage_buffer_shape: StageBufferShapeRoute) -> &'static str {
    match stage_buffer_shape {
        StageBufferShapeRoute::TooMany => "stage_buffer_shape_gt4",
        StageBufferShapeRoute::Duplicate => "stage_buffer_shape_duplicate",
        StageBufferShapeRoute::VertexLayout => "stage_buffer_shape_vertex_layout",
    }
}

/// Which of [`stage_buffer_shape_route`]'s three rules a draw answered.
///
/// An enum rather than a `&'static str` at the call sites for the reason the
/// routes themselves exist: three arms that answer with different census names
/// are three facts, and a typo in a bare string would file two of them under
/// one name with nothing failing. The gate is the only constructor, one arm
/// per `return`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageBufferShapeRoute {
    /// The statement carries more declarations than
    /// [`MAX_RENDER_STAGE_BUFFERS`] states.
    TooMany,
    /// One `(stage, index)` is declared twice.
    Duplicate,
    /// A vertex-stage declaration occupies an index the pipeline's own vertex
    /// layout already describes (`0..vertex_streams`, where the count is the
    /// request's own fetch tables — `research/docs/26` §31).
    VertexLayout,
}

/// Charge one `stage_buffer_shape` arm's route beside the refusal it answers.
#[inline]
fn note_stage_buffer_shape(stage_buffer_shape: StageBufferShapeRoute) {
    crate::runtime::drain::note_store_route(stage_buffer_shape_route(stage_buffer_shape));
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
    // R22: a production names a registered pipeline and the passes of one
    // device epoch; a rebuilt device holds neither, so the recorded
    // productions go with them rather than being re-run against a provider
    // that never registered their pipelines.
    clear_productions();
}

/// One admitted vertex stream: one fetch table, and the attributes read out of
/// it at their own offsets.
///
/// One entry per *table* rather than one per attribute, because that is the
/// shape the canonical contract states: a [`VertexBufferLayout`] carries one
/// stride and a list of attributes, which is Metal's own
/// `MTLVertexBufferLayoutDescriptor`, and the entry's position in
/// [`VertexLayout::Buffers`] is the binding both rails resolve it under. The
/// request the seam hands this rail holds one entry per *attribute* (the engine
/// numbers one Vulkan binding per attribute *location*), so the attributes that
/// read one table — the runtime shares one staged allocation per guest vertex
/// buffer, [`BufferContent::Bytes`]'s own contract — collapse into one entry
/// here (`research/docs/26` §31).
struct NarrowVertexStream<'a> {
    /// Bytes between consecutive vertices of this table.
    stride: u64,
    /// The request attribute whose bytes this table holds: the group's identity,
    /// which the next attribute of the table is matched against
    /// ([`one_vertex_stream`]) — and the source below is that head's own, which
    /// is why a table several attributes read still travels once.
    head: usize,
    /// Where this table's bytes come from: the owner's staged copy, or the
    /// registered window a zero-copy bind was cut from (R9q).
    source: StreamSource<'a>,
    /// Every attribute this table carries, in the request's own order.
    attributes: Vec<NarrowVertexAttribute>,
}

/// One attribute read out of one canonical vertex stream.
///
/// The location is the guest's and is carried unchanged, because that is what
/// the pipeline's vertex input state and the translated shader's own inputs are
/// keyed on; the offset is inside the stream's records.
struct NarrowVertexAttribute {
    location: u32,
    offset: u64,
    format: VertexFormat,
}

/// The admitted index stream.
///
/// The same two-armed source a vertex stream has (`R11`): the draw's own index
/// buffer is resolved by the same zero-copy rail, so its bytes may be the
/// request's staged copy or the registered window its bind was cut from.
struct NarrowIndexStream<'a> {
    format: IndexFormat,
    source: StreamSource<'a>,
}

/// The sampling half of one admitted pass (R10/R12): the textures it binds,
/// beside the runtime sampler states the request states for them.
struct NarrowSampling<'a> {
    textures: Vec<NarrowTexture<'a>>,
    /// One entry per runtime `[[sampler(n)]]` argument an admitted texture reads
    /// through, ascending by Metal index.
    runtime_samplers: Vec<NarrowRuntimeSampler>,
    /// The productions the pass's own trace has to carry ahead of it (R22):
    /// one entry per distinct trace-produced identity the textures bind.
    productions: Vec<Arc<RecordedProduction>>,
}

/// Where one admitted texture's texels come from (R10/R22).
#[derive(Clone)]
enum NarrowTextureSource<'a> {
    /// The request's own tightly packed copy, the arm every earlier increment
    /// carried.
    Bytes(&'a [u8]),
    /// The trace's own production (R22, E-TX3): the view an earlier pass of
    /// this record's own trace stores, named by the guest target the bind's
    /// `SampledSource::Target` resolved to. The bytes do not exist before the
    /// trace runs, so the view travels as `TextureSource::TraceView` and the
    /// production rides the same trace ahead of this pass.
    Produced { production: Arc<RecordedProduction> },
}

/// One admitted texture: the canonical binding (the Metal index it states, in
/// the contract's ascending order), the bytes the fragment stage reads, and the
/// sampler form its declaration states (R10/R12/R15/R16).
struct NarrowTexture<'a> {
    /// The Metal `[[texture(n)]]` index the entry states — the fact the
    /// contract pairs the pass's view and the pipeline's declaration by
    /// (E-RS3), rather than the entry's position.
    index: u32,
    width: u64,
    height: u64,
    /// The texel the bind's own view names, resolved to the contract's format
    /// (E-TX1, `research/docs/23` §107): one of the two four-byte 8-bit UNORM
    /// byte orders, which is the provider's whole `RENDER_SAMPLED` window. The
    /// declaration, the view and the byte count all read it, so the bytes and
    /// the name they are uploaded under cannot drift apart.
    format: TextureFormat,
    /// The sampler form the declaration states, or the sampler-free fetched
    /// arm (R15).
    sampler: NarrowSampler,
    /// Where the texels come from: the request's own tightly packed copy —
    /// four bytes per texel, in the byte order [`Self::format`] names — or the
    /// trace's own production of the identity the bind resolved to (R22).
    source: NarrowTextureSource<'a>,
}

/// Which sampler form one admitted texture's declaration states (R10/R12/R15).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NarrowSampler {
    /// The AIR static state the module carries: the declaration states it, the
    /// canonical rail creates its `VkSampler` from it, and the draw's bind has
    /// to repeat it (R10).
    Static(SamplerPolicy),
    /// The runtime `[[sampler(n)]]` argument the texture reads through: the
    /// declaration states the pair, and the pass states the state, which is the
    /// request's own fact (R12).
    Runtime { index: u32 },
    /// No sampler at all (R15): the module's own sites texel-fetch the image,
    /// so the canonical declaration binds the `SAMPLED_IMAGE` descriptor alone
    /// and the pass states no sampler state for it.
    Fetched,
}

/// One runtime sampler state the request states for an admitted pass (R12):
/// the Metal `[[sampler(n)]]` index the declaration pairs with a texture, and
/// the policy the draw's own bind resolves to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NarrowRuntimeSampler {
    index: u32,
    policy: SamplerPolicy,
}

/// One admitted stage buffer: a `[[buffer(N)]]` argument a stage's own
/// translation declares, with the bytes the canonical pass binds into it
/// (`research/docs/26` §R9d, §R9j).
struct NarrowStageBuffer<'a> {
    stage: RenderPipelineStage,
    /// The Metal index inside `stage`'s own buffer namespace.
    index: u32,
    /// The access the contract declares for this slot — the arm the stage's own
    /// reflection reported, so the pass's view carries the same one
    /// (`StageBufferAccessMismatch` is the contract's own pairing of the two).
    access: BufferAccess,
    /// The proof the contract declares: the static ceiling, or the affine
    /// access set the reflection reported (R9f/v86). The bind's bytes are
    /// proven to cover it before the view exists.
    proof: FootprintProof,
    /// The owner's staged bytes, when the bind's content is a copy the owner
    /// holds. `None` is a bind whose bytes exist only behind its window.
    bytes: Option<&'a [u8]>,
    /// The registered window the bind's bytes came from, when the seam stated
    /// one: the borrowed no-copy arm.
    window: Option<StageBufferWindow>,
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
    load: NarrowLoad<'a>,
    /// Where this pass's frame goes ([`NarrowStore`]).
    store: NarrowStore,
    /// Whether the caller withheld its readback and this pass answered by
    /// publishing the frame anyway, because the caller cannot read a frame this
    /// rail keeps ([`RenderRailInputs::resident_frames_fetchable`]). Counted as
    /// `render_provider_publish_held_resident`, so the held population is a
    /// number rather than the silence a route counter that only fires on the
    /// elected arm would leave.
    published_held_resident: bool,
    /// Whether this pass's previous contents are the packet's own chain value,
    /// handed over by the caller (R25). Counted as
    /// `render_provider_chain_middle_source_bytes` beside R23's
    /// `render_provider_resident_source_bytes`, so the two byte arms are two
    /// numbers rather than one: R23 carries the frame the *engine's* registry
    /// holds, R25 the frame the *walk* holds.
    carried_chain_middle: bool,
    bgra: bool,
    width: u64,
    height: u64,
    extent: u64,
    index_count: u32,
    /// The admitted streams, in the request's own attribute order: entry `i`
    /// becomes canonical binding `i`. One entry per fetch table, and one
    /// attribute per *location* inside it: the engine's request numbers one
    /// Vulkan binding per attribute location, while a guest's interleaved
    /// stream is several attributes off one table — which is the table entry `i`
    /// states here, with each attribute at its own offset.
    vertex_streams: Vec<NarrowVertexStream<'a>>,
    index_stream: NarrowIndexStream<'a>,
    /// The read-only stage buffers the request binds, in the contract's own
    /// canonical order (vertex bindings first by index, then fragment), empty
    /// for every request whose stages declare no `[[buffer(N)]]` argument.
    stage_buffers: Vec<NarrowStageBuffer<'a>>,
    /// The sampled textures the pass binds, in the contract's own canonical
    /// order (R10/R16): ascending by the Metal index each view states, which is
    /// the index the same entry's declaration states — the two lists pair by
    /// that index rather than by position (E-RS3, `research/docs/23` §104), so
    /// entry `i` may be any `[[texture(n)]]` argument and a list may skip an
    /// index. Empty for every request whose fragment stage declares no sampled
    /// texture.
    textures: Vec<NarrowTexture<'a>>,
    /// The runtime sampler states the pass states (R12): one entry per
    /// `[[sampler(n)]]` argument an admitted texture reads through, ascending
    /// by Metal index — the canonical contract's own order. Empty for every
    /// stage that samples through its own AIR static state.
    runtime_samplers: Vec<NarrowRuntimeSampler>,
    /// The passes this record's own trace has to carry ahead of it so the
    /// textures above can sample them (R22): one entry per distinct
    /// trace-produced identity the pass binds, in the order the textures state
    /// them — which is what makes the trace's pass order the production's own
    /// order.
    productions: Vec<Arc<RecordedProduction>>,
    /// The scissor rectangle the pass states, or `None` for the whole
    /// attachment — the canonical pass's own default
    /// ([`RenderPassDescriptor::scissor`]).
    scissor: Option<[u32; 4]>,
    /// The viewport rect the pass states (`research/docs/23` §100), or `None`
    /// for the attachment-covering default the contract published before v100
    /// — the same default a request that binds no viewport keeps.
    viewport: Option<[u32; 4]>,
    /// The attachment's own blend state the pass states (`research/docs/23`
    /// §100), or `None` for the no-blend, all-writes entry the contract
    /// published before v40 — the same shape a request with
    /// `MTLRenderPipelineColorAttachmentDescriptor.blendingEnabled` clear
    /// keeps.
    blend: Option<RenderPassBlend>,
    /// The present target this pass hands on, or `None` for the pooled/resident
    /// arms (R4b). Present whenever the caller stated a present tail; the class
    /// conditions above are what make that the *only* shape a presenting record
    /// can reach this point with.
    present: Option<PresentAttachment>,
}

impl NarrowPass<'_> {
    /// The views this pass spends the trace's serial pool on: its attachment,
    /// its vertex streams, its index stream, its stage buffers and its sampled
    /// textures. The trace's own budget is
    /// [`MAX_SERIAL_RESOURCES`](metal_api_core::provider::MAX_SERIAL_RESOURCES)
    /// views, and a record that carries productions spends theirs beside this
    /// pass's — [`serial_views_admit`] is the one place that sum is compared.
    fn views(&self) -> usize {
        1 + self.vertex_streams.len() + 1 + self.stage_buffers.len() + self.textures.len()
    }

    /// Every vertex stream this pass states as a registered guest RAM window
    /// (`R9q`): the stream's canonical binding index beside the window its bind
    /// was cut from. The label is the *stream's* own numbering — one entry per
    /// fetch table, `research/docs/26` §31 — and never the guest's binding
    /// number, which is the same rule the layout's entry positions follow.
    fn vertex_windows(&self) -> impl Iterator<Item = (usize, StageBufferWindow)> + '_ {
        self.vertex_streams
            .iter()
            .enumerate()
            .filter_map(|(binding, stream)| match &stream.source {
                StreamSource::Window(window) => Some((binding, *window)),
                StreamSource::Staged(_) => None,
            })
    }

    /// The window the draw's own index bind was cut from, when the index
    /// stream travels as the owner's mapping (`R11`).
    ///
    /// One index stream per draw, so there is one window and no numbering: the
    /// canonical binding is the contract's own zero
    /// ([`IndexBufferBinding`]'s view is the pass's one index view), and the
    /// owner label is the namespace below.
    fn index_window(&self) -> Option<StageBufferWindow> {
        match &self.index_stream.source {
            StreamSource::Window(window) => Some(*window),
            StreamSource::Staged(_) => None,
        }
    }

    /// The contract's vertex layout for the admitted streams.
    ///
    /// One entry per admitted fetch table, with the attributes that read it at
    /// their own offsets — the shape a `MTLVertexBufferLayoutDescriptor` has,
    /// and the one the canonical pipeline's vertex input state is built from.
    /// The locations are the guest's, unchanged: they are what the shader reads
    /// and what the vertex input state is keyed on. The entry's *position* is
    /// the binding both rails resolve the stream under
    /// (`provider.rs::validate_vertex_buffer_binding`), which is this class's
    /// own numbering and not the guest's binding number
    /// (`research/docs/26` §31).
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
                    attributes: stream
                        .attributes
                        .iter()
                        .map(|attribute| VertexAttribute {
                            location: attribute.location,
                            offset: attribute.offset,
                            format: attribute.format,
                        })
                        .collect(),
                })
                .collect(),
        )
    }

    /// The render texture declarations this pass's pipeline is registered with
    /// (R10/R16, `research/docs/23` §101, §104).
    ///
    /// One entry per admitted texture, in the same canonical order the pass
    /// states its views — ascending by the Metal index each entry states, which
    /// is the index the contract pairs the two lists by (E-RS3). A
    /// static-sampled entry carries the module's own AIR state — the state the
    /// canonical rail creates its `VkSampler` from and the state the translated
    /// arm compares against its own translation of the same module — while a
    /// runtime-sampled one names the `[[sampler(n)]]` argument it reads through
    /// and states no state, which the pass states instead
    /// (`NarrowPass::runtime_samplers`, R12).
    fn texture_declarations(&self) -> Vec<TextureBindingContract> {
        self.textures
            .iter()
            .map(|texture| match texture.sampler {
                NarrowSampler::Static(policy) => {
                    TextureBindingContract::sampled(texture.index, texture.format, policy)
                }
                NarrowSampler::Runtime { index } => {
                    TextureBindingContract::sampled_runtime(texture.index, texture.format, index)
                }
                // The sampler-free arm (R15): the declaration states the image
                // alone, and the module's own `OpImageFetch` reads it — a rail
                // that bound a sampler would be filling a descriptor nothing
                // samples through.
                NarrowSampler::Fetched => {
                    TextureBindingContract::fetched(texture.index, texture.format)
                }
            })
            .collect()
    }
}

/// Whether one request is the narrow class, and the facts the trace is built
/// from when it is.
///
/// Pure over the request and this rail's own recorded state, and ordered
/// cheapest-first so a refused shape costs nothing: no provider call, no
/// translation, no registration, and no device read. Every refusal names the
/// condition that kept the shape on the engine, because that string is what the
/// observer reports when a class boundary moves.
///
/// Two facts this gate reads are not properties of the request. One is the
/// attachment window, which belongs to the device's provider and is
/// [`declared_attachment_window`]'s — asked by [`submit_render`] after this
/// gate has accepted the shape. The other is the production registry
/// ([`recorded_production`], R22): a trace-produced sampled texture is in class
/// exactly when a pass of this rail has already declared that guest target's
/// production, which is the arm E-TX3's `TextureSource::TraceView` states — the
/// registry read is a lock and a lookup, and it is the whole of what "the
/// trace's own production" means on this side of the seam.
fn narrow_class<'a>(
    inputs: &'a RenderRailInputs<'a>,
    req: &'a DrawRequest,
) -> Result<NarrowPass<'a>, OutOfClass> {
    // R25: the packet's own chain value, when the caller hands it over for the
    // record that continues the chain. Role-gated here so the class states the
    // election once: only a middle has a predecessor whose frame the walk
    // carries, and the load gate below trusts this admission instead of asking
    // the role a second time. A record of any other position that arrives with
    // bytes keeps the refusals its own position gives it.
    let chain_middle_source_bytes = match inputs.role {
        RenderChainRole::Middle => inputs.chain_middle_source_bytes,
        RenderChainRole::Head | RenderChainRole::SoleOrTail => None,
    };
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
        // R25: the middle whose previous contents are the frame its predecessor
        // produced, handed over by the caller that carries it. As with R7b's
        // arm, the admission is the request's own statement of where those
        // contents come from (`target_rgba8` — the exec walk's chain value, the
        // one writer of a continuing record's seed) beside the caller's bytes,
        // and nothing more: the deeper conditions below still answer for a
        // middle that states them and then fails on its streams, scissor or
        // format. A middle that states any other source, or one whose caller
        // hands nothing over, keeps the refusal below.
        RenderChainRole::Middle
            if req.target_rgba8.is_some()
                && req.target_guest_seed.is_none()
                && !req.load_guest_target_backing
                && chain_middle_source_bytes.is_some() => {}
        RenderChainRole::Middle => {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_chain_middle",
                "a record in the middle of a multi-record packet stays on the engine while the \
                 caller does not hand it the frame the record before it produced: the middle \
                 begins from that frame, and this class carries it only as the caller's own \
                 bytes",
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
    // Three arms, and each is one of the canonical attachment's own loads: the
    // *provider's* image under the attachment's own identity
    // (`LoadOp::Resident`), the caller's own bytes declared for that same view
    // (`LoadOp::Load`, R23), or the guest's byte-exact clear. A record whose
    // previous contents are guest bytes is still a shape the class does not
    // carry (one load op per attachment, and the rail cannot confirm a seed is
    // the live image), so it keeps the engine by name.
    let mut carried_chain_middle = false;
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
        if !inputs.resident_frames_fetchable {
            // The resident the caller's own chain names lives in the *engine*'s
            // registry (`render_chain_identity`, `gva_chain_identity`), and
            // while the arms are held this rail writes no image of its own for
            // it — so `LoadOp::Resident` here would name an image no record
            // ever stored, which the contract refuses by name
            // (`resident_target_undeclared`) and which this rail must not turn
            // into a dropped draw.
            //
            // R23: the caller that owns that registry can read the frame out
            // and hand it over, and then the contents *are* nameable — by the
            // trace's own view declaration, the `LoadOp::Load` arm the contract
            // has carried since R5b. That is the same fact `target_rgba8`
            // cannot express: a seed is bytes of unknown provenance, while this
            // is the frame under the identity the request itself named, read by
            // the caller from the registry that holds it. The record then
            // enters the class with the identity it stated — the identity is
            // still required above, because the engine's own answer to this
            // record still names it.
            let Some(bytes) = inputs.resident_source_bytes else {
                return Err(OutOfClass::new(
                    "render_provider_out_of_class_resident_source",
                    "a record whose previous contents are the live GPU image stays on the engine \
                     while the caller can neither read a frame this rail keeps nor hand the \
                     frame over: the resident the chain names is the engine's own, and this rail \
                     defines no image under it",
                ));
            };
            if u64::try_from(bytes.len()).ok() != Some(extent) {
                return Err(OutOfClass::new(
                    "render_provider_out_of_class_resident_source_shape",
                    "a record whose previous contents are handed over at a width or extent other \
                     than the attachment's own stays on the engine: the canonical attachment \
                     uploads exactly the view's declared bytes, so a shorter or longer buffer \
                     would begin the pass from bytes no record wrote",
                ));
            }
            NarrowLoad::Bytes(bytes)
        } else {
            NarrowLoad::Resident(resident)
        }
    } else if let Some(bytes) = chain_middle_source_bytes {
        // R25: the middle's previous contents are the frame its predecessor
        // produced, handed over by the caller that carries it. The role gate
        // above admitted exactly this pair — the request's own `target_rgba8`
        // beside the caller's bytes — so this arm states the contract's
        // trace-owned load for it, the same load R23's arm states for the
        // frame the engine holds. The bytes have to be the attachment's own
        // tightly packed extent for the same reason R23's do: the canonical
        // attachment uploads the view's declared bytes verbatim, and a shorter
        // or longer buffer would begin the pass from bytes no record wrote.
        if u64::try_from(bytes.len()).ok() != Some(extent) {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_chain_middle_shape",
                "a middle record whose predecessor's frame is handed over at a width or extent \
                 other than the attachment's own stays on the engine: the canonical attachment \
                 uploads exactly the view's declared bytes, so a shorter or longer buffer would \
                 begin the pass from bytes no record wrote",
            ));
        }
        carried_chain_middle = true;
        NarrowLoad::Bytes(bytes)
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
    //
    //   **The arm is the caller's capability, not the shape's.** The frame it
    //   keeps is one only a caller that can fetch it can use
    //   ([`RenderRailInputs::resident_frames_fetchable`]); a caller that cannot
    //   gets the frame *published* instead — the pooled arm's own answer, which
    //   is a superset of what it asked for and the only one its readers (the
    //   engine's registry, the deferred GVA debt, the mapper-ref-texture store,
    //   the next record of its packet) can consume. Census v15 measured what
    //   the other choice costs: 11 892 kept frames and 6 824 frames the guest
    //   never got.
    // - `UnpublishedStore` is a store action that publishes nothing: no
    //   resident, no reader, and no writeback the caller could land. It keeps
    //   the engine until its route is reviewed, exactly as before.
    let mut published_held_resident = false;
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
                    if inputs.resident_frames_fetchable {
                        NarrowStore::Resident(resident)
                    } else {
                        // The caller asked to skip the readback; this rail
                        // declines the *optimisation* and answers with the
                        // bytes, never with a frame the caller cannot reach.
                        // The identity is still required above: a resident
                        // store that names none is the caller's wiring bug
                        // either way.
                        published_held_resident = true;
                        NarrowStore::Writeback
                    }
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
    //
    // The rule belongs to the state where the arms are *elected* — a caller
    // that can read kept frames. While they are held every answer publishes its
    // bytes through the pooled pair, so the name is one the caller keys its own
    // landing on and not an image this rail has to own: the rule would refuse
    // the very shapes census v15 measured it must answer (the head record of
    // every split packet is a named identity with a `Clear` load and a withheld
    // readback).
    if inputs.resident_frames_fetchable
        && resident.is_some()
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
             comes from the provider's own image — the resident load and store arms; the R23 \
             byte arm carries the caller's own four-byte frame, which no wide readback produces",
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
    // when that frame is one the class can name: the provider's own image
    // (`LoadOp::Resident`), or the bytes the caller read out of the rail that
    // holds it (R23's `LoadOp::Load`, elected above from `load_from_target`
    // beside `resident_source_bytes`). A continuing record that would begin
    // from guest bytes of unknown provenance, or from nothing at all, keeps the
    // engine.
    if req.continues_render_pass && !matches!(load, NarrowLoad::Resident(_) | NarrowLoad::Bytes(_))
    {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_encoder",
            "a record that continues a multi-record encoder stays on the engine: it begins from \
             the frame the record before it produced, which this class can only name when that \
             frame is the provider's own image or the bytes the caller hands over for it",
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
    // R9b/R9d: whether the request *binds* buffers is no longer the condition.
    // What decides which rail executes the draw is whether either stage's own
    // translation *declares* a `[[buffer(N)]]` argument, and — once one does —
    // whether the canonical pair (declaration beside view) can be stated from
    // what this request carries. Both answers live in `stage_buffer_gate`. The
    // stream count the gate's vertex-layout rule reads is the number of the
    // request's own *fetch tables* — one canonical binding per admitted stream,
    // which is the layout this request will be registered with — rather than
    // its attribute list, because an attribute list is one entry per location
    // and a layout is one entry per stream (`research/docs/26` §31).
    let stage_buffers = stage_buffer_gate(
        inputs,
        req,
        req.storage_buffers.len(),
        canonical_vertex_stream_count(&req.vertex_attributes),
    )?;
    // The sampled textures the fragment stage reads (v101, `research/docs/23`
    // §101): the class states the module's own declarations beside the draw's
    // binds — the canonical pass carries both, and the wire carries the pass's
    // texture list like any other view — and every shape the provider or the
    // module refuses keeps the engine under its own name ([`sampled_textures`]).
    let sampling = sampled_textures(inputs, req)?;
    if req.occlusion_query.is_some() {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_visibility",
            "a visibility-armed draw stays on the engine",
        ));
    }
    // The attachment's own blend state (v100, `research/docs/23` §100): a
    // request that blends is in class exactly when the shape can be stated
    // *and* carried — the canonical pass states the entry, the v40 wire
    // section carries it — and every other shape keeps the engine under its
    // own name ([`declared_blend`], which also answers the three factor
    // families the provider always refuses).
    let blend = declared_blend(req)?;
    if req.blend_color != [0.0; 4] {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_blend_color",
            "a draw with a blend constant stays on the engine",
        ));
    }
    // The viewport the canonical pass states is a rect of its own since v100
    // (`research/docs/23` §100, E-RV1): the pass body carries
    // `[origin_x, origin_y, width, height]` in framebuffer coordinates, so a
    // request that binds *one* rect the contract can name — integer, non-empty,
    // inside the attachment, 0..1 depth — is in class and the pass states it
    // verbatim. Everything else is the engine's own answer to give, one bucket
    // per shape ([`viewport_admits`]).
    let viewport = match req.viewports.as_slice() {
        [] => None,
        [viewport] => Some(viewport_admits(*viewport, req.width, req.height)?),
        _ => {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_viewport_count",
                "a draw that binds more than one viewport stays on the engine: the canonical \
                 pass carries at most one rect, and one rect cannot state the others",
            ))
        }
    };
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
    // The index stream's source, arm by arm (R11), exactly as a vertex
    // stream's (`R9q`): the request's own staged bytes, or the one registered
    // window its zero-copy bind was cut from. The window arm is what the
    // census read as `index_staging` on every draw whose index bind the draw
    // path had already imported, and a gather this rail cannot state as one
    // window keeps the engine under the same slug as before.
    let index_source = index_stream_source(&index.content)?;
    if index_source.len() == 0 {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_index_empty",
            "an empty index stream stays on the engine",
        ));
    }

    // One canonical stream per *fetch table* the request's attributes read, and
    // one attribute per location inside it: the engine numbers one Vulkan
    // binding per attribute *location* (`runtime/draw/vulkan.rs` builds
    // `binding: a.location`), while the canonical layout this class states has
    // one entry per stream with the attributes that read it at their own
    // offsets (`research/docs/26` §31). Both rails fetch the same bytes — the
    // entry's stride and each attribute's offset are the descriptor's own — and
    // the canonical shape is Metal's, which is what makes a draw whose vertex
    // stage *also* declares a `[[buffer(N)]]` argument statable: the streams it
    // occupies are the tables it has, not the attributes it reads. Four is the
    // canonical contract's `max_vertex_buffers` — the axis 97.7 % of the
    // measured draw stream lives on — and a fifth table is a layout the
    // contract cannot state, so it stays on the engine rather than in a trace
    // admission would refuse.
    if canonical_vertex_stream_count(&req.vertex_attributes) > MAX_VERTEX_BUFFERS {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_vertex_stream_limit",
            format!(
                "a draw whose attributes read more than {MAX_VERTEX_BUFFERS} vertex streams stays \
                 on the engine: {} is the canonical contract's `max_vertex_buffers`, and a wider \
                 layout is a shape admission refuses by name",
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
    let mut vertex_streams: Vec<NarrowVertexStream<'_>> =
        Vec::with_capacity(req.vertex_attributes.len());
    for (head, attribute) in req.vertex_attributes.iter().enumerate() {
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
                 (Float32x2, Float32x3, Float32x4, Uint32, Unorm8x2, Unorm8x4, Unorm16x2, \
                 Unorm16x4)",
            ));
        };
        let stride = u64::from(attribute.stride);
        if stride < u64::from(attribute.offset) + format.bytes() {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_vertex_stride",
                "a vertex layout whose attribute does not fit its stride stays on the engine",
            ));
        }
        // The stream's source, arm by arm (R9q): the request's own staged bytes,
        // or the one registered window a zero-copy bind was cut from. The
        // record-length rule is stated on the arm's own length, so a window
        // shorter than one record is `vertex_short` exactly as a staged copy
        // is, and a gather that is no window at all is the named refusal
        // [`vertex_stream_source`] states.
        let source = vertex_stream_source(&attribute.content)?;
        if source.len() < stride {
            return Err(OutOfClass::new(
                "render_provider_out_of_class_vertex_short",
                "a vertex stream shorter than one record stays on the engine",
            ));
        }
        // The table this attribute reads, or a new one: the walk that decides
        // that is [`one_vertex_stream`], which the gate's own stream count is
        // built from, so the layout stated here and the number the gate read are
        // one measurement.
        let stated = NarrowVertexAttribute {
            location: attribute.location,
            offset: u64::from(attribute.offset),
            format,
        };
        match vertex_streams
            .iter_mut()
            .find(|stream| one_vertex_stream(&req.vertex_attributes[stream.head], attribute))
        {
            Some(stream) => stream.attributes.push(stated),
            None => vertex_streams.push(NarrowVertexStream {
                stride,
                head,
                source,
                attributes: vec![stated],
            }),
        }
    }

    // R12: a pass whose runtime `[[sampler(n)]]` states the command channel
    // cannot carry stays on the engine by name. The frame format states the
    // pass's sampled *textures* (the v70 channel) but not its runtime sampler
    // list yet (`research/docs/23` §3.3, v102: the decoder states an empty
    // list), so a trace that crosses the owner→provider wire would reach
    // admission with the states dropped and be refused there
    // (`render_runtime_sampler_missing`) — a decline, not a fallback. The same
    // rule the v40 blend section's two shapes keep: the class answers the wire
    // question before the frame exists.
    let wire_carries_samplers = stage_buffers.is_empty()
        && vertex_streams
            .iter()
            .all(|stream| !matches!(stream.source, StreamSource::Window(_)))
        && !matches!(index_source, StreamSource::Window(_));
    if !sampling.runtime_samplers.is_empty() && !wire_carries_samplers {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_texture_sampler_wire",
            format!(
                "a draw whose runtime `[[sampler(n)]]` states would travel the owner→provider \
                 frame stays on the engine: the frame format does not carry the pass's runtime \
                 sampler list yet (`research/docs/23` §3.3, v102), so the decoded pass would \
                 state no sampler and the provider would refuse the declaration by name \
                 (`render_runtime_sampler_missing`) rather than execute a pass that samples \
                 through a state nobody stated — {} runtime sampler state(s) beside a declared \
                 stage buffer or a window-backed stream are that shape",
                sampling.runtime_samplers.len(),
            ),
        ));
    }
    // The same wire question for the pass's sampled *textures* themselves
    // (`research/docs/23` §101.5): the frame's pipeline entry carries a render
    // contract with an **empty** texture list — the codec has a kind for the
    // compute half's declarations (`PIPELINE_KIND_COMPUTE_TEXTURES`) and none
    // for the render half's — so a sampled pass that crosses the frame reaches
    // admission with its declarations dropped and is refused by name
    // (`UndeclaredTextureBinding`, measured on this rail *before* this
    // condition existed: `evidence/r22-reims-target-source-<sha>/`'s
    // pre-guard reading). A decline is not a fallback, so the class answers
    // the wire question before the frame exists, exactly as the runtime
    // sampler condition above does — and the shapes this keeps on the engine
    // are the ones whose binds the frame has to carry (a window-backed stream,
    // the index stream, or a declared stage buffer of either arm).
    if !sampling.textures.is_empty() && !wire_carries_samplers {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_texture_wire",
            format!(
                "a draw whose {} sampled texture(s) would travel the owner→provider frame stays \
                 on the engine: the frame's render contract carries no texture declarations yet \
                 (`research/docs/23` §101.5), so the decoded pass would bind views no contract \
                 declares and admission would refuse it by name \
                 (`UndeclaredTextureBinding`) — a decline, not a fallback, and the trace-produced \
                 arm above is exactly the shape this increment adds to that population",
                sampling.textures.len(),
            ),
        ));
    }
    // R22: a record that carries productions spends the trace's serial pool on
    // *both* passes' views. The pool's own bound is the contract's
    // (`serial_resource_limit` at admission), and a shape above it is one the
    // provider always refuses — so the class answers it here, by name, rather
    // than handing admission a trace it can only decline.
    let views = 1
        + vertex_streams.len()
        + 1
        + stage_buffers.len()
        + sampling.textures.len()
        + sampling
            .productions
            .iter()
            .map(|production| production.views)
            .sum::<usize>();
    if views > MAX_SERIAL_RESOURCES {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_texture_production_budget",
            format!(
                "a draw whose trace would declare {views} views stays on the engine: the \
                 canonical trace's serial pool holds {MAX_SERIAL_RESOURCES} \
                 (`serial_resource_limit`), and a draw that carries a production beside its own \
                 pass spends the pool on both — a shape the provider always refuses is not one \
                 this class executes",
            ),
        ));
    }
    Ok(NarrowPass {
        vertex_entry: vertex_entry.to_owned(),
        fragment_entry: fragment_entry.to_owned(),
        format,
        load,
        store,
        published_held_resident,
        carried_chain_middle,
        bgra: format == AttachmentFormat::Bgra8Unorm,
        width: u64::from(req.width),
        height: u64::from(req.height),
        extent,
        index_count: index.index_count,
        vertex_streams,
        index_stream: NarrowIndexStream {
            format: index_format,
            source: index_source,
        },
        stage_buffers,
        textures: sampling.textures,
        runtime_samplers: sampling.runtime_samplers,
        productions: sampling.productions,
        scissor,
        viewport,
        blend,
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
///
/// The four normalized 8/16-bit storages are the widening E-VF1 landed on the
/// canonical contract (`research/docs/23` §103): each is mapped here by the
/// Metal storage of the same meaning, and the read the contract names (`c / 255`
/// for the 8-bit pair, `c / 65535` for the 16-bit one) is the one both rails
/// already apply. Every other storage — the signed normalized family, the
/// three-channel 8/16-bit shapes, the packed words, the `_bgra` channel order —
/// keeps the engine under the gate's own name.
///
/// The *component shape* a storage pairs with is not this map's answer: an
/// `unorm*4` declaration beside a `float4` AIR member is the reviewed pairing,
/// and the same declaration beside a `float2` member is refused by the
/// provider's own registration gate (`render_stage_reflection_mismatch`), the
/// shape rule [`VertexFormat`]'s Vulkan side states.
fn vertex_format(
    format: crate::backend::vulkan::engine::VertexAttributeFormat,
) -> Option<VertexFormat> {
    use crate::protocol::vertex_format as raw;
    match format.ordinal() {
        raw::MTL_VERTEX_FORMAT_FLOAT2 => Some(VertexFormat::Float32x2),
        raw::MTL_VERTEX_FORMAT_FLOAT3 => Some(VertexFormat::Float32x3),
        raw::MTL_VERTEX_FORMAT_FLOAT4 => Some(VertexFormat::Float32x4),
        raw::MTL_VERTEX_FORMAT_U_INT => Some(VertexFormat::Uint32),
        raw::MTL_VERTEX_FORMAT_U_CHAR2_NORMALIZED => Some(VertexFormat::Unorm8x2),
        raw::MTL_VERTEX_FORMAT_U_CHAR4_NORMALIZED => Some(VertexFormat::Unorm8x4),
        raw::MTL_VERTEX_FORMAT_U_SHORT2_NORMALIZED => Some(VertexFormat::Unorm16x2),
        raw::MTL_VERTEX_FORMAT_U_SHORT4_NORMALIZED => Some(VertexFormat::Unorm16x4),
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
///
/// R23's byte arm rides the pooled pair with the other held record: it carries
/// its own contents in and lands its own frame out, so it names no image of its
/// own and must not touch the resident registry the chain's identity belongs to.
fn attachment_identity(pass: &NarrowPass<'_>) -> AttachmentIdentity {
    if let Some(present) = pass.present {
        return present.into();
    }
    let resident = match (pass.load, pass.store) {
        (NarrowLoad::Resident(resident), _) => resident,
        (_, NarrowStore::Resident(resident)) => resident,
        (NarrowLoad::Clear(_) | NarrowLoad::Bytes(_), NarrowStore::Writeback) => {
            ResidentAttachment {
                allocation: ATTACHMENT_ALLOCATION,
                view: ATTACHMENT_VIEW,
            }
        }
    };
    resident.into()
}

fn submit_narrow(
    inputs: &RenderRailInputs<'_>,
    req: &DrawRequest,
    pass: &NarrowPass<'_>,
    copies: &WindowCopies,
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
    // on the emulator side. R23's byte arm is the exception the rule was always
    // shaped for: it *is* a `LoadOp::Load` attachment, so these bytes are the
    // frame the pass begins from and have to be the caller's own.
    let attachment_source = match pass.load {
        NarrowLoad::Bytes(bytes) => bytes.to_vec(),
        _ => vec![0u8; usize::try_from(pass.extent).unwrap_or(0)],
    };
    let declaration = BufferView {
        view_id: attachment.view,
        metal_binding: 0,
        allocation_id: attachment.allocation,
        offset: 0,
        length: pass.extent,
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(attachment_source),
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
    // The owner's leases first (R9d/R9q): every window-backed binding this
    // pass states — a stage buffer's, and a vertex stream's since R9q — is
    // imported here, and the trace's views below name the allocation and lease
    // the plan minted for each. Importing before the trace exists is the same
    // order the compute rail keeps: a refused import never reaches admission.
    let mut leases = plan_owner_leases(provider, pass, &mut resources, copies)?;
    // R22: the productions this record's trace carries, restated in this
    // trace's own view namespace. Built before the views below because the
    // consuming declarations name the very views these passes store.
    let in_flight: Vec<ProductionInFlight> = pass
        .productions
        .iter()
        .enumerate()
        .map(|(index, production)| production_in_flight(production, index))
        .collect();
    for production in &in_flight {
        for (allocation_id, size) in production
            .allocations
            .iter()
            .copied()
            .chain(std::iter::once((production.allocation, production.extent)))
        {
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
    }
    let mut vertex_buffers = Vec::new();
    let mut next_view = FIRST_INPUT_VIEW;
    for (binding, stream) in pass.vertex_streams.iter().enumerate() {
        // The canonical binding index is the stream's position in the request,
        // not the guest's own binding number: the contract requires entry `i` to
        // carry `metal_binding == i` (`VertexBufferBindingMismatch` otherwise),
        // and the pipeline's vertex input state is keyed on attribute
        // *locations*, which are the guest's and are carried unchanged. Two
        // rails that agree on every location, format, offset and stride fetch
        // the same bytes whatever the binding numbers are called — which is why
        // the canonical block is the request's own fetch tables and can be
        // shorter than the attribute list (`research/docs/26` §31).
        let view = match &stream.source {
            StreamSource::Staged(bytes) => BufferView {
                view_id: ViewId::new(next_view),
                metal_binding: u32::try_from(binding).unwrap_or(u32::MAX),
                allocation_id: input_allocation(next_view),
                offset: 0,
                length: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(bytes.to_vec()),
            },
            // R9q: the stream's bytes are the owner's own mapping, imported
            // for this submission under the label the plan cut for this
            // stream. The view is the pair the owner's reservation covers —
            // the window's own offset and the bind's own length — and the
            // access is the read the class admitted. R18 adds the second arm
            // here: a stream whose view pointer the device's granules turn away
            // is *copied* by the gate, so the plan minted a staged lease over
            // those bytes and the view names it — the channel the plan took is
            // the fact that decides which source this view states, so the two
            // arms cannot drift apart.
            StreamSource::Window(_) => {
                let owner = leases
                    .as_ref()
                    .and_then(|plan| plan.view(vertex_stream_owner_binding(binding)))
                    .expect("the owner plan covers every admitted vertex window");
                BufferView {
                    view_id: ViewId::new(next_view),
                    metal_binding: u32::try_from(binding).unwrap_or(u32::MAX),
                    allocation_id: owner.allocation,
                    offset: owner.view_offset,
                    length: owner.view_length,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: match owner.channel {
                        provider_owner::Channel::Borrowed => {
                            BufferSource::BorrowedNoCopy(owner.lease)
                        }
                        provider_owner::Channel::Staged => BufferSource::StagedLease(owner.lease),
                    },
                }
            }
        };
        vertex_buffers.push(view);
        next_view += 1;
    }
    let index_view = ViewId::new(next_view);
    // R11: the index stream's own arm decides the view. Staged bytes stay
    // trace-owned, exactly as before; a window-backed index bind names the
    // lease the owner plan imported for it — the borrowed arm, or since R18 the
    // staged lease over the copy the gate made — with the view offset and
    // length the owner's own reservation covers. The same channel switch the
    // vertex stream above keeps, for the same reason.
    let (index_allocation, index_offset, index_length, index_source) = match &pass
        .index_stream
        .source
    {
        StreamSource::Staged(bytes) => (
            input_allocation(next_view),
            0,
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            BufferSource::OwnedBytes(bytes.to_vec()),
        ),
        StreamSource::Window(_) => {
            let owner = leases
                .as_ref()
                .and_then(|plan| plan.view(index_stream_owner_binding()))
                .expect("the owner plan covers every admitted index window");
            (
                owner.allocation,
                owner.view_offset,
                owner.view_length,
                match owner.channel {
                    provider_owner::Channel::Borrowed => BufferSource::BorrowedNoCopy(owner.lease),
                    provider_owner::Channel::Staged => BufferSource::StagedLease(owner.lease),
                },
            )
        }
    };
    let indices = IndexBufferBinding {
        view: BufferView {
            view_id: index_view,
            metal_binding: 0,
            allocation_id: index_allocation,
            offset: index_offset,
            length: index_length,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: index_source,
        },
        format: pass.index_stream.format,
    };
    // The index view has an identity of its own: the two namespaces are the
    // same one (`ViewId`), and R9j made the collision visible — a stage
    // buffer's pool entry is keyed by this identity, and an index view that
    // reused it was answered as "the same view changed its allocation, range
    // or source bytes" (`SerialBufferRebinding`). Harmless while every stage
    // buffer lived only in the render pass and nothing declared these views in
    // the pool; a landing declares them.
    next_view += 1;
    // The v83 stage-buffer half, filled (R9d): one view per declaration, at the
    // view offset and length the owner's own lease covers — the pair the pass
    // has to state for the canonical registration to execute it. The leases
    // themselves were imported above, beside the vertex windows, because both
    // halves name the same plan.
    let mut stage_buffers = Vec::with_capacity(pass.stage_buffers.len());
    // The view identities the completion's writebacks are paired with, kept
    // beside the views for the same reason the compute rail keeps its binding
    // table: the writeback names `(view, allocation)`, and only the trace's own
    // construction knows which bind that is.
    let mut stage_buffer_slots: Vec<StageBufferSlot> = Vec::with_capacity(pass.stage_buffers.len());
    for buffer in &pass.stage_buffers {
        let binding = stage_buffer_owner_binding(buffer.stage, buffer.index);
        let view = leases
            .as_ref()
            .and_then(|plan| plan.view(binding))
            .expect("the owner plan covers every admitted stage buffer");
        stage_buffer_slots.push(StageBufferSlot {
            stage: buffer.stage,
            index: buffer.index,
            view: ViewId::new(next_view),
            allocation: view.allocation,
            view_offset: view.view_offset,
            view_length: view.view_length,
            access: buffer.access,
        });
        stage_buffers.push(StageBufferView {
            stage: buffer.stage,
            view: BufferView {
                view_id: ViewId::new(next_view),
                metal_binding: buffer.index,
                allocation_id: view.allocation,
                offset: view.view_offset,
                length: view.view_length,
                // The access the pipeline declares for this slot (R9j): the
                // contract pairs the two field by field
                // (`StageBufferAccessMismatch`), so a writable declaration is
                // what makes this view a landing rather than a read.
                access: buffer.access,
                attribute_stride: None,
                source: match view.channel {
                    provider_owner::Channel::Borrowed => BufferSource::BorrowedNoCopy(view.lease),
                    provider_owner::Channel::Staged => BufferSource::StagedLease(view.lease),
                },
            },
        });
        next_view += 1;
    }

    // The v101 sampled-texture half (R10): one view per admitted declaration,
    // in the contract's canonical order, each carrying the request's own
    // tightly packed copy as a trace-owned source, the byte order its own bind
    // states (E-TX1, `research/docs/23` §107), and the Metal index its
    // declaration states (E-RS3). The sampler state is *not* a view field: the
    // declaration states it, and the canonical rail creates its `VkSampler`
    // from that state — which is why the class gate requires the draw's bind to
    // repeat it rather than trusting either half.
    let mut textures = Vec::with_capacity(pass.textures.len());
    for texture in &pass.textures {
        // Where the texels come from decides the identity, and so the source
        // arm: the request's own copy is a view this rail mints, while a
        // trace-produced texture *is* the identity the production stored under
        // (`Research/docs/23` §110: the view's own `(allocation, view)` pair
        // is the production's identity, so the declaration and its producer
        // agree by construction). The view number still advances for a
        // produced texture, so the numbering `input_allocations` derives stays
        // in lockstep with the views stated here.
        let (view_id, allocation_id, source) = match &texture.source {
            NarrowTextureSource::Bytes(bytes) => (
                ViewId::new(next_view),
                input_allocation(next_view),
                TextureSource::OwnedBytes(bytes.to_vec()),
            ),
            NarrowTextureSource::Produced { production, .. } => {
                let position = pass
                    .productions
                    .iter()
                    .position(|known| Arc::ptr_eq(known, production))
                    .expect("every produced texture names a production of this pass");
                let in_flight = &in_flight[position];
                (
                    in_flight.attachment,
                    in_flight.allocation,
                    TextureSource::TraceView,
                )
            }
        };
        textures.push(TextureView {
            view_id,
            metal_binding: texture.index,
            allocation_id,
            texture_type: TextureType::D2,
            format: texture.format,
            width: texture.width,
            height: texture.height,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            // The access the declaration states (R15): a fetched texture's
            // view is the image alone, and the canonical admission holds the
            // view's access to the declaration's field by field
            // (`TextureAccessMismatch`), so the two halves cannot drift.
            access: match texture.sampler {
                NarrowSampler::Fetched => TextureAccess::Fetched,
                NarrowSampler::Static(_) | NarrowSampler::Runtime { .. } => TextureAccess::Sampled,
            },
            source,
        });
        next_view += 1;
    }

    let pass_descriptor = RenderPassDescriptor {
        pipeline: render_pipeline.pipeline_id,
        // The runtime `[[sampler(n)]]` bindings the fragment stage executes
        // with (R12, `research/docs/23` §102): one state per argument an
        // admitted texture reads through, in the contract's canonical order,
        // each read off the draw's own bind at the device slot the module's
        // reflection resolves that argument at. Empty for every stage that
        // samples through its own AIR static state, which is the shape every
        // pre-R12 pass keeps byte for byte.
        samplers: pass
            .runtime_samplers
            .iter()
            .map(|sampler| RenderSamplerBinding::new(sampler.index, sampler.policy))
            .collect(),
        color_attachments: vec![RenderAttachment {
            view_id: attachment.view,
            allocation_id: attachment.allocation,
            format: pass.format,
            width: pass.width,
            height: pass.height,
            load: match pass.load {
                NarrowLoad::Clear(clear) => LoadOp::Clear(clear),
                NarrowLoad::Resident(_) => LoadOp::Resident,
                // R23: the previous contents are the caller's bytes, declared
                // for this attachment's view in the trace above and uploaded by
                // the rail before the pass opens (`LoadOp::Load`).
                NarrowLoad::Bytes(_) => LoadOp::Load,
            },
            store: match pass.store {
                NarrowStore::Writeback => StoreOp::Store,
                NarrowStore::Resident(_) => StoreOp::Resident,
            },
        }],
        // The pass's own rect when the request bound one the contract can name
        // (`research/docs/23` §100), and the attachment-covering default the
        // pre-v100 contract published for a request that bound none — the same
        // default both rails rasterized with before this face existed.
        viewport: pass.viewport.unwrap_or([
            0,
            0,
            u32::try_from(pass.width).unwrap_or(u32::MAX),
            u32::try_from(pass.height).unwrap_or(u32::MAX),
        ]),
        scissor: pass.scissor,
        vertices: pass.index_count,
        vertex_buffers,
        indices: Some(indices),
        base_vertex: 0,
        cull: None,
        // The attachment's own blend state, stated since v100: `Some` exactly
        // when the request declares `blendingEnabled` and the shape travels
        // ([`declared_blend`]), `None` for the no-blend, all-writes entry every
        // earlier increment stated.
        blend: pass.blend.clone(),
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
        // The v70 sampler channel, filled since v101 (R10): one view per
        // admitted declaration, in the contract's canonical order, paired with
        // the declaration that states the same Metal index by `validate_against`
        // (E-RS3) — which is the fragment stage's own `[[texture(n)]]` argument
        // by the contract's own rule, whether or not the list skips an index.
        textures,
        // The v83 stage-buffer half (R9d): one view per declaration the
        // pipeline's own contract carries, in the same canonical order
        // (`stage_buffer_gate` built both from the same declarations), each
        // naming the lease the owner rail imported for it. Empty for every
        // request whose stages declare no `[[buffer(N)]]` argument — the R9b
        // population, whose binds fill indices no descriptor can be needed for
        // and which therefore needs no declaration either.
        // Cloned for the declaring half below, which states one pool entry per
        // *writable* view from this same list, so the render pass's views and
        // the pool entries they land through are one set of bytes rather than
        // two spellings of them.
        stage_buffers: stage_buffers.clone(),
    };
    // R22: the pass this record states is a candidate production of the guest
    // target it stores into, and a later record may sample it. The *copy* is
    // paid here rather than after the completion because the trace takes the
    // descriptor, and it is paid only for the shapes
    // [`production_recordable`] admits — the record this rail can restate
    // inside another record's trace. A shape that is not one keeps no copy.
    let recordable_descriptor = production_recordable(req, pass).then(|| pass_descriptor.clone());
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(NEXT_OPERATION_ID.fetch_add(1, Ordering::Relaxed)),
        // R22: one render pipeline per pass this trace states — the consumer's
        // own, and the registered pipeline of every production it carries.
        // Deduplicated by pipeline id, because two records of one module pair
        // share the registration and a trace states each pipeline once.
        pipelines: {
            let mut pipelines = vec![declaring.clone(), render_pipeline.clone()];
            for production in &pass.productions {
                if !pipelines
                    .iter()
                    .any(|known| known.pipeline_id == production.pipeline.pipeline_id)
                {
                    pipelines.push(production.pipeline.clone());
                }
            }
            pipelines
        },
        encoder_dispatch_type: DispatchType::Serial,
        passes: {
            let mut passes =
                declaring_passes_with(declaring.pipeline_id, declaration, &stage_buffers);
            // The productions' own declarations and passes, in the order the
            // textures named them: a trace-produced view has to be declared
            // before a render pass stores into it (the trace's pool is what
            // admission resolves an attachment view against), and the store
            // has to precede the sampling pass — which is the same order
            // `validate_serial_buffer_reuse` walks.
            for production in &in_flight {
                passes.push(declaring_pass(
                    declaring.pipeline_id,
                    vec![production_declaration(production)],
                ));
            }
            for production in &in_flight {
                passes.push(TracePass::Render(production.descriptor.clone()));
            }
            passes.push(TracePass::Render(pass_descriptor));
            passes
        },
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    // R9j/R9q: a pass that declares a stage buffer — or, since R9q, carries a
    // vertex stream through the owner's window — crosses the owner→provider
    // wire before anything is admitted. The frame is the payload (every
    // declaration, view and source this rail stated), it is decoded again with
    // the provider's own decoder, and it is what the admission below sees — so
    // a field the wire cannot carry is a typed decline here at the seam rather
    // than a difference discovered when the owner and the provider are two
    // processes. A trace whose stages declare no buffer *and* whose streams are
    // all staged keeps the exact path (and bytes) it had before the declaration
    // half existed; the frames of the shapes this increment does not touch are
    // R9h's 7/7 regression. The gate is the plan itself rather than the
    // declaration list, because the plan is exactly the set of bindings that
    // travel as leases — a vertex window without a declaration crosses the wire
    // on the same rule (R9q).
    //
    // R10 adds two faces to the *trace* and none to this gate: a sampled
    // texture's view travels in the pass's own texture list (the v70 channel)
    // and a v40-shaped blend entry travels in the pass's blend block, so both
    // cross with every frame that crosses at all — while the *registration's*
    // own frame does not carry the render texture declarations yet
    // (`research/docs/23` §101.5: the provider's `get_render_pipeline_contract`
    // still decodes an empty list), so this rail's sampled passes are the
    // in-process arm the class gate's declarations make exact. The two shapes
    // the v40 blend section cannot carry are answered by the class gate before
    // this point (`declared_blend`), not framed as another state here.
    let (trace, resources) = if leases.is_none() {
        (trace, resources)
    } else {
        let frame = provider_wire::submit_frame(&trace, &resources).map_err(|decline| {
            abort_stage_buffer_leases(&mut leases, provider);
            ProviderRenderDecline::StageBufferWire {
                step: decline.step,
                detail: decline.detail,
            }
        })?;
        provider_wire::note_submit_frame();
        match provider_wire::carried_submission(&frame) {
            Ok((trace, resources)) => {
                note_wire_stage_buffers(&trace);
                (trace, resources)
            }
            Err(decline) => {
                abort_stage_buffer_leases(&mut leases, provider);
                return Err(ProviderRenderDecline::StageBufferWire {
                    step: decline.step,
                    detail: decline.detail,
                });
            }
        }
    };
    let validated = match provider.capabilities().validate_trace(trace, resources) {
        Ok(validated) => validated,
        Err(error) => {
            abort_stage_buffer_leases(&mut leases, provider);
            return Err(ProviderRenderDecline::TraceAdmission {
                detail: provider_error_detail(&error),
            });
        }
    };
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
    let result = match provider.submit(validated) {
        Ok(result) => result,
        Err(error) => {
            // The refusal is mapped first: a `DeviceLost` refusal runs the
            // contract's teardown over the owner ledger, and the census it
            // reports is the state the teardown found — before this plan's own
            // unwind releases what it imported.
            let decline = refusal_decline(&error, "submission").into_render();
            abort_stage_buffer_leases(&mut leases, provider);
            return Err(decline);
        }
    };
    if !matches!(
        result.completion,
        CompletionDisposition::CompletedVisible { .. }
    ) {
        abort_stage_buffer_leases(&mut leases, provider);
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
                abort_stage_buffer_leases(&mut leases, provider);
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
    // The retirement chain (`research/docs/20` §3.4), in the order the owner
    // rail states it: bind the completion's token to every lease, observe that
    // the completion retires it, retire and reclaim each window, release the
    // provider's import. It runs before this rail reads the completion's
    // writebacks, exactly as the compute rail's does, so no provider-visible
    // byte reaches the caller while a lease that covers it is still held.
    if let Some(plan) = leases.take() {
        plan.settle(provider, result.completion)
            .map_err(ProviderRenderDecline::Owner)?;
    }
    // R9j: the writable stage buffers' bytes, on the same `BufferWriteback`
    // channel the stored attachment lands through (R9f). One writeback per
    // writable view, in the contract's canonical order, re-based onto the bind
    // the pass stated with the range checked against the view the trace
    // carried — a provider-reported interval the view cannot hold is never
    // truncated into place. The bytes leave here for the caller, which owns the
    // guest destination (`StageBufferLanding`).
    let stage_writebacks = stage_buffer_writebacks(&stage_buffer_slots, &result.writebacks)?;
    // R22: every production this trace carried has to have landed its bytes in
    // the trace's own writeback channel, or the consuming declaration sampled a
    // view the provider never produced. The contract refuses that shape at
    // admission (`render_texture_source_unwritten`) and the producer's store is
    // `StoreOp::Store`, so this is a check rather than an expectation: a
    // submission whose trace states a production and no writeback for it is a
    // wiring defect and a typed decline, never a frame.
    for production in &in_flight {
        let landed = result.writebacks.iter().any(|writeback| {
            writeback.view_id == production.attachment
                && writeback.allocation_id == production.allocation
        });
        if !landed {
            return Err(ProviderRenderDecline::AttachmentWritebackMissing);
        }
    }
    // A production is recorded only once it has actually run: the re-run the
    // consuming trace states is the pass this submission sent, and the bytes a
    // later consumer samples are the ones this completion published.
    if let Some(descriptor) = recordable_descriptor.as_ref() {
        record_production(req, pass, &render_pipeline, descriptor);
    }
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
    // The two byte arms' own populations, counted where the answer happens for
    // the reason above: the frame was carried into the pass as bytes, so
    // neither resident arm moved and these are the only names that say the
    // record was answered this way. R23's is the frame the caller read out of
    // the engine's registry; R25's is the packet's own chain value, handed on
    // by the walk.
    if matches!(pass.load, NarrowLoad::Bytes(_)) {
        crate::runtime::drain::note_store_route(if pass.carried_chain_middle {
            "render_provider_chain_middle_source_bytes"
        } else {
            "render_provider_resident_source_bytes"
        });
    }
    // The held arm's own population: a caller withheld its readback and this
    // rail published the frame because the caller cannot fetch a kept one.
    // Counted here rather than where the arm was elected, so a submission that
    // never reached the caller is not counted as an answer.
    if pass.published_held_resident {
        crate::runtime::drain::note_store_route("render_provider_publish_held_resident");
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
        stage_writebacks,
    }))
}

/// The trace's compute half: the pass that declares the attachment's bytes,
/// plus one declare pass per *writable* stage buffer (R9f/R9j).
///
/// A landing has to name a view the trace declares, and the views a trace
/// declares are the ones a compute pass binds — the trace's pool
/// ([`ComputeTrace::serial_resources`]). So each writable stage buffer rides in
/// a declare pass of its own, carrying the *same* bytes, source and identity as
/// the render pass's view of the same slot (`SerialBufferRebinding` holds the
/// two to each other field by field), at the read access the declaring kernel's
/// own reflection states. One pass per view rather than one pass with several
/// bound views, because a compute pass's bound set has to be exactly what its
/// pipeline declares (`UnknownBinding` / `MissingBinding`), and the declaring
/// kernel declares one buffer.
///
/// The render half follows, in the same order the render pass always had.
fn declaring_passes_with(
    pipeline: metal_api_core::provider::PipelineId,
    attachment: BufferView,
    stage_buffers: &[StageBufferView],
) -> Vec<TracePass> {
    let mut passes = vec![declaring_pass(pipeline, vec![attachment])];
    for buffer in stage_buffers
        .iter()
        .filter(|buffer| buffer.view.access.is_writable())
    {
        let mut pool = buffer.view.clone();
        // The declaring kernel's own interface: one buffer, read.
        pool.metal_binding = 0;
        pool.access = BufferAccess::Read;
        passes.push(declaring_pass(pipeline, vec![pool]));
    }
    passes
}

/// One declaring compute pass: the one-thread kernel that reads a view, which
/// is how a trace declares the bytes a render attachment or a writable stage
/// buffer lands in (`research/docs/23` §3.6).
fn declaring_pass(
    pipeline: metal_api_core::provider::PipelineId,
    buffers: Vec<BufferView>,
) -> TracePass {
    TracePass::Compute(ComputePass {
        pipeline,
        buffers,
        textures: Vec::new(),
        dispatch: Dispatch {
            kind: DispatchKind::ThreadsExact,
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
        },
    })
}

/// One recorded production restated inside a consuming trace (R22): the pass
/// descriptor in this trace's own view namespace, the view its own production
/// lands in, and the trace-owned allocations the rest of the trace has to
/// declare.
struct ProductionInFlight {
    /// The identity's allocation, one per guest target (`resident_attachment`).
    allocation: AllocationId,
    /// The extent of the surface this production stores.
    extent: u64,
    /// The view this production's own pass stores into — minted here, and the
    /// same view the consuming declaration of the identity names.
    attachment: ViewId,
    descriptor: RenderPassDescriptor,
    allocations: Vec<(AllocationId, u64)>,
}

/// Restate one recorded production inside the consuming trace.
///
/// Three rewrites, each holding one invariant of the trace it lands in:
///
/// * every view moves into the production's own block of
///   [`PRODUCTION_VIEW_BASE`], because the contract's view-id namespace is one
///   per trace: the conflict and declaration walks are keyed by [`ViewId`]
///   alone, so two productions — and a production beside the consuming pass's
///   own views — have to be distinct ids and not merely distinct pairs. A
///   *staged* view's allocation is re-derived from its minted view number,
///   exactly as [`input_allocation`] derives the consuming pass's;
/// * every bind is already the trace's own bytes ([`production_bytes`] read
///   them when the production was recorded), so the consuming trace states no
///   lease and needs no owner plan;
/// * the colour attachment becomes the identity's allocation at this trace's
///   view with [`StoreOp::Store`], which is what makes the frame land in the
///   trace's writeback channel for the consumer to sample.
fn production_in_flight(production: &RecordedProduction, index: usize) -> ProductionInFlight {
    let mut descriptor = production.descriptor.clone();
    let offset = PRODUCTION_VIEW_BASE + u64::try_from(index).unwrap_or(0) * PRODUCTION_VIEW_STRIDE;
    let mint = |view: ViewId| ViewId::new(view.get() + offset);
    let mut allocations = Vec::new();
    for vertex in descriptor.vertex_buffers.iter_mut() {
        vertex.view_id = mint(vertex.view_id);
        if let BufferSource::OwnedBytes(bytes) = &vertex.source {
            vertex.allocation_id = input_allocation(vertex.view_id.get());
            allocations.push((
                vertex.allocation_id,
                u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            ));
        }
    }
    if let Some(indices) = descriptor.indices.as_mut() {
        indices.view.view_id = mint(indices.view.view_id);
        if let BufferSource::OwnedBytes(bytes) = &indices.view.source {
            indices.view.allocation_id = input_allocation(indices.view.view_id.get());
            allocations.push((
                indices.view.allocation_id,
                u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            ));
        }
    }
    for stage in descriptor.stage_buffers.iter_mut() {
        stage.view.view_id = mint(stage.view.view_id);
    }
    for texture in descriptor.textures.iter_mut() {
        texture.view_id = mint(texture.view_id);
        if let TextureSource::OwnedBytes(bytes) = &texture.source {
            texture.allocation_id = input_allocation(texture.view_id.get());
            allocations.push((
                texture.allocation_id,
                u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            ));
        }
    }
    for color in descriptor.color_attachments.iter_mut() {
        color.view_id = mint(color.view_id);
        color.allocation_id = production.allocation;
        color.store = StoreOp::Store;
    }
    let attachment = descriptor
        .color_attachments
        .first()
        .map(|color| color.view_id)
        .expect("a recorded production states its one colour attachment");
    ProductionInFlight {
        allocation: production.allocation,
        extent: production.extent,
        attachment,
        descriptor,
        allocations,
    }
}

/// The declaration one production's view needs before a render pass can store
/// into it: the same one-word read the consuming attachment's declaration is,
/// at the identity's own extent and allocation (R22).
fn production_declaration(production: &ProductionInFlight) -> BufferView {
    BufferView {
        view_id: production.attachment,
        metal_binding: 0,
        allocation_id: production.allocation,
        offset: 0,
        length: production.extent,
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(vec![
            0u8;
            usize::try_from(production.extent)
                .unwrap_or(usize::MAX)
        ]),
    }
}

/// One admitted stage buffer's view identity, kept from the trace construction
/// so the completion's writebacks can be paired with the bind they belong to
/// (`research/docs/26` §R9j).
struct StageBufferSlot {
    stage: RenderPipelineStage,
    index: u32,
    view: ViewId,
    allocation: AllocationId,
    view_offset: u64,
    view_length: u64,
    access: BufferAccess,
}

/// The provider's writebacks for this submission's writable stage buffers, in
/// the contract's canonical order.
///
/// The pairing is the same `(view, allocation)` identity the compute rail's
/// `staged_writebacks` uses, and so are the two refusals: a writable view the
/// completion named no writeback for is a landing that did not happen, and one
/// whose interval leaves the view is a range the staged bytes cannot hold —
/// neither is truncated into place, and both are typed because an in-class
/// shape is never re-run somewhere else.
fn stage_buffer_writebacks(
    slots: &[StageBufferSlot],
    writebacks: &[BufferWriteback],
) -> Result<Vec<StageWriteback>, ProviderRenderDecline> {
    let mut out = Vec::new();
    for slot in slots.iter().filter(|slot| slot.access.is_writable()) {
        let Some(writeback) = writebacks.iter().find(|writeback| {
            writeback.view_id == slot.view && writeback.allocation_id == slot.allocation
        }) else {
            return Err(ProviderRenderDecline::StageBufferWritebackMissing {
                stage: slot.stage,
                index: slot.index,
            });
        };
        let length = u64::try_from(writeback.bytes.len()).unwrap_or(u64::MAX);
        let inside = slot
            .view_offset
            .checked_add(slot.view_length)
            .zip(writeback.offset.checked_add(length))
            .is_some_and(|(view_end, writeback_end)| {
                writeback.offset >= slot.view_offset && writeback_end <= view_end
            });
        if !inside {
            return Err(ProviderRenderDecline::StageBufferWritebackShape {
                stage: slot.stage,
                index: slot.index,
                offset: writeback.offset,
                length,
                view_offset: slot.view_offset,
                view_length: slot.view_length,
            });
        }
        out.push(StageWriteback {
            stage: slot.stage,
            index: slot.index,
            offset: writeback.offset - slot.view_offset,
            bytes: writeback.bytes.clone(),
        });
    }
    Ok(out)
}

/// Every trace-owned input allocation the render half carries: one per *staged*
/// vertex stream and one for the index stream, with the view's own length as the
/// extent. One per stream, so a table several attributes read travels once —
/// the bytes are the same allocation the request already holds
/// ([`one_vertex_stream`]).
///
/// A stream that travels as the owner's window is deliberately absent (R9q,
/// R11 — a vertex stream or the index stream): its allocation is the one
/// [`plan_owner_leases`] minted for the lease, and the trace's view names that
/// allocation rather than one of this rail's own. The view *identities* below
/// still advance once per stream, so a stream's identity never depends on
/// which arm it took.
fn input_allocations(pass: &NarrowPass<'_>) -> Vec<(AllocationId, u64)> {
    let mut out = Vec::with_capacity(pass.vertex_streams.len() + 1 + pass.textures.len());
    let mut next_view = FIRST_INPUT_VIEW;
    for stream in &pass.vertex_streams {
        if let StreamSource::Staged(bytes) = &stream.source {
            out.push((
                input_allocation(next_view),
                u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            ));
        }
        next_view += 1;
    }
    if let StreamSource::Staged(bytes) = &pass.index_stream.source {
        out.push((
            input_allocation(next_view),
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        ));
    }
    // The sampled textures' views are stated after the index view and the
    // stage buffers' (which claim a view number each but mint their allocation
    // through the owner rail), so their own allocations start one past the
    // index and every declared stage buffer — the same walk `submit_narrow`
    // makes, one number at a time.
    let texture_base = next_view + 1 + u64::try_from(pass.stage_buffers.len()).unwrap_or(u64::MAX);
    for (index, texture) in pass.textures.iter().enumerate() {
        // A trace-produced texture names the identity its production stored
        // under (R22): the allocation is the production's, declared from that
        // side of the trace, and re-declaring it here would be a second record
        // for one view.
        let NarrowTextureSource::Bytes(bytes) = &texture.source else {
            continue;
        };
        let view_number = texture_base + u64::try_from(index).unwrap_or(u64::MAX);
        out.push((
            input_allocation(view_number),
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        ));
    }
    out
}

/// The allocation identity of one render input view: the view identity offset
/// into this rail's own namespace, so a stream can never alias the attachment.
fn input_allocation(view: u64) -> AllocationId {
    AllocationId::new(0x7265_6e64_6572_0100 + view)
}

/// The label one stage buffer travels under in the owner rail.
///
/// [`provider_owner`] labels each request and each view with one `binding`
/// number, and it was written for the compute rail's single namespace. A render
/// pass has two: a vertex `[[buffer(0)]]` and a fragment `[[buffer(0)]]` are
/// different arguments, so the label has to be a function of the pair rather
/// than of the index alone. The stage's own canonical ordinal carries the high
/// half and the Metal index the low one, and the label stays a small integer
/// because that is what the owner rail's log lines print.
fn stage_buffer_owner_binding(stage: RenderPipelineStage, index: u32) -> u32 {
    (u32::from(stage.code()) << 16) | index
}

/// The label one vertex stream travels under in the owner rail (`R9q`).
///
/// [`stage_buffer_owner_binding`] owns the high halves `0` and `1` — the two
/// stages' `[[buffer(N)]]` namespaces — and a vertex stream is not an argument
/// in either of them: the canonical binding it is resolved under is the
/// layout's own stream numbering (`research/docs/26` §31). The third high half
/// keeps the two namespaces from ever being compared. The gate refuses a
/// vertex-stage declaration that lands inside `0..vertex_streams`
/// (`stage_buffer_shape_vertex_layout`), so the labels could not collide while
/// that rule holds — but a label is what the plan looks a view up by, and a
/// lookup answered by the other namespace's lease would be a wrong frame
/// rather than a refusal.
fn vertex_stream_owner_binding(stream: usize) -> u32 {
    (0x0002 << 16) | u32::try_from(stream).unwrap_or(u32::MAX)
}

/// The label the index stream travels under in the owner rail (`R11`).
///
/// The fourth namespace of the same three ([`stage_buffer_owner_binding`]'s
/// two stages, then [`vertex_stream_owner_binding`]): a draw has exactly one
/// index buffer and it is no stage's `[[buffer(N)]]` argument, so the label is
/// the namespace alone. A constant rather than a function of the canonical
/// binding, because there is only one index view per pass and the contract
/// spells it with `metal_binding` zero — the label still has to be distinct,
/// since a lookup answered by another namespace's lease would be a wrong frame
/// rather than a refusal.
fn index_stream_owner_binding() -> u32 {
    0x0003 << 16
}

/// One window-backed binding's coordinates in the owner rail's own shape, under
/// the label the plan looks its view up by.
///
/// The three shapes that state windows — a stage buffer, a vertex stream and
/// the index stream — differ only in the label, which is why it is a parameter
/// and the coordinates are built in one place. The borrowed *request* and the
/// staged *copy* ([`provider_owner::window_bytes`], R18) are the two readers of
/// this one spelling of the coordinates.
fn owner_window(binding: u32, window: StageBufferWindow) -> provider_owner::Window {
    provider_owner::Window {
        binding,
        import: window.import,
        host_va: window.host_va,
        length: window.length,
        head: window.head,
        bytes_len: window.bytes_len,
    }
}

/// One window-backed binding's owner request: the borrowed arm, at the window's
/// own coordinates.
fn window_request(binding: u32, window: StageBufferWindow) -> provider_owner::Request<'static> {
    provider_owner::Request::Window(owner_window(binding, window))
}

/// The arm one window-backed binding's owner request states (`R18`): the
/// borrowed window, or the owner-issued staged lease over the copy this pass
/// made of it.
///
/// `copies` decides alone, because it is filled only for the binds whose view
/// pointer the device's granules turned away — and a bind that has a copy is by
/// construction one the borrowed arm cannot state.
fn window_arm_request<'a>(
    binding: u32,
    window: StageBufferWindow,
    copies: &'a WindowCopies,
) -> provider_owner::Request<'a> {
    match copies.bytes(binding) {
        Some(bytes) => provider_owner::Request::Staged(provider_owner::Staged { binding, bytes }),
        None => window_request(binding, window),
    }
}

/// Import the owner's leases for one submission (`R9d`, `R9q`, `R11`, `R18`).
///
/// The three arms `research/docs/26` §13.7 left open are decided here, one
/// binding at a time, exactly as the compute rail decides them
/// ([`super::provider_compute`]'s owner plan): a bind whose bytes the seam cut
/// out of a registered guest RAM window is imported without copying
/// (`BufferSource::BorrowedNoCopy`), and a bind the owner holds as staged bytes
/// is imported as an owner-issued staged lease (`BufferSource::StagedLease`).
/// The third arm — trace-owned bytes — is what a *staged* stream travels as,
/// and it is not a stage buffer's arm: a stage buffer's bytes belong to the
/// guest's buffer, which is why the gate admits only the two owner arms
/// (`..._stage_buffer_gather` is the name for everything else).
///
/// R9q puts the *window* arm of the vertex streams in the same plan, and R11
/// the index stream's beside it: a stream the request holds as a gather whose
/// one registered window covers it is stated as `BorrowedNoCopy` too, under
/// [`vertex_stream_owner_binding`] / [`index_stream_owner_binding`]. Two
/// bindings of one registration — a stage buffer and a stream, the index stream
/// and a stream, or two streams — share the one lease the plan imports for that
/// registration, which is the contract's own grouping (`research/docs/14`'s
/// "one allocation, many views"). A *staged* stream is deliberately not here:
/// it is not a bind of the guest's buffer, it is the bytes the runtime already
/// read, and it stays trace-owned exactly as before.
///
/// R18 gives a window-backed bind a *second* owner arm. The gate copies the
/// bytes of any window whose view pointer the device's granules turn away
/// ([`window_arm`], [`WindowCopies`]), and this plan states those binds as
/// `StagedLease` over the copy — the same arm a bind with no window behind it
/// takes, and the arm the copy is *for*. The window's own coordinates are then
/// not part of the plan at all: nothing about this binding is imported, so
/// nothing can be read through a pointer the device will not take.
///
/// Every lease is imported before the trace exists, so a refused import never
/// reaches admission, and the allocation table is extended with what the plan
/// minted before the trace is validated against it — the same order the compute
/// rail uses, for the same reason: an allocation the trace names must be in the
/// snapshot admission checks.
fn plan_owner_leases(
    provider: &metal_api_vulkan::VulkanComputeProvider,
    pass: &NarrowPass<'_>,
    resources: &mut ResourceTableSnapshot,
    copies: &WindowCopies,
) -> Result<Option<provider_owner::Plan>, ProviderRenderDecline> {
    if pass.stage_buffers.is_empty()
        && pass.vertex_windows().next().is_none()
        && pass.index_window().is_none()
    {
        return Ok(None);
    }
    let mut requests: Vec<provider_owner::Request<'_>> = pass
        .stage_buffers
        .iter()
        .map(|buffer| {
            let binding = stage_buffer_owner_binding(buffer.stage, buffer.index);
            match buffer.window {
                Some(window) => window_arm_request(binding, window, copies),
                None => provider_owner::Request::Staged(provider_owner::Staged {
                    binding,
                    // The gate admits a bind with neither a window nor staged
                    // bytes only under its own name, so this arm is total here.
                    bytes: buffer
                        .bytes
                        .expect("a stage buffer without a window carries the owner's staged bytes"),
                }),
            }
        })
        .collect();
    requests.extend(pass.vertex_windows().map(|(binding, window)| {
        window_arm_request(vertex_stream_owner_binding(binding), window, copies)
    }));
    requests.extend(
        pass.index_window()
            .map(|window| window_arm_request(index_stream_owner_binding(), window, copies)),
    );
    let plan = provider_owner::plan(provider, &requests).map_err(ProviderRenderDecline::Owner)?;
    for (allocation, size, reservation) in plan.leases() {
        if let Err(error) = resources.insert_allocation(AllocationRecord {
            allocation_id: allocation,
            owner_epoch: provider.device_epoch(),
            size,
        }) {
            plan.abort(provider);
            return Err(ProviderRenderDecline::TraceAdmission {
                detail: error.to_string(),
            });
        }
        if let Err(error) = resources.insert_lease(reservation) {
            plan.abort(provider);
            return Err(ProviderRenderDecline::TraceAdmission {
                detail: error.to_string(),
            });
        }
    }
    Ok(Some(plan))
}

/// Give up the leases one submission imported, on a path that never reached a
/// completion.
fn abort_stage_buffer_leases(
    leases: &mut Option<provider_owner::Plan>,
    provider: &metal_api_vulkan::VulkanComputeProvider,
) {
    if let Some(plan) = leases.take() {
        plan.abort(provider);
    }
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
        // The v83 half, stated (R9d/R9j): one declaration per `[[buffer(N)]]`
        // argument the two stages' own translations reported, in the
        // contract's canonical order, with the access the reflection reported
        // and the footprint it reaches (a static ceiling, or the affine access
        // set R9f landed). The gate admitted this request by pairing each of
        // these declarations with a bind, so the pass below states exactly one
        // view per entry — carrying the same access — and core holds the two
        // lists to each other (`validate_against`). An empty list is the
        // pre-R9d shape where neither stage declares a buffer at all.
        stage_buffers: pass
            .stage_buffers
            .iter()
            .map(|buffer| StageBufferBinding {
                stage: buffer.stage,
                index: buffer.index,
                access: buffer.access,
                footprint: buffer.proof.clone(),
            })
            .collect(),
        // The render texture declarations (E-RS1, `research/docs/23` §101): one
        // entry per admitted sampled texture, each restating the module's own
        // AIR sampler state at the Metal index the pass's own view states
        // (E-RS3, §104), in the same canonical order those views are stated in.
        textures: pass.texture_declarations(),
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

    /// The three `stage_buffer_shape` arms' routes are three names, and they
    /// are *these* three: the census reads those keys by hand, so a rename has
    /// to fail here rather than silently re-file a population (R9o).
    #[test]
    fn the_stage_buffer_shape_arms_charge_three_distinct_routes() {
        let routes = [
            stage_buffer_shape_route(StageBufferShapeRoute::TooMany),
            stage_buffer_shape_route(StageBufferShapeRoute::Duplicate),
            stage_buffer_shape_route(StageBufferShapeRoute::VertexLayout),
        ];
        assert_eq!(
            routes,
            [
                "stage_buffer_shape_gt4",
                "stage_buffer_shape_duplicate",
                "stage_buffer_shape_vertex_layout",
            ]
        );
        let mut distinct = routes.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            routes.len(),
            "one name per rule: {routes:?}"
        );
    }
}

#[cfg(test)]
mod vertex_stream_tests {
    use super::*;

    /// One request attribute off one staged table.
    fn attribute(
        location: u32,
        offset: u32,
        stride: u32,
        table: &std::sync::Arc<Vec<u8>>,
    ) -> VertexAttributeResource {
        VertexAttributeResource {
            location,
            binding: location,
            format: crate::backend::vulkan::engine::VertexAttributeFormat::parse(29)
                .expect("Float2 is a vertex format"),
            offset,
            stride,
            step_function: VertexStepFunction::PerVertex,
            step_rate: 1,
            content: BufferContent::Bytes(std::sync::Arc::clone(table)),
        }
    }

    /// One request attribute off one zero-copy bind (`R9q`).
    ///
    /// The source is passed in rather than built here for the same reason the
    /// staged helper passes its allocation in: the identity *is* the bind, and
    /// a clone of one `GuestRunSource` is exactly what the runtime hands two
    /// attributes that read one guest vertex buffer.
    fn guest_attribute(
        location: u32,
        offset: u32,
        stride: u32,
        source: &crate::backend::vulkan::engine::GuestRunSource,
    ) -> VertexAttributeResource {
        VertexAttributeResource {
            location,
            binding: location,
            format: crate::backend::vulkan::engine::VertexAttributeFormat::parse(29)
                .expect("Float2 is a vertex format"),
            offset,
            stride,
            step_function: VertexStepFunction::PerVertex,
            step_rate: 1,
            content: BufferContent::GuestRuns(source.clone()),
        }
    }

    /// One zero-copy bind's source over a synthetic mapping: what the draw path
    /// hands this rail after `bound_buffer_content`.
    ///
    /// `runs` is the identity the grouping reads [`one_vertex_stream`] through,
    /// and `source_offset` / `total_len` are the window inside it — the two
    /// facts a packed resource's per-offset bind varies. Nothing here reads the
    /// mapping, so the address is a constant the grouping never dereferences.
    fn guest_bind(
        source_offset: u64,
        total_len: u64,
    ) -> crate::backend::vulkan::engine::GuestRunSource {
        let run = crate::backend::vulkan::engine::GuestRun::in_mapping(
            0x1000,
            1 << 20,
            source_offset,
            total_len,
        )
        .expect("the bind's bytes are inside the synthetic mapping");
        crate::backend::vulkan::engine::GuestRunSource {
            runs: std::sync::Arc::new(vec![run]),
            source_offset,
            total_len,
            row_length_texels: 0,
            pages: None,
            direct_image: None,
        }
    }

    /// R9q: four attributes over two zero-copy guest binds are two canonical
    /// streams, exactly as four over two staged tables are.
    ///
    /// The census v7 shape, in the arm the boot actually resolved: `attrs=4`
    /// with two bound vertex buffers, every byte of them arriving as a
    /// `GuestRuns` bind. R9p's key answered `_ => false` for those, so the
    /// count stayed 4 and the `[[buffer(2)]]` argument stayed inside the
    /// streams' block.
    #[test]
    fn attributes_of_one_guest_bind_are_one_canonical_stream() {
        let first = guest_bind(0, 48);
        let second = guest_bind(0, 48);
        let interleaved = vec![
            guest_attribute(0, 0, 16, &first),
            guest_attribute(1, 8, 16, &first),
            guest_attribute(2, 0, 16, &second),
            guest_attribute(3, 8, 16, &second),
        ];
        assert_eq!(
            canonical_vertex_stream_count(&interleaved),
            2,
            "two guest binds, four attributes"
        );
    }

    /// The window is half of the guest bind's identity, not decoration: two
    /// attributes that share one `runs` allocation but bind different windows
    /// inside it — what a packed resource's per-offset binds look like
    /// (`held_buffer_content` slices one retained alias) — are two tables.
    #[test]
    fn two_windows_of_one_guest_allocation_stay_two_streams() {
        let whole = guest_bind(0, 64);
        let mut tail = whole.clone();
        tail.source_offset = 16;
        tail.total_len = 48;
        let attributes = vec![
            guest_attribute(0, 0, 16, &whole),
            guest_attribute(1, 0, 16, &tail),
        ];
        assert_eq!(canonical_vertex_stream_count(&attributes), 2);
    }

    /// The two origins never join: a staged copy and a zero-copy bind are two
    /// streams even when their bytes are the same bytes. The staged arm has no
    /// `runs` allocation to compare, and inventing one would be a byte
    /// comparison this rail does not perform.
    #[test]
    fn a_staged_table_and_a_guest_bind_stay_two_streams() {
        let staged = std::sync::Arc::new(vec![0u8; 48]);
        let guest = guest_bind(0, 48);
        let attributes = vec![
            attribute(0, 0, 16, &staged),
            guest_attribute(1, 0, 16, &guest),
        ];
        assert_eq!(canonical_vertex_stream_count(&attributes), 2);
    }

    /// The census v6 door's own shape, in the terms the seam hands this rail:
    /// four attributes at locations 0..3 and *two* staged tables — locations 0
    /// and 1 off the first, 2 and 3 off the second, which is what an
    /// interleaved pair of guest vertex buffers looks like once the runtime has
    /// resolved each attribute through its own buffer's bind.
    ///
    /// Two canonical streams, not four: the count the stage-buffer gate's
    /// vertex-layout rule reads is the request's own fetch tables, so a vertex
    /// stage argument at Metal index 2 is clear of them — where the attribute
    /// count the rule used to read (4) put it inside. The second half of the
    /// test is that control: the same four attributes with four allocations are
    /// four streams, which is the number the same draw had before this
    /// increment and the reason the door held 77103 of them (R9p).
    #[test]
    fn attributes_of_one_staged_table_are_one_canonical_stream() {
        let first = std::sync::Arc::new(vec![0u8; 48]);
        let second = std::sync::Arc::new(vec![0u8; 48]);
        let interleaved = vec![
            attribute(0, 0, 16, &first),
            attribute(1, 8, 16, &first),
            attribute(2, 0, 16, &second),
            attribute(3, 8, 16, &second),
        ];
        assert_eq!(
            canonical_vertex_stream_count(&interleaved),
            2,
            "two tables, four attributes"
        );

        let separate = vec![
            attribute(0, 0, 8, &first),
            attribute(1, 0, 8, &second),
            attribute(2, 0, 8, &std::sync::Arc::new(vec![0u8; 24])),
            attribute(3, 0, 8, &std::sync::Arc::new(vec![0u8; 24])),
        ];
        assert_eq!(
            canonical_vertex_stream_count(&separate),
            4,
            "one table per attribute is the attribute count"
        );
    }

    /// Sharing the staged allocation is not enough: a stream is the *fetch
    /// table*, so the stride is the other half of the identity. Two attributes
    /// that read one allocation at two strides are two tables, and a rail that
    /// folded them would fetch one of them at the wrong stride.
    #[test]
    fn one_allocation_at_two_strides_is_two_streams() {
        let table = std::sync::Arc::new(vec![0u8; 64]);
        let attributes = vec![attribute(0, 0, 16, &table), attribute(1, 0, 32, &table)];
        assert_eq!(canonical_vertex_stream_count(&attributes), 2);
    }

    /// Two attributes of one table *are* one stream even when the caller built
    /// them separately: only the staged allocation's identity joins them, which
    /// is the runtime's own sharing contract — a request that resolves one
    /// buffer twice hands this rail two allocations, and it then states two
    /// streams (which the gate answers, conservatively, as it always did).
    #[test]
    fn equal_bytes_from_two_allocations_stay_two_streams() {
        let bytes = vec![0u8; 32];
        let attributes = vec![
            attribute(0, 0, 16, &std::sync::Arc::new(bytes.clone())),
            attribute(1, 8, 16, &std::sync::Arc::new(bytes)),
        ];
        assert_eq!(canonical_vertex_stream_count(&attributes), 2);
    }

    /// R11: the index stream's label is a namespace of its own.
    ///
    /// [`plan_owner_leases`] looks a view up by label, and a lookup answered by
    /// another namespace's lease would be a *wrong frame* rather than a
    /// refusal — which is why the label sits beside the vertex stream's here:
    /// the index stream's `0x30000` has to miss both stages' namespaces
    /// (`0x0000` / `0x0001`) and every vertex stream's (`0x20000 | stream`),
    /// however many streams a request states.
    #[test]
    fn the_index_stream_label_misses_every_other_namespace() {
        let index = index_stream_owner_binding();
        for stream in 0..MAX_VERTEX_BUFFERS + 1 {
            assert_ne!(
                index,
                vertex_stream_owner_binding(stream),
                "the index stream's label must not name vertex stream {stream}'s view"
            );
        }
        for stage in [RenderPipelineStage::Vertex, RenderPipelineStage::Fragment] {
            for metal_index in 0..u32::try_from(MAX_RENDER_STAGE_BUFFERS).unwrap_or(u32::MAX) {
                assert_ne!(
                    index,
                    stage_buffer_owner_binding(stage, metal_index),
                    "the index stream's label must not name {}'s [[buffer({metal_index})]]",
                    stage.name(),
                );
            }
        }
    }
}
