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
//! - **one draw**, indexed or not, `instance_count == 1`, `base_vertex == 0`,
//!   `first_vertex == 0`, triangle list, single-sample. The canonical contract
//!   states one arm per shape — [`RenderPassDescriptor::indices`] is an
//!   `Option`, and `None` reads as `0..vertices` — and both rails execute both
//!   arms (Vulkan's `DrawShape::Vertices` beside `DrawShape::Indexed`, the
//!   native rail's own `plan_vertex_input` arm), so a draw that names its
//!   vertices directly is in class beside the one that indexes them (R39,
//!   [`nonindexed_vertex_span`] is the span proof the second arm owes);
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
//! - **up to four vertex streams** and one index stream on the indexed arm —
//!   none on the arm that names its vertices directly (R39) — one canonical
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
//!   across. The declared layout may name attribute locations the vertex stage
//!   never reads (R-VI1): the surplus entries are bound with their streams and
//!   ignored, exactly as Metal's own descriptor binds and ignores them, while a
//!   location the stage *reads* that no declared entry covers keeps the draw on
//!   the engine by name. A surplus entry's own stream also has to cover the
//!   draw's own span (`render_provider_out_of_class_vertex_span`), because
//!   admission proves the surplus streams too and a declaration the stage never
//!   reads may not be the reason a draw the class took is skipped;
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
//!   window since E-TX1/§107, **and, while the device's own capability frame
//!   lists them, the two narrow lanes E's `render-sampled-narrow-lanes`
//!   appended to that same list** — `r8_unorm` and `r8g8_unorm`, R39, whose
//!   texels the canonical rail uploads at their own width and samples as the
//!   lane's own channels name: `(r,0,0,1)` and `(r,g,0,1)`, which is the texel
//!   the format declares rather than a four-channel summary of its bytes — at
//!   the render area's own extent unless the device's frame declares the gather
//!   below, R37) — beside the draw's own bind, and requires the bind's sampler
//!   state to repeat the module's. Every other shape is a named exit
//!   ([`sampled_textures`]): a resource family the
//!   translated rail does not execute, a texture index at or above the
//!   contract's own bound or a list that repeats an index or walks backwards, a
//!   reflected shape or state outside the family, an unbound declaration, a
//!   bind whose texels are a resident image, a texture of another extent, a lane
//!   the frame does not list — the fail-closed half of R39, which reads the
//!   sentence this class shipped before the increment, verbatim — and a draw
//!   whose sampler says another state;
//! - **a sampled texture whose texels are the guest's own pages** (R28): the
//!   bind a real boot resolves through the zero-copy rail
//!   (`SampledSource::GuestRuns`) leaves for the provider through the owner
//!   rail's *window* arm, the sampled sibling of the vertex and index streams'
//!   R9q/R11 — one page run whose registered window covers the texture's
//!   tightly packed extent ([`sampled_gather_window`]), stated as the borrowed
//!   no-copy window when the texture starts at the reservation's own first byte
//!   and as the owner's staged copy of exactly the extent when it does not
//!   ([`texture_window_arm`]). A gather whose *rows are padded* (`R36`) is the
//!   one shape the two window arms cannot state — a lease names a tightly
//!   packed extent, and the reservation holds the guest's rows with their
//!   padding — so the class repacks the texture's own extent out of the
//!   registration ([`PaddedRows::depad`], [`RowCopies`]) and states it as the
//!   trace's own bytes (`TextureSource::OwnedBytes`), the arm the request's own
//!   copy and R24's frame already take; a padded stride narrower than the
//!   texture's own row and a padded span the stated row count cannot tile are
//!   refusals by name. Everything else — a gather scattered over
//!   stretches, one no registration covers, a window that does not
//!   reach the span — keeps the draw on the engine under
//!   `..._texture_source`, and a pass whose such bind crosses the
//!   owner→provider frame also needs the frame to carry its declarations, which
//!   the class reads out of the frame's own capability answer
//!   (`..._texture_wire`);
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
//!   (`..._texture_source_shape`) all stay on the engine. A sampled pass whose
//!   binds cross the owner→provider frame stays on the engine exactly while the
//!   frame's own capability answer does not carry the declarations
//!   (`..._texture_wire`, which E-TX4's wire retires for the shapes it
//!   carries);
//!
//!   # The frame the registry holds, carried in (R24)
//!
//!   Census v18 (`evidence/gate3-census-v18-2026-09-18`) read R22's Target arm
//!   split into two named refusals — `..._texture_source_undeclared` (893
//!   records) and `..._texture_source_order` (618) — and they are two different
//!   questions. A target no pass of this rail declared a production for is one
//!   whose bytes *exist*: the engine's registry holds them (`SampledSource::
//!   Target` is a resident bind), and the caller can read them out with the
//!   same `read_target` R23's arm reads a chain frame with. The caller hands
//!   that frame over ([`RenderRailInputs::sampled_target_frames`]) and the
//!   declaration states the request's own copy (`TextureSource::OwnedBytes`),
//!   exactly as every pre-R22 sampled texture does — so the arm is a transfer
//!   of the *same* bytes, not a second reading of them. A frame the caller
//!   never read, and a frame that is not the declaration's own extent, keep
//!   the record on the engine under their own names
//!   (`..._texture_source_undeclared`, `..._texture_source_frame_shape`).
//!
//!   The self-sampling record is *not* that question. Its read is the live
//!   attachment — the arm the engine itself takes on a device with
//!   `VK_EXT_attachment_feedback_loop_layout` (`sampled_self_feedback_loop` in
//!   the census) — and every arm the canonical contract can state resolves
//!   *before the pass opens*: a trace-produced view is an earlier pass's landed
//!   store (`render_texture_source_order_unsupported` refuses the other order),
//!   and a declaration that names the attachment's own view is
//!   `RenderTextureAttachmentConflict`, "anything but a race". Answering that
//!   read with pre-pass bytes would state the engine's *fallback* (its snapshot
//!   arm), which differs from its answer wherever the shader reads a texel this
//!   pass has already written, so the shape keeps its own name and its frame is
//!   never read.
//!   Frozen boundary (census v29: 550 records; engine arm: sampled_self_feedback_loop). Reopen only with a native oracle.
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
//!   name (measured in `provider_render_rail.rs`). A pair whose two stages read
//!   one index is the fold R31 named and R33 retires for the devices that
//!   declare the split (see below);
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
//!   # The surface's own resident, carried in (R26)
//!
//!   Census v19 (`evidence/gate3-census-v19-2026-09-18`, same rig) left
//!   `resident_source` at 919 records (26.9 %), and 47.8 % of them are one
//!   group the two rounds above do not touch: `fmt=0x50`, `load=Load`,
//!   `door=mapping`, `skip=resident`, `store=1`, **`seed=none`**, `wb=1`,
//!   `continues=0`, `pass_cont=0` — the mapper-ref-texture composite whose
//!   previous contents the *engine's* registry already holds under the
//!   surface's own identity. Its LOAD was elided by the engine
//!   (`runtime::draw::vulkan`'s `mapper_ref_texture_load_currency_query`),
//!   which is a *different naming of the same fact* R23's caller hands over:
//!   the frame is readable by the caller that owns the registry
//!   (`read_target`), and the class already carries the trace-owned load for
//!   it. So this door gets its own input
//!   ([`RenderRailInputs::surface_resident_source_bytes`]) rather than
//!   borrowing R23's: the caller states a *different* obligation with it —
//!   the landing that consumes this record's own frame must be one that
//!   advances the surface's `surface_content_epoch`, so that the resident's
//!   older stamp can no longer vouch for pixels the provider never wrote
//!   (`mapper_ref_texture_load_resident_is_current` reads exactly that pair).
//!   The two elisions this door does **not** carry keep their refusal by
//!   name: the attachment's own GVA resident
//!   (`honour_gva_load_elision`) is witnessed by `resident_content_ready` on
//!   that identity and no in-tree API can un-ready a render resident, and a
//!   mapper-ref-texture record *without* a guest writeback has no landing that
//!   moves the epoch at all.
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
//! R32 puts the attachment's own load seed in the same channel. The two shapes
//! that carry one are the two the request's seed door resolves: the surface's
//! own guest pages, stated as the contract's ordered run list
//! (`BufferSource::GuestRuns`, `research/docs/23` §113 / E-TX6) under
//! [`load_seed_owner_binding`] — a list arm rather than a view arm, because
//! what the list names is one lease per registration and many runs inside it —
//! and the seed door's own bytes, which stay trace-owned
//! (`BufferSource::OwnedBytes`) exactly as R23/R25's hand-overs do. Everything
//! the class cannot state keeps the engine by name: padded rows, a list that is
//! not the attachment's extent, one split across two registrations, and a seed
//! the caller does not hand over.
//!
//! # The two stages' buffer namespaces, and the fold (R31, R33)
//!
//! The two stages' Metal buffer index spaces are independent — a
//! `setVertexBuffer(_:offset:index:)` at one stage and a
//! `setFragmentBuffer(_:offset:index:)` at the other are two arguments — while
//! `metal2vulkan`'s default descriptor layout binds either stage's
//! `[[buffer(n)]]` at set 0's own binding `n`. A pair that reads one index from
//! both stages therefore folds both declarations onto one descriptor, and the
//! provider refuses such a pair when the merged set is built
//! (`render_stage_buffer_layout_unsupported`). That refusal is a *decline* on
//! an admitted draw, which is fail-closed, so R31 answered the shape here
//! first, by name (`render_provider_out_of_class_stage_buffer_shape`, route
//! `stage_buffer_shape_folded`) and the engine drew it. It was the largest
//! single bucket of census v23c: 12,262 rows, 42.8% of that boot's seam
//! failures.
//!
//! R33 retires the exit for the devices that can execute the shape. E-TX9
//! (`metal-api-emulator`'s `stage-buffer-namespace-split`) publishes the
//! arrangement that stops the fold — the *vertex* stage's whole layout in the
//! canonical namespace set, `metal_api_vulkan::stage_buffer_namespace_layout()`,
//! with the fragment half left at the translator's default — and declares it as
//! the shape bit
//! `ProviderCapabilities::supports_render_stage_buffer_namespace_split`, which
//! defaults to `false` and decodes as `false` out of any frame that does not
//! carry it. This rail reads that bit out of the provider's own capability
//! frame ([`provider_wire::stage_buffer_namespace_split`]), asks it exactly when
//! the request's statement makes the pair ([`folded_stage_buffer_pair`]), and
//! translates the vertex half under that layout when the answer is yes. An
//! answer of no — every device that arranges nothing apart, and every frame
//! written before the bit existed — keeps R31's refusal, its sentence and its
//! own census route, byte for byte, at the same point in the same walk.
//!
//! # The sampled source of another extent (R35, R37, R40)
//!
//! One extent rule answered this shape until R37: a draw whose sampled bind is
//! not the pass's own extent stayed on the engine by name
//! (`render_provider_out_of_class_texture_extent`), because the two rails E
//! ships answer the shape differently and no capability bit told a frame's
//! reader which of the two answers produced it. R35 split the *population* of
//! that refusal into the two arms E's question turns on — `texture_extent_host_bytes`
//! for every source a rail reads off the host, `texture_extent_borrowed_no_copy`
//! for the owner's no-copy window — so the next census could size them instead
//! of the bare slug: v26 read 5 679 host-bytes rows (93.2 %) beside 416
//! borrowed ones (6.8 %).
//!
//! R37 retires the exit for the arm the Vulkan rail already executes. E-TX10
//! (`metal-api-emulator`'s `render-texture-gathered-extent`) publishes the
//! difference as the shape bit
//! `ProviderCapabilities::supports_render_texture_gathered_extent` — `true` on
//! the Vulkan snapshot, `false` on the native one and on any frame written
//! before the bit existed, which decodes as `false` out of the escape family
//! E-TX9 opened. This rail reads the bit out of the provider's own capability
//! frame ([`provider_wire::render_texture_gathered_extent`]), asks it exactly
//! when the request names a sampled bind of another extent
//! ([`sampled_source_of_another_extent`]), and hands the host-bytes arm of the
//! rule to the provider when the answer is yes. The arm is read where the rule
//! is ([`texture_extent_arm`]), because one request may state both arms: the
//! owner's no-copy window of another extent was the arm E-TX10's bit did *not*
//! cover — its own sentence says the host gather's bit must not be read as "the
//! snapshot executes the owner's no-copy window" — so it kept the refusal,
//! under its own route, until R40.
//!
//! Nothing about the admitted shape changes beside the verdict: the
//! declaration already states the bind's own extent (the extent the module's
//! own sample sites were lowered against, and the extent the Vulkan rail binds
//! by), no copy is made, and the pass's trace is the trace it was. An answer of
//! no — the native rail, and every frame written before the bit existed — keeps
//! R35's refusal, sentence, route and gate position, byte for byte.
//!
//! R40 retires the exit for the arm E-TX12 opened, and it is read out of its
//! *own* bit for that same reason. The owner's no-copy window of another extent
//! is the one source R35's split names as having no bytes a rail can gather, and
//! E-TX12 answers it with code of its own rather than with the host gather: a
//! *translated* fragment stage — the registration path this fork takes, whose
//! module states its own sample coordinates — binds the window at the source's
//! own extent exactly as it binds the trace's own bytes, while the reviewed
//! sampling pair's *gathered* sibling reads the window in place on the device,
//! at the destination grid's own integer index
//! (`metal-api-vulkan`'s `gathered_fetch.frag.spv`, `research/docs/23` §111).
//! `ProviderCapabilities::supports_render_texture_gathered_extent_no_copy` is
//! that answer, written into the capability tail as the escape family's fourth
//! tag (`0x00 0x04 <bool>`, after E-TX11's `0x00 0x03 <bool>`), and this rail
//! reads it out of the provider's own frame
//! ([`provider_wire::render_texture_gathered_extent_no_copy`]) under the same
//! shape test as E-TX10's bit. Each arm of the rule is then weighed against the
//! declaration of *that* arm: a device that declares one and not the other
//! keeps the other on the engine, and a frame written before either bit existed
//! keeps both, under R35's own slug, sentence, route and gate position.
//!
//! What R40 does *not* add is a copy. The arm's whole statement is that the
//! device reads the owner's mapping: no host bytes are gathered, the bind's
//! declaration is the same `TextureSource::BorrowedNoCopy` window it was, and
//! the frame the provider lands is the frame this project's oracle states for
//! the source's own extent.
//!
//! # The vertex interface, and the declared superset (R-VI1)
//!
//! One rule answered every disagreement between a request's declared vertex
//! attributes and the vertex stage's own reflection until this increment:
//! `render_provider_out_of_class_vertex_interface`. The two sides were compared
//! field by field, because the canonical registration gate
//! (`metal-api-vulkan`'s `validate_translated_vertex_attributes`) refuses a
//! mismatch, and a refusal there is a decline rather than a fallback. The rule
//! therefore had to answer it here first — but the two directions it answered
//! are priced completely differently:
//!
//! * a **declared** superset — `declared ⊋ reflected`, every location the
//!   vertex stage reads covered by a declared entry, and at least one declared
//!   entry naming a location it does not read — is a shape Metal answers. A
//!   `MTLVertexDescriptor` may name a location the vertex function never reads,
//!   and the surplus stream is *bound and ignored*. The canonical Vulkan rail's
//!   own vertex input state is built from the declared layout
//!   (`create_pipeline` emits one `VkVertexInputAttributeDescription` per
//!   declared attribute), so it executes the shape exactly as Metal does;
//! * a **reflected** superset — a location the vertex stage *reads* that no
//!   declared entry covers — is a value Metal defines nowhere, and a location
//!   mismatch is neither direction. No device can admit either, so both keep
//!   their refusal on every per-snapshot answer.
//!
//! R-VI1 split the bucket's *population* by direction, so the size of the
//! winnable half could be read before anything was widened: census v27b
//! (`evidence/gate3-census-v27b-2026-09-18`) counted 5 241 rows of
//! `vertex_interface_declared_superset` against zero of
//! `vertex_interface_reflected_superset` and zero of
//! `vertex_interface_location_mismatch` — 100 % of the bucket, and the largest
//! single bucket of that boot's seam failures (50.2 % of every refused draw).
//!
//! E-TX11 (`metal-api-emulator`'s `render-vertex-interface-superset`) makes the
//! one rule directional and publishes the answer as the shape bit
//! `ProviderCapabilities::supports_render_vertex_interface_superset` — `true`
//! on the Vulkan snapshot, `false` on the native one and on any frame written
//! before the bit existed, which decodes as `false` out of the escape family
//! E-TX9 opened. This rail reads the bit out of the provider's own capability
//! frame ([`provider_wire::render_vertex_interface_superset`]), asks it exactly
//! when the request's own walk names the declared-superset direction
//! ([`vertex_interface_declared_superset`]), and admits that arm — and only
//! that arm — when the answer is yes.
//!
//! Nothing about the admitted shape changes beside the verdict: the request's
//! whole declared list already travels to the provider, the surplus entry keeps
//! the stream grouping, stride, offset and format the descriptor declared, no
//! copy is made, and the pass's trace is the trace it was. An answer of no — the
//! native rail, and every frame written before the bit existed — keeps R-VI1's
//! refusal, sentence, route and gate position, byte for byte.
//!
//! The one thing the widening adds beside the bit is the span gate
//! (`render_provider_out_of_class_vertex_span`): a declared attribute the vertex
//! stage never reads may not be able to *refuse* the draw it is ignored by.
//! Admission proves every declared stream against the draw's own span — the
//! highest vertex the index stream names, over that stream's length divided by
//! its stride (`render_vertex_buffer_footprint_unsupported`) — and it proves the
//! surplus streams too, because they are bound with the rest of the layout. A
//! refusal there is a decline, and a decline on an in-class draw is fail-closed:
//! the draw is *skipped* rather than run on the engine. The same proof over a
//! stream the module *does* read is the gap this class has always had for the
//! shapes it has always carried, and this increment does not move it — closing
//! it would answer shapes the widening never touches. The number needs the index
//! *values*, so the gate runs on the staged index arm alone: a zero-copy index
//! window (`R11`) is guest RAM this process has not read.
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
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
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
    VertexStep, ViewId, FULL_SCREEN_TRIANGLE_VERTICES, MAX_RENDER_SAMPLERS,
    MAX_RENDER_STAGE_BUFFERS, MAX_RENDER_TEXTURES, MAX_RENDER_TEXTURE_INDEX, MAX_SERIAL_RESOURCES,
    MAX_VERTEX_BUFFERS, PROVIDER_SCHEMA_VERSION,
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

/// The sampled-texture half of the window-backed shapes (`R28`).
///
/// The fourth shape of the same rail, and the one whose second device answer
/// is *not* an alignment: a lease window names a texture's extent at the
/// reservation's own start, so what the device's granules turn away here is not
/// an off-granule view pointer (E reads the reservation's own base) but the
/// texture's first byte sitting anywhere but that base — the copy is the answer
/// and this is its name. The import answer is the stream arms' own
/// ([`window_binding_admits`]): a device that cannot import host pointers has
/// no window arm at all, and the class refuses rather than silently copying.
const TEXTURE_WINDOW: WindowShape = WindowShape {
    name: "sampled texture",
    import_slug: "render_provider_out_of_class_texture_import",
    alignment_slug: "render_provider_out_of_class_texture_window",
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

/// The arm one sampled texture's registered window takes on this device (`R28`).
///
/// The sampled sibling of [`window_arm`], and deliberately not the same
/// function: a lease window names a *texture's* bytes as its own tightly packed
/// extent at the reservation's start, so the second answer here is not "is the
/// view pointer a whole number of the device's granules" but "is the texture's
/// first byte the reservation's own first byte". `head` is that distance
/// ([`sampled_gather_window`]); zero means the borrowed arm states exactly the
/// texture's bytes, and non-zero means the class copies the extent out of the
/// registration ([`provider_owner::window_bytes`]) and states it as the owner's
/// staged lease — the same two arms a buffer view takes, decided by the one
/// fact a texture's window rule adds.
///
/// `reservation_starts_at_window` is the second condition, and it is about the
/// *submission* rather than the bind: one registration's windows share one
/// lease whose reservation covers their union
/// ([`provider_owner::plan`]), and E reads a texture at the *reservation's*
/// start — so a texture whose window is not the earliest window of its own
/// registration in this pass would be read from another bind's bytes. That
/// shape is the copy arm's, not the borrowed one's.
///
/// `Ok(None)` is the borrowed arm, `Ok(Some(bytes))` the staged copy, and `Err`
/// the two answers this rail cannot state: a device that cannot import host
/// pointers has no window arm at all ([`window_binding_admits`]'s own refusal,
/// named for the sampled shape), and a window the owner rail cannot read out of
/// its registration keeps the draw on the engine under its own name — a
/// refusal, because a decline is not a fallback and there is no copy to state.
fn texture_window_arm(
    binding: u32,
    window: StageBufferWindow,
    alignment: u64,
    reservation_starts_at_window: bool,
) -> Result<Option<Vec<u8>>, OutOfClass> {
    window_binding_admits(TEXTURE_WINDOW, window, alignment)?;
    if window.head == 0 && reservation_starts_at_window {
        return Ok(None);
    }
    match provider_owner::window_bytes(owner_window(binding, window)) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(decline) => Err(OutOfClass::owned(
            TEXTURE_WINDOW.alignment_slug,
            format!(
                "a draw whose sampled texture is covered by a registered guest RAM window stays \
                 on the engine when the window's bytes cannot be copied out of the registration \
                 that names it: the texture's extent starts {} byte(s) into the window (or that \
                 window is not the first of its registration this pass binds), so the \
                 canonical rail's no-copy arm — which reads a texture at the reservation's own \
                 start — would name bytes another bind owns, and the owner rail refused the copy \
                 this class states instead (`{}`)",
                window.head,
                decline.slug(),
            ),
        )),
    }
}

/// The bytes one pass copies out of its window-backed binds (`R18`, `R28`),
/// under the owner binding label the plan looks each view up by.
///
/// The copy itself is made by the class gate, where the device's import
/// alignment is read — before the pass is submitted and before any lease exists
/// — and this is what carries it to [`plan_owner_leases`], which states those
/// binds through the owner's staged arm instead of the borrowed one. Keyed by
/// the owner label rather than by position because the shapes' namespaces
/// overlap (`stage_buffer_owner_binding`'s two stages, a stream's index, the
/// index stream's constant, and since R28 a sampled texture's Metal index), and
/// a copy read back under another namespace's label would be a wrong frame
/// rather than a refusal.
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

/// The bytes one pass repacked out of a padded-row gather (`R36`), under the
/// Metal `[[texture(n)]]` index the declaration is keyed by.
///
/// The sibling of [`WindowCopies`], with the same division of labour and for
/// the same reason: the copy is made in the class gate, where the bytes behind
/// a registered window are readable and where the device answers live, and
/// [`submit_narrow`] states them. It is a second carrier rather than a second
/// use of the first because the bytes are not a window's own range — they are
/// the texture's tightly packed extent, assembled one row at a time — and
/// because nothing about them is imported by the owner plan: a padded gather's
/// declaration is the trace's own bytes (`TextureSource::OwnedBytes`), so no
/// lease is minted and the owner binding label a [`WindowCopies`] entry travels
/// under has no reader.
///
/// The key is the texture's own Metal index — the fact the contract pairs a
/// declaration and a view by — rather than the entry's position, exactly as
/// [`NarrowTexture::index`] is.
#[derive(Default)]
struct RowCopies {
    entries: Vec<(u32, Vec<u8>)>,
}

impl RowCopies {
    fn insert(&mut self, index: u32, bytes: Vec<u8>) {
        self.entries.push((index, bytes));
    }

    /// The repacked bytes for one sampled texture, when this pass padded one.
    fn bytes(&self, index: u32) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|(label, _)| *label == index)
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
/// the registration by name), and how many AIR static samplers it carries
/// (R37) — the canonical rail pairs those positionally, one per sampled texture
/// that reads through one, while the runtime half pairs by the index a
/// declaration names. One stage may carry both forms; each texture is declared
/// in the form the module's own sample sites name, and a module whose sites name
/// *two* samplers for one image — which no per-texture declaration can state —
/// stays on the engine by name.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RenderSamplerFamily {
    pub runtime: Arc<[RenderRuntimeSampler]>,
    /// The Metal index of every `ResourceKind::StaticSampler` binding, in the
    /// reflection's own order — the order the canonical rail's positional
    /// pairing counts against.
    pub statics: Arc<[u32]>,
    /// What the module's own sample sites state (`R37`): the pairing the
    /// runtime half of every declaration is read off, one texture at a time.
    pub sample_sites: RenderSampleSites,
}

/// What one module's sample sites state about the sampler beside each of its
/// textures (`R37`, `research/docs/23` §102).
///
/// The class gate's per-texture declarations are read off this walk
/// (`runtime::spirv_bind::sampled_image_pairs`), so the one answer that leaves
/// a texture with no single form is a stage-level fact of its own:
/// [`Self::MultipleSamplers`] is a module whose own sites name two samplers for
/// one image, which no declaration can state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RenderSampleSites {
    /// Every sampled image reads through one sampler — or the module samples
    /// nothing at all, which the empty pairing states the same way.
    #[default]
    Paired,
    /// One image's sample sites name more than one sampler: the shape that has
    /// no per-texture declaration.
    MultipleSamplers,
    /// A sample site's operand is not a descriptor the walk can follow, so the
    /// pairing is not one this walk names. The declarations fall back to the
    /// positional static rule, and a stage that also binds runtime
    /// `[[sampler(n)]]` arguments stays on the engine through the unpaired
    /// half of the class.
    Unresolved,
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
    // *sampled* textures that read through one by position — a texel-fetched
    // texture states no sampler at all (R15) and a runtime-sampled one states
    // the `[[sampler(n)]]` argument its own sample sites name (R37), so neither
    // takes a position in that pairing — and so does the loop below.
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
            // The runtime `[[sampler(n)]]` argument the module's own sample
            // sites pair this texture with, as `(metal index, device slot)`.
            // Read *before* the static half, because the module's own pairing
            // is the one statement that says which form this texture reads
            // through (R37): a texture whose sites name a runtime argument is
            // declared runtime however the AIR static samplers sit beside it,
            // and it takes no position in the positional static pairing —
            // which is the rule the canonical rail's own walk applies (its
            // static counter advances only for the textures their declaration
            // states static, `metal-api-vulkan/src/render.rs`).
            let runtime_pair = pairing.sampler_of(binding).and_then(|slot| {
                runtime_samplers
                    .iter()
                    .find(|(_, runtime_slot)| *runtime_slot == slot)
                    .copied()
            });
            // The static half's positional pairing: the AIR static samplers, in
            // the reflection's own order, against the sampled textures that do
            // not read through a runtime argument — a texel-fetched texture
            // states no sampler at all, so it takes no position in that pairing
            // either (R15).
            let paired = if fetched || runtime_pair.is_some() {
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
            // The sampler-free declaration states no slot: the canonical
            // contract's fetched arm carries no sampler, so a number here
            // would be a binding nothing reads.
            let (sampler_binding, sampler) = if fetched {
                (0, RenderSamplerState::Fetched)
            } else if let Some((index, slot)) = runtime_pair {
                // The runtime half (R12): the pair the module's own sample
                // sites name, which the declaration states and the pass fills
                // with the request's own state.
                (slot, RenderSamplerState::Runtime { index })
            } else {
                match paired.and_then(|sampler| sampler.static_sampler.as_ref()) {
                    Some(state) => (
                        static_slot,
                        match air_sampler_policy(state) {
                            Some(policy) => RenderSamplerState::Policy(policy),
                            None => RenderSamplerState::Unsupported(RenderSamplerRefusal::AirState),
                        },
                    ),
                    // No decoded AIR state beside this texture: the AIR static
                    // sampler carries none the translator could read, which the
                    // provider refuses by name.
                    None if paired.is_some() => (
                        static_slot,
                        RenderSamplerState::Unsupported(RenderSamplerRefusal::AirState),
                    ),
                    // No AIR static sampler left for this texture and no
                    // runtime argument its sites name: the module samples
                    // through a form neither half of the pairing states.
                    None => (
                        0,
                        RenderSamplerState::Unsupported(RenderSamplerRefusal::SampleSite),
                    ),
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
/// by name — and how many AIR static samplers it carries, because the canonical
/// rail pairs those positionally (one per sampled texture that reads through
/// one) while the runtime half pairs by the index a declaration names. A stage
/// may carry both forms at once (R37); the pairing the module's own sample sites
/// state is what the per-texture declarations are read off, and its one
/// unstateable answer keeps the stage on the engine by name.
#[cfg(feature = "provider-render")]
pub fn sampler_family(
    fragment: &metal2vulkan::reflect::ShaderReflection,
    words: &[u32],
) -> RenderSamplerFamily {
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
        // The pairing the per-texture declarations are read off (`R12`), in
        // its own three answers (R37): the class keeps the stage on the engine
        // when it is not one sampler per image.
        sample_sites: match crate::runtime::spirv_bind::sampled_image_pairs(words) {
            crate::runtime::spirv_bind::SamplePairing::Paired(_) => RenderSampleSites::Paired,
            crate::runtime::spirv_bind::SamplePairing::MultipleSamplers => {
                RenderSampleSites::MultipleSamplers
            }
            crate::runtime::spirv_bind::SamplePairing::Unresolved => RenderSampleSites::Unresolved,
        },
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

/// The texel lanes the bind sentence names as the provider's window (R39).
///
/// The two four-byte orders are the window this class has stated since E-TX1
/// and are named whatever the frame says: a device that omitted one of them is
/// answered by R28's own wire reading when the pass crosses the frame, not by a
/// second spelling of that rule here, so that half of the sentence keeps the
/// reading it always had. The two narrow lanes are named exactly while the
/// device's own frame lists them — and with neither lane stated the string
/// below is the sentence this class shipped, byte for byte, which is what keeps
/// a pre-increment device's refusal (and the census's reading of it) unchanged.
///
/// One phrase per shape of the answer rather than a list assembled at run time:
/// the sentence is the census's own key for this door, so it is written out.
fn sampled_bind_window(lanes: NarrowLanes) -> &'static str {
    match (lanes.r8, lanes.rg8) {
        (false, false) => {
            "`rgba8_unorm` or `bgra8_unorm` texels (the provider's whole `RENDER_SAMPLED` window)"
        }
        (true, false) => {
            "`rgba8_unorm`, `bgra8_unorm` or `r8_unorm` texels (the format lanes the provider's \
             own frame lists)"
        }
        (false, true) => {
            "`rgba8_unorm`, `bgra8_unorm` or `r8g8_unorm` texels (the format lanes the provider's \
             own frame lists)"
        }
        (true, true) => {
            "`rgba8_unorm`, `bgra8_unorm`, `r8_unorm` or `r8g8_unorm` texels (the provider's whole \
             `RENDER_SAMPLED` window)"
        }
    }
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
///    own names: a stage whose own sample sites name more than one sampler for
///    one texture, which no per-texture declaration can state (R37,
///    `..._texture_sampler_family`), more runtime arguments than the Metal
///    sampler table holds (`..._texture_sampler_count`), a runtime argument no
///    texture reads through (`..._texture_sampler_unpaired`), and a declaration
///    whose index, device slot or module the reflection does not back
///    (`..._texture_sampler_mismatch`).
/// 2. **What the draw bound.** Declaration `i` pairs with the request's own
///    bind at the device binding the runtime resolved it at; a declaration
///    without one is `..._texture_unbound`, a bind whose shape the pass cannot
///    state (dimensionality, layers, descriptor count, a texel outside the
///    lanes the device's own frame lists — the two 8-bit byte orders
///    `rgba8_unorm`/`bgra8_unorm`, and since R39 the one- and two-byte UNORM
///    lanes beside them — a view swizzle) is
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
    // The inputs' *inner* lifetime, not only the borrow of them: R24's arm hands
    // the caller's frame to the trace the same way the request's own bytes
    // travel (`research/docs/26` §48), so the slice has to live as long as the
    // consuming pass the caller builds from this answer.
    inputs: &RenderRailInputs<'a>,
    req: &'a DrawRequest,
    // The device's own answer to the extent rule's one question (R37), read by
    // [`submit_render`] out of the provider's capability frame exactly when the
    // request names a sampled bind of another extent
    // ([`sampled_source_of_another_extent`]) — the gate below is pure and reads
    // no device answer of its own.
    render_texture_gathered_extent: bool,
    // The device's own answer to the extent rule's *other* question (R40), read
    // by [`submit_render`] out of the same frame under the same condition — the
    // two arms E answers with different code state themselves separately, so
    // the walk below weighs each arm against its own bit.
    render_texture_gathered_extent_no_copy: bool,
    // The device's own answer to the narrow-lane rule's one question (R39),
    // read by [`submit_render`] out of the same frame exactly when the request
    // names a sampled bind whose view format is one of the two lanes
    // ([`sampled_bind_of_a_narrow_lane`]) — the gate below is pure and reads no
    // device answer of its own.
    render_texture_narrow_lanes: NarrowLanes,
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
    // The stage's sampler family (R12, `research/docs/23` §102; R37), before
    // any texture is weighed against its bind: more runtime `[[sampler(n)]]`
    // arguments than Metal's own sampler table holds is a shape the canonical
    // rail's registration refuses by name, so the class answers it here
    // instead of letting the provider decline a draw the engine could run.
    //
    // The two forms in one stage are *not* that shape (R37). The canonical
    // side falsified the rule this gate used to state — a stage carrying one
    // AIR static sampler beside one runtime `[[sampler(n)]]` registers,
    // submits and executes there, with each half reading its own way
    // (`metal-api-emulator`'s `render_sampler_family_e2e.rs`, E `41308a1`) —
    // so the stage is handed to the provider and the per-texture declarations
    // below state the form each texture's own sample sites name.
    let family = inputs.sampler_family;
    // What stays out of class by name (R37): a stage whose own sample sites
    // name *two* samplers for one image. A declaration names exactly one form
    // per texture, so such a stage has no per-texture statement this class can
    // make — and a walk that fell back to the positional pairing would declare
    // the texture through a sampler state its own sample sites do not name,
    // which the canonical rail then refuses at registration, leaving a draw
    // the engine could run declined rather than drawn. The pairing walk's own
    // answer is the fact, so the class reads it rather than re-deriving it.
    if matches!(family.sample_sites, RenderSampleSites::MultipleSamplers) {
        return Err(OutOfClass::owned(
            "render_provider_out_of_class_texture_sampler_family",
            "a fragment stage whose own sample sites name more than one sampler for one \
             texture stays on the engine: a declaration names exactly one sampler form per \
             texture — the AIR static state the module carries, or the runtime \
             `[[sampler(n)]]` argument the texture reads through — so a texture the module \
             samples two ways is not a per-texture statement this class can make. The \
             canonical rail's registration looks for one sampler per texture and refuses the \
             stage by name when its AIR static samplers and the declarations disagree \
             (`render_stage_reflection_mismatch`) or when a runtime `[[sampler(n)]]` the \
             module binds is paired with nothing (`render_runtime_sampler_undeclared`) rather \
             than executing a sample through a state nothing declared"
                .to_owned(),
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
        // provider's window on this device is the two four-byte 8-bit UNORM
        // byte orders and — when the device's own frame lists them (R39, E's
        // `render-sampled-narrow-lanes`) — the one- and two-byte UNORM lanes
        // beside them, and the bytes travel verbatim into the image the *name*
        // selects, so the fragment stage reads the channels the guest's own
        // view states (`r8_unorm` reads `(r,0,0,1)`, `r8g8_unorm` reads
        // `(r,g,0,1)`, which is the texel each lane's own format declares and
        // not a four-channel summary of it). Any other format — the wide
        // half-float, the sRGB spellings of the two byte orders, the integer
        // lanes the census counts — is refused by the provider under
        // `render_texture_format_unsupported`, so this class answers it here
        // rather than handing the provider a draw the engine would have run.
        let format = match image.format {
            ash::vk::Format::R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
            ash::vk::Format::B8G8R8A8_UNORM => Some(TextureFormat::Bgra8Unorm),
            // The two lanes E appended to `RENDER_SAMPLED`, each admitted only
            // while the device's own frame lists it: the answer is asked once
            // per request and travels as a value, so a frame that does not
            // carry a lane leaves this arm's `None` exactly where it was.
            narrow => render_texture_narrow_lanes.admits(narrow),
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
                     with one descriptor, {} and an identity channel mapping, and the bind \
                     is {}x{} {:?} (kind {:?}, layers {}, descriptors {}, element {}, multisampled \
                     {}) — a texel outside that window is refused by the provider by name \
                     (`render_texture_format_unsupported`) rather than uploaded under another \
                     format",
                    declaration.index,
                    sampled_bind_window(render_texture_narrow_lanes),
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
                // The count comes from the format the bind states rather than
                // from a constant here — four bytes per texel for the two
                // original orders, one or two for the narrow lanes R39
                // admitted — so a widening of the provider's window has one
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
                // production *after* the read, and this record's read is the
                // live attachment: the engine states that read itself whenever
                // the device carries `VK_EXT_attachment_feedback_loop_layout`
                // (`sampled_self_feedback_loop` in the census), and every arm
                // the canonical contract can state resolves *before* the pass
                // opens — a trace-produced view is an earlier pass's landed
                // store (`render_texture_source_order_unsupported` is the
                // contract's name for the other order), and a declaration that
                // names the pass's own attachment view is refused as
                // `RenderTextureAttachmentConflict`, "anything but a race".
                // Serving the read from bytes a copy captured before the pass
                // would state the engine's *fallback* (its snapshot arm) rather
                // than its answer, so the shape is answered by name rather than
                // reordered — and its frame is not read at all (R24's caller
                // skips it), because nothing would use it.
                if req.writes_attachment(identity) {
                    return Err(OutOfClass::owned(
                        "render_provider_out_of_class_texture_source_order",
                        format!(
                            "a draw whose `[[texture({})]]` samples the very attachment it \
                             renders into stays on the engine: the read this record states is the \
                             live frame it is writing, while every arm this class can declare \
                             resolves before the pass opens — the trace's own earlier production \
                             of that identity would put its store after the read, which the \
                             contract refuses by name (`RenderTextureAttachmentConflict`), and a \
                             copy taken before the pass is the engine's fallback arm, not its \
                             answer",
                            declaration.index,
                        ),
                    ));
                }
                match recorded_production(identity) {
                    Some(production) => {
                        // The sampled declaration has to restate the stored
                        // surface's format and extent, exactly as the contract
                        // holds it (`RenderTextureSourceShapeMismatch`): the
                        // bytes the trace produces are the *stored* surface's,
                        // so a declaration that names another texel order or
                        // another extent would sample texels the production
                        // never wrote.
                        if production.format.as_texture_format() != format
                            || [production.width, production.height]
                                != [u64::from(image.width), u64::from(image.height)]
                        {
                            return Err(OutOfClass::owned(
                                "render_provider_out_of_class_texture_source_shape",
                                format!(
                                    "a draw whose `[[texture({})]]` is {}x{} {:?} stays on the \
                                     engine when the target's own production stored {:?} at {}x{}: \
                                     the canonical rail pairs a trace-produced declaration with \
                                     the store that defines its bytes field by field and refuses \
                                     a disagreement by name \
                                     (`render_texture_source_shape_mismatch`), and a declaration \
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
                    // R24: no pass of this rail ever stated that target's
                    // production, but the *bytes* are not the caller's to
                    // invent — the engine's registry holds them (`SampledSource::
                    // Target` is a resident bind), and the caller that owns that
                    // registry reads them out with the same `read_target` R23's
                    // arm reads a chain frame with. The declaration then states
                    // the request's own copy (`TextureSource::OwnedBytes`),
                    // which is the arm every pre-R22 sampled texture takes, so
                    // the two rails read one set of bytes through one view.
                    None => {
                        let Some(frame) = inputs.sampled_target_frame(identity) else {
                            return Err(OutOfClass::owned(
                                "render_provider_out_of_class_texture_source_undeclared",
                                format!(
                                    "a draw whose `[[texture({})]]` texels come from a GPU target \
                                     stays on the engine when this rail has no production to \
                                     restate for it and the caller hands no frame over: the \
                                     canonical rail samples either a trace-produced view \
                                     (`TextureSource::TraceView`, the pass that stored it restated \
                                     in the same trace) or a copy the trace carries, and a target \
                                     whose bytes neither of those names is one this class cannot \
                                     state — the resident and guest-gather arms are the increments \
                                     after this one",
                                    declaration.index,
                                ),
                            ));
                        };
                        // The frame is the target's own tightly packed extent
                        // read out of its image, and the declaration states the
                        // *bind's* view over exactly those bytes (E-TX1/§107:
                        // the byte order is the name's, not the memory's), so a
                        // length that is not that extent is a caller wiring bug
                        // and is refused under its own name rather than uploaded
                        // and refused by the contract — a decline is never a
                        // fallback.
                        let expected = u64::from(image.width)
                            .checked_mul(u64::from(image.height))
                            .and_then(|texels| texels.checked_mul(format.bytes_per_texel()));
                        if expected != u64::try_from(frame.len()).ok() {
                            return Err(OutOfClass::owned(
                                "render_provider_out_of_class_texture_source_frame_shape",
                                format!(
                                    "a draw whose `[[texture({})]]` view carries the caller's {} \
                                     byte(s) frame for its {}x{} {:?} surface stays on the engine: \
                                     the frame read out of the registry has to be the target's own \
                                     tightly packed extent ({} byte(s)), the length the declaration \
                                     states, or the two rails would read two different windows of \
                                     one image",
                                    declaration.index,
                                    frame.len(),
                                    image.width,
                                    image.height,
                                    image.format,
                                    u64::from(image.width)
                                        * u64::from(image.height)
                                        * format.bytes_per_texel(),
                                ),
                            ));
                        }
                        NarrowTextureSource::Frame(frame)
                    }
                }
            }
            // The zero-copy guest gather: the bytes are the guest's own pages,
            // read inside the draw's command buffer (R28): the texture sibling
            // of the arms R9q/R11 state for the vertex and index streams, on
            // the same fact — one page run whose registered window the
            // registration ledger derived ([`sampled_gather_window`]).
            //
            // The contract's arm for a lease-backed texture is narrower than a
            // buffer's: E's window rule names *the texture's tightly packed
            // extent at the reservation's own start*, so a gather the window
            // rail can state is one whose first byte is the window's own and
            // whose rows are tight. The declaration states the no-copy window
            // when the reservation starts there, and the owner's copy of
            // exactly those bytes (`StagedLease`) when it does not — both
            // decided where the device answer lives ([`texture_window_arm`]),
            // because the second is what the device's import granules turn the
            // first into. Every other gather keeps the engine under this
            // bucket.
            crate::backend::vulkan::engine::SampledSource::GuestRuns(source, _vouch) => {
                // The texture's own view — its extent in texels beside the
                // bytes one texel takes — rather than one multiplied-out
                // number, because a padded source's span is a product of the
                // stride the guest states and the row *this* view states
                // (R36). One arithmetic feeds the lease window, the check and
                // the copy, so the three cannot disagree.
                let gather = sampled_gather_window(
                    source,
                    TextureExtent {
                        width: image.width,
                        height: image.height,
                        bytes_per_texel: format.bytes_per_texel(),
                    },
                );
                let gather = match gather {
                    Ok(gather) => gather,
                    Err(exit) => {
                        return Err(OutOfClass::owned(
                            "render_provider_out_of_class_texture_source",
                            format!(
                                "a draw whose `[[texture({})]]` texels are gathered from guest \
                                 memory stays on the engine when the gather is not one registered \
                                 window covering the texture's own tightly packed extent: the \
                                 canonical contract states a lease-backed sampled texture as the \
                                 texture's extent at the reservation's own start \
                                 (`TextureSource::BorrowedNoCopy`, or the owner's staged copy of \
                                 exactly those bytes), and {}",
                                declaration.index,
                                exit.sentence(),
                            ),
                        ));
                    }
                };
                match gather.rows {
                    // The guest's own rows are padded (R36): the class gate
                    // repacks them into the texture's tightly packed extent
                    // before anything is declared, because every lease window
                    // this rail can cut names the reservation's own bytes —
                    // which for a padded gather are the rows *with* their
                    // padding.
                    Some(rows) => NarrowTextureSource::Depadded {
                        window: gather.window,
                        rows,
                    },
                    None => NarrowTextureSource::Window {
                        binding: texture_owner_binding(declaration.index),
                        window: gather.window,
                    },
                }
            }
        };
        // The canonical render sampler of *this* class executes one texture
        // extent — the render area's own, so every fragment's sample stands on
        // a texel centre of the surface it reads — unless the device's own frame
        // declares that it gathers a source of another extent, and the two arms
        // of that shape are the two rails' own disagreement (R35, R37).
        // `research/docs/23` §111 widened the *Vulkan* rail at E-TX5: a source
        // of another extent whose bytes the rail reads off the host is gathered
        // into the render area's own grid, and
        // `render_texture_extent_unsupported` is kept for the owner's no-copy
        // window alone — the one arm with no host bytes to gather. The native
        // rail's own extent rule answers *every* source of another extent with
        // that same name, so no Apple-side frame exists to reconcile a widened
        // answer against. E-TX10 publishes the difference as a shape bit
        // (`ProviderCapabilities::supports_render_texture_gathered_extent`,
        // false by default and false out of any frame that does not carry it),
        // and [`submit_render`] hands this walk the frame's own reading of it.
        //
        // So the shape leaves for the provider one arm at a time, and each arm
        // only where its own fact holds: the host-bytes arm
        // ([`texture_extent_arm`]'s `HostBytes` — the request's own copy, the
        // caller's frame, the trace's production, the window the class gate
        // copies, and R36's repacked rows) where the frame declares the gather
        // (R37), and the owner's no-copy window where the frame declares that
        // *it* reads the window in place (R40). A device whose frame declares
        // neither keeps this refusal for both arms, and a device that declares
        // one keeps it for the other: R35's slug, sentence, gate and route, byte
        // for byte. Nothing about an admitted arm changes shape, trace or copy —
        // the declaration states the bind's own extent, which is what the Vulkan
        // rail binds by — and the refusal is counted under the arm the bind
        // would have stated ([`texture_extent_route`]).
        //
        // R40 is that same reading for the arm E-TX12 opened, and it is read out
        // of its *own* bit rather than out of E-TX10's (`research/docs/23` §111,
        // E-TX10's own sentence says the host-gather bit must not be read as
        // "the snapshot executes the owner's no-copy window"): the window whose
        // first byte is the texture's carries no host bytes to gather, so the
        // canonical rail answers it with code of its own — a *translated*
        // fragment stage binds the window at the source's own extent exactly as
        // it binds the trace's own bytes, which is the arm this rail registers
        // (`metal-api-vulkan`'s `register_translated_render_pipeline`), and the
        // reviewed sampling pair's gathered sibling reads it in place at the
        // destination grid's own index. Each arm is therefore weighed against
        // the frame's own declaration of *that* arm, and an arm whose bit the
        // frame leaves out keeps R35's slug, sentence, gate and route, byte for
        // byte, exactly as it did before either bit existed.
        if image.width != req.width || image.height != req.height {
            let arm = texture_extent_arm(&source);
            let declared = match arm {
                // The owner's no-copy window: the arm E-TX12 publishes, whose
                // statement is that the device reads the owner's mapping and
                // makes no host copy of it (R40).
                TextureExtentRoute::BorrowedNoCopy => render_texture_gathered_extent_no_copy,
                // Every source a rail reads off the host: the arm E-TX10
                // publishes, whose statement is the gather into the render
                // area's own grid (R37).
                TextureExtentRoute::HostBytes => render_texture_gathered_extent,
            };
            if !declared {
                note_texture_extent(arm);
                return Err(OutOfClass::owned(
                    "render_provider_out_of_class_texture_extent",
                    format!(
                        "a draw whose `[[texture({})]]` is {}x{} in a {}x{} pass stays on the \
                         engine: the two rails E ships answer this shape differently and no \
                         capability this class reads tells them apart — the Vulkan provider \
                         gathers a source of another extent into the render area's own grid and \
                         keeps `render_texture_extent_unsupported` for the owner's no-copy window \
                         alone (`research/docs/23` §111, E-TX5), while the native rail answers \
                         *every* other extent with that same name — so the draw goes to the \
                         engine, the rail that runs the shape on both, rather than to a provider \
                         whose answer the Apple-side oracle does not state",
                        declaration.index, image.width, image.height, req.width, req.height,
                    ),
                ));
            }
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

/// How a request's declared attributes and the vertex stage's own translation
/// of them disagree, or `None` when they name the same locations (R-VI1).
///
/// The request's attribute list is the engine's own shape — one entry per
/// attribute *location* — and the reflection lists one input per location, so
/// the comparison is a walk of one list against the other rather than a set
/// against a set. Two counts decide it, both taken over the entries rather than
/// over a de-duplicated view of them:
///
/// * the **surplus**: declared entries naming a location the vertex stage does
///   not read;
/// * the **uncovered**: locations the vertex stage reads that no declared entry
///   names.
///
/// Refusal is `surplus > 0 || uncovered > 0 || attributes.len() != reflected.len()`,
/// and the third term is not redundant: it is what catches one location
/// declared twice, which the reflection cannot mirror because its own locations
/// are distinct by translation. Every refusal is exactly one of the three
/// directions [`VertexInterfaceRoute`] names, and which one it is decides
/// whether the shape is a candidate for widening or a permanent boundary —
/// which is why this answers with a route rather than the `bool` it used to,
/// and why the walk is spelled out here rather than at the gate.
fn vertex_interface_mismatch(
    attributes: &[VertexAttributeResource],
    reflected: &[u32],
) -> Option<VertexInterfaceMismatch> {
    let surplus = attributes
        .iter()
        .filter(|attribute| !reflected.contains(&attribute.location))
        .count();
    let uncovered = reflected
        .iter()
        .filter(|location| !attributes.iter().any(|a| a.location == **location))
        .count();
    let route = if surplus > 0 && uncovered == 0 {
        VertexInterfaceRoute::DeclaredSuperset
    } else if uncovered > 0 && surplus == 0 {
        VertexInterfaceRoute::ReflectedSuperset
    } else if surplus > 0 || uncovered > 0 || attributes.len() != reflected.len() {
        VertexInterfaceRoute::LocationMismatch
    } else {
        return None;
    };
    let distance = match route {
        VertexInterfaceRoute::DeclaredSuperset => surplus,
        VertexInterfaceRoute::ReflectedSuperset => uncovered,
        VertexInterfaceRoute::LocationMismatch => surplus + uncovered,
    };
    Some(VertexInterfaceMismatch {
        route,
        distance: u64::try_from(distance).unwrap_or(u64::MAX),
    })
}

/// One `vertex_interface` disagreement: the direction it answers in, and the
/// distance that direction states beside it.
///
/// The distance is the number of entries on the side that is wider, which is a
/// different quantity per route and is why [`vertex_interface_route_distance`]
/// names one counter per route rather than one total. Zero is a reading rather
/// than a silence, and the only shape that charges the
/// [`VertexInterfaceRoute::LocationMismatch`] route with nothing beside it is
/// one location declared twice: [`vertex_interface_route`] says which direction
/// answered, and a companion that did not fire beside the mismatch route says
/// there was no wider side at all.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct VertexInterfaceMismatch {
    route: VertexInterfaceRoute,
    distance: u64,
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

/// The attachment's own previous contents as the ordered list of owner windows
/// the contract's `BufferSource::GuestRuns` arm states (`R32`,
/// `research/docs/23` §113 / E-TX6).
///
/// The request's seed is a [`GuestRunSource`] — the engine's own carrier for a
/// mapper-ref-texture surface's prior contents — and this is the one place the
/// class turns it into provider-shaped windows: one per maximal
/// import-contiguous stretch of the surface's registered pages, in window
/// order, each carrying the provider window the registration ledger derived for
/// that stretch and the binding's own first byte inside it.
///
/// The `head` a run needs is *not* [`WindowStretch::skip`] alone: the window is
/// the page-aligned range the ledger cut, so the byte the surface starts at is
/// `GuestRef::head` bytes into it as well — the first run of a window that does
/// not begin on a granule boundary is the one that shows it. Both coordinates
/// are the ledger's own, which is why they are added here rather than being
/// re-derived from an address.
///
/// Every shape this arm cannot state is a named exit rather than a partial
/// list: the contract's run list *is* the view's byte range, so a list that is
/// short, padded or unwindowed would be a declaration about bytes no record
/// wrote.
pub(crate) fn load_seed_run_windows(
    source: &GuestRunSource,
    extent: u64,
) -> Result<Vec<StageBufferWindow>, LoadSeedRunExit> {
    // Padded rows first, because the byte count check below would also catch
    // most of them and would name the wrong fact: a padded window's span is
    // larger than the extent, and what the contract lacks is the stride.
    if source.row_length_texels != 0 {
        return Err(LoadSeedRunExit::PaddedRows {
            row_length_texels: source.row_length_texels,
        });
    }
    if source.total_len != extent {
        return Err(LoadSeedRunExit::Extent {
            span: source.total_len,
            extent,
        });
    }
    let stretches = source
        .window_stretches()
        .ok_or(LoadSeedRunExit::Unregistered)?;
    let mut windows: Vec<StageBufferWindow> = Vec::new();
    let mut total = 0_u64;
    for stretch in stretches {
        let Some(window) = stretch.window else {
            return Err(LoadSeedRunExit::Unwindowed);
        };
        let Some(head) = stretch.guest.head().checked_add(stretch.skip) else {
            return Err(LoadSeedRunExit::Unwindowed);
        };
        windows.push(StageBufferWindow {
            import: window.import.get(),
            host_va: window.base,
            length: window.length,
            head,
            bytes_len: stretch.len,
        });
        total = total.saturating_add(stretch.len);
    }
    if windows.is_empty() {
        return Err(LoadSeedRunExit::Unregistered);
    }
    if total != extent {
        return Err(LoadSeedRunExit::Total { total, extent });
    }
    // One registration per list: the contract pairs every run's reservation
    // with the declaring view's own allocation, so a list that lives in two
    // registrations is a shape no single declaration can state. The engine's
    // run walk does split at an import seam, so this is a real shape rather than
    // a hypothetical one.
    let first = windows[0].import;
    if let Some(second) = windows.iter().find(|window| window.import != first) {
        return Err(LoadSeedRunExit::Registrations {
            first,
            second: second.import,
        });
    }
    Ok(windows)
}

/// Why an attachment's own guest seed is not a run list this class can state
/// (`R32`).
///
/// One variant per distinct fact, so the census reads which condition a shape
/// is behind rather than one sentence for five of them — the rule
/// [`SampledGatherExit`] follows for the sampled sibling of this arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LoadSeedRunExit {
    /// The surface's rows are padded (`bufferRowLength`): the byte span the
    /// runs tile is not the attachment's tightly packed extent, and the
    /// contract's list carries no stride.
    PaddedRows { row_length_texels: u32 },
    /// The seed's own span is not the attachment's tightly packed extent.
    Extent { span: u64, extent: u64 },
    /// The seed carries no run list at all: a synthetic source, or a host that
    /// cannot import the pages behind it.
    Unregistered,
    /// One run's bytes have no registered window: the registration ledger does
    /// not hold that import under the current epoch.
    Unwindowed,
    /// The list's runs do not add up to the attachment's extent.
    Total { total: u64, extent: u64 },
    /// The runs live in more than one registration.
    Registrations { first: u64, second: u64 },
}

impl LoadSeedRunExit {
    /// The census bucket this exit keeps its shape on.
    fn slug(&self) -> &'static str {
        match self {
            Self::Registrations { .. } => "render_provider_out_of_class_load_seed_registrations",
            Self::PaddedRows { .. } => "render_provider_out_of_class_load_seed_rows",
            Self::Extent { .. } | Self::Total { .. } => {
                "render_provider_out_of_class_load_seed_extent"
            }
            Self::Unregistered | Self::Unwindowed => "render_provider_out_of_class_load_seed_runs",
        }
    }

    /// The route this exit answers under when the *elision* door is the one that
    /// cut the window (B1).
    ///
    /// One name per fact and the same facts the seed door counts in its own
    /// bucket: the two doors read the same window shape, so the class answers
    /// the same way about both — what differs is which door the census has to
    /// charge, because one is a seed the caller chose and the other is a frame
    /// the engine's own elision said it already held.
    pub(crate) fn window_route(self) -> ResidentSourceRoute {
        match self {
            Self::PaddedRows { .. } => ResidentSourceRoute::WindowPaddedRows,
            Self::Extent { .. } | Self::Total { .. } => ResidentSourceRoute::WindowExtent,
            Self::Unregistered => ResidentSourceRoute::WindowUnregistered,
            Self::Unwindowed => ResidentSourceRoute::WindowUnwindowed,
            Self::Registrations { .. } => ResidentSourceRoute::WindowRegistrations,
        }
    }

    /// The fact, stated as the refusal's second half.
    fn sentence(&self) -> String {
        match self {
            Self::PaddedRows { row_length_texels } => format!(
                "the surface's rows are padded (`row_length_texels`={row_length_texels}), while a \
                 run list names a tightly packed extent and carries no stride",
            ),
            Self::Extent { span, extent } => format!(
                "the seed's own span is {span} byte(s) for a {extent} byte tightly packed extent",
            ),
            Self::Unregistered => String::from(
                "the seed carries no run list: its pages have no registration under the current \
                 epoch, so there is no owner window to read them through",
            ),
            Self::Unwindowed => String::from(
                "one of the seed's runs has no registered window under the current epoch, so that \
                 stretch of the surface's bytes cannot be named",
            ),
            Self::Total { total, extent } => format!(
                "the seed's runs add up to {total} byte(s) rather than the attachment's {extent} \
                 byte extent",
            ),
            Self::Registrations { first, second } => format!(
                "the seed's runs live in two registrations (imports {first} and {second}), and \
                 the contract pairs every run's reservation with the declaring view's own \
                 allocation",
            ),
        }
    }
}

/// The bucket one **seed-door** window fact answers under, when that fact has a
/// name in R32's own run-list family (R38).
///
/// Both seed doors ask one question of one list — does the surface's own page
/// range cut into the ordered windows the contract's `BufferSource::GuestRuns`
/// states — so the facts that can stop the cut answer under the four names
/// [`LoadSeedRunExit::slug`] already gives the door that reads a
/// `DrawRequest::target_guest_seed` list. `None` is the other half: the facts
/// the door answers *before* it reaches the list at all (the mapping's geometry
/// or texel format is not the attachment's, the owed frame could not be landed,
/// the debt's identity moved, the host promises no stable alias, the pages would
/// not cut into importable runs). Those keep the door's own bucket and are
/// charged under [`resident_source_route`] — the B1 window's own name for the
/// same fact — rather than being spelled a second time here.
fn load_seed_window_slug(route: ResidentSourceRoute) -> Option<&'static str> {
    use ResidentSourceRoute as Route;
    match route {
        Route::WindowPaddedRows => Some("render_provider_out_of_class_load_seed_rows"),
        Route::WindowExtent => Some("render_provider_out_of_class_load_seed_extent"),
        Route::WindowRegistrations => Some("render_provider_out_of_class_load_seed_registrations"),
        // The window that is not there: no registered import, no owner window
        // for a stretch, or pages the owner rail cannot cut into runs at all.
        // `LoadSeedRunExit` states those three as `Unregistered`, `Unwindowed`
        // and (`AttachmentWindowMiss::Untileable`) the same route name, and
        // R32's own family folds them into one bucket — the bytes are not
        // nameable as an owner window, whether the refusal is one page or all
        // of them.
        Route::WindowUnregistered | Route::WindowUnwindowed => {
            Some("render_provider_out_of_class_load_seed_runs")
        }
        _ => None,
    }
}

/// The bucket and sentence one **seed-door** window refusal answers with (R38).
///
/// The bucket is [`load_seed_window_slug`] where the fact has a name in R32's
/// run-list family, and the door's own bucket otherwise — the same split the
/// measurement round counted, now stated as the refusal it always was. The
/// sentence names the fact the door answered with, because the census reads
/// sentences as well as buckets: a padded row, a window with no owner, a mapping
/// whose geometry is not the attachment's and an unpaid landing are four
/// different reasons to keep a record on the engine, and one sentence for all of
/// them would be a bucket nobody can size.
fn load_seed_window_refusal(route: ResidentSourceRoute) -> (&'static str, String) {
    use ResidentSourceRoute as Route;
    let fact = match route {
        Route::WindowPaddedRows => {
            "the surface's rows are padded (`bufferRowLength`), while a run list names a tightly \
             packed extent and carries no stride"
                .to_owned()
        }
        Route::WindowExtent => {
            "the window's span is not the attachment's tightly packed extent".to_owned()
        }
        Route::WindowRegistrations => {
            "the window's runs live in more than one registration, and the contract pairs every \
             run's reservation with the declaring view's own allocation"
                .to_owned()
        }
        Route::WindowUnregistered => {
            "the window's pages have no registration this process can import, so there is no \
             owner window to read or land through"
                .to_owned()
        }
        Route::WindowUnwindowed => {
            "one stretch of the window has no registered window, so that stretch of the \
             surface's bytes cannot be named"
                .to_owned()
        }
        Route::WindowGeometry => {
            "the mapping's own geometry or texel format is not the attachment's, so those pages \
             are not the previous contents this record declared"
                .to_owned()
        }
        Route::WindowLandingRefused => {
            "the frame the mapping's pages are owed could not be landed — the guest wrote over \
             it, or the rail refused the Store — so the pages do not hold what the door read \
             (`INV-LAND`: the debt is paid before the pages may be named)"
                .to_owned()
        }
        Route::WindowIdentityMoved => {
            "the mapping's identity moved between the window's declaration and this submission, \
             so the pages this window names are no longer the ones the door resolved"
                .to_owned()
        }
        Route::WindowSpanUnmapped => "a page of the window's span is unmapped".to_owned(),
        // The seam states a window only through `AttachmentWindowMiss`, so a
        // route from the door vocabulary is a wiring bug rather than a shape —
        // and a wiring bug is named, never folded into a neighbour.
        other => format!(
            "the door's own answer is not a window fact this class knows (`{}`), which is a \
             wiring bug rather than a shape",
            resident_source_route(other)
        ),
    };
    (
        load_seed_window_slug(route).unwrap_or("render_provider_out_of_class_load_seed"),
        format!(
            "a record whose previous contents are the attachment's own guest backing stays on \
             the engine: the class states the mapper-ref-texture surface's own pages as the \
             canonical attachment's ordered run list (`BufferSource::GuestRuns`), and {fact}"
        ),
    )
}

/// The extent one sampled texture's own view states: its extent in texels
/// beside the bytes one texel of its format takes (`R36`).
///
/// The three numbers travel together because every rule below reads at least
/// two of them: one tightly packed row, the tightly packed extent, and the byte
/// stride a guest `bufferRowLength` states are all products of these.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TextureExtent {
    width: u32,
    height: u32,
    bytes_per_texel: u64,
}

impl TextureExtent {
    /// Bytes one tightly packed row of this extent holds.
    fn tight_row(self) -> Option<u64> {
        u64::from(self.width).checked_mul(self.bytes_per_texel)
    }

    /// Bytes the whole tightly packed extent holds — the number a lease window
    /// names and the number the declaration's `OwnedBytes` arm carries.
    fn tight(self) -> Option<u64> {
        self.tight_row()?.checked_mul(u64::from(self.height))
    }
}

/// The stride one padded-row gather states, beside the tight row and the row
/// count the texture's own extent names (`R36`).
///
/// `bufferRowLength` is a texel count, so the guest's byte stride is its
/// product with the view's own bytes per texel — the same arithmetic
/// `runtime::draw::vulkan`'s `strided_window_extent` made when it built the
/// source, re-derived here because the two have to agree and this rail is the
/// one that states the bytes. A gather whose span is not `stride * (rows - 1) +
/// tight_row` is refused rather than repacked: the last row's trailing padding
/// is outside the gather's window by construction, so a span that does not end
/// there is a source whose rows this rail cannot locate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PaddedRows {
    /// The guest's own `bufferRowLength`, in texels — the fact the refusal and
    /// the census both name.
    row_length_texels: u32,
    /// Bytes one guest row strides: `row_length_texels * bytes_per_texel`.
    stride: u64,
    /// Bytes one tightly packed row of the texture holds.
    tight_row: u64,
    /// The rows the gather's window holds — the texture's own height.
    rows: u32,
}

impl PaddedRows {
    /// Bytes the gather's own window holds: every row but the last at the
    /// guest's stride, then one tight row.
    fn span(self) -> u64 {
        self.stride
            .saturating_mul(u64::from(self.rows.saturating_sub(1)))
            .saturating_add(self.tight_row)
    }

    /// Bytes the tightly packed copy holds.
    fn tight(self) -> u64 {
        self.tight_row.saturating_mul(u64::from(self.rows))
    }

    /// Bytes the copy drops: every row but the last carries this much padding.
    fn padding(self) -> u64 {
        self.span().saturating_sub(self.tight())
    }

    /// The tightly packed rows, assembled out of the guest's padded window.
    ///
    /// `None` when the window is not the span this layout named, which is a
    /// caller-side wiring bug rather than a shape the guest's bytes can state:
    /// every reader of this type goes through this one function, so the copy
    /// and the byte count the declaration states cannot disagree.
    fn depad(self, padded: &[u8]) -> Option<Vec<u8>> {
        if u64::try_from(padded.len()).ok() != Some(self.span()) {
            return None;
        }
        let row = usize::try_from(self.tight_row).ok()?;
        let stride = usize::try_from(self.stride).ok()?;
        let mut out = Vec::with_capacity(usize::try_from(self.tight()).ok()?);
        for index in 0..usize::try_from(self.rows).ok()? {
            let start = index.checked_mul(stride)?;
            out.extend_from_slice(padded.get(start..start.checked_add(row)?)?);
        }
        Some(out)
    }
}

/// What one sampled texture's zero-copy gather is on this device (`R28`,
/// `R36`): the registered window its bytes live in, and the guest stride when
/// its rows are padded.
///
/// `rows: None` is the tight shape [`texture_window_arm`] decides between the
/// borrowed and the staged lease arm; `rows: Some` is the padded shape the
/// class gate repacks into the texture's own extent, which is stated as the
/// trace's own bytes ([`NarrowTextureSource::Depadded`]) rather than as any
/// lease — a lease window names the reservation's own bytes, which for a padded
/// gather are the guest's rows *with* their padding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SampledGather {
    /// The window the bind's bytes live in, with the texture's own coordinates
    /// inside it. `bytes_len` is the span the gather states: the tightly packed
    /// extent when the rows are tight, the guest's padded span when they are
    /// not.
    window: StageBufferWindow,
    /// The stride when the guest's rows are padded (`bufferRowLength`).
    rows: Option<PaddedRows>,
}

/// The registered window one sampled texture's zero-copy gather was cut from
/// (`R28`), and the stride the class repacks its rows out of when the guest's
/// own rows are padded (`R36`).
///
/// The sampled sibling of [`gather_window`], on the same fact — one page run
/// whose registered window covers the bind's bytes — read for the window rule
/// the *contract* states for a texture. E's lease channel names a texture's
/// bytes as "the texture's own tightly packed extent at the reservation's own
/// start" (`TextureSource`'s doc, `research/docs/23` §75/R5c), and two
/// conditions follow from that rule that a buffer's window does not have:
///
/// - **the rows have to be tight** for the *lease* arms (`row_length_texels ==
///   0`). The extent a lease window names is `width * height *
///   bytes_per_texel`; a window whose guest rows are padded is still a bind the
///   engine gathers, but it is not one a lease can describe, because the
///   reservation carries no stride. R36 answers that shape with the trace's own
///   bytes instead of a lease: the class gate copies the guest's padded rows
///   into the texture's tightly packed extent ([`PaddedRows::depad`],
///   [`RowCopies`]) and the declaration states `TextureSource::OwnedBytes`, the
///   arm the request's own copy and R24's frame already take. The
///   stride-carrying *zero-copy* window — E's `BorrowedNoCopy` rule read for a
///   padded source — is a contract-level increment and not one this rail can
///   state;
/// - **the window has to start at the texture's first byte**, which is what
///   [`StageBufferWindow::head`] measures here: the distance from the window's
///   base to the texture's first byte is the run's own in-granule head
///   ([`GuestRef::head`], the bytes the ledger's page-aligned window is cut
///   *below*) plus the source's own `source_offset` (a packed resource's plane
///   or level offset inside the run). A caller that finds a non-zero head
///   copies the extent out of the registration instead of binding it
///   ([`texture_window_arm`]), because a borrowed reservation would name the
///   window's own bytes as the texture's.
///
/// `extent` is the texture's own view — its extent in texels beside the bytes
/// one texel takes — so the returned window's `bytes_len` is the span the
/// gather states: the tightly packed extent when the rows are tight (what a
/// lease reads, and the *same* byte count the class states on its copy arm),
/// and the guest's own padded span when they are not.
///
/// `Err` is every gather that is not one such window, and it names the fact
/// that met it ([`SampledGatherExit`]) rather than one sentence for five
/// shapes: the census reads the bucket, and a reader of the refusal reads which
/// condition the stream is behind.
fn sampled_gather_window(
    source: &GuestRunSource,
    extent: TextureExtent,
) -> Result<SampledGather, SampledGatherExit> {
    let span = source.total_len;
    // The texture's own tightly packed row and extent: the two numbers a lease
    // window names, and the two a padded gather's copy is assembled from. An
    // extent of zero bytes is nothing a view can state, whatever the source
    // carries.
    let tight_row = extent
        .tight_row()
        .filter(|row| *row != 0)
        .ok_or(SampledGatherExit::Span { span, extent: 0 })?;
    let tight = extent
        .tight()
        .filter(|tight| *tight != 0)
        .ok_or(SampledGatherExit::Span {
            span,
            extent: tight_row,
        })?;
    let Some(runs) = source.pages.as_ref() else {
        return Err(SampledGatherExit::Unregistered);
    };
    let [only] = runs.as_slice() else {
        return Err(SampledGatherExit::Scattered { runs: runs.len() });
    };
    // `single_stretch`'s own rule, one fact at a time: the window has to be the
    // *first* run's bytes, which is what `window_offset == 0` names.
    if only.window_offset != 0 {
        return Err(SampledGatherExit::Scattered { runs: runs.len() });
    }
    let Some(window) = only.window else {
        return Err(SampledGatherExit::Unregistered);
    };
    // The span the gather's own window covers, and the stride when the guest's
    // rows are padded: the tightly packed extent is the whole answer when
    // `bufferRowLength` is zero, and a padded source states a row count this
    // rail has to agree with before it can locate a single row's bytes.
    let rows = match source.row_length_texels {
        0 => {
            if span != tight {
                return Err(SampledGatherExit::Span {
                    span,
                    extent: tight,
                });
            }
            None
        }
        row_length_texels => {
            let stride = u64::from(row_length_texels)
                .checked_mul(extent.bytes_per_texel)
                .ok_or(SampledGatherExit::PaddedRowNarrow {
                    row_length_texels,
                    tight_row,
                    stride: u64::MAX,
                })?;
            // A stride narrower than one tightly packed row is not padding:
            // those rows would overlap, so there are no bytes to repack.
            if stride < tight_row {
                return Err(SampledGatherExit::PaddedRowNarrow {
                    row_length_texels,
                    tight_row,
                    stride,
                });
            }
            let rows = PaddedRows {
                row_length_texels,
                stride,
                tight_row,
                rows: extent.height,
            };
            // The span the guest's own translation derived
            // (`strided_window_extent`), re-derived here because the statement
            // below is about bytes.
            if span != rows.span() {
                return Err(SampledGatherExit::PaddedRowCount {
                    span,
                    stride,
                    tight_row,
                    rows: extent.height,
                });
            }
            Some(rows)
        }
    };
    // The run states how many bytes it asked for, and this rail reads the whole
    // span out of it: a padded gather's window carries its padding, so the
    // number checked here is the span rather than the extent.
    if source
        .source_offset
        .checked_add(span)
        .is_none_or(|end| end > only.guest.requested())
    {
        return Err(SampledGatherExit::Span {
            span,
            extent: tight,
        });
    }
    // The run's first byte is the window the gather asked for; the texture's
    // first byte is `source_offset` further in, and the *registration's* window
    // is aligned below the run by the run's own head.
    let head = only
        .guest
        .head()
        .checked_add(source.source_offset)
        .unwrap_or(u64::MAX);
    if head.checked_add(span).is_none_or(|end| end > window.length) {
        return Err(SampledGatherExit::Window {
            head,
            span,
            window: window.length,
        });
    }
    Ok(SampledGather {
        window: StageBufferWindow {
            import: window.import.get(),
            host_va: window.base,
            length: window.length,
            head,
            bytes_len: span,
        },
        rows,
    })
}

/// Why one sampled texture's zero-copy gather is not a window this class can
/// state (`R28`), the error half of [`sampled_gather_window`].
///
/// One variant per condition, because the refusal's own sentence is what a
/// reader of a boot's fail log has: the bucket (`..._texture_source`) says the
/// family, and this says which fact in it the stream is behind. R36 retired the
/// `PaddedRows` exit that used to answer *every* padded gather with one
/// sentence: a padded source whose stride, row count and window agree is
/// repacked and leaves for the provider, and the two padded variants below are
/// the shapes that still do not — a stride narrower than the texture's own row,
/// and a span the stated row count cannot tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SampledGatherExit {
    /// The guest's `bufferRowLength` names fewer texels than one tightly packed
    /// row of the texture holds (`R36`): a stride narrower than the extent
    /// would make the rows overlap rather than be padded, so there are no bytes
    /// this class could repack.
    PaddedRowNarrow {
        row_length_texels: u32,
        tight_row: u64,
        stride: u64,
    },
    /// The gather's span is not `stride * (rows - 1) + tight_row` (`R36`): the
    /// guest's rows do not tile the window this rail read, so no single row's
    /// own bytes can be located inside it.
    PaddedRowCount {
        span: u64,
        stride: u64,
        tight_row: u64,
        rows: u32,
    },
    /// More than one stretch tiles the bind (or the one run does not begin at
    /// the gather's own first byte), so no single host range is its bytes.
    Scattered { runs: usize },
    /// No registration names these bytes under the current epoch: the run
    /// carries no provider-shaped window, and the gather carries no runs at
    /// all in the synthetic case.
    Unregistered,
    /// The gather's own span is not the texture's tightly packed extent (or
    /// reaches past the run it names): the lease reads exactly the extent.
    Span { span: u64, extent: u64 },
    /// The window does not cover the span this gather reads from the texture's
    /// first byte.
    Window { head: u64, span: u64, window: u64 },
}

impl SampledGatherExit {
    /// The fact, stated as the refusal's second half.
    fn sentence(&self) -> String {
        match self {
            Self::PaddedRowNarrow {
                row_length_texels,
                tight_row,
                stride,
            } => format!(
                "the gather's rows are padded (`row_length_texels`={row_length_texels}, {stride} \
                 byte(s) of stride) and that stride is narrower than the {tight_row} byte(s) one \
                 tightly packed row of the texture holds, so the rows overlap rather than carry \
                 padding this class could drop",
            ),
            Self::PaddedRowCount {
                span,
                stride,
                tight_row,
                rows,
            } => format!(
                "the gather's rows are padded ({stride} byte(s) of stride over {rows} row(s), \
                 whose last row is {tight_row} byte(s) tight), which tiles {} byte(s) rather \
                 than the {span} byte(s) the gather carries",
                stride
                    .saturating_mul(u64::from(rows.saturating_sub(1)))
                    .saturating_add(*tight_row),
            ),
            Self::Scattered { runs } => format!(
                "the gather is scattered over {runs} stretch(es), so no single registered window \
                 is the texture's bytes",
            ),
            Self::Unregistered => String::from(
                "the gather carries no registered window: its import has no registration under \
                 the current epoch, so there is no provider-shaped window to cut a lease from",
            ),
            Self::Span { span, extent } => format!(
                "the gather's own span is {span} byte(s) for a {extent} byte tightly packed \
                 extent, and a lease reads exactly the extent",
            ),
            Self::Window { head, span, window } => format!(
                "the texture starts {head} byte(s) into a {window} byte window, which does not \
                 hold the {span} byte(s) this gather reads from its own first byte",
            ),
        }
    }
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

/// The render-texture half of the same device answer (R28).
///
/// One snapshot, two readings, exactly as [`declared_stage_buffer_support`]:
/// the bits the *frame* carries are the ones the class gate gets, because a
/// provider whose capability answer cannot hold them is a provider whose
/// remote owner never sees them. What this answers is whether the provider
/// declares the sampled-render-texture shape at all (and up to which count and
/// in which formats); the frame's own carriage of *this pass's* declarations is
/// read back from the produced frame before every lease-carrying submission
/// (`note_wire_render_textures`), so the two halves of the question are two
/// readings of the same bytes.
fn declared_render_texture_support(
) -> Result<provider_wire::RenderTextureSupport, ProviderRenderDecline> {
    let rail = rail().map_err(IntoRender::into_render)?;
    provider_wire::render_texture_support(
        rail.provider.device_epoch(),
        &rail.provider.capabilities(),
    )
    .map_err(|decline| ProviderRenderDecline::StageBufferWire {
        step: decline.step,
        detail: decline.detail,
    })
}

/// The runtime-sampler half of the same device answer (R36).
///
/// One snapshot, two readings, exactly as [`declared_render_texture_support`]:
/// the bit the *frame* carries is the one the class gets, because a provider
/// whose capability answer cannot hold it is a provider whose remote owner
/// never sees it. What this answers is whether the device executes the
/// render-sampler shape the pass's `[[sampler(n)]]` states belong to — the
/// section v70 gave the capability tail and R28 already reads for sampled
/// textures. A pass whose runtime states travel a frame declares at least one
/// sampled texture (the pairing is the family's own rule), so the two R36
/// readings of one snapshot — this one and R28's — are the two halves of one
/// question: the device that states the section executes both, and one that
/// does not keeps the draw on the engine under the name its own shape owns.
///
/// The one thing that ever replaces the device's own snapshot is the test
/// instrument below, and it replaces it *before* the frame is written, so what
/// this function answers is always the frame's own reading of a snapshot —
/// never a second opinion read beside it.
fn declared_render_sampler_carriage() -> Result<bool, ProviderRenderDecline> {
    let rail = rail().map_err(IntoRender::into_render)?;
    let capabilities = {
        let declared = rail.provider.capabilities();
        match RENDER_SAMPLER_CARRIAGE_ANSWER.load(Ordering::Relaxed) {
            RENDER_SAMPLER_CARRIAGE_DEVICE => declared,
            answer => {
                let mut declared = declared;
                // The section is one presence-tagged block, so "not declared"
                // is the *absent* section rather than a section carrying a
                // false bit: the three fields go back to the defaults the codec
                // writes the block for, so the reading below is the reading an
                // old provider's frame gives.
                let declared_now = answer == RENDER_SAMPLER_CARRIAGE_DECLARED;
                declared.supports_render_texture_sampling = declared_now;
                if !declared_now {
                    declared.max_render_textures = 0;
                    declared.supported_render_texture_formats.clear();
                }
                declared
            }
        }
    };
    provider_wire::render_sampler_carriage(rail.provider.device_epoch(), &capabilities).map_err(
        |decline| ProviderRenderDecline::StageBufferWire {
            step: decline.step,
            detail: decline.detail,
        },
    )
}

/// The device's own answer for the render-sampler carriage (R36), and the
/// states the test instrument below can put it in.
const RENDER_SAMPLER_CARRIAGE_DEVICE: u8 = 0;
const RENDER_SAMPLER_CARRIAGE_NOT_DECLARED: u8 = 1;
const RENDER_SAMPLER_CARRIAGE_DECLARED: u8 = 2;

/// Whether the render-sampler carriage is read from the device's own frame
/// ([`RENDER_SAMPLER_CARRIAGE_DEVICE`], what production runs) or from an answer
/// a test stated.
static RENDER_SAMPLER_CARRIAGE_ANSWER: AtomicU8 = AtomicU8::new(RENDER_SAMPLER_CARRIAGE_DEVICE);

/// A test's own answer for the render-sampler carriage, restored when it drops
/// (R36).
///
/// The rail reads the bit out of the provider's capability frame, and a test
/// that has to see the fail-closed arm cannot make an admitted device stop
/// declaring the shape. While this guards an answer, the capability question is
/// asked of a snapshot carrying it — written, encoded and decoded through the
/// same frame — so the arm a test sees is the arm an old frame gives (`absent`
/// reads as undeclared), and the reading is still the wire's.
///
/// A guard rather than a plain setter for the reason
/// [`StageBufferNamespaceSplitOverride`] is one: this changes a *decision* and
/// not an observation, so a test that unwound through a failed assertion would
/// otherwise leave the next shape in the same binary answering from a device
/// that is not its own.
pub struct RenderSamplerCarriageOverride {
    previous: u8,
}

impl Drop for RenderSamplerCarriageOverride {
    fn drop(&mut self) {
        RENDER_SAMPLER_CARRIAGE_ANSWER.store(self.previous, Ordering::Relaxed);
    }
}

/// Ask the render-sampler carriage as `declared` until the returned guard
/// drops, or as the device's own answer for `None` (R36).
pub fn override_render_sampler_carriage(declared: Option<bool>) -> RenderSamplerCarriageOverride {
    let answer = match declared {
        None => RENDER_SAMPLER_CARRIAGE_DEVICE,
        Some(false) => RENDER_SAMPLER_CARRIAGE_NOT_DECLARED,
        Some(true) => RENDER_SAMPLER_CARRIAGE_DECLARED,
    };
    RenderSamplerCarriageOverride {
        previous: RENDER_SAMPLER_CARRIAGE_ANSWER.swap(answer, Ordering::Relaxed),
    }
}

/// The folded-pair half of the same device answer (R33).
///
/// One snapshot, two readings, exactly as [`declared_stage_buffer_support`]:
/// the bit the *frame* carries is the one the class gate gets. What it answers
/// is whether this provider executes a pass whose two stages each read a
/// `[[buffer(n)]]` argument of the same index — the shape R31 kept on the
/// engine by name, because the canonical rail's default layout folds both
/// descriptors onto set 0's own binding `n`. `false` is the fail-closed
/// answer *and* what a frame written before the bit existed decodes to
/// (`provider_wire::stage_buffer_namespace_split`), so this read can only ever
/// widen the class by what the device states.
fn declared_stage_buffer_namespace_split() -> Result<bool, ProviderRenderDecline> {
    let rail = rail().map_err(IntoRender::into_render)?;
    // The one thing that ever replaces the device's own snapshot is the test
    // instrument below, and it replaces it *before* the frame is written, so
    // what this function answers is always the frame's own reading of a
    // snapshot — never a second opinion read beside it.
    let capabilities = {
        let declared = rail.provider.capabilities();
        match NAMESPACE_SPLIT_ANSWER.load(Ordering::Relaxed) {
            NAMESPACE_SPLIT_DEVICE => declared,
            answer => {
                let mut declared = declared;
                declared.supports_render_stage_buffer_namespace_split =
                    answer == NAMESPACE_SPLIT_DECLARED;
                declared
            }
        }
    };
    provider_wire::stage_buffer_namespace_split(rail.provider.device_epoch(), &capabilities)
        .map_err(|decline| ProviderRenderDecline::StageBufferWire {
            step: decline.step,
            detail: decline.detail,
        })
}

/// The device's own answer for the folded-pair capability (R33), and the
/// states the test instrument below can put it in.
const NAMESPACE_SPLIT_DEVICE: u8 = 0;
const NAMESPACE_SPLIT_NOT_DECLARED: u8 = 1;
const NAMESPACE_SPLIT_DECLARED: u8 = 2;

/// Whether the folded-pair capability is read from the device's own frame
/// ([`NAMESPACE_SPLIT_DEVICE`], what production runs) or from an answer a test
/// stated.
static NAMESPACE_SPLIT_ANSWER: AtomicU8 = AtomicU8::new(NAMESPACE_SPLIT_DEVICE);

/// A test's own answer for the folded-pair capability, restored when it drops
/// (R33).
///
/// The rail reads the bit out of the provider's capability frame, and a test
/// that has to see the fail-closed arm cannot make an admitted device stop
/// declaring the shape. While this guards an answer, the capability question
/// is asked of a snapshot carrying it — written, encoded and decoded through
/// the same frame — so the arm a test sees is the arm an old frame gives
/// (`absent` reads as undeclared), and the reading is still the wire's.
///
/// A guard rather than a plain setter because this one changes a *decision*
/// and not an observation: a test that unwound through a failed assertion
/// would otherwise leave the next shape in the same binary answering from a
/// device that is not its own.
pub struct StageBufferNamespaceSplitOverride {
    previous: u8,
}

impl Drop for StageBufferNamespaceSplitOverride {
    fn drop(&mut self) {
        NAMESPACE_SPLIT_ANSWER.store(self.previous, Ordering::Relaxed);
    }
}

/// Ask the folded-pair capability as `declared` until the returned guard drops,
/// or as the device's own answer for `None` (R33).
pub fn override_stage_buffer_namespace_split(
    declared: Option<bool>,
) -> StageBufferNamespaceSplitOverride {
    let answer = match declared {
        None => NAMESPACE_SPLIT_DEVICE,
        Some(false) => NAMESPACE_SPLIT_NOT_DECLARED,
        Some(true) => NAMESPACE_SPLIT_DECLARED,
    };
    StageBufferNamespaceSplitOverride {
        previous: NAMESPACE_SPLIT_ANSWER.swap(answer, Ordering::Relaxed),
    }
}

/// The gathered-extent half of the same device answer (R37).
///
/// One snapshot, two readings, exactly as [`declared_stage_buffer_namespace_split`]:
/// the bit the *frame* carries is the one the class gate gets. What it answers
/// is whether this provider executes a sampled source of *another* extent whose
/// bytes the rail reads off the host — the half of E's two rails' disagreement
/// (R35) the Vulkan rail has had since E-TX5, where a source of another extent
/// is gathered into the render area's own grid and
/// `render_texture_extent_unsupported` is kept for the owner's no-copy window
/// alone. `false` is the fail-closed answer *and* what a frame written before
/// the bit existed decodes to (`provider_wire::render_texture_gathered_extent`,
/// the second tag of the escape family E-TX9 opened), so this read can only
/// ever widen the class by what the device states.
///
/// The bit is asked by arm and not by shape: the owner's no-copy window of
/// another extent is the arm a rail reads *without* a host copy, so E-TX12
/// publishes it as a bit of its own ([`declared_render_texture_gathered_extent_no_copy`],
/// R40) and that arm keeps R35's refusal whatever *this* reading answers.
fn declared_render_texture_gathered_extent() -> Result<bool, ProviderRenderDecline> {
    let rail = rail().map_err(IntoRender::into_render)?;
    // The one thing that ever replaces the device's own snapshot is the test
    // instrument below, and it replaces it *before* the frame is written, so
    // what this function answers is always the frame's own reading of a
    // snapshot — never a second opinion read beside it.
    let capabilities = {
        let declared = rail.provider.capabilities();
        match GATHERED_EXTENT_ANSWER.load(Ordering::Relaxed) {
            GATHERED_EXTENT_DEVICE => declared,
            answer => {
                let mut declared = declared;
                declared.supports_render_texture_gathered_extent =
                    answer == GATHERED_EXTENT_DECLARED;
                declared
            }
        }
    };
    provider_wire::render_texture_gathered_extent(rail.provider.device_epoch(), &capabilities)
        .map_err(|decline| ProviderRenderDecline::StageBufferWire {
            step: decline.step,
            detail: decline.detail,
        })
}

/// The device's own answer for the gathered-extent capability (R37), and the
/// states the test instrument below can put it in.
const GATHERED_EXTENT_DEVICE: u8 = 0;
const GATHERED_EXTENT_NOT_DECLARED: u8 = 1;
const GATHERED_EXTENT_DECLARED: u8 = 2;

/// Whether the gathered-extent capability is read from the device's own frame
/// ([`GATHERED_EXTENT_DEVICE`], what production runs) or from an answer a test
/// stated.
static GATHERED_EXTENT_ANSWER: AtomicU8 = AtomicU8::new(GATHERED_EXTENT_DEVICE);

/// A test's own answer for the gathered-extent capability, restored when it
/// drops (R37).
///
/// The rail reads the bit out of the provider's capability frame, and a test
/// that has to see the fail-closed arm cannot make an admitted device stop
/// declaring the shape. While this guards an answer, the capability question is
/// asked of a snapshot carrying it — written, encoded and decoded through the
/// same frame — so the arm a test sees is the arm an old frame gives (`absent`
/// reads as undeclared), and the reading is still the wire's.
///
/// A guard rather than a plain setter for the reason
/// [`StageBufferNamespaceSplitOverride`] is one: this changes a *decision* and
/// not an observation, so a test that unwound through a failed assertion would
/// otherwise leave the next shape in the same binary answering from a device
/// that is not its own.
pub struct RenderTextureGatheredExtentOverride {
    previous: u8,
}

impl Drop for RenderTextureGatheredExtentOverride {
    fn drop(&mut self) {
        GATHERED_EXTENT_ANSWER.store(self.previous, Ordering::Relaxed);
    }
}

/// Ask the gathered-extent capability as `declared` until the returned guard
/// drops, or as the device's own answer for `None` (R37).
pub fn override_render_texture_gathered_extent(
    declared: Option<bool>,
) -> RenderTextureGatheredExtentOverride {
    let answer = match declared {
        None => GATHERED_EXTENT_DEVICE,
        Some(false) => GATHERED_EXTENT_NOT_DECLARED,
        Some(true) => GATHERED_EXTENT_DECLARED,
    };
    RenderTextureGatheredExtentOverride {
        previous: GATHERED_EXTENT_ANSWER.swap(answer, Ordering::Relaxed),
    }
}

/// The gathered extent's *no-copy* half of the same device answer (R40).
///
/// One snapshot, two readings, exactly as [`declared_render_texture_gathered_extent`]:
/// the bit the *frame* carries is the one the class gate gets, because a
/// provider whose capability answer cannot hold it is a provider whose remote
/// owner never sees it. What it answers is whether this provider executes the
/// other arm of R35's split — a sampled bind whose source is the owner's
/// no-copy window at an extent other than the pass's, which E-TX12's rail reads
/// *in place*: the translated fragment stage binds the window at the source's
/// own extent exactly as it binds the trace's own bytes, and the reviewed
/// sampling pair's gathered sibling fetches the destination grid's texel on the
/// device (`research/docs/23` §111). Neither arm copies the owner's mapping to
/// the host, which is the whole statement of the bit.
///
/// `false` is the fail-closed answer *and* what a frame written before the bit
/// existed decodes to (`provider_wire::render_texture_gathered_extent_no_copy`,
/// the fourth tag of the escape family E-TX9 opened), so this read can only ever
/// widen the class by what the device states — and it widens the no-copy arm
/// alone: E-TX10's reading is the host-bytes arm's own bit, and a device that
/// declares one keeps the other arm on the engine.
fn declared_render_texture_gathered_extent_no_copy() -> Result<bool, ProviderRenderDecline> {
    let rail = rail().map_err(IntoRender::into_render)?;
    // The one thing that ever replaces the device's own snapshot is the test
    // instrument below, and it replaces it *before* the frame is written, so
    // what this function answers is always the frame's own reading of a
    // snapshot — never a second opinion read beside it.
    let capabilities = {
        let declared = rail.provider.capabilities();
        match GATHERED_EXTENT_NO_COPY_ANSWER.load(Ordering::Relaxed) {
            GATHERED_EXTENT_NO_COPY_DEVICE => declared,
            answer => {
                let mut declared = declared;
                declared.supports_render_texture_gathered_extent_no_copy =
                    answer == GATHERED_EXTENT_NO_COPY_DECLARED;
                declared
            }
        }
    };
    provider_wire::render_texture_gathered_extent_no_copy(
        rail.provider.device_epoch(),
        &capabilities,
    )
    .map_err(|decline| ProviderRenderDecline::StageBufferWire {
        step: decline.step,
        detail: decline.detail,
    })
}

/// The device's own answer for the no-copy half of the gathered extent (R40),
/// and the states the test instrument below can put it in.
const GATHERED_EXTENT_NO_COPY_DEVICE: u8 = 0;
const GATHERED_EXTENT_NO_COPY_NOT_DECLARED: u8 = 1;
const GATHERED_EXTENT_NO_COPY_DECLARED: u8 = 2;

/// Whether the no-copy half of the gathered extent is read from the device's own
/// frame ([`GATHERED_EXTENT_NO_COPY_DEVICE`], what production runs) or from an
/// answer a test stated.
static GATHERED_EXTENT_NO_COPY_ANSWER: AtomicU8 = AtomicU8::new(GATHERED_EXTENT_NO_COPY_DEVICE);

/// A test's own answer for the no-copy half of the gathered extent, restored
/// when it drops (R40).
///
/// The sibling of [`RenderTextureGatheredExtentOverride`] and for the same
/// reason: the rail reads the bit out of the provider's capability frame, and a
/// test that has to see the fail-closed arm cannot make an admitted device stop
/// declaring the shape. While this guards an answer, the capability question is
/// asked of a snapshot carrying it — written, encoded and decoded through the
/// same frame — so the arm a test sees is the arm a frame written before the bit
/// gives (`absent` reads as undeclared), and the reading is still the wire's.
///
/// A guard rather than a plain setter for the reason
/// [`StageBufferNamespaceSplitOverride`] is one: this changes a *decision* and
/// not an observation, so a test that unwound through a failed assertion would
/// otherwise leave the next shape in the same binary answering from a device
/// that is not its own.
pub struct RenderTextureGatheredExtentNoCopyOverride {
    previous: u8,
}

impl Drop for RenderTextureGatheredExtentNoCopyOverride {
    fn drop(&mut self) {
        GATHERED_EXTENT_NO_COPY_ANSWER.store(self.previous, Ordering::Relaxed);
    }
}

/// Ask the no-copy half of the gathered extent as `declared` until the returned
/// guard drops, or as the device's own answer for `None` (R40).
pub fn override_render_texture_gathered_extent_no_copy(
    declared: Option<bool>,
) -> RenderTextureGatheredExtentNoCopyOverride {
    let answer = match declared {
        None => GATHERED_EXTENT_NO_COPY_DEVICE,
        Some(false) => GATHERED_EXTENT_NO_COPY_NOT_DECLARED,
        Some(true) => GATHERED_EXTENT_NO_COPY_DECLARED,
    };
    RenderTextureGatheredExtentNoCopyOverride {
        previous: GATHERED_EXTENT_NO_COPY_ANSWER.swap(answer, Ordering::Relaxed),
    }
}

/// The declared-superset half of the same device answer (R-VI1).
///
/// One snapshot, two readings, exactly as [`declared_render_texture_gathered_extent`]:
/// the bit the *frame* carries is the one the class gate gets, because a
/// provider whose capability answer cannot hold it is a provider whose remote
/// owner never sees it. What it answers is whether this provider executes a
/// request whose declared attribute layout names a location the vertex stage
/// never reads — the direction R-VI1 split out of the `vertex_interface`
/// bucket, and the one the canonical Vulkan rail has always been able to
/// execute (`create_pipeline` emits one `VkVertexInputAttributeDescription` per
/// *declared* attribute, so the surplus stream is bound and ignored exactly as
/// Metal's own descriptor binds and ignores it). `false` is the fail-closed
/// answer *and* what a frame written before the bit existed decodes to
/// (`provider_wire::render_vertex_interface_superset`, the third tag of the
/// escape family E-TX9 opened), so this read can only ever widen the class by
/// what the device states.
///
/// The direction it does *not* answer is R-VI1's reflected superset, which no
/// contract can admit: a location the vertex stage reads that no declared entry
/// covers has no defined value, so the walk keeps that refusal whatever this
/// answers.
fn declared_render_vertex_interface_superset() -> Result<bool, ProviderRenderDecline> {
    let rail = rail().map_err(IntoRender::into_render)?;
    // The one thing that ever replaces the device's own snapshot is the test
    // instrument below, and it replaces it *before* the frame is written, so
    // what this function answers is always the frame's own reading of a
    // snapshot — never a second opinion read beside it.
    let capabilities = {
        let declared = rail.provider.capabilities();
        match VERTEX_INTERFACE_SUPERSET_ANSWER.load(Ordering::Relaxed) {
            VERTEX_INTERFACE_SUPERSET_DEVICE => declared,
            answer => {
                let mut declared = declared;
                declared.supports_render_vertex_interface_superset =
                    answer == VERTEX_INTERFACE_SUPERSET_DECLARED;
                declared
            }
        }
    };
    provider_wire::render_vertex_interface_superset(rail.provider.device_epoch(), &capabilities)
        .map_err(|decline| ProviderRenderDecline::StageBufferWire {
            step: decline.step,
            detail: decline.detail,
        })
}

/// The device's own answer for the declared-superset vertex interface (R-VI1),
/// and the states the test instrument below can put it in.
const VERTEX_INTERFACE_SUPERSET_DEVICE: u8 = 0;
const VERTEX_INTERFACE_SUPERSET_NOT_DECLARED: u8 = 1;
const VERTEX_INTERFACE_SUPERSET_DECLARED: u8 = 2;

/// Whether the declared-superset capability is read from the device's own frame
/// ([`VERTEX_INTERFACE_SUPERSET_DEVICE`], what production runs) or from an
/// answer a test stated.
static VERTEX_INTERFACE_SUPERSET_ANSWER: AtomicU8 = AtomicU8::new(VERTEX_INTERFACE_SUPERSET_DEVICE);

/// A test's own answer for the declared-superset capability, restored when it
/// drops (R-VI1).
///
/// The rail reads the bit out of the provider's capability frame, and a test
/// that has to see the fail-closed arm cannot make an admitted device stop
/// declaring the shape. While this guards an answer, the capability question is
/// asked of a snapshot carrying it — written, encoded and decoded through the
/// same frame — so the arm a test sees is the arm an old frame gives (`absent`
/// reads as undeclared), and the reading is still the wire's.
///
/// A guard rather than a plain setter for the reason
/// [`RenderTextureGatheredExtentOverride`] is one: this changes a *decision*
/// and not an observation, so a test that unwound through a failed assertion
/// would otherwise leave the next shape in the same binary answering from a
/// device that is not its own.
pub struct RenderVertexInterfaceSupersetOverride {
    previous: u8,
}

impl Drop for RenderVertexInterfaceSupersetOverride {
    fn drop(&mut self) {
        VERTEX_INTERFACE_SUPERSET_ANSWER.store(self.previous, Ordering::Relaxed);
    }
}

/// Ask the declared-superset vertex interface as `declared` until the returned
/// guard drops, or as the device's own answer for `None` (R-VI1).
pub fn override_render_vertex_interface_superset(
    declared: Option<bool>,
) -> RenderVertexInterfaceSupersetOverride {
    let answer = match declared {
        None => VERTEX_INTERFACE_SUPERSET_DEVICE,
        Some(false) => VERTEX_INTERFACE_SUPERSET_NOT_DECLARED,
        Some(true) => VERTEX_INTERFACE_SUPERSET_DECLARED,
    };
    RenderVertexInterfaceSupersetOverride {
        previous: VERTEX_INTERFACE_SUPERSET_ANSWER.swap(answer, Ordering::Relaxed),
    }
}

/// The two narrow sampled lanes the device's own frame lists (R39).
///
/// The render-sampler section's format list read one lane at a time, by the
/// same one-snapshot-two-readings rule [`declared_render_texture_support`]
/// states: E's `render-sampled-narrow-lanes` appended `r8_unorm` and
/// `r8g8_unorm` to the contract's `RENDER_SAMPLED` list (design decision: the
/// list is widened rather than a second list added), so a frame that carries
/// them is a frame whose owner sees a provider that creates those views, and
/// the class may state a bind of one.
///
/// `false` is the fail-closed answer for each lane: a frame written before the
/// two codes existed decodes to a list without them, and an older *decoder*
/// refuses the frame by name (`UnknownEnumValue`) rather than reading one of
/// the new codes as another format, so this read can only ever widen the class
/// by what the device states.
///
/// The bit is asked by shape and not by format: the class asks for the answer
/// exactly when the request names a sampled bind whose view format is one of
/// the two lanes ([`sampled_bind_of_a_narrow_lane`]), so a request whose binds
/// are all four-byte texels reaches the rail's provider no earlier than it did.
fn declared_render_texture_narrow_lanes() -> Result<NarrowLanes, ProviderRenderDecline> {
    let rail = rail().map_err(IntoRender::into_render)?;
    // The one thing that ever replaces the device's own snapshot is the test
    // instrument below, and it replaces it *before* the frame is written, so
    // what this function answers is always the frame's own reading of a
    // snapshot — never a second opinion read beside it. The list is widened or
    // narrowed *in place*: the section is one presence-tagged block whose
    // format list is the field in question, so a test's answer travels as the
    // list an old (narrower) or a new (wider) frame carries.
    let capabilities = {
        let declared = rail.provider.capabilities();
        match NARROW_LANES_ANSWER.load(Ordering::Relaxed) {
            NARROW_LANES_DEVICE => declared,
            answer => {
                let mut declared = declared;
                let state_the_lanes = answer == NARROW_LANES_DECLARED;
                declared.supported_render_texture_formats.retain(|format| {
                    state_the_lanes
                        || !matches!(format, TextureFormat::R8Unorm | TextureFormat::R8G8Unorm)
                });
                if state_the_lanes {
                    for lane in [TextureFormat::R8Unorm, TextureFormat::R8G8Unorm] {
                        if !declared.supported_render_texture_formats.contains(&lane) {
                            declared.supported_render_texture_formats.push(lane);
                        }
                    }
                }
                declared
            }
        }
    };
    let support =
        provider_wire::render_texture_narrow_lanes(rail.provider.device_epoch(), &capabilities)
            .map_err(|decline| ProviderRenderDecline::StageBufferWire {
                step: decline.step,
                detail: decline.detail,
            })?;
    Ok(NarrowLanes {
        r8: support.r8,
        rg8: support.rg8,
    })
}

/// Which of the two narrow sampled lanes this class may state for a bind (R39),
/// as the device's own frame lists them.
///
/// A copy value rather than a borrow of the reading: the class gate is pure and
/// takes it as an argument ([`submit_render`] reads it), exactly as the R37
/// extent answer travels.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct NarrowLanes {
    /// Whether the frame's format list carries `r8_unorm`.
    r8: bool,
    /// Whether the same list carries `r8g8_unorm`.
    rg8: bool,
}

impl NarrowLanes {
    /// The answer a device that lists neither lane gives, which is also what
    /// every request whose binds state no narrow view reads: the class gate
    /// then answers exactly as it did before this increment.
    const NONE: Self = Self {
        r8: false,
        rg8: false,
    };

    /// The contract format this device's answer states for one bind's own
    /// Vulkan view of a narrow lane, or `None` when the frame does not list it.
    ///
    /// The two wide lanes are deliberately **not** here: they are the window
    /// the class has stated since E-TX1, and a device whose frame omitted one of
    /// them is answered by R28's own wire reading when the pass crosses the
    /// frame rather than by a second spelling of that rule in this gate.
    fn admits(self, format: ash::vk::Format) -> Option<TextureFormat> {
        match format {
            ash::vk::Format::R8_UNORM if self.r8 => Some(TextureFormat::R8Unorm),
            ash::vk::Format::R8G8_UNORM if self.rg8 => Some(TextureFormat::R8G8Unorm),
            _ => None,
        }
    }
}

/// The device's own answer for the narrow lanes (R39), and the states the test
/// instrument below can put it in.
const NARROW_LANES_DEVICE: u8 = 0;
const NARROW_LANES_NOT_DECLARED: u8 = 1;
const NARROW_LANES_DECLARED: u8 = 2;

/// Whether the narrow lanes are read from the device's own frame
/// ([`NARROW_LANES_DEVICE`], what production runs) or from an answer a test
/// stated.
static NARROW_LANES_ANSWER: AtomicU8 = AtomicU8::new(NARROW_LANES_DEVICE);

/// A test's own answer for the two narrow lanes, restored when it drops (R39).
///
/// The rail reads the lanes out of the provider's capability frame, and a test
/// that has to see the fail-closed arm cannot make an admitted device stop
/// listing them. While this guards an answer, the capability question is asked
/// of a snapshot carrying (or missing) the two lanes — written, encoded and
/// decoded through the same frame — so the arm a test sees is the arm an old
/// frame gives (`absent` reads as undeclared), and the reading is still the
/// wire's.
///
/// A guard rather than a plain setter for the reason
/// [`RenderTextureGatheredExtentOverride`] is one: this changes a *decision*
/// and not an observation, so a test that unwound through a failed assertion
/// would otherwise leave the next shape in the same binary answering from a
/// device that is not its own.
pub struct RenderTextureNarrowLanesOverride {
    previous: u8,
}

impl Drop for RenderTextureNarrowLanesOverride {
    fn drop(&mut self) {
        NARROW_LANES_ANSWER.store(self.previous, Ordering::Relaxed);
    }
}

/// Ask the narrow lanes as `declared` until the returned guard drops, or as the
/// device's own answer for `None` (R39).
///
/// `Some(true)` states both lanes, `Some(false)` states neither: the contract's
/// list is one list, so a device that lists one of the two lanes and not the
/// other is not a shape this instrument has to be able to spell — and the class
/// reads each lane separately from whatever the frame carries, so a frame that
/// did carry one would be answered lane by lane rather than by a pair.
pub fn override_render_texture_narrow_lanes(
    declared: Option<bool>,
) -> RenderTextureNarrowLanesOverride {
    let answer = match declared {
        None => NARROW_LANES_DEVICE,
        Some(false) => NARROW_LANES_NOT_DECLARED,
        Some(true) => NARROW_LANES_DECLARED,
    };
    RenderTextureNarrowLanesOverride {
        previous: NARROW_LANES_ANSWER.swap(answer, Ordering::Relaxed),
    }
}

/// The highest vertex one indexed draw's own index bytes name, over the first
/// `count` indices of the declared width, or `None` when the bytes stop short of
/// them.
///
/// One reader for the two proofs that need it: the class's own span gate
/// (`render_provider_out_of_class_vertex_span`) and the affine stage-buffer
/// bound beside it ([`stage_buffer_affine_counts`]). Both read the *staged*
/// arm's bytes, because the number is a value and not a length — a zero-copy
/// index window is guest RAM this process has not read, and a caller that cannot
/// read it gets `None` rather than a bound nothing states.
#[inline]
fn highest_index(
    bytes: &[u8],
    index_type: crate::backend::vulkan::engine::IndexType,
    count: u32,
) -> Option<u64> {
    use crate::backend::vulkan::engine::IndexType;
    let width = match index_type {
        IndexType::U16 => 2,
        IndexType::U32 => 4,
    };
    let count = usize::try_from(count).ok()?;
    let readable = count.checked_mul(width)?;
    if bytes.len() < readable {
        return None;
    }
    let mut highest = 0u64;
    for chunk in bytes.chunks_exact(width).take(count) {
        let value = match index_type {
            IndexType::U16 => u64::from(u16::from_ne_bytes([chunk[0], chunk[1]])),
            IndexType::U32 => {
                u64::from(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            }
        };
        highest = highest.max(value);
    }
    Some(highest)
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
    let index = req.indexed.as_ref()?;
    let bytes = staged_bytes(&index.content)?;
    // The same reader the class's own span gate uses, so the highest index an
    // affine proof is evaluated over and the highest index that gate weighs a
    // stream's capacity against cannot be two different numbers.
    let highest = highest_index(bytes, index.index_type, index.index_count)?;
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
            // E-TX6 (`metal-api-core` v113) added the guest-runs source arm
            // (`BufferSource::GuestRuns`); the seam does not declare it yet, so
            // the wire note names the population rather than inventing a view.
            Some(BufferSource::GuestRuns(runs)) => format!("guest_runs={}", runs.len()),
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

/// What one frame carries of a pass's sampled textures (R28), printed from the
/// *decoded* trace.
///
/// The render sibling of [`note_wire_stage_buffers`], and it exists for the
/// same reason: the wire question the class asks ("does the frame carry this
/// pass's texture declarations?") is a statement about bytes, and the only
/// reading that can falsify it is the one taken from the frame this rail
/// produced. One line per texture — the pipeline's own declaration (Metal
/// index, access, sampler form) beside the pass's view (format, extent, source
/// arm) — read back out of the decoded value rather than out of the values this
/// rail built.
///
/// The source arm is the fact this increment turns on: a borrowed lease and a
/// staged lease are the two arms [`texture_window_arm`] decides between, and
/// `view=absent` is the shape the pre-E-TX4 frame would have decoded to — the
/// declarations dropped, which is what the class's old pure-gate refusal named.
fn note_wire_render_textures(trace: &ComputeTrace) {
    let declarations = trace
        .pipelines
        .iter()
        .find_map(|pipeline| pipeline.render.as_ref())
        .map(|render| render.textures.as_slice())
        .unwrap_or(&[]);
    let views = trace
        .passes
        .iter()
        .find_map(TracePass::as_render)
        .map(|pass| pass.textures.as_slice())
        .unwrap_or(&[]);
    for declaration in declarations {
        let view = views
            .iter()
            .find(|view| view.metal_binding == declaration.metal_binding);
        let source = match view.map(|view| &view.source) {
            Some(TextureSource::OwnedBytes(bytes)) => format!("owned_bytes={}", bytes.len()),
            Some(TextureSource::StagedLease(lease)) => format!("staged_lease={}", lease.get()),
            Some(TextureSource::BorrowedNoCopy(lease)) => {
                format!("borrowed_lease={}", lease.get())
            }
            Some(TextureSource::TraceView) => "trace_view".to_owned(),
            None => "view=absent".to_owned(),
        };
        crate::observe::line(format!(
            "render_provider_wire render_texture index={} access={} sampler={} {} \
             view_format={} view_extent={}x{}",
            declaration.metal_binding,
            texture_access_name(declaration.access),
            match (declaration.sampler, declaration.runtime_sampler) {
                (Some(_), _) => "static".to_owned(),
                (_, Some(index)) => format!("runtime={index}"),
                (None, None) => "none".to_owned(),
            },
            source,
            view.map(|view| format!("{:?}", view.format))
                .unwrap_or_else(|| "none".to_owned()),
            view.map(|view| view.width).unwrap_or(0),
            view.map(|view| view.height).unwrap_or(0),
        ));
    }
}

/// The name one texture access is reported under, in the spelling the
/// contract's own fields use.
fn texture_access_name(access: TextureAccess) -> &'static str {
    match access {
        TextureAccess::Sampled => "sampled",
        TextureAccess::Fetched => "fetched",
        TextureAccess::Storage => "storage",
        TextureAccess::Unused => "unused",
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

/// The folded pair one request's statement makes, in the walk's own order
/// (R31/R33).
///
/// Two declarations the walk states — the ones whose access the translation
/// classifies, which is what [`RenderRailInputs::stage_buffer_statement`]
/// carries — from different stages, naming one Metal buffer index. The pair is
/// read in the same canonical order the walk reads it in (vertex bindings
/// first by index, then the fragment ones), so the first stage it names is the
/// one the walk would have stated first and the index is the one both name.
///
/// This is the walk's fold test and nothing else: it decides nothing about the
/// shape, and it is what lets the class ask the *device's* own answer to the
/// rule (R33) before the walk runs — the rule sits inside the canonical walk,
/// and a walk that ran first would have answered by name before the answer was
/// in hand. A request that makes no folded pair never reaches the question, so
/// no other shape asks anything of the rail that it did not ask before.
fn folded_stage_buffer_pair(
    inputs: &RenderRailInputs<'_>,
) -> Option<(RenderPipelineStage, RenderPipelineStage, u32)> {
    let mut ordered = inputs.stage_buffer_statement().declared;
    ordered.sort_by_key(|(stage, declaration, _)| (stage.code(), declaration.index));
    let mut stated: Vec<(RenderPipelineStage, u32)> = Vec::with_capacity(ordered.len());
    for (stage, declaration, _) in ordered {
        let folded_onto = stated
            .iter()
            .find(|(other, index)| *other != stage && *index == declaration.index);
        if let Some((other, index)) = folded_onto {
            return Some((*other, stage, *index));
        }
        stated.push((stage, declaration.index));
    }
    None
}

/// The sampled bind of another extent one request's declaration walk states, in
/// the walk's own order (R35/R37).
///
/// The first declaration the fragment stage states (which is the walk's own
/// order, and the order the extent rule reaches) whose bind — the view the
/// request names at that declaration's device binding, the pairing the rule
/// reads — is not the pass's own extent. The Metal index it returns is the one
/// the extent rule's sentence names.
///
/// This is the extent rule's own test and nothing else: it decides nothing
/// about the shape, and it is what lets the class ask the *device's* own answer
/// (R37) before the pure gate runs — the rule sits inside the declaration walk,
/// and a walk that ran first would have answered by name before the answer was
/// in hand. Every other shape reaches the question no earlier than it did: a
/// request whose sampled binds are all the pass's own extent never asks the
/// rail anything.
fn sampled_source_of_another_extent(
    inputs: &RenderRailInputs<'_>,
    req: &DrawRequest,
) -> Option<u32> {
    inputs
        .fragment_texture_declarations
        .iter()
        .find_map(|declaration| {
            let image = req
                .sampled_images
                .iter()
                .find(|image| image.binding == declaration.binding)?;
            (image.width != req.width || image.height != req.height).then_some(declaration.index)
        })
}

/// The declared-superset direction one request's vertex interface states, in
/// the rule's own terms (R-VI1).
///
/// The surplus [`vertex_interface_mismatch`] counts when the walk answers
/// [`VertexInterfaceRoute::DeclaredSuperset`] — how many declared entries name
/// a location the vertex stage never reads — and `None` for every other shape,
/// including the two disagreements the walk refuses under routes of their own.
///
/// This is the vertex-interface rule's own test and nothing else: it decides
/// nothing about the shape, and it is what lets the class ask the *device's*
/// own answer (R-VI1) before the pure gate runs — the rule sits inside the
/// gate, and a gate that ran first would have answered by name before the
/// answer was in hand. Every other shape reaches the question no earlier than
/// it did: a request whose declared layout and reflection name the same
/// locations never asks the rail anything, and neither does one whose
/// disagreement is a reflected superset or a location mismatch.
fn vertex_interface_declared_superset(
    inputs: &RenderRailInputs<'_>,
    req: &DrawRequest,
) -> Option<u32> {
    match vertex_interface_mismatch(&req.vertex_attributes, inputs.vertex_attribute_locations) {
        Some(VertexInterfaceMismatch {
            route: VertexInterfaceRoute::DeclaredSuperset,
            distance,
        }) => Some(u32::try_from(distance).unwrap_or(u32::MAX)),
        _ => None,
    }
}

/// The narrow-lane rule's own test (R39): the Metal index of the first
/// declaration whose bind states one of the two narrow texel lanes this
/// increment opened.
///
/// The sibling of [`sampled_source_of_another_extent`] and for the same reason:
/// this is the shape test that lets the class ask the *device's* own answer
/// before the pure gate runs, so a request whose sampled binds are all the
/// window the class has always stated reaches the rail's provider no earlier
/// than it did. It decides nothing about the shape — the gate weighs the bind's
/// format against the device's answer and answers under
/// `..._texture_bind` when the lane is not there.
///
/// The test is on the bind's *Vulkan view format* rather than on the contract
/// format, because the contract format is what the answer decides: the mapping
/// from one to the other is the gate's and asking it here would be a second
/// spelling of it.
fn sampled_bind_of_a_narrow_lane(inputs: &RenderRailInputs<'_>, req: &DrawRequest) -> Option<u32> {
    inputs
        .fragment_texture_declarations
        .iter()
        .find_map(|declaration| {
            let image = req
                .sampled_images
                .iter()
                .find(|image| image.binding == declaration.binding)?;
            matches!(
                image.format,
                ash::vk::Format::R8_UNORM | ash::vk::Format::R8G8_UNORM
            )
            .then_some(declaration.index)
        })
}

/// The request's two stages' `[[buffer(N)]]` statement, every rule the class
/// answers it under, in the order the census reads them.
///
/// `vertex_streams` is the number of canonical vertex streams the request's
/// attributes form — its fetch tables, not its attributes
/// ([`canonical_vertex_stream_count`]) — because that is the number of
/// bindings the canonical layout will occupy and the number the contract's own
/// vertex-layout rule is written against (`research/docs/26` §31).
///
/// `stage_buffer_namespace_split` is the device's own answer to the one
/// question the folded-pair rule asks (`R33`): whether the provider arranges
/// the two stages' buffer namespaces in different descriptor slots. `true`
/// states the pair — the walk keeps both declarations and this rail translates
/// the vertex half under the canonical namespace layout — and `false` answers
/// it exactly as R31 did, by name, at the same point in the same order.
fn stage_buffer_gate<'a>(
    inputs: &'a RenderRailInputs<'a>,
    req: &DrawRequest,
    binds: usize,
    vertex_streams: usize,
    stage_buffer_namespace_split: bool,
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
        // The two stages' Metal buffer namespaces are independent — a
        // `[[buffer(N)]]` at the vertex stage and one at the fragment stage are
        // two arguments (`StageBufferDeclaration`'s own docs) — while the
        // canonical rail's merged set 0 is not: `metal2vulkan`'s default
        // resource layout binds either stage's `[[buffer(N)]]` at set 0's own
        // binding `n`, so a pair that reads one index from both stages folds
        // both writes onto one descriptor. The rail refuses that pair when the
        // merged set is built (`render_stage_buffer_layout_unsupported`)
        // instead of executing one stage under the other's declaration; a
        // decline on an admitted draw is fail-closed, so the shape stays on the
        // engine here by name (R31). Read inside the same canonical walk as the
        // duplicate rule, so the two rules answer in one order whatever order
        // the reflections stated their declarations in.
        //
        // R33 lifts that answer for the devices that declare the split: the
        // provider translates the vertex half of such a pair under
        // `metal_api_vulkan::stage_buffer_namespace_layout`, which moves its
        // whole layout into the canonical namespace set, so the two index
        // spaces keep their own descriptors and the fold the rail used to
        // refuse is a shape it now states. The rule reads the *device's* answer
        // and not the shape's, so the branch below is the same test in the same
        // place: a device that does not declare the split — an older frame, a
        // rail that arranges nothing apart — keeps R31's refusal, its sentence
        // and its own route, byte for byte.
        if let Some(other) = out
            .iter()
            .find(|stated| stated.stage != stage && stated.index == declaration.index)
        {
            // The pair is stated rather than refused when the device declares
            // the split: `other` is the half the vertex stage's namespace
            // layout moves, and the declarations are paired by (stage, index),
            // which no layout changes — so the bind and the contract keep
            // naming the slots they named.
            if !stage_buffer_namespace_split {
                // The folded-pair rule's own route (R31), beside the unchanged
                // slug.
                note_stage_buffer_shape(StageBufferShapeRoute::Folded);
                return Err(OutOfClass::owned(
                    "render_provider_out_of_class_stage_buffer_shape",
                    format!(
                        "a draw whose {} and {} stages each read [[buffer({})]] stays on the engine: \
                     the two stages' Metal buffer namespaces are independent, while the canonical \
                     rail binds both at one descriptor — set 0's own binding for that index — and \
                     refuses the pair when the merged set is built \
                     (`render_stage_buffer_layout_unsupported`) rather than execute one stage \
                     under the other's declaration. The engine, which keeps the two namespaces \
                     apart, draws it",
                        other.stage.name(),
                        stage.name(),
                        declaration.index,
                    ),
                ));
            }
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

/// Whether this rail has a production a sampled GPU target could be served
/// from (R22's arm) — the caller's own pre-check for R24's frame read.
///
/// The caller that owns the engine's registry reads a sampled target's frame
/// out of it before every submission, and that read is a full image→host copy:
/// asking for one under an identity this rail could have restated as
/// `TextureSource::TraceView` would be a payment with no answer behind it. So
/// the caller asks this first, and `false` is the only answer that leads to a
/// read. The class re-checks the registry itself — this is a cost gate, not the
/// authority — so a production that appears (or is evicted) between the two
/// reads changes nothing about which arm answers.
pub fn sampled_target_declared(identity: &TargetIdentity) -> bool {
    recorded_production(identity).is_some()
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
    // R39: both halves of this pair are absent on the non-indexed arm — the
    // descriptor states no index view, and the pass has no index stream — so
    // the arm needs no rule of its own here.
    if let (Some(indices), Some(stream)) = (descriptor.indices.as_mut(), pass.index_stream.as_ref())
    {
        if let StreamSource::Window(window) = &stream.source {
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
    // R28: a sampled texture's window is a lease in both of its arms — the
    // borrowed no-copy window and the owner's staged copy of the extent — and
    // its bytes are exactly the extent starting at the window's own `head`. The
    // consuming trace states them the way it states every other sampled
    // texture this rail read out of a registry (R24's frame): as the trace's
    // own bytes.
    for (index, texture) in pass.textures.iter().enumerate() {
        let NarrowTextureSource::Window { binding, window } = &texture.source else {
            continue;
        };
        let bytes = provider_owner::window_bytes(owner_window(*binding, *window)).ok()?;
        descriptor.textures[index].source = TextureSource::OwnedBytes(bytes);
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
    // Both byte-bearing arms are restatable: the descriptor keeps the bytes
    // themselves ([`NarrowTextureSource::Bytes`] from the request,
    // [`NarrowTextureSource::Frame`] from the registry, R24), so re-running the
    // pass in a later trace states the same inputs. A pass that itself samples
    // a *trace-produced* view is the one shape that would need its own
    // production restated recursively, and it is not recorded.
    //
    // R28's window arm joins them: its lease cannot cross submissions either,
    // and [`production_bytes`] restates the extent out of the registration the
    // window names — the same bytes the borrowed arm would have read.
    //
    // R36's repacked rows join them on the same fact one step further in: the
    // descriptor the producing submission built already carries the texture's
    // tightly packed extent as bytes (`TextureSource::OwnedBytes`), because
    // that is the arm the padded gather left through — so re-running the pass
    // in a later trace states exactly the bytes this one sampled, with no
    // window left to re-read.
    if pass.textures.iter().any(|texture| {
        !matches!(
            texture.source,
            NarrowTextureSource::Bytes(_)
                | NarrowTextureSource::Frame(_)
                | NarrowTextureSource::Window { .. }
                | NarrowTextureSource::Depadded { .. }
        )
    }) {
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
        // B3: the frame landed in the owner's own registered window, and the
        // view that names it is a lease the submission retires with it — so
        // there is no view a later trace could sample this production through.
        // The record is still answered (its bytes are in the guest's pages,
        // which is where the seed ladder reads them); what is not recorded is a
        // production, exactly as for a frame the caller kept.
        NarrowStore::Borrowed => return None,
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// * R32: a record whose previous contents the *seed door* resolved to
    ///   bytes the caller holds — the linear GVA target's cached frame, or the
    ///   mapper-ref-texture surface's host fallback. Both are the engine's own
    ///   seed (`DrawRequest::target_rgba8`, in the order
    ///   `DrawRequest::target_seed_order` states), and the caller that built
    ///   the request hands them over folded into the attachment's own order
    ///   ([`RenderRailInputs::load_seed_source_bytes`]).
    ///
    /// Every other byte-bearing shape is still refused by name: a record whose
    /// previous contents are its own guest backing, and a GVA seed wider than
    /// four bytes per texel, are two the class cannot state, and an assertion
    /// this rail cannot check is not a load arm it may state.
    Bytes(&'a [u8]),
    /// Begin from the guest's own pages, declared as the contract's ordered run
    /// list (`BufferSource::GuestRuns`, `research/docs/23` §113, E-TX6).
    ///
    /// This is the mapper-ref-texture surface's seed door: the request carries
    /// the surface's bytes as a run list ([`GuestRunSource`]) rather than as a
    /// host copy, which is what the engine's own `GuestTargetSeed` arm draws
    /// from. The class states the list as the attachment's own view — the runs
    /// concatenate to the attachment's tightly packed extent — and the owner
    /// rail imports each run's registration so the provider can gather the
    /// bytes out of the guest's live pages at resolution
    /// ([`plan_owner_leases`], [`load_seed_owner_binding`]).
    ///
    /// A seed that is not one such list keeps the engine by name: padded rows
    /// (`row_length_texels`), a run whose bytes the registration ledger does
    /// not hold, a list whose total is not the attachment's extent, and a list
    /// split across two registrations are four different facts and four
    /// different sentences (the contract keys a list on the declaring view's
    /// own allocation, so the fourth is a shape it cannot state at all).
    GuestRuns(Vec<StageBufferWindow>),
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
    /// The frame lands in the attachment's own guest window, which is the same
    /// view the pass began from (`StoreOp::Borrowed`, E-TX8, B3).
    ///
    /// Elected only for the record whose frame the guest's pages are owed —
    /// the packet's only record, or its last one
    /// ([`RenderChainRole::SoleOrTail`]) — and only when the door declared a
    /// window for the load. The bytes still come back through the completion's
    /// writeback channel (`RenderAttachment::publishes_bytes` stays true for
    /// this arm), because the seam's own Store route reads them; what changes is
    /// that the owner's pages receive them from the provider, so the seam lands
    /// nothing a second time.
    Borrowed,
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

/// One sampled GPU target's frame, as the caller read it out of the registry
/// that holds it (R24).
///
/// Census v18 named these two populations apart: `..._texture_source_
/// undeclared` (893 records) is a bind whose texels resolved to a guest target
/// no pass of this rail recorded a production for, and the bytes of that target
/// are not a mystery — [`crate::backend::vulkan::engine::SampledSource::Target`]
/// is a resident bind, so the caller that owns the registry can read the image
/// out (`engine::read_target_four_byte_color`) and hand it over here. The
/// class then declares it exactly as a bind whose bytes the request carried
/// (the pre-R22 arm), which is what makes the two rails read one set of bytes
/// rather than two readings of one image.
///
/// The bytes travel in the *image's* own order and at its tightly packed
/// extent, which is the readback's own shape; the *bind's* view format over
/// them is a separate fact the declaration states (E-TX1, `research/docs/23`
/// §107), so one frame serves every bind of that identity.
#[derive(Clone, Debug)]
pub struct SampledTargetFrame<'a> {
    /// The guest target the frame belongs to — the identity the request's own
    /// sampled bind states, not a second name for it.
    pub identity: crate::backend::vulkan::engine::TargetIdentity,
    /// The frame's bytes: the target image's level 0, four-byte colour, at the
    /// image's own width.
    pub bytes: &'a [u8],
}

/// The attachment's own guest window, as the door that opened one record's LOAD
/// found it (`research/docs/26` §Rn, B1-B3).
///
/// The two LOAD elisions (`honour_gva_load_elision`,
/// `mapper_ref_texture_load_currency_query`) both say the same thing — the
/// engine's own image already holds the frame this record begins from — and
/// both are made by a *deferred* Store: the frame is in the engine's resident,
/// the mapping's or the plane's guest pages are owed it, and
/// `pending_writebacks` holds that debt (`arm_surface_writeback_debt` /
/// `arm_gva`). The window is the page range those two names cover, so
/// "land-then-borrow" is one answer to both halves of the record's obligation:
/// paying the debt *is* the landing (`INV-LAND`), the run list *is* the load
/// source (`INV-BORROW`, E-TX6), and the record that owes the guest its frame
/// lands it back in the same window (`INV-RETURN`, E-TX8).
///
/// A window the door could not cut is a *named* answer rather than a silent
/// absence ([`Self::Refused`]): the class's one `resident_source` refusal is
/// where that name is charged, so the census reads which fact held the shape
/// instead of folding it into the door it came in through.
pub enum AttachmentGuestWindow<'a> {
    /// The window's runs, ready to be stated as the attachment's own
    /// declaration: one [`StageBufferWindow`] per maximal import-contiguous
    /// stretch, in window order, exactly the list
    /// [`load_seed_run_windows`] states for the R32 seed door.
    Runs(&'a [StageBufferWindow]),
    /// The door applied and the window could not be declared. The class charges
    /// this route beside its refusal, so the sum over the routes is still the
    /// bucket.
    Refused(ResidentSourceRoute),
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
    /// The frame of the surface the *LOAD elision* names, when the caller can
    /// read it out of the registry that holds it (R26).
    ///
    /// This is R23's fact in another naming, and it is a separate input
    /// because the caller states a different obligation with it. A
    /// mapper-ref-texture composite's LOAD is elided when the engine's
    /// registry already holds the surface's contents under
    /// `mapper_ref_texture_render_identity` — the same identity this record's
    /// own `target_identity` names — so the frame the pass must begin from is
    /// readable (`read_target`), in the attachment's own texel order and at
    /// its own tightly packed extent, exactly as R23's is.
    ///
    /// What the caller has to vouch for is the *currency witness*: the
    /// elision's test compares the mapping's `surface_content_epoch` with the
    /// epoch stamped on the resident, so a frame answered by this rail (which
    /// writes no image into the engine's registry) must land through a Store
    /// that advances the mapping's epoch — the guest writeback — or the
    /// resident's old stamp would keep vouching for pixels no draw wrote here.
    /// The caller states exactly that by handing the frame over only for a
    /// record whose `writeback_guest` is set; every other record of the same
    /// elision keeps the class's refusal by name, and so does the GVA
    /// elision, whose witness is a readiness flag no in-tree API can clear.
    pub surface_resident_source_bytes: Option<&'a [u8]>,
    /// Which door left this record's chain uncarried, when neither byte arm
    /// above could hand its frame over (R34).
    ///
    /// The class's refusal for that shape is one slug and one sentence — the
    /// caller can neither read a frame this rail keeps nor hand the frame over
    /// — but the reasons behind it are not one fact: the *serialized* packet
    /// chain fails on a read-side condition the caller could repair, while the
    /// two LOAD elisions fail by design, because not paying that read is what
    /// they exist for. Only the seam can tell them apart: it is the layer that
    /// elected the door and holds the read that declined, and the request the
    /// class sees carries no field that names which elision fired.
    ///
    /// `None` is every record that never named a chain at all
    /// (`!DrawRequest::load_from_target`), and those never reach the refusal
    /// this route is charged beside. The class charges the route only where it
    /// refuses, so a record refused by an earlier gate — for naming no
    /// identity, or for stating guest bytes beside the live image — leaves the
    /// route uncounted rather than filing it under a refusal it did not get.
    pub resident_source_route: Option<ResidentSourceRoute>,
    /// The attachment's own guest window the door cut for this record, when the
    /// record's `load_from_target` chain is one of the two LOAD elisions and the
    /// door could name the pages its frame lives in (B1-B3,
    /// [`AttachmentGuestWindow`]).
    ///
    /// Asked **before** the two byte arms below, and that order is the point:
    /// this is the door's own declaration, so a record it answers costs no
    /// readback at all — the provider gathers the guest's live pages (E-TX6) and
    /// the packet's last record lands its frame back in the same window
    /// (E-TX8). The byte arms stay for the records this door could not state
    /// (`Runs` absent or `Refused`), which is exactly the population the census
    /// separates by route.
    ///
    /// `None` is every record whose chain is not an elision door's — the
    /// serialized packet chain (R23's arm), a middle (R25's), a seed (R32's) —
    /// and every record that never named a chain at all.
    pub attachment_guest_window: Option<AttachmentGuestWindow<'a>>,
    /// The attachment's own guest window the **seed door** cut for this record
    /// (R38), when the record's previous contents are the mapper-ref-texture
    /// surface's own guest backing (`DrawRequest::load_guest_target_backing`).
    ///
    /// This is the same declaration [`Self::attachment_guest_window`] states,
    /// asked by the other door that has a window to state. The two are separate
    /// inputs because they answer two different requests: the elision door's
    /// record *loads the live GPU image* and the window is the pages its
    /// deferred Store still owes a frame, while this door's record loads the
    /// attachment's **own guest backing** and the window is simply the pages
    /// that backing is — no image is named at all, so the window is not an
    /// optimisation of a readback but the only way to state the contents.
    ///
    /// `Runs` is the door's declaration and admits the record as the contract's
    /// ordered run list ([`NarrowLoad::GuestRuns`]); `Refused(route)` is the
    /// name of the fact that stopped the door from cutting one, charged beside
    /// the door's refusal. The routes are the *same* vocabulary
    /// [`AttachmentGuestWindow::Refused`] states — one name per fact for the
    /// same question — and the class splits them: the run-list facts answer
    /// under the four names [`LoadSeedRunExit::slug`] already gives R32's arm,
    /// and the facts the door answers *before* it cuts (geometry, an unpaid
    /// landing, a moved identity, no stable alias, an untilable span) keep the
    /// door's own bucket and are charged as routes beside it.
    ///
    /// `None` is every record whose previous contents are not that backing. A
    /// record whose request states the backing and whose seam hands no answer
    /// over is a wiring bug rather than a shape: the door is the one producer of
    /// this input and it states one for every record it elects.
    pub seed_guest_window: Option<AttachmentGuestWindow<'a>>,
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
    /// The bytes the request's own **seed door** resolved, when the caller
    /// hands them over (R32).
    ///
    /// A record whose attachment loads (`MTLLoadActionLoad` /
    /// `DontCare`) and whose previous contents are not the live GPU image
    /// arrives with `DrawRequest::target_rgba8` set: the linear GVA target's
    /// cached frame (`seed_color_load`), or the mapper-ref-texture surface's
    /// host fallback. Those bytes are the engine's own seed and they are in the
    /// order `DrawRequest::target_seed_order` states, so the *caller* — the
    /// seam, which reads both fields — folds them into the attachment's own
    /// texel order and hands them over here, exactly as it does for R25's chain
    /// value and for the same reason: the canonical attachment uploads the
    /// bytes verbatim into the pass's own image, and an image is the
    /// attachment's own view.
    ///
    /// What the caller has to vouch for is the same pair R23/R25's callers
    /// vouch for: these *are* the record's previous contents, at the
    /// attachment's own tightly packed extent, in the attachment's own order.
    /// The class refuses a hand-over that is not that extent under
    /// `render_provider_out_of_class_load_seed_shape` rather than uploading it
    /// and letting the contract answer `render_attachment_initial_mismatch`,
    /// because a decline is never a fallback.
    ///
    /// `None` is every caller that does not hand the bytes over — including the
    /// one whose seed is wider than four-byte colour, which this arm cannot
    /// fold — and those records keep the class's refusal by name.
    pub load_seed_source_bytes: Option<&'a [u8]>,
    /// The frames the caller read out of the registry that holds them, for
    /// sampled GPU targets this rail has no production to restate (R24).
    ///
    /// A bind whose texels resolved to a guest target (`SampledSource::Target`)
    /// is a bind whose bytes *exist* somewhere: the engine's registry holds
    /// them, because the resolution that produced that arm was a resident bind.
    /// R22's arm serves the identities this rail's own passes recorded a
    /// production for; for the rest, the caller that owns the registry can read
    /// the target's frame out (`engine::read_target_four_byte_color`: the
    /// image's own bytes, four-byte colour only) and hand it over here, and the
    /// declaration then states the request's own copy — the arm every pre-R22
    /// sampled texture takes.
    ///
    /// The frames are keyed by the identity the request's own bind states, not
    /// by binding: the bytes are the image's, and the bind's view format over
    /// them is a separate fact the declaration already states (E-TX1/§107), so
    /// two binds of one target read one frame. A frame that is not the
    /// declaration's own tightly packed extent is a caller wiring bug and keeps
    /// its own slug ([`sampled_textures`]); a target the caller hands nothing
    /// for keeps R22's own refusal by name. Neither is a fallback.
    pub sampled_target_frames: &'a [SampledTargetFrame<'a>],
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
    /// The caller's frame for one sampled GPU target, when it handed one over
    /// (R24's arm; see [`Self::sampled_target_frames`]).
    pub fn sampled_target_frame(
        &self,
        identity: &crate::backend::vulkan::engine::TargetIdentity,
    ) -> Option<&'a [u8]> {
        self.sampled_target_frames
            .iter()
            .find(|frame| &frame.identity == identity)
            .map(|frame| frame.bytes)
    }

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
    /// Whether this pass's frame *already landed* in the attachment's own
    /// guest window (`StoreOp::Borrowed`, E-TX8, B3).
    ///
    /// `true` is the one answer the seam's Store route has to read before it
    /// lands anything itself: the owner's pages hold the frame by construction
    /// (`resolve_attachment_landing` writes exactly the readback the bytes above
    /// are), so a second write is a copy of bytes that are already there — and
    /// the *account* that says the page changed (`mark_mapping_written` and its
    /// siblings) belongs to whoever knows the landing happened, which is the
    /// seam. `false` is every other arm: the caller's store route lands the
    /// frame exactly as it always has.
    pub landed_in_window: bool,
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
    /// Whether the vertex module was translated under the canonical namespace
    /// layout (R33).
    ///
    /// Part of the key because it is part of what the registration *did*: the
    /// same AIR translated under the two layouts is two modules with two
    /// descriptor arrangements, and a hit that answered with the other one's
    /// compiled pipeline would bind the stage's buffer at a set the module does
    /// not read. The fold is a function of the two modules' own reflections, so
    /// a given pair is folded or not independently of the request — this field
    /// states that rather than leaving it to be inferred from the AIR.
    vertex_stage_buffer_namespace_split: bool,
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
    // R33: the folded pair's own class condition, and the one device answer
    // this rail asks *before* the gate. The rule it lifts sits inside the
    // canonical walk — the second declaration of the index is the one that
    // folds — so an answer read after the walk would arrive after the walk had
    // already answered by name. The ask is gated on the request's own statement
    // being folded ([`folded_stage_buffer_pair`], the same intersection the
    // rule tests), so no other shape reaches the rail's provider any earlier
    // than it did: the ~99.9% of draws that make no folded pair still answer
    // the pure gate first, exactly as the comment below states.
    //
    // A device that declares the split executes the pair — the vertex half's
    // whole layout moves to the canonical namespace set
    // (`metal_api_vulkan::stage_buffer_namespace_layout`) and the walk states
    // both declarations — and a device (or a frame) that does not keeps R31's
    // answer, by name, at the same point in the same order.
    let stage_buffer_namespace_split = match folded_stage_buffer_pair(inputs) {
        None => false,
        Some(_) => match declared_stage_buffer_namespace_split() {
            Ok(declared) => declared,
            // A provider that cannot be reached cannot answer the question the
            // walk needs, and an unanswerable candidate is an in-class
            // candidate: fail closed, exactly as `submit_narrow` does for the
            // shapes it refuses, rather than running the shape on a rail the
            // class never named.
            Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
        },
    };
    // R37: the extent rule's own class condition, and the second device answer
    // this rail asks *before* the gate. The rule it lifts sits inside the
    // declaration walk — the walk states the module's textures one by one and
    // weighs each bind against the pass's own extent — so an answer read after
    // the walk would arrive after the walk had already answered by name. The
    // ask is gated on the request's own statement naming a sampled bind of
    // another extent ([`sampled_source_of_another_extent`], the same
    // intersection the rule tests), so no other shape reaches the rail's
    // provider any earlier than it did.
    //
    // A device (or a frame) that declares the gathered extent executes the
    // host-bytes arm of the shape — the canonical Vulkan rail gathers such a
    // source into the render area's own grid (`research/docs/23` §111, E-TX5)
    // — and a device whose frame leaves the bit out keeps R35's answer for that
    // arm, by name, at the same point in the same order. The bit is asked for
    // the *shape* here and read by arm inside the walk, because one request may
    // state both arms.
    let render_texture_gathered_extent = match sampled_source_of_another_extent(inputs, req) {
        None => false,
        Some(_) => match declared_render_texture_gathered_extent() {
            Ok(declared) => declared,
            // A provider that cannot be reached cannot answer the question the
            // walk needs, and an unanswerable candidate is an in-class
            // candidate: fail closed, exactly as `submit_narrow` does for the
            // shapes it refuses, rather than running the shape on a rail the
            // class never named.
            Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
        },
    };
    // R40: the same condition and the same gate position, asked of the extent
    // rule's *other* arm. E-TX12 publishes the owner's no-copy window as a bit
    // of its own rather than as a second reading of E-TX10's, because the two
    // arms are answered by different code on the canonical side — and E-TX10's
    // own sentence says so: its bit states the host gather and must not be read
    // as "the snapshot executes the owner's no-copy window". This rail therefore
    // reads each arm's declaration out of the arm's own bit, under the one shape
    // test both arms share ([`sampled_source_of_another_extent`]), so a request
    // whose sampled binds are all the pass's own extent still reaches the rail's
    // provider no earlier than it did. A frame that leaves this bit out keeps
    // R35's answer for the no-copy arm alone, by name, at the same point in the
    // same order as before this bit existed.
    let render_texture_gathered_extent_no_copy = match sampled_source_of_another_extent(inputs, req)
    {
        None => false,
        Some(_) => match declared_render_texture_gathered_extent_no_copy() {
            Ok(declared) => declared,
            // The same fail-closed rule as every answer above: a provider
            // that cannot answer is not a provider this rail may widen the
            // class for.
            Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
        },
    };
    // R-VI1: the vertex interface's own class condition, and the third device
    // answer this rail asks *before* the gate. The rule it lifts sits inside
    // the gate itself, so an answer read after the gate would arrive after the
    // gate had already answered by name. The ask is gated on the request's own
    // statement being the direction a contract can widen onto
    // ([`vertex_interface_declared_superset`], the same walk the rule runs), so
    // no other shape reaches the rail's provider any earlier than it did.
    //
    // A device (or a frame) that declares the shape executes the declared
    // superset — the canonical Vulkan rail's vertex input state is built from
    // the declared layout (`create_pipeline` emits one
    // `VkVertexInputAttributeDescription` per declared attribute), which is
    // what makes the surplus stream bound and ignored rather than undefined —
    // and a device whose frame leaves the bit out keeps R-VI1's answer, by
    // name, at the same point in the same order. The two directions no
    // declaration can admit never reach the question at all.
    let render_vertex_interface_superset = match vertex_interface_declared_superset(inputs, req) {
        None => false,
        Some(_) => match declared_render_vertex_interface_superset() {
            Ok(declared) => declared,
            // A provider that cannot be reached cannot answer the question the
            // gate needs, and an unanswerable candidate is an in-class
            // candidate: fail closed, exactly as `submit_narrow` does for the
            // shapes it refuses, rather than running the shape on a rail the
            // class never named.
            Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
        },
    };
    // R39: the fourth device answer this rail asks *before* the gate, on the
    // same terms as the three above. Which texel lanes the class may state for a
    // sampled bind is the provider's own declaration — E's narrow-lane change
    // appended `r8_unorm`/`r8g8_unorm` to the contract's `RENDER_SAMPLED` list,
    // and the frame's format list is where a device states that it creates
    // those views — so the answer is read out of the capability frame exactly
    // when the request names a sampled bind whose view format is one of the two
    // lanes ([`sampled_bind_of_a_narrow_lane`]). A request whose binds are all
    // four-byte texels never asks, and the gate it reaches is the one this rail
    // shipped: `NarrowLanes::NONE` states neither lane, so the narrow formats
    // keep the refusal they had, by name, at the same point in the same order.
    let render_texture_narrow_lanes = match sampled_bind_of_a_narrow_lane(inputs, req) {
        None => NarrowLanes::NONE,
        Some(_) => match declared_render_texture_narrow_lanes() {
            Ok(declared) => declared,
            // The same fail-closed rule as the three answers above: a provider
            // that cannot answer is not a provider this rail may widen the
            // class for.
            Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
        },
    };
    // The class gate is pure and runs first: an out-of-class shape never
    // touches the rail (no provider, no compile, no registration).
    let pass = match narrow_class(
        inputs,
        req,
        stage_buffer_namespace_split,
        render_texture_gathered_extent,
        render_texture_gathered_extent_no_copy,
        render_vertex_interface_superset,
        render_texture_narrow_lanes,
    ) {
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
    // R32: the attachment's own guest-run seed is read through the owner's
    // imported mappings, so a device that cannot import host pointers has no
    // list arm at all — the same capability the window-backed binds ask, read
    // here for the same reason (a class answer, not a decline: the engine
    // gathers these runs on its own). The runs themselves carry no alignment
    // rule — the provider gathers them with the host, so a run's own offset
    // never becomes a view pointer a driver has to accept.
    if pass.load_seed_runs().is_some() {
        let alignment = match declared_host_import() {
            Ok(alignment) => alignment,
            Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
        };
        if alignment == 0 {
            let reason = OutOfClass::new(
                "render_provider_out_of_class_load_seed_import",
                "a record whose previous contents are the surface's own guest bytes stays on the \
                 engine on a device that cannot import host pointers: the canonical attachment \
                 states them as an ordered list of owner windows, and the owner rail refuses a \
                 lease-backed declaration on such a device rather than silently copying it",
            );
            reason.note();
            return RenderRailOutcome::NotInNarrowClass(reason);
        }
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
    // R36: the pass's runtime `[[sampler(n)]]` states under a frame. R12's
    // accounting kept every such pass on the engine because the frame of that
    // increment could not carry the list; E-TX4 taught the codec to carry it
    // (`PASS_KIND_RENDER_SAMPLERS`'s four tags) and the pinned provider has
    // carried it since `7d544d4`, so what is left is the device's own answer —
    // asked here, out of the capability frame, rather than stated by the pure
    // gate. The ask is the render-sampler section R28 reads too, because the
    // pass's states are the half of a pairing whose other half is that
    // section's texture declarations: a device that does not state the section
    // executes neither, and keeping the draw under R12's own bucket is what
    // makes the census count the population that moved rather than a new slug.
    //
    // The order matters: this answer is asked before the texture one below, so
    // a shape that is both (every runtime-sampler pass is, by the family's own
    // pairing rule) keeps the bucket its own fact owns.
    if !pass.runtime_samplers.is_empty() && pass.crosses_the_frame() {
        let carried = match declared_render_sampler_carriage() {
            Ok(carried) => carried,
            Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
        };
        if !carried {
            let states = pass.runtime_samplers.len();
            let reason = OutOfClass::owned(
                "render_provider_out_of_class_texture_sampler_wire",
                format!(
                    "a draw whose runtime `[[sampler(n)]]` states would travel the owner→provider \
                     frame stays on the engine when the provider's own capability answer does not \
                     carry the shape: the frame declares \
                     supports_render_texture_sampling=false, so a frame that does not carry the \
                     pass's runtime sampler list decodes to a pass that states no sampler — the \
                     provider would refuse the declaration by name \
                     (`render_runtime_sampler_missing`) rather than execute a pass that samples \
                     through a state nobody stated — {states} runtime sampler state(s) beside a \
                     declared stage buffer or a window-backed stream are that shape",
                ),
            );
            reason.note();
            return RenderRailOutcome::NotInNarrowClass(reason);
        }
    }
    // R28: the pass's sampled textures under a frame. Until E-TX4 the frame's
    // render contract carried no texture declarations — the codec had a kind
    // for the compute half's and none for the render half's — so the pure gate
    // kept every sampled pass that crossed the frame on the engine
    // (`..._texture_wire`). The frame now carries them, and this asks the frame
    // rather than assuming either answer: a provider whose capability answer
    // does not hold the render-sampler section, or one whose section covers
    // fewer textures (or other formats) than this pass declares, keeps the draw
    // on the engine under the same bucket. Every texture this class admits is
    // in the section's own format family — the two four-byte 8-bit UNORM byte
    // orders, plus the two narrow lanes R39 admits *only* while this same
    // frame lists them (`declared_render_texture_narrow_lanes`) — so the
    // format half is a check and not a widening.
    if !pass.textures.is_empty() && pass.crosses_the_frame() {
        let support = match declared_render_texture_support() {
            Ok(support) => support,
            Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
        };
        let declared = pass.textures.len();
        let formats_covered = pass
            .textures
            .iter()
            .all(|texture| support.formats.contains(&texture.format));
        if !support.supported || declared > support.maximum as usize || !formats_covered {
            let reason = OutOfClass::owned(
                "render_provider_out_of_class_texture_wire",
                format!(
                    "a draw whose {declared} sampled texture(s) would travel the owner→provider \
                     frame stays on the engine when the provider's own capability answer does not \
                     carry the shape: the frame declares \
                     supports_render_texture_sampling={} and max_render_textures={} over {} \
                     format(s), and a pass above either bound — or one binding a format the \
                     section does not list — is a declaration the provider admits no view for \
                     (`render_texture_unsupported` / `render_texture_limit` / \
                     `render_texture_format_unsupported`) rather than one this rail states",
                    support.supported,
                    support.maximum,
                    support.formats.len(),
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
    // R28: the sampled textures' window arm, on the same device answer and in
    // the same order — after the binds above, so a copy the granules forced on
    // one of *them* is already in `copies` and cannot be read as a borrowed
    // window below.
    //
    // The second question a texture's window asks is the reservation's own
    // start. One registration's windows share one lease whose reservation
    // covers their union (`provider_owner::plan`), and E reads a texture at
    // that reservation's start, so a texture whose window is not the earliest
    // window of its own registration in this pass is a copy rather than a
    // borrow — the shape [`texture_window_arm`] states. The walk below keeps
    // the same two arms and the same named refusals, and it is also what gives
    // the census one reading per arm: a borrowed sampled window and a copied
    // one are two routes, not one silent choice.
    if pass.texture_windows().next().is_some() {
        let alignment = match declared_host_import() {
            Ok(alignment) => alignment,
            Err(decline) => return RenderRailOutcome::ProviderDeclined(decline),
        };
        // The earliest borrowed window of each registration this pass will
        // state, read from the binds the walk above has already decided.
        let mut reservation_starts: std::collections::BTreeMap<u64, u64> =
            std::collections::BTreeMap::new();
        for (_, binding, window) in window_backed() {
            if copies.bytes(binding).is_some() {
                continue;
            }
            let start = reservation_starts.entry(window.import).or_insert(u64::MAX);
            *start = (*start).min(window.host_va);
        }
        for texture in &pass.textures {
            let NarrowTextureSource::Window { binding, window } = &texture.source else {
                continue;
            };
            let starts_at_window = reservation_starts
                .get(&window.import)
                .is_none_or(|start| *start >= window.host_va);
            match texture_window_arm(*binding, *window, alignment, starts_at_window) {
                Ok(Some(bytes)) => {
                    crate::runtime::drain::note_store_route(
                        "render_provider_sampled_window_staged",
                    );
                    crate::runtime::drain::note_store_route_n(
                        "render_provider_sampled_window_bytes",
                        u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                    );
                    copies.insert(*binding, bytes);
                }
                Ok(None) => {
                    crate::runtime::drain::note_store_route(
                        "render_provider_sampled_window_borrowed",
                    );
                    // A borrowed texture is one more window of its
                    // registration: the next texture of that import reads the
                    // union this one just joined.
                    let start = reservation_starts.entry(window.import).or_insert(u64::MAX);
                    *start = (*start).min(window.host_va);
                }
                Err(reason) => {
                    reason.note();
                    return RenderRailOutcome::NotInNarrowClass(reason);
                }
            }
        }
    }
    // R36: the sampled gathers whose guest rows are padded. The one arm the
    // class states for those bytes is a copy — the contract's window arms name
    // a *tightly packed* extent at the reservation's own start, and what the
    // reservation holds here is the guest's rows with their padding — so the
    // class makes it here, where the bytes behind a registered window are
    // readable, and states the texture's own extent through the same trace-owned
    // arm every other copy this rail read out of a registry takes
    // (`TextureSource::OwnedBytes`).
    //
    // The read goes through the owner rail's registration rather than the
    // gather's own host runs, for the reason the R18/R28 staging copies do:
    // `provider_owner::window_bytes` is the one reader that answers whether
    // these bytes are still this process's to read (the import's registration
    // under the current epoch), and the copy is made before the submission that
    // names it. No device *capability* is asked, because no lease is minted for
    // this arm: a device that cannot import host pointers executes it exactly
    // as this one does.
    let mut rows = RowCopies::default();
    for texture in &pass.textures {
        let NarrowTextureSource::Depadded {
            window,
            rows: layout,
        } = &texture.source
        else {
            continue;
        };
        // The padded span, read out of the registration the window names. A
        // window this rail cannot read keeps the draw on the engine *by name* —
        // a decline is not a fallback, and a shape the class cannot state is
        // not one the engine's own answer may be inferred from.
        let padded = match provider_owner::window_bytes(owner_window(
            texture_owner_binding(texture.index),
            *window,
        )) {
            Ok(bytes) => bytes,
            Err(decline) => {
                let reason = OutOfClass::owned(
                    "render_provider_out_of_class_texture_source",
                    format!(
                        "a draw whose `[[texture({})]]` texels are gathered from a guest \
                             window with padded rows stays on the engine when the window's bytes \
                             cannot be copied out of the registration that names it: the class \
                             states the texture's tightly packed extent as the trace's own bytes, \
                             and the rows this rail repacks it from are the registration's own \
                             (`{}`)",
                        texture.index,
                        decline.slug(),
                    ),
                );
                reason.note();
                return RenderRailOutcome::NotInNarrowClass(reason);
            }
        };
        // The repack's own shape check: the registration read has to be the
        // span the guest's stride and row count tile, or the window this
        // declaration names is not the one the gather measured. A refusal here
        // is the same answer as the pure gate's — the shape is not this class's
        // — and it is never a shorter copy.
        let Some(tight) = layout.depad(&padded) else {
            let reason = OutOfClass::owned(
                "render_provider_out_of_class_texture_source",
                format!(
                    "a draw whose `[[texture({})]]` texels are gathered from a guest window with \
                     padded rows stays on the engine when the window does not hold the span the \
                     guest's own stride and row count tile: the gather states {} byte(s) of \
                     stride over {} row(s) ending in a {} byte tight row ({} byte(s)), while the \
                     registration handed back {} byte(s)",
                    texture.index,
                    layout.stride,
                    layout.rows,
                    layout.tight_row,
                    layout.span(),
                    padded.len(),
                ),
            );
            reason.note();
            return RenderRailOutcome::NotInNarrowClass(reason);
        };
        // The reading a boot's census takes: one route per repacked texture,
        // the bytes the copy kept beside the bytes it dropped, so the two halves
        // of the arm are countable from the log rather than inferred from a
        // frame that looks right.
        crate::runtime::drain::note_store_route("render_provider_sampled_rows_depadded");
        crate::runtime::drain::note_store_route_n(
            "render_provider_sampled_rows_bytes",
            u64::try_from(tight.len()).unwrap_or(u64::MAX),
        );
        crate::runtime::drain::note_store_route_n(
            "render_provider_sampled_rows_padding",
            layout.padding(),
        );
        rows.insert(texture.index, tight);
    }
    match submit_narrow(inputs, req, &pass, &copies, &rows) {
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
/// and the door's four buckets are the same draws.
///
/// The bands break at four — the ceiling the *first* increment stated, kept
/// here as a fixed census breakpoint rather than re-read from
/// [`MAX_RENDER_STAGE_BUFFERS`] — so the arms either side of it stay comparable
/// across the widening E-SB1 landed (`research/docs/23` §108, which moved the
/// contract's own ceiling to eight). The arm above four is therefore the
/// population the widening *split*: the lists it admits (5..=8, which this
/// file's `a_widened_stage_buffer_declaration_leaves_for_the_provider_and_
/// agrees_with_the_engine` lands) beside the ones still past the contract's
/// ceiling, and [`stage_buffer_shape_route`]'s own `gt` route is what tells the
/// two apart in the same census.
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
/// The refusal slug is one name four rules answer under — more declarations
/// than the canonical contract states ([`MAX_RENDER_STAGE_BUFFERS`]), one
/// `(stage, index)` declared twice, one index read from both stages (R31, the
/// canonical rail's merged set 0 folds the two Metal namespaces onto one
/// descriptor), and a vertex declaration inside the canonical layout's own
/// `0..vertex_streams` bindings — and until this
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
        StageBufferShapeRoute::Folded => "stage_buffer_shape_folded",
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
    /// One index is read from both stages, which the canonical rail's merged
    /// set 0 folds onto one descriptor slot (R31).
    Folded,
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

/// The census route of one `resident_source` refusal (R34).
///
/// The refusal slug is one name three doors answer under, and census v24
/// (`evidence/gate3-census-v24-2026-09-18`) counted 646 records (11.3 % of that
/// boot's draws) under the bare slug while nothing in the log could say which
/// door had produced them: the recon that precedes this increment could only
/// decompose the bucket arithmetically (`gvaseed_elided` 330 + the mapper-ref
/// elision's unlanded half 316 = 646), because no counter separates the two.
///
/// That separation is the point, because the populations are priced completely
/// differently. A record whose *serialized* chain the caller failed to hand
/// over is one read-side repair away (fold the order, carry the wider texel,
/// wait for the readback, agree on the extent), while both LOAD elisions fail
/// here *by design*: their whole purpose is not to pay the frame copy this
/// refusal is about, so answering them means reversing a policy rather than
/// fixing a read.
///
/// Each arm charges its own route beside the refusal, [`resident_source_route`]
/// names it, `note_store_route` counts it, and
/// [`resident_source_route_bytes`] prices it. The sentence every arm answers
/// with is unchanged: this increment moves no admit/refuse edge.
pub fn resident_source_route(route: ResidentSourceRoute) -> &'static str {
    match route {
        ResidentSourceRoute::ChainOrderMismatch => "resident_source_chain_order_mismatch",
        ResidentSourceRoute::ChainGeometryMismatch => "resident_source_chain_geometry_mismatch",
        ResidentSourceRoute::ChainReadUnavailable => "resident_source_chain_read_unavailable",
        ResidentSourceRoute::ChainTexelUnavailable => "resident_source_chain_texel_unavailable",
        ResidentSourceRoute::GvaElision => "resident_source_gva_elision",
        ResidentSourceRoute::MapperRefNoLanding => "resident_source_mapper_ref_no_landing",
        ResidentSourceRoute::MapperRefIdentityMismatch => {
            "resident_source_mapper_ref_identity_mismatch"
        }
        ResidentSourceRoute::MapperRefFrameUnavailable => {
            "resident_source_mapper_ref_frame_unavailable"
        }
        ResidentSourceRoute::Undeclared => "resident_source_undeclared",
        // B1-B3: the elision doors' own window, when the door applied and the
        // record's attachment could not be declared as it.
        ResidentSourceRoute::WindowUnregistered => "resident_source_window_unregistered",
        ResidentSourceRoute::WindowUnwindowed => "resident_source_window_unwindowed",
        ResidentSourceRoute::WindowPaddedRows => "resident_source_window_padded_rows",
        ResidentSourceRoute::WindowExtent => "resident_source_window_extent",
        ResidentSourceRoute::WindowRegistrations => "resident_source_window_registrations",
        ResidentSourceRoute::WindowGeometry => "resident_source_window_geometry",
        ResidentSourceRoute::WindowIdentityMoved => "resident_source_window_identity_moved",
        ResidentSourceRoute::WindowLandingRefused => "resident_source_window_landing_refused",
        ResidentSourceRoute::WindowSpanUnmapped => "resident_source_window_span_unmapped",
    }
}

/// The byte price of one [`resident_source_route`], charged beside it.
///
/// One name per route rather than one total, because the two families the
/// bucket mixes sit at the two ends of the range: the mapper-ref-texture
/// composites are 1280x1024 (5 MiB a frame) while the chained GVA records are
/// 64..186 wide (16..74 KiB), so a single sum would be dominated by whichever
/// population happens to be larger. The number is the *whole* attachment the
/// record's previous contents are ([`NarrowPass::extent`], the same length the
/// byte arms have to carry), which is what a readback — or any other witness
/// standing in for one — would have to move.
pub fn resident_source_route_bytes(route: ResidentSourceRoute) -> &'static str {
    match route {
        ResidentSourceRoute::ChainOrderMismatch => "resident_source_chain_order_mismatch_bytes",
        ResidentSourceRoute::ChainGeometryMismatch => {
            "resident_source_chain_geometry_mismatch_bytes"
        }
        ResidentSourceRoute::ChainReadUnavailable => "resident_source_chain_read_unavailable_bytes",
        ResidentSourceRoute::ChainTexelUnavailable => {
            "resident_source_chain_texel_unavailable_bytes"
        }
        ResidentSourceRoute::GvaElision => "resident_source_gva_elision_bytes",
        ResidentSourceRoute::MapperRefNoLanding => "resident_source_mapper_ref_no_landing_bytes",
        ResidentSourceRoute::MapperRefIdentityMismatch => {
            "resident_source_mapper_ref_identity_mismatch_bytes"
        }
        ResidentSourceRoute::MapperRefFrameUnavailable => {
            "resident_source_mapper_ref_frame_unavailable_bytes"
        }
        ResidentSourceRoute::Undeclared => "resident_source_undeclared_bytes",
        ResidentSourceRoute::WindowUnregistered => "resident_source_window_unregistered_bytes",
        ResidentSourceRoute::WindowUnwindowed => "resident_source_window_unwindowed_bytes",
        ResidentSourceRoute::WindowPaddedRows => "resident_source_window_padded_rows_bytes",
        ResidentSourceRoute::WindowExtent => "resident_source_window_extent_bytes",
        ResidentSourceRoute::WindowRegistrations => "resident_source_window_registrations_bytes",
        ResidentSourceRoute::WindowGeometry => "resident_source_window_geometry_bytes",
        ResidentSourceRoute::WindowIdentityMoved => "resident_source_window_identity_moved_bytes",
        ResidentSourceRoute::WindowLandingRefused => "resident_source_window_landing_refused_bytes",
        ResidentSourceRoute::WindowSpanUnmapped => "resident_source_window_span_unmapped_bytes",
    }
}

/// Which of [`resident_source_route`]'s doors and misses a record answered.
///
/// An enum rather than a `&'static str` at the call sites for the reason the
/// routes themselves exist: arms that answer with different census names are
/// different facts, and a typo in a bare string would file two of them under
/// one name with nothing failing. The seam (`runtime::draw::vulkan`) is the
/// only producer — it is the only place the door is known — and the class's one
/// refusal is the only charger, so a route can only be counted for a record
/// this class actually refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentSourceRoute {
    /// The serialized packet chain named the engine's registry resident
    /// (`render_chain_identity`), and the caller's readback declined because
    /// the attachment's own texel order and the identity's are not one value —
    /// or because the request states no colour attachment at all, which the
    /// class refuses earlier by name (`render_provider_out_of_class_
    /// attachment_state`) and which therefore cannot be charged here.
    ChainOrderMismatch,
    /// The same door, refused because the identity's extent is not the
    /// attachment's (`identity.width/height != req.width/height`).
    ChainGeometryMismatch,
    /// The same door, refused because the registry read itself failed
    /// (`read_target`) — the ready/content question, not a shape question.
    ChainReadUnavailable,
    /// The same door, refused because the frame read back is wider than the
    /// four-byte colour this arm narrows (an `Rgba16Float` attachment, or any
    /// other resident the class would have to quantize to carry).
    ChainTexelUnavailable,
    /// The GVA LOAD elision chained (`honour_gva_load_elision`): the engine
    /// still holds the frame and its witness is a readiness flag no in-tree API
    /// can clear, so this door deliberately hands nothing over.
    GvaElision,
    /// The mapper-ref-texture composite's LOAD elision fired for a record whose
    /// own frame does *not* land in the mapping's guest pages (`writeback_guest`
    /// clear), so there is no landing to advance the `surface_content_epoch`
    /// the elision's currency test reads — the door hands nothing over.
    MapperRefNoLanding,
    /// The elision fired and the record does land, but the identity the elision
    /// returned is no longer the one the record's own attachment names.
    MapperRefIdentityMismatch,
    /// The elision fired, the identities agree, and the read itself declined
    /// (order, geometry, readiness, or a texel wider than four-byte colour —
    /// the four [`Self::ChainOrderMismatch`] and siblings name for the chain
    /// door).
    MapperRefFrameUnavailable,
    /// A record reached this refusal on a `load_from_target` chain that no door
    /// the seam knows produced. Unreachable by construction (three sites set
    /// it), and named rather than folded into a neighbour so that a fourth door
    /// which forgets to state its route reads as a number in the census instead
    /// of quietly inflating one.
    Undeclared,
    /// B1-B3: the door applied and its window's pages have no registration this
    /// process can import (`WindowRefusal::NoAlias`, or a source whose
    /// `pages` came back empty), so there is no owner window to read or land
    /// through.
    WindowUnregistered,
    /// The same door, one stretch of whose window has no registered window.
    WindowUnwindowed,
    /// The same door on a surface whose rows are padded (`bufferRowLength`):
    /// the byte span the window tiles is not the attachment's tightly packed
    /// extent, and the contract's run list carries no stride
    /// (`research/docs/26` §Rn — the follow-up change
    /// `render-attachment-window-stride` is what would lift this).
    WindowPaddedRows,
    /// The same door whose window's span is not the attachment's extent.
    WindowExtent,
    /// The same door whose runs live in more than one registration, a shape one
    /// declaration cannot state (`render-guest-run-multi-allocation`).
    WindowRegistrations,
    /// The same door whose mapping/GVA geometry or texel format is not the
    /// attachment's own.
    WindowGeometry,
    /// The same door whose `map_generation` / GVA generation moved between the
    /// declaration and the submission.
    WindowIdentityMoved,
    /// The same door whose owed frame could not be landed: the guest wrote over
    /// it, or the rail refused the Store, so the pages do *not* hold the content
    /// the elision read.
    WindowLandingRefused,
    /// The same door whose task GVA span has an unmapped page.
    WindowSpanUnmapped,
}

/// Charge one `resident_source` arm's route, and its byte price, beside the
/// refusal it answers.
#[inline]
fn note_resident_source(route: ResidentSourceRoute, extent: u64) {
    crate::runtime::drain::note_store_route(resident_source_route(route));
    crate::runtime::drain::note_store_route_n(resident_source_route_bytes(route), extent);
}

/// The census route of one `texture_extent` refusal (R35).
///
/// Census v25b (`evidence/gate3-census-v25b-2026-09-18`) reads 10 368 of that
/// boot's 30 373 seam first-failures (34.1 %) under the bare
/// `render_provider_out_of_class_texture_extent` slug, and nothing in the log
/// says which source those records named — while that is the one question the
/// bucket's next decision turns on, because E's two rails are not of one mind
/// about the shape. The Vulkan rail gathers a source of another extent into the
/// render area's own grid and keeps `render_texture_extent_unsupported` for the
/// owner's no-copy window alone (`research/docs/23` §111, E-TX5); the
/// `metal-api-native` rail refuses *every* source of another extent by that
/// same name. A bucket that is mostly host bytes is a widening the Vulkan rail
/// can already execute and the class has not followed; a bucket that is mostly
/// the no-copy window is a shape this fork's refusal already agrees with the
/// provider about.
///
/// Each arm charges its own route beside the refusal, [`texture_extent_route`]
/// names it, and `note_store_route` counts it. The refusal itself moves in this
/// increment only in its sentence, which used to claim the provider refuses
/// another extent by name — a rule the Vulkan rail stopped having at E-TX5 —
/// and now states what the two rails answer; the slug, the gate's position and
/// its verdict are unchanged.
pub fn texture_extent_route(route: TextureExtentRoute) -> &'static str {
    match route {
        TextureExtentRoute::BorrowedNoCopy => "texture_extent_borrowed_no_copy",
        TextureExtentRoute::HostBytes => "texture_extent_host_bytes",
    }
}

/// Which of [`texture_extent_route`]'s two arms one refused draw's sampled
/// texture names.
///
/// The arms are E's question rather than this rail's: the one source the
/// Vulkan rail cannot gather into the render area's own grid is the owner's
/// no-copy window (`RenderInputSource::Borrowed` is its only arm with no host
/// bytes), and that is the arm [`texture_window_arm`] leaves at the owner's own
/// mapping — the window whose first byte *is* the texture's, which is that
/// function's own first condition (`head == 0`). Every other source is bytes a
/// rail holds and can gather: the request's own copy, the caller's frame out of
/// the engine's registry (R24), the trace's own production (R22), and the
/// window whose extent does not start at the window's first byte, which the
/// class gate copies out of the registration.
///
/// One residual, named rather than hidden: [`texture_window_arm`]'s second
/// condition is about the *submission* — a window that is not its
/// registration's earliest in this pass is copied too — and this gate runs
/// before the class gate's window walk, so a `head == 0` window that another
/// bind of the same registration precedes is counted here under the borrowed
/// arm while the class gate would have copied it. The direction is the one that
/// matters for the bucket's question: a window whose `head` is not zero is
/// *always* copied, so `texture_extent_host_bytes` is never charged for a
/// source E's Vulkan rail refuses — the borrowed arm reads as an upper bound
/// and the host-bytes arm as the matching lower bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureExtentRoute {
    /// The owner's no-copy window: the one arm E's Vulkan rail refuses another
    /// extent for, and the one this fork's refusal already agrees with.
    BorrowedNoCopy,
    /// Every source a rail reads off the host — the request's own copy, the
    /// caller's frame, the trace's production, and the window the class gate
    /// copies — which the Vulkan rail gathers into the render area's grid.
    HostBytes,
}

/// The arm one refused draw's sampled source would be declared under, as the
/// gate that answers it can read it.
///
/// One constructor for the two arms ([`texture_extent_route`]'s own reason for
/// being an enum): the extent gate, which is the refusal's only charger.
fn texture_extent_arm(source: &NarrowTextureSource<'_>) -> TextureExtentRoute {
    match source {
        // The window whose first byte is the texture's: the only shape
        // [`texture_window_arm`] can leave at the owner's own mapping, and so
        // the only source whose declaration can name
        // `TextureSource::BorrowedNoCopy` — the arm the Vulkan rail has no host
        // bytes to gather another extent from.
        NarrowTextureSource::Window { window, .. } if window.head == 0 => {
            TextureExtentRoute::BorrowedNoCopy
        }
        // Everything else carries bytes a rail can read: the request's own
        // copy, the caller's frame out of the engine's registry (R24), the
        // trace's own production (R22), and every window the class gate copies
        // because the texture's extent does not start at its first byte.
        _ => TextureExtentRoute::HostBytes,
    }
}

/// Charge one `texture_extent` arm's route beside the refusal it answers.
#[inline]
fn note_texture_extent(route: TextureExtentRoute) {
    crate::runtime::drain::note_store_route(texture_extent_route(route));
}

/// The census route of one `vertex_interface` refusal (R-VI1).
///
/// The refusal slug is one name three directions answer under, and until this
/// increment the log could not say which. Census v25b
/// (`evidence/gate3-census-v25b-2026-09-18`) counted 3 894 rows (12.8 % of that
/// boot's seam rows) under the bare slug, and every one of the shapes it
/// latched read `attrs=4 streams=3` — a field pair that says how many
/// attributes a draw declared and how many fetch tables they read, and nothing
/// at all about *which side* named a location the other does not. The recon
/// that precedes this increment could not recover the direction from the
/// evidence, so it could not say whether the bucket was winnable.
///
/// That question is the whole increment, because the two directions are priced
/// completely differently:
///
/// * a **declared** superset is a shape Metal answers. `MTLVertexDescriptor` is
///   free to name an attribute location the vertex function never reads; the
///   canonical rail's own vertex input state is built from the declared layout
///   (`metal_api_vulkan`'s `create_vertex_inputs` emits one
///   `VkVertexInputAttributeDescription` per declared attribute) exactly as
///   Metal's descriptor is, so the surplus stream is *bound and ignored*
///   rather than undefined. Admitting it is a widening of one rule in two
///   places — the canonical registration gate and this rail's mirror of it —
///   and it needs no wire change, because the request's whole declared list
///   already travels;
/// * a **reflected** superset is a location the vertex stage *reads* that no
///   declared entry covers. Metal defines no value for it, so no oracle can
///   state what the frame should hold, and the refusal there is a permanent
///   boundary rather than a backlog item.
///
/// The residual is named rather than folded into a neighbour for the reason
/// [`ResidentSourceRoute::Undeclared`] is: a shape whose two sides disagree on
/// both faces at once, and one location declared twice, are not either
/// direction, and filing them under one would make the census's own reading of
/// "declared ⊋ reflected" false rather than merely incomplete.
///
/// Each arm charges its own route beside the refusal, [`vertex_interface_route`]
/// and [`vertex_interface_route_distance`] name the two keys,
/// `note_store_route` counts them, and the sentence the refusal answers with is
/// unchanged: this increment moves no admit/refuse edge.
pub fn vertex_interface_route(route: VertexInterfaceRoute) -> &'static str {
    match route {
        VertexInterfaceRoute::DeclaredSuperset => "vertex_interface_declared_superset",
        VertexInterfaceRoute::ReflectedSuperset => "vertex_interface_reflected_superset",
        VertexInterfaceRoute::LocationMismatch => "vertex_interface_location_mismatch",
    }
}

/// The distance one [`vertex_interface_route`] states beside it, charged as its
/// own sum.
///
/// One name per route rather than one total, and a *number* rather than only a
/// count, because the count alone cannot say how wide the disagreement was: a
/// four-attribute draw that declares one location its stage never reads and a
/// two-attribute draw that declares one are one record each and no distance at
/// all. Dividing the sum by the count is what turns the census line into "how
/// many locations per refused record", which is the number a widening order
/// sizes on — and the two routes' numbers are not the same quantity, so they
/// cannot share a counter.
///
/// The `_locations` suffix names the unit, as `_bytes` does for
/// [`resident_source_route_bytes`]. A route the walk did not charge a distance
/// for prints nothing, which [`VertexInterfaceMismatch`] states is itself a
/// reading.
pub fn vertex_interface_route_distance(route: VertexInterfaceRoute) -> &'static str {
    match route {
        VertexInterfaceRoute::DeclaredSuperset => {
            "vertex_interface_declared_superset_extra_locations"
        }
        VertexInterfaceRoute::ReflectedSuperset => {
            "vertex_interface_reflected_superset_uncovered_locations"
        }
        VertexInterfaceRoute::LocationMismatch => {
            "vertex_interface_location_mismatch_unpaired_locations"
        }
    }
}

/// Which direction a request's declared attributes and the vertex stage's own
/// reflection disagreed in.
///
/// An enum rather than a `&'static str` at the call site for the reason the
/// routes themselves exist: directions that answer with different census names
/// are different facts, and a typo in a bare string would file two of them
/// under one name with nothing failing. The gate is the only constructor, one
/// arm per direction, so a route can only be counted for a record this class
/// actually refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VertexInterfaceRoute {
    /// Every location the vertex stage reads is declared, and at least one
    /// declared entry names a location it does not read — `declared ⊋
    /// reflected`. Metal ignores the surplus, so this is the direction a
    /// widening can admit.
    DeclaredSuperset,
    /// Every declared entry names a location the vertex stage reads, and the
    /// stage reads at least one location no declared entry covers —
    /// `reflected ⊋ declared`. Metal leaves such a location undefined, so this
    /// is the direction no contract can admit.
    ReflectedSuperset,
    /// Neither: the two sides each name a location the other does not, or one
    /// location is declared twice (which the length walk catches and the
    /// reflection cannot mirror). Neither face is a widening candidate on its
    /// own terms, and the distance beside this route is `0` for exactly the
    /// duplicate shape.
    LocationMismatch,
}

/// Charge one `vertex_interface` arm's route, and the distance it states,
/// beside the refusal it answers.
#[inline]
fn note_vertex_interface(mismatch: VertexInterfaceMismatch) {
    crate::runtime::drain::note_store_route(vertex_interface_route(mismatch.route));
    crate::runtime::drain::note_store_route_n(
        vertex_interface_route_distance(mismatch.route),
        mismatch.distance,
    );
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

/// Where one admitted texture's texels come from (R10/R22/R24).
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
    /// The frame the caller read out of the registry that holds a sampled GPU
    /// target (R24), declared in the same place in the trace as
    /// [`Self::Bytes`]: the declaration states `TextureSource::OwnedBytes` over
    /// exactly those bytes, because that is what they are — a copy the trace
    /// carries, read out of the image the engine's own bind would have sampled.
    /// The variant exists so the arm that answered is countable at the
    /// completion (the census reads its own positive counter beside R23's), and
    /// so a pass that samples one is still restatable as a production: its
    /// descriptor carries the bytes, not a lease.
    Frame(&'a [u8]),
    /// The registered guest RAM window the bind's own zero-copy gather was cut
    /// from (R28): the texels live in the guest's pages, and the run carries
    /// the provider-shaped window the registration ledger derived for it — the
    /// fact R9q/R11 read for a vertex and an index stream. How those bytes
    /// travel is the *device's* answer and is decided in
    /// [`texture_window_arm`]: the borrowed no-copy window when the texture's
    /// extent starts at the reservation's own first byte, and the owner's
    /// staged copy of exactly the extent otherwise. Both are leases, so a pass
    /// that samples one crosses the owner→provider frame and its declaration
    /// names the lease the plan minted for this label.
    Window {
        /// The label the owner plan states this texture's window under
        /// ([`texture_owner_binding`]).
        binding: u32,
        /// The window, with the texture's own coordinates inside it.
        window: StageBufferWindow,
    },
    /// The registered guest RAM window the bind's own zero-copy gather was cut
    /// from when the guest's rows are padded (R36): the same one page run the
    /// [`Self::Window`] arm names, with a `bufferRowLength` the guest's own
    /// translation stated.
    ///
    /// The window arms cannot state it, either of them: a lease names the
    /// texture's tightly packed extent *at the reservation's own start*, and
    /// what the reservation holds here is the guest's rows with their padding.
    /// The class gate therefore repacks the rows into the texture's own extent
    /// — one row at a time, out of the registration the window names
    /// ([`RowCopies`]) — and the declaration states those bytes the way it
    /// states every other copy this rail read out of a registry (the request's
    /// own copy, R24's frame): as the trace's own bytes
    /// (`TextureSource::OwnedBytes`). No lease is minted for this arm, so a
    /// pass whose only guest-backed bind is one of these keeps the in-process
    /// path.
    ///
    /// `rows` is the stride the guest stated beside the tight row and row count
    /// the texture's own view states, so the repack is total: the source has
    /// been read, and its span measured against the stride, before this arm
    /// exists.
    Depadded {
        /// The window the rows live in, with the texture's own coordinates
        /// inside it. Its `bytes_len` is the guest's padded span, the number of
        /// bytes [`RowCopies`] read out of the registration.
        window: StageBufferWindow,
        /// The guest stride, the tight row and the row count.
        rows: PaddedRows,
    },
}

impl NarrowTextureSource<'_> {
    /// The bytes this source carries when it is one of the trace-owned arms
    /// (R36): the request's own copy, the caller's frame out of the engine's
    /// registry, or the tightly packed rows the class gate repacked out of a
    /// padded gather.
    ///
    /// One function for the two readers that have to agree about a texture's
    /// byte count — [`input_allocations`] when the view's allocation is minted
    /// and the declaration when the bytes are stated — so a padded gather's
    /// copy cannot be sized by one and stated by the other.
    fn owned_bytes<'a>(&'a self, rows: &'a RowCopies, index: u32) -> Option<&'a [u8]> {
        match self {
            Self::Bytes(bytes) | Self::Frame(bytes) => Some(bytes),
            Self::Depadded { .. } => rows.bytes(index),
            Self::Produced { .. } | Self::Window { .. } => None,
        }
    }
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
    /// (E-TX1, `research/docs/23` §107; the narrow lanes since R39): one of the
    /// two four-byte 8-bit UNORM byte orders, or one of the one- and two-byte
    /// lanes the device's own frame lists beside them. The declaration, the
    /// view and the byte count all read it, so the bytes and the name they are
    /// uploaded under cannot drift apart.
    format: TextureFormat,
    /// The sampler form the declaration states, or the sampler-free fetched
    /// arm (R15).
    sampler: NarrowSampler,
    /// Where the texels come from: the request's own tightly packed copy —
    /// at the texel width [`Self::format`] names, in the byte order it names —
    /// or the trace's own production of the identity the bind resolved to
    /// (R22).
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
    /// Whether this pass's previous contents are the seed door's own bytes,
    /// handed over by the caller (R32). Counted as
    /// `render_provider_load_seed_bytes` beside the three arms above, for the
    /// same reason they are three numbers: the census reads populations, and
    /// this door's caller is the request builder rather than a registry read.
    carried_load_seed_bytes: bool,
    /// Whether this pass's previous contents are the frame of the surface the
    /// LOAD *elision* named, handed over by the caller that owns the registry
    /// (R26). Counted as `render_provider_surface_resident_source_bytes`
    /// beside the two arms above, for the same reason they are two numbers:
    /// the census reads populations, and the obligation the caller states with
    /// these bytes is not the one the other two state (see
    /// [`RenderRailInputs::surface_resident_source_bytes`]).
    carried_surface_resident: bool,
    /// Whether this pass's previous contents are the *elision door's own guest
    /// window* or the *seed door's own guest backing* rather than any of the
    /// byte arms above (B1-B3, R38). Counted as
    /// `render_provider_attachment_guest_window_bytes` at the completion,
    /// beside R32's `render_provider_load_seed_runs`: the declaration the two
    /// state is the same one (`BufferSource::GuestRuns`), so without this flag
    /// the census could not tell the seed door's population from the elision
    /// door's — which is the whole question this increment's readings turn on.
    carried_attachment_guest_window: bool,
    /// Whether that declaration is the **seed door's** own (`R38`), rather than
    /// the elision door's. Counted as
    /// `render_provider_seed_attachment_guest_window_bytes` beside the name
    /// above, for the reason that name exists: a census that read one number for
    /// both doors could see the window population move without knowing which
    /// door moved it, and the two doors are read for different reasons (the
    /// elision door's population is a policy that already holds; the seed
    /// door's is the door the class had no arm for at all).
    carried_seed_guest_window: bool,
    /// Whether at least one of this pass's sampled textures is the frame the
    /// caller read out of the registry (R24's arm). Counted as
    /// `render_provider_sampled_target_frames` at the completion, beside R23's
    /// own counter, for the same reason: the population is what the census
    /// reads, and a submission that never reached the caller is not an answer.
    sampled_target_frames: bool,
    /// Whether the vertex stage's whole layout is the canonical namespace
    /// layout rather than the translator's default (R33).
    ///
    /// Set when this request's statement is the folded pair *and* the provider
    /// declared that it arranges the two stages' buffer namespaces apart:
    /// [`narrow_class`] reads the pair's existence from the statement and the
    /// device's answer from the frame, and this field is where the two meet.
    /// The registration below translates the vertex half under
    /// [`metal_api_vulkan::stage_buffer_namespace_layout`] exactly when it is
    /// set, so the module's own descriptor layout and the pass's binds state
    /// one arrangement rather than two.
    vertex_stage_buffer_namespace_split: bool,
    bgra: bool,
    width: u64,
    height: u64,
    extent: u64,
    /// How many vertices' worth of primitives this draw names: the index count
    /// on the indexed arm and the request's own `vertex_count` on the
    /// non-indexed one (R39). One field rather than two, because the number
    /// below the gate is one number: it is the descriptor's `vertices`, and the
    /// pass's own shape, whichever arm named it.
    draw_count: u32,
    /// The admitted streams, in the request's own attribute order: entry `i`
    /// becomes canonical binding `i`. One entry per fetch table, and one
    /// attribute per *location* inside it: the engine's request numbers one
    /// Vulkan binding per attribute location, while a guest's interleaved
    /// stream is several attributes off one table — which is the table entry `i`
    /// states here, with each attribute at its own offset.
    vertex_streams: Vec<NarrowVertexStream<'a>>,
    /// The index stream, on the indexed arm alone: `None` is a draw that names
    /// its vertices directly (R39), which is the contract's own `indices: None`
    /// arm — read as `0..vertices`, with no index view, no index lease and no
    /// `baseVertex` to add.
    index_stream: Option<NarrowIndexStream<'a>>,
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
    /// its vertex streams, its index stream *when it has one* (R39: a draw that
    /// names its vertices directly declares no index view, so the pool is one
    /// view cheaper), its stage buffers and its sampled textures. The trace's
    /// own budget is
    /// [`MAX_SERIAL_RESOURCES`](metal_api_core::provider::MAX_SERIAL_RESOURCES)
    /// views, and a record that carries productions spends theirs beside this
    /// pass's — [`serial_views_admit`] is the one place that sum is compared.
    fn views(&self) -> usize {
        1 + self.vertex_streams.len()
            + usize::from(self.index_stream.is_some())
            + self.stage_buffers.len()
            + self.textures.len()
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
    /// At most one index stream per draw, so there is one window and no
    /// numbering: the canonical binding is the contract's own zero
    /// ([`IndexBufferBinding`]'s view is the pass's one index view), and the
    /// owner label is the namespace below. `None` covers both arms a window
    /// cannot be cut from — a staged index stream, and the whole non-indexed
    /// arm (R39), which has no index stream at all.
    fn index_window(&self) -> Option<StageBufferWindow> {
        match &self.index_stream.as_ref()?.source {
            StreamSource::Window(window) => Some(*window),
            StreamSource::Staged(_) => None,
        }
    }

    /// The attachment's own previous contents as the owner windows they live in
    /// (`R32`, E-TX6), for a pass whose load is the guest-runs seed.
    ///
    /// The list is stated under one label ([`load_seed_owner_binding`]), so a
    /// registration that backs both this seed and a stream is still one lease —
    /// the ordering, the per-run coordinates and the one-registration rule are
    /// all [`load_seed_run_windows`]'s.
    fn load_seed_runs(&self) -> Option<&[StageBufferWindow]> {
        match &self.load {
            NarrowLoad::GuestRuns(runs) => Some(runs.as_slice()),
            NarrowLoad::Clear(_) | NarrowLoad::Resident(_) | NarrowLoad::Bytes(_) => None,
        }
    }

    /// The windows the pass's own sampled textures were cut from (`R28`), one
    /// entry per texture that travels as a lease, under the owner label the
    /// plan and the trace both key it by.
    ///
    /// The fourth namespace of [`Self::vertex_windows`] and
    /// [`Self::index_window`]: the texture's Metal `[[texture(n)]]` index is
    /// what the label carries ([`texture_owner_binding`]), because a pass's
    /// sampled textures are arguments of their own rather than stage buffers.
    fn texture_windows(&self) -> impl Iterator<Item = (u32, StageBufferWindow)> + '_ {
        self.textures
            .iter()
            .filter_map(|texture| match &texture.source {
                NarrowTextureSource::Window { binding, window } => Some((*binding, *window)),
                _ => None,
            })
    }

    /// Whether this pass's trace crosses the owner→provider frame at all.
    ///
    /// One condition, three readers, all in `submit_render`: the frame's
    /// texture-declaration reading (R28), the runtime-sampler carriage reading
    /// (R36), and — through the plan — every "does this binding travel as a
    /// lease" question below. A pass that stays in process has no frame to ask
    /// about. It is exactly [`plan_owner_leases`]'s
    /// "no binding travels as a lease" predicate — a declared stage buffer, a
    /// window-backed stream or index stream, or a window-backed sampled texture
    /// each make one — spelled from the pass rather than from the plan so the
    /// pure gate can read it.
    fn crosses_the_frame(&self) -> bool {
        !self.stage_buffers.is_empty()
            || self.vertex_windows().next().is_some()
            || self.index_window().is_some()
            || self.texture_windows().next().is_some()
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

/// The span proof the non-indexed arm owes, and the index arm does not (R39).
///
/// A draw that names its vertices directly reads `0..vertices`, so three facts
/// about it are already stated somewhere outside this gate — and every one of
/// them is a *refusal* rather than a fallback:
///
/// - a pipeline that declares a vertex layout names at least
///   [`FULL_SCREEN_TRIANGLE_VERTICES`] vertices
///   (`ContractError::DrawVertexCountBelowMinimum`);
/// - a pipeline that declares none — the `vertex_id` shape, whose positions the
///   module generates — names exactly that many
///   (`ContractError::DrawVertexCountMismatch`);
/// - every per-vertex stream covers `vertices * stride`
///   (`render_vertex_buffer_footprint_unsupported` on the Vulkan rail,
///   `render_vertex_footprint_unsupported` on the native one — one proof spelled
///   twice, and the same proof the class's `index_staged` door already sends
///   `vertex_short` shapes down a stricter form of).
///
/// The first two are the contract's own shape rules, so a draw outside them is
/// one admission refuses; the third is the provider's own coverage proof, which
/// the *indexed* arm does not owe in this form: an index names the vertex it
/// reads, so that arm's proof is over `base_vertex + highest index + 1` rather
/// than over the whole count. Answering all three here is what keeps the class's
/// one promise — everything it admits is a shape the provider executes
/// (`provider_render.rs`'s own words at the vertex-interface door above) —
/// because an in-class refusal is a typed decline and never re-runs the engine
/// (`runtime/draw/vulkan.rs`): a shape this gate let through and the provider
/// refused would be a hard failure, not a fallback.
///
/// `None` is a shape the provider executes; `Some` is this class's own sentence
/// for one it would decline, which keeps the draw on the engine instead.
fn nonindexed_vertex_span(
    vertex_streams: &[NarrowVertexStream<'_>],
    draw_count: u32,
) -> Option<OutOfClass> {
    const ROUTE: &str = "render_provider_out_of_class_vertex_span";
    if vertex_streams.is_empty() {
        if draw_count == FULL_SCREEN_TRIANGLE_VERTICES {
            return None;
        }
        return Some(OutOfClass::owned(
            ROUTE,
            format!(
                "a non-indexed draw whose pipeline declares no vertex layout stays on the engine \
                 unless it names exactly the {FULL_SCREEN_TRIANGLE_VERTICES} vertices the \
                 canonical `vertex_id` shape carries: the contract refuses any other count by \
                 name (`DrawVertexCountMismatch`), and a refusal is a decline rather than a \
                 fallback, so this class does not hand admission a pass it can only lose (this \
                 draw names {draw_count})",
            ),
        ));
    }
    if draw_count < FULL_SCREEN_TRIANGLE_VERTICES {
        return Some(OutOfClass::owned(
            ROUTE,
            format!(
                "a non-indexed draw with a vertex layout stays on the engine when it names fewer \
                 than {FULL_SCREEN_TRIANGLE_VERTICES} vertices: the canonical contract refuses \
                 such a pass by name (`DrawVertexCountBelowMinimum`), and a refusal is a decline \
                 rather than a fallback (this draw names {draw_count})",
            ),
        ));
    }
    for (binding, stream) in vertex_streams.iter().enumerate() {
        // Saturating, exactly as the native rail's own copy of this proof is: an
        // unrepresentable product is by definition larger than any buffer this
        // provider admits, so the comparison only has to decide coverage.
        let required = u64::from(draw_count).saturating_mul(stream.stride);
        let carried = stream.source.len();
        if carried < required {
            return Some(OutOfClass::owned(
                ROUTE,
                format!(
                    "a non-indexed draw stays on the engine when a per-vertex stream does not \
                     cover every vertex it names: the draw reads 0..{draw_count}, so the stream \
                     at binding {binding} owes {required} bytes at stride {} and the bind \
                     carries {carried}, which the provider proves before it executes the draw \
                     (`render_vertex_buffer_footprint_unsupported` / \
                     `render_vertex_footprint_unsupported`) — a proof this class answers here \
                     because a decline on an in-class draw is fail-closed",
                    stream.stride,
                ),
            ));
        }
    }
    None
}

/// Whether one request is the narrow class, and the facts the trace is built
/// from when it is.
///
/// Pure over the request and this rail's own recorded state — plus the one
/// device answers the caller hands it (`stage_buffer_namespace_split`, R33, and
/// the extent rule's two arms, `render_texture_gathered_extent`, R37, and
/// `render_texture_gathered_extent_no_copy`, R40) — and ordered cheapest-first
/// so a refused shape costs nothing: no provider call, no translation, and no
/// registration. Every refusal names the condition that kept the shape on the
/// engine, because that string is what the observer reports when a class
/// boundary moves.
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
    stage_buffer_namespace_split: bool,
    render_texture_gathered_extent: bool,
    render_texture_gathered_extent_no_copy: bool,
    render_vertex_interface_superset: bool,
    render_texture_narrow_lanes: NarrowLanes,
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
    // Four arms, and each is one of the canonical attachment's own loads: the
    // *provider's* image under the attachment's own identity
    // (`LoadOp::Resident`), the caller's own bytes declared for that same view
    // (`LoadOp::Load`, R23/R25/R32), the guest's own pages declared as an
    // ordered window list (R32, E-TX6), or the guest's byte-exact clear. What
    // is left — a record whose previous contents are the attachment's own guest
    // backing — keeps the engine by name.
    let mut carried_chain_middle = false;
    // R32: the two seed doors, counted where the bytes are chosen so the census
    // reads one positive number per arm rather than only the bucket that moved.
    let mut carried_load_seed_bytes = false;
    // R26: which of the two registry-byte doors this record came in through, so
    // the completion counts the population under its own name. Set where the
    // bytes are chosen below; false for every arm that carries none.
    let mut carried_surface_resident = false;
    // B: whether this record's load is the elision door's own guest window
    // rather than one of the byte arms ([`AttachmentGuestWindow::Runs`]). Set
    // where the window is chosen, so the completion counts the *window*
    // population under its own byte name instead of folding it into R32's seed
    // runs — the two share one declaration shape and have two different
    // callers.
    let mut carried_attachment_guest_window = false;
    // R38: which of the two doors that declaration came from. The flag above is
    // what the store election and the completion key on, and it covers both
    // doors by construction — but a census has to be able to say whether the
    // *elision* door or the *seed* door moved, so the seed door's own share
    // travels beside it and is counted under its own byte name rather than
    // inside the elision door's.
    let mut carried_seed_guest_window = false;
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
            // Two doors hand this frame over, and the two are kept apart
            // because the callers' obligations differ (see the two inputs):
            // R23's is the frame the record's own *chain* names, R26's is the
            // frame of the surface the LOAD *elision* named. Either one is the
            // same statement to this class — the caller's copy of the frame
            // the pass begins from — so both select the contract's trace-owned
            // load and only the population counters differ.
            // B1-B3: the elision door's own window is asked **first**, because
            // it is the one declaration that covers both halves of that record's
            // obligation — the pages the pass begins from (E-TX6's gather) and,
            // for the record the guest's pages are owed a frame, the landing
            // that frame makes (E-TX8's `StoreOp::Borrowed`, elected below it).
            // Nothing is read back to state it: `INV-LAND` says the door paid the
            // debt before it cut the window, so those pages hold exactly what the
            // elision's own currency test read.
            if let Some(AttachmentGuestWindow::Runs(runs)) = inputs.attachment_guest_window {
                carried_attachment_guest_window = true;
                NarrowLoad::GuestRuns(runs.to_vec())
            } else {
                let (bytes, from_surface_resident) = match (
                    inputs.resident_source_bytes,
                    inputs.surface_resident_source_bytes,
                ) {
                    (Some(bytes), _) => (bytes, false),
                    (None, Some(bytes)) => (bytes, true),
                    (None, None) => {
                        // R34: the one refusal these three doors share, counted by
                        // the door whose own read declined and priced by the frame
                        // a readback would have had to carry. `Undeclared` is the
                        // canary for a door that reached this point without naming
                        // itself; it can only be earned here, by a record this
                        // class refused, so a nonzero census reading is evidence
                        // about the seam rather than about the shape.
                        //
                        // B: a door that *did* state a window and could not cut it
                        // answers under the window's own route instead. The fact
                        // that held the shape is what the census has to read, and
                        // the nine R34 routes keep naming the doors that had no
                        // window to state at all.
                        let route = match inputs.attachment_guest_window {
                            Some(AttachmentGuestWindow::Refused(route)) => route,
                            _ => inputs
                                .resident_source_route
                                .unwrap_or(ResidentSourceRoute::Undeclared),
                        };
                        note_resident_source(route, extent);
                        return Err(OutOfClass::new(
                            "render_provider_out_of_class_resident_source",
                            "a record whose previous contents are the live GPU image stays on the \
                             engine while the caller can neither read a frame this rail keeps nor \
                             hand the frame over: the resident the chain names is the engine's \
                             own, and this rail defines no image under it",
                        ));
                    }
                };
                carried_surface_resident = from_surface_resident;
                if u64::try_from(bytes.len()).ok() != Some(extent) {
                    return Err(OutOfClass::new(
                        if from_surface_resident {
                            "render_provider_out_of_class_surface_resident_shape"
                        } else {
                            "render_provider_out_of_class_resident_source_shape"
                        },
                        "a record whose previous contents are handed over at a width or extent \
                         other than the attachment's own stays on the engine: the canonical \
                         attachment uploads exactly the view's declared bytes, so a shorter or \
                         longer buffer would begin the pass from bytes no record wrote",
                    ));
                }
                NarrowLoad::Bytes(bytes)
            }
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
    } else if req.load_guest_target_backing {
        // R38, R-B: the attachment's own **guest backing**, admitted. This is
        // the seed door's statement of the fact the elision door's window
        // carries — the pages the pass begins from *are* the attachment's
        // previous contents, so the contract states them as the ordered run list
        // `BufferSource::GuestRuns` (E-TX6) rather than as a copy of anything —
        // with the record's own frame landing back in that same window (E-TX8's
        // `StoreOp::Borrowed`, elected below by the
        // `carried_attachment_guest_window` flag this arm is the second producer
        // of; the seed door's own share of that population is counted beside it
        // under `render_provider_seed_attachment_guest_window_bytes`).
        //
        // The seam's answer is the whole condition of this arm, and that is
        // deliberate: `INV-LAND` is not a property of the request but of the
        // pages, and only the caller that holds the mapping can pay the debt the
        // surface's last Store armed and check the payment
        // (`runtime::writeback_debt::pay_for_mapping`). A window that reaches
        // this arm has already been landed-or-current, so "these pages are the
        // surface's" has become "these pages hold what the door read" — the one
        // conversion a declaration may not assume. A door that could not make it
        // says so by name, and the record keeps the engine under the name of the
        // fact that stopped it.
        match inputs.seed_guest_window {
            Some(AttachmentGuestWindow::Runs(runs)) => {
                carried_attachment_guest_window = true;
                carried_seed_guest_window = true;
                NarrowLoad::GuestRuns(runs.to_vec())
            }
            Some(AttachmentGuestWindow::Refused(route)) => {
                // The run-list facts answer under R32's own four names; the
                // facts the door answers before it reaches the list keep the
                // door's bucket and are charged as routes beside it, exactly as
                // the B1 window charges its own.
                if load_seed_window_slug(route).is_none() {
                    crate::runtime::drain::note_store_route(resident_source_route(route));
                }
                let (slug, sentence) = load_seed_window_refusal(route);
                return Err(OutOfClass::owned(slug, sentence));
            }
            // The seam elects this door only for a record it also states a
            // window for, so a missing answer is a wiring bug rather than a
            // shape — and a wiring bug is not a fallback: the record keeps the
            // engine under the door's own name, exactly as it did before.
            None => {
                return Err(OutOfClass::new(
                    "render_provider_out_of_class_load_seed",
                    "a record whose previous contents are the attachment's own guest backing \
                     stays on the engine while the seam states no window for it: this door's \
                     own pages are the only declaration that can carry those contents, and a \
                     caller that resolved the backing without cutting them is a wiring bug \
                     rather than a shape",
                ));
            }
        }
    } else {
        // R32: the two seed doors, in the order the two statements differ.
        //
        // The mapper-ref-texture surface's seed: the request carries the
        // bytes themselves, as the ordered list of windows inside the surface's
        // registered pages (`research/docs/23` §113, E-TX6). The class states
        // the list as the attachment's own view — its runs concatenate to the
        // attachment's tightly packed extent — and the owner rail imports their
        // registrations so the provider gathers the bytes out of the guest's
        // live pages (this is what the engine's own `GuestTargetSeed` arm
        // reads). Every shape the list cannot describe keeps the engine by name
        // rather than being restated as a copy of bytes nothing vouched for.
        if let Some(seed) = req.target_guest_seed.as_ref() {
            let runs = match load_seed_run_windows(&seed.source, extent) {
                Ok(runs) => runs,
                Err(exit) => {
                    return Err(OutOfClass::owned(
                        exit.slug(),
                        format!(
                            "a record whose previous contents are the mapper-ref-texture \
                             surface's own guest bytes stays on the engine: the canonical \
                             attachment states them as an ordered list of owner windows whose \
                             concatenation is the attachment's tightly packed extent \
                             (`BufferSource::GuestRuns`), and {}",
                            exit.sentence()
                        ),
                    ));
                }
            };
            NarrowLoad::GuestRuns(runs)
        } else if req.target_rgba8.is_some() {
            // The other seed door: the bytes are the caller's own
            // (`DrawRequest::target_rgba8`), already folded into the
            // attachment's order by the seam that read `target_seed_order`
            // beside them ([`RenderRailInputs::load_seed_source_bytes`]). The
            // class states the same trace-owned load R23/R25 state, so the
            // extension, the format and the upload path are the ones already
            // measured for those arms.
            let Some(bytes) = inputs.load_seed_source_bytes else {
                return Err(OutOfClass::new(
                    "render_provider_out_of_class_load_seed",
                    "a record whose previous contents are the seed door's own bytes stays on the \
                     engine while the caller hands no bytes over: the class states them as the \
                     canonical attachment's trace-owned `Load` arm, which needs the caller's \
                     own copy at the attachment's tightly packed extent and in the attachment's \
                     own texel order",
                ));
            };
            if u64::try_from(bytes.len()).ok() != Some(extent) {
                return Err(OutOfClass::new(
                    "render_provider_out_of_class_load_seed_shape",
                    "a record whose seed the caller hands over at a width or extent other than \
                     the attachment's own stays on the engine: the canonical attachment uploads \
                     exactly the view's declared bytes, so a shorter or longer buffer would \
                     begin the pass from bytes no record wrote",
                ));
            }
            carried_load_seed_bytes = true;
            NarrowLoad::Bytes(bytes)
        } else {
            if req.color0_declared != Some(crate::protocol::pass_action::LoadAction::Clear) {
                return Err(OutOfClass::new(
                "render_provider_out_of_class_load_action",
                "the canonical class loads by `Clear` or from the provider's own image; a Load \
                 or DontCare record whose previous contents this rail cannot name stays on the \
                 engine",
            ));
            }
            NarrowLoad::Clear(clear)
        }
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
                    } else if carried_attachment_guest_window
                        && inputs.role == RenderChainRole::SoleOrTail
                    {
                        // B3 (E-TX8): the record whose frame the guest's pages
                        // are owed, on a pass whose *load* is those same pages
                        // declared as the attachment's own view — so the frame
                        // lands in the window the pass began from, by
                        // construction and not by a second naming channel
                        // (`resolve_attachment_landing` reads the declaring
                        // view's own source). The caller asked to skip its
                        // readback: this arm honours that where the pooled
                        // answer could only undo it, because the frame's
                        // destination is not the provider's image at all.
                        NarrowStore::Borrowed
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
    //
    // E-TX8: what the rail cannot carry is the *landing*, and only the record
    // whose frame the guest's pages are owed performs one — the packet's last
    // record (or a lone record), which is exactly [`RenderChainRole::SoleOrTail`]
    // and the census's `wb=1`. Every other position's frame goes back to the
    // caller, and for a guest-backed attachment that frame is the chain value
    // the exec walk hands the record after it (R25's arm, which the role gate
    // above already admits): answering such a record with the *published* frame
    // is what the walk consumes, and the packet's last record is the one that
    // lands the complete frame in the guest's pages. Census v21 measured the
    // split this gate leaves: 21 of the bucket's 22 latched shapes are those
    // middles (`wb=0 continues=1 pass_cont=1`) and one is the tail.
    //
    // R38: the landing is carried for exactly one arm — the record whose load
    // *is* the attachment's own guest window
    // ([`RenderRailInputs::seed_guest_window`]'s runs, E-TX6). E-TX8's
    // `StoreOp::Borrowed` lands the frame back into that same declaration, so
    // the sentence above is no longer true of that record and it leaves for the
    // provider. Every other guest-backed tail keeps the refusal: a record whose
    // load is a `Clear` (or any other arm) states no window, and a landing with
    // no declaring view to land in is still a route this rail does not carry.
    if req.guest_target_memory.is_some()
        && !req.load_from_target
        && !carried_attachment_guest_window
        && inputs.role == RenderChainRole::SoleOrTail
    {
        return Err(OutOfClass::new(
            "render_provider_out_of_class_guest_backing",
            "a record whose attachment is backed by the guest's own pages stays on the engine \
             while its frame is the one the guest's pages are owed: the class renders into a \
             provider image, and writing the guest's pages from it is a landing this rail does \
             not carry",
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
        stage_buffer_namespace_split,
    )?;
    // The sampled textures the fragment stage reads (v101, `research/docs/23`
    // §101): the class states the module's own declarations beside the draw's
    // binds — the canonical pass carries both, and the wire carries the pass's
    // texture list like any other view — and every shape the provider or the
    // module refuses keeps the engine under its own name ([`sampled_textures`]).
    let sampling = sampled_textures(
        inputs,
        req,
        render_texture_gathered_extent,
        render_texture_gathered_extent_no_copy,
        render_texture_narrow_lanes,
    )?;
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
    // The two arms the canonical contract's own `indices: Option` states (R39).
    //
    // The indexed arm keeps every door it had, in the order it had them: the
    // zero-length draw first, then the `baseVertex` offset, then the index
    // format and the stream's source and its length. Nothing about that arm
    // moves, so every shape the class answered before answers the same way.
    //
    // The non-indexed arm has none of those doors to answer, and that is a fact
    // about the shape rather than a relaxation: there is no index count to be
    // zero, no `baseVertex` to add (the contract refuses that pairing by name,
    // `BaseVertexRequiresIndices`), and no index format or index source to
    // name. Its draw count is the request's own `vertex_count`, which the
    // contract reads as `0..vertices` — and what it owes instead of those doors
    // is the span proof [`nonindexed_vertex_span`] states, answered below
    // because it is a statement about the vertex streams.
    let (draw_count, index_stream, staged_highest_index) = match req.indexed.as_ref() {
        Some(index) => {
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
            // stream's (`R9q`): the request's own staged bytes, or the one
            // registered window its zero-copy bind was cut from. The window arm
            // is what the census read as `index_staging` on every draw whose
            // index bind the draw path had already imported, and a gather this
            // rail cannot state as one window keeps the engine under the same
            // slug as before.
            let index_source = index_stream_source(&index.content)?;
            if index_source.len() == 0 {
                return Err(OutOfClass::new(
                    "render_provider_out_of_class_index_empty",
                    "an empty index stream stays on the engine",
                ));
            }
            // The highest vertex this draw's *staged* index bytes name, read
            // once here because the surplus-stream span proof below needs it and
            // this arm is where the bytes are in scope: a zero-copy index window
            // (`R11`) is guest RAM this process has not read (`None`), exactly as
            // [`stage_buffer_affine_counts`] states for the proof it derives from
            // the same bytes.
            let staged_highest = match &index_source {
                StreamSource::Staged(bytes) => {
                    highest_index(bytes, index.index_type, index.index_count)
                }
                _ => None,
            };
            (
                index.index_count,
                Some(NarrowIndexStream {
                    format: index_format,
                    source: index_source,
                }),
                staged_highest,
            )
        }
        None => (req.vertex_count, None, None),
    };

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
    if let Some(mismatch) =
        vertex_interface_mismatch(&req.vertex_attributes, inputs.vertex_attribute_locations)
    {
        // R-VI1: the direction the walk named decides whether the shape is one a
        // contract can widen onto. The declared superset is — `MTLVertexDescriptor`
        // may name a location the vertex function never reads, and the surplus
        // stream is bound and ignored — so that arm, and only that arm, is
        // answered by the device's own declaration. The other two are not
        // candidates on any device: a reflected superset is a location Metal
        // defines no value for, and a location mismatch is neither direction.
        // The admitted arm charges no route and reaches the walk below with every
        // declared entry stated, so the surplus stream is declared, bound and
        // never read, exactly as Metal's own descriptor states it.
        let admitted = mismatch.route == VertexInterfaceRoute::DeclaredSuperset
            && render_vertex_interface_superset;
        if !admitted {
            // R-VI1: the direction is a route beside the refusal, never a
            // second slug — the census reads
            // `render_provider_out_of_class_vertex_interface` as one population
            // and this splits it without moving the edge.
            note_vertex_interface(mismatch);
            return Err(OutOfClass::new(
                "render_provider_out_of_class_vertex_interface",
                "a request whose declared vertex attributes do not name exactly the locations the \
                 vertex stage reads stays on the engine: the canonical registration gate compares \
                 the layout with the reflection field by field, so this is a shape the provider \
                 always refuses",
            ));
        }
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

    // R-VI1: a declared attribute the vertex stage never reads may not be able
    // to *refuse* the draw it is ignored by. Admission proves every stream the
    // pipeline declares against the draw's own span — the highest vertex the
    // index stream names, over that stream's length divided by its stride
    // (`render_vertex_buffer_footprint_unsupported`) — and it proves the
    // *surplus* streams too, because they are bound with the rest of the layout
    // even though the module never consumes them. So a surplus declaration whose
    // stream stops short of the draw's span is a draw admission refuses, and a
    // refusal there is a decline: the draw is *skipped* rather than run on the
    // engine (`runtime/draw/vulkan.rs`), which is the one outcome this widening
    // may not introduce. Answering it here keeps the class's promise for the
    // population this increment admits.
    //
    // Scoped to the surplus streams on purpose. The same proof over a stream the
    // module *does* read is a gap this class has always had, for every shape it
    // has always admitted — the two-attribute twin of the fixture in
    // `out_of_class_shapes_stay_on_the_self_contained_engine` reaches admission
    // and is a typed `render_vertex_buffer_footprint_unsupported` decline there
    // today, which the vertex-interface test pins — and closing it would move
    // shapes this increment does not widen (it would answer the two
    // `..._is_a_typed_decline` fixtures, which exist to read that very decline).
    // What the widening owes is only that its own declarations cannot be the
    // reason a draw the class took is thrown away.
    //
    // The number needs the index *values*, so it is available for the staged
    // index arm alone: a zero-copy index window (`R11`) is guest RAM this process
    // has not read, exactly as [`stage_buffer_affine_counts`] states for the proof
    // it derives from the same bytes. The read itself happens in the indexed arm
    // above, where those bytes are in scope.
    if let Some(highest) = staged_highest_index {
        for stream in &vertex_streams {
            let ignored = stream.attributes.iter().all(|attribute| {
                !inputs
                    .vertex_attribute_locations
                    .contains(&attribute.location)
            });
            if !ignored {
                continue;
            }
            let capacity = stream.source.len() / stream.stride.max(1);
            if highest >= capacity {
                return Err(OutOfClass::owned(
                    "render_provider_out_of_class_vertex_span",
                    format!(
                        "a draw whose ignored vertex stream holds {capacity} record(s) of \
                         {stride} bytes while its index stream names vertex {highest} stays \
                         on the engine: the stream is declared, so admission proves it \
                         against the draw's own span \
                         (`render_vertex_buffer_footprint_unsupported`) and refuses the draw, \
                         and an attribute the vertex stage never reads may not be the reason \
                         a draw the class took is thrown away instead of drawn",
                        stride = stream.stride,
                    ),
                ));
            }
        }
    }
    // R39: the non-indexed arm's own span, answered here because it is a
    // statement about the streams just built. The door is fail-closed by
    // construction rather than by taste: an in-class draw the provider refuses
    // ends as a typed decline (`runtime/draw/vulkan.rs` does not re-run the
    // engine for one), so a shape this gate admits and the provider declines is
    // a hard failure — and the provider's footprint proof is *stricter* on this
    // arm than on the indexed one, which only owes coverage of the vertices its
    // indices name. Every fact the door answers is one the canonical contract
    // or one of the two rails states; [`nonindexed_vertex_span`] holds them and
    // their names.
    if req.indexed.is_none() {
        if let Some(refusal) = nonindexed_vertex_span(&vertex_streams, draw_count) {
            return Err(refusal);
        }
    }

    // about the request: the frame format of that increment stated the pass's
    // sampled *textures* (the v70 channel) but not its runtime `[[sampler(n)]]`
    // list, so a trace that crossed the owner→provider wire would have reached
    // admission with the states dropped and been refused there
    // (`render_runtime_sampler_missing`) — a decline, not a fallback. The frame
    // carries the list since E-TX4 (`research/docs/23` §3.3, v102/v111: the
    // `PASS_KIND_RENDER_SAMPLERS` family, tags `0x15..=0x18`, which the pinned
    // provider has carried since `7d544d4`), so what is left of the question is
    // a fact about the *frame* and it moved to the one place that reads frames:
    // `submit_render` asks the device's own capability answer
    // (`declared_render_sampler_carriage`) and keeps the draw on the engine
    // under this same bucket when the answer does not carry the shape. The
    // "does this trace cross the frame at all" half of that reading is
    // [`NarrowPass::crosses_the_frame`], the one predicate the plan and the
    // device answers already share.
    // The pass's sampled *textures* under a frame are the third wire question,
    // and it is the one R28 moves out of this pure gate: the frame's own
    // capability answer decides it (`submit_render`'s
    // `declared_render_texture_support`), because v102/v109's wire carries the
    // render contract's texture declarations and the pass block that pairs them
    // with the views — E-TX4's half of the frozen pair — and which of those
    // sections a provider's frame actually carries is a fact about the frame.
    // R22: a record that carries productions spends the trace's serial pool on
    // *both* passes' views. The pool's own bound is the contract's
    // (`serial_resource_limit` at admission), and a shape above it is one the
    // provider always refuses — so the class answers it here, by name, rather
    // than handing admission a trace it can only decline.
    // R39: the index view is one of the pool's views on the indexed arm and
    // absent on the non-indexed one, so the sum this gate compares with the
    // pool's own bound is the number of views the trace really declares —
    // [`NarrowPass::views`] is the same sum read back at completion time.
    let views = 1
        + vertex_streams.len()
        + usize::from(index_stream.is_some())
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
    // The pipeline-layout face of the interface the two doors above admitted
    // one face at a time is *in* the class since R31. The canonical rail builds
    // one layout list for a pass: set 0 is the sampled pipeline's own — the
    // layout `metal-api-vulkan`'s `create_render_textures` builds beside the
    // texture binds — and the stage buffers' sets follow it positionally
    // (`create_stage_buffers`). A translated module reads each `[[buffer(n)]]`
    // from set 0 (`metal2vulkan`'s default resource layout binds it at
    // `BUFFER_BINDING_BASE + n`), so a pass that binds a sampled texture *and*
    // states a stage buffer used to be a shape whose two faces both wanted a
    // slot that list held once: the rail answered it when the pipeline layout
    // was built, by name (`render_texture_layout_unsupported`: "a stage buffer
    // occupies descriptor set 0, which is where this pipeline's combined image
    // samplers live; the two faces need different slots for one layout to hold
    // both").
    //
    // That answer is a *decline*, and a decline on an in-class draw is
    // fail-closed (`runtime/draw/vulkan.rs` does not re-run the engine), so the
    // class kept the shape on the engine instead of handing admission a trace
    // it could only lose. Census v22 read the population: 1,234
    // `render_texture_layout_unsupported` rows, every one of them this class's
    // own by-name exit, which is the login window's own icon layers (64x64,
    // 96x64, 144x64, 186x100) missing from the frame.
    //
    // E-TX7 (`metal-api-emulator` `115f01f`, E main `ab21f10`) taught the rail
    // to hold both faces — `create_stage_buffers` merges the sampled pipeline's
    // own set-0 layout with the stage buffers a module reads there
    // (`create_merged_set_zero`, the translator's bands disjoint by
    // construction) — so the overlap crosses into the provider now. The one
    // boundary that stays is the merge's own, and
    // [`stage_buffer_statement`] answers it by name before the frame exists:
    // two stages' buffers folding onto one set-0 slot (see the folded rule
    // there), which `metal-api-emulator` refuses at submission
    // (`render_stage_buffer_layout_unsupported`).
    Ok(NarrowPass {
        vertex_entry: vertex_entry.to_owned(),
        fragment_entry: fragment_entry.to_owned(),
        format,
        load,
        store,
        published_held_resident,
        carried_chain_middle,
        carried_load_seed_bytes,
        carried_surface_resident,
        carried_attachment_guest_window,
        carried_seed_guest_window,
        sampled_target_frames: sampling
            .textures
            .iter()
            .any(|texture| matches!(texture.source, NarrowTextureSource::Frame(_))),
        bgra: format == AttachmentFormat::Bgra8Unorm,
        width: u64::from(req.width),
        height: u64::from(req.height),
        extent,
        draw_count,
        vertex_streams,
        index_stream,
        stage_buffers,
        textures: sampling.textures,
        runtime_samplers: sampling.runtime_samplers,
        productions: sampling.productions,
        scissor,
        viewport,
        blend,
        present,
        vertex_stage_buffer_namespace_split: stage_buffer_namespace_split,
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
///
/// R32's run-list arm is the one route whose pair is not this rail's to choose:
/// the contract pairs every run's reservation with the *declaring view's* own
/// allocation, so the view — and with it the pass's colour attachment — has to
/// name the lease allocation the owner plan minted for the seed's registration
/// (`provider_owner::Plan::run_allocation`). The view identity is still this
/// rail's own constant, so a seed's attachment is the same view in every arm
/// and only the allocation moves.
fn attachment_identity(
    pass: &NarrowPass<'_>,
    leases: Option<&provider_owner::Plan>,
) -> AttachmentIdentity {
    if let Some(present) = pass.present {
        return present.into();
    }
    let resident = match (&pass.load, pass.store) {
        (NarrowLoad::Resident(resident), _) => *resident,
        (_, NarrowStore::Resident(resident)) => resident,
        (NarrowLoad::GuestRuns(_), _) => {
            let (allocation, _, _) = leases
                .and_then(|plan| plan.run_allocation(load_seed_owner_binding()))
                .expect("the owner plan covers an admitted guest-runs seed");
            ResidentAttachment {
                allocation,
                view: ATTACHMENT_VIEW,
            }
        }
        (NarrowLoad::Clear(_) | NarrowLoad::Bytes(_), NarrowStore::Writeback) => {
            ResidentAttachment {
                allocation: ATTACHMENT_ALLOCATION,
                view: ATTACHMENT_VIEW,
            }
        }
        // Unreachable by construction: `NarrowStore::Borrowed` is elected only
        // beside a `GuestRuns` load (the window *is* the load's source). The arm
        // is stated rather than left out so the pair is total, and it names the
        // pooled view rather than panicking: a future election that broke the
        // pair would declare a borrowed store beside a view no owner window
        // backs, which the provider refuses by name
        // (`render_attachment_landing_unsupported`) instead of writing
        // anywhere.
        (NarrowLoad::Clear(_) | NarrowLoad::Bytes(_), NarrowStore::Borrowed) => {
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
    rows: &RowCopies,
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
    let loads_resident = matches!(pass.load, NarrowLoad::Resident(_));

    let mut resources = ResourceTableSnapshot::new();
    for allocation in input_allocations(pass, rows) {
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
    // The owner's leases first (R9d/R9q): every window-backed binding this
    // pass states — a stage buffer's, and a vertex stream's since R9q — is
    // imported here, and the trace's views below name the allocation and lease
    // the plan minted for each. Importing before the trace exists is the same
    // order the compute rail keeps: a refused import never reaches admission.
    //
    // R32 puts this call ahead of the attachment's own declaration: a
    // guest-runs seed's view has to name the *lease's* allocation (the contract
    // pairs every run's reservation with the declaring view's allocation), and
    // only the plan knows which allocation its registration minted.
    let mut leases = plan_owner_leases(provider, pass, &mut resources, copies)?;
    let attachment = attachment_identity(pass, leases.as_ref());
    // The trace's own declaration of the attachment view. `OwnedBytes` of the
    // packed extent is what the frozen contract asks a storing attachment to
    // land through; neither the clear nor either resident arm reads them, and
    // the byte-less declaration that would lift the copy is the named follow-up
    // on the emulator side. R23's byte arm is the exception the rule was always
    // shaped for: it *is* a `LoadOp::Load` attachment, so these bytes are the
    // frame the pass begins from and have to be the caller's own — and R32's
    // run list is the second: its bytes are the guest's own pages, read through
    // the lease the plan just imported.
    let attachment_source = match &pass.load {
        NarrowLoad::Bytes(bytes) => BufferSource::OwnedBytes(bytes.to_vec()),
        NarrowLoad::GuestRuns(_) => BufferSource::GuestRuns(
            leases
                .as_ref()
                .and_then(|plan| plan.guest_runs(load_seed_owner_binding()))
                .expect("the owner plan covers an admitted guest-runs seed")
                .to_vec(),
        ),
        NarrowLoad::Clear(_) | NarrowLoad::Resident(_) => {
            BufferSource::OwnedBytes(vec![0u8; usize::try_from(pass.extent).unwrap_or(0)])
        }
    };
    let declaration = BufferView {
        view_id: attachment.view,
        metal_binding: 0,
        allocation_id: attachment.allocation,
        offset: 0,
        length: pass.extent,
        access: BufferAccess::Read,
        attribute_stride: None,
        source: attachment_source,
    };
    // The attachment's own allocation record. A guest-runs seed's attachment
    // *is* the registration the plan just recorded — same allocation, and its
    // size is the registration's, not the extent's — so only the pooled and
    // resident arms mint their own here.
    if pass.load_seed_runs().is_none() {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: attachment.allocation,
                owner_epoch: provider.device_epoch(),
                size: pass.extent,
            })
            .map_err(|error| ProviderRenderDecline::TraceAdmission {
                detail: error.to_string(),
            })?;
    }
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
    let index_slot = next_view;
    let index_view = ViewId::new(index_slot);
    // R39: the index view's number is claimed whatever the arm is — the step is
    // a property of the pass, not of the draw — so every view after this slot
    // (the stage buffers' and the sampled textures') keeps the identity the
    // indexed arm gives it. What the non-indexed arm does not do is *declare*
    // anything there: the contract's `indices: None` arm reads `0..vertices`,
    // and a view nothing names is not one the trace states.
    //
    // The index view has an identity of its own: the two namespaces are the
    // same one (`ViewId`), and R9j made the collision visible — a stage
    // buffer's pool entry is keyed by this identity, and an index view that
    // reused it was answered as "the same view changed its allocation, range
    // or source bytes" (`SerialBufferRebinding`). Harmless while every stage
    // buffer lived only in the render pass and nothing declared these views in
    // the pool; a landing declares them.
    next_view += 1;
    let indices = match &pass.index_stream {
        Some(stream) => {
            // R11: the index stream's own arm decides the view. Staged bytes
            // stay trace-owned, exactly as before; a window-backed index bind
            // names the lease the owner plan imported for it — the borrowed
            // arm, or since R18 the staged lease over the copy the gate made —
            // with the view offset and length the owner's own reservation
            // covers. The same channel switch the vertex stream above keeps,
            // for the same reason.
            let (index_allocation, index_offset, index_length, index_source) = match &stream.source
            {
                StreamSource::Staged(bytes) => (
                    input_allocation(index_slot),
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
                            provider_owner::Channel::Borrowed => {
                                BufferSource::BorrowedNoCopy(owner.lease)
                            }
                            provider_owner::Channel::Staged => {
                                BufferSource::StagedLease(owner.lease)
                            }
                        },
                    )
                }
            };
            Some(IndexBufferBinding {
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
                format: stream.format,
            })
        }
        None => None,
    };
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
            // R24's frame is the same arm one step further in: the bytes are
            // the caller's copy of the target image, declared for this view
            // exactly as a request-carried copy is.
            //
            // R36's repacked rows are the third spelling of that one arm: the
            // bytes are the class gate's own copy of the texture's tightly
            // packed extent, assembled row by row out of the guest's padded
            // window ([`RowCopies`]), so they are stated exactly as the other
            // two are. Its byte count comes from the one accessor the
            // allocation above reads, so the view and its allocation are sized
            // by one number.
            NarrowTextureSource::Bytes(_)
            | NarrowTextureSource::Frame(_)
            | NarrowTextureSource::Depadded { .. } => (
                ViewId::new(next_view),
                input_allocation(next_view),
                TextureSource::OwnedBytes(
                    texture
                        .source
                        .owned_bytes(rows, texture.index)
                        .expect("an admitted byte-bearing texture carries its bytes")
                        .to_vec(),
                ),
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
            // R28: the lease the plan minted for this texture's window, under
            // the label both halves key it by. Its *channel* is the device's
            // answer ([`texture_window_arm`]): the borrowed no-copy window, or
            // the owner-issued staged copy of the texture's own extent — the
            // same two arms a window-backed stream states, and the reason this
            // pass crosses the owner→provider frame at all.
            NarrowTextureSource::Window { binding, .. } => {
                let view = leases
                    .as_ref()
                    .and_then(|plan| plan.view(*binding))
                    .expect("the owner plan covers every admitted sampled window");
                (
                    ViewId::new(next_view),
                    view.allocation,
                    match view.channel {
                        provider_owner::Channel::Borrowed => {
                            TextureSource::BorrowedNoCopy(view.lease)
                        }
                        provider_owner::Channel::Staged => TextureSource::StagedLease(view.lease),
                    },
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
                // R32 adds the guest-runs arm to the same load op: the bytes
                // come from the owner's pages rather than from the caller, and
                // only the provider's own resolution differs.
                NarrowLoad::Bytes(_) | NarrowLoad::GuestRuns(_) => LoadOp::Load,
            },
            store: match pass.store {
                NarrowStore::Writeback => StoreOp::Store,
                NarrowStore::Resident(_) => StoreOp::Resident,
                // B3 (E-TX8): the frame lands in the attachment view's own
                // declared window. The writeback channel still carries the
                // bytes (`store_publishes` keeps both arms publishing), so the
                // seam's own Store route reads a completion exactly as it does
                // for `Store` — it just has nothing left to land.
                NarrowStore::Borrowed => StoreOp::Borrowed,
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
        vertices: pass.draw_count,
        vertex_buffers,
        // R39: `Some` on the indexed arm and `None` for a draw that names its
        // vertices directly — the contract's own two arms, where `None` reads
        // the count above as `0..vertices`.
        indices,
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
                note_wire_render_textures(&trace);
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
    // The byte arms' own populations, counted where the answer happens for the
    // reason above: the frame was carried into the pass as bytes, so neither
    // resident arm moved and these are the only names that say the record was
    // answered this way. R23's is the frame the caller read out of the
    // engine's registry under the record's own chain, R25's is the packet's
    // own chain value handed on by the walk, and R26's is the frame of the
    // surface the LOAD elision named — three doors, three numbers, because the
    // callers' obligations differ even though the load does not.
    if matches!(pass.load, NarrowLoad::Bytes(_)) && pass.carried_load_seed_bytes {
        // R32: the one byte arm whose caller is not a registry read — the
        // seed door's own bytes, folded into the attachment's order by the
        // seam. Its own name, for the same reason the three above have theirs.
        crate::runtime::drain::note_store_route("render_provider_load_seed_bytes");
    } else if matches!(pass.load, NarrowLoad::Bytes(_)) {
        crate::runtime::drain::note_store_route(if pass.carried_chain_middle {
            "render_provider_chain_middle_source_bytes"
        } else if pass.carried_surface_resident {
            "render_provider_surface_resident_source_bytes"
        } else {
            "render_provider_resident_source_bytes"
        });
    }
    // R32's other arm: the attachment's previous contents are the surface's own
    // guest pages, declared as the contract's ordered run list and gathered by
    // the provider. Counted where the answer happens, like every arm above.
    //
    // B: the elision door states the *same* declaration shape from its own
    // caller (the window it cut after paying the debt), so the two populations
    // are counted apart by the flag the load arm set — a census that folded them
    // together could not say whether the seed door or the elision door had
    // moved, which is exactly the reading this increment is judged on.
    if matches!(pass.load, NarrowLoad::GuestRuns(_)) {
        if pass.carried_attachment_guest_window {
            crate::runtime::drain::note_store_route_n(
                "render_provider_attachment_guest_window_bytes",
                pass.extent,
            );
        } else {
            crate::runtime::drain::note_store_route("render_provider_load_seed_runs");
        }
    }
    // R38: the same declaration, counted again under the door that produced it.
    // The seed door's own population is the one this increment moved, and it is
    // read *beside* the merged name above rather than instead of it: the merged
    // number is what says the window population grew, this one is what says
    // which door grew.
    if pass.carried_seed_guest_window {
        crate::runtime::drain::note_store_route_n(
            "render_provider_seed_attachment_guest_window_bytes",
            pass.extent,
        );
    }
    // B3: the frame this pass landed in the owner's own window (E-TX8), priced
    // beside the arm above. Counted where the answer happens for the same reason
    // every arm here is: a submission that never reached the caller is not an
    // answer.
    if matches!(pass.store, NarrowStore::Borrowed) {
        crate::runtime::drain::note_store_route_n(
            "render_provider_borrowed_landing_bytes",
            pass.extent,
        );
    }
    // R24's own population, counted where the answer happens for the same
    // reason as R23's above: the caller read a sampled GPU target's frame out
    // of the registry that holds it and this pass declared it as the trace's
    // own bytes, so the record left the engine without a production to restate.
    if pass.sampled_target_frames {
        crate::runtime::drain::note_store_route("render_provider_sampled_target_frames");
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
        landed_in_window: matches!(pass.store, NarrowStore::Borrowed),
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
fn input_allocations(pass: &NarrowPass<'_>, rows: &RowCopies) -> Vec<(AllocationId, u64)> {
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
    // R39: the index slot is walked whether or not the arm states a stream, so
    // the textures' own numbering below does not move with the draw's arm — and
    // a non-indexed draw, which declares no index view, contributes no
    // allocation to this list.
    if let Some(stream) = &pass.index_stream {
        if let StreamSource::Staged(bytes) = &stream.source {
            out.push((
                input_allocation(next_view),
                u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            ));
        }
    }
    // The sampled textures' views are stated after the index view and the
    // stage buffers' (which claim a view number each but mint their allocation
    // through the owner rail), so their own allocations start one past the
    // index and every declared stage buffer — the same walk `submit_narrow`
    // makes, one number at a time.
    let texture_base = next_view + 1 + u64::try_from(pass.stage_buffers.len()).unwrap_or(u64::MAX);
    for (index, texture) in pass.textures.iter().enumerate() {
        // Both byte-bearing arms — the request's own copy and the caller's
        // frame (R24), beside R36's repacked rows, which
        // [`NarrowTextureSource::owned_bytes`] reads out of the gate's own copy
        // — mint their own view's allocation here. A trace-produced texture
        // names the identity its production stored under (R22): that allocation
        // is declared from the production's side of the trace, and re-declaring
        // it here would be a second record for one view.
        // R28 adds the third arm to that rule: a window-backed texture's
        // allocation is the one [`plan_owner_leases`] minted for its lease, so
        // the trace's view names that allocation rather than one of this
        // rail's own — its view *identity* below still advances, so a texture's
        // identity never depends on which arm it took.
        let Some(bytes) = texture.source.owned_bytes(rows, texture.index) else {
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

/// The label one sampled texture's window travels under in the owner rail
/// (`R28`).
///
/// The fifth namespace of the same four: a sampled texture is a fragment
/// stage's `[[texture(n)]]` argument, which is neither stage's `[[buffer(N)]]`
/// namespace ([`stage_buffer_owner_binding`]), nor a vertex stream's
/// ([`vertex_stream_owner_binding`]), nor the index stream's
/// ([`index_stream_owner_binding`]) — and `MAX_RENDER_TEXTURE_INDEX` keeps the
/// index small enough for the low half, so the whole `[[texture(n)]]` space has
/// a namespace of its own. A lookup answered by another namespace's lease would
/// be a wrong frame rather than a refusal, which is the rule every one of these
/// labels is keyed for.
fn texture_owner_binding(index: u32) -> u32 {
    (0x0004 << 16) | index
}

/// The label one attachment's own guest-run seed travels under in the owner
/// rail (`R32`).
///
/// The sixth namespace of the same five ([`stage_buffer_owner_binding`]'s two
/// stages, then the vertex stream's, the index stream's and the sampled
/// texture's): an attachment's previous contents are no stage's argument at
/// all, so the namespace is the whole label. A lookup answered by another
/// namespace's lease would be a wrong frame rather than a refusal, which is the
/// rule every one of these labels is keyed for.
///
/// The label grants nothing on its own: [`provider_owner::plan`] mints one
/// lease per registration for whichever label asks, so a pass that states this
/// seed beside a vertex stream of the same registration shares that stream's
/// lease, exactly as two streams of one registration do.
fn load_seed_owner_binding() -> u32 {
    0x0005 << 16
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
///
/// R32 puts the attachment's own guest-run seed in the same plan, under
/// [`load_seed_owner_binding`]. The seed is not a view of its own: its runs are
/// stated *inside* the registration's lease, so the request is the list arm
/// ([`provider_owner::Request::Runs`]) and what comes back is the ordered
/// per-run coordinates the contract's `BufferSource::GuestRuns` declaration
/// carries. A registration that backs both this seed and a stream is still one
/// lease and one import.
fn plan_owner_leases(
    provider: &metal_api_vulkan::VulkanComputeProvider,
    pass: &NarrowPass<'_>,
    resources: &mut ResourceTableSnapshot,
    copies: &WindowCopies,
) -> Result<Option<provider_owner::Plan>, ProviderRenderDecline> {
    if pass.stage_buffers.is_empty()
        && pass.vertex_windows().next().is_none()
        && pass.index_window().is_none()
        && pass.texture_windows().next().is_none()
        && pass.load_seed_runs().is_none()
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
    // R28: the sampled textures' windows are the fourth shape of the same
    // plan, under their own label namespace. Both of their arms are leases —
    // the borrowed window, or the owner's staged copy of the extent
    // ([`texture_window_arm`]) — so a sampled pass whose bind is a guest
    // gather crosses the owner→provider frame exactly as a window-backed
    // stream does, and its declaration names the lease this plan imports.
    requests.extend(
        pass.texture_windows()
            .map(|(binding, window)| window_arm_request(binding, window, copies)),
    );
    // R32: the attachment's own guest-run seed, as the list arm of the same
    // plan. Its windows are read through the borrowed lease only: the provider
    // gathers the runs with the host at resolution (`research/docs/23` §113), so
    // a run needs no device import of its own and the class does not copy one —
    // the device answer this arm does ask is the registration's own import,
    // which the gate asks before the plan is reached.
    let load_seed_windows: Vec<provider_owner::Window> = pass
        .load_seed_runs()
        .map(|runs| {
            runs.iter()
                .map(|window| owner_window(load_seed_owner_binding(), *window))
                .collect()
        })
        .unwrap_or_default();
    if !load_seed_windows.is_empty() {
        requests.push(provider_owner::Request::Runs(provider_owner::Runs {
            binding: load_seed_owner_binding(),
            windows: &load_seed_windows,
        }));
    }
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
        vertex_stage_buffer_namespace_split: pass.vertex_stage_buffer_namespace_split,
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
    // R33: the descriptor layout each stage is translated against is the
    // module's own Vulkan ABI, and it is the *other* half of the folded pair's
    // answer. The default layout puts every Metal resource of a stage in set 0,
    // so a pair that reads one `[[buffer(n)]]` index from both stages folds the
    // two descriptors onto one slot; the canonical namespace layout moves the
    // *vertex* half's whole layout to set 1 and leaves the fragment half where
    // it was, and the provider reads each bound slot back out of the module's
    // own reflection. Only the folded request whose device declared the split
    // takes that layout, so every other pair is translated exactly as it was.
    let stage = |air: &[u8], entry: &str, which: &'static str, stage, layout| {
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
        TranslatedRenderStage::translate_with_policy_and_layout(stage, &function, policy, layout)
            .map_err(|error| ProviderRenderDecline::PipelineCompile {
                step: which,
                detail: error.to_string(),
            })
    };
    let vertex_layout = if pass.vertex_stage_buffer_namespace_split {
        metal_api_vulkan::stage_buffer_namespace_layout()
    } else {
        metal2vulkan::reflect::DescriptorLayout::default()
    };
    let vertex = stage(
        inputs.vertex_air,
        &pass.vertex_entry,
        "vertex_stage",
        RenderStage::Vertex,
        vertex_layout,
    )?;
    let fragment = stage(
        inputs.fragment_air,
        &pass.fragment_entry,
        "fragment_stage",
        RenderStage::Fragment,
        metal2vulkan::reflect::DescriptorLayout::default(),
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
mod padded_rows_tests {
    use super::*;

    /// The stride one padded gather states, spelled the way the draw path
    /// spells it: `bufferRowLength` in texels times the view's own bytes per
    /// texel, beside the tight row and the row count its extent names.
    fn rows(texels_per_row: u32, width: u32, height: u32) -> PaddedRows {
        PaddedRows {
            row_length_texels: texels_per_row,
            stride: u64::from(texels_per_row) * 4,
            tight_row: u64::from(width) * 4,
            rows: height,
        }
    }

    /// R36: the three numbers one padded gather is measured by — the span the
    /// guest's window holds, the extent the copy keeps, and the padding it drops
    /// — are one arithmetic, and the padding is the guest's stride minus the
    /// tight row for every row but the last.
    ///
    /// The shape is census v25b's head: a 135-texel-wide view at
    /// `bufferRowLength` 144, which is the reading that bucket's 444 records
    /// carried (`evidence/gate3-census-v25b-2026-09-18`).
    #[test]
    fn a_padded_sources_spans_are_the_guest_stride_and_the_tight_extent() {
        let layout = rows(144, 135, 16);
        assert_eq!(layout.tight_row, 540);
        assert_eq!(layout.tight(), 8_640);
        assert_eq!(layout.span(), 144 * 4 * 15 + 540);
        assert_eq!(layout.padding(), (144 * 4 - 540) * 15);
        // One row: the gather's span *is* the extent and no padding is
        // reachable at all, which is what "the last row's trailing padding is
        // outside the gather's window" means for the smallest case.
        let single = rows(144, 135, 1);
        assert_eq!(single.span(), single.tight());
        assert_eq!(single.padding(), 0);
    }

    /// R36: the repack reads one tight row per stride and writes them back to
    /// back, so the padding between the rows never reaches the copy; and a
    /// window that is not the span this layout named is a caller-side wiring bug
    /// rather than a shorter copy.
    #[test]
    fn the_repack_reads_one_tight_row_per_stride() {
        const PAD: u8 = 0xee;
        let layout = rows(4, 2, 2);
        assert_eq!(layout.stride, 16);
        assert_eq!(layout.tight_row, 8);
        let mut padded = Vec::new();
        for row in 0..2u8 {
            padded.extend_from_slice(&[row, 1, 2, 3, 4, 5, 6, 7]);
            padded.extend_from_slice(&[PAD; 8]);
        }
        // The last row's trailing padding is outside the gather's window.
        padded.truncate(usize::try_from(layout.span()).expect("the span fits usize"));
        let tight = layout
            .depad(&padded)
            .expect("the window is the span this layout named");
        assert_eq!(
            tight.len(),
            usize::try_from(layout.tight()).expect("the extent fits usize")
        );
        assert_eq!(&tight[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(&tight[8..], &[1, 1, 2, 3, 4, 5, 6, 7]);
        assert!(
            !tight.contains(&PAD),
            "the padding between the rows never reaches the copy: {tight:?}"
        );
        assert_eq!(
            layout.depad(&padded[..padded.len() - 1]),
            None,
            "a window short of the span names no rows"
        );
    }

    /// R36: a guest that states its own tight row as `bufferRowLength` is the
    /// degenerate padded shape — the source says the rows stride and the stride
    /// *is* the row — so the copy is the identity rather than a refusal. The
    /// draw path never builds one (`runtime::draw::vulkan`'s
    /// `strided_window_extent` answers zero for it), and census v25b's
    /// `row_length_texels` buckets are the ones that do; this rail reads the
    /// source rather than the builder, so both shapes have to answer.
    #[test]
    fn a_stride_that_is_the_row_is_an_identity_copy() {
        let layout = rows(8, 8, 4);
        assert_eq!(layout.stride, layout.tight_row);
        assert_eq!(layout.span(), layout.tight());
        assert_eq!(layout.padding(), 0);
        let texels: Vec<u8> = (0..layout.tight()).map(|byte| byte as u8).collect();
        assert_eq!(layout.depad(&texels), Some(texels));
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

#[cfg(test)]
mod vertex_interface_tests {
    use super::*;
    use crate::backend::vulkan::engine::VertexAttributeFormat;

    /// One declared attribute at each of `locations`, built from the reviewed
    /// `float2` storage. The classifier reads locations alone, so every other
    /// field only has to be a value that builds.
    fn declared(locations: &[u32]) -> Vec<VertexAttributeResource> {
        locations
            .iter()
            .map(|location| VertexAttributeResource {
                location: *location,
                binding: *location,
                format: VertexAttributeFormat::parse(29).expect("Float2 is a vertex format"),
                offset: 0,
                stride: 8,
                step_function: VertexStepFunction::PerVertex,
                step_rate: 1,
                content: BufferContent::Bytes(Arc::new(vec![0_u8; 24])),
            })
            .collect()
    }

    /// One row of [`every_vertex_interface_disagreement_has_a_direction`]'s
    /// table: the declared locations, the reflection's own, and the route the
    /// pair has to answer in with the distance beside it — `None` for the pairs
    /// the gate admits.
    type Case = (
        &'static [u32],
        &'static [u32],
        Option<(VertexInterfaceRoute, u64)>,
    );

    /// The classifier is *total*: every shape the gate refuses is exactly one of
    /// the three routes, a shape it admits is none of them, and the distance
    /// beside each route is the count that route's own definition names (R-VI1).
    ///
    /// The table is the whole truth table over the two counts, plus the two
    /// shapes only the length walk catches: one location declared twice at a
    /// location the stage reads, and a swap between two locations of one width.
    /// The admit rows are the point of the first two entries — a route that
    /// fired for a shape the old predicate passed would be a behaviour change,
    /// not a reading.
    #[test]
    fn every_vertex_interface_disagreement_has_a_direction() {
        use VertexInterfaceRoute as Route;
        let cases: &[Case] = &[
            // The two lists name the same location set: no route, and the gate
            // admits — including the shape whose *order* differs, which the old
            // predicate passed and this one must keep passing.
            (&[0], &[0], None),
            (&[0, 1], &[1, 0], None),
            // Declared ⊋ reflected: the direction Metal answers, and the only
            // one a widening can admit. The distance is the surplus.
            (&[0, 1], &[0], Some((Route::DeclaredSuperset, 1))),
            (&[0, 1, 2], &[0], Some((Route::DeclaredSuperset, 2))),
            // Reflected ⊋ declared: the direction Metal leaves undefined. The
            // distance is the uncovered location count.
            (&[0], &[0, 1], Some((Route::ReflectedSuperset, 1))),
            (&[], &[0], Some((Route::ReflectedSuperset, 1))),
            // Neither: each side names a location the other does not.
            (&[0, 2], &[0, 1], Some((Route::LocationMismatch, 2))),
            // Neither: one location declared twice, which the length walk
            // catches and the reflection cannot mirror. There is no wider side,
            // so the route fires with no distance charged beside it.
            (&[0, 0], &[0], Some((Route::LocationMismatch, 0))),
        ];
        for (declared_locations, reflected, expected) in cases {
            match (
                vertex_interface_mismatch(&declared(declared_locations), reflected),
                expected,
            ) {
                (None, None) => (),
                (Some(mismatch), Some((route, distance))) => {
                    assert_eq!(
                        mismatch.route, *route,
                        "route for declared={declared_locations:?} reflected={reflected:?}"
                    );
                    assert_eq!(
                        mismatch.distance, *distance,
                        "distance for declared={declared_locations:?} reflected={reflected:?}"
                    );
                }
                (answered, expected) => panic!(
                    "declared={declared_locations:?} reflected={reflected:?}: {answered:?} \
                     against {expected:?}"
                ),
            }
        }
    }

    /// The three arms' routes and the three distances beside them are these six
    /// names: the census reads those keys by hand, so a rename has to fail here
    /// rather than silently re-file a population (R-VI1).
    #[test]
    fn the_vertex_interface_arms_charge_their_own_routes() {
        let routes = [
            vertex_interface_route(VertexInterfaceRoute::DeclaredSuperset),
            vertex_interface_route(VertexInterfaceRoute::ReflectedSuperset),
            vertex_interface_route(VertexInterfaceRoute::LocationMismatch),
        ];
        assert_eq!(
            routes,
            [
                "vertex_interface_declared_superset",
                "vertex_interface_reflected_superset",
                "vertex_interface_location_mismatch",
            ]
        );
        let distances = [
            vertex_interface_route_distance(VertexInterfaceRoute::DeclaredSuperset),
            vertex_interface_route_distance(VertexInterfaceRoute::ReflectedSuperset),
            vertex_interface_route_distance(VertexInterfaceRoute::LocationMismatch),
        ];
        assert_eq!(
            distances,
            [
                "vertex_interface_declared_superset_extra_locations",
                "vertex_interface_reflected_superset_uncovered_locations",
                "vertex_interface_location_mismatch_unpaired_locations",
            ]
        );
        let mut distinct: Vec<&str> = routes.iter().chain(distances.iter()).copied().collect();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            6,
            "one name per arm and one per distance: {routes:?} {distances:?}"
        );
    }
}

#[cfg(test)]
mod nonindexed_span_tests {
    use super::*;

    /// One admitted stream over `bytes` at `stride`. The proof reads the source
    /// and the stride alone, so the attributes are empty and the head is the
    /// position this list states the table at.
    fn stream(stride: u64, bytes: &[u8]) -> NarrowVertexStream<'_> {
        NarrowVertexStream {
            stride,
            head: 0,
            source: StreamSource::Staged(bytes),
            attributes: Vec::new(),
        }
    }

    /// The facts the non-indexed arm's span door answers (R39), one shape each:
    /// the layout-bearing count, the layout-free shape's own count, and the
    /// coverage proof both rails run over every per-vertex stream.
    ///
    /// The arm's rail case drives the third one end to end
    /// (`provider_render_rail.rs::a_non_indexed_draw_the_provider_would_refuse_stays_on_the_engine_by_name`);
    /// the first two are the contract's own shape rules, and the rail corpus
    /// carries no module the layout-free arm can be reached with, so they are
    /// pinned here, where the boundary values can be.
    #[test]
    fn the_span_door_answers_the_contracts_count_beside_the_streams_coverage() {
        let bytes = [0u8; 24];
        let slug = |draw_count: u32, streams: &[NarrowVertexStream<'_>]| {
            nonindexed_vertex_span(streams, draw_count).map(|refusal| refusal.slug())
        };
        const SPAN: Option<&str> = Some("render_provider_out_of_class_vertex_span");

        // A draw with a vertex layout owes at least the contract's own
        // full-screen triangle (`DrawVertexCountBelowMinimum`); exactly the
        // three the reviewed stream carries is the admitted shape.
        assert_eq!(slug(3, &[stream(8, &bytes)]), None);
        for too_few in [0, 1, 2] {
            assert_eq!(
                slug(too_few, &[stream(8, &bytes)]),
                SPAN,
                "a layout-bearing draw that names {too_few} vertices is one the contract \
                 refuses, so it stays on the engine here"
            );
        }

        // The layout-free `vertex_id` shape owes exactly that count
        // (`DrawVertexCountMismatch`) — on both sides of it.
        assert_eq!(slug(3, &[]), None);
        for other in [0, 1, 2, 4, 6] {
            assert_eq!(
                slug(other, &[]),
                SPAN,
                "the layout-free shape names exactly {FULL_SCREEN_TRIANGLE_VERTICES} vertices, \
                 and this one names {other}"
            );
        }

        // Coverage, at the boundary and on both sides of it: `vertices * stride`
        // bytes is covered, one byte less is not, and the stride is the stream's
        // own rather than a constant of this module.
        assert_eq!(slug(3, &[stream(8, &bytes[..24])]), None);
        assert_eq!(slug(3, &[stream(8, &bytes[..23])]), SPAN);
        assert_eq!(slug(3, &[stream(8, &bytes[..16])]), SPAN);
        assert_eq!(
            slug(3, &[stream(16, &bytes[..24])]),
            SPAN,
            "the same bytes cover fewer vertices at a wider stride"
        );
        assert_eq!(slug(4, &[stream(4, &bytes[..24])]), None);

        // One short stream refuses the draw, and the sentence names which one —
        // the census reads the shape, and the shape here is a position in the
        // request's own fetch-table order.
        let streams = [stream(8, &bytes[..24]), stream(8, &bytes[..16])];
        let refusal = nonindexed_vertex_span(&streams, 3).expect("the second stream is short");
        assert_eq!(refusal.slug(), "render_provider_out_of_class_vertex_span");
        assert!(
            refusal.detail().contains("binding 1"),
            "the refusal names the stream it measured: {}",
            refusal.detail()
        );
    }
}
