; Owned synthetic fixture for the reims render rail's static-sampler pairing
; increment (R45, census v39): the R10 sampled fixture's own body with the
; sampler state moved to `coord::pixel` — the *one* shape census v39's four
; `draws_skipped_after_engine_refusal` carry.
;
; The state is the corpus's nearest + clamp word (`34901797601017929`, every
; sibling fixture's) with bit 15 — the coordinate field — set, which is
; `34901797601050697`: nearest filtering, one mip level, clampToEdge addressing,
; `coord::pixel`. That is the state the engine's own conformed line names beside
; the registration's refusal (`min=0 mag=0 mip=0 address_u=0 address_v=0
; aniso=1` for a clampToEdge sibling), and it is the state the translation
; lowers with shader-side fetches rather than a sampler (`metal2vulkan`'s
; `validate_lowering`).
;
; What makes this fixture the red line: the lowering is a *fetch*, so this
; rail's own read of the module's sample sites says the texture is fetched and
; declares it with no sampler at all — while the reflection still reports the
; AIR `constexpr sampler` it was lowered *from*. R45 read that leftover state as
; the class's own business — the registration then weighed every AIR sampler a
; stage carried, so it refused this module twice: by the state
; (`render_stage_unsupported_interface`, "an AIR sampler with pixel coordinates
; is outside the reviewed family") and by the counts
; (`render_stage_reflection_mismatch`, "this rail pairs one AIR static sampler
; with one sampled texture") — and R48 re-reads it the way E-RS5/v118 narrowed
; the registration: a sampler no `OpSampledImage` names is weighed by neither
; rail, so this fixture is the shape the class now *admits* to the provider. Its
; `render_frag_mixed_filters.ll` sibling is the bound half that still stays on
; the engine by name.
;
; The sample's coordinates are the centre of texel `(6, 3)` in *both* the
; conventions a pixel-coordinate read can use — `floor(6.5, 3.5)` is that texel,
; and so is the nearest texel centre — so the engine's fetch and any other
; lowering of the same state name one texel, and the frame this fixture lands is
; the sibling fixture's own red channel at that texel.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_pixel_sampler.air \
;           tests/fixtures/air/render_frag_pixel_sampler.ll
;
; The vertex half of the pair is the existing `reims_indexed_tri.air` (entry
; `reims_indexed_vertex`), whose stage-in position is the fragment's only
; interface — the fragment takes no varying, which is why the two compose.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_pixel_sampler.air"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8

define <4 x float> @reims_pixel_sampler_frag(ptr addrspace(1) readonly captures(none) %texture) local_unnamed_addr {
entry:
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 6.500000e+00, float 3.500000e+00>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value = extractvalue { <4 x float>, i8 } %sample, 0
  ret <4 x float> %value
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @reims_pixel_sampler_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"texture"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
