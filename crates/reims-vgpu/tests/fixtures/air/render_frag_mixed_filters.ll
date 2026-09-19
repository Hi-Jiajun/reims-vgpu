; Owned synthetic fixture for the reims render rail's static-sampler pairing
; increment (R48, the R-side follow-up to E-RS5/v118): the *bound* sibling of
; `render_frag_pixel_sampler.ll` — one fragment stage that samples its texture
; through the AIR `constexpr sampler` it declares, whose state is outside the
; canonical family on the min/mag filter axis (nearest minification, linear
; magnification).
;
; The state is the corpus's nearest + clamp word (`34901797601017929`, every
; sibling fixture's) with the magnification filter raised to `linear`, which is
; the field at bits 9-10: `34901797601018441`. The coordinates stay normalized
; (the centre of texel `(6, 3)` of an 8x4 surface, `0.8125, 0.875`), so the
; pinned translator lowers this sample to a genuine `OpSampledImage` — the
; module really reads through this sampler, and a walk that bypassed its state
; would execute a filter the module named and did not get.
;
; Why this fixture exists beside the pixel-coordinate one: the class weighs the
; AIR samplers the module's own instructions read through (E-RS5/v118; R48), so
; the two fixtures are the two sides of that predicate. The pixel-coordinate
; state is a sample site the translator turned into texel fetches, so no
; `OpSampledImage` names it and the class admits the draw to the provider (its
; own case). This one *is* named by the module, so the class weighs it, reads
; its state outside the family on this axis, and keeps the draw on the engine by
; name (`render_provider_out_of_class_texture_sampler`, the per-declaration
; `RenderSamplerRefusal::AirState` arm) rather than handing the provider a
; registration refusal the engine never gets to answer (census v39's red line,
; R45's counting rule).
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_mixed_filters.air \
;           tests/fixtures/air/render_frag_mixed_filters.ll
;
; The vertex half of the pair is the existing `reims_indexed_tri.air` (entry
; `reims_indexed_vertex`), whose stage-in position is the fragment's only
; interface — the fragment takes no varying, which is why the two compose.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_mixed_filters.air"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601018441, i64 0], align 8

define <4 x float> @reims_mixed_filters_frag(ptr addrspace(1) readonly captures(none) %texture) local_unnamed_addr {
entry:
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 8.125000e-01, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value = extractvalue { <4 x float>, i8 } %sample, 0
  ret <4 x float> %value
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @reims_mixed_filters_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"texture"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
