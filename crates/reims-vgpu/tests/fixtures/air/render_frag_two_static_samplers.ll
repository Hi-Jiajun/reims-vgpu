; Owned synthetic fixture for the reims render rail's sampler-family increment
; (R37): the shape the class still answers by name after the two sampler forms
; stopped being a refusal — a fragment stage whose AIR static samplers and
; sampled textures state *different pairings*.
;
; `[[texture(0)]]` is sampled twice, once through each of the module's two AIR
; `constexpr samplers` (nearest + clamp, then linear + clamp), while
; `[[texture(1)]]` is sampled through the runtime `[[sampler(0)]]` argument.
; The two pairing rules the canonical render rail holds disagree on the counts:
; one AIR static sampler pairs positionally with one sampled texture that reads
; through one, so this module's two AIR samplers would need two such textures
; where the stage declares one — the registration refuses the stage by name
; (`render_stage_reflection_mismatch`, "this rail pairs one AIR static sampler
; with one sampled texture") rather than executing a texture through a sampler
; state nothing pairs it with.
;
; The two constexpr states are the corpus's own words: `34901797601017929` is
; the nearest + clamp state every sibling fixture carries, and
; `34901797601020489` is the linear + clamp state of the canonical rail's
; `sample_texture_2d_linear_clamp.ll` (the same owned-synthetic encoding).
;
; Both samplers stay live in the emitted module — the second sample is the blue
; channel — because the refusal is about what the *stage* declares, not about a
; dead call the emitter would drop. The frame is never read: the class answers
; the shape before the provider is asked.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_two_static_samplers.air \
;           tests/fixtures/air/render_frag_two_static_samplers.ll
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_two_static_samplers.air"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state_linear = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0], align 8

define <4 x float> @reims_two_static_samplers_frag(ptr addrspace(1) readonly captures(none) %static_texture, ptr addrspace(1) readonly captures(none) %runtime_texture, ptr addrspace(2) readonly captures(none) %runtime_sampler) local_unnamed_addr {
entry:
  %nearest_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %static_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 8.125000e-01, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %nearest_value = extractvalue { <4 x float>, i8 } %nearest_pair, 0
  %nearest = extractelement <4 x float> %nearest_value, i64 0
  %addressed_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %runtime_texture, ptr addrspace(2) readonly captures(none) %runtime_sampler, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %addressed_value = extractvalue { <4 x float>, i8 } %addressed_pair, 0
  %addressed = extractelement <4 x float> %addressed_value, i64 0
  %linear_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %static_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state_linear, <2 x float> <float 3.125000e-01, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %linear_value = extractvalue { <4 x float>, i8 } %linear_pair, 0
  %linear = extractelement <4 x float> %linear_value, i64 0
  %with_red = insertelement <4 x float> undef, float %nearest, i32 0
  %with_green = insertelement <4 x float> %with_red, float %addressed, i32 1
  %with_blue = insertelement <4 x float> %with_green, float %linear, i32 2
  %alpha = insertelement <4 x float> %with_blue, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!6, !8}
!0 = !{ptr @reims_two_static_samplers_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !7}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"static_texture"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"runtime_texture"}
!7 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"runtime_sampler"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!8 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state_linear}
