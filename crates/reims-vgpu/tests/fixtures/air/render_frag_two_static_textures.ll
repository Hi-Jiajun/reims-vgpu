; Owned synthetic fixture for the reims render rail's static-sampler pairing
; increment (R45, census v39): the *positive* half of that rule — one fragment
; entry that carries two AIR `constexpr samplers` beside the two sampled
; textures that read through them, one each, so the two counts the registration
; compares agree.
;
; This is the shape the increment must keep admitting: the refusal beside it
; (`render_frag_unpaired_static_sampler.ll`) carries the same two AIR samplers
; with one of them named by no sample site, which the registration refuses
; (`render_stage_reflection_mismatch`). A walk that answered "two AIR samplers"
; instead of "one sampler per sampled texture" would refuse both, and a walk
; that paired them by *position* alone would be right here by accident — so the
; two halves are separately observable in the attachment's bytes:
;
;   red   = nearest_texture.sample(constexpr nearest, (0.8125, 0.875)).x -> texel (6, 3)
;   green = 0.0
;   blue  = linear_texture.sample(constexpr linear, (0.3125, 0.875)).x   -> texel (2, 3)
;   alpha = 1.0
;
; Against the rail's own 8x4 test texture (`sampled_texels`, whose red channel is
; `16 * x` per column) the fragment lands `96 0 32 ff`: moving the first
; texture's texel `(6, 3)` moves red alone and moving the second texture's texel
; `(2, 3)` moves blue alone, which is what makes "each texture read through the
; sampler its own sample site names" a reading of the frame.
;
; The two states are the corpus's own words: `34901797601017929` is the nearest
; + clamp state every sibling fixture carries and `34901797601020489` the linear
; + clamp state of the canonical rail's `sample_texture_2d_linear_clamp.ll`. Both
; sample points are exact texel centres, where a linear tap of one texel is that
; texel's bytes.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_two_static_textures.air \
;           tests/fixtures/air/render_frag_two_static_textures.ll
;
; The vertex half of the pair is the existing `reims_indexed_tri.air` (entry
; `reims_indexed_vertex`); the fragment takes no varying, which is why the two
; compose.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_two_static_textures.air"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state_linear = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0], align 8

define <4 x float> @reims_two_static_textures_frag(ptr addrspace(1) readonly captures(none) %nearest_texture, ptr addrspace(1) readonly captures(none) %linear_texture) local_unnamed_addr {
entry:
  %nearest_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %nearest_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 8.125000e-01, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %nearest_value = extractvalue { <4 x float>, i8 } %nearest_pair, 0
  %nearest = extractelement <4 x float> %nearest_value, i64 0
  %linear_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %linear_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state_linear, <2 x float> <float 3.125000e-01, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %linear_value = extractvalue { <4 x float>, i8 } %linear_pair, 0
  %linear = extractelement <4 x float> %linear_value, i64 0
  %with_red = insertelement <4 x float> undef, float %nearest, i32 0
  %with_green = insertelement <4 x float> %with_red, float 0.000000e+00, i32 1
  %with_blue = insertelement <4 x float> %with_green, float %linear, i32 2
  %alpha = insertelement <4 x float> %with_blue, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!6, !8}
!0 = !{ptr @reims_two_static_textures_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"nearest_texture"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"linear_texture"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!8 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state_linear}
