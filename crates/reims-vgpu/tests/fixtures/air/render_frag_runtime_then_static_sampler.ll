; Owned synthetic fixture for the reims render rail's sampler-family increment
; (R37): the sampler-family fixture's sibling with the two texture arguments
; *swapped* — `[[texture(0)]]` is the one the runtime `[[sampler(0)]]` argument
; samples, `[[texture(1)]]` the one the module's own AIR `constexpr sampler`
; samples.
;
; Reading the two forms means reading two different pairings: the AIR static
; sampler pairs *positionally* with the sampled textures that read through it
; (the rule C1b states), while a runtime `[[sampler(n)]]` argument pairs with
; the texture whose own sample site names it. This fixture is where the two
; rules give different answers for the same texture — the first sampled texture
; is the runtime one — so a walk that reached for the positional static pairing
; first would declare `[[texture(0)]]` as static-sampled and hand the pass a
; declaration the module contradicts. The declarations test pins the walk's
; answer on both fixtures.
;
; The readings are the sibling fixture's exactly: red is the static half's
; sample at (0.8125, 0.875) -> texel (6, 3), green the runtime half's at
; (1.375, 0.875) -> texel (7, 3) clamped and (3, 3) repeated, blue the runtime
; half's at (0.3125, 0.875) -> texel (2, 3). The two textures are bound with
; different bytes by the tests, which is what makes "each half reads its own
; texture" observable in the frame rather than assumed.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_runtime_then_static_sampler.air \
;           tests/fixtures/air/render_frag_runtime_then_static_sampler.ll
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_runtime_then_static_sampler.air"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define <4 x float> @reims_runtime_then_static_sampler_frag(ptr addrspace(1) readonly captures(none) %runtime_texture, ptr addrspace(1) readonly captures(none) %static_texture, ptr addrspace(2) readonly captures(none) %runtime_sampler) local_unnamed_addr {
entry:
  %static_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %static_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 8.125000e-01, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %static_value = extractvalue { <4 x float>, i8 } %static_pair, 0
  %static = extractelement <4 x float> %static_value, i64 0
  %addressed_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %runtime_texture, ptr addrspace(2) readonly captures(none) %runtime_sampler, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %addressed_value = extractvalue { <4 x float>, i8 } %addressed_pair, 0
  %addressed = extractelement <4 x float> %addressed_value, i64 0
  %filtered_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %runtime_texture, ptr addrspace(2) readonly captures(none) %runtime_sampler, <2 x float> <float 3.125000e-01, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %filtered_value = extractvalue { <4 x float>, i8 } %filtered_pair, 0
  %filtered = extractelement <4 x float> %filtered_value, i64 0
  %with_red = insertelement <4 x float> undef, float %static, i32 0
  %with_green = insertelement <4 x float> %with_red, float %addressed, i32 1
  %with_blue = insertelement <4 x float> %with_green, float %filtered, i32 2
  %alpha = insertelement <4 x float> %with_blue, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @reims_runtime_then_static_sampler_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !7}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"runtime_texture"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"static_texture"}
!7 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"runtime_sampler"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
