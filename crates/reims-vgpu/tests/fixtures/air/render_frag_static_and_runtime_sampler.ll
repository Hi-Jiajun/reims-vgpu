; Owned synthetic fixture for the reims render rail's sampler-family increment
; (R37, `research/docs/23` §102): one fragment stage that carries *both* sampler
; forms at once — `[[texture(0)]]` is sampled through the module's own AIR
; `constexpr sampler` (`@__air_sampler_state`), `[[texture(1)]]` through a
; runtime `[[sampler(0)]]` argument whose state only the pass states.
;
; The class gate this increment removes
; (`render_provider_out_of_class_texture_sampler_family`) answered every stage
; of this shape by name on the sentence "the canonical rail's registration
; rules refuse it by name". The canonical side falsified that sentence
; (`metal-api-emulator` `crates/metal-api-vulkan/tests/render_sampler_family_e2e.rs`,
; E `41308a1`): each binding is admitted in exactly one of the two forms, and
; nothing refuses the two forms in one stage. What the fixture pins on this
; rail is that each texture is declared in the form the module's own sample
; site names, one texture at a time.
;
; The two halves are separately observable in the attachment's bytes:
;
;   red   = static_texture.sample(constexpr, (0.8125, 0.875)).x  -> texel (6, 3)
;   green = runtime_texture.sample(sampler 0, (1.375, 0.875)).x  -> clamp: texel (7, 3)
;                                                                  repeat: texel (3, 3)
;   blue  = runtime_texture.sample(sampler 0, (0.3125, 0.875)).x -> texel (2, 3), either mode
;   alpha = 1.0
;
; Against the rail's own 8x4 test texture (`sampled_texels`, whose red channel
; is `16 * x` per column) the fragment lands `60 70 20 ff` under runtime
; nearest + clamp and `60 30 20 ff` under runtime nearest + repeat: the red
; channel is the *static* half's reading — the value the static-only sibling
; fixture lands for the same sample and state — and the green/blue pair is the
; *runtime* half's. Moving the static texture's texel (6, 3) therefore moves
; red alone, and moving the runtime texture's texel (7, 3) moves green alone.
;
; The vertex half of the pair is the existing `reims_indexed_tri.air` (entry
; `reims_indexed_vertex`); the fragment takes no varying, which is why the two
; compose.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_static_and_runtime_sampler.air \
;           tests/fixtures/air/render_frag_static_and_runtime_sampler.ll
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_static_and_runtime_sampler.air"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define <4 x float> @reims_static_and_runtime_sampler_frag(ptr addrspace(1) readonly captures(none) %static_texture, ptr addrspace(1) readonly captures(none) %runtime_texture, ptr addrspace(2) readonly captures(none) %runtime_sampler) local_unnamed_addr {
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
!0 = !{ptr @reims_static_and_runtime_sampler_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !7}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"static_texture"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"runtime_texture"}
!7 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"runtime_sampler"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
