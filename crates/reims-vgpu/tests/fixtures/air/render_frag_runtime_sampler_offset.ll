; Owned synthetic fixture for the texel space (2026-09-19, census v43's
; `texture_state` axis): the runtime-sampler fixture one axis over — its one
; fixed sample point carries a **non-zero constant offset**.
;
; The offset is what an unnormalized `VkSampler` may not be used with
; (`VUID-vkCmdDraw-None-08611`), so this module has no explicit-LOD sibling:
; the class answers a draw that binds the texel space to it by keeping it on
; the engine (`render_provider_out_of_class_texture_state`, the census's own
; sentence) instead of handing the provider a draw it would refuse by name.
; The normalized arms keep running unchanged, offset and all.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_runtime_sampler_offset.air \
;           tests/fixtures/air/render_frag_runtime_sampler_offset.ll
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_runtime_sampler_offset.air"

define <4 x float> @reims_runtime_sampled_offset_frag(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) %sampler) local_unnamed_addr {
entry:
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) %sampler, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> <i32 1, i32 0>, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value = extractvalue { <4 x float>, i8 } %sample, 0
  ret <4 x float> %value
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!0 = !{ptr @reims_runtime_sampled_offset_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"texture"}
!5 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
