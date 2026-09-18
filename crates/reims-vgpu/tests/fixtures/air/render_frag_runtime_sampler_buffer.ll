; AIR fixture (R12): the runtime-sampler fixture's shape beside one
; `[[buffer(0)]]` argument — the fragment stage a guest draw carries when its
; sampling state and its constants travel together.
;
; The buffer is read, so the seam states its declaration, and a stated stage
; buffer is what makes the trace cross the owner→provider wire. The command
; channel carries the pass's sampled textures (the v70 tag) *and*, since E-TX4
; (`research/docs/23` §3.3, v102/v111, tags `0x15..=0x18`), its runtime
; `[[sampler(n)]]` list, so the shape leaves for the provider on a device that
; declares the render-sampler section — `provider_render_rail.rs` pins both
; arms of that answer, and the bucket R12 gave the shape
; (`render_provider_out_of_class_texture_sampler_wire`) now counts the devices
; that do not declare it rather than the shape itself.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_runtime_sampler_buffer.air \
;           tests/fixtures/air/render_frag_runtime_sampler_buffer.ll
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_runtime_sampler_buffer.air"

define <4 x float> @reims_runtime_sampled_buffer_frag(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) %sampler, ptr addrspace(1) %tint) local_unnamed_addr {
entry:
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) %sampler, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value = extractvalue { <4 x float>, i8 } %sample, 0
  %word = load i32, ptr addrspace(1) %tint, align 4
  %red = bitcast i32 %word to float
  %out = insertelement <4 x float> %value, float %red, i32 0
  ret <4 x float> %out
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!0 = !{ptr @reims_runtime_sampled_buffer_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !6}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"texture"}
!5 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"tint"}
