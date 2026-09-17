; AIR fixture: a fragment stage that reads its colour through a `[[buffer(0)]]`
; argument (R9b).
;
; The twin of `render_frag.air`: the same single `Location 0` store of a
; four-component colour, with the red component loaded from the stage's own
; Metal buffer instead of being an immediate. It is the one fixture in this
; directory whose translation *declares* a buffer, which is the fact the
; canonical render rail's class gate answers on
; (`backend/provider_render.rs::stage_buffer_gate`): the provider's translated
; registration refuses such a stage by name
; (`render_stage_unsupported_interface`, field `bindings`) because its Vulkan
; rail executes pipeline-level buffers through its reviewed fixture pair alone.
;
; Assembled exactly as the rest of this directory is, from this source:
;
;   llvm-as render_frag_buffer.ll -o render_frag_buffer.air
;
; (`llvm-as` 22.1.8 reproduces every committed `.air` here byte for byte, which
; is what makes this source the fixture's own definition rather than a note
; beside a binary nobody can rebuild.)
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_buffer_frag(ptr addrspace(1) %tint) {
entry:
  %word = load i32, ptr addrspace(1) %tint, align 4
  %red = bitcast i32 %word to float
  %r = insertelement <4 x float> undef, float %red, i32 0
  %g = insertelement <4 x float> %r, float 0.000000e+00, i32 1
  %b = insertelement <4 x float> %g, float 0.000000e+00, i32 2
  %a = insertelement <4 x float> %b, float 1.000000e+00, i32 3
  ret <4 x float> %a
}

!air.fragment = !{!0}
!0 = !{ptr @reims_buffer_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"tint"}
