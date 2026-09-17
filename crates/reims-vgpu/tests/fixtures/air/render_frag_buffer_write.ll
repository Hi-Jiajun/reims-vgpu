; AIR fixture: a fragment stage that declares a *writable* `[[buffer(0)]]`
; argument and writes it (R9b).
;
; The `Writable` arm of the class gate's split: the writeback landing a
; `BufferAccess::Write`/`ReadWrite` stage buffer would need is an increment of
; its own on both rails (the canonical contract refuses the writable arms by
; name before admission), so the door has to name this population apart from
; the read-only one rather than folding the two into "the draw binds a buffer".
;
;   llvm-as render_frag_buffer_write.ll -o render_frag_buffer_write.air
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_write_buffer_frag(ptr addrspace(1) %tint) {
entry:
  %word = load i32, ptr addrspace(1) %tint, align 4
  %red = bitcast i32 %word to float
  store i32 %word, ptr addrspace(1) %tint, align 4
  %r = insertelement <4 x float> undef, float %red, i32 0
  %g = insertelement <4 x float> %r, float 0.000000e+00, i32 1
  %b = insertelement <4 x float> %g, float 0.000000e+00, i32 2
  %a = insertelement <4 x float> %b, float 1.000000e+00, i32 3
  ret <4 x float> %a
}

!air.fragment = !{!0}
!0 = !{ptr @reims_write_buffer_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"tint"}
