; AIR fixture: a fragment stage with two `[[buffer(N)]]` arguments, exactly one
; of which its entry point reaches (R9n).
;
; The mixed shape the unread population actually has: the translations a draw
; carries name several buffers, and only *some* of them are unread. Here
; `[[buffer(0)]]` is read — the red component comes from it, exactly as in
; `render_frag_buffer.air` — while `[[buffer(1)]]` is declared by the metadata
; and never dereferenced, exactly as in `render_frag_buffer_unused.air`.
;
; The seam's statement drops the unread slot (R9m) and, since R9n, still states
; the reached one: the wire frame has to carry the read declaration with its own
; view beside it while the unread slot takes no part in the pairing. That is the
; shape `provider_render_rail.rs`'s R9n mixed-slot test pins.
;
;   llvm-as render_frag_buffer_read_and_unused.ll -o render_frag_buffer_read_and_unused.air
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_read_and_unused_buffer_frag(ptr addrspace(1) %tint, ptr addrspace(1) %extra) {
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
!0 = !{ptr @reims_read_and_unused_buffer_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"tint"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"extra"}
