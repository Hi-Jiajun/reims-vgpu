; AIR fixture: a fragment stage that stores two colour locations while the pass
; attaches one (2026-09-20, the third door behind census v46's
; `stage_buffer_footprint` bucket).
;
; The twin of `render_frag.air`: the same unconditional `Location 0` store of a
; four-component colour — `(64/255, 128/255, 192/255, 1)`, which an 8-bit UNORM
; attachment reads back as `40 80 c0 ff` — plus a *second*, equally
; unconditional `Location 1` store the pass's one attachment has no destination
; for. Vulkan discards a fragment store whose location has no attachment behind
; it, so the frame is the first location's texel; a rail that bound the second
; store to the attached location would land `ff 00 00 ff` instead, which is what
; makes the shape falsifiable rather than assumed.
;
; The byte/255 constants are the reviewed fixture's own: a half-integer tie such
; as `0.5 * 255` is resolved differently by different drivers
; (`research/docs/23` §3.5), while byte/255 values sit far from a tie.
;
; Assembled exactly as the rest of this directory is, from this source:
;
;   llvm-as render_frag_two_outputs.ll -o render_frag_two_outputs.air
;
; (`llvm-as` 22.1.8 reproduces every committed `.air` here byte for byte, which
; is what makes this source the fixture's own definition rather than a note
; beside a binary nobody can rebuild.)
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <{ <4 x float>, <4 x float> }> @reims_two_outputs_frag() {
entry:
  %r = insertelement <4 x float> undef, float 0.250980406999588, i32 0
  %g = insertelement <4 x float> %r, float 0.501960813999176, i32 1
  %b = insertelement <4 x float> %g, float 0.7529411911964417, i32 2
  %a = insertelement <4 x float> %b, float 1.000000e+00, i32 3
  %d0 = insertelement <4 x float> undef, float 1.000000e+00, i32 0
  %d1 = insertelement <4 x float> %d0, float 0.000000e+00, i32 1
  %d2 = insertelement <4 x float> %d1, float 0.000000e+00, i32 2
  %dropped = insertelement <4 x float> %d2, float 1.000000e+00, i32 3
  %out0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %a, 0
  %out1 = insertvalue <{ <4 x float>, <4 x float> }> %out0, <4 x float> %dropped, 1
  ret <{ <4 x float>, <4 x float> }> %out1
}

!air.fragment = !{!0}
!0 = !{ptr @reims_two_outputs_frag, !1, !2}
!1 = !{!3, !4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{}
