; Owned synthetic fixture for the reims render rail's texel-fetch increment
; (R15, `research/docs/23` §3.3, v105): the sibling of
; `render_frag_fetch_texture_2d.ll`, differing only in the two texels it reads.
;
; Same stage, same binding, same declaration — the coordinates move from
; (1, 0).x / (0, 1).y to (3, 0).x / (0, 3).y, so against the same
; `(16 * i, 64 * j, 8 * (i + j), 255)` texture the frame moves from
; `10 40 00 ff` to `30 c0 00 ff`. That is the falsification the fetch face
; owes: the declaration and the module are unchanged between the two fixtures,
; so the bytes can only have followed the coordinates.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_fetch_texture_2d_far.air \
;           tests/fixtures/air/render_frag_fetch_texture_2d_far.ll
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_fetch_texture_2d_far.frag.metal"

define <4 x float> @reims_fetch_far_frag(ptr addrspace(1) readonly captures(none) %tex) local_unnamed_addr {
entry:
  %left_pair = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, <2 x i32> <i32 3, i32 0>, i32 0, i32 0)
  %left_value = extractvalue { <4 x float>, i8 } %left_pair, 0
  %left = extractelement <4 x float> %left_value, i64 0
  %right_pair = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, <2 x i32> <i32 0, i32 3>, i32 0, i32 0)
  %right_value = extractvalue { <4 x float>, i8 } %right_pair, 0
  %right = extractelement <4 x float> %right_value, i64 1
  %red = insertelement <4 x float> undef, float %left, i32 0
  %green = insertelement <4 x float> %red, float %right, i32 1
  %blue = insertelement <4 x float> %green, float 0.000000000000000e+00, i32 2
  %alpha = insertelement <4 x float> %blue, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), <2 x i32>, i32, i32) local_unnamed_addr

!air.fragment = !{!0}
!0 = !{ptr @reims_fetch_far_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"tex"}
