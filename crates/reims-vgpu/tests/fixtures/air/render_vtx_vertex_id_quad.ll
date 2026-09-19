; Owned synthetic fixture for the reims render rail's *layout-free count*
; increment (2026-09-19, census v45's `vertex_span` bucket): the vertex stage
; computes its clip position from `[[vertex_id]]` alone — no `[[stage_in]]`
; attribute and no `[[buffer(n)]]` argument — which is exactly the module the
; census shape's pipelines carry (`attrs=0`, the bound streams nobody reads).
;
; The six vertices are the left half of a quad, in two triangles that share the
; seam `x = 0`:
;
;   id 0 -> (-1,-1)   id 1 -> (0,-1)   id 2 -> (-1,1)     lower triangle
;   id 3 -> (0,-1)    id 4 -> (0,1)    id 5 -> (-1,1)     upper triangle
;
; so a three-vertex draw rasterizes the first triangle alone and a six-vertex
; draw covers the whole left half. The seam and both hypotenuses pass *between*
; the declared window's pixel centres, which is what makes "the rail issued the
; count the draw named" a byte-for-byte reading instead of a fill rule's answer.
;
; `render_vtx_vertex_id_quad.air` is `llvm-as` output of this file:
;
;     llvm-as -o tests/fixtures/air/render_vtx_vertex_id_quad.air \
;             tests/fixtures/air/render_vtx_vertex_id_quad.ll
;
; The fragment half of the pair is the existing `render_frag.air` (entry
; `fmain`, `float4(0.25, 0.5, 0.75, 1)`), so the covered texels are the
; fragment's own colour and the rest keep the pass's clear.
source_filename = "render_vtx_vertex_id_quad.air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_vertex_id_quad(i32 %vertex_id) local_unnamed_addr {
entry:
  %is_one = icmp eq i32 %vertex_id, 1
  %is_two = icmp eq i32 %vertex_id, 2
  %is_three = icmp eq i32 %vertex_id, 3
  %is_four = icmp eq i32 %vertex_id, 4
  %is_five = icmp eq i32 %vertex_id, 5
  %x_one = select i1 %is_one, float 0.000000e+00, float -1.000000e+00
  %x_three = select i1 %is_three, float 0.000000e+00, float %x_one
  %x_shift = select i1 %is_four, float 0.000000e+00, float %x_three
  %y_two = select i1 %is_two, float 1.000000e+00, float -1.000000e+00
  %y_four = select i1 %is_four, float 1.000000e+00, float %y_two
  %y_shift = select i1 %is_five, float 1.000000e+00, float %y_four
  %p0 = insertelement <4 x float> undef, float %x_shift, i32 0
  %p1 = insertelement <4 x float> %p0, float %y_shift, i32 1
  %p2 = insertelement <4 x float> %p1, float 0.000000e+00, i32 2
  %clip = insertelement <4 x float> %p2, float 1.000000e+00, i32 3
  ret <4 x float> %clip
}

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}
!llvm.ident = !{!7}
!air.version = !{!8}
!air.language_version = !{!9}
!air.compile_options = !{!10, !11, !12}
!air.vertex = !{!13}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 7, !"air.max_device_buffers", i32 31}
!3 = !{i32 7, !"air.max_constant_buffers", i32 31}
!4 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!5 = !{i32 7, !"air.max_textures", i32 128}
!6 = !{i32 7, !"air.max_read_write_textures", i32 8}
!7 = !{!"Apple metal version 32023.884 (metalfe-32023.884)"}
!8 = !{i32 2, i32 8, i32 0}
!9 = !{!"Metal", i32 4, i32 0, i32 0}
!10 = !{!"air.compile.denorms_disable"}
!11 = !{!"air.compile.fast_math_enable"}
!12 = !{!"air.compile.framebuffer_fetch_enable"}
!13 = !{ptr @reims_vertex_id_quad, !14, !15}
!14 = !{!16}
!15 = !{!17}
!16 = !{!"air.position", !"air.arg_type_name", !"float4"}
!17 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vertex_id"}
