; Owned synthetic fixture for the reims render rail's scalar vertex-lane
; increment (2026-09-20, census v46's `vertex_format` bucket). It is not
; derived from a third-party metallib: it is a hand-written AIR module whose
; interface is the shape that increment admits — one `float2` vertex attribute
; at location 0 that becomes the clip position, and one *scalar* `float`
; attribute at location 1 whose value divides it.
;
; It exists because the canonical contract's appended `float32x1` lane pairs
; with a scalar AIR member, while every fixture before this one reads vectors
; at its position: a scalar storage declared beside `reims_indexed_tri.air`'s
; `float2`-only interface is answered by the provider's own shape rule
; (`render_stage_reflection_mismatch`) rather than executed, and a vector
; storage declared beside this fixture's scalar member is refused the same way.
;
; The divisor is load-bearing rather than decorative: `w = 1` is the reviewed
; triangle, and `w = 2` halves both components, so a rail that dropped the
; scalar fetch (or read it as the sentinel) lands the `w = 1` frame for a
; request that moved *only* the scalar, which is what the rail test's moved
; arm asserts. The division is exact in binary (`1.0`, `2.0`, `0.5`), so both
; rails' own arithmetic rounds it identically and parity stays an assertion
; about stream plumbing.
;
; `reims_indexed_tri_scalar.air` is `llvm-as` output of this file:
;
;     llvm-as -o tests/fixtures/air/reims_indexed_tri_scalar.air \
;             tests/fixtures/air/reims_indexed_tri_scalar.ll
;
; The fragment half of the pair is the existing `render_frag.air` (entry
; `fmain`, `float4(0.25, 0.5, 0.75, 1)`).
source_filename = "reims_indexed_tri_scalar.air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_scalar_vertex(<2 x float> %pos, float %w) local_unnamed_addr {
entry:
  %w0 = insertelement <2 x float> poison, float %w, i32 0
  %w2 = insertelement <2 x float> %w0, float %w, i32 1
  %scaled = fdiv <2 x float> %pos, %w2
  %padded = shufflevector <2 x float> %scaled, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %clip = shufflevector <4 x float> %padded, <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
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
!13 = !{ptr @reims_scalar_vertex, !14, !15}
!14 = !{!16}
!15 = !{!17, !18}
!16 = !{!"air.position", !"air.arg_type_name", !"float4"}
!17 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"pos"}
!18 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"w"}
