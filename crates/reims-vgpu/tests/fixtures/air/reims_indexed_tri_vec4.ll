; Owned synthetic fixture for the reims render rail's normalized-storage
; increment (`research/docs/26` §37, R14). It is not derived from a
; third-party metallib: it is a hand-written AIR module whose interface is the
; four-component corner of the shape that increment admits — one `float4`
; vertex attribute at location 0 whose `x` and `y` become the clip position
; and whose `w` is the clip `w` the position divides by.
;
; It exists because the canonical contract's four-component normalized
; storages (`unorm8x4` / `unorm16x4`, `research/docs/23` §103) pair with a
; `float4` AIR member, while every fixture before this one reads a `float2` at
; its position: `reims_indexed_tri.air`'s interface is the two-component
; corner, so a `unorm8x4` declaration beside it is answered by the provider's
; own shape rule (`render_stage_reflection_mismatch`) rather than executed.
;
; The `w` component is load-bearing rather than decorative: the position it
; divides stays inside the viewport while the attribute's fourth byte is the
; sentinel, so a rail that fetched two components — or read the storage as
; `unorm8x2` — lands the same frame for a request that moved *only* the fourth
; component, and the test that moves it is what makes the four-component fetch
; falsifiable rather than asserted.
;
; `reims_indexed_tri_vec4.air` is `llvm-as` output of this file:
;
;     llvm-as -o tests/fixtures/air/reims_indexed_tri_vec4.air \
;             tests/fixtures/air/reims_indexed_tri_vec4.ll
;
; The fragment half of the pair is the existing `render_frag.air` (entry
; `fmain`, `float4(0.25, 0.5, 0.75, 1)`).
source_filename = "reims_indexed_tri_vec4.air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_vec4_vertex(<4 x float> %pos) local_unnamed_addr {
entry:
  %clip = shufflevector <4 x float> %pos, <4 x float> <float poison, float poison, float 0.000000e+00, float poison>, <4 x i32> <i32 0, i32 1, i32 6, i32 3>
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
!13 = !{ptr @reims_vec4_vertex, !14, !15}
!14 = !{!16}
!15 = !{!17}
!16 = !{!"air.position", !"air.arg_type_name", !"float4"}
!17 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
