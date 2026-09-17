; Owned synthetic fixture for the reims render rail's vertex-stream numbering
; (`research/docs/26` §31, R9p): a vertex stage that reads four `[[stage_in]]`
; attributes *and* a `[[buffer(2)]]` argument.
;
; It is the shape the 2026-09-17 census v6 measured as the stage-buffer door's
; 99.23 %: four attributes, and a vertex-stage buffer argument whose Metal index
; (2) sits under the attribute count but above the number of *streams* the
; descriptor really reads. The four attributes arrive from two interleaved
; tables in the rail's own request — locations 0 and 1 out of the first,
; locations 2 and 3 out of the second — so a rail that numbers one canonical
; stream per attribute occupies 0..3 and collides with the argument, while one
; that numbers one stream per table occupies 0..1 and does not.
;
; The `[[buffer(2)]]` bytes reach the frame (`+ tail`), so a rail that stated
; the declaration without binding the view, or that dropped the stream the
; attributes read, lands a different frame rather than the same one by accident.
; The `fast` flag runs on the adds for the reason the fixtures beside this one
; state — a withheld permission turns the module into one whose admission is the
; executing device's own answer (R8/R8b).
;
; `reims_indexed_tri_two_stream_buffer2.air` is `llvm-as` output of this file:
;
;     llvm-as -o tests/fixtures/air/reims_indexed_tri_two_stream_buffer2.air \
;             tests/fixtures/air/reims_indexed_tri_two_stream_buffer2.ll
;
; The fragment half of the pair is the existing `render_frag.air` (entry
; `fmain`, `float4(0.25, 0.5, 0.75, 1)`).
source_filename = "reims_indexed_tri_two_stream_buffer2.air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_two_stream_buffer2_vertex(<2 x float> %pos, <2 x float> %a, <2 x float> %b, <2 x float> %c, ptr addrspace(1) %tail) local_unnamed_addr {
entry:
  %ab = fadd fast <2 x float> %a, %b
  %abc = fadd fast <2 x float> %ab, %c
  %slot = getelementptr <2 x float>, ptr addrspace(1) %tail, i32 0
  %t = load <2 x float>, ptr addrspace(1) %slot, align 8
  %tail_shift = fadd fast <2 x float> %abc, %t
  %shifted = fadd fast <2 x float> %pos, %tail_shift
  %padded = shufflevector <2 x float> %shifted, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
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
!13 = !{ptr @reims_two_stream_buffer2_vertex, !14, !15}
!14 = !{!16}
!15 = !{!17, !18, !19, !20, !21}
!16 = !{!"air.position", !"air.arg_type_name", !"float4"}
!17 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"pos"}
!18 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"a"}
!19 = !{i32 2, !"air.vertex_input", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"b"}
!20 = !{i32 3, !"air.vertex_input", !"air.location_index", i32 3, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"c"}
!21 = !{i32 4, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float2*", !"air.arg_name", !"tail"}
