; Owned synthetic fixture for the reims render rail's *affine* stage-buffer
; increment (`research/docs/26` §R9j): the vertex stage reads its clip position
; out of `positions[vertex_id]` — one `float2` per vertex — instead of taking a
; `[[stage_in]]` attribute.
;
; The address is `0 + vertex_id * 8` bytes, which is the one
; `constant + stride * index` shape the canonical contract states as
; `FootprintProof::Affine` over a draw's own vertex index (axis 0,
; `research/docs/23` §3.3 v86): the translator reports one strided access with
; stride 8, and both ends of the pairing state that same access set. A
; declaration whose proof cannot be evaluated over the draw's counts — a
; first-vertex offset it does not have, an unbounded reach — is a shape the
; contract refuses by name, which is what the gate answers before this rail
; ever submits.
;
; `render_vtx_buffer_positions.air` is `llvm-as` output of this file:
;
;     llvm-as -o tests/fixtures/air/render_vtx_buffer_positions.air \
;             tests/fixtures/air/render_vtx_buffer_positions.ll
;
; The fragment half of the pair is the existing `render_frag.air` (entry
; `fmain`, `float4(0.25, 0.5, 0.75, 1)`), so the frame is a function of the
; bytes this stage reads: a rail that dropped the binding or bound the wrong
; element rasterizes a different triangle.
source_filename = "render_vtx_buffer_positions.air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_buffer_positions_vertex(ptr addrspace(1) %positions, i32 %vertex_id) local_unnamed_addr {
entry:
  %slot = getelementptr <2 x float>, ptr addrspace(1) %positions, i32 %vertex_id
  %xy = load <2 x float>, ptr addrspace(1) %slot, align 8
  %padded = shufflevector <2 x float> %xy, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
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
!13 = !{ptr @reims_buffer_positions_vertex, !14, !15}
!14 = !{!16}
!15 = !{!17, !18}
!16 = !{!"air.position", !"air.arg_type_name", !"float4"}
!17 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float2*", !"air.arg_name", !"positions"}
!18 = !{i32 1, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vertex_id"}
