; Owned synthetic fixture for the per-stage stage-buffer ceiling (R46, E-SB2):
; the *vertex* half of the pair that declares thirteen `[[buffer(n)]]`
; arguments between its two stages — seven here and six in
; `render_frag_stage_buffers_six.air`.
;
; The stage is the affine shape `render_vtx_buffer_positions.ll` states
; (`positions[vertex_id]`, one `float2` per vertex, stride 8 — the
; `constant + stride * index` access that translates to an `AffineAccess` over
; axis 0) beside six more `[[buffer(n)]]` arguments at indices 1..6, each read
; as one whole `float4` and summed into the clip position. With the fixture's
; own bytes the six offsets are zero, so the triangle is exactly the one
; `b0` carries; they are there because the *count* is the shape this fixture
; exists for, and because a rail that bound the wrong slot, dropped one of the
; seven descriptors, or truncated the list at the old merged ceiling is
; readable two ways — the six slots' bytes move the triangle, and the fragment
; half's sum changes with its own payloads.
;
; The pair's two stages both name indices 0..5, which is the folded shape R31
; kept on the engine and R33 lifted for devices that declare the namespace
; split: the provider translates this stage under
; `metal_api_vulkan::stage_buffer_namespace_layout`, so its whole layout moves
; into the canonical namespace set and the fragment half keeps the
; translator's own set 0 (`research/docs/23` §3.3, E-TX9).
;
; `render_vtx_stage_buffers_seven.air` is `llvm-as` output of this file:
;
;     llvm-as -o tests/fixtures/air/render_vtx_stage_buffers_seven.air \
;             tests/fixtures/air/render_vtx_stage_buffers_seven.ll
source_filename = "render_vtx_stage_buffers_seven.air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_stage_buffers_seven_vertex(ptr addrspace(1) %positions, ptr addrspace(1) %o1, ptr addrspace(1) %o2, ptr addrspace(1) %o3, ptr addrspace(1) %o4, ptr addrspace(1) %o5, ptr addrspace(1) %o6, i32 %vertex_id) local_unnamed_addr {
entry:
  %slot = getelementptr <2 x float>, ptr addrspace(1) %positions, i32 %vertex_id
  %xy = load <2 x float>, ptr addrspace(1) %slot, align 8
  %v1 = load <4 x float>, ptr addrspace(1) %o1, align 16
  %v2 = load <4 x float>, ptr addrspace(1) %o2, align 16
  %v3 = load <4 x float>, ptr addrspace(1) %o3, align 16
  %v4 = load <4 x float>, ptr addrspace(1) %o4, align 16
  %v5 = load <4 x float>, ptr addrspace(1) %o5, align 16
  %v6 = load <4 x float>, ptr addrspace(1) %o6, align 16
  %s1 = fadd fast <4 x float> %v1, %v2
  %s2 = fadd fast <4 x float> %s1, %v3
  %s3 = fadd fast <4 x float> %s2, %v4
  %s4 = fadd fast <4 x float> %s3, %v5
  %offsets = fadd fast <4 x float> %s4, %v6
  %offset_xy = shufflevector <4 x float> %offsets, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %moved = fadd fast <2 x float> %xy, %offset_xy
  %padded = shufflevector <2 x float> %moved, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
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
!13 = !{ptr @reims_stage_buffers_seven_vertex, !14, !15}
!14 = !{!16}
!15 = !{!17, !18, !19, !20, !21, !22, !23, !24}
!16 = !{!"air.position", !"air.arg_type_name", !"float4"}
!17 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float2*", !"air.arg_name", !"positions"}
!18 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"o1"}
!19 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"o2"}
!20 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"o3"}
!21 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"o4"}
!22 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"o5"}
!23 = !{i32 6, !"air.buffer", !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"o6"}
!24 = !{i32 7, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vertex_id"}
