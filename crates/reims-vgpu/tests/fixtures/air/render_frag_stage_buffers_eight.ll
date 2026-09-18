; AIR fixture: a fragment stage that declares eight `[[buffer(k)]]` arguments,
; every one of them read (R9q).
;
; The largest list the canonical contract states (`MAX_RENDER_STAGE_BUFFERS` =
; eight, E-SB1), and the list whose per-set descriptor count is exactly the
; Vulkan minimum the ceiling is defended by. See
; `render_frag_stage_buffers_five.ll` for what the family is and why every
; argument is summed into the attachment.
;
;   llvm-as render_frag_stage_buffers_eight.ll -o render_frag_stage_buffers_eight.air
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_stage_buffers_eight_frag(ptr addrspace(1) %b0, ptr addrspace(1) %b1, ptr addrspace(1) %b2, ptr addrspace(1) %b3, ptr addrspace(1) %b4, ptr addrspace(1) %b5, ptr addrspace(1) %b6, ptr addrspace(1) %b7) {
entry:
  %v0 = load <4 x float>, ptr addrspace(1) %b0, align 16
  %v1 = load <4 x float>, ptr addrspace(1) %b1, align 16
  %v2 = load <4 x float>, ptr addrspace(1) %b2, align 16
  %v3 = load <4 x float>, ptr addrspace(1) %b3, align 16
  %v4 = load <4 x float>, ptr addrspace(1) %b4, align 16
  %v5 = load <4 x float>, ptr addrspace(1) %b5, align 16
  %v6 = load <4 x float>, ptr addrspace(1) %b6, align 16
  %v7 = load <4 x float>, ptr addrspace(1) %b7, align 16
  %s1 = fadd fast <4 x float> %v0, %v1
  %s2 = fadd fast <4 x float> %s1, %v2
  %s3 = fadd fast <4 x float> %s2, %v3
  %s4 = fadd fast <4 x float> %s3, %v4
  %s5 = fadd fast <4 x float> %s4, %v5
  %s6 = fadd fast <4 x float> %s5, %v6
  %s7 = fadd fast <4 x float> %s6, %v7
  ret <4 x float> %s7
}

!air.fragment = !{!0}
!0 = !{ptr @reims_stage_buffers_eight_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !6, !7, !8, !9, !10, !11}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b0"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b1"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b2"}
!7 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b3"}
!8 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b4"}
!9 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b5"}
!10 = !{i32 6, !"air.buffer", !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b6"}
!11 = !{i32 7, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b7"}
