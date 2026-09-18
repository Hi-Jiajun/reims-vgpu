; AIR fixture: a fragment stage that declares five `[[buffer(k)]]` arguments,
; every one of them read (R9q).
;
; The shape census v29's tail names (`stage_buffer_shape` = 184 rows, every one
; of them the `gt4` route) is a declaration list longer than the canonical
; contract's own ceiling. E-SB1 raised `MAX_RENDER_STAGE_BUFFERS` from four to
; eight (`research/docs/23` §108), so five declarations is the smallest list the
; widened ceiling has to carry and the shape the old one refused by name. Four
; fixtures — five, six, seven and eight — span the two ceilings, and each sums
; *every* argument into the attachment, so a seam that states four declarations
; and drops the rest cannot land the colour this module states.
;
; The payloads are byte/255 constants (20 in every slot, 40 in the one slot the
; mutant draw moves), which keeps the sum — 20*N/255, and 20*(N+1)/255 in the
; mutant — far from a quantisation tie on every driver (`research/docs/23` §3.5).
; The adds carry `fast`, the flag run a Metal module compiled with the default
; math mode carries and the spelling this corpus's own arithmetic fixtures use
; (`reims_indexed_tri_two_stream.ll`): an add that withholds the permission
; demands `FloatControls2` of the device, which is a different question from the
; one this family is about.
;
;   llvm-as render_frag_stage_buffers_five.ll -o render_frag_stage_buffers_five.air
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_stage_buffers_five_frag(ptr addrspace(1) %b0, ptr addrspace(1) %b1, ptr addrspace(1) %b2, ptr addrspace(1) %b3, ptr addrspace(1) %b4) {
entry:
  %v0 = load <4 x float>, ptr addrspace(1) %b0, align 16
  %v1 = load <4 x float>, ptr addrspace(1) %b1, align 16
  %v2 = load <4 x float>, ptr addrspace(1) %b2, align 16
  %v3 = load <4 x float>, ptr addrspace(1) %b3, align 16
  %v4 = load <4 x float>, ptr addrspace(1) %b4, align 16
  %s1 = fadd fast <4 x float> %v0, %v1
  %s2 = fadd fast <4 x float> %s1, %v2
  %s3 = fadd fast <4 x float> %s2, %v3
  %s4 = fadd fast <4 x float> %s3, %v4
  ret <4 x float> %s4
}

!air.fragment = !{!0}
!0 = !{ptr @reims_stage_buffers_five_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !6, !7, !8}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b0"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b1"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b2"}
!7 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b3"}
!8 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b4"}
