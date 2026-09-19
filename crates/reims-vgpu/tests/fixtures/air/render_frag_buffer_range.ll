; AIR fixture: a fragment stage whose `[[buffer(0)]]` argument is read at an
; *index the translation cannot express* (R48, E-SB3).
;
; The twin of `render_frag_buffer.ll` one arm over: the same single
; `Location 0` store of a four-component colour, with the red component loaded
; from the stage's own Metal buffer — but from the entry the buffer's *own
; first word* names rather than from its head:
;
;   %word  = load i32, ptr addrspace(1) %tint           ; the entry to read
;   %index = add i32 (and i32 %word, 3), 1              ; past the entry word
;   %slot  = getelementptr i32, ptr addrspace(1) %tint, i32 %index
;   %red   = bitcast i32 %bits to float
;
; The address the second load reaches is therefore a function of a value read
; out of memory, which is exactly what the translator's footprint schema cannot
; state: the reflection answers `has_unbounded_access` for this binding and asks
; every consumer to keep the caller's whole window available
; (`metal2vulkan::reflect::BufferFootprint`). That is the declaration this
; rail's class gate states as `StageBufferFootprint::Unstated`, and the shape
; E-SB3 executes as the contract's `FootprintProof::BindingRange`.
;
; What makes the frame falsifiable: the bind carries the index in its first
; word and the colour words behind it, so changing either word changes the
; texel the fragment lands — a bind whose table moved is a different frame.
;
; Assembled exactly as the rest of this directory is, from this source:
;
;   llvm-as render_frag_buffer_range.ll -o render_frag_buffer_range.air
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_buffer_range_frag(ptr addrspace(1) %tint) {
entry:
  %word = load i32, ptr addrspace(1) %tint, align 4
  %masked = and i32 %word, 3
  %index = add i32 %masked, 1
  %slot = getelementptr i32, ptr addrspace(1) %tint, i32 %index
  %bits = load i32, ptr addrspace(1) %slot, align 4
  %red = bitcast i32 %bits to float
  %r = insertelement <4 x float> undef, float %red, i32 0
  %g = insertelement <4 x float> %r, float 0.000000e+00, i32 1
  %b = insertelement <4 x float> %g, float 0.000000e+00, i32 2
  %a = insertelement <4 x float> %b, float 1.000000e+00, i32 3
  ret <4 x float> %a
}

!air.fragment = !{!0}
!0 = !{ptr @reims_buffer_range_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"tint"}
