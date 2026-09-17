; AIR fixture: a fragment stage that *declares* a `[[buffer(0)]]` argument and
; never reads it (R9b).
;
; The `Unused` arm of the class gate's split, and the reason that arm has to
; exist: a stage whose entry point never dereferences its buffer still carries
; the declaration, and the canonical provider's translated registration refuses
; the *stage* on the declaration's presence
; (`render_stage_unsupported_interface`, field `bindings`) — the admission is
; not cheaper for a binding nothing reads. The engine's own bind census already
; counts this population (`runtime::bind_phase`'s `access_unused`, served the
; neutral page rather than the guest's bytes), so the seam has to name it
; instead of folding it into the read arm.
;
; Metal's own reflection behaves the same way: `metal2vulkan`'s
; `refine_buffer_access_from_entry` classifies a parameter absent from the
; entry's body `Unused`, which is exactly what this module is.
;
;   llvm-as render_frag_buffer_unused.ll -o render_frag_buffer_unused.air
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_unused_buffer_frag(ptr addrspace(1) %tint) {
entry:
  %r = insertelement <4 x float> undef, float 0.250980406999588, i32 0
  %g = insertelement <4 x float> %r, float 0.501960813999176, i32 1
  %b = insertelement <4 x float> %g, float 0.7529411911964417, i32 2
  %a = insertelement <4 x float> %b, float 1.000000e+00, i32 3
  ret <4 x float> %a
}

!air.fragment = !{!0}
!0 = !{ptr @reims_unused_buffer_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"tint"}
