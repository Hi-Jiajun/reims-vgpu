; Owned synthetic fixture for the reims render rail's widened runtime-sampler
; state family (R21, `research/docs/26` §44): the R12 fixture's module with the
; one fixed sample point on the *negative* side of the surface, where the two
; mirroring address modes part from the two they are next to.
;
; The module carries no `@__air_sampler_state` and no `air.sampler_states` root
; at all: Metal binds the `MTLSamplerState` when the draw is encoded, so the
; state is a *request* fact and the pass states it.
;
; The sample point is `(-0.375, 0.875)` of an 8x4 surface: outside the texture
; in u, the centre of row 3 in v. The two halves of the address-mode family
; that only move a *negative* coordinate land different texels, so the
; attachment's bytes separate the modes that the positive-side fixture's
; coordinate cannot:
;
;   nearest + clamp-to-edge:       u clamps to 0      -> texel (0, 3)
;   linear  + clamp-to-edge:       the same clamp     -> texel (0, 3)
;   nearest + mirror-clamp-to-edge: |u| = 0.375       -> texel (3, 3)
;   linear  + mirror-clamp-to-edge: the same mirror   -> texels (2, 3) and
;                                                        (3, 3) blended
;   nearest + mirror-repeat:       the same mirror    -> texel (3, 3)
;   linear  + mirror-repeat:       the same mirror    -> the same half blend
;   nearest + repeat:              u wraps to 0.625   -> texel (5, 3)
;   linear  + repeat:              the same wrap      -> texels (4, 3) and
;                                                        (5, 3) blended
;
; Read beside the positive-side fixture this pins the two mirroring modes'
; whole rule: mirror-clamp-to-edge *is* mirror-repeat inside `[-1, 1]` and
; clamps (not mirrors) outside it, which is the one thing a single coordinate
; cannot state.
;
; The tests' texture carries multiples of sixteen in every channel, so that
; half step — the widest a blend gets — lands an exact eight-bit value on any
; driver that filters linearly at all.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_runtime_sampler_mirror.air \
;           tests/fixtures/air/render_frag_runtime_sampler_mirror.ll
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_runtime_sampler_mirror.air"

define <4 x float> @reims_runtime_sampled_mirror_frag(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) %sampler) local_unnamed_addr {
entry:
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) %sampler, <2 x float> <float -3.750000e-01, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value = extractvalue { <4 x float>, i8 } %sample, 0
  ret <4 x float> %value
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!0 = !{ptr @reims_runtime_sampled_mirror_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"texture"}
!5 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
