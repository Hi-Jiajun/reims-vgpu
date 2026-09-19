; Owned synthetic fixture for the reims render rail's one-dimensional LUT
; increment (2026-09-19, census b10's `texture_shape` bucket): one fragment
; entry with a single `[[texture(0)]]` argument declared as
; `texture1d_array<float, sample>`, sampled at two texel centres of its own
; single row through the AIR `constexpr sampler` the module carries.
;
;   red   = texel 0.x     sample at u = 0.0625
;   green = texel 4.x     sample at u = 0.5625
;   blue  = 0.0, alpha = 1.0 — constants, so the whole attachment texel is
;   defined by the two LUT reads and nothing else.
;
; The coordinates are texel centres of an eight-texel row — (i + 0.5) / 8 — so
; the nearest clamp-to-edge sample is the texel itself rather than a blend. A
; module that read the row as a 2D surface, flipped it, or shifted it by a texel
; lands another texel's number; and because the LUT's texel 4 carries `2.5` —
; whose own encoding is `00 00 20 40` — a rail that read the texel's first byte
; as a normalised component lands `0x00` where the frame must carry the 8-bit
; clamp's `0xff`.
;
; The layer index is the array's own second coordinate and is `0`: the census's
; LUTs are one-slice arrays (`MTLTextureType1DArray` views whose own
; `sampled_ref_backing` reads `16384x1` and `1024x1`), and the canonical
; contract admits exactly that shape.
;
; The sampler state word is the compute fixture's own
; (`sample_texture_2d_nearest_clamp.ll`): nearest filtering with clamped
; addressing, the one state the canonical render sampler's reviewed family
; carries. Both files are kept in step by re-assembling this text, never by
; editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_sampled_1d_array.air \
;           tests/fixtures/air/render_frag_sampled_1d_array.ll
;
; The vertex half of the pair is the existing `reims_indexed_tri.air` (entry
; `reims_indexed_vertex`), whose stage-in position is the fragment's only
; interface — the fragment takes no varying, which is why the two compose.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_sampled_1d_array.air"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define <4 x float> @reims_one_dim_lut_frag(ptr addrspace(1) readonly captures(none) %texture) local_unnamed_addr {
entry:
  %red_pair = tail call { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, float 6.250000e-02, i32 0, i1 false, i32 0, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %red_value = extractvalue { <4 x float>, i8 } %red_pair, 0
  %red = extractelement <4 x float> %red_value, i64 0
  %green_pair = tail call { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, float 5.625000e-01, i32 0, i1 false, i32 0, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %green_value = extractvalue { <4 x float>, i8 } %green_pair, 0
  %green = extractelement <4 x float> %green_value, i64 0
  %first = insertelement <4 x float> undef, float %red, i32 0
  %second = insertelement <4 x float> %first, float %green, i32 1
  %third = insertelement <4 x float> %second, float 0.000000e+00, i32 2
  %fourth = insertelement <4 x float> %third, float 1.000000e+00, i32 3
  ret <4 x float> %fourth
}

declare { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), float, i32, i1, i32, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @reims_one_dim_lut_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture1d_array<float, sample>", !"air.arg_name", !"texture"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
