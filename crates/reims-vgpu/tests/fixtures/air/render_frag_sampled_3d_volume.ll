; Owned synthetic fixture for the reims render rail's three-dimensional sampled
; texture arm (2026-09-20, the `D3` arm): one fragment entry with a single
; `[[texture(0)]]` argument declared `texture3d<float, sample>`, sampled at four
; texel centres of its own `4 x 4 x 2` volume through the AIR `constexpr
; sampler` the module carries.
;
;   red   = slice 0, (0, 0)   sample at (0.125, 0.125, 0.25)
;   green = slice 1, (2, 0)   sample at (0.625, 0.125, 0.75)
;   blue  = slice 0, (0, 2)   sample at (0.125, 0.625, 0.25)
;   alpha = slice 1, (2, 2)   sample at (0.625, 0.625, 0.75)
;
; The coordinates are texel centres of a `4 x 4 x 2` volume — `(i + 0.5) / 4`
; in x and y and `(k + 0.5) / 2` in z — so the nearest sample of that point is
; the texel itself rather than a blend.
;
; The two *z* coordinates are the arm's own statement. A rail that read the
; volume as a two-dimensional grid of *layers* — depth as an array axis, so the
; third coordinate selected a slice of a `4 x 4 x 2` array rather than the
; volume's own third axis — lands the same texels only if the two happen to
; agree, and a rail that uploaded only the first slice lands the fill value in
; the two lanes this fixture samples out of slice 1. The four positions differ
; in x and y as well, so a transposed or row-shifted read lands another texel's
; number too. The green texel's own value is `2.5`, whose leading byte is `0x00`
; against the frame's `0xff`: a rail that read the texel's first byte as a
; normalised component lands the wrong eight-bit colour.
;
; Every sample takes `.x` because a single-component texture carries exactly
; one channel: Vulkan fills the components an `R32_SFLOAT` view does not have
; (green and blue zero, alpha one), so a fixture reading `.y` would read the
; fill rather than the volume.
;
; The sampler state word is the corpus's own
; (`render_frag_sampled_2d.ll`, `sample_texture_2d_nearest_clamp.ll`): nearest
; filtering with clamped addressing on all three axes, which is one of the
; states the canonical render sampler's reviewed family carries and the state
; the tests state beside the bind. Both files are kept in step by re-assembling
; this text, never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_sampled_3d_volume.air \
;           tests/fixtures/air/render_frag_sampled_3d_volume.ll
;
; The vertex half of the pair is the existing `reims_indexed_tri.air` (entry
; `reims_indexed_vertex`), whose stage-in position is the fragment's only
; interface — the fragment takes no varying, which is why the two compose.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_sampled_3d_volume.air"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define <4 x float> @reims_volume_sample_frag(ptr addrspace(1) readonly captures(none) %texture) local_unnamed_addr {
entry:
  %red_pair = tail call { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 1.250000e-01, float 1.250000e-01, float 2.500000e-01>, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %red_value = extractvalue { <4 x float>, i8 } %red_pair, 0
  %red = extractelement <4 x float> %red_value, i64 0
  %green_pair = tail call { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 6.250000e-01, float 1.250000e-01, float 7.500000e-01>, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %green_value = extractvalue { <4 x float>, i8 } %green_pair, 0
  %green = extractelement <4 x float> %green_value, i64 0
  %blue_pair = tail call { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 1.250000e-01, float 6.250000e-01, float 2.500000e-01>, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %blue_value = extractvalue { <4 x float>, i8 } %blue_pair, 0
  %blue = extractelement <4 x float> %blue_value, i64 0
  %alpha_pair = tail call { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 6.250000e-01, float 6.250000e-01, float 7.500000e-01>, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %alpha_value = extractvalue { <4 x float>, i8 } %alpha_pair, 0
  %alpha = extractelement <4 x float> %alpha_value, i64 0
  %first = insertelement <4 x float> undef, float %red, i32 0
  %second = insertelement <4 x float> %first, float %green, i32 1
  %third = insertelement <4 x float> %second, float %blue, i32 2
  %fourth = insertelement <4 x float> %third, float %alpha, i32 3
  ret <4 x float> %fourth
}

declare { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <3 x float>, i1, <3 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @reims_volume_sample_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture3d<float, sample>", !"air.arg_name", !"texture"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
