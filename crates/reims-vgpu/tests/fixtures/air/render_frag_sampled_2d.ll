; Owned synthetic fixture for the reims render rail's sampled-texture
; increment (R10, `research/docs/23` §101): one fragment entry with a single
; `[[texture(0)]]` argument, sampled at one fixed point through the AIR
; `constexpr sampler` the module carries.
;
; The sample lands on one fixed texel centre: `(0.8125, 0.875)` is the centre
; of texel `(6, 3)` of an 8x4 surface, the extent the tests draw at. The
; canonical render sampler executes a texture of the render area's own extent
; (`render_texture_extent_unsupported`), so a fragment that samples one texel
; centre of that surface reads exactly the texel the pass bound there — which
; is what makes "the draw samples the texture the pass binds" a byte reading
; rather than a claim about plumbing.
;
; The sampler state word is the compute fixture's own
; (`sample_texture_2d_nearest_clamp.ll`): nearest filtering with clamped
; addressing, the one state the canonical render sampler's reviewed family
; carries. Both files are kept in step by re-assembling this text, never by
; editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_sampled_2d.air \
;           tests/fixtures/air/render_frag_sampled_2d.ll
;
; The vertex half of the pair is the existing `reims_indexed_tri.air` (entry
; `reims_indexed_vertex`), whose stage-in position is the fragment's only
; interface — the fragment takes no varying, which is why the two compose.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_sampled_2d.air"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define <4 x float> @reims_sampled_frag(ptr addrspace(1) readonly captures(none) %texture) local_unnamed_addr {
entry:
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 8.125000e-01, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value = extractvalue { <4 x float>, i8 } %sample, 0
  ret <4 x float> %value
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @reims_sampled_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"texture"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
