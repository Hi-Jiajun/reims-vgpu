; Owned synthetic fixture for the reims render rail's sampled-sampler-reuse
; increment (R50, 2026-09-20; census v48's fifth door): one fragment entry that
; carries **two** AIR `constexpr samplers` while **four** sampled textures read
; through them — `[[texture(0)]]`, `[[texture(1)]]` and `[[texture(3)]]` all
; read through the first state and `[[texture(2)]]` through the second.
;
; The shape is the census's own. v48's residual bucket is 26 records × 2 of one
; LPF-family fragment stage (`pipe=58`, `fw=109140`, `fdecl=25`) whose thirteen
; sampled textures read through four sampler descriptors: two runtime
; `[[sampler(n)]]` arguments and the module's own two AIR states, with the state
; reused across the textures past the first. The refusal the census counted is
; the fourth `[[texture(11)]]`: a walk that pairs one AIR static sampler with one
; sampled texture *by position* runs out of states there, and answers
; `RenderSamplerRefusal::SampleSite` for a texture whose own sample site names
; one of them.
;
; The two states are the corpus's own words, and they are the pair the census
; read (`smpl=4[s832:gNNnee,s833:gNNnee,s834:cLLnee,s835:cNNnrr]`):
; `34901797601018002` is Nearest + Repeat and `34901797601020489` the Linear +
; ClampToEdge state of the canonical rail's `sample_texture_2d_linear_clamp.ll`.
; Every texture is sampled at `u = 1.375`, one half column past the surface's
; right edge, which is exactly where those two states part:
;
;   red   = repeat_texture.sample(repeat,   (1.375, 0.875)).x -> repeat: texel (3, 3)
;   green = reuse_texture.sample(repeat,    (1.375, 0.875)).x -> the same state, reused
;   blue  = linear_texture.sample(linear,   (1.375, 0.875)).x -> clamp: texel (7, 3)
;   alpha = reuse_texture_b.sample(repeat,  (1.375, 0.875)).x -> the same state again
;
; Against the rail's own 8x4 test texture (`sampled_texels`, whose red channel is
; `16 * x` per column) the fragment lands `30 30 70 30`: the green channel is the
; reading a *positional* pairing would hand to the second AIR state instead
; (linear + clamp would land texel 7's `70`), blue is the reading a positional
; pairing would hand to the first, and a walk that ran out of states at the
; third texture would refuse the stage before the frame existed at all.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_sampler_reuse.air \
;           tests/fixtures/air/render_frag_sampler_reuse.ll
;
; The vertex half of the pair is the existing `reims_indexed_tri.air` (entry
; `reims_indexed_vertex`); the fragment takes no varying, which is why the two
; compose.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_sampler_reuse.air"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601018002, i64 0], align 8
@__air_sampler_state.1 = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0], align 8

define <4 x float> @reims_sampler_reuse_frag(ptr addrspace(1) readonly captures(none) %repeat_texture, ptr addrspace(1) readonly captures(none) %reuse_texture, ptr addrspace(1) readonly captures(none) %linear_texture, ptr addrspace(1) readonly captures(none) %reuse_texture_b) local_unnamed_addr {
entry:
  %repeat_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %repeat_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %repeat_value = extractvalue { <4 x float>, i8 } %repeat_pair, 0
  %repeat = extractelement <4 x float> %repeat_value, i64 0
  %reuse_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %reuse_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %reuse_value = extractvalue { <4 x float>, i8 } %reuse_pair, 0
  %reuse = extractelement <4 x float> %reuse_value, i64 0
  %linear_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %linear_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state.1, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %linear_value = extractvalue { <4 x float>, i8 } %linear_pair, 0
  %linear = extractelement <4 x float> %linear_value, i64 0
  %reuse_b_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %reuse_texture_b, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %reuse_b_value = extractvalue { <4 x float>, i8 } %reuse_b_pair, 0
  %reuse_b = extractelement <4 x float> %reuse_b_value, i64 0
  %with_red = insertelement <4 x float> undef, float %repeat, i32 0
  %with_green = insertelement <4 x float> %with_red, float %reuse, i32 1
  %with_blue = insertelement <4 x float> %with_green, float %linear, i32 2
  %with_alpha = insertelement <4 x float> %with_blue, float %reuse_b, i32 3
  ret <4 x float> %with_alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!6, !8}
!0 = !{ptr @reims_sampler_reuse_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !9, !10}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"repeat_texture"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"reuse_texture"}
!9 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"linear_texture"}
!10 = !{i32 3, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"reuse_texture_b"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!8 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.1}
