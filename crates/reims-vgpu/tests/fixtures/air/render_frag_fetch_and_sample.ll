; Owned synthetic fixture for the reims render rail's texel-fetch increment
; (R15, `research/docs/23` §3.3, v105): one fragment entry with two
; `[[texture(n)]]` arguments — a fetched `access::read` one at Metal index 0 and
; a sampled one at Metal index 1 — beside the runtime `[[sampler(0)]]` argument
; the second texture samples through (R12's family).
;
; The two arguments are the two accesses the canonical render contract states,
; in one module and one draw, which is the shape the guest's own captures have
; (`evidence/r13-reims-sample-site-...`: one fetch-only texture at index 0
; beside runtime-sampled textures after it). The declaration list therefore has
; to carry both arms at once: entry 0 sampler-free, entry 1 paired with the
; sampler argument.
;
; The fetched half reads texel (1, 0).x into red, exactly as
; `render_frag_fetch_texture_2d.ll` does; the sampled half reads the centre of
; texel (3, 2) through the request's own state into green. Against the tests'
; `(16 * i, 64 * j, 8 * (i + j), 255)` texture of 4x4, the frame is
; `10 80 00 ff`, and the two halves move independently when the texels they
; read move.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the bitcode:
;
;   llvm-as -o tests/fixtures/air/render_frag_fetch_and_sample.air \
;           tests/fixtures/air/render_frag_fetch_and_sample.ll
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_frag_fetch_and_sample.air"

define <4 x float> @reims_fetch_and_sample_frag(ptr addrspace(1) readonly captures(none) %fetch_tex, ptr addrspace(1) readonly captures(none) %sample_tex, ptr addrspace(2) readonly captures(none) %sampler) local_unnamed_addr {
entry:
  %fetch_pair = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %fetch_tex, <2 x i32> <i32 1, i32 0>, i32 0, i32 0)
  %fetch_value = extractvalue { <4 x float>, i8 } %fetch_pair, 0
  %red = extractelement <4 x float> %fetch_value, i64 0
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %sample_tex, ptr addrspace(2) readonly captures(none) %sampler, <2 x float> <float 8.750000e-01, float 6.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %sample_value = extractvalue { <4 x float>, i8 } %sample, 0
  %green = extractelement <4 x float> %sample_value, i64 1
  %v0 = insertelement <4 x float> undef, float %red, i32 0
  %v1 = insertelement <4 x float> %v0, float %green, i32 1
  %v2 = insertelement <4 x float> %v1, float 0.000000000000000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i32 3
  ret <4 x float> %v3
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), <2 x i32>, i32, i32) local_unnamed_addr

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!0 = !{ptr @reims_fetch_and_sample_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !6}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"fetch_tex"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"sample_tex"}
!6 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
