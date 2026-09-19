; AIR fixture: a fragment stage whose own SPIR-V declares the 16-bit shader
; capability pair (2026-09-20, census v48's LPF pipeline).
;
; The body is the census's own narrowing shape, and the reason this fixture
; exists: `fixed_frag_lpf_cpf` narrows a float to `half` and reads the bits back
; through an `i16` shift, which the pinned translator emits as
; `OpCapability Float16` beside `OpCapability Int16` — the two capabilities the
; canonical provider's SPIR-V subset admits exactly on a device created with
; `shaderFloat16` and `shaderInt16` (E's `HalfShaderSupport`, published as the
; capability frame's own `0x00 0x10 <bool>` section). Before that face existed
; the module could not be translated at all, so the class gate had nothing to
; read and a device whose frame does not answer for the pair would have been
; handed a registration the provider refuses by name — which is the census red
; line `draws_skipped_after_engine_refusal`.
;
; The texel is a function of the narrowing path rather than a constant:
; `half(64/255) == 0x3404`, `lshr 1` => `0x1a02` = `6658`, and the module
; compares the `i16` it derives against that value — the red channel is the
; round-tripped half when they agree and `1.0` when they do not, with `alpha`
; the comparison's own select. `green` and `blue` are the round-tripped half of
; `128/255` and `192/255`, so the frame is `40 80 c0 ff` in every texel of an
; eight-bit UNORM attachment — the reviewed solid stage's own texel — and a rail
; that lost the conversion, the shift or the comparison lands `ff 80 c0 00`.
;
; Assembled exactly as the rest of this directory is, from this source:
;
;   llvm-as render_frag_half_truncated.ll -o render_frag_half_truncated.air
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_half_truncated_frag() {
entry:
  %r = insertelement <3 x float> undef, float 0.250980406999588, i32 0
  %g = insertelement <3 x float> %r, float 0.501960813999176, i32 1
  %b = insertelement <3 x float> %g, float 0.7529411911964417, i32 2
  %h = fptrunc <3 x float> %b to <3 x half>
  %bits = bitcast <3 x half> %h to <3 x i16>
  %trunc = lshr <3 x i16> %bits, splat (i16 1)
  %low = extractelement <3 x i16> %trunc, i64 0
  %index = zext i16 %low to i64
  %exact = icmp eq i64 %index, 6658
  %r0 = extractelement <3 x half> %h, i64 0
  %g1 = extractelement <3 x half> %h, i64 1
  %b2 = extractelement <3 x half> %h, i64 2
  %rf = fpext half %r0 to float
  %gf = fpext half %g1 to float
  %bf = fpext half %b2 to float
  %red = select i1 %exact, float %rf, float 1.000000e+00
  %alpha = select i1 %exact, float 1.000000e+00, float 0.000000e+00
  %out0 = insertelement <4 x float> undef, float %red, i32 0
  %out1 = insertelement <4 x float> %out0, float %gf, i32 1
  %out2 = insertelement <4 x float> %out1, float %bf, i32 2
  %out3 = insertelement <4 x float> %out2, float %alpha, i32 3
  ret <4 x float> %out3
}

!air.fragment = !{!0}
!0 = !{ptr @reims_half_truncated_frag, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{}
