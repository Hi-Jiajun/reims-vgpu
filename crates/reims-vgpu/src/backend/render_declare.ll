; Owned synthetic fixture. Not derived from a third-party metallib.
;
; The reims render rail's *declaring* compute kernel (`research/docs/26` §3,
; R3). The frozen render contract resolves every colour attachment against a
; buffer view the trace declares, and only a compute pass can declare one; so
; the narrow class carries this one-thread kernel ahead of its render pass to
; declare the attachment view the render pass stores into.
;
; It reads one word of that view and nothing else: the read is what fixes the
; binding the declaration is made through, and a kernel whose work is a read
; cannot race the store the render pass then performs over the same bytes
; (core admission refuses a compute pass that *writes* the view an attachment
; stores into — `AttachmentComputeConflict`).
source_filename = "reims_render_declare.metal"

define void @reims_declare(ptr addrspace(1) %attachment) {
  %value = load i32, ptr addrspace(1) %attachment, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @reims_declare, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"attachment"}
