# mtlb-rail: drive a reviewed `.mtlb` fixture through the guest

`guest_mtlb_dispatch.rb` runs *inside the macOS guest*. It loads a
`newLibraryWithData:` library from the reviewed fixture
(`crates/reims-vgpu/tests/fixtures/compute_mul3add1.mtlb`), builds the compute
pipeline for `apv_cs`, and dispatches it twice — `dispatchThreadgroups` 1x1x1
groups of 4x1x1 threads, then `dispatchThreads` 4x1x1 in 4x1x1 threadgroups —
reading the buffer back after each arm.

The guest has no compiler and no working `python3` (both are Xcode CLT stubs on
the frozen `macos.img`), so the driver goes through Ruby 2.6 + Fiddle, with the
two `MTLSize`-by-value dispatch calls built through libffi.

```sh
scp -P 2322 -i ~/.ssh/macos_x86_guest \
  crates/reims-vgpu/tests/fixtures/compute_mul3add1.mtlb \
  hiliang@127.0.0.1:/tmp/gate3_mul3add1.mtlb
scp -P 2322 -i ~/.ssh/macos_x86_guest guest_mtlb_dispatch.rb \
  hiliang@127.0.0.1:/tmp/
ssh -p 2322 -i ~/.ssh/macos_x86_guest hiliang@127.0.0.1 \
  'cd /tmp && ruby guest_mtlb_dispatch.rb /tmp/gate3_mul3add1.mtlb'
```

Expected (both arms green, readback is `1*3+1, 2*3+1, 3*3+1, 4*3+1`):

```text
GATE3 library ok functions=["apv_cs"]
GATE3 pipeline ok entry=apv_cs
GATE3 arm=threadgroups dispatch=dispatchThreadgroups status=4 error=nil readback=[4, 7, 10, 13] correct=true
GATE3 arm=threads      dispatch=dispatchThreads      status=4 error=nil readback=[4, 7, 10, 13] correct=true
```

On the host the same run shows the seam it exists to exercise:

```text
get_compute_info ok task=4 pipe=2 ...
linux_m2v_async queued pipe=2 stage=Kernel tg=[4,1,1] air=2704
linux_m2v_async done   pipe=2 stage=Kernel tg=[4,1,1] ok spv=1760
compute_provider dispatch pipe=2
```

`--payload-check` builds the payload without touching Metal and fails unless the
`dispatch_data`'s size equals the file's. The payload type matters:
`newLibraryWithData:` takes a `dispatch_data_t`, and handing it an `NSData`
segfaults the guest inside Metal. `kb/mtlb-from-data-is-dispatch-data.md` has
the measurements.
