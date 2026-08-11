# sustained-animation-probe

Drives the guest with a **sustained, full-rate** animation and captures the
device census for exactly that window.

```sh
scripts/sustained-animation-probe/sustained-animation-probe.sh /tmp/out 40
```

Takes `(outdir, seconds)`, writes `<outdir>/window.log`, and ends by running the
analysis over it — the same interface the other driven-boot probes take, so it
is interchangeable with them in a multi-boot harness.

## Why it exists

An undriven boot measures an idle device; `AGENTS.md` already says so. This
probe exists because a **bursty** driven boot measures the bursts' *gaps*, which
is a different error and reads as a device result rather than as an idle one.

A window-server probe that opens Mission Control and Launchpad spends ~2 s of
wall clock per round waiting for their animations, so whole seconds of it have
literally zero draws. Its `present_hz` median came out at **2.8 Hz** on a device
observed sustaining **78.8 Hz** (peak 92.2) under a frame-rate test page in the
same VM, minutes apart. Nothing in the bursty capture said it was idle: the
counters were self-consistent and the log well-formed.

The consequence is not a scale factor, it is a different ranking. Same guest
rail (macos-13), same build, same quiesced host:

| `chain_phase` share | bursty window-server probe | sustained animation |
|---|---|---|
| `store` | 10.3 % | **34.9 %** |
| `engine` | 49.0 % | 28.2 % |
| `sampled` | 18.5 % | 20.9 % |
| `pipeline` | 6.2 % | 8.5 % |
| per-chain total | 129 µs | 87 µs |
| drain worker duty | 0.00 median, 0.39 peak | **0.22 median, 0.88 peak** |

The last row is the one that decides what is worth fixing. Only the sustained
arm ever makes the drain worker the bottleneck, so it is the only arm on which
a per-draw CPU saving can become frames — which is why several CPU-side wins on
the bursty probe (a bounded pipeline cache, a 39 % cut in submissions, a
twentyfold cut in `stage_us`) each bought real microseconds and **zero** frames.

Run both before any "faster" claim. A change can help one and hurt the other,
and neither is the whole workload.

## The page is served by the host, on purpose

`anim.html` is served over QEMU's user-net gateway (`10.0.2.2:8123`), not fetched
from the internet. A probe whose workload can change under it cannot be A/B'd,
and the guest rails have no reason to have working DNS. Override the port with
`ANIM_PORT`.

Everything the page draws steps per *frame*, never per wall-clock millisecond,
so a slow boot and a fast boot draw identical content per frame number and
differ only in how many frames they complete. It loads both rails that matter:
eight `will-change` layers the window server composites separately, and a canvas
repainted every frame so texture content is uploaded rather than only
re-composited.

## What it does not do

No host input lands inside the measured window — the page animates itself — so
nothing in the capture is the probe's own cost. It also cannot report a verdict
the way a drag probe can: there is no "the window never moved" check, because
there is no host-driven motion to check. Confirm the page is live from the
screenshot the surrounding harness takes, and from `present_hz` being nowhere
near zero.
