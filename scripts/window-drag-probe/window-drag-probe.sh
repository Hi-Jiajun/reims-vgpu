#!/usr/bin/env bash
# window-drag-probe.sh — what does this device do while a window is being dragged?
#
# Goal 6 is "window-dragging performance of Safari should be a stable 120 fps".
# Nothing measured it. `scripts/browser-probe` measures rAF inside a page, which
# is the wrong instrument twice over: a drag is window-server compositing rather
# than page script, and AGENTS.md records that Safari's rAF here is bimodal at
# ~59 and ~118 with nothing between, so a single rAF figure cannot support a
# claim about a code change in either direction.
#
# The device-side counters do not share that bimodality, so they are the result
# and this script is the harness that makes them mean something:
#
#   host_window_cadence  present_hz, offered_hz  — frames this device put out
#   drain_duty           duty, draw_us, flush_us — what the drain worker spent
#   readback_split       fence_us, bar_us, gpu_us
#
# Two motions, and the difference is not cosmetic. `--motion drag` posts a real
# Quartz pointer stream (`drag.c`), which is what goal 6 names; a pointer held
# across a title bar and a window teleported by the AX API do not take the same
# path through the window server. But `CGEventPost` to the HID tap is silently
# discarded for a process that is not trusted for Accessibility, and that trust
# cannot be arranged on this guest — no passwordless sudo, and SIP's Filesystem
# Protections leave TCC.db unwritable. So `--motion reposition` (the default)
# moves the window through System Events, which *is* trusted. It measures a
# large window moving at ~100 Hz with everything behind it recomposited, and it
# omits whatever the window server does specifically for a drag session.
#
# Either way the motion reports what it actually did, and the harness samples
# the window's real position mid-run and refuses a verdict if it never moved.
# Both guards exist because a stressor that produces nothing reports the idle
# device's counters as this device's ceiling — which is how
# `scripts/web-content-probe` once passed a static page off as a churn test, and
# is exactly what the drag mode does here.
#
# Usage:
#   scripts/window-drag-probe/window-drag-probe.sh [--seconds N] [--hz N]
#     [--app "Safari"] [--motion drag|reposition] [--keep DIR]
#
# Exits 0 when the motion ran and the counters were collected, 2 on a setup
# failure — which includes the window never moving. It does not fail on a slow
# device: this is an instrument, and the number it prints is the result.
set -euo pipefail
export LC_ALL=C

SECONDS_RUN=15
HZ=120
APP="Safari"
# `drag` posts a real pointer drag and is what goal 6 is about; it needs the
# posting process to be trusted for Accessibility, which cannot be arranged on
# this guest (no passwordless sudo, SIP Filesystem Protections on, so TCC.db is
# unwritable). Its events are then silently discarded and the window never
# moves — which the harness detects rather than reporting the idle device.
# `reposition` moves the window through System Events, which is trusted. It is a
# weaker stressor and the default only because it is the one that runs.
MOTION=reposition
KEEP=""
GUEST="${GUEST:-macos-vm}"
FAILLOG="${REIMS_FAIL_LOG:-/tmp/reims-vgpu-fail.log}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --seconds) SECONDS_RUN="$2"; shift 2 ;;
    --hz) HZ="$2"; shift 2 ;;
    --app) APP="$2"; shift 2 ;;
    --motion) MOTION="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    -h|--help) sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "window-drag-probe: unknown argument $1" >&2; exit 2 ;;
  esac
done

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"
[ -n "$KEEP" ] || trap 'rm -rf "$WORK"' EXIT
say() { echo "window-drag-probe: $*"; }
# shellcheck source=../lib/guest-display.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/guest-display.sh"
osa() { guest_osa "$GUEST" "$1"; }

ssh -o ConnectTimeout=8 -o BatchMode=yes "$GUEST" true 2>/dev/null || {
  say "no guest at $GUEST" >&2; exit 2; }
[ -f "$FAILLOG" ] || { say "no fail log at $FAILLOG — is a boot running?" >&2; exit 2; }

# Built on the guest, not shipped: nothing here commits or copies a binary.
# C rather than the obvious Python because the guest's /usr/bin/python3 (3.9.6,
# Command Line Tools) has no PyObjC, so `import Quartz` fails, while clang and
# the ApplicationServices headers are present.
case "$MOTION" in
  drag)
    scp -q "$SCRIPT_DIR/drag.c" "$GUEST:/tmp/reims-drag.c"
    ssh -o BatchMode=yes "$GUEST" \
      'clang -O2 -o /tmp/reims-drag /tmp/reims-drag.c -framework ApplicationServices -lm' \
      2>"$WORK/build.err" || {
      say "could not build the drag poster on the guest:" >&2
      sed 's/^/  /' "$WORK/build.err" >&2; exit 2; } ;;
  reposition)
    scp -q "$SCRIPT_DIR/reposition.applescript" "$GUEST:/tmp/reims-reposition.applescript" ;;
  *) say "--motion takes drag or reposition" >&2; exit 2 ;;
esac

# A window that fills the screen leaves the compositor almost nothing to
# recomposite behind it, and one that is tiny damages almost nothing. Half the
# screen, placed off-centre so the drag path stays on-screen throughout.
ssh -o BatchMode=yes "$GUEST" "open -a '$APP' about:blank" >/dev/null 2>&1 || true
sleep 5
osa "tell application \"System Events\" to tell process \"$APP\" to set position of window 1 to {320, 180}" >/dev/null || true
osa "tell application \"System Events\" to tell process \"$APP\" to set size of window 1 to {1000, 640}" >/dev/null || true
sleep 2

# Grab the title bar. Read the window's real frame rather than assuming the
# reposition took: if the app refused it, dragging at the assumed point grabs
# the desktop and the run measures nothing.
POS=$(osa "tell application \"System Events\" to tell process \"$APP\" to get position of window 1" || true)
SIZ=$(osa "tell application \"System Events\" to tell process \"$APP\" to get size of window 1" || true)
WX=$(echo "$POS" | awk -F', *' '{print $1}')
WY=$(echo "$POS" | awk -F', *' '{print $2}')
WW=$(echo "$SIZ" | awk -F', *' '{print $1}')
case "${WX:-}${WY:-}${WW:-}" in
  ''|*[!0-9-]*) say "could not read $APP's window frame (pos '$POS' size '$SIZ')" >&2; exit 2 ;;
esac
# Middle of the title bar. 14 px down is inside the bar for every macOS window
# style and clear of the traffic lights, which sit at the left.
GX=$((WX + WW / 2))
GY=$((WY + 14))
if [ "$MOTION" = drag ]; then
  say "motion=drag on $APP window 1 at ($GX,$GY) for ${SECONDS_RUN}s at ${HZ} Hz"
else
  say "motion=reposition on $APP window 1 for ${SECONDS_RUN}s (--hz does not \
apply: System Events sets the rate, and the run reports what it achieved)"
fi

# Mark the fail log so only lines the drag produced are read. Byte offset rather
# than a timestamp: the log's `t=` is device time and this shell's clock is not.
OFF=$(stat -c %s "$FAILLOG")

if [ "$MOTION" = drag ]; then
  ssh -o BatchMode=yes "$GUEST" \
    "/tmp/reims-drag $GX $GY $SECONDS_RUN $HZ 180 90" >"$WORK/drag.json" 2>"$WORK/drag.err" &
else
  # The accessibility route cannot pace itself — each `set position` is a
  # synchronous round trip through System Events — so it runs for the duration
  # and reports the rate it achieved. `--hz` has no meaning here and the harness
  # says so above rather than appearing to honour it.
  ssh -o BatchMode=yes "$GUEST" \
    "python3 -c \"
import subprocess, time, json
t0 = time.time()
r = subprocess.run(['osascript', '/tmp/reims-reposition.applescript',
                    '$APP', '$GX', '$WY', '$SECONDS_RUN', '180', '90'],
                   capture_output=True, text=True)
el = time.time() - t0
n = int((r.stdout or '0').strip() or 0)
print(json.dumps({'posted': n, 'elapsed': round(el, 3),
                  'posted_hz': round(n / el, 1) if el else 0.0,
                  'late': 0, 'worst_late_s': 0.0, 'stderr': r.stderr[-200:]}))
\"" >"$WORK/drag.json" 2>"$WORK/drag.err" &
fi
DRAG_PID=$!

# Posting an event is not moving a window. `CGEventPost` to the HID tap is
# silently dropped for a process that is not trusted for Accessibility, and a
# run where every event went nowhere reports the idle device's counters as the
# device's ceiling — which is exactly the shape of failure that made a static
# page pass as a churn test this morning. So sample the window's real position
# mid-drag and require it to have left where it started.
sleep 3
MID=$(osa "tell application \"System Events\" to tell process \"$APP\" to get position of window 1" || true)
wait "$DRAG_PID" || {
  say "the drag did not run — see $WORK/drag.err:" >&2
  sed 's/^/  /' "$WORK/drag.err" >&2; exit 2; }
DRAG=$(cat "$WORK/drag.json")

tail -c "+$((OFF + 1))" "$FAILLOG" >"$WORK/window.log"
say "drag: $DRAG"
say "window at start ($WX,$WY), mid-drag ($MID)"
if [ "$MID" = "$WX, $WY" ] || [ -z "$MID" ]; then
  say "the window never moved — the events were posted but not delivered, most \
likely because the posting process is not trusted for Accessibility. The counters \
below would be an idle device's, not a dragging one's." >&2
  exit 2
fi

posted_hz=$(echo "$DRAG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["posted_hz"])')
# Short of the ask by more than a fifth and the drag, not the device, is the
# slow thing. Say so rather than reporting the device's rate as its ceiling.
if awk -v p="$posted_hz" -v h="$HZ" 'BEGIN{exit !(p < 0.8 * h)}'; then
  say "the drag was posted at ${posted_hz} Hz against a requested ${HZ} Hz — the \
counters below are bounded by the drag, not by this device" >&2
fi

python3 - "$WORK/window.log" <<'PY'
import re, statistics, sys

text = open(sys.argv[1], errors="replace").read()


def rows(family, keys, leg_of=False):
    """Rows of `family` carrying every key in `keys`.

    A line missing any key is dropped rather than half-read, which is why a
    reader that asks for a field the build no longer emits sees "(no samples)"
    for the whole family. `leg_of` additionally carries the line's `leg=` label
    through, for families emitted once per leg.
    """
    out = []
    for line in text.splitlines():
        if f" {family} " not in f" {line} ":
            continue
        got = {}
        for k in keys:
            m = re.search(rf"\b{k}=([0-9.]+)", line)
            if m:
                got[k] = float(m.group(1))
        if len(got) != len(keys):
            continue
        if leg_of:
            m = re.search(r"\bleg=(\w+)", line)
            if not m:
                continue
            got["leg"] = m.group(1)
        out.append(got)
    return out


def show(label, vals, unit=""):
    if not vals:
        print(f"  {label:<22} (no samples)")
        return
    vals = sorted(vals)
    med = statistics.median(vals)
    print(f"  {label:<22} n={len(vals):<4} min={vals[0]:.2f} med={med:.2f} "
          f"max={vals[-1]:.2f}{unit}")


cad = rows("host_window_cadence", ["present_hz", "offered_hz", "window_ms"])
duty = rows("drain_duty", ["duty", "draw_us", "draws", "flush_us", "flushes",
                           "max_tranche_us", "tranches"])
rb = rows("readback_split", ["fence_us", "fence"])
pub = rows("window_publish", ["fresh", "same_key"])
lock = rows("engine_lock", ["window", "window_blocked", "window_wait_us",
                            "window_wait_max_us", "worker_hold_us",
                            "worker_hold_max_us"])
wl = rows("host_window_loop", ["ticks", "redraws_asked", "draws",
                               "draws_fresh", "draws_stale"])
bp = rows("bind_phase", ["binds", "vertex_us", "fragment_us", "attrs_us"])
cp = rows("chain_phase", ["chains", "binds_us"])
ws = rows("write_split", ["bytes", "land_us", "land"])

print("host_window_cadence — frames this device put out")
show("present_hz", [r["present_hz"] for r in cad], " Hz")
show("offered_hz", [r["offered_hz"] for r in cad], " Hz")
print("drain_duty — what the drain worker spent")
show("duty", [r["duty"] for r in duty])
show("max_tranche_us", [r["max_tranche_us"] for r in duty], " us")
show("draw_us/draw", [r["draw_us"] / r["draws"] for r in duty if r["draws"]], " us")
show("flush_us/flush", [r["flush_us"] / r["flushes"] for r in duty if r["flushes"]], " us")
print("readback_split — the device's GPU cost")
show("fence_us/fence", [r["fence_us"] / r["fence"] for r in rb if r["fence"]], " us")

# What the writeback moved. The share of it the destination already held is no
# longer sampled here: it was measured at 70-99% and that finding is banked in
# the GPU tile-difference pass that acts on it.
print("write_split — bytes moved")
show("MB/s written", [r["bytes"] / 1e6 for r in ws], " MB")

# The per-landing CPU scatter cost. It is
# reported here rather than left to a reader because a CPU tile-skipping rail
# was built against the redundancy fractions and made it *worse* — 744/769 us per landing
# without it against 802 with it, while declining 92 % of the bytes — since a
# full-cache-line store never reads its destination and the compare adds a read
# the eager path never paid. Unlike `fence_us` above, this is a CPU cost and so
# does not depend on the host GPU's power state.
show("land_us/land", [r["land_us"] / r["land"] for r in ws if r["land"]], " us")

# The delivery chain between the composite and the screen, which the three
# families above do not cover. `publish_window_frame` runs once per drain
# tranche, so `fresh + same_key` must equal `tranches` — a pair that does not is
# the first thing to explain. `fresh` is then what this device offered the
# window, and `present_hz` what reached the screen; `engine_lock` says whether
# the difference is the window thread blocked on the mutex the worker holds.
# The draw path is what caps this device once the writeback is accounted for,
# and `binds_us` is its largest column. These three divide it; they are not
# claimed to sum to it.
print("bind_phase — inside chain_phase's binds_us, per draw")
show("vertex_us/bind", [r["vertex_us"] / r["binds"] for r in bp if r["binds"]], " us")
show("fragment_us/bind", [r["fragment_us"] / r["binds"] for r in bp if r["binds"]], " us")
show("attrs_us/bind", [r["attrs_us"] / r["binds"] for r in bp if r["binds"]], " us")
show("binds_us/chain", [r["binds_us"] / r["chains"] for r in cp if r["chains"]], " us")
print("window_publish — frames offered, sampled once per drain tranche")
show("fresh", [r["fresh"] for r in pub], "/s")
show("fresh+same_key", [r["fresh"] + r["same_key"] for r in pub])
show("tranches", [r["tranches"] for r in duty])
print("host_window_loop — how often the window loop looked, and what it found")
show("ticks", [r["ticks"] for r in wl])
show("redraws_asked", [r["redraws_asked"] for r in wl])
show("draws", [r["draws"] for r in wl])
show("draws_fresh", [r["draws_fresh"] for r in wl])
show("draws_stale", [r["draws_stale"] for r in wl])
print("engine_lock — the window thread against the worker's hold")
show("window acquires", [r["window"] for r in lock])
show("window blocked", [r["window_blocked"] for r in lock])
show("window wait_us", [r["window_wait_us"] for r in lock], " us")
show("window wait_max_us", [r["window_wait_max_us"] for r in lock], " us")
show("worker hold_us", [r["worker_hold_us"] for r in lock], " us")
show("worker hold_max_us", [r["worker_hold_max_us"] for r in lock], " us")

# The pacing claim goal 6 is about, stated as the counters see it rather than as
# a mean. A "stable 120" that spends a second at 60 is not stable, and a median
# hides exactly that.
hz = [r["present_hz"] for r in cad]
if hz:
    low = sum(1 for v in hz if v < 100)
    print(f"\nseconds below 100 Hz: {low}/{len(hz)}"
          f"   worst second: {min(hz):.1f} Hz")
PY

[ -n "$KEEP" ] && say "counters kept in $WORK/window.log"
exit 0
