#!/usr/bin/env bash
# Drive the guest with a load whose weight is a dial, served from the host.
# Fourth probe alongside `maps.sh`, `hammer.sh` and `sustained-animation-probe`;
# same (outdir, seconds) interface, so it drops into `ab.sh` as `AB_PROBE`.
#
#   gpu-load-probe.sh <outdir> <seconds>
#
# Why it exists. The sustained-animation probe has **saturated** on macos-13:
# the guest produces ~26 800 draws and ~76 presented frames a second whatever
# the device does, the drain worker sits at duty 0.83, and a measured
# 3.9 us/chain CPU saving moved `draws_s` by less than 0.1 % because the worker
# spent what it saved waiting for guest work. A probe the device cannot fall
# behind cannot rank a device change, in either direction — it reports "no
# effect" for a real win and for a real regression alike.
#
# So this probe makes each guest frame heavier instead of asking for more
# frames, which is also the only thing a page *can* do: the guest's frame rate
# is its own display cadence and no content raises it.
#
# The load is three independent dials, set through `GPU_LOAD_ARGS`, because they
# land on three different rails:
#
#   layers=N   composited surfaces  -> the writeback / store rail
#   verts=K    per-frame WebGL buffer upload -> the **guest buffer gather** rail
#   tex=W      per-frame 2D canvas repaint   -> the sampled-image rail
#
#   GPU_LOAD_ARGS='layers=16&boxes=8&verts=90000&tex=1024' \
#     kb/harness/ab.sh /tmp/out on macos-13 25
#
# Name the dials in any result. Two boots at different `GPU_LOAD_ARGS` are two
# workloads, not two readings, exactly as two rails are two guest drivers.
#
# The page is served by the host over QEMU's user-net gateway (10.0.2.2), not
# fetched from the internet: a probe whose workload can change under it cannot
# be A/B'd, and the rails have no reason to have working DNS.
set -u
OUT="${1:?outdir}"; SECS="${2:-40}"
REPO=/home/aneesiqbal/Projects/steelbrain/reims-vgpu
export QMP_SOCK="${QMP_SOCK:-$REPO/vm/disks/run/qmp.sock}"
Q="$REPO/scripts/qmp/qmp.py"
FAILLOG=/tmp/reims-vgpu-fail.log
# Fixed port: the guest is told the URL once and a random port would have to be
# plumbed through `open`. A different default from the sustained probe's 8123,
# so a stray server left behind by one cannot serve the other's page.
PORT="${GPU_LOAD_PORT:-8124}"
GATEWAY=10.0.2.2
ARGS="${GPU_LOAD_ARGS:-layers=8&boxes=6}"
mkdir -p "$OUT"

# Serve from the page's own directory so nothing else in the repo is exposed.
python3 -m http.server "$PORT" --directory "$REPO/scripts/gpu-load-probe" \
  >"$OUT/httpd.log" 2>&1 &
HTTPD=$!
trap 'kill $HTTPD 2>/dev/null' EXIT
sleep 1
kill -0 $HTTPD 2>/dev/null || { echo "http server failed to start:"; cat "$OUT/httpd.log"; exit 2; }

URL="http://$GATEWAY:$PORT/load.html?$ARGS"
echo "serving $URL (pid $HTTPD) secs=$SECS"
# The dials are the workload, so they go in the output directory rather than
# only into this script's stdout — a result read later has to be able to say
# which load produced it.
printf '%s\n' "$URL" >"$OUT/load-url.txt"

# `open -a` at setup and never at drive time, so a rail with flaky ssh still
# produces a measurable window. Safari because it is on every rail from macos-11
# up and its compositor path is the one the window server actually serves.
timeout 60 ssh -o BatchMode=yes macos-vm "open -a Safari '$URL'" 2>/dev/null \
  || { echo "could not open Safari on the guest"; exit 3; }

# Safari has to fetch, lay out, compile the WebGL program and get a first frame
# composited before the measurement means anything. Longer than the sustained
# probe's 12 s because a shader compile is on this page's warm-up path.
sleep 16

read -r W H < <("$Q" size) || { echo "no display size"; exit 2; }
echo "display ${W}x${H}"
# Full-screen it, so the load drives the whole scanout rather than a window
# inset in a static desktop. ctrl+cmd+f is Safari's binding and needs no chrome
# geometry to be guessed — `meta_l` is Command in QEMU's qcode names, which is
# not free choice. Then park the pointer in a corner: a cursor resting over a
# moving layer adds hover work that is not the workload under test.
"$Q" key ctrl+meta_l+f >/dev/null 2>&1
sleep 3
"$Q" move 4 4 >/dev/null 2>&1
sleep 2

# Mark the log so only the driven window is measured.
OFFSET=$(stat -c %s "$FAILLOG")
# Nothing to drive: the page animates itself. That is the point — no host input
# lands during the window, so nothing in the measurement is the probe's own cost.
sleep "$SECS"

tail -c "+$(( OFFSET + 1 ))" "$FAILLOG" >"$OUT/window.log"
# Reading a captured window is not specific to this probe, so the analysis is
# not carried here. `ANIM_ANALYZE` names it — the same variable the sustained
# probe uses, because the window format is the device's and not the probe's.
ANALYZE="${ANIM_ANALYZE:-}"
[ -n "$ANALYZE" ] && [ -f "$ANALYZE" ] && python3 "$ANALYZE" "$OUT/window.log"
exit 0
