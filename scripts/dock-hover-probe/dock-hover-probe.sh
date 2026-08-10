#!/usr/bin/env bash
# dock-hover-probe — photograph the guest's dock hover effect, and record what
# the device did while it was on screen.
#
# The bug this exists for: on macos-26 a dock hover tooltip renders as a flat
# untextured polygon with no icon highlight, and the dock's own background comes
# out mottled rather than blurred. That was reported from a hand-taken
# screenshot, which is not a regression gate — `AGENTS.md` asks for a log- or
# test-level proxy for a bug class before a visual fix lands, and there was no
# way to ask for the effect on demand at all.
#
# # Why this drives the pointer from the host
#
# The first version of this probe compiled a Quartz pointer poster on the guest,
# the way `window-drag-probe`'s `drag.c` does. That cannot work on the rail with
# the bug: **macOS 26 has no command line developer tools**, so `clang` is absent
# and the build step requests an installer dialog instead. The other guest-side
# routes are no better — `screencapture` fails outright there ("could not create
# image from display", so `guest_display_size` cannot answer), and the
# `osascript` desktop-bounds queries need Apple Events consent a fresh ssh
# session does not have.
#
# QMP's `input-send-event` reaches the machine's usb-tablet from outside the
# guest, so it needs no guest tooling, no consent and no permission, and it works
# identically on all six rails. It is the same transport `vibrancy-latch-probe`
# already drives its gestures over. The consequence worth noting is that this
# probe **does not use ssh at all** — a guest whose sshd never came up can still
# be photographed.
#
# A hover is an *arrival followed by rest*, not a coordinate: the window server
# starts its tooltip timer when the pointer stops, so the probe glides in over
# several sub-moves and then stops sending events entirely. Re-asserting the
# position on a timer would restart that timer forever and the tooltip would
# never appear.
#
# It does not judge the picture. It produces a screenshot per slot and a fail-log
# slice per slot; comparing those against a known-good rail is the reading.
#
# Usage:
#   scripts/dock-hover-probe/dock-hover-probe.sh [--slots N] [--rest SECONDS]
#                                                [--keep DIR] [--qmp SOCK]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SLOTS=5
REST=2.5
KEEP=""
FAILLOG="${REIMS_FAIL_LOG:-/tmp/reims-vgpu-fail.log}"
# The x86 boot's stable per-boot symlink. qmp.py defaults to the arm64 path, so
# this must be passed explicitly here — the same override vibrancy-latch-probe
# makes, for the same reason.
QMP_SOCK="${QMP_SOCK:-$REPO_ROOT/vm/disks/run/qmp.sock}"

while [ $# -gt 0 ]; do
  case "$1" in
    --slots) SLOTS="$2"; shift 2 ;;
    --rest) REST="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --qmp) QMP_SOCK="$2"; shift 2 ;;
    -h|--help) sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "dock-hover-probe: unknown option $1" >&2; exit 2 ;;
  esac
done

say() { echo "dock-hover-probe: $*"; }

# The screenshot is taken partway into each hover's rest, so a shorter rest would
# photograph a pointer that has already left. Refused rather than clamped: a run
# that silently ignored `--rest` would produce dock shots with no hover in them
# and nothing would say why.
if ! awk -v r="$REST" 'BEGIN { exit !(r > 1.5) }'; then
  say "--rest must be greater than 1.5 s (the shot lands 1.2 s into the rest)" >&2
  exit 2
fi

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"

QMP="$REPO_ROOT/scripts/qmp/qmp.py"
[ -S "$QMP_SOCK" ] || { say "no QMP socket at $QMP_SOCK — is a boot running?" >&2; exit 2; }
[ -f "$FAILLOG" ] || { say "no fail log at $FAILLOG — is a boot running?" >&2; exit 2; }

read -r SW SH < <(QMP_SOCK="$QMP_SOCK" "$QMP" size 2>"$WORK/size.err")
case "${SW:-}" in
  [0-9]*) ;;
  *) say "QMP did not report a display size — see $WORK/size.err" >&2; exit 2 ;;
esac
say "guest display ${SW}x${SH}"

# The dock's own geometry is not queryable without assistive access, which five
# of six rails do not have. It does not need to be: the probe hovers a band of
# slots across the bottom centre of the screen, which is where a default dock
# sits, and reports the coordinates it used. A rail whose dock is hidden or
# repositioned produces screenshots with no dock in them — visibly a miss rather
# than a silent one.
#
# 44 px above the bottom edge is the centre of a default 64 pt icon row plus its
# margin, in points; these guests are non-Retina so a point is a pixel.
HOVER_Y=$(( SH - 44 ))
APPROACH_Y=$(( SH - 260 ))
[ "$APPROACH_Y" -lt 0 ] && APPROACH_Y=0

SHOT="$REPO_ROOT/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh"

span=$(( SW / 2 ))
left=$(( SW / 4 ))
captured=0
for i in $(seq 1 "$SLOTS"); do
  if [ "$SLOTS" -gt 1 ]; then
    x=$(( left + (span * (i - 1)) / (SLOTS - 1) ))
  else
    x=$(( SW / 2 ))
  fi

  before=$(wc -c < "$FAILLOG")

  # The approach: a handful of sub-moves so the window server sees a pointer
  # entering the target rather than teleporting into it. Then nothing, so the
  # tooltip timer can run.
  for step in 1 2 3 4 5 6; do
    y=$(( APPROACH_Y + (HOVER_Y - APPROACH_Y) * step / 6 ))
    QMP_SOCK="$QMP_SOCK" "$QMP" move "$x" "$y" >/dev/null 2>>"$WORK/move-$i.err"
  done

  sleep 1.2
  "$SHOT" -o "$WORK/dock-slot-$i.png" > "$WORK/shot-$i.log" 2>&1 \
    && captured=$(( captured + 1 )) \
    || say "slot $i at x=$x: host screenshot failed — see $WORK/shot-$i.log"

  # Let the rest finish before cutting the log slice, so the slice covers the
  # whole time the effect was on screen rather than only up to the shot.
  sleep "$(awk -v r="$REST" 'BEGIN { d = r - 1.2; print (d > 0 ? d : 0) }')"

  after=$(wc -c < "$FAILLOG")
  if [ "$after" -gt "$before" ]; then
    tail -c "$(( after - before ))" "$FAILLOG" > "$WORK/faillog-slot-$i.txt"
  else
    : > "$WORK/faillog-slot-$i.txt"
  fi
  say "slot $i at x=$x,y=$HOVER_Y — $(wc -l < "$WORK/faillog-slot-$i.txt") device lines"
done

if [ "$captured" -eq 0 ]; then
  say "no screenshot was captured — this run is not a measurement" >&2
  say "artifacts in $WORK" >&2
  exit 1
fi

# The union of what the device said across every hover, ranked. Fail-channel
# records only: an `OFF ` record carries `reason=` too, for ordering and
# control-flow events that are not losses, so ranking without this filter
# inverts the queue.
cat "$WORK"/faillog-slot-*.txt 2>/dev/null \
  | grep -v '^OFF ' \
  | grep -o 'reason=[a-z_0-9]*' \
  | sort | uniq -c | sort -rn > "$WORK/reasons.txt"

say "captured $captured/$SLOTS screenshots in $WORK"
if [ -s "$WORK/reasons.txt" ]; then
  say "device refusals during the hovers:"
  sed 's/^/  /' "$WORK/reasons.txt"
else
  say "no fail-channel refusal was emitted during any hover"
fi
