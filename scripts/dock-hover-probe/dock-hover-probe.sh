#!/usr/bin/env bash
# dock-hover-probe — photograph the guest's dock hover effect, and record what
# the device did while it was on screen.
#
# The bug this exists for: on macos-26 a dock hover tooltip renders as a flat
# untextured polygon with no icon highlight, and the dock's own background comes
# out mottled rather than blurred. That was reported from a hand-taken
# screenshot, which is not a regression gate — `AGENTS.md` asks for a log- or
# test-level proxy for the bug class before a visual fix lands, and there was no
# way to ask for the effect on demand at all.
#
# So this does two things a screenshot cannot. It *asks for* the hover
# reproducibly, by gliding the pointer onto a dock slot and resting it there
# (see `hover.c` — a hover is an arrival followed by rest, not a coordinate).
# And it brackets each hover with the device's own fail log, so the lines the
# device emitted while that effect was being composited are separated from the
# whole boot's noise.
#
# It does not judge the picture. It produces a screenshot per slot and a fail-log
# slice per slot; comparing those against a known-good rail is the reading.
#
# Usage:
#   scripts/dock-hover-probe/dock-hover-probe.sh [--slots N] [--rest SECONDS]
#                                                [--keep DIR] [--guest HOST]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/guest-display.sh
. "$SCRIPT_DIR/../lib/guest-display.sh"

GUEST=macos-vm
SLOTS=5
REST=2.5
KEEP=""
FAILLOG=/tmp/reims-vgpu-fail.log

while [ $# -gt 0 ]; do
  case "$1" in
    --slots) SLOTS="$2"; shift 2 ;;
    --rest) REST="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --guest) GUEST="$2"; shift 2 ;;
    *) echo "dock-hover-probe: unknown option $1" >&2; exit 2 ;;
  esac
done

say() { echo "dock-hover-probe: $*"; }

# The screenshot is taken two seconds into each hover's rest, so a shorter rest
# would photograph a pointer that has already left. Refused rather than clamped:
# a run that silently ignored `--rest` would produce dock shots with no hover in
# them and nothing would say why.
if ! awk -v r="$REST" 'BEGIN { exit !(r > 2.2) }'; then
  say "--rest must be greater than 2.2 s (the shot lands 2 s into the rest)" >&2
  exit 2
fi

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"

ssh -o ConnectTimeout=8 -o BatchMode=yes "$GUEST" true 2>/dev/null || {
  say "no guest at $GUEST — run vm/guest-authorize.sh first" >&2; exit 2; }
[ -f "$FAILLOG" ] || { say "no fail log at $FAILLOG — is a boot running?" >&2; exit 2; }

read -r SW SH < <(guest_display_size "$GUEST") || exit 2
say "guest desktop ${SW}x${SH}"

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

scp -q "$SCRIPT_DIR/hover.c" "$GUEST:/tmp/reims-hover.c" || {
  say "could not copy the hover poster to the guest" >&2; exit 2; }
ssh -o BatchMode=yes "$GUEST" \
  'clang -O2 -o /tmp/reims-hover /tmp/reims-hover.c -framework ApplicationServices -lm' \
  2>"$WORK/build.err" || {
  say "could not build the hover poster on the guest:" >&2
  sed 's/^/  /' "$WORK/build.err" >&2; exit 2; }

SHOT="$REPO_ROOT/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh"

# Slots across the middle half of the screen, which is where a default dock's
# icons fall whatever its item count.
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
  # The hover runs in the background and the screenshot is taken *during* its
  # rest, because the effect exists only while the pointer is resting. Waiting
  # for the poster to return and then shooting photographs a dock that is still
  # hovered but has had its tooltip up for the whole rest — which works, but
  # gives no control over when in the effect's life the shot lands.
  #
  # Bounded host-side: an unattended harness cannot tell a wedged guest command
  # from a wedged boot, and a hover that never returns must not hang a sweep.
  timeout 90 ssh -o BatchMode=yes "$GUEST" \
    "/tmp/reims-hover $x $HOVER_Y $APPROACH_Y $REST" \
    > "$WORK/hover-$i.json" 2>"$WORK/hover-$i.err" &
  HOLD=$!

  # Let the approach finish (~200 ms) and the tooltip timer elapse before
  # shooting. The poster rests for $REST after arriving, so this must land
  # inside that window.
  sleep 2
  "$SHOT" -o "$WORK/dock-slot-$i.png" > "$WORK/shot-$i.log" 2>&1 \
    && captured=$(( captured + 1 )) \
    || say "slot $i at x=$x: host screenshot failed — see $WORK/shot-$i.log"

  wait $HOLD 2>/dev/null
  if [ ! -s "$WORK/hover-$i.json" ]; then
    say "slot $i at x=$x: hover poster produced no record — see hover-$i.err"
  fi

  # The device's own account of that interval, and nothing else's.
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
