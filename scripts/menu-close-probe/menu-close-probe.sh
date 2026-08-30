#!/usr/bin/env bash
# menu-close-probe.sh [TRIALS] — photograph a context menu's close-in
# animation and say whether the region it vacates is the desktop or is black.
#
# Usage:
#   scripts/menu-close-probe/menu-close-probe.sh [TRIALS]
#     QPID=PID     the guest to drive (default: the one running qemu-system-x86_64)
#     OUT=DIR      where frames land (default: $TMPDIR/menu-close-probe)
#     MENU_X/Y     empty desktop point to right-click, in guest pixels
#     AWAY_X/Y     where to click to dismiss, in guest pixels
#
# # Why this no longer right-clicks the dock
#
# It did, at a coordinate documented as "the Finder icon, leftmost in the dock".
# On this rail's current snapshot that coordinate is not Finder, and the
# snapshot boots with an application window already open and frontmost — so the
# probe's other anchor, "somewhere with no window and no menu", was beside that
# window rather than on bare desktop. Five consecutive boots reported "the menu
# box did not change when the menu opened" and exited saying nothing. Reading
# the captures afterwards: no context menu was ever created. The right-click
# raised an application; the dismiss click sent it behind Finder.
#
# Both anchors were positional guesses about a desktop this probe cannot see in
# advance, and dock layout is exactly the thing that moves between snapshots,
# guest versions and running-application sets.
#
# A **desktop** right-click removes the guess. macOS opens the desktop context
# menu with its corner at the cursor, so the menu's rectangle is derived from
# the coordinate this script itself clicked rather than measured off a dock. The
# interaction is one step removed from the reported one — the report names a
# dock icon's menu — but the defect is a property of the menu layer's close-in
# animation and not of which menu it is, and this is the version of it that can
# actually be aimed. What is lost is stated rather than papered over: if the
# defect turns out to be specific to a dock menu, this probe will read CLEAN and
# be wrong, and the dock variant then has to be aimed by survey rather than by
# assumption.
#
# Exits 0 VERDICT=CLEAN, 1 VERDICT=BLACK_RECTANGLE, 2 when the run did not
# sample the animation and therefore says nothing about the device.
#
# The reported symptom is "right-click something in the dock, then left-click to
# dismiss, and a BLACK RECTANGLE is left behind for the duration of the close
# animation, where the desktop should show through".
#
# # Why the first version of this probe never produced a reading
#
# It captured "immediately and repeatedly" after the dismiss click and reported
# the darkest frame. Three things were wrong with that, and each alone was
# fatal:
#
#   1. The host capture costs **1 427 ms** on this rig (six timed captures,
#      8 562 ms). The animation is ~250 ms. A loop of captures cannot sample it;
#      every frame lands after it is over, which is why every trial reported the
#      identical mean.
#   2. `dark=%[fx:mean(lightness)<0.06]` is not a magick expression —
#      `mean(...)` is not an fx operator — so that column was empty on every
#      frame of every run and the probe's actual verdict never computed.
#   3. The menu rectangle was a hardcoded guess. It does not have to be.
#
# # What replaces it: aim the capture instead of chasing the animation
#
# The capture's ~1 427 ms is nearly all setup; the frame is grabbed **late**,
# about 500 ms in. Measured by starting a capture and opening the menu 150 ms
# later — the menu is in the resulting frame, so the grab is after that — and
# then scanning the dismiss delay:
#
#   dismiss at 0.20 s  mean 124.7   already closed
#   dismiss at 0.42 s  mean 130.3   closing
#   dismiss at 0.46 s  mean 151.0   mid-animation
#   dismiss at 0.50 s  mean 161.4   still open
#   (menu open 161.4, no menu 231.6)
#
# So the animation is reachable: start the capture, wait, *then* dismiss, and
# the grab lands inside the fade. This probe bisects for that delay per boot
# rather than assuming it, because it is a property of the host's compositor
# and not of the guest.
#
# The rectangle is anchored to the icon rather than differenced out of two
# captures; see the note above section 1 for why differencing cannot work on a
# desktop that is repainting between the two frames.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHOT="$REPO/scripts/screenshot/screenshot.sh"
QMP="$REPO/scripts/qmp/qmp.py"

TRIALS="${1:-24}"
OUT="${OUT:-${TMPDIR:-/tmp}/menu-close-probe}"
mkdir -p "$OUT"

# QPID from the caller when there is one, otherwise found by argv[0]. Never by
# `pgrep -f` on a pattern: that also matches any shell whose command line
# mentions qemu, which is how a probe once picked its own caller as the guest.
if [ -z "${QPID:-}" ]; then
  QPID=""
  for p in $(pgrep -f qemu-system-x86_64 2>/dev/null); do
    exe="$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)"
    case "$exe" in */qemu-system-x86_64|qemu-system-x86_64) QPID="$p"; break ;; esac
  done
fi
[ -n "$QPID" ] || { echo "menu-close-probe: no running guest"; exit 2; }

# A right-click on this rail has been observed killing the guest about four
# seconds later, so every stage below has to be able to say "the guest went
# away" rather than scoring the frames it happened to collect. A dead guest
# produces frozen, identical captures, and identical captures score as no new
# black at all — which reads as CLEAN. That is the one wrong answer this probe
# must not give, because it would report the defect repaired by a boot that died
# before it could show it.
SER="$(ls -t "$REPO"/vm/disks/run/serial-*.log 2>/dev/null | head -1)"
SER_MARK=0
[ -n "$SER" ] && SER_MARK="$(wc -c < "$SER")"
guest_gone() {
  kill -0 "$QPID" 2>/dev/null || return 0
  [ -n "$SER" ] || return 1
  tail -c +"$((SER_MARK + 1))" "$SER" 2>/dev/null \
    | grep -q 'Debugger called: <panic>\|panic(cpu '
}

# Where to open the desktop context menu, in GUEST pixels on this rail's
# 1920x1080 desktop. Chosen with room for the menu to open down-and-right
# without meeting a screen edge — macOS flips the menu to the other side of the
# cursor when it would not fit, which would move the rectangle out from under
# the crop derived below.
MENU_X=${MENU_X:-1450}; MENU_Y=${MENU_Y:-250}
# Somewhere with no window and no menu, to dismiss into. Far from the menu, so
# the dismiss click cannot land inside it and choose an item.
AWAY_X=${AWAY_X:-1750}; AWAY_Y=${AWAY_Y:-900}

shot() { "$SHOT" -o "$1" >/dev/null 2>&1; }
open_menu()    { python3 "$QMP" rclick "$MENU_X" "$MENU_Y" >/dev/null 2>&1; }
dismiss_menu() { python3 "$QMP" click "$AWAY_X" "$AWAY_Y" >/dev/null 2>&1; }
# Mean of a crop, x255.
crop_mean() { magick "$1" -crop "$2" +repage -format '%[fx:mean*255]' info: 2>/dev/null; }

# The scoring metric, as two functions the self-test and the real measurement
# both call.
#
# They are functions rather than two spellings of the same magick pipeline
# because the two spellings are exactly what went wrong: the mask was built with
# one polarity and scored as if it had the other, and nothing could catch that
# while the self-test built its own mask. Sharing the derivation means a
# self-test that passes is a statement about the pipeline the verdict is
# computed with, not about a copy of it.

# White where the reference frame is **not** near-black: the region a later
# frame could newly darken. CROP may be empty for a whole image.
base_mask() {
  local src="$1" crop="$2" out="$3"
  if [ -n "$crop" ]; then
    magick "$src" -crop "$crop" +repage -threshold 6% "$out" 2>/dev/null
  else
    magick "$src" -threshold 6% "$out" 2>/dev/null
  fi
}

# Fraction of the box that is near-black in this frame and was not near-black in
# the reference. `Darken` is `min`, so intersecting "dark now" with "was not dark
# before" is the difference this probe is named for.
new_black_frac() {
  local frame="$1" crop="$2" mask="$3"
  if [ -n "$crop" ]; then
    magick "$frame" -crop "$crop" +repage -threshold 6% -negate \
      "$mask" -compose Darken -composite -format '%[fx:mean]' info: 2>/dev/null
  else
    magick "$frame" -threshold 6% -negate \
      "$mask" -compose Darken -composite -format '%[fx:mean]' info: 2>/dev/null
  fi
}

echo "menu-close-probe: qemu=$QPID menu=$MENU_X,$MENU_Y away=$AWAY_X,$AWAY_Y trials=$TRIALS"

# ---- 0a. Prove the scoring metric can see the defect, before trusting a verdict.
#
# This probe has now shipped two scoring metrics that could not measure
# anything. The first used `%[fx:mean(lightness)<0.06]`, which is not a magick
# expression, so the column was empty on every frame. The second composited
# against a mask of the wrong polarity and returned the wallpaper's own dark
# fraction for every trial of every boot -- 0.0829293 on four boots across two
# binaries, candidate and control alike. Both failed *silently*, and both
# reported CLEAN, which is the defect's own answer.
#
# A verdict of CLEAN is only worth reading from an instrument that would have
# said BLACK_RECTANGLE if the rectangle were there. So the metric is run
# against a synthetic frame that is half black, on every invocation, and the run
# is abandoned if it does not measure what it was built to measure. This costs
# two 100x100 images and no guest interaction.
self_test() {
  local d="$OUT/selftest"
  mkdir -p "$d"
  magick -size 100x100 xc:'rgb(150,150,150)' "$d/base.png" 2>/dev/null || return 1
  magick "$d/base.png" -fill black -draw 'rectangle 0,0 49,99' "$d/half.png" 2>/dev/null || return 1
  base_mask "$d/base.png" "" "$d/mask.png" || return 1
  local got
  got="$(new_black_frac "$d/half.png" "" "$d/mask.png")"
  echo "menu-close-probe: self-test half-black frame scores ${got:-<none>} (want ~0.50)"
  awk -v g="${got:-0}" 'BEGIN{ exit !(g > 0.45 && g < 0.55) }'
}
if ! self_test; then
  echo "menu-close-probe: VERDICT=METRIC_BLIND — the scoring metric does not \
respond to a synthetic black rectangle, so no verdict from it means anything"
  exit 2
fi

# ---- 0. The two settled states this run is scored against.
dismiss_menu; sleep 1.5
shot "$OUT/closed.png"
open_menu; sleep 1.2
shot "$OUT/open.png"
dismiss_menu; sleep 1.2
[ -s "$OUT/closed.png" ] && [ -s "$OUT/open.png" ] || {
  echo "menu-close-probe: could not capture the host window"; exit 2; }

# ---- 1. The menu's rectangle, derived from the click that opens it.
#
# Not by differencing two captures. That was tried and it does not work here:
# the two are 1.5 s apart on a live desktop that is repainting damage rects the
# whole time, so the changed-pixel bounding box came back 1279x704 — the whole
# screen — and on a boot with the blank-field defect the field itself is
# churning between the frames.
#
# The desktop context menu opens with its top-left corner at the cursor, so the
# rectangle follows from `MENU_X,MENU_Y` by construction rather than by
# measurement. The crop is inset from that corner and kept well inside the
# smallest menu this desktop produces, because what is being scored is the
# region the animation vacates — a crop that overhangs the menu scores desktop
# that never had a menu over it, in both directions.
#
# Expressed against the capture's own dimensions so a different capture cap does
# not silently move it.
read -r CAPW CAPH <<<"$(magick identify -format '%w %h' "$OUT/closed.png")"
CROP="$(awk -v mx="$MENU_X" -v my="$MENU_Y" -v cw="$CAPW" -v ch="$CAPH" 'BEGIN{
  x = int((mx + 15) * cw / 1920); y = int((my + 15) * ch / 1080);
  w = int(150 * cw / 1920); h = int(150 * ch / 1080);
  printf "%dx%d+%d+%d", w, h, x, y }')"
BASE="$(crop_mean "$OUT/closed.png" "$CROP")"
OPEN="$(crop_mean "$OUT/open.png" "$CROP")"
echo "menu-close-probe: menu box=$CROP  no_menu_mean=$BASE  menu_open_mean=$OPEN"
# Absolute, not signed. The menu is a light material, so it reads *darker* than
# the white field of a blank-desktop boot and *lighter* than the dark foliage of
# a painted one. Both were measured on this rail — 231.6 against 157.5 on one
# boot and 96.1 against 147.5 on another — and a signed test passes on one and
# fails on the other for no reason that concerns the guest.
awk -v b="$BASE" -v o="$OPEN" 'BEGIN{ d = b - o; if (d < 0) d = -d; exit !(d > 8) }' || {
  if guest_gone; then
    echo "menu-close-probe: VERDICT=GUEST_GONE — the guest died during the \
opening captures; this run says nothing about the animation"
    exit 2
  fi
  echo "menu-close-probe: the menu box did not change when the menu opened \
(no_menu=$BASE menu_open=$OPEN); the right-click may not have opened one"
  exit 2; }

# ---- 2. Aim the capture so its grab lands inside the fade.
#
# Wanted: a dismiss delay whose frame is strictly between the two settled
# states. Scanned rather than bisected because the response is not monotone at
# the edges, and eight captures cost under fifteen seconds. Compared with
# `mid_between` so the test does not assume which of the two states is brighter.
mid_between() {
  awk -v m="$1" -v a="$2" -v b="$3" 'BEGIN{
    lo = (a < b ? a : b); hi = (a < b ? b : a);
    exit !(m > lo + 3 && m < hi - 3) }'
}
BESTD=""; BESTM=""
for D in 0.34 0.38 0.42 0.46 0.50; do
  open_menu; sleep 0.9
  ( sleep "$D"; dismiss_menu ) &
  shot "$OUT/aim-$D.png"; wait
  M="$(crop_mean "$OUT/aim-$D.png" "$CROP")"
  echo "menu-close-probe: aim delay=$D mean=$M"
  if mid_between "$M" "$BASE" "$OPEN"; then BESTD="$D"; BESTM="$M"; fi
  sleep 0.8
done
if [ -z "$BESTD" ]; then
  echo "menu-close-probe: no delay landed inside the fade; the animation was \
not sampled and this run says nothing about the defect"
  exit 2
fi
echo "menu-close-probe: aiming at delay=$BESTD (mid-animation mean=$BESTM)"

# ---- 3. Repeat, and score how dark the vacated region got.
#
# The score is the fraction of the menu box that is near-black in the frame but
# was NOT near-black with no menu on screen. A wallpaper with dark foliage in it
# reads near-black honestly, and subtracting the no-menu frame is what keeps the
# probe from reporting the desktop as the defect.
mkdir -p "$OUT/frames"
# White where the no-menu frame is **not** near-black.
#
# The `-negate` that used to be here made this "white where the no-menu frame
# *is* near-black", and the `Darken` composite below then measured
# `near-black now AND near-black before` -- the intersection, not the
# difference. Over a static wallpaper crop that quantity does not depend on the
# trial frame at all, so it returned the same number for every trial of every
# boot: 0.0829293, the wallpaper's own dark fraction, on four boots across two
# different binaries. The probe could not see a black rectangle and reported
# CLEAN regardless.
#
# Demonstrated rather than reasoned: a synthetic frame with half the box painted
# black scores 0 under the old pipeline and 0.50 under this one. That case is
# `self_test` below, and it now runs on every invocation.
base_mask "$OUT/closed.png" "$CROP" "$OUT/basemask.png"
WORST=0; WORSTI=""; N=0
for i in $(seq 1 "$TRIALS"); do
  open_menu; sleep 0.9
  ( sleep "$BESTD"; dismiss_menu ) &
  shot "$OUT/frames/t-$i.png"; wait
  [ -s "$OUT/frames/t-$i.png" ] || continue
  # Scored only while the guest is still the thing being photographed. A frame
  # grabbed after it died is a photograph of a frozen host window.
  if guest_gone; then
    echo "menu-close-probe: guest went away during trial $i; scoring the \
$N trial(s) that preceded it"
    break
  fi
  N=$((N + 1))
  M="$(crop_mean "$OUT/frames/t-$i.png" "$CROP")"
  # near-black in this frame AND not near-black in the no-menu frame
  BLACK="$(new_black_frac "$OUT/frames/t-$i.png" "$CROP" "$OUT/basemask.png")"
  echo "trial=$i mean=$M new_black_frac=$BLACK"
  if awk -v a="$BLACK" -v b="$WORST" 'BEGIN{ exit !(a > b) }'; then
    WORST="$BLACK"; WORSTI="$i"
  fi
  sleep 0.6
done

if [ "$N" -eq 0 ]; then
  echo "menu-close-probe: VERDICT=GUEST_GONE — no trial completed against a \
live guest"
  exit 2
fi
echo "menu-close-probe: $N frames in $OUT/frames"
echo "menu-close-probe: worst new-black fraction=$WORST (trial $WORSTI) over box $CROP"
# A tenth of the menu's own box turning black that was not black before is not
# wallpaper and is not the menu's material.
if awk -v w="$WORST" 'BEGIN{ exit !(w > 0.10) }'; then
  echo "menu-close-probe: VERDICT=BLACK_RECTANGLE"; exit 1
fi
echo "menu-close-probe: VERDICT=CLEAN"
