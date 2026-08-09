# guest-display.sh — the one place a probe asks the guest how big its desktop is.
#
# Source it and call `guest_display_size <ssh-host>`; it prints `WIDTH HEIGHT`
# on stdout and returns non-zero if the guest could not answer.
#
# # Why not system_profiler
#
# `system_profiler SPDisplaysDataType | grep Resolution:` is the obvious
# spelling and it is not portable across the guest OS lines this device
# supports. A macOS 12 guest on this device prints the GPU and **no display at
# all** — no `Displays:` section, no `Resolution:` line — so every probe that
# read it exited 2 with "guest reported no display resolution", which reads like
# a wedged guest rather than like one report being unavailable. It has also been
# observed hanging indefinitely on a macOS 11 guest, which an unattended harness
# cannot tell from a wedged boot.
#
# Nor from Finder: `tell application "Finder" to get bounds of window of desktop`
# answers `AppleEvent timed out (-1712)` on these guests.
#
# `screencapture` writes the framebuffer and `sips` reads its header. Both ship
# with every macOS, neither needs root, and the answer is the strongest form of
# the question a two-observation probe is asking — not what the guest's
# *configuration* claims the display is, but how many pixels its compositor just
# produced. A probe that then draws at that size is drawing at the size the
# frame it will measure actually has.
#
# The capture is bounded host-side by `timeout`: an unattended harness cannot
# tell a wedged guest command from a wedged boot, and this one runs against
# guests that are sometimes wedged by construction.

# Run one AppleScript in the guest, bounded.
#
# `osascript -e 'tell application "System Events" to ...'` over ssh **hangs
# forever** on a guest that has not granted the ssh session Automation access:
# the request raises a consent prompt on the guest's own desktop and nobody is
# there to answer it. Measured on the macos-12 rail, where every System Events
# call in every probe hung until the probe's own harness gave up, and the probe
# then reported the empty answer as "the guest says the desktop picture is ''".
#
# So the bound is not defensive tidiness, it is the difference between a probe
# that reports "this rail cannot be scripted" in 15 seconds and one that stalls
# an unattended sweep. The host-side `timeout` does not kill the remote
# osascript; it only stops us waiting on it.
guest_osa() {
  local guest="$1" script="$2" secs="${GUEST_OSA_TIMEOUT:-15}"
  timeout "$secs" ssh -o BatchMode=yes "$guest" "osascript -e '$script'" 2>/dev/null
}

# Print "WIDTH HEIGHT" for $1's desktop, or return 1 having said why on stderr.
guest_display_size() {
  local guest="$1" out w h
  out=$(timeout 90 ssh -o BatchMode=yes "$guest" 'bash -s' 2>/dev/null <<'GUEST_EOF'
set -e
out=/tmp/reims-guest-display-size.png
rm -f "$out"
/usr/sbin/screencapture -x -t png "$out" >/dev/null 2>&1
/usr/bin/sips -g pixelWidth -g pixelHeight "$out" 2>/dev/null
rm -f "$out"
GUEST_EOF
  ) || true
  w=$(printf '%s\n' "$out" | sed -n 's/.*pixelWidth: *\([0-9][0-9]*\).*/\1/p' | head -1)
  h=$(printf '%s\n' "$out" | sed -n 's/.*pixelHeight: *\([0-9][0-9]*\).*/\1/p' | head -1)
  if [ -z "$w" ] || [ -z "$h" ]; then
    echo "guest-display: $guest did not report a desktop size (got '$out')" >&2
    return 1
  fi
  printf '%s %s\n' "$w" "$h"
}
