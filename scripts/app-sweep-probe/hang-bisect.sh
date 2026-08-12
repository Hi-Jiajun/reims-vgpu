#!/usr/bin/env bash
# hang-bisect.sh — which rail of this device hangs the host GPU?
#
# A `VK_ERROR_DEVICE_LOST` here is not a Vulkan-level refusal, it is the host
# kernel resetting the GPU: `i915 … GPU HANG: ecode … context reset due to GPU
# hang`. The device survives it — it recreates and the guest keeps running — but
# every recreate loses the writebacks in flight (`render_store_lost`) and stalls
# the drain for ten to twenty seconds, which is what `app-sweep-probe` reports as
# FREEZE.
#
# The device's own log cannot say which submission hung: by the time anything is
# observable the fence has already failed and the batch that caused it is gone.
# The kernel's `ecode` is a hash of the hanging context's state and is stable per
# cause, so counting hangs per arm is the instrument, and the arms are the
# `REIMS_VGPU_*` narrowing switches — each one turns a rail off without turning
# anything on, which is the only direction an override is allowed to move.
#
# Each arm is one driven boot. The verdict is the number of `GPU HANG` lines the
# kernel logged during it, read from the journal by timestamp so a hang from a
# previous arm cannot be counted twice.
#
# Usage:
#   scripts/app-sweep-probe/hang-bisect.sh [--rail NAME] [--seconds N]
#     [--arms "shipping GUEST_IMPORT=off COMPUTE_GATHER=off ..."]
#
# An arm is either the word `shipping` or one or more comma-joined `NAME=value`
# pairs, each exported as `REIMS_VGPU_NAME=value` for that boot. A comma-joined
# arm is how the whole set of narrowing switches gets tested in one boot, which
# is the first question worth asking: if the hang survives every optimization
# being switched off, no optimization causes it and there is nothing to bisect.
#
# Every `REIMS_VGPU_*` variable in the environment is cleared before each arm, so
# an arm cannot inherit the previous one's switch. The names come from the
# environment itself rather than a list kept here, because a list would go stale
# the first time `env.rs` grows a switch and the drift is invisible.
set -uo pipefail
export LC_ALL=C

RAIL="macos-11"
SECONDS_PER_APP=12
ARMS="shipping GUEST_IMPORT=off COMPUTE_GATHER=off COMPUTE_SCATTER=off"
OUT="/tmp/reims-hang-bisect"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --rail) RAIL="$2"; shift 2 ;;
    --seconds) SECONDS_PER_APP="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "hang-bisect: unknown argument $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
say() { echo "hang-bisect: $*"; }
RESULTS="$OUT/results.tsv"
: >"$RESULTS"

# The kernel's own count, bounded to this arm's wall clock. `--since` takes a
# local timestamp, which is what `date` prints, so the two agree without a
# timezone argument.
hangs_since() {
  journalctl -k --since "$1" --no-pager 2>/dev/null | grep -c 'GPU HANG' || true
}

for arm in $ARMS; do
  say "=== arm $arm ==="
  # Not a fixed sleep: an arm that just hung the GPU takes longer to die than
  # one that did not, and this bisect exists to run exactly those arms.
  "$REPO/scripts/app-sweep-probe/stop-previous-vm.sh" || \
    say "$arm: previous VM still holds :2222"
  rm -f /tmp/reims-vgpu-fail.log
  # Exported, never passed as argv: a command line naming the arm would be
  # matched by the `pkill` above on the next round.
  for stale in $(env | sed -n 's/^\(REIMS_VGPU_[A-Z0-9_]*\)=.*/\1/p'); do
    unset "$stale"
  done
  if [ "$arm" != shipping ]; then
    # Comma-joined, so one arm can carry every switch at once.
    old_ifs=$IFS; IFS=,
    for pair in $arm; do
      export "REIMS_VGPU_${pair%%=*}=${pair##*=}"
    done
    IFS=$old_ifs
  fi

  started=$(date '+%Y-%m-%d %H:%M:%S')
  BOOTLOG="$OUT/$arm-boot.log"
  TESTING_TIMEOUT=900 nohup "$REPO/vm/boot-x86.sh" --device reims-vgpu-pci \
    --rail "$RAIL" --testing >"$BOOTLOG" 2>&1 &

  up=no
  for _ in $(seq 1 60); do
    [ -f /tmp/reims-vgpu-fail.log ] && { up=yes; break; }
    sleep 5
  done
  [ "$up" = yes ] || { say "$arm: no device"; printf '%s\tNO-BOOT\t-\n' "$arm" >>"$RESULTS"; continue; }

  timeout 300 "$REPO/vm/guest-authorize.sh" >/dev/null 2>&1
  # Shared with `sweep-rails.sh`: it separates "still booting" from "stopped at
  # the login window" and logs in for the second, which a bare `pgrep -x Dock`
  # loop cannot do. An arm that never reaches a desktop drives nothing, so its
  # hang count would be a measurement of an idle GPU.
  dock=yes
  "$REPO/scripts/app-sweep-probe/wait-for-desktop.sh" --timeout 400 || dock=no
  [ "$dock" = yes ] || { say "$arm: no desktop"; printf '%s\tNO-DESKTOP\t-\n' "$arm" >>"$RESULTS"; continue; }
  sleep 8

  QMP_SOCK="$REPO/vm/disks/run/qmp.sock" timeout 700 \
    "$REPO/scripts/app-sweep-probe/app-sweep-probe.sh" --rail "$RAIL" \
    --seconds "$SECONDS_PER_APP" --torture-seconds 30 \
    --keep "$OUT/$arm-work" >"$OUT/$arm-probe.log" 2>&1
  probe=$?

  pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
  sleep 6
  hangs=$(hangs_since "$started")
  freezes=$(grep -c 'FREEZE' "$OUT/$arm-probe.log" || true)
  # The capability line says whether the arm took, so an override that was
  # ignored cannot be scored as an arm that ran.
  caps=$(grep -m1 -o 'host_pointer_import=[a-z_]*' "$OUT/$arm-work"/*.log 2>/dev/null | head -1)
  printf '%s\thangs=%s\tfreezes=%s\tprobe_exit=%s\t%s\n' \
    "$arm" "$hangs" "$freezes" "$probe" "${caps:-caps=?}" >>"$RESULTS"
  say "$arm: GPU hangs=$hangs freezes=$freezes probe_exit=$probe ${caps:-}"
done

echo
echo "=== hang bisect on $RAIL ==="
cat "$RESULTS"
