#!/usr/bin/env bash
#
# vm/guest-authorize.sh — make the running x86 guest reachable as ssh host
# `macos-vm`, whichever rail it came from.
#
# WHY THIS EXISTS. Every probe under `scripts/` reaches the guest as
# `ssh -o BatchMode=yes macos-vm`, which is key auth and nothing else. Exactly
# one rail was provisioned with that key; the rest authenticate by password. A
# probe run against them fails at the first hop, and `BatchMode=yes` makes that
# failure look like "guest not up yet" rather than "guest has no key".
#
# Installing the key into each rail's snapshot would work and is the wrong
# shape: a `--testing` boot is a byte-identical COW clone of an immutable
# snapshot, so authorizing the *clone* costs one second, survives exactly as
# long as the boot it belongs to, and mutates nothing that outlives it. Run
# this once after a boot comes up and every existing probe works unchanged.
#
# Idempotent. On a rail that already has the key it does not need the password
# at all, so it is safe to run unconditionally in a harness.
#
#   vm/guest-authorize.sh                 # wait for sshd, authorize, verify
#   vm/guest-authorize.sh --timeout 300   # bound the wait differently
#
# Environment:
#   SSH_PORT              host port forwarded to the guest's 22 (default 2222)
#   REIMS_GUEST_USER      guest account (default aneesiqbal)
#   REIMS_GUEST_PASSWORD  its password (default aneesiqbal) — a throwaway
#                         credential for a local development VM, not a secret
#   REIMS_GUEST_KEY       private key whose .pub gets installed
#                         (default ~/.ssh/macos_x86_guest, the key `macos-vm`
#                         names in ~/.ssh/config)
#
# Every guest-side step is bounded by `timeout` on the host side. A wedged
# guest — a stuck `sshd`, a login shell waiting on something — must cost this
# script its deadline and no more, because the harness that calls it is
# unattended and a hang here reads as a hung boot.
set -euo pipefail

SSH_PORT="${SSH_PORT:-2222}"
GUEST_USER="${REIMS_GUEST_USER:-aneesiqbal}"
GUEST_PASSWORD="${REIMS_GUEST_PASSWORD:-aneesiqbal}"
GUEST_KEY="${REIMS_GUEST_KEY:-$HOME/.ssh/macos_x86_guest}"
WAIT_SECONDS=420

# One ssh attempt must never outlast this, password or key.
STEP_TIMEOUT=30

die() { echo "guest-authorize: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --timeout) shift; WAIT_SECONDS="${1:-}"; [ -n "$WAIT_SECONDS" ] || die "--timeout needs seconds"; shift ;;
    --timeout=*) WAIT_SECONDS="${1#--timeout=}"; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[ -f "$GUEST_KEY" ] || die "no private key at $GUEST_KEY (set REIMS_GUEST_KEY)"
[ -f "$GUEST_KEY.pub" ] || die "no public key at $GUEST_KEY.pub"
PUBKEY="$(cat "$GUEST_KEY.pub")"

# The guest's host key changes with every rail and every reprovision, and this
# is a loopback port on a development box — pinning it would only produce a
# mismatch to click through. Keep that decision out of the user's known_hosts.
SSH_COMMON=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ConnectTimeout=8
  -p "$SSH_PORT"
)

ssh_key() {
  timeout "$STEP_TIMEOUT" ssh "${SSH_COMMON[@]}" \
    -o BatchMode=yes -o IdentitiesOnly=yes -i "$GUEST_KEY" \
    "$GUEST_USER@127.0.0.1" "$@"
}

ssh_password() {
  command -v sshpass >/dev/null 2>&1 || die "sshpass not found (needed to authorize a rail that has no key yet)"
  timeout "$STEP_TIMEOUT" sshpass -p "$GUEST_PASSWORD" ssh "${SSH_COMMON[@]}" \
    -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 \
    "$GUEST_USER@127.0.0.1" "$@"
}

# --- Wait for sshd ---------------------------------------------------------
# Either credential answering means sshd is up; which one it was is the next
# question, not this one. A refused connection is "not yet", anything else is
# still "not yet" until the deadline, because a guest mid-login refuses in
# several different ways.
deadline=$(( $(date +%s) + WAIT_SECONDS ))
up=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ssh_key true 2>/dev/null || ssh_password true 2>/dev/null; then up=1; break; fi
  sleep 5
done
[ "$up" -eq 1 ] || die "guest sshd did not answer on port $SSH_PORT within ${WAIT_SECONDS}s"

# --- Authorize -------------------------------------------------------------
if ssh_key true 2>/dev/null; then
  echo "guest-authorize: key auth already works (port $SSH_PORT, user $GUEST_USER)"
else
  echo "guest-authorize: installing $GUEST_KEY.pub for $GUEST_USER ..."
  # Appending only when absent keeps a re-run from growing the file, and the
  # 0700/0600 modes are what macOS sshd requires before it will read it at all.
  ssh_password "set -e
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
    grep -qxF '$PUBKEY' ~/.ssh/authorized_keys || printf '%s\n' '$PUBKEY' >> ~/.ssh/authorized_keys" \
    || die "could not install the key over password auth"

  ssh_key true 2>/dev/null \
    || die "key installed but key auth still fails — check the guest's sshd config"
  echo "guest-authorize: key auth now works"
fi

# --- Report what a probe will see ------------------------------------------
# `macos-vm` is what the probes actually type, and it authenticates against the
# user's own ~/.ssh/known_hosts. That file cannot hold a useful entry for this
# endpoint: `127.0.0.1:2222` is a different machine on every rail, so whichever
# rail booted first wins and every other rail then fails the host-key check —
# with `BatchMode=yes` turning the mismatch into a silent probe failure. Forget
# the pin before verifying; `accept-new` re-learns it for the rail that is
# actually running.
for host in "[127.0.0.1]:$SSH_PORT" "[localhost]:$SSH_PORT"; do
  timeout "$STEP_TIMEOUT" ssh-keygen -R "$host" >/dev/null 2>&1 || true
done

# If ~/.ssh/config points `macos-vm` somewhere else, say so here rather than
# letting a probe fail obscurely.
if timeout "$STEP_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=8 \
     -o StrictHostKeyChecking=accept-new macos-vm true 2>/dev/null; then
  echo "guest-authorize: ssh macos-vm ok — probes under scripts/ will reach this guest"
else
  echo "guest-authorize: WARNING ssh macos-vm failed even though direct key auth works." >&2
  echo "guest-authorize: probes default to GUEST=macos-vm; check ~/.ssh/config names port $SSH_PORT and $GUEST_KEY." >&2
  exit 1
fi
