#!/usr/bin/env bash
# wait-for-desktop.sh — block until this rail has a composited desktop, logging
# in first if that is what it is waiting for.
#
# sshd answers well before the desktop composites, so every harness here waits on
# `pgrep -x Dock` rather than on port 2222. That wait has one failure it cannot
# distinguish from a slow boot, and macos-12 sits in it forever: **the guest
# reached the login window and stopped**. Its serial log says so —
# `IOConsoleUsers: ... lin 0, llk 1` and `gIOScreenLockState 3`, repeating — while
# ssh answers, `guest-authorize.sh` installs its key, and Apple Events report
# consented. Every signal a harness reads says the guest is healthy, because it
# is; nobody is logged in. A whole rail was reported NO-DESKTOP for it.
#
# The console user is the discriminator and it costs one ssh round trip:
# `stat -f%Su /dev/console` is `root` at the login window and the account name
# once a session owns the console. So this waits, and when it finds the console
# owned by root it types the password at the login window through QMP — host
# side, because the guest cannot log itself in.
#
# Usage:
#   scripts/app-sweep-probe/wait-for-desktop.sh [--timeout N] [--password P]
#
# Exit 0 once `pgrep -x Dock` succeeds. Exit 1 on timeout, having said which of
# the two states it timed out in, which is the part the old inline loop could not
# report.
set -uo pipefail
export LC_ALL=C

TIMEOUT=400
# The same throwaway default `vm/guest-authorize.sh` documents; these rails share
# one account and it is not a secret this repository is keeping.
PASSWORD="${REIMS_GUEST_PASSWORD:-aneesiqbal}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --password) PASSWORD="$2"; shift 2 ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "wait-for-desktop: unknown argument $1" >&2; exit 2 ;;
  esac
done

say() { echo "wait-for-desktop: $*"; }
gssh() { timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 macos-vm "$1" 2>/dev/null; }
qmp() {
  QMP_SOCK="${QMP_SOCK:-$REPO/vm/disks/run/qmp.sock}" \
    timeout 30 "$REPO/scripts/qmp/qmp.py" "$@" >/dev/null 2>&1
}

# At most twice. A password typed at a window that is not the login window goes
# somewhere else, and retrying a wrong guess forever is how a harness turns one
# failure into a locked account.
attempts=0
state=starting
deadline=$((SECONDS + TIMEOUT))

while [ "$SECONDS" -lt "$deadline" ]; do
  if gssh 'pgrep -x Dock >/dev/null'; then
    say "desktop up (console user $(gssh 'stat -f%Su /dev/console' || echo '?'))"
    exit 0
  fi

  console=$(gssh 'stat -f%Su /dev/console' || true)
  case "$console" in
    root)
      state=login-window
      if [ "$attempts" -lt 2 ]; then
        attempts=$((attempts + 1))
        say "console is owned by root — the login window; typing the password (attempt $attempts)"
        # A single-account login window comes up with the password field focused,
        # so attempt one just types. If that did not take, the window was showing
        # the user *list* instead and the characters went nowhere: a Return picks
        # the highlighted account and gives the field focus, so attempt two leads
        # with one.
        [ "$attempts" -ge 2 ] && { qmp key ret; sleep 2; }
        qmp type "$PASSWORD"
        sleep 1
        qmp key ret
        sleep 20
      fi
      ;;
    '') state=${state} ;;   # ssh did not answer this round; say nothing new
    *) state="logged-in-as-$console" ;;
  esac
  sleep 10
done

say "no Dock within ${TIMEOUT}s, last state: $state"
exit 1
