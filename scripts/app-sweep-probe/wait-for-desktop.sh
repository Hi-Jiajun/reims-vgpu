#!/usr/bin/env bash
# wait-for-desktop.sh — block until this rail has a composited desktop, and
# collect the guest's crash reports if it finds a login window instead.
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
# `stat -f%Su /dev/console` names the account once a session owns the console.
# Before that it is a *system* account, and which one is not stable — the usual
# answer is `root`, but a macos-11 guest whose WindowServer had aborted answered
# `_windowserver`, measured live. So the test is not "is it root": it is "is it
# an account a person could log in as", and every system answer — `root`, any
# leading-underscore daemon account, or an empty string from an ssh that did not
# land — counts as nobody being logged in.
#
# **A login window is evidence, and logging in destroys it.** The two states
# above are not equally innocent. `root` owning the console is a guest that never
# logged in; `_windowserver` owning it is a guest whose WindowServer *aborted*,
# and the guest wrote a crash report saying why. Typing the password restarts the
# session over the top of it, and the next boot's snapshot revert throws the
# report away — so a whole class of failure has been arriving as "the desktop
# took a while" for as long as this script has existed.
#
# So the login window is now a collection point. Every time one is seen this
# pulls `/Library/Logs/DiagnosticReports/*.ips` and the user's own copy to
# `--reports DIR` first, and **refuses to log in** when any of them names a
# crash. `--login-after-crash` is the override for a session that wants the
# desktop anyway, and it is off by default: an unattended sweep must not trade a
# crash report for a screenshot.
#
# Usage:
#   scripts/app-sweep-probe/wait-for-desktop.sh [--timeout N] [--password P]
#     [--reports DIR] [--login-after-crash]
#
# Exit 0 once `pgrep -x Dock` succeeds. Exit 1 on timeout, having said which of
# the two states it timed out in, which is the part the old inline loop could not
# report. Exit 3 when a crash report was collected and the login was refused.
set -uo pipefail
export LC_ALL=C

TIMEOUT=400
REPORTS="${REIMS_GUEST_REPORTS:-/tmp/reims-guest-reports}"
LOGIN_AFTER_CRASH=no
# The same throwaway default `vm/guest-authorize.sh` documents; these rails share
# one account and it is not a secret this repository is keeping.
PASSWORD="${REIMS_GUEST_PASSWORD:-aneesiqbal}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --password) PASSWORD="$2"; shift 2 ;;
    --reports) REPORTS="$2"; shift 2 ;;
    --login-after-crash) LOGIN_AFTER_CRASH=yes; shift ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "wait-for-desktop: unknown argument $1" >&2; exit 2 ;;
  esac
done

say() { echo "wait-for-desktop: $*"; }
gssh() { timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 macos-vm "$1" 2>/dev/null; }

# Copy every diagnostic report the guest holds into `$REPORTS`, and answer with
# the names of the ones that are crashes.
#
# Both directories, because they hold different halves of the same event: a
# WindowServer abort is a *system* process and lands in `/Library/Logs`, while
# anything the logged-in user ran lands under `~`. Read without `sudo` on
# purpose — `AGENTS.md` records a guest `sudo` wedging with its timestamp lock
# held, which queues every later `sudo` forever, and a report this account
# cannot read is worth less than a boot.
#
# `tar` over the pipe rather than `scp` per file: one round trip, no second
# authentication, and it needs no path quoting for names that carry spaces.
collect_reports() {
  mkdir -p "$REPORTS"
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=5 macos-vm \
    'tar -cf - -C / Library/Logs/DiagnosticReports 2>/dev/null; \
     tar -cf - -C "$HOME" Library/Logs/DiagnosticReports 2>/dev/null' \
    2>/dev/null | tar -xf - -C "$REPORTS" 2>/dev/null
  find "$REPORTS" -name '*.ips' -o -name '*.crash' -o -name '*.panic' 2>/dev/null
}
qmp() {
  QMP_SOCK="${QMP_SOCK:-$REPO/vm/disks/run/qmp.sock}" \
    timeout 30 "$REPO/scripts/qmp/qmp.py" "$@" >/dev/null 2>&1
}

# At most twice. A password typed at a window that is not the login window goes
# somewhere else, and retrying a wrong guess forever is how a harness turns one
# failure into a locked account.
attempts=0
collected=no
state=starting
deadline=$((SECONDS + TIMEOUT))

while [ "$SECONDS" -lt "$deadline" ]; do
  if gssh 'pgrep -x Dock >/dev/null'; then
    say "desktop up (console user $(gssh 'stat -f%Su /dev/console' || echo '?'))"
    # Collected on the way out too. A WindowServer that aborted and was restarted
    # by autologin never shows this loop a login window at all, so the console
    # check cannot see that class — the report is the only thing that can, and it
    # is sitting on the guest either way.
    crashes=$(collect_reports)
    [ -n "$crashes" ] && {
      say "crash reports present on a guest that reached the desktop — $REPORTS:"
      printf '  %s\n' $crashes
    }
    exit 0
  fi

  console=$(gssh 'stat -f%Su /dev/console' || true)
  # An empty answer is an ssh that did not land, not a verdict: say nothing and
  # come round again rather than typing a password at a guest we cannot see.
  [ -z "$console" ] && { sleep 10; continue; }
  case "$console" in
    root|_*)
      state="login-window (console $console)"
      # Collected before anything is typed, and only once: the reports are the
      # reason this state is interesting, and a login overwrites the session
      # that produced them.
      if [ "$collected" = no ]; then
        collected=yes
        crashes=$(collect_reports)
        if [ -n "$crashes" ]; then
          say "CRASH REPORTS collected into $REPORTS:"
          printf '  %s\n' $crashes
          if [ "$LOGIN_AFTER_CRASH" != yes ]; then
            say "refusing to log in — a login restarts the session over the evidence."
            say "pass --login-after-crash if the desktop is wanted anyway."
            exit 3
          fi
          say "--login-after-crash given; logging in over the crash anyway"
        else
          say "no crash reports on the guest; this is a guest that never logged in"
        fi
      fi
      if [ "$attempts" -lt 2 ]; then
        attempts=$((attempts + 1))
        say "console is owned by '$console' — nobody is logged in; typing the password (attempt $attempts)"
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
    # A real account owns the console, so someone is logged in and the desktop is
    # merely still coming up. Nothing to do but wait.
    *) state="logged-in-as-$console" ;;
  esac
  sleep 10
done

say "no Dock within ${TIMEOUT}s, last state: $state"
exit 1
