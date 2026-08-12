#!/usr/bin/env bash
# screenshot-when-kde-plasma-host.sh — host-side PNG of a live QEMU GTK/Wayland window.
#
# Captures the Plasma/KWin window for a running qemu-system-* (not guest QMP
# screendump). Safe over SSH: imports the graphical session's env from
# plasmashell / kwin_wayland rather than requiring WAYLAND_DISPLAY in the tty.
#
# Usage:
#   scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh [-o PATH] [--pid PID] [--decorations]
#
# Prints the PNG path on stdout. Errors on stderr and exits non-zero if no QEMU
# process/window, no graphical session, or capture tools are missing.
#
# Output is always downscaled to fit 1280x720 (aspect preserved, never upscaled)
# to keep the token cost of agent-read screenshots down. There is no flag for it.
set -euo pipefail

OUT=""
PID_FILTER=""
WINDOW_FILTER=""
MATCH_FILTER=""
ALLOW_ANY=0
ALLOW_BLACK=0
INCLUDE_DECORATIONS=0
SCRIPT_NAME="screenshot-when-kde-plasma-host"

# The host-owned window's caption, set in crates/reims-vgpu (device_window_start
# builds the WindowConfig with this title). It is a compile-time constant, so the
# common case needs no selector at all -- asking the caller to pass a string the
# tree already knows is what --match used to do, and it could not disambiguate
# two VMs anyway (both windows carry this same caption; use --window for that).
DEFAULT_TITLE="Reims vGPU"

usage() {
  cat <<'EOF'
usage: scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh [options]

Capture the host QEMU display window (GTK on Plasma Wayland) to a PNG.

Options:
  -o, --output PATH   Write PNG here (default: temp file under /tmp)
  --pid PID           Capture the window owned by this qemu-system PID.
                      STRICT: if no window matches, this errors out rather
                      than capturing some other QEMU window.
  --window UUID       Capture this exact KWin internalId. The only selector
                      that disambiguates two concurrent VMs, whose windows
                      share the same caption. Ids come from the candidate
                      dump an unmatched selector prints.
  --match SUBSTR      Capture the window whose caption/class contains SUBSTR.
                      Escape hatch for a non-default window -- e.g. QEMU's own
                      GTK window when comparing its DisplaySurface against the
                      host-owned one. Not needed for the normal case.
  --any               Allow falling back to the largest qemu-class window when
                      the selector matches nothing. Off by default: the silent
                      fallback used to capture the WRONG VM's window.
  --allow-black       Exit 0 even if the capture is uniformly black. Off by
                      default: a black frame is treated as a failed capture.
  --decorations       Include window decorations (default: content only)
  -h, --help          Show this help

Selector precedence: --window, then --pid, then --match. With none given, the
host-owned window is selected by its known caption -- the normal case needs no
selector. Among windows passing a selector, the largest wins (QEMU may expose a
small serial console alongside the GPU display).

Works from an SSH tty by attaching to the user's active Wayland session.
Prints the output path on stdout.
EOF
}

die() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o | --output)
      [[ $# -ge 2 ]] || die "$1 requires a path"
      OUT="$2"
      shift 2
      ;;
    --output=*)
      OUT="${1#--output=}"
      shift
      ;;
    --pid)
      [[ $# -ge 2 ]] || die "--pid requires a value"
      PID_FILTER="$2"
      shift 2
      ;;
    --pid=*)
      PID_FILTER="${1#--pid=}"
      shift
      ;;
    --window)
      [[ $# -ge 2 ]] || die "--window requires a KWin internalId"
      WINDOW_FILTER="$2"
      shift 2
      ;;
    --window=*)
      WINDOW_FILTER="${1#--window=}"
      shift
      ;;
    --match)
      [[ $# -ge 2 ]] || die "--match requires a substring"
      MATCH_FILTER="$2"
      shift 2
      ;;
    --match=*)
      MATCH_FILTER="${1#--match=}"
      shift
      ;;
    --any)
      ALLOW_ANY=1
      shift
      ;;
    --allow-black)
      ALLOW_BLACK=1
      shift
      ;;
    --decorations)
      INCLUDE_DECORATIONS=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

need_cmd spectacle
need_cmd pgrep

# Prefer Qt6 qdbus; fall back.
QDBUS=""
for c in qdbus6 qdbus-qt6 qdbus; do
  if command -v "$c" >/dev/null 2>&1; then
    QDBUS="$c"
    break
  fi
done
[[ -n "$QDBUS" ]] || die "required command not found: qdbus6 (or qdbus)"

# --- graphical session env (works when this shell is a plain SSH tty) ---------
import_session_env() {
  local pid cmd envfile key val
  for pid in $(pgrep -u "$(id -u)" -x plasmashell 2>/dev/null) \
             $(pgrep -u "$(id -u)" -x kwin_wayland 2>/dev/null); do
    envfile="/proc/${pid}/environ"
    [[ -r "$envfile" ]] || continue
    # shellcheck disable=SC2162
    while IFS= read -r -d '' line; do
      key="${line%%=*}"
      val="${line#*=}"
      case "$key" in
        DISPLAY | WAYLAND_DISPLAY | XDG_RUNTIME_DIR | DBUS_SESSION_BUS_ADDRESS | XDG_CURRENT_DESKTOP)
          export "${key}=${val}"
          ;;
      esac
    done <"$envfile"
    if [[ -n "${DBUS_SESSION_BUS_ADDRESS:-}" && -n "${XDG_RUNTIME_DIR:-}" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" || -z "${WAYLAND_DISPLAY:-}" || -z "${XDG_RUNTIME_DIR:-}" ]]; then
  import_session_env || true
fi

# Sensible defaults when still incomplete but same uid.
: "${XDG_RUNTIME_DIR:=/run/user/$(id -u)}"
if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" && -S "${XDG_RUNTIME_DIR}/bus" ]]; then
  export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"
fi
if [[ -z "${WAYLAND_DISPLAY:-}" && -S "${XDG_RUNTIME_DIR}/wayland-0" ]]; then
  export WAYLAND_DISPLAY=wayland-0
fi
if [[ -z "${DISPLAY:-}" ]]; then
  export DISPLAY=:0
fi

[[ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ]] || die "no session bus (is Plasma/KWin running for this user?)"
[[ -n "${XDG_RUNTIME_DIR:-}" && -d "${XDG_RUNTIME_DIR}" ]] || die "XDG_RUNTIME_DIR missing or unusable"
if [[ ! -S "${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY:-wayland-0}" && ! -S "${XDG_RUNTIME_DIR}/wayland-0" ]]; then
  die "no Wayland socket under ${XDG_RUNTIME_DIR} (is a graphical session active?)"
fi

# spectacle SIGABRTs (exit 134, core dumped, no output file) when WAYLAND_DISPLAY
# is unset even though the session bus is reachable — fail with the reason
# instead of letting the crash surface as an opaque capture failure.
[[ -n "${WAYLAND_DISPLAY:-}" ]] || die "WAYLAND_DISPLAY unresolved (spectacle aborts without it)"

# Probe KWin
if ! "$QDBUS" org.kde.KWin /KWin >/dev/null 2>&1; then
  die "cannot reach org.kde.KWin on the session bus (Plasma Wayland not available)"
fi

# --- live qemu-system PIDs ----------------------------------------------------
mapfile -t QEMU_PIDS < <(pgrep -u "$(id -u)" -x qemu-system-x86_64 2>/dev/null || true)
mapfile -t QEMU_PIDS_A64 < <(pgrep -u "$(id -u)" -x qemu-system-aarch64 2>/dev/null || true)
if [[ ${#QEMU_PIDS_A64[@]} -gt 0 ]]; then
  QEMU_PIDS+=("${QEMU_PIDS_A64[@]}")
fi
# Also match via full cmdline (custom install paths / names).
if [[ ${#QEMU_PIDS[@]} -eq 0 ]]; then
  mapfile -t QEMU_PIDS < <(pgrep -u "$(id -u)" -f 'qemu-system-(x86_64|aarch64)' 2>/dev/null || true)
fi

if [[ -n "$PID_FILTER" ]]; then
  [[ "$PID_FILTER" =~ ^[0-9]+$ ]] || die "--pid must be numeric"
  if ! kill -0 "$PID_FILTER" 2>/dev/null; then
    die "no process with pid ${PID_FILTER}"
  fi
  # Accept even if pgrep miss (custom binary name) as long as process lives.
  QEMU_PIDS=("$PID_FILTER")
fi

if [[ ${#QEMU_PIDS[@]} -eq 0 && -z "$WINDOW_FILTER" && -z "$MATCH_FILTER" ]]; then
  die "no qemu-system process found for user $(id -un)"
fi

# Dedup
declare -A _seen=()
_QEMU_UNIQ=()
for p in "${QEMU_PIDS[@]}"; do
  [[ -n "$p" ]] || continue
  [[ -z "${_seen[$p]:-}" ]] || continue
  _seen[$p]=1
  _QEMU_UNIQ+=("$p")
done
QEMU_PIDS=("${_QEMU_UNIQ[@]}")

# --- locate + activate KWin window -------------------------------------------
TOKEN="qemu-shot-$$-$(date +%s%N)"
PIDS_JS=$(printf '%s,' "${QEMU_PIDS[@]}")
PIDS_JS="${PIDS_JS%,}"

TMPDIR_SCRIPTS="$(mktemp -d "${TMPDIR:-/tmp}/screenshot-when-kde-plasma-host.XXXXXX")"
cleanup() {
  local id="${KWIN_SCRIPT_ID:-}"
  if [[ -n "$id" && -n "${QDBUS:-}" ]]; then
    "$QDBUS" org.kde.KWin "/Scripting/Script${id}" org.kde.kwin.Script.stop >/dev/null 2>&1 || true
    "$QDBUS" org.kde.KWin /Scripting org.kde.kwin.Scripting.unloadScript "screenshot-when-kde-plasma-host-${TOKEN}" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMPDIR_SCRIPTS"
}
trap cleanup EXIT

KWIN_JS="${TMPDIR_SCRIPTS}/activate.js"
# Selection is STRICT by design. The previous version fell back to "largest
# qemu-class window" whenever the requested PID matched nothing, which silently
# captured a different VM's window and reported success — the false-evidence
# mechanism behind the 2026-07-14 "false black" hunt. A selector that matches
# nothing is now an error naming every candidate, unless --any is passed.
#
# PID drift is real: the pid KWin attributes to the window can differ from the
# pid you launched or pgrep'd (observed 3365891 launched vs 3365895 on the
# window), so --match on the caption is the durable selector.
#
# Among windows that pass the selector, the largest wins: QEMU may expose a
# small serial console plus the GPU display, and the latter is the visual target.
# Build JS literals in bash: a heredoc cannot emit quotes via ${x:+\"$x\"}
# (backslash before " is not an escape there), so quote/escape them here.
js_string() {
  if [[ -z "$1" ]]; then
    printf 'undefined'
  else
    printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  fi
}
WANT_ID_JS="$(js_string "$WINDOW_FILTER")"
WANT_MATCH_JS="$(js_string "$MATCH_FILTER")"
DEFAULT_TITLE_JS="$(js_string "$DEFAULT_TITLE")"
STRICT_PID_JS="undefined"
[[ -n "$PID_FILTER" ]] && STRICT_PID_JS="1"

cat >"$KWIN_JS" <<EOF
const token = "${TOKEN}";
const wantPids = [${PIDS_JS}].map(function (x) { return Number(x); });
const wantId = ${WANT_ID_JS};
const wantMatch = ${WANT_MATCH_JS};
const defaultTitle = ${DEFAULT_TITLE_JS};
const allowAny = ${ALLOW_ANY};
const strictPid = ${STRICT_PID_JS};

function isQemu(c) {
    const cls = String(c.resourceClass || "").toLowerCase();
    const name = String(c.resourceName || "").toLowerCase();
    return cls.indexOf("qemu") >= 0 || name.indexOf("qemu") >= 0;
}
function textHas(c, needle) {
    const n = String(needle).toLowerCase();
    return String(c.caption || "").toLowerCase().indexOf(n) >= 0
        || String(c.resourceClass || "").toLowerCase().indexOf(n) >= 0;
}
function area(c) { return Number(c.width || 0) * Number(c.height || 0); }
function describe(c) {
    return "pid=" + c.pid + " id=" + c.internalId
        + " class=" + (c.resourceClass || "") + " caption=" + (c.caption || "")
        + " size=" + c.width + "x" + c.height + " minimized=" + c.minimized;
}

const clients = workspace.windowList();

// Always report the candidate set so a selection failure is diagnosable from
// one run instead of needing a second probe. The host-owned window is a winit
// window that sets no qemu app_id, so an isQemu-only dump can be blind exactly
// when the selector fails -- widen to the default title, and if even that finds
// nothing, list every window rather than print an empty candidate set.
let cands = clients.filter(function (c) { return isQemu(c) || textHas(c, defaultTitle); });
if (cands.length === 0) { cands = clients; }
for (let i = 0; i < cands.length; i++) {
    console.info(token + ":CAND " + describe(cands[i]));
}

let pool = [];
let selector = "";
if (typeof wantId !== "undefined") {
    selector = "--window " + wantId;
    pool = clients.filter(function (c) { return String(c.internalId) === String(wantId); });
} else if (typeof strictPid !== "undefined") {
    selector = "--pid " + wantPids.join(",");
    pool = clients.filter(function (c) { return wantPids.indexOf(Number(c.pid || 0)) >= 0; });
} else if (typeof wantMatch !== "undefined") {
    selector = "--match " + wantMatch;
    pool = clients.filter(function (c) { return textHas(c, wantMatch); });
} else {
    selector = "(default: caption \"" + defaultTitle + "\")";
    pool = clients.filter(function (c) { return textHas(c, defaultTitle); });
}

if (pool.length === 0 && allowAny) {
    console.info(token + ":FALLBACK selector=" + selector + " matched nothing; using largest qemu window");
    pool = clients.filter(isQemu);
}

if (pool.length === 0) {
    console.info(token + ":NONE selector=" + selector);
} else {
    let match = pool[0];
    for (let i = 1; i < pool.length; i++) {
        if (area(pool[i]) > area(match)) { match = pool[i]; }
    }
    if (match.minimized) { match.minimized = false; }
    workspace.activeWindow = match;
    // Verify activation actually landed: spectacle -a captures whatever KWin
    // considers active, NOT the window selected here. Without this check a
    // failed/raced activation silently captures a different window.
    const active = workspace.activeWindow;
    const activeId = active ? String(active.internalId) : "null";
    const ok = activeId === String(match.internalId);
    console.info(token + ":" + (ok ? "FOUND" : "ACTIVATEFAIL")
        + " selected=" + describe(match) + " active=" + activeId);
}
EOF

PLUGIN="screenshot-when-kde-plasma-host-${TOKEN}"
KWIN_SCRIPT_ID="$("$QDBUS" org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript "$KWIN_JS" "$PLUGIN" 2>/dev/null || true)"
[[ -n "$KWIN_SCRIPT_ID" && "$KWIN_SCRIPT_ID" =~ ^[0-9]+$ ]] || die "failed to load KWin script (loadScript)"

"$QDBUS" org.kde.KWin "/Scripting/Script${KWIN_SCRIPT_ID}" org.kde.kwin.Script.run >/dev/null 2>&1 \
  || die "failed to run KWin script"

# Pull the script result from the user journal (KWin console.info sink).
ALL_LINES=""
FOUND_LINE=""
for _try in 1 2 3 4 5 6 7 8 9 10; do
  sleep 0.15
  if command -v journalctl >/dev/null 2>&1; then
    # Strip journalctl's timestamp/host/unit prefix; keep the TOKEN:… payload.
    ALL_LINES="$(
      journalctl --user -n 200 --no-pager 2>/dev/null \
        | sed -n "s/.*\(${TOKEN}:.*\)/\1/p" || true
    )"
    # The verdict is the one non-CAND line the script emits last.
    FOUND_LINE="$(printf '%s\n' "$ALL_LINES" | grep -v ":CAND " | tail -1 || true)"
  fi
  [[ -n "$FOUND_LINE" ]] && break
done

CANDIDATES="$(printf '%s\n' "$ALL_LINES" | grep ":CAND " || true)"

if [[ -z "$FOUND_LINE" ]]; then
  die "KWin script produced no result (journalctl empty / delayed); is kwin_wayland logging to the user journal?"
fi
if [[ "$FOUND_LINE" == *":NONE"* ]]; then
  printf '%s: candidates on the compositor:\n' "$SCRIPT_NAME" >&2
  if [[ -n "$CANDIDATES" ]]; then
    printf '%s\n' "$CANDIDATES" >&2
  else
    printf '  (no qemu-class window at all)\n' >&2
  fi
  die "selector matched no window — refusing to capture a different one. \
Re-run with --window <id> from the candidates above (the only selector that \
separates two concurrent VMs), --match <caption-substring> for a non-default \
window, or --any to accept the largest qemu window. \
(A headless/-display none QEMU has no window unless the host-owned window is on.)"
fi
if [[ "$FOUND_LINE" == *":ACTIVATEFAIL"* ]]; then
  die "selected the right window but KWin did not make it active: ${FOUND_LINE}. \
spectacle -a would have captured a DIFFERENT window, so refusing to capture."
fi
if [[ "$FOUND_LINE" != *":FOUND"* ]]; then
  die "unexpected KWin script output: ${FOUND_LINE}"
fi

printf '%s: %s\n' "$SCRIPT_NAME" "$FOUND_LINE" >&2

# Unload before spectacle so we don't leave a script pinned.
"$QDBUS" org.kde.KWin "/Scripting/Script${KWIN_SCRIPT_ID}" org.kde.kwin.Script.stop >/dev/null 2>&1 || true
"$QDBUS" org.kde.KWin /Scripting org.kde.kwin.Scripting.unloadScript "$PLUGIN" >/dev/null 2>&1 || true
KWIN_SCRIPT_ID=""

# Brief settle so the window is active for spectacle -a.
sleep 0.25

# --- capture ------------------------------------------------------------------
if [[ -z "$OUT" ]]; then
  OUT="$(mktemp "${TMPDIR:-/tmp}/qemu-window-XXXXXX.png")"
else
  mkdir -p "$(dirname -- "$OUT")"
fi

SPECTACLE_ARGS=(-b -n -a -o "$OUT")
if [[ "$INCLUDE_DECORATIONS" -eq 0 ]]; then
  SPECTACLE_ARGS+=(-e)
fi

if ! spectacle "${SPECTACLE_ARGS[@]}" >/dev/null 2>&1; then
  die "spectacle failed to capture active window → ${OUT}"
fi

if [[ ! -s "$OUT" ]]; then
  die "capture produced empty/missing file: ${OUT}"
fi

# Basic PNG magic check
magic="$(head -c 8 "$OUT" | od -An -tx1 | tr -d ' \n')"
if [[ "$magic" != "89504e470d0a1a0a" ]]; then
  die "output is not a PNG: ${OUT}"
fi

# --- black-frame guard --------------------------------------------------------
# A valid PNG full of black used to exit 0 and read as a successful capture,
# which is how a capture artifact got mistaken for the device rendering black.
# Treat uniform black as a FAILED CAPTURE unless the caller opts out.
MAGICK=""
for c in magick convert; do
  command -v "$c" >/dev/null 2>&1 && {
    MAGICK="$c"
    break
  }
done

if [[ -n "$MAGICK" ]]; then
  # `-alpha off` is the whole guard. `maxima` and `mean` are taken over every
  # enabled channel, and spectacle writes RGBA with a fully opaque alpha, so on a
  # black frame they read max=255 and mean=63.75 — that is 255/4, the alpha
  # channel and three zeroes. The guard could not fire, and did not: a whole
  # driven boot of a solidly black window was captured, measured "max=255
  # mean=63.7673 colors=30", and passed as a good frame. The frames were used to
  # judge a rail sweep.
  #
  # Discarding alpha first makes both numbers describe the picture. The same
  # black capture then reads max=22 mean=0.02 and is refused.
  STATS="$("$MAGICK" "$OUT" -alpha off \
    -format 'max=%[fx:maxima*255] mean=%[fx:mean*255] colors=%k' info: 2>/dev/null || true)"
  if [[ -n "$STATS" ]]; then
    printf '%s: %s\n' "$SCRIPT_NAME" "$STATS" >&2
    MAXV="${STATS#max=}"
    MAXV="${MAXV%% *}"
    MAXV="${MAXV%%.*}"
    MEANV="${STATS#*mean=}"
    MEANV="${MEANV%% *}"

    # Two arms, because the frame this guard exists to catch does not trip the
    # obvious one. A black *window* capture is not uniformly black: the corners
    # are rounded and anti-aliased against what is behind them, so the real
    # black boot measured max=22 — over the max<=8 threshold, and passed.
    #
    # `mean` is what separates them, and by a wide margin: that black capture
    # reads 0.024 against 94-99 for a composited desktop, and its 717 non-black
    # pixels out of 920 320 are all corner artifacts. The threshold is 0.5, an
    # order of magnitude above the black frame and below the darkest picture
    # worth keeping — a macOS boot screen, black but for the Apple logo, is
    # around 1.0 by the logo's area and brightness alone.
    #
    # The arms report separately so the message says which one fired: "nothing
    # bright anywhere" and "almost nothing lit" are different failures.
    BLACK_WHY=""
    if [[ "$MAXV" =~ ^[0-9]+$ ]] && [[ "$MAXV" -le 8 ]]; then
      BLACK_WHY="max_rgb=${MAXV}, nothing brighter than near-black anywhere"
    elif awk -v m="$MEANV" 'BEGIN{exit !(m+0 < 0.5)}' 2>/dev/null; then
      BLACK_WHY="mean_rgb=${MEANV} with max_rgb=${MAXV}, i.e. black but for a \
handful of edge pixels"
    fi
    if [[ -n "$BLACK_WHY" ]]; then
      if [[ "$ALLOW_BLACK" -eq 1 ]]; then
        printf '%s: WARNING: capture is black (%s); --allow-black set\n' \
          "$SCRIPT_NAME" "$BLACK_WHY" >&2
      else
        die "capture is black (${BLACK_WHY}) → treating as a FAILED CAPTURE, \
not as evidence the guest rendered black. The window was selected and active, so suspect the \
capture path (cross-GPU dmabuf readback, or the window had not composited yet) — or the device, \
which has produced exactly this for a whole boot when guest RAM was being gathered as garbage. \
Corroborate with /tmp/reims-vgpu-fail.log before concluding anything about present correctness. \
Pass --allow-black to keep a black PNG anyway. Wrote: ${OUT}"
      fi
    fi
  fi
else
  printf '%s: WARNING: no magick/convert — black-frame guard SKIPPED\n' "$SCRIPT_NAME" >&2
fi

# --- downscale to ~720p -------------------------------------------------------
# These captures are read by agents, where a 4K PNG costs a large multiple of a
# 720p one in tokens for no added signal at the altitude they get read at. Shrink
# to fit a 1280x720 box; ImageMagick's trailing '>' preserves aspect ratio and
# only ever shrinks, so a window already smaller than 720p is left untouched.
#
# No flag: a lever every caller has to remember to pass is a lever most callers
# will forget, and 720p is the right default for a capture an agent reads.
# REIMS_SHOT_NATIVE=1 in the environment opts out, for the measurements where the
# resample is itself the confound -- a downscale averages sparse per-pixel errors
# into dense speckle and it hides the magnitude distribution the residue and
# colour classes are judged on (AGENTS.md). Environment rather than a flag so an
# existing repro can be re-run at native resolution without editing it.
# Runs AFTER the black-frame guard so the guard's max/mean/colors stats describe
# the pixels actually captured, not resampled ones.
if [[ -n "${REIMS_SHOT_NATIVE:-}" ]]; then
  printf '%s: REIMS_SHOT_NATIVE set — keeping full resolution %s\n' \
    "$SCRIPT_NAME" "$("$MAGICK" "$OUT" -format '%wx%h' info: 2>/dev/null || echo unknown)" >&2
elif [[ -n "$MAGICK" ]]; then
  DIM_BEFORE="$("$MAGICK" "$OUT" -format '%wx%h' info: 2>/dev/null || true)"
  if "$MAGICK" "$OUT" -resize '1280x720>' "$OUT" 2>/dev/null; then
    DIM_AFTER="$("$MAGICK" "$OUT" -format '%wx%h' info: 2>/dev/null || true)"
    printf '%s: size=%s -> %s (720p cap)\n' \
      "$SCRIPT_NAME" "${DIM_BEFORE:-unknown}" "${DIM_AFTER:-unknown}" >&2
  else
    printf '%s: WARNING: 720p downscale failed; keeping full resolution %s\n' \
      "$SCRIPT_NAME" "${DIM_BEFORE:-unknown}" >&2
  fi
else
  printf '%s: WARNING: no magick/convert — 720p downscale SKIPPED\n' "$SCRIPT_NAME" >&2
fi

printf '%s\n' "$OUT"
