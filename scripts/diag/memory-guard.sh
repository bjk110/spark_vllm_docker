#!/bin/bash
# =============================================================================
# memory-guard.sh — fast emergency host-memory guard for DGX Spark / GB10 UMA
#
# *** THIS IS MITIGATION, NOT A FIX. ***
#
# On GB10's unified host/GPU memory, the observed failure mode is not a
# clean Linux OOM-kill: available memory can collapse from tens of GB to
# ~0 in 15-20 seconds, after which the *host itself* becomes unschedulable
# (ping replies, but sshd cannot be scheduled -- "Connection timed out
# during banner exchange"). A 1s-interval watchdog can lose this race
# entirely. This script polls /proc/meminfo at sub-second intervals and
# tries to kill the vLLM/Ray containers *before* that point, converting a
# host freeze into a logged container failure -- but if the decline is
# steep enough, the host can still become unschedulable before this
# script's kill command is scheduled. Treat a successful kill as a data
# point, not as proof the underlying issue is fixed.
#
# See docs/diagnostics/dgx-spark-uma-memory-freeze.md for the runbook this
# script is part of.
#
# Usage:
#   scripts/diag/memory-guard.sh [options]
#
# Options:
#   --threshold-mb MB   Trip threshold for MemAvailable (default: 4096)
#   --interval SEC      Poll interval in seconds (default: 0.1)
#   --pattern REGEX     Fallback container-name regex if known names are not
#                        running (default: "vllm|ray|spark")
#   --containers LIST   Comma-separated known container names to target first
#                        (default: vllm-spark-head,vllm-spark-worker)
#   --log FILE          Log file (default: .local/diag/memory-guard-<ts>-<host>.log)
#   --dry-run           Log trips but never kill; keeps polling after a trip
#   -h, --help          show this help
#
# Exit: on a real (non-dry-run) trip, kills matching containers and exits.
# Stop early with Ctrl-C.
# =============================================================================
set -uo pipefail

THRESHOLD_MB=4096
INTERVAL=0.1
PATTERN="vllm|ray|spark"
CONTAINERS="vllm-spark-head,vllm-spark-worker"
LOG=""
DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --threshold-mb) THRESHOLD_MB="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        --containers) CONTAINERS="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            # Print everything between the first and second "# ====...="
            # delimiter lines (the full header block), regardless of how
            # many lines it grows to -- avoids hardcoded-line-range
            # truncation if the header is edited later.
            awk '/^# =====/{c++; next} c==1' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
TS="$(date '+%Y%m%d-%H%M%S')"
if [ -z "$LOG" ]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    LOG_DIR="${REPO_ROOT}/.local/diag"
    mkdir -p "$LOG_DIR"
    # Fresh filename per run (timestamp+host) -- avoids the stale
    # root-owned log file issue seen with the older watchdog.sh, where a
    # log left behind by a killed-as-root process blocked subsequent
    # appends and silently broke the kill pipeline.
    LOG="${LOG_DIR}/memory-guard-${TS}-${HOSTNAME_SHORT}.log"
fi

log() {
    echo "$(date '+%T.%3N') $*" | tee -a "$LOG"
}

log "memory-guard start host=${HOSTNAME_SHORT} threshold=${THRESHOLD_MB}MB interval=${INTERVAL}s dry_run=${DRY_RUN}"
log "WARNING: this is mitigation only. A steep enough decline can make the"
log "WARNING: host unschedulable before the kill command below runs."

mem_available_mb() {
    awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo
}

kill_containers() {
    if ! command -v docker >/dev/null 2>&1; then
        log "docker not found; skipping container kill"
        return
    fi
    local killed_any=0
    IFS=',' read -ra known <<< "$CONTAINERS"
    for name in "${known[@]}"; do
        name="$(echo "$name" | xargs)"
        [ -z "$name" ] && continue
        if docker ps -q --filter "name=^${name}\$" 2>/dev/null | grep -q .; then
            log "killing known container: ${name}"
            docker kill "$name" >> "$LOG" 2>&1
            killed_any=1
        fi
    done
    if [ "$killed_any" -eq 0 ]; then
        log "no known containers running; falling back to --pattern '${PATTERN}'"
        local matches
        matches="$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E "$PATTERN" || true)"
        if [ -n "$matches" ]; then
            echo "$matches" | while read -r name; do
                log "killing matched container: ${name}"
                docker kill "$name" >> "$LOG" 2>&1
            done
            killed_any=1
        fi
    fi
    if [ "$killed_any" -eq 0 ]; then
        log "no matching containers found to kill"
    fi
}

# Best-effort: ask Ray to tear down its local node processes (GCS, raylet,
# object store, dashboard) after the containers are gone. This is a slower,
# more graceful step and is NOT relied upon to free memory in time -- the
# docker kill above is the primary action. Safe no-op if `ray` is not on
# PATH (e.g. run from outside the container) or no Ray session is running.
ray_stop_force() {
    if ! command -v ray >/dev/null 2>&1; then
        log "ray CLI not found; skipping ray stop --force"
        return
    fi
    log "attempting ray stop --force"
    ray stop --force >> "$LOG" 2>&1 || log "ray stop --force failed/no-op (continuing)"
}

snapshot_meminfo() {
    local label="$1"
    {
        echo "--- /proc/meminfo @ ${label} ---"
        cat /proc/meminfo
    } >> "$LOG" 2>/dev/null
}

log "polling MemAvailable (threshold=${THRESHOLD_MB}MB)..."
while true; do
    avail="$(mem_available_mb)"
    if [ -z "$avail" ]; then
        avail=999999
    fi
    if [ "$avail" -lt "$THRESHOLD_MB" ]; then
        log "!!! TRIP avail=${avail}MB < threshold=${THRESHOLD_MB}MB"
        snapshot_meminfo "trip"
        if [ "$DRY_RUN" -eq 1 ]; then
            log "dry-run: NOT killing containers, continuing to poll"
        else
            # Emergency order: docker kill (fast) first, then best-effort
            # ray stop --force, then a post-kill meminfo snapshot so the
            # trace shows whether memory was actually reclaimed.
            kill_containers
            ray_stop_force
            snapshot_meminfo "post-kill"
            log "kill attempted. memory-guard exiting."
            break
        fi
    fi
    sleep "$INTERVAL"
done
