#!/bin/bash
# =============================================================================
# trace-memory.sh — dual-node host memory trace for DGX Spark / GB10 UMA
#
# Diagnostic-only. Run independently on each node (head and worker) before
# starting the vLLM/Ray dual-node deployment. Captures high-frequency
# /proc/meminfo, free -m, top-RSS processes, Docker container stats, Ray
# session logs, and (if permitted) kernel log output, so that a host-memory
# freeze can be reconstructed after the fact.
#
# See docs/diagnostics/dgx-spark-uma-memory-freeze.md for the runbook this
# script is part of.
#
# Usage:
#   scripts/diag/trace-memory.sh [options]
#
# Options:
#   --out-dir DIR          Output directory (default: .local/diag/memtrace-<ts>-<host>)
#   --meminfo-interval SEC /proc/meminfo sample interval (default: 0.2)
#   --free-interval SEC    free -m sample interval (default: 1)
#   --ps-interval SEC      top-RSS process snapshot interval (default: 5)
#   --docker-interval SEC  docker stats snapshot interval (default: 5)
#   --no-kernel-log        skip dmesg -w / journalctl -kf background capture
#   --duration SEC         auto-stop after SEC seconds (default: unlimited,
#                            run until Ctrl-C)
#   -h, --help             show this help
#
# Stop with Ctrl-C, or automatically after --duration seconds. On exit, a
# final Ray log snapshot and kernel-log tail (if collected) are copied into
# the output directory, and the directory path is printed.
#
# IMPORTANT: run this script, do not `source`/`.` it. It sets `set -uo
# pipefail`, installs a `trap cleanup INT TERM`, and (with --duration) sends
# SIGTERM to "$$" to stop itself -- if sourced, "$$" is your interactive
# shell's PID and --duration would terminate your shell.
# =============================================================================

# Refuse to run if sourced (see IMPORTANT note above) -- ${BASH_SOURCE[0]}
# differs from $0 only when sourced.
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    echo "[trace-memory] this script must be executed, not sourced." >&2
    echo "[trace-memory] run: ./scripts/diag/trace-memory.sh [options] &" >&2
    return 1 2>/dev/null || exit 1
fi

set -uo pipefail

MEMINFO_INTERVAL="0.2"
FREE_INTERVAL="1"
PS_INTERVAL="5"
DOCKER_INTERVAL="5"
NO_KERNEL_LOG=0
OUT_DIR=""
DURATION=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --meminfo-interval) MEMINFO_INTERVAL="$2"; shift 2 ;;
        --free-interval) FREE_INTERVAL="$2"; shift 2 ;;
        --ps-interval) PS_INTERVAL="$2"; shift 2 ;;
        --docker-interval) DOCKER_INTERVAL="$2"; shift 2 ;;
        --no-kernel-log) NO_KERNEL_LOG=1; shift ;;
        --duration) DURATION="$2"; shift 2 ;;
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
if [ -z "$OUT_DIR" ]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    OUT_DIR="${REPO_ROOT}/.local/diag/memtrace-${TS}-${HOSTNAME_SHORT}"
fi
mkdir -p "$OUT_DIR"

MEMINFO_LOG="${OUT_DIR}/meminfo.log"
FREE_LOG="${OUT_DIR}/free.log"
PS_LOG="${OUT_DIR}/ps_topRSS.log"
DOCKER_LOG="${OUT_DIR}/docker_stats.log"
KERNEL_LOG="${OUT_DIR}/kernel.log"
RAY_LOG_DIR="${OUT_DIR}/ray_logs"

echo "[trace-memory] host=${HOSTNAME_SHORT} output dir: ${OUT_DIR}"
echo "[trace-memory] meminfo every ${MEMINFO_INTERVAL}s, free every ${FREE_INTERVAL}s, ps/docker every ${PS_INTERVAL}/${DOCKER_INTERVAL}s"

PIDS=()

# ---------------------------------------------------------------------------
# /proc/meminfo high-frequency sampler
# ---------------------------------------------------------------------------
{
    echo "# timestamp avail free cached sreclaim shmem dirty writeback unevictable mlocked swaptotal swapfree commitlimit committedas (all kB unless noted)"
    while true; do
        ts="$(date '+%H:%M:%S.%3N')"
        awk -v ts="$ts" '
            /^MemAvailable:/   {a=$2}
            /^MemFree:/        {f=$2}
            /^Cached:/         {c=$2}
            /^SReclaimable:/   {sr=$2}
            /^Shmem:/          {sh=$2}
            /^Dirty:/          {d=$2}
            /^Writeback:/      {wb=$2}
            /^Unevictable:/    {u=$2}
            /^Mlocked:/        {ml=$2}
            /^SwapTotal:/      {st=$2}
            /^SwapFree:/       {sf=$2}
            /^CommitLimit:/    {cl=$2}
            /^Committed_AS:/   {cas=$2}
            END {
                printf "%s avail=%s free=%s cached=%s sreclaim=%s shmem=%s dirty=%s wb=%s unevict=%s mlocked=%s swaptotal=%s swapfree=%s commitlimit=%s committedas=%s\n", \
                    ts,a,f,c,sr,sh,d,wb,u,ml,st,sf,cl,cas
            }
        ' /proc/meminfo
        sleep "$MEMINFO_INTERVAL"
    done
} >> "$MEMINFO_LOG" 2>/dev/null &
PIDS+=("$!")

# ---------------------------------------------------------------------------
# free -m sampler
# ---------------------------------------------------------------------------
{
    while true; do
        echo "=== $(date '+%T.%3N') ==="
        free -m
        sleep "$FREE_INTERVAL"
    done
} >> "$FREE_LOG" 2>/dev/null &
PIDS+=("$!")

# ---------------------------------------------------------------------------
# top-RSS process snapshot
# ---------------------------------------------------------------------------
{
    while true; do
        echo "=== $(date '+%T.%3N') ==="
        ps -eo pid,ppid,comm,rss,shr,stat,wchan:32 --sort=-rss 2>/dev/null | head -50
        sleep "$PS_INTERVAL"
    done
} >> "$PS_LOG" 2>/dev/null &
PIDS+=("$!")

# ---------------------------------------------------------------------------
# Docker container memory stats (best-effort, skipped if docker unavailable)
# ---------------------------------------------------------------------------
if command -v docker >/dev/null 2>&1; then
    {
        while true; do
            echo "=== $(date '+%T.%3N') ==="
            docker stats --no-stream --format \
                'table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}' 2>/dev/null
            sleep "$DOCKER_INTERVAL"
        done
    } >> "$DOCKER_LOG" 2>/dev/null &
    PIDS+=("$!")
else
    echo "[trace-memory] docker not found; skipping docker_stats.log" >> "$OUT_DIR/notes.log"
fi

# ---------------------------------------------------------------------------
# Kernel log (dmesg -w, falling back to journalctl -kf). Best-effort; not all
# environments allow unprivileged dmesg.
# ---------------------------------------------------------------------------
if [ "$NO_KERNEL_LOG" -eq 0 ]; then
    if dmesg -w >> "$KERNEL_LOG" 2>/dev/null &
    then
        PIDS+=("$!")
    elif journalctl -kf >> "$KERNEL_LOG" 2>/dev/null &
    then
        PIDS+=("$!")
    else
        echo "[trace-memory] no permission for dmesg -w / journalctl -kf; skipping kernel.log" >> "$OUT_DIR/notes.log"
    fi
fi

# ---------------------------------------------------------------------------
# Snapshot current Ray session logs (if any), repeated periodically so we
# have a copy even if the host freezes before the run completes.
# ---------------------------------------------------------------------------
snapshot_ray_logs() {
    if [ -d /tmp/ray/session_latest/logs ]; then
        mkdir -p "$RAY_LOG_DIR"
        cp -r /tmp/ray/session_latest/logs/. "$RAY_LOG_DIR/" 2>/dev/null || true
    fi
}
{
    while true; do
        snapshot_ray_logs
        sleep 30
    done
} &
PIDS+=("$!")

# ---------------------------------------------------------------------------
# Cleanup: stop all background loggers, take a final snapshot
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "[trace-memory] stopping..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    snapshot_ray_logs
    {
        echo "=== meminfo (final) ==="
        cat /proc/meminfo
        echo "=== nvidia errors (dmesg, best-effort) ==="
        dmesg 2>/dev/null | grep -iE 'nvidia|nvrm|xid|oom' | tail -50
    } > "${OUT_DIR}/final_snapshot.log" 2>/dev/null
    echo "[trace-memory] output dir: ${OUT_DIR}"
}
# ---------------------------------------------------------------------------
# Optional auto-stop timer. $$ is the top-level script PID even from a
# background subshell, so this triggers the same `trap cleanup TERM` path
# as Ctrl-C.
# ---------------------------------------------------------------------------
if [ -n "$DURATION" ] && [ "$DURATION" != "0" ]; then
    (
        sleep "$DURATION"
        kill -TERM "$$" 2>/dev/null
    ) &
    PIDS+=("$!")
fi

if [ -n "$DURATION" ] && [ "$DURATION" != "0" ]; then
    echo "[trace-memory] tracing started. Will auto-stop after ${DURATION}s (or press Ctrl-C)."
else
    echo "[trace-memory] tracing started. Press Ctrl-C to stop."
fi
wait
