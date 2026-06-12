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
#   --proc-maps-interval SEC
#                          Normal-mode interval for the per-process
#                          /proc/<pid>/{status,smaps_rollup,limits,
#                          numa_maps,maps} sampler (default: 2)
#   --proc-maps-burst-interval SEC
#                          Faster interval used once MemAvailable drops below
#                          --proc-maps-threshold-mb (default: 0.5)
#   --proc-maps-threshold-mb MB
#                          MemAvailable threshold (MB) below which the
#                          proc_maps sampler switches to burst mode
#                          (default: 32768)
#   --proc-maps-pattern REGEX
#                          Extended-regex matched against `ps -eo
#                          pid,rss,comm,args` to pick candidate processes
#                          (default: "RayWorkerWrapper|EngineCore|vllm|python")
#   --proc-maps-max-pids N Max number of candidate PIDs sampled per tick,
#                          highest-RSS first (default: 8)
#   --proc-maps-max-snapshots N
#                          Rotation cap: number of timestamped snapshots kept
#                          per PID per file kind (default: 20)
#   --no-proc-maps         disable the proc_maps sampler entirely
#   --duration SEC         auto-stop after SEC seconds (default: unlimited,
#                            run until Ctrl-C)
#   -h, --help             show this help
#
# Stop with Ctrl-C, or automatically after --duration seconds. On exit, a
# final Ray log snapshot and kernel-log tail (if collected) are copied into
# the output directory, and the directory path is printed.
#
# PERMISSIONS / sudo: meminfo, free, docker stats, and top-RSS sampling work
# fine as a non-root user. However, the proc_maps sampler's
# smaps_rollup/maps/numa_maps capture for root-owned container processes
# (e.g. vLLM/Ray processes inside a Docker container, typically UID 0) is
# only readable by root or a process with CAP_SYS_PTRACE. If this script is
# run as a non-root user, those three files will be unreadable for UID-0
# PIDs; this is logged to proc_maps/notes.log (one line per unreadable path)
# rather than silently producing 0-byte files. Run with `sudo` if you need
# smaps_rollup/maps/numa_maps data for root-owned processes.
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
PROC_MAPS_INTERVAL="2"
PROC_MAPS_BURST_INTERVAL="0.5"
PROC_MAPS_THRESHOLD_MB="32768"
PROC_MAPS_PATTERN="RayWorkerWrapper|EngineCore|vllm|python"
PROC_MAPS_MAX_PIDS="8"
PROC_MAPS_MAX_SNAPSHOTS="20"
NO_PROC_MAPS=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --meminfo-interval) MEMINFO_INTERVAL="$2"; shift 2 ;;
        --free-interval) FREE_INTERVAL="$2"; shift 2 ;;
        --ps-interval) PS_INTERVAL="$2"; shift 2 ;;
        --docker-interval) DOCKER_INTERVAL="$2"; shift 2 ;;
        --no-kernel-log) NO_KERNEL_LOG=1; shift ;;
        --proc-maps-interval) PROC_MAPS_INTERVAL="$2"; shift 2 ;;
        --proc-maps-burst-interval) PROC_MAPS_BURST_INTERVAL="$2"; shift 2 ;;
        --proc-maps-threshold-mb) PROC_MAPS_THRESHOLD_MB="$2"; shift 2 ;;
        --proc-maps-pattern) PROC_MAPS_PATTERN="$2"; shift 2 ;;
        --proc-maps-max-pids) PROC_MAPS_MAX_PIDS="$2"; shift 2 ;;
        --proc-maps-max-snapshots) PROC_MAPS_MAX_SNAPSHOTS="$2"; shift 2 ;;
        --no-proc-maps) NO_PROC_MAPS=1; shift ;;
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
PROC_MAPS_DIR="${OUT_DIR}/proc_maps"

echo "[trace-memory] host=${HOSTNAME_SHORT} output dir: ${OUT_DIR}"
echo "[trace-memory] meminfo every ${MEMINFO_INTERVAL}s, free every ${FREE_INTERVAL}s, ps/docker every ${PS_INTERVAL}/${DOCKER_INTERVAL}s"

PIDS=()

# ---------------------------------------------------------------------------
# Helpers shared by the proc_maps sampler below
# ---------------------------------------------------------------------------
mem_available_mb() {
    awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null
}

# Extract just the smaps_rollup fields useful for UMA debugging.
extract_smaps_rollup() {
    local src="$1" dst="$2"
    grep -E '^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty|Anonymous|Locked|Swap):' \
        "$src" > "$dst" 2>/dev/null
}

# Summarize /proc/<pid>/maps into per-category mapping counts and total
# sizes (KB), instead of copying the (potentially huge) raw file. Categories:
# deleted files, /dev/nvidia*, /dev/infiniband*, /dev/shm, memfd, CUDA/NCCL
# shared libs, anonymous (no backing path), other.
summarize_maps_file() {
    local src="$1" dst="$2"
    awk '
        {
            split($1, range, "-")
            start = strtonum("0x" range[1])
            end   = strtonum("0x" range[2])
            size_kb = (end - start) / 1024

            cat = "other"
            if ($0 ~ /\(deleted\)/)                          cat = "deleted"
            else if ($0 ~ /\/dev\/nvidia/)                    cat = "nvidia_dev"
            else if ($0 ~ /\/dev\/infiniband/)                cat = "infiniband_dev"
            else if ($0 ~ /\/dev\/shm/)                       cat = "dev_shm"
            else if ($0 ~ /memfd:/)                           cat = "memfd"
            else if ($0 ~ /lib(cuda|nccl|cudart|nvidia)/)     cat = "cuda_nccl_lib"
            else if (NF < 6)                                  cat = "anonymous"

            count[cat]++
            size[cat] += size_kb
        }
        END {
            for (c in count) printf "%-16s count=%-6d size_kb=%d\n", c, count[c], size[c]
        }
    ' "$src" > "$dst" 2>/dev/null
}

# Capture one /proc/<pid>/<kind> file into $dst, dispatching to the right
# extractor. bash's `-r` test (access(R_OK)) can report /proc/<pid>/* as
# readable by file-mode bits even when the kernel's ptrace_scope / Yama LSM
# check makes the actual read() return EACCES for a UID-0 process read by a
# non-root user -- in that case `cp`/`grep`/`awk` silently produce an empty
# file with `2>/dev/null`. So after the capture attempt, check whether $dst
# actually has content; if not, log a permission note (with a sudo hint) to
# notes.log and remove the empty placeholder instead of leaving a silent
# 0-byte file.
capture_proc_file() {
    local kind="$1" src="$2" dst="$3" pid="$4" ts="$5"
    if [ -r "$src" ]; then
        case "$kind" in
            smaps_rollup) extract_smaps_rollup "$src" "$dst" ;;
            maps_summary) summarize_maps_file "$src" "$dst" ;;
            numa_maps)    head -c 1048576 "$src" > "$dst" 2>/dev/null ;;
            *)            cp "$src" "$dst" 2>/dev/null ;;
        esac
    fi
    if [ ! -s "$dst" ]; then
        rm -f "$dst"
        echo "${ts} pid=${pid} ${kind}: permission denied or unreadable (${src}, PID ${pid}, uid mismatch? try sudo)" >> "${OUT_DIR}/notes.log"
    fi
}

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
        # NOTE: "shr" is not a valid -o field on procps-ng 4.0.4 ("unknown
        # user-defined format specifier") and silently produced empty
        # snapshots when stderr was discarded. Use vsz instead, and merge
        # stderr into PS_LOG (no 2>/dev/null) so a future field error is
        # visible in the log rather than producing silent empty snapshots.
        ps -eo pid,ppid,user,comm,rss,vsz,stat,wchan:32 --sort=-rss | head -50
        sleep "$PS_INTERVAL"
    done
} >> "$PS_LOG" 2>&1 &
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
# Per-process /proc/<pid> sampler (status, smaps_rollup, limits, numa_maps,
# maps summary) for candidate RayWorkerWrapper/vLLM/EngineCore/python
# processes. Runs in the host PID namespace (this script is not containerized)
# so PIDs recorded here are host-namespace PIDs. Goal: capture per-process
# memory-mapping detail BEFORE Ray's memory monitor (or the host) kills these
# processes during a collapse.
#
# Lightweight by design (no `cp -r`, no raw /proc/<pid>/maps dumps -- maps and
# numa_maps are summarized/truncated) so the sampler itself does not
# meaningfully add to memory pressure during a collapse.
# ---------------------------------------------------------------------------
if [ "$NO_PROC_MAPS" -eq 0 ]; then
    mkdir -p "$PROC_MAPS_DIR"
    {
        while true; do
            avail_mb="$(mem_available_mb)"
            if [ -n "$avail_mb" ] && [ "$avail_mb" -lt "$PROC_MAPS_THRESHOLD_MB" ]; then
                interval="$PROC_MAPS_BURST_INTERVAL"
                mode="burst"
            else
                interval="$PROC_MAPS_INTERVAL"
                mode="normal"
            fi
            ts="$(date '+%H%M%S.%3N')"

            mapfile -t pids < <(
                ps -eo pid,rss,comm,args --no-headers 2>/dev/null \
                    | grep -E -- "$PROC_MAPS_PATTERN" \
                    | sort -k2 -rn \
                    | head -n "$PROC_MAPS_MAX_PIDS" \
                    | awk '{print $1}'
            )

            for pid in "${pids[@]}"; do
                [ -d "/proc/$pid" ] || continue
                comm="$(tr -d '\0\n' < "/proc/$pid/comm" 2>/dev/null || echo unknown)"
                comm="${comm//\//_}"
                pid_dir="${PROC_MAPS_DIR}/pid_${pid}_${comm}"
                mkdir -p "$pid_dir" 2>/dev/null

                capture_proc_file status        "/proc/$pid/status"       "${pid_dir}/status-${ts}.log"       "$pid" "$ts"
                capture_proc_file smaps_rollup   "/proc/$pid/smaps_rollup" "${pid_dir}/smaps_rollup-${ts}.log" "$pid" "$ts"
                capture_proc_file limits         "/proc/$pid/limits"       "${pid_dir}/limits-${ts}.log"       "$pid" "$ts"
                capture_proc_file numa_maps      "/proc/$pid/numa_maps"    "${pid_dir}/numa_maps-${ts}.log"    "$pid" "$ts"
                capture_proc_file maps_summary   "/proc/$pid/maps"         "${pid_dir}/maps_summary-${ts}.log" "$pid" "$ts"

                # Rotation: keep only the newest PROC_MAPS_MAX_SNAPSHOTS
                # snapshots per file kind for this PID.
                for kind in status smaps_rollup limits numa_maps maps_summary; do
                    ls -1t "${pid_dir}/${kind}-"*.log 2>/dev/null \
                        | tail -n "+$((PROC_MAPS_MAX_SNAPSHOTS + 1))" \
                        | xargs -r rm -f
                done
            done

            echo "${ts} mode=${mode} avail_mb=${avail_mb:-NA} candidates=${#pids[@]} pids=${pids[*]:-}" >> "${PROC_MAPS_DIR}/sampler.log"
            sleep "$interval"
        done
    } >> "${OUT_DIR}/notes.log" 2>&1 &
    PIDS+=("$!")
else
    echo "[trace-memory] proc_maps sampler disabled (--no-proc-maps)" >> "$OUT_DIR/notes.log"
fi

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
# Register the trap referenced by the header comment and the auto-stop timer
# below -- without this, SIGTERM/SIGINT terminate the main script via bash's
# default disposition without running cleanup(), leaving the background
# loggers as orphaned processes and skipping final_snapshot.log.
trap cleanup INT TERM
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
