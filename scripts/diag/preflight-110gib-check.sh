#!/bin/bash
# =============================================================================
# preflight-110gib-check.sh — path-specific 110 GiB preflight for
#   Step-3.7-Flash-NVFP4 v022 EP-on / mp-backend long-context path.
#
# WHY THIS EXISTS:
#   GB10 UMA (121.63 GiB shared) means VLLM_SKIP_INIT_MEMORY_CHECK=1 only
#   bypasses a guard check — it does NOT recover memory.  Running the server
#   when UMA is < 110 GiB free causes the profiling spike (~107 GiB peak) to
#   exhaust the pool, triggering kernel page-thrash and an unresponsive node.
#   Reboot is the only recovery.  This check must PASS before starting the
#   server with the memcheck-bypass image on this path.  It is a pre-start
#   gate only: do NOT run it while the server is loaded — serving-state
#   MemAvailable of ~12–15 GiB is normal and expected after model load.
#
#   Threshold derivation:
#     GPU_MEMORY_UTILIZATION=0.88 × 121.63 GiB = 107.03 GiB desired
#     Observed clean-boot MemAvailable: ~113.8–118 GiB
#     Gate: 110 GiB (5% headroom above 107 GiB requirement)
#
# Usage:
#   scripts/diag/preflight-110gib-check.sh [--threshold-gib N]
#                                           [--head HOST] [--worker HOST]
#
# Options:
#   --threshold-gib N   Required MemAvailable per node in GiB (default: 110)
#   --head HOST         Head node hostname/alias (default: spark01)
#   --worker HOST       Worker node hostname/alias (default: spark02)
#   -h, --help          Show this help
#
# Exit:
#   0 — both nodes pass (≥ threshold GiB)
#   1 — one or both nodes fail; prints which and how much is available
# =============================================================================
set -uo pipefail

THRESHOLD_GIB=110
HEAD_HOST=spark01
WORKER_HOST=spark02

while [ "$#" -gt 0 ]; do
    case "$1" in
        --threshold-gib) THRESHOLD_GIB="$2"; shift 2 ;;
        --head)          HEAD_HOST="$2";      shift 2 ;;
        --worker)        WORKER_HOST="$2";    shift 2 ;;
        -h|--help)
            awk '/^# =====/{c++; next} c==1' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# kB threshold: N GiB × 1048576
THRESHOLD_KB=$(( THRESHOLD_GIB * 1048576 ))

get_memavailable_kb() {
    local host="$1"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
        "awk '/^MemAvailable:/ {print \$2}' /proc/meminfo" 2>/dev/null
}

kb_to_gib_str() {
    local kb="$1"
    awk -v k="$kb" 'BEGIN { printf "%.1f GiB", k/1048576 }'
}

overall_pass=0

check_node() {
    local role="$1"
    local host="$2"
    local avail_kb
    avail_kb="$(get_memavailable_kb "$host")" || true
    if [ -z "$avail_kb" ]; then
        echo "  [FAIL] ${role} (${host}): unreachable or no MemAvailable"
        overall_pass=1
        return
    fi
    local avail_str
    avail_str="$(kb_to_gib_str "$avail_kb")"
    local thresh_str
    thresh_str="$(kb_to_gib_str "$THRESHOLD_KB")"
    if [ "$avail_kb" -ge "$THRESHOLD_KB" ]; then
        echo "  [PASS] ${role} (${host}): ${avail_str} available  (≥ ${thresh_str} required)"
    else
        echo "  [FAIL] ${role} (${host}): ${avail_str} available  (< ${thresh_str} required)"
        echo "         --> Reboot ${host} to recover GB10 UMA before starting the server."
        overall_pass=1
    fi
}

echo "=== Preflight: 110 GiB MemAvailable check (v022 EP-on/mp long-context path) ==="
echo "    Threshold: ${THRESHOLD_GIB} GiB per node"
echo ""
check_node "head  " "$HEAD_HOST"
check_node "worker" "$WORKER_HOST"
echo ""

if [ "$overall_pass" -eq 0 ]; then
    echo "=== PASS — safe to start server with VLLM_SKIP_INIT_MEMORY_CHECK=1 ==="
    exit 0
else
    echo "=== FAIL — do NOT start server; reboot failing node(s) first ==="
    exit 1
fi
