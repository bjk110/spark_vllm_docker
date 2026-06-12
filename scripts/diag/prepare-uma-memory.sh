#!/bin/bash
# =============================================================================
# prepare-uma-memory.sh — host memory headroom prep for DGX Spark / GB10 UMA
#
# *** DIAGNOSTIC MITIGATION, NOT A CORRECTNESS FIX. ***
#
# GB10 has no discrete VRAM: CPU and GPU allocations share one 121.63 GiB
# unified memory pool. The observed failure (rapid MemAvailable collapse
# during EP/MoE setup, host becoming unresponsive to SSH) may be made less
# likely, or shifted to an earlier/more diagnosable point, by reserving a
# larger kernel "do not touch" headroom (vm.min_free_kbytes) and reducing
# page-cache/swap pressure before launch. This does NOT address the root
# cause of the memory growth itself.
#
# Usage:
#   scripts/diag/prepare-uma-memory.sh [options]
#
# Options:
#   --apply              Actually change sysctls (requires root). Without
#                          this flag, the script only prints current values.
#   --min-free-kbytes N  vm.min_free_kbytes to set in --apply mode
#                          (default: 6291456, i.e. ~6 GiB)
#   --no-drop-caches     In --apply mode, skip `echo 3 > /proc/sys/vm/drop_caches`
#   -h, --help           show this help
# =============================================================================
set -uo pipefail

APPLY=0
MIN_FREE_KB=6291456
DROP_CACHES=1

while [ "$#" -gt 0 ]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --min-free-kbytes) MIN_FREE_KB="$2"; shift 2 ;;
        --no-drop-caches) DROP_CACHES=0; shift ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

show_current() {
    echo "--- current sysctl values ---"
    for key in vm.min_free_kbytes vm.swappiness vm.overcommit_memory vm.overcommit_ratio; do
        printf '%-24s = %s\n' "$key" "$(sysctl -n "$key" 2>/dev/null || echo '?')"
    done
}

show_current

if [ "$APPLY" -eq 0 ]; then
    echo ""
    echo "Dry run (no changes made). Re-run with --apply (as root/sudo) to set:"
    echo "  vm.min_free_kbytes = ${MIN_FREE_KB}"
    echo "  vm.swappiness      = 1"
    if [ "$DROP_CACHES" -eq 1 ]; then
        echo "  + sync && echo 3 > /proc/sys/vm/drop_caches"
    fi
    exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: --apply requires root (sudo)." >&2
    exit 1
fi

echo ""
echo "--- applying ---"
sysctl -w vm.min_free_kbytes="${MIN_FREE_KB}"
sysctl -w vm.swappiness=1

echo "sync..."
sync

if [ "$DROP_CACHES" -eq 1 ]; then
    echo "dropping page cache (echo 3 > /proc/sys/vm/drop_caches)..."
    echo 3 > /proc/sys/vm/drop_caches
else
    echo "skipping drop_caches (--no-drop-caches)"
fi

echo ""
show_current
echo ""
echo "NOTE: these settings are not persisted across reboot unless added to"
echo "/etc/sysctl.conf or /etc/sysctl.d/. This script intentionally does not"
echo "write to those files -- treat this as a per-attempt diagnostic prep step."
