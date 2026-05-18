#!/usr/bin/env bash
# =============================================================================
# sync_qwen36_base_to_sparks.sh
#
# Two-stage rsync for Qwen/Qwen3.6-27B base BF16:
#   Stage 1: homeserver  → spark01   via mgmt LAN 192.168.0.200
#   Stage 2: spark01     → spark02   via RDMA   10.10.10.2
#
# Inner ssh uses -n to avoid swallowing the parent heredoc's stdin (which
# is what made sync_qwen36_prismascout_to_sparks.sh's first run silently
# no-op Stage 2 — the bug was patched there and applied here from the start).
#
# Run on: homeserver only.
# =============================================================================
set -euo pipefail

SSH_USER="${SSH_USER:-bjk110}"

HOMESERVER_SRC="/mnt/data/llm-models/Qwen/Qwen3.6-27B"
SPARK01_DST="/home/bjk110/Documents/Models/Qwen/Qwen3.6-27B"
SPARK02_DST="${SPARK01_DST}"

SPARK01_MGMT="192.168.0.200"
SPARK02_RDMA="10.10.10.2"

RSYNC_OPTS=(-avh --info=progress2 --partial --append-verify)

SRC_TRAIL="${HOMESERVER_SRC%/}/"
DST_TRAIL_01="${SPARK01_DST%/}/"
DST_TRAIL_02="${SPARK02_DST%/}/"

if [[ ! -d "${HOMESERVER_SRC}" ]]; then
    echo "[sync-base] ERROR: source missing on homeserver: ${HOMESERVER_SRC}" >&2
    echo "[sync-base]        Run scripts/download_qwen36_base.sh first." >&2
    exit 1
fi

echo "[sync-base] === Pre-flight ==="
echo "[sync-base] homeserver source size: $(du -sh "${HOMESERVER_SRC}" | awk '{print $1}')"

echo ""
echo "[sync-base] === Stage 1: homeserver → spark01 (mgmt) ==="
ssh -n "${SSH_USER}@${SPARK01_MGMT}" "mkdir -p '${SPARK01_DST}'"
rsync "${RSYNC_OPTS[@]}" "${SRC_TRAIL}" "${SSH_USER}@${SPARK01_MGMT}:${DST_TRAIL_01}"
echo "[sync-base] spark01 receive size: $(ssh -n "${SSH_USER}@${SPARK01_MGMT}" "du -sh '${SPARK01_DST}'" | awk '{print $1}')"

echo ""
echo "[sync-base] === Stage 2: spark01 → spark02 (RDMA) ==="
ssh "${SSH_USER}@${SPARK01_MGMT}" bash -s <<REMOTE
set -euo pipefail
mkdir -p '${SPARK01_DST}'
ssh -n -o StrictHostKeyChecking=accept-new "${SSH_USER}@${SPARK02_RDMA}" "mkdir -p '${SPARK02_DST}'"
rsync ${RSYNC_OPTS[@]} \
    '${DST_TRAIL_01}' \
    "${SSH_USER}@${SPARK02_RDMA}:${DST_TRAIL_02}"
REMOTE

echo "[sync-base] spark02 receive size: $(ssh -n "${SSH_USER}@${SPARK01_MGMT}" ssh -o StrictHostKeyChecking=accept-new "${SSH_USER}@${SPARK02_RDMA}" "du -sh '${SPARK02_DST}'" | awk '{print $1}')"

echo ""
echo "[sync-base] === Done ==="
echo "[sync-base] Base model now on spark01 and spark02 at:"
echo "[sync-base]   ${SPARK01_DST}"
echo "[sync-base] Next: PRESET=qwen3.6-27b-base-bf16-tp2 bash scripts/start_qwen36_prismascout_tp2.sh"
