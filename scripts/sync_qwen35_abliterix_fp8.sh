#!/usr/bin/env bash
# =============================================================================
# sync_qwen35_abliterix_fp8.sh
#
# Distributes the FP8 quantization output from spark01 to:
#   - spark02 over RDMA (10.10.10.2)
#   - homeserver back-up (so spark01's local copy can be optionally cleared)
#
# Then optionally deletes the original BF16 on spark01 to reclaim disk
# (toggle with DELETE_BF16_ON_SPARK01=1).
#
# Run on: homeserver (orchestrates), or spark01 (skip the homeserver pull).
# =============================================================================
set -euo pipefail

SSH_USER="${SSH_USER:-bjk110}"

SPARK01_SRC="/home/bjk110/Documents/Models/wangzhang/Qwen3.5-122B-A10B-abliterix-FP8"
SPARK02_DST="${SPARK01_SRC}"
HOMESERVER_DST="/mnt/data/llm-models/wangzhang/Qwen3.5-122B-A10B-abliterix-FP8"

SPARK01_MGMT="192.168.0.200"
SPARK01_RDMA="10.10.10.1"
SPARK02_MGMT="192.168.0.201"
SPARK02_RDMA="10.10.10.2"

DELETE_BF16_ON_SPARK01="${DELETE_BF16_ON_SPARK01:-0}"
BF16_PATH_ON_SPARK01="/home/bjk110/Documents/Models/wangzhang/Qwen3.5-122B-A10B-abliterix"

RSYNC_OPTS=(-avh --info=progress2 --partial --append-verify)

echo "[sync-fp8] === Pre-flight ==="
SPARK01_SIZE=$(ssh -n "${SSH_USER}@${SPARK01_MGMT}" "du -sh '${SPARK01_SRC}' 2>/dev/null | awk '{print \$1}'" || echo "MISSING")
echo "[sync-fp8] spark01 FP8 size: ${SPARK01_SIZE}"
if [[ "${SPARK01_SIZE}" == "MISSING" ]]; then
    echo "[sync-fp8] ERROR: FP8 output not found on spark01 at ${SPARK01_SRC}" >&2
    echo "[sync-fp8]         Run scripts/quantize_qwen35_abliterix_fp8.sh on spark01 first." >&2
    exit 1
fi

echo ""
echo "[sync-fp8] === Stage 1: spark01 → spark02 (RDMA) ==="
# Run from spark01 via ssh (inner ssh uses -n to avoid heredoc stdin consumption).
ssh "${SSH_USER}@${SPARK01_MGMT}" bash -s <<REMOTE
set -euo pipefail
ssh -n -o StrictHostKeyChecking=accept-new "${SSH_USER}@${SPARK02_RDMA}" "mkdir -p '${SPARK02_DST}'"
rsync ${RSYNC_OPTS[@]} '${SPARK01_SRC%/}/' "${SSH_USER}@${SPARK02_RDMA}:${SPARK02_DST%/}/"
REMOTE
echo "[sync-fp8] spark02 receive size: $(ssh -n "${SSH_USER}@${SPARK01_MGMT}" ssh -o StrictHostKeyChecking=accept-new "${SSH_USER}@${SPARK02_RDMA}" "du -sh '${SPARK02_DST}'" | awk '{print $1}')"

echo ""
echo "[sync-fp8] === Stage 2: spark01 → homeserver (pull, mgmt) ==="
mkdir -p "${HOMESERVER_DST}"
rsync "${RSYNC_OPTS[@]}" "${SSH_USER}@${SPARK01_MGMT}:${SPARK01_SRC%/}/" "${HOMESERVER_DST%/}/"
echo "[sync-fp8] homeserver backup size: $(du -sh "${HOMESERVER_DST}" | awk '{print $1}')"

if [[ "${DELETE_BF16_ON_SPARK01}" == "1" ]]; then
    echo ""
    echo "[sync-fp8] === Stage 3: delete BF16 on spark01 (DELETE_BF16_ON_SPARK01=1) ==="
    ssh -n "${SSH_USER}@${SPARK01_MGMT}" "rm -rf '${BF16_PATH_ON_SPARK01}'; df -h /home/bjk110/Documents | tail -1"
fi

echo ""
echo "[sync-fp8] === Done ==="
echo "[sync-fp8] FP8 now on spark01 + spark02 + homeserver backup."
echo "[sync-fp8] Next: bash scripts/start_qwen35_abliterix_tp2.sh"
