#!/usr/bin/env bash
# =============================================================================
# sync_qwen35_abliterix_bf16_to_spark01.sh
#
# One-way rsync of BF16 source from homeserver to spark01 (quant target).
# Spark02 does NOT receive the BF16 — only the FP8 quantized output.
#
# Run on: homeserver only.
# =============================================================================
set -euo pipefail

SSH_USER="${SSH_USER:-bjk110}"
HOMESERVER_SRC="/mnt/data/llm-models/wangzhang/Qwen3.5-122B-A10B-abliterix"
SPARK01_DST="/home/bjk110/Documents/Models/wangzhang/Qwen3.5-122B-A10B-abliterix"
SPARK01_MGMT="192.168.0.200"

RSYNC_OPTS=(-avh --info=progress2 --partial --append-verify)

if [[ ! -d "${HOMESERVER_SRC}" ]]; then
    echo "[sync-bf16] ERROR: source missing on homeserver: ${HOMESERVER_SRC}" >&2
    echo "[sync-bf16]          Run scripts/download_qwen35_abliterix.sh first." >&2
    exit 1
fi

echo "[sync-bf16] === Pre-flight ==="
echo "[sync-bf16] source size: $(du -sh "${HOMESERVER_SRC}" | awk '{print $1}')"

echo ""
echo "[sync-bf16] === homeserver → spark01 (mgmt) ==="
ssh -n "${SSH_USER}@${SPARK01_MGMT}" "mkdir -p '${SPARK01_DST}'"
rsync "${RSYNC_OPTS[@]}" "${HOMESERVER_SRC%/}/" "${SSH_USER}@${SPARK01_MGMT}:${SPARK01_DST%/}/"
echo "[sync-bf16] spark01 receive size: $(ssh -n "${SSH_USER}@${SPARK01_MGMT}" "du -sh '${SPARK01_DST}'" | awk '{print $1}')"

echo ""
echo "[sync-bf16] === Done ==="
echo "[sync-bf16] Next: ssh to spark01 and run scripts/quantize_qwen35_abliterix_fp8.sh"
