#!/usr/bin/env bash
# =============================================================================
# download_qwen35_abliterix.sh
#
# Downloads wangzhang/Qwen3.5-122B-A10B-abliterix (BF16, ~244 GB) to homeserver
# at /mnt/data/llm-models/wangzhang/Qwen3.5-122B-A10B-abliterix.
#
# Repo is gated (auto-approve). Requires HF token at ~/.cache/huggingface/token.
#
# Run on: homeserver only.
# =============================================================================
set -euo pipefail

REPO="wangzhang/Qwen3.5-122B-A10B-abliterix"
DEST="/mnt/data/llm-models/wangzhang/Qwen3.5-122B-A10B-abliterix"
LOG="/tmp/abliterix_download.log"

if [[ ! -f "${HOME}/.cache/huggingface/token" ]]; then
    echo "[download] ERROR: HF token missing at ~/.cache/huggingface/token" >&2
    echo "[download]        Run: hf auth login" >&2
    exit 1
fi

mkdir -p "${DEST}"
echo "[download] target: ${DEST}"
echo "[download] log:    ${LOG}"
echo "[download] starting hf download (BF16 ~244 GB) ..."

# --local-dir keeps files outside the symlinked cache so rsync can transfer
# them directly. hf 1.11+ copies (no symlinks) by default when --local-dir is set.
hf download "${REPO}" \
    --local-dir "${DEST}" \
    --max-workers 8 \
    2>&1 | tee -a "${LOG}"

echo "[download] === Done ==="
du -sh "${DEST}"
