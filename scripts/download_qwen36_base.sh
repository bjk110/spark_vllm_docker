#!/usr/bin/env bash
# =============================================================================
# download_qwen36_base.sh
#
# Downloads the BASE BF16 model Qwen/Qwen3.6-27B (~54 GB, 15 shards) to
# homeserver at:
#   /mnt/data/llm-models/Qwen/Qwen3.6-27B
#
# This is the source of the PrismaSCOUT NVFP4-BF16 variant; we fall back to
# the base because PrismaSCOUT's NVFP4 quantization is incompatible with
# TP=2 on this model (vision intermediate_size=4304 splits to 2152 which
# violates the FP4 16-multiple alignment).
#
# Run on: homeserver only.
# =============================================================================
set -euo pipefail

MODEL_REPO="Qwen/Qwen3.6-27B"
LOCAL_DIR="/mnt/data/llm-models/Qwen/Qwen3.6-27B"

if [[ ! -d /mnt/data ]]; then
    echo "[download-base] ERROR: /mnt/data not found. Run on homeserver." >&2
    exit 1
fi

mkdir -p "$(dirname "${LOCAL_DIR}")"

HF_CMD=""
if command -v hf >/dev/null 2>&1; then HF_CMD="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then HF_CMD="huggingface-cli"
else
    echo "[download-base] ERROR: install huggingface-hub: pip install -U huggingface-hub" >&2
    exit 1
fi

echo "[download-base] CLI:       ${HF_CMD}"
echo "[download-base] Repo:      ${MODEL_REPO}"
echo "[download-base] Local dir: ${LOCAL_DIR}"
echo "[download-base] HF_TOKEN:  ${HF_TOKEN:+set}"
echo "[download-base] Expected:  ~54 GB BF16 weights (15 shards)"

ARGS=("${MODEL_REPO}" --local-dir "${LOCAL_DIR}")
[[ -n "${HF_TOKEN:-}" ]] && ARGS+=(--token "${HF_TOKEN}")

if "${HF_CMD}" download --help 2>&1 | grep -q -- "--local-dir-use-symlinks"; then
    "${HF_CMD}" download "${ARGS[@]}" --local-dir-use-symlinks False
else
    "${HF_CMD}" download "${ARGS[@]}"
fi

echo ""
echo "[download-base] === Download complete ==="
echo "[download-base] Total size: $(du -sh "${LOCAL_DIR}" | awk '{print $1}')"
echo "[download-base] Shards:     $(ls "${LOCAL_DIR}"/*.safetensors 2>/dev/null | wc -l)"
echo "[download-base] Next:       scripts/sync_qwen36_base_to_sparks.sh"
