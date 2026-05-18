#!/usr/bin/env bash
# =============================================================================
# download_qwen36_prismascout.sh
#
# Downloads rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm into the
# homeserver's central LLM model store at:
#   /mnt/data/llm-models/google/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm
#
# Path uses the project rule: "save under google/..." regardless of HF org.
#
# Behavior:
#   - Resumable: huggingface-cli handles partial blobs natively.
#   - Real files, not symlinks (--local-dir-use-symlinks False).
#   - Reads HF_TOKEN from env if set (private/gated models).
#   - Prints file list + total size on completion.
#
# Run on: homeserver (not on spark — Spark hosts have no /mnt/data).
# Requires: huggingface-hub installed (`pip install -U huggingface-hub`).
# =============================================================================
set -euo pipefail

MODEL_REPO="rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm"
LOCAL_DIR="/mnt/data/llm-models/google/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm"

# ---- Sanity: ensure we are on homeserver ----
if [[ ! -d /mnt/data ]]; then
    echo "[download] ERROR: /mnt/data not found. This script is meant to run on homeserver." >&2
    echo "[download]   spark01/spark02 should receive the model via sync_qwen36_prismascout_to_sparks.sh." >&2
    exit 1
fi

mkdir -p "$(dirname "${LOCAL_DIR}")"

# ---- Choose CLI: prefer modern `hf download`, fall back to `huggingface-cli` ----
HF_CMD=""
if command -v hf >/dev/null 2>&1; then
    HF_CMD="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD="huggingface-cli"
else
    echo "[download] ERROR: neither 'hf' nor 'huggingface-cli' on PATH. Install with: pip install -U huggingface-hub" >&2
    exit 1
fi

echo "[download] CLI:        ${HF_CMD}"
echo "[download] Repo:       ${MODEL_REPO}"
echo "[download] Local dir:  ${LOCAL_DIR}"
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[download] HF_TOKEN:   set (using authenticated download)"
else
    echo "[download] HF_TOKEN:   not set (anonymous download)"
fi

# ---- Build command per CLI variant ----
# `hf download` and `huggingface-cli download` accept the same positional args
# and the same --local-dir flag. The symlink flag name differs across versions
# of huggingface-cli; we pass it conditionally.
COMMON_ARGS=(
    "${MODEL_REPO}"
    --local-dir "${LOCAL_DIR}"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
    COMMON_ARGS+=(--token "${HF_TOKEN}")
fi

# Force real files. Newer huggingface-hub (>=0.23) removed --local-dir-use-symlinks
# entirely and writes real files by default; older versions still need it.
# We try with the flag, fall back to without.
echo "[download] Starting download (resumable)..."
if "${HF_CMD}" download --help 2>&1 | grep -q -- "--local-dir-use-symlinks"; then
    "${HF_CMD}" download "${COMMON_ARGS[@]}" --local-dir-use-symlinks False
else
    "${HF_CMD}" download "${COMMON_ARGS[@]}"
fi

echo ""
echo "[download] === Download complete ==="
echo "[download] Files:"
( cd "${LOCAL_DIR}" && find . -maxdepth 2 -type f -printf '  %p\t%s bytes\n' | sort )
echo ""
echo -n "[download] Total size: "
du -sh "${LOCAL_DIR}" | awk '{print $1}'
echo "[download] Next step: scripts/sync_qwen36_prismascout_to_sparks.sh"
