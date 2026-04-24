#!/usr/bin/env bash
# =============================================================================
# run_deepseek_v4_flash_spark01.sh — launch Ray head + vLLM API on spark01
#
# EXPERIMENTAL: DeepSeek-V4-Flash on 2× GB10 with TP=2 + EP.
#
# Run this ON spark01, from the vllm-spark repo root.
#   cd ~/docker/vllm-spark
#   ./scripts/run_deepseek_v4_flash_spark01.sh
#
# Paired with scripts/run_deepseek_v4_flash_spark02.sh (run on spark02).
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PRESET="${REPO_DIR}/models/deepseek-v4-flash-tp2.env"
if [ ! -f "${PRESET}" ]; then
    echo "FATAL: preset not found: ${PRESET}" >&2
    exit 1
fi

# Materialize .env
cp "${PRESET}" "${REPO_DIR}/.env"

# Extra V4 / tilelang runtime env — appended to docker-compose env via profile
# Wrote these as exported shell vars so docker compose picks them up when the
# compose file references ${VAR}.
export TILELANG_CLEANUP_TEMP_FILES=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=600000

echo "=== spark01 HEAD launch ==="
echo "preset: ${PRESET}"
echo "image:  $(grep ^VLLM_IMAGE "${REPO_DIR}/.env" | cut -d= -f2)"
echo "model:  $(grep ^MODEL_PATH "${REPO_DIR}/.env" | cut -d= -f2)"
echo

# Pre-flight: model present?
MODEL_PATH=$(grep ^MODEL_PATH "${REPO_DIR}/.env" | cut -d= -f2)
if [ ! -d "${MODEL_PATH}" ]; then
    echo "FATAL: model not present at ${MODEL_PATH}" >&2
    echo "Run scripts/rsync_deepseek_v4_from_homeserver.sh first." >&2
    exit 2
fi

# Pre-flight: tear down any existing stack
docker compose --profile head down --remove-orphans 2>/dev/null || true
docker compose --profile worker down --remove-orphans 2>/dev/null || true

# Bring up head
docker compose --env-file "${REPO_DIR}/.env" --profile head up -d

echo
echo "Tailing head logs (Ctrl-C to detach — container keeps running):"
docker logs -f vllm-spark-head
