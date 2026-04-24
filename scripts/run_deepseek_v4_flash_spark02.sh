#!/usr/bin/env bash
# =============================================================================
# run_deepseek_v4_flash_spark02.sh — launch Ray worker on spark02
#
# Must be started BEFORE spark01 head completes its "wait for workers" loop.
# In practice: start both in parallel. The head waits up to ~forever for the
# worker to join before launching vllm serve.
#
# Run this ON spark02, from the vllm-spark repo root.
#   cd ~/docker/vllm-spark
#   ./scripts/run_deepseek_v4_flash_spark02.sh
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PRESET="${REPO_DIR}/models/deepseek-v4-flash-tp2.env"
if [ ! -f "${PRESET}" ]; then
    echo "FATAL: preset not found: ${PRESET}" >&2
    exit 1
fi

cp "${PRESET}" "${REPO_DIR}/.env"

export TILELANG_CLEANUP_TEMP_FILES=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=600000

echo "=== spark02 WORKER launch ==="

MODEL_PATH=$(grep ^MODEL_PATH "${REPO_DIR}/.env" | cut -d= -f2)
if [ ! -d "${MODEL_PATH}" ]; then
    echo "FATAL: model not present at ${MODEL_PATH}" >&2
    echo "Run scripts/rsync_deepseek_v4_spark01_to_spark02_rdma.sh from spark01 first." >&2
    exit 2
fi

docker compose --profile head down --remove-orphans 2>/dev/null || true
docker compose --profile worker down --remove-orphans 2>/dev/null || true

docker compose --env-file "${REPO_DIR}/.env" --profile worker up -d

echo
echo "Tailing worker logs:"
docker logs -f vllm-spark-worker
