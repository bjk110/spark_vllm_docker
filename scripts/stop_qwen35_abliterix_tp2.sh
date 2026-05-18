#!/usr/bin/env bash
# =============================================================================
# stop_qwen35_abliterix_tp2.sh
#
# Stops the wangzhang/Qwen3.5-122B-A10B-abliterix FP8 TP=2 cluster.
# Does NOT touch the shared docker-compose.yml or qwen36 compose services.
#
# Auto-detects host by hostname; override with HOST=spark01|spark02|orchestrate
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/bjk110/docker/vllm-spark}"
COMPOSE_FILE="docker-compose.qwen35-abliterix-tp2.yml"
PRESET="${PRESET:-wangzhang-122b-abliterix-fp8-tp2}"
ENV_FILE="models/${PRESET}.env"
PROJECT="${PROJECT_NAME:-qwen35-abliterix-tp2}"

SPARK01_MGMT="192.168.0.200"
SPARK02_MGMT="192.168.0.201"
SSH_USER="${SSH_USER:-bjk110}"

HOST="${HOST:-}"
if [[ -z "${HOST}" ]]; then
    case "$(hostname)" in
        spark01*) HOST="spark01" ;;
        spark02*) HOST="spark02" ;;
        *) HOST="orchestrate" ;;
    esac
fi

bring_down() {
    local profile="$1"
    echo "[stop] === ${HOST}: docker compose down --profile ${profile} ==="
    cd "${REPO_DIR}"
    PRESET_ENV_FILE="${ENV_FILE}" docker compose \
        -p "${PROJECT}" \
        -f "${COMPOSE_FILE}" \
        --env-file "${ENV_FILE}" \
        --profile "${profile}" \
        down
}

orchestrate() {
    echo "[stop] Orchestrate: head first, then worker"
    ssh "${SSH_USER}@${SPARK01_MGMT}" \
        "cd '${REPO_DIR}' && PRESET='${PRESET}' bash scripts/stop_qwen35_abliterix_tp2.sh" <<<"" || true
    ssh "${SSH_USER}@${SPARK02_MGMT}" \
        "cd '${REPO_DIR}' && PRESET='${PRESET}' bash scripts/stop_qwen35_abliterix_tp2.sh" <<<"" || true
    echo "[stop] Done."
}

case "${HOST}" in
    spark01)     bring_down head ;;
    spark02)     bring_down worker ;;
    orchestrate) orchestrate ;;
    *) echo "[stop] ERROR: HOST must be spark01|spark02|orchestrate, got '${HOST}'" >&2; exit 1 ;;
esac
