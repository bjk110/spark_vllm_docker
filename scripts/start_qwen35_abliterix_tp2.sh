#!/usr/bin/env bash
# =============================================================================
# start_qwen35_abliterix_tp2.sh
#
# Brings up the wangzhang/Qwen3.5-122B-A10B-abliterix FP8 TP=2 cluster.
#
# Modes:
#   - On spark01 (head)
#   - On spark02 (worker)
#   - On homeserver (orchestrates BOTH via SSH, recommended)
#
# Auto-detects host by hostname; override with HOST=spark01|spark02|orchestrate
#
# Same-name cleanup: any prior vllm-qwen35-abliterix-{head,worker} is stopped
# before bringing up fresh. Containers from other compose files are NOT touched.
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

stop_stale() {
    local profile="$1"
    local name="vllm-qwen35-abliterix-${profile}"
    if docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
        echo "[start] Stopping stale container ${name}..."
        docker stop "${name}" 2>/dev/null || true
        docker rm   "${name}" 2>/dev/null || true
    fi
}

bring_up() {
    local profile="$1"
    echo "[start] === ${HOST}: docker compose up --profile ${profile} ==="
    cd "${REPO_DIR}"
    stop_stale "${profile}"
    # GB10 unified memory: page cache from prior runs counts against GPU
    # free memory in vLLM's request_memory check. Drop caches so vLLM
    # sees the full 121 GiB pool instead of failing with "Free memory
    # 8.1/121.63 GiB on startup is less than desired".
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || \
        echo "[start] (note: drop_caches needs passwordless sudo; skipped)"
    PRESET_ENV_FILE="${ENV_FILE}" docker compose \
        -p "${PROJECT}" \
        -f "${COMPOSE_FILE}" \
        --env-file "${ENV_FILE}" \
        --profile "${profile}" \
        up -d
    echo "[start] Up. Tail logs with: scripts/logs_qwen35_abliterix_tp2.sh"
}

orchestrate() {
    echo "[start] Orchestrate mode: launching head on ${SPARK01_MGMT}, worker on ${SPARK02_MGMT}"
    echo "[start] (Worker first so head's Ray wait loop sees both nodes.)"

    ssh "${SSH_USER}@${SPARK02_MGMT}" \
        "cd '${REPO_DIR}' && PRESET='${PRESET}' bash scripts/start_qwen35_abliterix_tp2.sh" <<<""
    sleep 3
    ssh "${SSH_USER}@${SPARK01_MGMT}" \
        "cd '${REPO_DIR}' && PRESET='${PRESET}' bash scripts/start_qwen35_abliterix_tp2.sh" <<<""

    cat <<EOF

[start] === Orchestrate done ===
[start] Health probes (allow ~5-10 min for engine init + spec-decode profiling):
[start]   curl http://${SPARK01_MGMT}:8000/health
[start]   curl http://${SPARK01_MGMT}:8000/v1/models
[start] Logs:
[start]   ssh ${SSH_USER}@${SPARK01_MGMT} 'docker logs -f vllm-qwen35-abliterix-head'
[start]   ssh ${SSH_USER}@${SPARK02_MGMT} 'docker logs -f vllm-qwen35-abliterix-worker'
EOF
}

case "${HOST}" in
    spark01)     bring_up head ;;
    spark02)     bring_up worker ;;
    orchestrate) orchestrate ;;
    *) echo "[start] ERROR: HOST must be spark01|spark02|orchestrate, got '${HOST}'" >&2; exit 1 ;;
esac
