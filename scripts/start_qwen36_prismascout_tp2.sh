#!/usr/bin/env bash
# =============================================================================
# start_qwen36_prismascout_tp2.sh
#
# Brings up the Qwen3.6-27B PrismaSCOUT NVFP4-BF16 TP=2 cluster.
#
# Modes:
#   - Run on spark01 (head):  spark01 launches head profile
#   - Run on spark02 (worker): spark02 launches worker profile
#   - Run on homeserver: orchestrates BOTH via SSH (recommended)
#
# Auto-detects the host by hostname; override with HOST=spark01|spark02|orchestrate
#
# Same-name container cleanup: any prior `vllm-qwen36-{head,worker}` is stopped
# before this brings up a fresh one. Containers from the shared
# docker-compose.yml (vllm-spark-head / vllm-spark-worker) are NOT touched.
# =============================================================================
set -euo pipefail

# --- Where the repo lives ---
REPO_DIR="${REPO_DIR:-/home/bjk110/docker/vllm-spark}"
COMPOSE_FILE="docker-compose.qwen36-prismascout-tp2.yml"
# Preset slug (filename without .env). Set PRESET to switch between:
#   qwen3.6-27b-prismascout-nvfp4-tp2  (default — NVFP4, blocked by TP=2)
#   qwen3.6-27b-base-bf16-tp2          (base BF16 fallback, works on TP=2)
PRESET="${PRESET:-qwen3.6-27b-prismascout-nvfp4-tp2}"
ENV_FILE="models/${PRESET}.env"
PROJECT="${PROJECT_NAME:-qwen36-prismascout-tp2}"

# --- Mgmt IPs (used by orchestrate mode) ---
SPARK01_MGMT="192.168.0.200"
SPARK02_MGMT="192.168.0.201"
SSH_USER="${SSH_USER:-bjk110}"

# --- Detect role ---
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
    local name="vllm-qwen36-${profile}"
    if docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
        echo "[start] Stopping stale container ${name}..."
        docker stop "${name}" 2>/dev/null || true
        docker rm   "${name}" 2>/dev/null || true
    fi
}

bring_up() {
    local profile="$1"     # head | worker
    echo "[start] === ${HOST}: docker compose up --profile ${profile} ==="
    cd "${REPO_DIR}"
    stop_stale "${profile}"
    # PRESET_ENV_FILE is consumed by compose's env_file: directive (so the
    # container gets the right preset's vars). --env-file feeds compose-level
    # ${VAR} substitution from the same file. Both point at the same env.
    PRESET_ENV_FILE="${ENV_FILE}" docker compose \
        -p "${PROJECT}" \
        -f "${COMPOSE_FILE}" \
        --env-file "${ENV_FILE}" \
        --profile "${profile}" \
        up -d
    echo "[start] Up. Tail logs with: scripts/logs_qwen36_prismascout_tp2.sh"
}

orchestrate() {
    echo "[start] Orchestrate mode: launching head on ${SPARK01_MGMT}, worker on ${SPARK02_MGMT}"
    echo "[start] (Worker comes up first so head's Ray wait loop sees both nodes.)"

    # spark02 (worker) first — head spins forever in 'waiting for nodes' otherwise.
    ssh "${SSH_USER}@${SPARK02_MGMT}" \
        "cd '${REPO_DIR}' && PRESET='${PRESET}' bash scripts/start_qwen36_prismascout_tp2.sh" <<<""

    # Small grace period; entrypoint will block on Ray join either way.
    sleep 3

    ssh "${SSH_USER}@${SPARK01_MGMT}" \
        "cd '${REPO_DIR}' && PRESET='${PRESET}' bash scripts/start_qwen36_prismascout_tp2.sh" <<<""

    cat <<EOF

[start] === Orchestrate done ===
[start] Health probes (run after engine init completes, ~5–10 min):
[start]   curl http://${SPARK01_MGMT}:8000/health
[start]   curl http://${SPARK01_MGMT}:8000/v1/models
[start] Logs:
[start]   ssh ${SSH_USER}@${SPARK01_MGMT} 'docker logs -f vllm-qwen36-head'
[start]   ssh ${SSH_USER}@${SPARK02_MGMT} 'docker logs -f vllm-qwen36-worker'
EOF
}

case "${HOST}" in
    spark01)     bring_up head ;;
    spark02)     bring_up worker ;;
    orchestrate) orchestrate ;;
    *) echo "[start] ERROR: HOST must be spark01|spark02|orchestrate, got '${HOST}'" >&2; exit 1 ;;
esac
