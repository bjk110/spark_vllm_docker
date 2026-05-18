#!/usr/bin/env bash
# =============================================================================
# logs_qwen36_prismascout_tp2.sh
#
# Tails logs for Qwen3.6 PrismaSCOUT TP=2 containers.
# Run on:
#   spark01 / spark02 -> tails the local container.
#   anywhere else      -> opens two SSH sessions (head + worker) side by side.
#
# Options:
#   --health   Print API health + /v1/models response and exit (no tail).
# =============================================================================
set -euo pipefail

SPARK01_MGMT="192.168.0.200"
SPARK02_MGMT="192.168.0.201"
SSH_USER="${SSH_USER:-bjk110}"
HEALTH_ONLY="0"

for arg in "$@"; do
    case "$arg" in
        --health) HEALTH_ONLY="1" ;;
        *) echo "[logs] Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

HOST="${HOST:-}"
if [[ -z "${HOST}" ]]; then
    case "$(hostname)" in
        spark01*) HOST="spark01" ;;
        spark02*) HOST="spark02" ;;
        *) HOST="orchestrate" ;;
    esac
fi

health_probe() {
    echo "[logs] === Health probe ==="
    echo "[logs] curl http://${SPARK01_MGMT}:8000/health"
    curl -sS --max-time 5 "http://${SPARK01_MGMT}:8000/health" || echo "  (no response)"
    echo ""
    echo "[logs] curl http://${SPARK01_MGMT}:8000/v1/models"
    curl -sS --max-time 5 "http://${SPARK01_MGMT}:8000/v1/models" || echo "  (no response)"
    echo ""
}

if [[ "${HEALTH_ONLY}" = "1" ]]; then
    health_probe
    exit 0
fi

case "${HOST}" in
    spark01)
        docker logs -f --tail 200 vllm-qwen36-head
        ;;
    spark02)
        docker logs -f --tail 200 vllm-qwen36-worker
        ;;
    orchestrate)
        echo "[logs] No native multiplex; pick one of:"
        echo "[logs]   ssh ${SSH_USER}@${SPARK01_MGMT} 'docker logs -f vllm-qwen36-head'"
        echo "[logs]   ssh ${SSH_USER}@${SPARK02_MGMT} 'docker logs -f vllm-qwen36-worker'"
        echo ""
        health_probe
        ;;
    *)
        echo "[logs] ERROR: HOST must be spark01|spark02|orchestrate, got '${HOST}'" >&2
        exit 1
        ;;
esac
