#!/usr/bin/env bash
# =============================================================================
# logs_qwen35_abliterix_tp2.sh — tail logs for the abliterix TP=2 containers.
#
# Args: -f (default), or pass --tail=N for snapshot. Pass -w to follow only
# the worker, -h to follow only the head.
# =============================================================================
set -euo pipefail

SPARK01_MGMT="192.168.0.200"
SPARK02_MGMT="192.168.0.201"
SSH_USER="${SSH_USER:-bjk110}"

FOLLOW_HEAD=1
FOLLOW_WORKER=1
EXTRA=("-f" "--tail=200")
for arg in "$@"; do
    case "$arg" in
        -h) FOLLOW_WORKER=0 ;;
        -w) FOLLOW_HEAD=0 ;;
        *)  EXTRA=("$arg") ;;
    esac
done

if [[ ${FOLLOW_HEAD} -eq 1 ]]; then
    echo "[logs] === HEAD (spark01: vllm-qwen35-abliterix-head) ===" >&2
    ssh "${SSH_USER}@${SPARK01_MGMT}" "docker logs ${EXTRA[*]} vllm-qwen35-abliterix-head 2>&1 | sed 's/^/[head] /'" &
    HEAD_PID=$!
fi
if [[ ${FOLLOW_WORKER} -eq 1 ]]; then
    echo "[logs] === WORKER (spark02: vllm-qwen35-abliterix-worker) ===" >&2
    ssh "${SSH_USER}@${SPARK02_MGMT}" "docker logs ${EXTRA[*]} vllm-qwen35-abliterix-worker 2>&1 | sed 's/^/[worker] /'" &
    WORKER_PID=$!
fi
wait
