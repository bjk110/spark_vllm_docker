#!/bin/bash
set -euo pipefail

# Apply Qwen3.5 MoE text-only patches (abliterated models only)
# Set APPLY_TEXT_ONLY_SHIM=1 in .env to enable
if [ "${APPLY_TEXT_ONLY_SHIM:-0}" = "1" ] && [ -f /patches/patch_qwen35_moe_text.py ]; then
    echo "[entrypoint] Applying TextOnlyShim patch (APPLY_TEXT_ONLY_SHIM=1)"
    python3 /patches/patch_qwen35_moe_text.py || true
fi

# =============================================================================
# vLLM Spark Unified Entrypoint
#
# Cluster modes (CLUSTER_MODE):
#   single     (default) — one DGX Spark, no RDMA, no Ray. The entrypoint
#              forces VLLM_HOST_IP=127.0.0.1 and clears NCCL/GLOO/UCX
#              interface bindings so vLLM/PyTorch c10d does not try to
#              bind to a missing 10.10.10.x RoCE IP and hang.
#   dual-rdma  Two DGX Sparks over 200 Gbps RoCE. Requires HEAD_ROCE_IP,
#              WORKER_ROCE_IP, ROCE_IF_NAME, IB_HCA_NAME, RAY_PORT.
#
# Role × TP_SIZE behavior:
#   head    + TP_SIZE=1   → direct vllm serve, no Ray, single-rank c10d.
#   head    + TP_SIZE>=2  → Ray head → wait for workers → vllm serve.
#                           Requires CLUSTER_MODE=dual-rdma.
#   worker  + (any)       → ray start --block. Requires CLUSTER_MODE=dual-rdma.
#
# Required environment:
#   ROLE                  - "head" or "worker" (compose profiles set this)
#   MODEL_CONTAINER_PATH  - model path inside container
#   SERVED_MODEL_NAME     - model name for API
#
# Optional (with sensible defaults):
#   CLUSTER_MODE          - single (default) | dual-rdma
#   TP_SIZE               - 1 (default) ; 2+ implies dual-rdma
#   HOST_PORT, MAX_MODEL_LEN, MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION,
#   MAX_NUM_BATCHED_TOKENS, VLLM_EXTRA_ARGS
# =============================================================================

: "${ROLE:?ROLE must be set to 'head' or 'worker'}"
: "${TP_SIZE:=1}"
: "${CLUSTER_MODE:=single}"
# DISTRIBUTED_BACKEND controls multi-node TP coordination when TP_SIZE>=2:
#   ray (default) — Ray actors, head waits for workers via `ray status`
#   mp            — vLLM SPMD: each node runs `vllm serve` with
#                   --nnodes/--node-rank/--master-addr/--master-port,
#                   workers add --headless. Matches forum/eugr no-ray path.
: "${DISTRIBUTED_BACKEND:=ray}"
: "${MASTER_PORT:=29501}"

# ---------------------------------------------------------------------------
# Cluster mode normalization
# ---------------------------------------------------------------------------
# In single mode we must aggressively clear any RDMA env that compose may
# have piped in from a stale .env, because PyTorch c10d / NCCL / GLOO will
# try to bind those interfaces / IPs and stall the server socket.
case "${CLUSTER_MODE}" in
    single)
        if [ "${TP_SIZE}" -gt 1 ]; then
            echo "[entrypoint] ERROR: CLUSTER_MODE=single but TP_SIZE=${TP_SIZE}." >&2
            echo "[entrypoint]   TP>=2 requires CLUSTER_MODE=dual-rdma + 2 Sparks." >&2
            exit 1
        fi
        if [ "${ROLE}" = "worker" ]; then
            echo "[entrypoint] ERROR: ROLE=worker is meaningless in CLUSTER_MODE=single." >&2
            echo "[entrypoint]   Use ROLE=head (docker compose --profile head)." >&2
            exit 1
        fi
        # Force loopback for c10d master store; loopback always exists,
        # so single-rank init completes immediately.
        export VLLM_HOST_IP=127.0.0.1
        # Clear interface bindings — let NCCL/GLOO/UCX pick whatever default
        # is sane for a single host. NCCL on a single-GPU host doesn't need
        # InfiniBand or a specific socket interface.
        unset NCCL_SOCKET_IFNAME
        unset GLOO_SOCKET_IFNAME
        unset UCX_NET_DEVICES
        unset NCCL_IB_HCA
        # Disable IB transport explicitly. Without this, NCCL probes IB
        # devices even on single-host setups and may print scary warnings.
        export NCCL_IB_DISABLE=1
        # Single GPU: P2P/SHM are within the same device, no cross-node.
        export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
        echo "[entrypoint] CLUSTER_MODE=single: VLLM_HOST_IP=127.0.0.1, NCCL_IB_DISABLE=1, NCCL/GLOO/UCX ifname cleared"
        ;;
    dual-rdma)
        # Validate the RDMA env compose passes through. Fail fast on missing
        # values so we don't silently fall back to a broken default.
        for v in HEAD_ROCE_IP WORKER_ROCE_IP ROCE_IF_NAME IB_HCA_NAME RAY_PORT; do
            if [ -z "${!v:-}" ]; then
                echo "[entrypoint] ERROR: CLUSTER_MODE=dual-rdma requires ${v} to be set in .env" >&2
                exit 1
            fi
        done
        # VLLM_HOST_IP / RAY_NODE_IP_ADDRESS are already set by compose
        # to HEAD_ROCE_IP (head) or WORKER_ROCE_IP (worker). Leave alone.
        echo "[entrypoint] CLUSTER_MODE=dual-rdma: head=${HEAD_ROCE_IP} worker=${WORKER_ROCE_IP} iface=${ROCE_IF_NAME} hca=${IB_HCA_NAME}"
        ;;
    *)
        echo "[entrypoint] ERROR: CLUSTER_MODE must be 'single' or 'dual-rdma', got '${CLUSTER_MODE}'" >&2
        exit 1
        ;;
esac

# ---- Worker dispatch (dual-rdma only) ----
if [ "${ROLE}" = "worker" ]; then
    case "${DISTRIBUTED_BACKEND}" in
        ray)
            # Join Ray cluster and block
            ray stop --force 2>/dev/null || true
            rm -rf /tmp/ray 2>/dev/null || true
            echo "[entrypoint] Starting Ray WORKER → ${HEAD_ROCE_IP}:${RAY_PORT}"
            exec ray start \
                --address="${HEAD_ROCE_IP}:${RAY_PORT}" \
                --node-ip-address="${WORKER_ROCE_IP}" \
                --block
            ;;
        mp)
            # SPMD: run vllm serve --headless on worker
            # Flags from eugr/spark-vllm-docker launch-cluster.sh no-ray mode.
            echo "[entrypoint] Starting vLLM MP WORKER (rank ${NODE_RANK:-1}) → ${HEAD_ROCE_IP}:${MASTER_PORT}"
            VLLM_CMD=(
                vllm serve "${MODEL_CONTAINER_PATH}"
                --served-model-name "${SERVED_MODEL_NAME}"
                --max-model-len "${MAX_MODEL_LEN:-32768}"
                --max-num-seqs "${MAX_NUM_SEQS:-8}"
                --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
                --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}"
                --trust-remote-code
                --host 0.0.0.0
                --port "${HOST_PORT:-8000}"
                --dtype auto
                --enable-prefix-caching
                --tensor-parallel-size "${TP_SIZE}"
                --nnodes "${TP_SIZE}"
                --node-rank "${NODE_RANK:-1}"
                --master-addr "${HEAD_ROCE_IP}"
                --master-port "${MASTER_PORT}"
                --headless
            )
            if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
                # shellcheck disable=SC2206
                VLLM_CMD+=(${VLLM_EXTRA_ARGS})
            fi
            echo "[entrypoint] Running (mp worker): ${VLLM_CMD[*]}"
            exec "${VLLM_CMD[@]}"
            ;;
        *)
            echo "[entrypoint] ERROR: unknown DISTRIBUTED_BACKEND=${DISTRIBUTED_BACKEND}"
            exit 1
            ;;
    esac
fi

# ---- Head: standalone or multi-node ----
if [ "${ROLE}" != "head" ]; then
    echo "[entrypoint] ERROR: ROLE must be 'head' or 'worker', got '${ROLE}'"
    exit 1
fi

# Build vllm serve command
VLLM_CMD=(
    vllm serve "${MODEL_CONTAINER_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --max-model-len "${MAX_MODEL_LEN:-32768}"
    --max-num-seqs "${MAX_NUM_SEQS:-8}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}"
    --trust-remote-code
    --host 0.0.0.0
    --port "${HOST_PORT:-8000}"
    --dtype auto
    --enable-prefix-caching
)

if [ "${TP_SIZE}" -ge 2 ]; then
    case "${DISTRIBUTED_BACKEND}" in
        ray)
            # ---- Ray: start head, wait for workers, then serve ----
            echo "[entrypoint] Starting Ray HEAD (TP_SIZE=${TP_SIZE})..."
            ray start --head --port="${RAY_PORT}" \
                --node-ip-address="${HEAD_ROCE_IP}" \
                --dashboard-host=0.0.0.0 \
                --disable-usage-stats

            echo "[entrypoint] Waiting for ${TP_SIZE} node(s) to join Ray cluster..."
            while true; do
                NODE_COUNT=$(ray status 2>/dev/null | grep -c 'node_' || echo 0)
                if [ "${NODE_COUNT}" -ge "${TP_SIZE}" ]; then
                    echo "[entrypoint] All ${TP_SIZE} nodes joined! Starting vLLM..."
                    break
                fi
                sleep 5
            done

            VLLM_CMD+=(
                --tensor-parallel-size "${TP_SIZE}"
                --distributed-executor-backend ray
            )
            ;;
        mp)
            # ---- MP SPMD head: torch.distributed rendezvous on MASTER_ADDR:MASTER_PORT ----
            echo "[entrypoint] Starting vLLM MP HEAD (rank 0) ${HEAD_ROCE_IP}:${MASTER_PORT} (TP_SIZE=${TP_SIZE})"
            VLLM_CMD+=(
                --tensor-parallel-size "${TP_SIZE}"
                --nnodes "${TP_SIZE}"
                --node-rank 0
                --master-addr "${HEAD_ROCE_IP}"
                --master-port "${MASTER_PORT}"
            )
            ;;
        *)
            echo "[entrypoint] ERROR: unknown DISTRIBUTED_BACKEND=${DISTRIBUTED_BACKEND}"
            exit 1
            ;;
    esac
else
    # ---- Standalone: direct serve, no Ray ----
    echo "[entrypoint] Starting vLLM standalone (TP_SIZE=1, CLUSTER_MODE=${CLUSTER_MODE})..."
fi

# Append model-specific extra args (split on whitespace)
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    VLLM_CMD+=(${VLLM_EXTRA_ARGS})
fi

echo "[entrypoint] Running: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
