#!/usr/bin/env bash
# =============================================================================
# run_deepseek_v4_flash_sm120_poc.sh
# DeepSeek-V4-Flash SM120 reference-attention POC launcher (PR #40852 image).
#
# Scope (correctness-first, NOT production):
#   - Image:   vllm-spark:deepseek-v4-sm120-poc  (built from Dockerfile.deepseek-v4)
#   - Model:   DeepSeek-V4-Flash (FP4 + FP8 mixed)
#   - Targets: 2× DGX Spark (GB10 / SM121) over 200 Gbps RoCE.
#
# Modes:
#   MODE=single   (default)   single-Spark TP=1 — smoke import + sanity start
#                              (model won't fully fit; useful only to surface
#                               import / KV-init errors quickly)
#   MODE=tp2-mn   multi-node  Ray HEAD/WORKER over RoCE, TP=2.
#                              The actual Ray cluster bring-up is handled by
#                              docker-compose head + worker profiles. This
#                              script simply applies SM120 reference env then
#                              calls docker compose.
#
# Tiers (--max-model-len budget):
#   TIER=8k    8192    (default — first-boot smoke)
#   TIER=32k   32768
#   TIER=256k  262144  (PR #40852 commit message claim — HTTP 200 on jasl host)
#
# Usage:
#   On spark01 (head):     MODE=tp2-mn TIER=8k bash scripts/run_deepseek_v4_flash_sm120_poc.sh
#   On spark02 (worker):   MODE=tp2-mn ROLE=worker bash scripts/run_deepseek_v4_flash_sm120_poc.sh
#   Single-host smoke:     MODE=single bash scripts/run_deepseek_v4_flash_sm120_poc.sh
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

MODE="${MODE:-single}"
TIER="${TIER:-8k}"
ROLE="${ROLE:-head}"

# ---- Tier → (MAX_MODEL_LEN, MAX_NUM_BATCHED_TOKENS) ----
case "${TIER}" in
    8k)   MAX_MODEL_LEN=8192    ; MAX_NUM_BATCHED_TOKENS=4096   ;;
    32k)  MAX_MODEL_LEN=32768   ; MAX_NUM_BATCHED_TOKENS=8192   ;;
    256k) MAX_MODEL_LEN=262144  ; MAX_NUM_BATCHED_TOKENS=16384  ;;
    *) echo "FATAL: unknown TIER=${TIER} (use 8k|32k|256k)" >&2; exit 1 ;;
esac
echo "[poc] TIER=${TIER} → MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"

# ---- Image / model ----
VLLM_IMAGE="${VLLM_IMAGE:-vllm-spark:deepseek-v4-sm120-poc}"
MODEL_PATH="${MODEL_PATH:-/home/bjk110/Documents/Models/deepseek-ai/deepseek-ai_DeepSeek-V4-Flash}"
MODEL_CONTAINER_PATH="${MODEL_CONTAINER_PATH:-/models/DeepSeek-V4-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Flash}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "FATAL: model not present at ${MODEL_PATH}" >&2
    echo "Run scripts/rsync_deepseek_v4_from_homeserver.sh or rsync_deepseek_v4_spark01_to_spark02_rdma.sh first." >&2
    exit 2
fi

# ---- SM120 reference path env (PR #40852) ----
# Both canonical (VLLM_TRITON_MLA_SPARSE_*) and legacy aliases (VLLM_SM120_*)
# are set so the same script runs across PR commit history. SM12x devices
# auto-select the reference path even without these, but explicit is safer
# during POC validation.
export VLLM_TRITON_MLA_SPARSE="${VLLM_TRITON_MLA_SPARSE:-1}"
export VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE="${VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE:-256}"
export VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE="${VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE:-128}"
export VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION="${VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION:-1}"
export VLLM_SM120_REFERENCE_TOPK_CHUNK_SIZE="${VLLM_SM120_REFERENCE_TOPK_CHUNK_SIZE:-256}"
export VLLM_SM120_REFERENCE_QUERY_CHUNK_SIZE="${VLLM_SM120_REFERENCE_QUERY_CHUNK_SIZE:-128}"

# ---- Common engine / RPC tuning for cold start of a 284B model ----
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"
export VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-1}"
export TILELANG_CLEANUP_TEMP_FILES="${TILELANG_CLEANUP_TEMP_FILES:-1}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"

# ---- Build env (also applied at runtime so JIT respects the right ptxas) ----
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/usr/local/cuda/bin:${PATH}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-120a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"

# ---- vLLM serve flag set (matches user POC plan, conservative for first boot) ----
# Diff vs PR #40852 commit-message example:
#   - --max-model-len driven by TIER (8k default, not 262144)
#   - --tensor-parallel-size honors MODE (1 in single, 2 in tp2-mn)
#   - --compilation-config kept as in plan
VLLM_EXTRA_ARGS=(
    --trust-remote-code
    --kv-cache-dtype fp8
    --block-size 256
    --max-num-seqs "${MAX_NUM_SEQS:-1}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.83}"
    --max-model-len "${MAX_MODEL_LEN}"
    --tokenizer-mode deepseek_v4
    --reasoning-parser deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --enable-expert-parallel
    --no-disable-hybrid-kv-cache-manager
    --disable-uvicorn-access-log
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
)

case "${MODE}" in
    single)
        echo "[poc] MODE=single → single-host smoke (TP=1, no Ray)"
        echo "[poc] WARNING: DeepSeek-V4-Flash 284B does NOT fit on 1× GB10 (~100GB FP4+FP8)."
        echo "[poc]          Use this MODE only to surface import/init errors fast."
        echo "[poc] image=${VLLM_IMAGE} model=${MODEL_PATH}"

        docker rm -f vllm-sm120-poc-single 2>/dev/null || true
        exec docker run --rm --name vllm-sm120-poc-single \
            --network host --ipc host \
            --gpus all \
            --ulimit memlock=-1:-1 \
            -v "${MODEL_PATH}:${MODEL_CONTAINER_PATH}:ro" \
            -v "${REPO_DIR}/.cache/vllm:/root/.cache/vllm" \
            -v "${REPO_DIR}/patches:/patches:ro" \
            -e VLLM_TRITON_MLA_SPARSE \
            -e VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE \
            -e VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE \
            -e VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION \
            -e VLLM_SM120_REFERENCE_TOPK_CHUNK_SIZE \
            -e VLLM_SM120_REFERENCE_QUERY_CHUNK_SIZE \
            -e VLLM_ENGINE_READY_TIMEOUT_S \
            -e VLLM_RPC_TIMEOUT \
            -e VLLM_LOG_STATS_INTERVAL \
            -e TILELANG_CLEANUP_TEMP_FILES \
            -e VLLM_DISABLE_COMPILE_CACHE \
            -e CUDA_ARCH_LIST -e TORCH_CUDA_ARCH_LIST -e TRITON_PTXAS_PATH \
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            -p 8000:8000 \
            "${VLLM_IMAGE}" \
            vllm serve "${MODEL_CONTAINER_PATH}" \
                --served-model-name "${SERVED_MODEL_NAME}" \
                --tensor-parallel-size 1 \
                --host 0.0.0.0 --port 8000 \
                "${VLLM_EXTRA_ARGS[@]}"
        ;;

    tp2-mn)
        # Multi-node Ray TP=2 — go through docker-compose so the existing
        # head/worker plumbing (RDMA env, NCCL_IB_HCA, RoCE IPs) is reused.
        PRESET="${REPO_DIR}/models/deepseek-v4-flash-tp2.env"
        if [ ! -f "${PRESET}" ]; then
            echo "FATAL: preset not found: ${PRESET}" >&2
            exit 1
        fi
        # Materialize .env with overrides for this POC tier.
        TMP_ENV="${REPO_DIR}/.env"
        sed -E \
            -e "s|^VLLM_IMAGE=.*|VLLM_IMAGE=${VLLM_IMAGE}|" \
            -e "s|^MAX_MODEL_LEN=.*|MAX_MODEL_LEN=${MAX_MODEL_LEN}|" \
            -e "s|^MAX_NUM_SEQS=.*|MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}|" \
            -e "s|^MAX_NUM_BATCHED_TOKENS=.*|MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}|" \
            -e "s|^GPU_MEMORY_UTILIZATION=.*|GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.83}|" \
            "${PRESET}" > "${TMP_ENV}"
        # Append the SM120 reference + serve extra args so docker-compose
        # passes them through to the engine.
        cat >> "${TMP_ENV}" <<EOF

# ---- SM120 reference path overrides (run_deepseek_v4_flash_sm120_poc.sh) ----
VLLM_TRITON_MLA_SPARSE=${VLLM_TRITON_MLA_SPARSE}
VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE=${VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE}
VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE=${VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE}
VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION=${VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION}
VLLM_SM120_REFERENCE_TOPK_CHUNK_SIZE=${VLLM_SM120_REFERENCE_TOPK_CHUNK_SIZE}
VLLM_SM120_REFERENCE_QUERY_CHUNK_SIZE=${VLLM_SM120_REFERENCE_QUERY_CHUNK_SIZE}
VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S}
VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT}
VLLM_LOG_STATS_INTERVAL=${VLLM_LOG_STATS_INTERVAL}
TILELANG_CLEANUP_TEMP_FILES=${TILELANG_CLEANUP_TEMP_FILES}
VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE}
EOF
        # Override VLLM_EXTRA_ARGS in .env. Quoted carefully; docker-compose's
        # env-file does NOT do shell parsing so single-line is safest.
        # Strip the existing VLLM_EXTRA_ARGS= line and append the new one.
        sed -i '/^VLLM_EXTRA_ARGS=/d' "${TMP_ENV}"
        EXTRA_JOINED="$(printf '%s ' "${VLLM_EXTRA_ARGS[@]}")"
        printf 'VLLM_EXTRA_ARGS=%s\n' "${EXTRA_JOINED}" >> "${TMP_ENV}"

        echo "[poc] MODE=tp2-mn ROLE=${ROLE}"
        echo "[poc] env-file: ${TMP_ENV}"
        echo "[poc] image:    ${VLLM_IMAGE}"

        # Pre-flight: tear down any lingering containers
        docker compose --profile head down --remove-orphans 2>/dev/null || true
        docker compose --profile worker down --remove-orphans 2>/dev/null || true

        case "${ROLE}" in
            head)
                docker compose --env-file "${TMP_ENV}" --profile head up -d
                echo "[poc] tailing head logs (Ctrl-C detaches)"
                exec docker logs -f vllm-spark-head
                ;;
            worker)
                docker compose --env-file "${TMP_ENV}" --profile worker up -d
                echo "[poc] tailing worker logs (Ctrl-C detaches)"
                exec docker logs -f vllm-spark-worker
                ;;
            *) echo "FATAL: ROLE must be head|worker, got '${ROLE}'" >&2; exit 1 ;;
        esac
        ;;

    *)
        echo "FATAL: unknown MODE=${MODE} (use single|tp2-mn)" >&2
        exit 1
        ;;
esac
