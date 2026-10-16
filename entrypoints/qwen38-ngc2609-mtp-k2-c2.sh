#!/usr/bin/env bash
set -euo pipefail

: "${ROLE:?ROLE=head|worker}"
: "${MODEL_CONTAINER_PATH:?MODEL_CONTAINER_PATH required}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME required}"
: "${MODEL_REVISION:?MODEL_REVISION required}"

case "${ROLE}" in
  head) NODE_RANK=0 ;;
  worker) NODE_RANK=${NODE_RANK:-1} ;;
  *) echo "invalid ROLE=${ROLE}" >&2; exit 64 ;;
esac
case "${ROLE}:${NODE_RANK}" in
  head:0|worker:1) ;;
  *) echo "invalid role/rank: ${ROLE}/${NODE_RANK}" >&2; exit 64 ;;
esac

: "${VLLM_HOST_IP:?VLLM_HOST_IP required}"
: "${HEAD_ROCE_IP:?HEAD_ROCE_IP required}"
: "${MASTER_PORT:?MASTER_PORT required}"
[[ -f "${MODEL_CONTAINER_PATH}/config.json" ]] || {
  echo "missing ${MODEL_CONTAINER_PATH}/config.json" >&2
  exit 65
}
revision_marker="${MODEL_CONTAINER_PATH}/.hf-revision"
[[ -f "${revision_marker}" ]] || {
  echo "missing ${revision_marker}" >&2
  exit 65
}
IFS= read -r actual_revision < "${revision_marker}"
[[ "${actual_revision}" == "${MODEL_REVISION}" ]] || {
  echo "model revision mismatch: ${actual_revision} != ${MODEL_REVISION}" >&2
  exit 65
}
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

args=(
  vllm serve "${MODEL_CONTAINER_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --port "${HOST_PORT:-8000}"
  --tensor-parallel-size "${TP_SIZE:-2}"
  --nnodes "${NNODES:-2}"
  --node-rank "${NODE_RANK}"
  --master-addr "${HEAD_ROCE_IP}"
  --master-port "${MASTER_PORT}"
  --distributed-executor-backend mp
  --trust-remote-code
  --dtype auto
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${MAX_NUM_SEQS:-2}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.83}"
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-8589934592}"
  --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}"
  --enforce-eager
  --enable-prefix-caching
  --enable-chunked-prefill
  --enable-expert-parallel
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  --no-enable-flashinfer-autotune
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
)
[[ "${ROLE}" == worker ]] && args+=(--headless)

printf '[qwen38-ngc2609-mtp-k2-c2] role=%s rank=%s model=%s\n' \
  "${ROLE}" "${NODE_RANK}" "${SERVED_MODEL_NAME}"
printf '[qwen38-ngc2609-mtp-k2-c2] argv:'
printf ' %q' "${args[@]}"
printf '\n'

if [[ "${PRELAUNCH_ONLY:-0}" == 1 ]]; then
  echo QWEN38_NGC2609_MTP_K2_C2_PRELAUNCH_PASS
  exit 42
fi
exec "${args[@]}"
