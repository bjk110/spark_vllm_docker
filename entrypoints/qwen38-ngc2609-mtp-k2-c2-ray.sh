#!/usr/bin/env bash
set -euo pipefail

: "${ROLE:?ROLE=head|worker}"
: "${MODEL_CONTAINER_PATH:?MODEL_CONTAINER_PATH required}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME required}"
: "${MODEL_REVISION:?MODEL_REVISION required}"
: "${VLLM_HOST_IP:?VLLM_HOST_IP required}"
: "${HEAD_ROCE_IP:?HEAD_ROCE_IP required}"
: "${RAY_PORT:?RAY_PORT required}"

case "${ROLE}" in
  head|worker) ;;
  *) echo "invalid ROLE=${ROLE}" >&2; exit 64 ;;
esac

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
  --distributed-executor-backend ray
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

printf '[qwen38-ngc2609-mtp-k2-c2-ray] role=%s model=%s\n' \
  "${ROLE}" "${SERVED_MODEL_NAME}"
printf '[qwen38-ngc2609-mtp-k2-c2-ray] argv:'
printf ' %q' "${args[@]}"
printf '\n'

if [[ "${PRELAUNCH_ONLY:-0}" == 1 ]]; then
  python - <<'PY'
import cupy
import ray
import vllm
print(f"ray={ray.__version__} cupy_runtime={cupy.cuda.runtime.runtimeGetVersion()} vllm={vllm.__version__}")
PY
  echo QWEN38_NGC2609_MTP_K2_C2_RAY_PRELAUNCH_PASS
  exit 42
fi

ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray

if [[ "${ROLE}" == worker ]]; then
  echo "[qwen38-ngc2609-mtp-k2-c2-ray] joining ${HEAD_ROCE_IP}:${RAY_PORT}"
  if [[ -n "${RAY_OBJECT_STORE_MEMORY_BYTES:-}" ]]; then
    exec ray start --address="${HEAD_ROCE_IP}:${RAY_PORT}" \
      --node-ip-address="${VLLM_HOST_IP}" \
      --object-store-memory="${RAY_OBJECT_STORE_MEMORY_BYTES}" --block
  fi
  exec ray start --address="${HEAD_ROCE_IP}:${RAY_PORT}" \
    --node-ip-address="${VLLM_HOST_IP}" --block
fi

head_cmd=(
  ray start --head --port="${RAY_PORT}"
  --node-ip-address="${VLLM_HOST_IP}"
  --disable-usage-stats
)
if [[ "${RAY_INCLUDE_DASHBOARD:-false}" == false || "${RAY_INCLUDE_DASHBOARD:-false}" == 0 ]]; then
  head_cmd+=(--include-dashboard=false)
else
  head_cmd+=(--dashboard-host=0.0.0.0)
fi
if [[ -n "${RAY_OBJECT_STORE_MEMORY_BYTES:-}" ]]; then
  head_cmd+=(--object-store-memory="${RAY_OBJECT_STORE_MEMORY_BYTES}")
fi
"${head_cmd[@]}"

join_deadline=$((SECONDS + ${RAY_JOIN_TIMEOUT_SEC:-600}))
while (( SECONDS < join_deadline )); do
  if HEAD_ROCE_IP="${HEAD_ROCE_IP}" RAY_PORT="${RAY_PORT}" NNODES="${NNODES:-2}" python - <<'PY'
import os
import ray
ray.init(address=f"{os.environ['HEAD_ROCE_IP']}:{os.environ['RAY_PORT']}", logging_level="ERROR")
alive = sum(bool(node.get("Alive")) for node in ray.nodes())
expected = int(os.environ["NNODES"])
print(f"Ray alive nodes: {alive}/{expected}")
ray.shutdown()
raise SystemExit(0 if alive >= expected else 1)
PY
  then
    break
  fi
  sleep 5
done
if (( SECONDS >= join_deadline )); then
  echo "Ray cluster join timeout after ${RAY_JOIN_TIMEOUT_SEC:-600}s" >&2
  ray status --address="${HEAD_ROCE_IP}:${RAY_PORT}" || true
  exit 70
fi
ray status --address="${HEAD_ROCE_IP}:${RAY_PORT}"
exec "${args[@]}"
