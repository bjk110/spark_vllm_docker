#!/usr/bin/env bash
# Entrypoint for aidendle94/sparkrun-vllm-ds4-gb10 (unholy-fusion) image.
# Supports both Ray (default) and mp backends, matching jasl0603 entrypoint logic.
set -euo pipefail

# ── conda env setup ──────────────────────────────────────────────────────────
export PATH="/opt/env/bin:/opt/env/nvvm/bin:/opt/env/targets/sbsa-linux/nvvm/bin:${PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/opt/env/targets/sbsa-linux}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-${CUDA_HOME}}"
export LD_LIBRARY_PATH="/opt/env/lib:/opt/env/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/opt/env/bin/aarch64-conda-linux-gnu-g++}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin ${CUDAHOSTCXX} -I${CUDA_HOME}/include/cccl -I${CUDA_HOME}/include}"
export DG_JIT_NVCC_COMPILER=/opt/env/bin/nvcc
export FLASHINFER_NVCC=/opt/env/bin/nvcc
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_NCCL_SO_PATH="${VLLM_NCCL_SO_PATH:-/opt/env/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2}"

# ── persistent JIT cache dirs ────────────────────────────────────────────────
export HF_HOME="/cache/huggingface"
export DG_JIT_CACHE_DIR="/cache/huggingface/deepgemm-cache-rank${NODE_RANK:-0}"
export TRITON_CACHE_DIR="/cache/huggingface/triton-cache-rank${NODE_RANK:-0}"
export TORCHINDUCTOR_CACHE_DIR="/cache/huggingface/torchinductor-rank${NODE_RANK:-0}"
export VLLM_CACHE_ROOT="/cache/huggingface/vllm-cache"
mkdir -p "${DG_JIT_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

: "${ROLE:?ROLE must be set to head or worker}"
: "${DISTRIBUTED_BACKEND:=mp}"
: "${MTP_NUM_TOKENS:=1}"

# ── NCCL GID index: auto-detect RoCE v2 IPv4-mapped entry ───────────────────
for HCA in rocep1s0f0 roceP2p1s0f0; do
  for i in $(seq 0 7); do
    t=$(cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i 2>/dev/null) || continue
    g=$(cat /sys/class/infiniband/$HCA/ports/1/gids/$i 2>/dev/null) || continue
    case "$t" in *"RoCE v2"*)
      case "$g" in *0000:0000:0000:0000:0000:ffff:*)
        export NCCL_IB_GID_INDEX=$i
        echo "[unholy] NCCL_IB_GID_INDEX=${i} (${HCA})"
        break 2
      ;; esac
    ;; esac
  done
done

# ── cutlass DSL install if missing ───────────────────────────────────────────
PY=/opt/env/bin/python
if ! $PY -c "import cutlass; assert cutlass.__version__ == '4.5.1'" 2>/dev/null; then
  echo "[unholy] Installing nvidia-cutlass-dsl 4.5.1 ..."
  $PY -m pip install --no-cache-dir \
    "nvidia-cutlass-dsl==4.5.1" \
    "nvidia-cutlass-dsl-libs-base==4.5.1" \
    "nvidia-cutlass-dsl-libs-cu13==4.5.1" > /tmp/cutlass-install.log 2>&1 || true
fi

# ── GB10 UMA patch 1: request_memory() pre-init check ───────────────────────
VLLM_UTILS=/opt/env/lib/python3.12/site-packages/vllm/v1/worker/utils.py
if [ -f "$VLLM_UTILS" ] && ! grep -q "VLLM_SKIP_INIT_MEMORY_CHECK" "$VLLM_UTILS"; then
  /opt/env/bin/python3 - "$VLLM_UTILS" <<'PYPATCH1'
import sys
target = sys.argv[1]
src = open(target).read()
OLD = '    if init_snapshot.free_memory < requested_memory:\n        raise ValueError('
NEW = ('    import os as _os\n'
       '    if _os.environ.get("VLLM_SKIP_INIT_MEMORY_CHECK") == "1":\n'
       '        import logging; logging.getLogger(__name__).warning(\n'
       '            "VLLM_SKIP_INIT_MEMORY_CHECK=1 — skipping startup free-memory check")\n'
       '        return requested_memory\n'
       '    if init_snapshot.free_memory < requested_memory:\n'
       '        raise ValueError(')
if OLD not in src:
    print("[unholy-patch] patch1 anchor not found — skipping")
    sys.exit(0)
open(target, 'w').write(src.replace(OLD, NEW, 1))
print("[unholy-patch] patch1 (request_memory) applied to " + target)
PYPATCH1
fi

# ── GB10 UMA patch 2: determine_available_memory() post-profile assertion ────
# GB10 page cache release during profiling → current_free > init_free → OOM.
# Use current_free (post-release) as safe KV budget (~34 GiB).
GPU_WORKER=/opt/env/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py
if [ -f "$GPU_WORKER" ] && ! grep -q "_uma_early_ret" "$GPU_WORKER"; then
  /opt/env/bin/python3 - "$GPU_WORKER" <<'PYPATCH2'
import sys
target = sys.argv[1]
src = open(target).read()
OLD = '        assert self.init_snapshot.free_memory >= free_gpu_memory, ('
NEW = (
    '        if __import__("os").environ.get("VLLM_SKIP_INIT_MEMORY_CHECK") == "1":\n'
    '            if free_gpu_memory > self.init_snapshot.free_memory:  # _uma_early_ret\n'
    '                _kv = max(0, free_gpu_memory)\n'
    '                self.available_kv_cache_memory_bytes = _kv\n'
    '                return _kv\n'
    '        assert self.init_snapshot.free_memory >= free_gpu_memory, ('
)
if OLD not in src:
    print("[unholy-patch] patch2 anchor not found — skipping")
    sys.exit(0)
open(target, 'w').write(src.replace(OLD, NEW, 1))
print("[unholy-patch] patch2 (determine_available_memory) applied to " + target)
PYPATCH2
fi

# ── ROLE=worker dispatch ─────────────────────────────────────────────────────
if [ "${ROLE}" = "worker" ]; then
  : "${NODE_RANK:=1}"
  echo "[unholy] ROLE=worker backend=mp NODE_RANK=${NODE_RANK} → ${HEAD_ROCE_IP}:${MASTER_PORT:-25000}"
  exec vllm serve "${MODEL_CONTAINER_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 --port "${HOST_PORT:-8000}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP_SIZE:-2}" \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len "${MAX_MODEL_LEN:-262144}" \
    --max-num-seqs "${MAX_NUM_SEQS:-4}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
    --enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --distributed-executor-backend mp \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
    --default-chat-template-kwargs '{"thinking":true}' \
    --enable-flashinfer-autotune \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_NUM_TOKENS}}" \
    --nnodes 2 \
    --node-rank "${NODE_RANK}" \
    --master-addr "${HEAD_ROCE_IP}" \
    --master-port "${MASTER_PORT:-25000}" \
    --headless
fi

# ── ROLE=head ─────────────────────────────────────────────────────────────────
: "${NODE_RANK:=0}"
echo "[unholy] ROLE=head backend=mp NODE_RANK=${NODE_RANK} → ${HEAD_ROCE_IP}:${MASTER_PORT:-25000}"
exec vllm serve "${MODEL_CONTAINER_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 --port "${HOST_PORT:-8000}" \
  --trust-remote-code \
  --tensor-parallel-size "${TP_SIZE:-2}" \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs '{"thinking":true}' \
  --enable-flashinfer-autotune \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_NUM_TOKENS}}" \
  --nnodes 2 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${HEAD_ROCE_IP}" \
  --master-port "${MASTER_PORT:-25000}"
