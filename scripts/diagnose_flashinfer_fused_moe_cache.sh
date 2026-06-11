#!/bin/bash
# Diagnostic (read-only by default): why does fused_moe_120 recompile at
# runtime despite an AOT-prebaked /root/.cache/flashinfer/.../fused_moe_120.so
# in the image?
#
# Runs a throwaway `docker run --rm` from the AOT image (no vLLM serve, no
# model load) and inspects the FlashInfer JIT cache state. Steps 1/2/4 do
# not trigger any compilation. Step 3 is a ninja dry-run (-n), also no
# compilation. Step 5 (optional, off by default) calls build_and_load() and
# CAN trigger nvcc/cicc compilation -- enable with RUN_STEP5=1.
#
# Usage (on spark01/spark02):
#   ./scripts/diagnose_flashinfer_fused_moe_cache.sh [IMAGE]
#   RUN_STEP5=1 ./scripts/diagnose_flashinfer_fused_moe_cache.sh [IMAGE]
set -euo pipefail

IMAGE="${1:-vllm-spark:v022-d568-fi-aot}"
RUN_STEP5="${RUN_STEP5:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm \
  --gpus all \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1 \
  -e RUN_STEP5="$RUN_STEP5" \
  -v "$SCRIPT_DIR/diagnose_flashinfer_fused_moe_cache.py:/diag.py:ro" \
  --entrypoint bash \
  "$IMAGE" -c '
set -x

echo "########## STEPS 1+2+4 (spec inspection, mtime diff, env) ##########"
python3 /diag.py

echo
echo "########## STEP 3: ninja -n -d explain for fused_moe_120 ##########"
CACHE_ROOT="/root/.cache/flashinfer"
NINJA_FILE=$(find "$CACHE_ROOT" -path "*fused_moe_120*build.ninja" 2>/dev/null | head -1)
echo "NINJA_FILE=$NINJA_FILE"
if [ -n "$NINJA_FILE" ]; then
  NINJA_DIR=$(dirname "$NINJA_FILE")
  echo "--- ls -la $NINJA_DIR ---"
  ls -la "$NINJA_DIR"
  echo "--- ninja -C $NINJA_DIR -n -d explain (tail -200) ---"
  ninja -C "$NINJA_DIR" -n -d explain 2>&1 | tail -200
else
  echo "build.ninja for fused_moe_120 not found under $CACHE_ROOT"
  echo "--- find $CACHE_ROOT -iname \"*fused_moe_120*\" ---"
  find "$CACHE_ROOT" -iname "*fused_moe_120*" 2>/dev/null
fi

echo
if [ "$RUN_STEP5" = "1" ]; then
  echo "########## STEP 5 (OPTIONAL): FLASHINFER_DISABLE_JIT=1 build_and_load() ##########"
  echo "WARNING: this can trigger nvcc/cicc compilation if is_aot is False."
  FLASHINFER_DISABLE_JIT=1 python3 - <<PY
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module
spec = gen_cutlass_fused_moe_sm120_module(use_fast_build=False)
for attr in ("is_aot", "is_compiled", "aot_path", "jit_library_path"):
    print(f"{attr}:", getattr(spec, attr, "N/A"))
try:
    m = spec.build_and_load()
    print("loaded:", m)
except Exception as ex:
    print("build_and_load failed:", repr(ex))
PY
else
  echo "########## STEP 5 skipped (set RUN_STEP5=1 to enable) ##########"
fi
'
