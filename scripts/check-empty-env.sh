#!/bin/bash
# =============================================================================
# check-empty-env.sh
#
# Lint presets/*.env against docker-compose.yml's `${VAR:-}` substitutions.
#
# entrypoints/entrypoint.sh now unsets any of these vars when docker-compose
# renders them as an empty string ("VAR=" -> unset), so a plain empty value is
# no longer a hard crash for any preset (see issue #14). However, NVFP4
# presets (--quantization nvfp4 / VLLM_USE_FLASHINFER_MOE_FP4=1) genuinely
# need explicit values for the FlashInfer/Triton-DG/CUDA-graph knobs below --
# if those render empty, the NVFP4 GEMM/attention/CUDA-graph paths fall back
# to upstream defaults that are not validated on GB10/sm_121.
#
# This script fails (exit 1) only when an NVFP4-sensitive preset is missing
# one of those explicit values. For all other presets it just reports which
# optional vars are empty (informational; entrypoint.sh sanitizes them).
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
    echo "[check-empty-env] docker not found, skipping" >&2
    exit 0
fi

# Vars that have an empty `${VAR:-}` default in docker-compose.yml and that
# NVFP4 presets must pin to a real value (see entrypoint.sh sanitizer list
# and issue #14).
NVFP4_REQUIRED_VARS=(
    VLLM_NVFP4_GEMM_BACKEND
    FLASHINFER_CUDA_ARCH_LIST
    TORCH_CUDA_ARCH_LIST
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
)

# Other optional vars worth reporting on (sanitized when empty, never fatal).
INFO_VARS=(
    VLLM_ATTENTION_BACKEND
)

fail=0

for preset in presets/*.env; do
    name=$(basename "$preset")

    config_out=$(MODEL_PATH=/tmp ENTRYPOINT_FILE=./entrypoints/entrypoint.sh \
        docker compose --env-file "$preset" --profile head config 2>/dev/null) || {
        echo "[check-empty-env] WARN ${name}: docker compose config failed, skipping" >&2
        continue
    }

    # Only flag presets that perform RUNTIME NVFP4 quantization
    # (--quantization nvfp4). Pre-quantized compressed-tensors/modelopt NVFP4
    # checkpoints (redhatai-*, wangzhang-*-nvfp4*, prismascout-*) use a
    # different quant path and are documented as verified-working without
    # these pins -- don't fail the lint for those.
    is_nvfp4=0
    if grep -qiE -- '--quantization[[:space:]]+nvfp4' "$preset"; then
        is_nvfp4=1
    fi

    for var in "${NVFP4_REQUIRED_VARS[@]}"; do
        value=$(echo "$config_out" | grep -E "^\s*${var}:" | head -1 | sed -E 's/^[^:]*:\s*//')
        if [ "$value" = '""' ] || [ -z "$value" ]; then
            if [ "$is_nvfp4" -eq 1 ]; then
                echo "[check-empty-env] ERROR ${name}: ${var} is empty in an NVFP4 preset"
                fail=1
            else
                echo "[check-empty-env] info ${name}: ${var} empty (sanitized by entrypoint.sh)"
            fi
        fi
    done

    for var in "${INFO_VARS[@]}"; do
        value=$(echo "$config_out" | grep -E "^\s*${var}:" | head -1 | sed -E 's/^[^:]*:\s*//')
        if [ "$value" = '""' ] || [ -z "$value" ]; then
            echo "[check-empty-env] info ${name}: ${var} empty (sanitized by entrypoint.sh)"
        fi
    done
done

if [ "$fail" -ne 0 ]; then
    echo "[check-empty-env] FAILED: NVFP4 preset(s) missing required env values" >&2
    exit 1
fi

echo "[check-empty-env] OK"
