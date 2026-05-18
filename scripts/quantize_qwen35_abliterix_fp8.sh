#!/usr/bin/env bash
# =============================================================================
# quantize_qwen35_abliterix_fp8.sh
#
# Runs the FP8 quantization for wangzhang/Qwen3.5-122B-A10B-abliterix on
# spark01 inside an ephemeral vllm-spark container (no Ray, no serving).
#
# Uses safetensors-level direct quantization
# (quantize/quantize_qwen35_abliterix_fp8_direct.py) rather than llmcompressor:
# llmcompressor 0.10 pins transformers <=4.57.6, but the Qwen3.5 MoE class
# is only in transformers >=5.5 and the model ships no modeling .py files.
# The direct path needs only torch + safetensors, both already in the image.
#
# Inputs:
#   /home/bjk110/Documents/Models/wangzhang/Qwen3.5-122B-A10B-abliterix
# Outputs:
#   /home/bjk110/Documents/Models/wangzhang/Qwen3.5-122B-A10B-abliterix-FP8
#
# Run on: spark01 only (it's the spark with the BF16 weights).
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/bjk110/docker/vllm-spark}"
IMAGE="${VLLM_IMAGE:-ghcr.io/bjk110/vllm-spark:v021-ngc2603}"
INPUT_HOST="/home/bjk110/Documents/Models/wangzhang/Qwen3.5-122B-A10B-abliterix"
OUTPUT_HOST="/home/bjk110/Documents/Models/wangzhang/Qwen3.5-122B-A10B-abliterix-FP8"

cd "${REPO_DIR}"

if [[ ! -d "${INPUT_HOST}" ]]; then
    echo "[quantize] ERROR: BF16 input missing at ${INPUT_HOST}" >&2
    echo "[quantize]         Run scripts/sync_qwen35_abliterix_bf16_to_spark01.sh first." >&2
    exit 1
fi

mkdir -p "${OUTPUT_HOST}"

NAME="abliterix-quant-fp8"
docker rm -f "${NAME}" 2>/dev/null || true

echo "[quantize] === Launching quantization container ==="
echo "[quantize] image:   ${IMAGE}"
echo "[quantize] input:   ${INPUT_HOST}"
echo "[quantize] output:  ${OUTPUT_HOST}"
echo "[quantize] (safetensors-level, per-channel FP8 W8A8 with dynamic act)"

docker run --rm \
    --name "${NAME}" \
    --gpus all \
    --ipc host \
    --shm-size 16g \
    -v /home/bjk110/Documents/Models:/models:rw \
    -v "${REPO_DIR}/quantize":/quantize:ro \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --entrypoint python3 \
    "${IMAGE}" /quantize/quantize_qwen35_abliterix_fp8_direct.py \
        --input  /models/wangzhang/Qwen3.5-122B-A10B-abliterix \
        --output /models/wangzhang/Qwen3.5-122B-A10B-abliterix-FP8

echo "[quantize] === Done ==="
du -sh "${OUTPUT_HOST}"
ls -la "${OUTPUT_HOST}" | head -20
