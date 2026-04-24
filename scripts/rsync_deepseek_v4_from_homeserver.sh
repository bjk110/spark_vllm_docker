#!/usr/bin/env bash
# =============================================================================
# rsync_deepseek_v4_from_homeserver.sh
#
# Stage 1 of 2: homeserver → spark01 (public network).
# Run this ON homeserver (where /mnt/data/llm-models/deepseek-ai exists).
#
# Prereqs:
#   - Model already at /mnt/data/llm-models/deepseek-ai/deepseek-ai_DeepSeek-V4-Flash on homeserver
#   - ssh spark01 reachable
#   - spark01 has /mnt/data/llm-models/deepseek-ai/ writable and >= 160 GB free
#
# This only handles the V4-Flash directory. Leaves other deepseek-ai models alone.
# =============================================================================
set -euo pipefail

SRC_ROOT="/mnt/data/llm-models/deepseek-ai"
MODEL_DIR="deepseek-ai_DeepSeek-V4-Flash"
SRC="${SRC_ROOT}/${MODEL_DIR}/"
DST_HOST="${1:-spark01}"
DST_ROOT="/mnt/data/llm-models/deepseek-ai"
DST="${DST_HOST}:${DST_ROOT}/${MODEL_DIR}/"

if [ ! -d "${SRC}" ]; then
    echo "FATAL: source directory not found: ${SRC}" >&2
    exit 1
fi

echo "Source:      ${SRC}"
echo "Destination: ${DST}"
echo "Size:        $(du -sh "${SRC}" | cut -f1)"
echo

ssh "${DST_HOST}" "mkdir -p ${DST_ROOT}/${MODEL_DIR}"

rsync -aHAX --numeric-ids --info=progress2 --partial --inplace \
    "${SRC}" "${DST}"

echo
echo "--- Verify on ${DST_HOST} ---"
ssh "${DST_HOST}" "du -sh ${DST_ROOT}/${MODEL_DIR} && \
    ls ${DST_ROOT}/${MODEL_DIR}/*.safetensors 2>/dev/null | wc -l && \
    ls ${DST_ROOT}/${MODEL_DIR}/model.safetensors.index.json && \
    ls ${DST_ROOT}/${MODEL_DIR}/config.json"
echo "done."
