#!/usr/bin/env bash
# =============================================================================
# rsync_deepseek_v4_spark01_to_spark02_rdma.sh
#
# Stage 2 of 2: spark01 → spark02 over RoCE (10.10.10.0/24).
# Run this ON spark01.
#
# Prereqs:
#   - Model already at /mnt/data/llm-models/deepseek-ai/deepseek-ai_DeepSeek-V4-Flash on spark01
#   - spark02 RoCE IP 10.10.10.2 reachable via ssh
#   - spark02 has /mnt/data/llm-models/deepseek-ai/ writable and >= 160 GB free
#
# Uses Compression=no / IPQoS=throughput to avoid CPU bottleneck on RoCE path.
# =============================================================================
set -euo pipefail

SRC_ROOT="/home/bjk110/Documents/Models/deepseek-ai"
MODEL_DIR="deepseek-ai_DeepSeek-V4-Flash"
SRC="${SRC_ROOT}/${MODEL_DIR}/"
DST_IP="${1:-10.10.10.2}"
DST_ROOT="/home/bjk110/Documents/Models/deepseek-ai"
DST="${DST_IP}:${DST_ROOT}/${MODEL_DIR}/"

if [ ! -d "${SRC}" ]; then
    echo "FATAL: source directory not found on spark01: ${SRC}" >&2
    exit 1
fi

echo "Source:      spark01:${SRC}"
echo "Destination: ${DST}  (over RoCE)"
echo "Size:        $(du -sh "${SRC}" | cut -f1)"
echo

ssh "${DST_IP}" "mkdir -p ${DST_ROOT}/${MODEL_DIR}"

rsync -aHAX --numeric-ids --info=progress2 --partial --inplace \
    -e "ssh -o Compression=no -o IPQoS=throughput" \
    "${SRC}" "${DST}"

echo
echo "--- Verify on ${DST_IP} ---"
ssh "${DST_IP}" "du -sh ${DST_ROOT}/${MODEL_DIR} && \
    ls ${DST_ROOT}/${MODEL_DIR}/*.safetensors 2>/dev/null | wc -l && \
    ls ${DST_ROOT}/${MODEL_DIR}/model.safetensors.index.json && \
    ls ${DST_ROOT}/${MODEL_DIR}/config.json"

echo
echo "--- Manifest check (shard count vs index) ---"
ssh "${DST_IP}" "python3 -c \"
import json, glob, os
idx = json.load(open('${DST_ROOT}/${MODEL_DIR}/model.safetensors.index.json'))
expected = set(idx['weight_map'].values())
present  = set(os.path.basename(p) for p in glob.glob('${DST_ROOT}/${MODEL_DIR}/*.safetensors'))
missing  = expected - present
extra    = present - expected
print(f'expected shards: {len(expected)}')
print(f'present shards:  {len(present)}')
print(f'missing:         {sorted(missing) if missing else \\\"none\\\"}')
print(f'extra:           {sorted(extra) if extra else \\\"none\\\"}')
\""

echo "done."
