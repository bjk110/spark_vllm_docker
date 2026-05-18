#!/usr/bin/env bash
# =============================================================================
# sync_qwen36_prismascout_to_sparks.sh
#
# Two-stage rsync distribution for Qwen3.6-27B PrismaSCOUT NVFP4-BF16:
#   Stage 1: homeserver  → spark01   via mgmt LAN 192.168.0.200
#   Stage 2: spark01     → spark02   via RDMA   10.10.10.2
#
# The Stage-2 hop is launched by SSH'ing into spark01 and running rsync there,
# so the bytes traverse the 200 Gbps RDMA link directly (not the mgmt LAN, not
# homeserver). The remote rsync has the same path on both Spark nodes.
#
# Run on: homeserver only.
# Auth:   key-based SSH for bjk110@spark01 (mgmt) and bjk110@10.10.10.2 (RDMA)
#         must already work from each origin host.
# =============================================================================
set -euo pipefail

SSH_USER="${SSH_USER:-bjk110}"

# ---- Path layout ----
HOMESERVER_SRC="/mnt/data/llm-models/google/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm"
SPARK01_DST="/home/bjk110/Documents/Models/google/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm"
SPARK02_DST="${SPARK01_DST}"

# ---- Network endpoints ----
SPARK01_MGMT="192.168.0.200"          # Stage 1: homeserver → spark01 (mgmt LAN)
SPARK02_RDMA="10.10.10.2"             # Stage 2: spark01 → spark02 (RDMA)

# rsync flags justification:
#   -a         archive (perms/times/links — but we expect real files only)
#   -v -h      verbose + human-readable
#   --info=progress2  single-line aggregate progress (good for big trees)
#   --partial  keep partial files on interrupt (resume support)
#   --append-verify  resume by appending + verify checksum of resumed portion
#                    (catches the rare case where the source changed mid-flight)
RSYNC_OPTS=(-avh --info=progress2 --partial --append-verify)

# Trailing slash matters: copy contents of SRC into DST (so DST itself is
# the model dir, not DST/Qwen3.6.../Qwen3.6.../).
SRC_TRAIL="${HOMESERVER_SRC%/}/"
DST_TRAIL_01="${SPARK01_DST%/}/"
DST_TRAIL_02="${SPARK02_DST%/}/"

# ---- Sanity ----
if [[ ! -d "${HOMESERVER_SRC}" ]]; then
    echo "[sync] ERROR: source missing on homeserver: ${HOMESERVER_SRC}" >&2
    echo "[sync]        Run scripts/download_qwen36_prismascout.sh first." >&2
    exit 1
fi

echo "[sync] === Pre-flight ==="
echo -n "[sync] homeserver source size:  "
du -sh "${HOMESERVER_SRC}" | awk '{print $1}'

# ---- Stage 1: homeserver → spark01 (mgmt LAN) ----
echo ""
echo "[sync] === Stage 1: homeserver → ${SPARK01_MGMT} (spark01, mgmt) ==="
ssh "${SSH_USER}@${SPARK01_MGMT}" "mkdir -p '${SPARK01_DST}'"
rsync "${RSYNC_OPTS[@]}" \
    "${SRC_TRAIL}" \
    "${SSH_USER}@${SPARK01_MGMT}:${DST_TRAIL_01}"

echo -n "[sync] spark01 receive size:    "
ssh "${SSH_USER}@${SPARK01_MGMT}" "du -sh '${SPARK01_DST}'" | awk '{print $1}'

# ---- Stage 2: spark01 → spark02 (RDMA) ----
# rsync from inside spark01 over RDMA. We must ensure rsync exists on both;
# the remote `rsync` is invoked by SSH'ing through spark01's perspective.
echo ""
echo "[sync] === Stage 2: spark01 → ${SPARK02_RDMA} (spark02, RDMA) ==="
# Stream the remote script via stdin and explicitly close stdin afterward
# (</dev/null on the outer ssh) so the inner ssh doesn't slurp the rest of
# the heredoc. The inner ssh ALSO gets -n so its own stdin is /dev/null and
# it can't consume parent bash's stdin (otherwise the rsync line below gets
# silently eaten and Stage 2 finishes as a no-op).
ssh "${SSH_USER}@${SPARK01_MGMT}" bash -s <<REMOTE
set -euo pipefail
mkdir -p '${SPARK01_DST}'
ssh -n -o StrictHostKeyChecking=accept-new "${SSH_USER}@${SPARK02_RDMA}" "mkdir -p '${SPARK02_DST}'"
rsync ${RSYNC_OPTS[@]} \
    '${DST_TRAIL_01}' \
    "${SSH_USER}@${SPARK02_RDMA}:${DST_TRAIL_02}"
REMOTE

echo -n "[sync] spark02 receive size:    "
ssh "${SSH_USER}@${SPARK01_MGMT}" \
    ssh -o StrictHostKeyChecking=accept-new "${SSH_USER}@${SPARK02_RDMA}" \
    "du -sh '${SPARK02_DST}'" | awk '{print $1}'

echo ""
echo "[sync] === Done ==="
echo "[sync] Model is now resident on spark01 and spark02 at:"
echo "[sync]   ${SPARK01_DST}"
echo "[sync] Next: scripts/start_qwen36_prismascout_tp2.sh"
