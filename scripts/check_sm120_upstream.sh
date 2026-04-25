#!/usr/bin/env bash
# =============================================================================
# check_sm120_upstream.sh — watch upstream refs that can unblock the
# DeepSeek-V4-Flash SM120 POC. Read NOTES.deepseek-v4-sm120-poc.md first.
#
# Compares the commits currently pinned in Dockerfile.deepseek-v4 (and the
# DeepGEMM tag pinned in the vLLM source's cmake/external_projects/deepgemm.cmake)
# against the heads of:
#   - vllm-project/vllm  pull/40852/head
#   - jasl/vllm          ds4-sm120-prototype
#   - jasl/DeepGEMM      sm120
#
# Also checks whether the fp8_gemm_nt SM12x blocker is gone. Specifically
# looks for an arch_major == 12 case in jasl/DeepGEMM csrc/utils/layout.hpp.
#
# Requires: gh CLI authenticated (`gh auth status`).
#
# Exit codes:
#   0   no movement, nothing to do
#   1   gh not available / not authenticated
#   2   one or more upstream refs moved (Dockerfile pin candidate to update)
#   3   blocker resolved AND refs moved — re-run the POC
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${REPO_DIR}/Dockerfile.deepseek-v4"
NOTES="${REPO_DIR}/NOTES.deepseek-v4-sm120-poc.md"

if ! command -v gh >/dev/null 2>&1; then
    echo "FATAL: gh CLI not installed" >&2
    exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
    echo "FATAL: gh auth status failed (run 'gh auth login')" >&2
    exit 1
fi
if [ ! -f "${DOCKERFILE}" ]; then
    echo "FATAL: ${DOCKERFILE} not found" >&2
    exit 1
fi

# ---- Pinned values from Dockerfile.deepseek-v4 ----
PINNED_VLLM_COMMIT=$(awk -F= '/^ARG VLLM_COMMIT=/{print $2}' "${DOCKERFILE}" | tr -d ' "')
PINNED_VLLM_REF=$(awk -F= '/^ARG VLLM_REF=/{print $2}' "${DOCKERFILE}" | tr -d ' "')

# ---- Heads of interest ----
PR_HEAD=$(gh api repos/vllm-project/vllm/pulls/40852 --jq .head.sha)
JASL_VLLM_HEAD=$(gh api 'repos/jasl/vllm/branches/ds4-sm120-prototype' --jq .commit.sha)
JASL_DG_HEAD=$(gh api 'repos/jasl/DeepGEMM/branches/sm120' --jq .commit.sha)

# ---- DeepGEMM commit pinned by the vLLM source we use ----
# The pin lives inside the vLLM source's cmake (cmake/external_projects/deepgemm.cmake).
# Fetch that file at the pinned vLLM commit and extract GIT_TAG.
PINNED_DG_COMMIT=$(gh api "repos/jasl/vllm/contents/cmake/external_projects/deepgemm.cmake?ref=${PINNED_VLLM_COMMIT}" \
    --jq .content 2>/dev/null | base64 -d 2>/dev/null \
    | awk '/jasl\/DeepGEMM/{flag=1} flag && /GIT_TAG/{print $2; exit}' | tr -d ' ' || echo "<unknown>")

echo "============================================================"
echo "Pinned in this branch:"
echo "  vLLM    ref=${PINNED_VLLM_REF}  commit=${PINNED_VLLM_COMMIT}"
echo "  DeepGEMM (via vLLM cmake)       commit=${PINNED_DG_COMMIT}"
echo
echo "Upstream now:"
echo "  vllm-project/vllm  pull/40852/head        = ${PR_HEAD}"
echo "  jasl/vllm          ds4-sm120-prototype    = ${JASL_VLLM_HEAD}"
echo "  jasl/DeepGEMM      sm120 head             = ${JASL_DG_HEAD}"
echo "============================================================"

MOVED=0
[ "${PINNED_VLLM_COMMIT}" != "${JASL_VLLM_HEAD}" ] && \
    echo "* jasl/vllm ds4-sm120-prototype moved: bump VLLM_COMMIT to ${JASL_VLLM_HEAD}" && MOVED=1
[ "${PINNED_VLLM_COMMIT}" != "${PR_HEAD}" ] && \
    echo "* PR #40852 head ${PR_HEAD} differs from pin (PR may have rebased onto vllm main)" && MOVED=1
[ "${PINNED_DG_COMMIT}" != "${JASL_DG_HEAD}" ] && \
    echo "* jasl/DeepGEMM sm120 moved: vLLM cmake pin should advance to ${JASL_DG_HEAD}" && MOVED=1

# ---- Blocker check ----
# fp8_gemm_nt SM12x is blocked while csrc/utils/layout.hpp::get_default_recipe
# does not handle arch_major == 12. Probe the head.
DG_LAYOUT=$(gh api "repos/jasl/DeepGEMM/contents/csrc/utils/layout.hpp?ref=${JASL_DG_HEAD}" \
    --jq .content | base64 -d)
BLOCKER_CLEARED=0
if echo "${DG_LAYOUT}" | grep -q "arch_major == 12"; then
    echo
    echo "*** BLOCKER LIKELY CLEARED ***"
    echo "    jasl/DeepGEMM ${JASL_DG_HEAD} now references arch_major == 12 in"
    echo "    csrc/utils/layout.hpp. Inspect get_default_recipe / "
    echo "    transform_sf_into_required_layout for fp8_gemm_nt support, then"
    echo "    re-run the POC per NOTES.deepseek-v4-sm120-poc.md."
    BLOCKER_CLEARED=1
fi

if [ ${MOVED} -eq 0 ] && [ ${BLOCKER_CLEARED} -eq 0 ]; then
    echo
    echo "No movement. Pin still matches upstream; SM12x fp8_gemm_nt still gated."
    echo "See ${NOTES} for re-test triggers."
    exit 0
fi

if [ ${BLOCKER_CLEARED} -eq 1 ]; then
    exit 3
fi
exit 2
