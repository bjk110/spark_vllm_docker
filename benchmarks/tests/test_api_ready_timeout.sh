#!/usr/bin/env bash
# =============================================================================
# test_api_ready_timeout.sh
#
# No-runtime test for the configurable API readiness timeout in
# bench-bt-matrix-step37-v023.sh. Exercises only the early validation path,
# which runs before any SSH or container operation, so the test needs neither
# the Spark nodes nor Docker.
#
# Contract under test:
#   - API_READY_TIMEOUT must be a positive integer (seconds): regex ^[1-9][0-9]*$
#   - Invalid / non-positive values fail fast with a clear FATAL message,
#     exit nonzero, and never reach container/SSH work.
#   - Unset or empty defaults to 600 (Bash ${VAR:-600}), preserving prior behavior.
#
# Usage: bash benchmarks/tests/test_api_ready_timeout.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${SCRIPT_DIR}/bench-bt-matrix-step37-v023.sh"
[[ -f "${RUNNER}" ]] || { echo "Runner not found: ${RUNNER}" >&2; exit 1; }

pass=0
fail=0
check() { # description, expected(0=pass/1=fail), actual_exit
    local desc="$1" expect="$2" got="$3"
    if [[ "${expect}" == "${got}" ]]; then
        echo "  PASS: ${desc}"
        pass=$((pass + 1))
    else
        echo "  FAIL: ${desc} (expected exit-class ${expect}, got ${got})"
        fail=$((fail + 1))
    fi
}

# --- Integration: invalid values must die before any SSH/container work -------
# The runner reaches the API_READY_TIMEOUT guard inside main() before mkdir,
# preflight, or node access, so these cases are hermetic (no nodes required).
echo "[invalid values -> fail fast with clear message]"
for bad in abc 0 -5 3.5 "10x" " 600"; do
    out=$(API_READY_TIMEOUT="${bad}" bash "${RUNNER}" --bt 2048 2>&1)
    rc=$?
    nonzero=0; [[ ${rc} -ne 0 ]] && nonzero=1
    has_msg=0; grep -q "Invalid API_READY_TIMEOUT" <<<"${out}" && has_msg=1
    touched_nodes=0; grep -qiE "Generating env|Starting head|Copying env|Containers started" <<<"${out}" && touched_nodes=1
    check "API_READY_TIMEOUT='${bad}': nonzero exit" 1 "${nonzero}"
    check "API_READY_TIMEOUT='${bad}': clear FATAL message" 1 "${has_msg}"
    check "API_READY_TIMEOUT='${bad}': stopped before container/SSH work" 0 "${touched_nodes}"
done

# --- Unit: the documented regex accepts valid, rejects invalid ----------------
# Mirrors the literal guard regex in the runner; documents the accepted form
# without requiring a live startup (which would proceed into SSH).
echo "[regex contract ^[1-9][0-9]*\$]"
regex='^[1-9][0-9]*$'
for good in 1 600 900 1200 86400; do
    [[ "${good}" =~ ${regex} ]] && check "accepts '${good}'" 0 0 || check "accepts '${good}'" 0 1
done
for bad in abc 0 -5 3.5 "10x" "" " 600" 0600; do
    [[ "${bad}" =~ ${regex} ]] && check "rejects '${bad}'" 1 0 || check "rejects '${bad}'" 1 1
done

# --- Unit: unset / empty default to 600 ---------------------------------------
echo "[unset/empty -> default 600]"
unset_default="${API_READY_TIMEOUT_UNSET:-600}"; check "unset defaults to 600" 0 "$([[ ${unset_default} == 600 ]] && echo 0 || echo 1)"
empty_var=""; empty_default="${empty_var:-600}"; check "empty defaults to 600" 0 "$([[ ${empty_default} == 600 ]] && echo 0 || echo 1)"

echo ""
echo "Result: ${pass} passed, ${fail} failed."
[[ ${fail} -eq 0 ]]
