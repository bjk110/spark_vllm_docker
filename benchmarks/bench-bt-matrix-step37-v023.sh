#!/usr/bin/env bash
# =============================================================================
# bench-bt-matrix-step37-v023.sh
#
# Purpose
#   Benchmark MAX_NUM_BATCHED_TOKENS effect on Step-3.7-Flash-NVFP4 prefill
#   throughput under vLLM 0.23.0 on dual DGX Spark GB10.
#
#   Background: bt=256 produced ~538 t/s vs ~1251 t/s on v022 baseline.
#   Single variable: only MAX_NUM_BATCHED_TOKENS changes across runs.
#   All other parameters fixed (see .local/env/step37/bt-matrix-base.env).
#
# Usage (run from homeserver in /home/bjk110/docker/vllm-spark/):
#   bash benchmarks/bench-bt-matrix-step37-v023.sh --bt <value> [OPTIONS]
#
# IMPORTANT: --bt is required for benchmark runs. Running a multi-value sweep
#   auto-stops containers between runs and requires both nodes to have >50 GiB
#   free AFTER container stop. Reboot between runs is strongly recommended to
#   avoid GB10 UMA driver memory accumulation.
#   Use --all to intentionally run the full matrix.
#
# Options:
#   --bt <values>              Comma-separated bt values to run (REQUIRED for
#                              normal run; optional with --preflight-only or
#                              --validate-existing-container)
#                              Example: --bt 2048
#                              Example: --bt 2048,4096,8192
#   --all                      Run all matrix values (256,512,...,32768).
#                              Requires explicit confirmation unless --dry-run.
#   --template <path>          Override template env file.
#                              Default: bt-matrix-base.env (Series B, EP-on)
#                              Use bt-matrix-series-a-ep-off.env for Series A.
#   --runs <n>                 llama-benchy runs per test (default: 3)
#   --skip-bt <vals>           Comma-separated bt values to skip (with --all)
#   --dry-run                  Print commands without executing
#   --no-stop                  Skip container stop between runs (testing only)
#   --result-dir <d>           Override result directory
#   --expected-ep <on|off|unknown>
#                              Expected EP state. Runner halts if startup logs
#                              contradict this. Checked against the entrypoint
#                              command line (--enable-expert-parallel presence).
#                              NOTE: vLLM 0.23 does not log expert_parallel_size=1
#                              when EP is disabled, so entrypoint command is
#                              the authoritative source.
#                              Default: unknown (log check only, no halt).
#   --config-label <str>       Short label for this run (recorded in metadata).
#                              Example: v023-triton-marlin-ep-off-bt2048
#   --continue-on-bench-fail   Do not halt matrix on llama-benchy request
#                              failure only (exit 3). NEVER applies to:
#                              topology mismatch, backend mismatch, EP mismatch,
#                              startup failure, memory threshold. These are
#                              always fatal.
#   --continue-on-fail         Deprecated alias for --continue-on-bench-fail.
#   --preflight-only           Read-only pre-start checks: SSH, boot ID, uptime,
#                              memory state, container state, stale process
#                              detection. No container ops. Exits after checks.
#                              Safe to run with containers up.
#   --validate-existing-container
#                              Read-only backend detection against the currently
#                              running head container. Applies MARLIN, TRITON_ATTN,
#                              and EP detection. No container stop/start.
#   --server-only              Start the server and validate backends, then exit
#                              without running any benchmark or stopping containers.
#                              Server keeps running after script exits (docker
#                              compose up -d). Useful for manual testing.
#                              Requires --bt.
#   --supplement               Run extended correctness (max_tokens=2048, 2 runs
#                              each) + decode-only bench (pp=1, tg=32) in a single
#                              server lifecycle. Saves to a separate supplement
#                              run ID. Requires --bt. Use --decode-runs to override
#                              the number of decode runs (default: 5).
#   --decode-runs <n>          Number of llama-benchy runs for decode-only bench in
#                              supplement mode (default: 5).
#   --supplement-of <run_id>   Original run ID that this supplement extends
#                              (recorded in metadata only; no runtime effect).
#                              Use --expected-ep to gate on EP state.
#
# Memory checks:
#   SAFE_MEM_GIB (default: 50) applies ONLY to the post-stop check. It is NOT
#   applied to the pre-start check. On GB10 UMA, a running vLLM server holding
#   ~75 GiB will leave only ~46 GiB free — this is expected and NOT a failure.
#   Pre-start memory is recorded in metadata (spark0X_free_gib_before) for
#   reference; it is informational only, not a gate.
#
# Safety rules (enforced):
#   - Never modifies production preset (presets/step37-flash-nvfp4-tp2.env)
#   - Never reboots, destroys volumes, or docker system prune
#   - Halts if post-stop memory does not recover above SAFE_MEM_GIB threshold
#   - Halts if --expected-ep contradicts observed EP state in startup logs
#   - Each run uses a separate disposable env file (deleted after success)
#   - Never auto-promotes a result to production recommendation
#   - Warns if running containers detected before starting
#
# Requirements:
#   - SSH aliases: spark01, spark02 (configured in ~/.ssh/config)
#   - llama-benchy installed on spark01: ~/.local/bin/llama-benchy
#   - Template env: .local/env/step37/bt-matrix-base.env
#   - Model weights: /home/bjk110/Documents/Models/stepfun-ai/Step-3.7-Flash-NVFP4
#     (on spark01 AND spark02, or NFS-shared)
#   - docker compose available on spark01 and spark02
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Default template: EP-on (Series B). For EP-off (Series A) runs, override with --template.
TEMPLATE_ENV="${REPO_ROOT}/.local/env/step37/bt-matrix-base.env"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.yml"
RESULT_DIR="${REPO_ROOT}/benchmarks/results/bt-matrix"
LOG_DIR="${REPO_ROOT}/benchmarks/results/bt-matrix/logs"

SPARK01=spark01
SPARK02=spark02

# Remote paths (same tree on each Spark node)
REMOTE_REPO_ROOT="/home/bjk110/docker/vllm-spark"
REMOTE_COMPOSE_FILE="${REMOTE_REPO_ROOT}/docker-compose.yml"

SERVED_MODEL_NAME="stepfun-ai/Step-3.7-Flash-NVFP4"
TOKENIZER_PATH="/home/bjk110/Documents/Models/stepfun-ai/Step-3.7-Flash-NVFP4"
API_BASE="http://localhost:8000/v1"
API_URL="http://localhost:8000"

# llama-benchy settings
LLAMA_BENCHY_BIN="${HOME}/.local/bin/llama-benchy"
BENCH_RUNS=3
# Depth sweep: skip d32768 because MAX_MODEL_LEN=32768 and llama-benchy
# adds output tokens (32), which overflows the limit by 1 token.
BENCH_PP=2048
BENCH_TG=32
BENCH_DEPTHS="0 4096 8192 16384"

# Post-stop memory safety threshold. Applied ONLY after container stop —
# NOT to the pre-start check. On GB10 UMA, a running vLLM server occupies
# ~75 GiB, leaving ~46 GiB free; this is expected and not a failure here.
SAFE_MEM_GIB=50

# Default bt matrix (run all unless --bt or --skip-bt override)
DEFAULT_BT_VALUES=(256 512 1024 2048 4096 8192 16384 32768)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BT_OVERRIDE=""
SKIP_BT=""
DRY_RUN=false
NO_STOP=false
RUN_ALL=false
EXPECTED_EP="unknown"
CONFIG_LABEL=""
CONTINUE_ON_BENCH_FAIL=false
PREFLIGHT_ONLY=false
VALIDATE_EXISTING=false
SERVER_ONLY_MODE=false
SUPPLEMENT_MODE=false
SUPPLEMENT_DECODE_RUNS=5
SUPPLEMENT_OF=""
# DECODE_CONTEXT_MODE: controls which decode contexts supplement runs measure.
#   pp1      - tg32 after pp=1 context (negligible prefill); default for --supplement
#   pp2048   - tg32 after pp=2048 context (same as full run); comparable to original bench
#   both     - pp=1 first, then pp=2048 in a second llama-benchy pass
# For full runs (run_bt), decode is always pp=2048 (embedded in the bench sweep).
DECODE_CONTEXT_MODE="pp1"
DECODE_WARMUP_RUNS=0
RUN_CORRECTNESS=true
AUTO_NEXT_BT=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bt)                    BT_OVERRIDE="$2"; shift 2 ;;
        --all)                   RUN_ALL=true; shift ;;
        --template)              TEMPLATE_ENV="$2"; shift 2 ;;
        --runs)                  BENCH_RUNS="$2"; shift 2 ;;
        --skip-bt)               SKIP_BT="$2"; shift 2 ;;
        --dry-run)               DRY_RUN=true; shift ;;
        --no-stop)               NO_STOP=true; shift ;;
        --result-dir)            RESULT_DIR="$2"; shift 2 ;;
        --expected-ep)           EXPECTED_EP="$2"; shift 2 ;;
        --config-label)          CONFIG_LABEL="$2"; shift 2 ;;
        --continue-on-bench-fail) CONTINUE_ON_BENCH_FAIL=true; shift ;;
        --continue-on-fail)      CONTINUE_ON_BENCH_FAIL=true
                                 echo "[WARN] --continue-on-fail is deprecated; use --continue-on-bench-fail" >&2
                                 shift ;;
        --preflight-only)        PREFLIGHT_ONLY=true; shift ;;
        --validate-existing-container) VALIDATE_EXISTING=true; shift ;;
        --server-only)           SERVER_ONLY_MODE=true; shift ;;
        --supplement)            SUPPLEMENT_MODE=true; shift ;;
        --decode-runs)           SUPPLEMENT_DECODE_RUNS="$2"; shift 2 ;;
        --supplement-of)         SUPPLEMENT_OF="$2"; shift 2 ;;
        --decode-context)        DECODE_CONTEXT_MODE="$2"; shift 2 ;;
        --decode-warmup-runs)    DECODE_WARMUP_RUNS="$2"; shift 2 ;;
        --no-correctness)        RUN_CORRECTNESS=false; shift ;;
        --no-auto-next-bt)       AUTO_NEXT_BT=false; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# --bt is required for normal benchmark, supplement, and server-only runs, but NOT for preflight/validate
if [[ -z "${BT_OVERRIDE}" ]] && ! ${RUN_ALL} && ! ${PREFLIGHT_ONLY} && ! ${VALIDATE_EXISTING}; then
    echo "ERROR: --bt <value(s)> is required." >&2
    echo "  Single run:       --bt 2048" >&2
    echo "  Multi-value:      --bt 2048,4096,8192" >&2
    echo "  Full matrix:      --all (requires confirmation)" >&2
    echo "  Pre-flight only:  --preflight-only (no --bt needed)" >&2
    echo "  Validate live:    --validate-existing-container (no --bt needed)" >&2
    exit 1
fi

# Validate --expected-ep value
case "${EXPECTED_EP}" in
    on|off|unknown) ;;
    *) echo "ERROR: --expected-ep must be 'on', 'off', or 'unknown'. Got: '${EXPECTED_EP}'" >&2; exit 1 ;;
esac

# Validate --decode-context value
case "${DECODE_CONTEXT_MODE}" in
    pp1|pp2048|both) ;;
    *) echo "ERROR: --decode-context must be 'pp1', 'pp2048', or 'both'. Got: '${DECODE_CONTEXT_MODE}'" >&2; exit 1 ;;
esac

# Build effective bt list (used for normal runs and informational in preflight)
if [[ -n "${BT_OVERRIDE}" ]]; then
    IFS=',' read -r -a BT_VALUES <<< "${BT_OVERRIDE}"
elif ${RUN_ALL}; then
    BT_VALUES=("${DEFAULT_BT_VALUES[@]}")
else
    BT_VALUES=()
fi

# Filter out skip list
if [[ -n "${SKIP_BT}" ]] && [[ ${#BT_VALUES[@]} -gt 0 ]]; then
    IFS=',' read -r -a SKIP_ARRAY <<< "${SKIP_BT}"
    FILTERED=()
    for bt in "${BT_VALUES[@]}"; do
        skip=false
        for s in "${SKIP_ARRAY[@]}"; do
            [[ "${bt}" == "${s}" ]] && skip=true && break
        done
        ${skip} || FILTERED+=("${bt}")
    done
    BT_VALUES=("${FILTERED[@]}")
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] WARN: $*" >&2; }
die() { echo "[$(date '+%H:%M:%S')] FATAL: $*" >&2; exit 1; }

run() {
    if ${DRY_RUN}; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

ssh_run() {
    local host="$1"; shift
    if ${DRY_RUN}; then
        echo "[DRY-RUN] ssh ${host}: $*"
    else
        ssh "${host}" "$@"
    fi
}

# Check free GiB on a Spark node using /proc/meminfo MemAvailable.
# This is the correct metric for GB10 UMA where nvidia-smi --query-gpu CSV
# returns N/A for memory fields (UMA pools are not tracked as separate GPU memory).
node_free_gib() {
    local host="$1"
    ssh "${host}" "awk '/MemAvailable/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo"
}

# Boot ID from /proc/sys/kernel/random/boot_id — changes on every reboot.
node_boot_id() {
    local host="$1"
    ssh "${host}" "cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown"
}

# Uptime in seconds from /proc/uptime (first field, integer part).
node_uptime_seconds() {
    local host="$1"
    ssh "${host}" "awk '{printf \"%d\", \$1}' /proc/uptime 2>/dev/null || echo unknown"
}

# Check if a container is running on a host
container_running() {
    local host="$1" name="$2"
    ssh "${host}" "docker inspect --format '{{.State.Running}}' '${name}' 2>/dev/null || echo false"
}

# Wait for API readiness (polls /health on spark01)
wait_for_api() {
    local timeout_s=${API_READY_TIMEOUT:-600}
    local elapsed=0
    local interval=10
    log "Waiting for API readiness (timeout=${timeout_s}s)..."
    while [[ ${elapsed} -lt ${timeout_s} ]]; do
        local status
        status=$(ssh "${SPARK01}" "curl -s -o /dev/null -w '%{http_code}' '${API_URL}/health' 2>/dev/null || echo 000")
        if [[ "${status}" == "200" ]]; then
            log "API ready after ${elapsed}s."
            return 0
        fi
        sleep "${interval}"
        elapsed=$((elapsed + interval))
    done
    warn "API not ready after ${timeout_s}s."
    return 1
}

# ---------------------------------------------------------------------------
# Preflight check (--preflight-only mode)
#
# Read-only checks against both Spark nodes:
#   - SSH connectivity
#   - boot ID (changes on every reboot — use to verify nodes were rebooted)
#   - uptime
#   - host memory state (/proc/meminfo, UMA-safe)
#   - running container list
#   - stale vllm processes outside containers
#   - template env file validity
#
# No container ops. Safe to run with containers up.
# Exit 0: all checks pass (or only warnings — running server with low free
#         memory is expected and is NOT a failure here).
# Exit 1: critical failure (SSH unreachable, template missing).
# ---------------------------------------------------------------------------
preflight_check() {
    local pf_dir="${RESULT_DIR}/.preflight-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "${pf_dir}"
    local pf_file="${pf_dir}/preflight.txt"
    local critical_fail=false

    log "=== Preflight check (read-only, no container ops) ==="
    log "Results will be saved to: ${pf_file}"

    {
        echo "preflight_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "bt_values=${BT_VALUES[*]:-<not specified>}"
        echo "template_env=${TEMPLATE_ENV}"
        echo "expected_ep=${EXPECTED_EP}"
        echo ""

        # 1. Template validity
        echo "## Template check"
        if [[ -f "${TEMPLATE_ENV}" ]]; then
            echo "template_exists=OK"
            if grep -q '__BT_PLACEHOLDER__' "${TEMPLATE_ENV}"; then
                echo "template_bt_placeholder=OK"
            else
                echo "template_bt_placeholder=WARN (no __BT_PLACEHOLDER__ in file)"
            fi
            local ep_in_template
            # Check VLLM_EXTRA_ARGS line only — comments may contain the flag text
            if grep '^VLLM_EXTRA_ARGS=' "${TEMPLATE_ENV}" 2>/dev/null | grep -q -- '--enable-expert-parallel'; then
                ep_in_template="yes (in VLLM_EXTRA_ARGS)"
            else
                ep_in_template="no (absent from VLLM_EXTRA_ARGS)"
            fi
            echo "template_has_enable_expert_parallel=${ep_in_template}"
        else
            echo "template_exists=FAIL (not found: ${TEMPLATE_ENV})"
        fi
        echo ""

        # 2. SSH connectivity
        echo "## SSH connectivity"
        for host in "${SPARK01}" "${SPARK02}"; do
            if ssh -o ConnectTimeout=5 "${host}" "echo ok" &>/dev/null; then
                echo "ssh_${host}=OK"
            else
                echo "ssh_${host}=FAIL"
                critical_fail=true
            fi
        done
        echo ""

    } > "${pf_file}"

    # Per-node checks (SSH required — skip if unreachable)
    {
        # 3. Boot ID + uptime
        echo "## Boot ID and uptime"
        for host in "${SPARK01}" "${SPARK02}"; do
            local boot_id uptime_s
            boot_id=$(ssh -o ConnectTimeout=5 "${host}" "cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown" 2>/dev/null || echo "ssh_fail")
            uptime_s=$(ssh -o ConnectTimeout=5 "${host}" "awk '{printf \"%d\", \$1}' /proc/uptime 2>/dev/null || echo unknown" 2>/dev/null || echo "ssh_fail")
            echo "${host}_boot_id=${boot_id}"
            echo "${host}_uptime_seconds=${uptime_s}"
        done
        echo ""

        # 4. Host memory state
        # NOTE: SAFE_MEM_GIB applies to post-stop checks, not here. A running vLLM
        # server on GB10 UMA holds ~75 GiB, leaving ~46 GiB free — below SAFE_MEM_GIB.
        # This is expected and is logged as INFO, not FAIL.
        echo "## Host memory (pre-start, informational only)"
        echo "## SAFE_MEM_GIB=${SAFE_MEM_GIB} applies to post-stop only, not here."
        for host in "${SPARK01}" "${SPARK02}"; do
            local free_gib
            free_gib=$(ssh -o ConnectTimeout=5 "${host}" "awk '/MemAvailable/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo "ssh_fail")
            echo "${host}_mem_available_gib=${free_gib}"
            if [[ "${free_gib}" == "ssh_fail" ]]; then
                echo "${host}_memory_prestart=ssh_fail"
            elif awk "BEGIN { exit (${free_gib} >= ${SAFE_MEM_GIB}) }"; then
                echo "${host}_memory_prestart=BELOW_THRESHOLD (${free_gib} GiB < ${SAFE_MEM_GIB} GiB threshold — EXPECTED if server is running)"
            else
                echo "${host}_memory_prestart=OK (${free_gib} GiB free — sufficient for cold-start)"
            fi
        done
        echo ""

        # 5. Running container list
        echo "## Running containers"
        for host in "${SPARK01}" "${SPARK02}"; do
            local running_containers
            running_containers=$(ssh -o ConnectTimeout=5 "${host}" \
                "docker ps --filter 'name=vllm-spark' --format '{{.Names}} ({{.Status}})' 2>/dev/null" \
                2>/dev/null || echo "docker_unavailable")
            echo "${host}_vllm_containers=${running_containers:-none}"
        done
        echo ""

        # 6. Stale vllm processes outside containers.
        # If a vllm-spark container is running, its internal vllm processes are
        # visible from the host via pgrep — these are NOT stale. Only flag processes
        # when no vllm-spark container is running on that host.
        echo "## Stale vllm processes (outside containers)"
        for host in "${SPARK01}" "${SPARK02}"; do
            local running_count stale_procs
            running_count=$(ssh -o ConnectTimeout=5 "${host}" \
                "docker ps --filter 'name=vllm-spark' --format '{{.Names}}' 2>/dev/null | wc -l" \
                2>/dev/null || echo "0")
            if [[ "${running_count}" =~ ^[1-9] ]]; then
                echo "${host}_stale_vllm_procs=skipped (${running_count} vllm-spark container(s) running — in-container vllm processes expected)"
            else
                stale_procs=$(ssh -o ConnectTimeout=5 "${host}" \
                    "pgrep -a -f 'vllm serve' 2>/dev/null | head -3 || echo none" \
                    2>/dev/null || echo "check_failed")
                echo "${host}_stale_vllm_procs=${stale_procs:-none}"
            fi
        done
        echo ""

    } >> "${pf_file}"

    # Print to stdout
    cat "${pf_file}"

    log "Preflight results saved: ${pf_file}"

    if ${critical_fail}; then
        log "Preflight: CRITICAL failures detected. Fix before running benchmark."
        return 1
    fi
    log "Preflight: No critical failures."
    log "Note: BELOW_THRESHOLD memory is expected if a server is currently running."
    log "After reboot, memory should exceed ${SAFE_MEM_GIB} GiB on both nodes."
    return 0
}

# ---------------------------------------------------------------------------
# Validate existing container (--validate-existing-container mode)
#
# Reads docker logs from the currently running head container and applies
# the same MARLIN, TRITON_ATTN, and EP detection as run_bt().
# No container start/stop. Read-only.
#
# Uses tail -1 to get the most recent occurrence of each log pattern
# (head may restart vllm serve internally without restarting the container).
#
# Exit 0: both MARLIN and TRITON_ATTN confirmed (EP check advisory only).
# Exit 1: container not running, log fetch failed, or backend not confirmed.
# ---------------------------------------------------------------------------
validate_existing_container() {
    local vc_dir="${RESULT_DIR}/.validate-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "${vc_dir}"
    local vc_file="${vc_dir}/validate-existing.txt"
    local log_file="${vc_dir}/head-startup.log"

    log "=== Validate existing container (read-only, no container ops) ==="

    # Check container is running
    local running
    running=$(container_running "${SPARK01}" "vllm-spark-head")
    if [[ "${running}" != "true" ]]; then
        warn "vllm-spark-head is not running on ${SPARK01} (state=${running})"
        echo "container_running=false" | tee "${vc_file}"
        return 1
    fi

    # Fetch logs
    log "Fetching container logs from ${SPARK01}:vllm-spark-head..."
    ssh "${SPARK01}" "docker logs vllm-spark-head 2>&1" > "${log_file}" 2>&1 || {
        warn "Failed to fetch container logs from ${SPARK01}"
        return 1
    }
    local log_lines
    log_lines=$(wc -l < "${log_file}")
    log "Fetched ${log_lines} log lines -> ${log_file}"

    # Apply same detection logic as run_bt (tail -1: most recent occurrence)
    local marlin_line triton_line entrypoint_cmd
    marlin_line=$(grep "Using 'MARLIN' NvFp4 MoE backend" "${log_file}" 2>/dev/null | tail -1 || echo "")
    triton_line=$(grep "Using AttentionBackendEnum.TRITON_ATTN backend" "${log_file}" 2>/dev/null | tail -1 || echo "")
    entrypoint_cmd=$(grep '\[entrypoint\] Running: vllm serve' "${log_file}" 2>/dev/null | tail -1 || echo "")

    local marlin_ok triton_ok ep_observed_str ep_evidence
    marlin_ok=$(echo "${marlin_line}" | grep -c "MARLIN" || echo 0)
    triton_ok=$(echo "${triton_line}" | grep -c "TRITON_ATTN" || echo 0)

    if [[ -z "${entrypoint_cmd}" ]]; then
        ep_observed_str="not_found_in_logs"
        ep_evidence="entrypoint command line not captured in log"
    elif echo "${entrypoint_cmd}" | grep -q -- '--enable-expert-parallel'; then
        ep_observed_str="enabled"
        ep_evidence="--enable-expert-parallel present in entrypoint command line"
    else
        ep_observed_str="disabled"
        ep_evidence="--enable-expert-parallel absent from entrypoint command line"
    fi

    local backend_valid="OK"
    [[ ${marlin_ok} -lt 1 ]] && backend_valid="FAIL_MARLIN_NOT_CONFIRMED"
    [[ ${triton_ok} -lt 1 ]] && backend_valid="FAIL_TRITON_NOT_CONFIRMED"

    {
        echo "validate_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "container_running=true"
        echo "log_lines_fetched=${log_lines}"
        echo "log_file=${log_file}"
        echo ""
        echo "## Backend detection"
        echo "marlin_confirmed=${marlin_ok}"
        echo "marlin_evidence=${marlin_line:-not found}"
        echo ""
        echo "triton_attn_confirmed=${triton_ok}"
        echo "triton_evidence=${triton_line:-not found}"
        echo ""
        echo "## EP detection"
        echo "expert_parallel_observed=${ep_observed_str}"
        echo "ep_evidence=${ep_evidence}"
        if [[ -n "${entrypoint_cmd}" ]]; then
            echo "entrypoint_cmd=${entrypoint_cmd}"
        fi
        echo ""
        echo "## Summary"
        echo "backend_valid=${backend_valid}"
    } | tee "${vc_file}"

    # EP check against --expected-ep (advisory in validate mode — no halt)
    if [[ "${EXPECTED_EP}" != "unknown" ]]; then
        local ep_match=true
        if [[ "${EXPECTED_EP}" == "on" ]] && [[ "${ep_observed_str}" != "enabled" ]]; then
            ep_match=false
        elif [[ "${EXPECTED_EP}" == "off" ]] && [[ "${ep_observed_str}" != "disabled" ]]; then
            ep_match=false
        fi
        if ! ${ep_match}; then
            warn "EP mismatch: expected=${EXPECTED_EP}, observed=${ep_observed_str}"
            echo "ep_validation=MISMATCH (expected=${EXPECTED_EP}, observed=${ep_observed_str})" | tee -a "${vc_file}"
        else
            echo "ep_validation=OK (expected=${EXPECTED_EP}, observed=${ep_observed_str})" | tee -a "${vc_file}"
        fi
    fi

    log "Validation results saved: ${vc_file}"

    if [[ ${marlin_ok} -lt 1 ]]; then
        warn "MARLIN not confirmed — container would fail backend gate in a benchmark run."
        return 1
    fi
    if [[ ${triton_ok} -lt 1 ]]; then
        warn "TRITON_ATTN not confirmed — container would fail backend gate in a benchmark run."
        return 1
    fi
    log "Backend validation: PASS (MARLIN confirmed, TRITON_ATTN confirmed, EP=${ep_observed_str})"
    return 0
}

# ---------------------------------------------------------------------------
# Correctness validation (lightweight — not a full evaluation suite)
# ---------------------------------------------------------------------------
correctness_check() {
    local bt="$1"
    local out_file="$2"
    local model="${SERVED_MODEL_NAME}"
    local api="http://localhost:8000/v1/chat/completions"

    log "Running correctness checks for bt=${bt}..."

    # Helper: POST a prompt and extract content
    ask() {
        local prompt="$1" max_tokens="${2:-800}"
        ssh "${SPARK01}" "curl -s -X POST '${api}' \
            -H 'Content-Type: application/json' \
            -d '{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${max_tokens},\"temperature\":0}'"
    }

    {
        echo "# Correctness check — bt=${bt} — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo ""

        # Test 1: Simple factual
        echo "## Test 1: Factual (largest prime < 100)"
        local r1
        r1=$(ask "What is the largest prime number less than 100? Answer with only the number." 100)
        local c1
        c1=$(echo "${r1}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message'].get('content','') or '')" 2>/dev/null || echo "ERROR")
        echo "Response: ${c1}"
        if echo "${c1}" | grep -qE '^[[:space:]]*97[[:space:]]*$'; then
            echo "Result: PASS (correct: 97)"
        else
            echo "Result: FAIL or UNCERTAIN (expected '97', got: '${c1:0:200}')"
        fi
        echo ""

        # Test 2: Multi-step arithmetic
        echo "## Test 2: Multi-step reasoning (15 factorial)"
        local r2
        r2=$(ask "What is 15 factorial (15!)? Give only the numeric answer." 600)
        local c2
        c2=$(echo "${r2}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message'].get('content','') or d['choices'][0]['message'].get('reasoning','')[:100])" 2>/dev/null || echo "ERROR")
        echo "Response (truncated): ${c2:0:300}"
        if echo "${c2}" | grep -q "1307674368000"; then
            echo "Result: PASS (found 1307674368000)"
        else
            echo "Result: UNCERTAIN (check if content is None — may need more max_tokens)"
        fi
        echo ""

        # Test 3: Unicode integrity
        echo "## Test 3: Unicode integrity (Korean)"
        local r3
        r3=$(ask "서울에서 부산까지 KTX 소요시간은?" 400)
        local c3
        c3=$(echo "${r3}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message'].get('content','') or '')" 2>/dev/null || echo "ERROR")
        echo "Response (truncated): ${c3:0:300}"
        if echo "${c3}" | grep -qP '[\x{FFFD}]|[^\x{0000}-\x{9FFF}\x{AC00}-\x{D7A3}\x{F900}-\x{FFEF}\x{1F000}-\x{1FFFF} \t\n\r\.,!?]' 2>/dev/null; then
            echo "Result: POSSIBLE_GARBLE (unexpected codepoints detected)"
        elif [[ -z "${c3}" ]] || echo "${c3}" | grep -q "ERROR"; then
            echo "Result: EMPTY_OR_ERROR"
        else
            echo "Result: PASS (Korean response received, no obvious garble)"
        fi
        echo ""

        # Test 4: Finish reason check
        echo "## Test 4: Finish reason check (simple 2+2)"
        local r4
        r4=$(ask "What is 2+2?" 100)
        local fr4
        fr4=$(echo "${r4}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['finish_reason'])" 2>/dev/null || echo "ERROR")
        local c4
        c4=$(echo "${r4}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message'].get('content',''))" 2>/dev/null || echo "ERROR")
        echo "finish_reason: ${fr4}"
        echo "content: ${c4:0:100}"
        if [[ "${fr4}" == "stop" ]] && echo "${c4}" | grep -qE '[4]'; then
            echo "Result: PASS"
        else
            echo "Result: UNCERTAIN (finish_reason=${fr4})"
        fi
        echo ""

        echo "## Summary"
        echo "bt=${bt} correctness checks complete. Manual review required for UNCERTAIN results."
        echo "CRITICAL garble indicators: broken Unicode, garbled tokens in factual answers,"
        echo "  systematic wrong answers, repeated token loops."

    } > "${out_file}" 2>&1

    log "Correctness check saved: ${out_file}"
}

# ---------------------------------------------------------------------------
# Extended correctness check (supplement mode)
# ---------------------------------------------------------------------------
# Same 4 prompts as correctness_check() but max_tokens=2048 and 2 runs each.
# Designed for Step-3.7-Flash reasoning model: short budgets exhaust the
# <think> chain before the answer token, producing empty content (not garble).
# Verdict taxonomy per run:
#   PASS                    — expected answer found in content; finish_reason=stop
#   PASS_WITH_LENGTH_LIMIT  — expected answer found before budget exhausted; finish_reason=length
#   FAIL_WRONG_ANSWER       — non-empty content but expected pattern absent; finish_reason=stop
#   FAIL_GARBLE             — U+FFFD/surrogate escape, PUA dominance (>10%), or pathological repetition
#   INCONCLUSIVE_OUTPUT_BUDGET — finish_reason=length; expected pattern not found in sampled content
#   FAIL_API                — HTTP error, parse error, or empty content with finish_reason=stop
# Garble detection targets specific corruption signals, not a character whitelist.
# Characters outside ASCII/Korean range (e.g. U+00B7 MIDDLE DOT in Korean punctuation)
# must not trigger FAIL_GARBLE. Only replacement chars, surrogate escapes, PUA dominance,
# and pathological repetition are flagged as garble.
correctness_check_extended() {
    local bt="$1"
    local out_file="$2"
    local model="${SERVED_MODEL_NAME}"
    local api="http://localhost:8000/v1/chat/completions"
    local MAX_T=2048
    local NRUNS=2

    log "Running extended correctness checks for bt=${bt} (max_tokens=${MAX_T}, ${NRUNS} runs each)..."

    # Run one API call; prompt must be safe for embedding in JSON (no bare quotes/backslashes)
    _ask_ext() {
        local prompt="$1"
        ssh "${SPARK01}" "curl -s --max-time 300 -X POST '${api}' \
            -H 'Content-Type: application/json' \
            -d '{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${MAX_T},\"temperature\":0}'"
    }

    # Parse raw JSON response to structured key=value lines
    _parse_ext() {
        python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ch = d['choices'][0]
    msg = ch.get('message', {})
    content = (msg.get('content') or '').strip()
    fr = ch.get('finish_reason', 'unknown')
    ct = d.get('usage', {}).get('completion_tokens', '?')
    r_content = msg.get('reasoning_content') or ''
    print('content=' + content[:600])
    print('finish_reason=' + str(fr))
    print('completion_tokens=' + str(ct))
    print('has_reasoning=' + str(bool(r_content)))
    print('reasoning_len=' + str(len(r_content)))
except Exception as e:
    print('content=PARSE_ERROR:' + str(e))
    print('finish_reason=error')
    print('completion_tokens=?')
    print('has_reasoning=False')
    print('reasoning_len=0')
" 2>/dev/null
    }

    # Classify a parsed response. Echoes verdict on last line as "verdict: RESULT".
    _classify() {
        local parsed="$1" expected_grep="$2" test_type="$3"
        local content fr ct has_r r_len
        content=$(echo "${parsed}" | grep '^content=' | head -1 | cut -c9-)
        fr=$(echo "${parsed}"  | grep '^finish_reason=' | cut -d= -f2-)
        ct=$(echo "${parsed}"  | grep '^completion_tokens=' | cut -d= -f2-)
        has_r=$(echo "${parsed}" | grep '^has_reasoning=' | cut -d= -f2-)
        r_len=$(echo "${parsed}" | grep '^reasoning_len=' | cut -d= -f2-)
        echo "  finish_reason: ${fr}"
        echo "  completion_tokens: ${ct}"
        echo "  has_reasoning_content: ${has_r} (len=${r_len})"
        echo "  content (first 400 chars): ${content:0:400}"
        local verdict
        if echo "${content}" | grep -q "PARSE_ERROR" || [[ "${fr}" == "error" ]]; then
            verdict="FAIL_API"
        elif [[ -z "${content}" ]] && [[ "${fr}" == "stop" ]]; then
            verdict="FAIL_API"
        elif [[ -z "${content}" ]]; then
            verdict="INCONCLUSIVE_OUTPUT_BUDGET"
        elif [[ "${test_type}" == "garble" ]]; then
            if echo "${content}" | python3 -c "
import sys, re
txt = sys.stdin.read()
# Replacement char (U+FFFD) or escaped surrogates (U+D800-U+DFFF) = definite garble
if re.search(r'[\ufffd\ud800-\udfff]', txt):
    sys.exit(1)
# Private Use Area dominance above 10% of text length suggests corrupted output
pua = len(re.findall(r'[\ue000-\uf8ff]', txt))
if pua > max(5, len(txt) * 0.1):
    sys.exit(1)
# Pathological token repetition: same 3-15 char sequence repeated 5+ times
for n in range(3, 16):
    if re.search(r'(.{' + str(n) + r'})\\1{4}', txt):
        sys.exit(1)
" 2>/dev/null; then
                [[ "${fr}" == "length" ]] && verdict="INCONCLUSIVE_OUTPUT_BUDGET" || verdict="PASS"
            else
                verdict="FAIL_GARBLE"
            fi
        elif [[ "${fr}" == "length" ]]; then
            echo "${content}" | grep -qE "${expected_grep}" && verdict="PASS_WITH_LENGTH_LIMIT" || verdict="INCONCLUSIVE_OUTPUT_BUDGET"
        elif echo "${content}" | grep -qE "${expected_grep}"; then
            verdict="PASS"
        else
            verdict="FAIL_WRONG_ANSWER"
        fi
        echo "  verdict: ${verdict}"
        echo "__verdict__:${verdict}"
    }

    {
        echo "# Extended correctness check — bt=${bt} — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# max_tokens=${MAX_T}, runs_per_test=${NRUNS}"
        echo "# supplement_of=${SUPPLEMENT_OF:-unknown}"
        echo ""

        local all_verdicts=()
        local prompt_bests=()

        for test_spec in \
            "1|What is the largest prime number less than 100? Answer with only the number.|97|exact" \
            "2|What is 15 factorial (15!)? Give only the numeric answer.|1307674368000|exact" \
            "3|서울에서 부산까지 KTX 소요시간은?|.|garble" \
            "4|What is 2+2?|[4]|exact"
        do
            local tid prompt expected ttype
            tid=$(echo "${test_spec}"    | cut -d'|' -f1)
            prompt=$(echo "${test_spec}" | cut -d'|' -f2)
            expected=$(echo "${test_spec}" | cut -d'|' -f3)
            ttype=$(echo "${test_spec}"  | cut -d'|' -f4)

            local label
            case "${tid}" in
                1) label="Factual (largest prime < 100)" ;;
                2) label="Multi-step reasoning (15 factorial)" ;;
                3) label="Unicode integrity (Korean KTX)" ;;
                4) label="Finish reason and simple answer (2+2)" ;;
            esac

            echo "## Test ${tid}: ${label}"
            echo "Prompt: '${prompt}'"
            echo "max_tokens: ${MAX_T}"
            local test_verdicts=()
            for n in $(seq 1 "${NRUNS}"); do
                echo ""
                echo "### Run ${n}/${NRUNS}"
                local raw parsed classified
                raw=$(_ask_ext "${prompt}" 2>&1)
                parsed=$(echo "${raw}" | _parse_ext)
                classified=$(_classify "${parsed}" "${expected}" "${ttype}")
                echo "${classified}" | grep -v "^__verdict__:"
                local v
                v=$(echo "${classified}" | grep "^__verdict__:" | cut -d: -f2)
                test_verdicts+=("${v}")
                all_verdicts+=("${v}")
            done
            echo ""
            echo "Test ${tid} verdict summary: ${test_verdicts[*]}"
            local best
            best="INCONCLUSIVE_OUTPUT_BUDGET"
            for v in "${test_verdicts[@]}"; do
                [[ "${v}" == "PASS" ]] && best="PASS" && break
                [[ "${v}" == "PASS_WITH_LENGTH_LIMIT" ]] && [[ "${best}" != "PASS" ]] && best="PASS_WITH_LENGTH_LIMIT"
                [[ "${v}" == "FAIL_GARBLE" ]] && [[ "${best}" != "PASS" && "${best}" != "PASS_WITH_LENGTH_LIMIT" ]] && best="FAIL_GARBLE"
                [[ "${v}" == "FAIL_WRONG_ANSWER" ]] && [[ "${best}" != "PASS" && "${best}" != "PASS_WITH_LENGTH_LIMIT" && "${best}" != "FAIL_GARBLE" ]] && best="FAIL_WRONG_ANSWER"
                [[ "${v}" == "FAIL_API" ]] && [[ "${best}" == "INCONCLUSIVE_OUTPUT_BUDGET" ]] && best="FAIL_API"
            done
            # prompt_best: coverage-oriented aggregation — did this prompt get at least
            # one passing result? Does NOT imply all duplicate runs were strict PASS.
            echo "Test ${tid} prompt_best: ${best}"
            prompt_bests+=("${best}")
            echo ""
        done

        echo "## Overall summary"
        echo "All verdicts: ${all_verdicts[*]}"
        local observed_garble=false all_runs_strict_pass=true all_prompts_have_pass=true
        local inconclusive_run_count=0 failed_run_count=0
        for v in "${all_verdicts[@]}"; do
            [[ "${v}" == "FAIL_GARBLE" ]] && observed_garble=true
            if [[ "${v}" != "PASS" && "${v}" != "PASS_WITH_LENGTH_LIMIT" ]]; then
                all_runs_strict_pass=false
            fi
            if [[ "${v}" == "INCONCLUSIVE_OUTPUT_BUDGET" ]]; then
                inconclusive_run_count=$(( inconclusive_run_count + 1 ))
            fi
            if [[ "${v}" == FAIL_* ]]; then
                failed_run_count=$(( failed_run_count + 1 ))
            fi
        done
        for b in "${prompt_bests[@]}"; do
            if [[ "${b}" != "PASS" && "${b}" != "PASS_WITH_LENGTH_LIMIT" ]]; then
                all_prompts_have_pass=false
            fi
        done
        local suite_status
        if ${all_prompts_have_pass} && ${all_runs_strict_pass} && ! ${observed_garble}; then
            suite_status="PASS_STRICT"
        elif ${all_prompts_have_pass} && [[ "${failed_run_count}" -eq 0 ]] && ! ${observed_garble}; then
            suite_status="PASS_WITH_INCONCLUSIVE_DUPLICATE"
        elif ${all_prompts_have_pass} && [[ "${failed_run_count}" -gt 0 ]] && ! ${observed_garble}; then
            suite_status="PASS_WITH_FAILED_DUPLICATE"
        elif ${all_prompts_have_pass} && ${observed_garble}; then
            suite_status="PASS_WITH_GARBLE_HISTORY"
        elif ! ${all_prompts_have_pass} && [[ "${failed_run_count}" -eq 0 ]] && ! ${observed_garble}; then
            suite_status="INCONCLUSIVE"
        else
            suite_status="FAIL_PROMPT"
        fi
        echo "all_prompts_have_pass: ${all_prompts_have_pass}"
        echo "all_runs_strict_pass: ${all_runs_strict_pass}"
        echo "inconclusive_run_count: ${inconclusive_run_count}"
        echo "failed_run_count: ${failed_run_count}"
        echo "observed_garble: ${observed_garble}"
        echo "suite_status: ${suite_status}"
        echo ""
        echo "Verdict taxonomy:"
        echo "  PASS                    - expected answer found; finish_reason=stop"
        echo "  PASS_WITH_LENGTH_LIMIT  - expected answer found before budget exhausted; finish_reason=length"
        echo "  FAIL_WRONG_ANSWER       - output present but expected pattern absent"
        echo "  FAIL_GARBLE             - U+FFFD/surrogates, PUA dominance (>10%), or pathological repetition"
        echo "  FAIL_API                - HTTP error, parse error, or empty stop response"
        echo "  INCONCLUSIVE_OUTPUT_BUDGET - finish_reason=length; expected pattern not found in sampled content"
        echo ""
        echo "Suite status taxonomy:"
        echo "  PASS_STRICT                   - all prompts covered, all runs strict pass, no garble"
        echo "  PASS_WITH_INCONCLUSIVE_DUPLICATE - all prompts covered, no failures, some runs inconclusive"
        echo "  PASS_WITH_FAILED_DUPLICATE    - all prompts covered, some runs failed (wrong answer/API)"
        echo "  PASS_WITH_GARBLE_HISTORY      - all prompts covered, garble observed in at least one run"
        echo "  INCONCLUSIVE                  - no prompt failures or garble, but not all prompts covered"
        echo "  FAIL_PROMPT                   - at least one prompt has no passing result"
        echo ""
        echo "Note: prompt_best (per test) indicates coverage: did this prompt produce at least"
        echo "  one PASS or PASS_WITH_LENGTH_LIMIT across its runs? It does NOT mean all runs passed."
        echo "  INCONCLUSIVE_OUTPUT_BUDGET is not a garble failure. Increase max_tokens for a definitive result."

    } > "${out_file}" 2>&1

    log "Extended correctness check saved: ${out_file}"
}

# ---------------------------------------------------------------------------
# Decode-only benchmark (supplement mode)
# ---------------------------------------------------------------------------
# Controlled by DECODE_CONTEXT_MODE: pp1 | pp2048 | both
#   pp1    - pp=1, tg=32 (short-context decode); negligible prefill; default
#   pp2048 - pp=2048, tg=32 (pp2048-context decode); comparable to full run
#   both   - pp1 pass first, then pp2048 pass
# Results: decode-bench.md (pp1), decode-bench-pp2048.md (pp2048)
# DECODE_WARMUP_RUNS warmup iterations are discarded before measured runs.
decode_bench_supplement() {
    local run_dir="$1"
    local meta_file="$2"
    local bt="$3"

    local depth_args=""
    for d in ${BENCH_DEPTHS}; do depth_args="${depth_args} ${d}"; done

    echo "decode_context_mode=${DECODE_CONTEXT_MODE}" >> "${meta_file}"
    echo "decode_bench_tg=32" >> "${meta_file}"
    echo "decode_bench_runs=${SUPPLEMENT_DECODE_RUNS}" >> "${meta_file}"
    echo "decode_bench_depths=${BENCH_DEPTHS}" >> "${meta_file}"
    echo "decode_bench_warmup_runs=${DECODE_WARMUP_RUNS}" >> "${meta_file}"

    _run_decode_pass() {
        local pp="$1"
        local out_file="$2"
        local log_file="${out_file%.md}-stdout.log"
        local tmp_remote="/tmp/decode-bench-supp-bt${bt}-pp${pp}.md"
        local mode_label="pp${pp}-context decode"
        local exit_code=0

        if [[ ${DECODE_WARMUP_RUNS} -gt 0 ]]; then
            log "  Warmup: ${DECODE_WARMUP_RUNS} run(s) before measured ${mode_label}..."
            # shellcheck disable=SC2086
            ssh "${SPARK01}" "PYTHONUNBUFFERED=1 ${LLAMA_BENCHY_BIN} \
                --base-url '${API_BASE}' \
                --model '${SERVED_MODEL_NAME}' \
                --tokenizer '${TOKENIZER_PATH}' \
                --pp ${pp} --tg 32 --depth ${depth_args} \
                --runs ${DECODE_WARMUP_RUNS} --latency-mode generation --format md" \
                > /dev/null 2>&1 || true
        fi

        log "  Running ${mode_label} (pp=${pp}, tg=32, runs=${SUPPLEMENT_DECODE_RUNS})..."
        # shellcheck disable=SC2086
        ssh "${SPARK01}" "PYTHONUNBUFFERED=1 ${LLAMA_BENCHY_BIN} \
            --base-url '${API_BASE}' \
            --model '${SERVED_MODEL_NAME}' \
            --tokenizer '${TOKENIZER_PATH}' \
            --pp ${pp} --tg 32 --depth ${depth_args} \
            --runs ${SUPPLEMENT_DECODE_RUNS} \
            --latency-mode generation \
            --format md \
            --save-result '${tmp_remote}'" \
            > "${log_file}" 2>&1 \
            && scp "${SPARK01}:${tmp_remote}" "${out_file}" \
            || exit_code=$?

        return ${exit_code}
    }

    local overall_ok=true

    if [[ "${DECODE_CONTEXT_MODE}" == "pp1" || "${DECODE_CONTEXT_MODE}" == "both" ]]; then
        local f="${run_dir}/decode-bench.md"
        echo "decode_bench_pp=1" >> "${meta_file}"
        echo "decode_bench_note=pp=1 gives negligible prefill; measures short-context decode t/s" >> "${meta_file}"
        if _run_decode_pass 1 "${f}"; then
            log "Decode bench (pp=1) saved: ${f}"
            echo "decode_bench_result=OK" >> "${meta_file}"
        else
            warn "Decode bench (pp=1) failed"
            echo "decode_bench_result=FAIL" >> "${meta_file}"
            overall_ok=false
        fi
    fi

    if [[ "${DECODE_CONTEXT_MODE}" == "pp2048" || "${DECODE_CONTEXT_MODE}" == "both" ]]; then
        local f="${run_dir}/decode-bench-pp2048.md"
        echo "decode_bench_pp2048_note=pp=2048 context; comparable to full run tg32 measurement" >> "${meta_file}"
        if _run_decode_pass 2048 "${f}"; then
            log "Decode bench (pp=2048) saved: ${f}"
            echo "decode_bench_pp2048_result=OK" >> "${meta_file}"
        else
            warn "Decode bench (pp=2048) failed"
            echo "decode_bench_pp2048_result=FAIL" >> "${meta_file}"
            overall_ok=false
        fi
    fi

    ${overall_ok}
}

# ---------------------------------------------------------------------------
# Server-only run: start server, validate, keep running (no bench, no stop)
# ---------------------------------------------------------------------------
# Starts the bt=<N> server, applies prometheus patch, waits for API, validates
# backends. Exits WITHOUT stopping containers — server keeps running for
# manual testing or follow-up supplement/correctness commands.
server_only_run() {
    local bt="$1"
    local run_id="bt${bt}-server-$(date +%Y%m%d-%H%M%S)"
    local run_dir="${RESULT_DIR}/${run_id}"
    local env_file="${REPO_ROOT}/.local/env/step37/bt-matrix-run.env"
    local meta_file="${run_dir}/metadata.txt"

    log "======================================================"
    log "SERVER-ONLY mode: bt=${bt} (id=${run_id})"
    log "  Server will keep running after script exits."
    log "======================================================"

    mkdir -p "${run_dir}"
    sed "s/__BT_PLACEHOLDER__/${bt}/" "${TEMPLATE_ENV}" > "${env_file}"

    {
        echo "run_id=${run_id}"
        echo "server_only_mode=true"
        echo "bt=${bt}"
        echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "config_label=${CONFIG_LABEL:-v023-server-only-bt${bt}}"
        echo "template_env=${TEMPLATE_ENV}"
        echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "model=${SERVED_MODEL_NAME}"
        echo "image=$(grep '^VLLM_IMAGE=' "${env_file}" | cut -d= -f2)"
        echo "max_num_batched_tokens=${bt}"
        echo "head_boot_id=$(node_boot_id "${SPARK01}" 2>/dev/null || echo unknown)"
        echo "worker_boot_id=$(node_boot_id "${SPARK02}" 2>/dev/null || echo unknown)"
        echo "spark01_free_gib_before=$(node_free_gib "${SPARK01}" 2>/dev/null || echo unknown)"
        echo "spark02_free_gib_before=$(node_free_gib "${SPARK02}" 2>/dev/null || echo unknown)"
    } > "${meta_file}"

    if ${DRY_RUN}; then
        log "[DRY-RUN] Would start bt=${bt} server and validate backends; no benchmark."
        return 0
    fi

    start_containers "${env_file}"

    if ! apply_prometheus_patch "${meta_file}"; then
        warn "Server-only: prometheus patch failed."
        echo "startup_result=PROMETHEUS_PATCH_FAILED" >> "${meta_file}"
        stop_containers
        return 1
    fi

    if ! wait_for_api; then
        warn "Server-only: API not ready after timeout."
        echo "startup_result=STARTUP_FAIL" >> "${meta_file}"
        stop_containers
        return 1
    fi
    echo "startup_result=OK" >> "${meta_file}"

    log "Fetching startup log..."
    ssh "${SPARK01}" "docker logs vllm-spark-head 2>&1" > "${run_dir}/head-startup.log" 2>&1 || true

    local marlin_ok marlin_line triton_ok triton_line
    marlin_line=$(grep "Using 'MARLIN' NvFp4 MoE backend" "${run_dir}/head-startup.log" 2>/dev/null | head -1 || echo "")
    marlin_ok=$(echo "${marlin_line}" | grep -c "MARLIN" || echo 0)
    triton_line=$(grep "Using AttentionBackendEnum.TRITON_ATTN backend" "${run_dir}/head-startup.log" 2>/dev/null | head -1 || echo "")
    triton_ok=$(echo "${triton_line}" | grep -c "TRITON_ATTN" || echo 0)
    {
        echo "marlin_confirmed=${marlin_ok}"
        echo "triton_attn_confirmed=${triton_ok}"
        echo "backend_validation_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >> "${meta_file}"

    if [[ "${marlin_ok}" -lt 1 ]]; then
        warn "Server-only: MARLIN NOT confirmed."
        echo "backend_validity=INVALID_MARLIN_NOT_CONFIRMED" >> "${meta_file}"
        stop_containers; return 1
    fi
    if [[ "${triton_ok}" -lt 1 ]]; then
        warn "Server-only: TRITON_ATTN NOT confirmed."
        echo "backend_validity=INVALID_TRITON_NOT_CONFIRMED" >> "${meta_file}"
        stop_containers; return 1
    fi
    echo "backend_validity=OK (MARLIN confirmed, TRITON_ATTN confirmed)" >> "${meta_file}"

    log "======================================================"
    log "SERVER READY — bt=${bt}"
    log "  API: ${API_URL}"
    log "  Metadata: ${meta_file}"
    log "  Server is running. Use 'docker compose --profile head/worker down' to stop."
    log "======================================================"
    return 0
}

# ---------------------------------------------------------------------------
# Supplement run: extended correctness + decode-only bench in one lifecycle
# ---------------------------------------------------------------------------
supplement_run() {
    local bt="$1"
    local run_id="bt${bt}-supp-$(date +%Y%m%d-%H%M%S)"
    local run_dir="${RESULT_DIR}/${run_id}"
    local env_file="${REPO_ROOT}/.local/env/step37/bt-matrix-run.env"
    local meta_file="${run_dir}/metadata.txt"
    local correctness_file="${run_dir}/correctness-extended.md"
    local summary_file="${run_dir}/summary.csv"

    log "======================================================"
    log "Supplement run (id=${run_id})"
    log "  Extended correctness: max_tokens=2048, 2 runs each"
    log "  Decode-only bench: pp=1, tg=32, runs=${SUPPLEMENT_DECODE_RUNS}"
    log "  supplement_of: ${SUPPLEMENT_OF:-unset}"
    log "======================================================"

    mkdir -p "${run_dir}"
    sed "s/__BT_PLACEHOLDER__/${bt}/" "${TEMPLATE_ENV}" > "${env_file}"

    local ep_requested
    if grep '^VLLM_EXTRA_ARGS=' "${TEMPLATE_ENV}" 2>/dev/null | grep -q -- '--enable-expert-parallel'; then
        ep_requested="true"
    else
        ep_requested="false (flag absent from VLLM_EXTRA_ARGS)"
    fi

    {
        echo "run_id=${run_id}"
        echo "supplement_mode=true"
        echo "supplement_of=${SUPPLEMENT_OF:-unset}"
        echo "bt=${bt}"
        echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "config_label=${CONFIG_LABEL:-v023-triton-marlin-ep-off-bt${bt}-supplement}"
        echo "template_env=${TEMPLATE_ENV}"
        echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "model=${SERVED_MODEL_NAME}"
        echo "image=$(grep '^VLLM_IMAGE=' "${env_file}" | cut -d= -f2)"
        echo "max_model_len=$(grep '^MAX_MODEL_LEN=' "${env_file}" | cut -d= -f2)"
        echo "max_num_batched_tokens=${bt}"
        echo "expert_parallel_requested=${ep_requested}"
        echo "expert_parallel_expected_ep_flag=${EXPECTED_EP}"
        echo "supplement_correctness_max_tokens=2048"
        echo "supplement_correctness_runs_per_test=2"
        echo "supplement_correctness_enabled=${RUN_CORRECTNESS}"
        echo "supplement_decode_runs=${SUPPLEMENT_DECODE_RUNS}"
        echo "supplement_decode_context_mode=${DECODE_CONTEXT_MODE}"
        echo "supplement_decode_tg=32"
        echo "supplement_decode_depths=${BENCH_DEPTHS}"
        echo "supplement_decode_warmup_runs=${DECODE_WARMUP_RUNS}"
        echo "head_boot_id=$(node_boot_id "${SPARK01}" 2>/dev/null || echo unknown)"
        echo "worker_boot_id=$(node_boot_id "${SPARK02}" 2>/dev/null || echo unknown)"
        echo "head_uptime_seconds=$(node_uptime_seconds "${SPARK01}" 2>/dev/null || echo unknown)"
        echo "worker_uptime_seconds=$(node_uptime_seconds "${SPARK02}" 2>/dev/null || echo unknown)"
        echo "spark01_free_gib_before=$(node_free_gib "${SPARK01}" 2>/dev/null || echo unknown)"
        echo "spark02_free_gib_before=$(node_free_gib "${SPARK02}" 2>/dev/null || echo unknown)"
        echo "memory_metric_source=proc_meminfo_MemAvailable (nvidia-smi N/A on GB10 UMA)"
    } > "${meta_file}"

    if ${DRY_RUN}; then
        log "[DRY-RUN] Would start bt=${bt} server, run extended correctness and decode bench, stop."
        log "[DRY-RUN]   decode_context_mode=${DECODE_CONTEXT_MODE}, warmup_runs=${DECODE_WARMUP_RUNS}"
        log "[DRY-RUN]   run_correctness=${RUN_CORRECTNESS}"
        log "[DRY-RUN] Run dir: ${run_dir}"
        echo "run_id,supplement_mode,bt,status" > "${summary_file}"
        echo "${run_id},true,${bt},DRY_RUN" >> "${summary_file}"
        return 0
    fi

    # Start containers
    start_containers "${env_file}"

    # Prometheus patch (required — otherwise /health returns 500)
    if ! apply_prometheus_patch "${meta_file}"; then
        warn "Supplement run: prometheus patch failed — aborting."
        echo "startup_result=PROMETHEUS_PATCH_FAILED" >> "${meta_file}"
        echo "run_id,supplement_mode,bt,status,failure_reason" > "${summary_file}"
        echo "${run_id},true,${bt},STARTUP_FAIL,prometheus patch failed" >> "${summary_file}"
        stop_containers
        return 1
    fi

    # Wait for API
    if ! wait_for_api; then
        warn "Supplement run: API not ready — aborting."
        echo "startup_result=STARTUP_FAIL" >> "${meta_file}"
        echo "run_id,supplement_mode,bt,status,failure_reason" > "${summary_file}"
        echo "${run_id},true,${bt},STARTUP_FAIL,API timeout" >> "${summary_file}"
        stop_containers
        return 1
    fi
    echo "startup_result=OK" >> "${meta_file}"

    log "Fetching startup log..."
    ssh "${SPARK01}" "docker logs vllm-spark-head 2>&1" > "${run_dir}/head-startup.log" 2>&1 || true

    # Backend validation (same gates as run_bt)
    local marlin_ok marlin_line triton_ok triton_line
    marlin_line=$(grep "Using 'MARLIN' NvFp4 MoE backend" "${run_dir}/head-startup.log" 2>/dev/null | head -1 || echo "")
    marlin_ok=$(echo "${marlin_line}" | grep -c "MARLIN" || echo 0)
    triton_line=$(grep "Using AttentionBackendEnum.TRITON_ATTN backend" "${run_dir}/head-startup.log" 2>/dev/null | head -1 || echo "")
    triton_ok=$(echo "${triton_line}" | grep -c "TRITON_ATTN" || echo 0)

    local entrypoint_cmd ep_observed_str ep_evidence
    entrypoint_cmd=$(grep '\[entrypoint\] Running: vllm serve' "${run_dir}/head-startup.log" 2>/dev/null | head -1 || echo "")
    if [[ -z "${entrypoint_cmd}" ]]; then
        ep_observed_str="not_found_in_logs"; ep_evidence="entrypoint line not captured"
    elif echo "${entrypoint_cmd}" | grep -q -- '--enable-expert-parallel'; then
        ep_observed_str="enabled"; ep_evidence="--enable-expert-parallel present in entrypoint"
    else
        ep_observed_str="disabled"; ep_evidence="--enable-expert-parallel absent from entrypoint"
    fi
    {
        echo "marlin_confirmed=${marlin_ok}"
        echo "marlin_evidence_line=${marlin_line:-not found}"
        echo "triton_attn_confirmed=${triton_ok}"
        echo "triton_evidence_line=${triton_line:-not found}"
        echo "expert_parallel_observed=${ep_observed_str}"
        echo "expert_parallel_evidence=${ep_evidence}"
        echo "backend_validation_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >> "${meta_file}"

    if [[ "${marlin_ok}" -lt 1 ]]; then
        warn "Supplement run: MARLIN NOT confirmed — FATAL."
        echo "backend_validity=INVALID_MARLIN_NOT_CONFIRMED" >> "${meta_file}"
        echo "run_id,supplement_mode,bt,status,failure_reason" > "${summary_file}"
        echo "${run_id},true,${bt},BACKEND_FAIL,MARLIN not confirmed" >> "${summary_file}"
        stop_containers; return 1
    fi
    if [[ "${triton_ok}" -lt 1 ]]; then
        warn "Supplement run: TRITON_ATTN NOT confirmed — FATAL."
        echo "backend_validity=INVALID_TRITON_NOT_CONFIRMED" >> "${meta_file}"
        echo "run_id,supplement_mode,bt,status,failure_reason" > "${summary_file}"
        echo "${run_id},true,${bt},BACKEND_FAIL,TRITON_ATTN not confirmed" >> "${summary_file}"
        stop_containers; return 1
    fi
    echo "backend_validity=OK (MARLIN confirmed, TRITON_ATTN confirmed)" >> "${meta_file}"

    if [[ "${EXPECTED_EP}" != "unknown" ]]; then
        local ep_match=true
        [[ "${EXPECTED_EP}" == "off" ]] && [[ "${ep_observed_str}" != "disabled" ]] && ep_match=false
        [[ "${EXPECTED_EP}" == "on"  ]] && [[ "${ep_observed_str}" != "enabled"  ]] && ep_match=false
        if ! ${ep_match}; then
            warn "Supplement run: EP mismatch (expected=${EXPECTED_EP}, observed=${ep_observed_str}) — FATAL."
            echo "ep_validation=MISMATCH (expected=${EXPECTED_EP}, observed=${ep_observed_str})" >> "${meta_file}"
            echo "run_id,supplement_mode,bt,status,failure_reason" > "${summary_file}"
            echo "${run_id},true,${bt},EP_MISMATCH,expected=${EXPECTED_EP} observed=${ep_observed_str}" >> "${summary_file}"
            stop_containers; return 1
        fi
        echo "ep_validation=OK (expected=${EXPECTED_EP}, observed=${ep_observed_str})" >> "${meta_file}"
    else
        echo "ep_validation=SKIPPED" >> "${meta_file}"
    fi

    # Extended correctness
    if ${RUN_CORRECTNESS}; then
        correctness_check_extended "${bt}" "${correctness_file}"
    else
        log "Skipping correctness check (--no-correctness)"
        echo "correctness_skipped=true" >> "${meta_file}"
    fi

    # Decode-only benchmark
    local decode_ok=true
    decode_bench_supplement "${run_dir}" "${meta_file}" "${bt}" || decode_ok=false

    # Summary
    local decode_result
    decode_result=$(grep '^decode_bench_result=' "${meta_file}" | tail -1 | cut -d= -f2-)
    local overall_status="COMPLETE"
    ${decode_ok} || overall_status="COMPLETE_DECODE_FAIL"
    {
        echo "run_id,supplement_mode,bt,status,decode_bench_result"
        echo "${run_id},true,${bt},${overall_status},${decode_result}"
    } > "${summary_file}"

    log "Supplement run complete. Results in ${run_dir}/"

    # Stop containers
    stop_containers

    # Post-stop memory (informational for supplement; log but no halt gate)
    log "Post-stop memory check..."
    local s01 s02
    s01=$(node_free_gib "${SPARK01}" 2>/dev/null || echo "unknown")
    s02=$(node_free_gib "${SPARK02}" 2>/dev/null || echo "unknown")
    echo "post_stop_free_spark01_gib=${s01}" >> "${meta_file}"
    echo "post_stop_free_spark02_gib=${s02}" >> "${meta_file}"
    if check_memory_safe "${SPARK01}"; then
        echo "post_stop_memory_safe_spark01=OK" >> "${meta_file}"
    else
        warn "spark01 post-stop memory below ${SAFE_MEM_GIB} GiB — reboot recommended before next run."
        echo "post_stop_memory_safe_spark01=UNSAFE" >> "${meta_file}"
    fi
    if check_memory_safe "${SPARK02}"; then
        echo "post_stop_memory_safe_spark02=OK" >> "${meta_file}"
    else
        warn "spark02 post-stop memory below ${SAFE_MEM_GIB} GiB — reboot recommended before next run."
        echo "post_stop_memory_safe_spark02=UNSAFE" >> "${meta_file}"
    fi

    log "Supplement run done."
    return 0
}

# ---------------------------------------------------------------------------
# Memory safety check (post-stop only)
# ---------------------------------------------------------------------------
check_memory_safe() {
    local host="$1"
    local free_gib
    free_gib=$(node_free_gib "${host}")
    log "${host}: post-stop free memory = ${free_gib} GiB (threshold: ${SAFE_MEM_GIB} GiB)"
    if awk "BEGIN { exit (${free_gib} < ${SAFE_MEM_GIB}) }"; then
        return 0  # safe
    else
        return 1  # unsafe
    fi
}

# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------
start_containers() {
    local env_file="$1"
    local remote_env="${REMOTE_REPO_ROOT}/.local/env/step37/bt-matrix-run.env"

    log "Copying env file to spark01 and spark02..."
    run scp "${env_file}" "${SPARK01}:${remote_env}"
    run scp "${env_file}" "${SPARK02}:${remote_env}"

    log "Starting head on spark01 and worker on spark02 simultaneously..."
    run ssh "${SPARK01}" "cd '${REMOTE_REPO_ROOT}' && docker compose --env-file '${remote_env}' --profile head up -d" &
    local pid_head=$!
    run ssh "${SPARK02}" "cd '${REMOTE_REPO_ROOT}' && docker compose --env-file '${remote_env}' --profile worker up -d" &
    local pid_worker=$!

    wait "${pid_head}" || die "Failed to start head container on spark01"
    wait "${pid_worker}" || die "Failed to start worker container on spark02"
    log "Containers started."
}

stop_containers() {
    log "Stopping head on spark01 and worker on spark02 simultaneously..."
    run ssh "${SPARK01}" "cd '${REMOTE_REPO_ROOT}' && docker compose --profile head down" &
    local pid_head=$!
    run ssh "${SPARK02}" "cd '${REMOTE_REPO_ROOT}' && docker compose --profile worker down" &
    local pid_worker=$!

    wait "${pid_head}" || warn "Head container stop may have failed (spark01)"
    wait "${pid_worker}" || warn "Worker container stop may have failed (spark02)"
    log "Containers stopped."
}

# Apply prometheus_fastapi_instrumentator 8.0.0 routing.py compatibility fix to the
# running head container. Affected image: v023-step3p7-fixed-kv-profile-skip-candidate.
#
# Root cause: vLLM 0.23.0 registers routes using FastAPI's include_router(), which
# produces _IncludedRouter objects that lack a .path attribute. routing.py's
# _get_route_name() calls route.path unconditionally, raising AttributeError inside
# the Prometheus middleware on every HTTP request — including /health — causing all
# requests to return HTTP 500 before the model handler runs.
#
# Fix: replace the two occurrences of `route_name = route.path` with
# `route_name = getattr(route, 'path', 'unknown')` in routing.py.
#
# Worker-side patch: NOT required — prometheus instrumentator is only loaded by the
# head API server (vllm/entrypoints/serve/instrumentator/metrics.py). The worker
# process does not start an HTTP server.
#
# Simultaneous restart: required after patching. Restarting only head breaks the
# TCPStore address that spark02 worker is connected to, causing a 601s timeout.
# Restarting both simultaneously forces a clean re-join.
#
# Args: meta_file (optional) — path to write patch state fields (key=value lines).
#       Returns 0 on success, 1 on failure. Containers are NOT stopped on failure;
#       caller is responsible for calling stop_containers and returning an error.
#
# Known checksums (prometheus-fastapi-instrumentator==8.0.0):
#   original SHA256: b90d08f601c5ec82245630667c0cbc031f00df038284b4e61f46945d182c85fb
#   patched  SHA256: a3addfd90d1132a5ab5dca54c788f4743fe180b9607a662bf34ef0453750848c
apply_prometheus_patch() {
    local meta_file="${1:-/dev/null}"
    local patch_src
    patch_src=$(mktemp /tmp/fix_prometheus_XXXXXX.py)
    cat > "${patch_src}" << 'PYEOF'
import os, sys, glob, hashlib, subprocess

EXPECTED_ORIG    = "b90d08f601c5ec82245630667c0cbc031f00df038284b4e61f46945d182c85fb"
EXPECTED_PATCHED = "a3addfd90d1132a5ab5dca54c788f4743fe180b9607a662bf34ef0453750848c"
fp = '/usr/local/lib/python3.12/dist-packages/prometheus_fastapi_instrumentator/routing.py'

with open(fp, 'rb') as f:
    raw = f.read()
sha_before = hashlib.sha256(raw).hexdigest()
print(f'prometheus_routing_sha256_before={sha_before}')

if sha_before == EXPECTED_PATCHED:
    print('prometheus_patch_status=already_patched')
    sys.exit(0)

if sha_before != EXPECTED_ORIG:
    print(f'prometheus_patch_status=PRECONDITION_FAILED (unexpected sha256_before={sha_before})')
    sys.exit(1)

txt = raw.decode()
fixed = txt.replace("route_name = route.path", "route_name = getattr(route, 'path', 'unknown')")
if fixed == txt:
    print('prometheus_patch_status=PRECONDITION_FAILED (pattern not found despite matching sha256)')
    sys.exit(1)

with open(fp, 'w') as f:
    f.write(fixed)

with open(fp, 'rb') as f:
    sha_after = hashlib.sha256(f.read()).hexdigest()
print(f'prometheus_routing_sha256_after={sha_after}')

if sha_after != EXPECTED_PATCHED:
    print(f'prometheus_patch_status=VERIFY_FAILED (sha_after={sha_after})')
    sys.exit(1)

result = subprocess.run([sys.executable, '-m', 'py_compile', fp], capture_output=True, text=True)
if result.returncode != 0:
    print(f'prometheus_patch_status=SYNTAX_CHECK_FAILED ({result.stderr.strip()})')
    sys.exit(1)

pycache = os.path.dirname(fp) + '/__pycache__'
for pyc in glob.glob(pycache + '/routing*.pyc'):
    os.remove(pyc)

print('prometheus_patch_status=patched')
PYEOF
    # shellcheck disable=SC2064
    trap "rm -f '${patch_src}'" RETURN

    log "Applying prometheus routing.py compatibility patch to head container..."
    scp "${patch_src}" "${SPARK01}:/tmp/fix_prometheus_routing.py" 2>/dev/null \
        || { warn "Failed to copy prometheus patch script to spark01"; return 1; }
    local patch_out
    patch_out=$(ssh "${SPARK01}" \
        "docker cp /tmp/fix_prometheus_routing.py vllm-spark-head:/tmp/fix_prometheus_routing.py \
         && docker exec vllm-spark-head python3 /tmp/fix_prometheus_routing.py" 2>&1)
    local patch_rc=$?
    log "Patch output: ${patch_out}"

    # Append patch state fields to meta_file
    echo "${patch_out}" >> "${meta_file}"

    if [[ ${patch_rc} -ne 0 ]]; then
        warn "prometheus patch script returned non-zero (${patch_rc})"
        return 1
    fi
    if echo "${patch_out}" | grep -q 'prometheus_patch_status=.*FAILED'; then
        warn "prometheus patch reported failure: $(echo "${patch_out}" | grep 'prometheus_patch_status=')"
        return 1
    fi

    # Restart both containers simultaneously — see function comment for rationale.
    log "Restarting both containers simultaneously to activate prometheus patch..."
    ssh "${SPARK01}" "cd '${REMOTE_REPO_ROOT}' && docker compose --profile head restart" &
    local pid_head=$!
    ssh "${SPARK02}" "cd '${REMOTE_REPO_ROOT}' && docker compose --profile worker restart" &
    local pid_worker=$!
    wait "${pid_head}" || { warn "Head container restart failed"; return 1; }
    wait "${pid_worker}" || warn "Worker container restart may have failed (non-fatal — worker will reconnect)"
    log "Containers restarted with prometheus patch active."
    return 0
}

# ---------------------------------------------------------------------------
# Single bt run
# ---------------------------------------------------------------------------
run_bt() {
    local bt="$1"
    local run_id="bt${bt}-$(date +%Y%m%d-%H%M%S)"
    local run_dir="${RESULT_DIR}/${run_id}"
    local env_file="${REPO_ROOT}/.local/env/step37/bt-matrix-run.env"
    local bench_out="${run_dir}/bench.md"
    local meta_file="${run_dir}/metadata.txt"
    local correctness_file="${run_dir}/correctness.md"
    local summary_file="${run_dir}/summary.csv"

    log "======================================================"
    log "Starting bt=${bt} run (id=${run_id})"
    log "======================================================"

    mkdir -p "${run_dir}"

    # --- Generate run env file ---
    log "Generating env file with MAX_NUM_BATCHED_TOKENS=${bt}..."
    sed "s/__BT_PLACEHOLDER__/${bt}/" "${TEMPLATE_ENV}" > "${env_file}"

    # Derive topology metadata from template env.
    # Check VLLM_EXTRA_ARGS line only — comments may contain the flag text.
    local ep_requested="unknown"
    if grep '^VLLM_EXTRA_ARGS=' "${TEMPLATE_ENV}" 2>/dev/null | grep -q -- '--enable-expert-parallel'; then
        ep_requested="true"
    elif grep '^VLLM_EXTRA_ARGS=' "${TEMPLATE_ENV}" 2>/dev/null | grep -q -- '--no-enable-expert-parallel'; then
        ep_requested="false"
    else
        ep_requested="false (flag absent from VLLM_EXTRA_ARGS)"
    fi
    local distributed_backend
    distributed_backend=$(grep '^DISTRIBUTED_BACKEND=' "${TEMPLATE_ENV}" 2>/dev/null | cut -d= -f2 || echo "unknown")
    local moe_backend_req
    moe_backend_req=$(grep -oP '(?<=--moe-backend )\S+' "${TEMPLATE_ENV}" 2>/dev/null | head -1 || echo "unknown")
    local attn_backend_req
    attn_backend_req=$(grep -oP '(?<=--attention-backend )\S+' "${TEMPLATE_ENV}" 2>/dev/null | head -1 || echo "unknown")

    # Derive config_label if not provided
    local effective_config_label="${CONFIG_LABEL}"
    if [[ -z "${effective_config_label}" ]]; then
        local ep_tag
        [[ "${ep_requested}" == "true" ]] && ep_tag="ep-on" || ep_tag="ep-off"
        effective_config_label="v023-${moe_backend_req,,}-${attn_backend_req,,}-${ep_tag}-bt${bt}"
    fi

    # Boot ID and uptime at run start (for reboot verification)
    local head_boot_id worker_boot_id head_uptime_s worker_uptime_s
    head_boot_id=$(node_boot_id "${SPARK01}" 2>/dev/null || echo "unknown")
    worker_boot_id=$(node_boot_id "${SPARK02}" 2>/dev/null || echo "unknown")
    head_uptime_s=$(node_uptime_seconds "${SPARK01}" 2>/dev/null || echo "unknown")
    worker_uptime_s=$(node_uptime_seconds "${SPARK02}" 2>/dev/null || echo "unknown")

    # Pre-start memory (informational only — NOT a gate).
    # On GB10 UMA, a running vLLM server holds ~75 GiB, leaving ~46 GiB free.
    # SAFE_MEM_GIB applies only to the post-stop check below.
    local spark01_free_before spark02_free_before
    spark01_free_before=$(node_free_gib "${SPARK01}" 2>/dev/null || echo "unknown")
    spark02_free_before=$(node_free_gib "${SPARK02}" 2>/dev/null || echo "unknown")

    # Record metadata
    {
        echo "run_id=${run_id}"
        echo "bt=${bt}"
        echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "config_label=${effective_config_label}"
        echo "template_env=${TEMPLATE_ENV}"
        echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "model=${SERVED_MODEL_NAME}"
        echo "image=$(grep '^VLLM_IMAGE=' "${env_file}" | cut -d= -f2)"
        echo "max_model_len=$(grep '^MAX_MODEL_LEN=' "${env_file}" | cut -d= -f2)"
        echo "max_num_seqs=$(grep '^MAX_NUM_SEQS=' "${env_file}" | cut -d= -f2)"
        echo "gpu_util=$(grep '^GPU_MEMORY_UTILIZATION=' "${env_file}" | cut -d= -f2)"
        echo "max_num_batched_tokens=${bt}"
        echo "distributed_backend=${distributed_backend}"
        echo "expert_parallel_requested=${ep_requested}"
        echo "expert_parallel_expected_ep_flag=${EXPECTED_EP}"
        echo "moe_backend_requested=${moe_backend_req}"
        echo "attention_backend_requested=${attn_backend_req}"
        echo "bench_pp=${BENCH_PP}"
        echo "bench_tg=${BENCH_TG}"
        echo "bench_depths=${BENCH_DEPTHS}"
        echo "bench_runs=${BENCH_RUNS}"
        echo "api_ready_timeout=${API_READY_TIMEOUT:-600}"
        echo "decode_context_mode=pp${BENCH_PP}"
        # Boot ID and uptime — use to verify both nodes were rebooted since prior run.
        # Compare head_boot_id / worker_boot_id against prior run's metadata to confirm
        # a fresh boot session. Different IDs = confirmed reboot between runs.
        echo "head_boot_id=${head_boot_id}"
        echo "worker_boot_id=${worker_boot_id}"
        echo "head_uptime_seconds=${head_uptime_s}"
        echo "worker_uptime_seconds=${worker_uptime_s}"
        # Pre-start memory (informational, not a gate — see SAFE_MEM_GIB comment above)
        echo "spark01_free_gib_before=${spark01_free_before}"
        echo "spark02_free_gib_before=${spark02_free_before}"
        # spark02 nvidia-smi note: GB10 UMA causes nvidia-smi --query-gpu CSV to return
        # N/A for GPU memory fields. /proc/meminfo MemAvailable is the correct metric.
        echo "memory_metric_source=proc_meminfo_MemAvailable (nvidia-smi N/A on GB10 UMA)"
    } > "${meta_file}"

    if ${DRY_RUN}; then
        log "[DRY-RUN] Would start containers, wait for API, run benchmark, stop containers."
        log "[DRY-RUN] Env file: ${env_file}"
        log "[DRY-RUN] head_boot_id=${head_boot_id} uptime=${head_uptime_s}s"
        log "[DRY-RUN] worker_boot_id=${worker_boot_id} uptime=${worker_uptime_s}s"
        log "[DRY-RUN] spark01 free before: ${spark01_free_before} GiB"
        log "[DRY-RUN] spark02 free before: ${spark02_free_before} GiB"
        echo "run_id,bt,status,pp2048_tps,tg32_tps" > "${summary_file}"
        echo "${run_id},${bt},DRY_RUN,N/A,N/A" >> "${summary_file}"
        return 0
    fi

    # --- Start containers ---
    start_containers "${env_file}"

    # --- Patch prometheus routing.py (required before API readiness check) ---
    # prometheus_fastapi_instrumentator raises AttributeError on every HTTP request
    # when _IncludedRouter objects lack the 'path' attribute. This returns 500 for
    # /health, blocking the readiness check. Patch immediately after container start
    # (before model load completes) so only one model load is needed.
    if ! apply_prometheus_patch "${meta_file}"; then
        warn "bt=${bt}: prometheus patch failed — aborting run."
        echo "startup_result=PROMETHEUS_PATCH_FAILED" >> "${meta_file}"
        echo "run_id,bt,status,failure_reason" > "${summary_file}"
        echo "${run_id},${bt},STARTUP_FAIL,prometheus routing.py patch failed" >> "${summary_file}"
        stop_containers
        return 1
    fi

    # --- Wait for API ---
    local api_ready=false
    if wait_for_api; then
        api_ready=true
    else
        warn "bt=${bt}: API not ready — recording STARTUP_FAIL and stopping."
        echo "startup_result=STARTUP_FAIL" >> "${meta_file}"
        echo "run_id,bt,status,failure_reason" > "${summary_file}"
        echo "${run_id},${bt},STARTUP_FAIL,API not ready after 600s" >> "${summary_file}"
        stop_containers
        return 1
    fi
    echo "startup_result=OK" >> "${meta_file}"

    # --- Extract server startup log ---
    log "Fetching startup log from head container..."
    ssh "${SPARK01}" "docker logs vllm-spark-head 2>&1" > "${run_dir}/head-startup.log" 2>&1 || true

    # --- Backend verification (garble-fix guards) ---
    # Both are hard gates — mismatch stops containers and exits 1 (never bench-continuable).
    local marlin_ok marlin_line triton_ok triton_line
    marlin_line=$(grep "Using 'MARLIN' NvFp4 MoE backend" "${run_dir}/head-startup.log" 2>/dev/null | head -1 || echo "")
    marlin_ok=$(echo "${marlin_line}" | grep -c "MARLIN" || echo 0)
    triton_line=$(grep "Using AttentionBackendEnum.TRITON_ATTN backend" "${run_dir}/head-startup.log" 2>/dev/null | head -1 || echo "")
    triton_ok=$(echo "${triton_line}" | grep -c "TRITON_ATTN" || echo 0)

    # EP detection: entrypoint command line is the authoritative source.
    # vLLM 0.23 does NOT log expert_parallel_size=1 when EP is disabled (default).
    local entrypoint_cmd ep_observed_str ep_evidence
    entrypoint_cmd=$(grep '\[entrypoint\] Running: vllm serve' "${run_dir}/head-startup.log" 2>/dev/null | head -1 || echo "")
    if [[ -z "${entrypoint_cmd}" ]]; then
        ep_observed_str="not_found_in_logs"
        ep_evidence="entrypoint command line not captured in head-startup.log"
    elif echo "${entrypoint_cmd}" | grep -q -- '--enable-expert-parallel'; then
        ep_observed_str="enabled"
        ep_evidence="--enable-expert-parallel present in entrypoint command line"
    else
        ep_observed_str="disabled"
        ep_evidence="--enable-expert-parallel absent from entrypoint command line"
    fi

    {
        echo "marlin_confirmed=${marlin_ok}"
        echo "marlin_evidence_line=${marlin_line:-not found}"
        echo "triton_attn_confirmed=${triton_ok}"
        echo "triton_evidence_line=${triton_line:-not found}"
        echo "expert_parallel_observed=${ep_observed_str}"
        echo "expert_parallel_evidence=${ep_evidence}"
        echo "backend_validation_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >> "${meta_file}"

    # MARLIN hard gate
    if [[ "${marlin_ok}" -lt 1 ]]; then
        warn "bt=${bt}: MARLIN MoE backend NOT confirmed in logs — FATAL."
        warn "  Expected: \"Using 'MARLIN' NvFp4 MoE backend\""
        warn "  Found:    ${marlin_line:-<nothing>}"
        echo "backend_validity=INVALID_MARLIN_NOT_CONFIRMED" >> "${meta_file}"
        stop_containers
        return 1
    fi

    # TRITON_ATTN hard gate
    if [[ "${triton_ok}" -lt 1 ]]; then
        warn "bt=${bt}: TRITON_ATTN attention backend NOT confirmed in logs — FATAL."
        warn "  Expected: \"Using AttentionBackendEnum.TRITON_ATTN backend\""
        warn "  Found:    ${triton_line:-<nothing>}"
        echo "backend_validity=INVALID_TRITON_NOT_CONFIRMED" >> "${meta_file}"
        stop_containers
        return 1
    fi

    echo "backend_validity=OK (MARLIN confirmed, TRITON_ATTN confirmed)" >> "${meta_file}"

    # EP validation against --expected-ep (always fatal, never bench-continuable)
    if [[ "${EXPECTED_EP}" != "unknown" ]]; then
        local ep_match=true
        if [[ "${EXPECTED_EP}" == "on" ]] && [[ "${ep_observed_str}" != "enabled" ]]; then
            ep_match=false
        elif [[ "${EXPECTED_EP}" == "off" ]] && [[ "${ep_observed_str}" != "disabled" ]]; then
            ep_match=false
        fi
        if ! ${ep_match}; then
            warn "bt=${bt}: EP state mismatch — FATAL."
            warn "  expected-ep=${EXPECTED_EP}, observed=${ep_observed_str}"
            warn "  evidence: ${ep_evidence}"
            warn "  Check template env and --expected-ep flag. Halting."
            echo "ep_validation=MISMATCH (expected=${EXPECTED_EP}, observed=${ep_observed_str})" >> "${meta_file}"
            stop_containers
            return 1
        else
            echo "ep_validation=OK (expected=${EXPECTED_EP}, observed=${ep_observed_str})" >> "${meta_file}"
        fi
    else
        echo "ep_validation=SKIPPED (--expected-ep not set; review expert_parallel_observed field)" >> "${meta_file}"
    fi

    # --- Correctness check ---
    correctness_check "${bt}" "${correctness_file}"

    # --- Benchmark ---
    log "Running llama-benchy (bt=${bt}, runs=${BENCH_RUNS})..."
    local depth_args=""
    for d in ${BENCH_DEPTHS}; do
        depth_args="${depth_args} ${d}"
    done

    # shellcheck disable=SC2086
    local bench_exit=0
    ssh "${SPARK01}" "PYTHONUNBUFFERED=1 ${LLAMA_BENCHY_BIN} \
        --base-url '${API_BASE}' \
        --model '${SERVED_MODEL_NAME}' \
        --tokenizer '${TOKENIZER_PATH}' \
        --pp ${BENCH_PP} \
        --tg ${BENCH_TG} \
        --depth ${depth_args} \
        --runs ${BENCH_RUNS} \
        --format md \
        --save-result '/tmp/bench-bt-${bt}.md'" \
        > "${run_dir}/bench-stdout.log" 2>&1 \
        && scp "${SPARK01}:/tmp/bench-bt-${bt}.md" "${bench_out}" \
        || bench_exit=$?

    if [[ ${bench_exit} -ne 0 ]]; then
        warn "Benchmark request failed for bt=${bt} (exit ${bench_exit}). This is a bench-continuable failure."
        echo "bench_result=FAIL (exit=${bench_exit})" >> "${meta_file}"
    elif [[ -f "${bench_out}" ]]; then
        echo "bench_result=OK" >> "${meta_file}"
    else
        warn "Benchmark output file not found for bt=${bt}."
        echo "bench_result=MISSING_OUTPUT" >> "${meta_file}"
        bench_exit=99
    fi

    # --- Extract key numbers for CSV summary ---
    {
        echo "run_id,bt,status,pp2048_tps,pp2048_tps_sd,tg32_tps,tg32_tps_peak,pp_d4096_tps,pp_d8192_tps,pp_d16384_tps,tg_d4096_tps,tg_d8192_tps,tg_d16384_tps"
        if [[ -f "${bench_out}" ]]; then
            python3 - <<'PYEOF' "${bench_out}" "${run_id}" "${bt}"
import sys, re

result_file = sys.argv[1]
run_id = sys.argv[2]
bt = sys.argv[3]

lines = open(result_file).readlines()
values = {}

for line in lines:
    line = line.strip()
    if not line.startswith('|') or 'model' in line or '---' in line:
        continue
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 4:
        continue
    test = parts[2].strip()
    tps_raw = parts[3].strip()
    tps_match = re.match(r'([\d.]+)\s*[±]\s*([\d.]+)', tps_raw)
    if tps_match:
        tps = tps_match.group(1)
        sd = tps_match.group(2)
    else:
        tps_match = re.match(r'([\d.]+)', tps_raw)
        tps = tps_match.group(1) if tps_match else 'N/A'
        sd = 'N/A'
    values[test] = (tps, sd)

for line in lines:
    line = line.strip()
    if not line.startswith('|') or 'model' in line or '---' in line:
        continue
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 5:
        continue
    test = parts[2].strip()
    peak_raw = parts[4].strip()
    peak_match = re.match(r'([\d.]+)', peak_raw)
    if peak_match and test in values:
        values[test] = (values[test][0], values[test][1], peak_match.group(1))

def get(test, idx=0, default='N/A'):
    v = values.get(test, ())
    return v[idx] if idx < len(v) else default

pp = get('pp2048', 0)
pp_sd = get('pp2048', 1)
tg = get('tg32', 0)
tg_peak = get('tg32', 2) if len(values.get('tg32', ())) > 2 else 'N/A'
pp_d4 = get('pp2048 @ d4096', 0)
pp_d8 = get('pp2048 @ d8192', 0)
pp_d16 = get('pp2048 @ d16384', 0)
tg_d4 = get('tg32 @ d4096', 0)
tg_d8 = get('tg32 @ d8192', 0)
tg_d16 = get('tg32 @ d16384', 0)

print(f"{run_id},{bt},OK,{pp},{pp_sd},{tg},{tg_peak},{pp_d4},{pp_d8},{pp_d16},{tg_d4},{tg_d8},{tg_d16}")
PYEOF
        else
            echo "${run_id},${bt},NO_BENCH_FILE,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A"
        fi
    } > "${summary_file}"

    log "bt=${bt} benchmark complete. Results in ${run_dir}/"

    # --- Stop containers ---
    if ! ${NO_STOP}; then
        stop_containers

        # Post-stop memory safety check (SAFE_MEM_GIB gate applies here only).
        # If memory does not recover, GB10 UMA driver is retaining pages.
        # Only reboot can recover them; rmmod nvidia_uvm does NOT help.
        log "Checking post-stop memory recovery (threshold: ${SAFE_MEM_GIB} GiB)..."
        local safe=true
        if ! check_memory_safe "${SPARK01}"; then
            warn "spark01 post-stop memory below ${SAFE_MEM_GIB} GiB — GB10 UMA driver leak."
            warn "Reboot spark01 before the next run. rmmod nvidia_uvm does not help."
            echo "post_stop_memory_safe_spark01=UNSAFE" >> "${meta_file}"
            safe=false
        else
            echo "post_stop_memory_safe_spark01=OK" >> "${meta_file}"
        fi

        if ! check_memory_safe "${SPARK02}"; then
            warn "spark02 post-stop memory below ${SAFE_MEM_GIB} GiB — GB10 UMA driver leak."
            warn "Reboot spark02 before the next run."
            echo "post_stop_memory_safe_spark02=UNSAFE" >> "${meta_file}"
            safe=false
        else
            echo "post_stop_memory_safe_spark02=OK" >> "${meta_file}"
        fi

        if ! ${safe}; then
            warn "Memory not recovered. Halting matrix run."
            warn "Reboot affected nodes and resume with --skip-bt for completed values."
            return 2
        fi
    fi

    # Propagate bench failure as exit 3 (only bench-continuable failure code)
    if [[ ${bench_exit} -ne 0 ]]; then
        log "bt=${bt} complete with bench failure (exit 3)."
        return 3
    fi

    log "bt=${bt} complete."
    return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log "bench-bt-matrix-step37-v023.sh"
    log "Repository:        ${REPO_ROOT}"
    log "Template:          ${TEMPLATE_ENV}"
    log "Results:           ${RESULT_DIR}"
    [[ ${#BT_VALUES[@]} -gt 0 ]] && log "bt values:         ${BT_VALUES[*]}"
    log "Runs/test:         ${BENCH_RUNS}"
    log "Depths:            ${BENCH_DEPTHS}"
    log "expected-ep:       ${EXPECTED_EP}"
    log "bench-continuable: ${CONTINUE_ON_BENCH_FAIL}"
    [[ -n "${CONFIG_LABEL}" ]] && log "config-label:      ${CONFIG_LABEL}"
    ${DRY_RUN}         && log "DRY-RUN MODE: no containers will be started."
    ${PREFLIGHT_ONLY}  && log "PREFLIGHT-ONLY MODE: read-only checks, no container ops."
    ${VALIDATE_EXISTING} && log "VALIDATE-EXISTING-CONTAINER MODE: read-only backend detection."
    ${SERVER_ONLY_MODE} && log "SERVER-ONLY MODE: start bt=${BT_VALUES[0]:-?} server, validate, keep running."
    ${SUPPLEMENT_MODE}  && log "SUPPLEMENT MODE: bt=${BT_VALUES[0]:-?}, extended correctness + decode-only bench."
    echo ""

    # --- Validate API_READY_TIMEOUT early (before any container/SSH work) ---
    # Default 600s preserves prior behavior when unset. bt=8192 startup measured
    # ~610s on dual GB10, exceeding the former hard-coded 600s readiness limit, so
    # API_READY_TIMEOUT makes only the readiness wait configurable (it does not
    # affect benchmark request timeouts). Reject non-positive / non-integer values
    # here so a bad value fails fast instead of after starting containers.
    local _api_timeout="${API_READY_TIMEOUT:-600}"
    if ! [[ "${_api_timeout}" =~ ^[1-9][0-9]*$ ]]; then
        die "Invalid API_READY_TIMEOUT='${_api_timeout}': must be a positive integer (seconds)."
    fi
    log "API readiness timeout: ${_api_timeout}s (default 600; override via API_READY_TIMEOUT)."

    mkdir -p "${RESULT_DIR}/logs"

    # --- Preflight-only mode ---
    if ${PREFLIGHT_ONLY}; then
        preflight_check
        exit $?
    fi

    # --- Validate-existing-container mode ---
    if ${VALIDATE_EXISTING}; then
        validate_existing_container
        exit $?
    fi

    # --- Server-only mode ---
    if ${SERVER_ONLY_MODE}; then
        [[ ${#BT_VALUES[@]} -eq 1 ]] || die "--server-only requires exactly one --bt value"
        [[ -f "${TEMPLATE_ENV}" ]] || die "Template env not found: ${TEMPLATE_ENV}"
        server_only_run "${BT_VALUES[0]}"
        exit $?
    fi

    # --- Supplement mode ---
    if ${SUPPLEMENT_MODE}; then
        [[ ${#BT_VALUES[@]} -eq 1 ]] || die "--supplement requires exactly one --bt value"
        [[ -f "${TEMPLATE_ENV}" ]] || die "Template env not found: ${TEMPLATE_ENV}"
        ${DRY_RUN} || ssh "${SPARK01}" "test -x '${LLAMA_BENCHY_BIN}'" \
            || die "llama-benchy not found on spark01: ${LLAMA_BENCHY_BIN}"
        supplement_run "${BT_VALUES[0]}"
        exit $?
    fi

    # --- Normal benchmark run ---

    # Validate prerequisites
    [[ -f "${TEMPLATE_ENV}" ]] || die "Template env not found: ${TEMPLATE_ENV}"
    [[ -f "${COMPOSE_FILE}" ]] || die "docker-compose.yml not found: ${COMPOSE_FILE}"
    ${DRY_RUN} || ssh "${SPARK01}" "test -x '${LLAMA_BENCHY_BIN}'" \
        || die "llama-benchy not found on spark01: ${LLAMA_BENCHY_BIN}"
    ${DRY_RUN} || ssh "${SPARK01}" "test -f '${TOKENIZER_PATH}/tokenizer.json'" \
        || die "Tokenizer not found on spark01: ${TOKENIZER_PATH}"

    # Sanity check: warn if any vllm containers are currently running
    if ! ${DRY_RUN}; then
        local running_head running_worker
        running_head=$(container_running "${SPARK01}" "vllm-spark-head")
        running_worker=$(container_running "${SPARK02}" "vllm-spark-worker")
        if [[ "${running_head}" == "true" ]] || [[ "${running_worker}" == "true" ]]; then
            warn "========================================================"
            warn "EXISTING CONTAINERS DETECTED:"
            warn "  spark01 vllm-spark-head:   ${running_head}"
            warn "  spark02 vllm-spark-worker: ${running_worker}"
            warn "The runner will stop these before each new run."
            warn "If this is unexpected, Ctrl-C now and verify state."
            warn "========================================================"
            sleep 3
        fi
    fi

    # --all confirmation prompt (skip in dry-run)
    if ${RUN_ALL} && ! ${DRY_RUN}; then
        warn "--all flag: will run ${#BT_VALUES[@]} bt values sequentially."
        warn "Each run stops and restarts containers. GB10 UMA may accumulate."
        warn "Strongly recommended: reboot nodes between runs to prevent UMA thrash."
        read -r -p "Confirm full matrix run? [yes/N]: " confirm
        [[ "${confirm}" == "yes" ]] || { echo "Aborted."; exit 0; }
    fi

    # Master CSV aggregation file
    local master_csv="${RESULT_DIR}/matrix-summary-$(date +%Y%m%d-%H%M%S).csv"
    echo "run_id,bt,status,pp2048_tps,pp2048_tps_sd,tg32_tps,tg32_tps_peak,pp_d4096_tps,pp_d8192_tps,pp_d16384_tps,tg_d4096_tps,tg_d8192_tps,tg_d16384_tps" > "${master_csv}"

    local completed=0
    local failed=0

    for bt in "${BT_VALUES[@]}"; do
        log ""
        local run_exit=0
        run_bt "${bt}" || run_exit=$?

        if [[ ${run_exit} -eq 0 ]]; then
            local run_csv
            run_csv=$(find "${RESULT_DIR}" -name "summary.csv" -newer "${master_csv}" | sort | tail -1)
            [[ -n "${run_csv}" ]] && tail -1 "${run_csv}" >> "${master_csv}" || true
            completed=$((completed + 1))
            if ! ${AUTO_NEXT_BT}; then
                log "bt=${bt}: --no-auto-next-bt set; halting after first completed run."
                break
            fi
        elif [[ ${run_exit} -eq 2 ]]; then
            warn "Halting matrix run due to memory safety failure (exit 2)."
            warn "Completed: ${completed}/${#BT_VALUES[@]} runs."
            warn "Resume after reboot: --bt <remaining_values> --skip-bt <completed_values>"
            break
        elif [[ ${run_exit} -eq 3 ]]; then
            failed=$((failed + 1))
            if ! ${CONTINUE_ON_BENCH_FAIL}; then
                warn "bt=${bt}: benchmark request failed (exit 3). Halting."
                warn "Use --continue-on-bench-fail to skip request failures and continue."
                break
            else
                warn "bt=${bt}: benchmark request failed. --continue-on-bench-fail set, continuing."
            fi
        else
            failed=$((failed + 1))
            warn "bt=${bt}: FATAL failure (exit ${run_exit}). Topology/startup/backend error — not continuable."
            warn "Fix the underlying issue before retrying."
            break
        fi
    done

    log ""
    log "======================================================"
    log "Matrix run complete."
    log "  Completed: ${completed}"
    log "  Failed:    ${failed}"
    log "  Results:   ${RESULT_DIR}/"
    log "  Master CSV: ${master_csv}"
    log "======================================================"
    log ""
    log "Next step: review ${master_csv} and generate analysis report."
    log "  bash benchmarks/analyze-bt-matrix.sh ${RESULT_DIR}"
}

main "$@"
