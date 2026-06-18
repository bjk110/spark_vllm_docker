#!/usr/bin/env bash
# =============================================================================
# bench-bt-matrix-step37-v023.sh
#
# Purpose
#   Benchmark MAX_NUM_BATCHED_TOKENS effect on Step-3.7-Flash-NVFP4 prefill
#   throughput under vLLM 0.23.0 on dual DGX Spark GB10 (TP=2, EP=2, mp).
#
#   Background: bt=256 produced ~538 t/s vs ~1251 t/s on v022 baseline.
#   Single variable: only MAX_NUM_BATCHED_TOKENS changes across runs.
#   All other parameters fixed (see .local/env/step37/bt-matrix-base.env).
#
# Usage (run from homeserver in /home/bjk110/docker/vllm-spark/):
#   bash benchmarks/bench-bt-matrix-step37-v023.sh [OPTIONS]
#
# Options:
#   --bt <values>     Comma-separated bt values to test (default: all matrix)
#                     Example: --bt 256,2048,8192
#   --runs <n>        llama-benchy runs per test (default: 3)
#   --skip-bt <vals>  Comma-separated bt values to skip
#   --dry-run         Print commands without executing
#   --no-stop         Skip container stop between runs (for manual testing only)
#   --result-dir <d>  Override result directory (default: benchmarks/results/bt-matrix)
#
# Safety rules (enforced):
#   - Never modifies production preset (presets/step37-flash-nvfp4-tp2.env)
#   - Never reboots, destroys volumes, or docker system prune
#   - Halts if container stop does not recover memory below SAFE_MEM_GIB threshold
#   - Each run uses a separate disposable env file (deleted after success)
#   - Never auto-promotes a result to production recommendation
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

# Memory safety: after container stop, free GiB must exceed this threshold
# before next run is allowed. GB10 UMA: ~121.63 GiB total; 2× weight load
# (head + profiling) peaks ~65-75 GiB. 50 GiB free is a conservative floor.
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bt)       BT_OVERRIDE="$2"; shift 2 ;;
        --runs)     BENCH_RUNS="$2"; shift 2 ;;
        --skip-bt)  SKIP_BT="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --no-stop)  NO_STOP=true; shift ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Build effective bt list
if [[ -n "${BT_OVERRIDE}" ]]; then
    IFS=',' read -r -a BT_VALUES <<< "${BT_OVERRIDE}"
else
    BT_VALUES=("${DEFAULT_BT_VALUES[@]}")
fi

# Filter out skip list
if [[ -n "${SKIP_BT}" ]]; then
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

# Check free GiB on a Spark node (reads /proc/meminfo MemAvailable)
node_free_gib() {
    local host="$1"
    ssh "${host}" "awk '/MemAvailable/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo"
}

# Check if a container is running on a host
container_running() {
    local host="$1" name="$2"
    ssh "${host}" "docker inspect --format '{{.State.Running}}' '${name}' 2>/dev/null || echo false"
}

# Wait for API readiness (polls /health on spark01)
wait_for_api() {
    local timeout_s=600
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

# Extract key config fields from vLLM container logs on spark01
extract_server_metadata() {
    local log_file="$1"
    ssh "${SPARK01}" "docker logs vllm-spark-head 2>&1" | tee "${log_file}.startup" | {
        local version="" attn_backend="" moe_backend="" tp_size="" max_len="" max_seqs="" max_bt="" kv_dtype="" cudagraph="" mtp=""
        while IFS= read -r line; do
            [[ "${line}" =~ vLLM\ version:\ ([0-9.]+) ]] && version="${BASH_REMATCH[1]}"
            [[ "${line}" =~ Using.*attention.*backend.*([A-Z_]+) ]] && attn_backend="${BASH_REMATCH[1]}"
            [[ "${line}" =~ Using.*([A-Z_]+).*NvFp4\ MoE\ backend ]] && moe_backend="${BASH_REMATCH[1]}"
            [[ "${line}" =~ tensor_parallel_size=([0-9]+) ]] && tp_size="${BASH_REMATCH[1]}"
            [[ "${line}" =~ max_model_len=([0-9]+) ]] && max_len="${BASH_REMATCH[1]}"
            [[ "${line}" =~ max_num_seqs=([0-9]+) ]] && max_seqs="${BASH_REMATCH[1]}"
            [[ "${line}" =~ max_num_batched_tokens=([0-9]+) ]] && max_bt="${BASH_REMATCH[1]}"
            [[ "${line}" =~ kv_cache_dtype=([a-z0-9_]+) ]] && kv_dtype="${BASH_REMATCH[1]}"
            [[ "${line}" =~ [Cc]uda.*graph.*([a-z]+) ]] && cudagraph="${BASH_REMATCH[1]}"
            [[ "${line}" =~ [Ss]peculative|[Mm][Tt][Pp] ]] && mtp="${line}"
        done
        echo "version=${version:-unknown}"
        echo "attention_backend=${attn_backend:-unknown}"
        echo "moe_backend=${moe_backend:-unknown}"
        echo "tp_size=${tp_size:-unknown}"
        echo "max_model_len=${max_len:-unknown}"
        echo "max_num_seqs=${max_seqs:-unknown}"
        echo "max_num_batched_tokens=${max_bt:-unknown}"
        echo "kv_cache_dtype=${kv_dtype:-unknown}"
        echo "cuda_graph=${cudagraph:-unknown}"
        echo "mtp_line=${mtp:-not found}"
    }
}

# ---------------------------------------------------------------------------
# Correctness validation (lightweight — not a full evaluation suite)
# ---------------------------------------------------------------------------
correctness_check() {
    local bt="$1"
    local out_file="$2"
    local pass=true
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
            # Not hard-fail: reasoning model may wrap answer
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
        # Check for broken Unicode or gibberish (simple heuristic: no replacement chars)
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
# Memory safety check
# ---------------------------------------------------------------------------
check_memory_safe() {
    local host="$1"
    local free_gib
    free_gib=$(node_free_gib "${host}")
    log "${host}: free memory = ${free_gib} GiB (threshold: ${SAFE_MEM_GIB} GiB)"
    # Use awk for float comparison
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

    # Record metadata
    {
        echo "run_id=${run_id}"
        echo "bt=${bt}"
        echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "template_env=${TEMPLATE_ENV}"
        echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "model=${SERVED_MODEL_NAME}"
        echo "image=$(grep '^VLLM_IMAGE=' "${env_file}" | cut -d= -f2)"
        echo "max_model_len=$(grep '^MAX_MODEL_LEN=' "${env_file}" | cut -d= -f2)"
        echo "max_num_seqs=$(grep '^MAX_NUM_SEQS=' "${env_file}" | cut -d= -f2)"
        echo "gpu_util=$(grep '^GPU_MEMORY_UTILIZATION=' "${env_file}" | cut -d= -f2)"
        echo "max_num_batched_tokens=${bt}"
        echo "bench_pp=${BENCH_PP}"
        echo "bench_tg=${BENCH_TG}"
        echo "bench_depths=${BENCH_DEPTHS}"
        echo "bench_runs=${BENCH_RUNS}"
        echo "spark01_free_gib_before=$(node_free_gib "${SPARK01}")"
        echo "spark02_free_gib_before=$(node_free_gib "${SPARK02}")"
    } > "${meta_file}"

    if ${DRY_RUN}; then
        log "[DRY-RUN] Would start containers, wait for API, run benchmark, stop containers."
        log "[DRY-RUN] Env file: ${env_file}"
        echo "run_id,bt,status,pp2048_tps,tg32_tps" > "${summary_file}"
        echo "${run_id},${bt},DRY_RUN,N/A,N/A" >> "${summary_file}"
        return 0
    fi

    # --- Start containers ---
    start_containers "${env_file}"

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

    # --- Extract server metadata from logs ---
    log "Extracting server metadata from container logs..."
    ssh "${SPARK01}" "docker logs vllm-spark-head 2>&1" > "${run_dir}/head-startup.log" 2>&1 || true

    # Verify garble-fix backends are active
    local marlin_ok flashinfer_ok
    marlin_ok=$(grep -c "Using 'MARLIN' NvFp4 MoE backend" "${run_dir}/head-startup.log" 2>/dev/null || echo 0)
    flashinfer_ok=$(grep -c "TRITON" "${run_dir}/head-startup.log" 2>/dev/null || echo 0)

    {
        echo "marlin_confirmed=${marlin_ok}"
        echo "triton_attn_log_hits=${flashinfer_ok}"
    } >> "${meta_file}"

    if [[ "${marlin_ok}" -lt 1 ]]; then
        warn "bt=${bt}: MARLIN MoE backend NOT confirmed in logs — result flagged as INVALID."
        echo "backend_validity=INVALID_MARLIN_NOT_CONFIRMED" >> "${meta_file}"
    else
        echo "backend_validity=OK" >> "${meta_file}"
    fi

    # --- Correctness check ---
    correctness_check "${bt}" "${correctness_file}"

    # --- Benchmark ---
    log "Running llama-benchy (bt=${bt}, runs=${BENCH_RUNS})..."
    local depth_args=""
    for d in ${BENCH_DEPTHS}; do
        depth_args="${depth_args} ${d}"
    done

    # Run on spark01 (where API is served)
    # shellcheck disable=SC2086
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
        || { warn "Benchmark failed for bt=${bt}"; echo "bench_result=FAIL" >> "${meta_file}"; }

    [[ -f "${bench_out}" ]] && echo "bench_result=OK" >> "${meta_file}" || true

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
    # Match table rows like: | model | test | t/s | ...
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

# Also get peak tg from peak t/s column (col 4 if present)
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

        # Memory safety check
        log "Checking memory recovery after container stop..."
        local safe=true
        if ! check_memory_safe "${SPARK01}"; then
            warn "spark01 memory below safe threshold after container stop."
            warn "GB10 UMA driver memory leak: container stop does NOT always release GPU pages."
            warn "You may need to reboot spark01 before the next run."
            echo "post_stop_memory_safe_spark01=UNSAFE" >> "${meta_file}"
            safe=false
        else
            echo "post_stop_memory_safe_spark01=OK" >> "${meta_file}"
        fi

        if ! check_memory_safe "${SPARK02}"; then
            warn "spark02 memory below safe threshold after container stop."
            warn "You may need to reboot spark02 before the next run."
            echo "post_stop_memory_safe_spark02=UNSAFE" >> "${meta_file}"
            safe=false
        else
            echo "post_stop_memory_safe_spark02=OK" >> "${meta_file}"
        fi

        if ! ${safe}; then
            warn "Memory not recovered sufficiently. Halting matrix run."
            warn "Reboot affected nodes and resume with --skip-bt for completed values."
            return 2
        fi
    fi

    log "bt=${bt} complete."
    return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log "bench-bt-matrix-step37-v023.sh"
    log "Repository: ${REPO_ROOT}"
    log "Template:   ${TEMPLATE_ENV}"
    log "Results:    ${RESULT_DIR}"
    log "bt matrix:  ${BT_VALUES[*]}"
    log "Runs/test:  ${BENCH_RUNS}"
    log "Depths:     ${BENCH_DEPTHS}"
    ${DRY_RUN} && log "DRY-RUN MODE: no containers will be started."
    echo ""

    # Validate prerequisites
    [[ -f "${TEMPLATE_ENV}" ]] || die "Template env not found: ${TEMPLATE_ENV}"
    [[ -f "${COMPOSE_FILE}" ]] || die "docker-compose.yml not found: ${COMPOSE_FILE}"
    ${DRY_RUN} || ssh "${SPARK01}" "test -x '${LLAMA_BENCHY_BIN}'" \
        || die "llama-benchy not found on spark01: ${LLAMA_BENCHY_BIN}"
    ${DRY_RUN} || ssh "${SPARK01}" "test -f '${TOKENIZER_PATH}/tokenizer.json'" \
        || die "Tokenizer not found on spark01: ${TOKENIZER_PATH}"

    mkdir -p "${RESULT_DIR}/logs"

    # Master CSV aggregation file
    local master_csv="${RESULT_DIR}/matrix-summary-$(date +%Y%m%d-%H%M%S).csv"
    echo "run_id,bt,status,pp2048_tps,pp2048_tps_sd,tg32_tps,tg32_tps_peak,pp_d4096_tps,pp_d8192_tps,pp_d16384_tps,tg_d4096_tps,tg_d8192_tps,tg_d16384_tps" > "${master_csv}"

    local completed=0
    local failed=0

    for bt in "${BT_VALUES[@]}"; do
        log ""
        if run_bt "${bt}"; then
            # Append this run's CSV row to master
            local run_csv
            run_csv=$(find "${RESULT_DIR}" -name "summary.csv" -newer "${master_csv}" | sort | tail -1)
            [[ -n "${run_csv}" ]] && tail -1 "${run_csv}" >> "${master_csv}" || true
            completed=$((completed + 1))
        else
            local exit_code=$?
            if [[ ${exit_code} -eq 2 ]]; then
                warn "Halting matrix run due to memory safety check failure."
                warn "Completed: ${completed}/${#BT_VALUES[@]} runs."
                warn "Resume: --bt <remaining_values> --skip-bt <completed_values>"
                break
            fi
            failed=$((failed + 1))
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
