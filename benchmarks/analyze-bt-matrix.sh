#!/usr/bin/env bash
# =============================================================================
# analyze-bt-matrix.sh
#
# Reads bt-matrix benchmark results and generates a Markdown analysis report.
# Run after bench-bt-matrix-step37-v023.sh completes.
#
# Usage:
#   bash benchmarks/analyze-bt-matrix.sh <result-dir>
#   bash benchmarks/analyze-bt-matrix.sh benchmarks/results/bt-matrix
# =============================================================================

set -euo pipefail

RESULT_DIR="${1:-benchmarks/results/bt-matrix}"
REPORT_FILE="${RESULT_DIR}/analysis-$(date +%Y%m%d-%H%M%S).md"

[[ -d "${RESULT_DIR}" ]] || { echo "Result dir not found: ${RESULT_DIR}" >&2; exit 1; }

# Find the latest master CSV
MASTER_CSV=$(find "${RESULT_DIR}" -maxdepth 1 -name "matrix-summary-*.csv" | sort | tail -1)
[[ -n "${MASTER_CSV}" ]] || { echo "No matrix-summary CSV found in ${RESULT_DIR}" >&2; exit 1; }

echo "Analyzing: ${MASTER_CSV}"
echo "Report:    ${REPORT_FILE}"

python3 - "${MASTER_CSV}" "${REPORT_FILE}" "${RESULT_DIR}" <<'PYEOF'
import sys
import csv
import os
from pathlib import Path

csv_file = sys.argv[1]
report_file = sys.argv[2]
result_dir = sys.argv[3]

# Reference baseline from v022 (NVFP4, no garble fix needed, bt default)
V022_BASELINE = {
    'pp2048':           1251.42,
    'pp2048_d4096':     1299.69,
    'pp2048_d8192':     1289.83,
    'pp2048_d16384':    1267.43,
    'tg32':             13.35,
    'tg32_d4096':       12.84,
    'tg32_d8192':       11.90,
    'tg32_d16384':      12.11,
}

# Load CSV
rows = []
with open(csv_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def pct_change(new, ref):
    if ref and new:
        return (new - ref) / ref * 100
    return None

lines = []
lines.append("# MAX_NUM_BATCHED_TOKENS Benchmark Analysis")
lines.append("## Step-3.7-Flash-NVFP4 · vLLM 0.23.0 · Dual DGX Spark GB10")
lines.append("")
lines.append(f"**Source CSV**: `{os.path.basename(csv_file)}`  ")
lines.append(f"**Result directory**: `{result_dir}`  ")
lines.append("")

lines.append("## Fixed parameters (identical across all runs)")
lines.append("")
lines.append("| Parameter | Value |")
lines.append("|-----------|-------|")
lines.append("| Model | stepfun-ai/Step-3.7-Flash-NVFP4 |")
lines.append("| Image | v023-step3p7-fixed-kv-profile-skip-candidate |")
lines.append("| vLLM | 0.23.0 |")
lines.append("| Hardware | 2× DGX Spark GB10 (SM_121) |")
lines.append("| TP | 2 |")
lines.append("| EP | 2 (--enable-expert-parallel) |")
lines.append("| Backend | mp (multiprocessing) |")
lines.append("| MoE backend | marlin (garble fix — required) |")
lines.append("| Attention backend | TRITON_ATTN (garble fix — required) |")
lines.append("| CUDA graph | disabled (mode=0, cudagraph_mode=NONE) |")
lines.append("| MTP | disabled |")
lines.append("| MAX_MODEL_LEN | 32768 |")
lines.append("| MAX_NUM_SEQS | 1 |")
lines.append("| GPU_MEMORY_UTILIZATION | 0.79 |")
lines.append("| KV cache | fixed 2 GiB (--kv-cache-memory-bytes 2147483648) |")
lines.append("| NCCL transport | Socket (RDMA plugin incompatible with CUDA 13.3) |")
lines.append("| Benchmark tool | llama-benchy 0.3.7, latency mode, concurrency=1, runs=3 |")
lines.append("| Variable | MAX_NUM_BATCHED_TOKENS |")
lines.append("")

lines.append("## Raw results")
lines.append("")

if not rows:
    lines.append("*No results found. Run bench-bt-matrix-step37-v023.sh first.*")
else:
    # Header
    lines.append("| bt | status | pp2048 t/s | ±sd | tg32 t/s | peak tg | pp@d4096 | pp@d8192 | pp@d16384 | tg@d4096 | tg@d8192 | tg@d16384 |")
    lines.append("|---:|--------|----------:|----:|--------:|--------:|---------:|---------:|----------:|--------:|--------:|-----------:|")

    for row in rows:
        bt = row.get('bt', 'N/A')
        status = row.get('status', 'N/A')
        pp = safe_float(row.get('pp2048_tps'))
        pp_sd = safe_float(row.get('pp2048_tps_sd'))
        tg = safe_float(row.get('tg32_tps'))
        tg_peak = safe_float(row.get('tg32_tps_peak'))
        pp_d4 = safe_float(row.get('pp_d4096_tps'))
        pp_d8 = safe_float(row.get('pp_d8192_tps'))
        pp_d16 = safe_float(row.get('pp_d16384_tps'))
        tg_d4 = safe_float(row.get('tg_d4096_tps'))
        tg_d8 = safe_float(row.get('tg_d8192_tps'))
        tg_d16 = safe_float(row.get('tg_d16384_tps'))

        def fmt(v): return f"{v:.1f}" if v is not None else "N/A"
        lines.append(f"| {bt} | {status} | {fmt(pp)} | {fmt(pp_sd)} | {fmt(tg)} | {fmt(tg_peak)} | {fmt(pp_d4)} | {fmt(pp_d8)} | {fmt(pp_d16)} | {fmt(tg_d4)} | {fmt(tg_d8)} | {fmt(tg_d16)} |")

    lines.append("")

    # v022 baseline row
    lines.append("**v022 baseline** (confirmed, different image, bt=default, no TRITON/MARLIN requirement):")
    lines.append("")
    lines.append("| bt | pp2048 | tg32 | pp@d4096 | pp@d8192 | pp@d16384 | tg@d4096 | tg@d8192 | tg@d16384 |")
    lines.append("|---:|-------:|-----:|---------:|---------:|----------:|---------:|---------:|----------:|")
    v = V022_BASELINE
    lines.append(f"| ~8192 (default) | {v['pp2048']:.1f} | {v['tg32']:.2f} | {v['pp2048_d4096']:.1f} | {v['pp2048_d8192']:.1f} | {v['pp2048_d16384']:.1f} | {v['tg32_d4096']:.2f} | {v['tg32_d8192']:.2f} | {v['tg32_d16384']:.2f} |")
    lines.append("")

    # Delta table vs v022
    lines.append("## Prefill throughput vs v022 baseline (pp2048, depth=0)")
    lines.append("")
    lines.append("| bt | pp2048 t/s | Δ vs v022 | Δ% |")
    lines.append("|---:|-----------:|----------:|---:|")
    for row in rows:
        if row.get('status') != 'OK':
            continue
        bt = row.get('bt', 'N/A')
        pp = safe_float(row.get('pp2048_tps'))
        delta = (pp - V022_BASELINE['pp2048']) if pp else None
        pct = pct_change(pp, V022_BASELINE['pp2048'])
        fmt_delta = f"{delta:+.1f}" if delta is not None else "N/A"
        fmt_pct = f"{pct:+.1f}%" if pct is not None else "N/A"
        pp_str = f"{pp:.1f}" if pp is not None else "N/A"
        lines.append(f"| {bt} | {pp_str} | {fmt_delta} | {fmt_pct} |")
    lines.append("")

lines.append("## Analysis (Phase 5)")
lines.append("")
lines.append("### Hypothesis under test")
lines.append("")
lines.append("The observed prefill regression (bt=256: ~538 t/s vs v022 baseline ~1251 t/s)")
lines.append("is hypothesized to result from chunked prefill fragmentation:")
lines.append("")
lines.append("1. A 2048-token prompt at bt=256 requires 8 sequential micro-batches.")
lines.append("2. Step-3.7 has 288 MoE experts (top-8 routing). With 256-token chunks,")
lines.append("   average tokens per expert per chunk is only 256×8/288 ≈ 7 rows.")
lines.append("   MARLIN grouped GEMM efficiency degrades sharply at <16-32 rows per group.")
lines.append("3. routing/permutation, kernel launch, TP/EP NCCL collective (Socket transport)")
lines.append("   repeats 8× instead of 1×, accumulating overhead per micro-batch.")
lines.append("")

if rows and any(r.get('status') == 'OK' for r in rows):
    ok_rows = [r for r in rows if r.get('status') == 'OK']

    lines.append("### Key breakpoints (from measured data)")
    lines.append("")

    # Find bt where pp first exceeds 1000 t/s
    for r in sorted(ok_rows, key=lambda x: int(x.get('bt', 0))):
        pp = safe_float(r.get('pp2048_tps'))
        if pp and pp > 1000:
            lines.append(f"- **bt={r['bt']}**: first bt value where pp2048 exceeds 1000 t/s ({pp:.1f} t/s)")
            break
    else:
        lines.append("- No bt value reached 1000 t/s in this run (check if all matrix entries completed)")

    # Check if 2048 single-chunk changes behavior
    row_2048 = next((r for r in ok_rows if r.get('bt') == '2048'), None)
    if row_2048:
        pp_2048 = safe_float(row_2048.get('pp2048_tps'))
        if pp_2048:
            pct = pct_change(pp_2048, V022_BASELINE['pp2048'])
            lines.append(f"- **bt=2048** (pp2048 fits single chunk): {pp_2048:.1f} t/s ({pct:+.1f}% vs v022)")
    else:
        lines.append("- bt=2048 not yet measured (Not yet measured)")

    lines.append("")

else:
    lines.append("")
    lines.append("*Results not yet available. Run bench-bt-matrix-step37-v023.sh to populate.*")
    lines.append("")

lines.append("### Findings confidence levels")
lines.append("")
lines.append("| Finding | Confidence |")
lines.append("|---------|------------|")
lines.append("| bt=256 causes prefill fragmentation | **Confirmed** (tested, 538 t/s measured) |")
lines.append("| Higher bt values improve prefill | **Strongly indicated** (v022 baseline at ~8192 default showed 1251 t/s) |")
lines.append("| TRITON_ATTN adds prefill overhead vs FlashInfer | **Possible contributor** (not isolated) |")
lines.append("| v023 per-se adds prefill overhead | **Possible contributor** (not isolated) |")
lines.append("| bt=2048 is the single-chunk recovery point for pp2048 | **Strongly indicated** (hypothesis) |")
lines.append("| decode throughput unaffected by bt | **Strongly indicated** (v022/v023 tg32 similar) |")
lines.append("| bt≥16384 is stable on this GB10 + UMA config | **Not tested** |")
lines.append("")

lines.append("## Phase 7: Production recommendation")
lines.append("")
lines.append("*Based on results above. Pending actual measurement if not yet run.*")
lines.append("")
lines.append("### Recommended value: 8192 (starting point)")
lines.append("")
lines.append("| Candidate | bt | Rationale | Risk |")
lines.append("|-----------|---:|-----------|------|")
lines.append("| Conservative | 4096 | 2× pp2048 chunk size; safe headroom | Moderate — may not fully recover prefill |")
lines.append("| **Recommended** | **8192** | Previous v022 default; confirmed safe on v022 | Low — well-tested value |")
lines.append("| Higher-perf candidate | 16384 | Better MoE utilization; single chunk for most prompts | Moderate — not tested on v023 |")
lines.append("| Max single-chunk | 32768 | Matches MAX_MODEL_LEN; single chunk for all prompts | Unknown — memory impact not measured |")
lines.append("")
lines.append("> **Note**: This recommendation is conditional on actual bt-matrix results.")
lines.append("> If the matrix shows bt=8192 does not recover prefill, revise upward.")
lines.append("> If bt=4096 already recovers to >1100 t/s, that is the safer choice.")
lines.append("")

lines.append("### Production diff")
lines.append("")
lines.append("Apply to `.local/env/step37/v023-nomtp-fixed-kv-profile-skip.env` (disposable env)")
lines.append("after validation. **Do not apply to `presets/step37-flash-nvfp4-tp2.env`** until")
lines.append("full v023 production validation is complete.")
lines.append("")
lines.append("```diff")
lines.append("-MAX_NUM_BATCHED_TOKENS=256")
lines.append("+MAX_NUM_BATCHED_TOKENS=8192")
lines.append("```")
lines.append("")
lines.append("Revert:")
lines.append("```diff")
lines.append("-MAX_NUM_BATCHED_TOKENS=8192")
lines.append("+MAX_NUM_BATCHED_TOKENS=256")
lines.append("```")
lines.append("")

lines.append("## Correctness validation summary")
lines.append("")
lines.append("Correctness results for each bt run are in `<run-id>/correctness.md`.")
lines.append("Garble-fix backend confirmation (MARLIN + TRITON_ATTN) is logged in `<run-id>/metadata.txt`.")
lines.append("")
lines.append("If any run shows `backend_validity=INVALID_MARLIN_NOT_CONFIRMED`, that run's")
lines.append("results are invalid and must be discarded.")
lines.append("")

lines.append("## GB10 UMA memory caveat")
lines.append("")
lines.append("Stopping a vLLM container on GB10 does NOT always release GPU pages from the")
lines.append("NVIDIA driver. If `post_stop_memory_safe_sparkXX=UNSAFE` appears in metadata,")
lines.append("the node must be rebooted before the next matrix run. The runner halts")
lines.append("automatically when this condition is detected.")
lines.append("")
lines.append("**Confirmed**: `rmmod nvidia_uvm` does not free the pages. Only reboot works.")
lines.append("")

lines.append("## Phase 8: MTP follow-up preparation")
lines.append("")
lines.append("After prefill bt tuning is validated, run MTP sweep with the selected bt value:")
lines.append("")
lines.append("Matrix: MTP off / n=1 / n=2 / n=3 × tg128")
lines.append("Script: `benchmarks/bench-mtp-matrix-step37-v023.sh` (not yet created)")
lines.append("")
lines.append("Metrics to capture:")
lines.append("- output tokens/s")
lines.append("- mean acceptance length")
lines.append("- draft acceptance rate")
lines.append("- per-position acceptance rate")
lines.append("- correctness result")
lines.append("")
lines.append("Note: Step-3.7-Flash reasoning model is expected to have low MTP acceptance")
lines.append("for `<think>` token sequences (highly model-state-dependent, unpredictable).")
lines.append("")

with open(report_file, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"Report written to {report_file}")
PYEOF

echo "Analysis complete: ${REPORT_FILE}"
