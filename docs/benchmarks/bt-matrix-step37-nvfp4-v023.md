# MAX_NUM_BATCHED_TOKENS Matrix Benchmark
## Step-3.7-Flash-NVFP4 · vLLM 0.23.0 · Dual DGX Spark GB10

**Purpose**: Validate that `MAX_NUM_BATCHED_TOKENS=256` (set during garble debugging)
is the primary cause of the observed prefill throughput regression in v023, and
determine a safe production value.

**Status**: Benchmark infrastructure ready. Results: Not yet measured.

---

## Background

During vLLM 0.23.0 garble investigation, `MAX_NUM_BATCHED_TOKENS` was reduced from
the v022 default to `256` for conservative testing. After resolving garble via
`--moe-backend marlin` and `--attention-backend TRITON_ATTN`, benchmarking at `bt=256`
produced:

| Metric | v023 bt=256 | v022 baseline | Delta |
|--------|------------|--------------|-------|
| pp2048 | 537.94 t/s | 1251.42 t/s | −57% |
| tg32   | 12.26 t/s  | 13.35 t/s   | −8%  |
| pp2048 @ d4096 | 563.47 t/s | 1299.69 t/s | −57% |
| tg32 @ d4096   | 11.25 t/s  | 12.84 t/s   | −12% |
| pp2048 @ d8192 | 564.00 t/s | 1289.83 t/s | −56% |
| tg32 @ d8192   | 11.47 t/s  | 11.90 t/s   | −4%  |
| pp2048 @ d16384 | 530.91 t/s | 1267.43 t/s | −58% |
| tg32 @ d16384   | 11.52 t/s  | 12.11 t/s   | −5%  |

Decode throughput (tg) is near-identical. Prefill (pp) is ~57% lower across all depths.

---

## Hypothesis

| # | Hypothesis | Confidence |
|---|-----------|------------|
| 1 | 2048-token prompt splits into 8 × 256-token micro-batches → 8× kernel launches | **Confirmed** (math) |
| 2 | Each micro-batch runs MoE routing with ~7 tokens/expert (256×8/288) → MARLIN GEMM row starvation | **Strongly indicated** |
| 3 | Per-micro-batch TP/EP NCCL collective (Socket) overhead accumulates | **Strongly indicated** |
| 4 | TRITON_ATTN prefill path is slower than FlashInfer for long prefill | **Possible contributor** (not isolated) |
| 5 | v023 per-se introduces prefill overhead independent of bt | **Possible contributor** (not isolated) |
| 6 | bt=2048 is the exact single-chunk recovery point for pp=2048 | **Strongly indicated** |

---

## Fixed environment

All matrix runs use identical settings except `MAX_NUM_BATCHED_TOKENS`.

| Parameter | Value |
|-----------|-------|
| Image | `vllm-spark:v023-step3p7-fixed-kv-profile-skip-candidate` |
| vLLM | 0.23.0 |
| Hardware | 2× DGX Spark GB10 (Blackwell SM_121) |
| Memory | 121.63 GiB UMA per node |
| Interconnect | 200 Gbps RoCE (NCCL_NET=Socket) |
| TP | 2 |
| EP | 2 (`--enable-expert-parallel`) |
| Distributed backend | mp (multiprocessing) |
| MoE backend | `marlin` (**required** — v023 CUTLASS auto-selects for NVFP4 and garbles on SM_121) |
| Attention backend | `TRITON_ATTN` (**required** — FlashInfer causes reasoning garble on SM_121) |
| CUDA graph | disabled (`--compilation-config {"mode":0,"cudagraph_mode":"NONE"}`) |
| MTP | disabled |
| MAX_MODEL_LEN | 32768 |
| MAX_NUM_SEQS | 1 |
| GPU_MEMORY_UTILIZATION | 0.79 |
| KV cache | fixed 2 GiB (`--kv-cache-memory-bytes 2147483648`) |
| Benchmark | llama-benchy 0.3.7, latency mode, concurrency=1, runs=3 |
| Benchmark depths | pp2048 @ d0/4096/8192/16384 + tg32 @ same depths |
| d32768 excluded | MAX_MODEL_LEN=32768 and llama-benchy adds 32 output tokens → overflow |

---

## bt matrix

| bt | pp2048 t/s | tg32 t/s | pp@d4096 | pp@d8192 | pp@d16384 | Notes |
|---:|----------:|--------:|---------:|---------:|----------:|-------|
| 256 | 537.94 | 12.26 | 563.47 | 564.00 | 530.91 | **Confirmed** (measured 2026-06-18) |
| 512 | Not yet measured | — | — | — | — | |
| 1024 | Not yet measured | — | — | — | — | |
| 2048 | Not yet measured | — | — | — | — | **Key: first single-chunk point for pp2048** |
| 4096 | Not yet measured | — | — | — | — | |
| 8192 | Not yet measured | — | — | — | — | **v022 production default** |
| 16384 | Not yet measured | — | — | — | — | |
| 32768 | Not yet measured | — | — | — | — | Matches MAX_MODEL_LEN |

---

## How to run

```bash
# From homeserver in /home/bjk110/docker/vllm-spark/
# Dry run (check commands without executing):
bash benchmarks/bench-bt-matrix-step37-v023.sh --dry-run

# Run full matrix:
bash benchmarks/bench-bt-matrix-step37-v023.sh

# Run subset only:
bash benchmarks/bench-bt-matrix-step37-v023.sh --bt 2048,4096,8192

# Skip already-completed values:
bash benchmarks/bench-bt-matrix-step37-v023.sh --skip-bt 256

# After completion, generate analysis report:
bash benchmarks/analyze-bt-matrix.sh benchmarks/results/bt-matrix
```

**Prerequisites**:
- Both spark01 and spark02 must be up and have free memory above 50 GiB.
- The current running container must be stopped before running the matrix.
- SSH aliases `spark01` and `spark02` must be configured.

---

## Correctness validation

Each bt run includes the following correctness tests:

1. **Factual** — "largest prime < 100" → expects 97
2. **Multi-step arithmetic** — "15 factorial" → expects 1307674368000
3. **Unicode integrity** — Korean KTX question → checks for broken codepoints
4. **Finish reason** — "2+2" → expects `stop` finish_reason

A run is flagged `backend_validity=INVALID` if the MARLIN MoE backend confirmation
log line (`Using 'MARLIN' NvFp4 MoE backend`) is not found. Such runs must be discarded.

---

## GB10 UMA memory caveat

On GB10 (Blackwell, UMA), stopping a vLLM container does **not** release GPU pages
from the NVIDIA driver. The runner checks free memory after each container stop.
If memory does not recover above 50 GiB, the run halts with a warning.

**Confirmed**: `rmmod nvidia_uvm` does not help. Only reboot recovers memory.

If halted mid-matrix:
1. Note which bt values completed (listed in master CSV).
2. Reboot affected nodes.
3. Resume with `--skip-bt <completed_values>`.

---

## Production recommendation

*Pending measurement. Based on v022 history and hypothesis:*

| Candidate | bt | Rationale | Risk |
|-----------|---:|-----------|------|
| Conservative | 4096 | 2× pp2048; safe headroom for other prompt sizes | Moderate |
| **Recommended starting point** | **8192** | v022 production default; well-tested | Low |
| Higher-perf candidate | 16384 | Better MoE utilization per chunk | Moderate — not tested on v023 |
| Maximum single-chunk | 32768 | Matches MAX_MODEL_LEN; eliminates chunking for all prompts | Unknown |

**Do not apply to `presets/step37-flash-nvfp4-tp2.env`** until full v023 production
validation is complete. Apply only to the disposable env under `.local/`.

Diff to apply (disposable env only):
```diff
-MAX_NUM_BATCHED_TOKENS=256
+MAX_NUM_BATCHED_TOKENS=8192
```

Revert:
```diff
-MAX_NUM_BATCHED_TOKENS=8192
+MAX_NUM_BATCHED_TOKENS=256
```

---

## Phase 8: MTP follow-up

After bt tuning is validated, run MTP sweep with the selected bt:

- MTP off / n=1 / n=2 / n=3 × tg128
- Metrics: output t/s, acceptance rate, mean acceptance length, correctness
- Note: reasoning `<think>` traces have low acceptance → expected low MTP gain

---

## Related files

| File | Description |
|------|-------------|
| `.local/env/step37/bt-matrix-base.env` | Template env (bt placeholder) |
| `benchmarks/bench-bt-matrix-step37-v023.sh` | Matrix runner |
| `benchmarks/analyze-bt-matrix.sh` | Analysis and report generator |
| `benchmarks/results/bt-matrix/` | Per-run results (gitignored via `.cache/`) |
| `presets/step37-flash-nvfp4-tp2.env` | Production preset — **do not modify** |
| `.local/env/step37/v023-nomtp-fixed-kv-profile-skip.env` | Current disposable env (bt=256) |
