# MAX_NUM_BATCHED_TOKENS Matrix Benchmark
## Step-3.7-Flash-NVFP4 · vLLM 0.23.0 · Dual DGX Spark GB10

**Purpose**: Validate that `MAX_NUM_BATCHED_TOKENS=256` (set during garble debugging)
is the primary cause of the observed prefill throughput regression in v023, and
determine a safe production value.

**Status**: bt=256 and bt=2048 measured (Series A, EP-off). bt=8192 pending.

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
| 1 | 2048-token prompt is chunked into up to 8 × 256-token scheduler iterations | **Expected from budget math; scheduler iteration count not directly instrumented** |
| 2 | Each chunk runs MoE routing with ~7.1 routed token assignments per expert on average (256×8/288) → MARLIN grouped GEMM below efficient row threshold | **Plausible mechanism; not directly profiled** |
| 3 | Per-chunk TP all-reduce and collective overhead accumulates (EP was **disabled** in the bt=256 run — TP-only synchronization, no expert distribution across ranks) | **Plausible contributor; not isolated** |
| 4 | TRITON_ATTN prefill path is slower than FlashInfer for long prefill | **Possible contributor; not isolated** |
| 5 | v023 per-se introduces prefill overhead independent of bt | **Possible contributor; not isolated** |
| 6 | bt=2048 is the exact single-chunk boundary for pp=2048 | **Expected from budget math; not yet measured on v023** |

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
| EP | **See series note below** |
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

> **EP series note**: The bt=256 measured run (`v023-triton-marlin-ep-off-bt256`) used
> `.local/env/step37/v023-nomtp-fixed-kv-profile-skip.env`, which has **no
> `--enable-expert-parallel`** (EP disabled, TP=2 only). This was confirmed from both
> the env file and live `docker inspect vllm-spark-head` output.
>
> Future matrix runs will use `bt-matrix-base.env`, which **includes
> `--enable-expert-parallel`** (EP enabled, 144 experts per rank). This is a different
> topology. Raw throughput numbers cannot be compared across these two series as a
> pure bt-variable change. To isolate bt as the single variable, future runs must also
> use an EP-off config, or a separate EP-on series must be tracked independently.

---

## bt matrix

**Series A — EP disabled** (bt=256 measured run, `v023-triton-marlin-ep-off-bt256`):

EP evidence classification: **A — Historical runtime evidence.**
Container 566574c5 was created at 2026-06-17T22:19:04Z, started at 22:39:38Z, and had
RestartCount=0 through the bt=256 benchmark at 2026-06-18T16:17:24Z — confirming the
inspected container is the same instance that served the benchmark. The entrypoint
command in the startup log shows no `--enable-expert-parallel` in the `vllm serve`
invocation (direct runtime observation, not configuration inference).
Note: vLLM 0.23 does not log `expert_parallel_size` when EP is at its default of 1,
so absence of the flag in the entrypoint command is the primary evidence source.

| bt | EP | pp2048 t/s | tg32 t/s | pp@d4096 | pp@d8192 | pp@d16384 | Notes |
|---:|:--:|----------:|--------:|---------:|---------:|----------:|-------|
| 256 | off | 537.94 | 12.26 | 563.47 | 564.00 | 530.91 | Measured 2026-06-18; config_label=v023-triton-marlin-ep-off-bt256 |
| 2048 | off | **1034.86** | 10.06 | **1088.21** | **1076.12** | **1050.69** | Measured 2026-06-18; config_label=v023-triton-marlin-ep-off-bt2048; run_id=bt2048-20260618-093121; pp +92% vs bt=256 |
| 8192 | off | Not yet measured | — | — | — | — | Provisional candidate; not yet measured on v023 |

**Series B — EP enabled** (future matrix runs, `bt-matrix-base.env`):

| bt | EP | pp2048 t/s | tg32 t/s | pp@d4096 | pp@d8192 | pp@d16384 | Notes |
|---:|:--:|----------:|--------:|---------:|---------:|----------:|-------|
| 256 | on | Not yet measured | — | — | — | — | For EP-on baseline comparison |
| 512 | on | Not yet measured | — | — | — | — | |
| 1024 | on | Not yet measured | — | — | — | — | |
| 2048 | on | Not yet measured | — | — | — | — | Expected single-chunk boundary for pp=2048 (not yet measured on v023) |
| 4096 | on | Not yet measured | — | — | — | — | |
| 8192 | on | Not yet measured | — | — | — | — | v022 production default bt (different EP state; direct comparison requires caution) |
| 16384 | on | Not yet measured | — | — | — | — | |
| 32768 | on | Not yet measured | — | — | — | — | Matches MAX_MODEL_LEN |

> Cross-series comparison (Series A bt=256 vs Series B) is **not** a pure bt variable
> test. To isolate bt as the single variable across all values, run all entries with the
> same EP setting.

---

## How to run

### Phase 0: Pre-run preflight (before reboot)

Capture boot IDs and memory state while the current server is still running:

```bash
# From homeserver in /home/bjk110/docker/vllm-spark/
bash benchmarks/bench-bt-matrix-step37-v023.sh \
  --preflight-only \
  --template .local/env/step37/bt-matrix-series-a-ep-off.env \
  --expected-ep off
```

Save the output. After rebooting, run preflight again and verify:
- `spark0X_boot_id` changed on both nodes
- `spark0X_uptime_seconds` is low (e.g. < 600) on both nodes
- `spark0X_mem_available_gib` > 50 GiB on both nodes (no server loaded)

### Phase 1: Validate current container (optional, read-only)

If the server is still up, confirm the running container already uses MARLIN + TRITON_ATTN:

```bash
bash benchmarks/bench-bt-matrix-step37-v023.sh \
  --validate-existing-container \
  --expected-ep off
```

Reads docker logs from the live head container. No container ops. Prints detection
results to stdout and saves to `benchmarks/results/bt-matrix/.validate-*/`.

### Phase 2: Dry-run (verify config before live run)

After both nodes have been rebooted and memory is confirmed clean:

```bash
bash benchmarks/bench-bt-matrix-step37-v023.sh \
  --bt 2048 \
  --template .local/env/step37/bt-matrix-series-a-ep-off.env \
  --expected-ep off \
  --config-label v023-triton-marlin-ep-off-bt2048 \
  --dry-run
```

### Phase 3: Series A — EP-off, bt=2048 (required first)

bt=2048 is the required next measurement. Run this **before** bt=8192.

```bash
bash benchmarks/bench-bt-matrix-step37-v023.sh \
  --bt 2048 \
  --template .local/env/step37/bt-matrix-series-a-ep-off.env \
  --expected-ep off \
  --config-label v023-triton-marlin-ep-off-bt2048
```

### Phase 4: Series A — EP-off, bt=8192 (provisional candidate, after bt=2048)

Run only after bt=2048 completes successfully. bt=8192 is the provisional candidate
but is NOT validated until measured on the v023 correctness-safe stack.

```bash
bash benchmarks/bench-bt-matrix-step37-v023.sh \
  --bt 8192 \
  --template .local/env/step37/bt-matrix-series-a-ep-off.env \
  --expected-ep off \
  --config-label v023-triton-marlin-ep-off-bt8192
```

### Series B — EP-on (separate series, separate template)

```bash
bash benchmarks/bench-bt-matrix-step37-v023.sh \
  --bt 2048 \
  --expected-ep on \
  --config-label v023-triton-marlin-ep-on-bt2048
```

### Full matrix (after individual runs validate)

```bash
bash benchmarks/bench-bt-matrix-step37-v023.sh \
  --all \
  --template .local/env/step37/bt-matrix-series-a-ep-off.env \
  --expected-ep off

# Generate analysis report after completion:
bash benchmarks/analyze-bt-matrix.sh benchmarks/results/bt-matrix
```

**Prerequisites**:
- Both spark01 and spark02 must be rebooted before each run (GB10 UMA driver retains pages after container stop — only reboot recovers them).
- After reboot, both nodes must have > 50 GiB free memory (verified by preflight).
- SSH aliases `spark01` and `spark02` must be configured.
- `--template` selects the topology series (Series A = EP-off, Series B = default EP-on).
- `--expected-ep off|on` gates the run on observed EP state from startup logs.

---

## Acceptance criteria for bt=2048

A bt=2048 result is valid for production consideration **only if all of the following hold**:

| Criterion | Metadata field | Required value |
|-----------|---------------|----------------|
| Both nodes rebooted before run | `head_boot_id` / `worker_boot_id` changed vs prior run metadata | Different from bt=256 run |
| Same image | `image` | `vllm-spark:v023-step3p7-fixed-kv-profile-skip-candidate` |
| EP confirmed off | `expert_parallel_observed` | `disabled` |
| MARLIN confirmed | `marlin_confirmed` | `1` |
| TRITON_ATTN confirmed | `triton_attn_confirmed` | `1` |
| Backend validity | `backend_validity` | `OK (MARLIN confirmed, TRITON_ATTN confirmed)` |
| EP validation | `ep_validation` | `OK (expected=off, observed=disabled)` |
| Bench result | `bench_result` | `OK` |
| Startup | `startup_result` | `OK` |
| Correctness | All checks in `correctness.md` | No garble, no systematic failures |

bt=8192 requires the same criteria. Do not run bt=8192 before bt=2048 completes.

---

## Pre-run validation state (2026-06-18)

### Re-validation of bt=256 historical container

A second `--validate-existing-container` run at `2026-06-18T08:45:44Z` against the same
container (ID `566574c5`, RestartCount=0, started `2026-06-17T22:39:38Z`) confirms all
three classification criteria remain consistent:

| Item | Status | Evidence |
|------|--------|---------|
| MARLIN backend | confirmed | `Worker_TP0 pid=304` (2026-06-17T22:40:26) |
| TRITON_ATTN backend | confirmed | `Worker_TP0 pid=304` (2026-06-17T22:40:26) |
| EP state | disabled | `--enable-expert-parallel` absent from entrypoint command line |
| `backend_valid` | OK | both gates pass |
| `ep_validation` | OK (expected=off, observed=disabled) | entrypoint cmd |

This is a third independent read of the same container instance. Classification A evidence
is unchanged. The result does not add new information about bt=256 throughput numbers.

### Preflight baseline (pre-reboot reference values)

Captured `2026-06-18T08:45:51Z` while the server is running. Use these boot IDs to verify
reboot occurred before the bt=2048 run.

| Field | spark01 | spark02 |
|-------|---------|---------|
| `boot_id` | `31eedf8e-c268-49e8-a029-5c1af037e09a` | `6ccea774-a076-407f-b791-da65cca9fe23` |
| `uptime_seconds` | ~146608 (~40.7 h) | ~146606 (~40.7 h) |
| `mem_available_gib` | 47.9 (server loaded) | 49.5 (server loaded) |
| container state | `vllm-spark-head` Up ~10 h | `vllm-spark-worker` Up ~10 h |

After reboot, both `boot_id` values must differ from the above before bt=2048 is executed.

### bt=2048 execution results (2026-06-18)

**Completed.** run_id=`bt2048-20260618-093121`, executed 2026-06-18T09:31Z.

#### Acceptance criteria check

| Criterion | Status | Evidence |
|-----------|--------|---------|
| Both nodes rebooted | PASS | spark01 `31eedf8e→9f01d96f`, spark02 `6ccea774→92bc7698` |
| Same image | PASS | `vllm-spark:v023-step3p7-fixed-kv-profile-skip-candidate` |
| EP confirmed off | PASS | `expert_parallel_observed=disabled` |
| MARLIN confirmed | PASS | `marlin_confirmed=1` (`nvfp4.py:231`) |
| TRITON_ATTN confirmed | PASS | `triton_attn_confirmed=1` (`cuda.py:331`) |
| Backend validity | PASS | `OK (MARLIN confirmed, TRITON_ATTN confirmed)` |
| EP validation | PASS | `OK (expected=off, observed=disabled)` |
| Bench result | PASS | `bench_result=OK` |
| Startup | PASS | `startup_result=OK` |
| Correctness | CONDITIONAL | Tests 1,2 correct (97, 1307674368000). Tests 3,4 EMPTY/UNCERTAIN due to max_tokens too small for reasoning model `<think>` chain — not a garble indicator (see vllm023_step37_garble_fix.md) |

#### Throughput results

| Depth | pp2048 t/s | tg32 t/s | pp TTFR (ms) | Notes |
|-------|-----------|---------|------------|-------|
| d0    | 1034.86 ± 49.61 | 10.06 ± 0.04 | 1985.92 | bt=256: 537.94 → +92.4% pp |
| d4096 | 1088.21 ± 7.28  | 10.06 ± 0.23 | 5648.48 | bt=256: 563.47 → +93.1% pp |
| d8192 | 1076.12 ± 0.89  | 9.71 ± 0.22  | 9517.92 | bt=256: 564.00 → +90.7% pp |
| d16384 | 1050.69 ± 2.16 | 9.70 ± 0.11  | 17545.09 | bt=256: 530.91 → +97.9% pp |

**Key observations:**
- Prefill (pp2048) improves by **~92-98%** vs bt=256. The 2048-token prompt is now processed
  in a single scheduler iteration instead of 8 × 256-token chunks.
- Decode (tg32) regresses by ~15-18% vs bt=256 (10.06 vs 12.26 t/s at d0). Mechanism not
  yet isolated (memory allocation difference, compilation effect, or sampling path change).
- pp peak is ~1088 t/s (d4096), still below v022 baseline of 1251 t/s at d4096.
  Remaining gap may be due to TRITON_ATTN prefill vs FlashInfer, MARLIN overhead, or
  v023 per-se changes — not isolated at this time.

#### Infrastructure note: prometheus patch added

An `apply_prometheus_patch()` step was added to `bench-bt-matrix-step37-v023.sh` as part
of this run. `prometheus_fastapi_instrumentator` routing.py raises `AttributeError: '_IncludedRouter'
object has no attribute 'path'` on every HTTP request including `/health`, blocking the
readiness check. The patch is applied automatically after containers start (before model
load completes) and both containers are restarted simultaneously to avoid TCPStore desync.

---

## Correctness validation

Each bt run includes the following correctness tests:

1. **Factual** — "largest prime < 100" → expects 97
2. **Multi-step arithmetic** — "15 factorial" → expects 1307674368000
3. **Unicode integrity** — Korean KTX question → checks for broken codepoints
4. **Finish reason** — "2+2" → expects `stop` finish_reason

A run is flagged `backend_validity=INVALID_MARLIN_NOT_CONFIRMED` if the MARLIN MoE
backend log line (`Using 'MARLIN' NvFp4 MoE backend out of potential backends: [...]`)
is not found. A run is flagged `INVALID_TRITON_NOT_CONFIRMED` if the TRITON_ATTN line
(`Using AttentionBackendEnum.TRITON_ATTN backend.`) is not found. Both are hard gates;
the benchmark does not execute and the containers are stopped. Such runs must be discarded.

**EP validation**: The runner detects EP state from the `[entrypoint] Running: vllm serve`
command line. If `--enable-expert-parallel` is absent, EP is classified as disabled.
vLLM 0.23 does not log `expert_parallel_size=1` when EP is at its default, so the entrypoint
command line is the authoritative source. Use `--expected-ep off|on` to gate on EP state.

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

> **Status: Not validated.** No production `MAX_NUM_BATCHED_TOKENS` value has been
> confirmed for the v023 correctness-safe configuration (MARLIN + TRITON_ATTN).
> The table below lists provisional candidates for measurement, not validated
> recommendations.

*Pending bt-matrix measurement results. All entries are unvalidated on v023.*

| Candidate | bt | Rationale | Risk |
|-----------|---:|-----------|------|
| Conservative candidate | 4096 | 2× pp2048; safe headroom for other prompt sizes | Moderate — may not fully recover prefill |
| **Provisional starting candidate** | **8192** | v022 production default (different EP state — direct comparison requires caution); not yet measured on v023 | Unknown — memory and throughput impact on v023 unmeasured |
| Higher-perf candidate | 16384 | Better MoE utilization per chunk; single chunk for most prompts | Moderate — not tested on v023 |
| Maximum single-chunk | 32768 | Matches MAX_MODEL_LEN; eliminates chunking for all prompts | Unknown — memory impact not measured |

**Do not apply to `presets/step37-flash-nvfp4-tp2.env`** until full v023 production
validation is complete. Apply only to the disposable env under `.local/`.

Provisional candidate test diff (disposable env only — requires correctness validation
before any production consideration):
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
