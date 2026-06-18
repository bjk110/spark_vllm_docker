# MAX_NUM_BATCHED_TOKENS Matrix Benchmark
## Step-3.7-Flash-NVFP4 · vLLM 0.23.0 · Dual DGX Spark GB10

**Purpose**: Validate that `MAX_NUM_BATCHED_TOKENS=256` (set during garble debugging)
is the primary cause of the observed prefill throughput regression in v023, and
determine a safe production value.

**Status**: bt=256 and bt=2048 measured (Series A, EP-off). bt=8192 not executed. Supplement correctness + decode-only run completed 2026-06-18 (`bt2048-supp-20260618-103333`): all 4 correctness tests PASS; decode regression at d0 not reproduced under pp=1 measurement methodology.

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
| 1 | 2048-token prompt spans up to 8 × 256-token scheduled batches at bt=256 | **Consistent with +92-98% prefill improvement at bt=2048; scheduling batch count not directly instrumented** |
| 2 | Each batch runs MoE routing with ~7.1 routed token assignments per expert on average (256×8/288) → MARLIN grouped GEMM below efficient row threshold | **Plausible mechanism; not directly profiled** |
| 3 | Per-batch TP all-reduce and collective overhead accumulates (EP was **disabled** in the bt=256 run — TP-only synchronization, no expert distribution across ranks) | **Plausible contributor; not isolated** |
| 4 | TRITON_ATTN prefill path is slower than FlashInfer for long prefill | **Possible contributor; not isolated; separating this from the bt effect requires a run with bt=2048 + FlashInfer, which conflicts with the SM_121 reasoning garble constraint** |
| 5 | v023 per-se introduces prefill overhead independent of bt | **Possible contributor; not isolated; confounded with attention backend change and EP state** |
| 6 | bt=2048 is the exact single-chunk boundary for pp=2048 | **Consistent with measured data (bt=2048, +92-98% prefill); not directly instrumented — scheduling batch count is inferred from budget math, not logged** |

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
| 2048 | off | **1034.86** | 10.06 | **1088.21** | **1076.12** | **1050.69** | Measured 2026-06-18; run_id=bt2048-20260618-093121; pp +92% vs bt=256. Supplement run (`bt2048-supp-20260618-103333`): all correctness tests PASS; decode-only (pp=1) 12.97 t/s at d0 (higher than original — see Phase 8 analysis). |
| 8192 | off | Not executed | — | — | — | — | Not executed in this session; pending separate run |

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
| Correctness | PASS (supplement) | Tests 1,2 PASS in original run. Tests 3,4 INCONCLUSIVE_OUTPUT_BUDGET in original run (max_tokens too small). Supplement run `bt2048-supp-20260618-103333` with max_tokens=2048: all 4 tests PASS×2. Confirmed budget exhaustion, not garble. |

#### Throughput results

| Depth | pp2048 t/s | tg32 t/s | pp TTFR (ms) | Notes |
|-------|-----------|---------|------------|-------|
| d0    | 1034.86 ± 49.61 | 10.06 ± 0.04 | 1985.92 | bt=256: 537.94 → +92.4% pp |
| d4096 | 1088.21 ± 7.28  | 10.06 ± 0.23 | 5648.48 | bt=256: 563.47 → +93.1% pp |
| d8192 | 1076.12 ± 0.89  | 9.71 ± 0.22  | 9517.92 | bt=256: 564.00 → +90.7% pp |
| d16384 | 1050.69 ± 2.16 | 9.70 ± 0.11  | 17545.09 | bt=256: 530.91 → +97.9% pp |

**Key observations:**
- Prefill (pp2048) improves by **~92-98%** vs bt=256. The 2048-token prompt is expected to
  fit within a single scheduling budget at bt=2048 instead of spanning 8 × 256-token
  scheduled batches. Scheduling batch count is inferred from budget math, not directly logged.
- Decode (tg32) regresses by ~18-22% vs bt=256 at d0 (10.06 vs 12.26 t/s). Regression is
  smaller at deeper contexts (d16384: 9.70 vs 11.52 t/s, ~16%). Mechanism not isolated
  — candidates include memory allocation difference from larger KV reservation, compilation
  effect, or sampling path change at higher batch token budget. A decode-only rerun would
  confirm whether the regression is reproducible; requires server restart (pending authorization).
- **v022 gap**: bt=2048 pp peak is ~1088 t/s (d4096) vs v022 baseline ~1300 t/s (d4096),
  approximately **−16%** below v022. The gap persists across all depths (−16–17%). This gap
  is not attributable to a single variable: confounders include attention backend
  (TRITON_ATTN vs FlashInfer in v022), MARLIN vs v022 MoE path, EP state difference
  (v022 production used EP=on), and any v023-per-se overhead. Isolating individual
  contributions requires controlled A/B runs not yet performed.

#### Infrastructure note: prometheus patch

**Root cause** (`prometheus-fastapi-instrumentator==8.0.0`): vLLM 0.23.0 registers routes
via FastAPI's `include_router()`, which produces `_IncludedRouter` objects. These lack a
`.path` attribute, so `_get_route_name()` in `routing.py` raises `AttributeError` inside
the Prometheus middleware on every HTTP request, including `/health`. All requests return
HTTP 500 before the model handler runs.

**Patch**: two occurrences of `route_name = route.path` changed to
`route_name = getattr(route, 'path', 'unknown')`. Only the head container needs patching
(prometheus instrumentator is only imported by the head API server).

**bt=2048 patch state**: Patch applied to head container at runtime via `apply_prometheus_patch()`
immediately after container start (before model load). Both containers restarted
simultaneously to avoid TCPStore desync.
- Pre-patch SHA256: `b90d08f601c5ec82245630667c0cbc031f00df038284b4e61f46945d182c85fb`
- Post-patch SHA256: `a3addfd90d1132a5ab5dca54c788f4743fe180b9607a662bf34ef0453750848c`
- Patch status: `patched` (confirmed by post-patch SHA256 check + `py_compile`)

**Runtime mutation caveat**: The patch modifies `routing.py` inside the running container.
It is ephemeral — a container restart without re-applying the patch would revert the file.
The current bench runner always re-applies the patch after each container start. To
eliminate runtime mutation, the fix should be baked into the Docker image at build time
(see immutable image path below).

**bt=256 patch state**: The bt=256 run used a long-running container (~18h uptime) that
pre-dated the `apply_prometheus_patch()` function. Whether the prometheus patch was applied
manually before that run is not recorded in `bt256-confirmed-20260618/metadata.txt`. The
bt=256 bench produced outputs and `/health` returned 200, so the patch was in effect;
the exact application method and timing are undocumented. This is a condition difference
between the two runs.

**Immutable image path**: To eliminate runtime mutation, add to the Docker build:
```dockerfile
RUN python3 -c "
import re, pathlib
p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/prometheus_fastapi_instrumentator/routing.py')
p.write_text(p.read_text().replace(\"route_name = route.path\", \"route_name = getattr(route, 'path', 'unknown')\"))
"
```
This is applicable to any image that pins `prometheus-fastapi-instrumentator==8.0.0`.

**Metadata discrepancy**: `metadata.txt` records `git_commit=98edf86a...` (the HEAD commit
at run time). The bench script on disk at run time contained the `apply_prometheus_patch()`
function (which had not yet been committed). The function was committed at `62edd3f` after
the run. The working-tree version executed was logically equivalent to `62edd3f` content.
`98edf86` is the git_commit recorded in metadata; `62edd3f` is the commit that captures the
actual script state. Both are noted here for provenance.

---

## Correctness validation

Each bt run includes the following correctness tests:

1. **Factual** — "largest prime < 100" → expects 97; max_tokens=100
2. **Multi-step arithmetic** — "15 factorial" → expects 1307674368000; max_tokens=600
3. **Unicode integrity** — Korean KTX question → checks for broken codepoints; max_tokens=400
4. **Finish reason** — "2+2" → expects `stop` finish_reason; max_tokens=100

A run is flagged `backend_validity=INVALID_MARLIN_NOT_CONFIRMED` if the MARLIN MoE
backend log line (`Using 'MARLIN' NvFp4 MoE backend out of potential backends: [...]`)
is not found. A run is flagged `INVALID_TRITON_NOT_CONFIRMED` if the TRITON_ATTN line
(`Using AttentionBackendEnum.TRITON_ATTN backend.`) is not found. Both are hard gates;
the benchmark does not execute and the containers are stopped. Such runs must be discarded.

**EP validation**: The runner detects EP state from the `[entrypoint] Running: vllm serve`
command line. If `--enable-expert-parallel` is absent, EP is classified as disabled.
vLLM 0.23 does not log `expert_parallel_size=1` when EP is at its default, so the entrypoint
command line is the authoritative source. Use `--expected-ep off|on` to gate on EP state.

### bt=2048 correctness classification (run_id=bt2048-20260618-093121)

| Test | Result | Classification |
|------|--------|---------------|
| 1 — largest prime < 100 | "97" | PASS |
| 2 — 15 factorial | "1307674368000" | PASS |
| 3 — Korean KTX question | Empty output | INCONCLUSIVE_OUTPUT_BUDGET |
| 4 — "What is 2+2?" | Truncated or empty | INCONCLUSIVE_OUTPUT_BUDGET |

**INCONCLUSIVE_OUTPUT_BUDGET**: Step-3.7-Flash is a reasoning model that generates a
`<think>...</think>` chain before the visible answer. The chain typically consumes
1000–3000+ tokens. Tests 3 and 4 used max_tokens=400 and max_tokens=100 respectively,
which are insufficient to complete the chain. The model ran out of output budget before
producing an answer token — output was empty or the sequence was cut mid-chain. This is
budget exhaustion, not garble or a backend correctness failure. Garble would produce
syntactically complete but semantically wrong text; these tests produced no text at all.

Tests 1 and 2 used max_tokens=100 and max_tokens=600 respectively. These sufficed because
the `<think>` chain for simple factual/arithmetic queries is shorter. Tests 3 and 4 require
at least max_tokens=2000 to reliably clear the reasoning chain (see vllm023_step37_garble_fix.md).

**Supplement run results** (`bt2048-supp-20260618-103333`, 2026-06-18T10:33Z, max_tokens=2048 for all tests):

| Test | max_tokens | Completion tokens | finish_reason | Result |
|------|-----------|------------------|--------------|--------|
| 1 — largest prime < 100 | 2048 | 71 (×2) | stop | PASS×2 |
| 2 — 15 factorial | 2048 | 209 (×2) | stop | PASS×2 |
| 3 — Korean KTX | 2048 | 1845 / 1420 | stop | PASS×2 |
| 4 — "What is 2+2?" | 2048 | 159 (×2) | stop | PASS×2 |

Test 3 required 1420–1845 tokens to complete the reasoning chain and return a full Korean
answer ("서울역에서 부산역까지 평균 2시간 30분~2시간 40분"). Arrows (`→`) in the output are
legitimate Unicode; not garble. Test 4 returned "2 + 2 equals 4." with finish_reason=stop.

**Conclusion**: All 4 correctness tests PASS. The original INCONCLUSIVE_OUTPUT_BUDGET
classification for Tests 3 and 4 was correct — the reasoning chain requires ≥1420 tokens,
well above the 400/100 token budgets used in the original run. No garble was observed.

Full results in `benchmarks/results/bt-matrix/bt2048-supp-20260618-103333/correctness-extended.md`
(gitignored; not committed).

### Decode-only rerun results (supplement `bt2048-supp-20260618-103333`)

**Configuration**: pp=1, tg=32, 5 runs per depth, depths d0/4096/8192/16384, `--latency-mode generation`.
Same server instance used for correctness supplement (started 2026-06-18T10:33Z; bench ran ~38 min after start).

#### Phase 8: Decode statistics and comparison

| Depth | Original bt=2048 (3 runs) | Decode-only (5 runs, pp=1) | bt=256 baseline (3 runs) |
|-------|--------------------------|---------------------------|--------------------------|
| d0    | 10.06 ± 0.04 t/s         | **12.97 ± 0.70 t/s**      | 12.26 ± 0.60 t/s         |
| d4096 | 10.06 ± 0.23 t/s         | **10.74 ± 0.50 t/s**      | 11.25 ± 0.20 t/s         |
| d8192 | 9.71 ± 0.22 t/s          | **10.63 ± 1.00 t/s**      | 11.47 ± 0.30 t/s         |
| d16384 | 9.70 ± 0.11 t/s         | **9.92 ± 0.23 t/s**       | 11.52 ± 1.58 t/s         |

Supplemental pp=1 prefill measurements (context extension cost):

| Depth | pp1@depth TTFR (ms) | t/s |
|-------|---------------------|-----|
| d4096 | 3969 ± 103 ms | 1079 ± 29 t/s |
| d8192 | 7854 ± 65 ms | 1066 ± 9 t/s |
| d16384 | 15987 ± 31 ms | 1036 ± 2 t/s |

**Verdict — d0 regression not reproduced under pp=1 methodology**: the decode-only bench
returns 12.97 t/s at d0, which is *higher* than both the original bt=2048 (10.06 t/s, +29%)
and the bt=256 baseline (12.26 t/s, +6%). At d4096–d16384, results fall between the two
reference runs (above original bt=2048, below bt=256 baseline).

#### Phase 9: Runtime identity comparison

The supplement run is not identical to the original bt=2048 run in all runtime dimensions.
Differences that may affect tg32 comparison:

| Property | Original bt=2048 | bt=2048 supplement |
|----------|----------------|--------------------|
| run_id | bt2048-20260618-093121 | bt2048-supp-20260618-103333 |
| image | v023-step3p7-fixed-kv-profile-skip-candidate | same |
| vLLM | 0.23.0 | same |
| git_commit | 98edf86 (runner script was 62edd3f state) | f9e09bd |
| EP state | disabled | disabled |
| MARLIN | confirmed | confirmed |
| TRITON_ATTN | confirmed | confirmed |
| bt | 2048 | 2048 |
| Prometheus patch | applied (SHA confirmed) | applied (SHA confirmed, hardened) |
| llama-benchy latency mode | api (default) | generation |
| Prefill in tg32 test | pp=2048 (bench used `--pp 2048 --tg 32`) | pp=1 |
| Server uptime at decode bench | ~9 min (fresh boot) | ~38 min (after correctness tests) |
| Boot IDs | 9f01d96f / 92bc7698 | same (no reboot between) |

**Root cause of d0 discrepancy**: The original bt=2048 tg32 measurement used pp=2048 prompts
(benchmark ran with `--pp 2048 --tg 32`). Under llama-benchy's concurrent test ordering, the
tg32 measurement at d0 incurs a 2048-token active context, not a 1-token context. Larger KV
cache pressure increases decode latency. The decode-only bench with pp=1 at d0 uses a 1-token
context, which explains the 12.97 vs 10.06 t/s difference.

The bt=256 baseline (12.26 t/s at d0) also used pp=2048 prompts, yet outperformed the original
bt=2048 (10.06 t/s). This residual gap (~18%) is real but not reproduced under pp=1. It may
reflect the server warm-up state (bt=256 container: ~18h uptime vs bt=2048: ~9 min), memory
allocation pattern under different bt values, or interaction between the pp=2048 and tg32
tests within the same benchmark session.

**Summary**: The decode regression observed in the original bt=2048 run is not confirmed as
a pure bt-setting effect. Confounders — particularly the effective context size at the tg32
measurement point (pp=2048 vs pp=1) — make the original bt=2048 tg32 values not directly
comparable to a pp=1 decode-only measurement. A controlled A/B (same methodology, same
prompts, bt=256 vs bt=2048) would be needed to isolate the bt contribution to decode rate.

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
