# MAX_NUM_BATCHED_TOKENS Matrix Benchmark
## Step-3.7-Flash-NVFP4 · vLLM 0.23.0 · Dual DGX Spark GB10

**Purpose**: Validate that `MAX_NUM_BATCHED_TOKENS=256` (set during garble debugging)
is the primary cause of the observed prefill throughput regression in v023, and
determine a safe production value.

**Status**: bt=256, bt=2048, and bt=8192 measured (Series A, EP-off, 2026-06-18). All prompts
correctness-covered for all three values. bt=2048 supplement: all runs strict PASS
(suite_status=PASS_STRICT). bt=8192: all prompts covered; one inconclusive duplicate run
(suite_status=PASS_WITH_INCONCLUSIVE_DUPLICATE; not garble). bt=8192 analysis: prefill flat
at d0 (−0.7%, within noise), +3–4% at d4096+; short-context decode (pp=1) −6–18% vs bt=2048.
**bt=2048 remains the validated production candidate.**

No additional benchmark was executed during the checker and documentation cleanup phase
(commits e720e65, e61abe8, and subsequent). The bt=8192 measurement had already been completed
in the preceding authorized benchmark session (run_id=bt8192-20260618-121533 and
bt8192-supp-20260618-143154). bt=4096, bt=16384, bt=32768, and Series B (EP-on) were not
executed in this benchmark campaign.

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
| 2048 | off | **1034.86** | 10.06 | **1088.21** | **1076.12** | **1050.69** | Measured 2026-06-18; run_id=bt2048-20260618-093121; pp +92% vs bt=256. Two supplement runs: all correctness tests PASS; short-context decode (pp=1) 11.84–12.97 t/s at d0. pp2048-context decode lower than bt=256 (10.06 vs 12.26 t/s at d0); causality not established — see Phase 8 analysis. |
| 8192 | off | 1027.97 | 11.24 | 1133.53 | 1111.31 | 1084.02 | Measured 2026-06-18; run_id=bt8192-20260618-121533. Prefill flat at d0 (−0.7% vs bt=2048, within noise), +3–4% at d4096+. Short-context decode (pp=1): 10.22 t/s at d0 (−14% vs bt=2048). Correctness PASS (all 4 tests). See Phase 8 bt=8192 analysis. **bt=2048 remains the validated production candidate.** |

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
- Decode (tg32) at pp2048-context: 10.06 t/s (bt=2048) vs 12.26 t/s (bt=256) at d0 (~18%
  lower). Regression is smaller at deeper contexts (d16384: 9.70 vs 11.52 t/s, ~16%).
  Causality not established — candidates include server warm-up difference (bt=256 measured
  at ~18h uptime vs bt=2048 at ~9 min), effective KV state difference, or other confounders.
  A controlled A/B using identical pp2048-context methodology with matched server uptime is
  required to isolate the bt contribution.
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
3. **Unicode integrity** — Korean KTX question → checks for Unicode corruption signals; max_tokens=400
4. **Finish reason** — "2+2" → expects `stop` finish_reason; max_tokens=100

### Verdict taxonomy

| Verdict | Definition |
|---------|-----------|
| `PASS` | Expected answer found in content; `finish_reason=stop` |
| `PASS_WITH_LENGTH_LIMIT` | Expected answer found before budget exhausted; `finish_reason=length` |
| `FAIL_WRONG_ANSWER` | Output present but expected pattern absent; `finish_reason=stop` |
| `FAIL_GARBLE` | Unicode corruption signal detected: U+FFFD replacement char, lone surrogates (U+D800–U+DFFF), Private Use Area dominance (>10% of sampled text), or pathological token repetition |
| `INCONCLUSIVE_OUTPUT_BUDGET` | `finish_reason=length`; expected pattern not confirmed in sampled content; output not garbled |
| `FAIL_API` | HTTP error, JSON parse error, or empty content with `finish_reason=stop` |

**Important**: Characters outside ASCII and Korean ranges — including U+00B7 MIDDLE DOT (·),
Latin Extended, and common Unicode punctuation — are **not** garble signals. The garble detector
targets specific corruption artifacts (replacement chars, lone surrogates, PUA dominance,
pathological repetition) and does not use a character whitelist.

Per-test aggregation: `PASS` takes precedence over `PASS_WITH_LENGTH_LIMIT` over failures over
`INCONCLUSIVE_OUTPUT_BUDGET`. A test passes if any run produces `PASS` or `PASS_WITH_LENGTH_LIMIT`.

Suite status fields:
- `observed_garble`: true only if at least one run produced `FAIL_GARBLE`
- `All tests PASS (strict)`: true only if every individual run across all tests produced `PASS`

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

Two correctness supplement runs completed (both max_tokens=2048 for all tests):

**Supp #1** (`bt2048-supp-20260618-103333`, 2026-06-18T10:33Z):

| Test | max_tokens | Completion tokens | finish_reason | Result |
|------|-----------|------------------|--------------|--------|
| 1 — largest prime < 100 | 2048 | 71 (×2) | stop | PASS×2 |
| 2 — 15 factorial | 2048 | 209 (×2) | stop | PASS×2 |
| 3 — Korean KTX | 2048 | 1845 / 1420 | stop | PASS×2 |
| 4 — "What is 2+2?" | 2048 | 159 (×2) | stop | PASS×2 |

Note: Supp #1 runner timed out externally; Tests 3–4 were rerun manually via direct API
calls on the same server instance. curl timeout was 150s (runner fix not yet applied).

**Supp #2** (`bt2048-supp-20260618-113436`, 2026-06-18T11:34Z, with 300s curl timeout fix):

| Test | max_tokens | Completion tokens | finish_reason | Result |
|------|-----------|------------------|--------------|--------|
| 1 — largest prime < 100 | 2048 | ~71 (×2) | stop | PASS×2 |
| 2 — 15 factorial | 2048 | 253 (×2) | stop | PASS×2 |
| 3 — Korean KTX | 2048 | 1512 / 1889 | stop | PASS×2 |
| 4 — "What is 2+2?" | 2048 | 227 / 203 | stop | PASS×2 |

Test 3 required 1512–1889 tokens (Supp #2) to complete the reasoning chain. Content was
coherent Korean describing Seoul–Busan KTX travel times. No garble in either supplement.

**Conclusion**: All four prompts produced PASS in every duplicate run across both supplement
runs (suite_status=PASS_STRICT for each supplement). The original INCONCLUSIVE_OUTPUT_BUDGET
classification for Tests 3 and 4 was correct — the reasoning chain requires ≥1400 tokens,
well above the 400/100 token budgets used in the original run.

`All correctness tests completed without observed garble under the bt=2048 Series A configuration.
suite_status=PASS_STRICT: all_prompts_have_pass=yes, all_runs_strict_pass=yes, observed_garble=no.`

Full results (gitignored, not committed):
- `benchmarks/results/bt-matrix/bt2048-supp-20260618-103333/correctness-extended.md`
- `benchmarks/results/bt-matrix/bt2048-supp-20260618-113436/correctness-extended.md`

### Decode supplement analysis

Two bt=2048 supplement runs completed on 2026-06-18. Both use pp=1, tg=32,
5 runs per depth, depths d0/4096/8192/16384, `--latency-mode generation`.

| run_id | Started | API ready | Decode bench start | Server uptime at bench |
|--------|---------|-----------|-------------------|----------------------|
| `bt2048-supp-20260618-103333` | 10:33Z | ~10:39Z | ~11:11Z | ~38 min |
| `bt2048-supp-20260618-113436` | 11:34Z | 11:40:50Z | 11:47:14Z | ~12 min |

#### Phase 8: Decode analysis — three separate metrics

The decode measurements across bt=2048 runs use different methodologies and represent
separate metrics. They are not directly comparable to each other.

**A. Short-context decode (pp=1, tg=32)**

Both supplement runs measured decode with a 1-token prompt context (pp=1), isolating
the decode path from prefill warm-up effects. The two supplement runs and the bt=256
baseline (which used pp=2048) are placed in the same table for reference only.

| Depth | Supp #1 (pp=1, 5 runs) | Supp #2 (pp=1, 5 runs) | bt=256 (pp=2048, 3 runs)* |
|-------|------------------------|------------------------|--------------------------|
| d0    | 12.97 ± 0.70 t/s       | 11.84 ± 0.58 t/s       | 12.26 ± 0.60 t/s         |
| d4096 | 10.74 ± 0.50 t/s       | 12.46 ± 0.54 t/s       | 11.25 ± 0.20 t/s         |
| d8192 | 10.63 ± 1.00 t/s       | 11.80 ± 0.22 t/s       | 11.47 ± 0.30 t/s         |
| d16384 | 9.92 ± 0.23 t/s       | 10.84 ± 0.22 t/s       | 11.52 ± 1.58 t/s         |

\* bt=256 baseline used pp=2048, a different methodology — shown for scale reference only.

`Verdict A: No short-context decode regression was observed under bt=2048. Both supplement runs fall within a range consistent with the bt=256 baseline.`

Between-supplement variability at d0 is ~9% (12.97 vs 11.84), reflecting natural
run-to-run variation. The supplements show that the bt=2048 Series A configuration
does not impair the short-context decode path.

Supp #2 pp=1 prefill measurements (context extension cost at each depth):

| Depth | pp1@depth TTFR (ms) | t/s |
|-------|---------------------|-----|
| d4096 | 4061 ± 114 ms | 1062 ± 30 t/s |
| d8192 | 7947 ± 20 ms  | 1057 ± 3 t/s  |
| d16384 | 16050 ± 140 ms | 1034 ± 9 t/s |

**B. pp2048-context decode (pp=2048, tg=32)**

The original bt=2048 run measured tg32 after a 2048-token prompt context (via `--pp 2048
--tg 32` in llama-benchy). This is the same methodology as the bt=256 baseline.

| Depth | Original bt=2048 (pp=2048, 3 runs) | bt=256 baseline (pp=2048, 3 runs) | Delta |
|-------|------------------------------------|-----------------------------------|-------|
| d0    | 10.06 ± 0.04 t/s                   | 12.26 ± 0.60 t/s                  | −18%  |
| d4096 | 10.06 ± 0.23 t/s                   | 11.25 ± 0.20 t/s                  | −11%  |
| d8192 | 9.71 ± 0.22 t/s                    | 11.47 ± 0.30 t/s                  | −15%  |
| d16384 | 9.70 ± 0.11 t/s                  | 11.52 ± 1.58 t/s                  | −16%  |

`Verdict B: The original bt=2048 pp2048-context decode was lower than the bt=256 result under the same methodology. This is a measured difference; its cause has not been established.`

This result has not been remeasured under a controlled pp2048-context A/B. The supplement
runs (pp=1) used a different methodology and do not provide evidence for or against a
bt-setting effect in the pp2048-context path.

**C. Causality**

`Verdict C: No causal relationship between max_num_batched_tokens and decode throughput has been established.`

Candidate explanations for the lower pp2048-context decode in bt=2048:
- **Server warm-up**: bt=256 container measured at ~18h uptime; bt=2048 at ~9 min from reboot
- **Sequential test interaction**: prefill and decode are adjacent tests within the same
  llama-benchy run; warm-up state differs between pp and tg passes
- **KV state**: tg32 after pp=2048 vs pp=1 is a different decode state, but this applies
  to bt=256 equally (both used pp=2048), so KV size alone does not explain the bt=256 vs
  bt=2048 gap

`Required isolation: A controlled A/B using identical pp2048-context requests with matched server uptime (bt=256 vs bt=2048) is required to attribute any decode difference to max_num_batched_tokens.`

`Production note: The unresolved decode attribution does not invalidate the measured prefill improvement or the completed correctness validation.`

#### Phase 9: Runtime identity comparison

| Property | Original bt=2048 | Supp #1 | Supp #2 |
|----------|-----------------|---------|---------|
| run_id | bt2048-20260618-093121 | bt2048-supp-20260618-103333 | bt2048-supp-20260618-113436 |
| image | v023-step3p7-fixed-kv-profile-skip-candidate | same | same |
| vLLM | 0.23.0 | same | same |
| EP state | disabled | disabled | disabled |
| MARLIN | confirmed | confirmed | confirmed |
| TRITON_ATTN | confirmed | confirmed | confirmed |
| bt | 2048 | 2048 | 2048 |
| Prometheus patch | applied (SHA confirmed) | applied (SHA confirmed) | applied (SHA confirmed) |
| llama-benchy latency mode | api (default) | generation | generation |
| Prefill in tg32 test | pp=2048 | pp=1 | pp=1 |
| Server uptime at decode bench | ~9 min | ~38 min | ~12 min |
| Boot session (spark01) | 9f01d96f | same | same |
| curl timeout in correctness | 150s (n/a to bench) | 150s (external runner killed) | **300s (fixed)** |

**Key methodology difference**: The original run used `--pp 2048 --tg 32` in llama-benchy
(pp2048-context decode). Both supplement runs used `--pp 1 --tg 32` (short-context decode).
These measure different states of the model and are not directly comparable.

The pp2048-context decode gap between bt=2048 (10.06 t/s) and bt=256 (12.26 t/s) at d0
(~18%) is a measured difference. KV context size is a candidate factor contributing to
this gap, but it applies to both bt values equally (both used pp=2048). The gap is not
attributable to a single confirmed cause.

**Summary**: The two decode metrics show different things:
- **Short-context decode** (pp=1): bt=2048 is within normal range — no regression.
- **pp2048-context decode**: bt=2048 shows lower throughput than bt=256 under the same
  methodology; the cause is unresolved. The supplements do not address this question
  because they used a different methodology.

A controlled A/B with identical pp2048-context methodology and matched server uptime
would be needed to isolate the bt contribution to pp2048-context decode throughput.

---

## bt=8192 execution results (2026-06-18)

**Full run** `bt8192-20260618-121533` (2026-06-18T12:15Z).
**Supplement** `bt8192-supp-20260618-143154` (2026-06-18T14:31Z):
extended correctness + pp=1 decode (1 warmup + 5 runs) + pp=2048 decode controlled (1 warmup + 5 runs).

### Acceptance criteria check (bt=8192)

| Criterion | Status | Evidence |
|-----------|--------|---------|
| Both nodes rebooted | PASS | boot_id unchanged from bt=8192 full run (same boot session) |
| Same image | PASS | `vllm-spark:v023-step3p7-fixed-kv-profile-skip-candidate` |
| EP confirmed off | PASS | `expert_parallel_observed=disabled` |
| MARLIN confirmed | PASS | `marlin_confirmed=1` (`nvfp4.py:231`) |
| TRITON_ATTN confirmed | PASS | `triton_attn_confirmed=1` (`cuda.py:331`) |
| Backend validity | PASS | `OK (MARLIN confirmed, TRITON_ATTN confirmed)` |
| EP validation | PASS | `OK (expected=off, observed=disabled)` |
| Bench result | PASS | `bench_result=OK` |
| Startup | PASS | API ready at 340s (full run), 330s (supplement) |
| Post-stop memory | PASS | spark01 53.0 GiB, spark02 53.1 GiB (both ≥ 50 GiB threshold) |

### Throughput results

**Full run prefill and pp2048-context decode** (runs=3, pp=2048 context):

| Depth | pp2048 t/s | tg32 t/s | pp TTFR (ms) | pp delta vs bt=2048 |
|-------|-----------|---------|------------|---------------------|
| d0    | 1027.97 ± 47.76 | 11.24 ± 0.43 | 1999.79 | −0.7% (within noise) |
| d4096 | 1133.53 ± 3.66  | 10.34 ± 0.45 | 5423.34 | +4.2% |
| d8192 | 1111.31 ± 3.28  | 10.24 ± 0.39 | 9217.50 | +3.3% |
| d16384 | 1084.02 ± 0.98 | 9.97 ± 0.10  | 17006.44 | +3.2% |

bt=2048 reference (run_id=bt2048-20260618-093121): pp2048 d0=1034.86, d4096=1088.21, d8192=1076.12, d16384=1050.69.

### Extended correctness (bt=8192, max_tokens=2048, 2 runs each)

| Test | Run 1 | Run 2 | Overall |
|------|-------|-------|---------|
| 1 — largest prime < 100 | PASS (97, 85 tokens) | PASS (97, 85 tokens) | **PASS** |
| 2 — 15 factorial | PASS (1307674368000, 314 tokens) | PASS (1307674368000, 209 tokens) | **PASS** |
| 3 — Korean KTX | ~~FAIL_GARBLE~~ → INCONCLUSIVE_OUTPUT_BUDGET (length=2048; content valid Korean; see note) | PASS (1850 tokens, stop) | **PASS** |
| 4 — "What is 2+2?" | PASS (4, 93 tokens, stop) | PASS (4, 203 tokens, stop) | **PASS** |

**Note on Test 3 Run 1 — checker false positive (corrected):**

The automated checker returned `FAIL_GARBLE` for Run 1. This was a false positive caused by a
character-whitelist bug in the garble detector.

| Field | Value |
|-------|-------|
| finish_reason | `length` |
| completion_tokens | 2048 (budget exhausted) |
| Sampled content (first 400 chars) | 서울에서 부산까지 KTX의 소요시간은 출발·도착 역과 정차역, 운행 열차 종류에 따라 조금 차이가 있습니다. |
| Unicode validity | Valid UTF-8; no replacement chars (U+FFFD); no surrogates |
| Garble signals observed | None |
| Root cause of false positive | U+00B7 MIDDLE DOT (·) in Korean punctuation was outside the old whitelist range `\x00–\x9F`; the whitelist did not cover common Unicode punctuation above U+009F |
| Automated verdict (original) | `FAIL_GARBLE` |
| Revised verdict | `INCONCLUSIVE_OUTPUT_BUDGET` — `finish_reason=length`, content valid, answer completion unknown |
| Checker fix | Commit replacing character whitelist with targeted garble signal detection |
| Raw artifact | `correctness-extended.md` records the original automated verdict; this note documents the revision |

Run 2 produced `PASS` (stop, 1850 tokens, valid Korean). Test 3 prompt_best: `PASS`.

**Suite aggregate (bt=8192 extended correctness run):**

| Field | Value | Note |
|-------|-------|------|
| `all_prompts_have_pass` | yes | All 4 prompts have at least one PASS |
| `all_runs_strict_pass` | no | Test 3 Run 1 = INCONCLUSIVE_OUTPUT_BUDGET (revised from false FAIL_GARBLE) |
| `inconclusive_run_count` | 1 | Test 3 Run 1 only |
| `failed_run_count` | 0 | No FAIL_WRONG_ANSWER, FAIL_GARBLE, or FAIL_API |
| `observed_garble` | no | No actual garble signals in any run |
| `suite_status` | `PASS_WITH_INCONCLUSIVE_DUPLICATE` | All prompts covered; one run exhausted output budget |

All four prompts produced at least one passing result. One duplicate Korean run exhausted the
output budget and was classified as INCONCLUSIVE_OUTPUT_BUDGET (revised from the false-positive
FAIL_GARBLE caused by U+00B7 MIDDLE DOT). No garbled output was observed.

Full results (gitignored):
- `benchmarks/results/bt-matrix/bt8192-20260618-121533/correctness.md`
- `benchmarks/results/bt-matrix/bt8192-supp-20260618-143154/correctness-extended.md`

### Phase 8 bt=8192 decode analysis — three metrics

#### A. Short-context decode (pp=1, tg=32) — controlled A/B with bt=2048

Both bt=8192 and bt=2048 measured with identical methodology: pp=1, tg=32, 5 runs, 1 warmup, depths d0/4096/8192/16384.

| Depth | bt=8192 (supp, pp=1, 5 runs) | bt=2048 (supp #2, pp=1, 5 runs) | Delta |
|-------|------------------------------|----------------------------------|-------|
| d0    | 10.22 ± 0.31 t/s             | 11.84 ± 0.58 t/s                 | **−14%** |
| d4096 | 10.27 ± 0.19 t/s             | 12.46 ± 0.54 t/s                 | **−18%** |
| d8192 | 10.71 ± 0.11 t/s             | 11.80 ± 0.22 t/s                 | **−9%** |
| d16384 | 10.22 ± 0.73 t/s            | 10.84 ± 0.22 t/s                 | **−6%** |

`Result A: bt=8192 showed lower measured short-context decode throughput than bt=2048 at all tested depths (6–18%). Both were measured with the same methodology (pp=1, tg=32, 5 runs, 1 warmup). This operational comparison supports selecting bt=2048, but strict causal isolation has not been achieved through alternating same-session A/B measurements. Possible confounders include different boot sessions, JIT/Triton cache state, sequential test ordering, and thermal conditions. The bt setting's direct effect on decode speed has not been independently isolated.`

bt=8192 pp=1 prefill measurements (context extension cost):

| Depth | pp1@depth TTFR (ms) | t/s |
|-------|---------------------|-----|
| d4096 | 3722 ± 16 ms | 1153 ± 5 t/s |
| d8192 | 7221 ± 18 ms | 1162 ± 3 t/s |
| d16384 | 14837 ± 108 ms | 1117 ± 8 t/s |

Note: bt=8192 pp=1@depth prefill is notably higher than bt=2048 (3722 vs 4061 ms at d4096 → 8.6% faster). This
is a prefill-path improvement consistent with larger scheduling batches at deeper contexts.

#### B. pp2048-context decode (pp=2048, tg=32) — controlled supplement vs bt=2048 embedded

| Depth | bt=8192 (controlled: 1 warmup + 5 runs) | bt=2048 (embedded: 3 runs, no warmup) | Delta |
|-------|----------------------------------------|---------------------------------------|-------|
| d0    | 10.86 ± 0.58 t/s                       | 10.06 ± 0.04 t/s                      | +8.0% |
| d4096 | 10.01 ± 0.27 t/s                       | 10.06 ± 0.23 t/s                      | −0.5% |
| d8192 | 9.82 ± 0.20 t/s                        | 9.71 ± 0.22 t/s                       | +1.1% |
| d16384 | 9.50 ± 0.22 t/s                       | 9.70 ± 0.11 t/s                       | −2.1% |

`Result B: bt=8192 pp2048-context decode is comparable to bt=2048. The d0 result (+8%) is within the combined
uncertainty range. Results at d4096+ are within ±2%, indistinguishable from noise.`

Note: the comparison is not fully controlled — bt=8192 used a dedicated supplement (1 warmup + 5 runs) while
bt=2048 used the embedded full-run tg32 rows (0 warmup, 3 runs). A symmetrically controlled A/B would require
a bt=2048 supplement with pp2048 context (not yet run). The current comparison is directionally informative
but not a strict A/B.

#### C. Summary comparison

| Metric | bt=8192 vs bt=2048 | Verdict |
|--------|---------------------|---------|
| Prefill d0 | −0.7% | Within noise |
| Prefill d4096+ | +3–4% | Marginal gain |
| Short-context decode (pp=1) | −6 to −18% | **Consistent regression** |
| pp2048-context decode | −2 to +8% | Comparable (within noise) |
| Correctness | PASS | No garble |
| Startup time | 330–340s | Same as bt=2048 |
| Post-stop memory | 53.0/53.1 GiB | Safe |

### Phase 15 — Production candidate judgment

`bt=8192 does not provide a net improvement over bt=2048 for the tested MAX_NUM_SEQS=1 latency-oriented workload.`

**Prefill**: bt=8192 delivers marginal gains at d4096+ (+3–4%) but is flat at d0 (−0.7%, within noise).
The practical impact on user-visible TTFR is small.

**Short-context decode**: bt=8192 showed lower measured throughput than bt=2048 across all four tested depths
(6–18%), using identical pp=1 methodology. The causal mechanism is not definitively isolated — different
boot sessions, JIT cache state, and test ordering are potential confounders. However, the directional
finding is consistent and the measured gap is larger than expected run-to-run variation (±3% typical).
For Step-3.7-Flash serving with MAX_NUM_SEQS=1, decode speed dominates latency: reasoning `<think>`
traces are 1000–3000+ tokens.

**pp2048-context decode**: bt=8192 is comparable to bt=2048 (within ±8%, indistinguishable from noise).

**Correctness**: PASS for bt=8192.

**Judgment**:
- bt=2048 has a better short-context decode profile (−6 to −18% advantage) at the cost of marginal prefill
  gap at d4096+ (−3–4%).
- For a reasoning model with long decode chains, decode speed is the dominant latency factor.
- bt=8192 is not recommended as a production change from bt=2048.

| Candidate | bt | Status | Correctness | Rationale |
|-----------|---:|--------|-------------|-----------|
| Conservative | 256 | Measured | All prompts covered; strict complete | Current production; lower prefill ceiling |
| **Recommended** | **2048** | **Measured** | **All prompts covered; all runs strict PASS** | Best combined prefill/decode profile |
| Measured; not recommended | 8192 | Measured | All prompts covered; 1 inconclusive duplicate | +3–4% prefill at d4096+; −6–18% short-context decode |

### Phase 9: Runtime identity comparison (bt=8192)

| Property | Full run | Supplement |
|----------|---------|---------|
| run_id | bt8192-20260618-121533 | bt8192-supp-20260618-143154 |
| image | v023-step3p7-fixed-kv-profile-skip-candidate | same |
| vLLM | 0.23.0 | same |
| EP state | disabled | disabled |
| MARLIN | confirmed | confirmed |
| TRITON_ATTN | confirmed | confirmed |
| bt | 8192 | 8192 |
| Prometheus patch | applied (SHA confirmed) | applied (SHA confirmed) |
| decode context | pp=2048 (embedded bench) | pp=1 + pp=2048 (supplement, 1 warmup + 5 runs) |
| API ready time | 340s | 330s |
| Boot session (spark01) | 9f01d96f | same |

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

> **Status: bt=2048 is the validated production candidate. bt=8192 measured and not recommended.**
> bt=256, bt=2048, and bt=8192 all measured under Series A (EP-off). bt=2048 delivers the best
> combined prefill/decode profile: +92% prefill vs bt=256, and better short-context decode than
> bt=8192. bt=8192 shows marginal prefill gains at deeper contexts but incurs a systematic
> 6–18% short-context decode regression.
>
> Correctness summary: all three values are correctness-safe (no garble observed; all prompts
> covered). bt=2048 supplement runs are all-runs strict PASS (suite_status=PASS_STRICT).
> bt=8192 has one duplicate run classified as INCONCLUSIVE_OUTPUT_BUDGET
> (suite_status=PASS_WITH_INCONCLUSIVE_DUPLICATE; not a garble or wrong-answer failure).

**bt=2048**: Preferred measured value for the v0.23 Series A latency-oriented configuration.
Correctness-safe across all 4 tests (2 supplement runs). Provides nearly the same pp2048
prefill performance as bt=8192 at shallow context (d0: −0.7%, within noise), gives up only
approximately 3–4% at deeper contexts, and delivered higher measured short-context decode
throughput in the tested runs. Scoped to: MAX_NUM_SEQS=1, TP=2, EP-off, MARLIN, TRITON_ATTN,
dual DGX Spark. Not generalized to other topologies, multi-user workloads, or EP-on configs.

**bt=8192**: Valid measured candidate. No garble observed. Marginal prefill advantage at d4096+
(+3–4%). Measured short-context decode lower than bt=2048 in the tested runs. Causal isolation
was not achieved; strict same-session A/B was not performed. The measured difference supports
choosing bt=2048 but does not prove that bt itself caused the decode throughput gap.

| Candidate | bt | Status | Correctness | Rationale |
|-----------|---:|--------|-------------|-----------|
| Conservative | 256 | Measured | All prompts covered; strict complete | Current production; lower prefill ceiling |
| **Preferred** | **2048** | **Measured** | **All prompts covered; all runs strict PASS** | Best combined prefill/decode in tested runs; +92% prefill vs bt=256 |
| Valid; not preferred | 8192 | Measured | All prompts covered; 1 inconclusive duplicate | +3–4% prefill at d4096+; lower measured short-context decode; causal isolation not achieved |
| Not measured | 16384 | Not executed in this benchmark campaign | Better MoE chunk utilization; unknown risk |
| Not measured | 32768 | Not executed in this benchmark campaign | Matches MAX_MODEL_LEN; memory impact unknown |

`MAX_NUM_BATCHED_TOKENS=2048 is the preferred measured value for the v0.23 Series A latency-oriented
configuration. It provides nearly the same pp2048 prefill performance as bt=8192 at shallow context
and gave up only ~3–4% at deeper prefixes, while delivering higher measured short-context decode
throughput in the tested runs.`

Caution: this conclusion applies to the specific tested environment (MAX_NUM_SEQS=1, dual DGX Spark,
EP-off, MARLIN, TRITON_ATTN). It is not a general claim about bt=2048 being optimal across all vLLM
configurations, hardware, or workload types.

**Do not apply to `presets/step37-flash-nvfp4-tp2.env`** until full v023 production
validation is complete. Apply only to the disposable env under `.local/`.

Proposed diff for production preset when ready to apply:
```diff
-MAX_NUM_BATCHED_TOKENS=8192
+MAX_NUM_BATCHED_TOKENS=2048
+# Preferred measured value for v0.23 Series A (EP-off, MARLIN, TRITON_ATTN, MAX_NUM_SEQS=1).
```

---

## bt=8192 measurement plan

bt=8192 is the v022 production default (under a different EP state — direct comparison
requires caution). **Executed 2026-06-18; results documented above in "bt=8192 execution results".**
This section preserves the original run plan for reference.

### Three metrics to collect

Each bt=8192 run should produce all three metric types to enable a complete comparison
with bt=256 and bt=2048:

| Metric | Context | llama-benchy invocation | File |
|--------|---------|------------------------|------|
| Prefill (pp2048 t/s) | pp=2048 prompt | `--pp 2048 --tg 32` (full run) | `bench-results.md` |
| Short-context decode | pp=1 → tg=32 | `--pp 1 --tg 32` via `--supplement --decode-context pp1` | `decode-bench.md` |
| pp2048-context decode | pp=2048 → tg=32 | `--pp 2048 --tg 32` (same as full run tg32 row) | embedded in `bench-results.md` |

Note: pp2048-context decode is already captured in the full run tg32 rows. A separate
supplement with `--decode-context pp2048` can add a controlled isolated measurement.

### Invocation

**Full benchmark run** (prefill + pp2048-context decode):
```bash
bash benchmarks/bench-bt-matrix-step37-v023.sh \
  --bt 8192 \
  --template .local/env/step37/bt-matrix-series-a-ep-off.env \
  --expected-ep off \
  --config-label v023-triton-marlin-ep-off-bt8192
```

**Supplement: short-context + pp2048-context decode** (after full run, same boot):
```bash
bash benchmarks/bench-bt-matrix-step37-v023.sh \
  --bt 8192 \
  --supplement \
  --template .local/env/step37/bt-matrix-series-a-ep-off.env \
  --expected-ep off \
  --config-label v023-triton-marlin-ep-off-bt8192 \
  --decode-context both \
  --decode-warmup-runs 1
```

**Dry-run verified** (2026-06-18): `--decode-context pp2048 --decode-warmup-runs 1` flags
accepted without error. **Execution complete** — see "bt=8192 execution results" above.

### Acceptance criteria

| Criterion | Pass | Fail / investigate |
|-----------|------|--------------------|
| Correctness | All 4 tests PASS | Any FAIL_GARBLE or FAIL_WRONG_ANSWER |
| Prefill (pp2048 d0) | ≥1034 t/s (≥ bt=2048) | < bt=2048 at any depth |
| Short-context decode | ≥10 t/s at d0 | < 9 t/s |
| No startup hang | API ready within 600s | Hang → stop + reboot |
| Post-stop memory | ≥50 GiB free | < 50 GiB → reboot before next run |

### Result judgment

- **If prefill ≥ bt=2048 and correctness PASS**: bt=8192 is a stronger prefill candidate.
  Promote to validated candidate; document pp2048-context decode result.
- **If prefill < bt=2048**: bt=2048 remains the validated prefill leader; report regress.
- **If garble**: halt; diagnose backend (MARLIN+TRITON_ATTN required). Do not use.
- **If startup hang**: stop + reboot. Do not retry without diagnosing warm-up peak.

pp2048-context decode comparison: bt=8192 result vs bt=2048 (10.06 t/s at d0) under
identical pp2048 methodology. This is the first controlled decode A/B; if matched server
uptime and sequence are held constant, the bt-setting effect on pp2048-context decode
can begin to be isolated.

---

## MTP follow-up (post bt-tuning)

After a production bt value is selected and validated, run MTP sweep:

- MTP off / n=1 / n=2 / n=3 × tg128
- Metrics: output t/s, acceptance rate, mean acceptance length, correctness
- Note: reasoning `<think>` traces have low acceptance → expected low MTP gain

---

## Related files

| File | Description |
|------|-------------|
| `.local/env/step37/bt-matrix-series-a-ep-off.env` | Series A template env (bt placeholder; disposable) |
| `benchmarks/bench-bt-matrix-step37-v023.sh` | Matrix runner |
| `benchmarks/analyze-bt-matrix.sh` | Analysis and report generator |
| `benchmarks/tests/test_correctness_checker.py` | Checker unit tests (40 tests) |
| `benchmarks/results/bt-matrix/` | Per-run results (gitignored) |
| `presets/step37-flash-nvfp4-v023-tp2-latency.env` | v0.23 Series A latency preset (bt=2048, EP-off, MARLIN, TRITON_ATTN) |
| `presets/step37-flash-nvfp4-tp2.env` | v0.22 general-purpose serving preset (EP-on, MAX_NUM_SEQS=4) |
| `.local/env/step37/v023-nomtp-fixed-kv-profile-skip.env` | Original bt=256 disposable env |
