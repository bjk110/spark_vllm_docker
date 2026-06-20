# Step-3.7-Flash-NVFP4 v0.22 Long-Context Validation

**Date**: 2026-06-19 / 2026-06-20  
**Preset**: `presets/step37-flash-nvfp4-tp2.env`  
**Status**: `EXPERIMENTAL — STAGE_D_PARTIALLY_VALIDATED_TO_245009. Stages A through C and Stage D single-sequence requests through 245,009 prompt tokens are runtime-validated from a clean-memory precondition. A 257,891-token prompt reproducibly caused an infrastructure hang under the tested EP-on/multiprocessing/CUDA-graph configuration. Retrieval correctness at that depth was not evaluated. Multi-sequence operation remains unvalidated.`

## Hardware

| Node | Role | GPU | Driver | RAM |
|---|---|---|---|---|
| spark01 | head | NVIDIA GB10 (SM_121) | 610.43.02 | 121.63 GiB UMA |
| spark02 | worker | NVIDIA GB10 (SM_121) | 610.43.02 | 121.63 GiB UMA |

Network: 200 Gbps RoCE (enp1s0f0np0 / rocep1s0f0), 10.10.10.0/24

## Image

### Validation image (Stages A–D)

| Field | Value |
|---|---|
| Tag | `vllm-spark:v022-d568-step3p7-memcheck-bypass` |
| Base | `v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release` |
| Dockerfile | `dockerfiles/active/Dockerfile.step3p7-memcheck-bypass` (commit `42d6f5f`) |
| Both nodes | identical (ID `0bac1cfc9fd2`) |
| vLLM | 0.22.1 |
| CUDA toolkit | 13.2 (NGC 26.05) |

### Operational image (adds prompt-token admission control)

| Field | Value |
|---|---|
| Tag | `vllm-spark:v022-d568-step3p7-memcheck-bypass-prompt-cap` |
| Base | `vllm-spark:v022-d568-step3p7-memcheck-bypass` |
| Dockerfile | `dockerfiles/active/Dockerfile.step3p7-memcheck-bypass-prompt-cap` |
| Both nodes | identical (ID `a73ea6723649`) — spark01 built; spark02 synchronized via `docker save` transfer 2026-06-20 |
| Built | 2026-06-20 |

### Patches applied

| # | Patch | Effect |
|---|---|---|
| 1 | `patch_envs_register_skip_memcheck.py` | Registers `VLLM_SKIP_INIT_MEMORY_CHECK` env var |
| 2 | `patch_skip_init_memory_check.py` | Bypasses pre-init `request_memory()` assertion when var=1 |
| 3 | `patch_relax_profile_assertion.py` | Relaxes post-profile free-memory assertion |
| 4 | `patch_envs_register_prompt_cap.py` | Registers `VLLM_SPARK_MAX_PROMPT_TOKENS` env var |
| 5 | `patch_prompt_token_admission.py` | Adds per-request prompt-token admission control (HTTP 400 if exceeded) |

**Safety caveat**: The bypass does not recover memory. It only skips the guard
check. If `MemAvailable` is below 110 GiB before starting, profiling will still
exhaust UMA and cause kernel page-thrash (node unresponsive, reboot-only
recovery). Always run `scripts/diag/preflight-110gib-check.sh` and confirm PASS
before starting the server with `VLLM_SKIP_INIT_MEMORY_CHECK=1`.

## Stage A — EP-on + mp dual-node minimal topology (eager, 32k, seq1)

**Env**: `.local/env/step37/v022-longctx-stage-a-memcheck-patched.env`  
**Final result**: `STAGE_A_VALIDATED` ✅

### Attempt 1 — Original image, no bypass (FAILED: ValueError)

**Image**: `v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release`  
**State**: spark01 47.87 GiB CUDA-free at idle (driver 610.43.02 baseline), spark02 similar.

Worker container exited within 30 s:

```
ValueError: Free memory on device cuda:0 (50.45/121.63 GiB) on startup is less
than desired GPU memory utilization (0.88, 107.03 GiB).
  at vllm/v1/worker/utils.py:413 in request_memory()
```

**Root cause**: The base image does not include the init-memory-check bypass patches.
Without the bypass, `request_memory()` performs a hard pre-init assertion:
`GPU_UTIL × 121.63 GiB = 107.03 GiB required > 50.45 GiB available → raises ValueError`.

**Fix**: Build `Dockerfile.step3p7-memcheck-bypass` layer over the base image.

---

### Attempt 2 — Bypass image, retained UMA (FAILED: kernel thrash)

**Image**: `v022-d568-step3p7-memcheck-bypass`  
**State**: spark01 had ~60.75 GiB CUDA-free — prior vLLM session left ~61 GiB
retained in NVIDIA driver (expected GB10 UMA behaviour; see
`feedback_gb10_uma_memory_recovery.md`).

`VLLM_SKIP_INIT_MEMORY_CHECK=1` allowed the server to pass the guard, but:

1. Weight load consumed ~58.58 GiB → ~2 GiB remaining on spark01
2. Post-load profiling spike pushed allocation past physical UMA limit
3. Kernel entered page-in/page-out loop; spark01 SSH became unresponsive
4. Recovery: physical power-cycle (reboot)

**Root cause**: The bypass circumvented the guard check, but retained UMA
left insufficient headroom. **The bypass is not a memory-recovery mechanism.**
Pre-start MemAvailable must be ≥ 110 GiB for safe operation on this path.

---

### Attempt 3 — Bypass image, clean reboot (SUCCESS)

**Image**: `v022-d568-step3p7-memcheck-bypass`  
**Pre-start state** (both nodes freshly rebooted):

| Node | MemAvailable | Swap |
|---|---|---|
| spark01 | 113.8 GiB | 64 GiB free |
| spark02 | 118.0 GiB | 64 GiB free |

Both nodes ≥ 110 GiB → preflight PASS.

**Config**: `VLLM_SPARK_SKIP_FIXED_KV_PROFILE_RUN=0` (dynamic KV allocation),
`--enforce-eager` (CUDA graph disabled for this stage), MAX_MODEL_LEN=32768,
MAX_NUM_SEQS=1, GPU_MEMORY_UTILIZATION=0.88.

**Bypass log confirmed**:
```
VLLM_SKIP_INIT_MEMORY_CHECK=1 — skipping startup free-memory check
(free_memory=122,176,217,088, requested=114,924,587,910 on cuda:0)
```

**Stage A metrics**:

| Metric | Value |
|---|---|
| Weight loading | 58.58 GiB, 430.70 s |
| KV cache — head (spark01) | 36.01 GiB |
| KV cache — worker (spark02) | 37.22 GiB |
| GPU KV cache (total) | 1,749,960 tokens |
| 32k concurrency | 53.40× |
| Engine init time | 262.72 s |
| spark01 RAM peak | 107 GiB used, 2 GiB swap (profiling transient) |

**API validation**:

| Test | Result |
|---|---|
| `GET /health` | 200 OK ✅ |
| `POST /v1/completions` "2+2" | "2+2 = 4" ✅ |
| `POST /v1/chat/completions` Korean | "1+1은 2입니다" + reasoning tokens ✅ |
| Garble check | none ✅ |

---

## Stage B — EP-on + mp + CUDA graph (32k, seq1)

**Env**: `.local/env/step37/v022-longctx-stage-b-cudagraph-32k-seq1.env`  
**Config delta from Stage A**: `--enforce-eager` removed → CUDA graph enabled.  
**Result**: `STAGE_B_VALIDATED_CUDAGRAPH_32K_SEQ1` ✅

**CUDA graph config**:
- `cudagraph_mode`: `FULL_AND_PIECEWISE`
- `cudagraph_capture_sizes`: `[1, 2]` (driven by MAX_NUM_SEQS=1)
- `cudagraph_num_of_warmups`: 1

**Stage B metrics**:

| Metric | Value | Stage A delta |
|---|---|---|
| Weight loading | 58.58 GiB, 403.05 s | identical |
| torch.compile | 84.19 s (cache hit) | — |
| Initial profiling/warmup | 97.86 s | +97.86 s (new in Stage B) |
| CUDA graph memory | 0.02 GiB | — |
| KV cache (head) | 36.08 GiB | +0.07 GiB |
| GPU KV cache tokens | 1,740,325 | −9,635 (CUDA graph reservation) |
| Peak RAM during startup | ~107 GiB | no thrash ✅ |

**API validation**:

| Test | Result |
|---|---|
| `GET /health` | 200 OK ✅ |
| `POST /v1/completions` "2+2=" | "4..." ✅ |
| Korean reasoning | No garble, correct Korean ✅ |
| `15!` | `1307674368000` ✅ |
| 4k context needle (code=9871) | retrieved ✅ (stop) |
| 16k context needle (28,839 tokens) | retrieved ✅ (stop) |
| 30k context needle (30,039 tokens) | retrieved ✅ (stop) |

---

## Stage C — EP-on + mp + CUDA graph (262k, seq1)

**Env**: `.local/env/step37/v022-longctx-stage-c-262k-seq1.env`  
**Config delta from Stage B**: `MAX_MODEL_LEN` 32768 → 262144.  
**Result**: `STAGE_C_VALIDATED_262K_STARTUP_SEQ1` ✅

**Stage C metrics**:

| Metric | Value | Stage B delta |
|---|---|---|
| torch.compile | 82.26 s (cache miss: different `max_seq_len` key) | −1.93 s |
| Initial profiling/warmup | 102.70 s | +4.84 s |
| CUDA graph mode | FULL_AND_PIECEWISE, sizes=[1,2] | identical |
| CUDA graph memory | −0.06 GiB (net freed) | −0.08 GiB |
| KV cache (head) | 36.16 GiB | +0.08 GiB |
| GPU KV cache tokens | **2,838,891** | +1,098,566 (+63%) |
| KV capacity @ 262144 | 10.8× (capacity >> requirement) | — |
| Peak RAM during startup | ~108 GiB | no thrash ✅ |

Note: torch.compile cache key includes `max_seq_len`, so Stage C (262144) produces
a different cache entry than Stage B (32768). Recompile took 82 s (same duration as
Stage B first compile — expected).

**API validation**:

| Test | Result |
|---|---|
| `GET /health` | 200 OK ✅ |
| `GET /v1/models` | `stepfun-ai/Step-3.7-Flash-NVFP4` ✅ |
| `2+2` | `4` ✅ |
| `15!` | `1307674368000` ✅ |
| Korean ("대한민국 수도") | `대한민국의 수도는 서울특별시입니다.` ✅ |
| Code generation | Python `sum_list` function ✅ |
| Garble check | none ✅ |

**Context needle tests** (max_tokens=2000, stop=stop):

| Context | prompt_tokens | Secret | Result |
|---|---|---|---|
| 4k needle | 4,882 | 코드7731 | `7731` ✅ |
| 16k needle | 19,382 | 코드4829 | `4829` ✅ |
| 30k needle | 29,039 | 코드9157 | `9157` ✅ |

All needle tests used ≤ 29,039 input tokens (< 32k constraint). No >32k requests
were sent during Stage C validation.

**Memory checkpoints**:

| Checkpoint | spark01 MemAvailable | spark02 MemAvailable |
|---|---|---|
| Stage C pre-start (post reboot) | 117.7 GiB | 118.0 GiB |
| Stage C server running (post-test) | 12.69 GiB | 12.99 GiB |
| Stage C post-stop | 18.34 GiB | 18.48 GiB |

Post-stop retention ~103 GiB per node — standard GB10 UMA behaviour. Reboot required
before next launch (≥ 110 GiB gate).

**Stage C scope note**: Stage C validated that a server configured with
`MAX_MODEL_LEN=262,144` starts successfully, warmup completes without thrash, CUDA
graphs are captured, the KV cache allocates correctly, and short inference requests
succeed. Stage C did **not** send any request approaching 262,144 tokens; all needle
tests were ≤ 29,039 tokens. The actual long-context request boundary was measured in
Stage D.

---

---

## Stage D — Single-sequence context ladder (D0–D5, seq1)

**Env**: `.local/env/step37/v022-longctx-stage-d-single-seq.env`  
**Config**: Identical to Stage C env (zero server-setting differences). Only request context length varies.  
**Image**: `vllm-spark:v022-d568-step3p7-memcheck-bypass` ID `0bac1cfc9fd2` (both nodes)  
**Final classification**: `STAGE_D_PARTIALLY_VALIDATED_TO_245009`  
**D5 sub-classification**: `D5_INFRASTRUCTURE_HANG_AT_257891; RETRIEVAL_NOT_EVALUATED`

### Stage D — Overview

Stage D ran in three sub-phases across 2026-06-19/20:

1. **Preliminary attempt** (max_tokens=128/256) — infrastructure healthy, output budget
   exhausted at D0 before markers could be emitted. Classified
   `STAGE_D_NOT_EVALUATED_D0_OUTPUT_BUDGET_EXHAUSTED`. Not a retrieval failure.
2. **Rerun (max_tokens=2048)** — D0–D4 all passed. D5 caused spark01 to become
   operationally unresponsive on the first attempt. That attempt occurred within a server
   session that had started cleanly from a passed pre-start preflight; approximately
   13 GiB available on spark02 at the time of the D5 request was the normal
   serving-state value (the same range observed in Stages B, C, and during D0–D4 in
   this run). This is valid D5 hang evidence.
3. **D5 independent reproduction** — both nodes were rebooted and passed the ≥110 GiB
   pre-start gate again. A fresh server session was started. D5 again caused spark01 to
   become operationally unresponsive. Together, the two D5 attempts establish that the
   hang is reproducible and is not explained by serving-state memory alone.

No third D5 attempt was performed. Stage E–G were not run. Multi-sequence was not tested.

---

### Preliminary Stage D attempt (2026-06-20, max_tokens=128/256)

**Classification**: `STAGE_D_NOT_EVALUATED_D0_OUTPUT_BUDGET_EXHAUSTED`

| Metric | Attempt 1 | Attempt 2 |
|---|---|---|
| server_prompt_tokens | 29,953 | 29,976 |
| max_tokens | 128 | 256 |
| completion_tokens | 128 | 256 |
| finish_reason | length | length |
| content_preview | (empty) | `NEEDLE_BEGIN_D0` (partial) |
| infrastructure | PASS | PASS |
| retrieval verdict | NOT EVALUATED — output budget exhausted |

Infrastructure was healthy in both attempts. The model's `<think>` reasoning phase
consumed the entire output budget before markers could be emitted. This is not a
retrieval correctness failure; the model began outputting the first marker in attempt 2.

---

### Stage D rerun — D0–D4 results (max_tokens=2048)

**Server startup** (rerun, after clean reboot of both nodes):
- torch.compile: 1.19 s (AOT cache HIT — same `max_seq_len=262144` key as Stage C)
- profiling/warmup: 8.92 s (no thrash)
- GPU KV cache: 2,863,987 tokens
- Maximum concurrency at 262,144 tokens: 10.93×
- CUDA graph: FULL_AND_PIECEWISE, sizes=[1,2], pool 0.15 GiB
- API readiness: `Application startup complete`

#### D0 through D4 — PASS

| Depth | Target tokens | Local tokens | Server prompt tokens | Completion tokens | Finish reason | All markers | Order OK | Garble | TTFT (s) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| D0r | 30,000 | 30,005 | 30,005 | 239 | stop | ✅ | ✅ | ✅ | 39.18 | **PASS** |
| D1r | 64,000 | 63,977 | 63,977 | 351 | stop | ✅ | ✅ | ✅ | 77.19 | **PASS** |
| D2r | 128,000 | 127,817 | 127,817 | 301 | stop | ✅ | ✅ | ✅ | 134.93 | **PASS** |
| D3r | 192,000 | 191,999 | 191,999 | 235 | stop | ✅ | ✅ | ✅ | 217.60 | **PASS** |
| D4r | 245,000 | 245,009 | **245,009** | 253 | stop | ✅ | ✅ | ✅ | 295.61 | **PASS** |

Each depth: HTTP 200, three markers (BEGIN/MIDDLE/END) retrieved in correct order,
`finish_reason=stop`, no garble. Server-reported prompt tokens match local tokenizer
counts (delta = 0 for all depths).

**Maximum validated prompt length: 245,009 server-reported prompt tokens (D4r)**

Marker strings: `NEEDLE_BEGIN_D4r-20260620`, `NEEDLE_MIDDLE_D4r-20260620`,
`NEEDLE_END_D4r-20260620`. Payload SHA256: `535cd9fedc72d2ad...`

---

### D4 first attempt — unsafe precondition (excluded from boundary conclusion)

Before the clean D4 pass above, an earlier D4 request was sent while spark02 had
retained UMA from a previous session (~13–18 GiB MemAvailable vs. the required ≥110 GiB
threshold). spark01 became operationally unresponsive during that attempt and required
a power-cycle. This is excluded from the primary boundary conclusion because the
precondition was not met.

---

### D5 infrastructure hang — characterization

**D5 payload**: local_token_count = 257,891 (delta −109 from 258,000 target).
SHA256: `1d8cc1b7236af887...`. No server-reported token count is available — spark01
became unresponsive before a response was returned.

#### D5 attempt 1 (first occurrence)

This attempt occurred within the same server session that successfully completed D0–D4.
That session had started from a clean pre-start condition (both nodes passed the ≥110 GiB
pre-start gate before startup; see D0–D4 startup entry in the timeline). Approximately
13 GiB MemAvailable on spark02 at the time of the D5 request is the normal serving-state
floor — the same range observed at the end of Stages B and C and throughout the D0–D4
sequence in this run. It does not indicate a failed or unsafe startup precondition.

- Server state: loaded and serving (clean-start session, post-D0–D4 sequence)
- spark02 serving-state memory: ~13 GiB MemAvailable (normal for a loaded server)
- D5 request initiated
- spark01 became operationally unresponsive (ping: unreachable, SSH: no route to host)
- spark02 worker container remained running
- Recovery: power-cycle of spark01; spark02 rebooted thereafter

#### D5 attempt 2 — clean-condition reproduction (primary evidence)

- Precondition: both nodes rebooted; both passed the ≥110 GiB preflight
  (spark01: 117.7 GiB, spark02: 118.0 GiB)
- Server started cleanly; `Application startup complete` confirmed; API healthy
- D5 request initiated
- spark01 again became operationally unresponsive
- Recovery: power-cycle of spark01; spark02 rebooted thereafter

**Post-recovery diagnostics (both nodes, post power-cycle):**

| Observation | spark01 | spark02 |
|---|---|---|
| Kernel OOM message | not present | not present |
| Kernel hung-task message | not present | not collected (rebooted) |
| NVIDIA Xid error | not present (driver load only) | not collected (rebooted) |
| Memory PSI (post-reboot) | 0.00 all windows | 0.00 all windows |
| Swap in use | 0 | 0 |
| vLLM container OOMKilled flag | not available (reboot cleared state) | not available |
| Last spark02 container state | not available (rebooted) | not available |

Kernel OOM and Xid messages were absent in post-recovery `dmesg`. This reflects the
post-reboot kernel ring buffer — messages from the failure window are not preserved.

#### Observed facts (D5)

- The D5 request was initiated twice from a healthy API endpoint
- spark01 became operationally unresponsive (ping unreachable, SSH unreachable) during
  or shortly after both D5 requests
- The same behavior occurred under a clean-start configuration (attempt 2) where both
  nodes had ≥110 GiB MemAvailable before startup
- No kernel OOM message was collected from the failure window (logs not preserved across
  reboot)
- No watchdog trigger occurred (watchdog was not running during validation)
- No garble or partial response was returned; no HTTP response was received

#### Hypotheses (unconfirmed)

The following are consistent with the observations but were not confirmed by direct log
evidence:

- Transient UMA pressure during very large prefill at 257,891 tokens
- KV cache population growth during prefill
- Prefill workspace or intermediate activation memory growth
- Expert Parallel routing or dispatch-buffer pressure
- CUDA allocator fragmentation under sustained pressure
- Distributed synchronization failure secondary to head-side memory pressure

No individual component was proven to be the root cause. The failure is classified as an
infrastructure hang of unknown precise cause.

---

### Stage D — Final decision

**Primary classification**: `STAGE_D_PARTIALLY_VALIDATED_TO_245009`  
**D5 sub-classification**: `D5_INFRASTRUCTURE_HANG_AT_257891; RETRIEVAL_NOT_EVALUATED`

- D0–D4 passed with correct needle retrieval at all depths up to 245,009 prompt tokens
- D5 did not produce a retrieval correctness result
- D5 failed at the infrastructure level (spark01 operationally unresponsive)
- The failure was reproducible under a clean-start configuration
- The exact stable boundary between D4 (245,009 tokens, PASS) and D5 (257,891 tokens,
  HANG) was not narrowed — no intermediate tests were performed
- No third D5 attempt was performed
- Stage E–G were not run
- Multi-sequence was not tested

**Correct interpretation**:

> A 245,009-token prompt is runtime-validated for this exact dual-DGX-Spark
> configuration. A 257,891-token prompt reproducibly caused an infrastructure hang. This
> establishes an observed operational stability boundary between the validated D4 depth
> and the failing D5 depth for the tested configuration. The failure mechanism is
> consistent with transient UMA pressure during very large prefill but was not confirmed
> by direct log evidence. Reboot or power-cycle erased the kernel ring buffer, so no
> kernel OOM, UVM fault, Xid, or hung-task cause was conclusively captured.

---

### Stage D timeline

All times are UTC 2026-06-19/20.

| Time (approx.) | Event |
|---|---|
| ~22:00 | Preliminary D0 attempt 1 (max_tokens=128) — finish_reason=length |
| ~22:05 | Preliminary D0 attempt 2 (max_tokens=256) — finish_reason=length |
| ~00:23 | Stage D rerun diag directory created (T002349) |
| ~00:35 | Payloads generated (D0r–D5r, max_tokens=2048) |
| ~01:15 | Both nodes rebooted (first clean boot for rerun) |
| ~01:15 | Preflight PASS: spark01 117.7 GiB, spark02 118.0 GiB |
| ~01:20 | Server started (AOT compile 1.19 s, profiling 8.92 s) |
| ~01:36 | Application startup complete |
| ~01:36 | D0r submitted (30,005 tokens) → PASS 39.18 s |
| ~01:37 | D1r submitted (63,977 tokens) → PASS 77.19 s |
| ~01:38 | D2r submitted (127,817 tokens) → PASS 134.93 s |
| ~01:41 | D3r submitted (191,999 tokens) → PASS 217.60 s |
| ~01:45 | D4r first attempt submitted (245,009 tokens, spark02 dirty UMA ~13 GiB) |
| ~01:50 | spark01 unresponsive — power-cycle; spark02 rebooted |
| ~02:15 | Both nodes rebooted (second clean boot) |
| ~02:15 | Preflight PASS: spark01 117.7 GiB, spark02 118.0 GiB |
| ~02:20 | Server restarted |
| ~02:36 | Application startup complete |
| ~02:36 | D4r submitted (245,009 tokens, both nodes clean) → PASS 295.61 s |
| ~02:41 | D5r first attempt submitted (257,891 tokens, spark02 had running worker) |
| ~02:46 | spark01 unresponsive — power-cycle (D5 first occurrence) |
| ~03:00 | spark02 rebooted |
| ~03:10 | Both nodes rebooted (third clean boot) |
| ~03:10 | Preflight PASS: spark01 117.7 GiB, spark02 118.0 GiB |
| ~03:15 | Server restarted |
| ~03:30 | Application startup complete |
| ~03:45 | D5r second attempt submitted (257,891 tokens, both nodes clean) |
| ~03:50 | spark01 unresponsive — power-cycle (D5 second occurrence, primary evidence) |
| ~04:00 | spark02 rebooted |
| ~04:15 | Both nodes back online; post-recovery diagnostics collected |
| ~04:20 | Stage D closed; no further D5 retry |

---

## Memory Checkpoints

| Checkpoint | spark01 MemAvailable | spark02 MemAvailable |
|---|---|---|
| Attempt 1 pre-run | 53.0 GiB | 53.1 GiB |
| Attempt 2 pre-run | ~60.75 GiB (retained UMA) | ~118 GiB |
| Attempt 3 pre-run (clean boot) | 113.8 GiB | 118.0 GiB |
| Stage A profiling peak | 107 GiB used | 84 GiB used |
| Post Stage A stop | ~19.7 GiB (retained) | ~19.8 GiB (retained) |
| After Stage A reboot (both) | 117.6 GiB | 118.0 GiB |
| Stage B post-stop (retained) | ~18.5 GiB | ~18.6 GiB |
| After Stage B reboot (both) | 117.7 GiB | 118.0 GiB |
| Stage C pre-start | 117.7 GiB | 118.0 GiB |
| Stage C server running | 12.69 GiB | 12.99 GiB |
| Stage C post-stop (retained) | 18.34 GiB | 18.48 GiB |
| After Stage C reboot (both) | 117.7 GiB | 118.0 GiB |
| Stage D rerun pre-start (D0–D4 run) | 117.7 GiB | 118.0 GiB |
| Stage D server running (post-D0) | ~12 GiB | ~13 GiB |
| D4 first attempt — spark02 dirty precondition | 117 GiB (clean) | ~13–18 GiB (retained UMA) |
| D4 second attempt (clean) — both nodes pre-start | 117.7 GiB | 118.0 GiB |
| D5 first attempt — loaded serving state (normal) | ~117 GiB | ~13 GiB (serving-state floor, not a startup failure) |
| D5 second attempt — both nodes pre-start (clean) | 117.7 GiB | 118.0 GiB |
| Post Stage D recovery (both rebooted) | 117 GiB | 117 GiB |

## Memory Thresholds

This path uses two separate memory thresholds that serve different purposes:

| Threshold | Value | Purpose | When it applies |
|---|---|---|---|
| Pre-start gate | **110 GiB** MemAvailable | Required headroom before startup to safely survive weight load + profiling peak (~107 GiB) | Before `docker compose up` — must pass `preflight-110gib-check.sh` |
| Live-serving floor | **2 GiB** MemAvailable | Emergency watchdog lower bound during inference | After startup, while serving requests |

The 110 GiB requirement does **not** apply after the server has started. Once the model
is loaded, approximately 12–15 GiB remaining was the observed normal serving-state floor
across Stages B, C, and D. A serving-state value below 110 GiB is expected and does not
indicate a precondition failure.

`VLLM_SKIP_INIT_MEMORY_CHECK=1` bypasses a startup guard check. It does **not** reclaim
or create memory. If the node does not have ≥110 GiB free before startup, enabling the
bypass will allow the server to start but the subsequent profiling spike will exhaust UMA
and may cause the node to become unresponsive. A reboot is the only confirmed recovery
after a GB10 UMA thrash event — model shutdown alone does not return all retained UMA.

## Path-Specific Preflight

Before starting any container on this path:

```bash
scripts/diag/preflight-110gib-check.sh   # must exit 0 before docker compose up
```

Threshold: 110 GiB per node (derived: 0.88 × 121.63 GiB = 107 GiB peak + 3 GiB margin).
If either node fails, reboot it — this is the only confirmed recovery for GB10 UMA
retention. Do not run this check while the model server is loaded; the loaded serving
state will show ~12–15 GiB and will fail the pre-start gate even though the server is
healthy.

## Validated Context Ceiling

| Parameter | Value |
|---|---|
| Configured model maximum (`MAX_MODEL_LEN`) | 262,144 tokens |
| Maximum runtime-validated request depth (D4r) | **245,009 prompt tokens** |
| Reproducible failing request depth (D5) | 257,891 locally calculated prompt tokens |
| Exact stable boundary between D4 and D5 | not measured |
| Server-reported D5 token count | not available (no response returned) |

The 262,144 `MAX_MODEL_LEN` configures the engine's KV cache and attention window.
Stage C validated that a server with `MAX_MODEL_LEN=262,144` can start and serve short
requests correctly. Stage D validated actual long-context requests up to 245,009 tokens.
No request approaching 262,144 tokens was successfully completed.

Requests at or below 245,009 tokens are validated under the exact Stage D workload
configuration. This does not guarantee that every possible prompt up to that length is
universally safe — it establishes the maximum tested successful depth.

## Preset Status

`presets/step37-flash-nvfp4-tp2.env` remains `EXPERIMENTAL`.

`STAGE_D_PARTIALLY_VALIDATED_TO_245009`: Stages A through C and Stage D
single-sequence requests through 245,009 prompt tokens are runtime-validated from a
clean-memory precondition. A 257,891-token prompt
(`D5_INFRASTRUCTURE_HANG_AT_257891`) reproducibly caused an infrastructure hang under
the tested EP-on/multiprocessing/CUDA-graph configuration; retrieval correctness at that
depth was not evaluated. Multi-sequence operation remains unvalidated. Stage E–G not run.

## Operator Preset Selection

### Recommended production path

**`presets/step37-flash-nvfp4-v023-tp2-latency.env`** (vLLM 0.23.0)

Use this preset for:
- Normal production serving
- Validated single-user latency operation up to the approximately 29–32k prompt-token
  range validated during development
- Lower operational risk (no bypass patches, no UMA preflight dependency)
- EP-off path with MARLIN MoE and TRITON_ATTN backends (required for correctness on
  SM_121 in v0.23)

### Experimental long-context path

**`presets/step37-flash-nvfp4-tp2.env`** (this preset, vLLM 0.22.1)

Use only when:
- Context above the v0.23 latency preset's validated range is required
- The operator accepts experimental status and the manual reboot/preflight requirement
- Both nodes pass `scripts/diag/preflight-110gib-check.sh` (≥110 GiB) before each startup
- `MAX_NUM_SEQS=1` is maintained (the only validated concurrency level)
- Requests remain at or below the validated 245,009-token depth
- The operator understands that a 257,891-token prompt caused a reproducible
  infrastructure hang and that the exact safe boundary above 245,009 tokens has not been
  measured

**For validated production serving**: `presets/step37-flash-nvfp4-v023-tp2-latency.env`
(vLLM 0.23.0, MARLIN MoE, TRITON_ATTN — see `vllm023_step37_garble_fix.md`).

## Prompt-Token Admission Control

### Motivation

Stage D established a validated context ceiling of 245,009 prompt tokens and a
reproducible infrastructure hang at 257,891 tokens. The engine limit
(`MAX_MODEL_LEN=262,144`) is higher than either value and does not prevent requests
that fall between them. Without an additional gate, a request between 245,009 and
257,891 tokens could reach the engine unpredictably.

### Implementation

`VLLM_SPARK_MAX_PROMPT_TOKENS` is a server-side admission control env var added to
the vLLM 0.22 serving layer by `patch_prompt_token_admission.py`. It:

- Is read once at server startup and stored as an instance attribute on `OpenAIServing`
- Applies after chat-template rendering and tokenization, before engine enqueue
- Covers both `/v1/chat/completions` and `/v1/completions`
- Returns HTTP 400 with the actual token count and configured limit if exceeded
- Logs `[spark-prompt-cap] rejected request <id>: prompt_tokens=<N> limit=<L>` at WARNING

**Default: 0 (disabled).** Enabling requires the `prompt-cap` image layer.

### Image

`vllm-spark:v022-d568-step3p7-memcheck-bypass-prompt-cap` adds patches 4 and 5 on
top of `v022-d568-step3p7-memcheck-bypass`. See the [Image section](#image) above.

### Preset configuration

`presets/step37-flash-nvfp4-tp2.env` sets:

```
VLLM_SPARK_MAX_PROMPT_TOKENS=245009
```

This rejects any request whose tokenized prompt exceeds 245,009 tokens (the Stage D
validated ceiling) with HTTP 400, before the request reaches prefill. The engine limit
(`MAX_MODEL_LEN=262,144`) remains the hard ceiling; the prompt cap is an operational
policy layer above it.

### Validation (2026-06-20)

Unit tests: `tests/test_patch_prompt_token_admission.py` — 22 tests, all pass (homeserver, inside image).

Runtime integration test (cap=32,000, dual-Spark GB10, `v022-d568-step3p7-memcheck-bypass-prompt-cap`):

| Test | Prompt tokens (server-reported) | Expected | Result |
|---|---|---|---|
| Accepted (`/v1/chat/completions`, short) | ~10 | HTTP 200 | ✅ |
| Rejected (`/v1/chat/completions`, 33k words) | 33,013 | HTTP 400 | ✅ |
| Rejected (`/v1/completions`, 33k words) | 33,002 | HTTP 400 | ✅ |

- Startup log (7× APIServer workers): `[spark-prompt-cap] admission control active: max_prompt_tokens=32000` ✅
- Rejection log: `[spark-prompt-cap] rejected request chatcmpl-bd501b077cf51b4f: prompt_tokens=33013 limit=32000` ✅
- Rejection response body: `"Prompt token count 33013 exceeds the configured VLLM_SPARK_MAX_PROMPT_TOKENS limit of 32000. Reduce your prompt to at most 32000 tokens."` ✅
- No request above 64k was sent during the integration test
- Post-stop UMA: spark01 19.2 GiB, spark02 19.4 GiB (expected GB10 retention)
