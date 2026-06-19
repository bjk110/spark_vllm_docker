# Step-3.7-Flash-NVFP4 v0.22 Long-Context Validation

**Date**: 2026-06-19  
**Preset**: `presets/step37-flash-nvfp4-tp2.env`  
**Status**: `EXPERIMENTAL — Stage A (eager, 32k, seq1) and Stage B (CUDA graph, 32k, seq1) validated with EP-on + mp backend from a clean boot. 262k context and multi-sequence operation remain unvalidated.`

## Hardware

| Node | Role | GPU | Driver | RAM |
|---|---|---|---|---|
| spark01 | head | NVIDIA GB10 (SM_121) | 610.43.02 | 121.63 GiB UMA |
| spark02 | worker | NVIDIA GB10 (SM_121) | 610.43.02 | 121.63 GiB UMA |

Network: 200 Gbps RoCE (enp1s0f0np0 / rocep1s0f0), 10.10.10.0/24

## Image

| Field | Value |
|---|---|
| Tag | `vllm-spark:v022-d568-step3p7-memcheck-bypass` |
| Base | `v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release` |
| Dockerfile | `dockerfiles/active/Dockerfile.step3p7-memcheck-bypass` (commit `42d6f5f`) |
| Both nodes | identical (ID `0bac1cfc9fd2`) |
| vLLM | 0.22.1 |
| CUDA toolkit | 13.2 (NGC 26.05) |

### Bypass patches applied in this image

| Patch | Effect |
|---|---|
| `patch_envs_register_skip_memcheck.py` | Registers `VLLM_SKIP_INIT_MEMORY_CHECK` env var |
| `patch_skip_init_memory_check.py` | Bypasses pre-init `request_memory()` assertion when var=1 |
| `patch_relax_profile_assertion.py` | Relaxes post-profile free-memory assertion |

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

## Stages Not Run

| Stage | Config | Status |
|---|---|---|
| Stage A | ep+mp, eager, 32k/seq1 | ✅ VALIDATED |
| Stage B | ep+mp, CUDA graph, 32k/seq1 | ✅ VALIDATED |
| Stage C | 262k/seq1 startup | NOT_RUN |
| Stage D | context ladder 32k→262k | NOT_RUN |
| Stage E | 262k/seq2 | NOT_RUN |
| Stage F/G | 262k/seq4 (exact preset) | NOT_RUN |

## Memory Checkpoints

| Checkpoint | spark01 MemAvailable | spark02 MemAvailable |
|---|---|---|
| Attempt 1 pre-run | 53.0 GiB | 53.1 GiB |
| Attempt 2 pre-run | ~60.75 GiB (retained UMA) | ~118 GiB |
| Attempt 3 pre-run (clean boot) | 113.8 GiB | 118.0 GiB |
| Stage A profiling peak | 107 GiB used | 84 GiB used |
| Post Stage A stop | ~19.7 GiB (retained) | ~19.8 GiB (retained) |
| After Stage A reboot (both) | 117.6 GiB | 118.0 GiB |

## Path-Specific Preflight

Before starting any container on this path:

```bash
scripts/diag/preflight-110gib-check.sh   # must exit 0
```

Threshold: 110 GiB per node (derived: 0.88 × 121.63 GiB = 107 GiB peak + 3 GiB margin).
If either node fails, reboot it — this is the only confirmed recovery for GB10 UMA retention.

## Preset Status

`presets/step37-flash-nvfp4-tp2.env` remains `EXPERIMENTAL`.
Stage A (eager, 32k, seq1, dynamic KV) validated with bypass image from clean boot.
CUDA graph, 262k context, and multi-sequence modes are unvalidated as of 2026-06-19.

**For validated production serving**: `presets/step37-flash-nvfp4-v023-tp2-latency.env`
(vLLM 0.23.0, MARLIN MoE, TRITON_ATTN — see `vllm023_step37_garble_fix.md`).
