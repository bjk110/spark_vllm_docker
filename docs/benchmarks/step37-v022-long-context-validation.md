# Step-3.7-Flash-NVFP4 v0.22 Long-Context Validation Attempt

**Date**: 2026-06-19  
**Preset**: `presets/step37-flash-nvfp4-tp2.env`  
**Status**: `BLOCKED — IMAGE_MISSING_INIT_MEMORY_CHECK_BYPASS`

## Hardware

| Node | Role | GPU | Driver | RAM |
|---|---|---|---|---|
| spark01 | head | NVIDIA GB10 (SM_121) | 610.43.02 | 121.63 GiB UMA |
| spark02 | worker | NVIDIA GB10 (SM_121) | 610.43.02 | 121.63 GiB UMA |

Network: 200 Gbps RoCE (enp1s0f0np0 / rocep1s0f0), 10.10.10.0/24

## Image

| Field | Value |
|---|---|
| Tag | `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release` |
| Image ID | `sha256:6e62e76c35a0fa57f669f4d5e0cc9fdcdd817414af97d720020963fd07f58777` |
| Both nodes | identical |
| vLLM | 0.22.1 |
| CUDA toolkit | 13.2 (NGC 26.05) |

## Target Configuration

| Key | Value |
|---|---|
| DISTRIBUTED_BACKEND | mp |
| TP_SIZE | 2 |
| EP | enabled (--enable-expert-parallel) |
| MAX_MODEL_LEN | 262144 |
| MAX_NUM_SEQS | 4 |
| GPU_MEMORY_UTILIZATION | 0.88 |
| MAX_NUM_BATCHED_TOKENS | 8192 |
| CUDA graph | enabled (VLLM_USE_BREAKABLE_CUDAGRAPH=0, no --enforce-eager) |
| KV cache | FP8 (--kv-cache-dtype fp8, dynamic) |
| MoE backend | CUTLASS (VLLM_NVFP4_GEMM_BACKEND=cutlass) |
| Attention | default (FlashInfer JIT; autotune disabled) |
| MASTER_PORT | 29500 |

## Pre-flight State

| Node | MemAvailable | Swap | Running containers | Port conflicts |
|---|---|---|---|---|
| spark01 | 53.0 GiB | 0 | portainer_agent only | none |
| spark02 | 53.1 GiB | 0 | portainer_agent only | none |

Both nodes: uptime 18h, no OOM/Xid in recent kernel log.

## Validation Ladder Attempt

### Stage A — EP-on + mp minimal topology (eager, 32k, seq1)

**Run**: 2026-06-19, Stage A disposable env  
**Result**: `FAILED`

#### Failure Detail

Worker container on spark02 exited within 30 seconds of startup with:

```
ValueError: Free memory on device cuda:0 (50.45/121.63 GiB) on startup is less
than desired GPU memory utilization (0.88, 107.03 GiB). Decrease GPU memory
utilization or reduce GPU memory used by other processes.
  at vllm/v1/worker/utils.py:413 in request_memory()
```

#### Root Cause Analysis

The `v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release` image
**does not include** `patches/common/patch_skip_init_memory_check.py`. This was
confirmed by:
1. Inspecting `dockerfiles/active/Dockerfile.step3p7` — only three patches are
   applied: `patch_registry_step3p7.py`, `patch_step3p7_nvfp4_input_scale.py`,
   `patch_step3p7_modelopt_cache_release.py`.
2. Runtime check: `VLLM_SKIP_INIT_MEMORY_CHECK` not present in
   `vllm.v1.worker.utils.request_memory` source inside the container.

Without the bypass patch, `vllm/v1/worker/utils.py:request_memory()` performs a
hard pre-init assertion:

```
requested_memory = GPU_MEMORY_UTILIZATION × cuda_total_memory
                 = 0.88 × 121.63 GiB = 107.03 GiB

if free_cuda_memory < requested_memory:
    raise ValueError(...)
```

On GB10 with driver 610.43.02, the CUDA-visible free memory at idle is ~50 GiB
(confirmed by the error: 50.45 GiB). This reflects the driver's UMA baseline
allocation, which remains ~71 GiB used even after a fresh reboot (see
`spark01_idle_uma_baseline_driver610.md`). A reboot does not increase CUDA free
memory beyond ~50 GiB.

**50.45 GiB < 107.03 GiB → unbypassable hard failure.**

#### Why the Preset Previously Appeared to Work

The `step37-flash-nvfp4-tp2.env` preset was originally authored and tested with
`MAX_MODEL_LEN=8192 / DISTRIBUTED_BACKEND=ray / --enforce-eager`. The README
notes "MAX_NUM_SEQS=4 verified" for that tracked configuration. The *current*
working-tree changes (`MAX_MODEL_LEN=262144 / DISTRIBUTED_BACKEND=mp`) were WIP
additions that had not been runtime-tested before this validation attempt.

The init memory check failure would also affect the original ray+eager path with
the same image and driver 610. It is possible the original validation was
performed with a different driver version (580 or earlier) or on freshly rebooted
nodes with less driver baseline UMA occupancy.

#### Required Fix

To validate this preset, the image must include the init-memory-check bypass.
Options:

1. **Rebuild the image** from `Dockerfile.step3p7` with the three additional
   patches from `dockerfiles/active/Dockerfile.step3p7-v023` appended:
   - `patches/common/patch_envs_register_skip_memcheck.py`
   - `patches/common/patch_skip_init_memory_check.py`
   - `patches/common/patch_relax_profile_assertion.py`
   Then set `VLLM_SKIP_INIT_MEMORY_CHECK=1` in the preset.

2. **Use the v0.23 image** (`v023-step3p7-fixed-kv-profile-skip-candidate`) which
   already includes the bypass. This would require updating the image tag in the
   preset and verifying CUDA-graph + EP-on + mp behavior with that image.

**Tuning `GPU_MEMORY_UTILIZATION` downward is not a valid workaround**: to pass
the check, `util × 121.63 < 50 GiB → util < 0.41`. At 0.41, only 49.9 GiB is
available for model weights + KV cache. The Step-3.7-Flash-NVFP4 model in
NVFP4 (4-bit) requires approximately 50 GiB per TP rank (198B × 0.5B / 2 ranks)
for weights alone, leaving zero headroom for KV or activations.

## Stages Not Run

All stages were blocked by the Stage A failure gate:

| Stage | Config | Status |
|---|---|---|
| Stage A | ep+mp, eager, 32k/seq1 | FAILED — init memory check |
| Stage B | ep+mp, CUDA graph, 32k/seq1 | NOT_RUN |
| Stage C | 262k/seq1 startup | NOT_RUN |
| Stage D | context ladder 32k→262k | NOT_RUN |
| Stage E | 262k/seq2 | NOT_RUN |
| Stage F/G | 262k/seq4 (exact preset) | NOT_RUN |

## Memory Summary

| Checkpoint | spark01 MemAvailable | spark02 MemAvailable |
|---|---|---|
| Pre-run | 53.0 GiB | 53.1 GiB |
| Stage A attempt (T+30s) | 51.7 GiB | 54.4 GiB (container never loaded) |
| Post-stop | 52.9 GiB | 53.1 GiB |

Worker container exited before model weights were loaded; no significant UMA
retained.

## Preset Status

`presets/step37-flash-nvfp4-tp2.env` remains `EXPERIMENTAL`. The EXPERIMENTAL
status note now includes the specific blocking reason (image missing bypass patch)
and the required fix path.

## Recommended Follow-up

1. Build a new image layer on `v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release`
   that applies the three missing bypass patches and adds `VLLM_SKIP_INIT_MEMORY_CHECK=1`
   to the preset.
2. Re-run Stage A with the patched image.
3. If Stage A passes, proceed through the full ladder (Stage B → C → D → E → F/G).

See `presets/step37-flash-nvfp4-tp2.env` header for validation scope and
limitations. Validated production path: `presets/step37-flash-nvfp4-v023-tp2-latency.env`.
