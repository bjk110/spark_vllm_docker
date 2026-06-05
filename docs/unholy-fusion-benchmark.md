# unholy-fusion (aidendle94) — Configuration & Benchmark Results

**Image**: `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`  
**Source fork**: `local-inference-lab/vllm:dev/unholy-fusion`  
**GHCR mirror**: `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`  
**Tested**: 2026-06-05 | **llama-benchy**: 0.3.7

---

## Background

The unholy-fusion fork adds custom GB10 (Blackwell sm_120/sm_121) kernels
unavailable in the jasl lineage:

| Env var | Kernel | Status in test |
|---------|--------|----------------|
| `VLLM_USE_B12X_MOE=1` | Custom MoE dispatcher for GB10 | **Enabled** |
| `VLLM_USE_B12X_MHC` | Multi-head compression | Disabled (unstable) |
| `VLLM_USE_B12X_FP8_GEMM` | FP8 GEMM override | Disabled |
| `VLLM_USE_B12X_SPARSE_INDEXER` | Sparse attention indexer | Disabled |
| `VLLM_USE_B12X_WO_PROJECTION` | Weight-output projection | Disabled |

The image uses a conda environment (`/opt/env`) instead of NGC, has no Ray
binary, and requires the `mp` (SPMD) distributed backend.

---

## GB10 UMA Memory Patches

The aidendle94 image does not include the GB10 UMA memory accounting fix.
During profiling, the OS releases page cache, causing `current_free >
init_free`. This triggers two assertion failures in vLLM v1. Both are
bypassed at startup via `VLLM_SKIP_INIT_MEMORY_CHECK=1` plus two inline
patches applied by `entrypoint.unholy.sh`:

**Patch 1** — `vllm/v1/worker/utils.py` `request_memory()`:  
Pre-init free-memory check is skipped when `VLLM_SKIP_INIT_MEMORY_CHECK=1`.

**Patch 2** — `vllm/v1/worker/gpu_worker.py` `determine_available_memory()`:  
When `current_free > init_free`, returns `current_free` (~34 GiB) as the KV
cache budget instead of firing the assertion. This gives a safe KV allocation
without overestimating (which caused OOM in earlier patch iterations).

**Effective KV cache**: ~34 GiB (vs ~93 GiB on a system with accurate UMA
accounting). This limits multi-request performance at depth ≥ 32k.

---

## Configuration

See `.env.unholy-fusion` for the full variable set. Key parameters:

```
VLLM_IMAGE=aidendle94/sparkrun-vllm-ds4-gb10:production-ready
GPU_MEMORY_UTILIZATION=0.80
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=8192
MAX_MODEL_LEN=262144
MTP_NUM_TOKENS=1          # speculative decoding depth (1=best, see §MTP)
VLLM_USE_B12X_MOE=1
VLLM_USE_BREAKABLE_CUDAGRAPH=0
VLLM_SKIP_INIT_MEMORY_CHECK=1
NCCL_NET=IB
NCCL_CUMEM_ENABLE=0
NCCL_CROSS_NIC=1
NCCL_IGNORE_CPU_AFFINITY=1
VLLM_NCCL_SO_PATH=/opt/env/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
```

JIT caches are persisted via volume mounts:
- `./cache/unholy-hf:/cache/huggingface` — DeepGEMM, Triton, torch.compile
- `./cache/unholy-jit:/cache/jit` — vLLM compile cache root

With warm cache, model startup takes ~5 min (weight load ~60 s, profiling ~17 s).

---

## Full Depth Sweep — MTP n=1 (2026-06-05 08:46 KST)

`pp=2048, tg=128, runs=3, latency-mode=generation`

### Prompt Processing — pp2048 t/s (total)

All depths and concurrencies show stable ~1820–1970 t/s prefill throughput.
Depth and concurrency have negligible effect on pp performance.

### Token Generation — tg128 t/s (total)

| depth | c=1 | c=2 | c=4 | c=6 | c=8 |
|------:|----:|----:|----:|----:|----:|
| 0 | 39.56 | 56.82 | 67.37 | 62.97 | **73.79** |
| 4096 | 37.40 | 21.20 | 31.44 | 28.21 | 33.28 |
| 8192 | 36.70 | 27.99 | 21.94 | 20.86 | 24.36 |
| 16384 | 33.08 | 23.30 | 14.03 | 13.53 | 14.32 |
| 32768 | 35.89 | 13.34 | 8.17 | 7.82 | 7.22 |
| 65536 | 36.25 | 6.82 | 3.13 | 3.85 | 3.83 |

### Token Generation — tg128 peak t/s

| depth | c=1 | c=2 | c=4 | c=6 | c=8 |
|------:|----:|----:|----:|----:|----:|
| 0 | 42.33 | 71.33 | 107.67 | 110.33 | **170.67** |
| 4096 | 43.00 | 39.33 | 85.33 | 96.67 | 134.33 |
| 8192 | 40.00 | 61.33 | 84.33 | 95.00 | 137.67 |
| 16384 | 37.67 | 62.67 | 87.33 | 93.67 | 136.67 |
| 32768 | 40.00 | 65.33 | 84.67 | 93.67 | 122.67 |
| 65536 | 43.50 | 62.67 | 70.00 | 90.00 | 120.00 |

**Observations:**
- Single-request tg (c=1) is stable at 35–40 t/s regardless of depth — no
  KV pressure at low concurrency.
- At depth ≥ 32k with c ≥ 4, total throughput collapses to 3–8 t/s. This is
  directly caused by the 34 GiB KV cache limit: available blocks per request
  drop below what the scheduler needs to maintain c=4+ concurrency.
- Peak t/s at d=0/c=8 reaches **170.67 t/s**, consistent with the forum
  reference (aidendle94: ~167 t/s peak).

---

## MTP Depth Comparison — n=1 / n=2 / n=3 (depth=0 only)

`pp=2048, tg=128, depth=0, runs=3, latency-mode=generation`

### tg128 t/s (total)

| c | n=1 | n=2 | n=3 |
|--:|----:|----:|----:|
| 1 | 39.56 | 40.25 | 34.71 |
| 2 | 56.82 | 59.66 | 50.04 |
| 4 | **67.37** | ~~20.12~~ | 65.95 |
| 6 | 62.97 | ~~1.91~~ | **70.56** |
| 8 | **73.79** | ~~2.22~~ | 70.37 |

### tg128 peak t/s

| c | n=1 | n=2 | n=3 |
|--:|----:|----:|----:|
| 1 | 42.33 | 47.00 | 40.67 |
| 2 | 71.33 | 70.33 | 64.33 |
| 4 | 107.67 | 36.33 | 104.00 |
| 6 | 110.33 | 9.33 | **134.33** |
| 8 | **170.67** | 14.33 | 168.67 |

### Analysis

**n=2 catastrophic failure at c ≥ 4**: vLLM warns that `num_speculative_tokens > 1`
runs the same MTP layer multiple times, lowering the acceptance rate. For n=2
specifically, this appears to collapse acceptance to near-zero at c ≥ 4,
triggering cascading re-generation. The `Server disconnected` error at c=6
suggests the server became unresponsive under speculative pipeline pressure.

**n=3 recovery**: Despite more speculation depth, n=3 performs close to n=1
across all concurrency levels. At c=6 it outperforms n=1 by +12%
(70.56 vs 62.97 t/s). The CUDA graph capture size range expands to
`[1, 2, 4, 8, 16, 24, 32, 40, 48, 64]` for n=3, which may provide better
batch packing efficiency than n=2's intermediate range.

**Recommendation**: Use `MTP_NUM_TOKENS=1` (operational default). n=3 is
viable for medium-concurrency workloads (c=4–6) but offers no net gain at c=8.
n=2 must not be used with B12X_MOE.

---

## Comparison vs. jasl0603

| metric | jasl0603 | unholy-fusion n=1 | delta |
|--------|----------:|------------------:|------:|
| tg total @ d=0, c=8 | 61.67 t/s | 73.79 t/s | **+19.6%** |
| tg peak @ d=0, c=8 | — | 170.67 t/s | ≈ forum 167 t/s |
| pp @ d=0, c=8 | ~1100 t/s | ~1950 t/s | — |
| model load time | ~64 s | ~60 s | comparable |
| KV cache budget | ~34 GiB | ~34 GiB | equal (same UMA limit) |

jasl0603 uses the Ray backend with expert parallelism. unholy-fusion uses mp
backend without expert parallelism (`--enable-expert-parallel` is incompatible
with `VLLM_USE_B12X_MOE`).
