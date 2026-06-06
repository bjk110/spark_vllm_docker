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
When OS page-cache release during profiling causes `current_free > init_free`,
the patch returns `current_free` as the post-profile free UMA memory (~34 GiB)
rather than firing the assertion. vLLM then applies `GPU_MEMORY_UTILIZATION`
against this budget; the resulting effective KV cache allocation is approximately
**~17 GiB** (1,144,306 tokens at GPU_UTIL=0.80).

> The UMA patch may expose roughly ~34 GiB of post-profile free memory in the
> worker memory path, but the stable benchmark run reports an effective vLLM KV
> cache allocation around ~17 GiB. Treat free memory after UMA page-cache release
> and the vLLM-reported KV cache budget as separate metrics.

---

## Configuration

See `.env.unholy-fusion` for the full variable set. Key parameters:

```
VLLM_IMAGE=aidendle94/sparkrun-vllm-ds4-gb10:production-ready
GPU_MEMORY_UTILIZATION=0.80
MAX_NUM_SEQS=4                # confirmed upper limit; ≥5 causes startup hang
MAX_NUM_BATCHED_TOKENS=8192
MAX_MODEL_LEN=262144          # confirmed upper limit; 524288 crashes at d≥131072
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

## Full Depth Sweep — MTP n=1, MAX_NUM_SEQS=4 (2026-06-06 22:14 KST)

`pp=2048, tg=128, runs=3, latency-mode=generation`  
Config: MAX_NUM_SEQS=4, MAX_MODEL_LEN=262144, GPU_UTIL=0.80, MTP n=1, KV=17.1 GiB (1,144,306 tokens)

### Prompt Processing — pp2048 t/s (total)

| depth | c=1 | c=2 | c=4 | c=8 |
|------:|----:|----:|----:|----:|
| 0 | 1919 | 1958 | 1931 | 1140 |
| 4096 | 2051 | 1984 | 2003 | 1561 |
| 8192 | 1925 | 1987 | 2007 | 1723 |
| 16384 | 2003 | 1795 | 2012 | 1853 |
| 32768 | 1977 | 1986 | 1987 | 1915 |
| 65536 | 1927 | 1933 | 1939 | 1904 |
| **131072** | **1813** | **1821** | **1821** | **1820** |

Prefill is consistently ~1900–2050 t/s across all depths. At d=131072, throughput
dips slightly to ~1813–1821 t/s due to attention cost scaling with context length.

### Token Generation — tg128 t/s (total)

| depth | c=1 | c=2 | c=4 | c=8 |
|------:|----:|----:|----:|----:|
| 0 | 39.76 | 62.25 | 83.68 | **67.20** |
| 4096 | 37.16 | 38.95 | 35.04 | 32.66 |
| 8192 | 35.66 | 34.62 | 27.70 | 20.42 |
| 16384 | 39.51 | 24.41 | 17.58 | 13.97 |
| 32768 | 35.73 | 11.13 | 9.82 | 8.27 |
| 65536 | 41.05 | 5.49 | 5.26 | 4.11 |
| **131072** | **34.59** | **3.17** | **2.56** | **1.94** |

### Token Generation — tg128 peak t/s

| depth | c=1 | c=2 | c=4 | c=8 |
|------:|----:|----:|----:|----:|
| 0 | 43.00 | 71.33 | 114.00 | **112.33** |
| 4096 | 41.00 | 61.33 | 88.67 | 97.67 |
| 8192 | 39.67 | 52.67 | 95.33 | 96.33 |
| 16384 | 42.67 | 67.33 | 99.33 | 96.67 |
| 32768 | 39.67 | 56.67 | 98.33 | 100.33 |
| 65536 | 43.00 | 52.00 | 93.67 | 100.67 |
| **131072** | **39.00** | **58.33** | **95.33** | **95.67** |

**Key pattern at d=131072**:
- **c=1 total (34.59 t/s)**: single-stream decode is depth-immune — no collapse at 128k context.
- **c=8 total (1.94 t/s)**: sustained multi-stream collapses completely. Root cause is O(n)
  per-step attention cost at 131k tokens × 4 batched sequences (MAX_NUM_SEQS=4).
- **c=8 peak (95.67 t/s)**: burst hardware ceiling is preserved — the first decode steps
  remain fast. The gap between peak (~96 t/s) and total (~2 t/s) reflects that sustained
  throughput degrades as each sequence accumulates 131k+ KV history per step.

**Note**: MAX_NUM_SEQS=4 caps actual concurrency — c=8 requests are queued in batches
of 4, so c=8 total is lower than expected. Compare to 2026-06-05 run (MAX_NUM_SEQS=8):
c8 total d=0 was 73.79 t/s vs 67.20 t/s here.

---

## MAX_MODEL_LEN=524288 Depth Probe (2026-06-06, FAILED)

`pp=2048, tg=128, runs=3, latency-mode=generation`  
Config: MAX_NUM_SEQS=4, MAX_MODEL_LEN=**524288**, GPU_UTIL=0.80, MTP n=1

**Outcome: server crash at first d=131072 request.**

| depth | result | note |
|------:|--------|------|
| 0 | ✓ success | pp ~1800 t/s, tg c1 39 t/s, tg c4 65 t/s |
| 131072 | ✗ crash | `sample_tokens` RPC timeout → EngineCore fatal → connection refused |
| 196608 | ✗ crash | server already dead |
| 262144+ | ✗ crash | server already dead |

**Why**: With MAX_MODEL_LEN=524288, CUDA graph sizes roughly double. The decode step
at d=131072 context length now exceeds the EngineCore RPC timeout limit. RAM hits
120/121 GiB (UMA fully consumed). Reboot required for recovery.

**Conclusion**: MAX_MODEL_LEN=262144 is the confirmed stable upper bound.
Raising it to 524288 starts the engine but makes it operationally unusable at any
significant depth. The depth ceiling remains **d=131072** (128k tokens).

---

## Full Depth Sweep — MTP n=1 (2026-06-05 08:46 KST)

> **Note**: This run used `MAX_NUM_SEQS=8`, which is now confirmed to cause
> startup hang on GB10 UMA (CUDA graph capture stall). The data is preserved
> for reference but reflects an **experimental configuration** not recommended
> for operational use. c=6 and c=8 results here are not reproducible at the
> current safe limit of `MAX_NUM_SEQS=4`. Use the 2026-06-06 22:14 KST run
> for the current stable reference.

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
  directly caused by the ~17 GiB effective KV cache allocation: available blocks
  per request drop below what the scheduler needs to maintain c=4+ concurrency
  at long context (note: the ~34 GiB post-profile UMA free memory is not the
  same as the KV cache — see §GB10 UMA Memory Patches).
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

## jasl0603 Forum-Spec Results (2026-06-05, for comparison)

`pp=2048, tg=128, runs=3, latency-mode=generation`  
Config: MAX_NUM_SEQS=6, MAX_MODEL_LEN=1,000,000, GPU_UTIL=0.82, MTP n=2, Ray + expert-parallel  
KV cache: 11.94 GiB (spark01) / 11.99 GiB (spark02)

### Prefill — pp2048 t/s (total)

| depth   | c=1 | c=2 | c=4 | c=8 |
|--------:|----:|----:|----:|----:|
| d0      | 949 | 923 | 789 | 716 |
| d4096   | 887 | 915 | 913 | 804 |
| d8192   | 868 | 815 | 837 | 781 |
| d16384  | 839 | 833 | 852 | 789 |
| d32768  | 795 | 690 | 784 | 718 |
| d65536  | 731 | 601 | 639 | 624 |
| d131072 | 648 | 619 | 613 | 619 |

Prefill at d=0/c=1 peaks at 949 t/s. At higher concurrency (c=4/8), scheduler
contention with MTP re-generation lowers throughput to 716–789 t/s. The jasl0603
prefill ceiling is ~950 t/s for a single stream, compared to ~2030 t/s for unholy-fusion.

### Decode — tg128 t/s (total)

| depth   | c=1  | c=2  | c=4  | c=8  |
|--------:|-----:|-----:|-----:|-----:|
| d0      | 36.0 | 39.2 | 38.2 | 40.2 |
| d4096   | 36.2 | 25.9 | 19.5 | 17.6 |
| d8192   | 36.5 | 14.8 | 12.4 | 10.0 |
| d16384  | 32.5 | 12.1 | 7.5  | 5.6  |
| d32768  | 32.2 | 4.0  | 2.9  | 2.8  |
| d65536  | 60.8†| 1.9  | 1.5  | 1.3  |
| d131072 | 25.4 | 1.0  | 0.8  | 0.7  |

† d65536/c=1 shows 60.8 t/s with σ=43.44 — high variance indicates MTP n=2 occasionally
accelerates single-stream generation, but it is not reproducibly stable.

**MTP n=2 collapse pattern**: At c=1, throughput stays 25–40 t/s across all depths.
At c≥2, throughput halves at d=4096 and collapses to ≤2 t/s by d=32768. This is the
MTP n=2 acceptance-rate collapse under queued requests: the speculative pipeline
re-generates tokens repeatedly when multiple requests compete for the same KV blocks.

### Decode — tg128 peak t/s

| depth   | c=1   | c=2   | c=4   | c=8   |
|--------:|------:|------:|------:|------:|
| d0      | 41.7  | 49.7  | 90.3  | **116.3** |
| d4096   | 41.7  | 66.7  | 89.7  | 108.3 |
| d8192   | 41.7  | 66.7  | 89.3  | 114.0 |
| d16384  | 38.0  | 61.0  | 85.7  | 106.7 |
| d32768  | 38.3  | 56.7  | 67.3  | 66.0  |
| d65536  | 77.4† | 38.7  | 34.0  | 37.7  |
| d131072 | 30.7  | 32.3  | 32.3  | 31.7  |

Peak (best-of-3) at d≤16k remains strong (90–116 t/s at c=8), showing the hardware
can deliver high burst decode even with MTP n=2. At d≥32k, peak collapses as KV cache
(11.9 GiB) is exhausted — cannot hold 8 concurrent sequences × 32k+ context.

---

## Comparison vs. jasl0603

### Config differences

| param | jasl0603 | unholy-fusion |
|-------|----------|---------------|
| MAX_NUM_SEQS | 6 | 4 |
| MAX_MODEL_LEN | 1,000,000 | 262,144 |
| GPU_UTIL | 0.82 | 0.80 |
| MTP | n=2 | n=1 |
| backend | Ray + expert-parallel | mp (no expert-parallel) |
| KV cache | ~11.9 GiB | ~16.9 GiB |

### Prefill (pp2048 total, c=1 and c=8)

| depth | jasl c1 | unholy c1 | delta c1 | jasl c8 | unholy c8 | delta c8 |
|------:|--------:|----------:|---------:|--------:|----------:|---------:|
| d0      | 949 | 1919      | **+102%**| 716 | 1140      | **+59%** |
| d4096   | 887 | 2051      | **+131%**| 804 | 1561      | **+94%** |
| d8192   | 868 | 1925      | **+122%**| 781 | 1723      | **+121%**|
| d16384  | 839 | 2003      | **+139%**| 789 | 1853      | **+135%**|
| d32768  | 795 | 1977      | **+149%**| 718 | 1915      | **+167%**|
| d65536  | 731 | 1927      | **+164%**| 624 | 1904      | **+205%**|
| d131072 | 648 | 1813      | **+180%**| 619 | 1820      | **+194%**|

**unholy-fusion prefill is 2–3× faster at all depths and concurrencies.**  
Root cause: B12X custom MoE dispatcher + mp SPMD backend eliminates Ray actor
overhead on expert routing. jasl0603's expert-parallel mode adds per-token
routing synchronization that B12X_MOE bypasses with a custom GB10 dispatch path.

### Decode total (tg128 total t/s)

| depth  | jasl c1 | unholy c1 | jasl c4 | unholy c4 | jasl c8 | unholy c8 |
|-------:|--------:|----------:|--------:|----------:|--------:|----------:|
| d0      | 36.0  | 39.76 | 38.2  | **83.68** | 40.2  | 67.20 |
| d4096   | 36.2  | 37.16 | 19.5  | 35.04     | 17.6  | 32.66 |
| d8192   | 36.5  | 35.66 | 12.4  | 27.70     | 10.0  | 20.42 |
| d16384  | 32.5  | 39.51 | 7.5   | 17.58     | 5.6   | 13.97 |
| d32768  | 32.2  | 35.73 | 2.9   | 9.82      | 2.8   | 8.27  |
| d65536  | 60.8† | 41.05 | 1.5   | 5.26      | 1.3   | 4.11  |
| d131072 | 25.4  | **34.59** | 0.8 | 2.56    | 0.7   | 1.94  |

- **c=1**: Nearly identical (32–41 t/s). Single-stream decode is hardware-bound; stable
  even at d=131072 (jasl 25.4 t/s vs unholy 34.59 t/s — slight unholy edge from B12X attention).
- **c=4**: unholy-fusion leads at d=0 (+119%); gap narrows at depth but persists.
  At d=0, unholy-fusion serves all 4 concurrent requests (MAX_NUM_SEQS=4), while
  jasl0603 with MTP n=2 spends cycles on speculative re-generation.
- **c=8**: unholy-fusion leads (+67% at d=0), but advantage narrows with depth.
  At d=131072, unholy 1.94 vs jasl 0.7 t/s (+177%) — both near-zero, unholy slightly better.
- **d≥4k, c≥2**: jasl0603 collapses due to MTP n=2 acceptance-rate drop. This is not
  a hardware or KV-cache limit — single-stream (c=1) stays at 25–41 t/s even at d=131k.

### Decode peak (tg128 peak t/s, best-of-3 run)

| depth   | jasl c4 | unholy c4 | jasl c8 | unholy c8 |
|--------:|--------:|----------:|--------:|----------:|
| d0      | 90.3    | 114.00    | **116.3** | 112.33  |
| d4096   | 89.7    | 88.67     | 108.3   | 97.67     |
| d8192   | 89.3    | 95.33     | **114.0** | 96.33   |
| d16384  | 85.7    | 99.33     | **106.7** | 96.67   |
| d32768  | 67.3    | 98.33     | 66.0    | **100.33** |
| d65536  | 34.0    | 93.67     | 37.7    | **100.67** |
| d131072 | 32.3    | **95.33** | 31.7    | **95.67** |

Peak throughput (best single run) tells a different story from total:
- At **d≤16k**, both are similar (~90–116 t/s at c=8). jasl0603 occasionally edges
  ahead because a lucky run has high MTP acceptance → burst decode.
- At **d≥32k**, jasl0603 drops sharply (KV exhausted at 11.9 GiB with 1M context
  model; peak collapses to ~32–37 t/s). unholy-fusion maintains ~95–100 t/s peak
  because 16.9 GiB KV + 262k context provides more headroom per block.
- At **d=131072**: jasl peak 31.7 t/s vs unholy peak **95.67 t/s** — 3× difference.
  Both KV caches are sufficient for the depth, but unholy's B12X kernels compute
  attention at ~3× higher throughput in the first token burst.

### Summary

| metric | winner | magnitude | note |
|--------|--------|-----------|------|
| Prefill all depths | **unholy-fusion** | 2–3× | B12X MoE + mp backend |
| Decode total c=1, all depths | tie | <20% | hardware-bound single stream |
| Decode total c=4, d=0 | **unholy-fusion** | +119% | MTP n=2 overhead in jasl |
| Decode total c=8, d=0 | **unholy-fusion** | +67% | NUM_SEQS cap limits both |
| Decode total d≥4k, c≥2 | **unholy-fusion** | +100–300% | MTP n=2 collapse in jasl |
| Decode peak d≤16k | tie | <10% | both near hardware ceiling |
| Decode peak d≥32k | **unholy-fusion** | +40–200% | jasl KV exhausted (11.9 GiB) |
| Decode peak d=131072 | **unholy-fusion** | **3×** | 95 vs 32 t/s; B12X attention kernel |
| Long-context | **jasl0603** | 4× more | 1M vs 262k tokens |
| Operational stability | **jasl0603** | — | unholy hangs at NUM_SEQS>4 |

**Key takeaway**: unholy-fusion's B12X_MOE kernel delivers a massive prefill advantage
(~2×) and eliminates MTP n=2 throughput collapse at depth. The decode peak hardware
ceiling is similar (~115–120 t/s at c=8, d=0). The practical gap is: unholy-fusion
cannot run with MAX_NUM_SEQS>4 or MAX_MODEL_LEN>262k, making it unsuitable as a
drop-in for long-context or high-concurrency production workloads.

jasl0603 uses the Ray backend with expert parallelism. unholy-fusion uses mp
backend without expert parallelism (`--enable-expert-parallel` is incompatible
with `VLLM_USE_B12X_MOE`).

**Production-readiness**: unholy-fusion is valuable as an **experimental
high-prefill-performance** alternative. It should not be used as a general
production default. The recommended safe operational profile is single or
few long-context streams (`c=1–2`), not many concurrent long-context users.
Long-context concurrency (`d≥4k`, `c≥4`) collapses decode throughput due to
O(n) attention cost and scheduler queuing under `MAX_NUM_SEQS=4`. For
workloads requiring more than 262k context or sustained high concurrency,
jasl0603 remains the appropriate choice.

### Operational limits of unholy-fusion

| limit | confirmed safe | confirmed broken | failure mode |
|-------|---------------|-----------------|--------------|
| MAX_NUM_SEQS | ≤ 4 | ≥ 5 | startup hang (CUDA graph capture, `gpu_model_runner.py:6290`) |
| MAX_MODEL_LEN | ≤ 262144 | 524288 | starts OK but crashes at d≥131072 (`sample_tokens` RPC timeout) |
| MTP n | 1 (or 3) | 2 | n=2 catastrophic throughput collapse at c≥4 |
| VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS | must be 0 | unset/1 | `ValueError: invalid literal for int() with base 10: ''` at engine init |

**MAX_NUM_SEQS ≥ 5** — Worker_TP* stalls in `gpu_model_runner.py:6290` during CUDA
graph memory profiling. EngineCore shm_broadcast times out indefinitely. Tested:
NUM_SEQS=5, 6, 8 all hang. Root cause: MTP + B12X_MOE interaction with larger
graph capture sizes on GB10.

**MAX_MODEL_LEN = 524288 (2026-06-06 test)** — Starts successfully with NUM_SEQS=4;
KV cache allocates to 3,698,516 tokens (vs 1,144,306 with 262144). However, the first
request at d=131072 triggers `sample_tokens` RPC timeout → EngineCore fatal error →
server crash. Root cause: CUDA graph sizes scale with MAX_MODEL_LEN; at 524288 the
decode step at 131k context exceeds the RPC timeout threshold. RAM is also driven to
120/121 GiB (UMA fully consumed). Requires reboot to recover.

**Effective depth limit: d=131072 (128k tokens)** with MAX_MODEL_LEN=262144.
Depth 131072 + pp 2048 = 133120 total context, which fits within the 262144 limit.
