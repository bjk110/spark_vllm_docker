# feat/turboquant — TurboQuant 4-bit KV Cache Compression

Experimental branch for [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874) KV cache compression on vLLM + Blackwell (SM121).

Based on vLLM [PR #38280](https://github.com/vllm-project/vllm/pull/38280) (lishunyang12, OPEN) with Blackwell-specific optimizations.

## What This Branch Does

Compresses vLLM KV cache from bf16 to 4-bit via Walsh-Hadamard rotation + Lloyd-Max quantization. Model weights are **unchanged** — only the KV cache storage format changes.

## Docker Image

```bash
# Pull pre-built image from GHCR
docker pull ghcr.io/bjk110/vllm-spark:turboquant

# Or build from source
docker buildx build -f Dockerfile.gemma4 -t vllm-spark:turboquant --load .
```

### Quick Start

```bash
cp models/nemotron-120b-nvfp4-tq.env .env
sed -i 's|\[model_path\]|/path/to/models|' .env
docker compose --profile head up -d
```

Enable TurboQuant by adding `--kv-cache-dtype turboquant` to `VLLM_EXTRA_ARGS`:

```bash
VLLM_EXTRA_ARGS=--enable-chunked-prefill --kv-cache-dtype turboquant
```

## Software Stack

| Component | Version |
|---|---|
| Docker Image | `ghcr.io/bjk110/vllm-spark:turboquant` |
| Base Image | NGC PyTorch 26.03 |
| vLLM | 0.19.1 (commit a7d79fa, source build) |
| CUDA | 13.2 (native) |
| PyTorch | 2.11.0a0 |
| FlashInfer | v0.6.7 (SM121 source build) |
| Transformers | 5.5.0 |
| TurboQuant WPH ext | AOT SM121 (BLOCK_D=128/256, 2/4/8 warps) |

## Test Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GB10 (Blackwell, SM121) x2 DGX Spark |
| Memory | 121 GB unified (GPU+CPU shared) |
| Benchmark | [llama-benchy](https://github.com/eugr/llama-benchy) v0.3.4, pp2048 tg32 |

### Tested Models

| Model | Params | Quantization | head_dim | block_d | Architecture | Status |
|---|---|---|---|---|---|---|
| RedHatAI/Qwen3.5-122B-A10B-NVFP4 | 122B MoE | NVFP4 | 256 | 256 | Hybrid (Attn+Mamba+GDN) | **Verified** |
| NVIDIA/Nemotron-3-Super-120B-A12B-NVFP4 | 120B MoE | NVFP4 | 128 | 128 | Hybrid (Attn+Mamba+MoE) | **Verified** |
| google/gemma-4-31B-it | 31B Dense | BF16 | 256+512 | — | Heterogeneous head_dim | **Blocked** (vLLM forces TRITON_ATTN) |
| Models with head_dim=128 | — | Any | 128 | 128 | Any | Verified (unit tests) |
| Models with head_dim=256 | — | Any | 256 | 256 | Any | Verified (unit tests) |

## Optimization History

### Phase 1: Initial TurboQuant Integration

Applied PR #38280 as source patches on vLLM 0.19.1 (a7d79fa).

| Issue | Fix |
|---|---|
| PR targets vLLM base `8c0b626`, we use `a7d79fa` | Python patch script `apply_turboquant.py` adapts to code differences |
| `is_quantized_kv_cache` location mismatch | Patched in `torch_utils.py` instead of `backend.py` |
| Page-size unification failure (`NotImplementedError`) | `slot_bytes` padded to `_next_pow2()` for Mamba/GDN compatibility |

**Result**: TurboQuant serving operational. KV cache 155K → 413K tokens (+2.6x).

| Metric | bf16 KV (baseline) | TQ Initial |
|---|---|---|
| tg32 c=1 | 17.0 t/s | 14.2 t/s |
| tg32 c=4 | 55.2 t/s | 31.2 t/s |
| KV cache | 155K tokens | 413K tokens |

### Phase 2: Incremental Decode

Per-block dict cache — only dirty blocks re-decoded, rest reused.

| Issue | Fix |
|---|---|
| `.tolist()` / `.unique()` during CUDA graph capture | Deferred to `@torch.compiler.disable` scope |
| `is_current_stream_capturing()` needed for capture-safe branching | Full decode fallback during capture, incremental only in non-capture |

**Result**: c=4 improved 31.2 → 36.3 t/s (+16%).

### Phase 3: Gather-Free Triton Decode + Early Exit

Eliminated `cache[flat_bt]` full gather (~34MB/layer memcpy).

| Optimization | Description |
|---|---|
| Gather-free Triton kernel | `_fused_paged_decode_direct_kernel` reads paged cache via `cache_ptr + stride` |
| Norms-only gather | 2 bytes/slot instead of full slot_bytes (64x smaller) |
| `max_seq_len` early exit | Skips Hadamard butterfly for padding slots (block_size=4176 >> actual ~2080) |
| `_safe_view_fp16` | Handles odd `norm_offset=93` safely (byte-level assembly for odd offsets) |

**Result**: c=4 improved 36.3 → 39.9 t/s (+10%). Cumulative +28% vs initial.

### Phase 4: Profiling

Added `TQ_PROFILE=1` env var for per-layer decode/attention timing.

```
[TQ profile] decode=2.62ms attn=0.47ms total=3.10ms (decode 85%)
[TQ profile] decode=2.58ms attn=0.43ms total=3.01ms (decode 86%)
```

**Key finding**: Decode = 85–86% of total time. Hadamard butterfly is the dominant bottleneck.

### Phase 5: CUDA WPH Kernel (AOT, SM121)

Warp-shuffle butterfly — register-only, no shared memory, no barriers.

| Issue | Fix |
|---|---|
| JIT compile fails on SM121/aarch64 | AOT build in Dockerfile via `setup.py` |
| `at::cuda::getCurrentCUDAStream` → namespace error | Changed to `c10::cuda::getCurrentCUDAStream` |
| `norm_offset=93` odd alignment → `view(fp16)` crash | `_safe_view_fp16` byte-level assembly |
| `BLOCK_D=128` hardcoded, Qwen3.5 needs 256 | Template dispatch `<BLOCK_D>` with 128/256 |
| 1-warp CTA (32 threads) → low occupancy | 4-warp CTA (128 threads), `WARPS_PER_CTA` template |
| `cache[flat_bt]` gather → CUDA graph output pointer mismatch | Gather-free direct paged-cache read inside CUDA kernel |
| Serving garbage despite MSE=0 in round-trip test | Root cause: CUDA graph captures Triton path, replay writes to Triton output tensor, but Python returns WPH output tensor (different pointer). Fixed by making WPH also gather-free (same output allocation path as Triton). |

**Butterfly design (Plan A)**: 1 warp/head, `ELEMS_PER_THREAD = BLOCK_D/32`.

| BLOCK_D | EPT | Intra-thread levels | Warp shuffle levels |
|---|---|---|---|
| 128 | 4 | h=1,2 (2 levels) | h=4..64 (5 levels) |
| 256 | 8 | h=1,2,4 (3 levels) | h=8..128 (5 levels) |

**CTA size experiment**:

| CTA | c=1 | c=2 | c=4 | Verdict |
|---|---|---|---|---|
| 1 warp (32 threads) | 14.0 | 20.8 | 38.1 | Baseline |
| **4 warps (128 threads)** | **14.0** | **24.5** | **40.6** | **Best** |
| 8 warps (256 threads) | 13.8 | 21.8 | 34.3 | Too much register pressure |

### Final Results

| Metric | bf16 KV | TQ Triton | TQ WPH v2 (4-warp) |
|---|---|---|---|
| tg32 c=1 | 17.0 t/s | 14.1 t/s | 14.0 t/s |
| tg32 c=2 | 33.3 t/s | 23.5 t/s | **24.5 t/s** |
| tg32 c=4 | 55.2 t/s | 39.7 t/s | **40.6 t/s** |
| peak c=4 | — | — | **45.7 t/s** |
| TTFT c=1 | 984 ms | 1,068 ms | 1,046 ms |
| KV cache | 155K tokens | 405K tokens | 405K tokens |

#### Nemotron-H 120B-A12B NVFP4 (TP1, head_dim=128, 88 layers, hybrid Mamba+MoE)

| Metric | TQ Triton | TQ WPH v2 (4-warp) | WPH vs Triton |
|---|---|---|---|
| tg32 c=1 | 14.9 t/s | **15.2 t/s** | **+1.5%** |
| pp2048 c=1 | 1,387 t/s | **1,396 t/s** | +0.6% |
| TTFT c=1 | 1,628 ms | 1,631 ms | ~0% |
| KV cache | 1,548K tokens | 1,423K tokens | — |

Note: c>=2 benchmarks are unstable on this model due to 88-layer decode latency causing llama-benchy timeouts. c=1 results are stable and reproducible.

**Korean QA benchmark** (Qwen3.5): 12/12 pass (censorship, reading comprehension, math, hangul analysis, roleplay, common sense). No quality degradation observed.

### Regression Tests

6/6 PASS (`scripts/test_wph_v2.py`):

| Test Case | BLOCK_D | Outliers | Result |
|---|---|---|---|
| Qwen3.5 config (256, outliers, 2 heads) | 256 | ON | MSE=0.00 |
| Multi-token multi-head (256, 16tok, 4heads) | 256 | ON | MSE=0.00 |
| No outliers (256) | 256 | OFF | MSE=0.00 |
| head_dim=128 + outliers | 128 | ON | MSE=0.00 |
| head_dim=128 no outliers | 128 | OFF | MSE=0.00 |
| Stress (64 tokens, 4 heads, 256) | 256 | ON | MSE=0.00 |

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `TQ_CUDA_WPH` | `0` | `1` = CUDA warp-shuffle decode, `0` = Triton |
| `TQ_WPH_WARPS` | `4` | Warps per CTA (2/4/8). 4 optimal for SM121 |
| `TQ_PROFILE` | `0` | `1` = Per-layer decode/attention timing log |
| `TQ_NO_INCREMENTAL` | `0` | `1` = Force full decode every step (debug) |

## File Structure

```
patches/
├── apply_turboquant.py              # Patches 8 vLLM source files to register TurboQuant
├── turboquant_src/
│   └── vllm/
│       ├── model_executor/layers/quantization/
│       │   └── turboquant.py        # Config, codebook, Hadamard state (972 lines)
│       └── v1/attention/
│           ├── backends/
│           │   └── turboquant_attn.py   # Attention backend, encode/decode routing (900+ lines)
│           └── ops/
│               ├── triton_fused_turboquant.py      # Gather-free Triton encode/decode
│               ├── triton_hadamard_turboquant.py   # Hadamard Triton kernels
│               └── cuda_turboquant_decode.py       # CUDA WPH wrapper (AOT)
└── turboquant_ext/
    ├── turboquant_wph_kernel.cu     # CUDA WPH kernel (BLOCK_D=128/256, 2/4/8 warps)
    └── setup.py                     # AOT build for SM121
```

## Known Limitations

- **Throughput trade-off**: +2.6x KV cache at -18~26% decode throughput vs bf16. Best for long-context / high-concurrency.
- **block_size=4176**: Mamba/GDN page-size unification forces oversized blocks. Early exit mitigates partially.
- **slot_bytes padding**: 95→128 bytes (26% waste) due to power-of-2 page-size alignment. Could be reduced with vLLM core changes.
- **WPH not default**: Slightly faster than Triton at c>=2 but needs more production validation.
- **Incremental decode**: CUDA graph capture uses full decode; incremental only in non-capture path.
- **CUDA graph + large block_size**: Models with `block_size >= 8320` (e.g., Nemotron-H) crash at c>=2 due to graph replay memory access issues. Use `--enforce-eager` as workaround. Qwen3.5 (`block_size=4176`) is stable with graph.
