# vLLM Spark — Unified Serving for DGX Spark (GB10)

**[한국어](README.ko.md)** | English

Unified vLLM serving configuration for NVIDIA DGX Spark dual-node cluster (GB10 x 2).
Supports multiple models (Qwen3.5, Gemma 4) with different quantizations via `.env` presets — one repo, one Dockerfile, one compose file.

## Hardware

| Node | Role | GPU | Memory | Interconnect |
|---|---|---|---|---|
| spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE |
| spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE |

## Software Stack

### v019-ngc2603 (latest, NGC 26.03)

Upgraded from v018 to vLLM 0.19.1 with Gemma 4 support, async scheduling, and CUDA 13.2 `cuMemcpyBatchAsync` compatibility patch. Transformers 5.5.0 enables native Gemma 4 architecture (MoE + Dense, multimodal, thinking mode). TTFT improved ~2x over v018 thanks to vLLM V1 engine optimizations. All Qwen3.5 quantization formats (FP8, NVFP4, INT4) remain fully supported.

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.03 |
| vLLM | 0.19.1 (main a7d79fa, source build) |
| FlashInfer | v0.6.7 (CUTLASS 4.4.2, SM121 source build) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (native) |
| NCCL | 2.29.7 |
| Python | 3.12 |
| Transformers | 5.5.0 |
| `_C_stable_libtorch` | Included (NVFP4/FP8/CUTLASS full ops) |

### v018-ngc2603 (previous, NGC 26.03)

First NGC 26.03 source build. Native CUDA 13.2 eliminated the compat layer overhead from 26.01, yielding **+23% KV cache**. PyTorch 2.11 enabled `_C_stable_libtorch` compilation — all NVFP4/FP8/CUTLASS ops built in a single image. Superseded by v019-ngc2603 which adds Gemma 4 support and improved TTFT.

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.03 |
| vLLM | 0.18.3 (main c494977, source build) |
| FlashInfer | v0.6.7 (SM121 source build) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (native) |
| Transformers | 5.2.0 |

## Supported Models

| Preset | Model | Quantization | TP | Image |
|---|---|---|---|---|
| `gemma4-26b-a4b.env` | google/gemma-4-26B-A4B-it | BF16 MoE (26B/4B active) | 1 | v019-ngc2603 |
| `gemma4-26b-a4b-tq.env` | google/gemma-4-26B-A4B-it | BF16 + **TurboQuant KV** | 1 | turboquant |
| `qwen3.5-122b-fp8.env` | Qwen/Qwen3.5-122B-A10B-FP8 | FP8 (multimodal) | 2 | v019-ngc2603 |
| `redhatai-122b-nvfp4.env` | RedHatAI/Qwen3.5-122B-A10B-NVFP4 | NVFP4 (pre-quantized) | 1 | v019-ngc2603 |
| `intel-122b-int4.env` | Intel/Qwen3.5-122B-A10B-int4-AutoRound | INT4 AutoRound (Marlin) | 1 | v019-ngc2603 |
| `wangzhang-122b-fp8.env` | wangzhang/Qwen3.5-122B-A10B-abliterated | FP8 (text-only, abliterated) | 2 | v019-ngc2603 |
| `wangzhang-122b-nvfp4.env` | wangzhang/Qwen3.5-122B-A10B-abliterated-NVFP4 | NVFP4 (text-only, abliterated) | 1 | v019-ngc2603 |
| `qwen3.5-397b-int4.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 AutoRound (Marlin) | 2 | v019-ngc2603 |
| `qwen3.5-122b-nvfp4.env` | Qwen3.5-122B-A10B | NVFP4 (runtime) | 1 | v019-ngc2603 |
| `qwen3.5-122b-nvfp4-tp2.env` | Qwen3.5-122B-A10B | NVFP4 (runtime) | 2 | v019-ngc2603 |

## Quick Start

### 0. Get the Docker Image

#### Option A: Pull pre-built image from GHCR

```bash
# NGC 26.03 + vLLM 0.19.1 (Gemma 4 + Qwen3.5, all quantizations)
docker pull ghcr.io/bjk110/vllm-spark:v019-ngc2603
```

#### Option B: Build from source

```bash
# NGC 26.03 source build (vLLM 0.19.1)
docker buildx build -f Dockerfile.gemma4 \
  -t vllm-spark:v019-ngc2603 --load .
```

Build arguments:

| Argument | Default | Description |
|---|---|---|
| `BUILD_JOBS` | 16 | Parallel build jobs |
| `FLASHINFER_REF` | v0.6.7 | FlashInfer git ref |
| `VLLM_COMMIT` | a7d79fa | vLLM source commit |
| `TORCH_CUDA_ARCH` | 12.1a | Target CUDA arch (Blackwell) |

### 1. Choose a Model Preset

```bash
cp models/qwen3.5-397b-int4.env .env
```

Edit `MODEL_PATH` in `.env` to point to your local model weights directory:

```bash
# Replace [model_path] with your actual path
sed -i 's|\[model_path\]|/home/user/models|' .env
```

### 2. Start Services

#### TP2 Multi-Node (e.g., 397B INT4)

```bash
# spark01 (head):
docker compose --profile head up -d

# spark02 (worker):
docker compose --profile worker up -d
```

The head node automatically waits for the worker to join the Ray cluster before launching vLLM.

#### TP1 Single-Node (e.g., NVFP4 122B)

```bash
cp models/qwen3.5-122b-nvfp4.env .env
docker compose --profile head up -d
```

When `TP_SIZE=1`, the entrypoint skips Ray entirely and runs `vllm serve` directly.

### 3. Verify

```bash
curl http://spark01:8000/health
```

## Architecture

```
spark01 (head)                    spark02 (worker)
┌─────────────────────┐          ┌─────────────────────┐
│  Ray Head (6379)    │          │  Ray Worker          │
│  vLLM API (:8000)   │◄────────►│                      │
│  GB10 GPU            │ 200Gbps │  GB10 GPU            │
│  TP rank 0           │  RoCE   │  TP rank 1           │
└─────────────────────┘          └─────────────────────┘
```

### How the Entrypoint Works

`entrypoint.sh` routes automatically based on `ROLE` and `TP_SIZE`:

| ROLE | TP_SIZE | Behavior |
|---|---|---|
| `head` | 1 | Direct `vllm serve` (no Ray) |
| `head` | 2+ | Ray head → wait for workers → `vllm serve --distributed-executor-backend ray` |
| `worker` | any | `ray start --block` (joins head) |

### Repository Structure

```
vllm-spark/
├── docker-compose.yml          # Unified compose (head + worker profiles)
├── entrypoint.sh               # Smart entrypoint (TP1/TP2 auto-routing)
├── .env.example                # Full configuration template
├── Dockerfile.gemma4           # v019-ngc2603 (NGC 26.03, latest)
├── Dockerfile.ngc2603-v3       # v018-ngc2603 (NGC 26.03, previous)
├── models/                     # Validated model presets
│   ├── gemma4-26b-a4b.env      # Gemma 4 26B MoE (TP1)
│   ├── gemma4-26b-a4b-tq.env   # Gemma 4 + TurboQuant KV
│   ├── redhatai-122b-nvfp4.env # RedHatAI NVFP4 (TP1)
│   └── ...                     # See models/ for full list
├── patches/                    # SM121 / PyTorch 2.11 compatibility
│   ├── apply_turboquant.py     # TurboQuant integration patch
│   ├── turboquant_src/         # TurboQuant source (PR #38280 + optimizations)
│   │   └── vllm/v1/attention/  # Triton + CUDA WPH decode kernels
│   ├── turboquant_ext/         # AOT CUDA WPH extension (SM121)
│   │   ├── turboquant_wph_kernel.cu
│   │   └── setup.py
│   └── ...                     # SM121 patches
└── scripts/
    ├── test_wph_v2.py          # WPH regression test (6 cases)
    ├── verify_imports.py       # Build/runtime verification
    └── ...
```

## Configuration

All configuration is via `.env`. See [`.env.example`](.env.example) for full documentation.

### Key Variables

| Variable | Description | Example |
|---|---|---|
| `VLLM_IMAGE` | Pre-built Docker image | `vllm-spark:v019-ngc2603` |
| `MODEL_PATH` | Host path to model weights | `/home/user/Models/Qwen/...` |
| `MODEL_CONTAINER_PATH` | Container mount point | `/models/Qwen3.5-397B-...` |
| `SERVED_MODEL_NAME` | API model name | `Qwen/Qwen3.5-397B-...` |
| `TP_SIZE` | Tensor parallel size (1=standalone, 2+=Ray) | `2` |
| `VLLM_EXTRA_ARGS` | Model-specific vllm serve flags | `--kv-cache-dtype fp8 --reasoning-parser qwen3` |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | Enable for INT4 AutoRound | `1` (or empty to disable) |

## Patches

The Dockerfile applies SM121 (Blackwell) compatibility patches:

| Patch | Purpose |
|---|---|
| `fix_cuda13_memcpy_batch` | `cuMemcpyBatchAsync` API fix for CUDA 13.0+ |
| `fastsafetensors_natural_sort` | Multi-node weight loading order fix |
| `qwen3_5_moe_rope_fix` | RoPE validation fix for transformers 5.x |
| `aot_cache_fix` | torch.fx.Node pickling fix for AOT cache |
| `nogds_force` | Force `nogds=True` (GB10 has no GDS support) |
| `apply_sm121_patches` | `is_blackwell_class`, NVFP4 split, TRITON_PTXAS |
| `moe_config_e256/e512` | GB10-tuned MoE kernel configs |
| `apply_turboquant` | TurboQuant KV cache 4-bit compression (PR #38280) |
| `turboquant_wph_ext` | AOT CUDA warp-shuffle decode kernel (SM121) |

## TurboQuant KV Cache Compression

Experimental 4-bit KV cache compression based on [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874) and vLLM [PR #38280](https://github.com/vllm-project/vllm/pull/38280).

Compresses the KV cache from bf16 to 4-bit via Walsh-Hadamard rotation + Lloyd-Max quantization, with outlier-aware channel splitting. Model weights are **unchanged** — only the KV cache storage format changes.

### How to Enable

Add `--kv-cache-dtype turboquant` to `VLLM_EXTRA_ARGS`:

```bash
VLLM_EXTRA_ARGS=--enable-chunked-prefill --kv-cache-dtype turboquant
```

Or use the TurboQuant preset:

```bash
cp models/gemma4-26b-a4b-tq.env .env
```

### What It Does

| Aspect | bf16 KV (default) | TurboQuant 4-bit KV |
|---|---|---|
| KV cache per token | 16 bits | ~4 bits + outlier bf16 |
| KV cache capacity | baseline | **+2.6x** (155K → 405K tokens) |
| Decode throughput | baseline | -18% (c=1), -26% (c=4) |
| Quality | lossless | near-lossless (4-bit quantization noise) |

### Implementation

Built from vLLM PR #38280 with the following additions for Blackwell/GB10:

| Feature | Description |
|---|---|
| **Gather-free Triton decode** | Reads directly from paged cache via `cache_ptr + strides` (no `cache[flat_bt]` copy) |
| **max_seq_len early exit** | Skips Hadamard butterfly for unused slots in oversized blocks |
| **CUDA WPH decode** | Warp-shuffle butterfly (no shared memory). AOT-compiled for SM121 |
| **BLOCK_D=128/256** | Template dispatch for different head dimensions |
| **4-warp CTA** | 128 threads/block for optimal occupancy on SM121 |
| **`_safe_view_fp16`** | Handles odd byte offsets (e.g. norm_offset=93) safely |
| **Page-size padding** | `slot_bytes` padded to next power-of-2 for Mamba page-size unification |
| **Incremental decode** | Reuses previously decoded blocks; only dirty blocks re-decoded |

### Decode Backend Selection

| Env Variable | Default | Description |
|---|---|---|
| `TQ_CUDA_WPH` | `0` | `1` = CUDA warp-shuffle decode, `0` = Triton |
| `TQ_WPH_WARPS` | `4` | Warps per CTA (2/4/8). 4 is optimal for SM121 |
| `TQ_PROFILE` | `0` | `1` = Log per-layer decode/attention timing |
| `TQ_NO_INCREMENTAL` | `0` | `1` = Disable incremental decode (debug) |

### Benchmark: TurboQuant (Qwen3.5-122B NVFP4 TP1, pp2048 tg32)

| Concurrency | bf16 KV | TQ Triton | TQ WPH (4-warp) |
|---|---|---|---|
| 1 | 17.0 t/s | 14.1 t/s | 14.0 t/s |
| 2 | 33.3 t/s | 23.5 t/s | **24.5 t/s** |
| 4 | 55.2 t/s | 39.7 t/s | **40.6 t/s** |
| KV cache | 155K tokens | 405K tokens | 405K tokens |
| TTFT c=1 | 984 ms | 1,068 ms | 1,046 ms |

### Supported Models

TurboQuant is independent of weight quantization and works with any model. Tested configurations:

| Model | head_dim | BLOCK_D | Status |
|---|---|---|---|
| Qwen3.5-122B-A10B (NVFP4/FP8/INT4) | 256 | 256 | Verified |
| Gemma 4 26B MoE | 256 | 256 | Compatible (untested) |
| Models with head_dim=128 | 128 | 128 | Verified (unit tests) |

### Caveats

- **Throughput trade-off**: TurboQuant increases KV cache capacity by 2.6x but reduces decode throughput by 18–26% compared to bf16 KV. Best suited for long-context or high-concurrency workloads where KV cache is the bottleneck.
- **Hybrid architecture**: Qwen3.5's Mamba/GDN layers force `block_size=4176` for page-size unification, causing decode overhead for short sequences. `max_seq_len` early exit mitigates this partially.
- **CUDA WPH status**: The CUDA warp-shuffle kernel (`TQ_CUDA_WPH=1`) is functional and slightly faster than Triton at c>=2, but is not the default. Set `TQ_CUDA_WPH=1` to enable.
- **Quality**: 4-bit quantization introduces small numerical noise. Round-trip MSE ≈ 0.11 (vs 0.0 for bf16). No observable quality degradation in Korean QA benchmarks (12/12 pass).

## Benchmark Results

All benchmarks measured with [llama-benchy](https://github.com/eugr/llama-benchy) v0.3.4.

### Gemma 4 — Single Node (TP1, BF16)

| Concurrency | 26B MoE (4B active) | 31B Dense |
|---|---|---|
| 1 | 25.0 (peak 26) | 4.0 (peak 5) |
| 2 | 45.9 (peak 49) | 7.9 (peak 8) |
| 4 | 67.2 (peak 77) | 14.1 (peak 17) |

| Metric | 26B MoE | 31B Dense |
|---|---|---|
| TTFT c=1 | 417 ms | 653 ms |
| KV cache | 224K tokens (51.3 GiB) | 77K tokens (35.2 GiB, FP8) |

### Qwen3.5 122B — Decode Throughput Comparison (t/s)

| Concurrency | FP8 TP2 (abliterated) | INT4 TP1 (Intel) | NVFP4 TP1 (abliterated) |
|---|---|---|---|
| 1 | 31.5 (peak 32.5) | 29.7 (peak 30) | 17.0 (peak 18) |
| 2 | 42.4 (peak 54) | 57.6 (peak 59) | 33.3 (peak 35) |
| 4 | 59.7 (peak 91) | 52.1 (peak 97) | 55.2 (peak 65) |

| Metric | FP8 TP2 | INT4 TP1 | NVFP4 TP1 |
|---|---|---|---|
| TTFT c=1 | 1,989 ms | 1,098 ms | 984 ms |
| KV cache | 839K tokens (38.5 GiB/node) | 789K tokens (36.2 GiB) | 155K tokens (14.3 GiB) |

### 397B INT4 TP2

#### Single Request (concurrency=1)

| Test | Throughput (t/s) | TTFT (ms) |
|---|---|---|
| pp512 | 967 ± 33 | 543 ± 25 |
| pp1024 | 1,349 ± 2 | 776 ± 2 |
| pp2048 | 1,704 ± 9 | 1,224 ± 7 |
| tg128 | 27.0 ± 0.1 | — |

#### Concurrent Requests — Total Decode Throughput (t/s)

| Concurrency | tg128 total | tg128 peak |
|---|---|---|
| 1 | 27.0 | 28 |
| 2 | 45.3 | 52 |
| 4 | 60~67 | 85~88 |
| 8 | 59~91 | 152~160 |

## System Tuning

Recommended OS-level settings for DGX Spark:

```bash
# Reduce swap pressure (unified memory)
sudo sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## License

Configuration files are provided as-is for reference. Models are subject to their respective licenses ([Qwen License](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)).
