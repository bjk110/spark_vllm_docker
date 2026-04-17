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

### v020-ngc2603 (latest, NGC 26.03)

Major update: vLLM main with **upstream TurboQuant KV cache compression** (PR #38479), FlashInfer v0.6.8 with SM121/GB10 optimizations (NVFP4 group GEMM, tile filtering, FP4 CUTLASS). Three upstream patches removed (cuMemcpyBatch, RoPE fix, PR #38423 — all merged). TurboQuant enables 2-4x KV cache capacity via `--kv-cache-dtype turboquant_k8v4`.

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.03 |
| vLLM | 0.20.0.dev (main 978a4462, source build, TurboQuant included) |
| FlashInfer | v0.6.8 (SM121 tile filtering, NVFP4 group GEMM, source build) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (native) |
| NCCL | 2.29.7 |
| Python | 3.12 |
| Transformers | 5.5.4 |
| `_C_stable_libtorch` | Included (NVFP4/FP8/CUTLASS full ops) |

### v019-ngc2603 (previous, NGC 26.03)

vLLM 0.19.1 with Gemma 4 support, async scheduling. Transformers 5.5.0. TTFT improved ~2x over v018. Superseded by v020-ngc2603 which adds TurboQuant and FlashInfer v0.6.8.

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.03 |
| vLLM | 0.19.1 (main a7d79fa, source build) |
| FlashInfer | v0.6.7.post3 (CUTLASS 4.4.2, SM121 source build) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (native) |
| Transformers | 5.5.0 |

## Supported Models

| Preset | Model | Quantization | TP | Image |
|---|---|---|---|---|
| `gemma4-26b-a4b.env` | google/gemma-4-26B-A4B-it | BF16 MoE (26B/4B active) | 1 | v020-ngc2603 |
| `qwen3.5-122b-fp8.env` | Qwen/Qwen3.5-122B-A10B-FP8 | FP8 (multimodal) | 2 | v020-ngc2603 |
| `redhatai-122b-nvfp4.env` | RedHatAI/Qwen3.5-122B-A10B-NVFP4 | NVFP4 (pre-quantized) | 1 | v020-ngc2603 |
| `intel-122b-int4.env` | Intel/Qwen3.5-122B-A10B-int4-AutoRound | INT4 AutoRound (Marlin) | 1 | v020-ngc2603 |
| `wangzhang-122b-fp8.env` | wangzhang/Qwen3.5-122B-A10B-abliterated | FP8 (text-only, abliterated) | 2 | v020-ngc2603 |
| `wangzhang-122b-nvfp4.env` | wangzhang/Qwen3.5-122B-A10B-abliterated-NVFP4 | NVFP4 (text-only, abliterated) | 1 | v020-ngc2603 |
| `qwen3.5-397b-int4.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 AutoRound (Marlin) | 2 | v020-ngc2603 |
| `qwen3.5-122b-nvfp4.env` | Qwen3.5-122B-A10B | NVFP4 (runtime) | 1 | v020-ngc2603 |
| `qwen3.5-122b-nvfp4-tp2.env` | Qwen3.5-122B-A10B | NVFP4 (runtime) | 2 | v020-ngc2603 |

## Quick Start

### 0. Get the Docker Image

#### Option A: Pull pre-built image from GHCR

```bash
# NGC 26.03 + vLLM 0.20.0.dev (TurboQuant + Gemma 4 + Qwen3.5)
docker pull ghcr.io/bjk110/vllm-spark:v020-ngc2603
```

#### Option B: Build from source

```bash
# NGC 26.03 source build (vLLM main, TurboQuant included)
docker buildx build -f Dockerfile.gemma4 \
  -t vllm-spark:v020-ngc2603 --load .
```

Build arguments:

| Argument | Default | Description |
|---|---|---|
| `BUILD_JOBS` | 16 | Parallel build jobs |
| `FLASHINFER_REF` | v0.6.8 | FlashInfer git ref |
| `VLLM_COMMIT` | 978a4462 | vLLM source commit |
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
├── Dockerfile.gemma4           # v020-ngc2603 (NGC 26.03, latest)
├── Dockerfile.ngc2603-v3       # v018-ngc2603 (NGC 26.03, archived)
├── models/                     # Validated model presets
│   ├── gemma4-26b-a4b.env      # Gemma 4 26B MoE (TP1)
│   ├── redhatai-122b-nvfp4.env # RedHatAI NVFP4 (TP1)
│   ├── intel-122b-int4.env     # Intel INT4 AutoRound (TP1)
│   ├── wangzhang-122b-fp8.env  # abliterated FP8 (TP2)
│   ├── wangzhang-122b-nvfp4.env # abliterated NVFP4 (TP1)
│   ├── qwen3.5-397b-int4.env   # 397B INT4 (TP2)
│   ├── qwen3.5-122b-fp8.env
│   ├── qwen3.5-122b-nvfp4.env
│   └── qwen3.5-122b-nvfp4-tp2.env
├── benchmarks/                 # llama-benchy benchmark results
│   ├── results_intel-int4-tp1.json
│   ├── results_wangzhang-fp8-tp2.json
│   └── results_wangzhang-nvfp4-tp1.json
├── patches/                    # SM121 / PyTorch 2.11 compatibility
│   ├── fix_pytorch211_compat.py  # hoist=True removal (PyTorch 2.11)
│   └── ...
└── scripts/
    ├── run-cluster-node.sh     # Manual Ray cluster bootstrap
    ├── verify_imports.py       # Build/runtime verification
    └── verify_runtime.sh       # Full GPU verification
```

## Configuration

All configuration is via `.env`. See [`.env.example`](.env.example) for full documentation.

### Key Variables

| Variable | Description | Example |
|---|---|---|
| `VLLM_IMAGE` | Docker image (local or GHCR) | `ghcr.io/bjk110/vllm-spark:v020-ngc2603` |
| `MODEL_PATH` | Host path to model weights | `/home/user/Models/Qwen/...` |
| `MODEL_CONTAINER_PATH` | Container mount point | `/models/Qwen3.5-397B-...` |
| `SERVED_MODEL_NAME` | API model name | `Qwen/Qwen3.5-397B-...` |
| `TP_SIZE` | Tensor parallel size (1=standalone, 2+=Ray) | `2` |
| `VLLM_EXTRA_ARGS` | Model-specific vllm serve flags | `--kv-cache-dtype fp8 --reasoning-parser qwen3` |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | Enable for INT4 AutoRound | `1` (or empty to disable) |

## Patches

The Dockerfile applies SM121 (Blackwell) compatibility patches:

| Patch | Purpose | Status |
|---|---|---|
| `fix_pytorch211_compat` | `hoist=True` removal for PyTorch 2.11 | Active |
| `fastsafetensors_natural_sort` | Multi-node weight loading order fix | Active |
| `aot_cache_fix` | torch.fx.Node pickling fix for AOT cache | Active |
| `nogds_force` | Force `nogds=True` (GB10 has no GDS support) | Active |
| `apply_sm121_patches` | `is_blackwell_class`, NVFP4 split, TRITON_PTXAS | Active |
| `moe_config_e256/e512` | GB10-tuned MoE kernel configs | Active |
| ~~`fix_cuda13_memcpy_batch`~~ | `cuMemcpyBatchAsync` API fix | Removed (upstream) |
| ~~`qwen3_5_moe_rope_fix`~~ | RoPE validation fix | Removed (upstream) |
| ~~`pr38423_nvfp4_spark`~~ | NVFP4 DGX Spark fixes | Removed (upstream) |

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
