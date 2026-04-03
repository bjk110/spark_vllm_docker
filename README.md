# vLLM Spark — Unified Serving for DGX Spark (GB10)

**[한국어](README.ko.md)** | English

Unified vLLM serving configuration for NVIDIA DGX Spark dual-node cluster (GB10 x 2).
Supports multiple Qwen3.5 models with different quantizations via `.env` presets — one repo, one Dockerfile, one compose file.

## Hardware

| Node | Role | GPU | Memory | Interconnect |
|---|---|---|---|---|
| spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE |
| spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE |

## Software Stack

### v018-ngc2603 (latest, NGC 26.03)

Full source build on NGC 26.03. Native CUDA 13.2 eliminates the compat layer overhead from 26.01, yielding **+23% KV cache** (37.4 GiB vs 30.4 GiB). PyTorch 2.11 enables `_C_stable_libtorch` compilation — all NVFP4/FP8/CUTLASS ops are built in, so a single image covers every quantization format (FP8, NVFP4, INT4). Two Python patches (`hoist=True` removal, `__fx_repr__` dict fix) are applied at build time for PyTorch 2.11 compatibility.

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.03 |
| vLLM | 0.18.3 (main c494977, source build) |
| FlashInfer | v0.6.7 (CUTLASS 4.4.2, SM121 source build) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (native) |
| NCCL | 2.29.7 |
| Python | 3.12 |
| Transformers | 5.2.0 |
| `_C_stable_libtorch` | Included (NVFP4/FP8/CUTLASS full ops) |

### v018-fi067 (previous, NGC 26.01)

Attempted upgrade path: NGC 26.01 base with vLLM 0.18.1rc1 nightly wheel. CUDA 13.2 ran via compat layer on top of native 13.1, causing ~23% KV cache penalty. `_C_stable_libtorch` could not be compiled — PyTorch 2.10 failed the `stableivalue_conversions.h` static assertion (`const Tensor&` violates `trivially_copyable`). As a result, NVFP4 ops were missing and only FP8/INT4 serving was possible.

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.01 |
| vLLM | 0.18.1rc1 (nightly, cu130 wheel) |
| FlashInfer | v0.6.7 (SM121 source build) |
| PyTorch | 2.10.0a0 |
| CUDA | 13.1 + 13.2 compat |
| `_C_stable_libtorch` | Not included (wheel limitation) |

### v020-fi064 (legacy, NGC 26.01)

First working image for DGX Spark. Built around vLLM 0.17 nightly wheel with FlashInfer v0.6.1. Required a separate `Dockerfile.nvfp4` layer for NVFP4 support since the base wheel lacked those ops. Served as the production image for initial 397B INT4 and 122B NVFP4 benchmarks. Superseded by v018-ngc2603 which unifies all quantizations into a single image.

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.01 |
| vLLM | 0.17.0rc1.dev212 (nightly, cu130) |
| FlashInfer | v0.6.1 (SM121 build) |
| PyTorch | 2.10.0a0 |
| CUDA | 13.1 |

## Supported Models

| Preset | Model | Quantization | TP | Image |
|---|---|---|---|---|
| `qwen3.5-122b-fp8.env` | Qwen/Qwen3.5-122B-A10B-FP8 | FP8 (multimodal) | 2 | v018-ngc2603 |
| `redhatai-122b-nvfp4.env` | RedHatAI/Qwen3.5-122B-A10B-NVFP4 | NVFP4 (pre-quantized) | 1 | v018-ngc2603 |
| `intel-122b-int4.env` | Intel/Qwen3.5-122B-A10B-int4-AutoRound | INT4 AutoRound (Marlin) | 1 | v018-ngc2603 |
| `wangzhang-122b-fp8.env` | wangzhang/Qwen3.5-122B-A10B-abliterated | FP8 (text-only, abliterated) | 2 | v018-ngc2603 |
| `wangzhang-122b-nvfp4.env` | wangzhang/Qwen3.5-122B-A10B-abliterated-NVFP4 | NVFP4 (text-only, abliterated) | 1 | v018-ngc2603 |
| `qwen3.5-397b-int4.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 AutoRound (Marlin) | 2 | v018-ngc2603 |
| `qwen3.5-122b-nvfp4.env` | Qwen3.5-122B-A10B | NVFP4 (runtime) | 1 | v018-ngc2603 |
| `qwen3.5-122b-nvfp4-tp2.env` | Qwen3.5-122B-A10B | NVFP4 (runtime) | 2 | v018-ngc2603 |

## Quick Start

### 0. Get the Docker Image

#### Option A: Pull pre-built image from GHCR

```bash
# NGC 26.03 base image (FP8 / INT4 / NVFP4)
docker pull ghcr.io/bjk110/vllm-spark:v018-ngc2603
```

#### Option B: Build from source

```bash
# NGC 26.03 source build
docker buildx build -f Dockerfile.ngc2603-v3 \
  -t vllm-spark:v018-ngc2603 --load .
```

Build arguments:

| Argument | Default | Description |
|---|---|---|
| `BUILD_JOBS` | 16 | Parallel build jobs |
| `FLASHINFER_REF` | v0.6.7 | FlashInfer git ref |
| `VLLM_COMMIT` | c494977 | vLLM source commit |
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
├── Dockerfile                  # v020-fi064 base (NGC 26.01, legacy)
├── Dockerfile.nvfp4            # NVFP4 extension (legacy)
├── Dockerfile.ngc2603-v3       # v018-ngc2603 (NGC 26.03, latest)
├── models/                     # Validated model presets
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
│   ├── fix_pytorch211_compat.py  # hoist + __fx_repr__ fix
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
| `VLLM_IMAGE` | Pre-built Docker image | `vllm-spark:v018-ngc2603` |
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
| `fastsafetensors_natural_sort` | Multi-node weight loading order fix |
| `qwen3_5_moe_rope_fix` | RoPE validation fix for transformers 5.x |
| `aot_cache_fix` | torch.fx.Node pickling fix for AOT cache |
| `nogds_force` | Force `nogds=True` (GB10 has no GDS support) |
| `apply_sm121_patches` | `is_blackwell_class`, NVFP4 split, TRITON_PTXAS |
| `moe_config_e256/e512` | GB10-tuned MoE kernel configs |

## Benchmark Results

All benchmarks measured with [llama-benchy](https://github.com/eugr/llama-benchy) v0.3.4.

### 122B Models — Decode Throughput Comparison (t/s)

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
