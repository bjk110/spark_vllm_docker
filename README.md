# vLLM Spark — Unified Serving for DGX Spark (GB10)

**[한국어](README.ko.md)** | English

Unified vLLM serving configuration for NVIDIA DGX Spark (GB10), supporting two
topologies from the same repo / Dockerfile / compose file:

- **Single Spark** (default, zero RDMA setup) — one GB10 box, TP=1.
- **Dual Spark + 200 Gbps RoCE/IB** — two GB10 boxes, Ray, TP=2.

Pick the topology by setting `CLUSTER_MODE=single` (default) or
`CLUSTER_MODE=dual-rdma` in your `.env`. See [`Quick Start`](#quick-start) below.

For release-by-release detail and patch-by-patch status, see
[`CHANGELOG.md`](CHANGELOG.md) and [`PATCH_STATUS.md`](PATCH_STATUS.md).

## Hardware

| Topology | Node | Role | GPU | Memory | Interconnect |
|---|---|---|---|---|---|
| single | one Spark | vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB unified | n/a |
| dual-rdma | spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE |
| dual-rdma | spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE |

## Software Stack

### v021-ngc2603 (latest, NGC 26.03)

vLLM main bumped from 978a4462 to **95995bbe** (+236 commits incl. upstream merges of TQ backend selection #40060, FA3/FA4 prefill #40092, prior-art random-signs cleanup #40194). FlashInfer bumped **v0.6.8 → v0.6.9** with SM121 b12x FP4 GEMM (#3113) and b12x CuTe DSL fused MoE for SM120 (#3066). TurboQuant enables 2-4x KV cache capacity via `--kv-cache-dtype turboquant_k8v4`.

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.03 |
| vLLM | 0.20.0.dev (main 95995bbe, source build, TurboQuant included) |
| FlashInfer | v0.6.9 (SM121 b12x FP4 GEMM, b12x CuTe DSL MoE, source build) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (native) |
| NCCL | 2.29.7 |
| Python | 3.12 |
| Transformers | 5.5.4 |
| `_C_stable_libtorch` | Included (NVFP4/FP8/CUTLASS full ops) |

### v019-ngc2603 (previous, NGC 26.03)

vLLM 0.19.1 with Gemma 4 support, async scheduling. Transformers 5.5.0. TTFT improved ~2x over v018. Superseded by v021-ngc2603 (vLLM main 95995bbe + TurboQuant + FlashInfer v0.6.9).

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.03 |
| vLLM | 0.19.1 (main a7d79fa, source build) |
| FlashInfer | v0.6.7.post3 (CUTLASS 4.4.2, SM121 source build) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (native) |
| Transformers | 5.5.0 |

## Supported Models

| Preset | Model | Quantization / dtype | Topology | TP | Image | Notes |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b.env` | google/gemma-4-26B-A4B-it | BF16 MoE (26B/4B active) | single | 1 | v021-ngc2603 | — |
| `gemma4-26b-a4b-tq.env` | google/gemma-4-26B-A4B-it | BF16 MoE + **TurboQuant KV** (`turboquant_k8v4`) | single | 1 | v021-tq | TQ baked-in |
| `qwen3.5-122b-fp8.env` | Qwen/Qwen3.5-122B-A10B-FP8 | FP8 (multimodal) | dual-rdma | 2 | v021-ngc2603 | — |
| `qwen3.5-122b-nvfp4.env` | Qwen/Qwen3.5-122B-A10B | NVFP4 (runtime, FlashInfer) | single | 1 | v021-ngc2603 | — |
| `qwen3.5-122b-nvfp4-tp2.env` | Qwen/Qwen3.5-122B-A10B | NVFP4 (runtime, FlashInfer) | dual-rdma | 2 | v021-ngc2603 | — |
| `qwen3.5-122b-prismaquant.env` | rdtand/Qwen3.5-122B-A10B-PrismaQuant-4.75bit-vllm | PrismaQuant 4.76bpp (NVFP4+MXFP8+BF16 mixed) | single | 1 | v021-ngc2603 | MTP `n=1` default |
| `redhatai-122b-nvfp4.env` | RedHatAI/Qwen3.5-122B-A10B-NVFP4 | NVFP4 (pre-quantized) | single | 1 | v021-ngc2603 | — |
| `redhatai-122b-nvfp4-tq.env` | RedHatAI/Qwen3.5-122B-A10B-NVFP4 | NVFP4 + **TurboQuant KV** | single | 1 | v021-tq | TQ baked-in |
| `intel-122b-int4.env` | Intel/Qwen3.5-122B-A10B-int4-AutoRound | INT4 AutoRound (Marlin) | single | 1 | v021-ngc2603 | — |
| `wangzhang-122b-fp8.env` | wangzhang/Qwen3.5-122B-A10B-abliterated | FP8 (text-only, abliterated) | dual-rdma | 2 | v021-ngc2603 | `APPLY_TEXT_ONLY_SHIM=1` |
| `wangzhang-122b-nvfp4.env` | wangzhang/Qwen3.5-122B-A10B-abliterated-NVFP4 | NVFP4 (text-only, abliterated) | single | 1 | v021-ngc2603 | `APPLY_TEXT_ONLY_SHIM=1` |
| `qwen3.5-397b-int4.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 AutoRound (Marlin) | dual-rdma | 2 | v021-ngc2603 | — |
| `qwen3.5-397b-int4-tq.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 AutoRound + **TurboQuant KV** (`turboquant_3bit_nc` cascade) | dual-rdma | 2 | v021-tq | TQ baked-in; uses `--compilation-config {"use_inductor_graph_partition":true}` |
| `qwen3.6-35b-fp16.env` ⚗️ | Qwen/Qwen3.6-35B-A3B | **FP16 original** (KV fp8) | single | 1 | v021-ngc2603 | Experimental |

## Quick Start

### 0. Get the Docker Image

#### Option A: Pull pre-built image from GHCR

```bash
# Base image (all models, no TQ patches)
docker pull ghcr.io/bjk110/vllm-spark:v021-ngc2603

# TurboQuant image (base + upstream TQ bugfix patches for hybrid models)
docker pull ghcr.io/bjk110/vllm-spark:v021-tq
```

#### Option B: Build from source

```bash
# NGC 26.03 source build (vLLM main, TurboQuant included)
docker buildx build -f Dockerfile.gemma4 \
  -t vllm-spark:v021-ngc2603 --load .
```

Build arguments:

| Argument | Default | Description |
|---|---|---|
| `BUILD_JOBS` | 16 | Parallel build jobs |
| `FLASHINFER_REF` | v0.6.9 | FlashInfer git ref |
| `VLLM_COMMIT` | 95995bbe | vLLM source commit |
| `TORCH_CUDA_ARCH` | 12.1a | Target CUDA arch (Blackwell) |

### 1. Choose a Model Preset

Single-Spark presets ship with `CLUSTER_MODE=single` and TP=1 (zero RDMA setup).
Dual-Spark presets ship with `CLUSTER_MODE=dual-rdma` and TP=2.

```bash
# Single Spark (no RDMA needed):
cp models/redhatai-122b-nvfp4.env .env

# Dual Spark + RoCE:
cp models/qwen3.5-397b-int4.env .env
```

Edit `MODEL_PATH` in `.env` to point to your local model weights directory:

```bash
sed -i 's|\[model_path\]|/home/user/models|' .env
```

### 2. Start Services

#### Single Spark — TP=1 (default, no Ray, no RDMA)

```bash
docker compose --profile head up -d
docker logs -f vllm-spark-head
```

`entrypoint.sh` reads `CLUSTER_MODE=single` and **forces** `VLLM_HOST_IP=127.0.0.1`,
unsets `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `UCX_NET_DEVICES` / `NCCL_IB_HCA`,
and sets `NCCL_IB_DISABLE=1`. This is what avoids the
`tcp://10.10.10.1:<port> server socket has timed out` c10d hang
(see [Troubleshooting](#troubleshooting) below).

#### Dual Spark — TP=2 (Ray + RoCE)

```bash
# spark01 (head):
docker compose --profile head up -d

# spark02 (worker):
docker compose --profile worker up -d
```

The head waits for the worker to join the Ray cluster, then launches vLLM
with `--distributed-executor-backend ray`. Requires `HEAD_ROCE_IP`,
`WORKER_ROCE_IP`, `ROCE_IF_NAME`, `IB_HCA_NAME`, `RAY_PORT` in `.env`
(uncomment the block in any preset shipped with `CLUSTER_MODE=dual-rdma`).

### 3. Verify

```bash
curl http://localhost:8000/health      # single
curl http://spark01:8000/health        # dual-rdma
```

## Troubleshooting

### `SyntaxError: invalid syntax` in `compilation/codegen.py` during EngineCore init (Qwen3.5 hybrid + torch.compile)

```
File "<string>", line 5
    gdn_attention_core = torch.ops.vllm.gdn_attention_core(..., <vllm.utils.torch_utils.LayerName object at 0x...>)
                                                                ^
SyntaxError: invalid syntax
```

vLLM main since `951dca80` (PR #38657 "[compile] Invoke split FX graph by codegen") emits the default `repr()` of opaque arguments like `LayerName` into the generated execution function source. The hybrid GDN attention path used by Qwen3.5 takes a `LayerName` and trips this every cold start.

**Workaround (recommended, no source patch)** — pass `use_inductor_graph_partition=True` so torch.compile uses Inductor's own partitioning instead of vLLM's split-by-codegen:

```
VLLM_EXTRA_ARGS=... --compilation-config {"use_inductor_graph_partition":true}
```

This keeps torch.compile + CUDAGraph (`FULL_AND_PIECEWISE`) enabled. Cold-start engine init is roughly 2× longer (≈ 440 s vs 250 s for `--enforce-eager` on 397B INT4 TP=2) due to the extra Inductor compile, but steady-state inference benefits from CUDA graph capture.

**Last-resort workaround** — `--enforce-eager`. Disables torch.compile and CUDAGraph entirely; loses inference performance but guaranteed to bypass the codegen path.

**Hot-patch (kept on standby)** — `patches/patch_codegen_fx_repr.py` rewrites `_node_ref()` to honor `__fx_repr__()` and merges its namespace into the `exec()` scope. Apply only if a future vLLM bump regresses the Inductor partition path or a different opaque type triggers the same SyntaxError:

```bash
docker exec vllm-spark-head python3 /patches/patch_codegen_fx_repr.py
docker compose --profile head restart
```

### `[c10d] The server socket on [::ffff:10.10.10.1]:<port> has timed out, will retry.`

This means a single-Spark setup is leaking RDMA env (`VLLM_HOST_IP=10.10.10.1`,
`GLOO_SOCKET_IFNAME=enp1s0f0np0`, etc.) into the container, and PyTorch c10d
can't bind to a RoCE IP that doesn't exist on this host. Fix:

1. Confirm `.env` (or the preset you copied) has `CLUSTER_MODE=single`.
2. Make sure the `HEAD_ROCE_IP=…` / `ROCE_IF_NAME=…` / `IB_HCA_NAME=…` lines
   are **commented out** (lines starting with `#`) in single-mode presets.
3. Recreate the container:
   ```bash
   docker compose --profile head down
   docker compose --profile head up -d
   ```
   `entrypoint.sh` will print
   `CLUSTER_MODE=single: VLLM_HOST_IP=127.0.0.1, NCCL_IB_DISABLE=1, NCCL/GLOO/UCX ifname cleared`
   on a clean single-Spark start.

## Architecture

### Single Spark (CLUSTER_MODE=single, TP=1)

```
single Spark
┌──────────────────────────────────┐
│  vLLM API (:8000)                │
│  c10d binds 127.0.0.1            │
│  GB10 GPU, TP rank 0             │
└──────────────────────────────────┘
```

### Dual Spark (CLUSTER_MODE=dual-rdma, TP=2)

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

`entrypoint.sh` normalizes the environment based on `CLUSTER_MODE`, then
dispatches on `ROLE` × `TP_SIZE`:

| CLUSTER_MODE | ROLE | TP_SIZE | Behavior |
|---|---|---|---|
| `single`     | `head`   | 1   | Force `VLLM_HOST_IP=127.0.0.1`, clear NCCL/GLOO/UCX ifname, set `NCCL_IB_DISABLE=1`, then direct `vllm serve` (no Ray) |
| `single`     | `head`   | 2+  | Fail-fast (`single` cannot host TP≥2) |
| `single`     | `worker` | any | Fail-fast (worker is meaningless in single mode) |
| `dual-rdma`  | `head`   | 1   | Reject (use `single` for TP=1) |
| `dual-rdma`  | `head`   | 2+  | Validate RDMA env → Ray head → wait for workers → `vllm serve --distributed-executor-backend ray` |
| `dual-rdma`  | `worker` | any | `ray start --address=$HEAD_ROCE_IP:$RAY_PORT --block` |

### Repository Structure

```
vllm-spark/
├── docker-compose.yml             # Unified compose (head + worker profiles)
├── entrypoint.sh                  # CLUSTER_MODE-aware entrypoint
├── .env.example                   # Full configuration template
├── Dockerfile.gemma4              # v021-ngc2603 unified build (historical name)
├── Dockerfile.ngc2603-v3          # v018-ngc2603 archived build
├── Dockerfile.nvfp4               # NVFP4 runtime defaults overlay
├── CHANGELOG.md                   # Release-by-release history
├── PATCH_STATUS.md                # Per-patch purpose / status / removal condition
├── models/                        # Validated model presets
│   ├── gemma4-26b-a4b.env             # Gemma 4 26B MoE (single, TP1)
│   ├── gemma4-26b-a4b-tq.env          # Gemma 4 + TurboQuant KV (single, TP1)
│   ├── redhatai-122b-nvfp4.env        # RedHatAI NVFP4 (single, TP1)
│   ├── redhatai-122b-nvfp4-tq.env     # RedHatAI NVFP4 + TurboQuant (single, TP1)
│   ├── intel-122b-int4.env            # Intel INT4 AutoRound (single, TP1)
│   ├── wangzhang-122b-fp8.env         # abliterated FP8 (dual-rdma, TP2)
│   ├── wangzhang-122b-nvfp4.env       # abliterated NVFP4 (single, TP1)
│   ├── qwen3.5-397b-int4.env          # 397B INT4 (dual-rdma, TP2)
│   ├── qwen3.5-397b-int4-tq.env       # 397B INT4 + TurboQuant (dual-rdma, TP2)
│   ├── qwen3.5-122b-fp8.env           # 122B FP8 multimodal (dual-rdma, TP2)
│   ├── qwen3.5-122b-nvfp4.env         # 122B NVFP4 runtime (single, TP1)
│   ├── qwen3.5-122b-nvfp4-tp2.env     # 122B NVFP4 runtime (dual-rdma, TP2)
│   ├── qwen3.5-122b-prismaquant.env   # PrismaQuant 4.76bpp mixed (single, TP1)
│   └── qwen3.6-35b-fp16.env           # ⚗️ Qwen3.6 FP16 experimental (single, TP1)
├── benchmarks/                    # llama-benchy benchmark results
├── patches/                       # SM121 / PyTorch 2.11 / TurboQuant patches
│   ├── fix_pytorch211_compat.py       # hoist=True removal (PyTorch 2.11)
│   ├── fastsafetensors_natural_sort.patch
│   ├── aot_cache_fix.patch
│   ├── nogds_force.patch
│   ├── apply_sm121_patches.py
│   ├── moe_config_e256.json / moe_config_e512.json
│   ├── apply_turboquant_fixes.py      # v021-tq only
│   ├── patch_qwen35_moe_text.py       # APPLY_TEXT_ONLY_SHIM=1 only
│   ├── patch_codegen_fx_repr.py       # On-standby hot-patch (see Troubleshooting)
│   └── ...                            # See PATCH_STATUS.md for the full inventory
└── scripts/
    ├── run-cluster-node.sh        # Manual Ray cluster bootstrap
    ├── verify_imports.py          # Build-time import verification
    └── verify_runtime.sh          # Full GPU verification
```

## Configuration

All configuration is via `.env`. See [`.env.example`](.env.example) for full documentation.

### Key Variables

| Variable | Description | Example |
|---|---|---|
| `VLLM_IMAGE` | Docker image (local or GHCR) | `ghcr.io/bjk110/vllm-spark:v021-ngc2603` |
| `MODEL_PATH` | Host path to model weights | `/home/user/Models/Qwen/...` |
| `MODEL_CONTAINER_PATH` | Container mount point | `/models/Qwen3.5-397B-...` |
| `SERVED_MODEL_NAME` | API model name | `Qwen/Qwen3.5-397B-...` |
| `CLUSTER_MODE` | Topology: `single` (default) or `dual-rdma` | `single` |
| `TP_SIZE` | Tensor parallel size (1=single, 2+=dual-rdma) | `1` |
| `HEAD_ROCE_IP` | (`dual-rdma` only) head node RoCE IP | `10.10.10.1` |
| `WORKER_ROCE_IP` | (`dual-rdma` only) worker node RoCE IP | `10.10.10.2` |
| `ROCE_IF_NAME` | (`dual-rdma` only) RoCE interface name | `enp1s0f0np0` |
| `IB_HCA_NAME` | (`dual-rdma` only) InfiniBand HCA name | `rocep1s0f0` |
| `RAY_PORT` | (`dual-rdma` only) Ray head port | `6379` |
| `VLLM_EXTRA_ARGS` | Model-specific vllm serve flags | `--kv-cache-dtype fp8 --reasoning-parser qwen3` |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | Enable for INT4 AutoRound | `1` (or empty to disable) |

## Patches

Quick summary — see [`PATCH_STATUS.md`](PATCH_STATUS.md) for purpose, scope,
upstream tracking, and removal conditions per patch.

| Patch | Purpose | Status |
|---|---|---|
| `fix_pytorch211_compat` | `hoist=True` removal for PyTorch 2.11 | Active (build) |
| `fastsafetensors_natural_sort` | Multi-node weight loading order fix | Active (build) |
| `aot_cache_fix` | torch.fx.Node pickling fix for AOT cache | Active (build) |
| `nogds_force` | Force `nogds=True` (GB10 has no GDS support) | Active (build) |
| `apply_sm121_patches` | `is_blackwell_class`, NVFP4 split, TRITON_PTXAS | Active (build) |
| `moe_config_e256/e512` | GB10-tuned MoE kernel configs | Active (build) |
| `apply_turboquant_fixes` | TQ KV cherry-picks (PRs #40074, #39988, #39931) | Active (`v021-tq` only) |
| `patch_qwen35_moe_text` | Text-only shim for abliterated Qwen3.5 MoE | Conditional (`APPLY_TEXT_ONLY_SHIM=1`) |
| `patch_codegen_fx_repr` | `__fx_repr__()` honoring in `compilation/codegen.py` | On-standby (hot-patch — see Troubleshooting) |
| ~~`fix_cuda13_memcpy_batch`~~ | `cuMemcpyBatchAsync` API fix | Removed (upstream — base-refresh-20260417) |
| ~~`qwen3_5_moe_rope_fix`~~ | RoPE validation fix | Removed (upstream — base-refresh-20260417) |
| ~~`pr38423_nvfp4_spark`~~ | NVFP4 DGX Spark fixes | Removed (upstream — base-refresh-20260417) |

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

### Qwen3.5-122B-A10B PrismaQuant — Single Node (TP1, mixed-precision + fp8 KV)

4.76bpp mixed-precision checkpoint (NVFP4 bulk MoE / MXFP8 high-sensitivity Linears / BF16 router+embed).
Weights 72 GB, peak VRAM ~86 GB (fp8 KV @ 32k) on a single GB10.
Model ships with MTP speculative-decoding heads — this preset defaults to `n=1` after local tuning.

**Decode throughput vs MTP setting (llama-benchy, 3 runs each, tg32):**

| Concurrency | MTP=3 total / peak | MTP=1 total / peak | MTP=0 total / peak |
|---|---:|---:|---:|
| 1 | 11.2 / 12.5 | 15.7 / 16.4 | **19.1 / 20.0** |
| 2 | 20.5 / 23.0 | 25.7 / 28.7 | **30.4 / 38.0** |
| 3 | 21.1 / 24.0 | 30.3 / 34.0 | **39.8 / 49.0** |
| 4 | 29.2 / 33.7 | 45.1 / 50.7 | **65.1 / 72.3** |

**Prefill throughput (pp2048 total t/s) and TTFT (c=1):**

| MTP | pp c=1 | pp c=4 | TTFT c=1 |
|---|---:|---:|---:|
| n=3 | 1,744 | 2,262 | 1,026 ms |
| n=1 | 1,825 | 2,318 | 1,033 ms |
| n=0 | **1,989** | **2,555** | **947 ms** |

MTP speculative decoding adds per-step overhead; on tg32 microbursts (32 generated tokens) the overhead dominates and MTP=0 wins. For longer natural-text generation the acceptance rate rises and MTP=1 matches or beats MTP=0. MTP=3 (model-card default) measured worst in every throughput bucket on this hardware — the extra speculative tokens lower acceptance and amortize poorly on GB10.

**vs Intel INT4 / RedHatAI NVFP4 (same TP=1, c=1, prior runs):**

| Quant | Disk | pp2048 c=1 | tg32 c=1 | tg32 c=4 peak |
|---|---:|---:|---:|---:|
| Intel INT4 AutoRound | ~65 GB | 2,084 | 29.8 | 96.0 |
| RedHatAI NVFP4 | ~60 GB | 2,027 | 16.2 | 60.0 |
| PrismaQuant (MTP=1) | 72 GB | 1,825 | 15.7 | 50.7 |
| PrismaQuant (MTP=0) | 72 GB | 1,989 | 19.1 | 72.3 |

Intel INT4 remains fastest on GB10. PrismaQuant's value is **quality-per-bit** via Fisher-weighted per-Linear allocation (NVFP4 bulk + MXFP8 for sensitive Linears + BF16 for router/embed) — see the model card for the methodology.

### Qwen3.6-35B-A3B — Single Node (TP1, FP16 + fp8 KV) ⚗️

Experimental test preset (see [Experimental: Qwen3.6-35B-A3B FP16 test preset](#experimental-qwen36-35b-a3b-fp16-test-preset)).
Original bf16/fp16 weights, fp8 KV cache, 32K context, `spark01` single-node.

| Concurrency | pp2048 total t/s | tg32 total t/s | tg32 per-req t/s | peak tg t/s |
|---|---|---|---|---|
| 1 | 3,032 ± 825 | 32.4 ± 0.1 | 32.4 | 33 |
| 2 | 4,724 ± 75 | 63.9 ± 2.2 | 32.0 | 66 |
| 3 | 4,783 ± 439 | 61.1 ± 10.8 | 21.5 | 72 |
| 4 | 5,206 ± 444 | 80.1 ± 19.2 | 22.4 | 101 |

TTFT c=1: ~746 ms (pp2048).

### 397B INT4 TP2 — TurboQuant KV Cache Sweep

Same 397B INT4 AutoRound model on `v021-tq`, TP=2 (spark01+spark02 over 200 Gbps RoCE), `max_model_len=32768`, `gpu_memory_utilization=0.90`. Only `--kv-cache-dtype` varies. Measured 2026-04-17.

#### Capacity & Quality Profile

| Mode | Compression | KV tokens | Max conc @ 32K | PPL vs bf16* |
|---|---:|---:|---:|---:|
| `turboquant_3bit_nc` | 4.9x | 75,488 | 3.00x | +20.6% |
| `turboquant_k3v4_nc` | 3.5x | 64,960 | 3.00x | +10.6% |
| `turboquant_4bit_nc` | 3.8x | 57,120 | 2.82x | +2.7% |
| `turboquant_k8v4`    | 2.6x | 38,528 | 2.50x | +1.2% |

*PPL figures are the upstream reference values from `TurboQuantConfig` docstring.

Note: `k3v4_nc` is strictly dominated by `4bit_nc` — higher compression (3.8x > 3.5x) *and* lower PPL (+2.7% < +10.6%) — because 3-bit keys cost more quality than 4-bit keys cost capacity.

#### Prefill Throughput — `t/s (total)`

| Mode       | pp512 c1 | pp1024 c1 | pp2048 c1 | pp2048 c4 |
|---|---:|---:|---:|---:|
| 3bit_nc    | 916.1 | 1,313.4 | 1,673.4 | 1,928.9 |
| k3v4_nc    | 898.0 | 1,304.1 | 1,663.2 | 2,013.1 |
| 4bit_nc    | 873.8 | 1,300.7* | 1,642.7 | 1,930.8 |
| k8v4       | 901.8 | 1,295.4* | 1,662.7 | 1,931.7 |

\* approx — see full tables in `benchmarks/llama-benchy/results_397b-int4-tq-*-c1to4.md`

#### Decode Throughput — tg128 `t/s (total)` / peak

| Mode       | c1 | c2 | c3 | c4 peak |
|---|---:|---:|---:|---:|
| 3bit_nc    | 26.7 | 42.1 | 50.1 | 72.0 |
| k3v4_nc    | 26.8 | 44.4 | 55.4 | 80.0 |
| 4bit_nc    | 26.6 | 44.7 | 55.2 | **84.0** |
| k8v4       | 26.7 | 45.0 | 56.1 | 78.7 |

#### Analysis

- **Decode throughput (c1) is identical across modes** (26.6-26.8 t/s). Single-request workload is compute-bound on the MoE matmul, not KV memory-bound.
- **High concurrency (c4) amplifies differences**: `4bit_nc` reaches peak 84 t/s tg128 at c4 — **+17% vs 3bit_nc** — because 4-bit value dequant has better arithmetic intensity than 3-bit.
- **KV capacity ≠ throughput**: `3bit_nc` has 2x the KV capacity of `k8v4` but *lower* peak throughput, counter-intuitively. Dequant cost dominates.
- **Prefill is essentially flat** (±3%) across modes — attention read/write is a small fraction of prefill compute for this model.

#### Korean QA Quality (12 questions, mt=30000, thinking off)

Scored on factual correctness of each answer (O=정답, △=부분정답, X=오답). Details in `benchmarks/results/*_Qwen3.5-397B-A17B-int4-AutoRound_mt30000_*.txt`.

| Mode | O | △ | X | Timeout | Score |
|---|---:|---:|---:|---:|---:|
| `3bit_nc` | 7 | 2 | 3 | 0 | **66.7%** |
| `k3v4_nc` | 8 | 3 | 1 | 0 | 79.2% |
| `4bit_nc` | 8 | 3 | 1 | 0 | 79.2% |
| `k8v4`    | 8 | 3 | 0 | 1 | 79.2% (Q6 제외) |

`3bit_nc` shows real quality degradation on logic/syllable-decomposition tasks — matches the +20.6% PPL prediction. The other three modes are indistinguishable on this benchmark (12 questions is too small to separate +1% vs +10% PPL). `k8v4` had one client-side timeout on an overlong answer (seahorse-emoji question, urllib 900 s limit) — not a vLLM/model issue.

#### Recommendation

**`turboquant_4bit_nc` is the operational default** for this model:
- Best peak decode throughput at c4 (84 t/s tg128)
- 3.8x KV compression (~2x concurrency headroom vs bf16)
- Only +2.7% PPL penalty — imperceptible in actual responses
- Strictly better than `k3v4_nc` on every axis

Use `k8v4` only if highest answer fidelity is required and KV capacity is not the bottleneck. Avoid `3bit_nc` — quality loss is measurable.

## System Tuning

Recommended OS-level settings for DGX Spark:

```bash
# Reduce swap pressure (unified memory)
sudo sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## Experimental: Qwen3.6-35B-A3B FP16 test preset

> This is an **experimental test preset** added for quick evaluation of the original
> upstream Qwen3.6 weights on a single DGX Spark. It is **not** a base-stack change —
> the main image, vLLM, FlashInfer, transformers, and CUDA versions are unchanged.

- **Preset file**: `models/qwen3.6-35b-fp16.env`
- **Scope**: `single DGX Spark / TP=1` (designed to fit one GB10 node with headroom)
- **Model**: original Qwen3.6-35B-A3B weights (bf16/fp16, **not quantized**).
  `--kv-cache-dtype fp8` is an optional KV-cache-only optimization and does **not**
  change the model weights.
- **Recommended options** (already in the preset):
  - `--kv-cache-dtype fp8` (KV cache compression only)
  - `--reasoning-parser qwen3`
  - `--enable-chunked-prefill`
  - `--enable-prefix-caching` (added by the entrypoint by default)

### Before launching: stop the running 397B TP=2 stack

```bash
ssh spark01 'cd ~/docker/vllm-spark && docker compose --profile head down'
ssh spark02 'cd ~/docker/vllm-spark && docker compose --profile worker down'
# Clear unified-memory residue between model switches (GB10)
ssh spark01 'sync && sudo sysctl -w vm.drop_caches=3'
```

### Model placement

The model is assumed to exist at
`/mnt/data/llm-models/Qwen/Qwen_Qwen3.6-35B-A3B` on the homeserver. Transfer it to
the chosen Spark node (recommended: `spark01`, same node as the 397B head) before
launch, then point `MODEL_PATH` at the local copy:

```bash
# From homeserver (~67 GB, ~6 min over the RoCE link)
rsync -av /mnt/data/llm-models/Qwen/Qwen_Qwen3.6-35B-A3B/ \
    spark01:/home/bjk110/Documents/Models/Qwen/Qwen_Qwen3.6-35B-A3B/

# On spark01: materialize the preset and substitute the local model root
ssh spark01 'cd ~/docker/vllm-spark && \
    cp models/qwen3.6-35b-fp16.env .env && \
    sed -i "s|\[model_path\]|/home/bjk110/Documents/Models/Qwen|" .env'
```

### Launch (single Spark, TP=1)

```bash
ssh spark01 'cd ~/docker/vllm-spark && \
    docker compose --env-file .env --profile head up -d'
```

### If the first boot fails

Adjust these values in `qwen3.6-35b-fp16.env` in this order (each step lowers
memory pressure):

1. `GPU_MEMORY_UTILIZATION=0.80`
2. `MAX_MODEL_LEN=16384`
3. `MAX_NUM_SEQS=4`
4. Only if the above still fails: consider a TP=2 variant across `spark01` +
   `spark02` (no preset ships for this — this experimental preset is TP=1 only).

## Image tags & Git tags

GHCR image tags (`ghcr.io/bjk110/vllm-spark:<tag>`) and Git tags do **not**
march in lockstep yet — only `v018-ngc2603` exists as a Git tag. The mapping
below documents what each image tag corresponds to in the Git history. Use
this table when you need to reproduce or roll back to a specific image.

| Image tag | Git ref (commit) | Stack | Notes |
|---|---|---|---|
| `v021-tq` (latest) | `3070f9a` | base + TQ patches + Inductor-graph-partition fix | Required for any `*-tq.env` preset |
| `v021-ngc2603` (latest) | `8623187` | vLLM `95995bbe` + FlashInfer `v0.6.9` | Used by every non-TQ preset |
| `v020-ngc2603` (transient) | `8efdf0b` (base-refresh-20260417 base bump) | vLLM `978a4462` + FlashInfer `v0.6.8` | Superseded by v021; only kept on GHCR for historical reproduction |
| `v019-ngc2603` | `7736716` (Gemma 4 + vLLM 0.19.1 upgrade) | vLLM `0.19.1` `a7d79fa` + FlashInfer `v0.6.7.post3` | Superseded by v021 |
| `v018-ngc2603` | `feb5993` (NGC 26.03 source build intro) — Git tag `v018-ngc2603` exists | vLLM `0.18.3` `c494977` + FlashInfer `v0.6.7` | The only currently-tagged release in Git |

### Recommended Git tags to create

Only `v018-ngc2603` is currently tagged. The maintainer can create the
following tags to align Git tags with GHCR image tags. Run from a clean
checkout of `main`; do **not** run blindly — verify the SHAs first.

    git tag -a v019-ngc2603 7736716 -m "v019-ngc2603 — Gemma 4 + vLLM 0.19.1"
    git tag -a v020-ngc2603 8efdf0b -m "v020-ngc2603 — base-refresh-20260417 (vLLM 978a4462, FlashInfer 0.6.8)"
    git tag -a v021-ngc2603 8623187 -m "v021-ngc2603 — vLLM 95995bbe + FlashInfer v0.6.9"
    git tag -a v021-tq      3070f9a -m "v021-tq — base + TurboQuant cherry-picks + codegen workaround"
    git push origin v019-ngc2603 v020-ngc2603 v021-ngc2603 v021-tq

**Verify commit before tagging.** The four SHAs above were extracted from
`git log --oneline` at the time this README was last updated; if subsequent
work reshuffles `main`, re-locate the boundary commits with:

    git log --oneline --grep='base.refresh\|bump base.*v021\|0.19.1\|use Inductor graph partition'

## Branch structure

`main` is the only long-lived branch. All previously separate work
streams (base stack refresh, TurboQuant rebase, single-Spark CLUSTER_MODE)
have been merged in and their feature branches deleted.

### Archived branch history

The legacy TurboQuant branch is preserved as a tag for reference:

- **`archive/feat-turboquant`**

If needed, it can be restored with:

```bash
git checkout -b feat/turboquant archive/feat-turboquant
```

## License

Configuration files are provided as-is for reference. Models are subject to their respective licenses ([Qwen License](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)).
