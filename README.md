# spark_vllm_docker

## Overview

Unified vLLM serving configuration for NVIDIA DGX Spark (GB10), supporting two
topologies from the same repo / Dockerfile / compose file:

- **Single Spark** (default, zero RDMA setup) — one GB10 box, TP=1.
- **Dual Spark + 200 Gbps RoCE/IB** — two GB10 boxes, Ray, TP=2.

Pick the topology by setting `CLUSTER_MODE=single` (default) or
`CLUSTER_MODE=dual-rdma` in your `.env`. See [`Quick Start`](#quick-start) below.

For release-by-release detail and patch-by-patch status, see
[`CHANGELOG.md`](CHANGELOG.md) and [`PATCH_STATUS.md`](PATCH_STATUS.md).

## Current serving paths

| Path | Status | Image | Backend | Use case |
|---|---|---|---|---|
| `dsv4-d568` | **Frozen** primary DSV4 baseline | `ghcr.io/bjk110/vllm-spark:dsv4-d568` | `ray` (default) or `mp` | DeepSeek-V4-Flash on dual DGX Spark — see [`docs/dsv4-flash-tp2.md`](docs/dsv4-flash-tp2.md) |
| `unholy-fusion` | Experimental (mp-only, DSV4 only) | `aidendle94/sparkrun-vllm-ds4-gb10:production-ready` (mirror: `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`) | `mp` (hardcoded) | Higher-prefill DSV4 alternative — see [`docs/unholy-fusion-benchmark.md`](docs/unholy-fusion-benchmark.md) |
| `v022-d568-ngc2605-tx5102-vllm022` | Active forward-stack (NGC 26.05, vLLM 0.22.1) | `ghcr.io/bjk110/vllm-spark:v022-d568-ngc2605-tx5102-vllm022` | `ray` | Qwen3.5-122B-FP8 and other forward-stack models |
| `v022-d568` | Stable general base (NGC 26.04, vLLM 0.21.0) | `ghcr.io/bjk110/vllm-spark:v022-d568` | `ray` or direct | Qwen3.6, Gemma 4 31B, abliterix NVFP4 presets |
| `v021-ngc2603` / `v021-tq` | Stable base for most existing presets | `ghcr.io/bjk110/vllm-spark:v021-ngc2603` / `v021-tq` | `ray` or direct | Most non-DSV4 presets in `presets/*.env` |

> **Image policy**: `dsv4-d568` is intentionally **frozen** and will not be rebased
> onto NGC 26.05 or any later stack. Forward-stack upgrades (e.g.
> `v022-d568-ngc2605-tx5102-vllm022`) are a **separate parallel path** for non-DSV4
> models. Do not modify or rebase `dsv4-d568` without an explicit decision to do so.

Component versions, stack lineage, and image digests → [`docs/software-stack.md`](docs/software-stack.md).
Image tag → Git-ref mapping → [`docs/images.md`](docs/images.md).

> **Optional**: `v022-d568-fi-aot` is a validated drop-in replacement for
> `v022-d568` with FlashInfer SM120/SM121 kernels AOT-prebaked (avoids runtime
> ninja/nvcc for those kernels) — see
> [`docs/flashinfer-aot-prebake.md`](docs/flashinfer-aot-prebake.md).

## Hardware and topology

| Topology | Node | Role | GPU | Memory | Interconnect | Backend |
|---|---|---|---|---|---|---|
| `single` | one Spark | vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB unified | n/a | direct (no Ray, no `mp`) |
| `dual-rdma` (`ray`, default) | spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE | `ray` |
| `dual-rdma` (`ray`, default) | spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE | `ray` |
| `dual-rdma` (`mp`) | spark01 | vLLM head (`--node-rank 0`, no Ray) | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE | `mp` |
| `dual-rdma` (`mp`) | spark02 | vLLM worker (`--headless --node-rank 1`, no Ray) | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE | `mp` |

> **Backend note**: `dual-rdma` deployments support two coordination backends,
> selected via `DISTRIBUTED_BACKEND=ray` (default) or `DISTRIBUTED_BACKEND=mp`
> (SPMD, no Ray). The primary `dsv4-d568` path uses `ray`; the experimental
> `unholy-fusion` path hardcodes `mp` (its image ships no Ray binary). See
> [Quick Start § Backend selection](#backend-selection--distributed_backendray--mp)
> for the full comparison and switching steps.

## Quick Start

> **Note**: The `presets/` directory contains `.env` preset files only.
> It does **not** store actual model weights. Keep model weights outside the repository
> and point `MODEL_PATH` / `MODEL_CONTAINER_PATH` to the correct host/container paths.
> See [`presets/README.md`](presets/README.md) for details.

### 0. Choose a serving path

Select the path matching your model and use case from the
[Current serving paths](#current-serving-paths) table above:

- **Non-DSV4 models** on NGC 26.05 + vLLM 0.22.1 (e.g. Qwen3.5-122B-FP8):
  use `v022-d568-ngc2605-tx5102-vllm022`
- **Non-DSV4 models** on the stable NGC 26.04 base (Qwen3.6, Gemma 4 31B, etc.):
  use `v022-d568`
- **Non-DSV4 models** on the legacy stable base: use `v021-ngc2603` (or `v021-tq`
  for TurboQuant KV presets)
- **DeepSeek-V4-Flash** (decode-optimized, stable): use `dsv4-d568`
- **DeepSeek-V4-Flash** (higher prefill, experimental): use `unholy-fusion`

### 1. Get the Docker Image

#### Option A: Pull pre-built image from GHCR

```bash
# Stable base for most existing presets:
docker pull ghcr.io/bjk110/vllm-spark:v021-ngc2603

# TurboQuant image (required for *-tq presets):
docker pull ghcr.io/bjk110/vllm-spark:v021-tq

# v022-d568 general base (NGC 26.04 + vLLM 0.21.0+PR#35568):
# Digest: sha256:88b544ed69476f3785ea7ce37fc8b99f0f064cc299eef35cda1535c68e7a9501
docker pull ghcr.io/bjk110/vllm-spark:v022-d568

# Forward-stack (NGC 26.05 + vLLM 0.22.1 + FlashInfer v0.6.12 + Transformers 5.10.2):
# Digest: sha256:2c52c885e48cffd15097291a0926a51401ecb95d169e6831bb63a4fda6dda8a5
docker pull ghcr.io/bjk110/vllm-spark:v022-d568-ngc2605-tx5102-vllm022

# DeepSeek-V4-Flash only (FROZEN — NOT rebased onto NGC 26.05):
docker pull ghcr.io/bjk110/vllm-spark:dsv4-d568
```

Intermediate bisection-only build variants (`v022-vllm021`, `v022-fi0611`, etc.) are
local-build only — see [`docs/stack-v022.md`](docs/stack-v022.md).

#### Option B: Build from source

All builds must run on spark01 or spark02.

```bash
# Current forward-stack (NGC 26.05 + vLLM 0.22.1 + FlashInfer v0.6.12 + Transformers 5.10.2):
docker buildx build -f dockerfiles/active/Dockerfile.v022-d568 \
  -t vllm-spark:v022-d568-ngc2605-tx5102-vllm022 --load .

# Stable legacy base (NGC 26.03 / vLLM main + TurboQuant):
docker buildx build -f dockerfiles/legacy/Dockerfile.gemma4 \
  -t vllm-spark:v021-ngc2603 --load .

# DeepSeek-V4-Flash derivative (FROZEN — reproduce only; build on spark node):
docker buildx build -f dockerfiles/active/Dockerfile.dsv4-d568 \
  -t vllm-spark:dsv4-d568 --load .
```

Build arguments for `Dockerfile.v022-d568`:

| Argument | Default | Description |
|---|---|---|
| `NGC_TAG` | `26.05-py3` | NGC PyTorch base image tag |
| `BUILD_JOBS` | `16` | Parallel build jobs |
| `VLLM_COMMIT` | `ad7125a431e176d4161099480a66f0169609a690` | vLLM source commit |
| `VLLM_VERSION` | `0.22.1` | Version for setuptools-scm pretend |
| `FLASHINFER_REF` | `v0.6.12` | FlashInfer git ref |
| `TRANSFORMERS_VER` | `5.10.2` | Transformers version |
| `TORCH_CUDA_ARCH` | `12.1a` | Target CUDA arch (Blackwell) |

### 2. Choose a Model Preset

Single-Spark presets ship with `CLUSTER_MODE=single` and TP=1 (zero RDMA setup).
Dual-Spark presets ship with `CLUSTER_MODE=dual-rdma` and TP=2.

```bash
# Single Spark (no RDMA needed):
cp presets/redhatai-122b-nvfp4.env .env

# Dual Spark + RoCE:
cp presets/qwen3.5-397b-int4.env .env
```

Edit `MODEL_PATH` in `.env` to point to your local model weights directory:

```bash
sed -i 's|\[model_path\]|/home/user/models|' .env
```

### 3. Start Services

#### Single Spark — TP=1 (default, no Ray, no RDMA)

```bash
docker compose --profile head up -d
docker logs -f vllm-spark-head
```

`entrypoint.sh` reads `CLUSTER_MODE=single` and **forces** `VLLM_HOST_IP=127.0.0.1`,
unsets `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `UCX_NET_DEVICES` / `NCCL_IB_HCA`,
and sets `NCCL_IB_DISABLE=1`. This is what avoids the
`tcp://10.10.10.1:<port> server socket has timed out` c10d hang
(see [`docs/troubleshooting.md`](docs/troubleshooting.md)).

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

#### Backend selection — `DISTRIBUTED_BACKEND=ray | mp`

For `dual-rdma` deployments, the entrypoint chooses how the two nodes coordinate based on the `DISTRIBUTED_BACKEND` env var (default `ray`):

| Mode | How it works | When to use |
|---|---|---|
| `ray` (default) | head starts `ray start --head`, worker joins via `ray start --address=…`, vLLM serves with `--distributed-executor-backend ray` | Existing validated path for all multi-node presets; matches established behavior |
| `mp` | head + worker both run `vllm serve` simultaneously (SPMD); head uses `--nnodes N --node-rank 0 --master-addr <head> --master-port <port>`; worker adds `--headless`. No Ray. Matches the eugr/spark-vllm-docker `--no-ray` path. | Avoids Ray Compiled DAG cross-node bug [#36237](https://github.com/vllm-project/vllm/issues/36237) when it surfaces; required by some forum-published recipes (e.g. DeepSeek-V4-Flash) |

Switching is a single env line — same image, same compose, just restart:

```
# in presets/<preset>.env
DISTRIBUTED_BACKEND=mp   # or ray (default)
MASTER_PORT=29501        # only used in mp mode
```

For DSV4 measurements, Ray and mp showed similar decode peak only in the measured no-MTP configuration on our GB10 setup. Stronger claims require additional latency, prefill, and stability metrics. See [`docs/dsv4-flash-tp2.md`](docs/dsv4-flash-tp2.md) §5 for the data.

### 4. Verify

```bash
curl http://localhost:8000/health      # single
curl http://spark01:8000/health        # dual-rdma
```

## Presets and model paths

The table below covers the currently shipped presets in `presets/`. For the
complete list, see [`presets/`](presets/) — each preset file documents its own
recipe / image / topology in its header comment.

> Some preset notes below name the script used to produce a given checkpoint
> (e.g. `quantize_qwen35_abliterix_fp8_direct.py`, `convert_bf16_to_nvfp4.py`)
> for provenance only. Quantization tooling is intentionally out of scope for
> this public serving repository — this repository focuses on DGX Spark / GB10
> vLLM container serving, presets, runtime patches, and validation notes.

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
| `wangzhang-122b-abliterix-fp8-tp2.env` | wangzhang/Qwen3.5-122B-A10B-abliterix | FP8 W8A8 (text-only, custom safetensors-level quant) | dual-rdma | 2 | v021-ngc2603 | `APPLY_TEXT_ONLY_SHIM=1`; BF16→FP8 via `quantize_qwen35_abliterix_fp8_direct.py` |
| `wangzhang-122b-abliterix-nvfp4-tp2.env` | wangzhang/Qwen3.5-122B-A10B-abliterix | NVFP4 W4A4 (text-only, custom safetensors-level quant with fused-group shared `weight_global_scale`) | dual-rdma | 2 | v022-d568 | `APPLY_TEXT_ONLY_SHIM=1`; BF16→NVFP4 via `fp8-quantizer/convert_bf16_to_nvfp4.py`. q/k/v_proj, expert gate/up, and shared_expert gate/up share one `weight_global_scale` per fused group — required to avoid the FlashInfer NVFP4 fused-Linear precision-loss warning and corresponding garbage outputs. `model_type=qwen3_5_moe_text` (flat config) to avoid the wrapper config's default `text_config.hidden_size=2048` |
| `qwen3.5-397b-int4.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 AutoRound (Marlin) | dual-rdma | 2 | v021-ngc2603 | — |
| `qwen3.5-397b-int4-tq.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 AutoRound + **TurboQuant KV** (`turboquant_3bit_nc` cascade) | dual-rdma | 2 | v021-tq | TQ baked-in; uses `--compilation-config {"use_inductor_graph_partition":true}` |
| `qwen3.6-35b-fp16.env` ⚗️ | Qwen/Qwen3.6-35B-A3B | **FP16 original** (KV fp8) | single | 1 | v021-ngc2603 | Experimental |
| `qwen3.6-35b-a3b.env` | Qwen/Qwen3.6-35B-A3B | BF16 hybrid Mamba/Attention MoE (KV fp8) | single | 1 | v022-d568 | `--reasoning-parser qwen3` + `--compilation-config {"use_inductor_graph_partition":true}` for hybrid arch |
| `gemma4-31b-it.env` | google/gemma-4-31B-it | BF16 dense multimodal | single | 1 | v022-d568 | `--limit-mm-per-prompt {"image":1,"audio":0,"video":0}` (audio still beta in vLLM 0.21) |
| `qwen3.6-27b-prismascout-nvfp4-tp2.env` (+ `-v022`) | rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm | NVFP4 mixed-precision (ViT NVFP4 + LM NVFP4 + BF16 sidecars) | dual-rdma | 2 | v022-vllm021 | MTP `n=3`; **v022 preset requires `--mm-encoder-tp-mode data`** (see [`docs/software-stack.md`](docs/software-stack.md)) for ViT MLP K-align |
| `dsv4-flash-fp8-tp2.env` | deepseek-ai/DeepSeek-V4-Flash | FP8 (E4M3 128×128 block, official) | dual-rdma | 2 | **dsv4-d568** or **unholy-fusion** | DSV4 sparse MLA + Lightning Indexer + fp8_ds_mla KV + MTP heads. Full guide: [`docs/dsv4-flash-tp2.md`](docs/dsv4-flash-tp2.md). Alternative: unholy-fusion image (`aidendle94/sparkrun-vllm-ds4-gb10:production-ready`) — 2× prefill speedup via B12X_MOE kernel, capped at MAX_NUM_SEQS=4 / MAX_MODEL_LEN=262144. See [`docs/unholy-fusion-benchmark.md`](docs/unholy-fusion-benchmark.md). |
| `step37-flash-fp8-tp2.env` | stepfun-ai/Step-3.7-Flash-FP8 | FP8 block (e4m3, 128×128, official) | dual-rdma | 2 | v022-d568-ngc2605-tx5102-vllm022-step3p7 | 198B sparse MoE VLM. `MAX_NUM_SEQS=1` (GB10 121GiB UMA memory ceiling, §3 of guide). Full guide: [`docs/step3.7-flash-tp2.md`](docs/step3.7-flash-tp2.md) |
| `step37-flash-nvfp4-tp2.env` | stepfun-ai/Step-3.7-Flash-NVFP4 | NVFP4 (modelopt) + FP8 KV | dual-rdma | 2 | v022-d568-ngc2605-tx5102-vllm022-step3p7 | Requires `patches/patch_step3p7_nvfp4_input_scale.py` (missing `expert_params_mapping.input_scale` causes NaN logits otherwise). `MAX_NUM_SEQS=4` verified. Full guide: [`docs/step3.7-flash-tp2.md`](docs/step3.7-flash-tp2.md) |

## Repository layout

```
vllm-spark/
├── docker-compose.yml             # Unified compose (head + worker profiles)
├── entrypoints/                   # Container entrypoint scripts (selected via ENTRYPOINT_FILE — see entrypoints/README.md)
│   ├── entrypoint.sh                  # CLUSTER_MODE-aware entrypoint (standard dsv4-d568 path)
│   └── entrypoint.unholy.sh           # mp-only entrypoint for the unholy-fusion path
├── .env.example                   # Full configuration template
├── dockerfiles/                   # All Dockerfiles; see dockerfiles/README.md
│   ├── active/                        # Current active build targets (build from repo root with . as context)
│   │   ├── Dockerfile.v022-d568           # Forward-stack base (NGC 26.05 + vLLM 0.22.1 + FlashInfer v0.6.12 + Transformers 5.10.2)
│   │   └── Dockerfile.dsv4-d568           # DeepSeek-V4-Flash derivative (FROM v022-d568; FROZEN)
│   └── legacy/                        # Historical / intermediate / specialized variants
│       ├── Dockerfile                     # NGC 26.01 era (vLLM 0.18.x, historical)
│       ├── Dockerfile.gemma4              # v021-ngc2603 unified build
│       ├── Dockerfile.ngc2603-v3          # v018-ngc2603 archived build
│       ├── Dockerfile.nvfp4               # NVFP4 runtime defaults overlay
│       └── Dockerfile.v022(-fi0611/-ngc2604/-tx581/-trt37/-nccl234)  # v022 stack intermediates
├── CHANGELOG.md                   # Release-by-release history
├── PATCH_STATUS.md                # Per-patch purpose / status / removal condition
├── presets/                       # .env preset files for model-serving configs (not model weights — see presets/README.md)
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
│   ├── wangzhang-122b-abliterix-fp8-tp2.env  # abliterix FP8 W8A8 text-only (dual-rdma, TP2)
│   ├── wangzhang-122b-abliterix-nvfp4-tp2.env # abliterix NVFP4 W4A4 text-only (dual-rdma, TP2; v022-d568)
│   ├── gemma4-31b-it.env             # Gemma 4 31B IT BF16 dense multimodal (single, TP1; v022-d568)
│   ├── qwen3.6-35b-a3b.env           # Qwen3.6-35B-A3B BF16 hybrid MoE (single, TP1; v022-d568)
│   ├── dsv4-flash-fp8-tp2.env        # DeepSeek-V4-Flash official FP8 (dual-rdma, TP2; dsv4-d568)
│   ├── step37-flash-fp8-tp2.env      # Step-3.7-Flash FP8 block (dual-rdma, TP2; step3p7)
│   ├── step37-flash-nvfp4-tp2.env    # Step-3.7-Flash NVFP4 + FP8 KV (dual-rdma, TP2; step3p7)
│   └── qwen3.6-35b-fp16.env           # ⚗️ Qwen3.6 FP16 experimental (single, TP1)
├── .env.unholy-fusion             # unholy-fusion config (MAX_NUM_SEQS=4, mp backend, B12X_MOE=1)
├── docs/                          # Technical notes, stack guides, and status documents
│   ├── dsv4-flash-tp2.md             # DSV4-Flash: build, recipe, 9-way benchmark sweep
│   ├── step3.7-flash-tp2.md          # Step-3.7-Flash: FP8/NVFP4 dual-Spark serving + benchmark comparison
│   └── unholy-fusion-benchmark.md    # Interpreted unholy-fusion serving result analysis and DSV4 comparison
├── benchmarks/                    # Raw benchmark artifacts and experiment outputs; see benchmarks/README.md
│   └── llama-benchy/              # Raw llama-benchy result files; see benchmarks/llama-benchy/README.md
├── patches/                       # Build/runtime patch scripts; grouped by purpose — see patches/README.md
│   ├── common/                        # Common runtime/build compatibility patches
│   ├── sm121/                         # SM121 / Blackwell / FP8 / NVFP4 patches
│   ├── dsv4/                          # DeepSeek-V4 specific patches and MoE config files
│   ├── qwen/                          # Qwen-specific compatibility patches
│   ├── turboquant/                    # TurboQuant-specific patches
│   ├── flashinfer/                    # FlashInfer-specific patches
│   ├── archive/                       # Historical patches retained for reproducibility
│   └── unknown/                       # Unverified early bring-up helpers
└── scripts/
    ├── run-cluster-node.sh        # Manual Ray cluster bootstrap
    ├── verify_imports.py          # Build-time import verification
    ├── verify_runtime.sh          # Full GPU verification
    └── check-empty-env.sh         # Lint presets for empty optional env vars (issue #14)
```

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
dispatches on `ROLE` × `TP_SIZE` × `DISTRIBUTED_BACKEND` (dual-rdma only):

| CLUSTER_MODE | ROLE | TP_SIZE | Backend | Behavior |
|---|---|---|---|---|
| `single` | `head` | 1 | — | Force `VLLM_HOST_IP=127.0.0.1`, clear NCCL/GLOO/UCX ifname, set `NCCL_IB_DISABLE=1`, then direct `vllm serve` (no Ray) |
| `single` | `head` | 2+ | — | Fail-fast (`single` cannot host TP≥2) |
| `single` | `worker` | any | — | Fail-fast (worker is meaningless in single mode) |
| `dual-rdma` | `head` | 1 | — | Reject (use `single` for TP=1) |
| `dual-rdma` | `head` | 2+ | `ray` (default) | Validate RDMA env + `RAY_PORT` → Ray head → wait for workers → `vllm serve --distributed-executor-backend ray` |
| `dual-rdma` | `worker` | any | `ray` (default) | `ray start --address=$HEAD_ROCE_IP:$RAY_PORT --block` |
| `dual-rdma` | `head` | 2+ | `mp` | Validate RDMA env + `MASTER_*`/`NNODES`/`NODE_RANK` (defaults: `MASTER_ADDR=$HEAD_ROCE_IP`, `MASTER_PORT=29501`, `NNODES=$TP_SIZE`, `NODE_RANK=0`) → `vllm serve --distributed-executor-backend mp --nnodes $NNODES --node-rank 0 --master-addr $MASTER_ADDR --master-port $MASTER_PORT` |
| `dual-rdma` | `worker` | any | `mp` | `vllm serve --distributed-executor-backend mp --headless --nnodes $NNODES --node-rank 1 --master-addr $MASTER_ADDR --master-port $MASTER_PORT` |

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
| `RAY_PORT` | (`dual-rdma` + `ray` backend only) Ray head port | `6379` |
| `DISTRIBUTED_BACKEND` | (`dual-rdma` only) `ray` (default) or `mp` (SPMD no-Ray) | `ray` |
| `MASTER_PORT` | (`mp` backend only) torch.distributed master port | `29501` |
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
| `patch_codegen_fx_repr` | `__fx_repr__()` honoring in `compilation/codegen.py` | On-standby (hot-patch — see [`docs/troubleshooting.md`](docs/troubleshooting.md)) |
| ~~`fix_cuda13_memcpy_batch`~~ | `cuMemcpyBatchAsync` API fix | Removed (upstream — base-refresh-20260417) |
| ~~`qwen3_5_moe_rope_fix`~~ | RoPE validation fix | Removed (upstream — base-refresh-20260417) |
| ~~`pr38423_nvfp4_spark`~~ | NVFP4 DGX Spark fixes | Removed (upstream — base-refresh-20260417) |

## Benchmark Results

Historical throughput benchmarks for the Gemma 4, Qwen3.5 122B, 397B INT4, PrismaQuant,
Qwen3.6-35B, and TurboQuant KV sweep model families are in
[`docs/model-serving-validation-history.md`](docs/model-serving-validation-history.md).

## System Tuning

Recommended OS-level settings for DGX Spark:

```bash
# Reduce swap pressure (unified memory)
sudo sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## Documentation

| Document | Covers |
|---|---|
| [`docs/software-stack.md`](docs/software-stack.md) | Full image/stack lineage: `v022-d568-ngc2605-tx5102-vllm022`, `dsv4-d568`, `v022-d568`, `v021` series, and legacy stacks, with component versions and digests |
| [`docs/dsv4-flash-tp2.md`](docs/dsv4-flash-tp2.md) | DeepSeek-V4-Flash (`dsv4-d568`) build, deployment recipe, and 9-way benchmark sweep |
| [`docs/unholy-fusion-benchmark.md`](docs/unholy-fusion-benchmark.md) | `unholy-fusion` configuration, switching procedure, operational limits, and benchmark comparison vs `dsv4-d568` |
| [`docs/model-serving-validation-history.md`](docs/model-serving-validation-history.md) | Historical stack validation notes and benchmark results (Gemma 4, Qwen3.5 122B, 397B INT4, PrismaQuant, Qwen3.6-35B, TurboQuant KV sweep) |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Model-path and stack-specific troubleshooting (Docker Compose checks, `dsv4-d568` / `unholy-fusion` / Qwen issues, logs and verification commands) |
| [`docs/images.md`](docs/images.md) | Container image tag history and image-to-preset / Git-ref mapping |
| [`docs/release-management.md`](docs/release-management.md) | Maintainer-only Git tag creation, branch structure, and archived branch notes |

## License

The source code, Dockerfiles, scripts, presets, and documentation in this repository are licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).

This repository does **not** distribute model weights. Presets may reference upstream models, but users are responsible for obtaining model weights and complying with the applicable upstream model licenses and terms.

Container images and dependencies remain governed by their respective upstream licenses and terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
