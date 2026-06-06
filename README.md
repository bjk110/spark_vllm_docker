# vLLM Spark — Unified Serving for DGX Spark (GB10)

Unified vLLM serving configuration for NVIDIA DGX Spark (GB10), supporting two
topologies from the same repo / Dockerfile / compose file:

- **Single Spark** (default, zero RDMA setup) — one GB10 box, TP=1.
- **Dual Spark + 200 Gbps RoCE/IB** — two GB10 boxes, Ray, TP=2.

Pick the topology by setting `CLUSTER_MODE=single` (default) or
`CLUSTER_MODE=dual-rdma` in your `.env`. See [`Quick Start`](#quick-start) below.

For release-by-release detail and patch-by-patch status, see
[`CHANGELOG.md`](CHANGELOG.md) and [`PATCH_STATUS.md`](PATCH_STATUS.md).

For a high-level overview of the current repository state and deferred cleanup roadmap,
see [`docs/repository-status.md`](docs/repository-status.md).

## Hardware

| Topology | Node | Role | GPU | Memory | Interconnect |
|---|---|---|---|---|---|
| single | one Spark | vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB unified | n/a |
| dual-rdma | spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE |
| dual-rdma | spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE |

## Software Stack

### v022-d568 (NGC 26.04, vLLM v0.21.0+#35568, FlashInfer 0.6.11.post3, Transformers 5.8.1, Triton 3.7.0, NCCL 2.30.4) — final forward-stack

Stacked-upgrade image built 2026-05-18, the deepest in the v022 series. Each layer was booted and verified against the PrismaSCOUT NVFP4 TP=2 preset (text + image inference, MTP n=3); the final `-d568` layer was additionally verified against `wangzhang-122b-abliterix-fp8-tp2` to confirm the SM121 FP8 kernel path now activates, on 2026-05-19 against `wangzhang-122b-abliterix-nvfp4-tp2` (custom BF16 → NVFP4 W4A4 with fused-group shared `weight_global_scale`) to confirm the SM121 NVFP4 path (FlashInfer-CUTLASS NVFP4 GEMM + MoE) is live end-to-end, and on 2026-05-20 against `gemma4-31b-it` (dense BF16 multimodal, TP=1) and `qwen3.6-35b-a3b` (hybrid Mamba/Attention MoE BF16 with `--reasoning-parser qwen3` + `--compilation-config use_inductor_graph_partition=true`, TP=1) to confirm dense/hybrid single-node paths. Use `v022-d568` to validate behavior on the released v0.21.0 plus the cherry-pick.

Current image roles:
- `v021-ngc2603`: stable base for most existing presets (non-TQ)
- `v021-tq`: TurboQuant preset base (required for `*-tq.env` presets)
- `v022-d568`: forward-stack validation base (NGC 26.04 + vLLM 0.21.0)
- `dsv4-d568`: primary DeepSeek-V4-Flash path
- `unholy-fusion`: experimental high-prefill DeepSeek-V4-Flash path

| Component | Version |
|---|---|
| Base Image | NGC PyTorch **26.04-py3** |
| vLLM | **0.21.0 + PR #35568** (release tag `ad7125a4` + cherry-pick of commit `06d020bb6`, source rebuild) |
| FlashInfer | **v0.6.11.post3** (SM120/121 XQA MLA bug fixes #2689, CUTLASS Small Tile N Blockscaled GEMMs #3152, Blackwell GDN accuracy #3156, SM120 cuDNN NaN #3192, NVFP4 KV prefill #3097) |
| PyTorch | **2.12.0a0** |
| CUDA | 13.2 (native) |
| Transformers | **5.8.1** |
| Triton | **3.7.0** (vanilla PyPI; NGC 26.04 still bundles 3.6.0) |
| NCCL | **2.30.4** (runtime via `nvidia-nccl-cu13` pip + `LD_LIBRARY_PATH`; NGC 26.04 system NCCL stays at 2.29.7) |
| tokenizers | 0.22.2 (Transformers 5.8.1 pins `<=0.23.0`; PyPI has no `0.23.0` stable, so 0.22.2 is the highest compatible) |
| Image tag | `ghcr.io/bjk110/vllm-spark:v022-d568` (**on GHCR**, digest `sha256:88b544ed`) |

Intermediate stacked images (**local-build only**, kept for bisection / rollback — not pushed to GHCR):
- `ghcr.io/bjk110/vllm-spark:v022-fi0611` — v022-vllm021 + FlashInfer 0.6.11.post3
- `ghcr.io/bjk110/vllm-spark:v022-ngc2604` — v022-fi0611 + NGC 26.04 (PyTorch 2.12.0a0) + `patch_split_module_compat.py`
- `ghcr.io/bjk110/vllm-spark:v022-tx581` — v022-ngc2604 + Transformers 5.8.1
- `ghcr.io/bjk110/vllm-spark:v022-trt37` — v022-tx581 + Triton 3.7.0
- `ghcr.io/bjk110/vllm-spark:v022-nccl234` — v022-trt37 + NCCL 2.30.4

**Runtime patches added during the stack:**
- `patches/common/patch_split_module_compat.py` (since `-ngc2604`): swaps vLLM's static `is_torch_equal_or_newer("2.12.0.dev")` gate around `torch.fx.passes.split_module.split_module(tuple_return=True)` for an `inspect.signature(...).parameters` probe. NGC 26.04 ships a PyTorch 2.12 alpha that predates the upstream `tuple_return` commit, so the version gate would otherwise fire false-positive and PyTorch would raise `TypeError`.
- `patches/sm121/apply_sm121_fp8_pr35568.py` (only on `-d568`): build-time cherry-pick of vLLM PR #35568. Widens four `enable_sm120_only` / `arch in [89, 120]` gates to `SM12x family` in the Marlin/CUTLASS FP8 codepaths so the DGX Spark GB10 (SM121) is no longer excluded. Confirmed live by the abliterix-FP8 boot logging `Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8`.

Verified preset overrides:
- `presets/qwen3.6-27b-prismascout-nvfp4-tp2-v022-{fi0611,ngc2604,tx581,trt37,nccl234,d568}.env` — PrismaSCOUT NVFP4 (text + image)
- `presets/wangzhang-122b-abliterix-fp8-tp2-v022-d568.env` — abliterix FP8 (text, confirms FP8 kernel path activation)
- `presets/wangzhang-122b-abliterix-nvfp4-tp2.env` — abliterix NVFP4 (text, **custom BF16 → NVFP4 with fused-group shared `weight_global_scale`**; confirms FlashInfer-CUTLASS NVFP4 GEMM + MoE path activation)
- `presets/gemma4-31b-it.env` — Gemma 4 31B IT (dense BF16 multimodal, single TP=1; confirms dense Gemma 4 path on the forward stack)
- `presets/qwen3.6-35b-a3b.env` — Qwen3.6-35B-A3B (hybrid Mamba/Attention MoE BF16, single TP=1; confirms `--reasoning-parser qwen3` + Inductor graph-partition path)

### dsv4-d568 — Primary DeepSeek-V4-Flash path

**This is the primary documented path for DeepSeek-V4-Flash on 2× DGX Spark / GB10.**

Layered on top of `v022-d568`. Uses a fork of vLLM with SM12x DSV4 support (sparse MLA, Lightning Indexer, fp8_ds_mla KV cache, MTP heads). Preset: `presets/dsv4-flash-fp8-tp2.env`.

| Component | Version |
|---|---|
| Base Image | `ghcr.io/bjk110/vllm-spark:v022-d568` |
| vLLM | source rebuild with SM12x DSV4 patches (sparse MLA, Lightning Indexer, fp8_ds_mla KV, MTP) |
| Other layers | unchanged from v022-d568 |
| Additional patches | `apply_dsv4_packed_mapping.py`, `patch_split_module_compat.py` (re-applied), `moe_config_e256/e512.json` (re-staged), `instanttensor` pip dep |
| Image tag | `ghcr.io/bjk110/vllm-spark:dsv4-d568` (**on GHCR**, digest `sha256:b18da2a0`) |

Verified preset: `presets/dsv4-flash-fp8-tp2.env` — DeepSeek-V4-Flash dual-rdma TP=2, 200K ctx, fp8 KV cache + Lightning Indexer.

**Full guide + 9-way benchmark sweep + MTP/backend analysis**: [`docs/dsv4-flash-tp2.md`](docs/dsv4-flash-tp2.md).

> **DSV4 path summary**: For DeepSeek-V4-Flash, use `dsv4-d568` as the primary path. For users who specifically want higher prefill throughput, `unholy-fusion` is available as an experimental alternative (see below). Earlier jasl-based DSV4 image notes are deferred and kept only for historical reference.

### unholy-fusion — Experimental high-prefill DSV4 path

> **Experimental**: unholy-fusion is an alternative for users who specifically want higher
> prefill throughput on 2× DGX Spark / GB10. It is not a general production default.
> For general use, start with `dsv4-d568` above.

Third-party image from `local-inference-lab/vllm:dev/unholy-fusion`
(Docker Hub: `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`, also mirrored as
`ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`). Adds custom GB10 (Blackwell
sm_120/sm_121) kernels unavailable in the dsv4-d568 image.

| Component | Detail |
|---|---|
| Image | `aidendle94/sparkrun-vllm-ds4-gb10:production-ready` (Docker Hub) |
| Mirror | `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` |
| Backend | `mp` (SPMD, no Ray) |
| Runtime env | conda (`/opt/env`) — no NGC base |
| MTP | n=1 (n=2 causes catastrophic collapse with B12X_MOE at c≥4) |

Key B12X kernel switches (GB10-specific, not available in the dsv4-d568 image):

| Env var | Kernel | Setting |
|---|---|---|
| `VLLM_USE_B12X_MOE=1` | Custom MoE dispatcher for GB10 | **On** — delivers 2× prefill speedup vs dsv4-d568 |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | | Required (=1 causes garbled output) |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | | Required (missing causes `ValueError` at engine init) |
| `VLLM_USE_B12X_MHC` | Multi-head compression | Off (unstable) |

Operational limits vs dsv4-d568:

| Limit | Safe | Broken | Failure |
|---|---|---|---|
| MAX_NUM_SEQS | ≤ 4 | ≥ 5 | CUDA graph capture hang at startup |
| MAX_MODEL_LEN | ≤ 262144 | 524288 | Starts OK but crashes at d≥131072 (RPC timeout) |
| Max context depth | d=131072 (128k tokens) | — | effective ceiling with MAX_MODEL_LEN=262144 |

KV cache: ~17.1 GiB / 1,144,306 tokens at GPU_UTIL=0.80.

**Full benchmark analysis + comparison vs dsv4-d568**: [`docs/unholy-fusion-benchmark.md`](docs/unholy-fusion-benchmark.md)

See [§ Applying unholy-fusion for DSV4](#applying-unholy-fusion-for-dsv4) for the switching procedure.

### Older / legacy stacks

Earlier images and the v022 intermediate layers are documented separately:

| Stack | When to use | Details |
|---|---|---|
| `v021-ngc2603` / `v021-tq` | Production default for most presets (`presets/*.env` images column = `v021-ngc2603`); required for `*-tq` (TurboQuant) presets | [`docs/stack-v021.md`](docs/stack-v021.md) |
| `v022-vllm021` / `v022-tx581` / `v022-{fi0611,ngc2604,trt37,nccl234}` | v022 stack intermediates (local-build only, kept for bisection / rollback against `v022-d568`) | [`docs/stack-v022.md`](docs/stack-v022.md) |
| `v019-ngc2603` | Archived (vLLM 0.19.1 + Gemma 4 + async scheduling). Historical reproduction only. | [`docs/stack-v019.md`](docs/stack-v019.md) |

See [`CHANGELOG.md`](CHANGELOG.md) for release-by-release detail and [`PATCH_STATUS.md`](PATCH_STATUS.md) for the per-patch upstream tracking matrix.

## Supported Models

The table below covers the currently shipped presets in `presets/`. For the
complete list, see [`presets/`](presets/) — each preset file documents its own
recipe / image / topology in its header comment.

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
| `qwen3.6-27b-prismascout-nvfp4-tp2.env` (+ `-v022`) | rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm | NVFP4 mixed-precision (ViT NVFP4 + LM NVFP4 + BF16 sidecars) | dual-rdma | 2 | v022-vllm021 | MTP `n=3`; **v022 preset requires `--mm-encoder-tp-mode data`** (see Software Stack §v022) for ViT MLP K-align |
| `dsv4-flash-fp8-tp2.env` | deepseek-ai/DeepSeek-V4-Flash | FP8 (E4M3 128×128 block, official) | dual-rdma | 2 | **dsv4-d568** or **unholy-fusion** | DSV4 sparse MLA + Lightning Indexer + fp8_ds_mla KV + MTP heads. Full guide: [`docs/dsv4-flash-tp2.md`](docs/dsv4-flash-tp2.md). Alternative: unholy-fusion image (`aidendle94/sparkrun-vllm-ds4-gb10:production-ready`) — 2× prefill speedup via B12X_MOE kernel, capped at MAX_NUM_SEQS=4 / MAX_MODEL_LEN=262144. See [`docs/unholy-fusion-benchmark.md`](docs/unholy-fusion-benchmark.md). |

## Quick Start

> **Note**: The `presets/` directory contains `.env` preset files only.
> It does **not** store actual model weights. Keep model weights outside the repository
> and point `MODEL_PATH` / `MODEL_CONTAINER_PATH` to the correct host/container paths.
> See [`presets/README.md`](presets/README.md) for details.

### 0. Get the Docker Image

#### Option A: Pull pre-built image from GHCR

```bash
# Base image (all models, no TQ patches)
docker pull ghcr.io/bjk110/vllm-spark:v021-ngc2603

# TurboQuant image (base + upstream TQ bugfix patches for hybrid models)
docker pull ghcr.io/bjk110/vllm-spark:v021-tq

# Final forward-stack image (NGC 26.04 + vLLM 0.21.0+PR#35568 + FlashInfer 0.6.11.post3
# + Transformers 5.8.1 + Triton 3.7.0 + NCCL 2.30.4). General production base.
# Manifest digest: sha256:88b544ed69476f3785ea7ce37fc8b99f0f064cc299eef35cda1535c68e7a9501
docker pull ghcr.io/bjk110/vllm-spark:v022-d568

# DeepSeek-V4-Flash derivative image (FROM v022-d568, with SM12x DSV4 support).
# DSV4-specific only. See presets/dsv4-flash-fp8-tp2.env and docs/dsv4-flash-tp2.md.
docker pull ghcr.io/bjk110/vllm-spark:dsv4-d568
```

**Intermediate stacked variants are local-build only** (kept on a build node for bisection / rollback). Rebuild from source via the matching `dockerfiles/legacy/Dockerfile.v022-*` if you need to bisect:

| Tag | Dockerfile | Diff from previous layer |
|---|---|---|
| `v022-vllm021` | `dockerfiles/legacy/Dockerfile.v022` | vLLM v0.21.0 release pin (off `95995bbe`) |
| `v022-fi0611` | `dockerfiles/legacy/Dockerfile.v022-fi0611` | FlashInfer 0.6.11.post3 |
| `v022-ngc2604` | `dockerfiles/legacy/Dockerfile.v022-ngc2604` | NGC 26.04 (PyTorch 2.12.0a0) + `patch_split_module_compat.py` |
| `v022-tx581` | `dockerfiles/legacy/Dockerfile.v022-tx581` | Transformers 5.8.1 |
| `v022-trt37` | `dockerfiles/legacy/Dockerfile.v022-trt37` | Triton 3.7.0 |
| `v022-nccl234` | `dockerfiles/legacy/Dockerfile.v022-nccl234` | NCCL 2.30.4 (pip override) |
| `v022-d568` | `dockerfiles/active/Dockerfile.v022-d568` | vLLM PR #35568 cherry-pick (SM121 FP8) — **on GHCR; forward-stack validation base** |
| `dsv4-d568` | `dockerfiles/active/Dockerfile.dsv4-d568` | DeepSeek-V4-Flash derivative — `FROM v022-d568` + SM12x DSV4 vLLM patches (DSV4-specific). **On GHCR.** |

#### Option B: Build from source

```bash
# NGC 26.03 source build (vLLM main, TurboQuant included)
docker buildx build -f dockerfiles/legacy/Dockerfile.gemma4 \
  -t vllm-spark:v021-ngc2603 --load .

# vLLM v0.21.0 release-pinned source build
# (build on a Spark node only — low-RAM hosts can OOM during vLLM C++/CUDA compile)
docker buildx build -f dockerfiles/legacy/Dockerfile.v022 \
  -t vllm-spark:v022-vllm021 --load .

# Stacked-upgrade builds (each cached layer-by-layer; rebuild only the diff)
docker buildx build -f dockerfiles/legacy/Dockerfile.v022-fi0611  -t vllm-spark:v022-fi0611  --load .
docker buildx build -f dockerfiles/legacy/Dockerfile.v022-ngc2604 -t vllm-spark:v022-ngc2604 --load .
docker buildx build -f dockerfiles/legacy/Dockerfile.v022-tx581   -t vllm-spark:v022-tx581   --load .
docker buildx build -f dockerfiles/legacy/Dockerfile.v022-trt37   -t vllm-spark:v022-trt37   --load .
docker buildx build -f dockerfiles/legacy/Dockerfile.v022-nccl234 -t vllm-spark:v022-nccl234 --load .

# Active builds:
docker buildx build -f dockerfiles/active/Dockerfile.v022-d568    -t vllm-spark:v022-d568    --load .
# DeepSeek-V4-Flash derivative (FROM v022-d568 + SM12x DSV4 patches).
# Build on a Spark node; see docs/dsv4-flash-tp2.md §1.
docker buildx build -f dockerfiles/active/Dockerfile.dsv4-d568    -t vllm-spark:dsv4-d568    --load .
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
cp presets/redhatai-122b-nvfp4.env .env

# Dual Spark + RoCE:
cp presets/qwen3.5-397b-int4.env .env
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

**Hot-patch (kept on standby)** — `patches/archive/patch_codegen_fx_repr.py` rewrites `_node_ref()` to honor `__fx_repr__()` and merges its namespace into the `exec()` scope. Apply only if a future vLLM bump regresses the Inductor partition path or a different opaque type triggers the same SyntaxError:

```bash
docker exec vllm-spark-head python3 /patches/archive/patch_codegen_fx_repr.py
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

### Repository Structure

```
vllm-spark/
├── docker-compose.yml             # Unified compose (head + worker profiles)
├── entrypoints/                   # Container entrypoint scripts (selected via ENTRYPOINT_FILE — see entrypoints/README.md)
│   ├── entrypoint.sh                  # CLUSTER_MODE-aware entrypoint (standard dsv4-d568 path)
│   └── entrypoint.unholy.sh           # mp-only entrypoint for the unholy-fusion path
├── .env.example                   # Full configuration template
├── dockerfiles/                   # All Dockerfiles; see dockerfiles/README.md
│   ├── active/                        # Current active build targets (build from repo root with . as context)
│   │   ├── Dockerfile.v022-d568           # Forward-stack validation base (NGC 26.04 stack)
│   │   └── Dockerfile.dsv4-d568           # DeepSeek-V4-Flash derivative (FROM v022-d568)
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
│   └── qwen3.6-35b-fp16.env           # ⚗️ Qwen3.6 FP16 experimental (single, TP1)
├── .env.unholy-fusion             # unholy-fusion config (MAX_NUM_SEQS=4, mp backend, B12X_MOE=1)
├── docs/                          # Technical notes, stack guides, and status documents
│   ├── repository-status.md          # Current recommended paths + cleanup roadmap
│   ├── dsv4-flash-tp2.md             # DSV4-Flash: build, recipe, 9-way benchmark sweep
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

## Applying unholy-fusion for DSV4

unholy-fusion uses its own entrypoint (`entrypoints/entrypoint.unholy.sh`) and config
(`.env.unholy-fusion`). The entrypoint hardcodes the `mp` (SPMD) backend —
Ray is not used. Switching from `dsv4-d568` requires no file overwriting.

**Safe defaults** (documented in `.env.unholy-fusion`):

| Variable | Value | Notes |
|---|---|---|
| `DISTRIBUTED_BACKEND` | `mp` | hardcoded in entrypoint; Ray not used |
| `MAX_MODEL_LEN` | `262144` | 524288 starts OK but crashes at d≥131072 |
| `MAX_NUM_SEQS` | `4` | hard limit — ≥5 causes CUDA graph capture hang |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | |
| `GPU_MEMORY_UTILIZATION` | `0.80` | |
| `MTP_NUM_TOKENS` | `1` | n=2 causes catastrophic throughput collapse at c≥4; n=3 is experimental |
| `VLLM_USE_B12X_MOE` | `1` | required for 2× prefill speedup |

### Switching from dsv4-d568 → unholy-fusion (primary path)

> **GB10 UMA note**: stopping a vLLM container leaves ~100 GiB stuck in the NVIDIA driver.
> `rmmod nvidia_uvm` does **not** free it — a full reboot is required. Stop first, then reboot.

The compose override uses `compose/docker-compose.unholy.yml` and `--env-file .env.unholy-fusion`.
No files need to be copied or overwritten. The override sets `ENTRYPOINT_FILE=./entrypoints/entrypoint.unholy.sh`
which the base compose resolves without touching `entrypoints/entrypoint.sh` or `.env`.

```bash
# 1. Stop existing containers on both nodes
# On spark01:
docker compose --profile head down || true
# On spark02:
docker compose --profile worker down || true

# 2. Reboot both nodes to recover GB10 UMA memory
sudo reboot
```

After both nodes are back up:

```bash
# 3. On spark02 — start worker first
docker compose \
  -f docker-compose.yml \
  -f compose/docker-compose.unholy.yml \
  --env-file .env.unholy-fusion \
  --profile worker up -d

# 4. On spark01 — start head
docker compose \
  -f docker-compose.yml \
  -f compose/docker-compose.unholy.yml \
  --env-file .env.unholy-fusion \
  --profile head up -d

# 5. Verify on spark01
curl http://localhost:8000/health
docker logs vllm-spark-head 2>&1 | grep "Application startup complete"
docker logs vllm-spark-worker 2>&1 | grep -E "startup|ready|error" | tail -5
# Or follow live:
# docker logs -f vllm-spark-head
```

> **Startup time**: ~5 min with warm JIT cache (~60 s weight load + ~17 s profiling).
> Cold cache (first boot after image change) is significantly longer due to JIT recompilation.

#### Manual fallback only

If your Docker Compose version does not support the `${ENTRYPOINT_FILE:-}` variable in volume specs,
set `ENTRYPOINT_FILE` explicitly in your shell before running compose, or export it in `.env`:

```env
ENTRYPOINT_FILE=./entrypoints/entrypoint.unholy.sh
```

Do not overwrite entrypoint files. If a copy-based workaround is unavoidable, use the new paths
and restore immediately after testing (see "Switching back" below).

### Switching back to dsv4-d568

#### Primary override path

No files were modified, so simply stop the unholy containers and restart with the normal command:

```bash
# On spark01:
docker compose \
  -f docker-compose.yml \
  -f compose/docker-compose.unholy.yml \
  --profile head down || true

# On spark02:
docker compose \
  -f docker-compose.yml \
  -f compose/docker-compose.unholy.yml \
  --profile worker down || true

# Reboot to reclaim GB10 UMA memory, then start the normal dsv4-d568 path:
# docker compose --env-file presets/dsv4-flash-fp8-tp2.env --profile worker|head up -d
```

#### Manual fallback path

If the compose override path was used (no files were modified), no restore is needed — just start
the normal dsv4-d568 containers as shown above.

### Key differences: dsv4-d568 vs unholy-fusion

| Parameter | dsv4-d568 (primary path) | unholy-fusion (experimental) |
|---|---|---|
| Image | `ghcr.io/bjk110/vllm-spark:dsv4-d568` | `aidendle94/sparkrun-vllm-ds4-gb10:production-ready` |
| Backend | Ray (default) or mp | mp only (hardcoded) |
| MTP | configurable | n=1 recommended (n=2 broken with B12X_MOE) |
| MAX_NUM_SEQS | up to 8 | ≤ 4 (hard limit — ≥5 hangs) |
| MAX_MODEL_LEN | up to 1,000,000 | ≤ 262,144 |
| KV cache | ~11.9 GiB | ~17.1 GiB |
| Prefill burst (c=1) | ~950 t/s | ~1900–2050 t/s (2× via B12X_MOE) |
| Decode burst peak c=8, d=0 | ~116 t/s | ~112 t/s (similar burst ceiling) |
| Decode burst peak d=131072 | ~32 t/s | ~96 t/s (3×, B12X attention kernel) |
| Sustained decode d≥4k, c≥4 | collapses (MTP n=2 overhead) | collapses (KV/attention + queue under NUM_SEQS=4) |

> **Burst peak vs sustained decode**: decode values in this table are burst peak (best-of-3
> run first-tokens). Sustained total throughput collapses at long context and high concurrency
> regardless of image — see [`docs/unholy-fusion-benchmark.md`](docs/unholy-fusion-benchmark.md)
> for full depth×concurrency sweep data.

> **When to use unholy-fusion**: single or few long-context streams (`c=1–2`) where prefill
> throughput matters. For workloads requiring more than 262k context, sustained high concurrency,
> or operational stability, use `dsv4-d568`.

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

- **Preset file**: `presets/qwen3.6-35b-fp16.env`
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
# On <head_node>:
docker compose --profile head down
# On <worker_node>:
docker compose --profile worker down
# Clear unified-memory residue between model switches (GB10) — on each node:
sync && sudo sysctl -w vm.drop_caches=3
```

### Model placement

Transfer the model to your chosen Spark node before launch, then point
`MODEL_PATH` at the local copy:

```bash
# From the build/source host (~67 GB, ~6 min over the RoCE link):
rsync -av <source_dir>/Qwen/Qwen_Qwen3.6-35B-A3B/ \
    <head_node>:<spark_model_dir>/Qwen/Qwen_Qwen3.6-35B-A3B/

# On <head_node>: materialize the preset and substitute the local model root
cd <repo>
cp presets/qwen3.6-35b-fp16.env .env
sed -i "s|\[model_path\]|<spark_model_dir>/Qwen|" .env
```

### Launch (single Spark, TP=1)

On `<head_node>`:

```bash
cd <repo>
docker compose --env-file .env --profile head up -d
```

### If the first boot fails

Adjust these values in `qwen3.6-35b-fp16.env` in this order (each step lowers
memory pressure):

1. `GPU_MEMORY_UTILIZATION=0.80`
2. `MAX_MODEL_LEN=16384`
3. `MAX_NUM_SEQS=4`
4. Only if the above still fails: consider a TP=2 variant across both Spark
   nodes (no preset ships for this — this experimental preset is TP=1 only).

## Image tags & Git tags

GHCR image tags (`ghcr.io/bjk110/vllm-spark:<tag>`) and Git tags do **not**
march in lockstep yet — only `v018-ngc2603` exists as a Git tag. The mapping
below documents what each image tag corresponds to in the Git history. Use
this table when you need to reproduce or roll back to a specific image.

| Image tag | Git ref (commit) | Stack | Notes |
|---|---|---|---|
| `dsv4-d568` (active, DSV4-specific) | current HEAD | `FROM v022-d568` + SM12x DSV4 vLLM patches | DeepSeek-V4-Flash primary path; used only by `presets/dsv4-flash-fp8-tp2.env`. See [`docs/dsv4-flash-tp2.md`](docs/dsv4-flash-tp2.md). |
| `v022-d568` (active, general base) | current HEAD | NGC 26.04 + vLLM 0.21.0+PR#35568 + FlashInfer 0.6.11.post3 + Triton 3.7.0 + NCCL 2.30.4 + Transformers 5.8.1 | General production base for v022-series presets and the dsv4-d568 derivative. |
| `v021-tq` | `3070f9a` | base + TQ patches + Inductor-graph-partition fix | Required for any `*-tq.env` preset (production default for TurboQuant presets). |
| `v021-ngc2603` | `8623187` | vLLM `95995bbe` + FlashInfer `v0.6.9` | Production default for non-TQ presets (most `presets/*.env` files reference this). |
| `v020-ngc2603` (superseded) | `8efdf0b` (base-refresh-20260417 base bump) | vLLM `978a4462` + FlashInfer `v0.6.8` | Superseded by v021; only kept on GHCR for historical reproduction. |
| `v019-ngc2603` (superseded) | `7736716` (Gemma 4 + vLLM 0.19.1 upgrade) | vLLM `0.19.1` `a7d79fa` + FlashInfer `v0.6.7.post3` | Superseded by v021. |
| `v018-ngc2603` (archive) | `feb5993` (NGC 26.03 source build intro) — Git tag `v018-ngc2603` exists | vLLM `0.18.3` `c494977` + FlashInfer `v0.6.7` | The only currently-tagged release in Git. |

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
streams (base stack refresh, TurboQuant rebase, single-Spark CLUSTER_MODE,
unholy-fusion integration) have been merged in and their feature branches deleted.

### Archived branch history

The legacy TurboQuant branch is preserved as a tag for reference:

- **`archive/feat-turboquant`**

If needed, it can be restored with:

```bash
git checkout -b feat/turboquant archive/feat-turboquant
```

## License

Configuration files are provided as-is for reference. Models are subject to their respective licenses ([Qwen License](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)).
