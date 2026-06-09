# Software Stack

This document describes the full image/stack lineage used by this repository:
the primary DeepSeek-V4-Flash derivative (`dsv4-d568`), the forward-stack
validation base it is built on (`v022-d568`), the `v021` production-default
series, and older/legacy stacks.

The top-level [`README.md`](../README.md) only lists the current recommended
serving paths — see [`README.md` § Current serving paths](../README.md#current-serving-paths)
for the short summary.

## Current stack summary

Current image roles:
- `v021-ngc2603`: stable base for most existing presets (non-TQ)
- `v021-tq`: TurboQuant preset base (required for `*-tq.env` presets)
- `v022-d568`: stable general base (NGC 26.04 + vLLM 0.21.0) for Qwen3.6/Gemma/abliterix presets
- `v022-d568-ngc2605-tx5102-vllm022`: active forward-stack (NGC 26.05 + vLLM 0.22.1 + FlashInfer v0.6.12 + Transformers 5.10.2)
- `dsv4-d568`: primary DeepSeek-V4-Flash path — **frozen, not rebased onto NGC 26.05**
- `unholy-fusion`: experimental high-prefill DeepSeek-V4-Flash path

`unholy-fusion` is a third-party image with custom GB10 (Blackwell sm_120/sm_121)
kernels (B12X_MOE etc.) not present in `dsv4-d568`. Its full stack/configuration
detail, operational limits, and benchmark comparison live in
[`docs/unholy-fusion-benchmark.md`](unholy-fusion-benchmark.md) rather than in
this document — see [Relationship to images and presets](#relationship-to-images-and-presets)
below.

## dsv4-d568 — Primary DeepSeek-V4-Flash path

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

**Full guide + 9-way benchmark sweep + MTP/backend analysis**: [`docs/dsv4-flash-tp2.md`](dsv4-flash-tp2.md).

> **DSV4 path summary**: For DeepSeek-V4-Flash, use `dsv4-d568` as the primary path. For users who specifically want higher prefill throughput, `unholy-fusion` is available as an experimental alternative (see [`docs/unholy-fusion-benchmark.md`](unholy-fusion-benchmark.md)). Earlier jasl-based DSV4 image notes are deferred and kept only for historical reference.

## v022-d568-ngc2605-tx5102-vllm022 (NGC 26.05, vLLM 0.22.1, FlashInfer v0.6.12, Transformers 5.10.2) — active forward-stack

| Component | Version |
|---|---|
| Base Image | NGC PyTorch **26.05-py3** (CUDA 13.2) |
| vLLM | **0.22.1** (commit `ad7125a431e` — v0.22.1 + DSV4 MTP HC bugfix PR#42320, source rebuild) |
| FlashInfer | **v0.6.12** (SM12x kernel fixes, DGX Spark CI fix) |
| PyTorch | **2.12.0a0** |
| CUDA | 13.2 (native) |
| Transformers | **5.10.2** |
| Triton | **3.7.0** |
| NCCL | **2.30.4+cuda13.2** (bundled in NGC 26.05) |
| Image tag | `ghcr.io/bjk110/vllm-spark:v022-d568-ngc2605-tx5102-vllm022` (**on GHCR**, digest `sha256:2c52c885e48c`) |

Verified preset: Qwen3.5-122B-A10B-FP8 (TP=2, ray, MAX_MODEL_LEN=32768, KV cache ~45 GiB). Korean QA and English math verified. `--reasoning-parser qwen3` separates reasoning field correctly.

> **Image policy**: This image is NOT a replacement for `dsv4-d568`. It is a separate forward-stack path for non-DSV4 models. `dsv4-d568` is frozen and will not be rebased onto this stack.

Required `.env` variables beyond standard presets (NGC 26.05 specifics):

```
NCCL_NET=Socket                              # HPC-X RDMA plugin init() fails in NGC 26.05; TCP fallback
FLASHINFER_CUDA_ARCH_LIST=12.1               # NGC 26.05 base exports "" causing FlashInfer crash
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1   # docker-compose :- default passes "" to int()
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0          # v2 executor not yet stable on GB10
VLLM_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/libnccl.so.2
```

> **Note**: `VLLM_SKIP_INIT_MEMORY_CHECK` is not present in vLLM 0.22.1. If the GB10
> UMA memory was not fully released from a prior container run, the startup
> `request_memory()` check will fail. Reboot is the only recovery.

## v022-d568 (NGC 26.04, vLLM v0.21.0+#35568, FlashInfer 0.6.11.post3, Transformers 5.8.1, Triton 3.7.0, NCCL 2.30.4) — stable general base

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

For detailed stack validation notes, intermediate image list, runtime patches, and verified preset
overrides, see [`docs/model-serving-validation-history.md`](model-serving-validation-history.md).

## v021 series

| Stack | When to use | Details |
|---|---|---|
| `v021-ngc2603` / `v021-tq` | Production default for most presets (`presets/*.env` images column = `v021-ngc2603`); required for `*-tq` (TurboQuant) presets | [`docs/stack-v021.md`](stack-v021.md) |

## Legacy stacks

Earlier images and the v022 intermediate layers are documented separately:

| Stack | When to use | Details |
|---|---|---|
| `v022-vllm021` / `v022-tx581` / `v022-{fi0611,ngc2604,trt37,nccl234}` | v022 stack intermediates (local-build only, kept for bisection / rollback against `v022-d568`) | [`docs/stack-v022.md`](stack-v022.md) |
| `v019-ngc2603` | Archived (vLLM 0.19.1 + Gemma 4 + async scheduling). Historical reproduction only. | [`docs/stack-v019.md`](stack-v019.md) |

See [`CHANGELOG.md`](../CHANGELOG.md) for release-by-release detail and [`PATCH_STATUS.md`](../PATCH_STATUS.md) for the per-patch upstream tracking matrix.

## Relationship to images and presets

- Each `presets/*.env` file documents which image/stack it expects, both in its
  header comment and in the "Image" column of
  [`README.md` § Presets and model paths](../README.md#presets-and-model-paths).
- `v021-ngc2603` / `v021-tq` are the base for most existing (non-`v022`) presets.
- `v022-d568` is the stable general base for `v022-*` presets (Qwen3.6, Gemma 4 31B,
  abliterix NVFP4) and is the base on which `dsv4-d568` was originally built.
- `v022-d568-ngc2605-tx5102-vllm022` is the active forward-stack for new models.
- `dsv4-d568` is used only by `presets/dsv4-flash-fp8-tp2.env`.
- `unholy-fusion` serves the same model/preset via its own override path
  (`.env.unholy-fusion` + `compose/docker-compose.unholy.yml`) rather than by
  copying a preset to `.env` — see [`docs/unholy-fusion-benchmark.md`](unholy-fusion-benchmark.md).

For exact image tags, digests, and Git-ref → image mapping, see [`docs/images.md`](images.md).
