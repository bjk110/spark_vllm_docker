# Container Images

This document describes the container image tags used by this repository and
how they relate to runtime presets and Git history.

The top-level README only lists the current recommended runtime paths. This
document provides the detailed image-tag history.

## Current recommended paths

| Path | Status | Config | Image source |
|---|---|---|---|
| `dsv4-d568` | **Frozen** primary DSV4 baseline (NOT rebased onto NGC 26.05) | `presets/dsv4-flash-fp8-tp2.env` | `ghcr.io/bjk110/vllm-spark:dsv4-d568` — see the image mapping below |
| `unholy-fusion` | Experimental high-prefill DSV4 path | `.env.unholy-fusion` + `compose/docker-compose.unholy.yml` | External/upstream image (`aidendle94/sparkrun-vllm-ds4-gb10:production-ready`) or GHCR mirror — see [`docs/unholy-fusion-benchmark.md`](unholy-fusion-benchmark.md) |
| `v022-d568-ngc2605-tx5102-vllm022` | Active forward-stack (NGC 26.05 + vLLM 0.22.1) | `.env` with `VLLM_IMAGE=ghcr.io/bjk110/vllm-spark:v022-d568-ngc2605-tx5102-vllm022` | `ghcr.io/bjk110/vllm-spark:v022-d568-ngc2605-tx5102-vllm022` |
| `v022-d568` | Stable general base (NGC 26.04 + vLLM 0.21.0) | Various `presets/*.env` | `ghcr.io/bjk110/vllm-spark:v022-d568` |

## Image tag mapping

GHCR image tags (`ghcr.io/bjk110/vllm-spark:<tag>`) and Git tags do **not**
march in lockstep yet — only `v018-ngc2603` exists as a Git tag. The mapping
below documents what each image tag corresponds to in the Git history. Use
this table when you need to reproduce or roll back to a specific image.

| Image tag | Git ref (commit) | Stack | Notes |
|---|---|---|---|
| `v022-d568-ngc2605-tx5102-vllm022` (active, forward-stack) | `022924e` | NGC 26.05 + vLLM 0.22.1 (`ad7125a431e`) + FlashInfer v0.6.12 + Transformers 5.10.2 | GHCR digest `sha256:2c52c885e48c`. Validated: Qwen3.5-122B-A10B-FP8 (TP=2, ray). Requires `NCCL_NET=Socket`, `FLASHINFER_CUDA_ARCH_LIST=12.1`. |
| `v022-d568-ngc2605-tx5102-vllm022-step3p7` (active, Step-3.7-Flash) | `FROM v022-d568-ngc2605-tx5102-vllm022` | + `patches/patch_step3p7_nvfp4_input_scale.py` (NVFP4 `expert_params_mapping` fix) | Step-3.7-Flash FP8/NVFP4 dual-rdma TP=2. FP8 path uses standard `fp8.py` (patch is a no-op there); NVFP4 path requires the patch to avoid NaN logits. See [`docs/step3.7-flash-tp2.md`](step3.7-flash-tp2.md). |
| `dsv4-d568` (active, DSV4-specific) | repository HEAD at build time | `FROM v022-d568` + SM12x DSV4 vLLM patches | DeepSeek-V4-Flash primary path; used only by `presets/dsv4-flash-fp8-tp2.env`. See [`docs/dsv4-flash-tp2.md`](dsv4-flash-tp2.md). **Frozen — NOT rebased onto NGC 26.05.** |
| `v022-d568` (active, stable general base) | repository HEAD at build time | NGC 26.04 + vLLM 0.21.0+PR#35568 + FlashInfer 0.6.11.post3 + Triton 3.7.0 + NCCL 2.30.4 + Transformers 5.8.1 | Stable base for Qwen3.6, Gemma 4 31B, abliterix NVFP4 presets. Digest `sha256:88b544ed`. |
| `v021-tq` | `3070f9a` | base + TQ patches + Inductor-graph-partition fix | Required for any `*-tq.env` preset (legacy preset base for TurboQuant presets). |
| `v021-ngc2603` | `8623187` | vLLM `95995bbe` + FlashInfer `v0.6.9` | Retained for older presets — most non-TQ `presets/*.env` files reference this. |
| `v020-ngc2603` (superseded) | `8efdf0b` (base-refresh-20260417 base bump) | vLLM `978a4462` + FlashInfer `v0.6.8` | Superseded by v021; only kept on GHCR for historical reproduction. |
| `v019-ngc2603` (superseded) | `7736716` (Gemma 4 + vLLM 0.19.1 upgrade) | vLLM `0.19.1` `a7d79fa` + FlashInfer `v0.6.7.post3` | Superseded by v021. |
| `v018-ngc2603` (archive) | `feb5993` (NGC 26.03 source build intro) — Git tag `v018-ngc2603` exists | vLLM `0.18.3` `c494977` + FlashInfer `v0.6.7` | The only currently-tagged release in Git. |

## Notes

- GHCR image tags and Git tags may not always move in lockstep.
- Prefer release notes and image digests when exact image reproducibility
  matters — see this repository's GitHub Releases for independently verified
  digests.
- Some older tags are retained for historical reproduction only.
- Do not infer current recommended runtime settings from raw image tags alone.
- For maintainer-only Git tag creation and archived branch notes, see
  [`docs/release-management.md`](release-management.md).
