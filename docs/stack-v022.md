# Software Stack — v022 series (forward-stack lineage)

The v022 series is the NGC 26.04 / vLLM v0.21.0 forward-stack. The final layer,
`v022-d568`, is the current main image and is documented in the top-level
[README](../README.md#software-stack). The intermediate variants here are
**local-build only** (kept on a build node for bisection / rollback;
not pushed to GHCR).

| Tag | Dockerfile | Stack diff |
|---|---|---|
| `v022-vllm021` | `dockerfiles/legacy/Dockerfile.v022` | vLLM v0.21.0 release pin off `95995bbe` (NGC 26.03 base) |
| `v022-fi0611` | `dockerfiles/legacy/Dockerfile.v022-fi0611` | + FlashInfer 0.6.11.post3 |
| `v022-ngc2604` | `dockerfiles/legacy/Dockerfile.v022-ngc2604` | + NGC 26.04 (PyTorch 2.12.0a0) + `patch_split_module_compat.py` |
| `v022-tx581` | `dockerfiles/legacy/Dockerfile.v022-tx581` | + Transformers 5.8.1 |
| `v022-trt37` | `dockerfiles/legacy/Dockerfile.v022-trt37` | + Triton 3.7.0 |
| `v022-nccl234` | `dockerfiles/legacy/Dockerfile.v022-nccl234` | + NCCL 2.30.4 (pip override) |
| `v022-d568` (current) | `dockerfiles/active/Dockerfile.v022-d568` | + vLLM PR #35568 cherry-pick (SM121 FP8). **Only this is on GHCR.** |

## v022-vllm021 — NGC 26.03 / vLLM v0.21.0 release-pinned

The foundation of the series. Built from `dockerfiles/legacy/Dockerfile.v022`, pinned
to the vLLM v0.21.0 release tag (`ad7125a4`). Three upstream-absorbed runtime
patches drop out of the build (`aot_cache_fix.patch`,
`fastsafetensors_natural_sort.patch`, `nogds_force.patch`). Preset overrides
live alongside the base preset as `presets/*-v022.env`. Use this image to
validate behavior on the released v0.21.0 before bumping the default image
off `95995bbe`.

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.03 |
| vLLM | **0.21.0** (release tag, commit `ad7125a4`, source build) |
| FlashInfer | v0.6.9 (same as v021-ngc2603) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (native) |
| Transformers | 5.5.4 |
| Image tag | `ghcr.io/bjk110/vllm-spark:v022-vllm021` (local-build only) |

### Verified presets (v022 override env files)

| Override env | Model | Notes |
|---|---|---|
| `presets/wangzhang-122b-abliterix-fp8-tp2-v022.env` | wangzhang/Qwen3.5-122B-A10B-abliterix (FP8) | text-only shim, dual-rdma TP=2 |
| `presets/qwen3.6-27b-prismascout-nvfp4-tp2-v022.env` | rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16 | NVFP4 mixed-precision, **adds `--mm-encoder-tp-mode data`** so the ViT MLP fc2 (`hidden=4304`) is not split across TP=2 (would yield K=2152, breaking NVFP4 GEMM K-align(16)); MTP `n=3`, dual-rdma TP=2 |

### Caveat — AOT compile cache poisoning across config changes

vLLM persists AOT-compiled forward functions in
`./.cache/<preset>/torch_compile_cache/torch_aot_compile/`. Switching a
preset's CLI args in a way that changes the encoder profile path (e.g.
toggling `--mm-encoder-tp-mode` or `--limit-mm-per-prompt`) without clearing
the cache can surface a `'NoneType' object has no attribute 'size'` failure
deep inside the compiled forward (qwen3_next.py / qwen3_5.py). Workaround:

```bash
sudo mv .cache/<preset>/torch_compile_cache .cache/<preset>/torch_compile_cache.backup_$(date +%s)
```

on both nodes, then restart. Fresh compile takes ~2-3 min (vs <10s with a
warm cache).

## v022-tx581 — NGC 26.04 / vLLM v0.21.0 / FlashInfer v0.6.11.post3 / Transformers 5.8.1

Intermediate image in the stacked-upgrade chain; superseded by `v022-trt37` →
`v022-nccl234` → `v022-d568` (the latter is the current main). Kept for
bisection. Triton 3.6.0 / NCCL 2.29.7 here (older than the final `-d568`).
