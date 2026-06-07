# Software Stack — v021-ngc2603 (previous main, NGC 26.03)

Superseded by [v022-d568](software-stack.md) on the forward-stack
lineage. `v021-ngc2603` remains the **production default for non-TQ presets**
in this repo (preset table column "Image: v021-ngc2603") because it's the
most-tested broad-coverage image. Use v022-d568 to validate v0.21.0 release
plus the cherry-pick.

`v021-tq` is the TurboQuant sibling that adds the
`apply_turboquant_fixes.py` patch stack; required by any `*-tq.env` preset.

## v021-ngc2603

vLLM main bumped from `978a4462` to **`95995bbe`** (+236 commits incl.
upstream merges of TQ backend selection #40060, FA3/FA4 prefill #40092,
prior-art random-signs cleanup #40194). FlashInfer bumped
**v0.6.8 → v0.6.9** with SM121 b12x FP4 GEMM (#3113) and b12x CuTe DSL fused
MoE for SM120 (#3066). TurboQuant enables 2-4x KV cache capacity via
`--kv-cache-dtype turboquant_k8v4`.

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.03 |
| vLLM | 0.20.0.dev (main `95995bbe`, source build, TurboQuant included) |
| FlashInfer | v0.6.9 (SM121 b12x FP4 GEMM, b12x CuTe DSL MoE, source build) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (native) |
| NCCL | 2.29.7 |
| Python | 3.12 |
| Transformers | 5.5.4 |
| `_C_stable_libtorch` | Included (NVFP4 / FP8 / CUTLASS full ops) |
| Image tag | `ghcr.io/bjk110/vllm-spark:v021-ngc2603` |

## v021-tq

Same base as `v021-ngc2603` plus the TurboQuant cherry-pick stack
(`patches/turboquant/apply_turboquant_fixes.py`). Required by every `*-tq.env` preset
(e.g. `gemma4-26b-a4b-tq.env`, `redhatai-122b-nvfp4-tq.env`,
`qwen3.5-397b-int4-tq.env`).

| Component | Version |
|---|---|
| Base | identical to v021-ngc2603 |
| Additional patches | `apply_turboquant_fixes.py` (PRs #40074, #39988, #39931, #40060, #40092 — some upstreamed since) |
| Image tag | `ghcr.io/bjk110/vllm-spark:v021-tq` |

Build:

```bash
docker buildx build -f dockerfiles/legacy/Dockerfile.gemma4 \
  -t vllm-spark:v021-ngc2603 --load .
# (v021-tq is built by layering the apply_turboquant_fixes.py patch on top
#  during the runner stage; see CHANGELOG for the exact recipe.)
```
