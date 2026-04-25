# DeepSeek-V4-Flash on 2× DGX Spark (SM121 / GB10) — POC debrief

**Status: PAUSED.** Upstream blocker beyond PR #40852 scope.
**Date**: 2026-04-25
**Branch**: `feat/deepseek-v4-sm120-poc`
**Image**: `vllm-spark:deepseek-v4-sm120-poc` (sha `399a5eccd03e`, 25.6 GB,
built on spark01, transferred to spark02 over RoCE)

The image and patches in this branch reflect [vLLM PR #40852](https://github.com/vllm-project/vllm/pull/40852) on top
of the existing spark_vllm_docker base. The work is preserved as a known-good
build environment — re-enable it the moment upstream lands SM12x general FP8
GEMM support (see "Re-test triggers" below).

---

## Pinned upstream commits

| Component | Repo | Ref | Commit |
|---|---|---|---|
| vLLM source | `jasl/vllm` | branch `ds4-sm120-prototype` (PR #40852 head) | `1523228e62b92d877df05a9034fbfeb32aeaf308` (2026-04-25) |
| vLLM PR base | `vllm-project/vllm` | branch `main` | `bc2ae5a3d6b59690b6a3312f0ed63842e8bc600b` |
| DeepGEMM | `jasl/DeepGEMM` | branch `sm120` | `959f1df759ed591cadb463ab9af68222428de5df` (2026-04-25) — fetched by vLLM CMake `external_projects/deepgemm.cmake` |
| FlashInfer | `flashinfer-ai/flashinfer` | tag | `v0.6.8.post1` |
| Transformers | PyPI | | `5.5.4` |
| tilelang | GitHub release | aarch64 | `0.1.9` |
| apache-tvm-ffi | PyPI | | `0.1.9` |
| Base image | NGC | | `nvcr.io/nvidia/pytorch:26.03-py3` (CUDA 13.2, PyTorch 2.11) |

`scripts/check_sm120_upstream.sh` polls these for movement.

---

## What worked

1. **Image build (spark01)** — see `Dockerfile.deepseek-v4`. Build succeeded
   on retry after defensive `safe_float8_e8m0fnu.py` patch was made
   idempotent. Build log line confirmed:
   `DeepGEMM CUDA architectures: 12.0a`.
2. **Vendored DeepGEMM in image** —
   `/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/_C.cpython-312-aarch64-linux-gnu.so`
   loads cleanly, `import vllm.third_party.deep_gemm as dg` OK.
3. **Build-time import smoke** — `scripts/check_deepseek_v4_env.sh --build-time`
   reported **21 ok / 0 failed**, including:
   - `vllm.model_executor.models.deepseek_v4`
   - `vllm.model_executor.models.deepseek_v4_mtp`
   - `vllm.model_executor.layers.deepseek_v4_attention`
   - `vllm.model_executor.layers.mhc`
   - `vllm.tokenizers.deepseek_v4`, `..deepseek_v4_encoding`, `vllm.renderers.deepseek_v4`
   - `envs.SM120_path: 3 canonical (VLLM_TRITON_MLA_SPARSE_*) + 3 legacy (VLLM_SM120_REFERENCE_*) registered`
   - `registry: DeepseekV4ForCausalLM`
   - `tokenizer_mode: deepseek_v4`
4. **Image distribution** — `docker save vllm-spark:deepseek-v4-sm120-poc | ssh -o Compression=no 10.10.10.2 docker load`
   over RoCE (≈ 250 MB/s) completed successfully.
5. **Multi-node Ray TP=2 cluster** — head on spark01 (`10.10.10.1`),
   worker on spark02 (`10.10.10.2`) joined via RoCE, `ray status` showed
   `Active: 2 nodes, 2.0 GPU, 40 CPU, 170.04 GiB memory`.
   NCCL reported `ncclCommInitRank ... Init COMPLETE`, channels routed
   `via NET/IBext_v11/0`.
6. **Weight load** — 46/46 safetensors checkpoint shards loaded
   (≈ 100 s) without errors. Per-rank weight footprint ≈ 85 GB on each
   GB10 (TP=2 + EP=2; non-experts replicated).

## What did not work — the run loop on the engine startup

| Attempt | Knob | Outcome | Key error |
|---|---|---|---|
| 1 | `gpu_memory_utilization=0.83`, `--compilation-config FULL_AND_PIECEWISE` | KV cache memory pre-check fail | `ValueError: Free memory on device cuda:0 (33.99/121.63 GiB) on startup is less than desired GPU memory utilization (0.83, 100.95 GiB).` Cause: 80 GB of host page cache from prior 149 GB model rsync. Fixed by `sync && echo 3 > /proc/sys/vm/drop_caches`. |
| 2 | drop_caches → `gpu_memory_utilization=0.25` | torch.compile / inductor crash | `RuntimeError: cutlass_gemm_caller, …/cutlass_gemm_caller.cuh:61, Error Internal` during `profile_run` |
| 3 | + `--enforce-eager` (skip torch.compile/inductor) | Same crash | identical `cutlass_gemm_caller Error Internal` — the crash is in the C++ kernel, not in inductor codegen |
| 4 | + `patch_sm12x_force_deep_gemm.py` (rewrite `cuda.support_deep_gemm` to accept SM12x) | Routes to DeepGEMM. Crashes on weight post-process | `RuntimeError: Assertion error (.../csrc/apis/layout.hpp:59): Unknown SF transformation` inside `transform_sf_into_required_layout` |
| 5 | + `VLLM_USE_DEEP_GEMM_E8M0=0` (let `disable_ue8m0_cast=True` route SM90 branch) | Past weight post-process. Crashes during the first `fp8_gemm_nt` call | `RuntimeError: Assertion error (.../utils/layout.hpp:76): Unknown recipe` from `get_default_recipe(arch_major=12)` — function only has `arch_major == 9` and `arch_major == 10` cases |

Failure traces (preserved in this branch):
- `logs/dsv4_sm120_poc/traces/01_cutlass_scaled_mm_error_internal.txt`
- `logs/dsv4_sm120_poc/traces/01_cutlass_scaled_mm_stack.txt`
- `logs/dsv4_sm120_poc/traces/02_deepgemm_unknown_sf_transform.txt`
- `logs/dsv4_sm120_poc/traces/03_deepgemm_unknown_recipe_fp8_gemm_nt.txt`
- `logs/dsv4_sm120_poc/traces/04_kv_cache_memory_check_fail.txt`

Full container head logs:
- `logs/dsv4_sm120_poc/head_full.log` — first attempt (cutlass crash + restart loop)
- `logs/dsv4_sm120_poc/head_eager.log` — `--enforce-eager` attempt
- `logs/dsv4_sm120_poc/head_e8m0.log` — `VLLM_USE_DEEP_GEMM_E8M0=0` attempt

## Common failing call site

```
vllm/model_executor/models/deepseek_v4.py:528  self.attn(positions, x, None)
vllm/model_executor/models/deepseek_v4.py:406  return self.mla_attn(...)
vllm/model_executor/layers/deepseek_v4_attention.py:484  qr_kv, _ = self.fused_wqa_wkv(hidden_states)
vllm/model_executor/layers/linear.py:581  output_parallel = self.quant_method.apply(...)
vllm/model_executor/layers/quantization/fp8.py:476  return self.fp8_linear.apply_weights(layer, x, bias)
vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:132
   ─ via CUTLASS:  scaled_mm/cutlass.py:268  ops.cutlass_scaled_mm(...)        → Error Internal
   ─ via DeepGEMM: scaled_mm/deep_gemm.py:122 → fp8_gemm_nt → layout.hpp:76    → Unknown recipe
```

The DeepSeek-V4 `fused_wqa_wkv` projection (combined Q-projection + KV-projection
in MLA) is implemented as an **FP8 BlockScaledMM**. SM12x has no working
backend for that op:
- vLLM CUTLASS scaled_mm SM120: `ENABLE_SCALED_MM_SM120=1` is set at build
  time and `_C_stable_libtorch.abi3.so` includes `scaled_mm_sm120_fp8`,
  but the kernel returns `cutlass::Status::kErrorInternal` at runtime for
  the shapes V4 produces.
- DeepGEMM `fp8_gemm_nt`: `arch_major == 12` is **deliberately not
  implemented** in `jasl/DeepGEMM @ 959f1df`. From its commit `2206a1d7`
  message:
  > "FP4 paged MQA, **non-paged MQA, MegaMoE, and general FP8/FP4 GEMM
  > remain gated until they have separate SM12x implementations**."

  jasl's `sm120` branch added SM12x reference fallbacks for
  `tf32_hc_prenorm_gemm`, `fp8_paged_mqa_logits`, and (in `959f1df`)
  `fp8_einsum`, but **not** for `fp8_gemm_nt`.

## Re-test triggers

Re-enable this POC only when **at least one** of the following lands:

1. **`jasl/DeepGEMM` `sm120` branch** gets a new commit that adds an SM12x
   case to `csrc/utils/layout.hpp::get_default_recipe` and an SM12x
   implementation of `fp8_gemm_nt`.
2. **`vllm-project/vllm`** ships an SM12x-aware fix for
   `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh`
   so the existing `ENABLE_SCALED_MM_SM120` kernel does not return
   `Error Internal` on V4 shapes.
3. A V4-Flash variant is published with **non-FP8 attention/projection
   weights** (e.g. BF16 or W4A16) so the `fused_wqa_wkv` does not enter
   the BlockScaledMM path. (Out of repo scope.)

The build environment in this branch is otherwise ready. The only
expected rebuild step is bumping `Dockerfile.deepseek-v4`'s
`VLLM_COMMIT` and the DeepGEMM `GIT_TAG` in `cmake/external_projects/deepgemm.cmake`
(the latter is inside the vLLM source we fetch — bumping `VLLM_COMMIT`
implicitly bumps DeepGEMM since the cmake reference is checked into vLLM).

## How to revert this branch's effects on the Spark cluster

The image is local-only (not pushed to GHCR). It does not interfere with
other workflows unless you run `docker compose --profile head` after
manually setting `VLLM_IMAGE=vllm-spark:deepseek-v4-sm120-poc` in `.env`.
Other model presets (`models/redhatai-122b-nvfp4.env` etc.) point to the
production image and are unaffected.

To remove the POC image entirely:
```bash
ssh spark01 docker rmi vllm-spark:deepseek-v4-sm120-poc
ssh spark02 docker rmi vllm-spark:deepseek-v4-sm120-poc
```

## Files added/changed by this branch

- `Dockerfile.deepseek-v4` — `VLLM_REF` switched to `ds4-sm120-prototype`
- `entrypoint.sh` — added `APPLY_SM12X_DEEP_GEMM` hook
- `docker-compose.yml` — added `APPLY_SM12X_DEEP_GEMM` and `VLLM_USE_DEEP_GEMM_E8M0` plumbing
- `models/deepseek-v4-flash-tp2.env` — POC defaults (lower `GPU_MEMORY_UTILIZATION`)
- `patches/patch_sm12x_force_deep_gemm.py` — rewrites `cuda.support_deep_gemm` to accept SM12x
- `patches/safe_float8_e8m0fnu.py` — defensive (no-op on PyTorch 2.11+ that has the dtype)
- `patches/patch_deepseek_v4_config.py` — transformers 5.5.4 strict-dataclass compat
- `scripts/run_deepseek_v4_flash_sm120_poc.sh` — single-host smoke / multi-node TP=2 launcher
- `scripts/check_deepseek_v4_env.sh` — adds SM120-path env-var registration check + vendored DeepGEMM check
- `scripts/check_sm120_upstream.sh` — periodic upstream-commit watcher
- `logs/dsv4_sm120_poc/` — preserved failure traces and head logs

`Dockerfile.deepseek-v4` and `models/deepseek-v4-flash-tp2.env` are the
two files that need to change when this POC is re-attempted.
