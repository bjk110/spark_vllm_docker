# Patch Files

This directory contains patch scripts, compatibility shims, configuration helpers,
and build-time hot-patches used to build or validate DGX Spark / GB10 vLLM images.

**Not every file is active for every image.** Check the table below and the
Dockerfile comments for the correct context before using or removing any file.

Do not apply patches manually unless a Dockerfile or `README.md` explicitly
instructs you to.

---

## File classification

Status labels:

- **Active** — copied and applied by a current active Dockerfile (`dockerfiles/active/Dockerfile.v022-d568` or `dockerfiles/active/Dockerfile.dsv4-d568`).
- **Conditional** — applied at container runtime only under a specific env-var flag.
- **Standby** — not in Dockerfiles; retained for on-demand manual use and documented in `README.md` troubleshooting.
- **Historical** — referenced only in `dockerfiles/legacy/` intermediate builds, or superseded upstream (noted in Dockerfile comments); retained for bisection / rollback.
- **Unknown / needs verification** — no confirmed reference found at time of classification; likely an experiment or partial draft.

| File | Area | Status | Used by / Notes |
|---|---|---|---|
| `flashinfer_cache.patch` | FlashInfer / build | Active | `dockerfiles/active/Dockerfile.v022-d568` — patches FlashInfer build cache to prevent stale shared-library conflicts on aarch64 |
| `fix_pytorch211_compat.py` | Common / runtime | Active | `dockerfiles/active/Dockerfile.v022-d568`, `dockerfiles/active/Dockerfile.dsv4-d568` — removes PyTorch 2.11 incompatible `hoist=True` kwarg from `torch.fx` transformer calls |
| `apply_sm121_fp8_pr35568.py` | SM121 / Blackwell | Active | `dockerfiles/active/Dockerfile.v022-d568` only — build-time cherry-pick of vLLM PR #35568; widens FP8/Marlin/CUTLASS arch gates to include SM121. Confirmed live when `Selected CutlassFP8ScaledMMLinearKernel` appears in boot log |
| `apply_sm121_patches.py` | SM121 / Blackwell | Active | `dockerfiles/active/Dockerfile.v022-d568` — dispatcher that applies SM121-specific vLLM runtime patches (attention, MoE, FP8 codepaths) |
| `apply_turboquant_fixes.py` | TurboQuant | Active | `dockerfiles/active/Dockerfile.v022-d568` — applies upstream TurboQuant bugfix patches for hybrid quantization models (Gemma 4, Qwen3.5) |
| `patch_split_module_compat.py` | Common / runtime | Active | `dockerfiles/active/Dockerfile.v022-d568`, `dockerfiles/active/Dockerfile.dsv4-d568` — swaps PyTorch 2.12 static version gate around `split_module(tuple_return=True)` for an `inspect.signature` probe; needed because NGC 26.04 ships a PyTorch 2.12 alpha predating the upstream `tuple_return` commit |
| `moe_config_e256.json` | SM121 / Blackwell | Active | `dockerfiles/active/Dockerfile.v022-d568`, `dockerfiles/active/Dockerfile.dsv4-d568` — pre-tuned GB10 fused-MoE config for E=256 experts (FP8 W8A8, block 128×128) |
| `moe_config_e512.json` | SM121 / Blackwell | Active | `dockerfiles/active/Dockerfile.v022-d568`, `dockerfiles/active/Dockerfile.dsv4-d568` — pre-tuned GB10 fused-MoE config for E=512 experts |
| `apply_dsv4_packed_mapping.py` | DeepSeek-V4 / DSV4 | Active | `dockerfiles/active/Dockerfile.dsv4-d568` — adds `packed_modules_mapping` to `DeepseekV4ForCausalLM`; needed for FP8 block-scale attention to work with compressed-tensors DSV4 checkpoints; defensive (skips if already present) |
| `patch_skip_init_memory_check.py` | DeepSeek-V4 / DSV4 | Active | `dockerfiles/active/Dockerfile.dsv4-d568` — patches `vllm/v1/worker/utils.py` to skip the startup free-memory pre-check when `VLLM_SKIP_INIT_MEMORY_CHECK=1`; required on GB10 UMA where OS page-cache reclaim inflates apparent free memory |
| `patch_envs_register_skip_memcheck.py` | DeepSeek-V4 / DSV4 | Active | `dockerfiles/active/Dockerfile.dsv4-d568` — registers `VLLM_SKIP_INIT_MEMORY_CHECK` in `vllm/envs.py` so vLLM does not warn about an unknown env var |
| `patch_relax_profile_assertion.py` | DeepSeek-V4 / DSV4 | Active | `dockerfiles/active/Dockerfile.dsv4-d568` — relaxes the post-profiling free-memory assertion in `gpu_worker.py` when `VLLM_SKIP_INIT_MEMORY_CHECK=1` is active (GB10 UMA page-cache drop during profiling causes `current_free > init_free`) |
| `patch_fadvise_safetensors.py` | DeepSeek-V4 / DSV4 | Active | `dockerfiles/active/Dockerfile.dsv4-d568` — patches `weight_utils.py` to call `fadvise(POSIX_FADV_DONTNEED)` after each safetensors file is fully loaded, releasing OS page cache and recovering GB10 UMA headroom |
| `patch_qwen35_moe_text.py` | Qwen | Conditional | `entrypoint.sh` — applied at container startup when `APPLY_TEXT_ONLY_SHIM=1`; patches Qwen3.5-MoE model config to use `model_type=qwen3_5_moe_text` (flat config) to avoid the wrapper default `text_config.hidden_size=2048` |
| `patch_codegen_fx_repr.py` | Common / runtime | Standby | Not in any Dockerfile; retained as a documented hot-patch. Apply manually via `docker exec` only if a future vLLM bump regresses the Inductor partition path (`torch.fx` opaque type `__repr__` SyntaxError). See `README.md` §Troubleshooting |
| `aot_cache_fix.patch` | SM121 / Blackwell | Historical | Referenced in `dockerfiles/active/Dockerfile.v022-d568` comments as superseded upstream (AOT compile cache `source_fn_stack`/`nn_module_stack` fix merged into vLLM main). Still applied in historical `dockerfiles/legacy/` builds (`dockerfiles/legacy/Dockerfile.gemma4`, `dockerfiles/legacy/Dockerfile.ngc2603-v3`, etc.) |
| `fastsafetensors_natural_sort.patch` | Common / runtime | Historical | Referenced in `dockerfiles/active/Dockerfile.v022-d568` comments as superseded upstream (`_natural_sort_key()` merged into fastsafetensors main). Applied in historical `dockerfiles/legacy/` builds |
| `nogds_force.patch` | SM121 / Blackwell | Historical | Referenced in `dockerfiles/active/Dockerfile.v022-d568` comments as superseded upstream (GDS fallback merged into vLLM main). Applied in historical `dockerfiles/legacy/` builds |
| `qwen3_5_moe_rope_fix.py` | Qwen | Historical | Applied only in `dockerfiles/legacy/Dockerfile` and `dockerfiles/legacy/Dockerfile.ngc2603-v3` (v021 era). Fixes Qwen3.5-MoE RoPE compatibility for the NGC 26.01/26.03 stack; no longer needed in v022+ |
| `pr38423_nvfp4_spark.py` | FP8 / NVFP4 | Historical | Applied only in `dockerfiles/legacy/Dockerfile` (v021 NGC 26.01 era). NVFP4 GB10 runtime patch corresponding to vLLM PR #38423; superseded by the v022 FlashInfer-CUTLASS NVFP4 path |
| `apply_hotpatch.sh` | build helper | Unknown / needs verification | Shell wrapper for manually hot-patching running containers (applies AOT cache fix, nogds, MoE tuning). No Dockerfile reference; likely a manual recovery tool from early SM121 bringup. Verify before removing |
| `apply_patch2_fp8_moe.py` | FP8 / NVFP4 | Unknown / needs verification | FP8 block-scale MoE enablement on SM121 / GB10. No Dockerfile reference; may be an early experiment predating `apply_sm121_patches.py`. Verify before removing |
| `apply_patches_in_container.py` | build helper | Unknown / needs verification | Manual patch orchestrator designed to run inside a container (`docker exec`). No Dockerfile reference. Likely an early SM121 bringup tool before patches were baked into Dockerfiles |
| `apply_patches_round2.py` | build helper | Unknown / needs verification | Batch patch tool for a second round of SM121 patches (PR #2, #3, #6, #7, #9 in early numbering). No Dockerfile reference; predates current unified `apply_sm121_patches.py` |
| `fix_cuda13_memcpy_batch.py` | SM121 / Blackwell | Unknown / needs verification | Patches `cuMemcpyBatchAsync` API change in CUDA 13.0+ (removed `failIdx` parameter). No Dockerfile reference. May be a forward-looking patch for a future CUDA upgrade or an unreleased experiment |

---

## Cleanup policy

Do not delete unreferenced patches solely because `grep` does not find them in
Dockerfiles. Some files may be retained for:

- reproducing older builds or forum/debug experiments
- bisect experiments across image versions
- archived test branches
- manual recovery flows when a running container needs an emergency fix
- tracking upstream issues before a proper upstream fix lands

Mark candidates as **Historical** or **Unknown / needs verification** first, then
remove them only in a later cleanup after confirming they are not needed for any
of the above purposes.

---

## Future cleanup

A later structural cleanup may split this directory by purpose:

- `common/` — stack-agnostic PyTorch / Python compat shims
- `sm121/` — Blackwell / GB10-specific kernel and runtime patches
- `dsv4/` — DeepSeek-V4 / DSV4-specific patches
- `qwen/` — Qwen-family patches
- `turboquant/` — TurboQuant upstream fixes
- `archive/` — historical patches retained only for bisection

That migration should update all Dockerfile `COPY` directives and `README.md`
references in the same commit.
