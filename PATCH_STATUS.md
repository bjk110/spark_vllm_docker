# Patch Status

Inventory of `patches/` files for `vllm-spark`. The "Required for" column
captures which presets / features actually trigger the patch, and the
"Removal condition" column captures the upstream signal that lets us drop
the patch from `Dockerfile.gemma4` (or the entrypoint).

If a removal condition says **Unknown / verify before removal**, the
maintainer should reproduce the failure case before dropping the patch.

## Images and pinned vLLM commits

| Image tag | Built from | vLLM commit | Notes |
|---|---|---|---|
| `ghcr.io/bjk110/vllm-spark:v021-tq` | `Dockerfile.gemma4` | `95995bbe` (vLLM pre-0.21 dev) | Default / production image |
| `ghcr.io/bjk110/vllm-spark:v022-vllm021` | `Dockerfile.v022` | `ad7125a431e176d4161099480a66f0169609a690` (vLLM **v0.21.0** tag) | Forward-looking release-pinned build. Drops `aot_cache_fix.patch`, `fastsafetensors_natural_sort.patch`, `nogds_force.patch` (absorbed upstream by v0.21.0). Preset overrides live in `models/*-v022.env` (`wangzhang-122b-abliterix-fp8-tp2-v022.env`, `qwen3.6-27b-prismascout-nvfp4-tp2-v022.env`). |
| `ghcr.io/bjk110/vllm-spark:v022-fi0611` | `Dockerfile.v022-fi0611` | same v0.21.0 | **Stacked bump #1** on v022-vllm021. FlashInfer `v0.6.9 → v0.6.11.post3` (98 commits incl. SM120/121 XQA MLA bug fixes #2689, CUTLASS Small Tile N Blockscaled GEMMs for SM120/121 #3152, Blackwell GDN accuracy #3156, SM120 cuDNN NaN #3192, NVFP4 KV prefill+batch attention #3097). PrismaSCOUT NVFP4 TP=2 validated text + image. |
| `ghcr.io/bjk110/vllm-spark:v022-ngc2604` | `Dockerfile.v022-ngc2604` | same v0.21.0 | **Stacked bump #2** on v022-fi0611. NGC base `26.03-py3 → 26.04-py3` (PyTorch `2.11.0a0 → 2.12.0a0`). Adds **`patch_split_module_compat.py`** — NGC 26.04's PyTorch 2.12 alpha snapshot lacks `tuple_return` kwarg on `torch.fx.passes.split_module.split_module()`, but vLLM's static `is_torch_equal_or_newer("2.12.0.dev")` check returns True and tries to pass it. Patch swaps the version check for `inspect.signature(...).parameters` probe so the kwarg is only added when actually accepted. Validated text + image. |
| `ghcr.io/bjk110/vllm-spark:v022-tx581` | `Dockerfile.v022-tx581` | same v0.21.0 | **Stacked bump #3** on v022-ngc2604. Transformers `5.5.4 → 5.8.1` (5.6.1 / 5.6.2 / 5.7.0 / 5.8.0 / 5.8.1 — 4 minors). Only deprecation warnings emitted (`Qwen2VLImageProcessorFast`, `use_fast` kwarg); AOT compile cache reused unchanged from v022-ngc2604. Validated text + image. |
| `ghcr.io/bjk110/vllm-spark:v022-trt37` | `Dockerfile.v022-trt37` | same v0.21.0 | **Stacked bump B** on v022-tx581. Triton `3.6.0 → 3.7.0` (NGC 26.04 bundles 3.6.0; vanilla PyPI wheel overrides). PyTorch 2.12 `torch._inductor` still loads; FLA GDN prefill kernel selected. `tl.make_block_ptr` deprecation warning emitted but works. Validated text + image. |
| `ghcr.io/bjk110/vllm-spark:v022-nccl234` | `Dockerfile.v022-nccl234` | same v0.21.0 | **Stacked bump C** on v022-trt37. NCCL `2.29.7 → 2.30.4` via `nvidia-nccl-cu13==2.30.4` pip + `LD_LIBRARY_PATH` override (NGC 26.04 still bundles 2.29.7 under `/usr/lib/aarch64-linux-gnu/`). `ldd libtorch_cuda.so` confirms PyTorch loads the pip lib at runtime; `ncclGetVersion()` returns 23004 (=2.30.4). RDMA TP=2 + IBext_v11 plugin compatible. `torch.cuda.nccl.version()` still reports compile-time (2,29,7) — cosmetic, ignore. Validated text + image. |
| `ghcr.io/bjk110/vllm-spark:v022-d568` | `Dockerfile.v022-d568` | v0.21.0 + cherry-pick of `06d020bb6` | **Stacked bump D** on v022-nccl234. Cherry-picks vLLM PR #35568 ("Fix SM121 (DGX Spark) exclusion from Marlin/CUTLASS FP8 paths"). Requires vLLM C++ wheel recompile — `patches/apply_sm121_fp8_pr35568.py` widens four `enable_sm120_only`/`arch in [89, 120]` checks to `SM12x family` in `csrc/libtorch_stable/.../scaled_mm{,_sm120_fp8_dispatch}.cuh`, `csrc/moe/marlin_moe_wna16/{generate_kernels.py, ops.cu}`, `csrc/quantization/marlin/generate_kernels.py`. Wheel rebuild ~217s with ccache. Validated: PrismaSCOUT NVFP4 (regression-clean) **and** abliterix-FP8 TP=2 which now reports `Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8` — the SM12x FP8 path the PR unlocks. Final stacked image for the 2026-05-18 upgrade test run. **Pushed to GHCR** 2026-05-18 (digest `sha256:88b544ed`). The five intermediate stacked images (`v022-vllm021`, `-fi0611`, `-ngc2604`, `-tx581`, `-trt37`, `-nccl234`) are **local-build only** — kept on spark01/spark02 for bisection but never pushed. |

## Active patches (applied at build time by `Dockerfile.gemma4`)

| Patch file | Purpose | Applies to | Required for | Upstream status | Removal condition |
|---|---|---|---|---|---|
| `fix_pytorch211_compat.py` | Removes the `hoist=True` kwarg from `torch._library.utils.register_opaque_type()` calls in vLLM source — the kwarg was deleted in PyTorch 2.11 | `/workspace/vllm-src/vllm/utils/torch_utils.py` (build time, before wheel build) | NGC 26.03 (PyTorch 2.11) builds of any vLLM commit older than the upstream fix | Open (vLLM has not deleted the `hoist=True` kwarg upstream as of `95995bbe`, 2026-04-25) | vLLM commit removes the `hoist=True` argument; verify by grepping `register_opaque_type` in vLLM main and confirming no `hoist=` remains |
| `fastsafetensors_natural_sort.patch` | Sorts shard filenames with natural-number ordering before `fastsafetensors` round-robin assigns them across TP ranks; otherwise lexicographic ordering puts `model-00010-of-00012.safetensors` before `model-00002-...` and weights load to the wrong rank | `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py` (runtime, build-baked) | Multi-shard checkpoints with ≥10 shards on multi-node TP (e.g. 397B INT4 TP=2, hybrid Qwen3.5 with `fastsafetensors` enabled) | **Absorbed in v0.21.0** — `Dockerfile.v022` drops this patch. Gemma4 image still applies it. | Bump `Dockerfile.gemma4` past v0.21.0 (then this patch can be dropped from gemma4 too) |
| `aot_cache_fix.patch` | Drops every `node.meta` value that contains a raw `torch.fx.Node` reference, plus a serializer fallback that returns `None` for stray Node objects, before `GraphPickler` runs. Without this, AOT cache pickling raises `Unexpected raw Node during pickling` on PyTorch nightlies / NGC 26.03 | `/usr/local/lib/python3.12/dist-packages/vllm/compilation/caching.py` (runtime, build-baked) | Any preset that exercises the AOT cache (i.e. anything using `torch.compile` + persistent cache — basically all current models) | **Absorbed in v0.21.0** — `Dockerfile.v022` drops this patch. Gemma4 image still applies it. | Same — bump gemma4 past v0.21.0 |
| `nogds_force.patch` | Forces `nogds=True` in `fastsafetensors_weights_iterator()`. GB10 has no GDS support, but vLLM's default `nogds = pg.size() > 1` enables GDS for single-rank single-Spark loads, which then fails to open `cuFileDriverOpen()` | `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py` (runtime, build-baked) | Any single-Spark / TP=1 preset that uses `fastsafetensors` weight loading | **Absorbed in v0.21.0** — `Dockerfile.v022` drops this patch. Gemma4 image still applies it. | Same — bump gemma4 past v0.21.0 |
| `apply_sm121_patches.py` | Bundles three SM121-specific runtime fixes from `seli-equinix/vllm:feature/sm121-gb10-support` — (#3) splits `has_flashinfer_nvfp4` from `has_flashinfer_cutlass_fused_moe`, (#6) auto-configures `TRITON_PTXAS_PATH` for SM121, (#9) adds `is_blackwell_class()` covering SM10x/SM11x/SM12x | `/usr/local/lib/python3.12/dist-packages/vllm/{utils,platforms,model_executor/...}` (build time) | All SM121 / GB10 builds — every preset depends on these | Open — none of these have been merged into vLLM main as of `95995bbe` | Each individual change merges upstream (track the three sub-changes separately; the script logs each) |
| `moe_config_e256.json` | GB10-tuned fused-MoE kernel config for E=256 (256 experts, e.g. Qwen3.5 122B-A10B variants) at FP8 W8A8, block_shape `[128,128]` | `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_GB10,...json` | Qwen3.5 122B-A10B FP8 (multimodal + abliterated FP8 + INT4 + NVFP4) | Permanent — these are device-specific tuning artifacts; vLLM ships per-device tuned configs | None (delete only if the model class stops using fused-MoE on this device, which would be unusual) |
| `moe_config_e512.json` | GB10-tuned fused-MoE kernel config for E=512 (512 experts, e.g. Qwen3.5 397B-A17B) | Same path as above with `E=512` | Qwen3.5 397B-A17B INT4 / future 512-expert MoE | Permanent (same reasoning) | None |
| `apply_turboquant_fixes.py` | Cherry-picks open vLLM PRs needed for TurboQuant on DGX Spark / Qwen3.5 hybrid: PR #40074 (Triton decode index OOB), #39988 (BF16 FP8 cast), #39931 (hybrid model support). Also documents two PRs already merged upstream (#40060, #40092) so the next bump can drop those entries | `/usr/local/lib/python3.12/dist-packages/vllm/...` (runtime, build-baked into `v021-tq` only) | Any `*-tq.env` preset (`gemma4-26b-a4b-tq.env`, `redhatai-122b-nvfp4-tq.env`, `qwen3.5-397b-int4-tq.env`) | Open — three PRs (40074 / 39988 / 39931) still open upstream | All three PRs merge to vLLM main, then bump `VLLM_COMMIT` past their merge SHAs and delete the patch |
| `patch_split_module_compat.py` | Replaces vLLM's static `is_torch_equal_or_newer("2.12.0.dev")` gate around `torch.fx.passes.split_module.split_module(tuple_return=True)` with a runtime `inspect.signature(...).parameters` probe. NGC 26.04's PyTorch 2.12.0a0 alpha is *newer* than `2.12.0.dev` by version-string ordering but *older* than the upstream commit that added `tuple_return`, so the version gate fires and PyTorch raises `TypeError: split_module() got an unexpected keyword argument 'tuple_return'` | `/usr/local/lib/python3.12/dist-packages/vllm/compilation/backends.py` (runtime, build-baked into `v022-ngc2604`, `v022-tx581`, `v022-trt37`, `v022-nccl234`, `v022-d568`) | Any image built on NGC 26.04-py3 with vLLM v0.21.0 (or any vLLM that gates `tuple_return` on a static version check) | Open — needs either an NGC PyTorch snapshot that already has `tuple_return` in `split_module`, or a vLLM commit that probes the signature instead of the version | Either NGC bumps to a PyTorch snapshot past the upstream `tuple_return` commit, **or** vLLM's `backends.py` learns to probe `split_module`'s signature directly |
| `apply_sm121_fp8_pr35568.py` | **Build-time** cherry-pick of vLLM PR #35568 (commit `06d020bb6`) onto a vLLM source checkout *before* `bdist_wheel`. Widens four `enable_sm120_only` / `arch in [89, 120]` / `major_capability * 10 + minor_capability == 120` gates to `SM12x family` so the DGX Spark GB10 (SM121) gets compiled into the Marlin/CUTLASS FP8 kernel variants. Five C++ edits: `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm.cuh`, `.../scaled_mm_sm120_fp8_dispatch.cuh`, `csrc/moe/marlin_moe_wna16/{generate_kernels.py, ops.cu}`, `csrc/quantization/marlin/generate_kernels.py`. The PR also touches `vllm/model_executor/layers/quantization/utils/marlin_utils.py`, but the v0.21.0 source is laid out differently — the Python edit is best-effort and prints a warning if the marker is missing; the C++ side is sufficient to activate the SM121 FP8 path | `/workspace/vllm-src/csrc/...` (build-time, before wheel build; baked into `v022-d568`) | Any FP8 preset on GB10 (`wangzhang-122b-abliterix-fp8-tp2*.env` confirmed; future GLM/Qwen FP8 variants) | Drop once `VLLM_COMMIT` is bumped past `06d020bb6` (already merged to vLLM main) | `VLLM_COMMIT` in `Dockerfile.v022*` advances past `06d020bb6` |

## Step-3.7-Flash patches (applied at build time in `Dockerfile.v022-d568-ngc2605-step3p7-fi-aot`)

| Patch file | Purpose | Affected class / file | Upstream status | Removal condition |
|---|---|---|---|---|
| `patch_registry_step3p7.py` | Registers the Step-3.7-Flash model class (`Step3p7ForConditionalGeneration`) in the vLLM model registry so the model can be loaded by name | `vllm/model_executor/models/registry.py` | Not upstreamed; model class registration handled locally | Remove if Step-3.7-Flash is natively registered in the base vLLM image |
| `patch_step3p7_nvfp4_input_scale.py` | Adds missing `.input_scale` entries to `expert_params_mapping` for ModelOpt NVFP4 quantization. Without this, `w13/w2_input_scale` tensors are uninitialized, producing NaN logits on every NVFP4 MoE layer call | Step-3.7-Flash MoE layer weights loader | Not upstreamed | Remove if ModelOpt NVFP4 input-scale handling is corrected upstream or in a future Step-3.7-Flash checkpoint release |
| `patch_step3p7_modelopt_cache_release.py` | Prevents cumulative CUDA caching-allocator reserved-memory growth during ModelOpt NVFP4 MoE MARLIN post-load conversion on unified-memory (UMA) devices. Calls `torch.cuda.empty_cache()` after the MARLIN conversion of each `ModelOptNvFp4FusedMoE` module (42/42 modules on Step-3.7-Flash). Feature-gated via env var `VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE` (default: disabled; enabled by the Step-3.7 NVFP4 preset). Upstream relation: vLLM PR #45179 (merged to upstream main; **not included in v0.23.0**; first released version not yet confirmed). Downstream targets `ModelOptNvFp4FusedMoE` specifically with unconditional per-module release; upstream `release_device_memory_under_pressure()` is generic across quantization methods and conditional on a UMA pressure threshold | `ModelOptNvFp4FusedMoE` MARLIN conversion path | PR #45179 merged to vLLM main; not yet in any released vLLM tag | Remove only after rebasing the Step-3.7 image onto a released vLLM version that contains PR #45179 **and** validating Step-3.7-Flash-NVFP4 TP=2 EP=2 without the downstream patch |

## Step-3.7-Flash-NVFP4 native MTP patches (Candidate 1, 2026-06-15)

Stacked on top of the production NVFP4 image
(`ghcr.io/bjk110/vllm-spark@sha256:08ae8f2ab5597afd577977ce2700eff2cc024c3e6e781f6c8df6e1115963bf1b`)
in `Dockerfile.step37-nvfp4-mtp-candidate1` as 6 sequential patches.  All 6
were applied inside the candidate image
(`vllm-spark:step37-nvfp4-mtp-candidate1-canonical`, sha256 `b25400c3013e`)
and acceptance-tested on dual DGX Spark GB10 (spark01+spark02, TP=2 EP=2 mp
backend) with `num_speculative_tokens=3`.

| Patch file | Purpose | Affected class / file | Upstream status | Removal condition |
|---|---|---|---|---|
| `patch_step3p7_speculative_mtp.py` | Backport of vLLM commit `c621af16908f` (`hf_config_override()` lines 496-508): maps `step3p5`/`step3p7` outer configs to the `Step3p5MTP` draft architecture while preserving the outer ModelOpt `quantization_config` | `vllm/config/speculative.py` | Open (not in vLLM 0.22.1) | Merge into production NVFP4 image once upstream absorbs the step3p5/step3p7 MTP config mapping |
| `patch_step3p7_mtp_hfconfig.py` | Promotes `hf_config.text_config` (inner `Step3p5Config`) to a top-level attribute on the `Step3p5AMultiTokenPredictor`, fixing a `AttributeError: 'Step3p7Config' has no attribute 'num_hidden_layers'` during MTP model init | `vllm/model_executor/models/step3p5_mtp.py` | Not upstreamed | Remove if Step3p5AMultiTokenPredictor is updated to handle VLM outer configs natively |
| `patch_step3p7_mtp_draft_vllm_config.py` | In `SpecDecodeBaseProposer._get_model()`, overrides `model_config=draft_model_config` and forces `quant_config=None` so the MTP draft model is not given the NVFP4-packed parameter shapes that belong to the target model | `vllm/v1/spec_decode/llm_base_proposer.py` | Not upstreamed | Remove if vLLM properly separates draft model quantization config from target |
| `patch_step3p7_mtp_speculators_local_path.py` | When `speculative_config.model` is already an absolute local path (not a Hub repo id), skips the Hub-format validation that raises `ValueError: Repo id must be in the form 'repo_name'` | `vllm/transformers_utils/config.py` | Not upstreamed | Remove if vLLM's speculator config loader accepts absolute paths natively |
| `patch_step3p7_mtp_image_token_index.py` | Adds `Step3p7ForConditionalGeneration` to the list of model architectures that populate `image_token_index` from `hf_config.image_token_id`, fixing `AttributeError: 'Step3p7Config' object has no attribute 'image_token_index'` during draft proposer initialization | `vllm/v1/spec_decode/llm_base_proposer.py` | Not upstreamed | Remove if Step3p7 config is updated upstream to expose `image_token_index` |
| `patch_step3p7_mtp_kv_cache_grouping.py` | Two-file fix for the `AssertionError: All drafting layers should belong to the same kv cache group` crash. **Root cause**: Step3p5 has 12 full-attention + 33 sliding-attention layers; adding 3 MTP sliding layers (indices 45-47) gives 36 sliding total. `_get_kv_cache_groups_uniform_page_size` uses round-robin (`layers[i::3]`) to distribute 36 sliding layers into 3 groups of 12, placing MTP layers into groups 0, 1, 2 respectively. `validate_same_kv_cache_group` then fails because `kv_cache_gid` differs across MTP layers. **Fix**: (1) stores `_eagle_draft_attn_layer_names` on `compilation_config` for debugging; (2) in `get_kv_cache_groups()` (EngineCore process), detects draft layers by `.layers.N.` index ≥ `hf_config.text_config.num_hidden_layers` (45) and moves all of them into a single dedicated KV cache group | `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/spec_decode/llm_base_proposer.py` | Not upstreamed | Remove if vLLM's `_get_kv_cache_groups_uniform_page_size` or `validate_same_kv_cache_group` is updated to handle hybrid-attention models where spec-decode draft layers span multiple groups |

### Acceptance test results (2026-06-15, sha256:b25400c3)

- Image: `vllm-spark:step37-nvfp4-mtp-candidate1-canonical` (sha256 `b25400c3013e`)
- Config: TP=2, EP=2, mp backend, `MAX_MODEL_LEN=32768`, `MAX_NUM_SEQS=4`, `GPU_MEM_UTIL=0.79`, `num_speculative_tokens=3`
- KV cache: 1,063,006 tokens (32.44× at 32 KiB/request)
- MTP metrics: 1353 drafts, 4059 draft tokens, 72 accepted (acceptance rate 1.77% — low due to reasoning model trace unpredictability)
- Korean generation: `대한민국의 수도는 서울특별시입니다.` — PASS, no garbling
- English generation: `4` (for "What is 2+2?") — PASS
- Benchmark: depth sweep pp2048@d0/4096/8192/16384, tg32, runs3 — see `benchmarks/llama-benchy/results_step37-flash-nvfp4-tp2-mtp3-DEPTH.md`

## Conditional / opt-in patches (applied at runtime via `entrypoint.sh`)

| Patch file | Purpose | Applies to | Required for | Upstream status | Removal condition |
|---|---|---|---|---|---|
| `patch_qwen35_moe_text.py` | Registers a text-only shim for `Qwen3_5MoeForConditionalGeneration` so vLLM skips multimodal warmup, fixes the hybrid cache-spec page-size bug, and registers as `Qwen3_5MoeForCausalLM` | `vllm/model_executor/models/qwen3_5.py` and `registry.py` | `wangzhang-122b-fp8.env` and `wangzhang-122b-nvfp4.env` (both have `APPLY_TEXT_ONLY_SHIM=1`) | Unknown / verify before removal — abliterated / text-only Qwen3.5 variants may need this until vLLM ships a text-only loader path | vLLM upstream supports loading Qwen3.5 MoE as text-only via a flag, **and** the page-size bug is gone |

## On-standby (not applied automatically — invoke manually if symptom returns)

| Patch file | Purpose | Applies to | Required for | Upstream status | Removal condition |
|---|---|---|---|---|---|
| `patch_codegen_fx_repr.py` | Hot-patch for `vllm/compilation/codegen.py::_node_ref()` so it honors `__fx_repr__()` on opaque types like `LayerName`. Solves `SyntaxError: invalid syntax` in `EngineCore` init | `vllm/compilation/codegen.py` (runtime hot-patch via `docker exec`) | Qwen3.5 hybrid + `torch.compile` cold start (the GDN attention path takes a `LayerName` opaque arg) **only if** the Inductor-graph-partition workaround stops working | Open — vLLM has not landed an `__fx_repr__`-aware `_node_ref()` upstream | The `--compilation-config {"use_inductor_graph_partition":true}` workaround keeps working, so this patch is dormant. Drop it once vLLM's `_node_ref()` consults `__fx_repr__()` upstream **and** all `*-tq.env` presets have removed the inductor-graph-partition flag |

## Removed (already merged upstream — files retained for archive)

| Patch file | Purpose | Removed in | Upstream status | Notes |
|---|---|---|---|---|
| `fix_cuda13_memcpy_batch.py` | Adapt `cuMemcpyBatchAsync` call to CUDA 13.0+ API (drops the `failIdx` parameter) in `csrc/cache_kernels.cu` | base-refresh-20260417 (`a7bb0ef`) | Merged upstream | Comment in `Dockerfile.gemma4` confirms removal. File kept in `patches/` for reproducibility — no consumer references it. Safe to delete from tree once main has been pinned past the merge for ≥1 release. |
| `qwen3_5_moe_rope_fix.py` | Convert `ignore_keys_at_rope_validation: list` → `set` in `qwen3_5_moe.py` so the transformers 5.x `set | set` union works | base-refresh-20260417 (`a7bb0ef`) | Merged upstream | Same handling as above — orphan in tree. |
| `pr38423_nvfp4_spark.py` | Cherry-pick of vLLM PR #38423 (NVFP4 backend selection, FlashInfer CUTLASS quant_scales, trtllm_nvfp4_moe import order, FlashInfer use_ep removal) | base-refresh-20260417 (`a7bb0ef`) | PR #38423 merged | Same handling. |

## Historical / superseded scripts (not invoked by Dockerfile or entrypoint)

These predate the in-Dockerfile patch flow (they were `docker exec` hot-patch
scripts used in March 2026 before the patches were baked into the image).
Kept in `patches/` for archeology.

| File | What it did | Replaced by |
|---|---|---|
| `apply_hotpatch.sh` | Push the AOT / nogds / MoE-config patches into already-running spark01 / spark02 containers via `ssh + docker exec + docker cp + patch` | These three patches are now baked into the Dockerfile (`aot_cache_fix.patch`, `nogds_force.patch`, `moe_config_e256/e512.json`). Run path is build-time, not runtime. |
| `apply_patches_in_container.py` | Apply patches #1 (AOT cache), #4 (nogds), #5 (MoE tuning) via `docker exec` | Same — baked into Dockerfile. |
| `apply_patches_round2.py` | Apply patches #2, #3, #6, #7, #9 from `seli-equinix/vllm:feature/sm121-gb10-support` via `docker exec` | Patches #3, #6, #9 are now in `apply_sm121_patches.py`; patches #2 and #7 are in `apply_patch2_fp8_moe.py` (still hot-patch only — not in active build path). |
| `apply_patch2_fp8_moe.py` | Hot-patch script enabling FP8 Block-Scale MoE on SM121 (requires FlashInfer ≥ 0.6.4) | Folded into the FlashInfer build itself starting in v019/v020; this script no longer needs to run. |

## Container-resident static assets (not patches, but in `patches/`)

| File | Purpose |
|---|---|
| `flashinfer_cache.patch` | Optional patch applied to the FlashInfer source tree during the FlashInfer wheel build (Stage 1 of `Dockerfile.gemma4`). Adds an offline cubin checksum check that skips re-download when the cubin already exists. Non-fatal — the build silently skips it if it does not apply. |

## Solar-Open2 r4 BF16 production patches (`patches/solar/`, not yet tracked)

These four patches are the source-level provenance for the three named production gates in the
promoted Solar-Open2 r4 BF16 image
(`vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rawg1-pread-b12xsw-r4-exp`, local image ID
`sha256:ecb7bfe3978a5241c5c304d52ce91e061e22b750178d21a4ef7788a08e86e774`). They exist locally at
`patches/solar/` on the build hosts (spark01/spark02), following the repository's established
per-vendor `patches/<vendor>/` convention (same pattern as `patches/dsv4/`, `patches/sm121/`,
`patches/qwen/`), but are **not yet added to Git** as of this hygiene pass — they are
working-tree-ready, hash-verified, and staged for a future tracking commit, not fabricated: content
was read directly from the build hosts and matched to the exact gate names cited throughout the
production fast-track and promotion evidence.

| Patch file | Purpose | Applies to | Production status | Removal condition |
|---|---|---|---|---|
| `solar-open2-rawg1-contract-v025.patch` (sha256 `f252383c949de06cf14d2fadf6ff6c0a5ae76fa7408e7beb5b30c8db823dd6ca`) | **Raw-g1 KDA correction.** vLLM v0.25.x's `KimiGatedDeltaNetAttention` applies the KDA decay gate in-kernel (`fused_kda_gate`/`chunk_kda_with_fused_gate`); the v0.22-contract Solar-Open2 model code pre-gated `g1` before calling in, double-applying the gate and diverging at layer 1's kda_attention output. This patch passes the raw `g1` (reshaped, not pre-gated), matching the v0.25.x parent-forward contract. | `vllm/model_executor/models/solar_open2.py` (`SolarOpen2KimiDeltaAttention.forward`) | **Active in production.** Confirmed present in the live r4 image via source hash + bit-identical kernel-level conformance rerun (Gate 1, `SOLAR_OPEN2_V0251_R4_BF16_C2_TARGETED_CORRECTNESS_PASS`). | Remove only if Upstage's upstream Solar-Open2 vLLM integration ships this fix natively for the pinned vLLM revision (`752a3a504485`) — verify by reproducing Gate 1's kernel-level conformance check without the patch before removing. |
| `vllm-safetensors-pread-env-gate.patch` (sha256 `39b26822f2bfb85dba1fb75e8e90a62ee2aabb32014d4b26ce963b6e2b1371ea`) | **`VLLM_SPARK_ST_PREAD` gate.** Opt-in `backend="pread"` for the lazy safetensors weight iterator, avoiding GB10 UMA swap pressure from the default mmap read path during weight loading. | `safetensors_weights_iterator()` (vLLM weight loader) | **Active in production** (`VLLM_SPARK_ST_PREAD=1` in the production preset). Measured effect: loader-phase swap 7.33 GB / 3.12 GB -> near-zero; TP0/TP1 load time roughly halved (2026-07-25 diagnostics, summarized in `docs/solar-open2-production.md` section 6). | Not applicable for removal — this is a downstream, env-gated optimization, not a bug workaround; keep as long as GB10 UMA mmap pressure remains a factor. |
| `vllm-flashinfer-b12x-shared-workspace-env-gate.patch` (sha256 `2b31f3a873e7a29c991cefc39599b01b10f41f4dbdef096fc55f23c175382c83`) | **`VLLM_SPARK_B12X_SHARED_WORKSPACE` gate.** Opt-in sharing of FlashInfer `B12xMoEWrapper` pre-allocated scratch workspaces across MoE layers (default: one full scratch buffer per layer, ~27.9 GiB/rank for a 48-layer model; shared: one bounded set, ~0.6 GiB). | FlashInfer B12X MoE wrapper construction path | **Active in production** (`VLLM_SPARK_B12X_SHARED_WORKSPACE=1` in the production preset). Measured effect: torch allocated at engine-ready 103.5 -> 76.2 GiB/rank; deterministic output bit-identical to the unshared path (C3 conformance); decode throughput unchanged (2026-07-26 overlay validation). | Same as ST_PREAD — env-gated memory optimization, not a bug workaround; keep as long as the per-layer scratch allocation remains the FlashInfer default. |
| `solar-open2-support-v0251.patch` (sha256 `1e73ef5b5d70aa29957975dab39ece83ec32b1e542f3d51cbb7c0e857d519575`) | General Solar-Open2 vLLM 0.25.1 support overlay (r2 lineage) — adds fused-MoE tuning configs and base model-class support, not one of the three named gates above. | `vllm/model_executor/...` (multiple files, largely new fused-MoE config JSONs) | **Active in production** (base layer all three gates above are built on). Not independently gated by an env var — always active for the Solar-Open2 architecture. | Not applicable — this is the base architecture-support overlay, not a point fix. |

**Build-time provenance**: the r4 Dockerfile
(`dockerfiles/active/Dockerfile.solar-open2-nvfp4-v0251-rawg1-pread-b12xsw-r4-exp`, local/untracked)
applies these patches as `COPY` + in-image patch steps layered `r2 -> r3 (raw-g1 + ST_PREAD) -> r4
(+ B12X shared-workspace)`. The Dockerfile itself, like the patches, is **not tracked** as of this
pass — it is real, exact build source for the production image (not merely a claim), but its
promotion to tracked status was out of scope for this hygiene pass (image reproducibility from
source was not requested and was not verified against a fresh rebuild, which this pass explicitly
does not perform). If the implementation for any of the three gates above is ever needed and cannot
be reconstructed from `patches/solar/` plus the Dockerfile, say so explicitly rather than assuming
the repository can rebuild it — as of this audit, the source **does** exist locally for all three
gates; only the *tracked* (in-Git) status is what's missing.
