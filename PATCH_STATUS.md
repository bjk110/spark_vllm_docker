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
