# DeepSeek-V4-Flash native DSpark (k=7 greedy) — validated promotion CANDIDATE

**Status: validated promotion candidate — NOT yet production-active.**
This document describes the reproducible build + runtime for the native-DSpark speculative-decoding
route on dual DGX Spark (GB10 / SM121, TP=2 mp over RoCE), validated in the DS3H → DS3I → DS3J/DS3J-R1 arc.
It is experimental until explicitly promoted; it does not replace the current production baseline.

## Build definition (tracked, reproducible from a clean context)
- Dockerfile: `dockerfiles/active/Dockerfile.ds3j-r1-v025-native-dspark-k7-repro-exp`
  (single consolidated from-NGC build; documented build tag
   `vllm-spark:ds3j-r1-v025-native-dspark-k7-repro-exp-5cb57f7`).
- Validated image ID (both nodes): `sha256:75bdf3d810558f1738927996f448056b196f83d4e09e55b23fffecfe904ead24`.

## Pinned sources (exact, fail-fast verified in the build)
- Base: NGC 26.05 (`nvcr.io/nvidia/pytorch:26.05-py3`; PyTorch 2.12.0a0, CUDA 13.2, NCCL 2.30.4).
- vLLM v0.25.0 @ `702f4814fe54fabff350d43cb753ae3e47c0c276` (native DSpark PR #46995 + #47093).
- FlashInfer v0.6.15 @ `8eccd0c1352165302840c0e19066bc42d36dbd7a`.
- FlashInfer PR #3989 head `1459c5d337a13954e679ecc56566f194ecbbe85f`
  (`patches/flashinfer/pr3989-sm120-dsv4-topk256.patch`; SM120 DSV4 topk=256 dispatch; OPEN, not merged).
- DeepGEMM 2.5.0 (vendored in vLLM); apache-tvm-ffi 0.1.9; tilelang 0.1.9; tokenspeed-mla 0.1.2.
- Cutlass DSL coherent family: nvidia-cutlass-dsl / -libs-base / -libs-cu13 all == 4.5.2
  (libs-core / libs-cu12 4.6.0 removed); quack-kernels 0.5.0.

## SM121 numerical contract (mandatory)
- `patches/ds3/ds3j_sm121_oproj_scale_recipe.py`: compute_fp8_einsum_recipe major==12 →
  `einsum_recipe=(1,1,128), tma_aligned_scales=True` (SM100-style per-row UE8M0). SM90/SM100 preserved.
- Semantic gate: `scripts/verify_ds3j_oproj_recipe.py` (major 9/10/12 returned values).
- **`VLLM_USE_DEEP_GEMM_E8M0=1` is REQUIRED at runtime** — a numerical-correctness setting, NOT a
  performance tuning flag. Without it the sm120 FP8 o_proj is numerically wrong (overflows to Inf).

## Validated runtime envelope (do not deviate for this candidate)
model DeepSeek-V4-Flash-DSpark (48 shards); method dspark; num_speculative_tokens 7; draft_sample_method
greedy; TP=2; backend mp; transport RoCE (10.10.10.1 / 10.10.10.2); max_model_len 16384;
max_num_batched_tokens 8192; max_num_seqs 1; KV 4 GiB fp8; eager; prefix caching OFF; CUDA graphs OFF;
concurrency OFF. MoE backend MARLIN. Preset:
`presets/deepseek-v4-ds3j-native-dspark-k7-repro-exp-tp2.env`.

## Validated results (DS3J-R1; measurement = non-streaming, ignore_eos, fixed length, wall-clock)
- Runtime dispatch: indexer next_n=8, model-level index_topk=512, no `layout.hpp:97`.
- Acceptance (combined, 2 cold starts): aggregate 35.3–37.1%, MAL 3.47–3.60, position-1 75.7–77.9%.
- Throughput: overall 27.1–28.4 tok/s (code 31.3–32.3, mixed 22.9–24.5); ~3.3× the target-only control.
- 60-min soak: 120/120 HTTP 200, 0 EngineDeadError / CUDA / NaN / guard trip.

## Known limitations (unverified — do NOT present as supported)
higher k (>7); stochastic sampling; CUDA graphs; prefix caching; concurrency (>1 seq); long context
(>16384). UMA may remain pinned after teardown and require a node reboot. Validation is specific to
dual GB10 SM121, TP=2 mp/RoCE.

## Rollback
Current production baseline is unchanged: preset
`presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env` (graph-safe recipe). This candidate does
not alter it. Rollback = keep using the production preset.
