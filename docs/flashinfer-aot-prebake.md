# FlashInfer AOT Prebake (`v022-d568-fi-aot`)

`vllm-spark:v022-d568-fi-aot` is a thin overlay on `v022-d568` (same vLLM /
FlashInfer / Transformers versions) that pre-builds a fixed set of FlashInfer
SM120/SM121 CUTLASS kernel modules at **image build time** and promotes each
built `.so` from FlashInfer's JIT cache path (`spec.jit_library_path`) to its
AOT path (`spec.aot_path`).

## Why this is needed

FlashInfer's `JitSpec.is_aot` is simply `aot_path.exists()`. When `is_aot` is
`True`, `build_and_load()` returns immediately via `self.load(aot_path)` —
ninja is never invoked. When it's `False` (the default for a module that was
only JIT-built into the cache, not "AOT" in FlashInfer's own sense),
`build_and_load()` re-evaluates the module's `build.ninja`. For
`fused_moe_120` on GB10 this re-evaluation finds the `.ninja_deps` graph
incomplete and triggers a near-full ~96-source recompile (50-60 concurrent
`nvcc`/`cicc`/`ptxas` processes), which on a 121 GiB UMA + 63 GiB swap GB10
node pushes memory usage close to the safety ceiling.

The fix: after building each module's `.so` into the JIT cache at image-build
time (`dockerfiles/scripts/flashinfer_aot_build.py`), copy it to
`spec.aot_path`. This makes `spec.is_aot=True` for that module at runtime, so
`build_and_load()` skips ninja entirely.

## Validated specs

The following 7 specs are prebuilt and AOT-promoted in `v022-d568-fi-aot`
(`is_aot=True`, `is_compiled=True` for all 7, on both head and worker):

- `fp4_quantization_121`
- `fused_moe_120`
- `gemm_sm120`
- `fp4_gemm_cutlass_sm120`
- `mxfp8_gemm_cutlass_sm120`
- `batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False`
- `sampling`

This list covers the SM120/SM121 MoE/GEMM/FP4 kernels used across the
`v022-d568` presets (Qwen3.5/3.6, Gemma 4, abliterix FP8/NVFP4, PrismaSCOUT
NVFP4) plus the two general FA2/sampling modules observed to JIT-compile at
runtime for `Qwen/Qwen3.6-35B-A3B` (`kv_cache_dtype=fp8`,
`head_dim_qk=head_dim_vo=256`). It is **not** FlashInfer's full
`gen_all_modules()` set (696 specs) — other module/dtype/shape combinations
not in this list will still JIT-compile on first use as before.

## Validation result (2026-06-11, dual-node RDMA TP=2)

Ran `Qwen/Qwen3.6-35B-A3B` (BF16, `kv_cache_dtype=fp8`,
`max_model_len=32768`, `gpu_memory_utilization=0.85`) on `spark01` (head) +
`spark02` (worker), `CLUSTER_MODE=dual-rdma`, `TP_SIZE=2`,
`DISTRIBUTED_BACKEND=ray`:

- Ray cluster joined (2/2 nodes), vLLM confirmed `tensor_parallel_size=2`.
- Model load, post-weight-load profiling, and CUDA graph capture (PIECEWISE +
  FULL) all completed normally on both nodes.
- All 7 specs above reported `is_aot=True` / `is_compiled=True` on **both**
  the head and worker container.
- No `ninja`, `nvcc`, `cicc`, or `ptxas` process appeared on either node
  during model load, profiling, CUDA graph capture, or the first request.
- `/health` returned `200`; a `/v1/completions` request returned a valid
  completion.
- Memory/swap stayed stable on both nodes throughout.

## Build

```bash
# spark01 or spark02 only — see CLAUDE.md build-location rule
docker buildx build -f dockerfiles/active/Dockerfile.v022-d568-fi-aot \
  -t vllm-spark:v022-d568-fi-aot --load .
```

`Dockerfile.v022-d568-fi-aot-extra` is an idempotent addendum (re-promotes the
two general FA2/sampling specs) kept for incremental rebuilds; the main
Dockerfile already covers all 7 specs via
`dockerfiles/scripts/flashinfer_aot_build.py`.

## Usage

Drop-in replacement for `v022-d568` — set `VLLM_IMAGE=vllm-spark:v022-d568-fi-aot`
(or the GHCR tag) in the relevant preset `.env`. No other config changes
required.
