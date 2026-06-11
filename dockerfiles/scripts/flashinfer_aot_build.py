#!/usr/bin/env python3
"""
Pre-build FlashInfer's SM120/SM121 CUTLASS kernel modules into the JIT cache
at image-build time, so vLLM doesn't trigger nvcc/ninja JIT compilation
(and the associated memory spike) on first MoE-model startup on GB10.

Scope: the SM120/SM121-specific generators from flashinfer.aot.gen_all_modules()
- fp4_quantization_121   (FP4 KV/weight quantization, sm121)
- fused_moe_120          (CUTLASS grouped-GEMM MoE, sm120/121 -- the module
                           observed JIT-compiling at runtime for Qwen3.6-35B-A3B)
- gemm_sm120             (CUTLASS GEMM, sm120/121)
- fp4_gemm_cutlass_sm120 (CUTLASS FP4 GEMM, used by VLLM_USE_FLASHINFER_MOE_FP4
                           presets, e.g. wangzhang-122b-abliterix-nvfp4-tp2)
- mxfp8_gemm_cutlass_sm120

Avoids flashinfer.aot's main()/compile_and_package_modules() CLI path, which
has a project_root path-resolution bug for installed (non-source-tree) wheels.
gen_*_module() + build_jit_specs() use flashinfer.jit.env's package-relative
defaults directly, which are correct for an installed package.

Additionally prebuilds two general (non-SM120-specific) modules observed to
still JIT-compile at runtime for Qwen3.6-35B-A3B (kv_cache_dtype=fp8,
head_dim_qk=head_dim_vo=256, FA2 backend):
- batch_prefill_with_kv_cache (fa2, q=bf16, kv=e4m3, o=bf16, idx=i32,
  head_dim_qk=256, head_dim_vo=256, posenc=NONE, no SWA/logits-cap/f16qk)
- sampling

After building, every spec's .so is additionally promoted from its JIT cache
path (spec.jit_library_path, under /root/.cache/flashinfer/.../cached_ops/)
to its AOT path (spec.aot_path, under flashinfer/data/aot/). FlashInfer's
JitSpec.is_aot is just `aot_path.exists()`, and build_and_load() returns
immediately via `self.load(self.aot_path)` when is_aot is True -- skipping
ninja entirely. Without this promotion, runtime re-invocation of e.g.
gen_cutlass_fused_moe_sm120_module() re-evaluates fused_moe_120's build.ninja,
finds its .ninja_deps incomplete (deps missing for ~45/96 objects, the rest
dirty), and triggers a near-full ~96-file recompile (52-60 concurrent
nvcc/cicc/ptxas) despite the .so already existing in the JIT cache.
"""
import shutil
import sys
from pathlib import Path

import torch

import flashinfer.aot as aot
import flashinfer.jit as fjit
import flashinfer.jit.env as jit_env

sm = aot.detect_sm_capabilities()
print(f"sm_capabilities: {sm}", flush=True)

specs = []
if sm.get("sm121", False):
    specs.append(aot.gen_fp4_quantization_sm121_module())
if sm.get("sm120", False) or sm.get("sm121", False):
    specs += [
        aot.gen_cutlass_fused_moe_sm120_module(),
        aot.gen_gemm_sm120_module(),
        aot.gen_gemm_sm120_module_cutlass_fp4(),
        aot.gen_gemm_sm120_module_cutlass_mxfp8(),
    ]

# General modules observed to JIT-compile at runtime on first request
# (post-weight-load profiling) for the Qwen3.6-35B-A3B preset (fp8 KV cache,
# head_dim 256, FA2 prefill backend on GB10).
specs += [
    aot.gen_batch_prefill_module(
        "fa2",
        torch.bfloat16,  # dtype_q
        torch.float8_e4m3fn,  # dtype_kv
        torch.bfloat16,  # dtype_o
        torch.int32,  # dtype_idx
        256,  # head_dim_qk
        256,  # head_dim_vo
        0,  # pos_encoding_mode (NONE)
        False,  # use_sliding_window
        False,  # use_logits_soft_cap
        False,  # use_fp16_qk_reduction
    ),
    aot.gen_sampling_module(),
]

if not specs:
    print("No SM120/SM121 capability detected -- nothing to pre-build.", flush=True)
    sys.exit(0)

print(f"Building {len(specs)} module(s) into {jit_env.FLASHINFER_JIT_DIR}:", flush=True)
for s in specs:
    print(f"  - {s.name} ({len(s.sources)} sources)", flush=True)

fjit.build_jit_specs(specs, verbose=True, skip_prebuilt=True)

missing = [s for s in specs if not s.jit_library_path.exists()]
if missing:
    for s in missing:
        print(f"FAILED: {s.name} -> {s.jit_library_path} not found", file=sys.stderr)
    sys.exit(1)

print("All modules built successfully:", flush=True)
for s in specs:
    print(f"  {s.jit_library_path}", flush=True)

# Promote each JIT-built .so to its AOT path, so FlashInfer's
# JitSpec.is_aot becomes True and build_and_load() at runtime returns via
# `self.load(self.aot_path)` without ever invoking ninja.
print("[FlashInfer AOT promote] promoting JIT-built FlashInfer modules to AOT paths...", flush=True)
for spec in specs:
    src = Path(spec.jit_library_path)
    dst = Path(spec.aot_path)

    print(f"[FlashInfer AOT promote] {spec.name}", flush=True)
    print(f"  jit: {src}", flush=True)
    print(f"  aot: {dst}", flush=True)

    if not src.exists():
        raise FileNotFoundError(
            f"JIT-built .so not found for {spec.name}: {src}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Always overwrite. A stale AOT .so from a previous layer/cache must not survive.
    shutil.copy2(src, dst)

    if not dst.exists():
        raise RuntimeError(
            f"Failed to create AOT .so for {spec.name}: {dst}"
        )

    if dst.stat().st_size != src.stat().st_size:
        raise RuntimeError(
            f"AOT .so size mismatch for {spec.name}: "
            f"src={src.stat().st_size}, dst={dst.stat().st_size}"
        )

    if not spec.is_aot:
        raise RuntimeError(
            f"spec.is_aot is still false after promotion for {spec.name}: {dst}"
        )

    print(f"  ok: {dst.stat().st_size} bytes", flush=True)

print("[FlashInfer AOT verify]", flush=True)
for spec in specs:
    print(
        f"{spec.name}: "
        f"is_aot={spec.is_aot}, "
        f"is_compiled={spec.is_compiled}, "
        f"aot_path={spec.aot_path}",
        flush=True,
    )
    if not spec.is_aot:
        raise RuntimeError(f"AOT verify failed: {spec.name}")
