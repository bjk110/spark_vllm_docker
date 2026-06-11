#!/usr/bin/env python3
"""
Incremental addendum to flashinfer_aot_build.py: prebuilds 2 small general
(non-SM120-specific) FlashInfer modules observed to still JIT-compile at
runtime during post-weight-load profiling for Qwen3.6-35B-A3B
(kv_cache_dtype=fp8, head_dim_qk=head_dim_vo=256, FA2 prefill backend):
- batch_prefill_with_kv_cache (fa2, q=bf16, kv=e4m3, o=bf16, idx=i32,
  head_dim 256/256, posenc=NONE, no SWA/logits-cap/f16qk) -- 10 sources
- sampling -- 3 sources

Run on top of an image that already has the SM120/SM121 cache from
flashinfer_aot_build.py (vllm-spark:v022-d568-fi-aot), so only these
13 sources are compiled here.

After building, each spec's .so is promoted from spec.jit_library_path to
spec.aot_path (same as flashinfer_aot_build.py) so build_and_load() bypasses
ninja entirely at runtime. These two specs are also already covered by
flashinfer_aot_build.py's main spec list, so this is normally a no-op
re-promotion -- kept for idempotency in case this script is ever run alone.
"""
import shutil
import sys
from pathlib import Path

import torch

import flashinfer.aot as aot
import flashinfer.jit as fjit
import flashinfer.jit.env as jit_env

specs = [
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

# Promote each JIT-built .so to its AOT path (see flashinfer_aot_build.py).
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
