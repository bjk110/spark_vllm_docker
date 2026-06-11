#!/usr/bin/env python3
"""
Diagnostic (read-only, no compilation triggered): inspect FlashInfer's
fused_moe_120 JitSpec state and AOT cache layout to determine why a
prebaked /root/.cache/flashinfer/.../cached_ops/fused_moe_120/fused_moe_120.so
still appears to trigger a near-full runtime recompile.

Covers:
  - Step 1: spec.is_aot / spec.is_compiled / aot_path / jit_library_path /
    num sources, plus root .ninja_log / .ninja_deps presence.
  - Step 2: mtime/size diff of generated/cutlass_instantiations/120/**
    before vs after calling gen_cutlass_fused_moe_sm120_module() again.
  - Step 4: relevant env vars (MAX_JOBS, FLASHINFER_*, CUDA_HOME, nvcc).

Does NOT call build_jit_specs()/build_and_load() -- that's step 5,
handled separately (optional, may trigger compilation).
"""
import os
import subprocess
from pathlib import Path

import torch

import flashinfer
import flashinfer.jit.env as e
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module


def hr(title):
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


hr("STEP 4: env")
for var in [
    "MAX_JOBS",
    "FLASHINFER_NVCC_THREADS",
    "FLASHINFER_CUDA_ARCH_LIST",
    "FLASHINFER_DISABLE_JIT",
    "FLASHINFER_AOT_DIR",
    "CUDA_HOME",
]:
    print(f"{var}={os.environ.get(var)!r}")
print("which nvcc:", subprocess.run(["which", "nvcc"], capture_output=True, text=True).stdout.strip())
nvcc_ver = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
print("nvcc --version:\n" + nvcc_ver.stdout)

hr("STEP 1: env paths")
print("flashinfer.__version__:", getattr(flashinfer, "__version__", "unknown"))
print("FLASHINFER_JIT_DIR:", e.FLASHINFER_JIT_DIR)
print("FLASHINFER_AOT_DIR:", getattr(e, "FLASHINFER_AOT_DIR", "N/A"))
print("FLASHINFER_GEN_SRC_DIR:", e.FLASHINFER_GEN_SRC_DIR)
print("FLASHINFER_CSRC_DIR:", getattr(e, "FLASHINFER_CSRC_DIR", "N/A"))
print("FLASHINFER_WORKSPACE_DIR:", getattr(e, "FLASHINFER_WORKSPACE_DIR", "N/A"))

hr("STEP 1: gen_cutlass_fused_moe_sm120_module() spec (1st call)")
spec = gen_cutlass_fused_moe_sm120_module(use_fast_build=False)
print("spec.name:", spec.name)
print("type(spec):", type(spec))

# Print every attribute that looks relevant -- attribute names may differ
# across flashinfer versions, so don't assume is_aot/aot_path exist.
interesting_substrings = ("aot", "compil", "path", "lib", "source", "name", "ninja")
for attr in sorted(dir(spec)):
    if attr.startswith("_"):
        continue
    low = attr.lower()
    if any(s in low for s in interesting_substrings):
        try:
            val = getattr(spec, attr)
        except Exception as ex:
            val = f"<error: {ex!r}>"
        if callable(val):
            continue
        if attr == "sources":
            try:
                print(f"spec.{attr}: <{len(val)} items>")
            except Exception:
                print(f"spec.{attr}: {val!r}")
        else:
            print(f"spec.{attr}: {val!r}")

hr("STEP 1: first 20 sources")
try:
    for s in list(spec.sources)[:20]:
        print("  ", s)
except Exception as ex:
    print("  <error reading sources>:", ex)

hr("STEP 1: cached_ops/fused_moe_120 contents")
cached_dir = Path(e.FLASHINFER_JIT_DIR) / "fused_moe_120"
# Some versions key FLASHINFER_JIT_DIR at .../121a (arch-specific) and put
# modules directly under it; others may nest under cached_ops/. Check both.
candidates = [
    cached_dir,
    Path(e.FLASHINFER_JIT_DIR) / "cached_ops" / "fused_moe_120",
]
for c in candidates:
    print(f"\n-- {c} (exists={c.exists()}) --")
    if c.exists():
        for p in sorted(c.iterdir()):
            try:
                st = p.stat()
                print(f"  {p.name}\tsize={st.st_size}\tmtime={st.st_mtime}")
            except Exception as ex:
                print(f"  {p.name}\t<error: {ex}>")

hr("STEP 1: root .ninja_log / .ninja_deps")
for root_candidate in {Path(e.FLASHINFER_JIT_DIR), Path(e.FLASHINFER_JIT_DIR).parent}:
    for fname in (".ninja_log", ".ninja_deps"):
        p = root_candidate / fname
        if p.exists():
            print(f"{p}: exists=True size={p.stat().st_size}")
        else:
            print(f"{p}: exists=False")

hr("STEP 2: mtime/size diff of generated/cutlass_instantiations/120 across two spec-gen calls")
gen_dir = Path(e.FLASHINFER_GEN_SRC_DIR) / "cutlass_instantiations" / "120"
print("gen_dir:", gen_dir, "exists=", gen_dir.exists())


def snapshot(d: Path):
    out = {}
    if not d.exists():
        return out
    for p in d.rglob("*"):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(d))] = (st.st_mtime, st.st_size)
    return out


before = snapshot(gen_dir)
print(f"before: {len(before)} files")

print("\ncalling gen_cutlass_fused_moe_sm120_module() again (2nd call)...")
spec2 = gen_cutlass_fused_moe_sm120_module(use_fast_build=False)

after = snapshot(gen_dir)
print(f"after: {len(after)} files")

added = sorted(set(after) - set(before))
removed = sorted(set(before) - set(after))
changed = sorted(
    k for k in (set(before) & set(after)) if before[k] != after[k]
)

print(f"\nadded: {len(added)}")
for k in added[:20]:
    print("  +", k)
print(f"removed: {len(removed)}")
for k in removed[:20]:
    print("  -", k)
print(f"changed (mtime/size): {len(changed)}")
for k in changed[:20]:
    print(f"  ~ {k}: before={before[k]} after={after[k]}")

print("\nDone.")
