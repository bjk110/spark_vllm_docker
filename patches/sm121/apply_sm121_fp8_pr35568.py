"""
Apply vLLM PR #35568 ("Fix SM121 (DGX Spark) exclusion from Marlin/CUTLASS FP8 paths"),
commit 06d020bb6, to a vLLM source checkout *before* bdist_wheel.

The change widens four hard-coded `SM120-only` checks to `SM12x family`, so the
DGX Spark GB10 (SM121) is no longer excluded from Marlin/CUTLASS FP8 kernel
codegen and dispatch. Required at build time — the C++ kernels must be
recompiled for the SM121 target.

This is a literal-string replacement. Idempotent.

Run with cwd set to the vLLM source root (i.e. /workspace/vllm-src in the
Dockerfile).
"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("VLLM_SRC", "/workspace/vllm-src"))

EDITS = [
    # 1. CUTLASS scaled_mm sm120 → sm12x
    {
        "file": "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm.cuh",
        "old": "  using GemmKernel = enable_sm120_only<cutlass::gemm::kernel::GemmUniversal<\n      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>>;\n};",
        "new": "  using GemmKernel = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<\n      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>>;\n};",
    },
    # 2. CUTLASS scaled_mm sm120 FP8 dispatch
    {
        "file": "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh",
        "old": "  using GemmKernel = enable_sm120_only<cutlass::gemm::kernel::GemmUniversal<\n      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>>;\n};",
        "new": "  using GemmKernel = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<\n      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>>;\n};",
    },
    # 3. Marlin MoE kernel generator — broaden arch list
    {
        "file": "csrc/moe/marlin_moe_wna16/generate_kernels.py",
        "old": '    # only SM89 and SM120 fully support\n    # mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32.\n    # SM90 and SM100 can use this PTX, but it’s simulated\n    # with FP16 MMA, so it cannot achieve any acceleration.\n    if arch in [89, 120]:',
        "new": '    # SM89 and the SM12x family (SM120 RTX 5090, SM121 DGX Spark GB10)\n    # fully support mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32.\n    # SM90 and SM100 can use this PTX, but it’s simulated\n    # with FP16 MMA, so it cannot achieve any acceleration.\n    if arch == 89 or arch // 10 == 12:',
    },
    # 4. Marlin MoE ops.cu — dispatch check
    {
        "file": "csrc/moe/marlin_moe_wna16/ops.cu",
        "old": "    TORCH_CHECK(\n        major_capability * 10 + minor_capability == 89 ||\n            major_capability * 10 + minor_capability == 120,\n        \"Marlin W4A8-FP8 only support SM89 or SM120 device (It is slower than \"",
        "new": "    TORCH_CHECK(\n        major_capability * 10 + minor_capability == 89 ||\n            major_capability == 12,\n        \"Marlin W4A8-FP8 only support SM89 or SM12x device (It is slower than \"",
    },
    # 5. Marlin (non-MoE) kernel generator — same broadening
    {
        "file": "csrc/quantization/marlin/generate_kernels.py",
        "old": '    # only SM89 and SM120 fully support\n    # mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32.\n    # SM90 and SM100 can use this PTX, but it’s simulated\n    # with FP16 MMA, so it cannot achieve any acceleration.\n    if arch in [89, 120]:',
        "new": '    # SM89 and the SM12x family (SM120 RTX 5090, SM121 DGX Spark GB10)\n    # fully support mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32.\n    # SM90 and SM100 can use this PTX, but it’s simulated\n    # with FP16 MMA, so it cannot achieve any acceleration.\n    if arch == 89 or arch // 10 == 12:',
    },
]

# Python file edit — vllm/model_executor/layers/quantization/utils/marlin_utils.py
# Only show diff via gh api if needed; the commit only changed two lines in this file.
PY_EDIT = {
    "file": "vllm/model_executor/layers/quantization/utils/marlin_utils.py",
    "old": '            "vllm:fp8_marlin requires SM89 or SM120 devices.",\n        )\n        capability = current_platform.get_device_capability()\n        capability_val = capability.major * 10 + capability.minor\n        if capability_val not in (89, 120):',
    "new": '            "vllm:fp8_marlin requires SM89 or SM12x devices.",\n        )\n        capability = current_platform.get_device_capability()\n        if capability.major * 10 + capability.minor != 89 and capability.major != 12:',
}


def apply(edit) -> str:
    path = ROOT / edit["file"]
    if not path.exists():
        return f"SKIP (not found): {path}"
    src = path.read_text()
    if edit["new"].split("\n")[0].strip() in src and edit["old"].split("\n")[0].strip() not in src:
        return f"OK already patched: {edit['file']}"
    if edit["old"] not in src:
        return f"WARN old marker missing: {edit['file']}"
    path.write_text(src.replace(edit["old"], edit["new"], 1))
    return f"OK patched: {edit['file']}"


def main() -> int:
    if not ROOT.exists():
        print(f"FATAL: VLLM source not at {ROOT}", file=sys.stderr)
        return 1
    print(f"[sm121_fp8_pr35568] vLLM source: {ROOT}")
    failed = 0
    for edit in EDITS:
        msg = apply(edit)
        print(f"  {msg}")
        if msg.startswith("WARN") or msg.startswith("SKIP"):
            failed += 1
    # Python file is best-effort (may not match exactly across vLLM versions)
    msg = apply(PY_EDIT)
    print(f"  [python] {msg}")
    if failed:
        print(
            f"\n[sm121_fp8_pr35568] {failed} C++ edits did not apply — "
            "the cherry-pick is likely already absorbed upstream or the "
            "file layout has changed."
        )
        # do not fail the build; report and continue
    print("[sm121_fp8_pr35568] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
