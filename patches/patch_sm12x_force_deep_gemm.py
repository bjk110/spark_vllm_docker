"""
Force DeepGEMM linear path on SM12x (GB10 / SM121).

Background
----------
PR #40852 enables DeepGEMM CMake to build for SM12x (12.0f arch entry) and
ships the jasl/DeepGEMM fork with experimental SM120/SM121 fallback kernels.
However the **Python-side gate** in `vllm.platforms.cuda.support_deep_gemm`
still returns True only for SM90 (Hopper) and SM10x (Blackwell datacenter):

    return cls.is_device_capability(90) or cls.is_device_capability_family(100)

Consequence on GB10: `is_deep_gemm_supported()` is False, so
`BlockScaledMMLinearKernel` selection picks the **CUTLASS** path. That path
crashes inside `cutlass_gemm_caller.cuh:61 Error Internal` for the
DeepSeek-V4 fused_wqa_wkv FP8 GEMM:

    File ".../layers/quantization/fp8.py", line 476, in apply
    File ".../kernels/linear/scaled_mm/cutlass.py", line 268
        return ops.cutlass_scaled_mm(...)
    RuntimeError: cutlass_gemm_caller, ... Error Internal

PR #40852's tip was validated on a single-host SM120 box; we don't know
exactly how that environment selected DeepGEMM (no such patch in the PR).
For our 2× GB10 setup we need to force it explicitly.

Patch
-----
Rewrite `support_deep_gemm` in vllm/platforms/cuda.py to also accept
SM12x (GB10 / RTX Pro 6000). This routes the FP8 BlockScaledMM kernel
selection to DeepGEMM, which has SM12x JIT support in jasl/DeepGEMM @
959f1df (the version pinned by PR #40852's CMake FetchContent).

This is a POC-scope change. Production tuning is out of scope.
"""
import os
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py"

OLD = """    @classmethod
    def support_deep_gemm(cls) -> bool:
        \"\"\"Currently, only Hopper and Blackwell GPUs are supported.\"\"\"
        return cls.is_device_capability(90) or cls.is_device_capability_family(100)
"""

NEW = """    @classmethod
    def support_deep_gemm(cls) -> bool:
        \"\"\"Hopper (SM90) + Blackwell datacenter (SM10x) + Blackwell GB10 (SM12x).

        SM12x added by spark_vllm_docker SM120 POC patch — relies on the
        jasl/DeepGEMM SM12x fallback kernels pinned via PR #40852 CMake.
        \"\"\"
        return (
            cls.is_device_capability(90)
            or cls.is_device_capability_family(100)
            or cls.is_device_capability_family(120)
        )
"""


def main() -> int:
    if not os.path.exists(TARGET):
        print(f"[sm12x_force_dg] target not found: {TARGET}", file=sys.stderr)
        return 1
    with open(TARGET) as f:
        src = f.read()
    if "is_device_capability_family(120)" in src:
        print("[sm12x_force_dg] already patched, skipping")
        return 0
    if OLD not in src:
        print(f"[sm12x_force_dg] anchor not found in {TARGET}", file=sys.stderr)
        return 1
    backup = TARGET + ".sm12x_force_dg.orig"
    if not os.path.exists(backup):
        with open(backup, "w") as f:
            f.write(src)
        print(f"[sm12x_force_dg] backup written: {backup}")
    with open(TARGET, "w") as f:
        f.write(src.replace(OLD, NEW, 1))
    print(f"[sm12x_force_dg] patched: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
