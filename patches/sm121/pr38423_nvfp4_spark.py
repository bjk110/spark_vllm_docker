#!/usr/bin/env python3
"""
Apply Python-level patches from vllm-project/vllm#38423.
Fixes NVFP4 on DGX Spark (SM121) and RTX 50 (SM120).

Changes:
1. nvfp4_utils.py: Fix backend selection — check cutlass_fp4_supported() first
2. flashinfer_cutlass_moe.py: Fix quant_scales=None → [] for non-quantized
3. trtllm_nvfp4_moe.py: Add logger, fix import order
4. unquantized.py: Remove use_ep requirement for FlashInfer CUTLASS
5. unquantized_fused_moe_method.py: Remove use_ep arg
"""
import os
import glob
import sys

VLLM_PKG = None
for p in glob.glob("/usr/local/lib/python3.*/dist-packages/vllm"):
    VLLM_PKG = p
    break

if not VLLM_PKG:
    print("ERROR: vllm package not found")
    sys.exit(1)

print(f"Applying PR #38423 patches to: {VLLM_PKG}")
applied = []
skipped = []


def patch_file(rel_path, old, new, description):
    fpath = os.path.join(VLLM_PKG, rel_path)
    if not os.path.exists(fpath):
        skipped.append(f"{rel_path}: file not found")
        return False
    with open(fpath) as f:
        content = f.read()
    if old not in content:
        if new in content:
            skipped.append(f"{rel_path}: already patched")
        else:
            skipped.append(f"{rel_path}: anchor not found")
        return False
    content = content.replace(old, new)
    with open(fpath, "w") as f:
        f.write(content)
    applied.append(f"{rel_path}: {description}")
    return True


# 1. nvfp4_utils.py — Fix backend auto-selection
#    Check cutlass_fp4_supported() before has_flashinfer()
patch_file(
    "model_executor/layers/quantization/utils/nvfp4_utils.py",
    """        # Auto-select best available backend
        if current_platform.has_device_capability(100) and has_flashinfer():
            backend = NvFp4LinearBackend.FLASHINFER_CUTLASS
        elif cutlass_fp4_supported():""",
    """        # Auto-select best available backend.
        # cutlass_fp4_supported() checks that the vLLM NVFP4 kernels (both
        # quantization and GEMM) were compiled for the current GPU arch.
        if cutlass_fp4_supported() and has_flashinfer():
            backend = NvFp4LinearBackend.FLASHINFER_CUTLASS
        elif cutlass_fp4_supported():""",
    "fix NVFP4 backend selection for SM121",
)

# 2. flashinfer_cutlass_moe.py — Fix quant_scales
patch_file(
    "model_executor/layers/fused_moe/flashinfer_cutlass_moe.py",
    "quant_scales = None",
    "quant_scales = []",
    "fix quant_scales None→[] for non-quantized path",
)

# 3. unquantized.py — Remove use_ep requirement
patch_file(
    "model_executor/layers/fused_moe/oracle/unquantized.py",
    """def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig,
    use_ep: bool,
    use_dp: bool,""",
    """def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig,
    use_dp: bool,""",
    "remove use_ep param from select_unquantized_moe_backend",
)

# Also remove use_ep from the flashinfer_cutlass_available check
patch_file(
    "model_executor/layers/fused_moe/oracle/unquantized.py",
    """    flashinfer_cutlass_available = (
        has_flashinfer_cutlass_fused_moe()
        and use_ep""",
    """    flashinfer_cutlass_available = (
        has_flashinfer_cutlass_fused_moe()""",
    "remove use_ep check from flashinfer_cutlass_available",
)

# 4. unquantized_fused_moe_method.py — Remove use_ep arg at call site
patch_file(
    "model_executor/layers/fused_moe/unquantized_fused_moe_method.py",
    """        self.unquantized_backend = select_unquantized_moe_backend(
            moe_config=self.moe,
            use_ep=self.moe.moe_parallel_config.use_ep,
            use_dp=""",
    """        self.unquantized_backend = select_unquantized_moe_backend(
            moe_config=self.moe,
            use_dp=""",
    "remove use_ep arg from call site",
)

# Summary
print(f"\nPR #38423 PATCH SUMMARY")
print(f"Applied ({len(applied)}):")
for a in applied:
    print(f"  + {a}")
if skipped:
    print(f"Skipped ({len(skipped)}):")
    for s in skipped:
        print(f"  - {s}")
if not applied:
    print("WARNING: No patches applied!")
