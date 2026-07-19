#!/usr/bin/env python3
# =============================================================================
# DS3J SM121 FP8 o_proj scale-recipe patch (== the proven DS3E+DS3H net result).
#
# Single-application, one-file, pure-Python patch on the INSTALLED vLLM tree
# (no recompile). Transforms compute_fp8_einsum_recipe() so that GB10 compute-12.x
# (sm_120 / sm_121) selects the SM100-style per-row UE8M0-packed FP8 scale recipe.
#
# Proven contract (DS3H):
#   - GB10 has no dedicated sm_121a FP8 kernel; DeepGEMM 2.5.0 dispatches
#     arch_major==12 to the sm120 FP8 family, whose kernel consumes the SM100-style
#     per-row (gran_mn=1) UE8M0-packed scale, not the SM90-style FP32 block scale.
#   - DSV4-Flash weights are UE8M0-quantized (config scale_fmt=ue8m0); with
#     VLLM_USE_DEEP_GEMM_E8M0=1 set at runtime, process_weights_after_loading emits
#     INT32-packed UE8M0 (gran_mn=1) weight + activation scales. The matching recipe
#     is (1,1,128) with tma_aligned_scales=True.
#   - Bounded real-checkpoint reproducer on GB10 (wo_a layers 0/1/2): this path gives
#     mean_rel_err ~0.027 (cosine 0.9997) vs a bf16 dequant reference. The alternative
#     (1,128,128)+FP32 recipe was layout-legal but numerically wrong (rel ~0.34) and
#     overflowed to Inf on real activations.
#
# SM90 (major<=9) and SM100 (major>=10) behavior is preserved byte-for-byte in the
# else branch. Applied exactly once; fail-fast on source drift or double application.
#
# REQUIRED runtime companion (NOT optional): VLLM_USE_DEEP_GEMM_E8M0=1.
# =============================================================================
import sys, ast

P = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py"

OLD = (
    "    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)\n"
    "    tma_aligned_scales = cap.major >= 10\n"
)
NEW = (
    "    # DS3J SM121 FP8 o_proj scale contract (= proven DS3H): GB10 compute-12.x\n"
    "    # (sm_120/sm_121) has no dedicated sm_121a FP8 kernel; DeepGEMM 2.5.0 dispatches\n"
    "    # arch_major==12 to the sm120 FP8 family, whose kernel consumes the SM100-style\n"
    "    # per-row (gran_mn=1) UE8M0-packed scale. DSV4-Flash weights are UE8M0-quantized\n"
    "    # (scale_fmt=ue8m0); with VLLM_USE_DEEP_GEMM_E8M0=1 the correct recipe is\n"
    "    # (1,1,128) tma_aligned=True. Bounded real-checkpoint reproducer: rel~0.027 vs\n"
    "    # bf16; the (1,128,128)+FP32 variant was layout-legal but wrong (rel~0.34 -> Inf).\n"
    "    if cap.major == 12:\n"
    "        einsum_recipe = (1, 1, 128)\n"
    "        tma_aligned_scales = True\n"
    "    else:\n"
    "        einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)\n"
    "        tma_aligned_scales = cap.major >= 10\n"
)

src = open(P).read()

# Idempotency / drift guards.
if "if cap.major == 12:" in src and "einsum_recipe = (1, 1, 128)\n        tma_aligned_scales = True" in src:
    print("FATAL: DS3J o_proj patch appears already applied (major-12 branch present)"); sys.exit(5)
if OLD not in src:
    print("FATAL: DS3J o_proj target lines not found (upstream source drift)"); sys.exit(3)
if src.count(OLD) != 1:
    print("FATAL: expected exactly 1 occurrence of the target lines, found", src.count(OLD)); sys.exit(4)

src2 = src.replace(OLD, NEW)
ast.parse(src2)  # syntactic validity
open(P, "w").write(src2)

# Post-conditions: major-12 selects (1,1,128)/True; else branch preserved.
assert "if cap.major == 12:" in src2
assert "einsum_recipe = (1, 1, 128)\n        tma_aligned_scales = True" in src2
assert "einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)" in src2
assert "tma_aligned_scales = cap.major >= 10" in src2
print("[DS3J] o_proj.py compute_fp8_einsum_recipe major-12 branch => (1,1,128), tma_aligned=True (applied once, verified)")
