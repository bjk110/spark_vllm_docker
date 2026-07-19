#!/usr/bin/env python3
# =============================================================================
# DS3J final o_proj semantic gate (runs AFTER the SM121 o_proj patch is applied).
#
# Validates compute_fp8_einsum_recipe() by RETURNED VALUES for compute majors
# 9 / 10 / 12, not by brittle string counting. Uses an isolated function-level
# test: the function source is extracted via AST and exec'd in a namespace whose
# `current_platform` is a mock — no GPU, no CUDA, no full vLLM import required.
#
# Contract (== proven DS3H):
#   major  9 -> ((1, 128, 128), False)   (SM90, unchanged)
#   major 10 -> ((1, 1, 128),   True)    (SM100, unchanged)
#   major 12 -> ((1, 1, 128),   True)    (SM121, DS3H fix)   and NOT ((1,128,128), False)
# The patch must be applied exactly once and the file must import syntactically.
# Mandatory runtime companion (documented, not optional): VLLM_USE_DEEP_GEMM_E8M0=1.
# =============================================================================
import ast, sys

P = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py"
src = open(P).read()

# Applied exactly once.
if src.count("def compute_fp8_einsum_recipe") != 1:
    print("FATAL: compute_fp8_einsum_recipe not defined exactly once"); sys.exit(2)
# Explicit major-12 branch present.
if "if cap.major == 12:" not in src:
    print("FATAL: explicit major-12 branch absent"); sys.exit(3)

tree = ast.parse(src)  # whole-file syntactic validity
fn_src = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "compute_fp8_einsum_recipe":
        fn_src = ast.get_source_segment(src, node)
if fn_src is None:
    print("FATAL: could not extract compute_fp8_einsum_recipe source"); sys.exit(4)

# Isolated function-level test with a mocked current_platform (no GPU / no vLLM import).
class _Cap:
    def __init__(self, major): self.major = major
class _CP:
    major = 0
    def get_device_capability(self, *a, **k): return _Cap(_CP.major)
ns = {"current_platform": _CP()}
exec(fn_src, ns)
f = ns["compute_fp8_einsum_recipe"]

EXPECT = {9: ((1, 128, 128), False), 10: ((1, 1, 128), True), 12: ((1, 1, 128), True)}
for maj, exp in EXPECT.items():
    _CP.major = maj
    got = f()
    if got != exp:
        print(f"FATAL: major {maj}: got {got}, expected {exp}"); sys.exit(5)
# major 12 must NOT be the obsolete DS3E recipe.
_CP.major = 12
if f() == ((1, 128, 128), False):
    print("FATAL: major 12 returned the obsolete DS3E recipe (1,128,128),False"); sys.exit(6)

print("[DS3J-R1 o_proj gate] semantic PASS — "
      "major9=((1,128,128),False) major10=((1,1,128),True) major12=((1,1,128),True); "
      "obsolete DS3E recipe absent; defined exactly once; source parses. "
      "Runtime companion VLLM_USE_DEEP_GEMM_E8M0=1 is MANDATORY.")
