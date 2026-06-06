#!/usr/bin/env python3
"""Add packed_modules_mapping to DeepseekV4ForCausalLM.

Without this 1-line shim, FP8_BLOCK on attn fails when loading
pastapaul/DeepSeek-V4-Flash-W4A16-FP8 (or any compressed-tensors W4A16
DSV4 build) with:

    ValueError: Unable to find matching target for
    model.layers.0.attn.fused_wqa_wkv in the compressed-tensors config.

kylesayrs's PR #41276 references `self.packed_modules_mapping` (line 205)
but does not define it on the class — that's what this patch supplies.

Source: pasta-paul/dsv4-flash-w4a16-fp8 / scripts/patch_v4_packed_mapping.py
        (verified on dual DGX Spark GB10 SM 12.1a, 2026-05-06)

Usage (in Dockerfile, after vLLM source clone):
  COPY patches/dsv4/apply_dsv4_packed_mapping.py /tmp/
  RUN python3 /tmp/apply_dsv4_packed_mapping.py /workspace/vllm-src/vllm/model_executor/models/deepseek_v4.py
"""
import sys

F = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/workspace/vllm-src/vllm/model_executor/models/deepseek_v4.py"
)

with open(F) as f:
    src = f.read()

# Match either base class signature seen in the wild:
#   PR #40991 / ds4-sm120(-full): class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
#   PR #41834 / codex/ds4-sm120-min-enable: class DeepseekV4ForCausalLM(nn.Module):
ANCHOR = '''    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")'''

INJECTION = '''    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    # PATCH (paul/dsv4): mapping from fused module names to their constituent
    # shard names. Used by is_layer_skipped() and the compressed-tensors loader
    # to determine the quantization scheme for fused layers (which are constructed
    # at vLLM init from the underlying ColumnParallelLinear shards). Without this,
    # FP8_BLOCK on attn fails with "Unable to find matching target for
    # model.layers.0.attn.fused_wqa_wkv".
    packed_modules_mapping = {
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
        "gate_up_proj": ["w1", "w3"],
    }'''

old = ANCHOR
new = INJECTION

if "fused_wqa_wkv" in src and "packed_modules_mapping = {" in src:
    print(f"SKIP (already applied): {F}")
    sys.exit(0)

if old not in src:
    print(f"FATAL: anchor (model_cls + hf_to_vllm_mapper block) not found in {F}")
    print("This may indicate the jasl branch has diverged. Inspect the file.")
    sys.exit(1)

# Sanity check — make sure the anchor is unique inside DeepseekV4ForCausalLM
# class scope (not somewhere else in the module).
occurrences = src.count(old)
if occurrences != 1:
    print(f"FATAL: expected exactly 1 anchor occurrence, found {occurrences}")
    sys.exit(1)

src = src.replace(old, new)
with open(F, "w") as f:
    f.write(src)
print(f"OK: patched {F} (added packed_modules_mapping with 3 entries)")
