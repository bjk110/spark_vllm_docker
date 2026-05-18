#!/usr/bin/env python3
"""Direct safetensors-level FP8_DYNAMIC quantization for
wangzhang/Qwen3.5-122B-A10B-abliterix.

Bypasses transformers/llmcompressor because:
- llmcompressor 0.10 requires transformers <=4.57.6
- Qwen3.5 MoE class is only available in transformers >=5.5
- The model repo does not ship modeling/configuration .py files

So we operate at safetensors level:
- Scan tensors, identify 2D BF16 Linear weights (with positive + negative filters)
- Apply per-channel FP8_E4M3FN scale: scale = absmax(W,dim=1)/448, W_q = (W/scale).to(fp8)
- Save weight + weight_scale in compressed-tensors-compatible layout
- Patch config.json with a quantization_config block

vLLM 0.20+ with --quantization compressed-tensors auto-detects from config.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_E4M3_MAX = 448.0

# Modules to KEEP in BF16 (not quantized). These match the module name
# (the tensor name with trailing '.weight' stripped).
IGNORE_LITERAL_MODULES = {"lm_head"}
IGNORE_MODULE_REGEX = [
    re.compile(r".*\.mlp\.gate$"),
    re.compile(r".*\.mlp\.shared_expert_gate$"),
]

# Tensor name substring exclusions (catches embeddings, norms, etc.)
NEVER_QUANTIZE_SUBSTR = (
    "embed_tokens",
    "lm_head",
)
NEVER_QUANTIZE_SUFFIX = (
    "norm.weight",
)


def module_from_weight(tensor_name: str):
    if tensor_name.endswith(".weight"):
        return tensor_name[: -len(".weight")]
    return None


def should_quantize(tensor_name: str, shape, dtype) -> bool:
    if not tensor_name.endswith(".weight"):
        return False
    if len(shape) != 2:
        return False
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    for s in NEVER_QUANTIZE_SUBSTR:
        if s in tensor_name:
            return False
    for s in NEVER_QUANTIZE_SUFFIX:
        if tensor_name.endswith(s):
            return False
    module = module_from_weight(tensor_name)
    if module is None:
        return False
    if module in IGNORE_LITERAL_MODULES:
        return False
    for rx in IGNORE_MODULE_REGEX:
        if rx.match(module):
            return False
    return True


def quantize_fp8(w: torch.Tensor):
    w_fp32 = w.to(torch.float32)
    absmax = w_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = absmax / FP8_E4M3_MAX
    w_scaled = (w_fp32 / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return w_scaled.to(torch.float8_e4m3fn), scale.to(torch.bfloat16)


def build_quant_config_block():
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "kv_cache_scheme": None,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "channel",
                    "symmetric": True,
                    "dynamic": False,
                    "observer": "minmax",
                    "block_structure": None,
                    "group_size": None,
                    "actorder": None,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                    "observer": None,
                    "block_structure": None,
                    "group_size": None,
                    "actorder": None,
                },
                "output_activations": None,
            },
        },
        "ignore": [
            "lm_head",
            r"re:.*\.mlp\.gate$",
            r"re:.*\.mlp\.shared_expert_gate$",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/home/bjk110/Documents/Models/wangzhang/Qwen3.5-122B-A10B-abliterix",
    )
    ap.add_argument(
        "--output",
        default="/home/bjk110/Documents/Models/wangzhang/Qwen3.5-122B-A10B-abliterix-FP8",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan and report counts without writing output")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        print(f"ERROR: input not found: {inp}", file=sys.stderr)
        return 1

    idx = json.loads((inp / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]

    shards = {}
    for tname, sname in weight_map.items():
        shards.setdefault(sname, []).append(tname)

    if args.dry_run:
        n_q = n_s = 0
        sample_q = []
        sample_s = []
        for sname in sorted(shards.keys()):
            with safe_open(inp / sname, framework="pt") as f:
                for tname in shards[sname]:
                    sl = f.get_slice(tname)
                    sh = tuple(sl.get_shape())
                    dt = sl.get_dtype()
                    dtype = {
                        "BF16": torch.bfloat16,
                        "F16": torch.float16,
                        "F32": torch.float32,
                    }.get(dt)
                    if dtype is None:
                        n_s += 1
                        continue
                    if should_quantize(tname, sh, dtype):
                        n_q += 1
                        if len(sample_q) < 12:
                            sample_q.append((tname, sh))
                    else:
                        n_s += 1
                        if len(sample_s) < 12:
                            sample_s.append((tname, sh))
        print(f"would quantize: {n_q}")
        print(f"would skip    : {n_s}")
        print("sample quantize:")
        for n, s in sample_q:
            print(f"  Q  {n} {s}")
        print("sample skip:")
        for n, s in sample_s:
            print(f"  S  {n} {s}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    new_weight_map = {}
    quant_count = skip_count = 0

    shard_names = sorted(shards.keys())
    for i, sname in enumerate(shard_names):
        print(f"[{i+1}/{len(shard_names)}] {sname} "
              f"({len(shards[sname])} tensors)…", flush=True)
        new_shard = {}
        with safe_open(inp / sname, framework="pt") as f:
            for tname in shards[sname]:
                t = f.get_tensor(tname)
                if should_quantize(tname, t.shape, t.dtype):
                    w_q, sc = quantize_fp8(t)
                    mod = module_from_weight(tname)
                    new_shard[f"{mod}.weight"] = w_q
                    new_shard[f"{mod}.weight_scale"] = sc
                    new_weight_map[f"{mod}.weight"] = sname
                    new_weight_map[f"{mod}.weight_scale"] = sname
                    quant_count += 1
                else:
                    new_shard[tname] = t
                    new_weight_map[tname] = sname
                    skip_count += 1
        save_file(new_shard, str(out / sname))
        del new_shard
        print(f"  written. running totals: Q={quant_count}, S={skip_count}",
              flush=True)

    total_size = sum((out / s).stat().st_size for s in shard_names)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size},
                    "weight_map": new_weight_map}, indent=2, sort_keys=True)
    )

    for fn in ("generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "chat_template.jinja",
               "README.md", ".gitattributes"):
        src = inp / fn
        if src.exists():
            shutil.copy2(src, out / fn)

    cfg = json.loads((inp / "config.json").read_text())
    cfg["quantization_config"] = build_quant_config_block()
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    print("\n=== Summary ===")
    print(f"Quantized tensors: {quant_count}")
    print(f"Skipped tensors  : {skip_count}")
    print(f"Total size       : {total_size / 1e9:.2f} GB")
    print(f"Output           : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
