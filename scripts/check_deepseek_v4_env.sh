#!/usr/bin/env bash
# =============================================================================
# check_deepseek_v4_env.sh — verify DeepSeek-V4 container has all needed bits
#
# Modes:
#   --build-time   : no GPU assumed. Import vLLM modules, skip GPU-only checks.
#   (default)      : full check. Assumes GPU + model mounted at /models/DeepSeek-V4-Flash.
#
# Exit codes:
#   0 = OK
#   1 = import / module missing
#   2 = config / tokenizer check failed
#   3 = GPU / CUDA capability unexpected
# =============================================================================
set -u
MODE="${1:-runtime}"

python3 - "${MODE}" <<'PY'
import sys, importlib
mode = sys.argv[1]

failed = []
ok = []

def check_import(m):
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        failed.append((m, repr(e)))

mods = [
    "torch",
    "vllm",
    "vllm.model_executor.models.deepseek_v4",
    "vllm.model_executor.models.deepseek_v4_mtp",
    "vllm.model_executor.layers.deepseek_v4_attention",
    "vllm.model_executor.layers.deepseek_compressor",
    "vllm.model_executor.layers.mhc",
    "vllm.tokenizers.deepseek_v4",
    "vllm.tokenizers.deepseek_v4_encoding",
    "vllm.renderers.deepseek_v4",
    "vllm.transformers_utils.configs.deepseek_v4",
    "vllm.v1.attention.ops.deepseek_v4_ops",
    "vllm.v1.attention.backends.mla.flashmla_sparse",
    "vllm.v1.attention.backends.mla.indexer",
    "tilelang",
    "apache_tvm_ffi",
    "flashinfer",
]
for m in mods:
    check_import(m)

# Registry check — DeepseekV4ForCausalLM must be registered
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    if "DeepseekV4ForCausalLM" in archs:
        ok.append("registry:DeepseekV4ForCausalLM")
    else:
        failed.append(("registry:DeepseekV4ForCausalLM", f"not in {[a for a in archs if 'deepseek' in a.lower()]}"))
except Exception as e:
    failed.append(("registry", repr(e)))

# TokenizerMode Literal must contain 'deepseek_v4'
try:
    import typing, vllm.config.model as cfgmod
    tm = cfgmod.TokenizerMode
    members = typing.get_args(tm)
    if "deepseek_v4" in members:
        ok.append("tokenizer_mode:deepseek_v4")
    else:
        failed.append(("tokenizer_mode:deepseek_v4", f"members={members}"))
except Exception as e:
    failed.append(("tokenizer_mode", repr(e)))

# GPU checks (skip in build-time)
if mode != "--build-time":
    try:
        import torch
        if not torch.cuda.is_available():
            failed.append(("cuda", "torch.cuda.is_available() == False"))
        else:
            cc = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            ok.append(f"cuda:device={name} cc={cc}")
            if cc[0] != 12:
                failed.append(("cuda_cc", f"expected sm_12x (GB10), got sm_{cc[0]}{cc[1]}"))
    except Exception as e:
        failed.append(("cuda_probe", repr(e)))

print("=" * 70)
print(f"DeepSeek-V4 env check ({mode})")
print("=" * 70)
for m in ok:
    print(f"  OK    {m}")
for m, err in failed:
    print(f"  FAIL  {m}  — {err}")
print("=" * 70)
print(f"result: {len(ok)} ok, {len(failed)} failed")
sys.exit(0 if not failed else 1)
PY
