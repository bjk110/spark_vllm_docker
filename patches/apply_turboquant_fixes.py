#!/usr/bin/env python3
"""
Apply upstream TurboQuant fixes to vLLM (post PR #38479 merge).

These patches are cherry-picked from still-OPEN PRs that fix issues or
add features needed for DGX Spark (GB10, SM121, Qwen3.5 hybrid models).

Applied PRs (still open as of vLLM main 95995bbe, 2026-04-25):
  1. PR #40074 — Triton decode index OOB fix
  2. PR #39988 — BF16 FP8 cast fix
  3. PR #39931 — Hybrid model support (Qwen3.5)

Removed (merged upstream — no patch needed):
  - PR #40060 — TURBOQUANT backend selection fix     (merged 2026-04-17)
  - PR #40092 — FA3/FA4 for prefill paths           (merged 2026-04-23)

Usage (in Dockerfile, after vLLM install):
  COPY patches/apply_turboquant_fixes.py /tmp/
  RUN python3 /tmp/apply_turboquant_fixes.py
"""

import glob
import os
import sys

SITE = "/usr/local/lib/python3.12/dist-packages"
applied = 0
failed = 0


def find_file(relpath):
    """Find a file in site-packages."""
    full = os.path.join(SITE, relpath)
    if os.path.exists(full):
        return full
    # Fallback: glob for different python versions
    for p in glob.glob(f"/usr/local/lib/python3.*/dist-packages/{relpath}"):
        return p
    return None


def patch_file(path, edits, pr_num, description):
    """Apply string replacements to a file. Returns True on success."""
    global applied, failed
    fpath = find_file(path)
    if not fpath:
        print(f"  SKIP (not found): {path}")
        failed += 1
        return False

    with open(fpath) as f:
        content = f.read()
    original = content

    for old, new in edits:
        if old not in content:
            print(f"  WARN PR#{pr_num}: pattern not found in {path}")
            print(f"    >>> {old[:80]}...")
            failed += 1
            return False
        content = content.replace(old, new, 1)

    if content == original:
        print(f"  SKIP (no changes): {path}")
        return True

    with open(fpath, "w") as f:
        f.write(content)
    print(f"  OK PR#{pr_num}: {path} — {description}")
    applied += 1
    return True


def append_to_file(path, text, pr_num, description):
    """Append text to end of file."""
    global applied, failed
    fpath = find_file(path)
    if not fpath:
        print(f"  SKIP (not found): {path}")
        failed += 1
        return False

    with open(fpath) as f:
        content = f.read()

    # Check if already applied
    if text.strip()[:60] in content:
        print(f"  SKIP (already applied): {path}")
        return True

    with open(fpath, "a") as f:
        f.write(text)
    print(f"  OK PR#{pr_num}: {path} — {description}")
    applied += 1
    return True


# =====================================================================
# PR #40074 — Triton decode index OOB fix
# =====================================================================
print("\n[PR #40074] Triton decode index OOB fix...")

patch_file(
    "vllm/v1/attention/ops/triton_turboquant_decode.py",
    [
        (
            "        block_nums = tl.load(\n"
            "            Block_table_ptr + bt_base + page_idx,\n"
            "            mask=kv_mask,",
            "        # Clamp OOB lanes to index 0 before pointer arithmetic so\n"
            "        # Triton's bounds checker does not fire on masked-out lanes.\n"
            "        safe_page_idx = tl.where(kv_mask, page_idx, 0)\n"
            "        block_nums = tl.load(\n"
            "            Block_table_ptr + bt_base + safe_page_idx,\n"
            "            mask=kv_mask,",
        ),
    ],
    40074,
    "clamp OOB page indices",
)


# =====================================================================
# PR #39988 — BF16 FP8 cast fix
# =====================================================================
print("\n[PR #39988] BF16 FP8 cast fix...")

patch_file(
    "vllm/v1/attention/ops/triton_turboquant_store.py",
    [
        (
            "k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0)",
            "k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)",
        ),
    ],
    39988,
    "BF16→FP32 before FP8 cast",
)


# =====================================================================
# PR #40060 (TURBOQUANT backend selection fix) merged upstream 2026-04-17.
# Section removed; vLLM main 95995bbe and later already include it.
# =====================================================================


# =====================================================================
# PR #39931 — Hybrid model support (Qwen3.5)
# =====================================================================
print("\n[PR #39931] Hybrid model support...")

# 4a. arg_utils.py — remove NotImplementedError, simplify boundary call
patch_file(
    "vllm/engine/arg_utils.py",
    [
        (
            "        # TurboQuant: auto-skip first/last 2 layers (boundary protection).\n"
            "        # These layers are most sensitive to quantization error.\n"
            "        # Users can add extra layers via --kv-cache-dtype-skip-layers.\n"
            "        if resolved_cache_dtype.startswith(\"turboquant_\"):\n"
            "            if model_config.is_hybrid:\n"
            "                raise NotImplementedError(\n"
            "                    \"TurboQuant KV cache is not supported for hybrid \"\n"
            "                    \"(attention + Mamba) models. Boundary layer protection \"\n"
            "                    \"requires uniform attention layers.\"\n"
            "                )\n"
            "            from vllm.model_executor.layers.quantization.turboquant.config import (\n"
            "                TurboQuantConfig,\n"
            "            )\n"
            "\n"
            "            num_layers = model_config.hf_text_config.num_hidden_layers\n"
            "            boundary = TurboQuantConfig.get_boundary_skip_layers(num_layers)\n"
            "            existing = set(cache_config.kv_cache_dtype_skip_layers)\n"
            "            merged = sorted(existing | set(boundary), key=lambda x: int(x))\n"
            "            cache_config.kv_cache_dtype_skip_layers = merged\n"
            "            logger.info(\n"
            "                \"TQ: skipping layers %s for boundary protection (num_layers=%d)\",\n"
            "                merged,\n"
            "                num_layers,\n"
            "            )",
            "        if resolved_cache_dtype.startswith(\"turboquant_\"):\n"
            "            from vllm.model_executor.layers.quantization.turboquant.config import (\n"
            "                TurboQuantConfig,\n"
            "            )\n"
            "\n"
            "            boundary = TurboQuantConfig.get_boundary_skip_layers(model_config)\n"
            "            existing = set(cache_config.kv_cache_dtype_skip_layers)\n"
            "            cache_config.kv_cache_dtype_skip_layers = sorted(\n"
            "                existing | set(boundary), key=int\n"
            "            )",
        ),
    ],
    39931,
    "remove hybrid NotImplementedError, simplify boundary",
)

# 4b. config.py — add imports
patch_file(
    "vllm/model_executor/layers/quantization/turboquant/config.py",
    [
        (
            '"""TurboQuant configuration."""\n'
            "\n"
            "import math\n"
            "from dataclasses import dataclass",
            '"""TurboQuant configuration."""\n'
            "\n"
            "from __future__ import annotations\n"
            "\n"
            "import logging\n"
            "import math\n"
            "from dataclasses import dataclass\n"
            "from typing import TYPE_CHECKING\n"
            "\n"
            "if TYPE_CHECKING:\n"
            "    from vllm.config import ModelConfig\n"
            "\n"
            "logger = logging.getLogger(__name__)",
        ),
    ],
    39931,
    "add hybrid model imports",
)

# 4c. config.py — refactor get_boundary_skip_layers
patch_file(
    "vllm/model_executor/layers/quantization/turboquant/config.py",
    [
        (
            "    @staticmethod\n"
            "    def get_boundary_skip_layers(num_layers: int, n: int = 2) -> list[str]:\n"
            '        """Get layer indices to skip TQ compression (boundary protection).\n'
            "\n"
            "        Returns first N and last N layer indices as strings, suitable for\n"
            "        kv_cache_dtype_skip_layers.\n"
            '        """\n'
            "        if n <= 0 or num_layers <= 0:",
            "    @staticmethod\n"
            "    def get_boundary_skip_layers(\n"
            "        model_config: ModelConfig,\n"
            "        n: int = 2,\n"
            "    ) -> list[str]:\n"
            '        """Layer indices to skip TQ compression (boundary protection).\n'
            "\n"
            "        For hybrid models (attention + Mamba/linear-attention), boundary\n"
            "        protection is disabled -- hybrids typically have only 8-12\n"
            "        full-attention layers and a hard n=2 on each side would cover\n"
            "        ~40%% of them.\n"
            "\n"
            "        For dense models, skips first N and last N attention layers.\n"
            '        """\n'
            "        if model_config.is_hybrid:\n"
            "            attn_indices = _get_full_attention_layer_indices(model_config)\n"
            "            if not attn_indices:\n"
            "                raise NotImplementedError(\n"
            '                    "TurboQuant KV cache requires identifiable "\n'
            '                    "full-attention layers, but none were found in "\n'
            '                    "the hybrid model config."\n'
            "                )\n"
            '            logger.info("TQ hybrid: full-attention layers %s", attn_indices)\n'
            "            return []\n"
            "\n"
            "        num_layers = model_config.hf_text_config.num_hidden_layers\n"
            "        if n <= 0 or num_layers <= 0:",
        ),
    ],
    39931,
    "refactor boundary skip for hybrid models",
)

# 4d. config.py — fix return type annotation
patch_file(
    "vllm/model_executor/layers/quantization/turboquant/config.py",
    [
        (
            'def from_cache_dtype(cache_dtype: str, head_dim: int) -> "TurboQuantConfig":',
            "def from_cache_dtype(cache_dtype: str, head_dim: int) -> TurboQuantConfig:",
        ),
    ],
    39931,
    "fix return type annotation",
)

# 4e. config.py — append _get_full_attention_layer_indices function
append_to_file(
    "vllm/model_executor/layers/quantization/turboquant/config.py",
    '''

def _get_full_attention_layer_indices(model_config: ModelConfig) -> list[int]:
    """Global indices of full-attention layers in a hybrid model.

    Covers conventions: ``layer_types`` (Qwen3.5/Next),
    ``layers_block_type`` (Jamba/Zamba2), ``attn_type_list`` (Minimax).
    """
    text_cfg = model_config.hf_text_config
    hf_cfg = model_config.hf_config

    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types is not None:
        return [
            i for i, t in enumerate(layer_types)
            if t in ("full_attention", "attention")
        ]

    layers_block_type = getattr(text_cfg, "layers_block_type", None)
    if layers_block_type is not None:
        return [
            i for i, t in enumerate(layers_block_type)
            if t in ("attention", "hybrid")
        ]

    attn_type_list = getattr(hf_cfg, "attn_type_list", None)
    if attn_type_list is not None:
        return [i for i, t in enumerate(attn_type_list) if t == 1]

    return []
''',
    39931,
    "add _get_full_attention_layer_indices",
)

# 4f. turboquant_attn.py — ROCm flash_attn wrapper
patch_file(
    "vllm/v1/attention/backends/turboquant_attn.py",
    [
        (
            "_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()\n"
            "if _HAS_FLASH_ATTN:\n"
            "    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func",
            "_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()\n"
            "if _HAS_FLASH_ATTN:\n"
            "    import inspect as _inspect\n"
            "\n"
            "    from vllm.v1.attention.backends.fa_utils import (\n"
            "        flash_attn_varlen_func as _flash_attn_varlen_func,\n"
            "    )\n"
            "\n"
            "    try:\n"
            "        _FA_SUPPORTS_OUT = (\n"
            '            "out" in _inspect.signature(_flash_attn_varlen_func).parameters\n'
            "        )\n"
            "    except (TypeError, ValueError):\n"
            "        _FA_SUPPORTS_OUT = False\n"
            "\n"
            "    def flash_attn_varlen_func(*args, out=None, **kwargs):\n"
            '        kwargs.pop("out", None)\n'
            "        if _FA_SUPPORTS_OUT and out is not None:\n"
            '            kwargs["out"] = out\n'
            "            return _flash_attn_varlen_func(*args, **kwargs)\n"
            "        result = _flash_attn_varlen_func(*args, **kwargs)\n"
            "        if out is not None:\n"
            "            out.copy_(result)\n"
            "            return out\n"
            "        return result\n",
        ),
    ],
    39931,
    "flash_attn ROCm wrapper (out= kwarg)",
)

# 4g. interface.py — TQ page size for hybrid block alignment
patch_file(
    "vllm/platforms/interface.py",
    [
        (
            "        else:\n"
            "            attn_page_size_1_token = FullAttentionSpec(\n"
            "                block_size=1,\n"
            "                num_kv_heads=model_config.get_num_kv_heads(parallel_config),\n"
            "                head_size=model_config.get_head_size(),\n"
            "                dtype=kv_cache_dtype,\n"
            "                kv_quant_mode=kv_quant_mode,\n"
            "            ).page_size_bytes",
            '        elif cache_config.cache_dtype.startswith("turboquant_"):\n'
            "            from vllm.model_executor.layers.quantization.turboquant.config import (\n"
            "                TurboQuantConfig,\n"
            "            )\n"
            "            from vllm.v1.kv_cache_interface import TQFullAttentionSpec\n"
            "\n"
            "            tq_cfg = TurboQuantConfig.from_cache_dtype(\n"
            "                cache_config.cache_dtype, model_config.get_head_size()\n"
            "            )\n"
            "            tq_page = TQFullAttentionSpec(\n"
            "                block_size=1,\n"
            "                num_kv_heads=model_config.get_num_kv_heads(parallel_config),\n"
            "                head_size=model_config.get_head_size(),\n"
            "                head_size_v=model_config.get_head_size(),\n"
            "                dtype=kv_cache_dtype,\n"
            "                kv_quant_mode=kv_quant_mode,\n"
            "                tq_slot_size=tq_cfg.slot_size_aligned,\n"
            "            ).page_size_bytes\n"
            "            if cache_config.kv_cache_dtype_skip_layers:\n"
            "                skip_page = FullAttentionSpec(\n"
            "                    block_size=1,\n"
            "                    num_kv_heads=model_config.get_num_kv_heads(parallel_config),\n"
            "                    head_size=model_config.get_head_size(),\n"
            "                    dtype=kv_cache_dtype,\n"
            "                    kv_quant_mode=kv_quant_mode,\n"
            "                ).page_size_bytes\n"
            "                attn_page_size_1_token = max(tq_page, skip_page)\n"
            "            else:\n"
            "                attn_page_size_1_token = tq_page\n"
            "        else:\n"
            "            attn_page_size_1_token = FullAttentionSpec(\n"
            "                block_size=1,\n"
            "                num_kv_heads=model_config.get_num_kv_heads(parallel_config),\n"
            "                head_size=model_config.get_head_size(),\n"
            "                dtype=kv_cache_dtype,\n"
            "                kv_quant_mode=kv_quant_mode,\n"
            "            ).page_size_bytes",
        ),
    ],
    39931,
    "TQ page size for hybrid block alignment",
)


# =====================================================================
# PR #40092 (FA3/FA4 for prefill paths) merged upstream 2026-04-23.
# Section removed; vLLM main 95995bbe and later already include it.
# =====================================================================


# =====================================================================
# Summary
# =====================================================================
print(f"\n{'='*60}")
print(f"TurboQuant fixes: {applied} applied, {failed} failed")
if failed > 0:
    print("WARNING: Some patches could not be applied.")
    print("This may be due to upstream code changes.")
    print("Review the warnings above.")
    sys.exit(1)
else:
    print("All patches applied successfully!")
    print("\nUsage: vllm serve <model> --kv-cache-dtype turboquant_k8v4")
