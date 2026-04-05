#!/usr/bin/env python3
"""
Apply TurboQuant KV cache quantization patch to vLLM.
Based on vllm-project/vllm PR #38280 (lishunyang12).

This script:
  1. Copies 5 new TurboQuant source files into the vLLM tree
  2. Applies small edits to 8 existing files to register and wire up TurboQuant

Usage (in Dockerfile, after vLLM checkout):
  COPY patches/apply_turboquant.py /tmp/
  COPY patches/turboquant_src/ /tmp/turboquant_src/
  RUN cd /workspace/vllm-src && python3 /tmp/apply_turboquant.py
"""

import os
import re
import shutil
import sys

VLLM_ROOT = os.getcwd()
SRC_DIR = "/tmp/turboquant_src"

# ── 1. Copy new source files ──────────────────────────────────────────

NEW_FILES = [
    "vllm/model_executor/layers/quantization/turboquant.py",
    "vllm/v1/attention/backends/turboquant_attn.py",
    "vllm/v1/attention/ops/cuda_turboquant_decode.py",
    "vllm/v1/attention/ops/triton_fused_turboquant.py",
    "vllm/v1/attention/ops/triton_hadamard_turboquant.py",
]

print("[turboquant] Copying new source files...")
for relpath in NEW_FILES:
    src = os.path.join(SRC_DIR, relpath)
    dst = os.path.join(VLLM_ROOT, relpath)
    if not os.path.exists(src):
        print(f"  FATAL: {src} not found")
        sys.exit(1)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  + {relpath}")


# ── Helper ────────────────────────────────────────────────────────────

def patch_file(path, edits, description=""):
    """Apply a list of (old, new) string replacements to a file."""
    fpath = os.path.join(VLLM_ROOT, path)
    if not os.path.exists(fpath):
        print(f"  SKIP (not found): {path}")
        return False
    with open(fpath, "r") as f:
        content = f.read()
    original = content
    for old, new in edits:
        if old not in content:
            print(f"  WARN: pattern not found in {path}:")
            print(f"    {old[:80]}...")
            continue
        content = content.replace(old, new, 1)
    if content == original:
        print(f"  SKIP (no changes): {path}")
        return False
    with open(fpath, "w") as f:
        f.write(content)
    print(f"  ~ {path} ({description})")
    return True


# ── 2. Patch existing files ───────────────────────────────────────────

print("\n[turboquant] Patching existing files...")

# 2a. vllm/config/cache.py — add "turboquant" to KVCacheDType
patch_file("vllm/config/cache.py", [
    (
        '    "fp8_per_token_head",\n]',
        '    "fp8_per_token_head",\n    "turboquant",\n]'
    ),
], "add turboquant to KVCacheDType")

# 2b. vllm/utils/torch_utils.py — add turboquant dtype mapping
patch_file("vllm/utils/torch_utils.py", [
    (
        '    "fp8_ds_mla": torch.uint8,\n}',
        '    "fp8_ds_mla": torch.uint8,\n    "turboquant": torch.uint8,\n}'
    ),
], "add turboquant dtype mapping")

# 2c. vllm/model_executor/layers/quantization/__init__.py — register TurboQuant
patch_file("vllm/model_executor/layers/quantization/__init__.py", [
    (
        '    "cpu_awq",\n]',
        '    "cpu_awq",\n    "turboquant",\n]'
    ),
    (
        "    from .torchao import TorchAOConfig\n",
        "    from .torchao import TorchAOConfig\n"
        "    from .turboquant import TurboQuantVLLMConfig\n"
    ),
    (
        '        "cpu_awq": CPUAWQConfig,\n    }',
        '        "cpu_awq": CPUAWQConfig,\n'
        '        "turboquant": TurboQuantVLLMConfig,\n    }'
    ),
], "register TurboQuant quantization")

# 2d. vllm/platforms/cuda.py — add TurboQuant backend priority
patch_file("vllm/platforms/cuda.py", [
    (
        '    else:\n        if device_capability.major == 10:',
        '    else:\n'
        '        if kv_cache_dtype == "turboquant":\n'
        '            # TurboQuant only supports DECODER attention type.\n'
        '            return [\n'
        '                AttentionBackendEnum.TURBOQUANT,\n'
        '                AttentionBackendEnum.FLASH_ATTN,\n'
        '                AttentionBackendEnum.TRITON_ATTN,\n'
        '            ]\n'
        '        if device_capability.major == 10:'
    ),
], "add TurboQuant backend priority")

# 2e. vllm/utils/torch_utils.py — extend is_quantized_kv_cache
#     (in a7d79fa this function lives in torch_utils, not backend.py)
patch_file("vllm/utils/torch_utils.py", [
    (
        'return kv_cache_dtype.startswith("fp8") or kv_cache_dtype.endswith("per_token_head")',
        'return kv_cache_dtype.startswith("fp8") or kv_cache_dtype.endswith("per_token_head") or kv_cache_dtype == "turboquant"'
    ),
], "extend is_quantized_kv_cache for turboquant")

# 2f. vllm/v1/attention/backends/registry.py — register TURBOQUANT enum
patch_file("vllm/v1/attention/backends/registry.py", [
    (
        '    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"',
        '    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"\n'
        '    TURBOQUANT = (\n'
        '        "vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend"\n'
        '    )'
    ),
], "register TURBOQUANT backend enum")

# 2g. vllm/v1/attention/backends/triton_attn.py — skip (comment already removed in a7d79fa)

# 2h. vllm/model_executor/layers/attention/attention.py — TurboQuant integration
# This is the largest edit: add TQ init logic + KV cache spec

ATTENTION_EDIT_1 = (
    '                sliding_window,\n'
    '            )\n'
    '\n'
    '        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(',
    '                sliding_window,\n'
    '            )\n'
    '\n'
    '        # TurboQuant only supports full DECODER attention. Auto-skip for\n'
    '        # encoder layers and sliding-window layers in hybrid models.\n'
    '        if kv_cache_dtype == "turboquant" and (\n'
    '            attn_type != AttentionType.DECODER or sliding_window is not None\n'
    '        ):\n'
    '            kv_cache_dtype = "auto"\n'
    '            calculate_kv_scales = False\n'
    '\n'
    '        backend_kv_cache_dtype = kv_cache_dtype\n'
    '        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype('
)

ATTENTION_EDIT_2 = (
    '            self.attn_backend = get_attn_backend(\n'
    '                head_size,\n'
    '                dtype,\n'
    '                kv_cache_dtype,\n'
    '                use_mla=False,',
    '            self.attn_backend = get_attn_backend(\n'
    '                head_size,\n'
    '                dtype,\n'
    '                backend_kv_cache_dtype,\n'
    '                use_mla=False,'
)

ATTENTION_EDIT_3 = (
    '        impl_cls = self.attn_backend.get_impl_cls()\n'
    '        self.impl = impl_cls(',
    '        impl_cls = self.attn_backend.get_impl_cls()\n'
    '        # Pass original kv_cache_dtype to impl so it can detect turboquant,\n'
    '        # even though backend selection used "auto" for compatibility.\n'
    '        self.impl = impl_cls('
)

ATTENTION_EDIT_4_AFTER = '        _init_kv_cache_quant(self, quant_config, prefix)\n'
ATTENTION_EDIT_4_INSERT = '''
        # Fallback: if user only passed --kv-cache-dtype turboquant
        # without --quantization, create default TurboQuantConfig
        if kv_cache_dtype == "turboquant" and not hasattr(self, "_turboquant_config"):
            import os

            from vllm.model_executor.layers.quantization.turboquant import (
                TurboQuantConfig,
            )

            tq_lite = os.environ.get("TQ_LITE", "0") in ("1", "true", "True")
            tq_bits = int(os.environ.get("TQ_BITS", "4"))
            tq_outlier = float(os.environ.get("TQ_OUTLIER_FRAC", "0.15"))
            self._turboquant_config = TurboQuantConfig(
                bit_width=tq_bits,
                outlier_fraction=tq_outlier,
                lite_mode=tq_lite,
            )

        # Initialize TurboQuantState eagerly (not in forward) to avoid
        # torch.compile graph breaks from torch.Generator in rotation matrix
        if kv_cache_dtype == "turboquant":
            from vllm.model_executor.layers.quantization.turboquant import (
                TurboQuantState,
            )
            from vllm.model_executor.models.utils import extract_layer_index

            layer_idx = extract_layer_index(prefix)
            # Initialize on CUDA if available, CPU otherwise
            init_device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self._tq_k_state = TurboQuantState(
                config=self._turboquant_config,
                head_size=head_size,
                layer_idx=layer_idx,
                device=init_device,
            )
            # Use a separate config for V if asymmetric bit allocation
            # is enabled (value_bit_width != key bit_width).
            v_cfg = self._turboquant_config
            if self._turboquant_config.value_bit_width is not None:
                from dataclasses import replace

                v_cfg = replace(
                    self._turboquant_config,
                    bit_width=self._turboquant_config.value_bit_width,
                    value_bit_width=None,
                )
            self._tq_v_state = TurboQuantState(
                config=v_cfg,
                head_size=head_size,
                layer_idx=layer_idx + 10000,
                device=init_device,
            )
            # Calibrate outlier channels on first batch
            self._tq_needs_calibration = self._turboquant_config.outlier_fraction > 0

'''

ATTENTION_EDIT_5 = (
    '        # Should not be called for enc-dec or encoder-only attention.\n'
    '        assert self.attn_type == AttentionType.DECODER\n'
    '        quant_mode = get_kv_quant_mode(self.kv_cache_dtype)\n'
    '        if self.sliding_window is not None:',
    '        # Should not be called for enc-dec or encoder-only attention.\n'
    '        assert self.attn_type == AttentionType.DECODER\n'
    '        quant_mode = get_kv_quant_mode(self.kv_cache_dtype)\n'
    '\n'
    '        # TurboQuant: packed uint8 storage with outlier-aware layout.\n'
    '        # Slot = [outlier_bf16_bytes | packed_tq_indices | norm_fp16_bytes]\n'
    '        if self.kv_cache_dtype == "turboquant":\n'
    '            import math\n'
    '\n'
    '            cfg = self._turboquant_config\n'
    '            n_outliers = (\n'
    '                max(1, int(self.head_size * cfg.outlier_fraction))\n'
    '                if cfg.outlier_fraction > 0\n'
    '                else 0\n'
    '            )\n'
    '            normal_size = self.head_size - n_outliers\n'
    '            # Asymmetric K/V: use max bit-width for slot sizing so both\n'
    '            # K and V fit in the same slot layout.\n'
    '            k_bits = int(cfg.bit_width)\n'
    '            v_bits = int(cfg.effective_value_bit_width)\n'
    '            bits = max(k_bits, v_bits)\n'
    '            outlier_bytes = n_outliers * 2  # bf16\n'
    '            packed_bytes = math.ceil(normal_size * bits / 8)\n'
    '            norm_bytes = 2  # fp16\n'
    '            slot_bytes = outlier_bytes + packed_bytes + norm_bytes\n'
    '            # Pad slot_bytes up to next power-of-2 that divides the\n'
    '            # standard page size (head_size * 2 for bf16).  This is\n'
    '            # required by vLLM page-size unification for hybrid models\n'
    '            # (e.g. Qwen3.5 with attention + Mamba layers).\n'
    '            def _next_pow2(n):\n'
    '                p = 1\n'
    '                while p < n:\n'
    '                    p <<= 1\n'
    '                return p\n'
    '            slot_bytes = _next_pow2(slot_bytes)\n'
    '            return FullAttentionSpec(\n'
    '                block_size=block_size,\n'
    '                num_kv_heads=self.num_kv_heads,\n'
    '                head_size=slot_bytes,\n'
    '                head_size_v=slot_bytes,\n'
    '                dtype=torch.uint8,\n'
    '            )\n'
    '\n'
    '        if self.sliding_window is not None:'
)

attn_path = os.path.join(VLLM_ROOT, "vllm/model_executor/layers/attention/attention.py")
if os.path.exists(attn_path):
    with open(attn_path, "r") as f:
        attn = f.read()

    original = attn
    ok = True

    # Edit 1: TQ auto-skip + backend_kv_cache_dtype
    old1, new1 = ATTENTION_EDIT_1
    if old1 in attn:
        attn = attn.replace(old1, new1, 1)
    else:
        print("  WARN: attention.py edit 1 (auto-skip) pattern not found")
        ok = False

    # Edit 2: backend_kv_cache_dtype in get_attn_backend
    old2, new2 = ATTENTION_EDIT_2
    if old2 in attn:
        attn = attn.replace(old2, new2, 1)
    else:
        print("  WARN: attention.py edit 2 (backend dtype) pattern not found")
        ok = False

    # Edit 3: comment before impl_cls
    old3, new3 = ATTENTION_EDIT_3
    if old3 in attn:
        attn = attn.replace(old3, new3, 1)
    else:
        print("  WARN: attention.py edit 3 (impl comment) pattern not found")
        ok = False

    # Edit 4: TQ init after _init_kv_cache_quant
    if ATTENTION_EDIT_4_AFTER in attn:
        attn = attn.replace(
            ATTENTION_EDIT_4_AFTER,
            ATTENTION_EDIT_4_AFTER + ATTENTION_EDIT_4_INSERT,
            1
        )
    else:
        print("  WARN: attention.py edit 4 (TQ init) pattern not found")
        ok = False

    # Edit 5: get_kv_cache_spec TQ branch
    old5, new5 = ATTENTION_EDIT_5
    if old5 in attn:
        attn = attn.replace(old5, new5, 1)
    else:
        print("  WARN: attention.py edit 5 (kv_cache_spec) pattern not found")
        ok = False

    if attn != original:
        with open(attn_path, "w") as f:
            f.write(attn)
        print(f"  ~ vllm/model_executor/layers/attention/attention.py (TurboQuant integration, {5 if ok else '?'}/5 edits)")
    else:
        print("  SKIP (no changes): attention.py")
else:
    print("  SKIP (not found): attention.py")


# ── Done ──────────────────────────────────────────────────────────────
print("\n[turboquant] Patch complete!")
print("  Usage: vllm serve <model> --kv-cache-dtype turboquant")
