#!/usr/bin/env python3
"""Add opt-in per-module CUDA cache release after ModelOpt NVFP4 MoE conversion.

Background
----------
Each ModelOptNvFp4FusedMoE.process_weights_after_loading() call allocates
approximately 6.68 GiB of scratch blocks in the CUDA caching allocator that
are not immediately returned to the driver after the NVFP4-to-MARLIN repacking
completes. On DGX Spark GB10 (UMA, 121.63 GiB shared CPU+GPU pool) with
42 MoE modules, the cumulative reserved pool grows monotonically until it
exceeds Ray's OOM threshold (~90 % of node RAM), killing the worker.

Fix
---
After each completed ModelOptNvFp4FusedMoE module in
process_weights_after_loading(), call torch.cuda.empty_cache() when
VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE is "1". This resets the reserved
pool back to the stable post-weight-load baseline (~65 GiB) after each of the
42 conversions, preventing cumulative growth.

Feature flag
------------
  VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE=0   disabled (default; upstream-equivalent)
  VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE=1   enabled
  any other value                                 warning printed, feature disabled

Class guard
-----------
Activation requires BOTH:
  type(quant_method).__name__ == "ModelOptNvFp4FusedMoE"
  "modelopt" in type(quant_method).__module__
Both conditions must hold to avoid accidental activation on other quant paths
that may share a partial name. Direct import of ModelOptNvFp4FusedMoE is
avoided here because utils.py is imported early in the vLLM import chain and a
cross-package import would risk circular dependency; the dual string check is
the safe equivalent.

Validated on
------------
Dual DGX Spark GB10 (SM121), driver 610.43.02, TP=2 EP=2, Ray distributed
executor, stepfun-ai/Step-3.7-Flash-NVFP4 (42 ModelOptNvFp4FusedMoE modules),
vLLM 0.22.1 NGC 26.05. Cumulative overhead per rank: ~402 ms / ~279 ms.
"""
import pathlib
import sys

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py"
)

GUARD = "_SPARK_EC_ENABLED"  # idempotency sentinel; absent in upstream source

src = TARGET.read_text()

if GUARD in src:
    print(
        f"[patch] {TARGET.name}: already patched ({GUARD} present) — no change.",
        file=sys.stderr,
    )
    sys.exit(0)

# ── 1. Module-level feature gate  ────────────────────────────────────────────
# Insert immediately after the module docstring and before `import inspect`.
MODULE_ANCHOR = '"""Utilities for selecting and loading models."""\n\nimport inspect'
if MODULE_ANCHOR not in src:
    print(f"[patch] ERROR: module-level anchor not found in {TARGET}", file=sys.stderr)
    sys.exit(1)

MODULE_GATE = '''\
import os as _spark_os
import time as _spark_time

# vllm-spark: feature gate for per-module CUDA cache release after ModelOpt
# NVFP4 MoE conversion. Evaluated once at import time.
_SPARK_EC_RAW = _spark_os.environ.get("VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE", "0")
if _SPARK_EC_RAW not in ("0", "1"):
    import sys as _spark_sys
    print(
        f"[vllm-spark] WARN: VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE={_SPARK_EC_RAW!r}"
        " is not '0' or '1' — feature disabled",
        file=_spark_sys.stderr, flush=True,
    )
    _SPARK_EC_ENABLED = False
else:
    _SPARK_EC_ENABLED = _SPARK_EC_RAW == "1"
del _SPARK_EC_RAW

'''

src = src.replace(
    MODULE_ANCHOR,
    '"""Utilities for selecting and loading models."""\n\n' + MODULE_GATE + "import inspect",
    1,
)

# ── 2. Function body: counter init + enable log  ─────────────────────────────
# Replace the opening of process_weights_after_loading to insert counters
# and the per-run INFO log before the main module loop.
FUNC_ANCHOR = (
    ") -> None:\n"
    "    for _, module in model.named_modules():\n"
    '        quant_method = getattr(module, "quant_method", None)\n'
)
if FUNC_ANCHOR not in src:
    print(
        f"[patch] ERROR: function-body anchor not found in {TARGET}", file=sys.stderr
    )
    sys.exit(1)

FUNC_REPLACEMENT = (
    ") -> None:\n"
    "    _spark_ec_count = 0\n"
    "    _spark_ec_total_ns = 0\n"
    "    if _SPARK_EC_ENABLED:\n"
    '        logger.info("ModelOpt MoE post-load cache release workaround enabled.")\n'
    "    for _, module in model.named_modules():\n"
    '        quant_method = getattr(module, "quant_method", None)\n'
)
src = src.replace(FUNC_ANCHOR, FUNC_REPLACEMENT, 1)

# ── 3. Per-module hook + loop summary  ───────────────────────────────────────
# Anchor: the end of the first quant-method loop body, immediately before the
# comment that opens the second (Attention/MLA) loop.  The second loop uses
# `module.process_weights_after_loading(model_config.dtype)`, not
# `quant_method.process_weights_after_loading(module)`, so this string is
# unique within the file.
HOOK_ANCHOR = (
    "            with device_loading_context(module, target_device):\n"
    "                quant_method.process_weights_after_loading(module)\n"
    "\n"
    "    # Initialize post-load attention weights"
)
if HOOK_ANCHOR not in src:
    print(f"[patch] ERROR: hook anchor not found in {TARGET}", file=sys.stderr)
    sys.exit(1)

HOOK_REPLACEMENT = """\
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)
            # vllm-spark: release CUDA caching-allocator blocks after each
            # ModelOptNvFp4FusedMoE module to prevent cumulative OOM on UMA.
            # Both the class name and module path are checked so the hook
            # does not fire on other quant methods that share a name fragment.
            if (
                _SPARK_EC_ENABLED
                and type(quant_method).__name__ == "ModelOptNvFp4FusedMoE"
                and "modelopt" in type(quant_method).__module__
            ):
                _t0 = _spark_time.monotonic_ns()
                import torch as _spark_torch
                _spark_torch.cuda.empty_cache()
                _spark_ec_total_ns += _spark_time.monotonic_ns() - _t0
                _spark_ec_count += 1
                logger.debug(
                    "ModelOpt MoE cache release: module %d complete", _spark_ec_count
                )

    if _SPARK_EC_ENABLED and _spark_ec_count > 0:
        logger.info(
            "ModelOpt MoE post-load cache release complete: "
            "%d modules flushed, cumulative %.0f ms",
            _spark_ec_count, _spark_ec_total_ns / 1e6,
        )

    # Initialize post-load attention weights"""

src = src.replace(HOOK_ANCHOR, HOOK_REPLACEMENT, 1)

TARGET.write_text(src)
print(f"[patch] Applied ModelOpt NVFP4 MoE cache-release hook to {TARGET}")
