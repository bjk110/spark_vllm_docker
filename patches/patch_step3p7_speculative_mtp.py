#!/usr/bin/env python3
"""Backport Step-3.7 native MTP config mapping to vllm/config/speculative.py.

Root cause of failure
---------------------
`hf_config_override()` in the pinned vLLM 0.22.1 build (NGC 26.05,
vllm-spark:v022-d568-*) handles only `model_type == "step3p5"` for the
Step3p5/Step3p7 family.  Step-3.7-Flash has outer model_type ``step3p7`` and
architecture ``Step3p7ForConditionalGeneration`` -- neither matched the old
single-string check, so the draft config was returned unchanged.  At
speculative.py:704-706 the draft model_type ``step3p7`` is not in
``MTPModelTypes``, which triggered::

    NotImplementedError: Unsupported speculative method: 'mtp'

Backport source
---------------
Upstream repository : https://github.com/vllm-project/vllm
Upstream commit     : c621af16908f05270e033afd4237509902b7ba4d
Upstream file       : vllm/config/speculative.py
Upstream lines      : 496-508

Target pinned build : vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release
Target source SHA256: 536d9ed7ef910207259efdb8cf370fbfa3e6b81794c271d575fa6b9bb13d41fd

Why full replacement is avoided
--------------------------------
The installed speculative.py contains several vLLM-0.22.1-specific layout
differences from upstream main.  Replacing the file wholesale would risk
silently breaking unrelated speculative paths (Eagle, DeepSeek V3/V4, ngram,
longcat, etc.) that may have received separate maintenance on this image.
A minimal anchor-replacement keeps every other block intact.

What this patch does
--------------------
1. Extends the condition from ``model_type == "step3p5"`` to also match
   ``model_type == "step3p7"`` and both outer architecture names
   (``Step3p5ForCausalLM``, ``Step3p7ForConditionalGeneration``).
2. Captures the outer ``quantization_config`` *before* the text_config
   promotion, then restores it on the promoted config if it was missing.
   This preserves the ModelOpt/NVFP4 ``quantization_config`` that Step-3.7
   carries at the top-level VL config level; the checkpoint's BF16 MTP layers
   (45-47) are already listed in ``modules_to_not_convert`` so ModelOpt will
   not quantize them.
3. Promotes ``text_config`` so MTP layer indices and ``num_nextn_predict_layers``
   are resolved from the inner LM config (identical semantics to Step3p5).
4. Sets ``model_type = "step3p5_mtp"``, ``n_predict``, and
   ``architectures = ["Step3p5MTP"]`` exactly as before.

What this patch explicitly does NOT do
---------------------------------------
- Does not set ``quantization = None`` or remove quantization_config.
- Does not touch ``Step3p5MTP`` weight-loading logic.
- Does not modify any other model family's mapping.
- Does not download files or make network calls.
"""

import sys
import pathlib

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py"
)

# ---------------------------------------------------------------------------
# Anchor: the old step3p5-only block (production source, exact whitespace).
# ---------------------------------------------------------------------------
OLD = """\
        if hf_config.model_type == "step3p5":
            hf_config.model_type = "step3p5_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})"""

# ---------------------------------------------------------------------------
# Replacement: upstream backport (commit c621af16908f05270e033afd4237509902b7ba4d,
# lines 496-508) expanded with exact upstream whitespace.
# ---------------------------------------------------------------------------
NEW = """\
        if hf_config.model_type in ("step3p5", "step3p7") or hf_config.architectures[
            0
        ] in ("Step3p5ForCausalLM", "Step3p7ForConditionalGeneration"):
            quantization_config = getattr(hf_config, "quantization_config", None)
            hf_config = getattr(hf_config, "text_config", hf_config)
            if (
                quantization_config is not None
                and getattr(hf_config, "quantization_config", None) is None
            ):
                hf_config.update({"quantization_config": quantization_config})
            hf_config.model_type = "step3p5_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})"""

# ---------------------------------------------------------------------------
# Sentinel: a short string that is present only in the patched form.
# Chosen to be unique: the old block never contained "step3p7" or text_config.
# ---------------------------------------------------------------------------
SENTINEL = '"step3p7") or hf_config.architectures['

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
src = TARGET.read_text()

# --- ALREADY_APPLIED ---------------------------------------------------------
if SENTINEL in src:
    print(
        f"[patch] {TARGET.name}: already patched (sentinel present) — no change.",
        file=sys.stderr,
    )
    sys.exit(0)

# --- UNSUPPORTED_SOURCE -------------------------------------------------------
if OLD not in src:
    print(
        f"[patch] ERROR: expected anchor not found in {TARGET}.\n"
        "The source does not match the known pre-backport form.\n"
        "State: UNSUPPORTED_SOURCE — file not modified.",
        file=sys.stderr,
    )
    sys.exit(1)

# Safety check: ensure we are replacing exactly one occurrence.
count = src.count(OLD)
if count != 1:
    print(
        f"[patch] ERROR: anchor found {count} times (expected 1) in {TARGET}.\n"
        "State: UNSUPPORTED_SOURCE — file not modified.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- APPLY -------------------------------------------------------------------
patched = src.replace(OLD, NEW, 1)

# Verify the sentinel is present after replacement and the old anchor is gone.
assert SENTINEL in patched, "BUG: sentinel missing after replacement"
assert OLD not in patched, "BUG: old anchor still present after replacement"

TARGET.write_text(patched)
print(f"[patch] Applied Step-3.7 native MTP config mapping backport to {TARGET}")
