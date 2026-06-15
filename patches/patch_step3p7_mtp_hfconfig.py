#!/usr/bin/env python3
"""Backport Step-3.7 VLM hf_config text_config promotion to step3p5_mtp.py.

Root cause
----------
When Step-3.7-Flash (VLM, model_type=step3p7) is used with MTP speculative
decoding, the draft VllmConfig is built from the target (head) VllmConfig as a
base in _create_draft_vllm_config(). This means draft_vllm_config.model_config
still carries the outer Step3p7Config (not the text_config-promoted Step3p5Config
that speculative.py's hf_config_override() stored in draft_model_config.hf_config).

step3p5_mtp.py's three __init__ methods all do:
    config = vllm_config.model_config.hf_config
which yields Step3p7Config, which lacks vocab_size, hidden_size, etc.
(those attributes live in text_config).

Fix
---
For each of the three sites, promote to text_config if the outer object is a
VLM config (i.e. has a text_config attribute):

    _hf = vllm_config.model_config.hf_config
    config = getattr(_hf, "text_config", _hf)

This is a no-op for plain Step3p5 models (no text_config attribute).

Companion
---------
This patch must be applied together with patch_step3p7_speculative_mtp.py
(which handles speculative.py's hf_config_override step).

Target build : vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release
               (via Dockerfile.step37-nvfp4-mtp-candidate1)
"""

import sys
import pathlib

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/step3p5_mtp.py"
)

# ---------------------------------------------------------------------------
# Patch 1: Step3p5AMultiTokenPredictorLayer.__init__
# ---------------------------------------------------------------------------
OLD1 = """\
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.enorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)"""

NEW1 = """\
        _hf = vllm_config.model_config.hf_config
        config = getattr(_hf, "text_config", _hf)
        quant_config = vllm_config.quant_config
        self.enorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)"""

# ---------------------------------------------------------------------------
# Patch 2: Step3p5AMultiTokenPredictor.__init__
# ---------------------------------------------------------------------------
OLD2 = """\
        config = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding("""

NEW2 = """\
        _hf = vllm_config.model_config.hf_config
        config = getattr(_hf, "text_config", _hf)
        self.embed_tokens = VocabParallelEmbedding("""

# ---------------------------------------------------------------------------
# Patch 3: Step3p5MTP.__init__
# ---------------------------------------------------------------------------
OLD3 = """\
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config"""

NEW3 = """\
        _hf = vllm_config.model_config.hf_config
        self.config = getattr(_hf, "text_config", _hf)
        self.vllm_config = vllm_config"""

# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------
SENTINEL = 'config = getattr(_hf, "text_config", _hf)'

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
src = TARGET.read_text()

if SENTINEL in src:
    print(
        f"[patch] {TARGET.name}: already patched (sentinel present) — no change.",
        file=sys.stderr,
    )
    sys.exit(0)

errors = []
for i, (old, new) in enumerate([(OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3)], 1):
    count = src.count(old)
    if count != 1:
        errors.append(f"Patch {i}: anchor found {count} times (expected 1)")
if errors:
    for e in errors:
        print(f"[patch] ERROR: {e}", file=sys.stderr)
    print(
        "[patch] State: UNSUPPORTED_SOURCE — file not modified.",
        file=sys.stderr,
    )
    sys.exit(1)

patched = src
for old, new in [(OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3)]:
    patched = patched.replace(old, new, 1)

assert SENTINEL in patched, "BUG: sentinel missing after replacement"

TARGET.write_text(patched)
print(
    f"[patch] Applied Step-3.7 MTP hf_config text_config promotion to {TARGET}"
)
