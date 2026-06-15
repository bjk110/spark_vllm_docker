#!/usr/bin/env python3
"""Patch vllm/v1/spec_decode/llm_base_proposer.py: add Step3p7ForConditionalGeneration
to the image_token_id model list in load_model().

Root cause (5th patch layer)
-----------------------------
llm_base_proposer.py:load_model() sets self.model.config.image_token_index on
the draft model config for VLM targets that use speculative decoding.  The code
has explicit branches for models that name the field 'image_token_id' (e.g.
Qwen3VLMoeForConditionalGeneration, Gemma4ForConditionalGeneration) and falls
back to 'image_token_index' for all others.

Step3p7Config (Step-3.7-Flash-NVFP4 outer VLM config) uses 'image_token_id'
(value 128001) and does not have 'image_token_index'.  Without this patch the
fallback 'else' branch raises:
  AttributeError: 'Step3p7Config' object has no attribute 'image_token_index'.
    Did you mean: 'image_token_id'?

Fix
---
Add "Step3p7ForConditionalGeneration" to the list of models whose image token
field is accessed as config.image_token_id, matching the existing pattern for
Qwen3VLMoeForConditionalGeneration / Gemma4ForConditionalGeneration.
"""

import sys
import pathlib

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py"
)

OLD = """\
                "Gemma4ForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id"""

NEW = """\
                "Gemma4ForConditionalGeneration",
                "Step3p7ForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id"""

SENTINEL = '"Step3p7ForConditionalGeneration",'

src = TARGET.read_text()

if SENTINEL in src:
    print(
        f"[patch] {TARGET.name}: already patched (sentinel present) — no change.",
        file=sys.stderr,
    )
    sys.exit(0)

count = src.count(OLD)
if count != 1:
    print(
        f"[patch] ERROR: anchor found {count} times (expected 1) in {TARGET}.",
        file=sys.stderr,
    )
    sys.exit(1)

patched = src.replace(OLD, NEW, 1)
assert SENTINEL in patched, "BUG: sentinel missing after replacement"

TARGET.write_text(patched)
print(
    f"[patch] Applied Step-3.7 MTP image_token_index fix to {TARGET}"
)
