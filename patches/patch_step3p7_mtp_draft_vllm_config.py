#!/usr/bin/env python3
"""Patch llm_base_proposer.py: replace vllm_config.model_config with draft_model_config
and set quant_config=None for MTP speculative decoding when the target model is a VLM
(step3p7) whose MTP layers are stored as BF16.

Root cause (3rd patch layer)
-----------------------------
_create_draft_vllm_config() builds the draft VllmConfig by replacing individual
sub-configs on self.vllm_config (target).  The resulting draft_vllm_config:
  (a) draft_vllm_config.model_config → still points to the TARGET's ModelConfig (Step3p7)
  (b) draft_vllm_config.quant_config → still set to the target's NVFP4 ModelOpt quant

Issue (a): All model layer classes (Step3p5DecoderLayer, Step3p5AMultiTokenPredictorLayer, …)
in step3p5.py and step3p5_mtp.py read vllm_config.model_config.hf_config to get
att_impl_type, hidden_size, etc.  When that is Step3p7Config (outer VLM config),
those attributes are missing → AttributeError.

Issue (b): With NVFP4 quant_config, MergedColumnParallelLinear.create_weights() creates
FP4-packed parameter tensors with shape [output_size // pack_factor, input_size].
But the Step-3.7-Flash checkpoint stores MTP layers 45-47 as plain BF16 (in
model-mtp-bf16.safetensors) with shape [output_size, input_size].
When load_merged_column_weight() asserts param_data.shape == loaded_weight.shape, it
fails → AssertionError.  Setting quant_config=None for the draft makes all MTP linear
layers initialize as plain BF16, matching the checkpoint.

Fix
---
In _get_model(), immediately after _create_draft_vllm_config(), replace both
  draft_vllm_config.model_config → speculative_config.draft_model_config
  draft_vllm_config.quant_config → None
using a single dataclasses.replace() call.

Companion patches
-----------------
Must be applied after:
  patch_step3p7_speculative_mtp.py    (speculative.py hf_config_override)
  patch_step3p7_mtp_hfconfig.py       (step3p5_mtp.py local config accesses)

Target build : vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release
"""

import sys
import pathlib

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py"
)

OLD = """\
        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("eagle_head"):
            model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )"""

NEW = """\
        draft_vllm_config = self._create_draft_vllm_config()
        # Replace model_config so that vllm_config.model_config.hf_config seen
        # by all layer constructors is the draft (text_config-promoted) config,
        # not the target VLM outer config.  Required for step3p7 VLM target.
        # Also set quant_config=None: MTP layers 45-47 are stored as BF16 in
        # model-mtp-bf16.safetensors; creating them with NVFP4 quant_config
        # would produce FP4-packed parameter shapes that mismatch the checkpoint.
        from dataclasses import replace as _dc_replace
        draft_vllm_config = _dc_replace(
            draft_vllm_config,
            model_config=self.speculative_config.draft_model_config,
            quant_config=None,
        )
        with set_model_tag("eagle_head"):
            model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )"""

SENTINEL = "draft_vllm_config = _dc_replace("

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
    f"[patch] Applied Step-3.7 MTP draft_vllm_config model_config+quant_config replacement to {TARGET}"
)
