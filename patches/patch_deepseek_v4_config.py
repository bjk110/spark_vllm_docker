"""
Patch DeepseekV4Config to be compatible with transformers 5.5.4 strict dataclass.

The upstream `vllm/transformers_utils/configs/deepseek_v4.py` from branch
woosuk/dsv4-sync defines DeepseekV4Config as a minimal PretrainedConfig
subclass (no field declarations). transformers 5.5.4's PreTrainedConfig
__post_init__ runs convert_rope_params_to_dict BEFORE flushing kwargs onto
self, so it raises:

    AttributeError: 'DeepseekV4Config' object has no attribute 'max_position_embeddings'

Fix: reuse the well-defined DeepseekV3Config (same architecture family,
same field set, already declared with @strict) and only override model_type.
DeepSeek-V4-Flash config.json fields are a superset of V3's; unknown fields
are stored as additional attributes by PreTrainedConfig.
"""
import os
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/deepseek_v4.py"

NEW_BODY = '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Patched for transformers 5.5.4 compat:
#   1. Inherit from DeepseekV3Config (which has the full @strict dataclass
#      field set) instead of a bare PretrainedConfig with no fields declared.
#   2. Re-expose rope-related top-level attributes (rope_theta, rope_scaling,
#      compress_rope_theta) that PreTrainedConfig.__post_init__ moves into
#      rope_parameters. The V4 model code accesses these directly.
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)


class DeepseekV4Config(DeepseekV3Config):
    model_type = "deepseek_v4"

    def __post_init__(self, **kwargs):
        # Save originals from kwargs before super processes them.
        rope_scaling = kwargs.get("rope_scaling")
        rope_theta = kwargs.get("rope_theta", 10000.0)
        compress_rope_theta = kwargs.get("compress_rope_theta", rope_theta)
        super().__post_init__(**kwargs)
        rope_params = getattr(self, "rope_parameters", None) or {}
        if not hasattr(self, "rope_theta"):
            object.__setattr__(
                self, "rope_theta", rope_params.get("rope_theta", rope_theta)
            )
        if not hasattr(self, "rope_scaling") or self.rope_scaling is None:
            object.__setattr__(
                self,
                "rope_scaling",
                rope_scaling if rope_scaling is not None else (dict(rope_params) if rope_params else None),
            )
        if not hasattr(self, "compress_rope_theta"):
            object.__setattr__(self, "compress_rope_theta", compress_rope_theta)
'''


def main() -> int:
    if not os.path.exists(TARGET):
        print(f"[patch_dsv4_config] target not found: {TARGET}", file=sys.stderr)
        return 1
    with open(TARGET, "r") as f:
        cur = f.read()
    if cur == NEW_BODY:
        print("[patch_dsv4_config] already patched, skipping")
        return 0
    backup = TARGET + ".orig"
    if not os.path.exists(backup):
        with open(backup, "w") as f:
            f.write(cur)
        print(f"[patch_dsv4_config] backup written: {backup}")
    with open(TARGET, "w") as f:
        f.write(NEW_BODY)
    print(f"[patch_dsv4_config] patched: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
