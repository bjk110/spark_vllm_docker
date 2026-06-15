#!/usr/bin/env python3
"""Patch vllm/transformers_utils/config.py: handle local model paths in
maybe_override_with_speculators() for speculative decoding with native MTP.

Root cause (4th patch layer)
-----------------------------
When --speculative-config {"method":"mtp",...} is passed, vLLM calls
maybe_override_with_speculators() during engine config creation.  The function
calls PretrainedConfig.get_config_dict(model_path) to check if the model has
a HuggingFace "speculators_config" embedded in its config.json.

In transformers 5.10.x, PretrainedConfig.get_config_dict() validates the path
as a HuggingFace repo ID before reading it as a local file.  Absolute local
paths (starting with '/') fail validation:
  HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
    '/models/Step-3.7-Flash-NVFP4'. Use `repo_type` argument if needed.

This only triggers when speculative_config is provided (MTP mode); no-MTP
production mode never calls this function.

The Step-3.7-Flash-NVFP4 model does not have a speculators_config, so the
function would return early after the check anyway.  The only needed change is
to handle the case where get_config_dict() fails for local paths.

Fix
---
For local directory paths, read config.json directly instead of going through
PretrainedConfig.get_config_dict().  This avoids the HuggingFace Hub
path-validation codepath entirely.  Falls back to the original call for all
other cases (HuggingFace repo IDs, remote URLs, etc.).

Target build : vllm-spark:step37-nvfp4-mtp-candidate1-canonical (sha256:a23199cab4fb)
Source SHA256: f0350704a6d30caf9209ae498424935fef3d2746bfae6f4ea9f83798f93b52c5
"""

import sys
import pathlib

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/config.py"
)

OLD = """\
    kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE
    config_dict, _ = PretrainedConfig.get_config_dict(
        model if gguf_model_repo is None else gguf_model_repo,
        revision=revision,
        token=hf_token,
        **without_trust_remote_code(kwargs),
    )"""

NEW = """\
    kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE
    # For local directory paths, read config.json directly to avoid
    # HuggingFace Hub repo-ID validation in transformers 5.x which rejects
    # absolute paths like '/models/Step-3.7-Flash-NVFP4'.
    import os as _os, json as _json
    _model_for_cfg = model if gguf_model_repo is None else gguf_model_repo
    _local_cfg = (
        _os.path.join(_model_for_cfg, "config.json")
        if _os.path.isdir(_model_for_cfg)
        else None
    )
    if _local_cfg and _os.path.exists(_local_cfg):
        with open(_local_cfg) as _f:
            config_dict = _json.load(_f)
    else:
        config_dict, _ = PretrainedConfig.get_config_dict(
            _model_for_cfg,
            revision=revision,
            token=hf_token,
            **without_trust_remote_code(kwargs),
        )"""

SENTINEL = "# For local directory paths, read config.json directly"

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
    f"[patch] Applied Step-3.7 MTP local-path speculators fix to {TARGET}"
)
