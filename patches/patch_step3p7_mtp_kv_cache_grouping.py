#!/usr/bin/env python3
"""Patch 6: Step-3.7 MTP — consolidate draft attention layers into a single KV cache group.

Root cause (6th patch layer)
-------------------------------
Step3p5ForCausalLM uses a hybrid attention pattern: 12 full_attention layers
(indices 0,4,8,…,44) and 33 sliding_attention layers (all others in 0-44).
Step3p5MTP adds 3 more sliding_attention layers (indices 45, 46, 47).

When KV cache groups are allocated (`_get_kv_cache_groups_uniform_page_size`),
the algorithm sets group_size = min_count = 12 (the number of full_attention
layers) and distributes the 36 total sliding layers across 3 groups of 12
using round-robin (`layers[i::num_groups]`).  The 3 MTP layers (positions 33,
34, 35 in the sliding list) end up in groups s0, s1, s2 respectively — each
in a DIFFERENT group.

`validate_same_kv_cache_group` then asserts that ALL draft attention layers
belong to ONE KV cache group.  With three different groups the assertion fails:
  AssertionError: All drafting layers should belong to the same kv cache group

Architecture note on the fix location
--------------------------------------
`get_kv_cache_groups()` is called in the EngineCore process, NOT the Worker
process.  Any fix must read from the EngineCore's own `vllm_config` — shared
state written only in the Worker (e.g. `compilation_config` attributes) is
invisible here because Python multiprocessing does not share memory.

Fix (two-file patch)
--------------------
1. `llm_base_proposer.py` — store `_eagle_draft_attn_layer_names` on the
   Worker's `compilation_config` (kept for debugging / future use; harmless).

2. `kv_cache_utils.py` — in `get_kv_cache_groups()`, after the general-path
   groups are built, detect MTP draft layers purely from EngineCore-available
   data: if `vllm_config.speculative_config is not None`, identify layer names
   whose embedded layer index (`.layers.N.` pattern) is >= the target model's
   `num_hidden_layers`.  If those layers span multiple groups, remove them from
   those groups and place them all in a new dedicated group.  All three MTP
   layers end up in one group → assertion passes.

Memory-safety note
------------------
The physical tensors are still shared via the standard `group_size` stripe
mechanism in `get_kv_cache_config_from_groups`.  Moving MTP layers to a
separate group changes which "stripe position" their tensor slot occupies but
does NOT increase or decrease total memory — `num_blocks = avail / page_size /
max_group_size`, and max_group_size remains 12.  The 3 MTP layers receive the
same number of blocks as any other layer-slot in the pool.
"""

import sys
import pathlib

# ---------------------------------------------------------------------------
# File 1: llm_base_proposer.py — store draft layer names on compilation_config
# (kept for debugging / potential future use; not read by the EngineCore)
# ---------------------------------------------------------------------------
PROPOSER_TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py"
)

PROPOSER_OLD = """\
        self._draft_attn_layer_names = {
            name
            for name in (set(all_attn_layers.keys()) - target_attn_layer_names)
            if all_attn_layers[name].get_kv_cache_spec(self.vllm_config) is not None
        }

        if self.supports_mm_inputs:"""

PROPOSER_NEW = """\
        self._draft_attn_layer_names = {
            name
            for name in (set(all_attn_layers.keys()) - target_attn_layer_names)
            if all_attn_layers[name].get_kv_cache_spec(self.vllm_config) is not None
        }
        # Publish draft layer names on compilation_config for debugging.
        # NOTE: this attribute is NOT read by get_kv_cache_groups() because
        # that function runs in the EngineCore process and cannot see Worker
        # process memory.  Draft-layer consolidation is done via layer-index
        # detection in kv_cache_utils.py instead.
        self.vllm_config.compilation_config._eagle_draft_attn_layer_names = (
            frozenset(self._draft_attn_layer_names)
        )

        if self.supports_mm_inputs:"""

PROPOSER_SENTINEL = "_eagle_draft_attn_layer_names"

# ---------------------------------------------------------------------------
# File 2: kv_cache_utils.py — consolidate draft layers into one group.
# Detection uses layer-index in the name (>= target num_hidden_layers), which
# is available in the EngineCore's own vllm_config — no cross-process IPC.
# ---------------------------------------------------------------------------
UTILS_TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"
)

UTILS_OLD = """\
    filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)
    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)

    # Add hidden-state layers back with page aligned to the common page."""

UTILS_NEW = """\
    filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)
    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)

    # _step3p7_mtp_draft_consolidate
    # Consolidate spec-decode MTP draft layers into a single KV cache group.
    # Hybrid-attention models (e.g. Step3p7 MTP) scatter the extra MTP layers
    # across multiple sliding-window groups via the round-robin distribution in
    # _get_kv_cache_groups_uniform_page_size.  validate_same_kv_cache_group()
    # requires all draft layers to share exactly one kv_cache_gid, so we
    # re-group them here.  Draft layers are identified by layer index:
    # any layer whose embedded `.layers.N.` index is >= the target model's
    # num_hidden_layers is a draft layer.  This detection uses only
    # vllm_config data that is available in the EngineCore process.
    if vllm_config.speculative_config is not None:
        import re as _re
        _hf = getattr(vllm_config.model_config, "hf_config", None)
        _num_target = getattr(_hf, "num_hidden_layers", None)
        if _num_target is None:
            _text = getattr(_hf, "text_config", None)
            _num_target = getattr(_text, "num_hidden_layers", None)
        if _num_target is not None:
            _draft_names = frozenset(
                _ln for _ln in filtered_spec
                if (_mx := _re.search(r"\\.layers\\.(\\d+)\\.", _ln))
                and int(_mx.group(1)) >= _num_target
            )
            if _draft_names:
                _draft_gids = {
                    i
                    for i, _g in enumerate(groups)
                    if any(_ln in _draft_names for _ln in _g.layer_names)
                }
                if len(_draft_gids) > 1:
                    _ordered_draft = [
                        _ln for _ln in filtered_spec if _ln in _draft_names
                    ]
                    _draft_group = create_kv_cache_group_specs(
                        filtered_spec, [_ordered_draft]
                    )[0]
                    _rebuilt: list = []
                    for _g in groups:
                        _rem = [
                            _ln for _ln in _g.layer_names
                            if _ln not in _draft_names
                        ]
                        if _rem:
                            _rebuilt.append(
                                create_kv_cache_group_specs(filtered_spec, [_rem])[0]
                            )
                    _rebuilt.append(_draft_group)
                    groups = _rebuilt

    # Add hidden-state layers back with page aligned to the common page."""

UTILS_SENTINEL = "_step3p7_mtp_draft_consolidate"

# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------
def apply_patch(target: pathlib.Path, old: str, new: str, sentinel: str, name: str) -> None:
    src = target.read_text()
    if sentinel in src:
        print(f"[patch] {name}: already patched (sentinel present) — no change.",
              file=sys.stderr)
        return
    count = src.count(old)
    if count != 1:
        print(f"[patch] ERROR: anchor found {count} times (expected 1) in {target}.",
              file=sys.stderr)
        sys.exit(1)
    patched = src.replace(old, new, 1)
    assert sentinel in patched, f"BUG: sentinel missing after replacement in {name}"
    target.write_text(patched)
    print(f"[patch] Applied {name} to {target}")


apply_patch(
    PROPOSER_TARGET, PROPOSER_OLD, PROPOSER_NEW, PROPOSER_SENTINEL,
    "step3p7 MTP llm_base_proposer draft-layer annotation"
)
apply_patch(
    UTILS_TARGET, UTILS_OLD, UTILS_NEW, UTILS_SENTINEL,
    "step3p7 MTP kv_cache_utils group consolidation (layer-index detection)"
)
