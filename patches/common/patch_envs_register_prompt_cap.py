#!/usr/bin/env python3
"""
Patch vllm/envs.py to register VLLM_SPARK_MAX_PROMPT_TOKENS.

Without this, vLLM's startup validator emits one warning per worker process:
  WARNING [envs.py:...] Unknown vLLM environment variable detected:
  VLLM_SPARK_MAX_PROMPT_TOKENS

Patch is idempotent — re-running prints a notice and exits 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/envs.py")

NEW_ENTRY = '''    # vllm-spark patch: see patches/common/patch_prompt_token_admission.py —
    # per-request prompt-token admission control for the GB10 UMA long-context path.
    "VLLM_SPARK_MAX_PROMPT_TOKENS": lambda: int(
        os.getenv("VLLM_SPARK_MAX_PROMPT_TOKENS", "0") or "0"
    ),
'''

ANCHOR_CANDIDATES = [
    # jasl HEAD 2026-06-01 (0440ee5c22) — new last entry
    '    "VLLM_NIC_SELECTION_VARS": lambda: os.getenv("VLLM_NIC_SELECTION_VARS", ""),\n}',
    # jasl edc82b614f51 (pre-2026-06-01) — previous last entry
    '    "VLLM_USE_SPINLOOP_EXT": lambda: bool(int(os.getenv("VLLM_USE_SPINLOOP_EXT", "0"))),\n}',
    # vllm-spark registered VLLM_SKIP_INIT_MEMORY_CHECK may now be the last entry
    '    "VLLM_SKIP_INIT_MEMORY_CHECK": lambda: bool(int(os.getenv("VLLM_SKIP_INIT_MEMORY_CHECK", "0"))),\n}',
]

MARKER = '"VLLM_SPARK_MAX_PROMPT_TOKENS":'


def main() -> int:
    if not TARGET.exists():
        print(f"[patch_envs_register_prompt_cap] target not found: {TARGET}")
        return 1

    src = TARGET.read_text()

    if MARKER in src:
        print(
            "[patch_envs_register_prompt_cap] already applied "
            "(marker present) — no-op"
        )
        return 0

    matched_anchor = None
    for cand in ANCHOR_CANDIDATES:
        if cand in src:
            matched_anchor = cand
            break

    if matched_anchor is None:
        print(
            "[patch_envs_register_prompt_cap] no anchor matched in "
            f"{TARGET} — vLLM version mismatch. Update ANCHOR_CANDIDATES. "
            "Verify dict tail with:\n"
            f"  tail -10 {TARGET}"
        )
        return 1

    replacement = matched_anchor.replace("\n}", "\n" + NEW_ENTRY + "}")
    patched = src.replace(matched_anchor, replacement, 1)
    TARGET.write_text(patched)
    print(
        f"[patch_envs_register_prompt_cap] applied to {TARGET}: "
        "VLLM_SPARK_MAX_PROMPT_TOKENS registered in environment_variables."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
