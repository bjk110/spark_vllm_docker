#!/usr/bin/env python3
"""
Patch vllm/envs.py to register VLLM_SKIP_INIT_MEMORY_CHECK in the
`environment_variables` dict.

Why this patch exists
---------------------
patch_skip_init_memory_check.py adds the runtime hook for the env-var,
but vllm/envs.py iterates os.environ on startup and warns about any
VLLM_-prefixed variable that is not registered in `environment_variables`.
The warning is harmless but noisy (one line per worker process, per
container start):

  WARNING [envs.py:1942] Unknown vLLM environment variable detected:
  VLLM_SKIP_INIT_MEMORY_CHECK

This patch registers the var so vLLM's own validate_environ recognises it.

Anchor used: the last dict entry (VLLM_USE_SPINLOOP_EXT) immediately
followed by the dict's closing `}`. The new entry is inserted before
the closing brace.

Patch is idempotent — re-running prints a notice and exits 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/envs.py")

# New entry to insert (idempotent — MARKER short-circuits re-runs).
NEW_ENTRY = '''    # vllm-spark patch: see patches/patch_skip_init_memory_check.py —
    # bypasses request_memory() startup pre-check on GB10 UMA where host
    # RAM accumulation makes the static budget check fire spuriously.
    "VLLM_SKIP_INIT_MEMORY_CHECK": lambda: bool(int(os.getenv("VLLM_SKIP_INIT_MEMORY_CHECK", "0"))),
'''

# Multiple anchor candidates for robustness across jasl/vllm bumps.
# Each anchor is the LAST dict entry (its line + the dict's closing `}`).
# When jasl adds new entries after the previous last, this list grows.
ANCHOR_CANDIDATES = [
    # jasl HEAD 2026-06-01 (0440ee5c22) — new last entry
    '    "VLLM_NIC_SELECTION_VARS": lambda: os.getenv("VLLM_NIC_SELECTION_VARS", ""),\n}',
    # jasl edc82b614f51 (pre-2026-06-01) — previous last entry
    '    "VLLM_USE_SPINLOOP_EXT": lambda: bool(int(os.getenv("VLLM_USE_SPINLOOP_EXT", "0"))),\n}',
]

MARKER = '"VLLM_SKIP_INIT_MEMORY_CHECK":'


def main() -> int:
    if not TARGET.exists():
        print(f"[patch_envs_register_skip_memcheck] target not found: {TARGET}")
        return 1

    src = TARGET.read_text()

    if MARKER in src:
        print(
            "[patch_envs_register_skip_memcheck] already applied "
            f"(marker present) — no-op"
        )
        return 0

    matched_anchor = None
    for cand in ANCHOR_CANDIDATES:
        if cand in src:
            matched_anchor = cand
            break

    if matched_anchor is None:
        print(
            "[patch_envs_register_skip_memcheck] no anchor matched in "
            f"{TARGET} — vLLM version mismatch. Update ANCHOR_CANDIDATES. "
            "Verify dict tail with:\n"
            f"  tail -10 {TARGET}"
        )
        return 1

    # Build replacement: insert NEW_ENTRY before the closing `}` of the anchor.
    replacement = matched_anchor.replace("\n}", "\n" + NEW_ENTRY + "}")
    patched = src.replace(matched_anchor, replacement, 1)
    TARGET.write_text(patched)
    print(
        f"[patch_envs_register_skip_memcheck] applied to {TARGET}: "
        "VLLM_SKIP_INIT_MEMORY_CHECK registered in environment_variables."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
