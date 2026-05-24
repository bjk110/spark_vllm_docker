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

ANCHOR_OLD = '''    "VLLM_USE_SPINLOOP_EXT": lambda: bool(int(os.getenv("VLLM_USE_SPINLOOP_EXT", "0"))),
}'''

ANCHOR_NEW = '''    "VLLM_USE_SPINLOOP_EXT": lambda: bool(int(os.getenv("VLLM_USE_SPINLOOP_EXT", "0"))),
    # vllm-spark patch: see patches/patch_skip_init_memory_check.py —
    # bypasses request_memory() startup pre-check on GB10 UMA where host
    # RAM accumulation makes the static budget check fire spuriously.
    "VLLM_SKIP_INIT_MEMORY_CHECK": lambda: bool(int(os.getenv("VLLM_SKIP_INIT_MEMORY_CHECK", "0"))),
}'''

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

    if ANCHOR_OLD not in src:
        print(
            "[patch_envs_register_skip_memcheck] anchor not found in "
            f"{TARGET} — vLLM version mismatch. The dict tail layout may "
            "have changed. Verify with:\n"
            f"  grep -n VLLM_USE_SPINLOOP_EXT {TARGET}"
        )
        return 1

    patched = src.replace(ANCHOR_OLD, ANCHOR_NEW, 1)
    TARGET.write_text(patched)
    print(
        f"[patch_envs_register_skip_memcheck] applied to {TARGET}: "
        "VLLM_SKIP_INIT_MEMORY_CHECK registered in environment_variables."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
