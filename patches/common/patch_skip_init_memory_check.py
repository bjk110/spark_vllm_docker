#!/usr/bin/env python3
"""
Patch vllm/v1/worker/utils.py to allow bypassing the startup free-memory
pre-allocation check via VLLM_SKIP_INIT_MEMORY_CHECK=1.

Why this patch exists
---------------------
On GB10 (DGX Spark, sm_121, UMA) the nvidia driver + kernel page allocator
accumulate host-RAM reservations over the first ~5-15 minutes of uptime even
with no user-space workload. Measured pattern on our cluster (2026-05-23):

  uptime  2 min: free 117 GiB, available 117 GiB → vLLM init passes
  uptime 14 min: free  30 GiB, available  42 GiB → vLLM init rejects

vLLM's `request_memory()` (vllm/v1/worker/utils.py) compares
`MemorySnapshot.free_memory` (which on UMA is `psutil.virtual_memory().available`)
against `total_memory * gpu_memory_utilization` and raises ValueError when
short. On GB10 this rejects well-sized configs that would actually fit
(TP=2 puts only ~75 GiB of weights+KV per node, well under the 0.85 * 121 GiB
= 103 GiB budget the check demands).

This patch adds an env-var escape hatch — set VLLM_SKIP_INIT_MEMORY_CHECK=1
to log a warning and proceed; any genuine OOM still surfaces during the
weight-load step at its natural failure site.

Operational note
----------------
Without this patch the only workaround is "reboot the node and start vLLM
within 5 minutes." The patch removes that timing constraint and makes GB10
deployments reproducible from any host-RAM state.

Patch is idempotent — re-running prints a notice and exits 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py"
)

ANCHOR_OLD = '''def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    """
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    if init_snapshot.free_memory < requested_memory:
        raise ValueError('''

ANCHOR_NEW = '''def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.

    vllm-spark patch: GB10 UMA accumulates host-RAM reservations after a few
    minutes of uptime. Set VLLM_SKIP_INIT_MEMORY_CHECK=1 to bypass the
    pre-check; real OOM still surfaces during weight load.
    """
    import os as _os
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    if _os.environ.get("VLLM_SKIP_INIT_MEMORY_CHECK") == "1":
        from vllm.logger import init_logger as _init_logger
        _init_logger(__name__).warning(
            "VLLM_SKIP_INIT_MEMORY_CHECK=1 — skipping startup free-memory "
            "check (free_memory=%s, requested=%s on %s). Any genuine OOM "
            "will surface at the weight-load step.",
            init_snapshot.free_memory,
            requested_memory,
            init_snapshot.device_,
        )
        return requested_memory

    if init_snapshot.free_memory < requested_memory:
        raise ValueError('''

MARKER = "VLLM_SKIP_INIT_MEMORY_CHECK"


def main() -> int:
    if not TARGET.exists():
        print(f"[patch_skip_init_memory_check] target not found: {TARGET}")
        return 1

    src = TARGET.read_text()

    if MARKER in src:
        print(
            "[patch_skip_init_memory_check] already applied "
            f"(marker '{MARKER}' present) — no-op"
        )
        return 0

    if ANCHOR_OLD not in src:
        print(
            "[patch_skip_init_memory_check] anchor not found in "
            f"{TARGET} — vLLM version mismatch. Function `request_memory` "
            "may have changed shape; verify by running:\n"
            f"  grep -A4 'def request_memory' {TARGET}"
        )
        return 1

    patched = src.replace(ANCHOR_OLD, ANCHOR_NEW, 1)
    TARGET.write_text(patched)
    print(
        f"[patch_skip_init_memory_check] applied to {TARGET}: "
        "VLLM_SKIP_INIT_MEMORY_CHECK=1 env-var bypass available."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
