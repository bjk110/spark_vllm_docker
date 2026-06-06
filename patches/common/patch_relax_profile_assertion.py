#!/usr/bin/env python3
"""
Patch vllm/v1/worker/gpu_worker.py to relax the post-profiling free-memory
assertion when running under VLLM_SKIP_INIT_MEMORY_CHECK=1.

Why this patch exists
---------------------
After patch_skip_init_memory_check.py let init pass on GB10 UMA, a second
guard fires later in `determine_available_memory()`:

  assert self.init_snapshot.free_memory >= free_gpu_memory

vLLM assumes profiling can only *consume* memory (so reported free should
go down or stay flat). On GB10 the UMA free metric is
`psutil.virtual_memory().available`, and during the safetensors load the
Linux kernel may reclaim page cache between the init snapshot and the
post-profile snapshot, which legitimately raises that value:

  AssertionError: Error in memory profiling. Initial free memory 31.21 GiB,
    current free memory 32.66 GiB. ... other processes sharing the same
    container release GPU memory while vLLM is profiling ...

The diagnostic message is misleading on UMA — no other process touched
the GPU; the host RAM accounting just shifted. The downstream code path
does not depend on the assertion (`free_gpu_memory` is computed but
`self.requested_memory` and `profile_result.non_kv_cache_memory` are used
for the KV-cache budget), so relaxing it to a warning is safe.

The patch reuses the existing VLLM_SKIP_INIT_MEMORY_CHECK env-var as a
single GB10/UMA escape hatch — flipping the var on relaxes both the
pre-init and post-profile checks.

Patch is idempotent — re-running prints a notice and exits 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"
)

ANCHOR_OLD = '''        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory >= free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {format_gib(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )'''

ANCHOR_NEW = '''        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        # vllm-spark patch: VLLM_SKIP_INIT_MEMORY_CHECK=1 also relaxes this
        # post-profile assertion. On GB10 UMA the free metric is
        # psutil.virtual_memory().available; the Linux kernel can reclaim
        # page cache during profiling, raising it above the init snapshot.
        # The diagnostic about "other processes" is wrong on UMA — host RAM
        # accounting just shifted. Downstream KV-cache budget does not
        # depend on this assertion.
        import os as _vllm_spark_os
        if (
            _vllm_spark_os.environ.get("VLLM_SKIP_INIT_MEMORY_CHECK") == "1"
            and self.init_snapshot.free_memory < free_gpu_memory
        ):
            from vllm.logger import init_logger as _vllm_spark_init_logger
            _vllm_spark_init_logger(__name__).warning(
                "VLLM_SKIP_INIT_MEMORY_CHECK=1 — relaxing post-profile "
                "free-memory assertion (init=%s GiB, current=%s GiB). On "
                "UMA platforms the OS may reclaim page cache during "
                "profiling, legitimately raising the reported free metric.",
                format_gib(self.init_snapshot.free_memory),
                format_gib(free_gpu_memory),
            )
        else:
            assert self.init_snapshot.free_memory >= free_gpu_memory, (
                "Error in memory profiling. "
                f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "
                f"current free memory {format_gib(free_gpu_memory)} GiB. "
                "This happens when other processes sharing the same container "
                "release GPU memory while vLLM is profiling during initialization. "
                "To fix this, ensure consistent GPU memory allocation or "
                "isolate vLLM in its own container."
            )'''

MARKER = "vllm-spark patch: VLLM_SKIP_INIT_MEMORY_CHECK=1 also relaxes"


def main() -> int:
    if not TARGET.exists():
        print(f"[patch_relax_profile_assertion] target not found: {TARGET}")
        return 1

    src = TARGET.read_text()

    if MARKER in src:
        print(
            "[patch_relax_profile_assertion] already applied "
            "(marker present) — no-op"
        )
        return 0

    if ANCHOR_OLD not in src:
        print(
            "[patch_relax_profile_assertion] anchor not found in "
            f"{TARGET} — vLLM version mismatch. The assertion in "
            "determine_available_memory() may have changed shape. Verify with:\n"
            "  grep -n 'init_snapshot.free_memory >= free_gpu_memory' "
            f"{TARGET}"
        )
        return 1

    patched = src.replace(ANCHOR_OLD, ANCHOR_NEW, 1)
    TARGET.write_text(patched)
    print(
        f"[patch_relax_profile_assertion] applied to {TARGET}: "
        "post-profile assertion now respects VLLM_SKIP_INIT_MEMORY_CHECK=1."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
