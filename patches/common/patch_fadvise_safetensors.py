#!/usr/bin/env python3
"""
Patch vllm/model_executor/model_loader/weight_utils.py to drop the OS
page cache for each safetensors file right after it is fully read.

Why this patch exists
---------------------
On GB10 (DGX Spark, sm_121, UMA) the host RAM and GPU memory are unified.
Linux populates the OS page cache while reading safetensors files, then
keeps those pages around long after the weights are on the GPU. Those
pages reduce `psutil.virtual_memory().available`, which (per merged PR
vllm-project/vllm#35356) is what vLLM uses as `MemorySnapshot.free_memory`
on UMA platforms. The result is a slow, monotonic drop in reported free
memory over the lifetime of the container — the exact pattern documented
in docs/dsv4-flash-tp2.md §11.6 (uptime 2 min: free 117 GiB → uptime 14
min: free 30 GiB).

Port of the upstream patch from PR vllm-project/vllm#35929 (still open
at time of writing). After each safetensors file is read, call
`posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` so the kernel evicts the
file's pages from the cache. This is a best-effort hint — the kernel is
allowed to ignore it — but on Linux 6.x it reliably drops clean cached
pages on the next reclaim pass.

Anchor used: the end of the outer `for st_file in tqdm(...)` loop body
in safetensors_weights_iterator. Placing the call there (rather than
inside one specific branch) covers all three load strategies (eager,
torchao, default) with a single insertion.

Operational note
----------------
This patch reduces — but does not eliminate — the host RAM accumulation
on GB10. nvidia driver / kernel buffers still grow over uptime. The
`VLLM_SKIP_INIT_MEMORY_CHECK` escape hatch (patch_skip_init_memory_check.py)
remains the load-bearing fix for the init pre-check; this patch attacks
the same underlying problem from the supply side.

Patch is idempotent — re-running prints a notice and exits 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py"
)

ANCHOR_OLD = '''        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if _should_skip_safetensors_weight(
                        name, local_expert_ids, weight_name_filter
                    ):
                        continue
                    param = f.get_tensor(name)
                    yield name, param


def multi_thread_safetensors_weights_iterator('''

ANCHOR_NEW = '''        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if _should_skip_safetensors_weight(
                        name, local_expert_ids, weight_name_filter
                    ):
                        continue
                    param = f.get_tensor(name)
                    yield name, param

        # vllm-spark patch (port of PR vllm-project/vllm#35929): drop the
        # OS page cache for the safetensors file just read. On GB10 UMA
        # the populated page cache would otherwise deflate
        # psutil.virtual_memory().available, which on integrated GPUs is
        # vLLM's free-memory metric (see PR #35356). Placed outside the
        # branch-specific with-block so all three load strategies (eager,
        # torchao, default) get the same treatment.
        if hasattr(os, "posix_fadvise"):
            try:
                fd = os.open(st_file, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                pass


def multi_thread_safetensors_weights_iterator('''

MARKER = "vllm-spark patch (port of PR vllm-project/vllm#35929)"


def main() -> int:
    if not TARGET.exists():
        print(f"[patch_fadvise_safetensors] target not found: {TARGET}")
        return 1

    src = TARGET.read_text()

    if MARKER in src:
        print(
            "[patch_fadvise_safetensors] already applied "
            "(marker present) — no-op"
        )
        return 0

    if ANCHOR_OLD not in src:
        print(
            "[patch_fadvise_safetensors] anchor not found in "
            f"{TARGET} — vLLM version mismatch. The default branch of "
            "safetensors_weights_iterator may have moved. Verify with:\n"
            "  grep -n 'def safetensors_weights_iterator' "
            f"{TARGET}\n"
            "and check the tail of that function."
        )
        return 1

    patched = src.replace(ANCHOR_OLD, ANCHOR_NEW, 1)
    TARGET.write_text(patched)
    print(
        f"[patch_fadvise_safetensors] applied to {TARGET}: "
        "POSIX_FADV_DONTNEED hint installed after each safetensors read."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
