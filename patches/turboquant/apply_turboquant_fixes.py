#!/usr/bin/env python3
"""
Apply still-open upstream TurboQuant fixes to vLLM.

These patches are cherry-picked from PRs that remain open as of
vLLM v0.21.0 (commit ad7125a4, 2026-05-15) but are needed for
DGX Spark (GB10, SM121, Qwen3.5 hybrid) workloads.

Applied PRs (still open against vLLM main):
  1. PR #40074 — Triton decode index OOB fix
  2. PR #39988 — BF16 FP8 cast fix

Removed (merged upstream — no patch needed):
  - PR #40060 — TURBOQUANT backend selection fix     (merged 2026-04-17)
  - PR #40092 — FA3/FA4 for prefill paths           (merged 2026-04-23)
  - PR #39931 — Hybrid model + uniform quantization (merged 2026-05-05)

Usage (in Dockerfile, after vLLM install):
  COPY patches/turboquant/apply_turboquant_fixes.py /tmp/
  RUN python3 /tmp/apply_turboquant_fixes.py
"""

import glob
import os
import sys

SITE = "/usr/local/lib/python3.12/dist-packages"
applied = 0
failed = 0


def find_file(relpath):
    full = os.path.join(SITE, relpath)
    if os.path.exists(full):
        return full
    for p in glob.glob(f"/usr/local/lib/python3.*/dist-packages/{relpath}"):
        return p
    return None


def patch_file(path, edits, pr_num, description):
    global applied, failed
    fpath = find_file(path)
    if not fpath:
        print(f"  SKIP (not found): {path}")
        failed += 1
        return False

    with open(fpath) as f:
        content = f.read()
    original = content

    for old, new in edits:
        if old not in content:
            print(f"  WARN PR#{pr_num}: pattern not found in {path}")
            print(f"    >>> {old[:80]}...")
            failed += 1
            return False
        content = content.replace(old, new, 1)

    if content == original:
        print(f"  SKIP (no changes): {path}")
        return True

    with open(fpath, "w") as f:
        f.write(content)
    print(f"  OK PR#{pr_num}: {path} — {description}")
    applied += 1
    return True


# =====================================================================
# PR #40074 — Triton decode index OOB fix
# =====================================================================
print("\n[PR #40074] Triton decode index OOB fix...")

patch_file(
    "vllm/v1/attention/ops/triton_turboquant_decode.py",
    [
        (
            "        block_nums = tl.load(\n"
            "            Block_table_ptr + bt_base + page_idx,\n"
            "            mask=kv_mask,",
            "        # Clamp OOB lanes to index 0 before pointer arithmetic so\n"
            "        # Triton's bounds checker does not fire on masked-out lanes.\n"
            "        safe_page_idx = tl.where(kv_mask, page_idx, 0)\n"
            "        block_nums = tl.load(\n"
            "            Block_table_ptr + bt_base + safe_page_idx,\n"
            "            mask=kv_mask,",
        ),
    ],
    40074,
    "clamp OOB page indices",
)


# =====================================================================
# PR #39988 — BF16 FP8 cast fix
# =====================================================================
print("\n[PR #39988] BF16 FP8 cast fix...")

patch_file(
    "vllm/v1/attention/ops/triton_turboquant_store.py",
    [
        (
            "k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0)",
            "k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)",
        ),
    ],
    39988,
    "BF16→FP32 before FP8 cast",
)


# =====================================================================
# Summary
# =====================================================================
print(f"\n{'='*60}")
print(f"TurboQuant fixes: {applied} applied, {failed} failed")
if failed > 0:
    print("WARNING: Some patches could not be applied.")
    print("This may be due to upstream code changes.")
    print("Review the warnings above.")
    sys.exit(1)
else:
    print("All patches applied successfully!")
    print("\nUsage: vllm serve <model> --kv-cache-dtype turboquant_k8v4")
