"""
Defensive patch: replace direct torch.float8_e8m0fnu attribute access with
getattr(torch, "float8_e8m0fnu", None) so vLLM imports do not crash on PyTorch
builds where the dtype is missing or moved.

PR #40852 review feedback flagged direct attribute access in:
  - vllm/model_executor/layers/quantization/fp8.py
  - vllm/model_executor/layers/quantization/utils/fp8_utils.py
  - vllm/model_executor/models/deepseek_v4.py

NGC 26.03 (PyTorch 2.11) has the dtype, so this patch is a defensive no-op
on this image. It still runs to insulate against upstream churn.

Strategy: textual rewrite. Idempotent — if the file already uses getattr or
the anchor is missing, it's a silent skip.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DIST = Path("/usr/local/lib/python3.12/dist-packages")
TARGETS = [
    DIST / "vllm/model_executor/layers/quantization/fp8.py",
    DIST / "vllm/model_executor/layers/quantization/utils/fp8_utils.py",
    DIST / "vllm/model_executor/models/deepseek_v4.py",
]

# Replace `torch.float8_e8m0fnu` with `getattr(torch, "float8_e8m0fnu", None)`
# but skip lines that already wrap with getattr or reassign in a guard.
PAT = re.compile(r"\btorch\.float8_e8m0fnu\b")
SAFE = 'getattr(torch, "float8_e8m0fnu", None)'


def patch_file(path: Path) -> bool:
    if not path.exists():
        print(f"[safe_e8m0] skip (missing): {path}")
        return False
    src = path.read_text()
    if "float8_e8m0fnu" not in src:
        print(f"[safe_e8m0] no anchor: {path}")
        return False
    new_lines: list[str] = []
    changed = False
    for line in src.splitlines(keepends=True):
        if "getattr(torch" in line and "float8_e8m0fnu" in line:
            new_lines.append(line)
            continue
        if PAT.search(line):
            new_lines.append(PAT.sub(SAFE, line))
            changed = True
        else:
            new_lines.append(line)
    if not changed:
        print(f"[safe_e8m0] already safe: {path}")
        return False
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        backup.write_text(src)
    path.write_text("".join(new_lines))
    print(f"[safe_e8m0] patched: {path}")
    return True


def main() -> int:
    any_changed = False
    for t in TARGETS:
        any_changed = patch_file(t) or any_changed
    return 0 if any_changed or all(not p.exists() for p in TARGETS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
