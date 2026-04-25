"""
Hot-patch vllm/compilation/codegen.py to honor __fx_repr__() on opaque types
(specifically LayerName) instead of falling back to default repr().

Bug
---
After PR #38657 (codegen-based FX split) introduced 951dca80, the function
`_node_ref()` converts every non-Node argument via plain `repr(arg)`. That
embeds object addresses like `<vllm.utils.torch_utils.LayerName object at
0x...>` into the generated execution function source, which is then fed to
`exec(code, namespace)` and dies with `SyntaxError: invalid syntax`. The
opaque type already has a usable `__fx_repr__()` method that returns
`("LayerName('...')", {"LayerName": LayerName})`, but `_node_ref()` does
not consult it.

Trigger: torch.compile + any op taking a `LayerName` positional arg
(currently the GDN attention path for Qwen3.5 hybrid models) — the engine
crashes during `_initialize_kv_caches` → `determine_available_memory`.

Workaround prior to this patch: `--enforce-eager` (loses CUDAGraph and
torch.compile entirely).

Patch shape
-----------
1. `_node_ref()` checks for `__fx_repr__` and uses it. The (src, ns) tuple
   is unpacked; the namespace contribution accumulates into a module-level
   dict `_FX_REPR_NS` so the caller can pick it up.
2. `compile_execution_fn()` merges `_FX_REPR_NS` into the exec namespace,
   then clears the dict for the next call.

Both changes are minimal and idempotent: re-running this script on a file
already containing the markers leaves it untouched. Designed for runtime
hot-patch via `docker exec` (no image rebuild needed).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/compilation/codegen.py")

# --- Edit 1: insert module-level _FX_REPR_NS dict ---------------------------
OLD_HEADER = "import torch.fx\n"
NEW_HEADER = "import torch.fx\n\n# vllm-spark hot-patch (codegen __fx_repr__ support)\n_FX_REPR_NS: dict = {}\n"

# --- Edit 2: rewrite _node_ref to consult __fx_repr__ -----------------------
OLD_NODE_REF = (
    "def _node_ref(arg: Any) -> str:\n"
    "    \"\"\"Convert an FX node argument to a source code reference recursively.\"\"\"\n"
    "    if isinstance(arg, torch.fx.Node):\n"
    "        return arg.name\n"
    "    if isinstance(arg, list):\n"
    "        return f\"[{', '.join(_node_ref(x) for x in arg)}]\"\n"
    "    if isinstance(arg, tuple):\n"
    "        items = \", \".join(_node_ref(x) for x in arg)\n"
    "        return f\"({items},)\" if len(arg) == 1 else f\"({items})\"\n"
    "    if isinstance(arg, dict):\n"
    "        return (\n"
    "            \"{\"\n"
    "            + \", \".join(f\"{_node_ref(k)}: {_node_ref(v)}\" for k, v in arg.items())\n"
    "            + \"}\"\n"
    "        )\n"
    "    return repr(arg)\n"
)

NEW_NODE_REF = (
    "def _node_ref(arg: Any) -> str:\n"
    "    \"\"\"Convert an FX node argument to a source code reference recursively.\"\"\"\n"
    "    if isinstance(arg, torch.fx.Node):\n"
    "        return arg.name\n"
    "    if isinstance(arg, list):\n"
    "        return f\"[{', '.join(_node_ref(x) for x in arg)}]\"\n"
    "    if isinstance(arg, tuple):\n"
    "        items = \", \".join(_node_ref(x) for x in arg)\n"
    "        return f\"({items},)\" if len(arg) == 1 else f\"({items})\"\n"
    "    if isinstance(arg, dict):\n"
    "        return (\n"
    "            \"{\"\n"
    "            + \", \".join(f\"{_node_ref(k)}: {_node_ref(v)}\" for k, v in arg.items())\n"
    "            + \"}\"\n"
    "        )\n"
    "    # vllm-spark hot-patch: opaque types (e.g. LayerName) carry __fx_repr__\n"
    "    fx_repr = getattr(arg, \"__fx_repr__\", None)\n"
    "    if callable(fx_repr):\n"
    "        result = fx_repr()\n"
    "        if isinstance(result, tuple) and len(result) == 2:\n"
    "            src, ns = result\n"
    "            if isinstance(ns, dict):\n"
    "                _FX_REPR_NS.update(ns)\n"
    "            if isinstance(src, str):\n"
    "                return src\n"
    "        elif isinstance(result, str):\n"
    "            return result\n"
    "    return repr(arg)\n"
)

# --- Edit 3: compile_execution_fn merges _FX_REPR_NS into namespace ---------
OLD_EXEC = (
    "    namespace: dict[str, Any] = {}\n"
    "    exec(code, namespace)  # noqa: S102\n"
)

NEW_EXEC = (
    "    namespace: dict[str, Any] = {}\n"
    "    # vllm-spark hot-patch: merge namespace contributions from __fx_repr__\n"
    "    namespace.update(_FX_REPR_NS)\n"
    "    _FX_REPR_NS.clear()\n"
    "    exec(code, namespace)  # noqa: S102\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"[codegen_fx_repr] target not found: {TARGET}", file=sys.stderr)
        return 1
    src = TARGET.read_text()

    if "_FX_REPR_NS" in src:
        print("[codegen_fx_repr] already patched, skipping")
        return 0

    if OLD_HEADER not in src:
        print(f"[codegen_fx_repr] anchor 'import torch.fx' not found", file=sys.stderr)
        return 1
    if OLD_NODE_REF not in src:
        print("[codegen_fx_repr] anchor _node_ref body not found (drift?)", file=sys.stderr)
        return 1
    if OLD_EXEC not in src:
        print("[codegen_fx_repr] anchor 'exec(code, namespace)' not found", file=sys.stderr)
        return 1

    backup = TARGET.with_suffix(TARGET.suffix + ".fx_repr.orig")
    if not backup.exists():
        backup.write_text(src)
        print(f"[codegen_fx_repr] backup: {backup}")

    src = src.replace(OLD_HEADER, NEW_HEADER, 1)
    src = src.replace(OLD_NODE_REF, NEW_NODE_REF, 1)
    src = src.replace(OLD_EXEC, NEW_EXEC, 1)
    TARGET.write_text(src)
    print(f"[codegen_fx_repr] patched: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
