"""
Patch vllm/compilation/backends.py to detect torch.fx.passes.split_module.split_module()
tuple_return support via runtime signature inspection instead of a torch-version string check.

Motivation
----------
vLLM v0.21.0 keys the `tuple_return=True` kwarg on `is_torch_equal_or_newer("2.12.0.dev")`,
but NGC PyTorch 26.04 ships a `2.12.0a0` alpha snapshot whose `split_module` does NOT
yet have `tuple_return`. The version check returns True, the kwarg is passed, and
PyTorch raises `TypeError: split_module() got an unexpected keyword argument 'tuple_return'`.

This patch replaces the static version check with a dynamic signature probe so the kwarg
is only passed when the installed PyTorch's `split_module` actually accepts it.

Idempotent: safe to run multiple times.
"""
import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/compilation/backends.py"
)

OLD = '''        has_tuple_return = is_torch_equal_or_newer("2.12.0.dev")
        tuple_return_kwarg = {"tuple_return": True} if has_tuple_return else {}'''

NEW = '''        import inspect as _inspect_split
        has_tuple_return = (
            "tuple_return"
            in _inspect_split.signature(
                torch.fx.passes.split_module.split_module
            ).parameters
        )
        tuple_return_kwarg = {"tuple_return": True} if has_tuple_return else {}'''


def main() -> int:
    if not TARGET.exists():
        print(f"[split_module_compat] FATAL: {TARGET} not found", file=sys.stderr)
        return 1

    src = TARGET.read_text()
    if NEW.split("\n")[1].strip() in src:
        print("[split_module_compat] OK — already patched (dynamic signature probe present)")
        return 0
    if OLD.split("\n")[0].strip() not in src:
        print(
            "[split_module_compat] WARN — version-check line not found; vLLM source may have changed."
        )
        return 0

    patched = src.replace(OLD, NEW, 1)
    if patched == src:
        print("[split_module_compat] WARN — replace produced no change", file=sys.stderr)
        return 1

    TARGET.write_text(patched)
    print(f"[split_module_compat] OK — patched {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
