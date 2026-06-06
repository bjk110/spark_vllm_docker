#!/usr/bin/env python3
"""
Patch vLLM for PyTorch 2.11 (NGC 26.03) compatibility.

Fix: Remove hoist=True from register_opaque_type() — kwarg removed in PyTorch 2.11.

Note: The __fx_repr__ set→dict fix is no longer needed (fixed upstream).
Note: The class was renamed ModuleName → LayerName upstream (978a4462+).
"""
import glob
import sys

target = None
for pattern in [
    "/workspace/vllm-src/vllm/utils/torch_utils.py",
    "/tmp/vllm/vllm/utils/torch_utils.py",
    "/usr/local/lib/python3.*/dist-packages/vllm/utils/torch_utils.py",
]:
    for p in glob.glob(pattern):
        target = p
        break
    if target:
        break

if not target:
    print("FATAL: torch_utils.py not found")
    sys.exit(1)

with open(target) as f:
    code = f.read()

changes = 0

# Fix: hoist=True — removed in PyTorch 2.11's register_opaque_type()
if ", hoist=True" in code:
    code = code.replace(", hoist=True", "")
    changes += 1
    print("[fix] Removed hoist=True from register_opaque_type()")
elif "hoist=True" not in code:
    print("[fix] hoist=True already absent — OK")
else:
    print("FATAL: hoist=True in unexpected position")
    sys.exit(1)

if changes > 0:
    with open(target, "w") as f:
        f.write(code)
    print(f"PATCHED: {target} ({changes} changes)")
else:
    print(f"NO CHANGES needed: {target}")

# Verify
with open(target) as f:
    final = f.read()
assert "hoist=True" not in final, "FATAL: hoist=True still present"
print("VERIFIED: hoist=True patch applied correctly")
