#!/usr/bin/env python3
"""Adds Step3p7ForConditionalGeneration to vLLM model registry."""
path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py"
anchor = '    "Step3VLForConditionalGeneration": ("step3_vl", "Step3VLForConditionalGeneration"),'
entry  = '    "Step3p7ForConditionalGeneration": ("step3p7", "Step3p7ForConditionalGeneration"),'

with open(path) as f:
    content = f.read()

if entry in content:
    print("Step3p7ForConditionalGeneration already in registry — no change needed.")
else:
    assert anchor in content, f"Anchor not found:\n{anchor}"
    content = content.replace(anchor, anchor + "\n" + entry)
    with open(path, "w") as f:
        f.write(content)
    print("Step3p7ForConditionalGeneration added to vLLM registry.")
