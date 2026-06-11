#!/usr/bin/env python3
"""Fix Korean/non-ASCII tokenizer fallback for Step-3.7-Flash checkpoints.

The checkpoint's `tokenizer_config.json` declares
`"tokenizer_class": "LlamaTokenizerFast"`. Under transformers 5.10.2 (this
image's version), `AutoTokenizer.from_pretrained()` detects an "incorrect
regex pattern" (same family as the
mistralai/Mistral-Small-3.1-24B-Instruct-2503 tokenizer issue, HF discussion
#84) and silently falls back to the **slow** `LlamaTokenizer`
(SentencePiece-based). The slow tokenizer:

1. Drops non-ASCII (e.g. Korean) text during encoding -- `/tokenize` on
   "한국의 수도는 어디?" returned only `[0, 33]` (BOS + trailing "?").
2. Doesn't reverse GPT2's byte-to-unicode mapping during decoding, leaking
   raw `Ġ`/`Ċ` byte-level BPE markers into generated text.

Fix: change `tokenizer_class` to `PreTrainedTokenizerFast`, which makes
`AutoTokenizer` resolve to the Rust fast tokenizer instead. The underlying
`tokenizer.json` is correct and is not modified.

Unlike the other scripts in this directory, this is not a vLLM source patch
applied during `docker buildx build` -- it edits the model checkpoint's
`tokenizer_config.json` on disk. Run it once per model directory, on every
node that holds a copy of the weights, then restart the vLLM containers
(worker then head).

Usage:
    python3 patch_step3p7_tokenizer_class.py /path/to/Step-3.7-Flash-FP8/tokenizer_config.json

Verified on Step-3.7-Flash-FP8 (2026-06-11). Apply the same fix to the NVFP4
checkpoint if/when it's deployed, since it likely ships the same
tokenizer_config.json.
"""

import shutil
import sys

OLD = '"tokenizer_class": "LlamaTokenizerFast"'
NEW = '"tokenizer_class": "PreTrainedTokenizerFast"'


def apply(path: str) -> None:
    with open(path) as f:
        content = f.read()

    if NEW in content:
        print(f"{path}: already patched -- no change needed.")
        return

    assert OLD in content, f"{path}: anchor not found:\n{OLD}"

    backup_path = path + ".bak-tokenizer-class"
    shutil.copyfile(path, backup_path)

    content = content.replace(OLD, NEW, 1)
    with open(path, "w") as f:
        f.write(content)

    print(f"{path}: tokenizer_class -> PreTrainedTokenizerFast (backup: {backup_path}).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <path-to-tokenizer_config.json>")
    apply(sys.argv[1])
