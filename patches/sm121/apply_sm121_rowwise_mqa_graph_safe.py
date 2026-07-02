#!/usr/bin/env python3
"""Make `_fp8_paged_mqa_logits_rowwise_kernel` CUDA-graph-safe (de-constexpr).

Without this patch, the first decode step that hits a *novel* context-length
shape (typically >256K tokens on a 512K `--max-model-len` deployment) can
JIT-compile a brand-new Triton specialization of the rowwise paged-MQA logits
kernel while a FULL_DECODE_ONLY CUDA graph capture is in flight:

    File ".../ops/sm12x_mqa.py", line 464, in fp8_paged_mqa_logits_rowwise_triton
      _fp8_paged_mqa_logits_rowwise_kernel[grid](
    ...
    File ".../triton/compiler/compiler.py", line 468, in _init_handles
      ... driver.active.utils.load_binary(
    RuntimeError: Triton Error [CUDA]: operation not permitted
    -> EngineCore fatal -> vllm.v1.engine.exceptions.EngineDeadError

Root cause: `num_rows`, `logits_width` and all 17 stride parameters are
`tl.constexpr`, and `logits_width` follows the batch's max_seq_len with no
bucketing (`_decode_logits_width` returns `min(max_model_len, max_seq_len)`)
-> one cubin per novel shape, loaded lazily at first launch; a first launch
inside a capture issues cuModuleLoad, which is illegal mid-capture
(CUDA_ERROR_NOT_PERMITTED). Same failure class as the direct-topk kernel
hardened in 72261a7 - this fallback variant was not covered, and small-batch
decode *always* takes it because `logits_bytes` (<= ~16 MB) never crosses
`sparse_indexer_max_logits_bytes()` (256 MB default on SM12x).

The patch moves those 19 parameters to runtime arguments. Model constants
(`next_n`, `num_heads`, `head_dim`, `block_size`) and tile sizes
(`BLOCK_N/D/H`) stay `tl.constexpr`. Triton's light runtime-int
specialization (==1 / %16==0) bounds the variant space to a handful, all
loaded during startup warmup - nothing left to load mid-capture. Decode
stays fully CUDA-graph captured: no PIECEWISE, no eager fallback.

Validated on dual DGX Spark GB10 (SM 12.1) TP=2, DeepSeek-V4-Flash,
max-model-len 524288, FULL_DECODE_ONLY + MTP=1: decode at 250,292 and
462,529 prompt tokens clean, multi-hour multi-concurrency stress with zero
errors, prefill/decode throughput unchanged.
Full analysis: docs/deepseek-v4-sm121-rowwise-mqa-cudagraph-fix.md

Usage (Dockerfile, after vLLM source clone):
  COPY patches/sm121/apply_sm121_rowwise_mqa_graph_safe.py /tmp/
  RUN python3 /tmp/apply_sm121_rowwise_mqa_graph_safe.py \
      /workspace/vllm-src/vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py

Runtime alternative (already-built image, inside the container):
  python3 apply_sm121_rowwise_mqa_graph_safe.py \
      /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py
"""
import re
import sys

F = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/workspace/vllm-src/vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py"
)

with open(F) as f:
    src = f.read()

m = re.search(r"def _fp8_paged_mqa_logits_rowwise_kernel\((.*?)\):\n", src, re.S)
if not m:
    sys.exit(f"{F}: rowwise kernel signature not found - wrong file or layout changed")

sig = m.group(1)
new_sig, n = re.subn(r"\b(num_rows|logits_width|stride_\w+): tl\.constexpr", r"\1", sig)
if n == 0:
    print(f"{F}: already applied (runtime-variant params carry no tl.constexpr)")
    sys.exit(0)
if n != 19:
    sys.exit(f"{F}: expected 19 replacements, got {n} - layout changed, refusing to half-apply")

with open(F, "w") as f:
    f.write(src[: m.start(1)] + new_sig + src[m.end(1) :])
print(f"{F}: patched - 19 params de-constexpr'd, rowwise kernel is now CUDA-graph-safe")
