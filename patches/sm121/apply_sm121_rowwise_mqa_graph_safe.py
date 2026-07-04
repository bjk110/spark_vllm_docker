#!/usr/bin/env python3
"""Make `_fp8_paged_mqa_logits_rowwise_kernel` CUDA-graph-safe (de-constexpr).

Without this patch, the first decode step that hits a context-length shape
*novel to the engine process* can JIT-compile a brand-new Triton
specialization of the rowwise paged-MQA logits kernel while a
FULL_DECODE_ONLY CUDA graph capture is in flight:

    File ".../ops/sm12x_mqa.py", line 464, in fp8_paged_mqa_logits_rowwise_triton
      _fp8_paged_mqa_logits_rowwise_kernel[grid](
    ...
    File ".../triton/compiler/compiler.py", line 468, in _init_handles
      ... driver.active.utils.load_binary(
    RuntimeError: Triton Error [CUDA]: operation not permitted
    -> EngineCore fatal -> vllm.v1.engine.exceptions.EngineDeadError

In our production reproduction this fired at >256K prompt tokens on a 512K
`--max-model-len` deployment, but >256K is the *observed* territory, not an
intrinsic threshold: path selection is governed by logits bytes vs. the
SM12x sparse-indexer threshold (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, 256 MB
default), and small-batch decode stays below it at any context length, so
the rowwise fallback is always the decode path there.

Root cause: `num_rows`, `logits_width` and all 17 stride parameters are
`tl.constexpr`, and `logits_width` follows the batch's max_seq_len with no
bucketing (`_decode_logits_width` returns `min(max_model_len, max_seq_len)`)
-> one cubin per novel shape, loaded lazily at first launch; a first launch
inside a capture issues cuModuleLoad, which is illegal mid-capture
(CUDA_ERROR_NOT_PERMITTED). Same failure class as the direct-topk kernel
hardening already present in the pinned vLLM source tree (jasl `72261a7`,
the tree this image is built from) - the rowwise fallback variant was not
covered, and small-batch decode always takes it (see threshold note above).

The patch moves those 19 parameters to runtime arguments. Model constants
(`next_n`, `num_heads`, `head_dim`, `block_size`) and tile sizes
(`BLOCK_N/D/H`) stay `tl.constexpr`. Triton's light runtime-int
specialization (==1 / %16==0) bounds the variant space to a handful, all
loaded during startup warmup - nothing left to load mid-capture. Decode
stays fully CUDA-graph captured: no PIECEWISE, no eager fallback.

State detection (explicit allowlist of the 19 target parameters):
  - all 19 still `: tl.constexpr`          -> patch (exit 0)
  - all 19 already runtime arguments       -> no-op, "already applied" (exit 0)
  - anything else (missing / mixed / other
    annotation / signature not found)      -> refuse, exit nonzero

Replacement is atomic: patched content is built in memory, `ast.parse`d,
byte-compiled from a temp file in the target directory, given the original
file's mode, then swapped in with `os.replace()`. The target is never
opened for writing in place.

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
import ast
import os
import py_compile
import re
import stat
import sys
import tempfile

# The 19 parameters that legitimately vary at runtime (explicit allowlist -
# no broad stride_\w+ matching). Order follows the kernel signature.
RUNTIME_PARAMS = (
    "num_rows",
    "logits_width",
    "stride_qb",
    "stride_qn",
    "stride_qh",
    "stride_qd",
    "stride_kvb",
    "stride_kvs",
    "stride_kvd",
    "stride_sb",
    "stride_ss",
    "stride_wm",
    "stride_wh",
    "stride_clb",
    "stride_cln",
    "stride_btb",
    "stride_btk",
    "stride_lm",
    "stride_ln",
)

# Parameters that must remain tl.constexpr; their presence doubles as a
# layout check that we are looking at the expected kernel.
KEEP_CONSTEXPR_PARAMS = (
    "next_n",
    "num_heads",
    "head_dim",
    "block_size",
    "BLOCK_N",
    "BLOCK_D",
    "BLOCK_H",
)

KERNEL_RE = re.compile(
    r"def _fp8_paged_mqa_logits_rowwise_kernel\((.*?)\):\n", re.S
)


def fail(msg):
    print(f"{TARGET}: {msg}", file=sys.stderr)
    sys.exit(1)


def classify(sig):
    """Return (constexpr, runtime, missing, odd) subsets of RUNTIME_PARAMS."""
    constexpr, runtime, missing, odd = [], [], [], []
    for name in RUNTIME_PARAMS:
        if re.search(rf"\b{name}\s*:\s*tl\.constexpr\b", sig):
            constexpr.append(name)
        elif re.search(rf"\b{name}\s*:", sig):
            odd.append(name)  # present but with an unexpected annotation
        elif re.search(rf"\b{name}\b", sig):
            runtime.append(name)
        else:
            missing.append(name)
    return constexpr, runtime, missing, odd


TARGET = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/workspace/vllm-src/vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py"
)

if not os.path.isfile(TARGET):
    fail("target file does not exist - wrong path or image layout changed")

with open(TARGET, encoding="utf-8") as f:
    src = f.read()

m = KERNEL_RE.search(src)
if not m:
    fail("rowwise kernel signature not found - wrong file or layout changed")
if KERNEL_RE.search(src, m.end()):
    fail("multiple rowwise kernel definitions found - refusing to guess")

sig = m.group(1)

for name in KEEP_CONSTEXPR_PARAMS:
    if not re.search(rf"\b{name}\s*:\s*tl\.constexpr\b", sig):
        fail(
            f"expected constexpr parameter `{name}` not found in signature - "
            "layout changed, refusing to patch"
        )

constexpr, runtime, missing, odd = classify(sig)

if missing or odd:
    fail(
        "signature does not match the expected layout "
        f"(missing: {missing or 'none'}, unexpected annotation: {odd or 'none'}) - "
        "wrong vLLM version or layout changed, refusing to patch"
    )

if len(runtime) == len(RUNTIME_PARAMS):
    print(
        f"{TARGET}: already applied - all {len(RUNTIME_PARAMS)} target "
        "parameters are runtime arguments, nothing to do"
    )
    sys.exit(0)

if len(constexpr) != len(RUNTIME_PARAMS):
    fail(
        f"partial/mixed state: {len(constexpr)} constexpr + {len(runtime)} "
        f"runtime of {len(RUNTIME_PARAMS)} expected (runtime: {runtime}) - "
        "refusing to half-apply"
    )

# All 19 are still constexpr -> patch, one explicit substitution per name.
new_sig = sig
for name in RUNTIME_PARAMS:
    new_sig, k = re.subn(rf"\b{name}\s*:\s*tl\.constexpr\b", name, new_sig)
    if k != 1:
        fail(f"expected exactly 1 substitution for `{name}`, got {k}")

new_src = src[: m.start(1)] + new_sig + src[m.end(1):]

# Post-state validation: the patched signature must classify as fully applied.
m2 = KERNEL_RE.search(new_src)
if not m2:
    fail("internal error: kernel signature lost after substitution")
constexpr2, runtime2, missing2, odd2 = classify(m2.group(1))
if constexpr2 or missing2 or odd2 or len(runtime2) != len(RUNTIME_PARAMS):
    fail("internal error: patched signature failed post-state validation")

try:
    ast.parse(new_src, filename=TARGET)
except SyntaxError as e:
    fail(f"patched content does not parse: {e}")

# Atomic replacement: temp file in the same directory, byte-compile check,
# original mode preserved, then os.replace().
orig_mode = stat.S_IMODE(os.stat(TARGET).st_mode)
dir_name = os.path.dirname(os.path.abspath(TARGET))
fd, tmp_path = tempfile.mkstemp(
    prefix=".sm12x_mqa_patch_", suffix=".py", dir=dir_name
)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(new_src)
    cfile = tmp_path + "c"
    try:
        py_compile.compile(tmp_path, cfile=cfile, doraise=True)
    finally:
        if os.path.exists(cfile):
            os.remove(cfile)
    os.chmod(tmp_path, orig_mode)
    os.replace(tmp_path, TARGET)
except BaseException:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    raise

print(
    f"{TARGET}: patched - {len(RUNTIME_PARAMS)} params de-constexpr'd, "
    "rowwise kernel is now CUDA-graph-safe"
)
