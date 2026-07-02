# EngineDeadError at long context (>256K) on DGX Spark (GB10/SM121): Triton JIT load during CUDA graph capture in `_fp8_paged_mqa_logits_rowwise_kernel` - root cause + validated fix

**TL;DR** - On the current production image (`ghcr.io/bjk110/vllm-spark:dsv4-sm121-indexer-production`, digest `sha256:ade810fd…`, same manifest as `v023-dsv4-72261a7-sm121-deepgemm-indexer-prod-fa83457d`), serving DeepSeek-V4-Flash in dual-node TP=2 with `--max-model-len 524288` dies with `EngineDeadError` the first time a decode step encounters a *novel* context-length shape (in practice: agent workloads crossing ~256K tokens), even though stress tests up to 256K pass. The root cause is a Triton kernel in the SM12x fallback path whose **sizes and strides are all `tl.constexpr`**, forcing a new cubin per novel shape; when the first launch of a new specialization happens during CUDA graph capture, `cuModuleLoad` is issued mid-capture → `CUDA_ERROR_NOT_PERMITTED` → dead engine. A 19-line signature-only patch (de-constexprify the runtime-variant parameters) removes the crash class entirely while keeping decode fully CUDA-graph captured. Validated with decode at **462,529 prompt tokens** and multi-hour 3-way concurrency stress, zero errors.

---

## Environment

| Item | Value |
|---|---|
| Hardware | 2× NVIDIA DGX Spark (GB10, SM 12.1), 128 GB unified LPDDR5x each |
| Interconnect | ConnectX-7, 2× 200 Gb/s RoCE (NCCL `NET/IB` + GPUDirect RDMA/DMABUF confirmed in logs) |
| Image | `ghcr.io/bjk110/vllm-spark:dsv4-sm121-indexer-production` = `sha256:ade810fd637e30922a30d09f0fcf128fbeb2a757a27a64f8a77e3646fae223a7` |
| vLLM | `v0.24.0.dev0+dsv4.pr41834.72261a7` (PR #41834 stack) |
| Model | DeepSeek-V4-Flash, served TP=2, `--nnodes 2 --distributed-executor-backend mp` |
| Relevant flags | `--max-model-len 524288 --kv-cache-dtype fp8 --block-size 256 --max-num-seqs 4 --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching --gpu-memory-utilization 0.80` |
| Compilation | `{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_num_of_warmups":2}` (the validated SM121 preset), MTP spec-decode `num_speculative_tokens=1` |
| KV pool | 1,421,799 tokens (15.05 GiB fp8_ds_mla per node), max concurrency 2.71× at 524,288 |

## Symptom

Intermittent, hard engine death during long-context workloads. Everything is healthy for hours, then:

```
(Worker_TP0) ERROR [multiproc_executor.py:990] WorkerProc hit an exception.
  ...
  File ".../vllm/compilation/cuda_graph.py", line 254, in __call__
  File ".../vllm/models/deepseek_v4/attention.py", line 461, in attention_impl
  File ".../vllm/model_executor/layers/sparse_attn_indexer.py", line 422, in sparse_attn_indexer
    logits = fp8_fp4_paged_mqa_logits(
  File ".../vllm/utils/deep_gemm.py", line 505, in _fp8_paged_mqa_logits_sm12x
  File ".../vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py", line 546
  File ".../vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py", line 464, in fp8_paged_mqa_logits_rowwise_triton
    _fp8_paged_mqa_logits_rowwise_kernel[grid](
  File ".../triton/runtime/jit.py", line 760, in run
    launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
  File ".../triton/compiler/compiler.py", line 468, in _init_handles
    self.module, ... = driver.active.utils.load_binary(
RuntimeError: Triton Error [CUDA]: operation not permitted
(EngineCore) ERROR [core.py:1231] EngineCore encountered a fatal error.
→ vllm.v1.engine.exceptions.EngineDeadError → all requests 500
```

Key confusing property: **a stress test up to 256K tokens (incl. 2-way concurrency) passes**, then a real workload crashes later. This is because the trigger is *shape novelty within the process lifetime*, not load.

## Root cause chain (all file references are paths inside the image)

1. **Decode-side sparse indexer dispatch** - `vllm/model_executor/layers/sparse_attn_indexer.py:391`: the graph-safe *direct top-k* path (the one hardened by the `72261a7` "graph-safe topk padding" fix) is only taken when `logits_bytes > sparse_indexer_max_logits_bytes()`. That threshold defaults to **256 MB on SM12x** (`vllm/v1/attention/backends/mla/indexer.py:36`, override: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`). With small decode batches (`num_padded_tokens = batch × next_n ≤ 8` and `logits_width ≤ 524288`), `logits_bytes ≤ ~16 MB` - the threshold is **never crossed**, so decode always uses the fallback:

2. **The fallback materializes full logits via a Triton kernel** - `sm12x_mqa.py::fp8_paged_mqa_logits_rowwise_triton` (selected for any model with `head_dim % 64 == 0 and num_heads % 4 == 0`, i.e. always for DSv4-Flash). Its kernel declares **`num_rows`, `logits_width`, and all 17 stride parameters as `tl.constexpr`**:

3. **`logits_width` varies with context** - `sparse_attn_indexer.py::_decode_logits_width` returns `min(max_model_len, max_seq_len_of_batch)` with **no bucketing**. Every novel `(logits_width, num_rows, strides)` combination ⇒ a brand-new Triton specialization ⇒ compile + **`cuModuleLoad`** at first launch *in that process*.

4. **First launch during capture = death** - under `FULL_DECODE_ONLY`, decode batches are CUDA-graph captured (`AttentionCGSupport.UNIFORM_BATCH` lazy capture per batch descriptor). If the first launch of a new kernel specialization lands inside a capture, `cuModuleLoad` is illegal (`CUDA_ERROR_NOT_PERMITTED`, error 800) and the worker dies. This is the *same* failure class that `72261a7` fixed for the direct top-k kernel (`_fp8_(paged_)mqa_logits_kernel` constexpr specialization) - the rowwise variant simply wasn't covered.

Empirical confirmation: after the crash, the persistent Triton cache contained **zero** compiled variants of the rowwise kernel - its first-ever compilation in that process was the fatal one. This also explains the "mostly stable" feel: the more your workload explores novel context lengths (long documents, agents), the more often you roll the dice.

## The fix (19 changes, kernel signature only)

De-constexprify the parameters that legitimately vary at runtime. Model/config constants (`next_n`, `num_heads`, `head_dim`, `block_size`) and tile sizes (`BLOCK_N/D/H`) stay `tl.constexpr`. Triton still lightly specializes runtime ints (on `==1` / `%16==0`), so the variant space collapses from *unbounded* to a handful - all compiled and loaded during startup warmup, none left to load mid-capture. Decode stays **fully CUDA-graph captured**: no PIECEWISE, no eager fallback, no throughput cost.

```diff
--- a/vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py
+++ b/vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py
@@ -302,29 +302,29 @@  def _fp8_paged_mqa_logits_rowwise_kernel(
     block_tables_ptr,
     logits_ptr,
     token_start,
-    num_rows: tl.constexpr,
-    logits_width: tl.constexpr,
+    num_rows,
+    logits_width,
     next_n: tl.constexpr,
     num_heads: tl.constexpr,
     head_dim: tl.constexpr,
     block_size: tl.constexpr,
-    stride_qb: tl.constexpr,
-    stride_qn: tl.constexpr,
-    stride_qh: tl.constexpr,
-    stride_qd: tl.constexpr,
-    stride_kvb: tl.constexpr,
-    stride_kvs: tl.constexpr,
-    stride_kvd: tl.constexpr,
-    stride_sb: tl.constexpr,
-    stride_ss: tl.constexpr,
-    stride_wm: tl.constexpr,
-    stride_wh: tl.constexpr,
-    stride_clb: tl.constexpr,
-    stride_cln: tl.constexpr,
-    stride_btb: tl.constexpr,
-    stride_btk: tl.constexpr,
-    stride_lm: tl.constexpr,
-    stride_ln: tl.constexpr,
+    stride_qb,
+    stride_qn,
+    stride_qh,
+    stride_qd,
+    stride_kvb,
+    stride_kvs,
+    stride_kvd,
+    stride_sb,
+    stride_ss,
+    stride_wm,
+    stride_wh,
+    stride_clb,
+    stride_cln,
+    stride_btb,
+    stride_btk,
+    stride_lm,
+    stride_ln,
     BLOCK_N: tl.constexpr,
     BLOCK_D: tl.constexpr,
     BLOCK_H: tl.constexpr,
```

No body changes needed: `logits_width`/`num_rows` are only used in masking comparisons, strides only in address arithmetic - all fine as runtime scalars. Perf impact is negligible (constants become registers).

### Applying it without rebuilding the image

Generate the patched file from the stock one (idempotent, asserts exactly 19 replacements):

```bash
docker exec <container> cat /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py > sm12x_mqa.py.orig
python3 - <<'EOF'
import re
src = open("sm12x_mqa.py.orig").read()
m = re.search(r"def _fp8_paged_mqa_logits_rowwise_kernel\((.*?)\):\n", src, re.S)
new_sig, n = re.subn(r"\b(num_rows|logits_width|stride_\w+): tl\.constexpr", r"\1", m.group(1))
assert n == 19, n
open("sm12x_mqa.py", "w").write(src[:m.start(1)] + new_sig + src[m.end(1):])
EOF
```

Then place `sm12x_mqa.py` on a volume already mounted in the container (e.g. the models mount) and copy it over the stock file at serve time, *before* `vllm serve` starts (containers recreated per launch keep the stock image intact - rollback = stop copying):

```bash
# in the serve entry script, before exec vllm serve:
OPS=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops
[ -f /models/patches/sm12x_mqa.py ] && cp /models/patches/sm12x_mqa.py "$OPS/sm12x_mqa.py"
```

## Validation

- Before: crash signature above, triggered by an agent workload reading large documents (first decode at a novel >256K shape while a capture was in flight). Reproducible risk window on any fresh engine process.
- After (same flags, `FULL_DECODE_ONLY` + MTP=1 unchanged):
  - decode at **250,292 prompt tokens** → HTTP 200, coherent output;
  - decode at **462,529 prompt tokens** (novel-shape territory for the process) → HTTP 200, correct answer, ~7 min end-to-end (prefill-dominated);
  - new Triton compilations observed *during* the long request without incident;
  - multi-hour stress with 3 concurrent requests: **0 ERROR** lines on both nodes;
  - prefill ~1.3-1.6K tok/s and decode ~40-47 tok/s unchanged vs. pre-patch benchmarks.

## Workarounds if you can't patch

- `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1` - forces the graph-safe direct top-k path for contexts ≳128K (the risk zone). Caveat: the same threshold participates in prefill-side logits chunk sizing (`sparse_attn_indexer.py:174`); `0` is **not** recommended.
- `cudagraph_mode PIECEWISE` - keeps the indexer out of full-graph capture; costs decode throughput.
- Warming up every context-length range you'll ever serve, per engine start, "works" but is impractical at 512K (multi-minute prefills per launch) and is defeated by the unbounded specialization space anyway.

## Suggested upstream fix options

1. Take the signature patch above as-is (smallest diff, validated).
2. Or mirror the `72261a7` approach: pad `logits_width` to fixed power-of-two buckets clamped at `max_model_len`, keeping constexpr but bounding the variant space, and pre-compile the buckets at init.
3. Or lower the SM12x direct-topk threshold so small-batch decode also uses the already-graph-safe path (needs an eye on the prefill chunk-sizing interaction).

Happy to provide full logs, the exact stack trace, or run validation of a fixed image on the same dual-Spark setup.
