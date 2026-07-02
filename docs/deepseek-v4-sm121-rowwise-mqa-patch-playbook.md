# Agent Playbook - Fix `EngineDeadError` at long context on GB10/SM121 (rowwise indexer CUDA-graph crash)

> **How to use this file**: hand it to Claude Code (or any coding agent with shell access to your
> DGX Spark) and say *"apply this playbook"*. It is self-contained: detection, patch, deployment,
> validation, rollback. A human can also follow it step by step.
>
> **Scope**: vLLM images from this repo serving DeepSeek-V4-Flash on DGX Spark (GB10, SM 12.1),
> single- or dual-node. Known affected: `dsv4-sm121-indexer-production`
> (= `v023-dsv4-72261a7-sm121-deepgemm-indexer-prod-fa83457d`, digest `sha256:ade810fd…`).
> **Root cause & full analysis**: see the companion report (`docs/deepseek-v4-sm121-rowwise-mqa-cudagraph-fix.md`).

## What this fixes

First decode step at a *novel* context-length shape (typically >256K tokens) can trigger a Triton
JIT `cuModuleLoad` **inside a CUDA graph capture** → `RuntimeError: Triton Error [CUDA]: operation
not permitted` → `EngineDeadError` → all requests 500. The fix de-constexprifies the 19
runtime-variant parameters of `_fp8_paged_mqa_logits_rowwise_kernel` so the specialization space
collapses to a handful of variants, all loaded at warmup. Decode stays fully CUDA-graph captured
(`FULL_DECODE_ONLY` unchanged) - no eager fallback, no measured throughput change.

## Conventions

- `$CONTAINER` = your vLLM container name (this repo's runbooks use `vllm_ds4`).
- `$MODELS_MOUNT` = a host directory already bind-mounted into the container (the runbooks mount
  `~/models` at `/models`; adjust if yours differs).
- On multi-node (TP over 2 Sparks): **repeat steps 2-4 on every node**.

---

## Step 1 - Detect whether you are affected

```bash
docker exec $CONTAINER grep -c "num_rows: tl.constexpr" \
  /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py
```

- Output `1` (or more) → **affected**, continue.
- Output `0` or file missing → this image is not affected (already fixed or different layout). Stop.

Optional confirmation that your past crashes match this bug - look for this pair in your serve logs:

```
File ".../ops/sm12x_mqa.py", line 464, in fp8_paged_mqa_logits_rowwise_triton
RuntimeError: Triton Error [CUDA]: operation not permitted
```

## Step 2 - Generate the patched file (idempotent, self-verifying)

```bash
mkdir -p $MODELS_MOUNT/patches
docker exec $CONTAINER cat \
  /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py \
  > $MODELS_MOUNT/patches/sm12x_mqa.py.orig

python3 - <<'EOF'
import re, os
base = os.path.expanduser(os.environ.get("MODELS_MOUNT", "~/models")) + "/patches"
src = open(f"{base}/sm12x_mqa.py.orig").read()
m = re.search(r"def _fp8_paged_mqa_logits_rowwise_kernel\((.*?)\):\n", src, re.S)
assert m, "rowwise kernel signature not found - layout differs, do not proceed"
new_sig, n = re.subn(r"\b(num_rows|logits_width|stride_\w+): tl\.constexpr", r"\1", m.group(1))
assert n == 19, f"expected exactly 19 replacements, got {n} - layout differs, do not proceed"
open(f"{base}/sm12x_mqa.py", "w").write(src[:m.start(1)] + new_sig + src[m.end(1):])
print("patched file written, 19 replacements")
EOF

python3 -m py_compile $MODELS_MOUNT/patches/sm12x_mqa.py && echo "syntax OK"
```

**Safety properties**: the script only touches the *signature* of one kernel; it aborts (assert)
if the file doesn't match the expected layout, so it cannot half-apply on a future fixed image.
`tl.constexpr` is kept on model constants (`next_n`, `num_heads`, `head_dim`, `block_size`) and
tile sizes (`BLOCK_N/D/H`).

## Step 3 - Auto-apply at serve time (survives container recreation)

Containers are recreated from the stock image on every launch, so copy the patched file over the
stock one *before* `vllm serve` starts. In your serve entry script (this repo's runbooks use a
`serve.sh` that is `docker cp`'d into the container), insert **before the `exec vllm serve` line**:

```bash
# --- long-context CUDA-graph crash fix (rowwise indexer, see community report) ---
# Rollback: delete /models/patches/sm12x_mqa.py - the stock image is never modified.
OPS=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops
if [ -f /models/patches/sm12x_mqa.py ] && [ -d "$OPS" ]; then
  cp /models/patches/sm12x_mqa.py "$OPS/sm12x_mqa.py"
  echo "[serve] sm12x_mqa.py patch applied (rowwise indexer graph-safe)"
fi
```

Adjust `/models` if your mount differs. Then relaunch the serving stack as usual and confirm the
log line `sm12x_mqa.py patch applied` appears in each node's serve log.

## Step 4 - Validate

1. **Sanity**: a short chat completion returns 200 with coherent output.
2. **The regression test** (the shape class that used to kill the engine): send one request whose
   prompt exceeds 262,144 tokens (any filler text ≈ 1.1 MB), `max_tokens: 48`, `temperature: 0`.
   - Expected: HTTP 200 after the prefill (several minutes on GB10), **no** `ERROR` in serve logs.
   - Pre-patch, this class of request could return 500 with the `operation not permitted` trace.
3. Optional: repeat near your `--max-model-len` (e.g. ~460K prompt tokens at 512K max) and run a
   few concurrent requests for an hour. Reference results on a dual-Spark TP=2 setup: 250,292 and
   462,529 prompt-token decodes clean, multi-hour 3-way concurrency stress with zero errors,
   prefill/decode throughput unchanged.

## Rollback

```bash
rm $MODELS_MOUNT/patches/sm12x_mqa.py   # on every node
# relaunch - containers come back on the stock image file
```

## Notes for agents applying this

- Do not run destructive commands beyond what is listed; the stock image is never modified.
- If Step 1 returns `0`, report "not affected" and stop - do not force-apply.
- If the Step 2 asserts fail, the image layout changed: report the mismatch instead of improvising;
  the maintainer may have fixed the kernel differently.
- Multi-node: verify the patched file's checksum is identical on all nodes before relaunching.
