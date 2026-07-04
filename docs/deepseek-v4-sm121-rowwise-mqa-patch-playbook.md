# Agent Playbook - Fix `EngineDeadError` at long context on GB10/SM121 (rowwise indexer CUDA-graph crash)

> **How to use this file**: hand it to Claude Code (or any coding agent with shell access to your
> DGX Spark) and say *"apply this playbook"*. It covers detection, patch, deployment, validation,
> rollback. A human can also follow it step by step.
>
> **Scope**: vLLM images from this repo serving DeepSeek-V4-Flash on DGX Spark (GB10, SM 12.1),
> single- or dual-node. Known affected: `dsv4-sm121-indexer-production`
> (= `v023-dsv4-72261a7-sm121-deepgemm-indexer-prod-fa83457d`, digest `sha256:ade810fd…`).
> **Root cause & full analysis**: see the companion report (`docs/deepseek-v4-sm121-rowwise-mqa-cudagraph-fix.md`).

## What this fixes

The first decode step at a context-length shape *novel to the engine process* can trigger a Triton
JIT `cuModuleLoad` **inside a CUDA graph capture** → `RuntimeError: Triton Error [CUDA]: operation
not permitted` → `EngineDeadError` → all requests 500. In the reported production reproduction this
fired at >256K prompt tokens on a 512K deployment; that is the *observed* territory, not an
intrinsic threshold - the dispatch decision is logits bytes vs. the SM12x sparse-indexer threshold,
and small-batch decode stays below that threshold at any context length. The fix de-constexprifies
the 19 runtime-variant parameters of `_fp8_paged_mqa_logits_rowwise_kernel` so the specialization
space collapses to a handful of variants, all loaded at warmup. Decode stays fully CUDA-graph
captured (`FULL_DECODE_ONLY` unchanged) - no eager fallback, no measured throughput change.

## Conventions

- `$CONTAINER` = your vLLM container name (this repo's runbooks use `vllm_ds4`).
- `$MODELS_MOUNT` = a host directory already bind-mounted into the container (the runbooks mount
  `~/models` at `/models`; adjust if yours differs).
- `$SM12X` = `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py`
  (path inside the container).
- On multi-node (TP over 2 Sparks): **repeat steps 2-4 on every node**.

---

## Step 1 - Check the target file exists

```bash
docker exec $CONTAINER test -f $SM12X && echo present || echo MISSING
```

- `MISSING` → this image has a different layout (different vLLM build); **stop and report** -
  this playbook does not apply as-is.
- `present` → continue. A quick (non-authoritative) signal of the pre-patch state:

```bash
docker exec $CONTAINER grep -c "num_rows: tl.constexpr" $SM12X
```

`1` strongly suggests you are affected. **Do not treat `0` as "safe/already fixed"** - `0` also
happens if the kernel was renamed, the signature layout changed, or the file is in a partial
state. The authoritative check is the state detection built into the apply script in Step 2.

Optional confirmation that your past crashes match this bug - look for this pair in your serve logs:

```
File ".../ops/sm12x_mqa.py", line 464, in fp8_paged_mqa_logits_rowwise_triton
RuntimeError: Triton Error [CUDA]: operation not permitted
```

## Step 2 - Extract the stock file and generate the patched copy

The patch is applied by `patches/sm121/apply_sm121_rowwise_mqa_graph_safe.py` from this repo
(fetch it from the repo if you only have this playbook file). The script carries an **explicit
allowlist of the 19 target parameters** and distinguishes three states:

- all 19 still `tl.constexpr` → patches (atomic replace, `ast.parse` + `py_compile` validated);
- all 19 already runtime arguments → prints `already applied`, exits 0, writes nothing;
- anything else (missing/renamed parameters, unexpected annotations, kernel not found, layout
  drift) → **refuses with a nonzero exit**. Report the message; do not improvise.

```bash
mkdir -p $MODELS_MOUNT/patches
docker exec $CONTAINER cat $SM12X > $MODELS_MOUNT/patches/sm12x_mqa.py.orig
cp $MODELS_MOUNT/patches/sm12x_mqa.py.orig $MODELS_MOUNT/patches/sm12x_mqa.py
python3 apply_sm121_rowwise_mqa_graph_safe.py $MODELS_MOUNT/patches/sm12x_mqa.py
```

Interpret the outcome:

- `patched - 19 params de-constexpr'd` → you were affected; the patched copy is ready, continue.
- `already applied` → the image already carries the fix; remove
  `$MODELS_MOUNT/patches/sm12x_mqa.py` and stop - nothing to deploy.
- nonzero exit → the image layout changed; report the exact message to the maintainer instead of
  forcing anything. The maintainer may have fixed the kernel differently.

`tl.constexpr` is kept on model constants (`next_n`, `num_heads`, `head_dim`, `block_size`) and
tile sizes (`BLOCK_N/D/H`); only the signature of one kernel is touched.

## Step 3 - Auto-apply at serve time (survives container recreation)

Containers are recreated from the stock image on every launch, so copy the patched file over the
stock one *before* `vllm serve` starts. In your serve entry script (this repo's runbooks use a
`serve.sh` that is `docker cp`'d into the container), insert **before the `exec vllm serve` line**:

```bash
# --- long-context CUDA-graph crash fix (rowwise indexer, see companion report) ---
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
   prompt **exceeds 262,144 tokens as counted by the real model tokenizer** - do not assume a
   byte size maps to a token count (tokens-per-byte varies wildly with content and tokenizer).
   Count with the server's own tokenizer via the `/tokenize` endpoint before sending:

   ```bash
   python3 - <<'EOF'
   import json, urllib.request
   base = "http://YOUR_HEAD_NODE:PORT"      # the vLLM OpenAI endpoint
   prompt = open("prompt.txt").read()        # your filler prompt
   req = urllib.request.Request(
       base + "/tokenize",
       data=json.dumps({"model": "YOUR_SERVED_MODEL_NAME", "prompt": prompt}).encode(),
       headers={"Content-Type": "application/json"})
   print("prompt tokens:", json.load(urllib.request.urlopen(req))["count"])
   EOF
   ```

   Grow/trim the prompt until the count exceeds the target, then send it with `max_tokens: 48`,
   `temperature: 0`.
   - Expected: HTTP 200 after the prefill (several minutes on GB10), **no** `ERROR` in serve logs.
   - Pre-patch, this class of request could return 500 with the `operation not permitted` trace.
3. Optional: repeat near your `--max-model-len` (e.g. ~460K prompt tokens at 512K max) and run a
   few concurrent requests for an hour. Reference results on a dual-Spark TP=2 setup: 250,292 and
   462,529 prompt-token decodes clean, multi-hour 3-way concurrency stress with zero errors,
   prefill/decode throughput unchanged.

## Rollback

```bash
rm $MODELS_MOUNT/patches/sm12x_mqa.py   # on every node
# relaunch - containers are recreated from the stock image
```

Semantics, to be precise: the **image layers are immutable and are never modified** by this flow.
What the serve-time `cp` modifies is the **container's writable layer**. If your launch flow
recreates containers on every start (as this repo's runbooks do), removing the patched file from
the mount fully reverts to stock at the next launch. If you instead copied the file into a
long-lived container with `docker exec`/`docker cp`, that container's writable layer remains
modified until the container itself is recreated - rollback then requires recreating the
container, not just deleting the file from the mount.

## Workaround without patching (with a caveat)

`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1` forces the graph-safe direct top-k path for long contexts.
**Warning**: this threshold is not decode-only - it also participates in prefill-side logits
chunk sizing, so it may change prefill behavior/performance. Do **not** leave it set while
running prefill-performance attribution experiments, and do not set it to `0`.

## Notes for agents applying this

- Do not run destructive commands beyond what is listed; the stock image is never modified.
- If Step 1 reports the file missing, or the Step 2 script exits nonzero, report the exact
  output and stop - do not force-apply and do not improvise a different patch.
- A `grep` count of `0` in Step 1 is **not** proof the image is fixed; only the Step 2 state
  detection (explicit 19-parameter allowlist, three-state) is authoritative.
- Multi-node: verify the patched file's checksum is identical on all nodes before relaunching.
