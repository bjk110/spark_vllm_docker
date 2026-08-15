#!/usr/bin/env python3
"""Startup prewarm for the DeepSeek-V4-Flash-0731 / vLLM 0.27 / b43s promotion
candidate (MAX_NUM_SEQS=1 default profile).

Purpose
-------
vLLM's own boot-time CUDA-graph capture does NOT compile every kernel this
runtime touches. A fixed set of lazy-JIT kernels only compile on the first
real inference request that reaches their code path, which costs the FIRST
matching production request an extra ~15-30s. This script issues a small,
deterministic set of warmup requests against the already-healthy API server
so that cost is paid once, at startup, under operator control -- not on an
arbitrary user's first request.

This is the repository implementation of the prewarm sequence validated in
B4.4A (see docs/deepseek-v4-v027-b43s-promotion-candidate.md and
/home/bjk110/docker-build/deepseek-v4-0731-vllm027-b44a-prewarm-promotion-readiness-20260815T172634KST/).

Default (MS=1) targets
-----------------------
Target A -- short decode (~1,000 input tokens)
    Triggers the DSpark batch-decode Triton kernel family
    (_prepare_dflash_inputs_kernel, _compute_local_logits_stats_kernel,
    _rejection_kernel, _resample_kernel) plus the two TileLang fused-norm
    kernels (mhc_pre_big_fuse_broadcast_with_norm_tilelang,
    mhc_pre_big_fuse_with_norm_tilelang), which B4.4A found are NOT fully
    warmed by vLLM's own boot-time graph capture.

Target B -- non-8192-aligned long prefill (61,440 input tokens)
    Triggers BuildPrefillChunkMetadataKernel.kernel, the chunked-prefill
    remainder-shape metadata kernel. This depth is NOT arbitrary: B4.4A
    directly tested an ~8,300-token prompt first and it did NOT trigger this
    kernel (only Target A's kernels fired). 61,440 was confirmed to trigger
    it (matching the non-aligned remainder shape characterized in B4.3V's
    long-context validation). Do not shorten Target B without re-validating
    against live JIT log markers -- a shorter prompt may silently skip this
    kernel and reintroduce first-use latency on the first real long-context
    request instead.

Optional (MS>=4 only) target
-----------------------------
Target C -- c=4 concurrent short decodes, enabled with --concurrency-warmup.
    Triggers _compute_global_topk_indices_and_lens_kernel (DSpark global
    top-k across >2 simultaneously-decoding sequences). Unreachable at
    MAX_NUM_SEQS=1 -- do not run this against the default profile, it has
    no effect there. Only relevant for the optional MAX_NUM_SEQS>=4 profile
    (see presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-ms4-optional-tp2.env).

FlashInfer AutoTuner "outside tuning bucket" fallback and the DSpark
speculator/target CUDA-graph capture itself are NOT prewarm targets here --
the former is a performance-only per-shape cache-miss fallback (not a JIT
compile), the latter already happens during vLLM's own boot sequence.

Exit code: 0 if every requested stage returned HTTP 200 with a non-empty
completion; nonzero otherwise. Never produces a false success marker.
"""
import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_S = 180
HEALTH_TIMEOUT_S = 10
TARGET_A_TOKENS = 1000
TARGET_B_TOKENS = 61440
# Deterministic filler text (English, ASCII-only) repeated and trimmed to an
# exact token count at runtime -- no large literal prompt file is stored in
# the repository.
FILLER_SENTENCE = (
    "The DeepSeek-V4-Flash-0731 promotion candidate runs tensor-parallel "
    "rank {rank} of {tp_size} on a DGX Spark node over RoCE. "
)


def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def wait_for_health(base_url, timeout_s):
    deadline = time.monotonic() + timeout_s
    last_err = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(base_url + "/health")
            with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT_S) as resp:
                if resp.status == 200:
                    return True
        except Exception as e:  # noqa: BLE001 - report exact cause, keep polling
            last_err = e
        time.sleep(2)
    log(f"health check did not pass within {timeout_s}s (last error: {last_err})")
    return False


def build_prompt(tokenizer, target_tokens):
    """Build a deterministic prompt of exactly `target_tokens` input tokens.

    Repeats FILLER_SENTENCE, encodes, then trims to the exact token count so
    both Target A and Target B reach precise, reproducible depths (Target
    B's non-8192-aligned depth is the whole point of that trigger).

    Estimates the repeat count from a single unit encode (O(target_tokens)
    total, not O(target_tokens^2)) -- at 61,440 tokens, re-encoding the
    whole growing string on every iteration would dominate wall time far
    more than the model request itself.
    """
    unit = FILLER_SENTENCE.format(rank=0, tp_size=2)
    unit_len = max(len(tokenizer.encode(unit)), 1)
    reps = (target_tokens // unit_len) + 2  # small safety margin, trimmed below
    text = unit * reps
    ids = tokenizer.encode(text)
    while len(ids) < target_tokens:  # extremely unlikely with the margin above
        text += unit
        ids = tokenizer.encode(text)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids), len(ids)


def post_completion(base_url, model, prompt, timeout_s, max_tokens=16):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "seed": 0,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.URLError as e:
        return {"ok": False, "status": f"ERROR:{e}", "e2e_s": time.monotonic() - t0}
    e2e_s = time.monotonic() - t0
    try:
        parsed = json.loads(raw.decode("utf-8"))
        text = parsed["choices"][0]["text"]
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status": status, "e2e_s": e2e_s, "parse_error": str(e)}
    return {"ok": bool(text), "status": status, "e2e_s": e2e_s, "text_len": len(text)}


def run_stage(name, fn):
    log(f"STAGE_START {name}")
    t0 = time.monotonic()
    result = fn()
    result["duration_s"] = round(time.monotonic() - t0, 2)
    result["name"] = name
    log(f"STAGE_END {name} ok={result.get('ok')} duration_s={result['duration_s']} "
        f"detail={json.dumps({k: v for k, v in result.items() if k not in ('name',)})}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", required=True, help="SERVED_MODEL_NAME")
    ap.add_argument("--tokenizer-path", required=True,
                     help="Local path to the model directory (container-internal, e.g. MODEL_CONTAINER_PATH)")
    ap.add_argument("--out", default=None, help="Optional path to write JSON results")
    ap.add_argument("--health-timeout", type=int, default=60,
                     help="Seconds to wait for /health=200 before giving up")
    ap.add_argument("--request-timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--concurrency-warmup", action="store_true",
                     help="Also run Target C (c=4 global-topk warmup). Only meaningful "
                          "for the optional MAX_NUM_SEQS>=4 profile -- has no effect "
                          "at MAX_NUM_SEQS=1 and is NOT part of the default sequence.")
    args = ap.parse_args()

    log(f"prewarm starting: base_url={args.base_url} model={args.model}")

    if not wait_for_health(args.base_url, args.health_timeout):
        log("FAILED: engine did not become healthy before the health timeout")
        sys.exit(2)

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        log(f"FAILED: transformers not importable in this environment: {e}")
        sys.exit(3)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    results = []

    def target_a():
        prompt, actual = build_prompt(tokenizer, TARGET_A_TOKENS)
        r = post_completion(args.base_url, args.model, prompt, args.request_timeout, max_tokens=16)
        r["actual_input_tokens"] = actual
        return r

    results.append(run_stage("target_a_short_decode", target_a))

    def target_b():
        prompt, actual = build_prompt(tokenizer, TARGET_B_TOKENS)
        r = post_completion(args.base_url, args.model, prompt, args.request_timeout, max_tokens=16)
        r["actual_input_tokens"] = actual
        return r

    results.append(run_stage("target_b_nonaligned_chunk", target_b))

    if args.concurrency_warmup:
        def target_c():
            prompts = [build_prompt(tokenizer, TARGET_A_TOKENS)[0] for _ in range(4)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                futs = [ex.submit(post_completion, args.base_url, args.model, p,
                                   args.request_timeout, 16) for p in prompts]
                sub = [f.result() for f in futs]
            return {"ok": all(s["ok"] for s in sub), "status": 200 if all(s["ok"] for s in sub) else "PARTIAL_FAIL",
                    "n": len(sub), "sub_results": sub}

        results.append(run_stage("target_c_concurrent_topk", target_c))
    else:
        log("target_c_concurrent_topk SKIPPED (--concurrency-warmup not set; "
            "not part of the default MAX_NUM_SEQS=1 sequence)")

    all_ok = all(r["ok"] for r in results)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)

    log(f"prewarm {'COMPLETE' if all_ok else 'FAILED'}: "
        f"{sum(1 for r in results if r['ok'])}/{len(results)} stages ok")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
