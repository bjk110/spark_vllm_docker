# DeepSeek-V4-Flash Prefill-Optimization Campaign Plan

| Field | Value |
|---|---|
| Status | `Historical` |
| Scope | Completed prefill-optimization campaign plan (max-num-batched-tokens sweep) |
| Current replacement | [`deepseek-v4-sm121-indexer-production.md`](deepseek-v4-sm121-indexer-production.md) (current production) |
| Last validated | 2026-06-25 (campaign concluded; prefill8192 adopted, later superseded) |
| Runtime or image identity | Not applicable (planning document) |
| Historical relevance | Records the completed campaign that produced the prefill8192 baseline; the planned actions are not current operational instructions |

> **Status note:** this campaign is **complete**. The planned actions below must not
> be treated as current operational instructions; the current production baseline is
> the SM121 indexer (see current replacement above).

**Status:** EXPERIMENTAL / DISPOSABLE / untracked. NOT a production plan.

## Objective

Determine whether raising `max-num-batched-tokens` improves long-context prefill
throughput and TTFT without compromising correctness, 131K retrieval, MTP accounting,
CUDA-graph stability, decode throughput, unified-memory headroom, paging, or host
responsiveness. **Change only `max-num-batched-tokens`.**

## Confirmed baseline (do not rerun)

Long-context capability complete: LC0 16K, LC1 32K, LC2 64K, LC3 131K all PASS. The
LC3 `max-num-batched-tokens=2048` run is the reference baseline. A static audit (below)
proves its measurement definition is identical to this campaign's, so it is **not** rerun.

| Context | TTFT | TTFT-derived prefill | Streaming decode |
| --- | ---: | ---: | ---: |
| 32K (~32,737 tok) | 21.42 s | ~1,529 t/s | 21.67 t/s |
| 64K (~65,531 tok) | 46.54 s | ~1,408–1,412 t/s | 21.68 t/s |
| 131K (~131,043 tok) | 103.71 s | ~1,264 t/s | 21.38 t/s |

LC3 131K also: MTP acceptance 89.59%, retrieval en/ko/multi pass, fallback 0, recapture 0,
no memory pressure.

## Phase 0 — Source/runtime audit (DONE)

Pinned source `72261a7`, image config `4c41950c`. Findings:

- Field path: env `MAX_NUM_BATCHED_TOKENS` → entrypoint `--max-num-batched-tokens`
  (`entrypoints/entrypoint-rdma.sh:241,282`) → `scheduler_config.max_num_batched_tokens`.
- `config/vllm.py::_set_max_num_scheduled_tokens()`: with speculative decoding the scheduler
  sets `max_num_scheduled_tokens = max_num_batched_tokens - scheduled_token_delta`, where
  `scheduled_token_delta = max_num_new_slots_for_drafting * max_num_seqs`. For MTP
  (`uses_draft_model()=True`, serial drafting) `max_num_new_slots_for_drafting = 1` and
  `max_num_seqs = 1`, so **delta = 1**: scheduled = batched − 1 (2047 / 4095 / 8191).
- The value is **honored, not overridden or capped**. The `vllm.py` warning
  ("max_num_scheduled_tokens ... below 8192 ... suboptimal") is **informational only**;
  it confirms increasing the value is the intended prefill lever. It fires for all three
  values (2047/4095/8191 < 8192).
- **Chunked prefill stays enabled** (`enable_chunked_prefill=True`). A 131K prompt is split
  into prefill chunks of ~`max_num_scheduled_tokens`: ~64 chunks at 2048, ~32 at 4096,
  ~16 at 8192. Fewer, larger chunks is the throughput-gain mechanism.
- **CUDA-graph capture sizes are unaffected**: `_set_cudagraph_sizes()` uses
  `min(max_num_seqs*2, 512) = 2` plus the explicit `cudagraph_capture_sizes=[2]`. Independent
  of `max_num_batched_tokens`.
- **Decode is unaffected**: at concurrency 1 / `max_num_seqs=1` / MTP n=1 the decode batch is
  2 tokens, far below any of these limits.
- **Memory risk**: prefill activation and persistent scheduler input buffers scale ~linearly
  with `max_num_scheduled_tokens` (2048→8192 ≈ 4×). Must be monitored.
- No DeepSeek-V4-specific restriction on `max_num_batched_tokens` was found.

Effective-value proof required from runtime logs (per variant): entrypoint echo
`--max-num-batched-tokens <V>`; `non-default args: {... 'max_num_batched_tokens': <V> ...}`;
engine `scheduler_config` showing the value; the warning line value `<V-1>`.

## Phase 1 — Disposable presets (DONE)

Both derived from the corrected LC3 131K preset
(`...-longctx131k-exp-tp2.env`, SHA `45508585912cd927…`). Single semantic delta each:

- `presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill4096-exp-tp2.env`
  — `MAX_NUM_BATCHED_TOKENS=4096`, SHA `89c3ff4e73c0c153…`
- `presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-exp-tp2.env`
  — `MAX_NUM_BATCHED_TOKENS=8192`, SHA `06148eaeac6abe9d…`

Everything else identical: MML 135168, KV 4294967296 bytes fp8, MTP n=1, FULL_DECODE_ONLY,
capture [2], prefix cache off, concurrency 1, TP=2 mp, NET/IB, EP off, B12X off, parser/tool
off, Ray off, no memcheck bypass, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`.

## Phase 2 — Measurement definition

Identical tokenization, prompt construction, endpoint (`/v1/completions`), streaming, and
timing boundaries as LC2/LC3. Two prefill metrics recorded separately:

1. `TTFT` (first-token latency, perf_counter boundary).
2. `ttft_prefill_tps = actual_api_prompt_tokens / ttft_seconds` — **TTFT-derived**, using the
   server's `usage.prompt_tokens` (via `stream_options.include_usage`), labeled as such.

vLLM exposes no native prompt-processing-throughput gauge (only a TTFT histogram and a
`prompt_tokens` counter), so no native rate is merged with the TTFT-derived value. **No
comparison to unholy-fusion's ~2,000 t/s** (timing definition unproven/different).

## Phase 3 — Driver integrity (DONE)

Driver `scripts/diag/dsv4_mtp1_fullgraph_long_context_probe.py`, final SHA
`20631b1a0e1e52fe…` (fixed before the first reboot; **unchanged across both variants**).
Campaign extension only: `include_usage` capture, `ttft_prefill_tps`, perf/repeat
aggregation, and a `prefill` stage (`max_model_len 135168`, depths 32K/64K/131K). Exact-token
builder unchanged; static tests pass (32K 32737, 64K 65531, 131K 131043 std; 131K en 131064,
ko 131050, multi 131072 — all ≤ target and within 64). If the driver must change after the
4096 run, the campaign stops and re-requests authorization instead of running 8192.

## Variant order and gating

1. 4096 (clean boot) → screening 32K/64K → 131K confirmation → repetition → classify.
2. 8192 only if 4096 passes all safety/correctness gates (a no-gain-but-safe 4096 still
   permits 8192). If 4096 fails a safety or correctness gate, 8192 does not run.

Each variant: independent clean boot, no second reboot within a variant, no same-boot switch,
no config change inside a running session. Reboot/clean-memory/cache-clear gates per the
LC3 procedure (both boot IDs change, MemAvailable ≥ 110 GiB, swap 0, ports free, RoCE/RDMA
active, disk ≥ 25 GiB, clear only `./.cache/vllm`).

## Classification

Per variant: `PREFILL_VARIANT_{MAJOR_GAIN|MODERATE_GAIN|NO_MATERIAL_GAIN|REGRESSION|UNSTABLE|INVALID}`
on 131K prefill change vs the 2048 baseline (major ≥15%, moderate 5–<15%, no-gain ±5%,
regression < −5% prefill or decode, unstable on CV>5%/drift/memory/paging/graph/rank issues).

Campaign outcome (exactly one): `PROMOTE_PREFILL4096_AS_VALIDATED_CANDIDATE`,
`PROMOTE_PREFILL8192_AS_VALIDATED_CANDIDATE`, `RETAIN_PREFILL2048_BASELINE`,
`PREPARE_INTERMEDIATE_PREFILL_SIZE_TEST`, `PREPARE_B12X_EXPERIMENT`,
`INVESTIGATE_PREFILL_KERNEL_PATH`, `INVESTIGATE_MEMORY_OR_PAGING`,
`ABANDON_PREFILL_OPTIMIZATION`. No production promotion; a validated candidate may only be
recommended, not committed.

## Result directories (do not overwrite LC0–LC3)

- `benchmarks/results/dsv4-prefill4096-131k-<timestamp>/`
- `benchmarks/results/dsv4-prefill8192-131k-<timestamp>/`

---

## Validation follow-up (2026-06-25): 8192 formalized as validated candidate

Two independent runtime classifications now both PASS:
- cold reproduction (harness-corrected): `PREFILL8192_COLD_REPRO_PASS`
- 60-minute mixed-context stability: `PREFILL8192_60MIN_STABILITY_PASS`

Candidate decision: `FORMALIZE_PREFILL8192_VALIDATED_CANDIDATE` (NOT production).

A prior independent cold reproduction was `PREFILL8192_INCONCLUSIVE` due to a non-required
64K Korean retrieval probe (harness prompt-construction / output-budget artifact); that run is
preserved unchanged. The retrieval harness was corrected (probe matrix, 128-token budget,
strong response contract, strict-but-tolerant validator, offline unit tests including the prior
failed completion as a negative regression). Corrected driver SHA-256
`d67075b46a1bfd06e380d6db6baefa40b23129b850372a9a96d39d385e76f2b1`.

Formal candidate preset (untracked):
`presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-validated-candidate-tp2.env`
(SHA-256 `ed13ee5e17668509f8e08072bcc206981bf43f497121380dc0000d780aa1f88a`). Full record:
`docs/deepseek-v4-prefill8192-validated-candidate.md`.
