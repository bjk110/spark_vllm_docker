# DeepSeek-V4-Flash — long-output experimental profiles (H1Z-P46)

> **Status: EXPERIMENTAL — THROUGHPUT ONLY — NOT FOR INTERACTIVE SERVING — NOT
> PRODUCTION.** Two disposable long-output concurrency profiles (c2, c4) on the
> current H0 route (image `vllm-spark:h1z-p38c-overlay-rebuild-h0-20260706T125127`,
> byte-identical to production digest `ade810fd` / config `fa83457d`). They change
> ONLY concurrency + KV (max_num_seqs, cudagraph capture, KV bytes); all other
> route identity is the production route (MARLIN MoE, SM121 FP8 Lightning Indexer,
> DEEPSEEK_SPARSE_SWA, DeepGEMM PDL, MTP n=1, TP=2 mp/RoCE, bt8192). They are
> validated for 16K context, long output only; they do NOT replace the production
> baseline and must never be used for interactive/low-latency or production restore.

## Status
- Experimental, not production. Throughput/batch only. Not interactive (high TTFT).
- Smoke-validated (H1Z-P46G): both presets start, pass route proof + correctness 5/5, no OOM/drift.

## Profiles

### c2 — h1z-longout-c2-throughput-experimental
- Preset: `presets/deepseek-v4-h1z-longout-c2-throughput-experimental-tp2.env`
- max_num_seqs=2 ; cudagraph capture [4] ; KV 4 GiB (fp8) ; bt8192 ; max_model_len 40960
- KV capacity: 53,990 tokens / 1.32x. Two concurrent 16K jobs.
- Intended: 16K context, tg1024/tg2048 batch throughput, moderate TTFT.
- Expected: tg1024 ~47.7 tok/s ; tg2048 ~54 tok/s ; TTFT ~15.3s (H1Z-P46B).

### c4 — h1z-longout-c4-deepbatch-experimental
- Preset: `presets/deepseek-v4-h1z-longout-c4-deepbatch-experimental-tp2.env`
- max_num_seqs=4 ; cudagraph capture [8] ; KV 8 GiB (fp8) ; bt8192 ; max_model_len 40960
- KV capacity: 107,980 tokens / 2.64x. Four concurrent 16K jobs. (c4 REQUIRES 8 GiB KV — 4 GiB is INVALID.)
- Intended: 16K context, tg1024/tg2048 deep-batch / offline max throughput.
- Expected: tg1024 ~54 tok/s ; tg2048 ~67 tok/s ; TTFT ~30.6s (H1Z-P46D).

## Do NOT use for
- Short-output or interactive / low-TTFT serving (use the production baseline instead).
- 32K context. max_num_batched_tokens > 8192 (bt16384 / bt24576). KV > 8 GiB. Reduced max_model_len < 40960.
- Ray. DSpark. B12X. Production restore.

## Key design lesson (H1Z-P45/P46)
The correct concurrency lever is to keep max_num_batched_tokens=8192 and raise KV only as max_num_seqs grows
(capture moves with seqs: c2→[4], c4→[8]). Expanding batched tokens is strongly coupled with KV and wastes
the extra KV on prefill working buffers (bt16384 gives no gain). Doubling KV at bt8192 cleanly doubles
usable KV (4 GiB→53,990 ; 8 GiB→107,980).

## Evidence
Validation was run as internal H1Z-P46 tasks (16K context, disposable dual-node H0 route, port 8100):
- H1Z-P46A — first c2 long-output evidence (tg512 neutral, tg1024 1.20x c1).
- H1Z-P46B — c2 validation: tg1024 ~47.7 (1.22x c1), tg2048 ~54 (1.37x c1), reproducible.
- H1Z-P46C — c4 admission gate: 8 GiB KV admits four 16K jobs (KV 107,980 / 2.64x).
- H1Z-P46D — c4 validation: tg1024 ~54, tg2048 ~67 (~1.24x c2), same-session c2 head-to-head.
- H1Z-P46G — dual-node preset smoke: both presets pass route proof + correctness.

## Safety / usage
- Always start with an explicit `--env-file <preset>`, a unique compose project (`-p <name>`), and the
  non-production port 8100. NEVER use bare `docker compose`.
- Production restore MUST use `--env-file presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env`
  (never bare `.env`, which resolves to base checkpoint 4c41950c, NOT production).
- Keep production containers STOPPED while running an experimental profile. Reboot to reclaim UMA after an
  8 GiB-KV (c4) weight-load. Preserve the production baseline (image `ade810fd`, preset hash f1b049d5).
