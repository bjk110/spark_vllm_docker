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
- Expected: tg1024 ~54 tok/s ; tg2048 ~67 tok/s ; tg4096 ~76 tok/s (1.91x c1, H1Z-P48A) ; TTFT ~30.6s (H1Z-P46D).

## Do NOT use for
- Short-output or interactive / low-TTFT serving (use the production baseline instead).
- 32K context. max_num_batched_tokens > 8192 (bt16384 / bt24576). KV > 8 GiB. Reduced max_model_len < 40960.
- Ray. DSpark. B12X. Production restore.

## Key design lesson (H1Z-P45/P46)
The correct concurrency lever is to keep max_num_batched_tokens=8192 and raise KV only as max_num_seqs grows
(capture moves with seqs: c2→[4], c4→[8]). Expanding batched tokens is strongly coupled with KV and wastes
the extra KV on prefill working buffers (bt16384 gives no gain). Doubling KV at bt8192 cleanly doubles
usable KV (4 GiB→53,990 ; 8 GiB→107,980).

## c4 tg4096 asymptote (H1Z-P48A)
The c4 deep-batch profile was probed at longer output (tg4096) to locate its practical throughput asymptote.
Result: `H1Z_P48A_C4_TG4096_STRONG`. Same-day, same committed c4 route (preset hash `e49f96cb`, unchanged);
route proof PASS, correctness 5/5. The PR #18 SM121 rowwise-MQA standby patch was **not wired** for this run.

Convention: 16K context (`--depth 16384 --pp 2048`), `--exact-tg`, `--tg 4096`, `--runs 2`, identical c4 route
(max_num_seqs=4, capture [8], KV 8 GiB fp8, bt8192, max_model_len 40960).

| concurrency | total tok/s | peak tok/s | TTFT |
|---|---|---|---|
| c1 | 39.83 ± 0.41 | 45  | ~10.2 s |
| c2 | 58.64 ± 0.04 | 69  | ~18.7 s |
| c4 | 76.18 ± 0.30 | 102 | ~31.3 s |

- Ratios: **c4/c1 = 1.91x**, c4/c2 = 1.30x (same-day, same route).
- c4 aggregate scales with output length: tg1024 1.37x → tg2048 1.70x → **tg4096 1.91x** c1; wall/peak
  0.52 → 0.64 → 0.75 (peak stays ~102 tok/s). tg4096 is the **practical asymptote point** for c4 long-output
  deep-batch use — still climbing toward the ~2.56x theoretical ceiling but with diminishing headroom.
- Run health: exit 0, 1515 s, coherence PASS, HTTP 200 throughout, MemAvailable flat, no OOM / leak / preemption / route drift.
- **Still NOT interactive**: per-request ~21 tok/s and TTFT ~31 s at c4 — offline / deep-batch / long-output
  throughput only. Not a production baseline and not a production promotion.
- **tg8192**: optional only, not recommended by default — expected diminishing returns (~2.1–2.4x c1 at ~2x
  runtime); run only if an absolute upper-bound measurement is explicitly desired.

Evidence: `/home/bjk110/docker-build/h1z-p48a-c4-tg4096-asymptote-probe-20260710T063305/` (SHA256 ALL_OK, 26 files).

## Published metadata image tag (H1Z-P49D)

A GHCR tag was published for provenance and discoverability of the validated H0 runtime paired with this
documented long-output c4 profile. It is a **metadata-only** tag — do not read it as a new runtime.

- Tag: `ghcr.io/bjk110/vllm-spark:h1z-p48-longout-c4-exp-5131a63`
- Remote manifest digest: `sha256:04ff082e9e012924682dd95d77910b1f580a3debd0c8dda7350a82ba0e4a1077`
- Config / local image ID: `sha256:75c7e93b0dcc`
- Runtime delta: **NONE**. RootFS is **H0-equivalent** (106 layers, byte-identical to the production digest
  `sha256:ade810fd…` / config `fa83457d`). The image was built `FROM` H0 with provenance `LABEL`s only
  (no `COPY`/`RUN`/`ENV`, no filesystem change). Remote labels verified 12/12.
- The c4 profile is a **preset** (`presets/deepseek-v4-h1z-longout-c4-deepbatch-experimental-tp2.env`, hash
  `e49f96cb`) that runs ON this image. The P48A result (`H1Z_P48A_C4_TG4096_STRONG`, c4 tg4096 76.18 tok/s =
  1.91x c1) is **preset/profile-driven, not caused by new image bits**.
- Intended use: **offline / deep-batch / long-output** throughput only — not interactive, not a production
  baseline, not a production replacement.
- The PR #18 SM121 rowwise-MQA standby patch is **not wired / not active** in this image. **Issue #17 is not
  fixed** by this tag and remains open.
- Production users should keep using the digest-pinned production path
  (`presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env`, preset hash `f1b049d5`) unless they
  explicitly want this experimental long-output c4 metadata tag.
- Provenance evidence: `/home/bjk110/docker-build/h1z-p49d-ghcr-push-longout-c4-exp-20260710T093000/` (SHA256 ALL_OK, 16 files).

## Evidence
Validation was run as internal H1Z-P46 tasks (16K context, disposable dual-node H0 route, port 8100):
- H1Z-P46A — first c2 long-output evidence (tg512 neutral, tg1024 1.20x c1).
- H1Z-P46B — c2 validation: tg1024 ~47.7 (1.22x c1), tg2048 ~54 (1.37x c1), reproducible.
- H1Z-P46C — c4 admission gate: 8 GiB KV admits four 16K jobs (KV 107,980 / 2.64x).
- H1Z-P46D — c4 validation: tg1024 ~54, tg2048 ~67 (~1.24x c2), same-session c2 head-to-head.
- H1Z-P46G — dual-node preset smoke: both presets pass route proof + correctness.
- H1Z-P48A — c4 tg4096 asymptote (STRONG): c4 76.18 tok/s = 1.91x c1 / 1.30x c2 (peak ~102); see the tg4096 section above.

## Safety / usage
- Always start with an explicit `--env-file <preset>`, a unique compose project (`-p <name>`), and the
  non-production port 8100. NEVER use bare `docker compose`.
- Production restore MUST use `--env-file presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env`
  (never bare `.env`, which resolves to base checkpoint 4c41950c, NOT production).
- Keep production containers STOPPED while running an experimental profile. Reboot to reclaim UMA after an
  8 GiB-KV (c4) weight-load. Preserve the production baseline (image `ade810fd`, preset hash f1b049d5).
