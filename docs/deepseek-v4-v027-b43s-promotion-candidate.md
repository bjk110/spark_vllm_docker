# DeepSeek-V4-Flash-0731 / vLLM 0.27 (b43s) — Promotion Candidate

Status: **active production** (promoted 2026-08-15, task B4.4C). This preset is now the formal
DeepSeek-V4-0731 production baseline — see [`docs/deepseek-v4-production.md`](deepseek-v4-production.md)
for the canonical operations document (activation/rollback commands, rollback tiers, residual risks).
This document remains the detailed technical reference for the route's validation history and prewarm
design; it is retained under its original filename from task B4.4B to avoid unnecessary churn across
cross-references at promotion time.

This was implemented as a repository artifact in task B4.4B from the runtime defaults approved by
task B4.4A's promotion-readiness audit (`DEEPSEEK_V4_0731_VLLM027_B4_4A_PREWARM_VALIDATED_PROMOTION_READY`),
live-validated in B4.4B (`B4_4B_PROMOTED_PRESET_IMPLEMENTATION_VALIDATED`), and formally promoted +
published to GHCR in B4.4C. It does not repeat B4.3S-Z validation; it turns that validation's
conclusions into launchable, now-production artifacts.

## 1. Production defaults

- **Preset:** `presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env`
- **Model:** `deepseek-ai/DeepSeek-V4-Flash-0731`
- **Image (published to GHCR — see [`docs/images.md`](images.md)):**
  `ghcr.io/bjk110/vllm-spark:v027-ngc2607-dsv4-0731-dspark-k7-256k-production`
  (local image ID `sha256:a7f0f4b8a508c0b2510fc7e4dcb916491efa03c380c9c7b84dddd4c16ad6f38d`,
  identical on spark01/spark02, unchanged since B4.3S; vLLM 0.27 / NGC 26.07 line)
- **Runtime contract:** TP=2, `DISTRIBUTED_BACKEND=mp`, spark01 rank 0 / spark02 rank 1 (headless),
  master `10.10.10.1:29500`; native DSpark `method=dspark`, `num_speculative_tokens=7`, greedy draft;
  target `FULL_DECODE_ONLY`, `cudagraph_capture_sizes=[8]`, 2 warmups; `MAX_MODEL_LEN=262144`;
  `MAX_NUM_SEQS=1` (approved default — see §5); fixed 10 GiB FP8 KV cache; `max-num-batched-tokens
  8192`; prefix caching disabled; MoE backend MARLIN; `VLLM_USE_DEEP_GEMM_E8M0=1` mandatory (SM121
  numerical contract).
- **Launch:**
  ```bash
  # spark01 (head):
  docker compose --env-file presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env \
    -f docker-compose.yml -f compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml \
    --profile head up -d
  # spark02 (worker):
  docker compose --env-file presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env \
    -f docker-compose.yml -f compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml \
    --profile worker up -d
  ```
  Cold start is ~8-10 minutes (weight load + DeepGEMM/TileLang/CUDA-graph warmup). Wait for
  `vllm-spark-head` health then for `vllm-spark-prewarm` to exit 0 (see §3) before serving traffic
  under load.

## 2. Validation coverage (B4.3S → B4.4A, not repeated here)

| Dimension | Result | Task |
|---|---|---|
| Functional correctness (TP=2, DSpark k=7, API) | PASS | B4.3S |
| Decode throughput vs target-only | 46.75 vs 7.28 tok/s (6.42x), acceptance agg=0.2742 | B4.3T |
| Concurrency scaling | c=1..8, peak 86.5 tok/s @ MS=8/c=8, no acceptance degradation | B4.3U |
| Long context | 16K-64K single-needle 9/9 PASS | B4.3V |
| Long context | 96K-256K single-needle, 21/21 PASS combined w/ B4.3V | B4.3W |
| Combined long-context + concurrency | 7 arms, ~496K aggregate active tokens, 21/21 correct | B4.3X |
| Host stability soak | 4-cycle no-cooldown 128K×4, one boot ID, zero Xid/UVM | B4.3Z |
| Startup JIT inventory + prewarm design | 7 events cataloged, 6/7 covered by default prewarm | B4.4A |

Full evidence: `/home/bjk110/docker-build/deepseek-v4-0731-vllm027-b43{s,t,u,v,w,x,y,z}-*/` and
`/home/bjk110/docker-build/deepseek-v4-0731-vllm027-b44a-prewarm-promotion-readiness-20260815T172634KST/`
(host-local, not in this repository). Project-memory index: see
`dsv4_0731_vllm027_b44a_prewarm_promotion_ready_2026_08_15.md` and the linked chain in
`MEMORY.md` under this project's auto-memory.

## 3. Startup prewarm

`compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml` adds a `healthcheck` to `head` and a
one-shot `prewarm` service (head node only) that runs `scripts/prewarm_dsv4_v027_b43s.py` once
`head` is healthy, then exits 0/nonzero. Only the head/API side issues warmup requests; `worker`
does not duplicate them.

**Readiness semantics (deliberately unchanged):** `GET /health` continues to mean "engine alive",
exactly as before this candidate existed — it does **not** wait for prewarm. Prewarm completion is a
separate, observable signal: `docker compose ... ps prewarm` shows `Exited (0)` on success, and
`docker logs vllm-spark-prewarm` carries `STAGE_START`/`STAGE_END`/`prewarm COMPLETE|FAILED` markers.
A prewarm failure never produces a false success marker and never blocks `head` from serving — it
only means the first matching real request pays the JIT cost prewarm was meant to absorb. Treat a
failed/missing `prewarm` exit as an operational finding to investigate, not a silent no-op.

### Targets (default MAX_NUM_SEQS=1 profile: A + B only)

| Target | Trigger | Cold cost (B4.4A) | Post-prewarm | Rationale |
|---|---|---|---|---|
| A — short decode | ~1,000-token deterministic decode | ~32.0s | ~1.6s (17.6-20x) | DSpark batch-decode Triton kernels (4) + 2 TileLang fused-norm kernels not warmed by boot-time graph capture |
| B — non-aligned chunk | 61,440-token deterministic prefill | ~30.6s | ~30.1-30.9s (prefill-dominated; 0 new JIT lines is the coverage proof, not the timing) | `BuildPrefillChunkMetadataKernel.kernel`, the chunked-prefill remainder-shape metadata kernel |
| C — global top-k (optional, MS≥4 only) | c=4 concurrent ~1,000-token decodes | validated separately | 0 new JIT lines on repeat | `_compute_global_topk_indices_and_lens_kernel`; unreachable at MS=1, not part of the default sequence |

**Negative finding — do not shorten Target B casually.** An initial hypothesis that ~8,300 input
tokens would suffice to trigger the chunked-prefill remainder kernel was directly tested in B4.4A
and **disproved** — that depth triggered only Target A's kernels. 61,440 tokens (B4.3V's confirmed
non-8192-aligned trigger depth) is the validated minimum used here. A shorter Target B may silently
skip this kernel's warmup and reintroduce first-use latency on the first real long-context request.

FlashInfer's AutoTuner "outside tuning bucket range" fallback is intentionally **not** a prewarm
target — it is a performance-only, per-input-shape cache-miss fallback, not a JIT compile, and has
never affected correctness or CUDA graph capture in any B4.3-B4.4 task.

## 4. Optional MAX_NUM_SEQS=4 profile

`presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-ms4-optional-tp2.env` is byte-identical to the
default candidate preset except `MAX_NUM_SEQS=4`. It is **not** part of the default startup prewarm
(Target C must be run manually — see the preset's own header for the exact command) and is intended
for controlled, attended, higher-concurrency validation/benchmarking use only — not unattended
default operation. See §5, Residual Risk R1.

## 5. Residual risks (R1-R5, from B4.4A; none block promotion of the MS=1 default)

- **R1 — historical spark01 abrupt power loss.** Occurred once (B4.3X), during an immediate
  zero-cooldown repeat of a 128K×4 combined-pressure run under **MAX_NUM_SEQS=4**. Not reproduced
  across two dedicated follow-up tasks (B4.3Y controlled reproduction; B4.3Z four-cycle extended
  no-cooldown soak, ~4-5x the exposure). Root cause remains unknown. Does not block the MS=1 default;
  is the reason MAX_NUM_SEQS=4 stays an optional, attended-use-only profile (§4).
- **R2 — hardware watchdog unavailable.** Confirmed platform limitation (B4.3Y): the ARM SBSA
  watchdog exists but the kernel reports it permanently disabled (NMI not fully supported). A
  host-level hard hang or failure of any cause has no OS-level automatic recovery path on this
  platform — physical intervention is required. Relevant for anyone running this candidate
  unattended.
- **R3 — one-token empty-text anomaly.** Observed once (B4.3Z Run 3) under elevated no-cooldown soak
  load: one HTTP 200 response with a single empty-text completion token; immediate retry of the same
  arm passed cleanly; not reproduced since. Non-blocking, flagged for future characterization if it
  recurs.
- **R4 — startup lazy JIT.** Known behavior (7 cataloged events), now **mitigated** by the prewarm
  sequence in §3 for the default MS=1 profile (6/7 covered; the 7th is unreachable at MS=1).
- **R5 — FlashInfer AutoTuner "outside tuning bucket" fallback.** Performance-only, non-blocking, not
  practically prewarmable in general (per-shape cache miss). Documented for operational awareness.

## 6. Rollback

Reverting means stopping this route and restarting
`presets/deepseek-v4-flash-dspark-k7-64k-production-tp2.env` (primary rollback, vLLM 0.25.0, the
route this preset superseded) per [`docs/deepseek-v4-production.md`](deepseek-v4-production.md);
`presets/deepseek-v4-flash-mtp1-production-tp2.env` is the legacy second-tier fallback. Neither
rollback preset or image was modified by this promotion. As with every route switch on this
platform, GB10 UMA may not fully release on container stop — plan for a reboot before switching
routes (see §8 of that document).

## 7. Promotion history

- **B4.4B** — repository implementation + live validation
  (`DEEPSEEK_V4_0731_VLLM027_B4_4B_PROMOTED_PRESET_IMPLEMENTATION_VALIDATED`). Evidence:
  `/home/bjk110/docker-build/deepseek-v4-0731-vllm027-b44b-promoted-preset-*/`. Did not commit, push,
  or publish the image.
- **B4.4C** — final diff review, commit, push to `origin/main`, GHCR publication under
  `ghcr.io/bjk110/vllm-spark:v027-ngc2607-dsv4-0731-dspark-k7-256k-production`, and formal update of
  `docs/deepseek-v4-production.md`'s production pointer to this route
  (`DEEPSEEK_V4_0731_VLLM027_B4_4C_FORMALLY_PROMOTED_AND_PUBLISHED`). Evidence:
  `/home/bjk110/docker-build/deepseek-v4-0731-vllm027-b44c-formal-promotion-*/`.
