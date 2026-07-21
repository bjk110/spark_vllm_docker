# DS3 — Native DSpark k=7 64K Production Candidate

## 1. Status
- **Validated production candidate.** Native DSpark k=7, capped at `MAX_MODEL_LEN=65536` (64K).
- **Not currently active.** Explicit user approval is required before activation. The current MTP1
  production preset remains authoritative until then.
- Preset: `presets/deepseek-v4-ds3-native-dspark-k7-64k-production-candidate-tp2.env`.

## 2. Immutable image identity
- Manifest digest: `sha256:aacb06de60ecdc1bcafca5209aa5f0973eb86ab786212c988847ce53575ed84c`
- Config/image ID: `sha256:75bdf3d810558f1738927996f448056b196f83d4e09e55b23fffecfe904ead24`
- Platform: linux/arm64. vLLM 0.25.0. Runtime ref is digest-pinned (no mutable tag, no `latest`).

## 3. Runtime contract
- Native DSpark, speculative method `dspark`, `num_speculative_tokens=7`, greedy draft sampling.
- Target CUDA graph `FULL_DECODE_ONLY`, `cudagraph_capture_sizes=[8]`; draft path **eager**; draft
  CUDA graph **disabled**.
- KV cache fixed at 10 GiB, FP8 dtype. `MAX_NUM_SEQS=1`. Prefix caching disabled.
- `MAX_MODEL_LEN=65536`. `MAX_NUM_BATCHED_TOKENS=8192`.
- `VLLM_USE_DEEP_GEMM_E8M0=1` (mandatory SM121 numerical contract).
- TP=2, distributed backend `mp`, RoCE numeric-IP endpoints (10.10.10.1 / 10.10.10.2), one rank per node.

## 4. Validation provenance
### DS3U (corrected fixed-length E2E, candidate vs production)
- short E2E: 47.27 tok/s; LC32 E2E: 13.51 tok/s; LC64 E2E: 6.87 tok/s.
- Production-relative geometric mean: **1.188x**.
- 25-minute stability: PASS. Rollback rehearsal: PASS. Correctness + numerical validity: PASS.
### DS3V (isolated single-shot LC131, both routes)
- Candidate isolated LC131 single-shot: PASS (256 tokens, finish=length, sentinel correct).
- Repeated LC131: **NOT validated**.
- Candidate minimum MemAvailable during isolated LC131: **~17.07 GiB**.
- Post-container UVM: not fully released; **reboot required for full recovery**.

## 5. Why the 64K cap
- The candidate passed the corrected production gates from short through LC64 (multi-request).
- In DS3U, **repeated ~130K prefills caused a production host freeze** (cumulative GB10 UMA
  collapse faster than a 0.1 s guard). DS3V confirmed both routes retain large UVM after shutdown.
- The candidate has ~10 GiB less steady-state headroom than production (KV 10 GiB vs 4 GiB).
- A single isolated LC131 success (DS3V) does **not** prove repeated-request LC131 safety.
- Therefore default activation is capped at `65536`. The unrestricted `135168` maximum context is
  **not approved** for default activation and remains a gated/experimental configuration only.

## 6. Excluded configuration (do NOT enable in this default candidate)
- `cudagraph_capture_sizes=[7,8]` (the two-size draft-graph capture) — DS3R showed an lc32
  decode regression; excluded.
- Draft CUDA graph — disabled (draft stays eager).
- Unrestricted `MAX_MODEL_LEN=135168` — not approved for default activation.
- Concurrency / `MAX_NUM_SEQS>1` — not validated.
- Prefix caching — disabled.
- Higher `k` (>7) — not validated.
- Stochastic sampling — not validated (greedy only).

## 7. Rollback contract
- Authoritative rollback preset: `presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env`
  (production image `sha256:de69fa367137...`, config `sha256:5bb962a9055d...`, MTP n=1, capture
  `[2]`, KV 4 GiB FP8, `MAX_MODEL_LEN=135168`, E8M0 disabled).
- The production image remains present on both Spark nodes.
- Any activation MUST preserve the current production preset and image.
- Any activation MUST include a rollback rehearsal.
- A failed activation MUST return to the intentionally-stopped state.

## 8. Activation prerequisites (all required; none performed here)
- Explicit user approval.
- Clean Git state on all three hosts at the intended commit.
- Candidate + production image parity across both nodes.
- Candidate + production model parity (48 / 46 safetensors).
- RoCE / RDMA health; ~118 GiB memory baseline after a fresh reboot.
- A disposable activation canary (isolated port) validated before any default routing.
- OpenWebUI and Traefik unchanged until direct API validation passes.
- Final commit and push of this tracked package completed first, if separately approved.

This candidate does **not** claim unrestricted `135168` production readiness, and does **not**
claim repeated-request LC131 safety. Those require a separate repeated-request safety investigation.
