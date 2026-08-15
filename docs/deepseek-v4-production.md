# DeepSeek-V4 Production Operations

Canonical operations document for the DeepSeek-V4-Flash serving path on the dual DGX Spark (GB10,
SM121) cluster. Three DeepSeek-V4 presets are retained in `presets/`: the **active** production
preset (v0.27, native DSpark k=7, 256K), its **primary rollback** (v0.25.0, native DSpark k=7, 64K —
itself production-proven since 2026-07-22), and a **legacy rollback** (MTP1). All other DeepSeek-V4
presets are superseded and available only through Git history.

## 1. Current active production — vLLM 0.27, native DSpark k=7 (256K)

- **Preset:** `presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env`
- **Status:** Active production since 2026-08-15 (promotion task B4.4C). Promoted from
  `B4_4A_PREWARM_VALIDATED_PROMOTION_READY` / `B4_4B_PROMOTED_PRESET_IMPLEMENTATION_VALIDATED`.
- **Model:** `deepseek-ai/DeepSeek-V4-Flash-0731`.
- **Image:** `ghcr.io/bjk110/vllm-spark:v027-ngc2607-dsv4-0731-dspark-k7-256k-production`
  (local build ID `sha256:a7f0f4b8a508c0b2510fc7e4dcb916491efa03c380c9c7b84dddd4c16ad6f38d`, identical
  on spark01/spark02, unchanged since task B4.3S; NGC 26.07 base, vLLM 0.27, arm64). Immutable
  registry digest recorded once published — see [`docs/images.md`](images.md).
- **Runtime contract:** native DSpark `method=dspark`, `num_speculative_tokens=7`, greedy draft;
  target `FULL_DECODE_ONLY`, `cudagraph_capture_sizes=[8]`, 2 warmups; `MAX_MODEL_LEN=262144`;
  `MAX_NUM_BATCHED_TOKENS=8192`; `MAX_NUM_SEQS=1`; fixed **10 GiB FP8** KV cache; MoE backend MARLIN;
  `VLLM_USE_DEEP_GEMM_E8M0=1` (mandatory SM121 numerical contract); prefix caching disabled; TP=2,
  `DISTRIBUTED_BACKEND=mp`, RoCE (10.10.10.1 / 10.10.10.2).
- **Automatic startup prewarm:** launch together with
  `compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml`, which adds a `head` healthcheck and a
  one-shot `prewarm` service (`scripts/prewarm_dsv4_v027_b43s.py`) that runs automatically once the
  engine is healthy — a ~1K-token decode plus a 61,440-token non-8192-aligned prefill, covering 6 of 7
  cataloged first-use JIT kernels for this profile. `/health` continues to mean "engine alive", not
  "prewarm complete" — check `docker ps` for the `prewarm` container's exit code separately. Do not
  shorten the 61,440-token target; a shorter ~8,300-token depth was tested and found insufficient
  (task B4.4A).
- **Operational limits:** 256K context validated (single-needle + ~496K-token combined-pressure
  correctness, four-cycle no-cooldown 128K×4 soak — task B4.3W/X/Z). `MAX_NUM_SEQS=1` is the
  production default; a separate `MAX_NUM_SEQS=4` profile
  (`presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-ms4-optional-tp2.env`) is validated but is an
  **optional, attended-use-only profile — not production, not default** (see §7, Residual Risk R1).
- **Full candidate documentation:** [`docs/deepseek-v4-v027-b43s-promotion-candidate.md`](deepseek-v4-v027-b43s-promotion-candidate.md).

## 2. Primary rollback — vLLM 0.25.0, native DSpark k=7 (64K)

- **Preset:** `presets/deepseek-v4-flash-dspark-k7-64k-production-tp2.env`
- **Status:** Primary rollback (stopped). This was the active production route from 2026-07-22 until
  2026-08-15; it has the longest in-production track record of any DeepSeek-V4 route in this
  repository and is the preferred fallback if the v0.27/256K route needs to be reverted.
- **Model:** `deepseek-ai/DeepSeek-V4-Flash-DSpark`.
- **Image (immutable, digest-pinned):**
  `ghcr.io/bjk110/vllm-spark@sha256:aacb06de60ecdc1bcafca5209aa5f0973eb86ab786212c988847ce53575ed84c`
  (config `sha256:75bdf3d810558f1738927996f448056b196f83d4e09e55b23fffecfe904ead24`), vLLM 0.25.0, arm64.
- **Runtime contract:** native DSpark `method=dspark`, `num_speculative_tokens=7`, greedy draft;
  target `FULL_DECODE_ONLY`, `cudagraph_capture_sizes=[8]`; draft path **eager** (draft CUDA graph not
  captured); `MAX_MODEL_LEN=65536`; `MAX_NUM_BATCHED_TOKENS=8192`; `MAX_NUM_SEQS=1`; fixed **10 GiB FP8**
  KV cache (201,624 tokens); `VLLM_USE_DEEP_GEMM_E8M0=1` (mandatory SM121 numerical contract); prefix
  caching disabled; TP=2, `DISTRIBUTED_BACKEND=mp`, RoCE (10.10.10.1 / 10.10.10.2).
- **Operational limits:** 64K context only. No LC131 exposure. Unrestricted `135168` is **not
  supported**. Repeated large-context (≥131K) operation is unvalidated. Single sequence only
  (`MAX_NUM_SEQS=1`); no concurrency.

## 3. Legacy rollback — MTP1

- **Preset:** `presets/deepseek-v4-flash-mtp1-production-tp2.env`
- **Status:** Legacy rollback, currently stopped. Predates the native-DSpark lineage; retained as a
  second-tier fallback if §2's route is also unavailable.
- **Model:** `deepseek-ai/DeepSeek-V4-Flash`.
- **Image (immutable, digest-pinned):**
  `ghcr.io/bjk110/vllm-spark@sha256:de69fa367137c3c77df07c3dfff784dc0b0caec8d2c8c43cf2ad63608381ee4a`
  (config `sha256:5bb962a9055d3931b34c18af78ef962368c8edc10cd4dd84382d69e6599d3991`),
  vLLM `0.24.0.dev0+dsv4.pr41834.72261a7`.
- **Runtime contract:** MTP n=1; target `FULL_DECODE_ONLY`, `cudagraph_capture_sizes=[2]`; fixed
  **4 GiB FP8** KV cache; `MAX_NUM_SEQS=1`; prefix caching disabled; MARLIN MoE + FP8 Lightning Indexer;
  TP=2 mp/RoCE.
- **Limitation:** repeated large-context operation is not approved.

## 4. Supported presets

Three DeepSeek-V4 presets are retained:

- `presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env` — active production.
- `presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-ms4-optional-tp2.env` — optional
  higher-concurrency profile for the active route, **not** production/default.
- `presets/deepseek-v4-flash-dspark-k7-64k-production-tp2.env` — primary rollback.
- `presets/deepseek-v4-flash-mtp1-production-tp2.env` — legacy rollback.

Superseded DeepSeek-V4 presets (intermediate, experimental, candidate, legacy, deep-recovery,
historical reproduction) are available through Git history and are not retained in `presets/`.

## 5. Production status matrix

| Item | Status |
|---|---|
| vLLM 0.27, native DSpark k=7, 256K, MS=1 | **Active production** |
| MS=4 (same v0.27 image) | Optional validated profile — not default, not production |
| vLLM 0.25.0, native DSpark k=7, 64K | **Primary rollback** (stopped) |
| MTP1 | **Legacy rollback** (stopped) |
| 256K single-needle / ~496K combined-pressure correctness | Validated (B4.3W/X) |
| Four-cycle no-cooldown 128K×4 soak | Validated (B4.3Z) |
| Automatic startup prewarm (Targets A+B) | Validated (B4.4A/B4.4B), wired via the candidate Compose overlay |

## 6. Activation and rollback

Active production launches through the committed Compose project plus the candidate overlay
(network host, local-then-registry-pinned image):

```bash
# Active production (spark01 head + spark02 worker):
docker compose --env-file presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env \
  -f docker-compose.yml -f compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml \
  --profile head up -d
# on the worker node:
docker compose --env-file presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env \
  -f docker-compose.yml -f compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml \
  --profile worker up -d
```

Health validation: `GET http://127.0.0.1:8000/health` → 200, `GET /v1/models` lists
`deepseek-ai/DeepSeek-V4-Flash-0731` with `max_model_len: 262144`, and the `prewarm` container exits
0 (`docker ps -a --filter name=vllm-spark-prewarm`). Cold start is ~8-11 minutes; wait for a stable
health window plus prewarm completion before serving load-sensitive traffic.

Rollback (to §2, primary): stop the v0.27 containers, reboot both nodes if UVM remains pinned (see
§8), then launch `presets/deepseek-v4-flash-dspark-k7-64k-production-tp2.env` with the plain
`docker-compose.yml` (no candidate overlay needed) on port 8000 and re-validate health + a short
correctness check. If §2 is also unavailable, §3 (MTP1) is the second-tier fallback with the same
procedure. Preserve all rollback images on both nodes; do not delete or prune them.

## 7. Residual risks (carried from B4.4A/B4.4B; none block this promotion)

- **R1 — historical spark01 abrupt host power loss.** Occurred once (task B4.3X) during an immediate
  zero-cooldown repeat of a 128K×4 combined-pressure run under **MAX_NUM_SEQS=4**. Not reproduced
  across two dedicated follow-up tasks (B4.3Y, B4.3Z). Root cause unknown. This is why MAX_NUM_SEQS=4
  remains an optional, attended-use-only profile rather than the production default.
- **R2 — hardware watchdog unavailable.** Confirmed platform limitation: this platform's ARM SBSA
  watchdog is permanently disabled at the kernel level (NMI not fully supported). A host-level hard
  hang has no OS-level automatic recovery path — physical intervention is required.
- **R3 — one-token empty-text anomaly.** Observed once (task B4.3Z) under elevated no-cooldown soak
  load; did not reproduce on immediate retry. Non-blocking.
- **R4 — startup lazy JIT.** Mitigated for the MS=1 default by the automatic prewarm described in §1.
- **R5 — FlashInfer AutoTuner "outside tuning bucket" fallback.** Performance-only, non-blocking, not
  practically prewarmable in general.

## 8. Recovery note (GB10 UMA)

GB10 unified memory: after a vLLM container exits, large UVM (driver) allocations are **not fully
released** — host `MemAvailable` recovers only partially, and **full recovery may require a reboot**.
Plan any route switch on a fresh boot (~118 GiB `MemAvailable`).

## 9. Validation provenance (final milestones)

**v0.27/256K route (current active production):** B4.3S (first TP=2+DSpark k=7 success on v0.27) →
B4.3T (performance/acceptance baseline) → B4.3U (concurrency to c=8) → B4.3V/W (long-context to 256K)
→ B4.3X/Y/Z (combined-pressure soak, host-stability investigation) → B4.4A (JIT inventory + prewarm
design, promotion-readiness audit) → B4.4B (repository implementation, live-validated) → B4.4C (this
promotion). Full chain: [`docs/deepseek-v4-v027-b43s-promotion-candidate.md`](deepseek-v4-v027-b43s-promotion-candidate.md)
and this project's memory index.

**v0.25.0/64K route (primary rollback), historical:**

- **DS3U** — corrected fixed-length E2E comparison (short / LC32 / LC64) of the DSpark candidate vs
  the MTP1 production route; three-stage geometric mean 1.188× production; stability + rollback
  rehearsal PASS.
- **DS3V** — isolated single-shot LC131 characterization on both routes; single-shot safe (candidate
  min `MemAvailable` ~17.07 GiB), **repeated LC131 not validated**; reboot-only UVM recovery confirmed.
  Basis for the 64K cap.
- **DS3X** — isolated-port (18000) activation canary: startup + graph `[8]` + correctness 4/4 +
  short/LC32/LC64 + 25-minute stability + teardown/recovery + MTP1 rollback rehearsal PASS.
- **DS3Y** — default production activation on port 8000 (zero-config: OpenWebUI already targeted
  spark01:8000, Traefik fronts only OpenWebUI); direct-API + 30-minute active-production observation
  PASS; `DS3Y_64K_DEFAULT_PRODUCTION_ACTIVE`.
