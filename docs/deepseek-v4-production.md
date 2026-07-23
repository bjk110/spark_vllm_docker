# DeepSeek-V4 Production Operations

Canonical operations document for the DeepSeek-V4-Flash serving path on the dual DGX Spark (GB10,
SM121) cluster. Two production-operable presets exist: the **active** native DSpark 64K production
preset and its **authoritative rollback** MTP1 preset. All other DeepSeek-V4 presets are superseded
and available only through Git history.

## 1. Current active production — native DSpark k=7 (64K)

- **Preset:** `presets/deepseek-v4-flash-dspark-k7-64k-production-tp2.env`
- **Status:** Active production since 2026-07-22 (spark01, port 8000).
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
  supported** for default production. Repeated large-context (≥131K) operation is unvalidated. Single
  sequence only (`MAX_NUM_SEQS=1`); no concurrency.

## 2. Production rollback — MTP1

- **Preset:** `presets/deepseek-v4-flash-mtp1-production-tp2.env`
- **Status:** Production rollback, currently stopped. Authoritative rollback for the active DSpark route.
- **Model:** `deepseek-ai/DeepSeek-V4-Flash`.
- **Image (immutable, digest-pinned):**
  `ghcr.io/bjk110/vllm-spark@sha256:de69fa367137c3c77df07c3dfff784dc0b0caec8d2c8c43cf2ad63608381ee4a`
  (config `sha256:5bb962a9055d3931b34c18af78ef962368c8edc10cd4dd84382d69e6599d3991`),
  vLLM `0.24.0.dev0+dsv4.pr41834.72261a7`.
- **Runtime contract:** MTP n=1; target `FULL_DECODE_ONLY`, `cudagraph_capture_sizes=[2]`; fixed
  **4 GiB FP8** KV cache; `MAX_NUM_SEQS=1`; prefix caching disabled; MARLIN MoE + FP8 Lightning Indexer;
  TP=2 mp/RoCE.
- **Limitation:** repeated large-context operation is not approved.

## 3. Supported presets

Exactly two DeepSeek-V4 presets are retained:

- `presets/deepseek-v4-flash-dspark-k7-64k-production-tp2.env` — active production.
- `presets/deepseek-v4-flash-mtp1-production-tp2.env` — production rollback.

Superseded DeepSeek-V4 presets (intermediate, experimental, candidate, legacy, deep-recovery,
historical reproduction) are available through Git history and are not retained in `presets/`.

## 4. Production status matrix

| Item | Status |
|---|---|
| native DSpark k=7 64K | **Active production** |
| MTP1 | **Production rollback** (stopped) |
| DSpark 131K / unrestricted 135168 | **Not supported** for default production |
| repeated LC131 | **Unvalidated** |
| intermediate / experimental presets | Removed from the active preset surface (Git history only) |

## 5. Activation and rollback

Both presets launch through the committed Compose project (network host, digest-pinned image):

```bash
# Active production (spark01 head + spark02 worker):
docker compose --env-file presets/deepseek-v4-flash-dspark-k7-64k-production-tp2.env --profile head up -d
# on the worker node:
docker compose --env-file presets/deepseek-v4-flash-dspark-k7-64k-production-tp2.env --profile worker up -d
```

Health validation: `GET http://127.0.0.1:8000/health` → 200, and `GET /v1/models` lists
`deepseek-ai/DeepSeek-V4-Flash-DSpark`. Cold start is ~8 minutes; wait for "Application startup
complete" and a 60 s stable-health window before serving.

Rollback: stop the DSpark containers, reboot both nodes if UVM remains pinned (see §6), then launch
the MTP1 rollback preset on port 8000 with the same `--profile head` / `--profile worker` commands and
re-validate health + a short correctness check. Preserve both images on both nodes; do not delete or
prune the rollback image.

## 6. Recovery note (GB10 UMA)

GB10 unified memory: after a vLLM container exits, large UVM (driver) allocations are **not fully
released** — host `MemAvailable` recovers only partially, and **full recovery may require a reboot**.
Plan any route switch on a fresh boot (~118 GiB `MemAvailable`).

## 7. Validation provenance (final milestones)

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
