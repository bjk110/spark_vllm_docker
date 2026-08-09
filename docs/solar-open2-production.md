# Solar-Open2-250B Production Operations

Canonical operations document for the Solar-Open2-250B-Nota-NVFP4 serving path on the dual DGX
Spark (GB10, SM121) cluster. Two production-operable presets exist: the **active** r4 BF16
production preset and its **authoritative rollback** v0.22.1 preset. All other Solar-Open2 presets
are historical/experimental and are retained for provenance but are not production-operable.

## 1. Current active production — r4 BF16 (vLLM 0.25.1)

- **Preset:** `presets/solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env`
- **Status:** Active production since 2026-08-09 (spark01 head + spark02 worker, port 8000).
- **Model:** `nota-ai/Solar-Open2-250B-Nota-NVFP4`, pinned HF revision
  `de88f6226788077e2d340204fd79d37720c9eda0`, compressed-tensors NVFP4 (W4A4, group_size 16).
- **Image:** `vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rawg1-pread-b12xsw-r4-exp`, image ID
  (must match on BOTH nodes) `sha256:ecb7bfe3978a5241c5c304d52ce91e061e22b750178d21a4ef7788a08e86e774`.
  NGC 26.05 base; vLLM 0.25.1 @752a3a504485; FlashInfer 0.6.15; Upstage overlay 00907fc; raw-g1 KDA
  fix; ST_PREAD gate; B12X shared-workspace gate. No rebuild is required or expected for this
  promotion — the promoted preset launches the exact, already-built and already-validated image.
- **Runtime contract:** vLLM 0.25.1, TP=2, Ray backend, one rank per node, BF16 KV, fixed KV cache
  4 GiB/rank (`--kv-cache-memory-bytes 4294967296`), KV capacity 66,764 tokens; `MAX_MODEL_LEN=4096`;
  `MAX_NUM_SEQS=8`; `MAX_NUM_BATCHED_TOKENS=2048`; `GPU_MEMORY_UTILIZATION=0.80`; eager mode (no CUDA
  graphs); prefix caching enabled; chunked prefill enabled; attention backend auto-selects
  `FLASH_ATTN` (no pin); MoE backend `FLASHINFER_B12X`; shared B12X workspace enabled (3 scratch
  entries/rank, fallback 0); `pread` lazy safetensors loader enabled; RoCE 10.10.10.1 (head) /
  10.10.10.2 (worker) over `enp1s0f0np0`/`rocep1s0f0`.
- **Policy adapter:** port 8011, 4 served aliases (`solar-open2-250b`,
  `solar-open2-250b-bounded-low`, `solar-open2-250b-exact`, `solar-open2-250b-evidence-bound`).
- **Performance envelope:** c1 paired median ratio 0.9863 vs the v0.22.1 baseline (parity class, CI
  includes 1.0); c2 paired median ratio 0.9702 (approximately 1-3% below baseline, not full parity).
  Gate 4 matched-cell regression -0.8% to -6.0% (within the 10% acceptance threshold), TTFT improved
  in both matched c1 cells. The earlier single-instance 12-14% regression figure (AC arc, superseded)
  must not be cited as the current performance result.
- **Promotion basis:** readiness state `SOLAR_OPEN2_V0251_R4_BF16_PRODUCTION_READY`, evidence
  `solar-open2-v0251-r4-bf16-production-fasttrack-20260808T153704Z` (all six fast-track gates PASS —
  see section 7). Promotion evidence:
  `solar-open2-v0251-r4-bf16-production-promotion-20260809T053857Z`.

## 2. Production rollback — v0.22.1 KV4G matched baseline

- **Preset:** `presets/solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env`
- **Status:** Production rollback, currently stopped. Authoritative rollback for the active r4 route.
  Preset contents are immutable and must not be modified.
- **Model:** `nota-ai/Solar-Open2-250B-Nota-NVFP4` (same weights, same `MODEL_PATH`).
- **Image:** `vllm-spark:solar-open2-nvfp4-v022d568-vllm0221-upstage00907fc-ecfix-exp`, image ID
  `sha256:1873d2174691f67e16b5588fcef01680d21f1e7b42ac5587bd23d7503cae1366`, vLLM 0.22.1. Confirmed
  identical on both nodes (`docker images`, matching creation timestamp).
- **Runtime contract:** matched scheduler footprint to r4 (`MAX_MODEL_LEN=4096`, `MAX_NUM_SEQS=8`),
  BF16 KV, `HOST_PORT=8000`.
- **Rollback validated end-to-end** (Gate 6, 2026-08-08/09): stop r4 -> start rollback -> verify
  image/health -> 2 smoke requests (both HTTP 200) -> stop rollback -> restore r4 -> verify full
  state match against the pre-rollback r4 state. See section 5 for the exact procedure, including
  the reboot requirement discovered during this validation.
- **First-request warmup note:** the rollback baseline's first inference request after a fresh start
  may take up to several minutes due to one-time Triton JIT kernel compilation
  (`_zero_kv_blocks_kernel`, `_compute_slot_mapping_kernel` — logged explicitly by vLLM's own
  `jit_monitor` as an expected latency spike). This is a known v0.22.1 eager-mode characteristic, not
  a defect; subsequent requests complete normally (observed 1.5-26.6 s for a 16-token completion).

## 3. Supported presets

Two Solar-Open2 presets are production-operable:

- `presets/solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env` — active production.
- `presets/solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env` — production rollback.

The following presets are retained for provenance and must not be described as production targets:

- `presets/solar-open2-250b-nota-nvfp4-v0251-r4-active-test-tp2.env` — **historical.** The validated
  active-test baseline this production preset was promoted from (2026-07-27 activation, superseded
  as the production target on 2026-08-09; preserved unchanged, byte-identical runtime values to the
  production preset). Retained as validation provenance; do not delete or repurpose.
- `presets/solar-open2-250b-nota-nvfp4-v0251-r4-b12xsw-kv4g-exp-tp2.env`,
  `presets/solar-open2-250b-nota-nvfp4-v0251-r3-pread-kv4g-exp-tp2.env`,
  `presets/solar-open2-250b-nota-nvfp4-v0251-kv4g-diag-tp2.env`,
  `presets/solar-open2-250b-nota-nvfp4-v0251-kv4g-marlin-diag-tp2.env`,
  `presets/solar-open2-250b-nota-nvfp4-v0251-eager-4k-exp-tp2.env`,
  `presets/solar-open2-250b-nota-nvfp4-v022-marlin-c2diag-tp2.env`,
  `presets/solar-open2-250b-nota-nvfp4-eager-4k-exp-tp2.env` — **experimental/historical**
  intermediate presets from the r2/r3/r4 development arcs. Not production-operable.

## 4. Production status matrix

| Item | Status |
|---|---|
| r4 BF16 (vLLM 0.25.1) | **Active production** |
| v0.22.1 KV4G matched | **Production rollback** (stopped) |
| E4M3 KV calibration | **Experimental** — `artifact-generated-not-runtime-validated`, not part of production, no runtime test approved (see section 6) |
| r4 active-test preset | **Historical** — superseded as production target, preserved for provenance |
| other r2/r3/r4/v022 diagnostic presets | **Historical/experimental** — not production-operable |

## 5. Activation and rollback

Both presets launch through the established production launcher
(`/home/bjk110/docker-build/c2-instr/launch-at.sh` on spark01/spark02), which starts guards
(`fastguard3.py`, kernel-event watcher), sets `B12X_CACHE_DIR` from the path recorded in
`/home/bjk110/docker-build/CW_EV_PATH`, and brings up the Compose profile:

```bash
# Active production — head (spark01), then worker (spark02) once Ray port 6379 is listening:
/home/bjk110/docker-build/c2-instr/launch-at.sh head <evidence-dir>
/home/bjk110/docker-build/c2-instr/launch-at.sh worker <evidence-dir>
```

This resolves to (equivalent, direct form):

```bash
export B12X_CACHE_DIR=$(cat /home/bjk110/docker-build/CW_EV_PATH)/cache/$(hostname)
docker compose --env-file presets/solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env \
  -f docker-compose.yml -f docker-compose.solar-open2-hc-exp.yml -f docker-compose.pread-r3.yml \
  -f docker-compose.b12xsw-r4.yml -f docker-compose.b12x-cache.yml --profile head up -d   # spark01
# ... --profile worker up -d                                                              # spark02
```

Health validation: `GET http://192.168.0.200:8000/health` -> 200, and `GET
http://192.168.0.200:8011/v1/models` lists all 4 aliases. Cold start (no page cache, post-reboot) is
approximately 7 minutes (both TP ranks report "Model loading took 71.6 GiB memory"); wait for
`[entrypoint] All 2 nodes joined!` in the head container log and a stable 200 health response before
serving.

### Rollback procedure (validated 2026-08-08/09, reboot requirement is empirical, not precautionary)

**Observed during Gate 6 validation:** stopping the runtime container alone does *not* reliably
reclaim enough GB10 unified memory for the next large-model startup. This was reproduced twice in
one validation session:

- r4 -> v0.22.1 rollback: `docker compose down` alone left only ~25-31 GiB `MemAvailable` on each
  node; the v0.22.1 engine requires ~97.3 GiB free (`gpu_memory_utilization=0.80` of 121.63 GiB) and
  failed to start (`ValueError: Free memory on device cuda:0 ... is less than desired GPU memory
  utilization`).
- v0.22.1 -> r4 restoration: the same failure reproduced in the opposite direction (~33-40 GiB
  `MemAvailable`, same ~97.3 GiB requirement — r4 also passes `--gpu-memory-utilization 0.80`).
- In both cases, a **physical reboot** of both nodes restored `MemAvailable` to ~117 GiB and allowed
  the target baseline to start immediately.

**Do not treat an ordinary container stop/restart as a validated rollback procedure on this
platform.** The validated procedure is:

1. Stop the current runtime cleanly on both nodes:
   `docker compose --env-file <current-preset> -f ... --profile head down` (spark01), then
   `--profile worker down` (spark02).
2. Verify both containers are removed (`docker ps -a`) and Ray/vLLM processes are gone.
3. Reboot spark02 (worker) first.
4. Wait for spark02 SSH reachability, confirm a fresh `uptime -s` and `MemAvailable` near the node's
   full unified-memory capacity (~117-118 GiB) before proceeding.
5. Reboot spark01 (head).
6. Wait for spark01 SSH reachability and the same fresh-boot/`MemAvailable` verification.
7. Start the target baseline (rollback or production) via `launch-at.sh` (or `launch-sf.sh <role>
   <ev> baseline` for the v0.22.1 rollback specifically): head on spark01 first, wait for Ray port
   6379 to listen, then worker on spark02.
8. Verify Ray 2/2 by inspecting the head container log for `All 2 nodes joined!` (the in-container
   `ray status` CLI is not reliable — see section 6).
9. Verify `:8000/health` = 200.
10. Verify image ID on both nodes matches the target preset's documented image.
11. Verify model identity (`MODEL_PATH`/config.json hash) matches the target preset.
12. Verify KV capacity in the log (`GPU KV cache size: NN,NNN tokens`) matches the target's
    documented value.
13. Restore the policy adapter (`solar-policy-adapter` container) once the API is healthy.
14. Verify all 4 adapter aliases via `GET /v1/models` and adapter `:8011/health` = 200.

Preserve both images on both nodes; do not delete or prune either image.

## 6. Known operational caveats (non-blocking)

### Root filesystem pressure

Observed during validation: Ray raylet emits a recurring `file_system_monitor.cc:116: ... is over
95% full` warning on both nodes; host root filesystem was approximately 97-99% full
(`/dev/nvme0n1p2`) throughout the fast-track and promotion campaigns. This did not cause any request
failure and predates both campaigns. Production operations should maintain additional free space.
Recommended future (not performed as part of this promotion) targeted cleanup candidates: Ray
temp/log files, safe Docker build cache, obsolete disposable build artifacts, and explicitly
reviewed old evidence copies. Do not run an unqualified `docker system prune` — review targets
individually first.

### `ray status` CLI unreliable inside the container

Observed: `ray status` executed inside `vllm-spark-head` may segfault (exit code 139) even when the
underlying Ray cluster is healthy and both ranks have joined. Reproduced consistently during both
the fast-track and promotion campaigns. **Do not use `ray status` as the sole production health
check.** Combine, instead:

- API health (`GET :8000/health` = 200)
- expected Ray node participation, confirmed via the head container log
  (`[entrypoint] All 2 nodes joined!` / `Initializing a V1 LLM engine ... tensor_parallel_size=2`)
- port ownership (`8000`, `8011`, `6379` listening)
- a request smoke test (a real chat completion, not just `/health`)

### E4M3 KV calibration status

The E4M3 KV calibration work is a **separate, experimental** track:

- Status: `artifact-generated-not-runtime-validated`.
- Not part of the production baseline described in this document.
- No E4M3 runtime test has been approved.
- Production remains **BF16 KV** (see section 1).
- Do not mix E4M3 configuration values into the production preset.

## 7. Validation provenance

- **Production fast-track (2026-08-08/09)** — 6-gate campaign, evidence
  `solar-open2-v0251-r4-bf16-production-fasttrack-20260808T153704Z`:
  - Gate 1 — C2 targeted KDA correctness: PASS
    (`SOLAR_OPEN2_V0251_R4_BF16_C2_TARGETED_CORRECTNESS_PASS`).
  - Gate 2 — 24-prompt correctness suite: PASS 24/24.
  - Gate 3 — ~200-request mixed stability soak: PASS 200/200, 0 restarts, swap=0.
  - Gate 4 — six-case matched performance: PASS, -0.8% to -6.0% regression (within 10% threshold).
  - Gate 5 — dual-node cold start: PASS, full acceptance checklist, 4/4 smoke requests.
  - Gate 6 — rollback validation: PASS, full round-trip (r4 -> rollback -> r4) with exact state
    match, 3 physical reboots required across the round-trip (see section 5).
  - Final state: `SOLAR_OPEN2_V0251_R4_BF16_PRODUCTION_READY`.
- **Production promotion (2026-08-09)** — this document and the production preset, evidence
  `solar-open2-v0251-r4-bf16-production-promotion-20260809T053857Z`. Result:
  `SOLAR_OPEN2_V0251_R4_BF16_PRODUCTION_PROMOTED`. No image rebuild, no model change, zero functional
  difference between the promoted production preset and the validated active-test preset.
- Earlier validation arcs (r2/r3/r4 development, MI/AC/FIA-C multi-instance benchmarking, b12x
  shared-workspace, ST_PREAD): see `docs/solar-open2-st-pread.md` and
  `docs/solar-open2-b12x-shared-workspace.md` for detail.
