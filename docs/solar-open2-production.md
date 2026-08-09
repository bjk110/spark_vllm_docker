# Solar-Open2-250B Production Operations

Canonical operations document for the Solar-Open2-250B-Nota-NVFP4 serving path on the dual DGX
Spark (GB10, SM121) cluster. Two production-operable presets exist: the **active** r4 BF16
production preset and its **authoritative rollback** v0.22.1 preset. All other Solar-Open2 presets
are historical/experimental, are not tracked in the active preset directory, and are not
production-operable — see section 3 for the retention policy.

## 1. Current active production — r4 BF16 (vLLM 0.25.1)

- **Preset:** `presets/solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env` (tracked since
  commit `d966925fa4c6e5b270c37047b8c8ea000a57c9a9`; comment-only corrections from the 2026-08-09
  hygiene pass are pending in the working tree, not yet committed — zero functional/runtime diff,
  see section 7).
- **Status:** Promoted to production 2026-08-09 (spark01 head + spark02 worker, port 8000).
- **Model:** `nota-ai/Solar-Open2-250B-Nota-NVFP4`, pinned HF revision
  `de88f6226788077e2d340204fd79d37720c9eda0`, compressed-tensors NVFP4 (W4A4, group_size 16).
- **Image:** `vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rawg1-pread-b12xsw-r4-exp`, local
  Docker image ID (must match on BOTH nodes) `sha256:ecb7bfe3978a5241c5c304d52ce91e061e22b750178d21a4ef7788a08e86e774`.
  This is a **local build image ID**, not a registry digest — the image has not been published to
  GHCR (see `docs/images.md` for the distinction). NGC 26.05 base; vLLM 0.25.1 @752a3a504485;
  FlashInfer 0.6.15; Upstage overlay 00907fc; raw-g1 KDA fix; ST_PREAD gate; B12X shared-workspace
  gate — see `PATCH_STATUS.md` for per-gate source-patch provenance. No rebuild is required or
  expected for this promotion; this and all later hygiene passes launch the exact, already-built and
  already-validated image.
- **Runtime contract:** vLLM 0.25.1, TP=2, Ray backend, one rank per node, BF16 KV, fixed KV cache
  4 GiB/rank (`--kv-cache-memory-bytes 4294967296`), KV capacity 66,764 tokens; `MAX_MODEL_LEN=4096`;
  `MAX_NUM_SEQS=8`; `MAX_NUM_BATCHED_TOKENS=2048`; `GPU_MEMORY_UTILIZATION=0.80`; eager mode (no CUDA
  graphs); prefix caching enabled; chunked prefill enabled; attention backend auto-selects
  `FLASH_ATTN` (no pin); MoE backend `FLASHINFER_B12X`; shared B12X workspace enabled (3 scratch
  entries/rank, fallback 0); `pread` lazy safetensors loader enabled; RoCE 10.10.10.1 (head) /
  10.10.10.2 (worker) over `enp1s0f0np0`/`rocep1s0f0`.
- **Policy adapter:** port 8011, 4 served aliases (`solar-open2-250b`,
  `solar-open2-250b-bounded-low`, `solar-open2-250b-exact`, `solar-open2-250b-evidence-bound`).
- **Compose overlays (working-tree-ready, not yet staged/committed):** `docker-compose.yml`
  (already tracked) + `docker-compose.solar-open2-hc-exp.yml` + `docker-compose.pread-r3.yml` +
  `docker-compose.b12xsw-r4.yml` + `docker-compose.b12x-cache.yml` (all four Solar-specific overlays
  verified and hash-recorded during the 2026-08-09 hygiene pass, but still untracked pending a
  separate, explicitly authorized commit). All four are present at repository root, matching the
  exact paths used by the validated launch invocation (see section 5). Non-disruptively re-resolved
  with `docker compose ... config` during the hygiene pass — merge is clean, no errors, all runtime
  values consistent with the values above.
- **Performance envelope:** c1 paired median ratio 0.9863 vs the v0.22.1 baseline (parity class, CI
  includes 1.0); c2 paired median ratio 0.9702 (approximately 1-3% below baseline, not full parity).
  Gate 4 matched-cell regression -0.8% to -6.0% (within the 10% acceptance threshold), TTFT improved
  in both matched c1 cells. The earlier single-instance 12-14% regression figure (AC arc, superseded)
  must not be cited as the current performance result.
- **Promotion basis:** readiness state `SOLAR_OPEN2_V0251_R4_BF16_PRODUCTION_READY`, evidence
  `solar-open2-v0251-r4-bf16-production-fasttrack-20260808T153704Z` (all six fast-track gates PASS —
  see section 7). Promotion commit `d966925fa4c6e5b270c37047b8c8ea000a57c9a9`. Promotion evidence:
  `solar-open2-v0251-r4-bf16-production-promotion-20260809T053857Z` (local, `/home/bjk110/docker-build/`).

## 2. Production rollback — v0.22.1 KV4G matched baseline

- **Preset:** `presets/solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env` (working-tree-ready,
  not yet staged/committed — hash-verified against the validated rollback evidence during the
  2026-08-09 hygiene pass, sha256
  `9ff168273617f749e289934a138aeef46fae406cb592f4c7cc7fef46d0977b33`, exact match). Contents are
  immutable and must not be modified.
- **Status:** Production rollback, currently stopped. Authoritative rollback for the active r4
  route.
- **Model:** `nota-ai/Solar-Open2-250B-Nota-NVFP4` (same weights, same `MODEL_PATH`).
- **Image:** `vllm-spark:solar-open2-nvfp4-v022d568-vllm0221-upstage00907fc-ecfix-exp`, local Docker
  image ID `sha256:1873d2174691f67e16b5588fcef01680d21f1e7b42ac5587bd23d7503cae1366`, vLLM 0.22.1.
  Confirmed identical on both nodes (`docker images`, matching creation timestamp). Local build image
  ID only, not published to GHCR.
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

## 3. Preset retention policy and status

Two Solar-Open2 presets are the only production-operable presets for this family:

- `presets/solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env` — active production. Tracked
  since commit `d966925fa4c6e5b270c37047b8c8ea000a57c9a9`.
- `presets/solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env` — production rollback.
  Working-tree-ready and hash-verified as of the 2026-08-09 hygiene pass, but not yet staged or
  committed — see the "recommended commit boundary" in that pass's report for the follow-up
  action.

This mirrors the DeepSeek-V4 retention policy exactly (see `presets/README.md`): only the active
production preset and its authoritative rollback are retained as production-operable presets, with
everything else excluded from the active preset directory. Historical/intermediate development
presets (r2/r3/r4 diagnostic variants, eager-4k smoke configs, marlin c2 diagnostics) are **not**
tracked and are **not** production-operable. They
remain as local, untracked, validated-in-place artifacts on the build hosts for reproducibility
provenance only — referenced below by content hash, not by claiming repository retention:

| Local artifact (untracked, not a repository path) | SHA-256 | Role |
|---|---|---|
| `solar-open2-250b-nota-nvfp4-v0251-r4-active-test-tp2.env` | `2c8b0b2c46c9633d37a4500db335db449860920c216904507d859a027d1a9125` | The validated active-test baseline the production preset was promoted from (2026-07-27 activation through 2026-08-09). Runtime values are byte-identical (0 functional diff) to the current production preset. |
| `solar-open2-250b-nota-nvfp4-v0251-r4-b12xsw-kv4g-exp-tp2.env` | `d8285a4162e5ec2529a729ee0362f8e8fddfe73514a2f32337241dd89fa800fe` | Earlier r4 development preset; the active-test preset's runtime values are byte-identical to this one. |
| `solar-open2-250b-nota-nvfp4-v0251-r4-b12xsw-kv4g-no-template-logits-exp-tp2.env` | (not re-verified this pass) | r4 development variant, superseded. |
| `solar-open2-250b-nota-nvfp4-v0251-r3-pread-kv4g-exp-tp2.env` | (not re-verified this pass) | r3 development preset (pre-B12X-shared-workspace). |
| `solar-open2-250b-nota-nvfp4-v0251-kv4g-diag-tp2.env`, `-marlin-diag-tp2.env`, `-eager-4k-exp-tp2.env` | (not re-verified this pass) | Early v0.25.1 diagnostic/smoke presets. |
| `solar-open2-250b-nota-nvfp4-v022-marlin-c2diag-tp2.env`, `-eager-4k-exp-tp2.env` | (not re-verified this pass) | v0.22.1-lineage diagnostic presets, predate the rollback baseline's `di-matched` variant. |

None of the presets in this table exists in the Git repository. If exact byte-for-byte
reproduction of one of these intermediate steps is ever required, it must be sourced from the
original build hosts (spark01/spark02, `presets/` directory, untracked) — not from `git log` or
`origin/main`. This is a **local-only reproducibility limitation** for the historical/intermediate
development path only — the production preset is tracked and the rollback preset is
working-tree-ready and hash-verified (see section 1/2 above), neither depends on this table.

## 4. Production status matrix

| Item | Status |
|---|---|
| r4 BF16 (vLLM 0.25.1) | **Active production** |
| v0.22.1 KV4G matched | **Production rollback** (stopped) |
| E4M3 KV calibration | **Experimental** — `artifact-generated-not-runtime-validated`, not part of production, no runtime test approved (see section 6) |
| r4 active-test preset and all r2/r3/r4/v022 intermediate presets | **Historical, local-only** — not tracked, not production-operable, referenced above by hash for provenance only |

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

This resolves to (equivalent, direct form, using only repository files — one tracked, four
working-tree-ready pending a future commit — plus one external environment value; see the
reproducibility-limitation note below):

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

### Reproducibility limitation: launcher orchestration remains local-only

The production preset is tracked (commit `d966925`). The four Solar-specific Compose files and the
rollback preset are working-tree-ready and hash-verified as of the 2026-08-09 hygiene pass but not
yet staged or committed; once committed, all five plus the production preset will be reproducible
from a fresh clone. The **launcher orchestration layer** (`launch-at.sh`, `fastguard3.py`, the
kernel-event watcher, and the `B12X_CACHE_DIR` / `CW_EV_PATH` cache-warm wiring) is **not** tracked,
is **not** working-tree-ready in this repository (it lives entirely outside the repository, under
`/home/bjk110/docker-build/`), and is **not** promoted in this pass, for a specific reason rather
than an oversight:

- `launch-at.sh` and `fastguard3.py` are small, promotable scripts in isolation, but the launcher's
  `B12X_CACHE_DIR` resolves through `CW_EV_PATH` to
  `/home/bjk110/docker-build/solar-open2-v0251-b12x-cold-warm-20260726T013344Z/cache/<hostname>` — a
  **timestamped diagnostic evidence directory** from the 2026-07-26 b12x cold/warm cache experiment,
  containing per-node FlashInfer/Triton/CUDA-driver JIT compilation caches (binary, large,
  non-portable, GB10-arch/build-specific).
- Tracking the launcher as-is would silently embed a dependency on that one evidence directory
  continuing to exist at that exact path — not reproducible from a fresh clone.
- Changing the launcher to point at a new, stable, git-tracked-adjacent cache location would fix
  reproducibility but would refactor the validated launch behavior (a cold cache changes startup
  timing) — explicitly out of scope for this hygiene pass, which must preserve exact validated
  behavior.
- The overlay also mounts `/home/bjk110/docker-build/c2-obs` (an observability/telemetry sink) —
  also local-only, also not required for correctness, only for diagnostics.

**Consequence:** once the working-tree-ready assets above are committed, a fresh clone of this
repository will be able to reproduce the exact configuration (image reference, both presets, all
Compose overlays) but will not be able to reproduce a *warm-cache* launch without also
recreating (or accepting a cold, slower first start from) the `B12X_CACHE_DIR` target directory
structure (`dotcache/`, `triton/`, `nv/`, `cutedsl/` subdirectories — see
`docker-compose.b12x-cache.yml`'s header comment for the exact contract) and the guard/observability
scripts under `/home/bjk110/docker-build/c2-instr/`. This is recorded as a known limitation, not a
defect: a cold-cache launch through the Compose files alone still starts and serves
correctly (confirmed in Gate 5 — the b12x-cache overlay is only a JIT-cache warm-start
optimization, not a correctness dependency), it will simply pay the same JIT-compilation warmup cost
documented for the rollback baseline in section 2.

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
Recommended future (not performed as part of promotion or this hygiene pass) targeted cleanup
candidates: Ray temp/log files, safe Docker build cache, obsolete disposable build artifacts, and
explicitly reviewed old evidence copies. Do not run an unqualified `docker system prune` — review
targets individually first.

### `ray status` CLI unreliable inside the container

Observed: `ray status` executed inside `vllm-spark-head` may segfault (exit code 139) even when the
underlying Ray cluster is healthy and both ranks have joined. Reproduced consistently during the
fast-track campaign. **Do not use `ray status` as the sole production health check.** Combine,
instead:

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

### ST_PREAD and B12X shared-workspace gates are now production-active

Two vLLM environment gates that were introduced and validated as *experimental, not promoted*
(`VLLM_SPARK_ST_PREAD`, `VLLM_SPARK_B12X_SHARED_WORKSPACE`) are set (`=1`) in the production preset
as of this promotion — they are load-time/memory optimizations, not correctness changes, and were
exercised throughout the entire six-gate fast-track. Their original experimental deep-dive
write-ups predate the promotion and still carry an "EXPERIMENTAL, not promoted" banner; that banner
is now stale for the production context and is not linked from this document to avoid the
contradiction. Measured effect, for reference:

- **ST_PREAD** (lazy safetensors `pread` loader): loader-phase swap reduced from 7.33 GB / 3.12 GB
  to near-zero on spark01/spark02; peak `Active_file` 37.3 GiB -> 1.87 GiB; TP0 load 450 s -> ~176 s,
  TP1 255 s -> ~141 s (2026-07-25 diagnostics).
- **B12X shared workspace**: torch allocated at engine-ready 103.5 -> 76.2 GiB per rank; spark01
  warmup swap 6.9-7.5 GB -> ~0; deterministic output bit-identical to the unshared path; decode
  throughput unchanged (2026-07-26 overlay validation).

Full technical detail (contract, fail-fast behavior, measurement methodology) remains in the
untracked local files `docs/solar-open2-st-pread.md` and `docs/solar-open2-b12x-shared-workspace.md`
on the build hosts — not linked here because their status banners have not been updated to reflect
production status and would otherwise contradict this document.

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
- **Production promotion (2026-08-09)** — commit `d966925fa4c6e5b270c37047b8c8ea000a57c9a9`
  (`docs/README.md`, this document, and the production preset), evidence
  `solar-open2-v0251-r4-bf16-production-promotion-20260809T053857Z`. Result:
  `SOLAR_OPEN2_V0251_R4_BF16_PRODUCTION_PROMOTED`, then pushed to `origin/main`
  (`SOLAR_OPEN2_V0251_R4_BF16_PRODUCTION_PROMOTION_PUSHED`). No image rebuild, no model change, zero
  functional difference between the promoted production preset and the validated active-test preset.
- **Repository hygiene and reproducibility pass (2026-08-09)** — verified and left
  working-tree-ready (not staged, not committed) the rollback preset, four Solar-specific Compose
  overlays, and the four Solar patch-provenance files (`patches/solar/`); made comment-only
  corrections to the production preset (zero functional diff); corrected this document's
  file-retention claims; documented the launcher-orchestration reproducibility limitation (see
  section 5); updated `presets/README.md`, root `README.md`, `docs/README.md`, `docs/images.md`,
  `docs/software-stack.md`, `PATCH_STATUS.md`, and `CHANGELOG.md` for consistency. No runtime
  change; no image rebuild; no additional validation performed; nothing staged, committed, or
  pushed — see that pass's report for the recommended commit boundary.
- Earlier validation arcs (r2/r3/r4 development, MI/AC/FIA-C multi-instance benchmarking, b12x
  shared-workspace, ST_PREAD) are recorded by hash in section 3; their original detailed local
  documents (`docs/solar-open2-st-pread.md`, `docs/solar-open2-b12x-shared-workspace.md`) remain
  untracked build-host artifacts (see the note in section 6).
