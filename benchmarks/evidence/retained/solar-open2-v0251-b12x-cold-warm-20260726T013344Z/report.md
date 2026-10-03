# Solar-Open2 v0.25.1 r3 — b12x cold vs warm cache diagnostic (Phase-2)
Arc: solar-open2-v0251-b12x-cold-warm-20260726T013344Z

## FINAL STATE: SOLAR_OPEN2_V0251_B12X_PHASE2_MIXED_CAUSE

Warm cache RESOLVES the init/first-request MemAvailable spikes and most of
the latency cost (JIT/compile component), but DOES NOT change the persistent
~7 GB warmup swap growth nor decode throughput (persistent CUDA-UVM/workspace
component). Migration remains BLOCKED.

## Setup
- Repo main@fd61254 tracked-clean throughout; git diff --check clean.
- Image both nodes: sha256:001dcd2fb66d...425 (r3), never rebuilt/retagged.
- Config: r3 preset (KV4G, eager, b12x, ST_PREAD=1) + hc-exp + kv4g-debug +
  pread-r3 + b12x-cache overlays. Rendered head+worker configs BIT-IDENTICAL
  between arms (diff = empty; only cache directory CONTENT differed).
- Guards both arms: ma soft 5.0 / hard 4.0 GiB both nodes; swap hard s01
  7.9 GiB / s02 4.0 GiB (s01 soft 6.7 / s02 3.2 GB post-hoc marks) — chosen
  and documented before Arm A; unchanged.
- Controlled reboots before EACH arm (s02 then s01); clean baselines
  ~117.6 GiB / swap 0 verified all four times.

## Cache contract (discovered, Phase 1)
- flashinfer JIT: /root/.cache/flashinfer/0.6.15/121a (dotcache mount).
  Image ships NO prebaked JIT cache (8 KiB log only) — every prior startup
  of this lineage recompiled from scratch. flashinfer-cubin package provides
  prebuilt MoE cubins; only `sampling` was JIT-compiled (~6.5 MB).
- FlashInfer autotuner: NO file persistence (in-memory only) — autotune
  reruns in both arms by design.
- CUDA driver JIT: /root/.nv/ComputeCache (nv mount) — THE dominant artifact:
  ~244 MB per node, written during init warmup (cutlass-DSL/PTX->SASS for
  sm_121). First-request adds 2 more entries per node.
- Triton: /root/.triton (88 KiB). CUTE DSL: CUTE_DSL_CACHE_DIR=/cache/cutedsl
  (redirected from non-persistent /tmp default; 4 KiB). torchinductor:
  compilation mode NONE (unused). deep_gemm: module NOT installed. /tmp: Ray
  runtime state, intentionally not persisted.
- Arm A empty-start proven (0 files both nodes); Arm A manifest+SHA256 taken;
  pre-Arm-B diff = UNCHANGED both nodes; post-Arm-B diff = flashinfer_jit.log
  content ONLY (append) — ZERO recompiles, ZERO new artifacts in Arm B.
- Node caches are rank/node-local: same file structure, DIFFERENT content
  hashes (compiled separately in Arm A); never shared between nodes.

## A/B results (UTC 2026-07-26)
| metric | Arm A (cold) | Arm B (warm) |
|---|---|---|
| container start | 01:49:59 | 02:27:48 |
| load (TP0/TP1, pread) | 140.0 / 171.7 s | 141.6 / 171.8 s |
| init engine (profile+KV+warmup) | 320.23 s | 227.13 s (-29%) |
| time to first /health 200 | ~603 s (02:00:02) | ~491 s (02:35:59) (-19%) |
| s01 min MemAvailable (20 ms) | 4,395,748 kB = 4.19 GiB, SOFT crossed | 7,911,332 kB = 7.55 GiB, NO soft |
| s02 min MemAvailable | 6,207,880 kB = 5.92 GiB | 6,840,368 kB = 6.52 GiB |
| s01 swap at ready / peak | 6.94 / ~6.98 GB | 7.47 / ~7.53 GB (UNCHANGED-to-worse) |
| s02 swap | ~0 (348 kB) | ~0 (344 kB) |
| C3 16-tok first request | 195 s | 67 s (-66%) |
| Korean 120 tok | 36.7 s (3.27 t/s) | 37.7 s (3.18 t/s) |
| second request 40 tok | 30.3 s | 30.7 s |
| 5-min idle | ma/swap flat, health 200 x10 | ma/swap flat, health 200 x10 |
| new cache files during arm | 27/node (incl. 3/node at first request) | 0 (log append only) |
- Deviation note: Arm B C3 ran ~70 min after actual ready (session-restart
  notification delay). Swap/ma were FLAT for that entire idle (extra
  stability evidence); in-memory autotune state persisted; comparison of
  first-request behavior remains valid (all JIT/cache state is process- or
  file-resident, not time-decaying).
- Cold-arm first request wrote CUDA ComputeCache entries on BOTH nodes
  (driver-JIT of decode-path kernels) = the rank-local lazy specialization;
  warm arm reused them (no new files, 3x faster C3, s02 floor +0.6 GiB).

## b12x conformance observation (NOT a final conformance pass)
- C3 deterministic 16-tok: Arm A tokens == Arm B tokens (bit-repeatable);
  both 14/16 vs marlin/v0.22.1 reference; first divergence pos 14
  (" Question" -> " User", lp -0.237/-0.179); tok0 lp -0.027/-0.030 healthy;
  no garble, no first-token collapse — matches the pread-complete result.
- Korean 120-tok: A/B diverge at char 202 ("Key Information" vs "Key
  Characteristics") — mid-sequence low-margin flip, both fully coherent
  (b12x run-to-run numerics; early tokens deterministic).

## Interpretation (per pre-declared rules)
- JIT/autotune component CONFIRMED for: init memory floor (s01 +3.36 GiB,
  s02 +0.6 GiB, no soft crossings), engine-init time, first-request latency
  and first-request specialization — all correlate with 100% cache reuse.
- Persistent workspace/UVM component CONFIRMED for: s01 warmup swap
  (0 -> ~7.5 GB in BOTH arms, with Arm B MemAvailable never below 7.5 GiB —
  anon swap-out WITHOUT host-level shortage = driver/UVM-side reclaim
  pressure, outside the cgroup, consistent with all prior arcs) and decode
  (~3.2 t/s both arms vs 12.8 t/s on 0.22.1).
- Therefore: PHASE2_MIXED_CAUSE. Confidence: HIGH for the JIT half (direct
  A/B correlation), MEDIUM-HIGH for UVM attribution of the remainder
  (cgroup-external, ma-high swap-out; no per-allocation UVM telemetry yet).
- Decode note: prior "0.3 t/s collapse" (pread-complete arc) is now shown to
  have been dominated by the C2 instrumentation overlays; clean r3 decodes
  at ~3.2-3.3 t/s — still ~4x below the 0.22.1 baseline, cache-independent.

## Recommended next arc (exactly one)
PERSISTENT WORKSPACE ISOLATION — warm cache does not resolve swap or decode.
First sub-step: dummy-run tensor-lifetime audit within that arc (the swap
burst aligns with the init dummy-run window and persists), plus per-stage
UVM residency accounting (nvidia-smi per-proc, cudaMemGetInfo deltas, b12x
workspace sizing) to isolate the ~7 GB swap-forcing resident set.

## Remaining migration blockers
1. Persistent warmup swap ~7.5 GB (s01) + decode ~3.2 t/s (vs 12.8 baseline)
   — cache-independent, UVM/workspace suspect.
2. b12x conformance decision (repeatable 14/16 + late-token run-to-run
   nondeterminism needs a formal acceptance rule).
3. Matched acceptance benchmark (not run in this arc).

## State after arc
Both nodes: containers removed, port 8000 free, telemetry stopped, caches
PRESERVED at $EV/cache/<node>/ (untouched), no post-Arm-B reboot (per spec).
s01 final ma 12.5 GiB / swap 0.27 GB residual; s02 13.1 GiB / 0.3 MB.
No tracked file modified; nothing committed/pushed; no image change.
