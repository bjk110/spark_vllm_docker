# Solar Open 2 — Multi-Instance Low-Concurrency A/B (MI arc)

- Arc: controlled multi-instance A/B, v0.22.1 baseline vs v0.25.1 r4, pp512/tg128 c1+c2 only
- Date: 2026-07-27 (UTC)
- **Final state: `SOLAR_OPEN2_V0251_MULTI_INSTANCE_VARIANCE_DOMINATED`**

## 1. Final state

`SOLAR_OPEN2_V0251_MULTI_INSTANCE_VARIANCE_DOMINATED` (decision rule C; evaluation in §24).

## 2–4. Evidence, mirror, verification

- spark01: `/home/bjk110/docker-build/solar-open2-v0251-multi-instance-lowconc-20260727T004456Z/`
- Homeserver mirror: `/home/bjk110/docker/vllm-spark/benchmarks/results/solar-open2-v0251-multi-instance-lowconc-20260727T004456Z/` (untracked path)
- SHA-256: every mirrored file verified against `SHA256SUMS.txt` (result recorded in §mirror log).

## 5. Repository

spark01 `~/docker/vllm-spark`, branch `main`, HEAD `fd61254`, tracked state clean at preflight and
completion; `git diff --check` clean; no commit/push/stage. All harness files untracked and outside
the repository.

## 6–7. Images (identical on spark01 and spark02, arm64, not rebuilt)

- Baseline: `vllm-spark:solar-open2-nvfp4-v022d568-vllm0221-upstage00907fc-ecfix-exp`
  `sha256:1873d2174691f67e16b5588fcef01680d21f1e7b42ac5587bd23d7503cae1366`
- r4: `vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rawg1-pread-b12xsw-r4-exp`
  `sha256:ecb7bfe3978a5241c5c304d52ce91e061e22b750178d21a4ef7788a08e86e774`

## 8. Exact benchmark command (recovered from LS evidence; llama-benchy 0.3.8, `--exact-tg` confirmed)

```
/home/bjk110/.local/bin/llama-benchy --model nota-ai/Solar-Open2-250B-Nota-NVFP4 \
  --served-model-name solar-open2-250b --base-url http://localhost:8000/v1 \
  --tokenizer /home/bjk110/Documents/Models/upstage/nota-ai_Solar-Open2-250B-Nota-NVFP4 \
  --tg 128 --exact-tg --runs 3 --latency-mode generation --format json \
  --pp 512 --concurrency {1|2} --save-result result-c{N}.json
```
Per invocation: built-in warmup request (excluded) + exactly 3 measured repetitions.

## 9. Runtime commands per arm (via validated `launch-sf.sh`, guards-only, no instrumentation)

- Baseline: `presets/solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env` + `docker-compose.yml`
  + `docker-compose.solar-open2-hc-exp.yml`
- r4: `presets/solar-open2-250b-nota-nvfp4-v0251-r4-b12xsw-kv4g-exp-tp2.env` + same base +
  `pread-r3` + `b12xsw-r4` + `b12x-cache` overlays (CW warm caches; `VLLM_SPARK_ST_PREAD=1`,
  `VLLM_SPARK_B12X_SHARED_WORKSPACE=1`)
- Both: TP=2 Ray, RoCE 10.10.10.1/.2, enforce-eager, no CUDA graphs, `--kv-cache-memory-bytes
  4294967296`, max_model_len 4096, max_num_seqs 8, max_num_batched_tokens 2048, gmu 0.80,
  flashinfer_b12x, same parsers/logits processor/template. Rendered configs and the full
  `vllm serve` line preserved per instance (`rendered-*.yml`, `runtime-markers.txt`).
- Fresh-instance procedure: controlled reboot spark02→spark01 before EVERY instance (GB10 UMA
  residual made normal recovery fail: idle MemAvailable 33/28 GiB vs required ~117 GiB — the
  documented validated procedure). Post-reboot clean gate: MemAvailable ≥110 GiB, swap ≈0, port
  8000 free — passed 12/12 (actual ~117.7 GiB every time).

## 10. Execution order and completion (all 12 instances complete, 6/6 pairs valid)

| # | instance | cells | startup | bench | note |
|---|---|---|---|---|---|
| 1 | pair01/base | c1,c2 | 867 s | 455 s | |
| 2 | pair01/r4 | c1,c2 | 479 s | — | driver-bug stop after HEALTH_OK; bench resumed manually on the same fresh engine (see §23) |
| 3 | pair02/r4 | c2,c1 | 457 s | 293 s | |
| 4 | pair02/base | c2,c1 | 867 s | 455 s | |
| 5 | pair03/base | c2,c1 | 867 s | 454 s | |
| 6 | pair03/r4 | c2,c1 | 479 s | 294 s | |
| 7 | pair04/r4 | c1,c2 | 457 s | 291 s | |
| 8 | pair04/base | c1,c2 | 866 s | 447 s | |
| 9 | pair05/base | c1,c2 | 866 s | — | orchestrator killed by session restart during health wait; engine unaffected; bench resumed manually on the same fresh engine (see §23) |
| 10 | pair05/r4 | c1,c2 | 501 s | 285 s | |
| 11 | pair06/r4 | c2,c1 | 479 s | 286 s | |
| 12 | pair06/base | c2,c1 | 866 s | 447 s | |

Startup reproducibility: baseline 866–867 s (6/6), r4 457–501 s (6/6).

## 11–13. Per-instance raw results and medians (tg t/s; 3 reps each; median-of-3 = instance value)

### c1
| pair | base values | base med | r4 values | r4 med |
|---|---|---|---|---|
| 01 | 12.334/12.333/12.338 | 12.334 | 12.432/12.458/12.465 | 12.458 |
| 02 | 11.999/12.003/12.000 | 12.000 | 12.032/12.051/12.056 | 12.051 |
| 03 | 12.233/12.246/12.232 | 12.233 | 11.761/11.790/11.790 | 11.790 |
| 04 | 12.942/13.006/12.928 | 12.942 | 11.791/11.770/11.750 | 11.770 |
| 05 | 12.283/12.281/12.277 | 12.281 | 12.478/12.440/12.435 | 12.440 |
| 06 | 12.951/12.925/12.934 | 12.934 | 12.512/12.525/12.527 | 12.525 |

### c2
| pair | base values | base med | r4 values | r4 med |
|---|---|---|---|---|
| 01 | 24.218/22.950/24.266 | 24.218 | 23.830/23.794/23.958 | 23.830 |
| 02 | 22.793/22.558/23.721 | 22.793 | 22.557/22.602/22.539 | 22.557 |
| 03 | 23.021/23.077/24.121 | 23.077 | 22.090/22.087/22.121 | 22.090 |
| 04 | 24.628/25.856/25.929 | 25.856 | 22.673/22.624/22.668 | 22.668 |
| 05 | 24.314/24.141/24.077 | 24.141 | 23.798/23.694/23.736 | 23.736 |
| 06 | 24.555/24.759/24.759 | 24.759 | 23.704/23.677/23.676 | 23.677 |

Within-instance repetition CV: r4 ≤0.30% all instances; baseline c1 ≤0.26%, baseline c2 up to
2.56% (baseline c2 has a recurring one-rep dip pattern; median-of-3 robust to it).

## 14. Aggregate statistics (6 instance medians per arm/cell)

| cell/arm | mean | median | sd | MAD | min | max | CV |
|---|---|---|---|---|---|---|---|
| c1 base | 12.454 | 12.307 | 0.392 | 0.191 | 12.000 | 12.942 | 3.15% |
| c1 r4 | 12.172 | 12.245 | 0.347 | 0.246 | 11.770 | 12.525 | 2.85% |
| c2 base | 24.141 | 24.179 | 1.121 | 0.841 | 22.793 | 25.856 | 4.64% |
| c2 r4 | 23.093 | 23.173 | 0.745 | 0.589 | 22.090 | 23.830 | 3.22% |

Aggregate-median ratios r4/base: c1 **0.9950** (−0.062 t/s, −0.50%); c2 **0.9584** (−1.007 t/s,
−4.16%). (Aggregate ratio pairs unmatched instances; the paired analysis below is primary.)

## 15. Six paired ratios per cell (r4 instance median / base instance median)

| pair | c1 | c2 |
|---|---|---|
| 01 | 1.0100 | 0.9840 |
| 02 | 1.0042 | 0.9897 |
| 03 | 0.9638 | 0.9572 |
| 04 | 0.9094 | 0.8767 |
| 05 | 1.0130 | 0.9832 |
| 06 | 0.9684 | 0.9563 |

- c1: median **0.9863**, mean 0.9781, range [0.9094, 1.0130], 3/6 below 1.000, 3/6 ≤0.97
- c2: median **0.9702**, mean 0.9579, range [0.8767, 0.9897], **6/6 below 1.000**, 3/6 ≤0.97

## 16. Paired bootstrap 95% CI of median ratio (100,000 iterations, seed 20260727, pairs resampled — never within-instance reps)

- c1: median 0.9863, CI **[0.9366, 1.0115]** — includes 1.000
- c2: median 0.9702, CI **[0.9165, 0.9868]** — excludes 1.000

## 17. Temporal drift and order effects (descriptive only; no causality claimed)

- Chronological medians show no monotonic drift; the two highest baseline instances (12.942,
  12.934) occurred at positions 8 and 12, the lowest (12.000) at position 4.
- First-arm vs second-arm within pair: second-arm slightly higher on average (c1 12.199→12.427,
  c2 23.390→23.844); arm assignment was counterbalanced 3/3, so this does not bias the arm
  comparison.
- Cell order: c2 tested-second averages higher than tested-first (24.075 vs 23.159, +4.0%);
  counterbalanced 3/3 per arm.
- Pre-start MemAvailable was uniform (123.39–123.44 GB across all 12 instances) — no correlation
  with throughput is inferable (spread ~0.04%).

## 18. Pre-start memory-state comparison

All 12 instances started from an identical envelope: MemAvailable 123,387,276–123,438,476 kB,
swap ≈0, port 8000 free, no stale containers/Ray. Recorded per instance in `prestate-spark0{1,2}.txt`.

## 19. Runtime-continuity findings

- Per-instance min MemAvailable (20 ms fastguard sampling): baseline s01 ≥33.0 GiB / s02 ≥35.4 GiB;
  r4 s01 ≥21.7 GiB / s02 ≥22.2 GiB — all far above the 4 GiB floor; zero guard trips.
- Peak swap: r4 **0 MiB on both nodes, 6/6 instances**; baseline s01 ~5.5 GiB + s02 ~1.2–1.4 GiB
  **every instance** (inherent v0.22.1 load-path behavior, uniform across instances and therefore
  not a pair-level confound; identical to AC-arc observations).
- No OOM, no NVRM/Xid events, no worker restarts, no engine warnings materially different from the
  validated LS runs (recurring benign raylet "over 95% full" disk notice present in both arms —
  root fs ≥19.9 GB free throughout).

## 20. r4 shared-workspace / pread gates

6/6 r4 instances: pread marker ×2 (one per rank), shared-workspace entries exactly 3 per rank
(ws_static 451.6 MiB, ws_dynamic 126.2 MiB, moe_output 16 MiB), fallback 0. KV identical every
instance: `GPU KV cache size: 66,764 tokens` (both arms, matching LS).

## 21. Swap and minimum MemAvailable

See §19; per-instance table in `08_analysis/continuity-minma-swap.txt`.

## 22. Request validity

Coherence test PASSED 24/24 benchy invocations; `Run 3/3` completed in every cell; `--exact-tg 128`
enforced; no request failures (0 error marks across all stdout logs); API health 200 after every
instance's bench.

## 23. Excluded instances

**None.** All 12 instances are valid and included. Two protocol deviations are documented (not
exclusions; both were harness/orchestration faults external to the runtimes, with the engine
verified fresh, healthy, and unbenchmarked at resume):
1. pair01/r4 — orchestrator false-failed on a gate-parsing bug (`grep -c` zero-match duplication)
   AFTER health and all r4 gates had genuinely passed; benchmark was started manually ~6 min later
   on the same fresh engine. Single allowed retry/resume of that arm.
2. pair05/base — orchestrator process was killed by a controller session restart during the health
   wait; the engine reached ready normally; benchmark was started manually ~45 min later on the
   same fresh idle engine. The baseline's known post-idle first-request re-specialization is
   absorbed by the benchy warmup (excluded from measurement); the instance's measured medians
   (12.281 / 24.141) sit mid-distribution, consistent with no residual effect.

## 24. Decision-rule evaluation (rules applied exactly)

Rule A (c2-specific regression): (1) c2 median 0.9702 ≤ 0.970 → **FAIL** (misses by 0.0002);
(2) CI upper 0.9868 < 1.000 → pass; (3) 6/6 below 1.000 → pass. → **Not A.**
Rule B (general regression): c1 median 0.9863 > 0.970, CI includes 1.000, 3/6 below 1.000 →
**Not B.**
Rule C (variance dominates): (1) neither cell meets a confirmed-regression rule ✓; (2) c2 median
paired ratio 0.9702 > 0.970 ✓; (3) pair signs (c1: 3 up / 3 down) and magnitudes (c2: −1.0% to
−12.3%) are mixed ✓; (4) no systematic runtime-continuity difference explains the result (all
gates uniform; baseline swap is an arm-inherent constant, not a pair-level variable) ✓;
(5) observed spread (instance CV 2.85–4.64%) is consistent with the previously documented ±4–5%
fresh-instance variance ✓. → **C applies.**
Rule D: not applicable — the c2 CI is informative (excludes parity), the data are complete, all
six pairs are valid, and pre-start state was controlled.

**Honest characterization (recorded, not a rule override):** c2 shows a direction-consistent small
deficit — 6/6 pairs below 1.000, median −2.98%, bootstrap CI [−8.35%, −1.32%]. A real sub-material
c2 deficit of roughly 1–3% likely exists, but it is below the 3% materiality gate (rule A missed
solely on the median criterion, by 0.0002) and far below the single-instance AC measurement
(−11.5/−13.8%), which this experiment shows was substantially variance-inflated: baseline
instance-to-instance spread alone reaches 4.6% CV (c2) / 7.8% max-min (c1). c1 is parity within
noise. The dominant explanation of the AC acceptance failure is fresh-instance variance.

## 25. Recommended next action (per variance-dominated branch)

- Retain r4 as the validated experimental candidate (do not promote).
- Replace single-instance performance acceptance with a multi-instance protocol: per-arm fresh
  instances (≥5 pairs), per-instance median-of-3, paired-ratio median + bootstrap CI as the gate
  statistic, counterbalanced order — this arc's harness (`01_protocol/mi_driver.sh`) is reusable
  as-is.
- Reassess the frozen single-instance performance gate separately (user decision; not modified
  here). The residual sub-material c2 deficit (~1–3%) should be weighed against r4's confirmed
  advantages (no 5.5 GiB swap, 2–3× faster startup, stable post-idle behavior, stable c8).
- Do not open a kernel-isolation arc without new evidence.

## 26. Service cleanup status

No test containers on either node; ports 8000/6379 free; guards (fastguard3/dmesg watchers)
stopped; hosts responsive; final state spark01 MemAvailable 39.8 GiB / swap used 134 MB, spark02
40.0 GiB / swap used 20 MB (post-run UMA residual as expected; recover by reboot only if needed
later). Images, model files, caches, and all evidence preserved. Disposable presets/harness files
retained untracked.

## 27. Migration promotion

Remains **BLOCKED**. Nothing was promoted; production and promoted presets untouched; the v0.22.1
baseline image untouched; no GHCR push.

## 28. Git

No commit, no push, no tracked-file modification. `git diff --check` clean; HEAD `fd61254`;
`git status` tracked-clean (untracked files pre-existing + evidence only).

---

Final state: **`SOLAR_OPEN2_V0251_MULTI_INSTANCE_VARIANCE_DOMINATED`**
