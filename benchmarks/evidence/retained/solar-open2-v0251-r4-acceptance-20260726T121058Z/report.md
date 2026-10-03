# Solar-Open2 r4 — formal b12x conformance + matched acceptance benchmark
Arc: solar-open2-v0251-r4-acceptance-20260726T121058Z

## FINAL STATE: SOLAR_OPEN2_V0251_R4_PERFORMANCE_FAILED
(conformance PASS-with-documented-variance; safety PASS with r4 clearly
superior; performance gates — frozen before execution — NOT met.)

## Conformance (frozen 3-level spec, corpus C1-C5, hashes preserved)
- Level 1 integrity: PASS both runtimes.
- Level 2 baseline parity (PRIMARY): C1 16-token sequence IDENTICAL r4 vs
  v0.22.1 across 8 runs each (9th+ consecutive reproduction overall);
  C3 tool contract: finish=tool_calls, tool invocation parity, stable both;
  C2/C4/C5 (long/low-margin class): the BASELINE ITSELF is run-to-run
  unstable (5/4/2 distinct texts across 5 runs); r4 exhibits the SAME
  instability class with identical finish behavior and coherent
  thinking-preamble outputs; no garble, no first-token collapse anywhere.
- Level 3 Marlin reference: both 14/16, divergence pos 14 (documented
  backend property).
- Verdict: **PASS WITH DOCUMENTED LOW-MARGIN NUMERICAL VARIANCE**.
- Note: C5 shows the preserved-think template consumes the token budget
  (finish=length on a one-word task) in BOTH runtimes — a template
  behavior contract, matched.

## Acceptance benchmark (llama-benchy 0.3.8, matched KV4G geometry,
pp512/2048 x tg128 x c1/c2/c4/c8, runs=3, scrape-free; frozen thresholds)
| cell | base tg | r4 tg | ratio | base CV% | r4 CV% |
|---|---|---|---|---|---|
| pp512 c1 | 12.89 | 12.24 | 0.950 | 0.01 | 0.08 |
| pp512 c2 | 26.05 | 23.05 | **0.885** | 0.11 | 0.12 |
| pp512 c4 | 46.96 | 42.77 | 0.911 | 1.73 | 0.11 |
| pp512 c8 | 41.04* | 70.22 | (1.711) | **54.45*** | 0.63 |
| pp2048 c1 | 13.10 | 11.94 | **0.912** | 0.05 | 0.18 |
| pp2048 c2 | 24.63 | 21.24 | **0.862** | 0.02 | 0.07 |
| pp2048 c4 | 37.40 | 34.58 | 0.925 | 2.21 | 0.22 |
| pp2048 c8 | 48.40 | 46.97 | 0.970 | 0.12 | 0.26 |
(*) baseline pp512-c8 CV 54.5% -> INVALID under the frozen variance rule;
baseline also accumulated 5.5 GB swap during its benchmark; r4 same cell:
stable (CV 0.63%) and 71% faster.
- Gates (FROZEN): c1 >= 0.95 -> pp512 0.950 PASS-at-boundary, pp2048 0.912
  FAIL. No c1/c2/c4 cell < 0.90 -> pp512-c2 0.885 and pp2048-c2 0.862 FAIL.
  Geomean all cells 0.990 (>=0.93) but including an invalid baseline cell;
  geomean over VALID cells = 0.916 -> FAIL.
- TTFT: r4 equal-or-better at c1 (0.90/0.98), comparable elsewhere.
- Finding: the regression is CONCENTRATED IN LOW-CONCURRENCY BATCHED DECODE
  (c2: -12~-14%); single-stream is borderline (0.95/0.91); c4 -7..-9%;
  high concurrency c8 favors r4 strongly (baseline destabilizes with swap).

## Safety (frozen gates)
- r4: alloc 76.2 GiB, swap peak 32 kB, min ma > 20 GiB, 3 shared entries/
  rank, fallback 0, idle 5 min stable, post-idle WARM (C1 1.44 s,
  B 10.13 s). ALL PASS.
- v0.22.1: completed, but swap grew to 5.5 GB during the matrix (within
  the 7.9 GiB guard), pp512-c8 unstable, and post-idle re-specialization
  reappeared at scale (C1 26.4 s, B 40.9 s after 5 min idle).
- Foreign-request audit: 1 pre/mid-idle health check outside measured
  windows (own poller); measured windows clean.

## Operational latency classification
- v0.22.1: cold-shape specialization required + POST-IDLE RE-SPECIALIZATION
  OBSERVED (26-41 s after 5 min idle).
- r4: cold-shape specialization required (but 2-3x cheaper) + POST-IDLE
  WARM. Proposed production priming (NOT implemented): one bounded request
  per expected prompt-shape class at deploy time (C1-class, Korean-class,
  short-math-class), ~2.5 min total on r4.

## Decision (frozen rules; thresholds not altered post-hoc)
- Conformance: PASS (with documented variance)
- Safety: PASS (r4 superior)
- Performance: **FAIL** (c1 pp2048 0.912 < 0.95; two c2 cells < 0.90;
  valid-cell geomean 0.916 < 0.93)
=> Overall: SOLAR_OPEN2_V0251_R4_PERFORMANCE_FAILED. No migration package
prepared (acceptance did not succeed).

## Recommended path (user decision required)
Option A (technical): one targeted arc on the c2-class batched-decode
regression (0.25 scheduler/chunked-prefill batching path suspected;
concentrated, reproducible CV<0.2% signal), then re-run this acceptance.
Option B (policy): user re-evaluates thresholds in light of (i) r4 c8
superiority + baseline c8 instability/swap, (ii) r4 post-idle warmth and
2-3x cold advantage, (iii) 14/16-parity conformance — i.e., whether the
c2 -12~14% matters for the intended single-user OWUI workload (typical
c1-c2). Only the user may change frozen thresholds.

## Remaining blockers
1. Performance acceptance (c2/c1-pp2048 cells) — per frozen thresholds.
2. (After that) user-approved migration-promotion decision.
Migration remains BLOCKED.
