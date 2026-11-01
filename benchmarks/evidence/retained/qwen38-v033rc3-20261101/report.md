# Qwen3.8 NGC 26.09/e77 vs vLLM v0.33.0rc3 — 2026-11-01

## Classification

`BOUNDED_PASS_NOT_ADOPTED`

The evaluated runtime used the same uncensored FP8 model revision and the established dual-Spark
Ray TP2/EP2, eager, MTP k=2, fixed 8 GiB FP8 KV/rank, 262,144-token contract. The variable was the
vLLM runtime: the existing NGC 26.09/e77 image versus a locally built v0.33.0rc3 image.

## Functional and stability gates

- Canonical, semantic, Korean, code, required-tool-call, JSON Schema, and synchronized-C2 probes: PASS.
- 30,083-token and 120,195-token retrieval: PASS.
- Two simultaneous exact 262,112-token requests: HTTP 200, `KESTREL-74291`, `finish_reason=stop`,
  server running peak 2, errors 0.
- No container OOM or restart occurred during the bounded run.
- After evaluation, the e77 head/worker were restored and both health/model endpoints returned 200.

## Comparison

| Prompt / concurrency | RC3 decode tok/s | e77 decode tok/s | Delta | RC3 TTFR | e77 TTFR |
|---|---:|---:|---:|---:|---:|
| 512 / C1 | 45.617 | 42.882 | +6.38% | 342.3 ms | 357.0 ms |
| 512 / C2 | 61.204 | 67.155 | -8.86% | 481.3 ms | 479.5 ms |
| 2048 / C1 | 42.517 | 40.688 | +4.50% | 752.5 ms | 781.8 ms |
| 2048 / C2 | 55.154 | 63.995 | -13.81% | 1107.6 ms | 1264.4 ms |

- RC3 KV capacity: 911,141 tokens / 3.48x nominal 262K concurrency.
- e77 KV capacity: 943,718 tokens / 3.60x; RC3 was 3.45% lower.
- RC3 cold start: 2,239.3 s; e77 restoration: 1,663.7 s.
- Exact 262,112 C2 latencies were 90.50/186.48 s on RC3 and 95.65/175.33 s on e77.
- The inherited optional `b12x_fp6` plugin entry point logged a missing `b12x.integration` import;
  it was non-fatal for this FP8 path but remains unresolved in the evaluated image.

## Decision

Keep the immutable NGC 26.09/e77 recipe unchanged. v0.33.0rc3 passed the bounded correctness and
long-context envelope, but it did not establish an operational advantage: C2 decode regressed by up
to 13.81%, KV capacity was lower, cold start was longer, and an optional plugin error remained.
No image publication, preset change, automatic startup, or activation is authorized by this record.

Machine-readable identities and measurements: [`comparison.json`](comparison.json).
