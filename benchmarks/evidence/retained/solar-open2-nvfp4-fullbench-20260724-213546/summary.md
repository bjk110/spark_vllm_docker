# Solar-Open2-250B NVFP4 — llama-benchy safety-bounded benchmark (2026-07-24/25)

## Classification

**Safety-bounded benchmark (16 of 24 rows).** The pp8192 block (8 rows) was skipped:
the serving config is capped at `MAX_MODEL_LEN=4096`, and raising maxlen is prohibited
per the first-run validation (post-load UMA headroom is minimal; increasing maxlen/seqs
risks spark02 deep-stall requiring a power reset — see solar_open2_first_run memory).
This is NOT a completed full benchmark.

## Setup

- Model: `nota-ai/Solar-Open2-250B-Nota-NVFP4` served as `solar-open2-250b`
- Serving: dual-node TP=2 (spark01+spark02, Ray, RoCE), image
  `vllm-spark:solar-open2-nvfp4-v022d568-vllm0221-upstage00907fc-ecfix-exp`
- vLLM 0.22.1 / torch 2.12.0a0 (NGC 26.05) / FlashInfer 0.6.12
- Engine: enforce-eager (no CUDA graphs), KV bf16 (auto), maxlen 4096, seqs 8, bt 2048,
  gpu_mem_util 0.80, moe_backend flashinfer_b12x, prefix caching enabled server-side
  (benchmark ran with unique prompts; llama-benchy reported prefix_caching_enabled=False)
- Driver: llama-benchy 0.3.8 on spark01, localhost:8000
- Command: see `benchmark_command.txt`; `--exact-tg` used (min_tokens enforced, DS3T lesson)
- Coherence test: PASSED. Generation-mode latency probe: 8602.55 ms.
- Memory guard: min MemAvailable during run — spark01 15,755 MiB, spark02 19,238 MiB
  (logs `mem_s01.log` / `mem_s02.log`, 30 s interval). No OOM, no stall; health 200 after run.
- Note: tokenizer trust_remote_code prompt auto-answered N (stdin /dev/null); llama-benchy
  fell back to generic tokenizer loading with `--adapt-prompt` warmup correction
  (delta 9–18 tokens). Token counting is warmup-calibrated, not exact-tokenizer.

## Consolidated results (mean ± std over 3 runs)

| model | test | t/s (total) | t/s (req) | peak t/s | peak t/s (req) | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) |
|---|---|---|---|---|---|---|---|---|
| solar-open2-250b | pp512 (c1) | — | — | — | — | 483.61 ± 1.94 | 0.00 ± 0.00 | 483.61 ± 1.94 |
| solar-open2-250b | tg128 (c1) | 12.83 ± 0.02 | 12.83 ± 0.02 | 13.33 ± 0.47 | 13.33 ± 0.47 | — | — | — |
| solar-open2-250b | pp512 (c2) | 1341.03 ± 237.85 | — | — | — | 702.40 ± 158.76 | 0.00 ± 0.00 | 702.40 ± 158.76 |
| solar-open2-250b | tg128 (c2) | 25.21 ± 0.60 | 12.74 ± 0.17 | 26.67 ± 0.94 | 13.33 ± 0.47 | — | — | — |
| solar-open2-250b | pp512 (c4) | 1526.62 ± 239.88 | — | — | — | 1104.64 ± 333.62 | 0.00 ± 0.00 | 1104.64 ± 333.62 |
| solar-open2-250b | tg128 (c4) | 47.70 ± 2.00 | 12.38 ± 0.39 | 52.00 ± 0.00 | 13.00 ± 0.00 | — | — | — |
| solar-open2-250b | pp512 (c8) | 1623.44 ± 63.93 | — | — | — | 1776.70 ± 720.64 | 0.00 ± 0.00 | 1776.70 ± 720.64 |
| solar-open2-250b | tg128 (c8) | 40.64 ± 21.99 | 6.65 ± 3.13 | 101.33 ± 3.77 | 12.67 ± 0.47 | — | — | — |
| solar-open2-250b | pp2048 (c1) | — | — | — | — | 1289.67 ± 10.93 | 0.00 ± 0.00 | 1289.67 ± 10.93 |
| solar-open2-250b | tg128 (c1) | 12.88 ± 0.01 | 12.88 ± 0.01 | 13.00 ± 0.00 | 13.00 ± 0.00 | — | — | — |
| solar-open2-250b | pp2048 (c2) | 1702.95 ± 91.10 | — | — | — | 1992.33 ± 450.43 | 0.00 ± 0.00 | 1992.33 ± 450.43 |
| solar-open2-250b | tg128 (c2) | 23.73 ± 0.71 | 12.42 ± 0.50 | 26.00 ± 0.00 | 13.00 ± 0.00 | — | — | — |
| solar-open2-250b | pp2048 (c4) | 1667.27 ± 4.92 | — | — | — | 3152.19 ± 1221.39 | 0.00 ± 0.00 | 3152.19 ± 1221.39 |
| solar-open2-250b | tg128 (c4) | 38.08 ± 0.06 | 10.94 ± 1.04 | 52.00 ± 0.00 | 13.00 ± 0.00 | — | — | — |
| solar-open2-250b | pp2048 (c8) | 1613.47 ± 2.74 | 4968.92 ± 3656.78 | — | — | 5677.21 ± 2848.58 | 223.72 ± 508.00 | 5677.21 ± 2848.58 |
| solar-open2-250b | tg128 (c8) | 48.40 ± 0.34 | 7.88 ± 1.24 | 88.00 ± 0.00 | 11.50 ± 0.71 | — | — | — |

Notes:
- pp (c1) total throughput is null in llama-benchy 0.3.8 JSON output for this run
  (values reported only for c>=2); ttfr at c1 implies ~1.06k t/s (pp512) and ~1.59k t/s
  (pp2048) single-stream prefill, but per convention missing values are shown as `—`,
  not derived.
- Skipped rows: pp8192 (c1,c2,c4,c8) + tg128 pairs — safety-bounded (maxlen 4096 cap).
- tg128 pp512 c8 shows high variance (40.64 ± 21.99 total, one straggler run);
  raw per-run values preserved in `result.json`.

## Observations

- Single-stream decode ~12.8–12.9 t/s; total decode throughput scales to ~48 t/s at
  c4–c8 while per-request drops (6.6–12.4 t/s). Peak instantaneous decode 101 t/s (c8).
- Prefill total throughput plateaus at ~1.3–1.7k t/s from c2 onward (bt2048 serialization).
- Config is the first-run safety configuration (enforce-eager, bf16 KV); results are a
  floor, not a tuned figure.
