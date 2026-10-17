# Qwen3.8 Uncensored MTP k=2/C2 full benchmark

**Classification:** `PERFORMANCE-COMPLETE / PROTOCOL-COMPLETE / SYSTEM-STABLE`

## Immutable scope

- Model: `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8`
- Model revision: `8bebbc05cbe65099f37c520865463c1377d8050d`
- Image manifest: `ghcr.io/bjk110/vllm-spark@sha256:928d86903dfce3b11203f1f6d3118aa47998e55fee36fc73423e3bef8ab04cd6`
- Image ID: `sha256:8bb958ffb833c407cd642d72c3938c0f8f0ed31136851c1edcd868bf135651c4`
- vLLM: `e77daef89e18e08321ae7b8b24827eedd5fe8673`
- Runtime: eager, TP=2/EP=2, MTP k=2, `MAX_NUM_SEQS=2`, BT8K, fixed 8 GiB KV/rank, `MAX_MODEL_LEN=262144`
- Driver: `llama-benchy 0.4.0`

## Completion

- Matrix: `pp512/2048/8192 × tg128 × c1/c2/c4/c8`, three repeats
- Benchmark objects: `12/12`; standard consolidated pp/tg rows: `24/24`
- Requests: `3 repeats × (1+2+4+8 requests) × 3 prompt sizes = 135/135`
- Request errors: `0`
- Completion tokens: minimum `16`, maximum `128`; EOS-short `3/135` (valid because `--exact-tg` was not requested)
- MTP acceptance: `60.70%`; accepted tokens/iteration: `1.214`
- Recovered server gauges: `344` samples; running peak `2`; waiting peak `6`

## Results

Each row combines the prompt-processing and generation metrics from one benchmark object. Values are mean ± standard deviation.

| Prompt | Client concurrency | Prefill tok/s | Aggregate decode tok/s | Per-request decode tok/s | TTFT ms |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 1294.41 ± 123.19 | 20.70 ± 0.54 | 20.70 ± 0.54 | 637.40 ± 33.23 |
| 512 | 2 | 1179.94 ± 14.98 | 40.16 ± 1.16 | 20.79 ± 0.92 | 801.51 ± 9.63 |
| 512 | 4 | 225.86 ± 5.17 | 36.49 ± 1.50 | 19.96 ± 1.38 | 4372.31 ± 3597.64 |
| 512 | 8 | 167.07 ± 4.04 | 34.80 ± 1.37 | 19.94 ± 2.75 | 10631.60 ± 7777.89 |
| 2048 | 1 | 1844.11 ± 11.73 | 22.38 ± 1.45 | 22.38 ± 1.45 | 1358.18 ± 6.97 |
| 2048 | 2 | 1721.30 ± 138.15 | 37.48 ± 4.75 | 20.13 ± 1.93 | 2120.88 ± 415.16 |
| 2048 | 4 | 715.12 ± 22.92 | 32.37 ± 1.96 | 19.18 ± 2.51 | 6248.33 ± 4185.37 |
| 2048 | 8 | 567.83 ± 11.50 | 30.75 ± 0.09 | 18.64 ± 2.29 | 14588.99 ± 9496.88 |
| 8192 | 1 | 1795.29 ± 12.84 | 21.21 ± 0.78 | 21.21 ± 0.78 | 4810.60 ± 31.95 |
| 8192 | 2 | 1717.02 ± 35.24 | 27.79 ± 6.01 | 18.19 ± 4.55 | 7652.42 ± 2117.22 |
| 8192 | 4 | 1289.02 ± 26.32 | 21.89 ± 0.46 | 17.19 ± 4.60 | 15843.16 ± 7543.47 |
| 8192 | 8 | 1153.52 ± 33.84 | 18.20 ± 0.87 | 15.24 ± 5.18 | 30522.97 ± 17155.50 |

## Scope notes

- c4/c8 are client overlap and queue pressure on the C2 server; observed engine running peak is 2.
- Full performance/protocol completion is not a byte-identical concurrent semantic qualification.
- The initial gauge sampler emitted no rows. The recovered sampler started at 12:50:53 UTC, so early-row engine gauges are unavailable.
- The canonical correctness suite separately qualified multimodal/vision, tool calling, structured output, and simultaneous 262,112-token retrieval.
