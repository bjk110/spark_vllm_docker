# Qwen3.8 NGC 26.11 MTP 4/3 operational benchmark

- Date: 2026-11-22
- llama-benchy: 0.4.0
- Matrix: `pp512/2048/8192 × tg128 × client concurrency 1/2/4/8 × 3`
- Cardinality: **12 benchmark objects / 24 consolidated pp-tg rows / 135 requests**
- Errors: **0**
- Engine concurrency cap: **2**; client c4/c8 rows measure overlap and queue pressure, not engine C4/C8.

## Runtime

- Image ID: `sha256:491a7ae4cec7c706a5325d249ecae20d2ed41734d5b137d997ce3c77efaccb21`
- NGC 26.11; vLLM commit `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`
- Model revision: `8bebbc05cbe65099f37c520865463c1377d8050d`
- Dual DGX Spark Ray TP=2/EP=2 over RoCE
- Eager, FP8 KV, fixed 8 GiB/rank, default MoE
- MTP schedule: C1 depth 4, C2 depth 3

## Results

| Prompt | Client concurrency | Prefill tok/s | Aggregate decode tok/s | Per-request decode tok/s | E2E TTFT ms |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 18381.49 | 37.77 | 37.77 | 367.33 |
| 512 | 2 | 1439.79 | 59.97 | 32.78 | 532.48 |
| 512 | 4 | 367.32 | 59.12 | 32.47 | 2515.78 |
| 512 | 8 | 268.55 | 56.23 | 31.95 | 7108.78 |
| 2048 | 1 | 4309.80 | 34.70 | 34.70 | 800.47 |
| 2048 | 2 | 2792.57 | 60.07 | 32.41 | 1308.82 |
| 2048 | 4 | 1154.36 | 52.31 | 32.66 | 3799.93 |
| 2048 | 8 | 910.77 | 49.24 | 29.31 | 8998.60 |
| 8192 | 1 | 3657.63 | 37.96 | 37.96 | 2562.81 |
| 8192 | 2 | 3099.50 | 32.77 | 26.56 | 3898.05 |
| 8192 | 4 | 2314.79 | 31.67 | 24.46 | 8127.10 |
| 8192 | 8 | 1997.95 | 29.80 | 23.27 | 17460.01 |

The unusually high pp512/C1 prefill value has high run-to-run variance and must not be treated as a stable standalone target. The representative qualified operating rows are C1/C2; c4/c8 quantify queue pressure above the engine's two-slot cap.

## MTP and stability

- Draft tokens during benchmark: `29725`
- Accepted tokens: `12977`
- Acceptance: `43.66%`
- Post-benchmark exact response: `POST_BENCH_OK`
- Head/worker: running; OOMKilled=false; restart count 0
- Xid, API mismatch, Linux OOM kill, NCCL/engine fatal, and benchmark-window `NV_ERR_NO_MEMORY`: 0

Machine-readable retained evidence: [`summary.json`](summary.json). Qualification gates and source-artifact hashes: [`qualification.json`](qualification.json).
Raw logs and progress events remain outside Git at the path recorded in the JSON.
