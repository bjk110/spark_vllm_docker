# Qwen3.8 Flash Next Uncensored FP8 — NGC 26.11 MTP 4/3 recommended profile

| Field | Operational value |
|---|---|
| Status | **Recommended; manually activated 2026-11-22** |
| Model | `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8` |
| Model revision | `8bebbc05cbe65099f37c520865463c1377d8050d` |
| Image | local immutable ID `sha256:491a7ae4cec7c706a5325d249ecae20d2ed41734d5b137d997ce3c77efaccb21` |
| Base/runtime | NGC 26.11; vLLM commit `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca` |
| Topology | dual DGX Spark, Ray TP=2/EP=2 over RoCE |
| Context/concurrency | 262,144 tokens; engine C2 |
| Preset | [`presets/qwen3.8-flash-next-uncensored-fp8-ngc2611-mtp43-recommended.env`](../presets/qwen3.8-flash-next-uncensored-fp8-ngc2611-mtp43-recommended.env) |
| Overlay | [`compose/qwen3.8-flash-next/docker-compose.ngc2611-v034-mtp43.yml`](../compose/qwen3.8-flash-next/docker-compose.ngc2611-v034-mtp43.yml) |
| Entrypoint | [`entrypoints/qwen38-ngc2611-v034-mtp43-ray.sh`](../entrypoints/qwen38-ngc2611-v034-mtp43-ray.sh) |
| Rollback | retained NGC 26.10 `qwen38-ngc2610-v034-final-r1-*` containers |

## Adopted contract

```json
{
  "method": "mtp",
  "num_speculative_tokens": 4,
  "num_speculative_tokens_per_batch_size": [[1, 1, 4], [2, 2, 3]]
}
```

The runtime remains eager and uses FP8 KV with a fixed 8 GiB cache per rank, default fused-MoE heuristics, prefix caching, chunked prefill, and expert parallelism. CUDA graph and the generated GB10 MoE config were rejected because they did not improve the matched workload. The third-party b12x GDN overlay was not applied because it is incompatible with this pinned tree; the tree already contains the GB10-appropriate upstream FlashInfer GDN path.

## Qualification

The sequential A/B qualification passed:

- strict tool schema 8/8;
- 32K, 131K, and 262,112-token retrieval;
- exact response and required tool call;
- final C1 75.41 tok/s and C2 aggregate 121.65 tok/s in the qualification harness;
- one final 10-minute soak: 20 cycles, 40 requests, 0 failures;
- final RoCE gate: 108.84/109.01 Gbit/s, direction ratio 1.0016.

See the retained operational benchmark: [`benchmarks/evidence/retained/qwen38-ngc2611-mtp43-20261122/report.md`](../benchmarks/evidence/retained/qwen38-ngc2611-mtp43-20261122/report.md). The compact [qualification evidence](../benchmarks/evidence/retained/qwen38-ngc2611-mtp43-20261122/qualification.json) records the gate values and SHA-256 of the six maintainer-local source artifacts.

## Image portability boundary

The image is currently a local immutable artifact on `spark01` and `spark02`; it is not a pullable registry manifest. The recipe therefore fails closed for fresh hosts until the exact image is published or transferred and the image ID is verified. Do not substitute another NGC 26.11/vLLM build under this preset.

## Preflight and launch

1. Verify driver `615.71.10`, model revision marker, exact image ID, and bidirectional RoCE.
2. Render the Compose closure from a clean environment:

```bash
env -i PATH="$PATH" HOME="$HOME" \
  docker compose \
  --env-file presets/qwen3.8-flash-next-uncensored-fp8-ngc2611-mtp43-recommended.env \
  -f docker-compose.yml \
  -f compose/qwen3.8-flash-next/docker-compose.ngc2611-v034-mtp43.yml \
  --profile '*' config --quiet
```

3. Stop any service on port 8000. Start the Ray head, then the worker before the 600-second join timeout. No automatic restart policy is added; a failed initialization must remain visible instead of entering a memory-thrashing restart loop.
4. Require HTTP health 200, the expected model and `max_model_len=262144`, exact non-thinking output, a required tool call, increasing MTP counters, running containers, and no hard failure.

The currently activated containers are:

- `spark01`: `qwen38-ngc2611-v034-mtp43-prod-head`
- `spark02`: `qwen38-ngc2611-v034-mtp43-prod-worker`

## Operational benchmark

`llama-benchy 0.4.0` was run after activation with:

```text
pp512/2048/8192 × tg128 × client concurrency 1/2/4/8 × 3
```

It completed 12 benchmark objects, 24 consolidated pp/tg rows, and 135 requests with zero errors. Representative aggregate decode results were:

| Prompt | C1 | C2 |
|---:|---:|---:|
| 512 | 37.77 tok/s | 59.97 tok/s |
| 2048 | 34.70 tok/s | 60.07 tok/s |
| 8192 | 37.96 tok/s | 32.77 tok/s |

C4/C8 are client queue-pressure rows because the engine cap is C2. Benchmark-window MTP acceptance was 43.66% (`12,977/29,725`), and all post-benchmark health/fatal checks passed.

## Rollback

If health, correctness, tool use, or hard-failure checks fail:

1. Stop NGC 26.11 head, then worker.
2. Start retained NGC 26.10 head, then worker:
   - `qwen38-ngc2610-v034-final-r1-head`
   - `qwen38-ngc2610-v034-final-r1-worker`
3. Wait for health 200 and verify exact output plus required tool call.

Do not delete the rollback containers or the NGC 26.10 image while this recommended profile remains local-only.
