# Model runbook index

This index groups the model-specific documents referenced by the executable recipe manifest.
The root README owns the common clone, selection, render, topology, readiness, and shutdown workflow.
Each model runbook owns its exact model revision or identity check, image/runtime settings, required
overlays, test matrix, measured results, known limits, and rollback details.

A model runbook should use this stable outline when it is next revised:

1. supported recipe variants;
2. model source and immutable revision or explicit identity limitation;
3. image identity and distribution class;
4. one-Spark or two-Spark prerequisites;
5. model download and verification;
6. preset, overlays, entrypoint, and rendered contract;
7. backend-specific start and readiness checks;
8. test matrix and retained evidence;
9. known limits and rollback.

## DeepSeek-V4 Flash

- [`deepseek-v4-production.md`](../deepseek-v4-production.md) — four MP recipes, image and rollback details.

## Solar-Open2-250B

- [`solar-open2-production.md`](../solar-open2-production.md) — Ray recipes and the documented local-only image/orchestration limitation.

## Qwen3.8 Flash Next

- [`qwen3.8-flash-next-tp2.md`](../qwen3.8-flash-next-tp2.md) — official checkpoint MP recipe.
- [`qwen3.8-flash-next-pr55122-mtp1-c1-candidate.md`](../qwen3.8-flash-next-pr55122-mtp1-c1-candidate.md) — local-only MTP1 recipe evidence.
- [`qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md`](../qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md) — canonical Ray recipe and dedicated MP rollback.

## Step-3.7 Flash

- [`benchmarks/bt-matrix-step37-nvfp4-v023.md`](../benchmarks/bt-matrix-step37-nvfp4-v023.md) — NVFP4 v0.23 latency matrix recipe evidence.
- [`step3.7-tokenizer-overlay.md`](../step3.7-tokenizer-overlay.md) — FP8 v0.23 tokenizer-overlay recipe.
- [`step3.7-flash-tp2.md`](../step3.7-flash-tp2.md) — FP8 two-Spark recipe.

## Gemma 4 and Qwen3.6

- [`model-serving-validation-history.md`](../model-serving-validation-history.md) — bounded single-Spark validation records for Gemma 4 and Qwen3.6.
- [`flashinfer-aot-prebake.md`](../flashinfer-aot-prebake.md) — Qwen3.6 two-Spark FlashInfer AOT recipe.

The generated executable catalog is [`../../recipes/README.md`](../../recipes/README.md). General,
experimental, and historical preset references remain in [`../../presets/README.md`](../../presets/README.md).
