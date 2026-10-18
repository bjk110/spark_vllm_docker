# Executable recipe catalog

Generated from [`manifest.json`](manifest.json). Do not edit this table by hand.

| Recipe ID | Nodes | Backend | Image distribution | Model | Runbook |
|---|---:|---|---|---|---|
| `deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2` | 2 | `mp` | `portable` | `deepseek-ai/DeepSeek-V4-Flash-0731` | [deepseek-v4-production.md](../docs/deepseek-v4-production.md) |
| `deepseek-v4-flash-0731-dspark-k7-256k-v027-ms4-optional-tp2` | 2 | `mp` | `portable` | `deepseek-ai/DeepSeek-V4-Flash-0731` | [deepseek-v4-production.md](../docs/deepseek-v4-production.md) |
| `deepseek-v4-flash-dspark-k7-64k-production-tp2` | 2 | `mp` | `portable` | `deepseek-ai/DeepSeek-V4-Flash-DSpark` | [deepseek-v4-production.md](../docs/deepseek-v4-production.md) |
| `deepseek-v4-flash-mtp1-production-tp2` | 2 | `mp` | `portable` | `deepseek-ai/DeepSeek-V4-Flash` | [deepseek-v4-production.md](../docs/deepseek-v4-production.md) |
| `solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2` | 2 | `ray` | `local-only` | `solar-open2-250b` | [solar-open2-production.md](../docs/solar-open2-production.md) |
| `solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2` | 2 | `ray` | `local-only` | `solar-open2-250b` | [solar-open2-production.md](../docs/solar-open2-production.md) |
| `qwen3.8-flash-next-fp8-tp2-candidate` | 2 | `mp` | `portable` | `Qwen/Qwen3.8-Flash-Next-FP8` | [qwen3.8-flash-next-tp2.md](../docs/qwen3.8-flash-next-tp2.md) |
| `qwen3.8-flash-next-fp8-tp2-pr55122-mtp1-c1-candidate` | 2 | `mp` | `local-only` | `Qwen/Qwen3.8-Flash-Next-FP8` | [qwen3.8-flash-next-pr55122-mtp1-c1-candidate.md](../docs/qwen3.8-flash-next-pr55122-mtp1-c1-candidate.md) |
| `qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2` | 2 | `ray` | `portable` | `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8` | [qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md](../docs/qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md) |
| `qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2-mp-rollback` | 2 | `mp` | `portable` | `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8` | [qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md](../docs/qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md) |
| `step37-flash-nvfp4-v023-tp2-latency` | 2 | `mp` | `local-only` | `stepfun-ai/Step-3.7-Flash-NVFP4` | [bt-matrix-step37-nvfp4-v023.md](../docs/benchmarks/bt-matrix-step37-nvfp4-v023.md) |
| `step37-flash-fp8-v023-tp2` | 2 | `mp` | `tag-only` | `stepfun-ai/Step-3.7-Flash-FP8` | [step3.7-tokenizer-overlay.md](../docs/step3.7-tokenizer-overlay.md) |
| `gemma4-31b-it` | 1 | `direct` | `tag-only` | `google/gemma-4-31B-it` | [model-serving-validation-history.md](../docs/model-serving-validation-history.md) |
| `qwen3.6-35b-a3b` | 1 | `direct` | `tag-only` | `Qwen/Qwen3.6-35B-A3B` | [model-serving-validation-history.md](../docs/model-serving-validation-history.md) |
| `qwen3.6-35b-a3b-fi-aot-tp2` | 2 | `ray` | `tag-only` | `Qwen/Qwen3.6-35B-A3B` | [flashinfer-aot-prebake.md](../docs/flashinfer-aot-prebake.md) |
| `step37-flash-fp8-tp2` | 2 | `ray` | `local-only` | `stepfun-ai/Step-3.7-Flash-FP8` | [step3.7-flash-tp2.md](../docs/step3.7-flash-tp2.md) |

Use `./scripts/recipe list`, `show`, `pull`, and `render` from the repository root.
Local-only entries are configuration/evidence references and cannot be pulled from a registry.
