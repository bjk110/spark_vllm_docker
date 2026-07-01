# Model Environment Presets

Docker Compose environment preset files for model-serving configurations. This directory does
**not** contain Hugging Face model weights.

**Status is set by this index, not by the filename.** A filename containing `production` or
`validated` does not by itself make a preset current — check its group below. Each `.env` file also
documents its own recipe/image/topology in its header comment.

Serve a preset directly:

```bash
docker compose --env-file presets/<preset>.env --profile head up -d   # + --profile worker on the worker node
```

Navigation: [Current production](#1-current-production-presets) · [Rollback](#2-rollback-presets)
· [Validated](#3-validated-presets) · [General supported](#4-general-supported-presets)
· [Experimental](#5-experimental-presets) · [Historical and reproduction](#6-historical-and-reproduction-presets)

---

## 1. Current production presets

The current accepted DeepSeek-V4-Flash production serving path. Runs the promoted image by its
**immutable GHCR manifest digest** (`sha256:ade810fd…`, config `fa83457d`); the mutable alias
`dsv4-sm121-indexer-production` is provenance only and must not be used as the runtime pin. Details:
[`docs/deepseek-v4-sm121-indexer-production.md`](../docs/deepseek-v4-sm121-indexer-production.md).

| Preset | Model / stack | Topology | Status | Use |
|---|---|---|---|---|
| `deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env` | DeepSeek-V4-Flash · SM121 DeepGEMM FP8-Q indexer, MARLIN MoE | dual-rdma TP=2 mp | **Current production** | Recommended DSV4 serving path (concurrency 1, ≤131K, MTP n=1, FULL_DECODE_ONLY `[2]`, 4 GiB fp8 KV) |

## 2. Rollback presets

Immediate and layered rollback targets for the current production baseline. Preserved unchanged;
not the current serving path. Runbook:
[`docs/deepseek-v4-prefill8192-production-runbook.md`](../docs/deepseek-v4-prefill8192-production-runbook.md).

| Preset | Model / stack | Topology | Status | Use |
|---|---|---|---|---|
| `deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-production-tp2.env` | DeepSeek-V4-Flash prefill8192 (config `4c41950c`) | dual-rdma TP=2 mp | **Immediate rollback** | Prior production; roll back here from `dsv4-sm121-indexer` |
| `deepseek-v4-v023-stack-pr41834-fullgraph-validated-rollback-tp2.env` | DeepSeek-V4-Flash graph-only (L1) | dual-rdma TP=2 | Rollback (L1) | Graph-only fallback (~27.2 t/s) |
| `deepseek-v4-v023-stack-pr41834-eager-u0-rollback-tp2.env` | DeepSeek-V4-Flash eager U0-RDMA (L2) | dual-rdma TP=2 | Rollback (L2) | Eager fallback (~7.4 t/s) |

## 3. Validated presets

Validated, but **not** the current default serving path.

| Preset | Model / stack | Topology | Status | Use |
|---|---|---|---|---|
| `deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-validated-candidate-tp2.env` | DeepSeek-V4-Flash prefill8192 | dual-rdma TP=2 mp | `VALIDATED_CANDIDATE` (superseded) | Provenance for the rollback baseline; not current serving |
| `deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-validated-tp2.env` | DeepSeek-V4-Flash MTP n=1 + FULL_DECODE_ONLY | dual-rdma TP=2 mp | `VALIDATED_PRESET_CANDIDATE` | Validated MTP fullgraph baseline; not production ([provenance](../docs/deepseek-v4-mtp1-fullgraph-validated-preset.md)) |
| `step37-flash-nvfp4-v023-tp2-latency.env` | Step-3.7-Flash NVFP4 · v0.23 (EP-off, MARLIN, TRITON_ATTN) | dual-rdma TP=2 | Validated (Step-3.7 NVFP4 path) | Recommended Step-3.7 NVFP4 latency path ([bench](../docs/benchmarks/bt-matrix-step37-nvfp4-v023.md)) |

## 4. General supported presets

Production-usable presets for non-DeepSeek-V4 models on the stable/forward stacks. Image bases:
`v021-ngc2603` / `v021-tq` / `v022-d568` / `v022-d568-fi-aot` / step3p7 (see
[`docs/software-stack.md`](../docs/software-stack.md)).

| Preset | Model | Quant / dtype | Topology | Base image |
|---|---|---|---|---|
| `gemma4-26b-a4b.env` | google/gemma-4-26B-A4B-it | BF16 MoE | single TP1 | v021-ngc2603 |
| `gemma4-26b-a4b-tq.env` | google/gemma-4-26B-A4B-it | BF16 + TurboQuant KV | single TP1 | v021-tq |
| `gemma4-31b-it.env` | google/gemma-4-31B-it | BF16 dense multimodal | single TP1 | v022-d568 |
| `intel-122b-int4.env` | Intel/Qwen3.5-122B-A10B-int4-AutoRound | INT4 AutoRound (Marlin) | single TP1 | v021-ngc2603 |
| `qwen3.5-122b-fp8.env` | Qwen/Qwen3.5-122B-A10B-FP8 | FP8 multimodal | dual-rdma TP2 | v021-ngc2603 |
| `qwen3.5-122b-nvfp4.env` | Qwen/Qwen3.5-122B-A10B | NVFP4 runtime | single TP1 | v021-ngc2603 |
| `qwen3.5-122b-nvfp4-tp2.env` | Qwen/Qwen3.5-122B-A10B | NVFP4 runtime | dual-rdma TP2 | v021-ngc2603 |
| `qwen3.5-122b-prismaquant.env` | rdtand/…PrismaQuant-4.75bit | PrismaQuant 4.76bpp mixed | single TP1 | v021-ngc2603 |
| `qwen3.5-397b-int4.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 AutoRound | dual-rdma TP2 | v021-ngc2603 |
| `qwen3.5-397b-int4-tq.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 + TurboQuant KV | dual-rdma TP2 | v021-tq |
| `qwen3.6-35b-a3b.env` | Qwen/Qwen3.6-35B-A3B | BF16 hybrid MoE | single TP1 | v022-d568 |
| `qwen3.6-35b-a3b-fi-aot-tp2.env` | Qwen/Qwen3.6-35B-A3B | BF16 hybrid MoE | dual-rdma TP2 | v022-d568-fi-aot |
| `qwen3.6-27b-base-bf16-tp2.env` | Qwen/Qwen3.6-27B (base) | BF16 | dual-rdma TP2 | v022-d568 |
| `qwen3.6-27b-prismascout-nvfp4-tp2.env` | rdtand/Qwen3.6-27B-PrismaSCOUT-NVFP4 | NVFP4 mixed | dual-rdma TP2 | v022-vllm021 |
| `redhatai-122b-nvfp4.env` | RedHatAI/Qwen3.5-122B-A10B-NVFP4 | NVFP4 pre-quantized | single TP1 | v021-ngc2603 |
| `redhatai-122b-nvfp4-tq.env` | RedHatAI/Qwen3.5-122B-A10B-NVFP4 | NVFP4 + TurboQuant KV | single TP1 | v021-tq |
| `wangzhang-122b-fp8.env` | wangzhang/…abliterated | FP8 text-only | dual-rdma TP2 | v021-ngc2603 |
| `wangzhang-122b-nvfp4.env` | wangzhang/…abliterated-NVFP4 | NVFP4 text-only | single TP1 | v021-ngc2603 |
| `wangzhang-122b-abliterix-fp8-tp2.env` | wangzhang/…abliterix | FP8 W8A8 text-only | dual-rdma TP2 | v021-ngc2603 |
| `wangzhang-122b-abliterix-nvfp4-tp2.env` | wangzhang/…abliterix | NVFP4 W4A4 text-only | dual-rdma TP2 | v022-d568 |
| `step37-flash-fp8-v023-tp2.env` | stepfun-ai/Step-3.7-Flash-FP8 | FP8 block · v0.23 tokenizer overlay | dual-rdma TP2 | v023-step3p7 |
| `step37-flash-fp8-tp2.env` | stepfun-ai/Step-3.7-Flash-FP8 | FP8 block | dual-rdma TP2 | v022-d568…step3p7 |

## 5. Experimental presets

Not promoted; tuning/bring-up/A-B and stack-bisection variants.

| Preset | Model / stack | Topology | Note |
|---|---|---|---|
| `deepseek-v4-v023-stack-pr41834-bootstrap-tp2.env` | DeepSeek-V4 v0.23-stack bring-up | dual-rdma TP2 | Bootstrap variant |
| `deepseek-v4-v023-stack-pr41834-graph-tp2.env` | DeepSeek-V4 v0.23-stack graph | dual-rdma TP2 | Graph experiment |
| `deepseek-v4-v023-stack-pr41834-prefixcache-tp2.env` | DeepSeek-V4 v0.23-stack prefix cache | dual-rdma TP2 | Prefix-cache experiment |
| `deepseek-v4-v023-stack-pr41834-reasoning-parser-tp2.env` | DeepSeek-V4 v0.23-stack reasoning parser | dual-rdma TP2 | Reasoning-parser experiment |
| `deepseek-v4-v023-stack-pr41834-tool-parser-tp2.env` | DeepSeek-V4 v0.23-stack tool parser | dual-rdma TP2 | Tool-parser experiment |
| `step37-flash-nvfp4-tp2.env` | Step-3.7-Flash NVFP4 v0.22 (EP-on) | dual-rdma TP2 | Experimental long-context (`STAGE_D_PARTIALLY_VALIDATED_TO_245009`) |
| `qwen3.6-35b-fp16.env` | Qwen/Qwen3.6-35B-A3B FP16 | single TP1 | Experimental FP16 |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022.env` | Qwen3.6-27B PrismaSCOUT NVFP4 · v022 | dual-rdma TP2 | v022 stack A/B (requires `--mm-encoder-tp-mode data`) |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-fi0611.env` | …PrismaSCOUT NVFP4 · v022 FlashInfer 0.6.11 | dual-rdma TP2 | v022 stack-bisection variant |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-nccl234.env` | …PrismaSCOUT NVFP4 · v022 NCCL 2.34 | dual-rdma TP2 | v022 stack-bisection variant |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-ngc2604.env` | …PrismaSCOUT NVFP4 · v022 NGC 26.04 | dual-rdma TP2 | v022 stack-bisection variant |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-trt37.env` | …PrismaSCOUT NVFP4 · v022 TRT 3.7 | dual-rdma TP2 | v022 stack-bisection variant |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-tx581.env` | …PrismaSCOUT NVFP4 · v022 Transformers 5.8.1 | dual-rdma TP2 | v022 stack-bisection variant |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-d568.env` | …PrismaSCOUT NVFP4 · v022-d568 | dual-rdma TP2 | v022-d568 stack variant |

## 6. Historical and reproduction presets

Legacy / reproduction references. Preserved for provenance; not current or recommended.

| Preset | Model / stack | Topology | Note |
|---|---|---|---|
| `dsv4-flash-fp8-tp2.env` | DeepSeek-V4-Flash official FP8 (`dsv4-d568`, legacy/JASL) | dual-rdma TP2 | Legacy DSV4 reproduction ([`docs/dsv4-flash-tp2.md`](../docs/dsv4-flash-tp2.md)); not current production |
| `wangzhang-122b-abliterix-fp8-tp2-v022.env` | wangzhang/…abliterix FP8 · v022 | dual-rdma TP2 | v022 stack reproduction variant |
| `wangzhang-122b-abliterix-fp8-tp2-v022-d568.env` | wangzhang/…abliterix FP8 · v022-d568 | dual-rdma TP2 | v022-d568 stack reproduction variant |

---

## What these files are

Each `.env` file defines model-specific runtime settings passed to
`docker compose --env-file presets/<preset>.env`. Typical settings: `MODEL_PATH`,
`MODEL_CONTAINER_PATH`, `SERVED_MODEL_NAME`, `TP_SIZE`, `CLUSTER_MODE` (`single` or `dual-rdma`),
`VLLM_IMAGE`, plus quantization / MTP / other vLLM flags. See the current-production document and
[`docs/software-stack.md`](../docs/software-stack.md) for configuration internals rather than
duplicating them here.

## Where to store model weights

Keep model weights outside this repository (e.g. `/mnt/data/llm-models/<org>/<model>` or
`/home/<user>/Documents/Models/<model>`). Point the preset to that location by editing `MODEL_PATH`:

```bash
sed -i 's|/path/to/model|/mnt/data/llm-models/deepseek-ai/DeepSeek-V4-Flash|' \
  presets/dsv4-flash-fp8-tp2.env
```

## Directory name

This directory was previously named `models/`. It was renamed to `presets/` (Stage 3-D) to avoid
confusion with actual model weights and container-internal `/models/...` mount paths.

## License and model weights

Preset files are configuration references only. This repository does not distribute model weights;
users are responsible for obtaining weights and complying with upstream model licenses and terms.
