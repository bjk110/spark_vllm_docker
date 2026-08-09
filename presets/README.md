# Model Environment Presets

Docker Compose environment preset files for model-serving configurations. This directory does
**not** contain Hugging Face model weights.

**Status is set by this index, not by the filename.** Each `.env` file also documents its own
recipe/image/topology in its header comment.

Serve a preset directly:

```bash
docker compose --env-file presets/<preset>.env --profile head up -d   # + --profile worker on the worker node
```

Navigation: **[1. Production presets](#1-production-presets)** · [2. Validated (non-production)](#2-validated-presets-non-production)
· [3. General supported](#3-general-supported-presets) · [4. Experimental](#4-experimental-presets) · [5. Historical reproduction](#5-historical-reproduction-presets)

**Selection order (any production model family):** use the family's **active production** preset
for serving; use its **production rollback** preset only to revert. There is no other operational
choice for a promoted model family. Superseded presets (intermediate, experimental, candidate,
legacy, deep-recovery, historical) are **not** retained in the active preset directory — see each
family's subsection below for where that provenance actually lives (Git history for DeepSeek-V4,
local/untracked build-host artifacts referenced by hash for Solar-Open2).

**DeepSeek-V4 and Solar-Open2 are independent production baselines**, each promoted and rolled back
on its own schedule. They are not served simultaneously: both currently target the same physical
serving slot (spark01 head + spark02 worker, port 8000), so only one family's containers run at any
given moment. Which one is *actually* running right now is **not** determined by this index or by
either family's production document — check live container/health state directly (`docker ps`,
`GET :8000/health`, `GET :8000/v1/models`) before assuming either baseline is currently deployed.

---

## 1. Production presets

### 1a. DeepSeek-V4 production presets

DeepSeek-V4-Flash has exactly two production-operable presets: the active native DSpark 64K
production preset and its authoritative MTP1 rollback. Both pin their image by **immutable GHCR
manifest digest**. Full operations and validation provenance:
[`docs/deepseek-v4-production.md`](../docs/deepseek-v4-production.md).

| Preset | Model | Status | Notes |
|---|---|---|---|
| `deepseek-v4-flash-dspark-k7-64k-production-tp2.env` | DeepSeek-V4-Flash-DSpark | **Active production** (since 2026-07-22, spark01:8000) | native DSpark k=7 greedy, target FULL_DECODE_ONLY `[8]`, draft eager, `MAX_MODEL_LEN=65536`, KV 10 GiB FP8, `E8M0=1`, max_num_seqs 1, prefix caching off, TP=2 mp/RoCE. Digest `@sha256:aacb06de60ec…`. No LC131 exposure; unrestricted 135168 not supported |
| `deepseek-v4-flash-mtp1-production-tp2.env` | DeepSeek-V4-Flash | **Production rollback** (stopped) | Authoritative rollback for the active DSpark production. MTP n=1, target FULL_DECODE_ONLY `[2]`, KV 4 GiB FP8, max_num_seqs 1, TP=2 mp/RoCE. Digest `@sha256:de69fa367137…`. Repeated large-context operation not approved |

Superseded DeepSeek-V4 presets (intermediate, experimental, candidate, legacy, deep-recovery,
historical) are available through Git history and are **not** retained in this directory.

### 1b. Solar-Open2 production presets

Solar-Open2-250B has exactly two production-operable presets: the active r4 BF16 production preset
and its authoritative v0.22.1 rollback. Both pin their image by **local Docker image ID** (not yet
published to a registry — see [`docs/images.md`](../docs/images.md) for the distinction). Full
operations and validation provenance: [`docs/solar-open2-production.md`](../docs/solar-open2-production.md).

| Preset | Model | Status | Notes |
|---|---|---|---|
| `solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env` | Solar-Open2-250B-Nota-NVFP4 | **Active production** (since 2026-08-09, spark01:8000 when deployed) | vLLM 0.25.1, TP=2 Ray, BF16 KV fixed 4 GiB/rank (66,764 tok), `MAX_MODEL_LEN=4096`, `MAX_NUM_SEQS=8`, eager, FLASH_ATTN auto, FLASHINFER_B12X MoE, ST_PREAD + B12X shared-workspace gates on. Local image ID `sha256:ecb7bfe3978a…`. 6-gate fast-track PASS, promotion commit `d966925`. |
| `solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env` | Solar-Open2-250B-Nota-NVFP4 | **Production rollback** (stopped) | Authoritative rollback for the active r4 production. vLLM 0.22.1, matched scheduler footprint (`MAX_MODEL_LEN=4096`, `MAX_NUM_SEQS=8`), BF16 KV. Local image ID `sha256:1873d21746…`. Rollback round-trip validated end-to-end (Gate 6). |

Superseded Solar-Open2 presets (r2/r3/r4 diagnostic variants, eager-4k smoke configs, marlin c2
diagnostics, the superseded active-test baseline) are **not** tracked in Git and are **not**
retained in this directory — unlike DeepSeek-V4, this provenance is not recoverable from `git log`
(those files were never committed). They remain as local, untracked, validated-in-place artifacts
on the build hosts (spark01/spark02) and are referenced by content hash only in
[`docs/solar-open2-production.md` section 3](../docs/solar-open2-production.md#3-preset-retention-policy-and-status).
This is a local-only reproducibility limitation for the historical/intermediate development path
only — the production preset is tracked and the rollback preset is working-tree-ready and
hash-verified (not yet staged/committed; see `docs/solar-open2-production.md` sections 1-2).

## 2. Validated presets (non-production)

Validated, but **not** the current default serving path for any production model family.

| Preset | Model / stack | Topology | Status | Use |
|---|---|---|---|---|
| `step37-flash-nvfp4-v023-tp2-latency.env` | Step-3.7-Flash NVFP4 · v0.23 (EP-off, MARLIN, TRITON_ATTN) | dual-rdma TP=2 | Validated (Step-3.7 NVFP4 path) | Recommended Step-3.7 NVFP4 latency path ([bench](../docs/benchmarks/bt-matrix-step37-nvfp4-v023.md)) |

## 3. General supported presets

Production-usable presets for non-DeepSeek-V4/non-Solar-Open2 models on the stable/forward stacks.
Image bases: `v021-ngc2603` / `v021-tq` / `v022-d568` / `v022-d568-fi-aot` / step3p7 (see
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

## 4. Experimental presets

Not promoted; tuning/bring-up/A-B and stack-bisection variants. **None is a production path**, and
each requires separate validation before any operational use.

| Preset | Model / stack | Topology | Note |
|---|---|---|---|
| `step37-flash-nvfp4-tp2.env` | Step-3.7-Flash NVFP4 v0.22 (EP-on) | dual-rdma TP2 | Experimental long-context (`STAGE_D_PARTIALLY_VALIDATED_TO_245009`) |
| `qwen3.6-35b-fp16.env` | Qwen/Qwen3.6-35B-A3B FP16 | single TP1 | Experimental FP16 |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022.env` | Qwen3.6-27B PrismaSCOUT NVFP4 · v022 | dual-rdma TP2 | v022 stack A/B (requires `--mm-encoder-tp-mode data`) |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-fi0611.env` | …PrismaSCOUT NVFP4 · v022 FlashInfer 0.6.11 | dual-rdma TP2 | v022 stack-bisection variant |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-nccl234.env` | …PrismaSCOUT NVFP4 · v022 NCCL 2.34 | dual-rdma TP2 | v022 stack-bisection variant |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-ngc2604.env` | …PrismaSCOUT NVFP4 · v022 NGC 26.04 | dual-rdma TP2 | v022 stack-bisection variant |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-trt37.env` | …PrismaSCOUT NVFP4 · v022 TRT 3.7 | dual-rdma TP2 | v022 stack-bisection variant |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-tx581.env` | …PrismaSCOUT NVFP4 · v022 Transformers 5.8.1 | dual-rdma TP2 | v022 stack-bisection variant |
| `qwen3.6-27b-prismascout-nvfp4-tp2-v022-d568.env` | …PrismaSCOUT NVFP4 · v022-d568 | dual-rdma TP2 | v022-d568 stack variant |

## 5. Historical reproduction presets

Legacy / reproduction references for non-production-family models. Preserved for provenance;
**not current, not recommended for new deployments.**

| Preset | Model / stack | Topology | Note |
|---|---|---|---|
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
  presets/deepseek-v4-flash-mtp1-production-tp2.env
```

## Directory name

This directory was previously named `models/`. It was renamed to `presets/` (Stage 3-D) to avoid
confusion with actual model weights and container-internal `/models/...` mount paths.

## License and model weights

Preset files are configuration references only. This repository does not distribute model weights;
users are responsible for obtaining weights and complying with upstream model licenses and terms.
