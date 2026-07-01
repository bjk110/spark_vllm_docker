# Model Environment Presets

This directory contains Docker Compose environment preset files for model-serving
configurations.

It does **not** contain actual Hugging Face model weights.

> **Current main Step-3.7 FP8 path:** `step37-flash-fp8-v023-tp2.env` (vLLM 0.23,
> TP=2, EP off, `MAX_MODEL_LEN=8192`, tokenizer overlay enabled; image
> `ghcr.io/bjk110/vllm-spark:v023-step37-tokenizer-overlay-exp-07a2722`). This is
> the validated FP8 baseline, not a global default. The Step-3.7 NVFP4 preset and
> the historical v0.22 FP8 preset are unchanged. See
> [`docs/step3.7-tokenizer-overlay.md`](../docs/step3.7-tokenizer-overlay.md).

> **DeepSeek-V4-Flash MTP n=1 + FULL_DECODE_ONLY (validated candidate):**
> `deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-validated-tp2.env` (vLLM PR #41834,
> dual-Spark TP=2 mp, NET/IB, MTP n=1, FULL decode graph capture `[2]`). Status =
> `VALIDATED_PRESET_CANDIDATE`, **not** production and **not** a replacement for the
> frozen primary DSV4 baseline `dsv4-d568`. Passed safety, performance, a 4-hour soak,
> and an independent cold-start reproduction. Rollback levels: L1 graph-only
> `deepseek-v4-v023-stack-pr41834-fullgraph-validated-rollback-tp2.env` (~27.2 t/s);
> L2 eager U0-RDMA `deepseek-v4-v023-stack-pr41834-eager-u0-rollback-tp2.env` (~7.4 t/s).
> Provenance + operational gates:
> [`docs/deepseek-v4-mtp1-fullgraph-validated-preset.md`](../docs/deepseek-v4-mtp1-fullgraph-validated-preset.md).
> Requires a clean-boot + dedicated-cache-clear startup gate (not automated by the preset).

> **DeepSeek-V4-Flash — prefill-optimized (validated candidate, concurrency 1, up to 131K):**
> `deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-validated-candidate-tp2.env`.
> Status = `VALIDATED_CANDIDATE`, **not** production and **not** current/default serving. The
> only deltas from the validated baseline are `MAX_MODEL_LEN=135168`, fixed KV 4 GiB, and
> `MAX_NUM_BATCHED_TOKENS=8192`. Validated envelope: concurrency 1 only, prompts up to 131,072
> tokens, typical output 128 tokens, prefix cache disabled, MTP n=1, FULL_DECODE_ONLY, capture
> `[2]`, NET/IB over RoCE. Passed an independent harness-corrected cold-start reproduction and a
> 60-minute stability run. Runtime KV headroom must be revalidated before increasing concurrency
> or context. See
> [`docs/deepseek-v4-prefill8192-validated-candidate.md`](../docs/deepseek-v4-prefill8192-validated-candidate.md).

> **DeepSeek-V4-Flash — SM121 DeepGEMM indexer PRODUCTION (current baseline; digest-pinned):**
> `deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env`. Status = **`Current production baseline`**
> (`H1Z_B1AE_PROMOTION_CUTOVER_PASS`, 2026-07-01). This is the recommended DeepSeek-V4-Flash serving
> path. It runs the promoted image by its **immutable GHCR manifest digest**
> `ghcr.io/bjk110/vllm-spark@sha256:ade810fd637e30922a30d09f0fcf128fbeb2a757a27a64f8a77e3646fae223a7`
> (config `fa83457d`). The mutable alias `dsv4-sm121-indexer-production` resolves to the same digest
> but is **provenance only** — the runtime must stay pinned to the digest, not the alias. Runtime
> envelope is identical to the prefill8192 rollback baseline (concurrency 1, prompts up to 131K, MTP
> n=1, FULL_DECODE_ONLY `[2]`, fixed 4 GiB fp8 KV, prefix cache disabled, TP=2 mp, NET/IB over RoCE),
> with the DeepGEMM SM121 FP8-Q prefill indexer active (MARLIN MoE + production Triton dense/sparse-MLA
> retained; `VLLM_MOE_USE_DEEP_GEMM=0`, `VLLM_USE_DEEP_GEMM_E8M0=0`, explicit `--moe-backend marlin`).
> Validated by H1Z-B1AB/B1AC/B1AD/B1AE; prefill within ~1.5% of the B1AB reference (recovers ~48–55%
> of the H1C indexer uplift over the rollback baseline). Immediate rollback target = the prefill8192
> baseline below. SM121 source clones carry a maintenance guard (valid only while SM120 sources
> `cedcce47`/`b3a5d236` hold). Full identity, evidence, rollback, clone guard, and the ABI-hash
> provenance correction: [`docs/deepseek-v4-sm121-indexer-production.md`](../docs/deepseek-v4-sm121-indexer-production.md)
> and [`docs/deepseek-v4-sm121-indexer-promotion-manifest.md`](../docs/deepseek-v4-sm121-indexer-promotion-manifest.md).

> **DeepSeek-V4-Flash — prefill8192 (ROLLBACK BASELINE; prior production, runtime NOT auto-activated):**
> `deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-production-tp2.env`. Status =
> **`Rollback baseline`** (prior production `4c41950c`, superseded by `dsv4-sm121-indexer` on
> 2026-07-01; preserved unchanged as the immediate rollback target). Repository status does **not**
> activate the serving runtime; live activation requires the separate maintenance-window procedure in
> the production runbook. Runtime lines are byte-identical to the validated candidate. Approved envelope: concurrency 1 only,
> prompts up to 131,072 tokens, typical output allowance 128 tokens, fixed 4 GiB fp8 KV, prefix
> cache disabled, MTP n=1, FULL_DECODE_ONLY capture `[2]`, TP=2 multiprocessing, NET/IB over RoCE.
> Requires a clean-boot + dedicated-cache-clear startup gate (not automated by the preset),
> OpenWebUI backend timeout `AIOHTTP_CLIENT_TIMEOUT >= 180` (confirmed effective 240), and healthy
> Prometheus + both Spark node-exporters. Monitoring status is
> `VLLM_MONITORING_CONFIGURED_WITH_LOG_GAPS`: graph fallback, graph recapture, per-rank health,
> NCCL, and CUDA remain operator log checks; Spark cAdvisor/DCGM and homeserver cAdvisor are
> deferred and not active. Activation, acceptance, shutdown, and rollback procedure:
> [`docs/deepseek-v4-prefill8192-production-runbook.md`](../docs/deepseek-v4-prefill8192-production-runbook.md).
> Rollback chain: (1) validated baseline
> `deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-validated-tp2.env`; (2) graph-only L1
> `deepseek-v4-v023-stack-pr41834-fullgraph-validated-rollback-tp2.env`; (3) eager U0-RDMA L2
> `deepseek-v4-v023-stack-pr41834-eager-u0-rollback-tp2.env`.

## What these files are

Each `.env` file in this directory defines model-specific runtime settings passed to
`docker compose --env-file presets/<preset>.env`. Typical settings include:

- `MODEL_PATH` — host path to the model weight directory
- `MODEL_CONTAINER_PATH` — container-internal mount point
- `SERVED_MODEL_NAME` — name served on the OpenAI-compatible API
- `TP_SIZE` — tensor-parallel degree
- `CLUSTER_MODE` — `single` (one DGX Spark) or `dual-rdma` (two nodes over RoCE)
- `VLLM_IMAGE` — which container image to use
- Quantization options, MTP settings, and other vLLM flags

## Where to store actual model weights

Keep model weights outside this repository. Example locations:

```text
/mnt/data/llm-models/deepseek-ai/DeepSeek-V4-Flash
/mnt/data/llm-models/Qwen/<model-name>
/home/<user>/Documents/Models/<model-name>
```

Point the preset to that location by editing `MODEL_PATH` in the chosen `.env` file:

```bash
# Edit directly:
sed -i 's|/path/to/model|/mnt/data/llm-models/deepseek-ai/DeepSeek-V4-Flash|' \
  presets/dsv4-flash-fp8-tp2.env

# Or copy to .env and edit there:
cp presets/dsv4-flash-fp8-tp2.env .env
# then edit MODEL_PATH in .env
```

## Usage

```bash
# Launch with a preset directly (no copy needed):
docker compose --env-file presets/dsv4-flash-fp8-tp2.env --profile head up -d

# Or copy to .env and use the default:
cp presets/redhatai-122b-nvfp4.env .env
docker compose --profile head up -d
```

## Directory name

This directory was previously named `models/`. It was renamed to `presets/` in
Stage 3-D to avoid confusion with actual model weights and container-internal
`/models/...` mount paths. Container-internal paths such as
`MODEL_CONTAINER_PATH=/models/DeepSeek-V4-Flash` are unrelated to this directory
and remain unchanged.

## License and model weights

Preset files in this directory are configuration references only.

This repository does not distribute model weights. Users are responsible for obtaining model weights and complying with the applicable upstream model licenses and terms.
