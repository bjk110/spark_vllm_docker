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
