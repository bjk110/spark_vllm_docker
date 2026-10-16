# Qwen3.8 Flash Next Uncensored FP8 — dual DGX Spark, MTP k=2/C2

| Field | Qualified value |
|---|---|
| Status | **Validated (non-production); manual activation only; not auto-started** |
| Model | `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8` |
| Model revision | `8bebbc05cbe65099f37c520865463c1377d8050d` |
| Image | `ghcr.io/bjk110/vllm-spark@sha256:928d86903dfce3b11203f1f6d3118aa47998e55fee36fc73423e3bef8ab04cd6` |
| Image ID after pull | `sha256:8bb958ffb833c407cd642d72c3938c0f8f0ed31136851c1edcd868bf135651c4` |
| vLLM | commit `e77daef89e18e08321ae7b8b24827eedd5fe8673` |
| Topology | two DGX Spark nodes, TP=2, EP=2, `mp` over RoCE |
| Serving contract | MTP k=2, `MAX_NUM_SEQS=2`, `MAX_MODEL_LEN=262144`, batched tokens 8192 |
| Preset | [`presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env`](../presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env) |
| Compose overlay | [`compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml`](../compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml) |

This is the validated configuration for the measured profile. It preserves multimodal capability:
do **not** add `--language-model-only`. The recipe publishes exact configuration and source bytes,
and pins the runtime image by a pullable immutable GHCR manifest digest.

## 1. What was qualified

- KV capacity: 548,909 tokens; nominal 262K concurrency 2.09×.
- Canonical, reasoning, coding, tool calls, and 10/10 tool continuation checks passed.
- One 262,112-token prompt and an actual pair of simultaneous 262,112-token prompts both returned HTTP 200 and recovered `KESTREL-74291`.
- Every synchronized performance pair reached server-side `running=2`.
- MTP workload acceptance: 74.56%; 1.491 accepted tokens/iteration.
- No Docker/kernel OOM, `OOMKilled`, restart, CUDA/NCCL/engine fatal, Xid, or request failure occurred.
- Two readiness-time `NV_ERR_NO_MEMORY` messages were recorded on the worker node. They did not continue after readiness and did not coincide with an OOM, restart, fatal error, request error, or increasing idle memory use; they remain a warning, not a clean-start claim.

The published profile is **k=2/C2**. Its measured aggregate decode throughput was 58.94 tok/s
for prose, 73.87 tok/s for code, and 78.14 tok/s for JSON workloads.

C2 is an aggregate-throughput profile. Relative to C1, aggregate throughput increased about
50–53%, while per-request decode throughput fell about 22% and TTFT increased about 54–61%.
Choose it when two-request aggregate throughput matters; do not present it as a single-request
latency improvement.

## 2. Prerequisites

1. Two DGX Spark nodes with working RoCE addresses `10.10.10.1` and `10.10.10.2`.
2. The exact model revision on both nodes at the path in the preset.
3. The exact immutable registry manifest pulled on both nodes.
4. This repository at the same clean commit on both nodes.
5. No conflicting containers or listeners on ports 8000 and 29538.
6. A clean reboot before a cold launch when previous large-model UVM state may remain.
7. At least 35 GiB free in Docker's storage filesystem before a first pull. The manifest contains
   12.81 GiB of compressed layers and expands to a larger local image.

The repository does not distribute model weights.

Download the pinned model once on a storage host, record the revision, and then copy the completed
directory to both serving nodes using your normal verified transfer procedure:

```bash
MODEL_DIR=/mnt/data/llm-models/orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8
MODEL_REVISION=8bebbc05cbe65099f37c520865463c1377d8050d
hf download orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8 \
  --revision "${MODEL_REVISION}" --local-dir "${MODEL_DIR}"
hf cache verify orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8 \
  --revision "${MODEL_REVISION}" --local-dir "${MODEL_DIR}" \
  --fail-on-missing-files
printf '%s\n' "${MODEL_REVISION}" > "${MODEL_DIR}/.hf-revision"
```

## 3. Static verification

Run from the repository root:

```bash
python3 scripts/diag/verify_qwen38_ngc2609_mtp_k2_c2_recipe.py
```

The verifier checks pinned identities, exact source hashes, the full overlay shape, read-only source
and model mounts for both roles, the entrypoint flags, and the rendered Compose contract. It fails if
Docker Compose is unavailable or a conflicting ambient shell variable could override the preset. It
never starts a container.

On both nodes, pull and verify the image and model before launch:

```bash
docker pull \
  ghcr.io/bjk110/vllm-spark@sha256:928d86903dfce3b11203f1f6d3118aa47998e55fee36fc73423e3bef8ab04cd6

test "$(docker image inspect \
  ghcr.io/bjk110/vllm-spark@sha256:928d86903dfce3b11203f1f6d3118aa47998e55fee36fc73423e3bef8ab04cd6 \
  --format '{{.Id}}')" = \
  sha256:8bb958ffb833c407cd642d72c3938c0f8f0ed31136851c1edcd868bf135651c4

test -f /mnt/data/llm-models/orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8/config.json
test "$(cat /mnt/data/llm-models/orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8/.hf-revision)" = \
  8bebbc05cbe65099f37c520865463c1377d8050d
```

## 4. Render before launch

```bash
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose \
  --env-file presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env \
  -f docker-compose.yml \
  -f compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml \
  --profile '*' config --quiet
```

The clean environment prevents shell variables from silently overriding `--env-file`. Inspect the
rendered configuration if local paths or interface names differ. Changing the image, source overlay,
model revision, MTP depth, context, concurrency, KV budget, or prefill batch size requires
requalification.

### Container privilege boundary

The qualified closure adds `IPC_LOCK`, `SYS_PTRACE`, and `label=disable` to both roles. `IPC_LOCK`
supports the pinned-memory runtime path; `SYS_PTRACE` and disabled security labeling are retained
strictly for parity with the container configuration that was qualified. They increase container
privilege. Use these settings only on trusted serving nodes with trusted model artifacts. Removing
either setting creates a different, unqualified profile and requires a new validation run.

## 5. Worker-first launch

Run the worker command on the worker node first:

```bash
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose \
  --env-file presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env \
  -f docker-compose.yml \
  -f compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml \
  --profile worker up -d
```

Then run the head command on the head node:

```bash
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose \
  --env-file presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env \
  -f docker-compose.yml \
  -f compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml \
  --profile head up -d
```

Cold readiness was about 31–33 minutes with the qualified prefault overlay. Do not infer failure from elapsed time alone; require forward progress, bounded memory, and no hard-failure signal.

## 6. Required post-start checks

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/metrics | \
  grep -E 'vllm:num_requests_running|vllm:spec_decode_num_(draft|accepted)'
```

Confirm:

- model name and `max_model_len=262144`;
- head and worker are running, `OOMKilled=false`, `RestartCount=0`;
- logs report `num_spec_tokens=2`, `max_num_seqs=2`, batched tokens 8192, and KV capacity 548,909;
- a short non-thinking request succeeds;
- MTP draft and accepted-token counters increase;
- no post-readiness OOM/fatal/Xid/request error appears.

## 7. Stop and rollback

Stop head first, then worker:

```bash
# head node
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose --env-file presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env \
  -f docker-compose.yml -f compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml \
  --profile head down

# worker node
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose --env-file presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env \
  -f docker-compose.yml -f compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml \
  --profile worker down
```

This recipe does not alter repository defaults or configure autostart. Reboot before switching to another large model if UMA/swap pressure remains.
