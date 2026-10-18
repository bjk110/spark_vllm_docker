# Qwen3.8 Flash Next Uncensored FP8 — dual DGX Spark, MTP k=2/C2

| Field | Qualified value |
|---|---|
| Status | **Validated; manual activation only; not auto-started** |
| Model | `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8` |
| Model revision | `8bebbc05cbe65099f37c520865463c1377d8050d` |
| Image | `ghcr.io/bjk110/vllm-spark@sha256:1308dca36201a271adbb928b7b2927fb0c049e5a6d6748900991e03484a09de5` |
| Image ID after pull | `sha256:518c1b8f9bdaca735d793ffef1c7d6e050bdb62e6ffc9730e9428052cbc59512` |
| vLLM | commit `e77daef89e18e08321ae7b8b24827eedd5fe8673` |
| Ray | `2.59.0` with `cgraph` extra |
| Topology | two DGX Spark nodes, TP=2, EP=2, `ray` over RoCE |
| Serving contract | MTP k=2, `MAX_NUM_SEQS=2`, `MAX_MODEL_LEN=262144`, batched tokens 8192 |
| Preset | [`presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env`](../presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env) |
| Compose overlay | [`compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml`](../compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml) |

This is the validated configuration for the measured profile. It preserves multimodal capability:
do **not** add `--language-model-only`. The recipe publishes exact configuration and source bytes,
and pins the runtime image by a pullable immutable GHCR manifest digest.

## 1. What was qualified

### Ray canonical arm — 2026-10-18

- Execution contract: eager, TP=2/EP=2, MTP k=2, `MAX_NUM_SEQS=2`, BT8K, fixed 8 GiB KV/rank.
- Two-node Ray readiness, deterministic short/tool calls, and MTP draft/accepted-token activity passed.
- A 262,112-token semantic retrieval replay returned HTTP 200 and recovered `KESTREL-74291`.
- Nine synchronized C2 pairs passed across prose, code, and JSON; every pair reached
  `vllm:num_requests_running=2`.
- No Docker/kernel OOM, `OOMKilled`, restart, CUDA/NCCL fatal, Xid, traceback, or request failure
  occurred. Four `triton_kernels.matmul_ogs` optional-import warnings matched an over-broad scanner
  pattern but also occur on MP and did not coincide with an EngineCore death or service failure.
- The matched Ray/MP C2 gate passed for every workload. Ray retained 95.47–98.96% of MP aggregate
  decode, with E2E ratios 100.97–102.82% and TTFT ratios 96.18–98.40%. See the
  [matched report](../benchmarks/evidence/retained/qwen38-uncensored-k2-c2-ray-matched-20261018/report.md).

### MP rollback arm — 2026-10-16

- KV capacity: 548,909 tokens; nominal 262K concurrency 2.09×.
- Vision primitive/compositional requests, canonical text, reasoning, coding, required tool calls,
  structured output, and 10/10 tool continuation checks passed. This is not a text-only recipe.
- One 262,112-token prompt and an actual pair of simultaneous 262,112-token prompts both returned
  HTTP 200 and recovered `KESTREL-74291` with exact token accounting.
- Actual C2 was measured from the server: synchronized pairs reached
  `vllm:num_requests_running=2`. `MAX_NUM_SEQS=2` alone is not evidence of C2 execution.
- Full `llama-benchy 0.4.0` matrix: `pp512/2048/8192 × tg128 × c1/c2/c4/c8 × 3`, 12/12
  benchmark objects (24 consolidated pp/tg rows), 135/135 requests, errors 0. Because the server cap
  is C2, c4/c8 characterize client overlap and queue pressure, not four- or eight-request engine
  execution.
- Request count: `3 repeats × (1+2+4+8 requests) × 3 prompt sizes = 135`.
- Full-matrix MTP acceptance: 60.70%; accepted tokens/iteration: 1.214. Acceptance is activation
  evidence, not a standalone performance claim.
- No Docker/kernel OOM, `OOMKilled`, restart, CUDA/NCCL/engine fatal, Xid, or request failure occurred.
- Readiness-time `NV_ERR_NO_MEMORY` messages were observed in the MP cold-start epochs. They did
  not continue after readiness and did not coincide with OOM, restart, fatal, request failure, or
  increasing idle memory use. Treat them as startup warnings only under those same conditions.

### MP rollback full-matrix performance reference

Each row below summarizes one MP rollback benchmark object by showing its prompt-processing and
generation metrics together. The standard consolidated report contains the corresponding 24 pp/tg rows. These
are measured end-to-end reference values for the exact pinned MP rollback profile, not the Ray image
and not universal targets:

| Prompt | Client concurrency | Prefill tok/s | Aggregate decode tok/s | Per-request decode tok/s | TTFT ms |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 1294.41 | 20.70 | 20.70 | 637.40 |
| 512 | 2 | 1179.94 | 40.16 | 20.79 | 801.51 |
| 512 | 4 | 225.86 | 36.49 | 19.96 | 4372.31 |
| 512 | 8 | 167.07 | 34.80 | 19.94 | 10631.60 |
| 2048 | 1 | 1844.11 | 22.38 | 22.38 | 1358.18 |
| 2048 | 2 | 1721.30 | 37.48 | 20.13 | 2120.88 |
| 2048 | 4 | 715.12 | 32.37 | 19.18 | 6248.33 |
| 2048 | 8 | 567.83 | 30.75 | 18.64 | 14588.99 |
| 8192 | 1 | 1795.29 | 21.21 | 21.21 | 4810.60 |
| 8192 | 2 | 1717.02 | 27.79 | 18.19 | 7652.42 |
| 8192 | 4 | 1289.02 | 21.89 | 17.19 | 15843.16 |
| 8192 | 8 | 1153.52 | 18.20 | 15.24 | 30522.97 |

Best observed rows were 1844.11 prefill tok/s at pp2048/C1, 40.16 aggregate decode tok/s at
pp512/C2, 22.38 per-request decode tok/s at pp2048/C1, and 637.40 ms TTFT at pp512/C1. The c4/c8
rows show that pressure above the two-slot server cap increases queueing and tail latency; use C2 as
the qualified operating envelope.

The byte-pinned public summary is
[`benchmarks/evidence/retained/qwen38-uncensored-k2-c2-fullbench-20261016/report.md`](../benchmarks/evidence/retained/qwen38-uncensored-k2-c2-fullbench-20261016/report.md).

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

This recipe requires **Docker Compose 2.24.4 or newer**. Its overlay uses
`environment: !override` to replace the generic base environment with the exact qualified Ray
closure. Do not convert it back to ordinary merge semantics: Ray 2.59 and FlashInfer can treat
present-but-empty variables as malformed active settings, and unrelated base tuning can also change
the NCCL path. The verifier requires exact rendered environment equality for both roles.

On both nodes, pull and verify the image and model before launch:

```bash
docker pull \
  ghcr.io/bjk110/vllm-spark@sha256:1308dca36201a271adbb928b7b2927fb0c049e5a6d6748900991e03484a09de5

test "$(docker image inspect \
  ghcr.io/bjk110/vllm-spark@sha256:1308dca36201a271adbb928b7b2927fb0c049e5a6d6748900991e03484a09de5 \
  --format '{{.Id}}')" = \
  sha256:518c1b8f9bdaca735d793ffef1c7d6e050bdb62e6ffc9730e9428052cbc59512

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

## 5. Ray head-first launch

Start the head first. Its entrypoint starts the Ray control plane and waits for both nodes:

```bash
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose \
  --env-file presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env \
  -f docker-compose.yml \
  -f compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml \
  --profile head up -d
```

Then join the worker before the head's bounded join timeout expires:

```bash
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose \
  --env-file presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env \
  -f docker-compose.yml \
  -f compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml \
  --profile worker up -d
```

Recent qualified clean-boot and rollback epochs reached readiness in roughly 40–46 minutes. Do not
infer failure from elapsed time alone; require forward progress, bounded memory, and no hard-failure
signal. A materially different host/cache state can change this duration.

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

Run a canonical non-thinking request so a short output limit cannot be consumed by hidden reasoning:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8",
    "messages":[{"role":"user","content":"Reply with exactly QWEN38_BASELINE_OK"}],
    "temperature":0,
    "max_tokens":32,
    "chat_template_kwargs":{"enable_thinking":false}
  }'
```

Require HTTP 200, `finish_reason="stop"`, and exact content `QWEN38_BASELINE_OK`.

Use a forced tool call to verify the tool path:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8",
    "messages":[{"role":"user","content":"Call record_marker with marker KESTREL-74291."}],
    "tools":[{"type":"function","function":{"name":"record_marker","description":"Record a marker","parameters":{"type":"object","properties":{"marker":{"type":"string"}},"required":["marker"]}}}],
    "tool_choice":{"type":"function","function":{"name":"record_marker"}},
    "temperature":0,
    "max_tokens":128,
    "chat_template_kwargs":{"enable_thinking":false}
  }'
```

Require HTTP 200, a `record_marker` tool call, and JSON arguments containing
`"marker":"KESTREL-74291"`. For structured output, send the same marker task with an OpenAI
`response_format` JSON schema and require schema-valid JSON; do not accept free-form prose as a
structured-output pass.

For a multimodal smoke test, set `IMAGE_URL` to a locally controlled image reachable from the serving
container and ask a known-answer question using an OpenAI content array with one `image_url` item and
one text item. Require HTTP 200, nonempty content, and the expected image-specific answer. Do not use
`--language-model-only`; that would create a different profile.

For an operational C2 claim, launch two bounded overlapping requests and sample the exact
`vllm:num_requests_running` series while they are active; require a nonempty sample with peak 2 and
validate both responses independently. Changing a local path, image, source, model, or multimodal
setting requires re-running the applicable checks; the retained qualification does not cover modified
bytes.

### Hard-failure boundary

Stop and investigate if any of the following occurs: Docker/kernel OOM, `OOMKilled`, restart,
userspace CUDA allocation fatal, engine/NCCL fatal, Xid, fallen-off GPU, node unresponsiveness,
health/readiness/API failure, correctness/tool/context failure, repeated post-readiness allocation
errors, or continuously increasing memory/swap. A bounded readiness-time allocation warning without
any of those symptoms is a warning, not by itself a failed epoch.

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

After an authorized clean reboot, launch MP rollback worker-first with its dedicated preset and
overlay. The rollback overlay preserves the qualified MP environment and must not be replaced by the
Ray overlay:

```bash
# spark02 — MP worker first
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose \
  --env-file presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2-mp-rollback.env \
  -f docker-compose.yml -f compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2-mp-rollback.yml \
  --profile worker up -d

# spark01 — MP head second
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose \
  --env-file presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2-mp-rollback.env \
  -f docker-compose.yml -f compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2-mp-rollback.yml \
  --profile head up -d
```

Neither preset configures autostart. Reboot before switching backend or another large model when
UMA/swap pressure remains.
