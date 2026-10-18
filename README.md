# spark_vllm_docker

## Overview

Unified vLLM serving recipes for NVIDIA DGX Spark (GB10). The repository supports two deployment
topologies from a shared Compose base:

- **Single Spark** — one GB10 node, normally TP=1.
- **Dual Spark over 200 Gbps RoCE** — head + worker, normally TP=2 with Ray or `mp`/SPMD.

A runnable model path is a matched set of artifacts, not a single `.env` file:

1. a preset under [`presets/`](presets/);
2. `docker-compose.yml` plus every overlay named by the preset's **Required overlay** header;
3. the referenced image, entrypoint, patches, model files, and model-specific runbook.

Do not mix presets, overlays, or entrypoints from different model families. The preset catalog is
[`presets/README.md`](presets/README.md), and the documentation map is
[`docs/README.md`](docs/README.md). Release and patch details live in
[`CHANGELOG.md`](CHANGELOG.md) and [`PATCH_STATUS.md`](PATCH_STATUS.md).

The evidence-backed executable set is defined by [`recipes/manifest.json`](recipes/manifest.json).
Its image distribution field is separate from validation status:

- **`portable`** — pullable immutable registry digest;
- **`tag-only`** — pullable tag, but not an immutable content identity;
- **`local-only`** — image or orchestration exists only on the qualification hosts.

Use local-only entries as configuration/evidence references unless you have the documented host
artifacts. They are not advertised as fresh-clone downloads.

## Hardware and topology

| Topology | Nodes | GPU / memory | Interconnect | Coordination |
|---|---:|---|---|---|
| `single` | 1 | NVIDIA GB10, 119 GiB unified memory | n/a | direct |
| `dual-rdma` | 2 | 2× GB10, 119 GiB unified memory each | 200 Gbps RoCE | Ray or `mp`/SPMD |

The base entrypoint dispatches on `CLUSTER_MODE`, `ROLE`, `TP_SIZE`, and `DISTRIBUTED_BACKEND`.
Architecture and networking details are documented in [`docs/architecture.md`](docs/architecture.md).

## Representative serving paths

The table is an index, not a launch command or statement about live process state. Each row has its
own image and runtime contract; only one large-model stack should occupy the shared serving slot at a
time.

| Representative path | Model family | Typical topology / backend | Start here |
|---|---|---|---|
| `dsv4-flash-0731-v027` | DeepSeek-V4 Flash | dual-RDMA, `mp` | [Preset catalog](presets/README.md) · [model runbook](docs/deepseek-v4-production.md) |
| `solar-open2-r4-bf16` | Solar-Open2-250B; local-only reference | dual-RDMA, Ray | [`presets/solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env`](presets/solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env) · [model runbook](docs/solar-open2-production.md) |
| `qwen3.8-flash-next-c1-c2` | Qwen3.8 Flash Next official; manual profile, manual activation, not auto-started | dual-RDMA, `mp` | [Preset catalog](presets/README.md) · [model guide](docs/qwen3.8-flash-next-tp2.md) |
| `qwen3.8-uncensored-k2-c2` | Qwen3.8 Flash Next Uncensored FP8 | dual-RDMA, Ray | [`presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env`](presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env) · [`compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml`](compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml) · [model runbook](docs/qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md) |

Exact image identities, stack lineage, and registry/local-image distinctions are maintained in
[`docs/images.md`](docs/images.md) and [`docs/software-stack.md`](docs/software-stack.md). Detailed
performance results and qualification history belong in model runbooks and retained evidence, not in
this top-level launch guide.

## Quick Start

### 0. Clone and check prerequisites

```bash
git clone https://github.com/bjk110/spark_vllm_docker.git
cd spark_vllm_docker
docker compose version
python3 --version
```

Docker Compose **2.24.4 or newer** is required because some exact-environment overlays use
`!override`. On every participating Spark, verify the NVIDIA container runtime and the network/RDMA
devices required by the selected model runbook.

List only the evidence-backed executable recipes:

```bash
./scripts/recipe list --nodes 1
./scripts/recipe list --nodes 2
./scripts/recipe show <recipe-id>
```

For a pullable recipe, download the exact image on every participating node and render without
starting containers:

```bash
./scripts/recipe pull <recipe-id>
./scripts/recipe render <recipe-id> --model-path /absolute/path/to/model
```

`pull` fails closed for local-only recipes. Model weights are not stored in this repository; follow
the recipe's model runbook for source, revision, checksum or revision-marker requirements. The CLI
currently performs discovery, image pull, and config-only rendering. Launch remains an explicit
operator action using the reviewed render and the backend-specific order below.

### 1. Select a preset

Choose an executable recipe with [`scripts/recipe`](scripts/recipe), or consult
[`presets/README.md`](presets/README.md) for general, experimental, and historical references. Then
read the preset header and linked model runbook before running any command. Treat these fields as one
contract:

- `VLLM_IMAGE`, `ENTRYPOINT_FILE`, and model identity;
- `CLUSTER_MODE`, `TP_SIZE`, and `DISTRIBUTED_BACKEND`;
- model length, concurrency, batched-token, KV-cache, and backend settings;
- **Required overlay** and linked runbook;
- status, limits, startup order, and rollback path.

Historical filenames are identifiers, not instructions to substitute another image or overlay. If a
rollback is needed, use the rollback preset and its own required overlays as a complete set.

### 2. Prepare and render

Keep model weights outside this repository. On every participating node:

1. use the same clean repository revision;
2. make the preset's immutable image available;
3. verify `MODEL_PATH`, model revision markers, cache paths, and RoCE device names required by the
   selected runbook;
4. render the final Compose configuration without starting containers.

Example render from the repository root:

```bash
PRESET='presets/selected-model.env'
OVERLAY='compose/required-overlay.yml'  # replace both values; omit the second -f when none is required

env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose --env-file "$PRESET" \
  -f docker-compose.yml -f "$OVERLAY" \
  --profile '*' config
```

Before launch, inspect the rendered image, entrypoint, mounts, role addresses, backend, context,
concurrency, KV budget, and restart policy. A render failure or unexpected empty value is a
configuration error; do not compensate by inventing an environment override.

### 3. Start by topology and backend

Use the exact preset and overlay arguments that passed rendering.

#### Single Spark

Start only the head profile:

```bash
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose --env-file "$PRESET" \
  -f docker-compose.yml -f "$OVERLAY" \
  --profile head up -d
```

#### Dual Spark with `DISTRIBUTED_BACKEND=ray`

Start the head first, then join the worker within the runbook's bounded join timeout:

```bash
# head node
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose --env-file "$PRESET" -f docker-compose.yml -f "$OVERLAY" \
  --profile head up -d

# worker node
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose --env-file "$PRESET" -f docker-compose.yml -f "$OVERLAY" \
  --profile worker up -d
```

#### Dual Spark with `DISTRIBUTED_BACKEND=mp`

Start the worker first, then the head. Both containers are ranks of one TP service, not independent
servers:

```bash
# worker node
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose --env-file "$PRESET" -f docker-compose.yml -f "$OVERLAY" \
  --profile worker up -d

# head node
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose --env-file "$PRESET" -f docker-compose.yml -f "$OVERLAY" \
  --profile head up -d
```

A model runbook may add preflight, prewarm, revision, or cache requirements. Those requirements are
additive and take precedence over these generic examples.

### 4. Verify readiness

`/health` proves that the API process is responding; it does not by itself prove model identity,
capacity, correctness, or completion of an optional prewarm job. Verify all applicable gates:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/metrics | \
  grep -E 'vllm:num_requests_(running|waiting)|vllm:gpu_cache_usage_perc'

docker inspect vllm-spark-head \
  --format 'running={{.State.Running}} restart={{.RestartCount}} oom={{.State.OOMKilled}}'
```

Require:

- `/health` succeeds and `/v1/models` reports the expected served model and context length;
- head and required worker containers are running with `OOMKilled=false` and `RestartCount=0`;
- cluster membership matches the selected topology;
- a bounded non-thinking generation returns HTTP 200 and expected content;
- required tool, structured-output, multimodal, long-context, concurrency, MTP, or prewarm checks from
  the model runbook pass;
- container and kernel logs contain no fatal CUDA, NCCL, EngineCore, OOM, Xid, or fallen-GPU signal.

Cold loading can take a long time on unified memory. Forward progress without OOM, restart, API
failure, or fatal logs is not by itself a failure. Use the selected runbook's timeout and hard-failure
boundary rather than a timeout copied from another model.

### 5. Stop, switch, or roll back

Use the same preset and overlay set used for launch. Stop the head first, then the worker:

```bash
# head node
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose --env-file "$PRESET" -f docker-compose.yml -f "$OVERLAY" \
  --profile head down

# worker node, when present
env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" \
  docker compose --env-file "$PRESET" -f docker-compose.yml -f "$OVERLAY" \
  --profile worker down
```

Before loading another large model or changing backend:

1. confirm containers and distributed-runtime processes are gone;
2. inspect host memory and swap;
3. perform the runbook's cache cleanup or clean reboot when unified-memory state remains;
4. render the next preset and overlays again;
5. follow that backend's startup order.

Never activate a rollback by changing only `VLLM_IMAGE` or only the preset. A rollback may require a
different entrypoint, overlay, launch order, or validation boundary.

## Repository layout

```text
vllm-spark/
├── docker-compose.yml        # Shared head + worker Compose base
├── compose/                  # Model-specific Compose overlays
├── entrypoints/              # Runtime entrypoints selected by presets
├── dockerfiles/              # Image definitions and provenance
├── presets/                  # Model-serving .env contracts; no weights
├── patches/                  # Build/runtime patches grouped by purpose
├── scripts/                  # Bootstrap, verification, and diagnostics
├── benchmarks/               # Benchmark tooling and retained summaries
├── docs/                     # Architecture, runbooks, operations, and history
├── CHANGELOG.md
└── PATCH_STATUS.md
```

## Configuration principles

- Presets contain configuration, not model weights or credentials.
- Prefer immutable registry digests; distinguish them from local Docker image IDs.
- Keep model mounts and pinned source overlays read-only.
- Use a clean shell environment when rendering so ambient variables cannot override the preset.
- Treat image, model revision, source overlay, backend, topology, context, concurrency, KV budget, and
  decoding mode as one measured runtime contract.
- Do not add restart/autostart policy unless the selected recipe explicitly defines and verifies it.

The complete variable reference is [`.env.example`](.env.example). Common variables include
`VLLM_IMAGE`, `MODEL_PATH`, `MODEL_CONTAINER_PATH`, `SERVED_MODEL_NAME`, `CLUSTER_MODE`, `TP_SIZE`,
`DISTRIBUTED_BACKEND`, `HEAD_ROCE_IP`, `WORKER_ROCE_IP`, `ROCE_IF_NAME`, `IB_HCA_NAME`,
`MASTER_PORT`, and `VLLM_EXTRA_ARGS`.

## Documentation

- Documentation index: [`docs/README.md`](docs/README.md)
- Preset catalog: [`presets/README.md`](presets/README.md)
- Architecture and backend behavior: [`docs/architecture.md`](docs/architecture.md)
- Images and stack lineage: [`docs/images.md`](docs/images.md), [`docs/software-stack.md`](docs/software-stack.md)
- Troubleshooting: [`docs/troubleshooting.md`](docs/troubleshooting.md)
- DeepSeek-V4 model operations: [model runbook](docs/deepseek-v4-production.md)
- Solar-Open2 model operations: [`docs/solar-open2-production.md`](docs/solar-open2-production.md)
- Qwen3.8 Uncensored FP8 model operations: [`docs/qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md`](docs/qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md)

## Compatibility and safety notice

- GB10 uses unified memory. Follow the selected runbook's clean-start and recovery requirements before
  a full model load when UVM or swap pressure remains.
- A model recipe can require elevated container capabilities or host-device mounts. Review its rendered
  Compose configuration before launch and use it only on trusted nodes with trusted artifacts.
- Recommended host tuning: `sudo sysctl -w vm.swappiness=10`.

## License

Source code, Dockerfiles, scripts, presets, and documentation are licensed under Apache License 2.0
(see [`LICENSE`](LICENSE)). This repository does **not** distribute model weights; users are
responsible for obtaining weights and complying with upstream model licenses. Container images and
dependencies remain governed by their upstream licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
