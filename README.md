# spark_vllm_docker

## Overview

Unified vLLM serving configuration for NVIDIA DGX Spark (GB10), supporting two topologies from the
same repo / Dockerfile / compose file:

- **Single Spark** (default, zero RDMA setup) — one GB10 box, TP=1.
- **Dual Spark + 200 Gbps RoCE/IB** — two GB10 boxes, TP=2 (Ray or `mp`/SPMD backend).

Pick the topology with `CLUSTER_MODE=single` (default) or `CLUSTER_MODE=dual-rdma` in your `.env`.

**Start here:** the [documentation index](docs/README.md) is the canonical map of all docs and their
status. Preset catalog and status: [`presets/README.md`](presets/README.md). Release/patch detail:
[`CHANGELOG.md`](CHANGELOG.md), [`PATCH_STATUS.md`](PATCH_STATUS.md).

## Hardware and topology

| Topology | Nodes | GPU / memory | Interconnect | Backend |
|---|---|---|---|---|
| `single` | one Spark | NVIDIA GB10 (Blackwell), 119 GiB unified | n/a | direct (no Ray, no `mp`) |
| `dual-rdma` | spark01 (head) + spark02 (worker) | 2× GB10, 119 GiB unified each | 200 Gbps RoCE | `ray` (default) or `mp` |

`dual-rdma` supports two coordination backends via `DISTRIBUTED_BACKEND=ray` (default) or `mp`
(SPMD, no Ray). Full entrypoint dispatch (`CLUSTER_MODE` × `ROLE` × `TP_SIZE` × backend), topology
diagrams, and the backend comparison are in [`docs/architecture.md`](docs/architecture.md).

## Current serving paths

| Path | Status | Backend | Use case |
|---|---|---|---|
| `dsv4-sm121-indexer` | **Current DeepSeek-V4-Flash production baseline** | `mp` | Recommended DSV4 path — SM121 DeepGEMM FP8-Q prefill indexer, MARLIN MoE, production Triton dense/sparse-MLA, dual-node TP=2. Digest-pinned. |
| `dsv4-prefill8192` | **Immediate rollback baseline** (prior production) | `mp` | Rollback target for `dsv4-sm121-indexer` — same envelope without the SM121 indexer. |
| `dsv4-d568` | Frozen legacy/historical DSV4 baseline | `ray` or `mp` | Historical decode-optimized reproduction/reference only. |
| `unholy-fusion` | Experimental (DSV4 only) | `mp` | Higher-prefill DSV4 experimental alternative — not a recommended production path. |
| `v022-d568-ngc2605-tx5102-vllm022` | Active forward-stack (NGC 26.05, vLLM 0.22.1) | `ray` | Qwen3.5-122B-FP8 and other forward-stack models. |
| `v022-d568` | Stable general base (NGC 26.04, vLLM 0.21.0) | `ray` or direct | Qwen3.6, Gemma 4 31B, abliterix NVFP4 presets. |
| `v021-ngc2603` / `v021-tq` | Stable base for most existing presets | `ray` or direct | Most non-DSV4 presets. |

**Current DeepSeek-V4 production** runs the promoted image by its **immutable GHCR manifest digest**
(`sha256:ade810fd…`, config `fa83457d`) via the digest-pinned preset
[`presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env`](presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env).
The mutable alias `dsv4-sm121-indexer-production` is provenance only — not a runtime pin. Full
identity, routing, evidence, rollback, clone guard, and ABI provenance:
[`docs/deepseek-v4-sm121-indexer-production.md`](docs/deepseek-v4-sm121-indexer-production.md).
Rollback procedure: [`docs/deepseek-v4-prefill8192-production-runbook.md`](docs/deepseek-v4-prefill8192-production-runbook.md).

> **`dsv4-d568` is intentionally frozen** and will not be rebased onto NGC 26.05+. Forward-stack
> upgrades are a separate parallel path for non-DSV4 models. `dsv4-d568` (JASL-era) and
> `unholy-fusion` are historical/experimental references, not generally recommended production paths.

Component versions, stack lineage, and digests → [`docs/software-stack.md`](docs/software-stack.md).
Image tag → Git-ref mapping → [`docs/images.md`](docs/images.md). Optional FlashInfer-AOT drop-in
for `v022-d568` → [`docs/flashinfer-aot-prebake.md`](docs/flashinfer-aot-prebake.md).

## Quick Start

Presets live in [`presets/`](presets/) (`.env` files only — no model weights). Keep weights outside
the repo and point `MODEL_PATH` / `MODEL_CONTAINER_PATH` at them.

### 1. Get the image

```bash
# Pick the base for your path (see Current serving paths):
docker pull ghcr.io/bjk110/vllm-spark:v021-ngc2603                      # stable base for most presets
docker pull ghcr.io/bjk110/vllm-spark:v022-d568                        # NGC 26.04 general base
docker pull ghcr.io/bjk110/vllm-spark:v022-d568-ngc2605-tx5102-vllm022 # forward stack (NGC 26.05)
docker pull ghcr.io/bjk110/vllm-spark:dsv4-d568                        # frozen legacy DSV4
```

Building from source (all builds on spark01/spark02): see [`dockerfiles/`](dockerfiles/) and
[`docs/software-stack.md`](docs/software-stack.md).

### 2. Choose a preset

Pick from [`presets/README.md`](presets/README.md) (grouped by status). Single-Spark presets ship
`CLUSTER_MODE=single`/TP=1; dual-Spark presets ship `CLUSTER_MODE=dual-rdma`/TP=2.

```bash
# Current DeepSeek-V4 production (digest-pinned):
docker compose --env-file presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env --profile head up -d   # + worker on spark02

# Or copy any preset to .env and edit MODEL_PATH:
cp presets/redhatai-122b-nvfp4.env .env
```

### 3. Start services

```bash
# Single Spark (TP=1, no Ray/RDMA):
docker compose --profile head up -d

# Dual Spark (TP=2):
docker compose --profile head up -d      # spark01
docker compose --profile worker up -d    # spark02
```

`entrypoint.sh` normalizes the environment by `CLUSTER_MODE` and dispatches on
`ROLE`×`TP_SIZE`×`DISTRIBUTED_BACKEND`. In `single` mode it forces `VLLM_HOST_IP=127.0.0.1` and
`NCCL_IB_DISABLE=1` (avoids the c10d `server socket has timed out` hang — see
[`docs/troubleshooting.md`](docs/troubleshooting.md)). Backend selection (`ray` vs `mp`), the full
dispatch table, and RDMA env requirements are in [`docs/architecture.md`](docs/architecture.md).

### 4. Verify

```bash
curl http://localhost:8000/health      # single
curl http://spark01:8000/health        # dual-rdma
```

## Presets and model paths

All model-serving presets live in [`presets/`](presets/), grouped by status (current production,
rollback, validated, general supported, experimental, historical/reproduction) in
[`presets/README.md`](presets/README.md). Each `.env` documents its own model, image/stack,
topology, and flags in its header. Keep model weights outside the repository and point `MODEL_PATH`
/ `MODEL_CONTAINER_PATH` at them.

## Repository layout

```
vllm-spark/
├── docker-compose.yml        # Unified compose (head + worker profiles)
├── entrypoints/              # Container entrypoints (ENTRYPOINT_FILE); see entrypoints/README.md
├── dockerfiles/              # active/ + legacy/ Dockerfiles; see dockerfiles/README.md
├── presets/                  # .env model-serving presets (not weights); see presets/README.md
├── patches/                  # Build/runtime patches by purpose; see patches/README.md
├── scripts/                  # Cluster bootstrap, verification, diagnostics
├── benchmarks/               # Raw benchmark artifacts; see benchmarks/README.md
├── docs/                     # Documentation — start at docs/README.md
├── CHANGELOG.md              # Release-by-release history
└── PATCH_STATUS.md           # Per-patch purpose / status / removal condition
```

## Configuration

All configuration is via `.env` (see [`.env.example`](.env.example) for full documentation). Key
variables: `VLLM_IMAGE`, `MODEL_PATH`, `MODEL_CONTAINER_PATH`, `SERVED_MODEL_NAME`, `CLUSTER_MODE`,
`TP_SIZE`; for `dual-rdma`: `HEAD_ROCE_IP` / `WORKER_ROCE_IP` / `ROCE_IF_NAME` / `IB_HCA_NAME` and
`DISTRIBUTED_BACKEND` (`ray` default, or `mp` with `MASTER_PORT`); plus `VLLM_EXTRA_ARGS` for
model-specific flags. Active build/runtime patches are tracked in
[`PATCH_STATUS.md`](PATCH_STATUS.md).

## Documentation

The [**documentation index**](docs/README.md) classifies every document by status (current
production, rollback and operations, general stable stacks, model guides, diagnostics, benchmarks,
validated alternatives, experimental, historical/superseded). Start there. Frequently used:

- Current DeepSeek-V4 production: [`docs/deepseek-v4-sm121-indexer-production.md`](docs/deepseek-v4-sm121-indexer-production.md)
- Rollback runbook: [`docs/deepseek-v4-prefill8192-production-runbook.md`](docs/deepseek-v4-prefill8192-production-runbook.md)
- Stack lineage / images: [`docs/software-stack.md`](docs/software-stack.md), [`docs/images.md`](docs/images.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)
- Troubleshooting: [`docs/troubleshooting.md`](docs/troubleshooting.md)
- Presets: [`presets/README.md`](presets/README.md)

## Compatibility and safety notice

- All Docker/vLLM builds run on spark01 or spark02, never on the homeserver (GB10 template
  compilation needs 64–128 GiB peak).
- DeepSeek-V4 production runs at the validated envelope (concurrency 1, prompts up to 131K). Exceeding
  concurrency or context requires KV-headroom re-validation.
- GB10 uses unified memory; a clean reboot + dedicated-cache-clear startup gate is required before a
  full model load when UVM is retained (not automated by presets).
- Recommended OS tuning: `sudo sysctl -w vm.swappiness=10`.

## License

Source code, Dockerfiles, scripts, presets, and documentation are licensed under Apache License 2.0
(see [`LICENSE`](LICENSE)). This repository does **not** distribute model weights; users are
responsible for obtaining weights and complying with upstream model licenses. Container images and
dependencies remain governed by their upstream licenses — see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
