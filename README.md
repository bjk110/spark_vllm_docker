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
| `dsv4-sm121-indexer` | **Current DeepSeek-V4-Flash production baseline** (graph-safe, H1Z-P54A) | `mp` | Recommended DSV4 path — SM121 DeepGEMM FP8-Q prefill indexer, MARLIN MoE, production Triton dense/sparse-MLA, dual-node TP=2. Digest-pinned to the graph-safe image (`@de69fa367137`); previous `@ade810fd` baseline kept as the legacy rollback preset. |
| `dsv4-prefill8192` | **Immediate rollback baseline** (prior production) | `mp` | Rollback target for `dsv4-sm121-indexer` — same envelope without the SM121 indexer. |
| `dsv4-d568` | Frozen legacy/historical DSV4 baseline | `ray` or `mp` | Historical decode-optimized reproduction/reference only. |
| `unholy-fusion` | Experimental (DSV4 only) | `mp` | Higher-prefill DSV4 experimental alternative — not a recommended production path. |
| `v022-d568-ngc2605-tx5102-vllm022` | Active forward-stack (NGC 26.05, vLLM 0.22.1) | `ray` | Qwen3.5-122B-FP8 and other forward-stack models. |
| `v022-d568` | Stable general base (NGC 26.04, vLLM 0.21.0) | `ray` or direct | Qwen3.6, Gemma 4 31B, abliterix NVFP4 presets. |
| `v021-ngc2603` / `v021-tq` | Stable base for most existing presets | `ray` or direct | Most non-DSV4 presets. |

**Current DeepSeek-V4 production (default recommended recipe)** runs the validated **graph-safe**
image by its **immutable GHCR manifest digest**
(`sha256:de69fa367137…`, config `sha256:5bb962a9055d…`) via the digest-pinned preset
[`presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env`](presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env).
This preset was promoted from the previous `@ade810fd` baseline (H1Z-P54A) by changing **only**
`VLLM_IMAGE` to the graph-safe digest; all serving settings (`max_num_seqs=1`, capture `[2]`, 4 GiB
KV, TP=2 mp/RoCE, MTP n=1, MARLIN) are unchanged. The graph-safe delta is a **one-file installed
`ops/sm12x_mqa.py` signature patch** (PR #18, `tl.constexpr` 83→64) — a repository recipe-level
patched image, **not** an upstream vLLM fix, with **no performance-improvement claim** and **no**
GHCR `latest`/`stable`/`production` tag movement. Prefer this digest-pinned preset over floating tags.
The previous baseline is preserved as an explicit **legacy rollback** preset
[`presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-legacy-ade810fd-tp2.env`](presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-legacy-ade810fd-tp2.env)
(`@ade810fd`, config `fa83457d`) — use it to return to the pre-graph-safe image without a rebuild.
The mutable alias `dsv4-sm121-indexer-production` is provenance only — not a runtime pin. Full
identity, routing, evidence, rollback, clone guard, and ABI provenance:
[`docs/deepseek-v4-sm121-indexer-production.md`](docs/deepseek-v4-sm121-indexer-production.md).
Rollback procedure: [`docs/deepseek-v4-prefill8192-production-runbook.md`](docs/deepseek-v4-prefill8192-production-runbook.md).

> **`dsv4-d568` is intentionally frozen** and will not be rebased onto NGC 26.05+. Forward-stack
> upgrades are a separate parallel path for non-DSV4 models. `dsv4-d568` (JASL-era) and
> `unholy-fusion` are historical/experimental references, not generally recommended production paths.

> **DeepSeek-V4 B3 sparse-attention investigation is CLOSED (2026-07-03).** Backporting the
> experimental FlashInfer CUDA sparse-MLA *prefill-only* kernel was found **performance-neutral**
> (64K c1 prefill parity, −0.91% vs the MARLIN + SM121-indexer baseline) and is **not promoted**.
> The production baseline above is unchanged. Full record:
> [`docs/dsv4-sparse-mla-b3-investigation-closure.md`](docs/dsv4-sparse-mla-b3-investigation-closure.md).

> **P46 long-output experimental profiles.** Experimental **c2/c4** DeepSeek-V4-Flash profiles for
> **long-output / deep-batch throughput** testing are available — these are **not** the production
> baseline; use the digest-pinned production preset above for normal serving and restore. c4 passed
> `llama-benchy` c4-safe validation, but short-output / interactive serving stays unsuitable
> (prefill-stagger) — use only for long-output / throughput-oriented offline workloads. Presets:
> [`…longout-c2-throughput…`](presets/deepseek-v4-h1z-longout-c2-throughput-experimental-tp2.env),
> [`…longout-c4-deepbatch…`](presets/deepseek-v4-h1z-longout-c4-deepbatch-experimental-tp2.env);
> detail: [`docs/dsv4-longout-experimental-profiles.md`](docs/dsv4-longout-experimental-profiles.md).
> A published GHCR **metadata-only** tag for the c4 profile exists for provenance/discoverability —
> `ghcr.io/bjk110/vllm-spark:h1z-p48-longout-c4-exp-5131a63` (digest
> `sha256:04ff082e9e012924682dd95d77910b1f580a3debd0c8dda7350a82ba0e4a1077`). Its RootFS is
> **H0-equivalent** (byte-identical runtime to the production digest `sha256:ade810fd…`); it is **not**
> the production baseline or a replacement, and the P48A/P46 c4 throughput is **preset/profile-driven,
> not new image bits**. The PR #18 SM121 standby patch is **not wired** in it. For normal serving and
> restore, keep using the digest-pinned production preset above.

> **SM121 rowwise-MQA graph-safe experimental image (Issue #17).** A separate **experimental** image
> carrying the active PR #18 rowwise paged-MQA CUDA-graph-safety patch is published:
> `ghcr.io/bjk110/vllm-spark:h1z-p50-sm121-rowwise-mqa-graphsafe-exp-41d211f` (digest
> `sha256:4b8f650aa96e3af5f30d50b2f98890d6d4a04e0cac6acf800d8b08e30deabba8`). Unlike the c4 tag above,
> this is a **real one-file runtime delta** (installed `ops/sm12x_mqa.py` signature patch, `tl.constexpr`
> 83→64) over the H0/production-equivalent runtime. It passed static (P50B) + active staged validation
> (P50C) to a tokenizer-verified **256K context** with zero `EngineDeadError` / CUDA-graph fault. It is
> **not** itself the digest pinned by the default production preset (that is the equivalent `@de69fa367137`
> graph-safe image, see below), and carries **no performance claim**. [Issue #17](https://github.com/bjk110/spark_vllm_docker/issues/17)
> is **closed** (resolved at the repository recipe level — the graph-safe image is now the default recommended
> preset; see the promotion note below). A **repo-reproducible** build recipe for this
> path lives at
> [`dockerfiles/experimental/Dockerfile.h1z-p52-graphsafe-from-h0`](dockerfiles/experimental/Dockerfile.h1z-p52-graphsafe-from-h0)
> — a non-default FROM-H0 derivative that applies the patch to the installed vLLM package at build time
> (the from-source build path is blocked upstream, so this is a derivative, not a wheel rebuild). Its local
> image was re-validated to 256K (H1Z-P52C2). It is **not** published to GHCR yet and **not** production default.
> **This graph-safe image is now the default recommended production recipe (H1Z-P54A).** The default
> production preset
> [`presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env`](presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env)
> was updated in place to pin the validated graph-safe digest
> `ghcr.io/bjk110/vllm-spark@sha256:de69fa367137…` (config `sha256:5bb962a9055d…`) — the P53
> production-candidate image, promoted after P50C/P52C2 256K validation, a P52 full 24-row
> `llama-benchy` bench (24/24, incl. c8), a **performance-neutral** P52Bench-Baseline vs `@ade810fd`
> on the same disposable route, and P53C production-like validation (`max_num_seqs=1`, capture `[2]`,
> 4 GiB KV; correctness 5/5, c1 latency ~±2%, 32K/64K/128K sanity, zero `EngineDeadError`/CUDA/
> `cuModuleLoad`/OOM/preemption). The previous baseline is preserved verbatim as the **legacy rollback**
> preset
> [`presets/…-production-legacy-ade810fd-tp2.env`](presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-legacy-ade810fd-tp2.env)
> (`@ade810fd` / config `fa83457d`). The earlier
> [`…-graphsafe-production-candidate-tp2.env`](presets/deepseek-v4-h1z-b1ae-sm121-indexer-graphsafe-production-candidate-tp2.env)
> preset is retained for provenance and pins the **same** digest as the default preset — new users should
> use the default production preset; use the legacy preset only to roll back. This is a **repository
> recipe-level** promotion of a validated **patched** image — **not** an upstream vLLM fix, **no**
> performance-improvement claim, and **no** GHCR `latest`/`stable`/`production` tag was moved. The
> from-source build path remains blocked upstream (pinned source unavailable), so the reproducible
> recipe stays the FROM-H0 derivative Dockerfile above.
> [Issue #17](https://github.com/bjk110/spark_vllm_docker/issues/17) is **closed** as resolved at the
> repository recipe level (graph-safe image is now the default recommended preset).
> Detail: [`docs/deepseek-v4-sm121-rowwise-mqa-cudagraph-fix.md`](docs/deepseek-v4-sm121-rowwise-mqa-cudagraph-fix.md).

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
