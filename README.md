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

Two independent model families have a **promoted production baseline**: DeepSeek-V4-Flash and
Solar-Open2-250B. "Promoted" describes the repository-defined baseline (which preset/image is
authoritative), not which model is *physically running* at this instant — both currently target the
same serving slot (spark01 head + spark02 worker, port 8000) and are not served simultaneously.
Check live state (`docker ps`, `GET :8000/health`, `GET :8000/v1/models`) to see which one is
actually deployed right now; do not infer it from this document alone.

| Path | Status | Backend | Use case |
|---|---|---|---|
| `dsv4-flash-0731-v027` | **DeepSeek-V4-Flash promoted production** — native DSpark k=7, 256K (promoted 2026-08-15) | `mp` | Active DSV4 path — `MAX_NUM_SEQS=1`, FULL_DECODE_ONLY `[8]`, automatic startup prewarm, dual-node TP=2. Immutable manifest `@sha256:7a005243701c…`. |
| `dsv4-flash-dspark-64k` | DeepSeek-V4 primary rollback (stopped) | `mp` | vLLM 0.25.0 native DSpark k=7, 64K, digest `@sha256:aacb06de60ec…`. |
| `dsv4-flash-mtp1` | DeepSeek-V4 legacy rollback (stopped) | `mp` | Second-tier fallback — MTP n=1, capture `[2]`, 4 GiB FP8 KV, digest `@sha256:de69fa367137…`. |
| `dsv4-d568` | Frozen legacy/historical DSV4 baseline | `ray` or `mp` | Historical decode-optimized reproduction/reference only. |
| `unholy-fusion` | Historical/experimental (DSV4 only); config removed from active tree 2026-08-11 | `mp` | Higher-prefill DSV4 experimental alternative — not a recommended production path. Recoverable from Git history; see [`docs/unholy-fusion-benchmark.md`](docs/unholy-fusion-benchmark.md). |
| `solar-open2-r4-bf16` | **Solar-Open2-250B promoted production** — r4 BF16, vLLM 0.25.1 (promoted 2026-08-09) | `ray` | TP=2, BF16 KV fixed 4 GiB/rank (66,764 tok), `MAX_MODEL_LEN=4096`, eager, FLASHINFER_B12X MoE, ST_PREAD + B12X shared-workspace gates. Local image ID `sha256:ecb7bfe3…` (not yet published to a registry). Rollback = v0.22.1 KV4G preset. |
| `solar-open2-v022-rollback` | Solar-Open2-250B v0.22.1 production rollback (stopped) | `ray` | Authoritative rollback for the r4 production — matched scheduler footprint, BF16 KV. Local image ID `sha256:1873d217…`. |
| `v022-d568-ngc2605-tx5102-vllm022` | Active forward-stack (NGC 26.05, vLLM 0.22.1) | `ray` | Qwen3.5-122B-FP8 and other forward-stack models. |
| `v022-d568` | Stable general base (NGC 26.04, vLLM 0.21.0) | `ray` or direct | Qwen3.6, Gemma 4 31B, abliterix NVFP4 presets. |
| `v021-ngc2603` / `v021-tq` | Stable base for most existing presets | `ray` or direct | Most non-DSV4/non-Solar presets. |

**DeepSeek-V4 promoted production** runs `deepseek-ai/DeepSeek-V4-Flash-0731` on vLLM 0.27,
native DSpark k=7, 256K, and `MAX_NUM_SEQS=1`. Its immutable image identity is
`ghcr.io/bjk110/vllm-spark@sha256:7a005243701c5df6f8945ea56d509f747f2c63ecff6091a0169d8109a736d09f`
(local image ID `sha256:a7f0f4b8a508…`). Launch uses the production preset plus its health/prewarm
overlay; see [Quick Start](#quick-start) and
[`docs/deepseek-v4-production.md`](docs/deepseek-v4-production.md). The v0.25.0/64K DSpark route is the
primary rollback; MTP1 is the legacy second-tier rollback.

**Solar-Open2-250B promoted production** runs the r4 BF16 route (vLLM 0.25.1) by its **local Docker
image ID** (`sha256:ecb7bfe3978a…` — not yet published to a registry, see
[`docs/images.md`](docs/images.md) for the local-ID-vs-digest distinction), promoted 2026-08-09 via
[`presets/solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env`](presets/solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env).
Runtime contract: TP=2 Ray, BF16 KV fixed 4 GiB/rank (66,764 tok), `MAX_MODEL_LEN=4096`,
`MAX_NUM_SEQS=8`, eager mode, FLASH_ATTN (auto), FLASHINFER_B12X MoE, ST_PREAD + B12X
shared-workspace gates enabled.

The authoritative rollback is the v0.22.1 preset
[`presets/solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env`](presets/solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env)
(local image ID `sha256:1873d2174691…`, matched scheduler footprint, BF16 KV; currently stopped).
Full operations, activation, rollback (including the empirically required physical-reboot
sequence), and validation provenance (6-gate production fast-track):
[`docs/solar-open2-production.md`](docs/solar-open2-production.md).

> **`dsv4-d568` is intentionally frozen** and will not be rebased onto NGC 26.05+. `dsv4-d568`
> (JASL-era) and `unholy-fusion` are historical/experimental references, not recommended production paths.

> **DeepSeek-V4 B3 sparse-attention investigation is CLOSED (2026-07-03).** Backporting the experimental
> FlashInfer CUDA sparse-MLA prefill-only kernel was **performance-neutral** (64K c1 prefill parity,
> −0.91% vs the MARLIN baseline) and is **not promoted**. Full record:
> [`docs/dsv4-sparse-mla-b3-investigation-closure.md`](docs/dsv4-sparse-mla-b3-investigation-closure.md).

> **SM121 rowwise-MQA graph-safe fix (Issue #17, closed).** The graph-safe `ops/sm12x_mqa.py` signature
> patch (PR #18, `tl.constexpr` 83→64) is installed in the MTP1 rollback image (`@de69fa367137`);
> repository-recipe-level patched image, not an upstream fix, no performance claim. Technical record:
> [`docs/deepseek-v4-sm121-rowwise-mqa-cudagraph-fix.md`](docs/deepseek-v4-sm121-rowwise-mqa-cudagraph-fix.md).

Component versions, stack lineage, and digests → [`docs/software-stack.md`](docs/software-stack.md).
Image tag → Git-ref mapping → [`docs/images.md`](docs/images.md). Optional FlashInfer-AOT drop-in
for `v022-d568` → [`docs/flashinfer-aot-prebake.md`](docs/flashinfer-aot-prebake.md).

## Quick Start

Presets live in [`presets/`](presets/) (`.env` files only — no model weights). Keep weights outside
the repo and point `MODEL_PATH` / `MODEL_CONTAINER_PATH` at them.

### 1. Get the image

Pull the image referenced by the selected preset on every participating node. For the current
DeepSeek-V4 production route, use its immutable manifest digest:

```bash
docker pull ghcr.io/bjk110/vllm-spark@sha256:7a005243701c5df6f8945ea56d509f747f2c63ecff6091a0169d8109a736d09f
```

Other general bases are cataloged in [`docs/images.md`](docs/images.md). Building from source on
spark01/spark02 is documented in [`dockerfiles/`](dockerfiles/) and
[`docs/software-stack.md`](docs/software-stack.md).

### 2. Check the preset and prerequisites

Presets are cataloged by status in [`presets/README.md`](presets/README.md). Keep model weights outside
the repository. Before startup, ensure both nodes use the same clean tracked Git revision and that
`MODEL_PATH`, the RoCE addresses/interface, and the immutable image identity exist on both nodes. The
current DeepSeek-V4 production artifacts are:

- `presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env`;
- `compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml`;
- `scripts/prewarm_dsv4_v027_b43s.py`.

The preset retains `candidate` in its historical filename but is the active production preset. Its
published tag is mutable, so the commands below override `VLLM_IMAGE` with the recorded immutable
digest.

### 3. Start current DeepSeek-V4 production

Run from the repository root on both nodes. Start the MP worker before the head. This is TP=2 startup,
not two independent servers:

```bash
# spark02 — worker first
VLLM_IMAGE=ghcr.io/bjk110/vllm-spark@sha256:7a005243701c5df6f8945ea56d509f747f2c63ecff6091a0169d8109a736d09f \
docker compose \
  --env-file presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env \
  -f docker-compose.yml \
  -f compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml \
  --profile worker up -d

# spark01 — head plus one-shot automatic prewarm
VLLM_IMAGE=ghcr.io/bjk110/vllm-spark@sha256:7a005243701c5df6f8945ea56d509f747f2c63ecff6091a0169d8109a736d09f \
docker compose \
  --env-file presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env \
  -f docker-compose.yml \
  -f compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml \
  --profile head up -d
```

For generic single-Spark and other dual-Spark presets, follow the preset header and
[`presets/README.md`](presets/README.md); model-specific overlays must not be omitted. `entrypoint.sh`
dispatches on `CLUSTER_MODE`×`ROLE`×`TP_SIZE`×`DISTRIBUTED_BACKEND`; details are in
[`docs/architecture.md`](docs/architecture.md).

### 4. Verify readiness

`/health` means engine alive, not prewarm complete. Wait for all three checks before serving traffic:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models

docker ps -a --filter name=vllm-spark-prewarm \
  --format 'table {{.Names}}\t{{.Status}}'
docker logs vllm-spark-prewarm
```

The model list must contain `deepseek-ai/DeepSeek-V4-Flash-0731` with
`max_model_len: 262144`, and `vllm-spark-prewarm` must finish with exit code 0. Cold startup normally
takes about 8–11 minutes. See [`docs/deepseek-v4-production.md`](docs/deepseek-v4-production.md) for
rollback and GB10 UMA recovery.

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
├── compose/                  # Model-specific Compose overlays (e.g. compose/solar-open2/)
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

- DeepSeek-V4 production (active DSpark + MTP1 rollback, activation/rollback, provenance): [`docs/deepseek-v4-production.md`](docs/deepseek-v4-production.md)
- Stack lineage / images: [`docs/software-stack.md`](docs/software-stack.md), [`docs/images.md`](docs/images.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)
- Troubleshooting: [`docs/troubleshooting.md`](docs/troubleshooting.md)
- Presets: [`presets/README.md`](presets/README.md)

## Compatibility and safety notice

- All Docker/vLLM builds run on spark01 or spark02, never on the homeserver (GB10 template
  compilation needs 64–128 GiB peak).
- DeepSeek-V4 production runs only within the documented v0.27 envelope: `MAX_NUM_SEQS=1` and
  `MAX_MODEL_LEN=262144`. Any concurrency increase, runtime/image change, or context claim beyond that
  contract requires separate qualification.
- GB10 uses unified memory; a clean reboot + dedicated-cache-clear startup gate is required before a
  full model load when UVM is retained (not automated by presets).
- Recommended OS tuning: `sudo sysctl -w vm.swappiness=10`.

## License

Source code, Dockerfiles, scripts, presets, and documentation are licensed under Apache License 2.0
(see [`LICENSE`](LICENSE)). This repository does **not** distribute model weights; users are
responsible for obtaining weights and complying with upstream model licenses. Container images and
dependencies remain governed by their upstream licenses — see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
