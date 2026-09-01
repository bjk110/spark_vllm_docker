# Documentation index

This is the canonical index for `docs/`. **A document's authority is determined by
this index and by the status banner at the top of each document**, not by its
filename. A filename containing `production` or `validated` does not by itself make a
document current — always check its group here and its banner.

The DeepSeek-V4-Flash promoted production baseline is the **vLLM 0.27 native DSpark k=7 256K** path
with `MAX_NUM_SEQS=1`. Deploy with
`presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env` plus
`compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml`; the preset itself is pinned to the
immutable registry digest documented in `deepseek-v4-production.md`. The v0.25.0/64K preset
is the primary rollback; MTP1 is the legacy second-tier rollback.

The Solar-Open2-250B promoted production baseline is the **r4 BF16** (vLLM 0.25.1) path. Runtime
deployments use the **local-image-ID-pinned** production preset
(`presets/solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env`); the v0.22.1 KV4G matched preset
(`presets/solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env`) is the authoritative rollback.

All three families are independent promoted baselines that currently share the same physical serving
slot (spark01 head + spark02 worker, port 8000) rather than running simultaneously — this index
does not track which one is deployed at this instant; check live container/health state for that.

Groups: [Current production](#current-production) · [Rollback and operations](#rollback-and-operations)
· [General stable stacks](#general-stable-stacks) · [Model guides](#model-guides)
· [Diagnostics and troubleshooting](#diagnostics-and-troubleshooting) · [Benchmarks](#benchmarks)
· [Validated alternatives](#validated-alternatives) · [Experimental work](#experimental-work)
· [Historical and superseded records](#historical-and-superseded-records)

## Current production

Three independent model tracks are documented as "current production" below; each is authoritative
for its own model, not for the physical port 8000 slot simultaneously — spark01/spark02 run one
model at a time, and whichever track is not currently deployed remains stopped (see each document's
own status banner for what is actually running right now).

Runtime authority (DeepSeek-V4-Flash) = immutable manifest `sha256:7a005243…` (local image ID
`sha256:a7f0f4b8…`). Deploy the v0.27/256K preset with its required health/prewarm overlay and immutable
image override; see `deepseek-v4-production.md`.

Runtime authority (Solar-Open2-250B) = local Docker image ID `sha256:ecb7bfe3…` (not yet published
to a registry). Deploy via the local-ID-pinned preset.

| Document | Subject | Status | Use |
|---|---|---|---|
| [deepseek-v4-production.md](deepseek-v4-production.md) | Canonical DeepSeek-V4 production operations — active v0.27 native DSpark k=7 256K/MS1 route, v0.25.0/64K primary rollback, MTP1 legacy rollback, startup/prewarm and qualification provenance | `Current production` | Authoritative operations, runtime contracts, rollback |
| [deepseek-v4-v027-runtime-build-base.md](deepseek-v4-v027-runtime-build-base.md) | Frozen DSV4-specific NGC 26.07/vLLM 0.27 build base, GHCR identities, thin-derivative recipe | `Build/release reference` | Avoid rebuilding the validated common runtime closure |
| [solar-open2-production.md](solar-open2-production.md) | Canonical Solar-Open2-250B production operations — active r4 BF16 (vLLM 0.25.1) route and v0.22.1 rollback, runtime contracts, activation/rollback with empirical reboot procedure, validation provenance (6-gate production fast-track 2026-08-08/09) | `Current production` | Authoritative operations, runtime contracts, rollback |
| [qwen3.8-flash-next-tp2.md](qwen3.8-flash-next-tp2.md) | Qwen/Qwen3.8-Flash-Next-FP8 dual DGX Spark TP=2 recipe — production-qualified c1/c2 profile, `MAX_NUM_SEQS=2`, FULL_DECODE_ONLY capture sizes `[1,2]`, staged Gate0–Gate3 procedure, checksum/identity verification | `Production-qualified` (c1/c2; not auto-start) | Production-qualified c1/c2 route; not auto-started, activate manually |

## Rollback and operations

The primary rollback for the active v0.27/256K DSpark production is
`presets/deepseek-v4-flash-dspark-k7-64k-production-tp2.env` (v0.25.0, image
`@sha256:aacb06de60ec…`, stopped). MTP1 (`@sha256:de69fa367137…`) is the legacy second-tier rollback.
Rollback and recovery procedure: [deepseek-v4-production.md](deepseek-v4-production.md).

The authoritative rollback for the active Solar-Open2 r4 production is the v0.22.1 preset
(`presets/solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env`, image
`sha256:1873d2174691…`, stopped). Rollback and recovery procedure (including the empirically
required physical-reboot sequence): [solar-open2-production.md](solar-open2-production.md).

| Document | Subject | Status | Use |
|---|---|---|---|
| [release-management.md](release-management.md) | Maintainer-only Git tag creation, branch structure, archived branches | `Operational reference` | Release/branch maintenance |

## General stable stacks

| Document | Subject | Status | Use |
|---|---|---|---|
| [software-stack.md](software-stack.md) | Full image/stack lineage and component versions/digests (`v022-d568-ngc2605…`, `dsv4-d568`, `v022-d568`, `v021`) | `Stack reference` | Component versions and stack provenance |
| [stack-v022.md](stack-v022.md) | v022 series forward-stack lineage and intermediate build variants | `Stack reference` | v022 stack detail |
| [images.md](images.md) | Container image tag history and image-to-preset / Git-ref mapping | `Operational reference` | Image tag → preset/Git-ref lookup |
| [architecture.md](architecture.md) | Home-infrastructure and distributed-serving architecture | `Stack reference` | Topology, entrypoint dispatch, backend model |

## Model guides

| Document | Subject | Status | Use |
|---|---|---|---|
| [step3.7-flash-tp2.md](step3.7-flash-tp2.md) | Step-3.7-Flash FP8/NVFP4 dual-Spark TP=2 serving + benchmark comparison | `Model guide` | Step-3.7-Flash serving |
| [dsv4-flash-tp2.md](dsv4-flash-tp2.md) | Legacy JASL-era DeepSeek-V4-Flash TP=2 guide (`dsv4-d568`) | `Historical` model guide | Legacy `dsv4-d568` reproduction (NOT current production — see current replacement in its banner) |

## Diagnostics and troubleshooting

| Document | Subject | Status | Use |
|---|---|---|---|
| [troubleshooting.md](troubleshooting.md) | Model-path and stack-specific troubleshooting (compose checks, dsv4/unholy/Qwen issues) | `Diagnostic reference` | First-stop troubleshooting |
| [diagnostics/dgx-spark-uma-memory-freeze.md](diagnostics/dgx-spark-uma-memory-freeze.md) | DGX Spark UMA host-memory freeze during dual-node startup | `Diagnostic reference` | UMA freeze diagnosis |
| [diagnostics/soak-mem-snapshot-hardening.md](diagnostics/soak-mem-snapshot-hardening.md) | Hardened soak memory-snapshot handling (H1Z-B1AF) | `Diagnostic reference` | Soak-gate memory-telemetry design |

## Benchmarks

Benchmark documents index existing results only; they are not production-configuration authority.

| Document | Subject | Status | Use |
|---|---|---|---|
| [benchmarks/bt-matrix-step37-nvfp4-v023.md](benchmarks/bt-matrix-step37-nvfp4-v023.md) | Step-3.7-NVFP4 v0.23 `MAX_NUM_BATCHED_TOKENS` matrix benchmark | `Benchmark` | bt-matrix reference for the Step-3.7 NVFP4 path |
| [benchmarks/step37-v022-long-context-validation.md](benchmarks/step37-v022-long-context-validation.md) | Step-3.7-NVFP4 v0.22 long-context validation (to 245009 tokens) | `Benchmark` (experimental) | Long-context envelope evidence |

## Validated alternatives

Validated, but not the current default serving path.

| Document | Subject | Status | Use |
|---|---|---|---|
| [flashinfer-aot-prebake.md](flashinfer-aot-prebake.md) | FlashInfer AOT-prebaked image (`v022-d568-fi-aot`) validated specs | `Validated alternative` | Optional drop-in for `v022-d568` |

## Experimental work

Not promoted. Reference/experimental only.

| Document | Subject | Status | Use |
|---|---|---|---|
| [unholy-fusion-benchmark.md](unholy-fusion-benchmark.md) | `unholy-fusion` configuration, limits, and benchmark comparison | `Experimental` | Higher-prefill DSV4 experimental alternative (not a recommended production path) |
| [step3.7-tokenizer-overlay.md](step3.7-tokenizer-overlay.md) | Step-3.7 non-mutating runtime tokenizer overlay | `Experimental` | Tokenizer-overlay technique |
| [prometheus-routing-path-fix.md](prometheus-routing-path-fix.md) | Prometheus `routing.py` `.path` guard (experimental image) | `Experimental` | Monitoring routing-path fix notes |
| [dsv4-dspark-speculative-decoding-72261a7-closure.md](dsv4-dspark-speculative-decoding-72261a7-closure.md) | DeepSeek-V4-Flash-DSpark speculative decoding on vLLM `72261a7` — DS2 arc closure (defects found and fixed; acceptance unmoved) | `Experimental` (investigation CLOSED, NOT VALIDATED) | DSpark experimental status: DS2D14 preferred baseline / DS2D13 fallback / DS2D12 rollback; k=3 only validated envelope; local-only images; production unaffected (speculative decoding not used) |

## Historical and superseded records

Preserved for evidence and reproducibility. Superseded ≠ incorrect; do not treat as current.

| Document | Subject | Status | Use |
|---|---|---|---|
| [deepseek-v4-external-runtime-qualification-2026-08-28.md](deepseek-v4-external-runtime-qualification-2026-08-28.md) | eugr, MiaAI/Anemll, and Aiden/Tony replacement-candidate closure | `Qualification record` (no candidate promoted) | Immutable identities, lossless-gate results, benchmark fail-closed decision |
| [model-serving-validation-history.md](model-serving-validation-history.md) | Historical stack validation notes and benchmarks (Gemma 4, Qwen3.5 122B/397B, PrismaQuant, Qwen3.6-35B, TurboQuant) | `Historical` | Historical benchmark/validation archive |
| [stack-v021.md](stack-v021.md) | Software stack v021-ngc2603 (previous main, NGC 26.03) | `Superseded` | Prior main stack reference |
| [stack-v019.md](stack-v019.md) | Software stack v019-ngc2603 (archived) | `Historical` (archived) | Archived stack reference |
