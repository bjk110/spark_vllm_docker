# Documentation index

This is the canonical index for `docs/`. **A document's authority is determined by
this index and by the status banner at the top of each document**, not by its
filename. A filename containing `production` or `validated` does not by itself make a
document current — always check its group here and its banner.

The current DeepSeek-V4-Flash production baseline is the **SM121 DeepGEMM FP8-Q
indexer** path. Runtime deployments must use the **immutable digest-pinned** production
preset (`presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env`), never the
mutable alias.

Groups: [Current production](#current-production) · [Rollback and operations](#rollback-and-operations)
· [General stable stacks](#general-stable-stacks) · [Model guides](#model-guides)
· [Diagnostics and troubleshooting](#diagnostics-and-troubleshooting) · [Benchmarks](#benchmarks)
· [Validated alternatives](#validated-alternatives) · [Experimental work](#experimental-work)
· [Historical and superseded records](#historical-and-superseded-records)

## Current production

Runtime authority = immutable digest `sha256:ade810fd…` (image config `fa83457d`). Deploy via the
digest-pinned preset; the mutable alias `dsv4-sm121-indexer-production` is provenance only.

| Document | Subject | Status | Use |
|---|---|---|---|
| [deepseek-v4-sm121-indexer-production.md](deepseek-v4-sm121-indexer-production.md) | Current DeepSeek-V4-Flash production baseline — SM121 DeepGEMM FP8-Q indexer, MARLIN MoE, production Triton dense/sparse-MLA, TP=2 mp | `Current production` | Authoritative runtime identity, routing, rollback, clone guard, ABI provenance |
| [deepseek-v4-sm121-indexer-promotion-manifest.md](deepseek-v4-sm121-indexer-promotion-manifest.md) | Machine-oriented promotion record (digests, source/ABI SHAs, validation outcomes, rollback) | `Current production` | Quick identity/provenance lookup |

## Rollback and operations

Distinguish: current production (above), the immediate rollback baseline (config `4c41950c`, preset
`presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-production-tp2.env`, SHA `593ba898`),
and historical activation evidence.

| Document | Subject | Status | Use |
|---|---|---|---|
| [deepseek-v4-prefill8192-production-runbook.md](deepseek-v4-prefill8192-production-runbook.md) | prefill8192 activation / acceptance / shutdown / rollback runbook | `Rollback baseline` | Rollback activation and historical reproduction (not the normal current path) |
| [deepseek-v4-prefill8192-production-activation.md](deepseek-v4-prefill8192-production-activation.md) | prefill8192 production activation/acceptance record | `Rollback baseline` (historical activation evidence) | Immutable record of the prior production activation |
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
| [deepseek-v4-mtp1-fullgraph-validated-preset.md](deepseek-v4-mtp1-fullgraph-validated-preset.md) | DeepSeek-V4 MTP n=1 + FULL_DECODE_ONLY validated preset provenance | `Validated alternative` | Provenance/gates for the MTP fullgraph validated preset |
| [flashinfer-aot-prebake.md](flashinfer-aot-prebake.md) | FlashInfer AOT-prebaked image (`v022-d568-fi-aot`) validated specs | `Validated alternative` | Optional drop-in for `v022-d568` |

## Experimental work

Not promoted. Reference/experimental only.

| Document | Subject | Status | Use |
|---|---|---|---|
| [deepseek-v4-v023-stack-pr41834.md](deepseek-v4-v023-stack-pr41834.md) | DeepSeek-V4 SM12x v0.23-stack / PR #41834 experimental build | `Experimental` | v0.23-stack DSV4 build notes |
| [unholy-fusion-benchmark.md](unholy-fusion-benchmark.md) | `unholy-fusion` configuration, limits, and benchmark comparison | `Experimental` | Higher-prefill DSV4 experimental alternative (not a recommended production path) |
| [step3.7-tokenizer-overlay.md](step3.7-tokenizer-overlay.md) | Step-3.7 non-mutating runtime tokenizer overlay | `Experimental` | Tokenizer-overlay technique |
| [prometheus-routing-path-fix.md](prometheus-routing-path-fix.md) | Prometheus `routing.py` `.path` guard (experimental image) | `Experimental` | Monitoring routing-path fix notes |

## Historical and superseded records

Preserved for evidence and reproducibility. Superseded ≠ incorrect; do not treat as current.

| Document | Subject | Status | Use |
|---|---|---|---|
| [deepseek-v4-prefill8192-validated-candidate.md](deepseek-v4-prefill8192-validated-candidate.md) | prefill8192 validated-candidate testing record | `Superseded` | Provenance for the rollback baseline (replacement: SM121 indexer) |
| [deepseek-v4-prefill-optimization-campaign-plan.md](deepseek-v4-prefill-optimization-campaign-plan.md) | Completed prefill-optimization campaign plan | `Historical` | Campaign record (planned actions are not current instructions) |
| [deepseek-v4-mtp1-fullgraph-promotion-checklist.md](deepseek-v4-mtp1-fullgraph-promotion-checklist.md) | MTP + FULL-graph validated-preset promotion checklist | `Historical` operational reference | Promotion-checklist record |
| [model-serving-validation-history.md](model-serving-validation-history.md) | Historical stack validation notes and benchmarks (Gemma 4, Qwen3.5 122B/397B, PrismaQuant, Qwen3.6-35B, TurboQuant) | `Historical` | Historical benchmark/validation archive |
| [stack-v021.md](stack-v021.md) | Software stack v021-ngc2603 (previous main, NGC 26.03) | `Superseded` | Prior main stack reference |
| [stack-v019.md](stack-v019.md) | Software stack v019-ngc2603 (archived) | `Historical` (archived) | Archived stack reference |
