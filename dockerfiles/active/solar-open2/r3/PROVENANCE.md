# Solar incremental r3 byte provenance

Exact copies of local build evidence; hashes checked without a build. Historical
EXPERIMENTAL Dockerfile comments are preserved as original build-time provenance;
production promotion is documented in docs/solar-open2-production.md.
The source paths below use archive ID `2026-10-03-final-recipe-curation` and are
authenticated by [`benchmarks/evidence/archive-manifest.sha256`](../../../../benchmarks/evidence/archive-manifest.sha256).

| File | SHA256 | Read-only source evidence |
|---|---|---|
| `Dockerfile` | `4cebb9c2effb5565e1d4de478e630c28f337596b221634532837761bf92de627` | `archive:2026-10-03-final-recipe-curation/benchmarks/results/solar-open2-v0251-promotion-readiness-20260727T065646Z/01_provenance/Dockerfile.solar-open2-nvfp4-v0251-rawg1-pread-r3-exp` |
| `solar-open2-rawg1-contract-v025.patch` | `f252383c949de06cf14d2fadf6ff6c0a5ae76fa7408e7beb5b30c8db823dd6ca` | `archive:2026-10-03-final-recipe-curation/benchmarks/results/solar-open2-v0251-pread-r3-build-20260726T002411Z/build/context/solar-open2-rawg1-contract-v025.patch` |
| `solar_open2-rawg1.py` | `56b487479c06a4c23d19276c624cde2d462da0ac07e90e394bed580984761645` | `archive:2026-10-03-final-recipe-curation/benchmarks/results/solar-open2-v0251-pread-r3-build-20260726T002411Z/build/context/solar_open2-rawg1.py` |
| `vllm-safetensors-pread-env-gate.patch` | `39b26822f2bfb85dba1fb75e8e90a62ee2aabb32014d4b26ce963b6e2b1371ea` | `archive:2026-10-03-final-recipe-curation/benchmarks/results/solar-open2-v0251-pread-r3-build-20260726T002411Z/build/context/vllm-safetensors-pread-env-gate.patch` |
| `weight_utils-pread-gate.py` | `de48cbefb218b5a0c7160df0253c116bf5b84cf4bf4cd90aecdd4396bb5de7aa` | `archive:2026-10-03-final-recipe-curation/benchmarks/results/solar-open2-v0251-pread-r3-build-20260726T002411Z/build/context/weight_utils-pread-gate.py` |

This directory is the original COPY build context. Parent image tags remain local
references; neither a public-base reconstruction nor an image-ID reproduction is claimed.
