# Solar incremental r4 byte provenance

Exact copies of local build evidence; hashes checked without a build. Historical
EXPERIMENTAL Dockerfile comments are preserved as original build-time provenance;
production promotion is documented in docs/solar-open2-production.md.
The source paths below use archive ID `2026-10-03-final-recipe-curation` and are
authenticated by [`benchmarks/evidence/archive-manifest.sha256`](../../../../benchmarks/evidence/archive-manifest.sha256).

| File | SHA256 | Read-only source evidence |
|---|---|---|
| `Dockerfile` | `c2df5e9dcd5409cf26ec2f48d7fa2fd367a4dd4d3665b3f3e155b38076755b0b` | `archive:2026-10-03-final-recipe-curation/benchmarks/results/solar-open2-v0251-promotion-readiness-20260727T065646Z/01_provenance/Dockerfile.solar-open2-nvfp4-v0251-rawg1-pread-b12xsw-r4-exp` |
| `b12x_moe_r4.py` | `38c253a37c131b38713b92cea038bb6ff07f7b38a1c91fdd68f38567cb8180bc` | `archive:2026-10-03-final-recipe-curation/benchmarks/results/solar-open2-v0251-b12xsw-r4-20260726T053457Z/build/context/b12x_moe_r4.py` |
| `vllm-flashinfer-b12x-shared-workspace-env-gate.patch` | `2b31f3a873e7a29c991cefc39599b01b10f41f4dbdef096fc55f23c175382c83` | `archive:2026-10-03-final-recipe-curation/benchmarks/results/solar-open2-v0251-b12xsw-r4-20260726T053457Z/build/context/vllm-flashinfer-b12x-shared-workspace-env-gate.patch` |

This directory is the original COPY build context. Parent image tags remain local
references; neither a public-base reconstruction nor an image-ID reproduction is claimed.
