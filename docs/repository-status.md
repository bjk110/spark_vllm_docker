# Repository Status and Cleanup Roadmap

Last updated: 2026-06-06 (Stage 3-C).

This document summarises the current recommended paths, major directory roles,
completed documentation cleanup stages, and intentionally deferred structural work.

---

## Current recommended paths

### Primary DeepSeek-V4-Flash path — `dsv4-d568`

Image: `ghcr.io/bjk110/vllm-spark:dsv4-d568`

Built from `dockerfiles/active/Dockerfile.dsv4-d568`. Uses the Ray backend by default.
See the top-level `README.md` and `docs/dsv4-flash-tp2.md` for the full guide.

### Experimental high-prefill path — `unholy-fusion`

Image: `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`  
Mirrored: `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`

Run with the compose override so `.env` and `entrypoints/entrypoint.sh` are not overwritten:

```bash
# worker node (spark02):
docker compose \
  -f docker-compose.yml \
  -f compose/docker-compose.unholy.yml \
  --env-file .env.unholy-fusion \
  --profile worker up -d

# head node (spark01):
docker compose \
  -f docker-compose.yml \
  -f compose/docker-compose.unholy.yml \
  --env-file .env.unholy-fusion \
  --profile head up -d
```

This image is `mp` backend only (Ray is not available in its conda environment).
See `README.md §Applying unholy-fusion for DSV4` and `docs/unholy-fusion-benchmark.md`.

**Safe defaults** (aligned across `.env.unholy-fusion`, `entrypoints/entrypoint.unholy.sh`, and `README.md`):

| Variable | Safe default | Notes |
|---|---|---|
| `MAX_MODEL_LEN` | `262144` | 524288 crashes at d≥131072 |
| `MAX_NUM_SEQS` | `4` | ≥5 causes CUDA graph capture hang |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | |
| `GPU_MEMORY_UTILIZATION` | `0.80` | |
| `MTP_NUM_TOKENS` | `1` | n=2 causes throughput collapse at c≥4 |
| `VLLM_USE_B12X_MOE` | `1` | required for 2× prefill speedup |

### Historical / deferred path — jasl-based DSV4

Earlier jasl-based DSV4 image notes are kept only for historical reference and
benchmark traceability. `jasl` is not a currently recommended operational path.

---

## Directory roles

| Path | Current role |
|---|---|
| `dockerfiles/active/Dockerfile.v022-d568` | Active build — forward-stack validation base; on GHCR |
| `dockerfiles/active/Dockerfile.dsv4-d568` | Active build — primary DeepSeek-V4-Flash image; on GHCR. **Currently frozen.** |
| `dockerfiles/active/` | Current active build targets; see `dockerfiles/README.md` |
| `dockerfiles/legacy/` | Historical, intermediate, and specialized Dockerfile variants; see `dockerfiles/README.md` |
| `presets/` | `.env` model-serving preset files only — **not** actual model weights; see `presets/README.md`. Previously named `models/`; container-internal `/models/...` mount paths are unrelated and unchanged. |
| `patches/common/` | Common runtime/build compatibility patches; see `patches/README.md` |
| `patches/sm121/` | SM121 / Blackwell / FP8 patches; see `patches/README.md` |
| `patches/dsv4/` | DeepSeek-V4 specific patches and MoE config files; see `patches/README.md` |
| `patches/qwen/` | Qwen-specific compatibility patches; see `patches/README.md` |
| `patches/turboquant/` | TurboQuant-specific patches; see `patches/README.md` |
| `patches/flashinfer/` | FlashInfer-specific patches; see `patches/README.md` |
| `patches/archive/` | Historical patches retained for reproducibility; see `patches/README.md` |
| `patches/unknown/` | Unverified early bring-up helpers; see `patches/README.md` |
| `benchmarks/` | Raw benchmark artifacts and experiment outputs; see `benchmarks/README.md` |
| `entrypoints/` | Container entrypoint scripts; selected via `ENTRYPOINT_FILE` in `docker-compose.yml`; see `entrypoints/README.md` |
| `compose/` | Compose overrides (`docker-compose.unholy.yml`); referenced via `-f` flag |
| `docs/` | Interpreted technical notes, stack guides, and status documents |

---

## Completed cleanup stages

| Stage | Summary |
|---|---|
| **Stage 1** | README restructured: Korean README (`README.ko.md`) removed; jasl de-emphasised; DSV4 and unholy-fusion roles clarified; safe defaults aligned across `.env.unholy-fusion`, `entrypoint.unholy.sh`, and `README.md`; copy/overwrite procedure moved to manual fallback |
| **Stage 2-A** | Non-destructive unholy-fusion run path: `ENTRYPOINT_FILE` variable in base compose; `compose/docker-compose.unholy.yml` override; `--env-file .env.unholy-fusion` — no `cp` required |
| **Stage 2-B** | `models/` clarified as `.env` preset storage, not model weights; `models/README.md` added |
| **Stage 2-C** | Dockerfile roles documented; `dockerfiles/README.md` added; `v022-d568` table entry wording updated from "general production base" to "forward-stack validation base" |
| **Stage 2-D** | All 25 patch files classified (Active / Conditional / Standby / Historical / Unknown); `patches/README.md` added with cleanup policy and future-split plan |
| **Stage 2-E** | Benchmark artifacts documented; `benchmarks/README.md` and `benchmarks/llama-benchy/README.md` added with filename legend and 21-file inventory |
| **Stage 2-F** | This document |
| **Stage 3-A** | Entrypoint scripts moved from repo root to `entrypoints/`; `docker-compose.yml` default updated to `./entrypoints/entrypoint.sh`; `.env.unholy-fusion` updated to `./entrypoints/entrypoint.unholy.sh`; `entrypoints/README.md` added |
| **Stage 3-B** | Dockerfiles reorganized: active builds moved to `dockerfiles/active/`, legacy/intermediate variants moved to `dockerfiles/legacy/`; `dockerfiles/README.md` rewritten; build commands updated in `README.md` and `docs/` |
| **Stage 3-C** | `patches/` split into purpose-based subdirectories (`common/`, `sm121/`, `dsv4/`, `qwen/`, `turboquant/`, `flashinfer/`, `archive/`, `unknown/`); all Dockerfile `COPY` references, entrypoint script paths, and documentation updated simultaneously |
| **Stage 3-D** | `models/` renamed to `presets/`; all host-side preset path references updated in `README.md`, `docs/`, and `.env.example`. Container-internal `/models/...` mount paths are unrelated and unchanged. |

---

## Deferred structural cleanup

The following work was intentionally deferred. Each item should be done in a
single commit that updates Dockerfile `COPY` directives, compose references, and
README links in the same change.

| Deferred item | Notes |
|---|---|
| Reorganize benchmark outputs into `benchmarks/summary/` and `benchmarks/raw/` | Documentation-level impact only |
| Add CI checks for compose config syntax and shell script syntax | Low-risk addition; `bash -n` for scripts, `docker compose config` for compose |

---

## Do not infer recommendations from raw artifacts

- Raw benchmark filenames (e.g. `maxseq8`, `mtp2`) reflect experiments, not safe defaults.
- Dockerfiles under `dockerfiles/` are historical or intermediate — not active build targets.
- Unreferenced patches should not be deleted solely because `grep` finds no references;
  they may support bisect, forum reproduction, or manual recovery flows.
- `.env` preset files in `presets/` should be reviewed before use — `MODEL_PATH` must
  point to the correct local weight directory.
