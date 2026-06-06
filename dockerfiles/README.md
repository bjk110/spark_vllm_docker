# Dockerfile Variants

This directory is organized into two subdirectories:

```text
dockerfiles/
  active/    Current active build targets
  legacy/    Historical, intermediate, and specialized variants
```

---

## Active build targets — `dockerfiles/active/`

| Dockerfile | Image tag | Role |
|---|---|---|
| `Dockerfile.v022-d568` | `v022-d568` | Forward-stack validation base (NGC 26.04 + vLLM 0.21.0 + SM121 FP8 cherry-pick). General-purpose base for non-DSV4 model presets. **On GHCR.** |
| `Dockerfile.dsv4-d568` | `dsv4-d568` | Primary DeepSeek-V4-Flash image path. `FROM v022-d568` + SM12x DSV4 vLLM patches (sparse MLA, Lightning Indexer, fp8_ds_mla KV, MTP). **On GHCR. Currently frozen — do not update casually.** |

Build commands (always build from **repo root** with `.` as context):

```bash
# Build on a Spark node (spark01 or spark02) — homeserver has insufficient RAM
docker buildx build -f dockerfiles/active/Dockerfile.v022-d568  -t vllm-spark:v022-d568  --load .
docker buildx build -f dockerfiles/active/Dockerfile.dsv4-d568  -t vllm-spark:dsv4-d568  --load .
```

`COPY patches/...`, `COPY scripts/...`, and other relative paths inside these Dockerfiles
resolve from the repo root build context, not from the Dockerfile's location.

---

## Legacy / historical variants — `dockerfiles/legacy/`

| Dockerfile | Era / purpose | Status |
|---|---|---|
| `Dockerfile` | NGC 26.01 era, vLLM 0.18.x | Historical — not the current stack |
| `Dockerfile.gemma4` | v021-ngc2603 unified base build | Bisection / reproduction only |
| `Dockerfile.ngc2603-v3` | v018-ngc2603 archived build | Archived |
| `Dockerfile.nvfp4` | NVFP4 runtime defaults overlay | Specialized; layered on top of a base image |
| `Dockerfile.v022` | vLLM v0.21.0 release pin | v022 stack intermediate |
| `Dockerfile.v022-fi0611` | FlashInfer 0.6.11.post3 bump | v022 stack intermediate |
| `Dockerfile.v022-ngc2604` | NGC 26.04 + split_module compat patch | v022 stack intermediate |
| `Dockerfile.v022-tx581` | Transformers 5.8.1 bump | v022 stack intermediate |
| `Dockerfile.v022-trt37` | Triton 3.7.0 bump | v022 stack intermediate |
| `Dockerfile.v022-nccl234` | NCCL 2.30.4 pip override | v022 stack intermediate |

The v022 intermediate layers are kept for bisection and rollback if a regression is
found in `v022-d568`. They are not published to GHCR.

Legacy build commands (build from repo root):

```bash
docker buildx build -f dockerfiles/legacy/Dockerfile.gemma4       -t vllm-spark:v021-ngc2603  --load .
docker buildx build -f dockerfiles/legacy/Dockerfile.v022          -t vllm-spark:v022-vllm021  --load .
docker buildx build -f dockerfiles/legacy/Dockerfile.v022-fi0611   -t vllm-spark:v022-fi0611   --load .
docker buildx build -f dockerfiles/legacy/Dockerfile.v022-ngc2604  -t vllm-spark:v022-ngc2604  --load .
docker buildx build -f dockerfiles/legacy/Dockerfile.v022-tx581    -t vllm-spark:v022-tx581    --load .
docker buildx build -f dockerfiles/legacy/Dockerfile.v022-trt37    -t vllm-spark:v022-trt37    --load .
docker buildx build -f dockerfiles/legacy/Dockerfile.v022-nccl234  -t vllm-spark:v022-nccl234  --load .
```

---

## Notes

- Do not assume a Dockerfile in `legacy/` is a current recommended build path.
- Check the top-level `README.md` (§ Software Stack / § Build) for active targets.
- `dsv4-d568` is frozen — changes to `dockerfiles/active/Dockerfile.dsv4-d568` should
  be coordinated with the full runtime verification procedure.
- Any change to an active Dockerfile should update this document and `README.md` in the
  same commit.
