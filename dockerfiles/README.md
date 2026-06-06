# Dockerfile Variants

This directory contains historical, experimental, or intermediate Dockerfile variants
used while testing DGX Spark / GB10 vLLM stacks.

## Active build targets (repository root)

The currently active Dockerfiles are kept in the **repository root** for backward
compatibility with existing build commands:

| Dockerfile | Image tag | Role |
|---|---|---|
| `Dockerfile.v022-d568` | `v022-d568` | Forward-stack validation base (NGC 26.04 + vLLM 0.21.0 + SM121 FP8 cherry-pick). General-purpose base for non-DSV4 model presets. **On GHCR.** |
| `Dockerfile.dsv4-d568` | `dsv4-d568` | Primary DeepSeek-V4-Flash image path. `FROM v022-d568` + SM12x DSV4 vLLM patches (sparse MLA, Lightning Indexer, fp8_ds_mla KV, MTP). **On GHCR.** |

Build commands for these:

```bash
# Build on a Spark node (spark01 or spark02) — homeserver has insufficient RAM
docker buildx build -f Dockerfile.v022-d568  -t vllm-spark:v022-d568  --load .
docker buildx build -f Dockerfile.dsv4-d568  -t vllm-spark:dsv4-d568  --load .
```

## Files in this directory

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

## Notes

- Do not assume a Dockerfile in this directory is the current recommended build path.
- Check the top-level `README.md` (§ Software Stack / § Build) for the active targets.
- A future cleanup may reorganize this directory into `active/` and `legacy/`
  subdirectories, but this stage intentionally avoids moving files.
