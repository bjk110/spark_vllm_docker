# DeepSeek-V4 v0.27 frozen runtime build base

Status: **released build/reproducibility artifact; not a new runtime promotion**.

## Purpose

Avoid rebuilding the validated NGC 26.07 + vLLM 0.27 + FlashInfer/DeepGEMM/Quack closure for every
future DeepSeek-V4 derivative. This base is explicitly **DSV4-specific**, not model-neutral: it already
contains the B4.3R/B4.3S sparse-MLA dispatch changes and the production dependency closure.

## Immutable identity

- versioned tag: `ghcr.io/bjk110/vllm-spark:v027-ngc2607-dsv4-runtime-base-20260828`
- stable alias: `ghcr.io/bjk110/vllm-spark:v027-ngc2607-dsv4-runtime-base`
- manifest: `sha256:7a005243701c5df6f8945ea56d509f747f2c63ecff6091a0169d8109a736d09f`
- local image ID: `sha256:a7f0f4b8a508c0b2510fc7e4dcb916491efa03c380c9c7b84dddd4c16ad6f38d`
- architecture: `linux/arm64`

Both aliases are a registry retag of the already-qualified production artifact. Registry and Spark02
pull readback confirmed the same manifest and local image ID; no layer was rebuilt or altered.

## Included frozen contract

- NGC PyTorch 26.07 lineage; Torch `2.13.0a0+9186a08b2c.nv26.7.59513937`, CUDA ABI 13.3;
- vLLM `0.27.1.dev0+g4bdc8a788.d20260813.cu133`;
- FlashInfer Python/Cubin `0.6.16.post3`;
- DeepGEMM `2073ddb2814892014c33ef4cd1c7d4c148baf1fe` and Quack `99bd7973bf3dc6db40961e413d4bdfea6c6fee3e` pins;
- B4.3R `(32,128)@page_block_size=256` and B4.3S `(32,256)@page_block_size=64` dispatch entries;
- API/runtime dependency closure used by the active DeepSeek-V4 route.

## Build a thin derivative

Run on a DGX Spark node from a clean repository checkout:

```bash
./scripts/build/build-dsv4-v027-from-runtime-base.sh \
  vllm-spark:my-dsv4-v027-derivative
```

The script creates a temporary one-file build context, so unrelated local artifacts are never sent to
BuildKit. The Dockerfile pins `FROM` by manifest digest and fails closed on package, CLI/API import,
and sparse-MLA dispatch mismatches. Add future DSV4-specific `COPY`/`RUN` steps only after the existing
contract check; never mutate or republish the base tag for an experiment.

A measured first smoke derivative on spark01 completed in **9.03 seconds** (`maxrss_kb=65836`); the
base resolved locally and only the contract-check step ran. This is build-time evidence only, not a
promise for network-cold pulls or runtime startup.

## What this does not accelerate

The base removes repeated image construction of the common stack. It does not eliminate model shard
loading, TP/NCCL initialization, shape-specific JIT/autotuning, CUDA graph capture, or startup prewarm.
A derivative that changes filesystem content has a new image identity and must pass the normal static,
startup, canonical MS1, conditional MS8, and benchmark gates before promotion.

## Production recipe effect

The active MS1 and optional MS4 presets now pin the immutable manifest directly instead of a mutable
tag. Because the digest is unchanged, this is an identity-hardening change, not a new runtime image.
No production container was started or modified as part of this release.

Host-local release evidence:
`/home/bjk110/docker-build/dsv4-v027-runtime-base-release-20260828/`.
