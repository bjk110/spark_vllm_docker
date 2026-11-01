# Qwen3.8 v0.33.0rc3 build provenance

This directory preserves the exact Dockerfile shape used for the bounded 2026-11-01 evaluation. It
is a reproduction record, not an active image, preset, or publication target.

| Field | Qualified value |
|---|---|
| vLLM tag | `v0.33.0rc3` |
| Source archive SHA-256 | `e2685461826c8cfb9542214cd7cf6024886ea75b15d8a88d45f911f22067228f` |
| Base image ID | `sha256:518c1b8f9bdaca735d793ffef1c7d6e050bdb62e6ffc9730e9428052cbc59512` |
| Evaluated image ID | `sha256:d4c2ace4e497d89eee47671cd1aa92a8cedc0d3a5fa17f845781c0bfdfb1cf60` |
| Wheel | `vllm-0.33.0rc3+cu134-cp312-cp312-linux_aarch64.whl` |
| Wheel SHA-256 | `b485ecf24d6f7f7dfc5e2f3e55d1452d157bcc6fd0893c56fb19996e194216bd` |
| Platform | `linux/arm64`, CUDA 13.4, SM121 |
| Torch | `2.14.0a0+b2c75dd062.nv26.09` |
| Model revision | `8bebbc05cbe65099f37c520865463c1377d8050d` |

Build-specific corrections were limited to installing `libdw-dev`, accepting the wheel's `+cu134`
local-version suffix, and explicitly generating Marlin headers before the wheel build. No e77 source
overlay was mounted into the RC3 runtime.

The source archive and 503 MB wheel are intentionally not tracked. Reproduction requires providing
an archive that matches the pinned hash and independently verifying the base image ID before build.
The evaluated image is local-only and was not published. See the
[qualification record](../../docs/qwen3.8-flash-next-v033rc3-qualification-20261101.md) for the decision.
