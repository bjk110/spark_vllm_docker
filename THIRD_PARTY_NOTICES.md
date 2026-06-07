# Third-Party Notices

This repository contains Dockerfiles, Docker Compose configuration, entrypoint scripts, patch scripts, model-serving presets, benchmark artifacts, and documentation for DGX Spark / GB10 vLLM serving.

The source files, configuration files, scripts, presets, and documentation authored for this repository are licensed under the repository root [`LICENSE`](LICENSE) (Apache License 2.0).

## No model weights

This repository does **not** distribute model weights.

Preset files may reference upstream model names or expected local/container paths, but those references do not include or grant rights to the corresponding model weights.

Users are responsible for obtaining model weights from the appropriate upstream provider and complying with each model's license and terms.

## Container images and dependencies

Container images built from or referenced by this repository may include third-party software and base images, including but not limited to:

- NVIDIA CUDA / NGC container components
- PyTorch
- vLLM
- FlashInfer
- Triton
- NCCL
- Transformers
- other Python, CUDA, or system packages installed by the Dockerfiles

Those components remain governed by their respective upstream licenses and terms.

This repository's license does not relicense third-party components or base images.

## Upstream models

This repository may include presets or documentation for serving models such as Qwen, DeepSeek, Gemma, or other upstream models.

These presets and documents are configuration references only. They do not redistribute model weights and do not change the license terms of those upstream models.

## User responsibility

Before using any container image, dependency, or model referenced by this repository, users should review the applicable upstream license and terms.

This document is provided for clarity on scope and is not legal advice.
