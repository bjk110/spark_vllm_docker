# Software Stack — v019-ngc2603 (archived)

vLLM 0.19.1 with Gemma 4 support and async scheduling. Transformers 5.5.0.
TTFT improved ~2x over v018. **Superseded by**
[v021-ngc2603](stack-v021.md) (vLLM main `95995bbe` + TurboQuant + FlashInfer
v0.6.9), and further by [v022-d568](../README.md#software-stack) on the
forward-stack lineage. Kept for historical reproduction only.

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.03 |
| vLLM | 0.19.1 (main `a7d79fa`, source build) |
| FlashInfer | v0.6.7.post3 (CUTLASS 4.4.2, SM121 source build) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (native) |
| Transformers | 5.5.0 |
| Image tag | `ghcr.io/bjk110/vllm-spark:v019-ngc2603` |

Build:

```bash
docker buildx build -f dockerfiles/Dockerfile.ngc2603-v3 \
  -t vllm-spark:v019-ngc2603 --load .
```

(The Dockerfile is named `ngc2603-v3` for historical reasons — `v018` was
`ngc2603-v1`, `v019` was a `v2` rev that then settled as the `v3` snapshot.)
