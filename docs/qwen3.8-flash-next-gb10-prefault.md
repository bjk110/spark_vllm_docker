# Qwen3.8 Flash Next FP8 on GB10: safetensors copy-site prefault

| Field | Value |
|---|---|
| Status | **VALIDATED MECHANISM — per-checkpoint qualification required** |
| Scope | Qwen3.8 Flash Next FP8-family checkpoints loaded by vLLM's default mmap-backed safetensors path on NVIDIA GB10 integrated GPUs |
| Qualified runtime | `vllm-local:qwen38-flash-next-arm64-3b0e188f`, image ID `sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a`, vLLM `0.1.dev20073+g8e685d198` |
| Upstream basis | vLLM PR #58868, commit `857f70df4b7f3d96e19dc2b7361b56130bbe79b1` |
| Patch manifest | [`patches/qwen/prefault-pr58868/MANIFEST.json`](../patches/qwen/prefault-pr58868/MANIFEST.json) |
| Compose overlay | [`compose/qwen3.8-flash-next/docker-compose.prefault-pr58868.yml`](../compose/qwen3.8-flash-next/docker-compose.prefault-pr58868.yml) |
| Static verifier | [`scripts/diag/verify_qwen38_prefault_recipe.py`](../scripts/diag/verify_qwen38_prefault_recipe.py) |

This document defines a reusable **loading mechanism and qualification procedure**, not a universal
set of serving flags. The measured reference was a Qwen3.8-Flash-Next FP8 derivative with
`Qwen4ExpForConditionalGeneration` / `qwen4_exp`, dual GB10, TP=2, and expert parallel. The official
Qwen checkpoint and other fine-tunes still require a bounded checkpoint-specific clean-boot gate
before this overlay is adopted.

## 1. What the recipe changes

The overlay replaces six Python modules from the exact qualified vLLM image with byte-pinned source
files. Their shared `copy_weight_()` helper performs a page-stride first touch immediately before a
weight copy only when all of these are true:

```text
source is a CPU tensor
source and destination dtypes match
source is contiguous
destination is an integrated CUDA GPU
```

The helper touches one byte per host page and the final byte, then calls the original `dst.copy_(src)`.
It does **not** clone a tensor, prefetch the whole checkpoint, change checkpoint files, or retain an
anonymous second copy. Applying at copy sites also preserves TP rank-local slicing: each rank faults
only the slice it is about to copy rather than touching every full source tensor in the iterator.

Covered paths are the generic parameter loader, linear layers, vocabulary embedding, routed experts,
and the default weight loader. The mechanism is not conditioned on a Qwen model name. A
model-specific loader that bypasses these paths is outside the recipe and must be inspected before
use.

## 2. Measured reference

The matched clean-boot A–B–A–B series changed only prefault behavior while keeping the image,
checkpoint, TP/EP topology, KV policy, graph mode, and cache policy fixed.

| Metric | Baseline | Prefault |
|---|---:|---:|
| Mean readiness | 3452.30 s | 1810.15 s |
| Mean reduction | — | **47.567%** |
| Repeat 1 readiness | 3496.1 s | 1828.6 s |
| Repeat 2 readiness | 3408.5 s | 1791.7 s |

The promoted operational source then passed a separate clean boot at **1845.7** seconds. That run
also passed:

- deterministic c1: 5/5, content SHA-256
  `223af11cb89e535933eeddbb49bfddcc734cfe6c2d829ba4cde297805a338fc9`;
- 8 GiB KV reservation and 262,144-token concurrency `2.44x`;
- `MemAvailable` floors 9.244 GiB / 12.271 GiB;
- swap growth peaks 4.051 GiB / 2.474 GiB;
- zero container OOM kills/restarts and zero boot-scoped OOM/Xid/`NV_ERR_NO_MEMORY`;
- final health HTTP 200.

The result attributes the dominant improvement to mmap page faults accumulated across a very large
number of FP8/MoE tensors. It does not establish that every Qwen3.8 checkpoint will improve by the
same percentage.

## 3. Applicability

### High-confidence candidate

Use this recipe as a candidate when all of the following match:

- NVIDIA GB10 or another target for which `current_platform.is_integrated_gpu()` returns true;
- the exact qualified image/source ABI, including Python 3.12 site-package layout;
- default vLLM safetensors mmap loading, with no `--load-format` override;
- Qwen3.8 Flash Next FP8 architecture or a derivative using the covered shared/routed-expert copy
  paths;
- checkpoint size and host headroom compatible with the predeclared UMA safety gates.

### Requalification required

Run a fresh source port and full candidate gate when any of these change:

- vLLM image/version/commit or Python site-package path;
- quantization/layout (BF16, AWQ, GPTQ, NVFP4, different FP8 packing);
- loader (`fastsafetensors`, native sharded state, custom model loader);
- TP/EP topology, checkpoint shard layout, kernel/driver, or hardware;
- model-specific code that adds or bypasses weight-copy sites.

Do not mount these complete modules into another vLLM release. Re-extract that release's source,
port the helper against its actual copy sites, rerun the exact-byte/mmap fixture, and generate a new
manifest.

### Not applicable

- Discrete GPUs: the integrated-GPU gate disables prefault.
- Non-mmap loaders: this mechanism does not address their bottleneck.
- Post-load compilation or CUDA-graph stalls: prefault only changes weight-copy behavior.
- Serializer/export memory peaks: this is a load-time recipe, not a checkpoint conversion recipe.

## 4. Keep deployment-specific controls separate

**Graph-NONE and KV8 are not universal.** They were independently qualified for the measured
operational deployment:

- `cudagraph_mode=NONE` avoided a separate distributed post-KV graph-capture stall;
- `--kv-cache-memory-bytes 8589934592` preserved two 262,144-token sequences while maintaining the
  UMA floor.

Neither setting is required by the prefault helper. For another checkpoint, retain its qualified
Graph/KV settings and change only prefault for the first A/B. Test Graph-NONE or a fixed KV budget as
separate variables only when the target workload demonstrates the corresponding problem.

## 5. Static verification

Run from the repository root:

```bash
python3 scripts/diag/verify_qwen38_prefault_recipe.py
pytest -q scripts/diag/test_verify_qwen38_prefault_recipe.py
```

The verifier checks the six source hashes, helper guards, absence of cloning, source syntax, exact
head/worker read-only mount set, documentation contract, and Compose rendering. It never starts a
container.

## 6. Compose usage

Stack the prefault overlay after the model's normal overlay:

```bash
# Render only; does not start a container.
docker compose \
  --env-file presets/qwen3.8-flash-next-fp8-tp2-candidate.env \
  -f docker-compose.yml \
  -f compose/qwen3.8-flash-next/docker-compose.candidate.yml \
  -f compose/qwen3.8-flash-next/docker-compose.prefault-pr58868.yml \
  --profile head config
```

By default, the overlay resolves source files from
`./patches/qwen/prefault-pr58868`. Set `QWEN38_PREFAULT_ROOT` only when every node has an identical
absolute source root; verify hashes and rendered source/destination/read-only fields on both roles.

A separately authorized runtime launch remains worker-first. The overlay itself does not start,
restart, stop, promote, or enable auto-start for any service.

## 7. Per-checkpoint qualification gate

Use one clean-boot baseline and one clean-boot candidate when applying the mechanism to a new
checkpoint. A checkpoint with materially different copy-site coverage or memory behavior requires an
A–B–A confirmation before promotion.

Fail closed unless all gates pass:

1. **Static identity:** exact image digest, patch manifest, model revision, both rendered roles, and
   live read-only mounts.
2. **Fresh boot:** no existing containers, swap zero, boot-scoped fatal matches zero, and enough
   initial `MemAvailable` for the declared model.
3. **Readiness:** bounded total startup and bounded post-KV warmup; record rank-local weight-load
   timings.
4. **UMA:** predeclare an absolute `MemAvailable` floor and maximum swap growth. For the measured
   128 GiB nodes these were 8 GiB and 8 GiB respectively; do not weaken them to make a candidate pass.
5. **Safety:** zero OOM kill/restart, kernel OOM, Xid, and `NV_ERR_NO_MEMORY`.
6. **Capacity:** verify the target context/concurrency contract rather than copying the measured KV8
   setting blindly.
7. **Correctness:** deterministic c1 for the interactive profile and protocol/semantic validity at
   the representative batch profile.
8. **Promotion:** move qualified sources to a stable, checksummed operational path and verify one
   ordinary clean boot from the promoted source.

Stop after decisive evidence. Do not retry unsafe fast loaders, lower the memory floor, increase swap,
drop caches, or combine a vLLM/image upgrade with prefault qualification.
