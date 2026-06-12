# Troubleshooting

This document collects model-path-specific and stack-specific troubleshooting
notes that were previously spread across the top-level `README.md`. For the
general Quick Start flow (image pull/build, preset selection, starting
services), see [`README.md` § Quick Start](../README.md#quick-start).

For dual-node `TP_SIZE=2` + `DISTRIBUTED_BACKEND=ray` +
`--enable-expert-parallel` deployments where a node becomes unresponsive
(SSH banner-exchange timeout, ping still works) during startup, see the
diagnostic runbook:
[`docs/diagnostics/dgx-spark-uma-memory-freeze.md`](diagnostics/dgx-spark-uma-memory-freeze.md).

## General Docker Compose checks

### `[c10d] The server socket on [::ffff:10.10.10.1]:<port> has timed out, will retry.`

This means a single-Spark setup is leaking RDMA env (`VLLM_HOST_IP=10.10.10.1`,
`GLOO_SOCKET_IFNAME=enp1s0f0np0`, etc.) into the container, and PyTorch c10d
can't bind to a RoCE IP that doesn't exist on this host. Fix:

1. Confirm `.env` (or the preset you copied) has `CLUSTER_MODE=single`.
2. Make sure the `HEAD_ROCE_IP=…` / `ROCE_IF_NAME=…` / `IB_HCA_NAME=…` lines
   are **commented out** (lines starting with `#`) in single-mode presets.
3. Recreate the container:
   ```bash
   docker compose --profile head down
   docker compose --profile head up -d
   ```
   `entrypoint.sh` will print
   `CLUSTER_MODE=single: VLLM_HOST_IP=127.0.0.1, NCCL_IB_DISABLE=1, NCCL/GLOO/UCX ifname cleared`
   on a clean single-Spark start.

## Empty optional env vars (`${VAR:-}` defaults, issue #14)

`docker-compose.yml` forwards many optional knobs into the container as
`${VAR:-}`. If a preset doesn't set `VAR`, the container still gets
`VAR=""` — an env var that is **set to an empty string**, not unset. Some
vLLM / FlashInfer / Triton-DG env parsers treat those very differently from a
truly-unset var:

- `VLLM_NVFP4_GEMM_BACKEND=""` → `ValueError: Invalid value '' for
  VLLM_NVFP4_GEMM_BACKEND` (enum-validated env var)
- `FLASHINFER_CUDA_ARCH_LIST=""` → `flashinfer/compilation_context.py` does
  `arch.split(".")` → `ValueError: not enough values to unpack (expected 2,
  got 1)`
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=""` → `invalid literal for int()
  with base 10: ''`, which can leave CUDA-graph capture under-provisioned and
  the EngineCore gets OOM-killed during graph capture

**Fix (already applied as of this writing — pull latest `main`):**
`entrypoints/entrypoint.sh` runs `unset_empty_optional_envs()` at startup,
before anything else, and `unset`s any of ~24 known optional env vars
(`VLLM_NVFP4_GEMM_BACKEND`, `VLLM_ATTENTION_BACKEND`,
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, `FLASHINFER_CUDA_ARCH_LIST`,
`TORCH_CUDA_ARCH_LIST`, the B12X/NCCL/DG-JIT toggles, etc.) if they were
forwarded as an empty string. This makes vLLM/FlashInfer fall back to their
normal "unset" defaults instead of crashing on `""`. If you're still hitting
one of the errors above, confirm your image's `/entrypoint.sh` actually
contains `unset_empty_optional_envs` (older images baked the entrypoint in,
so a `docker compose pull`/restart with an old image won't pick up a `main`
fix — rebuild or bind-mount the updated `entrypoints/entrypoint.sh`).

**NVFP4 on GB10/sm_121 — pin explicit values, don't rely on defaults:**
presets that pass `--quantization nvfp4` (runtime NVFP4, e.g.
`qwen3.5-122b-nvfp4*.env`) genuinely exercise the FlashInfer NVFP4 GEMM and
CUDA-graph profiling paths, so set these explicitly rather than letting the
sanitizer unset them:

```bash
VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
FLASHINFER_CUDA_ARCH_LIST=12.1
TORCH_CUDA_ARCH_LIST=12.1a
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
```

**Diagnostic command** — render the final env that a preset produces (after
`${VAR:-}` substitution) without starting a container:

```bash
MODEL_PATH=/tmp docker compose --env-file presets/<preset>.env --profile head config \
  | grep -E "VLLM_NVFP4_GEMM_BACKEND|VLLM_ATTENTION_BACKEND|FLASHINFER_CUDA_ARCH_LIST|TORCH_CUDA_ARCH_LIST|VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"
```

A value of `""` here is fine for the entrypoint sanitizer to handle, but for
NVFP4 presets it should instead show the explicit values above. Run
`bash scripts/check-empty-env.sh` to check all presets at once — it fails
only if a runtime-NVFP4 preset is missing one of the four pins, and prints
informational notes for everything else.

## Model path and preset issues

- `MODEL_PATH` must point to a local model-weight directory — `presets/*.env`
  files are configuration references only and do not contain model weights.
  See [`presets/README.md`](../presets/README.md).
- If the container can't find the model, confirm the host path in `MODEL_PATH`
  and the container mount point in `MODEL_CONTAINER_PATH` agree, and that the
  preset's `SERVED_MODEL_NAME` matches what you intend to query.
- If you copied a preset with `cp presets/<preset>.env .env`, re-run the
  `sed -i 's|\[model_path\]|...|' .env` substitution shown in
  [`README.md` § Quick Start](../README.md#quick-start) — a literal
  `[model_path]` placeholder left in `.env` will fail to resolve.

## dsv4-d568 issues

For known pitfalls and fixes specific to the primary DeepSeek-V4-Flash path
(`dsv4-d568`) — including the PyTorch 2.12.0a0 alpha `split_module` issue, the
forum-recommended `dda4668b` pin's MTP graph hang, `max_num_scheduled_tokens`
warnings with MTP, the `--attention_config.use_fp4_indexer_cache=True` GB10
limitation, page-cache-induced GPU memory shortages, and KV pool sizing — see
[`docs/dsv4-flash-tp2.md` § 4. 알려진 함정과 픽스](dsv4-flash-tp2.md#4-알려진-함정과-픽스).

## unholy-fusion issues

For the experimental high-prefill `unholy-fusion` path's known operational
limits (`MAX_NUM_SEQS`, `MAX_MODEL_LEN`, MTP `n`, the
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` requirement, the
`VLLM_USE_BREAKABLE_CUDAGRAPH` garbled-output trap), the GB10 UMA memory-check
patches it requires, and the switching procedure to/from `dsv4-d568`, see
[`docs/unholy-fusion-benchmark.md`](unholy-fusion-benchmark.md).

## Qwen or other experimental preset issues

### `SyntaxError: invalid syntax` in `compilation/codegen.py` during EngineCore init (Qwen3.5 hybrid + torch.compile)

```
File "<string>", line 5
    gdn_attention_core = torch.ops.vllm.gdn_attention_core(..., <vllm.utils.torch_utils.LayerName object at 0x...>)
                                                                ^
SyntaxError: invalid syntax
```

vLLM main since `951dca80` (PR #38657 "[compile] Invoke split FX graph by codegen") emits the default `repr()` of opaque arguments like `LayerName` into the generated execution function source. The hybrid GDN attention path used by Qwen3.5 takes a `LayerName` and trips this every cold start.

**Workaround (recommended, no source patch)** — pass `use_inductor_graph_partition=True` so torch.compile uses Inductor's own partitioning instead of vLLM's split-by-codegen:

```
VLLM_EXTRA_ARGS=... --compilation-config {"use_inductor_graph_partition":true}
```

This keeps torch.compile + CUDAGraph (`FULL_AND_PIECEWISE`) enabled. Cold-start engine init is roughly 2× longer (≈ 440 s vs 250 s for `--enforce-eager` on 397B INT4 TP=2) due to the extra Inductor compile, but steady-state inference benefits from CUDA graph capture.

**Last-resort workaround** — `--enforce-eager`. Disables torch.compile and CUDAGraph entirely; loses inference performance but guaranteed to bypass the codegen path.

**Hot-patch (kept on standby)** — `patches/archive/patch_codegen_fx_repr.py` rewrites `_node_ref()` to honor `__fx_repr__()` and merges its namespace into the `exec()` scope. Apply only if a future vLLM bump regresses the Inductor partition path or a different opaque type triggers the same SyntaxError:

```bash
docker exec vllm-spark-head python3 /patches/archive/patch_codegen_fx_repr.py
docker compose --profile head restart
```

For the experimental `Qwen3.6-35B-A3B FP16` test preset's setup notes and
first-boot troubleshooting steps, see
[`docs/model-serving-validation-history.md` § Qwen3.6-35B-A3B FP16 — Experimental test preset](model-serving-validation-history.md#qwen36-35b-a3b-fp16--experimental-test-preset-setup-notes).

## Logs and verification commands

```bash
# Health check
curl http://localhost:8000/health      # single
curl http://spark01:8000/health        # dual-rdma

# Follow logs live
docker logs -f vllm-spark-head
docker logs -f vllm-spark-worker

# Check for startup completion / errors
docker logs vllm-spark-head 2>&1 | grep "Application startup complete"
docker logs vllm-spark-worker 2>&1 | grep -E "startup|ready|error" | tail -5
```

See [`README.md` § Quick Start → 3. Verify](../README.md#3-verify) for the
baseline health-check commands.
