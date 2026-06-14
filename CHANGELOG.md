# Changelog

All notable changes to `vllm-spark` (GHCR image + repo presets). Most recent on
top. See `git log` for the full commit history; this file is curated to describe
what users see (image tag, behavior, breaking changes) rather than every commit.

## `v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release` — Step-3.7-Flash validated NVFP4 image (2026-06-14)

GHCR immutable tag:
`ghcr.io/bjk110/vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release`

### Step-3.7-Flash FP8 serving (TP=2, dual DGX Spark GB10)

- **What**: Step-3.7-Flash (198B sparse MoE VLM) FP8 serving on dual DGX Spark GB10,
  TP=2 via Ray. Preset: `presets/step37-flash-fp8-tp2.env`. Image:
  `v022-d568-ngc2605-tx5102-vllm022-step3p7`.
- **Patches**: `patch_registry_step3p7.py` (model class registration).
- **Operational limit**: `MAX_NUM_SEQS=1` enforced by GB10 121 GiB UMA memory ceiling during
  FP8 warmup.
- **Reference**: [`docs/step3.7-flash-tp2.md`](docs/step3.7-flash-tp2.md).

### Step-3.7-Flash NVFP4 TP=2/EP=2 serving via ModelOpt quantization

- **What**: Step-3.7-Flash-NVFP4 serving on dual DGX Spark GB10, TP=2 EP=2 via Ray.
  Preset: `presets/step37-flash-nvfp4-tp2.env`. Image:
  `v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release`.
- **Patches**:
  - `patch_step3p7_nvfp4_input_scale.py`: adds missing `.input_scale` entries to
    `expert_params_mapping`; prevents NaN logits on all NVFP4 MoE layers.
  - `patch_step3p7_modelopt_cache_release.py`: feature-gated downstream workaround for
    cumulative CUDA caching-allocator reserved-memory growth during ModelOpt NVFP4 MoE
    MARLIN post-load conversion. Applies `torch.cuda.empty_cache()` after each
    `ModelOptNvFp4FusedMoE` module conversion (42/42 modules). Enabled via
    `VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE=1` in the NVFP4 preset.
- **Upstream relation**: vLLM PR #45179 merged to upstream main; **not included in
  v0.23.0**. First released version not yet confirmed. Downstream patch targets
  `ModelOptNvFp4FusedMoE` specifically (unconditional per-module release); upstream
  `release_device_memory_under_pressure()` is generic and threshold-conditional.
- **Formal acceptance**: 781,611 KV tokens generated, all HTTP 200, 5-minute
  continuous stability test. No NaN logits, no OOM, no inference errors.
- **Operational note**: Post-shutdown UMA residual (~19 GiB) remains on GB10 after
  container stop. Reboot is required to recover the full UMA pool before the next
  cold start.
- **Reference**: [`docs/step3.7-flash-tp2.md`](docs/step3.7-flash-tp2.md),
  [`docs/diagnostics/dgx-spark-uma-memory-freeze.md`](docs/diagnostics/dgx-spark-uma-memory-freeze.md).

## `v022-d568-fi-aot` — FlashInfer AOT kernel prebake (2026-06-11)

- **What**: New optional image `v022-d568-fi-aot`, a drop-in replacement for
  `v022-d568`. Pre-builds 7 FlashInfer SM120/SM121 kernel modules at image
  build time and promotes each `.so` from `spec.jit_library_path` to
  `spec.aot_path`, so `spec.is_aot=True` at runtime and `build_and_load()`
  skips ninja for those modules (notably `fused_moe_120`).
- **Validated**: dual-node RDMA TP=2 with `Qwen/Qwen3.6-35B-A3B` — Ray join,
  model load, profiling, and CUDA graph capture all completed with zero
  `ninja`/`nvcc`/`cicc`/`ptxas` processes for the 7 prebaked specs;
  `/health` and `/v1/completions` both OK on both nodes.
- **Scope**: only the 7 specs listed in
  [`docs/flashinfer-aot-prebake.md`](docs/flashinfer-aot-prebake.md) are
  AOT-promoted — other FlashInfer module/dtype/shape combinations still
  JIT-compile on first use as before.
- **User impact**: no required action. To use, set
  `VLLM_IMAGE=vllm-spark:v022-d568-fi-aot` (or the GHCR tag) in place of
  `v022-d568` in any preset `.env`.

## v021-tq — Qwen3.5 hybrid codegen workaround (`3070f9a`, 2026-04-26)

- **What**: All `*-tq.env` presets that exercise Qwen3.5 hybrid models (i.e.
  `qwen3.5-397b-int4-tq.env`) now pass
  `--compilation-config {"use_inductor_graph_partition":true}` in
  `VLLM_EXTRA_ARGS`.
- **Why**: After vLLM main `951dca80` (PR #38657, "Invoke split FX graph by
  codegen"), `compilation/codegen.py::_node_ref()` falls back to the default
  `repr()` for opaque arguments. Qwen3.5 hybrid GDN attention takes a
  `LayerName` opaque arg, so the generated execution function source ends up
  with `<vllm.utils.torch_utils.LayerName object at 0x...>`, which `exec(code)`
  rejects with `SyntaxError: invalid syntax` during `EngineCore` init.
- **User impact**: No image rebuild required — the workaround is a runtime
  flag and is already baked into the affected preset(s). Cold-start engine
  init takes roughly 2× longer (≈ 440 s vs 250 s for `--enforce-eager` on
  397B INT4 TP=2) because Inductor performs the partition itself, but
  CUDAGraph (`FULL_AND_PIECEWISE`) stays enabled for steady-state inference.
- **Last-resort fallback**: `--enforce-eager` (disables torch.compile and
  CUDAGraph entirely). Hot-patch alternative: `patches/patch_codegen_fx_repr.py`
  (apply via `docker exec` — does not require image rebuild).
- See [docs/troubleshooting.md](docs/troubleshooting.md) for details.

## v021-ngc2603 / v021-tq — base stack bump (`8623187`, 2026-04-25)

- **What**: `Dockerfile.gemma4` build args bumped to
  - `VLLM_COMMIT=95995bbef81292e3ee1ef0df5ca3989bb481bdd5` (vLLM main, +236
    commits since v020 baseline `978a4462`).
  - `FLASHINFER_REF=v0.6.9` (was `v0.6.8`).
  - `TRANSFORMERS_VER=5.5.4` (unchanged from base-refresh).
- **Why**:
  - **vLLM**: PR #40060 (TurboQuant backend selection), #40092 (FA3/FA4
    prefill), #40194 (random-signs cleanup) all merged upstream — the
    in-tree TurboQuant patch set in `apply_turboquant_fixes.py` shrinks
    accordingly (see [PATCH_STATUS.md](PATCH_STATUS.md)).
  - **FlashInfer**: PR #3113 adds b12x FP4 GEMM SM121 (GB10) support; #3066
    adds b12x CuTe DSL fused MoE for SM120.
- **User impact**: Two GHCR tags published from this commit:
  - `ghcr.io/bjk110/vllm-spark:v021-ngc2603` — base image, no TurboQuant
    bugfix patches (suitable for any non-TQ preset).
  - `ghcr.io/bjk110/vllm-spark:v021-tq` — base + `apply_turboquant_fixes.py`
    runtime patches required by `*-tq.env` presets that hit hybrid models.
  All `*.env` presets default to one of these two tags. No env file edits
  needed beyond `MODEL_PATH`.

## CLUSTER_MODE — single Spark without RDMA (`98af63f`, 2026-04-22)

- **What**: New environment variable `CLUSTER_MODE` (`single` or `dual-rdma`)
  read by `entrypoint.sh` and `docker-compose.yml`. `single` is the default
  in `.env.example` and in single-node presets so a fresh checkout boots
  on one Spark with no RDMA configuration.
- **Why**: PyTorch c10d would try to bind `VLLM_HOST_IP=10.10.10.1` (a RoCE
  IP that does not exist on a single host) and stall with
  `tcp://10.10.10.1:<port> server socket has timed out`. Setting
  `CLUSTER_MODE=single` makes the entrypoint force `VLLM_HOST_IP=127.0.0.1`,
  unset `NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME / UCX_NET_DEVICES /
  NCCL_IB_HCA`, and set `NCCL_IB_DISABLE=1`.
- **User impact**:
  - Single-Spark presets (`gemma4-26b-a4b.env`, `redhatai-122b-nvfp4.env`,
    `qwen3.5-122b-prismaquant.env`, `qwen3.6-35b-fp16.env`,
    `intel-122b-int4.env`, `qwen3.5-122b-nvfp4.env`,
    `wangzhang-122b-nvfp4.env`, all `-tq.env` for TP=1 models) ship with
    `CLUSTER_MODE=single` and have the RDMA lines commented out.
  - Dual-Spark presets (`qwen3.5-122b-fp8.env`, `qwen3.5-122b-nvfp4-tp2.env`,
    `qwen3.5-397b-int4.env`, `wangzhang-122b-fp8.env`,
    `qwen3.5-397b-int4-tq.env`) ship with `CLUSTER_MODE=dual-rdma`.
  - `entrypoint.sh` fail-fasts if you try to mix modes (e.g.
    `CLUSTER_MODE=single` with `TP_SIZE=2`).

## turboquant-rebase-20260417 — TQ KV bugfixes + 397B sweep (`ce7b437`, 2026-04-17)

- **What**:
  - New patch `patches/apply_turboquant_fixes.py` cherry-picks **open**
    upstream PRs needed for DGX Spark stability:
    - PR #40074 — Triton decode index OOB fix.
    - PR #39988 — BF16 FP8 cast fix.
    - PR #39931 — Hybrid model support (Qwen3.5).
    Two earlier PRs (#40060, #40092) had already merged upstream by the v021
    bump and are no longer cherry-picked.
  - Three new TurboQuant presets:
    `gemma4-26b-a4b-tq.env`, `redhatai-122b-nvfp4-tq.env`,
    `qwen3.5-397b-int4-tq.env`.
  - Full 397B INT4 + TurboQuant capacity / quality / throughput sweep
    (`turboquant_3bit_nc` / `turboquant_k3v4_nc` / `turboquant_4bit_nc` /
    `turboquant_k8v4`) — results in `benchmarks/llama-benchy/results_397b-int4-tq-*-c1to4.md`
    and `benchmarks/results/*_Qwen3.5-397B-A17B-int4-AutoRound_mt30000_*.txt`.
- **Why**: Upstream TQ shipped via PR #38479 in `978a4462` but several open
  follow-up PRs were needed before TQ was usable on Qwen3.5 hybrid + GB10.
- **User impact**: `--kv-cache-dtype turboquant_4bit_nc` is the
  recommended operational mode for 397B INT4 (3.8× KV compression, +2.7%
  PPL, peak 84 t/s tg128 c=4 — see [README.md → 397B INT4 TP2 — TurboQuant
  KV Cache Sweep](README.md#397b-int4-tp2--turboquant-kv-cache-sweep)).

## base-refresh-20260417 — base stack refresh (`a7bb0ef`, 2026-04-17, intermediate `v020-ngc2603`)

- **What**: `Dockerfile.gemma4` base bumped to vLLM main `978a4462`,
  FlashInfer `v0.6.8`, Transformers `5.5.4`. Three earlier compatibility
  patches removed because they merged upstream:
  `fix_cuda13_memcpy_batch`, `qwen3_5_moe_rope_fix`, `pr38423_nvfp4_spark`
  (see [PATCH_STATUS.md](PATCH_STATUS.md)).
- **Why**: Pull in upstream TurboQuant (PR #38479), Gemma 4 stability
  fixes, FlashInfer SM121 tile filtering / NVFP4 group GEMM / FP4 CUTLASS.
- **User impact**: GHCR tag `v020-ngc2603` was a transient release tag for
  this base stack; presets briefly carried `v020-ngc2603` before the v021
  bump. Functionally equivalent to v021 for any non-TQ preset; v021 is
  preferred because of the FlashInfer SM121 b12x FP4 GEMM addition.

## v019-ngc2603 — Gemma 4 + FlashInfer 0.6.7.post3 (`7736716` / `b0a1454`, mid-Apr 2026)

- **What**: vLLM bumped to `0.19.1` source build (commit `a7d79fa`),
  Transformers `5.5.0`, FlashInfer `v0.6.7.post3`. Adds Gemma 4 support
  and async scheduling. TTFT improved ~2× over v018 on Qwen3.5 122B FP8.
- **Why**: Production Gemma 4 26B-A4B-it serving + Qwen3.5 latency parity.
- **User impact**: Superseded by v020 / v021. Image tag remains pullable
  for users who want this stack.

## v018-ngc2603 — initial NGC 26.03 source build (`70afdc9`, early Apr 2026)

- **What**: First NGC 26.03 (CUDA 13.2, PyTorch 2.11) source build of
  vLLM `0.18.3` (commit `c494977`) with FlashInfer `v0.6.7`, Transformers
  `5.2.0`, and the SM121 / PyTorch 2.11 patch stack baked in (hoist
  removal, fastsafetensors natural sort, AOT cache fix, nogds, MoE
  configs, SM121 patches). Earlier NGC 26.01 + cu130 wheel attempts
  (`Dockerfile`) live alongside as historical references.
- **Image**: `ghcr.io/bjk110/vllm-spark:v018-ngc2603` (matching git tag
  `v018-ngc2603`).
- **User impact**: This is the only currently-published Git tag (see
  [README.md → Image tags & Git tags](README.md#image-tags--git-tags)).
  v019 onwards do not have matching Git tags yet — recommendations in
  the README.

## Earlier history

Pre-`v018-ngc2603` work (NGC 26.01 cu130 wheel image, single `Dockerfile`,
`Dockerfile.nvfp4` extension, initial Qwen3.5 122B INT4/FP8/NVFP4 preset
collection) is preserved in `git log` and the original
`README` (now superseded). Of note:

- `archive/feat-turboquant` — git tag preserving the legacy TurboQuant
  branch before the 2026-04-17 rebase.
- `Dockerfile.ngc2603-v3` — the v018-ngc2603 build file, kept for
  reproducibility.
- `Dockerfile.nvfp4` — small extension over a base image setting NVFP4
  runtime defaults; rarely used now (NVFP4 presets export the same env
  vars directly).
