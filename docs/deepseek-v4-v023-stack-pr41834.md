# DeepSeek-V4-Flash SM12x — v0.23-stack-derived, PR #41834-based (EXPERIMENTAL)

**Status: validated experimental** (2026-06-21). A **validated experimental**
baseline: the image is built, published, and runtime-confirmed on dual DGX Spark
(startup + correctness + reasoning parser + tool parser). It is **not** production,
not promoted production, not a stable release, and is **not** performance-,
long-context-, graph-, MTP-, or prefix-cache-validated. Use the phrase
"validated experimental" consistently.

## Published image (validated experimental)

| Field | Value |
|---|---|
| GHCR immutable tag | `ghcr.io/bjk110/vllm-spark:v023-stack-dsv4-sm12x-pr41834-exp-72261a7-tvmfam019` |
| GHCR manifest digest | `sha256:2f4a96283fc5b491d5e28cee607525e32e914615bb469978beb2336cf8e62c44` |
| Local/config image ID | `sha256:4c41950c47ecb771eead3e32147a750240d6605d15ee99c54b43e0a85ea26105` |
| Manifest media type | `application/vnd.docker.distribution.manifest.v2+json` |
| Architecture / OS | arm64 / linux |
| vLLM source commit | `72261a7af149fa5d3fe2ed2b9956e92590731012` (PR #41834, OPEN) |
| Repository baseline | `9ceabf3` (this commit adds the recipe) |
| Model | `deepseek-ai/DeepSeek-V4-Flash` (official FP8 base only) |
| apache-tvm-ffi / tilelang / FlashInfer | 0.1.9 / 0.1.9 / 0.6.12 |
| Image created | 2026-06-21T16:36:11+09:00 |
| OCI labels | `org.vllm-spark.experimental=true`, `org.opencontainers.image.source=https://github.com/bjk110/spark_vllm_docker` |
| Classification | validated experimental |

Reproducible pull (digest-qualified form is the canonical reference):

```
docker pull ghcr.io/bjk110/vllm-spark:v023-stack-dsv4-sm12x-pr41834-exp-72261a7-tvmfam019
docker pull ghcr.io/bjk110/vllm-spark@sha256:2f4a96283fc5b491d5e28cee607525e32e914615bb469978beb2336cf8e62c44
```

Rollback / lineage images (preserved, not published as the recipe): old pre-fix
`...-exp-72261a7` (image `3cb14c02`, TVM-FFI drift, aborts at sparse profiling — do
NOT use); intermediate apache-tvm-ffi-only `...-tvmffi019` (`22eefdcc`). The
suffixless base tag is NOT the validated image.

Classification legend used below: **Confirmed** (verified from an authoritative
Git source), **Current baseline** (the project's shipped v0.23 stack),
**Experimental**, **Historical**, **Unverified**.

## Objective

Create an isolated experimental DeepSeek-V4-Flash build path that preserves the
project's current v0.23 image toolchain (CUDA/PyTorch/ARM64/FlashInfer/NCCL/SM121
patches) but builds vLLM from the latest source-supported DeepSeek-V4 SM120/SM121
implementation, excluding the Step-3.7 tokenizer overlay, without bulk-cherry-picking
the historical unholy-fusion or aidendle94 forks. Stop before building or loading.

## Repository baseline (Confirmed)

- Canonical Git host: **homeserver** (`/home/bjk110/docker/vllm-spark`). Authoring
  happens here; spark01/spark02 are build/run nodes (operational clones).
- Branch `main`, HEAD `9ceabf3`, origin/main `9ceabf3`, ahead/behind `0/0`, clean.
- `9ceabf3` = the previous task's expected `6c06fc3` plus one approved, already-pushed
  Step-3.7 FP8 preset-promotion commit. `6c06fc3` is a clean historical ancestor.
- spark01's clone is at old `7777aebc` with seven tracked operational modifications
  (a dirty historical/operational clone) — not used for authoring; see below.

## Current v0.23 image-stack baseline (Current baseline)

Lowest reusable stage: `dockerfiles/active/Dockerfile.v023-d568` (multi-stage:
flashinfer-builder → vllm-builder → runner). It contains the v0.23 toolchain and
no Step-3.7 model behavior or tokenizer overlay.

| Component | Pin |
|---|---|
| NGC base | `nvcr.io/nvidia/pytorch:26.05-py3` |
| PyTorch | 2.12.0a0 (NGC-bundled; CUDA 13.2) |
| vLLM (base) | 0.23.0 @ `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| FlashInfer | v0.6.12 (built from source) |
| Transformers | 5.10.2 |
| Triton | 3.7.0 |
| NCCL | 2.30.4+cuda13.2 |
| TORCH_CUDA_ARCH | 12.1a (SM121/GB10) |
| SM121 patches kept | `apply_sm121_patches.py`, `apply_turboquant_fixes.py`, `patch_split_module_compat.py`, `fix_pytorch211_compat.py` |
| Prometheus fix | introduced only in the Step-3.7 promfix layer (NOT in this base) |
| Tokenizer overlay | introduced only in the Step-3.7 overlay layer (NOT in this base) |

## Selected vLLM source (Confirmed)

- Official vLLM 0.23.0 release: tag `v0.23.0` → `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- DeepSeek-V4 SM12x preview: **vLLM PR #41834** — title "[New Model][Nvidia] Add
  SM12x support for DeepSeek V4 Flash with essential fixes", head_repo `jasl/vllm`,
  head_ref `codex/ds4-sm120-min-enable`, **state OPEN**, base `main` @ `93bad119…`,
  126 commits, +15555/-372, 116 files.
- Pinned commit: **`72261a7af149fa5d3fe2ed2b9956e92590731012`**. The annotated tag
  `sm120-pr-41834-stable-preview-20260621` (tag object `5c5e8c18…`) dereferences
  (`^{}`) to exactly `72261a7…`; `refs/pull/41834/head` on vllm-project/vllm is the
  same SHA. Tag→commit resolution confirmed.
- Relationship to v0.23.0: `git compare 0fc695f…72261a7` = **diverged, ahead 586,
  behind 6**. It is a post-0.23.0 `main` snapshot plus the 126 DeepSeek-V4 commits.
  Therefore this image is **v0.23-stack-derived / PR41834-based / experimental —
  NOT vLLM 0.23 itself.**

## Model identity (Confirmed)

Two complete DeepSeek-V4-Flash checkpoints exist on homeserver (no partial files):

| | `deepseek-ai_DeepSeek-V4-Flash` (base) | `DeepSeek-V4-Flash-W4A16-FP8` (derived) |
|---|---|---|
| arch / model_type | DeepseekV4ForCausalLM / deepseek_v4 | same |
| dtype | bfloat16 | bfloat16 base |
| hidden / layers | 4096 / 43 | 4096 / 43 |
| routed/shared experts, top-k | 256 / 1, top-6 | same |
| max_position_embeddings | 1,048,576 | 1,048,576 |
| vocab | 129280 | 129280 |
| quantization | FP8 block [128,128], e4m3, dynamic act, scale ue8m0 | compressed-tensors **mixed-precision**: attn/indexer FP8 [128,128]; routed experts **INT4** pack-quantized (group 128, actorder static); shared_experts + lm_head bf16 |
| shards / size | 46 / ~159.6 GB | 4 / ~152.5 GB |
| tokenizer.json | present | present |

The bootstrap preset targets the **FP8 base** (simplest dispatch — pure block FP8,
analogous to the Step-3.7 FP8 path). The **W4A16-FP8** mixed INT4/FP8 checkpoint is
a separate later experiment (it exercises Marlin WNA16 INT4 experts + FP8 attention
via compressed-tensors). The "intended" single model is ambiguous (two are present);
this should be confirmed with the user before bring-up.

## Historical lineage (Historical / Unverified)

- **unholy-fusion** (`local-inference-lab/vllm` `dev/unholy-fusion`,
  `docs/unholy-fusion-benchmark.md`): the `VLLM_USE_B12X_MOE` family of feature
  gates. **The pinned PR #41834 source does NOT contain `VLLM_USE_B12X_MOE` in
  `envs.py`** — the B12X gate is gone / superseded by automatic backend selection.
  Do not add B12X flags. (Historical / SUPERSEDED_BY_UPSTREAM.)
- **aidendle94** `vllm` `DSV4-2`; images `sparkrun-vllm-ds4-gb10:production-ready`
  and `:production-v2` (digest `sha256:2ae96f30013ade72693aca00553d92a2bfe6759c5e94de5346d060639f54648e`).
  Do **not** use either Aiden image as a base. production-v2 source revision cannot
  be proven from an opaque image and is therefore **Unverified**; do not extract
  source from its binary layers and represent it as provenance.

## Source provenance / equivalence matrix

`v023` = present in vLLM 0.23.0 (`0fc695f`); `pr41834` = present in `72261a7`.

| Area | Source path (pr41834) | In v023 | In pr41834 | Classification | Recommendation |
|---|---|---|---|---|---|
| DeepSeek-V4 model arch (registry) | registry `DeepseekV4ForCausalLM` | yes (3 registry mentions) | yes (restructured to `vllm/models/deepseek_v4/...`) | ALREADY_IN_V023 (+ restructured) | use pr41834 layout |
| SM12x model impl | `vllm/models/deepseek_v4/nvidia/model.py` | no | yes | REQUIRED_SM121_KERNEL_CHANGE | include (built from source) |
| Sparse MLA (prefill/decode) | `…/sparse_mla.py`, `v1/attention/backends/mla/sparse_mla_kernels.py`, `sparse_swa.py` | no | yes | REQUIRED_SM121_KERNEL_CHANGE | include |
| Sparse indexer | `v1/attention/backends/mla/indexer.py`, `layers/sparse_attn_indexer.py` | no | yes | REQUIRED_SM121_KERNEL_CHANGE | include |
| Hybrid KV manager | `v1/core/kv_cache_coordinator.py`, `kv_cache_manager.py`, `single_type_kv_cache_manager.py` | partial | yes | PRESENT_IN_PR41834 | include |
| MTP | `…/nvidia/mtp.py`, `tests/v1/spec_decode/test_mtp.py` | partial | yes | OPTIONAL_PERFORMANCE_CHANGE | defer (off in bootstrap) |
| Marlin MoE WNA16 (INT4) | `csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu`, `kernels/linear/scaled_mm/marlin.py` | PR #40923 native SM12x cubins upstream in v023 | yes (extended) | SUPERSEDED_BY_UPSTREAM (base) + pr41834 extensions | rely on upstream; W4A16 later |
| FP8 Marlin selection | `tests/…/test_fp8_marlin_kernel_selection.py` | yes | yes | ALREADY_IN_V023 | n/a |
| FP8 einsum / o_proj | `…/nvidia/ops/fp8_einsum.py`, `o_proj.py` | no | yes | REQUIRED_SM121_KERNEL_CHANGE | include |
| SM12x MQA + packed-KV int64 offset | `…/nvidia/ops/sm12x_mqa.py` | no | yes (`block_idx…​.to(tl.int64)`) | **REQUIRED_CORRECTNESS_CHANGE** | include; verified present |
| FlashInfer SM120 decode | `…/nvidia/flashinfer_sm120_decode.py` | no | yes | OPTIONAL_PERFORMANCE_CHANGE | default off; opt-in later |
| FlashMLA | `…/nvidia/flashmla.py` | no | yes | PRESENT_IN_PR41834 | source default |
| DeepGEMM SM120 | `…/nvidia/ops/sm12x_deep_gemm_fallbacks.py` | n/a | fallbacks only (torch/triton) | REQUIRED_SM121_KERNEL_CHANGE (fallback) | do NOT force DeepGEMM |
| Prefix-cache selective retention | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` (envs.py) | PR #43447 | yes | SUPERSEDED_BY_UPSTREAM | use upstream env; no write fence |
| Reasoning parser / tokenizer | `vllm/reasoning/deepseek_v4_reasoning_parser.py`, `vllm/tokenizers/deepseek_v4_encoding.py` | no | yes | PRESENT_IN_PR41834 | confirm registered names, then enable |
| B12X MoE gate | `VLLM_USE_B12X_MOE` | no | **no** | REJECTED_OR_DISPROVEN | do not add |
| Init memcheck bypass | `VLLM_SKIP_INIT_MEMORY_CHECK` (spark01 patch) | n/a | n/a | REJECTED (forbidden) | do not add |
| Breakable CUDA graph | `VLLM_USE_BREAKABLE_CUDAGRAPH` | yes (default on) | yes | REQUIRED_CORRECTNESS (disable) | keep `=0` |
| NVFP4 GEMM backend override | `VLLM_NVFP4_GEMM_BACKEND` | yes | yes | HISTORICAL_ONLY | leave unset |

### Required correctness condition (Confirmed)

`vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py` (726 lines) computes the packed
FP8 paged-MQA-logits KV offset with an int64-promoted block index in BOTH the
non-rowwise kernel (`_fp8_paged_mqa_logits_kernel`, `block_idx[:, :, None].to(tl.int64)
* stride_kvb`) and the rowwise kernel. This is the int64-compatible offset that
prevents int32 overflow on the non-rowwise paged MQA path. **Present — required
correctness condition satisfied.**

### DeepGEMM conclusion (Confirmed)

DeepGEMM is **not natively available on SM120/SM121**. The pinned source ships
`sm12x_deep_gemm_fallbacks.py` ("SM12x fallback implementations for DeepGEMM-only
interfaces"; Triton/torch fallbacks; `NotImplementedError` for non-FP8 Q). Do not
enable/force DeepGEMM by broadening an arch gate; rely on the fallbacks.

## Build-time compatibility (static review)

| Requirement (pr41834) | Image provides | Verdict |
|---|---|---|
| flashinfer-python/cubin == 0.6.12 | v0.6.12 | exact match |
| transformers >= 5.5.3 | 5.10.2 | OK |
| requires-python >=3.10,<3.15 | 3.12 | OK |
| torch == 2.11.0 (requirements pin) | NGC 2.12.0a0 (built from source) | RISK — same source-build-on-NGC pattern as v0.23.0; ABI/API drift possible |
| nvidia-cutlass-dsl[cu13] == 4.5.2 | new dependency | pinned in the Dockerfile; availability to confirm at build |
| nvidia-cudnn-frontend >= 1.19.1 | NGC-provided / pip | confirm at build |

## Selected implementation strategy

**Preferred strategy (selected):** reuse the v0.23 image toolchain (NGC 26.05 +
FlashInfer 0.6.12 + Transformers 5.10.2 + Triton 3.7.0 + NCCL 2.30.4 + SM121
patches) and build vLLM from the pinned PR #41834 commit `72261a7`. This avoids a
126-commit bulk backport: the whole PR tree IS the vLLM source. Static review
passes (FlashInfer/Transformers/Python compatible; correctness fix present;
DeepGEMM fallbacks present). The torch pin and the new cutlass-dsl dependency are
documented build-time risks, not blockers.

## New repository files (this preparation)

- `dockerfiles/active/Dockerfile.v023-stack-dsv4-sm12x-pr41834`
- `presets/deepseek-v4-v023-stack-pr41834-bootstrap-tp2.env`
- `presets/deepseek-v4-v023-stack-pr41834-graph-tp2.env`
- `presets/deepseek-v4-v023-stack-pr41834-prefixcache-tp2.env`
- `presets/deepseek-v4-v023-stack-pr41834-reasoning-parser-tp2.env`
- `presets/deepseek-v4-v023-stack-pr41834-tool-parser-tp2.env`
- `docs/deepseek-v4-v023-stack-pr41834.md` (this file)

## Runtime plans

- **Bootstrap:** FP8 base checkpoint, TP=2 mp, EP off, MTP off, prefix off, CUDA
  graph off (enforce-eager), `MAX_NUM_SEQS=1`, `MAX_NUM_BATCHED_TOKENS=2048`,
  `MAX_MODEL_LEN=8192`, fixed 2 GiB KV, numeric RoCE IPs, no Ray, no memcheck
  bypass. Record the auto-selected MoE/attention backends from the startup log.
- **Graph experiment:** only graph settings — `cudagraph_mode=FULL_AND_PIECEWISE`
  (confirmed in source), breakable graph stays off, profiler estimate cudagraphs
  on. MTP/prefix unchanged.
- **Prefix-cache experiment:** only prefix-cache settings — enable prefix caching;
  optionally `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` (confirmed in source); no write
  fence.
- **MTP experiment (deferred):** enable MTP only after graph + prefix are clean.
- **Long-context progression (deferred):** 8192 → 32768 → larger, re-validating
  memory headroom at each step. 1M is the model ceiling, not a starting point.

## Rollback

This path is purely additive (new files + a new image tag). The Step-3.7 FP8 and
NVFP4 presets, images, docs, and benchmarks are untouched. To abandon, delete the
new files / experimental image tag; nothing else changes.

## Known risks

- torch 2.11.0 (source pin) vs NGC 2.12.0a0 (build host) ABI/API drift.
- New `nvidia-cutlass-dsl[cu13]==4.5.2` dependency must resolve at build.
- W4A16-FP8 mixed INT4/FP8 dispatch (compressed-tensors + Marlin WNA16) is untested
  on GB10 here; the bootstrap deliberately uses the FP8 base.
- Auto-selected DeepSeek-V4 attention path on SM_121 could garble (as FlashInfer
  did for Step-3.7); verify output and record the backend before trusting it.
- PR #41834 is OPEN and may force-push; the commit pin (`72261a7`) and the
  build-time commit-mismatch check guard against silent drift.
- GB10 UMA: a clean-memory preflight (~116-117 GiB CUDA-free/node) is required;
  `drop_caches` does not recover UVM retention; reboot needs approval.

## First dual-node TP=2 model-load result + fixes (2026-06-21)

The first bring-up of the FP8 base checkpoint on dual GB10 reached weight load
(73.82 GiB/rank, ~153 s, TP=2 mp rendezvous OK) and selected the expected
backends (TritonFp8BlockScaledMM linear, DeepSeek `fp8_ds_mla` KV, MARLIN MXFP4
experts, FP8 Lightning Indexer, `DEEPSEEK_SPARSE_SWA` attention). Two issues were
found:

1. **Required `--kv-cache-dtype fp8` (fixed).** `auto` failed with `DeepseekV4
   FlashMLA fp8 layout only supports fp8 kv-cache` (`models/deepseek_v4/
   attention.py`). All three presets now set `--kv-cache-dtype fp8`.

2. **TVM-FFI duplicate registration — `apache-tvm-ffi` version drift (fixed).**
   Engine init aborted during the `DEEPSEEK_SPARSE_SWA` profiling forward pass
   (worker rank1) with a C++ abort: `tvm::ffi::Error: TypeAttr __ffi_repr__ is
   already registered for type index 130`. Forensics: plain imports do NOT
   reproduce it (it is JIT-time, in FlashInfer 0.6.12's bundled CuTeDSL
   `tvm_ffi_builder`). `pip check` shows the vLLM PR source declares
   `apache-tvm-ffi==0.1.9`, but the image had **0.1.12** — the Dockerfile
   installed it via `apache-tvm-ffi<0.2` (→ 0.1.12) and a name-only `$DEPS`
   install that drops the source's exact pins. The vLLM source coordinates the
   whole TVM-FFI family at 0.1.9 (`apache-tvm-ffi==0.1.9` AND `tilelang==0.1.9`);
   the image drifted BOTH (0.1.12 / 0.1.11), and `tilelang>=0.1.10` in turn
   requires `apache-tvm-ffi>=0.1.10`. `tilelang` IS loaded on the DeepSeek-V4
   sparse path (`vllm.model_executor.kernels.mhc.tilelang`); `__ffi_repr__` is
   registered only by apache-tvm-ffi's `libtvm_ffi.so`, so the drift let a second
   mismatched consumer re-register it at JIT time. **Classification:
   APACHE_TVM_FFI_VERSION_DRIFT** (TVM-FFI family drift). **Fix (runner stage):
   pin `apache-tvm-ffi==0.1.9` AND `tilelang==0.1.9`** to the source-coordinated
   versions, with a build-time assertion + `pip check` that the family is
   coherent (no residual conflict). The flashinfer-builder line is kept
   byte-identical (build-env only; the wheel does not embed tvm-ffi, so the
   FlashInfer build cache is preserved). Imports (incl. sparse_swa / flashmla /
   indexer / tilelang mhc) pass and `pip check` shows no TVM-family conflict; the
   definitive sparse-attention JIT proof requires the next model-load stage. New
   image: `vllm-spark:v023-stack-dsv4-sm12x-pr41834-exp-72261a7-tvmfam019`
   (`sha256:4c41950c47ecb771eead3e32147a750240d6605d15ee99c54b43e0a85ea26105`).

Separately observed (NOT changed in this narrow fix — tracked as runtime risk):
`compressed-tensors` 0.15.0.1 vs source-pinned 0.17.0; `nvidia-nccl-cu13` 2.30.4
vs the validated-profile 2.30.7; `nvidia-cutlass-dsl-libs-cu13` absent (installed
`--no-deps`); `torch` 2.12.0a0 (NGC) vs source-pinned 2.11.0. These are relevant
to later paths (W4A16 / multi-node NCCL) but are not the `__ffi_repr__` cause.

## Future build workflow (do NOT build now)

The future build MUST NOT use the dirty spark01 clone. Use a clean source on
spark01: (1) transfer an exact homeserver snapshot to a new build directory, or
(2) a separate clean checkout/worktree at the approved commit, or (3) a
deterministic source archive from the approved homeserver state. Suggested next
stage: build once on spark01, then an import-only smoke test (`python3 -c "import
vllm; from vllm.model_executor.models.registry import ModelRegistry; assert
'DeepseekV4ForCausalLM' in ModelRegistry.get_supported_archs()"`) with no model
load, before any TP=2 bring-up.

## Reasoning-parser experiment (EXPERIMENTAL, 2026-06-21)

Goal: enable DeepSeek-V4's native tokenizer and reasoning parser on top of the
already-validated dual-node TP=2 bootstrap so server responses split the model's
`<think>...</think>` trace into `reasoning_content`. Tool parsing stays disabled
(a separate future preset). Built on image
`vllm-spark:v023-stack-dsv4-sm12x-pr41834-exp-72261a7-tvmfam019`
(`sha256:4c41950c47ecb771eead3e32147a750240d6605d15ee99c54b43e0a85ea26105`),
source `72261a7af149fa5d3fe2ed2b9956e92590731012`.

Preset: `presets/deepseek-v4-v023-stack-pr41834-reasoning-parser-tp2.env`
(disposable, EXPERIMENTAL). Derived verbatim from the bootstrap preset; the only
runtime delta is two `VLLM_EXTRA_ARGS` tokens — `--tokenizer-mode deepseek_v4`
and `--reasoning-parser deepseek_v4`. All other keys are byte-identical so any
difference isolates to the parser path.

### Static registry validation (Confirmed, image 4c41950c, `--rm`, no model/GPU)

- `ReasoningParserManager.get_reasoning_parser("deepseek_v4")` resolves to
  `vllm.reasoning.deepseek_v4_reasoning_parser.DeepSeekV4ReasoningParser`;
  registered keys containing `deepseek` = `['deepseek_v4']`.
- Parser module: `vllm/reasoning/deepseek_v4_reasoning_parser.py`; classes
  `DeepSeekV4ReasoningParser` (dispatcher), `DeepSeekV4ThinkingReasoningParser`
  (DeepSeek-R1-derived, implicit end-of-reasoning when `</think>` missing),
  `IdentityReasoningParser` (pass-through).
- Tokenizer mode `deepseek_v4` referenced in `vllm/config/model.py` and
  `vllm/tokenizers/registry.py`; encoding module
  `vllm/tokenizers/deepseek_v4_encoding.py` (exposes `REASONING_EFFORT_MAX`,
  `DS_TASK_SP_TOKENS`, etc.).
- `ChatMessage.reasoning_content` and `DeltaMessage.reasoning_content` fields
  exist (`vllm/entrypoints/openai/chat_completion/protocol.py`).

### Accepted request shape (from pinned parser source, lines 238-262)

`DeepSeekV4ReasoningParser.__init__` reads `chat_template_kwargs`:

```json
{"chat_template_kwargs": {"thinking": true}}         // -> thinking parser
{"chat_template_kwargs": {"enable_thinking": true}}  // -> thinking parser
// absent / false                                     // -> IdentityReasoningParser
```

`thinking = bool(thinking) or bool(enable_thinking)`. Default (no
`chat_template_kwargs`) = non-thinking (Identity), so `reasoning_content` is
populated **only** when thinking is requested. Top-level `thinking` and
`reasoning_effort` are NOT read by the reasoning parser. Validation sampling:
`temperature=1.0`, `top_p=1.0` (DeepSeek-V4 reference). Think Max budget deferred
(left at the bootstrap `MAX_MODEL_LEN=8192`).

### Runtime confirmation (Confirmed, 2026-06-21, dual GB10 TP=2 mp)

Static parser/registry/config validation PASS; preset created and statically
validated (normalized diff = exactly the two parser tokens, no tracked file
changed, no secrets, `git diff --check` clean). Runtime TP=2 PASS via the
disposable live env `.env.dsv4-tvmfam019-rp-live` (bootstrap live env + only the
two parser tokens). Clean-memory preflight 117 GiB/node after one reboot each;
backends identical to the bootstrap run (TritonFp8BlockScaledMM linear, FP8
indexer cache, DEEPSEEK_SPARSE_SWA, MARLIN Mxfp4 MoE); init engine 218.46 s; GPU
KV cache 21,413 tokens, 2.61x concurrency at 8,192.

Engine confirmed `tokenizer_mode=deepseek_v4` and
`reasoning_parser='deepseek_v4'`. Six reasoning-parser tests (temperature=1.0,
top_p=1.0) all PASS:

| # | Request | reasoning_content | content (correctness) |
|---|---------|-------------------|-----------------------|
| 1 | bare (no kwargs) | populated | 391 (17*23) — default = thinking-on |
| 2 | `chat_template_kwargs.thinking=true` | populated | ball $0.05 (correct) |
| 3 | `chat_template_kwargs.thinking=false` | null (Identity) | red/blue/yellow (direct) |
| 4 | Korean, thinking=true | populated, clean Korean (no garble) | 11 (7,9,11) |
| 5 | streaming, thinking=true | reasoning_content deltas precede content deltas | 1,2,3 |
| 6 | multi-turn, thinking=true | populated | 84 (42*2, context kept) |

Empirical refinement of the static reading: the served chat template defaults to
**thinking-on**, so an absent `chat_template_kwargs` yields reasoning_content;
`thinking=false` is what forces the IdentityReasoningParser path (test 3
confirms). Top-level `thinking`/`reasoning_effort` are still not read.

Benign warning (both processes, startup): `Auto-initialization of reasoning
token IDs failed ... reasoning_start_str/reasoning_end_str`. Expected — the
`deepseek_v4` reasoning parser is a per-request dispatcher with no top-level
start/end strings; extraction is delegated to the Thinking/Identity sub-parser,
and all six tests parsed correctly. Controlled shutdown clean (both containers
removed). GB10 UMA retains UVM after shutdown (post-shutdown MemAvailable ~39
GiB/node); the next bring-up requires a reboot.

### Future work (recommended, NOT created)

A tool-parser preset adding `--enable-auto-tool-choice --tool-call-parser
deepseek_v4` on top of this reasoning-parser preset, validated independently
after the reasoning parser is confirmed.

## Tool-parser experiment (EXPERIMENTAL, 2026-06-21)

Status: **Experimental** — not promoted, not production. Goal: validate DeepSeek-V4
DSML tool-call parsing through the OpenAI-compatible API on top of the validated
reasoning-parser baseline. Auto tool choice is enabled **only** in this dedicated
preset. The reasoning parser remains enabled; graph, MTP, prefix cache, EP, MTP,
Think Max, long-context, and benchmark workloads remain disabled. The official
`deepseek-ai/DeepSeek-V4-Flash` FP8 checkpoint is the only checkpoint used.

Preset: `presets/deepseek-v4-v023-stack-pr41834-tool-parser-tp2.env` (disposable,
EXPERIMENTAL). Derived verbatim from the reasoning-parser preset; the only runtime
delta is two `VLLM_EXTRA_ARGS` tokens — `--enable-auto-tool-choice` and
`--tool-call-parser deepseek_v4`. Image
`vllm-spark:v023-stack-dsv4-sm12x-pr41834-exp-72261a7-tvmfam019`
(`sha256:4c41950c47ecb771eead3e32147a750240d6605d15ee99c54b43e0a85ea26105`),
source `72261a7af149fa5d3fe2ed2b9956e92590731012`.

### Static parser/registry validation (Confirmed, image 4c41950c, `--rm`, no model/GPU)

- `ToolParserManager.get_tool_parser("deepseek_v4")` resolves to
  `vllm.tool_parsers.deepseekv4_tool_parser.DeepSeekV4ToolParser`
  (`supports_required_and_named = False`).
- `DeepSeekV4ToolParser` subclasses `DeepSeekV32ToolParser`
  (`vllm/tool_parsers/deepseekv32_tool_parser.py`, 562 lines) and overrides only
  the DSML wrapper: `tool_call_start_token = "<｜DSML｜tool_calls>"`,
  `tool_call_end_token = "</｜DSML｜tool_calls>"`, `structural_tag_model =
  "deepseek_v4"`. Manager: `vllm/tool_parsers/abstract_tool_parser.py`.
- Protocol (`vllm/entrypoints/openai/chat_completion/protocol.py` +
  `.../engine/protocol.py`): `ChatCompletionRequest` exposes `tools`,
  `tool_choice`, `parallel_tool_calls`, `chat_template_kwargs`. `ChatMessage` and
  `DeltaMessage` each expose **both** `reasoning_content` and `tool_calls`.
  `DeltaToolCall(id, type, index, function)`, `DeltaFunctionCall(name,
  arguments)`. `tool_call_id` / `tool` role messages handled in
  `vllm/entrypoints/chat_utils.py` + `.../chat_completion/serving.py`.
- `coerce_to_schema_type` sanity (the `string="false"` path): `"7"`→int 7,
  `"true"`→bool True, `"3.5"`→float 3.5, `"[…]"`→list, `"{…}"`→dict.

### DSML grammar and parsing behavior (from pinned source)

Non-streaming `extract_tool_calls`: regex finds each
`<｜DSML｜tool_calls>…</｜DSML｜tool_calls>` block, then each
`<｜DSML｜invoke name="fn">…</｜DSML｜invoke>`, then each
`<｜DSML｜parameter name="p" string="true|false">VALUE</｜DSML｜parameter>`.
`string="true"` keeps the raw string; `string="false"` casts via the tool schema.
Returns `ExtractedToolCallInformation(tools_called, tool_calls, content)`; content
is the text before the first tool-call token. Multiple invokes → multiple
`ToolCall`s in order. A lone `arguments`/`input` wrapper is unwrapped **only** when
that key is absent from the tool schema (`_repair_param_dict` /
`_should_buffer_wrapper_param`), so a user field literally named `arguments` is
preserved.

Streaming `extract_tool_calls_streaming` / `_process_streaming_buffer`: buffers
DSML, emits incremental `DeltaToolCall`s (id, function name, then argument JSON
fragments) and reconstructs valid JSON; string params stream escaped, object/array
params buffer until complete, scalars buffer for coercion parity. Each invoke
advances `current_tool_index` with its own id.

tool_choice: `supports_required_and_named = False` → the pinned source treats
`tool_choice="required"` and named function choice the same as `"auto"` (no forced
constraint); `tool_choice="none"` disables tool emission (`adjust_request`).
`adjust_request` sets `skip_special_tokens=False` when tools are present and
`tool_choice != "none"` so DSML tokens survive decoding. Reasoning and tools
coexist: the reasoning parser fills `reasoning_content`; the tool parser extracts
DSML `tool_calls` from the post-think content; `chat_template_kwargs.{thinking|
enable_thinking}` still controls thinking (default thinking-on at runtime;
`thinking=false` → Identity).

### Planned test matrix and success criteria

`temperature=1.0`, `top_p=1.0`, concurrency 1, raw HTTP JSON preserved. No external
tools executed; deterministic local mock results for continuation. Tests: no-tool
regression; single string arg; boolean; integer; array; object; schema field
literally named `arguments`; multiple calls; `tool_choice="none"`; irrelevant tool
(no spurious call); thinking-disabled tool call; streaming single call; streaming
multiple calls (only if non-streaming multiple passed); tool-result continuation;
Korean tool request; incomplete/underspecified arguments. Because
required/named tool_choice is unsupported, calls are driven by auto + prompting,
and required/named is classified UNSUPPORTED rather than FAIL. Full PASS criteria:
ordinary chat valid; primitive/array/object JSON types parse; `arguments`
field-name not confused; multiple calls per supported semantics; `none` works;
irrelevant tools not called; thinking-disabled tool call works; streaming
reconstructs valid JSON; continuation works; Korean valid UTF-8; no DSML
(`<｜DSML｜…>`) or `<think>`/`</think>` leakage; no server crash.

### Runtime confirmation (Confirmed, 2026-06-21, dual GB10 TP=2 mp)

Static parser/registry/config validation PASS; preset created and statically
validated. Runtime TP=2 tool-parser validation **TOOL_PARSER_PASS** via the
disposable live env `.env.dsv4-tvmfam019-tool-live` (reasoning-parser live env +
only the two tool tokens). One reboot per node first (new boot ids, MemAvailable
117–118 GiB, swap 0, RoCE `enp1s0f0np0`/`rocep1s0f0` ACTIVE, ping 10.10.10.1→.2
OK, no stale vLLM/Ray, ports 8000/29500 free). Backends identical to the
reasoning-parser run (TritonFp8BlockScaledMM, fp8_ds_mla KV, MARLIN Mxfp4 MoE,
DEEPSEEK_SPARSE_SWA); model load 73.82 GiB / 150 s; init engine 219.18 s; GPU KV
21,413 tokens, 2.61x; no `__ffi_repr__` error. Engine confirmed
`enable_auto_tool_choice=True`, `tool_call_parser='deepseek_v4'`,
`reasoning_parser='deepseek_v4'`, `tokenizer_mode=deepseek_v4`.

Validation matrix (temperature=1.0, top_p=1.0, concurrency 1, raw JSON preserved;
all HTTP 200):

| Test | Result |
|------|--------|
| no-tool regression | PASS — finish=stop, "Paris", no tool, no DSML/think leak |
| 1 single string | PASS — get_weather `{"location":"Seoul"}`, finish=tool_calls |
| 2 boolean true/false | PASS — `{"enabled":true}` / `{"enabled":false}` (JSON bool) |
| 3 integer | PASS — `{"count":7}` (JSON int, unquoted) |
| 4 array of strings | PASS — `["apple","banana","cherry"]` order preserved |
| 5 nested object | PASS — `{"name":"Alice","age":30,"active":true}` typed |
| 6 field named `arguments` | PASS — `{"command":"ls","arguments":["-l","/tmp"]}` not confused |
| 7 multiple calls | PASS — add `{"a":2,"b":3}` + multiply `{"a":4,"b":5}`, distinct ids |
| 8 tool_choice none | PASS — normal content, 0 tool calls |
| 9 irrelevant tool | PASS — haiku, no spurious call |
| 10 thinking off + tool | PASS — reasoning_content null, get_weather call valid |
| 11 streaming single | PASS — args reconstruct to valid JSON, 22 reasoning deltas separate |
| 12 streaming multiple | PASS — indexes [0,1], distinct ids, both valid JSON |
| 13 tool-result continuation | PASS — tool role result consumed, final "Seoul clear 21°C", no dup call |
| 14 Korean tool | PASS — `{"location":"서울"}`, valid UTF-8 |
| 15 incomplete arguments | PASS (graceful) — asks for clarification, no malformed JSON, no crash |

`tool_choice="required"` / named function choice: **UNSUPPORTED** by this parser
(`supports_required_and_named = False`, treated as auto) — classified UNSUPPORTED,
not FAIL; calls were driven via auto + prompting. Stage F leak scan: 0 DSML
markers, 0 `<think>`/`</think>` in content, 0 error payloads, 0 server exceptions
across 33 saved request/response artifacts. **Final classification:
TOOL_PARSER_PASS.** Controlled shutdown clean; post-stop UVM retained ~39.5 GiB/
node (stable over 55 s; reboot required before next load). No image rebuilt or
pushed; no Step-3.7 asset, no tracked repo file, and the dirty spark01 repo
(7777aeb + 7) unchanged.
