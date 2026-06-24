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

## Performance equivalence roadmap (EXPERIMENTAL, 2026-06-21)

Goal: approach the historical unholy-fusion prefill/decode performance while keeping
the published PR41834 baseline's stability, official-model correctness, and parser
behaviour. The current and historical paths stay independently reproducible. This
is analysis + static preparation only — no image rebuild, no model load.

### Current vs historical configuration matrix

| Dimension | Current PR41834 `72261a7` (validated experimental) | Historical unholy-fusion (`aidendle94 / dev/unholy-fusion`) | Classification |
|---|---|---|---|
| Distributed backend | mp (SPMD) | mp | ALREADY_PRESENT_AUTOMATICALLY |
| Expert parallelism | off | off | ALREADY_PRESENT_AUTOMATICALLY |
| MTP | off (model ships `num_nextn_predict_layers=1`) | n=1 via `MTP_NUM_TOKENS=1` | POTENTIALLY_PORTABLE (use `--speculative-config {"method":"deepseek_mtp","num_speculative_tokens":1}`) |
| CUDA graph | off (eager); sparse-MLA declares `AttentionCGSupport.UNIFORM_BATCH`; `breakable_cudagraph` present | FULL/piecewise, bounded by `MAX_NUM_SEQS≤4` | POTENTIALLY_PORTABLE / UNVERIFIED |
| Chunked prefill | on (`enable_chunked_prefill=True`) | on | ALREADY_PRESENT_AUTOMATICALLY |
| MoE backend | MARLIN MXFP4 (auto fallback on SM121) | B12X MoE (`VLLM_USE_B12X_MOE=1`, ~2x prefill) | REPLACED_BY_NEW_BACKEND; B12X-equivalent kernel present in-source (see below) |
| B12X global env flags | NONE in pinned source | `VLLM_USE_B12X_*` (fork-only) | OBSOLETE (globals) / UNSAFE (do not restore) |
| FP8 linear | `TritonFp8BlockScaledMMKernel` | B12X FP8 GEMM (disabled in the historical stable run) | ALREADY_PRESENT_AUTOMATICALLY |
| Sparse attention | `DEEPSEEK_SPARSE_SWA` | B12X sparse indexer (disabled in stable run) | ALREADY_PRESENT_AUTOMATICALLY |
| Sparse indexer | FP8 Lightning Indexer | B12X (disabled) | ALREADY_PRESENT_AUTOMATICALLY |
| KV cache | 2 GiB fixed | ~17 GiB (1.14M tokens) | OPTIONAL (roadmap U5) |
| Max sequences | 1 | 4 | OPTIONAL (U5) |
| Max model length | 8192 | 262144 | OPTIONAL (U6) |
| Block size | 256 (`DEEPSEEK_SPARSE_SWA`) | n/a | ALREADY_PRESENT_AUTOMATICALLY |
| Max batched tokens | 2048 | n/a | OPTIONAL |
| Warm-up | engine warmup + llama-benchy internal warmup | ~5 min warm JIT cache | ALREADY_PRESENT_AUTOMATICALLY |

The two performance levers are MoE prefill (MARLIN vs a flashinfer FP4 backend) and
decode MTP n=1. Both are achievable inside the published image without restoring any
B12X global env hack.

### B12X-equivalent MoE opportunity (primary prefill investigation)

The historical `VLLM_USE_B12X_MOE` globals and `b12x_moe.py` (515 lines) /
`b12x.py` live only in the `local-inference-lab/vllm dev/unholy-fusion` fork and are
**absent** from the pinned source (no `VLLM_USE_B12X` refs, no `b12x_moe.py`). They
must NOT be restored as globals.

However, the equivalent kernels are **already present in the pinned source** as
optional MoE backends under
`vllm/model_executor/layers/fused_moe/experts/`:
`flashinfer_b12x_moe.py`, `flashinfer_cutlass_moe.py`, `flashinfer_cutedsl_moe.py`,
`flashinfer_cutedsl_batched_moe.py`, `cutlass_moe.py`. The DSV4 expert path is chosen
by `select_deepseek_v4_mxfp4_moe_backend()` in
`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, which honours an explicit
`--moe-backend` first and otherwise walks a priority list. On SM121 the higher-priority
FP4 backends (FLASHINFER_TRTLLM/CUTLASS MXFP4) currently fail `is_supported`, so it
falls back to **MARLIN** — the observed baseline. Valid `--moe-backend` names:
`deep_gemm, flashinfer_trtllm, flashinfer_trtllm_afp8, flashinfer_cutlass,
flashinfer_cutlass_afp8, triton, triton_unfused, humming, marlin, aiter*, xpu, cpu,
emulation`.

**This reframes the B12X "port" as a backend-selection + SM121-support question, not a
fork port.** Minimal optional-backend plan:
- Keep MARLIN MXFP4 as the default (auto). Introduce the alternative only via the
  existing `--moe-backend` flag in a separate disposable preset — no new global env,
  no fork cherry-pick.
- Determine, per candidate (`flashinfer_cutlass`, `flashinfer_trtllm`, `deep_gemm`,
  `humming`), the exact `is_supported` rejection reason on SM121 for the official FP4
  expert layout (dtype/activation/scale-format/workspace/SM-arch gate). Fix only the
  gate if the kernel is genuinely SM121-capable; otherwise classify the candidate
  UNSAFE/OBSOLETE for SM121.
- Operator-level validation BEFORE model correctness: numerical output vs MARLIN
  (max abs/rel error thresholds), representative expert batch sizes, cold/warm
  latency, workspace memory, deterministic routing comparison, SM121 compilation
  proof. Fail cleanly (fall back to MARLIN) when unsupported. Affect only the official
  DeepSeek-V4 FP4 expert path; preserve exact rollback to the published image.
- Only if no in-source backend is SM121-viable does a new immutable image with an
  adapted optional backend become stage U4.

### MTP n=1 (decode investigation)

Method `deepseek_mtp`, depth n=1 (the model ships `num_nextn_predict_layers=1`).
Preset `presets/deepseek-v4-v023-stack-pr41834-mtp1-tp2.env` adds only
`--speculative-config {"method":"deepseek_mtp","num_speculative_tokens":1}` over the
bootstrap. Acceptance is measurable from Prometheus counters
`vllm:spec_decode_num_accepted_tokens_total`,
`vllm:spec_decode_num_draft_tokens_total`, `vllm:spec_decode_num_drafts`
(acceptance rate = accepted / draft). The MTP run must record tg128 d0/d4096,
acceptance rate, output correctness, memory/startup delta, and ≥20 sequential short
requests for stability. Caveat (verify at U1 launch): the JSON `--speculative-config`
argument is embedded in `VLLM_EXTRA_ARGS`; confirm the entrypoint preserves the JSON
as a single token (same open question as the graph preset's `--compilation-config`).

### CUDA graph safety classification

The pinned source provides DSV4 graph infrastructure: `sparse_mla.py` declares
`AttentionCGSupport.UNIFORM_BATCH` and `attention.py` imports
`breakable_cudagraph` / `eager_break_during_capture` (attention runs eager during
capture). The existing `presets/...-graph-tp2.env` selects
`--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE"}` while the SM121 guard
keeps `VLLM_USE_BREAKABLE_CUDAGRAPH=0`. FULL_AND_PIECEWISE over sparse SWA is the mode
implicated in the historical GB10 hang and is unproven here.

**Classification of the existing graph preset: REQUIRES_PIECEWISE_FIRST** — only
UNIFORM_BATCH/piecewise capture is declared safe by the attention backend, so a first
soak must use PIECEWISE before FULL_AND_PIECEWISE. Any future graph validation must
run ≥20 sequential requests before benchmarking. (`UNVERIFIED` until that soak.)

### Staged equivalence roadmap (do not combine stages)

- **U0 — exact comparable baseline**: current image, eager, MTP off, generation
  latency, tg128, c1, depths 0 and 4096. Establishes an apples-to-apples decode
  number against the historical tg128 (current published baseline used tg32, which is
  NOT comparable to historical tg128).
- **U1 — MTP n=1 only**: current image, eager, MTP n=1, c1, d0/d4096, acceptance
  metrics, 20-request stability soak.
- **U2 — graph only**: current image, MTP off, PIECEWISE first (per audit), 20–50
  request soak, same canonical benchmark.
- **U3 — MTP n=1 + validated graph**: only if U1 and U2 independently pass.
- **U4 — B12X-equivalent MoE experimental image**: new immutable image, optional
  backend, operator correctness → model correctness → prefill benchmark.
- **U5 — KV / concurrency scaling**: only after single-stream equivalence — KV 4/8/16–17
  GiB, max sequences 2 then 4.
- **U6 — long-context**: only after stable c1/c4 — max model length 32768 / 65536 /
  131072 / 262144.

### U0 exact apples-to-apples commands

Server unchanged (parser-free bootstrap: MTP off, graph off, prefix off, max seqs 1,
fixed 2 GiB KV, max model length 8192). llama-benchy 0.3.7, generation latency, runs 3,
concurrency 1, explicit official tokenizer. New result namespace (do NOT reuse the
tg32 directory):
`benchmarks/results/deepseek-v4-flash-pr41834-tvmfam019-bootstrap-tp2-u0-tg128-<ts>/`.

```
# pp2048 + tg128 at depth 0 and 4096 (generation mode)
llama-benchy --base-url http://localhost:8000/v1 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer /home/bjk110/Documents/Models/deepseek-ai/DeepSeek-V4-Flash \
  --pp 2048 --tg 128 --depth 0 4096 --runs 3 --latency-mode generation \
  --format md --save-result /tmp/dsv4-u0-tg128.md
```

Comparison matrix (ratio/delta only after latency mode and shape are confirmed
identical; compare totals, never peak-best-of-run):

| case | current (to measure) | historical unholy (total) | target |
|---|---|---|---|
| pp2048 d0 c1 | TBD | ~1919 t/s | ≥1800 t/s |
| pp2048 d4096 c1 | TBD | ~2051 t/s | ≥1800 t/s |
| tg128 d0 c1 | TBD | ~34.59 t/s (peak 39.67) | ≥35 t/s |
| tg128 d4096 c1 | TBD | ~37.4 t/s region | ≥35 t/s |

Targets are experimental goals, not current capability. CV ≤ 5%, no corruption, no
rank failure, no hang, no parser-behaviour regression.

## Decode performance-attribution audit (forensic, 2026-06-21)

Analysis only — no model load, no benchmark, no reboot. Goal: attribute the
remaining decode gap between the validated PR41834 path and the historical
unholy-fusion path, and find the minimum safe route toward historical decode.

### Confirmed results (generation latency, tg128, c1, total t/s)

| run | config | tg128 d0 | tg128 d4096 | source |
|---|---|---|---|---|
| U0 | eager, MTP off | 7.07 | 7.19 | CONFIRMED_FROM_RAW_ARTIFACT (U0 dir) |
| U1 | eager, MTP n=1 (acc ~0.88) | 12.25 | 11.36 | CONFIRMED_FROM_RAW_ARTIFACT (U1 dir) |
| U2 | PIECEWISE graph, MTP off | 6.84 | 6.89 | CONFIRMED_FROM_RAW_ARTIFACT (U2 dir) |
| historical | unholy-fusion (below) | **34.59 total** / 39.67 peak | ~34.59 (depth-immune) | CONFIRMED_FROM_RAW_ARTIFACT (docs/unholy-fusion-benchmark.md) |

**Historical exactness (critical)**: `docs/unholy-fusion-benchmark.md` records c=1
**total 34.59 t/s** and c=1 **peak 39.67 t/s**. The historical headline is TOTAL
34.59 (the right A/B metric vs our total t/s); the "35–40" range conflates total and
peak. Historical config (CONFIRMED_FROM_RAW_ARTIFACT): image
`aidendle94/sparkrun-vllm-ds4-gb10:production-ready` (conda env, opaque — exact vLLM
commit UNVERIFIED), `local-inference-lab/vllm dev/unholy-fusion` fork, official
DeepSeek-V4-Flash, TP=2 mp, MTP n=1, `VLLM_USE_B12X_MOE=1`, MAX_NUM_SEQS=4,
MAX_MODEL_LEN=262144, GPU_UTIL=0.80, KV 17.1 GiB, CUDA graph ON (capture sizes scale
with MAX_NUM_SEQS), pp=2048 tg=128 runs=3 latency=generation, llama-benchy 0.3.7.

### Residual gap and decode cost model (assumptions marked)

Per-token-output time: U0 141.4 ms, U1 81.6 ms, historical 28.9 ms. Simple
batch-1 step model (ASSUMPTION: MTP yields 1 + acceptance tokens/step; ~0.88 accept):
- U0 model-step ≈ 141 ms → 1 token/step.
- U1 model-step ≈ 1.88 × 81.6 = **153 ms/step** → +12 ms draft+verify over U0; nets
  1.73× (matches observed 12.25/7.07). MTP efficiency is as theoretically expected;
  no hidden MTP suppression.
- Historical model-step ≈ 1.88 × 28.9 = **54 ms/step** — **~2.8× faster per model step
  than ours**. The residual gap (U1 12.25 → historical 34.59 = 2.82×) is in the
  per-step FORWARD cost, NOT in MTP efficiency.

### Why PIECEWISE gave no gain — and what it does/does not prove

PIECEWISE splitting_ops include `vllm::deepseek_v4_attention`,
`vllm::sparse_attn_indexer`, `vllm::unified_mla_attention_with_output`: the graph
captures **dense/MoE** layers and leaves **sparse attention + FP8 Lightning indexer
EAGER**. U2 no-gain therefore proves only that **dense/MoE launch overhead is NOT the
bottleneck**. It does **NOT** prove the eager attention is compute-bound: the sparse
attention path is many small kernels (sparse_mla.py has 7 kernel/jit refs + indexer
top-k/gather/FP8-cache ops), which could be **launch-bound**. A graph mode that
captures attention (FULL / FULL_AND_PIECEWISE) could remove that launch overhead —
but only profiling can distinguish launch-bound vs compute-bound. **Do not conclude
graph cannot help; conclude PIECEWISE specifically cannot, because it does not capture
the attention.**

### Highest-impact gap candidates (impact UNKNOWN without profiling)

| factor | likely decode impact | evidence |
|---|---|---|
| Attention/indexer captured by FULL graph (launch overhead of many tiny eager kernels) | HIGH (UNKNOWN) | PIECEWISE leaves them eager; sparse path is many small kernels; U2 ruled out dense launch overhead only |
| B12X MoE decode kernels vs MARLIN MXFP4 | MEDIUM (UNKNOWN) | historical used VLLM_USE_B12X_MOE=1; our path MARLIN; not isolated by any run yet |
| Historical async scheduling / engine-loop differences | MEDIUM (UNKNOWN) | `vllm/v1/core/sched/async_scheduler.py` exists, unused by us; fewer CPU sync points |
| Stack drift (Triton/CUDA/PyTorch/FlashInfer/NCCL) | LOW (UNKNOWN) | conda vs NGC; opaque historical image |
| MAX_NUM_SEQS 4 vs 1, KV 17 vs 2 GiB, MML 262144 vs 8192 | LOW at c=1 d0 | single-stream decode is depth-immune (historical note); batch-1 only |
| MTP efficiency | NONE | cost model shows MTP behaves as expected (1.73×) |

### Historical hang root-cause status

Historical SM121 corruption/delayed-hang was associated with aggressive graph capture
(FULL/breakable) + buffer aliasing. Current source adds **persistent buffers kept
outside the cudagraph pool** (`nvidia/model.py` copy_ in forward) =
PARTIALLY_FIXED for the buffer-aliasing class. The SM121 garble tied to
`VLLM_USE_BREAKABLE_CUDAGRAPH`/FULL capture is STILL_PRESENT-risk (we keep BREAKABLE=0
and PIECEWISE to avoid it). One buffer fix does NOT make FULL_AND_PIECEWISE proven
safe.

### Candidate paths toward attention capture

- **A — FULL_AND_PIECEWISE unchanged**: EXPECTED_GAIN HIGH-if-launch-bound/UNKNOWN;
  CORRECTNESS_RISK HIGH (SM121 garble); HANG_RISK MEDIUM (one buffer fix only);
  PORTABILITY low. RECOMMENDATION: not first; gated safety probe only.
- **B — restricted FULL decode-only, batch-1, fixed shape**: EXPECTED_GAIN
  HIGH-if-launch-bound; RISK lower than A (smaller capture surface); COMPLEXITY
  MEDIUM (needs source support for decode-only/shape-whitelist capture — to verify).
  RECOMMENDATION: promising IF profiling shows launch-bound attention.
- **C — PIECEWISE + manually expanded graph region (pull attention in)**: COMPLEXITY
  HIGH (source change), RISK HIGH (re-creates unsafe capture). RECOMMENDATION: avoid.
- **D — MTP-only graph (proposer/verifier/rejection sampler), attention eager**:
  EXPECTED_GAIN LOW (MTP overhead is only ~12 ms/step per the cost model; capturing it
  saves a fraction of that); COMPLEXITY MEDIUM. RECOMMENDATION: low value.
- **E — scheduler/engine-loop (async scheduling) without broader graph capture**:
  EXPECTED_GAIN MEDIUM (UNKNOWN); RISK LOW; PORTABILITY high (in-source
  async_scheduler.py). RECOMMENDATION: attractive low-risk lever; verify spec-decode
  compatibility first.

### Profiling plan (no profiling run yet)

Tools confirmed in-image (no install): torch.profiler, torch.cuda.nvtx, Nsight
Systems (`/usr/local/cuda/bin/nsys`), Nsight Compute (`/usr/local/bin/ncu`), vLLM
`start_profile`/`stop_profile` worker hooks (`VLLM_TORCH_PROFILER_DIR`), Prometheus
spec_decode counters. Plan: low-overhead torch.profiler/NVTX (not ncu, too intrusive)
on **U0 eager** and **U1 MTP n=1**, shapes tg128 d0 c1 and tg128 d4096 c1, breaking
decode-step time into attention / indexer / MoE / dense / sampler / MTP
proposer / verifier / rejection-sampler / TP-NCCL / host-sync. The single decisive
question: **is the eager sparse-attention+indexer launch-bound or compute-bound?** If
launch-bound → a gated decode-only FULL capture (Candidate B) is justified; if
compute-bound → the gap is kernel-level (B12X) and graph cannot close it.

### U3 (MTP n=1 + PIECEWISE) classification

**LOW_VALUE_CONFIRMATION**. The cost model + U2 no-gain predict U3 ≈ U1 (~12 t/s):
PIECEWISE adds no decode benefit on top of MTP because attention stays eager. A short
U3 control run would only confirm MTP+graph stability/interaction, not progress toward
the target. Do NOT predict U3 by multiplying U1×U2 ratios. Run only if combined-mode
stability evidence is independently wanted.

### Recommended next runtime gate

**profiler-only U0/U1 comparison** (torch.profiler + NVTX, tg128 d0/d4096 c1). It is
the lowest-risk action that resolves the launch-bound-vs-compute-bound question, which
gates every subsequent decision (Candidate B vs B12X vs async-scheduler). Defer any
FULL_AND_PIECEWISE probe until profiling justifies it; if later run, apply strict
gates: clean-memory reboot, c1, MML 8192, max-seqs 1, fixed 2 GiB KV, MTP off,
output 16–32 tokens, correctness guard, per-request health, 10-request gate then
50-request soak, immediate stop on first corruption/timeout/rank divergence/memory
loss, full logs, no auto-retry, no second auto-reboot.

### Safe performance ceiling (confirmed)

Current confirmed SAFE decode ceiling = **MTP n=1, ~12 t/s** (U1). PIECEWISE adds
nothing. The historical 34.59 (2.82× higher) is **not yet attributed** and depended on
an attention-capturing graph mode (FULL/breakable) that is unsafe on SM121 here, plus
B12X MoE and possibly async scheduling — none isolated. The gap is **OPEN**, not
solved; FULL_AND_PIECEWISE is **not proven** to be the sole missing factor.

## Async-scheduler / engine-loop forensic audit (2026-06-22)

Static source + raw-artifact audit only — no model load, no benchmark, no
profiler, no reboot. Goal: decide whether asynchronous scheduling is a usable
new decode lever (candidate "E" from the earlier attribution audit). **Decisive
result: it is NOT a new lever — async scheduling is ALREADY ACTIVE BY DEFAULT in
every run we have, including the U0 baseline and the U1 MTP run.**

### Source mechanism (CONFIRMED_FROM_SOURCE, image 4c41950c / commit 72261a7)

- `vllm/v1/core/sched/async_scheduler.py` defines `AsyncScheduler(Scheduler)`
  (spec-token placeholders + `_update_after_schedule` output handling). It is a
  shipped, first-class scheduler, not a fork patch.
- `vllm/config/scheduler.py`: `async_scheduling: bool | None = None` (default
  **None → auto**). `get_scheduler_cls()` returns `AsyncScheduler` when
  `async_scheduling` is truthy, else `Scheduler`.
- `vllm/config/vllm.py:955-1035` auto-select: with `async_scheduling is None`,
  async is enabled UNLESS (a) pooling model, (b) a speculative method **not** in
  `EagleModelTypes`/`NgramGPUTypes`, (c) `disable_padded_drafter_batch`, or (d)
  the executor does not support async. `EagleModelTypes` (config/speculative.py)
  includes `deepseek_mtp`; `MultiprocExecutor.supports_async_scheduling()` (the
  `mp` backend used here) returns `True`. So for both our U0 (no spec) and U1
  (`deepseek_mtp` n=1) configs the auto path resolves to **enabled**, logging
  `config/vllm.py:1031` "Asynchronous scheduling is enabled."

### Current path (CONFIRMED_FROM_RAW_ARTIFACT)

`[vllm.py:1031] Asynchronous scheduling is enabled.` appears in the raw head logs
of **all three** runs (APIServer + EngineCore processes):

| run | config | async line | source artifact |
|---|---|---|---|
| U0-RDMA | eager, MTP off | YES (07:11:00 / 07:17:43) | `…u0-rdma-fullmodel-20260622-073014/head_full.log:48,360` |
| U1 | eager, MTP n=1 | YES (13:31:07) | `…mtp1-tp2-u1-tg128-20260621-131844/head.log:51` |
| U0 | eager, MTP off | YES | `…u0-…/head.log` (same auto-select; line present) |

**The bootstrap and mtp1 preset header comments that list "async scheduling"
under "Deferred (NOT enabled here)" are INACCURATE** — async was active the whole
time via auto-select. (Corrected here; the preset comments are left as historical
record and superseded by this section.) Classification: **ASYNC_ALREADY_ACTIVE**.

### Historical unholy-fusion (Phase-4 reconstruction)

Did the historical 34.59 t/s c=1 result use async scheduling, cherry scheduling,
asynchronous output processing, a patched engine loop, or a custom scheduler
class? **No raw artifact supports any such claim.**

- The historical image `aidendle94/sparkrun-vllm-ds4-gb10:production-ready` is an
  opaque conda image; its exact vLLM commit is **UNVERIFIED** (cannot be proven
  from binary layers). (UNVERIFIED)
- The `local-inference-lab/vllm dev/unholy-fusion` fork differences that ARE
  documented (`docs/unholy-fusion-benchmark.md`, fork analysis) are all
  **kernel-level**: `b12x_moe.py`, `b12x.py` FP8 GEMM, B12X sparse indexer, and a
  heavily customized **all-reduce** (`+462/-12`). None of these is a scheduler,
  engine-loop, or async-output-processing change. (CONFIRMED_FROM_SOURCE for the
  fork; the claim "no scheduler change" is INFERRED from the absence of any
  scheduler diff in the documented fork delta.)
- Therefore attributing the historical advantage to async scheduling is
  **UNVERIFIED**, and is positively **contradicted** by our own data: async is
  already on in our U0/U1 and still yields only 7/12 t/s. Async cannot be the
  2.82× differentiator.

### Compatibility (Phase-5, for completeness)

Async is **SUPPORTED** with our exact config — non-pooling DeepSeek-V4, mp
executor (`supports_async_scheduling()==True`), TP=2, MTP n=1 (`deepseek_mtp` in
`EagleModelTypes`), sparse attention/indexer (orthogonal to the scheduler),
NET/IB (orthogonal), c=1, streaming, metrics, shutdown — all already exercised in
U0/U1 with async on, no async-attributable failure. The only nuance: an EXPLICIT
`--async-scheduling` takes the hard-fail-on-incompatibility branch rather than the
warn-and-disable auto branch (SUPPORTED_WITH_LIMITATIONS for the explicit form).

### Performance estimate (Phase-6)

Optimistic / realistic / lower-bound all collapse to the **same number: 0%**.
Async is already the active scheduler in U0 (7 t/s) and U1 (12 t/s), so explicitly
re-enabling it cannot add throughput. There is no removable CPU-sync cost left for
async to attack that is not already attacked — and the U0 nsys + U0-RDMA NO-GAIN
result locates the decode bound in the per-step forward/communication path, not in
scheduler overlap. (The earlier "candidate E, MEDIUM gain" estimate is now
**retired** — it assumed async was off, which the logs disprove.)

### Decision (Phase-10): **RUN_RDMA_DUAL_RANK_NSYS_FIRST**

`RUN_U4_ASYNC_ONLY` is rejected (zero delta — async already on).
`ASYNC_SOURCE_PATCH_REQUIRED` / `ASYNC_UNSUPPORTED_*` are rejected (async is
present, supported, and active). The async preset
`presets/deepseek-v4-v023-stack-pr41834-async-u4-tp2.env`
(SHA-256 `86cd0f2e2ef954a577ce6a1a38eaf2699c6ed0f5032268bd1590e438f42b7b9c`,
normalized delta vs bootstrap = exactly one added token `--async-scheduling`) is
preserved **only as an explicit no-op control**, not as a throughput experiment.

The genuinely open decode question is upstream of the scheduler: the U0 Socket
nsys attributed ~68% of GPU kernel time to TP NCCL all-reduce, yet U0-RDMA (12×
faster isolated 8 KiB all-reduce) produced **no c=1 decode gain**. That
contradiction — not async — is what gates the next lever choice (FULL-graph
attention capture vs B12X-equivalent MoE vs a comm/compute-overlap reality). The
lowest-risk action that resolves it is a **dual-rank RDMA Nsight P0** measuring
RDMA NCCL ms/step against the Socket ~110 ms / ~147 ms. Per-shape fresh `nsys
launch` (a single session cannot do two start/stop cycles), one clean-memory
reboot per node, c=1, MTP off, eager. The proven safe decode lever remains MTP
n=1 (U1, ~12 t/s); async adds nothing to it.

## Dual-rank RDMA Nsight P0 — result (2026-06-23)

Executed the dual-rank RDMA Nsight P0 above. One reboot per node (clean memory,
~117-118 GiB), both ranks launched under in-image Nsight 2026.2.1, RDMA NET/IB
proven on both ranks (`rocep1s0f0:1/RoCE` v2, GID 3, no Socket/external plugin),
one deferred start/stop per rank, one fixed P0 request (depth 0, 64 forced decode
tokens, greedy). Both reports VALID (170,943 kernels each, 5,632 NCCL kernels each).
Artifacts: `benchmarks/results/deepseek-v4-flash-pr41834-tvmfam019-u0-rdma-dual-nsys-p0-20260623-041932/`
(reports + sqlite + `ANALYSIS.md`).

### Contradiction RESOLVED
| per decode step | rank0 RDMA (head) | rank1 RDMA (worker) | Socket rank0 (prior) |
|---|---|---|---|
| wall | 173.9 ms | 173.8 ms | 148.5 ms |
| all-reduce share (exclusive) | **77.0%** | 17.7% | 73.7% |
| avg all-reduce kernel | 1,524 us | 350 us | 1,245 us |
| GPU idle | 1.5% | **61.8%** | 1.6% |
| NCCL/compute overlap | **0.0%** | 0.0% | 0.0% |

1. **NCCL all-reduce is the genuine exclusive critical path.** On this stack NCCL
   runs on the SAME single stream as compute (0.0% overlap), so summed == union ==
   exclusive. The prior "~110 ms/step, 67.7%" Socket figure was summed kernel time
   but, because of the zero overlap, it equals the exclusive critical-path time —
   the earlier skepticism ("maybe just overlapping-summed") is **REFUTED**; the
   67.7% attribution was essentially correct.
2. **RDMA gave no decode gain because the all-reduce is latency/sync-bound, not
   bandwidth-bound.** Isolated 8 KiB all-reduce = 32.6 us over RDMA, but the
   in-model per-call all-reduce is 350 us (worker) to 1,524 us (head); the 8 KiB
   transfer is <10% of even the worker's 350 us. rank0's per-call cost did NOT
   shrink (1,245 us Socket -> 1,524 us RDMA). A 12x faster wire is invisible against
   a ~1.5 ms per-collective sync/launch floor.
3. **COLLECTIVE_WAIT_ASYMMETRY.** Both ranks lockstep at ~174 ms/step but rank0
   spins 98.5% busy inside the all-reduce (host blocked 11.9 s in cuda*Synchronize)
   while rank1 idles 61.8%. rank0's extra ~1,170 us/call is cross-rank arrival wait,
   not transfer. (Correlated by collective count 5,632 = 88x64 and lockstep cadence,
   NOT absolute host clocks; sub-ms arrival ordering not claimed — timesyncd only.)
4. **Structural enemy = 88 synchronizing TP barriers per token** (43 layers x 2 +
   2) x ~1.5 ms head floor ≈ 132 ms/token -> ~7.5 t/s, matching observed ~7.4.

### Classification: RDMA_DUAL_NSYS_COMM_CRITICAL
Communication-critical, but the communication is synchronization/latency-bound with
a co-primary COLLECTIVE_WAIT_ASYMMETRY (rank0 spin vs rank1 idle); _MIXED_CRITICAL_PATH
also fits. COMPUTE_CRITICAL / HOST_SYNC_CRITICAL rejected (compute exclusive only
~21-25%; rank0 host sync overlaps the GPU all-reduce, not exclusive wall).

### Next lever (ranked)
Wall time scales with barriers/token, so the levers that move it: **(1) MTP n=1**
(validated +73%, multiple tokens per forward = fewer barriers/accepted token);
**(2) reduce all-reduce count per layer** (fuse the two per-layer collectives,
88->44 ~ halves the dominant term — needs B12X-style fused-reduction source work);
**(3) overlap comm on a dedicated stream** (today serialized on stream 16, 0%
overlap — needs engine change). RDMA stays the correct transport (fixes over-broad
NCCL_NET=Socket, may help prefill/concurrency) but is confirmed **not** a c=1 decode
lever.

## TP collective source audit + historical all-reduce forensics (2026-06-23)

Static source/trace audit of the 88 collectives, fusion/overlap legality, and the
historical unholy-fusion all-reduce. No model load, no reboot, no benchmark.

### Exact 88-collective ledger (source + trace verified)
Trace-exact per output token: **87 `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` +
1 `ncclDevKernel_AllGather_RING_LL` = 88** (NOT "43×2+2 all-reduce" — the extra two
are 1 AllReduce + 1 AllGather). DSV4-Flash has `num_hidden_layers=43`,
`first_k_dense_replace=None` (every layer is MoE). The model source contains NO
explicit `tensor_model_parallel_all_reduce`; all collectives come from vLLM layers:

| # | site | source | type | payload (c=1) |
|---|---|---|---|---|
| 43 | attn output proj `wo_b` | `deepseek_v4/attention.py:223` `RowParallelLinear` → `linear.py:1647` | AllReduce | hidden 4096 × bf16 = 8 KiB |
| 43 | MoE experts reduce | `deepseek_v4/nvidia/model.py:561` `reduce_results` → FusedMoE | AllReduce | 8 KiB |
| 1 | input embedding | `vocab_parallel_embedding.py:491` | AllReduce | 8 KiB |
| 1 | lm_head logits | `logits_processor.py:83` `_gather_logits` | AllGather | local vocab shard |

mhc residual-mix / norm (`mhc_*_tilelang`), sparse-MLA, Lightning indexer
(`ReplicatedLinear`, no TP), shared-expert add, and RoPE are all LOCAL (no collective).

### Fusion legality (Phase 3) — NOT_FUSIBLE_DATA_DEPENDENCY
The two per-layer collectives are **sequentially data-dependent**: attn AllReduce →
`mhc_fused_post_pre` (norm+residual) → MoE input → MoE AllReduce → next layer. The
reduced attn tensor is consumed immediately; MoE cannot start until it is full.
**88→44 by fusing the intra-layer pair is NOT legal** (would change semantics). The
only count reduction is fewer forward passes per accepted token (MTP), or eliminating
TP entirely (single-node). The deepseek_v2 comment "replace the end-of-attn all_reduce
with reduce_scatter" is sequence-parallel — it shrinks payloads, not the 88 barrier
COUNT, and is moot at c=1 (1 token).

### Overlap feasibility (Phase 4) — NO_INDEPENDENT_COMPUTE at c=1
All 88 collectives run on the single default compute stream (trace stream 16), 0%
overlap, result immediately consumed. The chain attn→norm→MoE→next-layer is fully
data-dependent; at c=1 there is no independent compute to hide a collective behind.
The model already allocates `aux_stream_list` (3 streams, model.py:977) for intra-MoE
compute, but NOT for hiding the TP all-reduce. Real overlap would require cross-token
pipelining (absent at c=1) → OVERLAP_REQUIRES_PIPELINING / NO_INDEPENDENT_COMPUTE.

### Historical unholy-fusion all-reduce (Phase 5-6, 10) — CONFIRMED_FROM_SOURCE
Fork `local-inference-lab/vllm@dev/unholy-fusion` (public). **Correction to prior
memory ("All-reduce 대폭 커스텀 +462/-12"): there is NO B12X custom all-reduce.** The
B12X feature set is COMPUTE only — `fused_moe/b12x_moe.py`, `experts/flashinfer_b12x_moe.py`,
`kernels/linear/scaled_mm/b12x.py`, `attention/backends/mla/b12x_mla_sparse.py`. There
is no `b12x_all_reduce`. The fork's `csrc/custom_all_reduce.cu` and
`device_communicators/custom_all_reduce.py` are the **standard upstream** one-shot
custom all-reduce, and they STILL carry the multi-node guard
(`custom_all_reduce.py:253` `if not all(in_the_same_node_as(...)): self.disabled=True`,
"No need to initialize custom allreduce for multi-node case"). Custom / symm-mem /
FlashInfer one-shot all-reduce are **all SAME-NODE-ONLY** (need NVLink/P2P/IPC). Our
dual-physical-node RoCE TP=2 is multi-node → these are hard-disabled (our engine config
already prints `disable_custom_all_reduce=True`; the trace shows PYNCCL). **Portability
classification: SINGLE_NODE_ONLY — there is no portable multi-node fast all-reduce to
port.** The unholy-fusion decode advantage comes from B12X COMPUTE kernels and/or
single-node deployment, not from a faster cross-node collective.

### Collective cost model (Phase 7)
Unprofiled c=1 token ~135 ms (7.41 t/s): ~77% collective (~104 ms) + ~21.5% compute
(~29 ms). Upper bounds (do NOT multiply; they share the collective budget):
- **A. per-call latency floor**: collective→0 ⇒ compute-only ~29 ms ⇒ ~34 t/s ceiling.
  Remove only the head WAIT (head 1.524→0.350 ms like worker) ⇒ ~55 ms ⇒ ~18 t/s.
  (Do NOT use the isolated 32.6 us as the in-model floor — it is not representative.)
- **B. count reduction**: 88→44 ⇒ ~83 ms ⇒ ~12 t/s — but fusion is NOT legal (see Phase 3).
- **C. overlap**: realistic <25% (data-dependent) ⇒ ~9-12 t/s.
- **D. MTP n=1**: measured ~1.88 tok/step ⇒ ~12.25 t/s (validated, the real lever).
The **12.25 → 34.59 t/s gap implies ~29 ms/token = compute-only = ZERO TP collective**.
**CORRECTION (2026-06-23): the approximately 34 t/s compute-only value is a theoretical
zero-communication upper bound. It is not a feasible single-node TP=1 configuration for
the official checkpoint because the full model weights exceed one DGX Spark's
unified-memory capacity.** The official DeepSeek-V4-Flash checkpoint is **~148.7 GiB**
(on-disk safetensors 159,617,149,040 B; index total_size 159,609,485,896 B; HF-reported
159,634,530,349 B; 46 shards, FP8/bf16). The earlier "73.82 GiB = whole model" reading
was wrong: **73.82 GiB is the per-rank TP=2 shard** (148.7 / 2 ≈ 74.3 GiB/rank). One
DGX Spark has only ~121.63 GiB total unified memory, so the full checkpoint cannot be
held with TP=1 on a single node, and runtime buffers + KV cache widen the deficit; CPU
offload does not raise the node's unified-memory capacity. **Single-node TP=1 is NOT an
available optimization path for the official checkpoint.** The 34.59 gap is therefore
explained by B12X COMPUTE kernels and/or an UNVERIFIED historical topology (see Historical
verification), NOT by a portable all-reduce patch and NOT by single-node TP=1.

### NCCL algo/protocol (Phase 8)
Trace shows `RING_LL` for both AllReduce and AllGather — NCCL already auto-selected the
**LL (low-latency) protocol**, which is the correct choice for an 8 KiB message; Ring≈Tree
for world_size=2. The unexplained quantity is the 10-47× gap between the isolated 8 KiB
all-reduce (32.6 us) and the in-model per-call (0.35-1.5 ms). Bounded A/B matrix for
LATER (NCCL 2.30.4, world_size=2 only): {Ring,Tree} × {LL,LL128,Simple}, plus
NCCL_MIN/MAX_NCHANNELS. Not run here.

### Exact-cadence reproducer (Phase 9)
`scripts/diag/dsv4_exact_cadence_allreduce_repro.py` (untracked, NOT run): replays 87
AllReduce + 1 AllGather/token × 64 tokens at 8 KiB bf16 on one rank per node, modes
{back-to-back, model-like gaps, rank-skew injection, dedicated comm-stream}, finite
timeout + correctness check, NO model load. Isolates whether the in-model per-call cost
is launch-serialization / sync-cadence / rank-skew vs NCCL protocol vs profiler overhead.

### Selected next action (Phase 11): **RUN_EXACT_CADENCE_NCCL_AB**
Source mapping is now CLEAR, but the isolated-vs-in-model per-call latency gap is the
quantity that gates whether per-call latency is reducible. The exact-cadence reproducer
(subsuming TEST_NCCL_LOW_LATENCY_PROTOCOL via its env-driven ALGO/PROTO sweep) is the
lowest-risk, no-model-load step. REJECTED: PORT_HISTORICAL_ALLREDUCE_PATCH (no portable
multi-node all-reduce exists — SINGLE_NODE_ONLY); IMPLEMENT_COLLECTIVE_FUSION_PROTOTYPE
(NOT_FUSIBLE_DATA_DEPENDENCY); IMPLEMENT_COMM_COMPUTE_OVERLAP_PROTOTYPE (NO_INDEPENDENT_
COMPUTE at c=1). Standing decode lever remains MTP n=1. **Single-node TP=1 is NOT a
candidate** (the ~148.7 GiB checkpoint exceeds one Spark's ~121.63 GiB unified memory);
the ~34 t/s compute-only figure is only a theoretical zero-communication ceiling, not a
deployable configuration.

## Exact-cadence NCCL A/B (2026-06-23)

Standalone reproduction of the 88-collective decode cadence, NO model load, one rank per
node, validated RDMA (NET/IB proven both ranks every run). Reproducer
`scripts/diag/dsv4_exact_cadence_allreduce_repro.py` (v2 SHA-256 8e4c663947…; v1
f961b451… preserved in the results dir). Artifacts:
`benchmarks/results/dsv4-exact-cadence-nccl-ab-20260623-060315/` (20 rank logs +
ANALYSIS.md + both reproducer versions + harness).

### Model-size correction (prerequisite)
The official checkpoint is **~148.7 GiB** (on-disk 159,617,149,040 B; index 159,609,485,896 B;
HF-reported 159,634,530,349 B). The 73.82 GiB observed at load is the **per-rank TP=2
shard** (148.7/2). **Single-node TP=1 is infeasible** for the official checkpoint
(148.7 GiB > one Spark's 121.63 GiB unified memory); the ~34 t/s compute-only value is a
theoretical zero-communication upper bound, NOT a deployable path.

### Historical (unholy-fusion 34.59 t/s) equivalence: OPEN
`docs/unholy-fusion-benchmark.md` headline 34.59 t/s = **tg128 c=1 TOTAL @ depth 131072**
(c=1 d0 total 39.76, peak 43.00), image `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`,
MAX_NUM_SEQS=4, MTP n=1, `.env.unholy-fusion` CLUSTER_MODE=dual-rdma TP=2 mp. Field
classification: served model name "deepseek-v4-flash" = INFERRED; local path = UNVERIFIED
(placeholder); config arch / quantization_config / index / shard count / checkpoint bytes
inside the opaque aidendle94 image = **UNVERIFIED**; node count (2) / TP size (2) =
INFERRED (from our reconstructed `.env`, not a captured run log); per-rank load / CPU
offload = UNVERIFIED; 34.59 = TOTAL (CONFIRMED_FROM_RAW_ARTIFACT, table labels "(total)");
MTP token accounting under llama-benchy synthetic generation = UNVERIFIED. The historical
checkpoint quantization (FP8 vs NVFP4) and exact run topology cannot be proven from
preserved artifacts ⇒ **historical equivalence stays OPEN**.

### Results (rank0)
| mode | token ms | all-reduce/call | note |
|---|---|---|---|
| E1 eager back-to-back | 5.84 | **64.9 us** | correct |
| E2 model-like 220us gaps (synthetic) | 6.62 | 69.3 us | correct |
| E4 dedicated comm stream | 5.52 | 61.0 us | no gain (c=1) |
| **G1 CUDA-graph captured** | **4.38** | **49.8 us** | CUDA_GRAPH_CADENCE_PASS |
| in-model reference (Nsight) | ~174 | **~1,524 us** | rank1 GPU idle 61.8% |

NCCL matrix (E1): default/Ring+LL/Ring+LL128/Ring+Simple all ~5.7 ms, ~63 us (within 3%).
**Tree+LL and Tree+LL128 REJECTED** — NCCL 2.30.4: *"No algorithm/protocol available for
function AllGather … NCCL_ALGO was set to Tree"* (Tree is AllReduce-only; cannot serve the
lm_head AllGather — this is why the trace uses `AllGather_RING_LL`). E3 rank-skew sweep:
injected rank1 delay grows the affected collectives' **p95** (60→324 us at 1170 us skew)
but not the mean (only 8/88 delayed) — mechanism direction confirmed, magnitude not.

### Findings
- **The in-model 1.524 ms/call is NOT reproduced by any standalone cadence, protocol,
  gap, or stream** (all 50–69 us/call, ~23× faster). It is NOT the wire and NOT NCCL.
- **Cause = cross-rank arrival skew driven by per-kernel host dispatch.** In-model rank1
  GPU idles 61.8% (host-dispatch starved) → rank1 reaches each collective late → rank0's
  all-reduce spins waiting → 1.5 ms. The standalone tight loop has no framework dispatch,
  so both ranks arrive together → 65 us. E3 confirms skew lengthens the waiting rank's
  collective; G1 shows graph removes ~25% of even the no-compute cadence (per-call host
  launch).
- **NCCL protocol is not a lever** (flat); **comm stream is not a lever** at c=1.
- **CUDA graph is graph-safe for the 88-collective sequence** (G1 PASS) and is the
  mechanism that would remove the dispatch-induced skew — but only via FULL/PIECEWISE
  MODEL graph, currently disabled (enforce_eager) due to the known graph-mode garble.

### Primary classification: EXACT_CADENCE_MIXED_CAUSE
Cross-rank arrival skew + per-kernel host dispatch; NOT NCCL protocol/cadence. No standalone
NCCL change meets the ≥20% full-model bar.

### Selected next action: PREPARE_FULL_GRAPH_SAFETY_PROBE
G1 proves the collective sequence is CUDA-graph-safe on this NCCL/PyTorch stack, and the
dominant in-model cost (host-dispatch-induced rank skew) is exactly what FULL/PIECEWISE
CUDA graph removes; the blocker is the known eager-vs-graph garble. Lowest-risk high-upside
next step = a bounded, correctness-first full-graph safety probe. This also makes the
historical graph-enabled hypothesis more plausible (decode 3× without a faster collective).
Standing decode lever remains MTP n=1. Single-node TP=1 is NOT a candidate. REJECTED:
PREPARE_NCCL_PROTOCOL_FULL_MODEL_TEST (matrix flat), RETAIN_MTP_ONLY_SAFE_PATH (graph probe
has higher upside and is now evidence-supported as safe at the collective level).

## Full-model CUDA-graph SAFETY probe — static preparation (2026-06-23)

Correctness-first, bounded, NOT yet executed. Performance deferred. The exact-cadence
result (host-dispatch rank skew, not NCCL) and G1 (the 88-collective NCCL sequence is
graph-capturable) motivate testing whether ATTENTION-CAPTURING decode CUDA graph is
correct/stable on SM121. G1 proves collective-level graph safety only — NOT full-model
correctness; this probe targets that gap.

### Supported graph modes (pinned source 72261a7, `config/compilation.py`)
`CUDAGraphMode`: NONE=0, PIECEWISE=1, FULL=2, FULL_DECODE_ONLY=(FULL,NONE),
FULL_AND_PIECEWISE=(FULL,PIECEWISE). PIECEWISE keeps "some attention ops outside the
cudagraph" (does NOT capture attention → does not remove the host-dispatch bottleneck;
this is why the prior U2 PIECEWISE gave no gain). FULL/FULL_DECODE_ONLY capture attention.

### Selected candidate: F2 = FULL_DECODE_ONLY (source-supported)
The DSV4 FlashMLA sparse-MLA metadata builder declares
`_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH` (`models/deepseek_v4/sparse_mla.py:134`;
compressor.py:87 = ALWAYS). UNIFORM_BATCH ⇒ full cudagraph is supported ONLY for uniform
decode batches, so **FULL (mixed) is unsupported; FULL_DECODE_ONLY is the correct, supported
attention-capturing mode**. `cudagraph_dispatcher.py:143` dispatches FULL for uniform_decode
batches with `uniform_decode_query_len = 1 + num_speculative_tokens` (=1 with MTP off) — an
exact match for our c=1 decode. Candidate ranking: **F2 chosen**; F1 FULL_AND_PIECEWISE
(higher risk, FULL on mixed unsupported by this backend); F3 shape-whitelist is folded into
F2 via `cudagraph_capture_sizes:[1]`; F4 (capture only attention region) is what PIECEWISE
already does inversely and does not capture attention.

### Graph configuration syntax
`--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1],"cudagraph_num_of_warmups":2}`
(no spaces ⇒ survives bash `VLLM_CMD+=(${VLLM_EXTRA_ARGS})` word-split as one token; JSON
validated). Remove `--enforce-eager`. capture sizes `[1]` ⇒ exactly ONE captured graph
(batch 1, query_len 1) ⇒ minimal graph-pool memory (single-token activations across 43
layers, << 1 GiB; weights/KV already resident; ESTIMATE_CUDAGRAPHS=1 accounts the pool;
fixed KV 2 GiB + ~110 GiB free start keeps the guard safe).

### Known SM121 garble/hang mechanisms (Phase 3) and mitigations
- `VLLM_USE_BREAKABLE_CUDAGRAPH=1` auto-enable → garble (jasl_breakable_cudagraph_garble_fix).
  STILL_PRESENT if =1 → preset keeps **=0**.
- FlashInfer SM121 reasoning garble (vllm023_step37) — UNRELATED to DSV4 default (FlashMLA
  sparse-MLA), but the probe MUST NOT switch to FlashInfer MLA backend.
- SM121 stale torch.compile cache → silent garble (sm121_stale_compile_cache_garble).
  STILL_PRESENT → operator MUST wipe `./.cache/vllm` on both nodes before the graph session.
- Warmup/UMA memory thrash hang (jasl_bump) — bounded by capture_sizes=[1] + memory guard.

### Disposable preset
`presets/deepseek-v4-v023-stack-pr41834-fullgraph-safety-f2-tp2.env`
(SHA-256 61468aeff7faf8f4acf30f764d3d6a09ca0d7affd371d0fb83845a111b4896b6, untracked).
Normalized delta from U0-RDMA baseline (`.env.dsv4-tvmfam019-u0rdma`, 24caeda8…) = exactly
TWO graph-only keys: (1) VLLM_EXTRA_ARGS drops `--enforce-eager` + adds the FULL_DECODE_ONLY
compilation-config; (2) VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS 0→1. All else identical
(RDMA NET/IB, TP=2 mp, MTP off, prefix off, fixed KV 2 GiB, max-len 8192, max-seqs 1,
max-batched 2048, c=1, async auto, BREAKABLE=0, NO init-memcheck bypass).

### Mandatory graph-activation proof (not a CLI flag alone)
Require, from source-supported evidence: resolved runtime `cudagraph_mode=FULL_DECODE_ONLY`
in the engine config dump; "Capturing CUDA graph" / capture-completion logs; the captured
size list = `[1]`; graph-pool allocation; replay evidence (no full eager fallback);
attention/indexer inside the captured region (UNIFORM_BATCH decode); rank-symmetric capture
completion on BOTH ranks.

### Correctness-first request ladder (client: `scripts/diag/dsv4_fullgraph_safety_probe.py`,
SHA-256 3acc56ee…, untracked, NOT executed)
- Gate 0: startup + graph-activation proof + no memory-guard event.
- Gate 1: 4 requests (arithmetic/English/Korean/Unicode), ≤16 tok, non-streaming.
- Gate 2: 10× 16-tok sequential, c=1, deterministic, health before+after each.
- Gate 3: 10× 32-tok sequential (only after Gate 2).
- Gate 4: 50× 32-tok soak — ONLY under a separate explicit approval.
Output integrity (not visual-only): HTTP 200, valid UTF-8, nonempty, finish_reason∈{length,stop},
no replacement-char flood (>2%), no degenerate repetition run (>24), nonzero completion
tokens; raw bytes + decoded + metadata preserved. Rank-health/memory gates before+after each
request. STOP at first: garble / timeout / HTTP 5xx / rank exit / NCCL error / graph replay
error / unexplained MemAvailable collapse / persistent health failure. NO automatic retry.

### Classification plan
FULL_GRAPH_SAFETY_PASS / _CORRUPTION_FAIL / _DELAYED_HANG_FAIL / _CAPTURE_FAIL /
_REPLAY_FAIL / _RANK_DIVERGENCE / _MEMORY_FAIL / _EAGER_FALLBACK_ONLY / _INCONCLUSIVE.
PASS requires: graph activation proved + Gate 1 pass + 10/10 Gate 2 + (Gate 3 if run) +
no corruption/rank-failure/delayed-hang/unexplained fallback. PASS ≠ performance validation.

### Deferred performance phase (prepared, not authorized)
Only after a safety PASS: tg128 d0 + d4096, runs 3, c=1, vs U0/U1/U2; optional MTP n=1
combination only AFTER graph-only performance validation. No U3 combined runtime in the
first probe.

### Runtime authorization boundary (requires explicit approval)
One reboot/node, one graph-safety session, Gates 0–2 (+3 if 2 passes), NO Gate 4, NO
benchmark, NO MTP, NO profiler, NO P4096, NO second reboot, NO auto-retry. Single-node TP=1
NOT a candidate (148.7 GiB > 121.63 GiB). Published presets unchanged.

## Full-model CUDA-graph safety probe — RESULT: FULL_GRAPH_SAFETY_PASS (2026-06-23)

Executed the authorized F2 probe. One reboot/node (clean-memory gate passed: boot IDs
changed 2a62ba9a→9d06f158 / 1f64852a→9c3f0a60, MemAvail 117.6/118.0 GiB, swap 0, ports
free, RoCE/RDMA ACTIVE, bidirectional OK). Cleared the verified disposable vLLM cache
(`…/build/.cache/vllm`, root-owned, only `modelinfos/*.json`, no symlinks/protected assets,
no torch.compile artifacts existed). Image config `4c41950c`, official 46-shard checkpoint,
preset `…fullgraph-safety-f2-tp2.env` (61468aef…). Artifacts:
`benchmarks/results/dsv4-fullgraph-safety-f2-20260623-071708/`.

### Graph activation proof (all satisfied)
- Resolved EngineCore config: `cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY: (2,0)>`,
  `enforce_eager=False`, `CompilationMode.VLLM_COMPILE`, `cudagraph_capture_sizes=[1]`,
  `max_cudagraph_capture_size=1`. splitting_ops include `vllm::deepseek_v4_attention`,
  `vllm::sparse_attn_indexer`, `vllm::unified_mla_attention_with_output` (attention/indexer
  inside the captured decode region).
- **Symmetric capture both ranks**: rank0 (Worker_TP0) "Capturing CUDA graphs (decode,
  FULL): 1/1" → "Graph capturing finished in 24 secs, took 0.45 GiB"; rank1 (Worker_TP1)
  "finished in 25 secs, took 0.47 GiB". NO eager fallback, NO capture exception.
- NET/IB both ranks (`rocep1s0f0:1/RoCE`, "Using network IB", no NET/Socket).
- Startup: weights 148.2s, model 73.82 GiB, KV 21,413 tokens, init engine 213s, /health 200.

### Gate results (correctness-first, deterministic, c=1, NO retry)
- **Gate 0**: PASS (both ranks alive, symmetric capture, health 200, no NCCL/CUDA/memory-guard
  event, graph mem 0.45/0.47 GiB, min MemAvail ~34 GiB).
- **Gate 1** (4 basic ≤16 tok): 4/4 PASS — arithmetic `4, 2+3=5…`, English ` Paris.`,
  Korean ` 서울입니다.`, Unicode OK; all HTTP 200, valid UTF-8, finish=length, no garble.
- **Gate 2** (10×16 tok): **10/10 PASS** — deterministic identical ` four, five, …, eleven,`,
  health 200 before/after each.
- **Gate 3** (10×32 tok): **10/10 PASS** — deterministic ` four … nineteen,`, all valid.
- Output-integrity (programmatic): valid UTF-8, nonempty, finish∈{length}, nonzero
  completion tokens, no replacement-char flood, no degenerate repetition — all pass.
- No delayed hang (post-gate logs clean both ranks), no rank divergence, no replay error.

### Classification: FULL_GRAPH_SAFETY_PASS
Graph activation proved + Gate 1 pass + Gate 2 10/10 + Gate 3 10/10 + no corruption /
rank-failure / delayed-hang / unexplained fallback. **This is SAFETY validation only, NOT
performance.** (Incidental, non-authoritative: graph decode steady ~723 ms/16 tok and
~1310 ms/32 tok ≈ 22–24 tok/s vs eager ~7.4 — consistent with the host-dispatch-skew
hypothesis, but formal performance is the deferred phase, not measured here.)

### Controlled shutdown
Head+worker down, containers 0, vLLM procs 0, ports free. Post-stop UVM retention:
immediate 39.4/39.6 GiB, delayed (≈min later) unchanged 39.4/39.6 GiB (no release without
reboot, expected GB10 behavior). No second reboot.

### Whether a separate graph performance experiment is justified: YES
Safety PASS clears the blocker that forced enforce_eager. A separate, explicitly-authorized
performance phase (tg128 d0/d4096, runs 3, c=1, vs U0/U1/U2; optional graph+MTP only after
graph-only perf) is now justified to quantify the decode gain the incidental latencies
suggest. Not run here.

## FULL_DECODE_ONLY graph-only PERFORMANCE — RESULT: FULL_GRAPH_PERF_MAJOR_GAIN (2026-06-23)

Executed the authorized graph-only performance session. One reboot/node (clean-memory gate
PASS: boot IDs 9d06f158→6a361e0b / 9c3f0a60→afef439e, MemAvail 117.8/118.0 GiB, swap 0, RoCE/
RDMA ACTIVE, bidirectional OK). Verified vLLM cache cleared (root-owned, no symlink/protected
assets). Preset `…fullgraph-safety-f2-tp2.env` (61468aef…, byte-identical both nodes). Local-
direct API path on spark01 (NOT mgmt-network, which caused the prior 3.21 t/s anomaly).
Artifacts: `benchmarks/results/dsv4-fullgraph-perf-f2-20260623-080241/`.

### Graph + transport activation (proved, both ranks)
`cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY:(2,0)>`, `cudagraph_capture_sizes=[1]`,
`enforce_eager=False`; symmetric capture (rank0 0.65 GiB/24s, rank1 0.19 GiB/25s; attention/
sparse-indexer in the captured decode region), NO eager fallback. NET/IB both ranks
(`rocep1s0f0:1/RoCE`, GID 3, "Using network IB", no NET/Socket). Backends: fp8_ds_mla KV,
MARLIN Mxfp4 MoE, DEEPSEEK_SPARSE_SWA (FlashMLA, not FlashInfer). Startup: weights 146.2s,
model 73.82 GiB, KV 21,413 tokens, init 213.5s, min MemAvail ~34 GiB.

### Correctness + bounded soak
Gate 1 4/4 PASS (deterministic, identical to the safety probe: `4, 2+3=5…` / ` Paris.` /
` 서울입니다.` / unicode OK). **20× 32-token soak 20/20 PASS** (each 32 tok, finish=length,
printable, no replacement char, health 200 before/after). No garble, no rank divergence.

### Benchmark (llama-benchy 0.3.7, generation mode, pp2048 tg128, runs 3, c=1, local-direct)
Warm-up (runs 1, excluded): tg128 d0 27.33, d4096 27.42. Measured (runs 3):
| test | t/s | std | CV |
|---|---|---|---|
| pp2048 d0 (prefill) | 1529.35 | ±2.91 | 0.19% |
| **tg128 d0 (decode)** | **27.27** | ±0.01 | **0.04%** |
| pp2048 @ d4096 (prefill) | 1595.24 | ±4.72 | 0.30% |
| **tg128 @ d4096 (decode)** | **27.23** | ±0.04 | **0.15%** |
All CV ≪ 5%; consistent generated-token counts; no per-run outlier; warm-up and measured agree
(27.33/27.27, 27.42/27.23). No direct-local vs benchmark-reported decode discrepancy.

### Decode comparison (primary metric)
| baseline | d0 | d4096 |
|---|---|---|
| U0 Socket eager 7.07/7.19 | **3.86×** (+286%) | 3.79× (+279%) |
| **U0 RDMA eager 7.41/7.36** | **3.68× (+268%)** | **3.70× (+270%)** |
| U1 eager MTP n=1 12.25/11.36 | 2.23× (+123%) | 2.40× (+140%) |
| U2 PIECEWISE graph 6.84/6.89 | 3.99× (+299%) | 3.95× (+295%) |
| Historical opaque 34.59 TOTAL | 27.27 = 78.8% (equivalence OPEN) | — |
Prefill also rose (1529 vs U0-RDMA 832, +84%) from inductor VLLM_COMPILE — reported but NOT used
for classification (warm-cache sensitive, and FULL_DECODE_ONLY does not graph prefill). Graph-only
(no MTP) decode 27.27 t/s **exceeds U1 MTP-eager (12.25) by 2.2×**, confirming the exact-cadence
hypothesis: removing per-kernel host dispatch collapses the cross-rank arrival skew that inflated
the all-reduce to ~1.5 ms; PIECEWISE gave nothing because it leaves attention/host-dispatch eager.

### Classification: FULL_GRAPH_PERF_MAJOR_GAIN
Graph activation proved + correctness + 20-soak passed + both d0 and d4096 decode ≥100% over
U0-RDMA (268%/270%) + no corruption/fallback/rank/memory instability. Graph replay/fallback
logs clean (0 eager-fallback, 0 replay error both ranks). Remaining stability limitation: only a
BOUNDED soak (4 + 20 + 26 requests, ~minutes) was run — long-duration stability is unproven.

### Controlled shutdown + UVM
Head+worker down, containers 0, vLLM procs 0, ports free. Post-stop UVM retention: immediate
39.4/39.6 GiB, delayed (~min) unchanged 39.4/39.6 GiB (no release without reboot, expected GB10).
No second reboot.

### Selected next action: PREPARE_MTP1_FULL_GRAPH_SAFETY_PROBE
Graph-only gain is material (MAJOR) with clean correctness and no delayed-hang/memory issue, so
the next milestone is to PREPARE (not run) a combined MTP n=1 + FULL_DECODE_ONLY safety probe —
graph-only 27.27 already beats U1 MTP-eager, and the combination targets the historical-34.59
range. NOTE: MTP changes `uniform_decode_query_len = 1 + num_speculative_tokens`, altering the
captured graph shape, so the combination requires its own safety validation; a longer graph-only
soak before promotion is also advisable. Graph-only ALSO qualifies as a
PROMOTE_GRAPH_ONLY_VALIDATED_PRESET_CANDIDATE (repeatable, CV 0.04%, clean) — recorded as a
candidate, not production. Combined mode NOT run in this task.

## MTP n=1 + FULL_DECODE_ONLY combined safety probe — STATIC PREPARATION (2026-06-23)

Correctness-first static preparation of a combined DeepSeek-MTP (`num_speculative_tokens=1`)
plus `FULL_DECODE_ONLY` attention-capturing decode-graph SAFETY probe. SAFETY ONLY: graph+MTP
activation, speculative/rejection correctness, rank consistency, capture/replay stability, bounded
memory, no garble/hang. Combined THROUGHPUT is a separate, deferred authorization. No model load,
no reboot, no benchmark/probe/profiler were run in this preparation.

### Source classification: EXPLICITLY_SUPPORTED (pinned image 72261a7)
The combination is explicitly handled by the pinned source, not merely tolerated:
- `method="deepseek_mtp"` is normalized to `method="mtp"` at config init
  (`config/speculative.py:589-593`, emits "deprecated and replaced with mtp"). `use_eagle()` then
  returns True, so the drafter is `EagleProposer` loading `DeepSeekV4MTPModel` as the MTP head
  (`gpu_model_runner.py:595-600` selection order: gemma4/step3p5 MTP guards are draft-model-type
  specific and fail, `use_dflash()` is False, `use_eagle()` True).
- MTP n=1 sets `uniform_decode_query_len = 1 + num_speculative_tokens = 2`
  (`cudagraph_dispatcher.py:37`, `gpu_model_runner.py:817`).
- `resolve_cudagraph_mode_and_sizes()` only downgrades FULL decode + spec-decode when
  `min_cg_support < UNIFORM_BATCH`. DSV4 FlashMLA sparse-MLA is `UNIFORM_BATCH`
  (`sparse_mla.py:134`), so the guard is False and `FULL_DECODE_ONLY` SURVIVES with MTP.

### Graph shape semantics: capture size MUST be [2], NOT [1] (Phase 3 resolved)
The pinned source requires the capture size to be a multiple of `uniform_decode_query_len`.
`adjust_cudagraph_sizes_for_spec_decode()` (`config/compilation.py:1466`) is invoked automatically
when `decode_mode()==FULL and uniform_decode_query_len>1`, with `multiple_of = uniform_decode_query_len
= 2` (TP=2 only raises `multiple_of` when sequence-parallelism is enabled; SP is OFF here).
- With `cudagraph_capture_sizes=[1]`: `round_up(1,2)=2 > max_cudagraph_capture_size(=1)` filters out,
  small-decode set empty, and the function RAISES `ValueError("No valid cudagraph sizes after rounding
  to multiple of 2 …")`. So `[1]` is HARD-REJECTED at engine init — not silently disabled. The
  graph-only fallback candidate `[1]` is therefore eliminated.
- `cudagraph_capture_sizes=[2]` yields exactly one FULL decode graph: dispatcher decode branch
  (`cudagraph_dispatcher.py:207-231`) keeps `x` with `2<=max_num_tokens(=2*1)=2` and `x>=2` → `[2]`;
  `_create_padded_batch_descriptor(2, uniform=True)` → `num_reqs=min(2//2,1)=1`, assert `2%2==0` OK →
  BatchDescriptor(num_tokens=2, num_reqs=1, uniform=True). One graph, one request, query_len 2.

### Draft vs verify graph keys (Phase 7 resolved): ONE graph total
- Target/verify model: `FULL_DECODE_ONLY` → ONE FULL decode graph at size [2].
- Draft (MTP head): `SpecDecodeBaseProposer.initialize_cudagraph_keys` (`llm_base_proposer.py:403-418`)
  keys on `cudagraph_mode.mixed_mode()`; for `FULL_DECODE_ONLY` `mixed_mode()==NONE` →
  `eagle_cudagraph_mode = NONE`. The MTP head runs EAGER, captures ZERO graphs. Net: a single FULL
  decode graph (target), no draft graph. This is simpler and lower-risk than a dual-capture design.

### Speculative-state graph safety (Phase 4): rejection runs OUTSIDE the graph
- The FULL graph wraps only `self.model()` target forward inside
  `set_forward_context(cudagraph_runtime_mode=FULL, batch_descriptor=…)` (`gpu_model_runner.py:4300-4316`).
- `RejectionSampler` (`v1/sample/rejection_sampler.py`) is invoked separately on the post-forward
  logits (`gpu_model_runner.py:3594`), using Triton recover/accept kernels — NOT inside any cudagraph
  (the draft path is eager, the target graph wraps only the forward). Sampling, acceptance counting,
  and recovered/bonus token selection all execute outside graph replay.
- Mutable speculative buffers are persistent (`_make_buffer`) and rewritten in `_prepare_inputs`
  BEFORE the forward, at fixed addresses the graph closed over: `self.input_ids` via
  `copy_to_gpu()`/`scatter_()` (`gpu_model_runner.py:1733/1791/1815/1839`); positions likewise;
  `self.num_accepted_tokens` set post-rejection (`:1516`) and consumed for next-step input shifting
  (`:2045-2056`). Because the captured shape is constant (num_tokens=2, num_reqs=1), the accepted-token
  count changes only buffer CONTENTS and KV/state shifting, never the tensor shape → no recapture.
- Bonus-token handling is active with n=1: "if all proposed tokens accepted, append the bonus token"
  (`rejection_sampler.py:763-767`); bonus is sampled from target probs at the draft position.

### Combined candidate MG2 (fixed) + disposable preset
`presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-safety-tp2.env`
(SHA-256 `f0bb73814dd600c18393599fee6ce40181de737957af6c11f9316117266db88a`). EXACT copy of the validated
graph-only preset (SHA `61468aeff7faf8f4…`, unchanged) with a normalized delta of exactly TWO
source-mandated changes on the `VLLM_EXTRA_ARGS` line:
1. add `--speculative-config {"method":"deepseek_mtp","num_speculative_tokens":1}`
2. `cudagraph_capture_sizes` `[1] → [2]`.
bash word-split verified: both JSON blobs remain single argv tokens (10 tokens total).

### Mandatory combined activation proof (Phase 10) — two INDEPENDENT facts
The probe and operator must confirm BOTH, separately (a nonzero acceptance rate ALONE does not prove
graph replay):
- (A) MTP active: Prometheus counters advance — `vllm:spec_decode_num_drafts_total`,
  `…num_draft_tokens_total`, `…num_accepted_tokens_total` (registered in `spec_decode/metrics.py:228-232`,
  Counters scrape with `_total` suffix). At least one request shows `drafts_delta>0`.
  Independent fallback: engine-log "Speculative metrics" acceptance line.
- (B) FULL graph replay active: from server logs (Gate 0) — the startup deprecation warning
  "method deepseek_mtp … replaced with mtp" (proves MTP path), the FULL decode cudagraph capture at
  the size-2 descriptor, MTP-head load, and decode dispatch hitting `CUDAGraphMode.FULL`. If `/metrics`
  returns 500 (the `_IncludedRouter` routing bug), apply the runtime Prometheus routing sed on BOTH
  nodes before the probe (non-image-mutating; see prometheus-routing-path-fix), else (A) fails closed.

### Request ladder + Gate 4 decision (Phase 11) — client `scripts/diag/dsv4_mtp1_fullgraph_safety_probe.py`
SHA-256 `6b7ab8798698a78eb4fb804c28e226025c901c0fbb4be6ca8ce05870ca881d9e`. Gates 0–3 mirror the
graph-only probe (basic / 10×16-tok / 10×32-tok, deterministic c=1, health around each, stop at first
anomaly, no retry) and ADD per-request spec-decode counter deltas + token-accounting checks. Gate 4
(rejection-heavy: incompressible hex / shuffled-token continuations) is DEFINED and JUSTIFIED by the
source audit (the recovered-token Triton path is distinct from uniform-accept traffic) but is gated
behind explicit approval (`DSV4_GATE4_APPROVED=1`), default-excluded.

### Token-accounting invariants for n=1 (Phase 12)
Per request, from counter deltas (drafts d, draft_tokens dt, accepted a) and `usage.completion_tokens` c:
- I2: `dt == d * 1` (each draft proposes exactly one token).
- I3: `0 <= a <= dt`.
- I4: per-step emission 1..2 → `d <= c <= 2*d`.
- I5: `c == d + a` within ±1 (a trailing bonus may be truncated at max_tokens).
- A1 (require-active): some request has `d>0`. Counters missing → fail closed.

### Memory impact estimate (Phase 9)
Versus the graph-only run (min MemAvailable spark01 ~34.1 / spark02 ~35.2 GiB), the combined mode
additionally resides: the MTP head weights (1 DSV4 layer, TP=2 sharded; est. ~1–2 GiB/rank), draft KV
(1 layer, tens of MB), and small speculative buffers. The FULL decode graph stays a single graph
(size [2], still 1 request) and the draft adds NO graph memory (eager). Estimated extra ~1–2 GiB/rank
→ comfortable headroom remains. Safe-stop threshold: abort if MemAvailable falls below ~8 GiB on either
node during load/capture, or on any host-unresponsive / swap-thrash signal (same conservative gate as
graph-only); preserve logs, controlled shutdown, no second reboot.

### Classification plan (Phase 15)
- MTP1_FULLGRAPH_SAFETY_PASS — graph[2] captured + MTP active (A&B) + Gates 0–3 + token-accounting
  invariants hold + no garble/rank/memory instability.
- MTP1_FULLGRAPH_GRAPH_DISABLED — engine starts but FULL graph not captured/replayed (e.g. dispatch
  falls to NONE); MTP may still run eager.
- MTP1_FULLGRAPH_CONFIG_REJECTED — engine init raises (capture-size / shape validation).
- MTP1_FULLGRAPH_MTP_INACTIVE — graph active but no drafts advance (spec path not running).
- MTP1_FULLGRAPH_ACCOUNTING_VIOLATION — an invariant I2–I5 fails (token bookkeeping inconsistent).
- MTP1_FULLGRAPH_GARBLE — output-integrity failure (invalid UTF-8 / replacement flood / repetition).
- MTP1_FULLGRAPH_RANK_ASYMMETRY — ranks disagree on graph/MTP activation or one rank stalls.
- MTP1_FULLGRAPH_MEMORY_UNSAFE — MemAvailable breach / swap thrash / host unresponsive.
- MTP1_FULLGRAPH_HANG — capture or decode hang (bounded timeout fires).
- MTP1_FULLGRAPH_RECAPTURE — unexpected graph recapture during decode (shape instability).
- MTP1_FULLGRAPH_REJECTION_PATH_FAIL — Gate 4 only: recovered-token path produces corruption.
- MTP1_FULLGRAPH_UNKNOWN — anomaly not matching the above; preserve all logs.

### Deferred combined performance design (Phase 16, NOT authorized here)
Only after a SAFETY_PASS and a separate explicit approval: llama-benchy 0.3.7 generation mode,
pp2048 tg128, runs 3, c=1, local-direct API path (NOT the management-net path that produced the
anomalous 3.21 t/s), warming the exact size-2 decode shape; compare combined decode t/s against
graph-only 27.27 (d0) / U1 MTP-eager 12.25 / historical-opaque 34.59. Acceptance-rate-adjusted
effective t/s reported alongside raw. Not prepared for immediate execution.

### Reboot approval gate (Phase 19): STATIC PREP COMPLETE — awaiting explicit approval
All static artifacts are in place and validated; the runtime phase (reboot both nodes → clean
≥110 GiB → start from the combined preset → Gate 0–3 probe) requires explicit user approval before
any reboot or model load.

## MTP n=1 + FULL_DECODE_ONLY combined safety probe — RESULT: MTP1_FULLGRAPH_SAFETY_PASS (2026-06-23)

Runtime execution of the combined safety probe under explicit authorization. SAFETY ONLY (no
throughput benchmark, no llama-benchy, no profiler, no soak beyond the gates, concurrency 1).
Single session, single reboot per node. Combined preset
`presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-safety-tp2.env` (SHA-256
`f0bb73814dd600c1…`), probe `scripts/diag/dsv4_mtp1_fullgraph_safety_probe.py` (SHA-256
`6b7ab8798698a78e…`); both staged byte-identical on both nodes from the non-repo runtime dir
`/home/bjk110/docker-build/vllm-spark-dsv4-9ceabf3-72261a7` (the spark01 git repo was untouched).
Image config ID `sha256:4c41950c47ecb771eead3e32147a750240d6605d15ee99c54b43e0a85ea26105`
(pinned `72261a7`). Results: `benchmarks/results/dsv4-mtp1-fullgraph-safety-20260623-121435/`.

### Reboot + clean-memory gate
New boot IDs (s01 `62e1263b…`, s02 `7c1d4af8…`); MemAvailable 117.7 / 118.0 GiB (≥110); swap 0;
no stale vllm/ray/nsys; no model containers; ports 8000+29500 free; RoCE UP (10.10.10.1/.2);
RDMA ACTIVE/LINK_UP; bidirectional RoCE ping OK; disk 32/74 GiB free. Dedicated vLLM cache
(`<runtime>/.cache/vllm`, verified abs path, not a symlink, only a stale `modelinfos` json — no
weights/tokenizer/AOT) cleared 12K→4.0K (0 files), dir preserved.

### /metrics — no patch needed
`/metrics` returned HTTP 200 on first start (no `_IncludedRouter` 500). Per authorization, NO
Prometheus patch was applied; the runtime was left unmodified.

### Graph activation proof (capture [2], both ranks)
Effective engine config: `cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY: (2,0)>`,
`cudagraph_capture_sizes=[2]`, `max_cudagraph_capture_size=2`, `enforce_eager=False`,
`VLLM_USE_BREAKABLE_CUDAGRAPH=0`, `disable_custom_all_reduce=True`. Capture log
`Capturing CUDA graphs (decode, FULL): 100%|██| 1/1` → exactly ONE graph (target/verify decode),
`uniform decode requests=[1]`, `next_n=2`. `Graph capturing finished in 2 secs, took -0.02 GiB`
on BOTH ranks (head Worker_TP0 pid=339 + worker Worker_TP1 pid=264) → rank-symmetric. ZERO
`falling back to eager`, ZERO recapture, ZERO `NET/Socket` (both logs). NET/IB:
`NET/IB : Using [0]rocep1s0f0:1/RoCE` + `Using network IB` (head 50 / worker 35 lines).
NOTE: requested `cudagraph_num_of_warmups=2` was honored in non-default args but the engine
normalized the resolved value to 1 for FULL_DECODE_ONLY (benign; warmup-count only).

### MTP activation proof (independent of graph proof)
`method deepseek_mtp is deprecated and replaced with mtp`; `Resolved architecture:
DeepSeekV4MTPModel`; `SpeculativeConfig(method='mtp', num_spec_tokens=1)`; `Loading drafter
model…` → `MTP draft model loaded: 39 params`; embedding + lm_head shared with draft;
`Warming up DeepSeek V4 MTP spec-decode kernels … 1 draft tokens`. Draft head captured NO graph
(only 1/1 = target) → runs eager, as predicted from source. Prometheus counters
`vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total` present and advanced per
request (proof independent of acceptance rate).

### Startup metrics
Model loading 75.52 GiB / 178.7 s (incl. MTP head; +~1.7 GiB/rank over graph-only 73.82 — the
39-param MTP layer, embed/lm_head shared). Initial free 114.2 GiB; KV fixed 2.0 GiB = 21,386
tokens (2.61× concurrency @8192). Graph capture 2 s / −0.02 GiB. init engine 218.56 s. Min
MemAvailable during startup: s01 29.7 / s02 30.9 GiB (≫16 warn, ≫12 stop). Swap stable ~2.9 GiB,
no thrash.

### Gate results (deterministic, c=1, NO retry)
- Gate 0 PASS: both ranks alive, graph[2] captured, MTP head+proposer init, API ready, /metrics
  ok, counters readable, no fallback/NCCL/CUDA/memory event.
- Gate 1 PASS 4/4 (arithmetic/English/Korean/Unicode, 16 tok). drafts>0 each; accept 0.25–1.0;
  Korean `서울입니다` clean (no Ġ/Ċ); accounting I2–I5 hold (`c−(d+a)=1` = prefill token).
- Gate 2 PASS 10/10 (16 tok) byte-identical, d8/a8, ~545 ms steady (first 3793 ms warm).
- Gate 3 PASS 10/10 (32 tok) byte-identical, d16/a16, ~910 ms.
- Gate 4 PASS (rejection-heavy, max 32 tok, 4 prepared prompts; authorization cap 32 applied,
  the prepared probe's 48 NOT used — invoked the probe's checks via import at 32). Rejection
  path EXERCISED: hex1 d20/a11, hex2 d21/a10, rand d19/a13, shuffled d20/a12 (6–11 rejections
  each, accept 0.48–0.68). All valid UTF-8 (DeepSeek special tokens are valid, not garble),
  accounting consistent, health 200, no duplicate/dropped token, no recapture.

### Token accounting + integrity
For n=1: `draft_tokens == drafts` held in every request (final totals 357/357, accepted 310 =
86.8%). `0 ≤ accepted ≤ draft_tokens` always; `drafts ≤ completion ≤ 2·drafts`;
`completion = drafts + accepted (±1 prefill)`. No invariant violation, no counter reset/scrape
race. Output integrity: all valid UTF-8, no replacement-char flood, no degenerate repetition,
finish_reason ∈ {length}, nonzero completion tokens. Determinism: temp 0 byte-identical across
repeats.

### Rank health / memory / fallback
Both ranks symmetric throughout; no rank exit, no NCCL/CUDA error, no eager fallback, no graph
recapture. Min MemAvailable across the whole session ~29.7 GiB; never approached 16 GiB warn or
12 GiB stop; swap flat (~2.9 GiB), no thrash, no host unresponsiveness.

### Controlled shutdown + UVM
Requests stopped; logs/metrics/probe outputs preserved; head then worker `down`; containers 0;
real vllm/ray procs 0 (a `pgrep -f 'vllm'` self-match false-positive was excluded via `ps`);
ports 8000+29500 free. Post-stop UVM retention: immediate s01 37.6 / s02 37.7 GiB MemAvailable,
delayed (~90 s) unchanged 37.6 / 37.7 GiB (no release without reboot, expected GB10). No second
reboot.

### Final classification: MTP1_FULLGRAPH_SAFETY_PASS
Graph activation at [2] proved; MTP activation proved independently; Gate 1 4/4; Gate 2 10/10;
Gate 3 10/10; Gate 4 rejection path exercised + passed; token accounting valid; no corruption,
fallback, recapture, rank divergence, memory instability, or delayed hang within the bounded
session. SAFETY validation only — no performance benchmark was run.

### Whether a separate combined performance experiment is justified: YES (deferred)
The combination is proven safe and correct with a meaningful global acceptance rate (86.8% on
this gate mix; n=1 ⇒ theoretical decode speedup up to ~1.87× over the graph-only 27.27 t/s if the
extra verify token is free). A separate, explicitly-authorized performance experiment
(llama-benchy generation, pp2048 tg128, runs 3, c=1, local-direct; acceptance-adjusted effective
t/s vs graph-only 27.27 / U1 MTP-eager 12.25 / historical 34.59) is justified but was NOT run here.

## MTP n=1 + FULL_DECODE_ONLY combined PERFORMANCE — RESULT: MTP1_FULLGRAPH_PERF_MAJOR_GAIN (2026-06-23)

Runtime performance validation of the combined path under explicit authorization. Single session,
single reboot per node. No profiler/Nsight, concurrency 1, no soak beyond the gates. Combined
preset SHA-256 `f0bb73814dd600c1…` (re-verified homeserver + both nodes); GHCR image
`ghcr.io/bjk110/vllm-spark@sha256:2f4a96283fc5b491…`, config ID `sha256:4c41950c47ecb771…`
(verified), pinned `72261a7`. Runtime dir = non-repo
`/home/bjk110/docker-build/vllm-spark-dsv4-9ceabf3-72261a7` (spark01 git repo untouched).
Results: `benchmarks/results/dsv4-mtp1-fullgraph-perf-20260623-125104/`.

### Reboot + clean-memory + cache
New boot IDs (s01 `48a401a8…`, s02 `d73760f6…`); MemAvailable 117.7/118.0 GiB (≥110); swap 0;
no stale procs/containers; ports 8000+29500 free; RoCE UP; RDMA ACTIVE/LINK_UP; bidirectional
ping OK; disk 32/74 GiB. Dedicated vLLM cache (verified path/no symlink/no weights/tokenizer/AOT)
cleared 16K→4.0K. `/metrics` 200 on first start — no patch applied.

### Activation proofs (all PASS, both ranks)
Graph: `FULL_DECODE_ONLY:(2,0)`, `cudagraph_capture_sizes=[2]`, `max=2`, `enforce_eager=False`;
`Capturing CUDA graphs (decode, FULL): 1/1` + `Graph capturing finished` on head (TP0) and worker
(TP1); zero eager-fallback/recapture/NET-Socket. MTP: `replaced with mtp`, `DeepSeekV4MTPModel`,
`method='mtp' num_spec_tokens=1`, `MTP draft model loaded: 39 params`; draft eager (only 1/1 graph).
Transport: `NCCL_NET_PLUGIN=none NCCL_IB_GID_INDEX=3`, `NET/IB : Using rocep1s0f0:1/RoCE`,
`Using network IB` (HCA rocep1s0f0, port 1, GID 3). `cudagraph_num_of_warmups` requested 2 →
engine-normalized 1 (benign, as before).

### Correctness + 20-request soak
Correctness gate 4/4 (deterministic, identical to safety run: arith d8/a7, eng d9/a6, kor
`서울입니다` d12/a3, uni d9/a7). 20-request soak (max 32 tok, c=1) PASS 20/20 byte-identical,
d16/a16, ~915 ms, health 200/200, no garble, accounting clean.

### Warm-up (excluded)
Unmeasured tg128 d0=39.91, d4096=37.16 t/s; coherence PASSED; health 200; no recapture; nonzero
acceptance. Excluded from results.

### Benchmark matrix (llama-benchy 0.3.7, generation, pp2048 tg128, runs 3, c=1, local-direct)
Primary metric = API-visible generated-token throughput. Two passes run (pass-1 = the authorized
matrix; pass-2 = an instrumented re-measurement to capture the accepted-token counter at exact
case boundaries after a grep-pattern miss on pass-1; pass-2 corroborates throughput and supplies
per-case acceptance).

| Case | metric | pass-1 (primary) | pass-2 (instrumented) | prefill pp2048 |
|---|---|---|---|---|
| A d0    | tg128       | 38.92 ± 0.90 t/s (CV 2.3%, peak 41) | 39.82 ± 0.44 (CV 1.1%) | 1490.48 ± 2.63 / 1500.95 ± 3.37 |
| B d4096 | tg128@d4096 | 38.81 ± 1.22 t/s (CV 3.1%, peak 42) | 39.79 ± 1.39 (CV 3.5%) | 1535.48 ± 8.05 / 1534.05 ± 2.82 |

Per-case speculative metrics (instrumented pass, n=1 so draft_tokens==drafts):
- d0:    drafts 220, accepted 180, **acceptance 81.8%**.
- d4096: drafts 223, accepted 187, **acceptance 83.9%**.
Token accounting: `draft_tokens==drafts` both cases; `accepted ≤ draft_tokens` both; no counter
reset; API completion tokens consistent. MTP work is explanatory only — NOT added to API
throughput.

### Comparison (pass-1 primary)
| baseline | d0 | d4096 |
|---|---|---|
| **graph-only** 27.27 / 27.23 | **1.428× (+42.8%)** | **1.425× (+42.5%)** |
| MTP n=1 eager 12.25 / 11.36 | 3.18× | 3.42× |
| U0-RDMA eager 7.41 / 7.36 | 5.25× | 5.27× |
| U0 Socket eager 7.07 / 7.19 | 5.50× | 5.40× |
| historical opaque 34.59 (c=1 TOTAL) | 1.13× (equivalence OPEN, NOT used for classification) | — |

### Memory + swap
Min MemAvailable s01 30.1 / s02 31.1 GiB (≫16 warn, ≫12 stop). Max SwapUsed s01 2.9 / s02 2.0 GiB.
Swap rose during init then stayed FLAT: pswpout frozen (731066/479797) and pswpin essentially
frozen (Δ≈15 pages) across the entire measured window → ZERO sustained swap-in/out during measured
decode, no major-fault burst, no host-responsiveness loss → classification "Acceptable". No slow
run attributable to paging.

### Stability
No eager fallback, no recapture, no replay error, no rank divergence, no NCCL/CUDA error
throughout. Both ranks symmetric.

### Controlled shutdown + UVM
head then worker `down`; containers 0; real vllm/ray procs 0; ports free. Post-stop: MemAvailable
immediate s01 37.6 / s02 37.8 GiB, delayed (~90 s) unchanged 37.6 / 37.8 (UVM retained, no release
without reboot — GB10). SwapUsed released to ~0.2/0.1 GiB. No second reboot.

### Final classification: MTP1_FULLGRAPH_PERF_MAJOR_GAIN
Correctness + 20-soak pass; graph + MTP activation proved; both d0 (+42.8%) and d4096 (+42.5%)
improve ≥20% over graph-only; no corruption, fallback, metric ambiguity, or memory instability.

### Selected follow-up: PROMOTE_MTP1_FULLGRAPH_VALIDATED_CANDIDATE
All criteria met: repeatable (two passes within ~1 t/s; CV 1.1–3.5%, all <5%), token accounting
clean, no fallback/recapture, swap stable, no correctness issue. Designated a VALIDATED CANDIDATE
(not production promotion). A longer-duration soak remains the natural next validation before any
production promotion.

## MTP n=1 + FULL_DECODE_ONLY — 4-hour bounded long-soak (2026-06-23/24)

Bounded endurance run of the VALIDATED CANDIDATE. Same immutable image config
(`vllm-spark:v023-stack-dsv4-sm12x-pr41834-exp-72261a7-tvmfam019`), same combined preset
(`presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-safety-tp2.env`, SHA-256
`f0bb73814dd600c18393599fee6ce40181de737957af6c11f9316117266db88a`), dual GB10 TP=2 mp over RDMA,
FULL_DECODE_ONLY + deepseek_mtp n=1, capture size [2], concurrency 1. Single reboot/node before
start; `.cache/vllm` wiped both nodes; clean-memory gate (>110 GiB free) passed; startup
correctness+activation gate passed before the soak loop. Runtime location = non-repo staging dir
`/home/bjk110/docker-build/vllm-spark-dsv4-9ceabf3-72261a7` (spark01 git repo untouched).

Driver `scripts/diag/dsv4_mtp1_fullgraph_long_soak.py` (SHA-256
`49e556be5f4f153e14956438f2c3fd997b316221aed0573531f26b4f810f4202`, paced variant). The 4h + ≤3000
requests + concurrency-1 ceiling implies a paced workload; inter-request pacing `--pace 4.5`
(~2000 req/4h) was used. Methodology note: the soak SERVING session was a single uninterrupted
session — only the host-side workload driver was relaunched once with pacing after an initial
unpaced prelude (~73 req) was discarded; that prelude is NOT counted and is not a retry of a failure.

### Run outcome
- Duration 14414.1 s = **4.004 h**, `stop_reason=duration_complete` (ran full 4h, not stopped at 1000).
- attempted = successful = **2040** requests, **102** full 20-request mixed cycles, concurrency 1.
- Per-request integrity: **2040/2040 ok, 0 violations** (UTF-8, finish_reason, repetition, empty,
  zero-completion all clean).
- Token accounting: **2040/2040 consistent** — `draft_tokens==drafts` every request,
  `accepted ≤ draft_tokens` every request; global drafts 83305 / draft_tokens 83305 / accepted 72383
  → **global acceptance 86.89 %** (per-request deltas; server cumulative counter monotonic
  3881→87417 incl. startgate+sentinels, no reset).
- Rejection evidence: **102/102 cycles exercised the rejection path** (rejection-heavy requests 18–19
  produced rejections every cycle).

### Stability over 4h
No eager fallback, no recapture, no replay/rank-divergence, no NCCL/CUDA error across the full
window (head log scan 0/0). Both ranks symmetric. health=200 on all 102 cycle snapshots.

### Performance sentinels (drift, API-visible, NOT llama-benchy)
8 sentinels @ ~30 min (1 short-128 + 1 d4096-128). decode_tps **d0 min 21.49 / max 21.73**
(spread 0.24), **d4096 min 21.58 / max 21.87** (spread 0.29) → flat across 4h, **zero drift /
degradation**. These are API end-to-end decode-tps under the paced mixed soak (httpd overhead,
concurrency 1) and are intentionally a stability monitor, not an absolute-throughput measurement;
they are NOT comparable to the llama-benchy generation-mode 38.9 t/s of the perf phase. Flatness is
the signal.

### Memory + swap (102 snapshots)
**Min MemAvailable 32.0 GiB** (≫12 GiB stop, ≫16 warn), flat 32.18→32.15 over 4h. **Max SwapUsed
2.87 GiB**, flat 2.870→2.864. `pswpout` frozen at 721410 the entire run (zero sustained swap-out);
`pswpin` Δ≈2384 pages over 4h (negligible) → no paging pressure, no host-responsiveness loss.

### Controlled shutdown + UVM
head then worker stopped (graceful `docker stop -t 30`, head first), containers removed; vllm/ray
procs **0**; ports 8000/6379/29500 **free** on both nodes. Post-stop MemAvailable immediate
s01 37.6 / s02 37.8 GiB, delayed (~90 s) **unchanged** 37.6 / 37.8 (UVM retained by driver, no
release without reboot — GB10 known behavior). SwapUsed released to 0 both nodes. **No second
reboot performed.**

### Final classification: MTP1_FULLGRAPH_LONG_SOAK_PASS
4h+ continuous (4.004 h) AND ≥1000 successful (2040) AND zero eager fallback AND zero recapture AND
zero accounting inconsistency AND zero output corruption AND memory/swap stable within bounds. All
PASS conditions met.

### Selected follow-up: PREPARE_COLD_START_REPRODUCIBILITY_RUN
Clean long-soak PASS → prepare a cold-start reproducibility run (fresh reboot, fresh cache wipe,
re-derive the startup/activation gate and a shortened sentinel ladder) to confirm the candidate
reproduces from cold across an independent boot before any production-promotion decision. Static
preparation only; no execution without separate explicit approval. Note: the next cold start
requires a reboot first — current post-stop MemAvailable ~37.6/37.8 GiB (UVM retained) is below the
≥110 GiB load gate.

## MTP n=1 + FULL_DECODE_ONLY — cold-start reproducibility (STATIC PREP, 2026-06-24)

Static preparation only. No model load, no reboot, no benchmark/soak/profiler/probe, no container
start, no commit/push/stage during this preparation.

### Objective
Determine whether the VALIDATED candidate (long-soak `MTP1_FULLGRAPH_LONG_SOAK_PASS`) reproduces
after a fully independent cold start: new clean boot, fresh dedicated vLLM cache, new serving
process, new graph capture, new MTP initialization, NO reuse of any prior serving session. This
verifies REPRODUCIBILITY, not new tuning — no runtime optimization is introduced.

### Immutable runtime identity (unchanged from the candidate)
Image `ghcr.io/bjk110/vllm-spark@sha256:2f4a96283fc5b491d5e28cee607525e32e914615bb469978beb2336cf8e62c44`
(config ID `sha256:4c41950c47ecb771eead3e32147a750240d6605d15ee99c54b43e0a85ea26105`), pinned source
`72261a7af149fa5d3fe2ed2b9956e92590731012`, model `deepseek-ai/DeepSeek-V4-Flash` (official 46-shard
checkpoint), candidate preset
`presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-safety-tp2.env` (SHA-256
`f0bb73814dd600c18393599fee6ce40181de737957af6c11f9316117266db88a`, verified intact). Retained
exactly: TP=2 mp, one rank/node, NET/IB, MTP n=1, FULL_DECODE_ONLY, capture size [2], target/verify
graph captured, draft path eager, FlashMLA, EP off, prefix off, parser off, tool parser off, B12X
off, Ray off, fixed KV 2 GiB, KV fp8, max-model-len 8192, max-seqs 1, max-batched-tokens 2048,
GPU util 0.87, concurrency 1, init-memory-check bypass DISABLED, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`.

### Independence criteria (reset-by-reboot vs reset-by-cache-clear)
PASS requires both nodes independent of the prior session:
- **Reset by reboot**: kernel `boot_id` (both nodes must change), retained-UVM driver state
  (~37–38 GiB freed → ≥110 GiB MemAvailable), swap drained to 0, all in-memory graph pools / KV /
  Ray-or-mp state, any preserved runtime container, the reused API process, in-process MTP counter
  state, and any stale benchmark/soak driver process. The post-stop GB10 UMA retention does NOT
  release without a reboot, so a reboot is mandatory before the cold load.
- **Reset by cache clear**: the dedicated `./.cache/vllm` (torch.compile / inductor artifacts) on
  BOTH nodes — wiped fresh so the graph capture is genuinely regenerated (guards the SM121
  stale-compile-cache garble). Boot does not clear this on-disk cache; the explicit wipe does.

### Reuse vs regenerate
- **Immutable (reused)**: image, model, candidate preset, probe/driver logic, workload templates,
  expected baseline ranges.
- **Fresh per run**: boot IDs, runtime container IDs, graph-capture logs, MTP counters, request
  logs, memory logs, performance sentinel logs, result directory.
- **Reserved unique result dir** (no collision; long-soak result NOT overwritten):
  `benchmarks/results/dsv4-mtp1-fullgraph-coldrepro-20260624/` (runtime may append a time suffix).

### Gates (runtime sequence R0→R1→R2→R3→R4)
- **R0 — clean-start proof** (`dsv4_mtp1_fullgraph_cold_repro.py --phase r0check
  --baseline-bootids <s01>,<s02>`): both boot IDs changed, MemAvailable ≥110 GiB, swap 0, dedicated
  cache empty, ports 8000/6379/29500 free, no stale model process, RoCE/RDMA healthy.
- **R1 — startup identity + activation** (caller-verified from head/worker startup logs): exact
  image config ID, exact model identity, exact candidate preset SHA, FULL_DECODE_ONLY active,
  capture size [2], one target/verify graph captured, draft path eager, MTP head loaded, proposer
  active, NET/IB active, no fallback/recapture/capture error.
- **R2 — correctness + rejection** (`--phase r2`): arithmetic/English/Korean/Unicode + 4 bounded
  rejection-heavy requests; require valid UTF-8, correct structure, ≥1 rejection, valid token
  accounting, no duplicate/dropped token, health 200 before+after. Stop on first anomaly.
- **R3 — performance sentinels** (run via llama-benchy, NOT this driver — see methodology note):
  exactly three fixed d0-128 and three fixed d4096-128 generation-mode requests, concurrency 1,
  direct-local API, record API-visible decode t/s.
- **R4 — bounded stability window** (`--phase r4 --pace 4.5 --duration 3600`): 60 minutes,
  concurrency 1, fixed paced mixed 20-request cycle (same cycle as the long soak), one continuous
  sequence, ~400–600 requests, no driver restart, no prelude, no discard, no retry.

### R3 sentinel methodology (static-audit conclusion)
The validated d0/d4096 throughput band (38.9–39.8 t/s) is a **llama-benchy 0.3.7 generation-mode**
number (`benchmark_command.txt`: `--pp 2048 --tg 128 --depth {0|4096} --runs 3 --concurrency 1
--latency-mode generation`). An in-script `post_stream` sentinel reproduces only ~21.6 t/s (the
long-soak sentinel value) because it measures a different end-to-end path; comparing it to the
38.9–39.8 band would spuriously fail the ±10% test. **Conclusion: Gate R3 MUST use the exact
llama-benchy generation-mode command above** so the comparison is methodology-matched. The cold-repro
driver therefore covers R0/R2/R4 only; R3 is the standalone llama-benchy invocation.

### Acceptance ranges
- **Performance**: mean d0 within ±10% of 38.9–39.8 t/s; mean d4096 within ±10% of 38.8–39.8 t/s;
  no monotonic throughput decline across the 3 runs. (Do NOT compare the paced R4 post_stream
  cadence to llama-benchy values.)
- **Acceptance**: normal-workload cumulative acceptance ≥75% (expected 80–90%); rejection path
  exercised; zero accounting violation. Lower acceptance permitted on rejection-heavy prompts.
- **Memory**: MemAvailable minimum ≈30–32 GiB; init swap ≈2–3 GiB then flat. Flag if MemAvailable
  differs >4 GiB from prior clean runs, SwapUsed grows >512 MiB after startup stabilization,
  sustained paging, or major-fault bursts.

### Reproducibility classifications
`MTP1_FULLGRAPH_COLD_REPRO_PASS`, `_STARTUP_FAIL`, `_GRAPH_FAIL`, `_MTP_FAIL`, `_CORRUPTION_FAIL`,
`_ACCOUNTING_FAIL`, `_PERF_MISMATCH`, `_MEMORY_MISMATCH`, `_RANK_FAIL`, `_INCONCLUSIVE`.
PASS requires: independent clean boot + independent fresh graph capture + independent MTP init +
R2 pass + R3 sentinels in-band + R4 60-min window complete + accounting violations 0 + fallback 0 +
recapture 0 + no sustained paging + no rank failure + no corruption.

### Methodology correction (vs long soak)
The long-soak report transparently documented an initial unpaced prelude + one driver restart (the
serving session stayed continuous). For the cold-start run this ambiguity is removed: the fixed pace
is computed BEFORE execution (long-soak measured 7.066 s/req at `--pace 4.5` → 3600 s / 7.066 ≈ 509
requests, inside the 400–600 band → **pace fixed at 4.5 s**), set in the first and only driver
invocation; the driver has NO unpaced mode (`--pace` is mandatory and >0), runs no prelude, never
restarts, never discards early requests, and writes the measured-start timestamp before the first
request into one result directory as one continuous sequence.

### Per-request + per-cycle monitoring (Gate R4 driver)
Per request: seq, workload type, timestamp, HTTP status, duration, prompt/completion tokens, drafts,
draft_tokens, accepted, acceptance, raw byte length, decoded text, health, MemAvailable, SwapUsed.
Per 20-request cycle: process tree, health, log fallback/recapture/NCCL/CUDA scan, MTP cumulative
counters, acceptance stats, vmstat paging (pswpin/out), major faults (pgmajfault), PSI memory
pressure, port state, worker container state, host-responsiveness probe.

### Stop conditions (no automatic retry)
graph capture failure, graph fallback, graph recapture, MTP inactivity, malformed output,
token-accounting violation, duplicate/dropped output, HTTP 5xx, timeout, rank exit, NCCL error,
CUDA error, sustained paging, MemAvailable <12 GiB, persistent health failure.

### Promotion decision plan (choose exactly one after the run)
`PREPARE_VALIDATED_PRESET_PROMOTION` (preferred after a clean PASS — repository + preset promotion
PLAN only, no production promotion), `RUN_SECOND_COLD_START_REPRO`, `RETAIN_VALIDATED_CANDIDATE`,
`INVESTIGATE_COLD_START_VARIANCE`, `INVESTIGATE_MEMORY_VARIANCE`, `ABANDON_COMBINED_CANDIDATE`.

### Promotion-readiness checklist (prepared, NOT executed)
candidate preset naming · validated-status metadata · rollback preset · immutable image digest ·
model identity · required clean-boot procedure · required cache-clear procedure · expected startup
memory · expected steady memory · expected swap behavior · expected d0/d4096 throughput · known
limitations · long-soak evidence · cold-start evidence · operational stop conditions. The published
preset is NOT modified during static preparation.

## MTP n=1 + FULL_DECODE_ONLY — cold-start reproducibility RUN (2026-06-24)

Executed under explicit authorization (one reboot/node, R0→R4, no profiler/retry/restart, no second
reboot, no production promotion). Same immutable identity verified pre-run: image config ID
`sha256:4c41950c…`, candidate preset SHA `f0bb73814dd600c1…`, pinned `72261a7`. Runtime location =
non-repo staging `/home/bjk110/docker-build/vllm-spark-dsv4-9ceabf3-72261a7`. Result dir
`benchmarks/results/dsv4-mtp1-fullgraph-coldrepro-20260624/`.

### Independent boot + cache proof (R0)
Single reboot/node. Boot IDs CHANGED: spark01 `e8bfb5d5…` → `0c186f46-372a-427a-8632-5fd97979174c`,
spark02 `f49863cf…` → `b4201cd3-7899-441a-af64-0e06dadad995`. Post-reboot MemAvailable 117.7 / 118.0
GiB (≥110), swap 0, ports free, RoCE bidirectional, RDMA ACTIVE. Dedicated `./.cache/vllm`
(modelinfos only; no symlink/weights/tokenizer/AOT/benchmark) cleared on both nodes (2 files → absent)
→ graph capture genuinely regenerated. (`r0_proof.json` flagged `stale_model_procs=1`, corrected in
`r0_correction.json`: a self-match — the driver runs from a staging path containing the substring
"vllm" so `pgrep -af vllm` matched its own process; zero real stale model/engine/ray processes.)

### Fresh graph + MTP initialization (R1)
Fresh cold capture: `cudagraph_mode=FULL_DECODE_ONLY (2,0)`, `cudagraph_capture_sizes=[2]`,
`max_cudagraph_capture_size=2`, indexer `next_n=2`. Exactly ONE target/verify graph captured per
rank — rank0 "Graph capturing finished in 2 secs, took -0.01 GiB", rank1 (node-rank 1 --headless)
"finished in 2 secs, took -0.02 GiB" (symmetric; draft path eager — MTP head served via kernel
warmup, not a captured graph). `MTP draft model loaded: 39 params` on BOTH ranks (fresh), method
`deepseek_mtp`→`mtp`, num_spec_tokens=1, embed/lm_head/topk_indices shared. Spec counters started
from 0. NET/IB: `NCCL_IB_HCA=rocep1s0f0`, `NET/IB: Using [0]rocep1s0f0:1/RoCE`, `NCCL_IB_GID_INDEX=3`,
`Using network IB`, no NET/Socket fallback. Startup: model load 75.52 GiB / 174.3 s, KV cache 21,386
tokens, max concurrency 2.61x, graph capture ~2 s / ~0 GiB, init engine 220.89 s. No fallback /
recapture / replay / CUDA / NCCL error on either rank.

### R2 — correctness + rejection
4 basics (arith/English/Korean/Unicode) + 4 rejection-heavy, all HTTP 200, integrity ok, accounting
ok, `draft_tokens==drafts` every request, 4 rejections observed, Korean/Unicode clean → **R2 PASS**.

### R3 — exact performance (llama-benchy 0.3.7 generation, methodology-matched)
Command (validated methodology): `--pp 2048 --tg 128 --depth {0|4096} --runs 3 --concurrency 1
--latency-mode generation`. Shape warm-up (excluded): d0 tg128 42.17, d4096 tg128 35.19. Coherence
PASSED. **Measured d0 tg128 = 40.15 ± 1.03 t/s** (peak 43.33; CV 2.57%), **measured d4096 tg128 =
37.11 ± 1.64 t/s** (peak 40.00; CV 4.42%); pp2048 1465.37 / pp2048@d4096 1531.83 t/s. Both within
±10% of the validated bands (d0 38.9–39.8, d4096 38.8–39.8), CV <5%, no monotonic decline, tg=128
consistent, counters monotonic (0→…, drafts==draft_tokens), no fallback/recapture → **R3 PASS**.
(Note: `r3_snapshots.txt` retained only the post-d4096 snapshot — a benign logging bug in the
runner's `snap()` redirect; final counters/health/mem/paging preserved.)

### R4 — one uninterrupted 60-minute stability interval
Single driver invocation `--phase r4 --pace 4.5 --duration 3600`; measured_start written before the
first request (`10:15:11`); no prelude, no discard, no restart, one continuous sequence, one result
dir. **Duration 60.1 min, attempted = successful = 507** (band 400–600), **26 cycles**, concurrency 1.
Workload distribution: short 207, tg128 150, d4096 75, reject 50, multiling 25. Per-request integrity
507/507 ok (0 violations); accounting 507/507 (`draft_tokens==drafts`, `accepted≤draft_tokens`).
Global acceptance 86.65%; **normal-workload acceptance 87.63%** (≥75 required); rejection path
exercised 25/26 cycles + R2. Cycle monitoring (26 snapshots): health 200 all; fallback 0, recapture
0, NCCL 0, CUDA 0; worker_state "running" all; proc_tree stable; **MemAvailable min 31.7** (30–32
band); **SwapUsed flat 2.8** (2–3 band); **pswpout Δ0** (zero sustained swap-out), pswpin Δ549
pages/60 min, pgmajfault Δ744 (isolated, non-sustained); PSI memory avg10/60/300 = 0 (zero pressure);
host-responsiveness 4.1–8.2 ms (no stall); disk free spark01 31 G / spark02 73–74 G (≥20 G) → **R4
PASS**.

### Controlled shutdown + UVM
head then worker graceful stop (`docker stop -t 30`, head first), removed; containers 0, real
vllm/ray procs 0, ports 8000/29500 free both nodes. Post-stop MemAvailable immediate s01 37.6 / s02
37.7 GiB, delayed (~90 s) unchanged 37.6 / 37.7 (UVM retained, no release without reboot — GB10),
swap → 0, disk 31 G / 74 G. No second reboot.

### Final classification: MTP1_FULLGRAPH_COLD_REPRO_PASS
Independent reboot both nodes + fresh cache + independent graph capture + independent MTP init + R2
PASS + R3 in-band + R4 full 60 min + no restart/prelude + accounting 0 + fallback/recapture 0 + no
sustained paging + no rank failure + no corruption. All PASS conditions met. The candidate
reproduces from an independent cold start within tolerance (d0 40.15 ≈ +0.9% over the validated
upper, d4096 37.11 ≈ −4.4% under the validated lower, both inside ±10%).

### Selected follow-up: PREPARE_VALIDATED_PRESET_PROMOTION
Clean cold-start PASS → prepare (NOT execute) a repository + preset promotion plan only. No
production promotion and no published-preset modification performed in this run. The candidate
remains a VALIDATED CANDIDATE.

The candidate remains a VALIDATED CANDIDATE, NOT production.
