# Solar-Open2 v0.25.1 pread opt-in patch — r3 experimental image build report
Arc: solar-open2-v0251-pread-r3-build-20260726T002411Z

## FINAL STATE: SOLAR_OPEN2_V0251_PREAD_R3_BUILD_VALIDATED

Build + static validation + no-model container validation ONLY. No model
weights loaded, no Ray, no serving, no benchmark. NOT runtime-validated.
A successful result authorizes only the next controlled Phase-2 runtime
diagnostic. v0.25.1 migration remains BLOCKED (Phase-2 b12x/UVM regression).

## Provenance and lineage
- Repo: ~/docker/vllm-spark, main@fd61254, tracked CLEAN before and after;
  git diff --check clean. No commit/push/stage. ahead/behind origin: 0/0.
- Disk preflight: 21 GiB free (98%); decision PROCEED (thin patch-only layer,
  no compile; actual image delta ~101 kB across 3 layers).
- Base: r2 image vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rebased-
  ecfix-upstream-r2-exp = sha256:7e282758328c7dab8fea50b3fcac18ae0eb9f9e05b661f
  208c9ca4bb1c37d29c (identical both nodes; vLLM 0.25.1@752a3a504485,
  FlashInfer 0.6.15, NGC 26.05 torch 2.12.0a0, transformers 5.10.2,
  triton 3.7.0, compressed-tensors 0.15.0.1, CUTLASS DSL 4.5.2,
  safetensors 0.8.0, upstage 00907fc overlay rebased).
- C3 lineage: raw-g1 contract fix baked from permanent-fix-design.md exactly
  as validated (C3 offline conformance bit-identical + 16/16 marlin +
  pread-complete runtime diagnostic). v0.25-line-only form (no capability
  branch needed inside a v0.25.1-only image).

## New image
- Tag: vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rawg1-pread-r3-exp
- ID (BOTH nodes, full match):
  sha256:001dcd2fb66d42889e292426d0c9c52f519bf7e6de1fea7f63883f1bcfcdf425
- arm64/linux, 31.2 GB (r2 + ~101 kB), created 2026-07-26T00:32Z.
- Transfer: docker save | ssh 10.10.10.2 docker load (5m42s), no rebuild on
  spark02. In-image canonical patches at /opt/spark-patches/r3/.

## Changes (the ONLY two source deltas vs r2)
1. vllm/model_executor/models/solar_open2.py
   4e5c907155a9400a... -> 56b487479c06a4c2...
   patches/solar/solar-open2-rawg1-contract-v025.patch (19 lines):
   model-side fused_kda_gate + unsqueeze removed; RAW g1 passed as
   rearrange(g1, "n (h d) -> 1 n h d", d=head_dim). No other change.
2. vllm/model_executor/model_loader/weight_utils.py
   ad98e4040aa78fe1... -> de48cbefb218b5a0...
   patches/solar/vllm-safetensors-pread-env-gate.patch (97 lines):
   VLLM_SPARK_ST_PREAD env gate; strict value parsing (true: 1/true/yes/on;
   false: unset/empty/0/false/no/off; else ValueError); capability check via
   packaging.version (safetensors>=0.8.0), RuntimeError fail-fast BEFORE any
   checkpoint open, NO silent fallback; single INFO activation log per
   process; kwargs applied ONLY to the lazy-branch safe_open. torchao/eager
   branches and all other loaders untouched (test-asserted).
   Rationale for a local strict parser: vllm.envs has no validated
   fail-on-invalid boolean parser (bool(int()) or in-("1","true") only).
- Patch application in build: canonical patches applied externally
  (patch -p1 --fuzz=0, dry-run verified), patched files COPYed with
  sha256sum -c of BOTH pre- and post-state in the RUN step (r2 ships no
  patch binary; nothing was installed into the image). Patch order recorded:
  rawg1 -> pread gate (independent files). Stale .pyc for both modules
  removed; AST parse gate in build.

## Tests (16/16 PASSED, disposable container, pytest installed ephemerally —
image NOT modified; upstream test framework absent from runtime image)
patches/solar/tests/test_st_pread_gate.py: unset default (no backend kwarg);
false forms 0/false/no/off/empty; true forms 1/true/yes/on (backend="pread"
exactly once); invalid "maybe" (ValueError, value included, zero loader
calls); unsupported runtime (mocked safetensors 0.7.0 -> RuntimeError, zero
loader calls, no fallback); supported succeeds; loader isolation (5 other
iterators contain no gate reference); patched call contract source
assertions; mmap-vs-pread bit-identical tensors.

## No-model validation (spark01 AND spark02, disposable containers)
- default / 0 / false / off: pread disabled, mmap default contract intact,
  synthetic tensors correct (values/dtype/shape), rc=0.
- VLLM_SPARK_ST_PREAD=1: backend=pread selected, exactly one INFO activation
  line across two iterator passes, synthetic tensors correct, rc=0.
- invalid ("maybe"): rc=1, ValueError listing accepted values. No fallback.
- unsupported: covered by mocked unit test (0.7.0 -> RuntimeError).
- maps probe: mmap arm maps the synthetic file (1 mapping), pread arm maps
  NOTHING (0 mappings) while producing identical tensors.
- Solar imports: solar_open2 module imports; raw-g1 present (0 model-side
  fused_kda_gate(g1..) calls, rearrange form present); solar_open2 reasoning
  parser + tool parser (vllm.tool_parsers) + logits processor import;
  compressed-tensors 0.15.0.1; flashinfer 0.6.15; FLASHINFER_B12X present in
  the NvFp4 MoE oracle. No CUDA kernels initialized.

## Repo artifacts (ALL untracked; synced to spark02 repo, hashes verified)
- dockerfiles/active/Dockerfile.solar-open2-nvfp4-v0251-rawg1-pread-r3-exp
- patches/solar/solar-open2-rawg1-contract-v025.patch (f252383c...)
- patches/solar/vllm-safetensors-pread-env-gate.patch (39b26822...)
- patches/solar/tests/test_st_pread_gate.py
- docs/solar-open2-st-pread.md
- docker-compose.pread-r3.yml (forwards the env var; base compose untouched)
- presets/solar-open2-250b-nota-nvfp4-v0251-r3-pread-kv4g-exp-tp2.env
  (disposable; = kv4g diag preset + r3 image tag + VLLM_SPARK_ST_PREAD=1
  with experimental comment; KV 4GiB/rank, eager, all C3 runtime values kept)
- No production/promoted preset, no v0.22.1 preset, no OWUI/Traefik change.
- No image deleted or overwritten (r1/r2/C3-era images + 0.22.1 baseline intact).

## Remaining runtime blockers (unchanged by this task)
Phase-2 b12x autotune/JIT + warmup anon/UVM pressure (s01 warmup swap
~6.6 GB, decode ~0.3 t/s in the pread-complete diagnostic); b12x conformance
decision (14/16 vs marlin); acceptance benchmark. Next authorized step:
controlled Phase-2 runtime diagnostic using this r3 image.
