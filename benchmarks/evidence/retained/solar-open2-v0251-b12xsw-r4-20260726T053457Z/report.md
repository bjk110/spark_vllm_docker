# Solar-Open2 v0.25.1 — r4 experimental image (b12x shared-workspace env gate)
Arc: solar-open2-v0251-b12xsw-r4-20260726T053457Z

## FINAL STATE: SOLAR_OPEN2_V0251_B12XSW_R4_RUNTIME_VALIDATED

r4 converts the validated shared-workspace overlay into an immutable,
env-gated image and passes the full controlled runtime smoke. It is an
experimental memory-safe baseline ONLY — no promotion, no acceptance
benchmark; decode regression unresolved; migration remains BLOCKED.

## Image
- Tag: vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rawg1-pread-b12xsw-r4-exp
- ID (BOTH nodes, full match): sha256:ecb7bfe3978a5241c5c304d52ce91e061e22b750178d21a4ef7788a08e86e774
- Base r3 (001dcd2fb66d...) + ONE source delta:
  flashinfer/fused_moe/cute_dsl/b12x_moe.py bcac8067... -> 38c253a3...
  (patches/solar/vllm-flashinfer-b12x-shared-workspace-env-gate.patch,
  302 lines; SHA-verified pre/post in build; AST gate; stale pyc removed;
  canonical patch baked at /opt/spark-patches/r4/). arm64, +~78 kB.
- Cumulative patch order: (1) Solar 00907fc overlay [r2] -> (2) C3 raw-g1
  [r3] -> (3) pread env gate [r3] -> (4) b12x shared-workspace gate [r4].
- No package changes (stack identical to r3), no instrumentation/pytest
  baked, no bind mount required, disk delta negligible (20 GB free, safe).

## VLLM_SPARK_B12X_SHARED_WORKSPACE contract (same style as ST_PREAD gate)
- OFF (unset/empty/0/false/no/off): original per-wrapper allocation path,
  no manager, no logs — verified per-layer distinct storages.
- ON (1/true/yes/on): bounded shared manager (process/rank/device-local,
  exact-size keys, max 8 entries fail-loud, no resize by design, no
  layer-local fallback); one activation log per worker; per-entry log with
  key+bytes+entry-count+stream. Fail-fast on CUDA graph capture at both
  allocation and run() (no silent fallback). Layer weight state
  (_weight_views/_padded_weights) never shared.
- Invalid: ValueError naming the value + accepted values, nonzero exit.

## Validation results
- Unit/synthetic (r4 container, GPU, ephemeral pytest): **23/23 PASSED**
  (parsing x3 classes, invalid, per-layer regression, identity sharing,
  incompatible-key separation, entry-limit fail-loud, no-alias across
  sizes, weight-state isolation, capture guard at alloc AND run,
  non-b12x isolation, process-local reload, single activation log).
- No-model (spark01 + spark02): OFF ok / ON ok (2 entries small config) /
  invalid rc=1 with clear message / combined ST_PREAD+B12XSW coexist /
  Solar imports + raw-g1 + parsers + logits + FLASHINFER_B12X + gate all ok.
- spark02 sync: docker save|load, ID identical, arch arm64, OFF/ON pass.

## Runtime smoke (controlled: reboots, clean 117.6 GiB baselines, warm CW
caches kernel-artifacts verified unchanged [flashinfer_jit.log excluded as
log-only], r4 preset = r3 KV4G preset + r4 tag + B12XSW=1, guards ma
5.0/4.0 GiB + swap s01 7.9 / s02 4.0 GiB hard as before)
| acceptance gate | target | measured | verdict |
|---|---|---|---|
| torch alloc @engine-ready | <= ~80 GiB | **76.24 GiB** | PASS |
| dummy-run persistent delta | <= ~3 GiB | warmup _dummy_run deltas ~0.05 GiB (init profile phase settles at 76.2 incl. KV4G+shared set) | PASS |
| peak swap (both nodes) | < 1.0 GB | **s01 344 kB / s02 340 kB** | PASS |
| min MemAvailable | > 4.0 GiB floor | ~22 GiB (no soft crossing at all) | PASS |
| shared entries | bounded, expected | exactly 3/rank: ws_static 451.6 MB + ws_dynamic 126.2 MB + moe_output 16 MB (~0.58 GiB) | PASS |
| layer-local fallback | 0 | 0 (6 total shared allocs, no re-alloc during requests) | PASS |
| C3 correctness | = b12x r3/SW baseline | **bit-identical tokens** (6th consecutive), 14/16 vs marlin, first div pos 14, no garble | PASS |
| Korean decode | >= 3.2 t/s | 120 tok / 37.2 s = **3.23 t/s** | PASS |
| 5-min idle | stable, health x10 | ma flat ~21.8 GiB, swap flat, 10/10 health 200 | PASS |
| second request | coherent | 40 tok / 30.5 s, coherent | PASS |
- Telemetry: usercustomize boundary sampler (lightweight; heavy C2
  instrumentation NOT used). No NVRM/UVM/OOM/Ray events; hosts responsive;
  clean teardown; port 8000 free; warm caches preserved.

## State decision (Phase 17)
r4 retained as immutable experimental baseline; r3 preserved as rollback;
overlay evidence preserved; READY for the decode-performance isolation arc.
No promotion, no acceptance benchmark, v0.22.1 untouched.

## Remaining migration blockers
1. Decode ~3.2-3.3 t/s vs 12.8 t/s 0.22.1 baseline (next arc).
2. b12x conformance rule (stable 14/16; late-token run nondeterminism).
3. Matched acceptance benchmark.

## Files
- Repo (untracked, both nodes): Dockerfile r4, patch, tests/test_b12x_
  shared_workspace.py, docs/solar-open2-b12x-shared-workspace.md,
  docker-compose.b12xsw-r4.yml, presets/...v0251-r4-b12xsw-kv4g-exp-tp2.env.
- Tracked state clean on spark01/spark02/homeserver; git diff --check clean;
  nothing committed/pushed/staged.
