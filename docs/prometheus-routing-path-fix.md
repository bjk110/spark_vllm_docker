# Prometheus routing.py `.path` guard (EXPERIMENTAL image)

Deterministic, guarded, build-time replacement for the ephemeral runtime sed
that `benchmarks/bench-bt-matrix-step37-v023.sh::apply_prometheus_patch` applied
to the running head container.

**Status: EXPERIMENTAL. Not production, not promoted.**

## Problem

`prometheus-fastapi-instrumentator==8.0.0` derives a metric label on every HTTP
request via `routing._get_route_name()`, which does `route_name = route.path`
unconditionally. Under this stack:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| FastAPI | 0.137.0 |
| Starlette | 1.3.1 |
| prometheus-fastapi-instrumentator | 8.0.0 |
| vLLM | 0.23.0 |

vLLM 0.23.0 registers its OpenAI routes with FastAPI `include_router()`, which
inserts a top-level `fastapi.routing._IncludedRouter` into `app.routes`. That
object has **no `.path` attribute**, so `_get_route_name()` raises inside the
metrics middleware before the request handler runs — every endpoint (including
`/health` and `/metrics`) returns **HTTP 500**.

Observed traceback (tail):

```
File ".../prometheus_fastapi_instrumentator/middleware.py", line 240, in _get_handler
  route_name = routing.get_route_name(request)
File ".../prometheus_fastapi_instrumentator/routing.py", line 55, in _get_route_name
  route_name = route.path
AttributeError: '_IncludedRouter' object has no attribute 'path'
```

## Selected solution — guarded source patch (Option B)

Two occurrences of `route_name = route.path` in `_get_route_name()` (the
`Match.FULL` branch and the `Match.PARTIAL` branch) become:

```python
route_name = getattr(route, 'path', 'unknown')
```

Routes exposing `.path` keep their label; path-less routes get the bounded
fallback `'unknown'`. Control flow and return values are unchanged (a
partial-only match still returns `None`). Single quotes are intentional — they
reproduce the validated patched SHA256.

| | SHA256 |
|---|---|
| original `routing.py` | `b90d08f601c5ec82245630667c0cbc031f00df038284b4e61f46945d182c85fb` |
| patched `routing.py` | `a3addfd90d1132a5ab5dca54c788f4743fe180b9607a662bf34ef0453750848c` |
| statements replaced | 2 |

### Rejected alternative — dependency pin (Option A)

Pinning a different `prometheus-fastapi-instrumentator` version was rejected:
the package sits in the vLLM 0.23.0 / NGC 26.05 pinned dependency set; bumping
it risks transitive changes to FastAPI/Starlette/vLLM behavior, has no
demonstrated compatible lock, and cannot be justified without broad dependency
testing outside this task's scope. The source patch has the smaller regression
surface — one function, one file, zero dependency-graph change, semantics
proven preserved by the offline regression test.

## Build-time guards

`patches/common/patch_prometheus_routing_path.py` refuses to run unless:

- package version is exactly `8.0.0`
- target file exists
- original SHA256 matches `b90d08f6…`
- the exact statement `route_name = route.path` appears exactly twice
- the `def _get_route_name(` context is present
- the guarded replacement is not already present

After applying it verifies: the vulnerable statement is gone, the replacement
appears exactly twice, the full patched SHA256 matches `a3addfd9…`, and
`py_compile` succeeds — exiting non-zero (failing the Docker build) on any
mismatch. Idempotent (re-run on a patched file exits 0).

## Validation

`patches/common/validate_prometheus_routing_path.py` (offline; no GPU, no
model) runs in the build and verifies: exact version, patched hash, module
import, original-failure reproduction, patched-success reproduction, normal
route-label preservation, and that unrelated exceptions are not suppressed.

Inspection command (re-runnable in any container from the image):

```bash
python3 /usr/local/bin/validate_prometheus_routing_path.py --inspect
# prints package version, target path, patched sha256, patch_status
python3 /usr/local/bin/validate_prometheus_routing_path.py
# runs the 7-check regression, exit 0 on success
```

## Image

- Dockerfile: `dockerfiles/active/Dockerfile.step3p7-v023-promfix`
- Base (unmodified): `vllm-spark:v023-d568-ngc2605-tx5102-vllm023-step3p7-relax`
  (`sha256:c272907c3761…`)
- Experimental tag: `vllm-spark:v023-d568-ngc2605-tx5102-vllm023-step3p7-relax-promfix-exp`
- Full image ID: `sha256:6db9acfd86965f13fb8cdb407b6c8264bc58a9c933ffe1a981c161af9eb84430`
  (identical on spark01 and spark02; in-image routing.py SHA256 `a3addfd9…` on both)
- Validated 2026-06-20: full dual-node FP8 regression PASS — /health 200, /metrics
  200, OpenAI endpoints 200, zero `_IncludedRouter` 500, backend selection
  unchanged (fp8 / TRITON Fp8 MoE / TRITON_ATTN), 6/6 correctness, perf within
  ±15% of baseline (prefill -4.3%, decode -11.0% 3-run, TTFT +4.5%).
- Changes exactly one file vs base (`routing.py`); everything else inherited.

## Rollback

This image is additive and isolated:

- Roll back by using the base tag
  `vllm-spark:v023-d568-ngc2605-tx5102-vllm023-step3p7-relax` again — it is
  untouched.
- Or re-apply the ephemeral runtime patch via `apply_prometheus_patch()`.
- No base image, NVFP4 asset, preset, or model file is modified by this work.

## Known limitations

- **Version-pinned guards.** The patch refuses to apply unless
  `prometheus-fastapi-instrumentator==8.0.0` and the original `routing.py`
  SHA256 matches `b90d08f6…`. This is intentional (it prevents silently
  patching an unexpected file), but it means any future base-image bump that
  changes that package or file will fail the build until the preimage hash and
  expected version in `patches/common/patch_prometheus_routing_path.py` are
  re-validated and updated.
- **Upstream scope.** This is a local compatibility shim for the
  FastAPI 0.137 / Starlette 1.3.1 `_IncludedRouter`-without-`.path` behavior, not
  an upstream fix. If `prometheus-fastapi-instrumentator` later ships its own
  guard, this patch becomes redundant and the build guard will need updating.
- **Experimental only.** Not a production or promoted image.
- **GB10 UMA, unrelated to this patch.** A full FP8 run still leaves
  ~59–85 GiB of UVM/driver-retained memory per node after shutdown; a reboot is
  required before the next large model load. This is a platform property, not a
  consequence of this patch.

## Provenance (immutable publication)

- Git commit: _to be recorded after commit (see follow-up provenance commit)_
- GHCR immutable tag: _to be recorded after push_
- GHCR manifest digest: _to be recorded after push_
- Local validated image ID: `sha256:6db9acfd86965f13fb8cdb407b6c8264bc58a9c933ffe1a981c161af9eb84430`
