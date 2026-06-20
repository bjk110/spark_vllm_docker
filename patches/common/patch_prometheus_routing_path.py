#!/usr/bin/env python3
"""
Patch prometheus-fastapi-instrumentator 8.0.0 routing.py for FastAPI 0.137 /
Starlette 1.3.1 route objects that lack a `.path` attribute.

Why this patch exists
---------------------
vLLM 0.23.0 registers its OpenAI API routes through FastAPI's
`include_router()`, which (since FastAPI 0.137.0 / Starlette 1.3.1) inserts a
top-level `fastapi.routing._IncludedRouter` object into `app.routes`. That
object has no `.path` attribute. prometheus-fastapi-instrumentator's metrics
middleware calls `routing._get_route_name()` on EVERY HTTP request to derive a
metric label; the function does `route_name = route.path` unconditionally, so
it raises `AttributeError: '_IncludedRouter' object has no attribute 'path'`
inside the middleware before the request handler runs. The result is HTTP 500
on every endpoint, including `/health` and `/metrics`.

Observed failing stack (head API server):
    prometheus_fastapi_instrumentator/middleware.py _get_handler
    -> routing.get_route_name -> _get_route_name
    -> route_name = route.path
    -> AttributeError: '_IncludedRouter' object has no attribute 'path'

Fix
---
The two occurrences of `route_name = route.path` in `_get_route_name()` become
`route_name = getattr(route, 'path', 'unknown')`. Routes that expose `.path`
keep their existing label; path-less routes get the bounded fallback label
`'unknown'`. No control flow or return semantics change (a partial-only match
still returns None). This is the exact transformation validated at runtime on
the dual-Spark FP8 baseline (2026-06-20).

Determinism / guards
--------------------
This script refuses to run unless every precondition matches the validated
evidence (package version, target path, original SHA256, exact vulnerable
statement count, function context, not-already-patched). After applying it
re-verifies the post-image SHA256 and a py_compile, then exits non-zero on any
mismatch so the Docker build fails loudly instead of shipping a wrong file.

Idempotent: re-running on an already-patched file prints a notice and exits 0.

Single quotes in the replacement are intentional — they reproduce the exact
validated patched SHA256.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import py_compile
import sys
from pathlib import Path

EXPECTED_PKG = "prometheus-fastapi-instrumentator"
EXPECTED_VERSION = "8.0.0"
TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "prometheus_fastapi_instrumentator/routing.py"
)
SHA_ORIGINAL = "b90d08f601c5ec82245630667c0cbc031f00df038284b4e61f46945d182c85fb"
SHA_PATCHED = "a3addfd90d1132a5ab5dca54c788f4743fe180b9607a662bf34ef0453750848c"

VULNERABLE = "route_name = route.path"
REPLACEMENT = "route_name = getattr(route, 'path', 'unknown')"
EXPECTED_COUNT = 2
FUNC_CONTEXT = "def _get_route_name("


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[patch_prometheus_routing_path] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    # --- precondition: package version ---
    try:
        ver = md.version(EXPECTED_PKG)
    except md.PackageNotFoundError:
        fail(f"package {EXPECTED_PKG} not installed")
    if ver != EXPECTED_VERSION:
        fail(f"version mismatch: found {ver}, expected {EXPECTED_VERSION}")

    # --- precondition: target file exists ---
    if not TARGET.is_file():
        fail(f"target file not found: {TARGET}")

    raw = TARGET.read_text()
    sha = _sha256(raw)

    # --- idempotency: already patched ---
    if sha == SHA_PATCHED:
        if raw.count(REPLACEMENT) == EXPECTED_COUNT and raw.count(VULNERABLE) == 0:
            print("[patch_prometheus_routing_path] already patched "
                  f"(sha256={sha}); nothing to do")
            return 0
        fail(f"sha256 matches patched ({sha}) but content guards disagree")

    # --- precondition: original SHA256 ---
    if sha != SHA_ORIGINAL:
        fail(f"original SHA256 mismatch: found {sha}, expected {SHA_ORIGINAL}")

    # --- precondition: function context present ---
    if FUNC_CONTEXT not in raw:
        fail(f"expected function context {FUNC_CONTEXT!r} not found")

    # --- precondition: vulnerable statement appears exactly twice ---
    count = raw.count(VULNERABLE)
    if count != EXPECTED_COUNT:
        fail(f"vulnerable statement count {count}, expected {EXPECTED_COUNT}")

    # --- precondition: replacement not already present ---
    if REPLACEMENT in raw:
        fail("replacement statement unexpectedly already present")

    # --- apply: only the two intended changes ---
    patched = raw.replace(VULNERABLE, REPLACEMENT)

    # --- postcondition: vulnerable statement no longer present ---
    # (REPLACEMENT contains 'route.path' as a substring via getattr text? No —
    #  getattr(route, 'path', ...) has no literal "route.path", so a plain
    #  count of the exact VULNERABLE string is correct.)
    if patched.count(VULNERABLE) != 0:
        fail("vulnerable statement still present after patch")
    # --- postcondition: replacement appears exactly twice ---
    if patched.count(REPLACEMENT) != EXPECTED_COUNT:
        fail(f"replacement count {patched.count(REPLACEMENT)}, expected {EXPECTED_COUNT}")
    # --- postcondition: full SHA256 ---
    new_sha = _sha256(patched)
    if new_sha != SHA_PATCHED:
        fail(f"post-image SHA256 mismatch: got {new_sha}, expected {SHA_PATCHED}")

    TARGET.write_text(patched)

    # --- postcondition: py_compile ---
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as e:
        fail(f"py_compile failed after patch: {e}")

    print("[patch_prometheus_routing_path] patched OK "
          f"({EXPECTED_PKG}=={ver}, sha256 {SHA_ORIGINAL[:8]} -> {new_sha[:8]}, "
          f"{EXPECTED_COUNT} statements replaced)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
