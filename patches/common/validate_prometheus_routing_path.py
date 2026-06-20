#!/usr/bin/env python3
"""
Build-time validation + inspection for the prometheus-fastapi-instrumentator
routing.py `.path` guard (see patch_prometheus_routing_path.py).

Runs entirely offline: no GPU, no vLLM engine, no model weights.

Default mode (no args): full validation, exit non-zero on any failure.
  1. package version is exactly the expected version
  2. installed routing.py SHA256 equals the validated patched hash
  3. the routing module imports
  4. ORIGINAL failure reproduction: the pre-patch statement (`route.path`)
     raises AttributeError on a route object that lacks `.path`
  5. PATCHED reproduction success: the real installed `_get_route_name`
     returns the bounded fallback label for a path-less route (no raise)
  6. normal routes that expose `.path` keep their original label
  7. the guard does not suppress unrelated exceptions

--inspect mode: prints package version, target path, patched SHA256, status.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import sys
import traceback
from pathlib import Path

EXPECTED_PKG = "prometheus-fastapi-instrumentator"
EXPECTED_VERSION = "8.0.0"
TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "prometheus_fastapi_instrumentator/routing.py"
)
SHA_ORIGINAL = "b90d08f601c5ec82245630667c0cbc031f00df038284b4e61f46945d182c85fb"
SHA_PATCHED = "a3addfd90d1132a5ab5dca54c788f4743fe180b9607a662bf34ef0453750848c"
SCOPE = {"type": "http", "path": "/x", "method": "GET"}


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _status() -> str:
    if not TARGET.is_file():
        return "MISSING"
    sha = _sha256_file(TARGET)
    if sha == SHA_PATCHED:
        return "patched"
    if sha == SHA_ORIGINAL:
        return "original"
    return "unknown"


def inspect() -> int:
    try:
        ver = md.version(EXPECTED_PKG)
    except md.PackageNotFoundError:
        ver = "not-installed"
    print(f"package: {EXPECTED_PKG}=={ver}")
    print(f"target: {TARGET}")
    print(f"sha256: {_sha256_file(TARGET) if TARGET.is_file() else 'n/a'}")
    print(f"patch_status: {_status()}")
    return 0


def validate() -> int:
    failures = []

    # 1. version
    ver = md.version(EXPECTED_PKG)
    if ver != EXPECTED_VERSION:
        failures.append(f"version {ver} != {EXPECTED_VERSION}")

    # 2. patched hash
    sha = _sha256_file(TARGET)
    if sha != SHA_PATCHED:
        failures.append(f"sha256 {sha} != patched {SHA_PATCHED}")

    # 3. import
    from starlette.routing import Match, Mount  # noqa: F401
    import prometheus_fastapi_instrumentator.routing as R

    class FullNoPath:
        def matches(self, scope):
            return (Match.FULL, {})

    class NormalRoute:
        path = "/healthz"
        def matches(self, scope):
            return (Match.FULL, {})

    class RaisingRoute:
        path = "/boom"
        def matches(self, scope):
            raise RuntimeError("unrelated-boom")

    # 4. ORIGINAL failure reproduction (inline, replicates pre-patch statement)
    def original_line(route):
        return route.path  # the exact pre-patch access
    try:
        original_line(FullNoPath())
        failures.append("original-repro did not raise")
    except AttributeError:
        pass

    # 5. PATCHED reproduction success via the real installed function
    try:
        r = R._get_route_name(SCOPE, [FullNoPath()])
        if r != "unknown":
            failures.append(f"patched fallback label {r!r} != 'unknown'")
    except Exception:
        failures.append("patched _get_route_name raised on path-less route:\n"
                        + traceback.format_exc())

    # 6. normal route label preserved
    try:
        r = R._get_route_name(SCOPE, [NormalRoute()])
        if r != "/healthz":
            failures.append(f"normal route label {r!r} != '/healthz'")
    except Exception:
        failures.append("patched _get_route_name raised on normal route:\n"
                        + traceback.format_exc())

    # 7. unrelated exception not suppressed
    try:
        R._get_route_name(SCOPE, [RaisingRoute()])
        failures.append("unrelated exception was suppressed")
    except RuntimeError:
        pass

    if failures:
        print("[validate_prometheus_routing_path] FAIL:")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"[validate_prometheus_routing_path] OK "
          f"({EXPECTED_PKG}=={ver}, sha256={sha}, status={_status()}, "
          "all 7 checks passed)")
    return 0


if __name__ == "__main__":
    if "--inspect" in sys.argv[1:]:
        sys.exit(inspect())
    sys.exit(validate())
