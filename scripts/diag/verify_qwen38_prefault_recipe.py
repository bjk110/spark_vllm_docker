#!/usr/bin/env python3
"""Fail-closed verifier for the Qwen3.8 GB10 prefault recipe."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_ROOT_REL = Path("patches/qwen/prefault-pr58868")
OVERLAY_REL = Path("compose/qwen3.8-flash-next/docker-compose.prefault-pr58868.yml")
DOC_REL = Path("docs/qwen3.8-flash-next-gb10-prefault.md")
PRESET_REL = Path("presets/qwen3.8-flash-next-fp8-tp2-candidate.env")
BASE_OVERLAY_REL = Path("compose/qwen3.8-flash-next/docker-compose.candidate.yml")
BASE_COMPOSE_REL = Path("docker-compose.yml")
EXPECTED_FILES = (
    "model_executor/utils.py",
    "model_executor/parameter.py",
    "model_executor/layers/linear.py",
    "model_executor/layers/vocab_parallel_embedding.py",
    "model_executor/layers/fused_moe/routed_experts.py",
    "model_executor/model_loader/weight_utils.py",
)
DEST_PREFIX = "/usr/local/lib/python3.12/dist-packages/vllm/"
EXPECTED_IMAGE_ID = "sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a"
EXPECTED_UPSTREAM_COMMIT = "857f70df4b7f3d96e19dc2b7361b56130bbe79b1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    patch_root = root / PATCH_ROOT_REL
    manifest_path = patch_root / "MANIFEST.json"
    overlay_path = root / OVERLAY_REL
    doc_path = root / DOC_REL

    for path in (manifest_path, overlay_path, doc_path):
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(root)}")
    if errors:
        return errors

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid manifest: {exc}"]

    declared = manifest.get("files")
    expected_status = "VALIDATED_MECHANISM_PER_CHECKPOINT_QUALIFICATION_REQUIRED"
    if manifest.get("status") != expected_status:
        errors.append(
            f"manifest status {manifest.get('status')!r} does not match {expected_status!r}"
        )
    source_runtime = manifest.get("source_runtime") or {}
    if source_runtime.get("image_id") != EXPECTED_IMAGE_ID:
        errors.append(
            f"manifest image ID {source_runtime.get('image_id')!r} does not match {EXPECTED_IMAGE_ID!r}"
        )
    upstream = manifest.get("upstream") or {}
    if upstream.get("commit") != EXPECTED_UPSTREAM_COMMIT:
        errors.append(
            f"manifest upstream commit {upstream.get('commit')!r} does not match {EXPECTED_UPSTREAM_COMMIT!r}"
        )
    if not isinstance(declared, dict):
        errors.append("manifest files must be an object")
        declared = {}
    if set(declared) != set(EXPECTED_FILES):
        errors.append(
            f"manifest file set mismatch: got={sorted(declared)} expected={sorted(EXPECTED_FILES)}"
        )

    for rel in EXPECTED_FILES:
        path = patch_root / rel
        if not path.is_file():
            errors.append(f"missing patch source: {rel}")
            continue
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"syntax error in {rel}: {exc}")
        actual = sha256(path)
        if declared.get(rel) != actual:
            errors.append(f"sha256 mismatch for {rel}: {actual} != {declared.get(rel)}")

    utils = (patch_root / "model_executor/utils.py").read_text()
    for needle in (
        "def copy_weight_(",
        'src.device.type == "cpu"',
        "src.dtype == dst.dtype",
        "src.is_contiguous()",
        "_is_integrated_gpu(dst.device)",
        "flat[:: mmap.PAGESIZE].sum()",
        "return dst.copy_(src)",
    ):
        if needle not in utils:
            errors.append(f"prefault helper missing contract fragment: {needle}")
    if ".clone(" in utils:
        errors.append("prefault helper must not clone the source tensor")

    for rel in EXPECTED_FILES[1:]:
        text = (patch_root / rel).read_text()
        if "copy_weight_" not in text:
            errors.append(f"patched copy site does not reference copy_weight_: {rel}")

    overlay = overlay_path.read_text()
    root_expr = "${QWEN38_PREFAULT_ROOT:-./patches/qwen/prefault-pr58868}"
    for rel in EXPECTED_FILES:
        mount = f"{root_expr}/{rel}:{DEST_PREFIX}{rel}:ro"
        count = overlay.count(mount)
        if count != 2:
            errors.append(f"expected two read-only head/worker mounts for {rel}, found {count}")
    if overlay.count("volumes:") != 2:
        errors.append("overlay must define exactly one volumes block per head/worker role")

    doc = doc_path.read_text()
    for needle in (
        "VALIDATED MECHANISM",
        "per-checkpoint qualification",
        "Graph-NONE and KV8 are not universal",
        "47.567%",
        "1845.7",
        "PR #58868",
    ):
        if needle not in doc:
            errors.append(f"documentation missing required statement: {needle}")

    return errors


def verify_compose_render(root: Path = REPO_ROOT) -> list[str]:
    if shutil.which("docker") is None:
        return ["docker executable unavailable; compose render not verified"]
    errors: list[str] = []
    configured_root = Path(
        os.environ.get("QWEN38_PREFAULT_ROOT", str(root / PATCH_ROOT_REL))
    ).expanduser()
    if not configured_root.is_absolute():
        configured_root = root / configured_root
    configured_root = configured_root.resolve()
    try:
        manifest = json.loads((root / PATCH_ROOT_REL / "MANIFEST.json").read_text())
        declared = manifest["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return [f"cannot validate Compose mount sources against manifest: {exc}"]
    for rel in EXPECTED_FILES:
        source = configured_root / rel
        if not source.is_file():
            errors.append(f"mount source missing: {source}")
            continue
        actual = sha256(source)
        if actual != declared.get(rel):
            errors.append(
                f"mount source sha256 mismatch for {source}: {actual} != {declared.get(rel)}"
            )
    cmd_base = [
        "docker",
        "compose",
        "--env-file",
        str(root / PRESET_REL),
        "-f",
        str(root / BASE_COMPOSE_REL),
        "-f",
        str(root / BASE_OVERLAY_REL),
        "-f",
        str(root / OVERLAY_REL),
    ]
    for profile, service in (("head", "head"), ("worker", "worker")):
        proc = subprocess.run(
            [*cmd_base, "--profile", profile, "config", "--format", "json"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode:
            errors.append(f"compose render failed for {profile}: {proc.stderr.strip()}")
            continue
        try:
            rendered = json.loads(proc.stdout)["services"][service]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid compose render for {profile}: {exc}")
            continue
        volumes = rendered.get("volumes", [])
        mounts = {
            item.get("target"): (item.get("source"), item.get("read_only"))
            for item in volumes
            if isinstance(item, dict)
        }
        for rel in EXPECTED_FILES:
            target = DEST_PREFIX + rel
            expected_source = str(configured_root / rel)
            source, read_only = mounts.get(target, (None, None))
            if source != expected_source:
                errors.append(
                    f"{profile} mount source for {target} is {source!r}, expected {expected_source!r}"
                )
            if read_only is not True:
                errors.append(f"{profile} mount is not read-only: {target}")
    return errors


def main() -> int:
    errors = verify(REPO_ROOT)
    render_errors = verify_compose_render(REPO_ROOT)
    errors.extend(render_errors)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[PASS] Qwen3.8 GB10 prefault recipe static contract")
    print("[PASS] head/worker Compose render contains six read-only mounts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
