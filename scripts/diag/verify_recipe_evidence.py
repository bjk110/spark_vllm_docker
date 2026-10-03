#!/usr/bin/env python3
"""Verify curated recipe evidence and the external-archive hash ledger."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


MANIFEST_PATH = Path("benchmarks/evidence/manifest.json")
ALLOWED_STATUS_CLASSES = {
    "production",
    "rollback",
    "production-qualified",
    "validated-non-production",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files(root):
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.split("\0")) - {""}


def safe_relative(value):
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def catalog_scope(root):
    text = (root / "presets/README.md").read_text()
    section = None
    presets = set()
    for line in text.splitlines():
        if line.startswith("## 1."):
            section = 1
        elif line.startswith("## 2."):
            section = 2
        elif line.startswith("## "):
            section = None
        if section in (1, 2) and line.startswith("|"):
            for target in re.findall(r"\]\(([^)#]+\.env)\)", line):
                presets.add((Path("presets") / Path(target).name).as_posix())
    return presets


def preset_status_class(path):
    match = re.search(r"^#\s*Status:\s*(.+)$", path.read_text(), re.MULTILINE)
    if not match:
        return None
    status = match.group(1).strip().lower()
    if status.startswith("active production"):
        return "production"
    if "rollback" in status:
        return "rollback"
    if status.startswith(("production-qualified", "optional validated profile",
                          "main_merge_approved_opt_in")):
        return "production-qualified"
    if status.startswith("validated"):
        return "validated-non-production"
    return None


def parse_archive_manifest(path, errors):
    entries = []
    if not path.is_file():
        errors.append(f"missing archive manifest: {path}")
        return entries
    lines = path.read_text().splitlines()
    for number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or not safe_relative(match.group(2)):
            errors.append(f"archive manifest malformed at line {number}")
            continue
        entries.append((match.group(2), match.group(1)))
    paths = [item[0] for item in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        errors.append("archive manifest paths must be sorted and unique")
    return entries


def verify(root, expected_count=14, archive_root=None):
    root = Path(root).resolve()
    tracked = tracked_files(root)
    errors = []
    manifest_rel = MANIFEST_PATH.as_posix()
    if manifest_rel not in tracked or not (root / MANIFEST_PATH).is_file():
        return [f"missing tracked evidence manifest: {manifest_rel}"]
    try:
        manifest = json.loads((root / MANIFEST_PATH).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid evidence manifest: {exc}"]

    if manifest.get("schema_version") != 1:
        errors.append("unsupported evidence manifest schema_version")
    recipes = manifest.get("recipes")
    if not isinstance(recipes, list):
        return errors + ["recipes must be a list"]
    if manifest.get("recipe_count") != len(recipes) or len(recipes) != expected_count:
        errors.append(
            f"recipe count mismatch: manifest={manifest.get('recipe_count')} "
            f"entries={len(recipes)} expected={expected_count}"
        )

    catalog = catalog_scope(root)
    listed = {item.get("preset") for item in recipes if isinstance(item, dict)}
    if listed != catalog:
        errors.append(
            "catalog scope mismatch: "
            f"missing={sorted(catalog - listed)} extra={sorted(listed - catalog)}"
        )
    if len(listed) != len(recipes):
        errors.append("recipe presets must be unique")

    for item in recipes:
        if not isinstance(item, dict):
            errors.append("recipe entry must be an object")
            continue
        preset = item.get("preset", "")
        if not safe_relative(preset) or preset not in tracked:
            errors.append(f"recipe preset is not a safe tracked path: {preset}")
        if item.get("status_class") not in ALLOWED_STATUS_CLASSES:
            errors.append(f"invalid status_class for {preset}: {item.get('status_class')}")
        elif safe_relative(preset) and preset in tracked:
            expected_status = preset_status_class(root / preset)
            if expected_status != item.get("status_class"):
                errors.append(
                    f"status_class disagrees with preset metadata for {preset}: "
                    f"{item.get('status_class')} != {expected_status}"
                )
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"recipe evidence must be a non-empty list: {preset}")
        else:
            for evidence_path in evidence:
                if not safe_relative(evidence_path) or evidence_path not in tracked:
                    errors.append(f"evidence is not a safe tracked path for {preset}: {evidence_path}")
        limits = item.get("known_limits")
        if not isinstance(limits, list) or not limits:
            errors.append(f"known_limits must be explicit for {preset}")

    retained = manifest.get("retained_files")
    if not isinstance(retained, dict):
        errors.append("retained_files must be an object")
        retained = {}
    retained_prefix = "benchmarks/evidence/retained/"
    tracked_retained = {path for path in tracked if path.startswith(retained_prefix)}
    if set(retained) != tracked_retained:
        errors.append(
            "retained_files coverage mismatch: "
            f"missing={sorted(tracked_retained - set(retained))} "
            f"extra={sorted(set(retained) - tracked_retained)}"
        )
    referenced_evidence = {
        path
        for item in recipes if isinstance(item, dict)
        for path in (item.get("evidence") if isinstance(item.get("evidence"), list) else [])
    }
    unreferenced = set(retained) - referenced_evidence
    if unreferenced:
        errors.append(f"retained files not referenced by a recipe: {sorted(unreferenced)}")
    for rel, expected_sha in retained.items():
        if not safe_relative(rel) or rel not in tracked or not (root / rel).is_file():
            errors.append(f"retained file is not a safe tracked path: {rel}")
            continue
        actual = sha256(root / rel)
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha)) or actual != expected_sha:
            errors.append(f"retained SHA256 mismatch: {rel}")

    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        return errors + ["archive must be an object"]
    archive_manifest_rel = archive.get("manifest", "")
    if not safe_relative(archive_manifest_rel) or archive_manifest_rel not in tracked:
        errors.append(f"archive manifest is not a safe tracked path: {archive_manifest_rel}")
        entries = []
    else:
        entries = parse_archive_manifest(root / archive_manifest_rel, errors)
    if archive.get("file_count") != len(entries):
        errors.append(
            f"archive file_count mismatch: {archive.get('file_count')} != {len(entries)}"
        )

    if archive_root is not None:
        archive_root = Path(archive_root).resolve()
        total_bytes = 0
        for rel, expected_sha in entries:
            path = archive_root / rel
            try:
                path.resolve().relative_to(archive_root)
            except ValueError:
                errors.append(f"archive path escapes root: {rel}")
                continue
            if not path.is_file():
                errors.append(f"archive file missing: {rel}")
                continue
            total_bytes += path.stat().st_size
            if sha256(path) != expected_sha:
                errors.append(f"archive SHA256 mismatch: {rel}")
        if archive.get("total_bytes") != total_bytes:
            errors.append(
                f"archive total_bytes mismatch: {archive.get('total_bytes')} != {total_bytes}"
            )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--expected-count", type=int, default=14)
    args = parser.parse_args()
    errors = verify(args.root, args.expected_count, args.archive_root)
    for error in errors:
        print(f"[FAIL] {error}")
    if errors:
        print(f"FAIL: {len(errors)} recipe-evidence error(s)")
        return 1
    print(f"PASS: {args.expected_count} recipes; curated evidence and archive ledger valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
