"""Tests for the canonical recipe-evidence manifest verifier."""
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "recipe_evidence", Path(__file__).with_name("verify_recipe_evidence.py")
)
recipe_evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe_evidence)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "presets").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "benchmarks/evidence/retained").mkdir(parents=True)
    presets = ["prod.env", "validated.env"]
    for name in presets:
        status = "Active production" if name == "prod.env" else "Validated (non-production)"
        (tmp_path / "presets" / name).write_text(f"# Status: {status}\nMODEL={name}\n")
    (tmp_path / "presets/README.md").write_text(
        "# Catalog\n"
        "## 1. Production presets\n"
        "| Preset | Evidence |\n|---|---|\n"
        "| [prod.env](prod.env) | [evidence](../docs/prod.md) |\n"
        "## 2. Validated presets (non-production)\n"
        "| Preset | Evidence |\n|---|---|\n"
        "| [validated.env](validated.env) | [evidence](../docs/validated.md) |\n"
        "## 3. General supported presets\n"
    )
    for name in ("prod", "validated"):
        (tmp_path / "docs" / f"{name}.md").write_text(f"# {name}\n")
    retained = tmp_path / "benchmarks/evidence/retained/prod-summary.md"
    retained.write_text("# bounded evidence\n")
    archive_manifest = tmp_path / "benchmarks/evidence/archive-manifest.sha256"
    archive_manifest.write_text(f"{'a' * 64}  benchmarks/results/raw/log.txt\n")
    manifest = {
        "schema_version": 1,
        "scope": "catalog-sections-1-and-2",
        "recipe_count": 2,
        "recipes": [
            {
                "preset": "presets/prod.env",
                "status_class": "production",
                "evidence": ["docs/prod.md", "benchmarks/evidence/retained/prod-summary.md"],
                "known_limits": ["bounded test only"],
            },
            {
                "preset": "presets/validated.env",
                "status_class": "validated-non-production",
                "evidence": ["docs/validated.md"],
                "known_limits": ["not production"],
            },
        ],
        "retained_files": {
            "benchmarks/evidence/retained/prod-summary.md": sha256(retained)
        },
        "archive": {
            "manifest": "benchmarks/evidence/archive-manifest.sha256",
            "file_count": 1,
            "total_bytes": 7,
        },
    }
    (tmp_path / "benchmarks/evidence/manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    return tmp_path


def test_valid_manifest_passes(repo):
    assert recipe_evidence.verify(repo, expected_count=2) == []


def test_catalog_scope_must_match_exactly(repo):
    manifest = repo / "benchmarks/evidence/manifest.json"
    data = json.loads(manifest.read_text())
    data["recipes"].pop()
    data["recipe_count"] = 1
    manifest.write_text(json.dumps(data))
    errors = recipe_evidence.verify(repo, expected_count=2)
    assert any("catalog scope" in error for error in errors)


def test_status_class_must_match_preset_status(repo):
    manifest = repo / "benchmarks/evidence/manifest.json"
    data = json.loads(manifest.read_text())
    data["recipes"][0]["status_class"] = "rollback"
    manifest.write_text(json.dumps(data))
    errors = recipe_evidence.verify(repo, expected_count=2)
    assert any("status_class disagrees" in error for error in errors)


def test_retained_file_hash_must_match(repo):
    (repo / "benchmarks/evidence/retained/prod-summary.md").write_text("tampered\n")
    errors = recipe_evidence.verify(repo, expected_count=2)
    assert any("retained SHA256 mismatch" in error for error in errors)


def test_every_retained_file_must_be_hashed_and_referenced(repo):
    stray = repo / "benchmarks/evidence/retained/stray.md"
    stray.write_text("unindexed evidence\n")
    subprocess.run(["git", "-C", str(repo), "add", str(stray.relative_to(repo))], check=True)
    errors = recipe_evidence.verify(repo, expected_count=2)
    assert any("retained_files coverage" in error for error in errors)

    stray.unlink()
    subprocess.run(
        ["git", "-C", str(repo), "rm", "--cached", "-q", str(stray.relative_to(repo))],
        check=True,
    )
    manifest = repo / "benchmarks/evidence/manifest.json"
    data = json.loads(manifest.read_text())
    data["recipes"][0]["evidence"] = ["docs/prod.md"]
    manifest.write_text(json.dumps(data))
    errors = recipe_evidence.verify(repo, expected_count=2)
    assert any("not referenced by a recipe" in error for error in errors)


def test_archive_manifest_must_be_sorted_unique_and_well_formed(repo):
    path = repo / "benchmarks/evidence/archive-manifest.sha256"
    path.write_text(
        f"{'b' * 64}  z/file\n"
        f"{'a' * 64}  a/file\n"
        f"{'a' * 64}  a/file\n"
        "not-a-hash  broken\n"
    )
    errors = recipe_evidence.verify(repo, expected_count=2)
    assert any("archive manifest" in error for error in errors)


def test_optional_archive_verification_detects_missing_or_changed_files(repo, tmp_path):
    archive = tmp_path / "external-archive"
    (archive / "benchmarks/results/raw").mkdir(parents=True)
    raw = archive / "benchmarks/results/raw/log.txt"
    raw.write_text("content")
    digest = sha256(raw)
    manifest = repo / "benchmarks/evidence/archive-manifest.sha256"
    manifest.write_text(f"{digest}  benchmarks/results/raw/log.txt\n")
    data_path = repo / "benchmarks/evidence/manifest.json"
    data = json.loads(data_path.read_text())
    data["archive"]["total_bytes"] = raw.stat().st_size
    data_path.write_text(json.dumps(data))
    assert recipe_evidence.verify(repo, expected_count=2, archive_root=archive) == []
    raw.write_text("changed")
    errors = recipe_evidence.verify(repo, expected_count=2, archive_root=archive)
    assert any("archive SHA256 mismatch" in error for error in errors)


def test_experimental_recipe_does_not_expand_evidence_scope(repo):
    path = repo / 'presets/README.md'
    path.write_text(path.read_text() +
                    '## 4. Experimental presets\n| [vision.env](vision.env) | UNVALIDATED |\n')
    (repo / 'presets/vision.env').write_text('# Status: Experimental — UNVALIDATED\n')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    assert len(recipe_evidence.catalog_scope(repo)) == 2
    assert recipe_evidence.verify(repo, expected_count=2) == []
