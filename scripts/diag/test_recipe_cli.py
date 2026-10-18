"""Behavior tests for the read-only/pull recipe CLI."""
import json
import importlib.machinery
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/recipe"
MANIFEST = json.loads((ROOT / "recipes/manifest.json").read_text())


def run(*args):
    return subprocess.run(
        [sys.executable, str(CLI), *args], cwd=ROOT, text=True, capture_output=True
    )


def recipe(distribution):
    return next(item for item in MANIFEST["recipes"]
                if item["image"]["distribution"] == distribution)


def test_list_filters_by_node_count_and_shows_distribution():
    cp = run("list", "--nodes", "1")
    assert cp.returncode == 0, cp.stderr
    lines = cp.stdout.splitlines()
    assert lines[0].startswith("ID\tNODES\tBACKEND\tDISTRIBUTION")
    assert all("\t1\tdirect\t" in line for line in lines[1:])
    assert lines[1:]


def test_show_emits_the_canonical_recipe_json():
    selected = MANIFEST["recipes"][0]
    cp = run("show", selected["id"])
    assert cp.returncode == 0, cp.stderr
    assert json.loads(cp.stdout) == selected


def test_pull_refuses_non_pullable_recipe():
    selected = recipe("local-only")
    cp = run("pull", selected["id"], "--dry-run")
    assert cp.returncode != 0
    assert "not pullable" in cp.stderr


def test_pull_dry_run_uses_exact_manifest_reference():
    selected = recipe("portable")
    cp = run("pull", selected["id"], "--dry-run")
    assert cp.returncode == 0, cp.stderr
    assert cp.stdout.strip() == f"docker pull {selected['image']['reference']}"


def test_render_dry_run_uses_clean_environment_and_exact_compose_files():
    selected = recipe("portable")
    cp = run("render", selected["id"], "--model-path", "/tmp/model", "--dry-run")
    assert cp.returncode == 0, cp.stderr
    command = cp.stdout.strip()
    assert command.startswith("env -i ")
    assert f"--env-file {selected['preset']}" in command
    for compose in selected["runtime"]["compose"]:
        assert f"-f {compose}" in command
    assert "--profile '*' config" in command
    assert "MODEL_PATH=/tmp/model" in command
    assert "192.0.2." not in command
    assert "ROCE_IF_NAME=eth0" not in command


def test_render_accepts_explicit_operator_overrides_without_synthetic_defaults():
    selected = recipe("local-only")
    cp = run(
        "render", selected["id"], "--model-path", "/tmp/model",
        "--set", "B12X_CACHE_DIR=/srv/solar-cache",
        "--set", "B12X_OBS_DIR=/srv/solar-obs", "--dry-run",
    )
    assert cp.returncode == 0, cp.stderr
    assert "B12X_CACHE_DIR=/srv/solar-cache" in cp.stdout
    assert "B12X_OBS_DIR=/srv/solar-obs" in cp.stdout


def test_render_rejects_unsafe_override_keys():
    selected = recipe("portable")
    cp = run(
        "render", selected["id"], "--model-path", "/tmp/model",
        "--set", "PATH=/tmp/attacker", "--dry-run",
    )
    assert cp.returncode != 0
    assert "override key is not allowed" in cp.stderr


def test_unknown_recipe_fails_closed():
    cp = run("show", "not-a-recipe")
    assert cp.returncode != 0
    assert "unknown recipe" in cp.stderr


def test_relative_ambient_path_cannot_select_repository_local_docker(tmp_path, monkeypatch):
    fake = tmp_path / "docker"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    loader = importlib.machinery.SourceFileLoader("recipe_cli", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", ".")
    selected = Path(module.docker_executable())
    assert selected.is_absolute()
    assert selected.resolve() != fake.resolve()
