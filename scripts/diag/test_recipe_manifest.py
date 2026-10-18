"""Fail-closed tests for the executable recipe manifest."""
import importlib.util
import copy
import json
import re
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "recipe_manifest", Path(__file__).with_name("verify_recipe_manifest.py")
)
recipe_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe_manifest)
ROOT = Path(__file__).resolve().parents[2]


def test_repository_manifest_is_complete_and_valid():
    assert recipe_manifest.verify(ROOT) == []


def test_manifest_covers_exactly_the_evidence_backed_recipe_set():
    manifest = json.loads((ROOT / "recipes/manifest.json").read_text())
    evidence = json.loads((ROOT / "benchmarks/evidence/manifest.json").read_text())
    assert {item["preset"] for item in manifest["recipes"]} == {
        item["preset"] for item in evidence["recipes"]
    }


def test_every_recipe_names_its_effective_entrypoint():
    manifest = json.loads((ROOT / "recipes/manifest.json").read_text())
    for recipe in manifest["recipes"]:
        preset = (ROOT / recipe["preset"]).read_text()
        match = re.search(r"^ENTRYPOINT_FILE=(.*)$", preset, re.M)
        expected = (match.group(1) if match else "./entrypoints/entrypoint.sh").removeprefix("./")
        assert recipe["runtime"]["entrypoint"] == expected
        assert (ROOT / expected).is_file()


def test_generated_catalog_is_exactly_derived_from_manifest():
    manifest = json.loads((ROOT / "recipes/manifest.json").read_text())
    assert (ROOT / "recipes/README.md").read_text() == recipe_manifest.render_catalog(manifest)


def verify_mutated_repository(monkeypatch, mutate):
    manifest_path = ROOT / "recipes/manifest.json"
    catalog_path = ROOT / "recipes/README.md"
    data = copy.deepcopy(json.loads(manifest_path.read_text()))
    mutate(data)
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == manifest_path:
            return json.dumps(data)
        if path == catalog_path:
            return recipe_manifest.render_catalog(data)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    return recipe_manifest.verify(ROOT)


def test_required_overlay_removal_is_rejected(monkeypatch):
    def mutate(data):
        data["recipes"][0]["runtime"]["compose"] = ["docker-compose.yml"]
    assert any("compose contract differs from preset metadata" in error
               for error in verify_mutated_repository(monkeypatch, mutate))


def test_pinned_model_revision_drift_is_rejected(monkeypatch):
    def mutate(data):
        recipe = next(item for item in data["recipes"] if item["model"].get("revision"))
        recipe["model"]["revision"] = "0" * 40
    assert any("model revision is not present" in error
               for error in verify_mutated_repository(monkeypatch, mutate))


def test_distribution_and_pullability_must_agree(monkeypatch):
    def mutate(data):
        data["recipes"][0]["image"].update(distribution="local-only", pullable=True)
    assert any("local-only image must not be pullable" in error
               for error in verify_mutated_repository(monkeypatch, mutate))


def test_distribution_contract_is_fail_closed(tmp_path):
    data = {
        "schema_version": 1,
        "recipe_count": 1,
        "recipes": [{
            "id": "demo",
            "preset": "presets/demo.env",
            "status": "validated",
            "model": {"served_name": "demo", "revision": None,
                      "revision_note": "upstream does not publish immutable revisions"},
            "image": {"reference": "sha256:local", "distribution": "portable",
                      "pullable": True},
            "runtime": {"nodes": 1, "topology": "single", "backend": "direct",
                        "compose": ["docker-compose.yml"], "start_order": ["head"],
                        "compose_min_version": "2.24.4"},
            "documentation": {"runbook": "docs/demo.md", "evidence": ["docs/demo.md"]},
        }],
    }
    (tmp_path / "recipes").mkdir()
    (tmp_path / "presets").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "recipes/manifest.json").write_text(json.dumps(data))
    (tmp_path / "presets/demo.env").write_text(
        "VLLM_IMAGE=sha256:local\nSERVED_MODEL_NAME=demo\nCLUSTER_MODE=single\nTP_SIZE=1\n"
    )
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / "docs/demo.md").write_text("# demo\n")
    errors = recipe_manifest.verify(tmp_path, require_evidence_scope=False)
    assert any("portable image must use an immutable registry digest" in error for error in errors)


@pytest.mark.parametrize(
    "backend,start_order",
    [("direct", ["worker"]), ("ray", ["worker", "head"]), ("mp", ["head", "worker"])],
)
def test_start_order_contract_is_fail_closed(tmp_path, backend, start_order):
    nodes = 1 if backend == "direct" else 2
    topology = "single" if nodes == 1 else "dual-rdma"
    data = {
        "schema_version": 1,
        "recipe_count": 1,
        "recipes": [{
            "id": "demo", "preset": "presets/demo.env", "status": "validated",
            "model": {"served_name": "demo", "revision": "abc"},
            "image": {"reference": "registry.example/demo@sha256:" + "a" * 64,
                      "distribution": "portable", "pullable": True},
            "runtime": {"nodes": nodes, "topology": topology, "backend": backend,
                        "compose": ["docker-compose.yml"], "start_order": start_order,
                        "compose_min_version": "2.24.4"},
            "documentation": {"runbook": "docs/demo.md", "evidence": ["docs/demo.md"]},
        }],
    }
    (tmp_path / "recipes").mkdir()
    (tmp_path / "presets").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "recipes/manifest.json").write_text(json.dumps(data))
    env = [
        "VLLM_IMAGE=registry.example/demo@sha256:" + "a" * 64,
        "SERVED_MODEL_NAME=demo", f"CLUSTER_MODE={topology}", f"TP_SIZE={nodes}",
    ]
    if backend != "direct":
        env.append(f"DISTRIBUTED_BACKEND={backend}")
    (tmp_path / "presets/demo.env").write_text("\n".join(env) + "\n")
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / "docs/demo.md").write_text("# demo\n")
    errors = recipe_manifest.verify(tmp_path, require_evidence_scope=False)
    assert any("start_order" in error for error in errors)
