#!/usr/bin/env python3
"""Validate the executable recipe manifest without building or starting containers."""
import argparse
import json
import re
from pathlib import Path

EXPECTED_START_ORDER = {
    "direct": ["head"],
    "ray": ["head", "worker"],
    "mp": ["worker", "head"],
}
DISTRIBUTIONS = {"portable", "tag-only", "build-required", "local-only", "historical"}


def render_catalog(data):
    lines = [
        "# Executable recipe catalog",
        "",
        "Generated from [`manifest.json`](manifest.json). Do not edit this table by hand.",
        "",
        "| Recipe ID | Nodes | Backend | Image distribution | Model | Runbook |",
        "|---|---:|---|---|---|---|",
    ]
    for recipe in data["recipes"]:
        runbook = recipe["documentation"]["runbook"]
        lines.append(
            f"| `{recipe['id']}` | {recipe['runtime']['nodes']} | "
            f"`{recipe['runtime']['backend']}` | `{recipe['image']['distribution']}` | "
            f"`{recipe['model']['served_name']}` | [{Path(runbook).name}](../{runbook}) |"
        )
    lines.extend((
        "",
        "Use `./scripts/recipe list`, `show`, `pull`, and `render` from the repository root.",
        "Local-only entries are configuration/evidence references and cannot be pulled from a registry.",
        "",
    ))
    return "\n".join(lines)


def parse_env(path):
    return dict(re.findall(r"^([A-Z][A-Z0-9_]*)=(.*)$", path.read_text(), re.M))


def compose_contract(preset_text):
    match = re.search(r"^# Required overlay:\s*(.*)$", preset_text, re.M)
    overlays = []
    for path in re.findall(r"[\w./-]+\.ya?ml", match.group(1) if match else ""):
        if path != "docker-compose.yml" and path not in overlays:
            overlays.append(path)
    return ["docker-compose.yml", *overlays]


def safe_relative(value):
    if not isinstance(value, str):
        return False
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def verify(root, require_evidence_scope=True):
    root = Path(root).resolve()
    errors = []
    path = root / "recipes/manifest.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"recipe manifest unreadable: {exc}"]
    recipes = data.get("recipes")
    if data.get("schema_version") != 1 or not isinstance(recipes, list):
        return ["recipe manifest schema_version/recipes invalid"]
    if data.get("recipe_count") != len(recipes):
        errors.append("recipe_count differs from recipes length")
    catalog = root / "recipes/README.md"
    if not catalog.is_file() or catalog.read_text() != render_catalog(data):
        errors.append("generated recipe catalog differs from manifest")
    ids, presets = [], []
    for index, recipe in enumerate(recipes):
        label = recipe.get("id") or f"recipe[{index}]"
        ids.append(recipe.get("id"))
        preset = recipe.get("preset")
        presets.append(preset)
        if not isinstance(recipe.get("id"), str) or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", recipe["id"]):
            errors.append(f"{label}: invalid id")
        if not safe_relative(preset):
            errors.append(f"{label}: preset must be a safe relative path")
            continue
        preset_path = root / preset
        if not preset_path.is_file():
            errors.append(f"{label}: preset missing: {preset}")
            continue
        preset_text = preset_path.read_text()
        env = parse_env(preset_path)
        model = recipe.get("model", {})
        if model.get("served_name") != env.get("SERVED_MODEL_NAME"):
            errors.append(f"{label}: served_name differs from preset")
        if not model.get("revision") and not model.get("revision_note"):
            errors.append(f"{label}: model revision or revision_note required")
        if model.get("revision"):
            runbook_path = recipe.get("documentation", {}).get("runbook")
            runbook_text = ((root / runbook_path).read_text()
                            if safe_relative(runbook_path) and (root / runbook_path).is_file() else "")
            if model["revision"] not in preset_text and model["revision"] not in runbook_text:
                errors.append(f"{label}: model revision is not present in preset or runbook")
        image = recipe.get("image", {})
        reference = image.get("reference")
        distribution = image.get("distribution")
        pullable = image.get("pullable")
        if reference != env.get("VLLM_IMAGE"):
            errors.append(f"{label}: image reference differs from preset")
        if distribution not in DISTRIBUTIONS or not isinstance(pullable, bool):
            errors.append(f"{label}: image distribution/pullable invalid")
        if distribution == "portable" and not re.fullmatch(r"[^\s]+/[^\s]+@sha256:[0-9a-f]{64}", reference or ""):
            errors.append(f"{label}: portable image must use an immutable registry digest")
        if distribution in ("portable", "tag-only") and pullable is not True:
            errors.append(f"{label}: {distribution} image must be pullable")
        if distribution in ("local-only", "build-required", "historical") and pullable is not False:
            errors.append(f"{label}: {distribution} image must not be pullable")
        if distribution == "tag-only" and (
                not re.fullmatch(r"[^\s]+/[^\s@:]+:[^\s@]+", reference or "")
                or "@sha256:" in (reference or "")):
            errors.append(f"{label}: tag-only image must use a pullable registry tag")
        runtime = recipe.get("runtime", {})
        topology = runtime.get("topology")
        nodes = runtime.get("nodes")
        backend = runtime.get("backend")
        expected_topology = "single" if nodes == 1 else "dual-rdma" if nodes == 2 else None
        if topology != expected_topology or topology != env.get("CLUSTER_MODE"):
            errors.append(f"{label}: nodes/topology differs from preset")
        expected_backend = "direct" if topology == "single" else env.get("DISTRIBUTED_BACKEND", "ray")
        if backend != expected_backend:
            errors.append(f"{label}: backend differs from preset")
        if runtime.get("start_order") != EXPECTED_START_ORDER.get(backend):
            errors.append(f"{label}: start_order invalid for backend={backend}")
        if not re.fullmatch(r"\d+\.\d+\.\d+", str(runtime.get("compose_min_version", ""))):
            errors.append(f"{label}: compose_min_version must be explicit semver")
        compose = runtime.get("compose")
        if not isinstance(compose, list) or not compose or compose[0] != "docker-compose.yml":
            errors.append(f"{label}: compose must start with docker-compose.yml")
        else:
            if compose != compose_contract(preset_text):
                errors.append(f"{label}: compose contract differs from preset metadata")
            for compose_path in compose:
                if not safe_relative(compose_path) or not (root / compose_path).is_file():
                    errors.append(f"{label}: compose file missing: {compose_path}")
        expected_entrypoint = env.get("ENTRYPOINT_FILE", "./entrypoints/entrypoint.sh").removeprefix("./")
        entrypoint = runtime.get("entrypoint")
        if (entrypoint != expected_entrypoint or not safe_relative(entrypoint)
                or not (root / entrypoint).is_file()):
            errors.append(f"{label}: entrypoint differs from preset or is missing")
        docs = recipe.get("documentation", {})
        runbook = docs.get("runbook")
        evidence = docs.get("evidence")
        if not safe_relative(runbook) or not (root / runbook).is_file():
            errors.append(f"{label}: runbook missing")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{label}: evidence list required")
        else:
            for evidence_path in evidence:
                if not safe_relative(evidence_path) or not (root / evidence_path).is_file():
                    errors.append(f"{label}: evidence missing: {evidence_path}")
    if len(ids) != len(set(ids)):
        errors.append("recipe ids must be unique")
    if len(presets) != len(set(presets)):
        errors.append("recipe presets must be unique")
    if require_evidence_scope:
        try:
            evidence_data = json.loads((root / "benchmarks/evidence/manifest.json").read_text())
            evidence_presets = {item["preset"] for item in evidence_data["recipes"]}
            if set(presets) != evidence_presets:
                errors.append("recipe manifest scope differs from canonical evidence scope")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"canonical evidence scope unreadable: {exc}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    errors = verify(args.root)
    for error in errors:
        print(f"[FAIL] {error}")
    if errors:
        return 1
    count = json.loads((args.root / "recipes/manifest.json").read_text())["recipe_count"]
    print(f"PASS: {count} executable recipes; identities, distribution, topology, docs, and evidence valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
