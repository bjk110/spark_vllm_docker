import importlib.util
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = REPO_ROOT / "scripts" / "diag" / "verify_qwen38_prefault_recipe.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("verify_qwen38_prefault_recipe", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def copy_recipe_tree(module, target_root):
    for rel in (module.PATCH_ROOT_REL, module.OVERLAY_REL, module.DOC_REL):
        source = REPO_ROOT / rel
        target = target_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def test_repository_recipe_passes_static_verification():
    module = load_verifier()
    assert module.verify(REPO_ROOT) == []


def mutate_overlay(module, target_root, old, new):
    copy_recipe_tree(module, target_root)
    overlay_path = target_root / module.OVERLAY_REL
    overlay = overlay_path.read_text()
    assert old in overlay
    overlay_path.write_text(overlay.replace(old, new, 1))


def test_overlay_rejects_unexpected_read_write_mount(tmp_path):
    module = load_verifier()
    mutate_overlay(
        module,
        tmp_path,
        "  worker:\n",
        "      - /tmp/unexpected:/tmp/unexpected:rw\n  worker:\n",
    )
    errors = module.verify(tmp_path)
    assert any("exact volumes" in error for error in errors)


def test_overlay_rejects_restart_policy(tmp_path):
    module = load_verifier()
    mutate_overlay(
        module,
        tmp_path,
        "  worker:\n",
        "    restart: always\n  worker:\n",
    )
    errors = module.verify(tmp_path)
    assert any("exact keys" in error for error in errors)


def test_overlay_rejects_additional_service(tmp_path):
    module = load_verifier()
    copy_recipe_tree(module, tmp_path)
    overlay_path = tmp_path / module.OVERLAY_REL
    with overlay_path.open("a") as stream:
        stream.write("  unexpected:\n    image: alpine:latest\n")
    errors = module.verify(tmp_path)
    assert any("exact services" in error for error in errors)


def test_overlay_rejects_duplicate_service_key(tmp_path):
    module = load_verifier()
    copy_recipe_tree(module, tmp_path)
    overlay_path = tmp_path / module.OVERLAY_REL
    with overlay_path.open("a") as stream:
        stream.write("  head:\n    volumes: []\n")
    errors = module.verify(tmp_path)
    assert any("duplicate YAML key" in error for error in errors)


def test_overlay_rejects_empty_document(tmp_path):
    module = load_verifier()
    copy_recipe_tree(module, tmp_path)
    (tmp_path / module.OVERLAY_REL).write_text("")
    errors = module.verify(tmp_path)
    assert any("overlay root must be an object" in error for error in errors)


def test_overlay_rejects_null_document(tmp_path):
    module = load_verifier()
    copy_recipe_tree(module, tmp_path)
    (tmp_path / module.OVERLAY_REL).write_text("null\n")
    errors = module.verify(tmp_path)
    assert any("overlay root must be an object" in error for error in errors)


def test_overlay_rejects_unhashable_yaml_key_without_exception(tmp_path):
    module = load_verifier()
    copy_recipe_tree(module, tmp_path)
    (tmp_path / module.OVERLAY_REL).write_text("services: {}\n? [bad]\n: value\n")
    errors = module.verify(tmp_path)
    assert any("invalid overlay YAML key" in error for error in errors)


def test_overlay_rejects_non_string_top_level_key_without_exception(tmp_path):
    module = load_verifier()
    copy_recipe_tree(module, tmp_path)
    overlay_path = tmp_path / module.OVERLAY_REL
    with overlay_path.open("a") as stream:
        stream.write("1: bad\n")
    errors = module.verify(tmp_path)
    assert any("top-level keys must be strings" in error for error in errors)


def test_manifest_status_is_fail_closed(tmp_path):
    module = load_verifier()
    copy_recipe_tree(module, tmp_path)

    manifest_path = tmp_path / module.PATCH_ROOT_REL / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "UNQUALIFIED"
    manifest_path.write_text(json.dumps(manifest))

    errors = module.verify(tmp_path)
    assert any("manifest status" in error for error in errors)


def test_manifest_runtime_identity_is_fail_closed(tmp_path):
    module = load_verifier()
    copy_recipe_tree(module, tmp_path)

    manifest_path = tmp_path / module.PATCH_ROOT_REL / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_runtime"]["image_id"] = "sha256:wrong"
    manifest["upstream"]["commit"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))

    errors = module.verify(tmp_path)
    assert any("image ID" in error for error in errors)
    assert any("upstream commit" in error for error in errors)


def test_compose_render_rejects_noncanonical_mount_source(monkeypatch):
    if shutil.which("docker") is None:
        return
    module = load_verifier()
    monkeypatch.setenv("QWEN38_PREFAULT_ROOT", "/tmp/noncanonical-prefault-root")
    errors = module.verify_compose_render(REPO_ROOT)
    assert any("mount source" in error for error in errors)


def test_compose_render_accepts_hash_identical_custom_root(tmp_path, monkeypatch):
    if shutil.which("docker") is None:
        return
    module = load_verifier()
    custom_root = tmp_path / "prefault"
    shutil.copytree(REPO_ROOT / module.PATCH_ROOT_REL, custom_root)
    monkeypatch.setenv("QWEN38_PREFAULT_ROOT", str(custom_root))
    assert module.verify_compose_render(REPO_ROOT) == []
