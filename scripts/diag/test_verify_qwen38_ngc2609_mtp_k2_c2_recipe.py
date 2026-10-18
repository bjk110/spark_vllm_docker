#!/usr/bin/env python3
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "scripts/diag/verify_qwen38_ngc2609_mtp_k2_c2_recipe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qwen38_recipe_verifier", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecipeVerifierTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "repo"
        shutil.copytree(REPO, self.root, ignore=shutil.ignore_patterns(".git", "__pycache__"))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_canonical_recipe_passes(self):
        failures, warnings = self.module.verify(self.root, render_compose=False)
        self.assertEqual([], failures)
        self.assertEqual([], warnings)

    def test_preset_uses_pullable_immutable_registry_manifest(self):
        preset = self.module.parse_env(
            self.root / "presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env"
        )
        self.assertRegex(
            preset["VLLM_IMAGE"],
            r"^ghcr\.io/bjk110/vllm-spark@sha256:[0-9a-f]{64}$",
        )

    def test_runbook_pins_pulled_image_id(self):
        doc = (self.root / "docs/qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md").read_text()
        self.assertIn("sha256:518c1b8f9bdaca735d793ffef1c7d6e050bdb62e6ffc9730e9428052cbc59512", doc)
        self.assertNotIn('org.vllm-spark.qualified.rootfs.image.id', doc)

    def test_runbook_verifies_remote_code_checkpoint_content(self):
        doc = (self.root / "docs/qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md").read_text()
        self.assertIn("hf cache verify orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8", doc)
        self.assertIn('--revision "${MODEL_REVISION}"', doc)
        self.assertIn('--fail-on-missing-files', doc)

    def test_rejects_runtime_contract_drift(self):
        preset = self.root / "presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env"
        preset.write_text(preset.read_text().replace("MAX_NUM_SEQS=2", "MAX_NUM_SEQS=3"))
        failures, _ = self.module.verify(self.root, render_compose=False)
        self.assertTrue(any("MAX_NUM_SEQS" in item for item in failures), failures)

    def test_rejects_model_revision_drift(self):
        preset = self.root / "presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env"
        preset.write_text(preset.read_text().replace(
            "MODEL_REVISION=8bebbc05cbe65099f37c520865463c1377d8050d",
            "MODEL_REVISION=main",
        ))
        failures, _ = self.module.verify(self.root, render_compose=False)
        self.assertTrue(any("MODEL_REVISION" in item for item in failures), failures)

    def test_rejects_source_mutation(self):
        source = self.root / "patches/qwen/ngc2609-e77-prefault/model_executor/utils.py"
        source.write_text(source.read_text() + "\n# mutation\n")
        failures, _ = self.module.verify(self.root, render_compose=False)
        self.assertTrue(any("SHA-256" in item for item in failures), failures)

    def test_rejects_extra_compose_service_property(self):
        overlay = self.root / "compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml"
        overlay.write_text(overlay.read_text().replace("  head:\n", "  head:\n    restart: always\n", 1))
        failures, _ = self.module.verify(self.root, render_compose=False)
        self.assertTrue(any("compose shape" in item for item in failures), failures)

    def test_rejects_mp_rollback_overlay_contract_drift(self):
        overlay = self.root / "compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2-mp-rollback.yml"
        overlay.write_text(overlay.read_text().replace("- NNODES=2", "- NNODES=3", 1))
        failures, _ = self.module.verify(self.root, render_compose=False)
        self.assertTrue(any("MP rollback compose shape" in item for item in failures), failures)

    def test_rejects_override_semantics_on_mp_rollback_environment(self):
        overlay = self.root / "compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2-mp-rollback.yml"
        overlay.write_text(overlay.read_text().replace(
            "    environment:\n", "    environment: !override\n"
        ))
        failures, _ = self.module.verify(self.root, render_compose=False)
        self.assertTrue(any("MP rollback compose shape" in item for item in failures), failures)

    def test_rejects_rendered_mp_rollback_contract_drift(self):
        base = self.root / "docker-compose.yml"
        base.write_text(base.read_text().replace(
            "      - MASTER_PORT=${MASTER_PORT:-29501}\n", ""
        ))
        failures, _ = self.module.verify(self.root)
        self.assertTrue(
            any("rendered MP rollback Compose contract invalid" in item for item in failures),
            failures,
        )

    def test_rejects_raw_head_security_option_drift(self):
        overlay = self.root / "compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml"
        overlay.write_text(overlay.read_text().replace(
            "security_opt: [label=disable]",
            "security_opt: [no-new-privileges]",
            1,
        ))
        failures, _ = self.module.verify(self.root, render_compose=False)
        self.assertTrue(any("head security option" in item for item in failures), failures)

    def test_rejects_raw_worker_security_option_drift(self):
        overlay = self.root / "compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml"
        content = overlay.read_text()
        marker = "security_opt: [label=disable]"
        index = content.rfind(marker)
        self.assertNotEqual(-1, index)
        overlay.write_text(
            content[:index] + "security_opt: [no-new-privileges]" + content[index + len(marker):]
        )
        failures, _ = self.module.verify(self.root, render_compose=False)
        self.assertTrue(any("worker security option" in item for item in failures), failures)

    def test_rejects_rendered_security_option_drift_for_each_role(self):
        cmd = [
            "docker", "compose", "--env-file", str(self.module.PRESET),
            "-f", "docker-compose.yml", "-f", str(self.module.OVERLAY),
            "--profile", "*", "config", "--format", "json",
        ]
        trusted_env = {
            key: os.environ[key]
            for key in ("HOME", "DOCKER_CONFIG") if key in os.environ
        }
        trusted_env["PATH"] = os.environ.get("PATH", os.defpath)
        canonical = self.module.subprocess.run(
            cmd,
            cwd=self.root,
            env=trusted_env,
            text=True,
            capture_output=True,
            check=True,
        )
        rendered = json.loads(canonical.stdout)
        for role in ("head", "worker"):
            with self.subTest(role=role):
                mutated = json.loads(json.dumps(rendered))
                mutated["services"][role]["security_opt"] = ["no-new-privileges"]
                fake = self.module.subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout=json.dumps(mutated),
                    stderr="",
                )
                with mock.patch.object(self.module.subprocess, "run", return_value=fake):
                    failures, _ = self.module.verify(self.root)
                self.assertTrue(
                    any(f"{role} rendered security option" in item for item in failures),
                    failures,
                )

    def test_expected_contract_covers_every_preset_assignment(self):
        preset = self.module.parse_env(
            self.root / "presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env"
        )
        self.assertEqual(preset, self.module.EXPECTED_ENV)

    def test_rejects_language_model_only(self):
        entrypoint = self.root / "entrypoints/qwen38-ngc2609-mtp-k2-c2.sh"
        entrypoint.write_text(entrypoint.read_text().replace("--enable-prefix-caching", "--language-model-only\n  --enable-prefix-caching"))
        failures, _ = self.module.verify(self.root, render_compose=False)
        self.assertTrue(any("multimodal" in item for item in failures), failures)

    def test_rejects_ucx_device_suffix_drift(self):
        overlay = self.root / "compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml"
        overlay.write_text(overlay.read_text().replace(
            'UCX_NET_DEVICES=${ROCE_IF_NAME}',
            'UCX_NET_DEVICES=${ROCE_IF_NAME}:1',
        ))
        failures, _ = self.module.verify(self.root, render_compose=False)
        self.assertTrue(any("environment list is not exact" in item for item in failures), failures)

    def test_rejects_ambient_compose_contract_override(self):
        poisoned = {
            "VLLM_IMAGE": "alpine:latest",
            "MAX_MODEL_LEN": "1",
            "MAX_NUM_SEQS": "99",
            "MAX_NUM_BATCHED_TOKENS": "7",
            "KV_CACHE_MEMORY_BYTES": "1",
            "NCCL_IB_DISABLE": "1",
            "WORKER_NODE_RANK": "9",
            "QWEN38_NGC2609_OVERLAY_ROOT": "/tmp/not-the-pinned-overlay",
        }
        with mock.patch.dict(os.environ, poisoned, clear=False):
            failures, _ = self.module.verify(self.root)
        self.assertTrue(any("rendered Compose contract" in item for item in failures), failures)

    def test_compose_unavailable_fails_closed(self):
        with mock.patch.object(self.module.shutil, "which", return_value=None):
            failures, warnings = self.module.verify(self.root)
        self.assertTrue(any("docker unavailable" in item for item in failures), failures)
        self.assertEqual([], warnings)

    def test_entrypoint_preserves_cache_and_role_rank_preflight(self):
        script = (self.root / "entrypoints/qwen38-ngc2609-mtp-k2-c2.sh").read_text()
        self.assertIn('case "${ROLE}:${NODE_RANK}"', script)
        self.assertIn('head:0|worker:1', script)
        self.assertIn('mkdir -p "${HF_HOME}"', script)

    def test_entrypoint_requires_exact_model_revision_marker(self):
        script = (self.root / "entrypoints/qwen38-ngc2609-mtp-k2-c2.sh").read_text()
        self.assertIn('${MODEL_REVISION:?MODEL_REVISION required}', script)
        self.assertIn('${MODEL_CONTAINER_PATH}/.hf-revision', script)
        self.assertIn('model revision mismatch', script)
        overlay = (self.root / "compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml").read_text()
        self.assertEqual(2, overlay.count('MODEL_REVISION=${MODEL_REVISION}'))


if __name__ == "__main__":
    unittest.main()
