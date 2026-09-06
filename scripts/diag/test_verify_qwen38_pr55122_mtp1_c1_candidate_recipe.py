#!/usr/bin/env python3
"""Adversarial mutation tests for the Gate4 PR55122/MTP1/c1 recipe."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
THIS_DIR = Path(__file__).resolve().parent
VERIFIER_PATH = THIS_DIR / "verify_qwen38_pr55122_mtp1_c1_candidate_recipe.py"
if not VERIFIER_PATH.is_file():
    raise FileNotFoundError(f"Gate4 verifier is missing (expected RED): {VERIFIER_PATH}")
_spec = importlib.util.spec_from_file_location("gate4_verifier", VERIFIER_PATH)
verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verifier)
AUTHORIZATION_STATEMENT = (
    "Local commit was subsequently explicitly user-authorized and applied; push, release, image "
    "tag/publication, build, launch, service activation, auto-start, and production-default change "
    "remain HOLD/unauthorized."
)
PROMOTED_CONTENT_PATHS = (
    *verifier.PROMOTION_INDEX_PATHS.values(),
    verifier.DOC_PATH,
    verifier.PRESET_PATH,
)
class MutationCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="qwen38-pr55122-gate4-")
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.saved = {}
    def tearDown(self):
        for name, value in self.saved.items(): setattr(verifier, name, value)
    def patch_path(self, name, content, filename):
        self.saved.setdefault(name, getattr(verifier, name))
        path = self.root / filename; path.write_text(content); setattr(verifier, name, path)
class TestDockerfileMutations(MutationCase):
    def test_syntax_parser_directive_is_rejected(self):
        text = "# syntax=attacker.example/frontend:latest\n" + verifier.DOCKERFILE_PATH.read_text()
        self.patch_path("DOCKERFILE_PATH", text, "Dockerfile")
        results = verifier.Results(); verifier.check_dockerfile(results)
        self.assertTrue(any("parser directive" in failure for failure in results.failures))

    def test_escape_parser_directive_with_leading_space_and_case_is_rejected(self):
        text = "   # EsCaPe=backtick\n" + verifier.DOCKERFILE_PATH.read_text()
        self.patch_path("DOCKERFILE_PATH", text, "Dockerfile")
        results = verifier.Results(); verifier.check_dockerfile(results)
        self.assertTrue(any("parser directive" in failure for failure in results.failures))

    def test_check_parser_directive_with_leading_space_and_case_is_rejected(self):
        text = " \t# CHECK=skip=JSONArgsRecommended\n" + verifier.DOCKERFILE_PATH.read_text()
        self.patch_path("DOCKERFILE_PATH", text, "Dockerfile")
        results = verifier.Results(); verifier.check_dockerfile(results)
        self.assertTrue(any("parser directive" in failure for failure in results.failures))

    def test_copy_flag_after_first_positional_is_rejected(self):
        text = verifier.DOCKERFILE_PATH.read_text().replace(
            "COPY --chmod=0644 patches/qwen/pr55122/_C_det.so",
            "COPY patches/qwen/pr55122/_C_det.so --chmod=0644",
            1,
        )
        self.patch_path("DOCKERFILE_PATH", text, "Dockerfile")
        results = verifier.Results(); verifier.check_dockerfile(results)
        self.assertTrue(any("contiguous prefix" in f for f in results.failures))

    def test_copy_from_external_stage_is_rejected(self):
        text = verifier.DOCKERFILE_PATH.read_text().replace("COPY --chmod=0644", "COPY --from=attacker --chmod=0644", 1)
        self.patch_path("DOCKERFILE_PATH", text, "Dockerfile")
        results = verifier.Results(); verifier.check_dockerfile(results)
        self.assertTrue(any("COPY" in f and "flag" in f for f in results.failures))
    def test_label_value_or_extra_label_is_rejected(self):
        text = verifier.DOCKERFILE_PATH.read_text().replace(
            'LABEL org.vllm-spark.candidate.purpose="LOCAL NON-PRODUCTION experimental deterministic persistent_topk evaluation; not promoted or pushed"',
            'LABEL org.vllm-spark.candidate.purpose="wrong"\nLABEL attacker.extra="value"',
            1,
        )
        self.patch_path("DOCKERFILE_PATH", text, "Dockerfile")
        results = verifier.Results(); verifier.check_dockerfile(results)
        self.assertTrue(any("LABEL key/value set" in f for f in results.failures))

    def test_copy_destination_swap_is_rejected(self):
        text = verifier.DOCKERFILE_PATH.read_text().replace(verifier.QSA_DEST, "/tmp/qsa.py", 1)
        self.patch_path("DOCKERFILE_PATH", text, "Dockerfile")
        results = verifier.Results(); verifier.check_dockerfile(results)
        self.assertTrue(any("destination" in f for f in results.failures))
    def test_pinned_upstream_kernel_source_hash_mutation_is_rejected(self):
        path = self.root / "persistent_topk.cuh"
        path.write_bytes(b"mutated upstream source")
        had = hasattr(verifier, "UPSTREAM_KERNEL_PATH")
        old = getattr(verifier, "UPSTREAM_KERNEL_PATH", None)
        setattr(verifier, "UPSTREAM_KERNEL_PATH", path)
        self.addCleanup(lambda: setattr(verifier, "UPSTREAM_KERNEL_PATH", old) if had else delattr(verifier, "UPSTREAM_KERNEL_PATH"))
        results = verifier.Results(); verifier.check_hashes(results)
        self.assertTrue(any("upstream kernel source SHA256" in f and "does not match" in f for f in results.failures))

class TestOverlayMutations(MutationCase):
    def test_overlay_name_wrong_value_or_type_is_rejected(self):
        original = verifier.OVERLAY_PATH.read_text()
        for replacement in ("name: wrong-candidate", "name: [qwen38-pr55122-mtp1-c1-candidate]"):
            with self.subTest(replacement=replacement):
                text = original.replace(f"name: {verifier.EXPECTED_PROJECT_NAME}", replacement, 1)
                self.patch_path("OVERLAY_PATH", text, "overlay.yml")
                results = verifier.Results(); verifier.check_overlay_structure(results)
                self.assertTrue(any("overlay name" in failure for failure in results.failures))

    def test_healthcheck_cmd_shell_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace("[\"CMD\", \"curl\", \"-f\", \"http://127.0.0.1:8000/health\"]", "[\"CMD-SHELL\", \"curl -f http://127.0.0.1:8000/health\"]", 1)
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("head healthcheck test" in f for f in results.failures))

    def test_healthcheck_external_url_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace("http://127.0.0.1:8000/health", "https://example.com/health", 1)
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("head healthcheck URL" in f for f in results.failures))

    def test_healthcheck_or_true_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace("\"http://127.0.0.1:8000/health\"]", "\"http://127.0.0.1:8000/health\", \"||\", \"true\"]", 1)
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("head healthcheck test list" in f for f in results.failures))

    def test_healthcheck_disable_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace("    healthcheck:\n", "    healthcheck:\n      disable: true\n", 1)
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("healthcheck disable" in f for f in results.failures))

    def test_healthcheck_extra_key_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace("      start_period: 900s", "      start_period: 900s\n      x-extra: nope", 1)
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("healthcheck key set" in f for f in results.failures))

    def test_healthcheck_missing_key_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace("      interval: 15s\n", "", 1)
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("healthcheck key set mismatch" in f for f in results.failures))

    def test_healthcheck_bool_retries_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace("      retries: 40", "      retries: true", 1)
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("healthcheck retries" in f for f in results.failures))

    def test_healthcheck_duration_wrong_types_are_rejected(self):
        original = verifier.OVERLAY_PATH.read_text()
        for key, value in (("interval", "15"), ("timeout", "5"), ("start_period", "900")):
            with self.subTest(key=key):
                text = original.replace(f"      {key}: {value}s", f"      {key}: {value}", 1)
                self.patch_path("OVERLAY_PATH", text, "overlay.yml")
                results = verifier.Results(); verifier.check_overlay_structure(results)
                self.assertTrue(any(f"healthcheck {key}" in f for f in results.failures))

    def test_healthcheck_wrong_scalar_values_are_rejected(self):
        original = verifier.OVERLAY_PATH.read_text()
        for key, old_value, bad_value in (("interval", "15s", "16s"), ("timeout", "5s", "6s"), ("retries", "40", "41"), ("start_period", "900s", "901s")):
            with self.subTest(key=key):
                text = original.replace(f"      {key}: {old_value}", f"      {key}: {bad_value}", 1)
                self.patch_path("OVERLAY_PATH", text, "overlay.yml")
                results = verifier.Results(); verifier.check_overlay_structure(results)
                self.assertTrue(any(f"healthcheck {key}" in f for f in results.failures))

    def test_worker_healthcheck_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace("  worker:\n    environment:", "  worker:\n    healthcheck: {}\n    environment:", 1)
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("worker direct-child key set" in f for f in results.failures))

    def test_gate_duplicated_under_head_and_absent_from_worker_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text()
        text = text.replace("  worker:\n    environment:\n      - VLLM_QSA_DET_TOPK=1\n", "  worker:\n    environment:\n").replace("      - VLLM_QSA_DET_TOPK=1\n", "      - VLLM_QSA_DET_TOPK=1\n      - VLLM_QSA_DET_TOPK=1\n", 1)
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("worker" in f and "VLLM_QSA_DET_TOPK" in f for f in results.failures))
    def test_arbitrary_extra_direct_service_key_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace(
            "  worker:\n    environment:", "  worker:\n    foo: bar\n    environment:", 1
        )
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("worker direct-child key set" in f for f in results.failures))

    def test_quoted_extra_direct_service_key_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace(
            "  worker:\n    environment:", "  worker:\n    'foo': bar\n    environment:", 1
        )
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("worker direct-child key set" in f for f in results.failures))

    def test_multiline_extra_direct_service_key_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace(
            "  worker:\n    environment:",
            "  worker:\n    ? >-\n      foo\n    : bar\n    environment:",
            1,
        )
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("worker direct-child key set" in f for f in results.failures))

    def test_extra_environment_entry_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text().replace(
            "      - VLLM_QSA_DET_TOPK=1",
            "      - VLLM_QSA_DET_TOPK=1\n      - FOO=bar",
            1,
        )
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("head environment entries" in f for f in results.failures))

    def test_extra_service_is_rejected(self):
        text = verifier.OVERLAY_PATH.read_text() + "  prewarm:\n    environment:\n      - FOO=bar\n"
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("service" in f and "prewarm" in f for f in results.failures))
    def test_inline_flow_numeric_yaml_alias_is_rejected(self):
        text = ("x: [&123 foo]\n" + verifier.OVERLAY_PATH.read_text()).replace(
            "      - VLLM_QSA_DET_TOPK=1", "      - [*123]", 1
        )
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("anchor" in f or "alias" in f for f in results.failures))

    def test_numeric_yaml_anchor_and_alias_are_rejected(self):
        text = "services:\n  head:\n    environment: &123 []\n    healthcheck: {}\n  worker:\n    environment: *123\n"
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("anchor" in f or "alias" in f for f in results.failures))

    def test_yaml_anchor_alias_and_merge_are_rejected(self):
        text = "x-env: &candidate-env\n  - VLLM_QSA_DET_TOPK=1\nservices:\n  head:\n    environment: *candidate-env\n  worker:\n    <<: {environment: *candidate-env}\n"
        self.patch_path("OVERLAY_PATH", text, "overlay.yml")
        results = verifier.Results(); verifier.check_overlay_structure(results)
        self.assertTrue(any("anchor" in f or "alias" in f or "merge" in f for f in results.failures))
class TestPresetMutations(MutationCase):
    def test_false_push_release_activation_or_scope_claim_in_header_comment_is_rejected(self):
        claims = (
            "PR55122 profile was PUSHED.",
            "PR55122 profile was RELEASED and published.",
            "PR55122 service activation is authorized.",
            "PR55122 MTP1 c2 is authorized.",
        )
        original = verifier.PRESET_PATH.read_text()
        for claim in claims:
            with self.subTest(claim=claim):
                text = original.replace("# ", f"# {claim}\n# ", 1)
                self.patch_path("PRESET_PATH", text, "candidate.env")
                results = verifier.Results(); verifier.check_preset(results)
                self.assertTrue(any("preset full-file SHA256" in f for f in results.failures))

    def test_bare_noncomment_directive_is_rejected(self):
        text = verifier.PRESET_PATH.read_text() + "\nBARE_DIRECTIVE\n"
        self.patch_path("PRESET_PATH", text, "candidate.env")
        results = verifier.Results(); verifier.check_preset(results)
        self.assertTrue(any("preset syntax" in f and "BARE_DIRECTIVE" in f for f in results.failures))

    def test_export_assignment_syntax_is_rejected(self):
        text = verifier.PRESET_PATH.read_text().replace("TP_SIZE=2", "export TP_SIZE=2", 1)
        self.patch_path("PRESET_PATH", text, "candidate.env")
        results = verifier.Results(); verifier.check_preset(results)
        self.assertTrue(any("preset syntax" in f and "export TP_SIZE=2" in f for f in results.failures))

    def test_value_containing_equals_is_preserved(self):
        path = self.root / "equals.env"
        path.write_text("TOKEN=left=middle=right\n")
        self.assertEqual({"TOKEN": "left=middle=right"}, verifier.parse_env(path))

    def test_duplicate_key_with_same_value_is_rejected(self):
        text = verifier.PRESET_PATH.read_text().replace("TP_SIZE=2", "TP_SIZE=2\nTP_SIZE=2", 1)
        self.patch_path("PRESET_PATH", text, "candidate.env")
        results = verifier.Results(); verifier.check_preset(results)
        self.assertTrue(any("duplicate preset key" in f and "TP_SIZE" in f for f in results.failures))

    def test_arbitrary_nonapproved_preset_value_change_is_rejected(self):
        text = verifier.PRESET_PATH.read_text().replace("NCCL_DEBUG=WARN", "NCCL_DEBUG=INFO", 1)
        self.patch_path("PRESET_PATH", text, "candidate.env")
        results = verifier.Results(); verifier.check_preset(results)
        self.assertTrue(any("differs from production" in f and "NCCL_DEBUG" in f for f in results.failures))

    def test_extra_args_with_unapproved_extra_token_are_rejected(self):
        text = verifier.PRESET_PATH.read_text().replace(
            ' --speculative-config={"method":"mtp","num_speculative_tokens":1}',
            ' --speculative-config={"method":"mtp","num_speculative_tokens":1} --evil',
            1,
        )
        self.patch_path("PRESET_PATH", text, "candidate.env")
        results = verifier.Results(); verifier.check_preset(results)
        self.assertTrue(any("exact production string plus approved suffix" in f for f in results.failures))

class TestDocMutations(MutationCase):
    def test_current_exact_doc_passes(self):
        results = verifier.Results(); verifier.check_doc(results)
        self.assertEqual([], results.failures)

    def test_each_appended_false_positive_claim_is_rejected(self):
        claims = (
            "PR55122 profile was PUSHED.",
            "PR55122 profile RELEASED.",
            "PR55122 is now the production default.",
            "PR55122 is active on the production service.",
            "PR55122 service activation is authorized.",
            "PR55122 MTP1 c8 is supported.",
        )
        original = verifier.DOC_PATH.read_text()
        for claim in claims:
            with self.subTest(claim=claim):
                self.patch_path("DOC_PATH", original + f"\n{claim}\n", "candidate.md")
                results = verifier.Results(); verifier.check_doc(results)
                self.assertTrue(any("doc full-file SHA256" in f for f in results.failures))

    def test_status_downgrade_to_controlled_candidate_is_rejected(self):
        promoted = "**Status: PROMOTED_LOCAL_CONFIG_NOT_RELEASED — opt-in interactive c1 only; NOT the production default; no auto-start.**"
        downgraded = "**Status: CONTROLLED-CANDIDATE — opt-in only; NOT the production default; no auto-start.**"
        text = verifier.DOC_PATH.read_text().replace(promoted, downgraded, 1)
        self.patch_path("DOC_PATH", text, "candidate.md")
        results = verifier.Results(); verifier.check_doc(results)
        self.assertTrue(any("exact status line location" in f for f in results.failures))

    def test_prepended_promoted_and_scope_authorized_status_is_rejected(self):
        text = "Status: PRODUCTION-PROMOTED; c2/c8 authorized\n" + verifier.DOC_PATH.read_text()
        self.patch_path("DOC_PATH", text, "candidate.md")
        results = verifier.Results(); verifier.check_doc(results)
        self.assertTrue(any("exact status line location" in f for f in results.failures))
        self.assertTrue(any("contradictory positive claim" in f for f in results.failures))

    def test_false_release_claim_is_rejected(self):
        text = verifier.DOC_PATH.read_text().replace(
            "\n\nThis Gate4 recipe", "\n\nThe MTP1 profile was RELEASED and published.\n\nThis Gate4 recipe", 1
        )
        self.patch_path("DOC_PATH", text, "candidate.md")
        results = verifier.Results(); verifier.check_doc(results)
        self.assertTrue(any("contradictory positive claim" in f for f in results.failures))

    def test_explicit_positive_promotion_and_scope_forms_are_rejected(self):
        original = verifier.DOC_PATH.read_text()
        claims = (
            "PRODUCTION-PROMOTED",
            "production promotion: PASS",
            "production promotion: APPROVED",
            "c2/c8 authorized",
            "c2/c8 qualified",
            "prefix-cache ON qualified",
            "prefix-cache ON authorized",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                text = original.replace("\n\nThis Gate4 recipe", f"\n\n{claim}\n\nThis Gate4 recipe", 1)
                self.patch_path("DOC_PATH", text, "candidate.md")
                results = verifier.Results(); verifier.check_doc(results)
                self.assertTrue(any("contradictory positive claim" in f for f in results.failures))

    def test_appended_negative_scope_disclaimer_is_rejected(self):
        text = verifier.DOC_PATH.read_text() + "\nNot PRODUCTION-PROMOTED; c2/c8 not authorized or qualified; prefix-cache ON is not qualified or authorized.\n"
        self.patch_path("DOC_PATH", text, "candidate.md")
        results = verifier.Results(); verifier.check_doc(results)
        self.assertTrue(any("doc full-file SHA256" in f for f in results.failures))

class TestPromotionIndexes(MutationCase):
    def _patch_indexes(self, mutate=None):
        self.saved.setdefault("PROMOTION_INDEX_PATHS", verifier.PROMOTION_INDEX_PATHS)
        paths = {}
        for name, required in verifier.PROMOTION_INDEX_REQUIRED_LINES.items():
            lines = list(required)
            if mutate is not None:
                lines = mutate(name, lines)
            path = self.root / name.replace("/", "_")
            path.write_text("\n".join(lines) + "\n")
            paths[name] = path
        verifier.PROMOTION_INDEX_PATHS = paths

    def _patch_one_exact_index(self, target, suffix):
        self.saved.setdefault("PROMOTION_INDEX_PATHS", verifier.PROMOTION_INDEX_PATHS)
        paths = dict(verifier.PROMOTION_INDEX_PATHS)
        path = self.root / target.replace("/", "_")
        path.write_bytes(verifier.PROMOTION_INDEX_PATHS[target].read_bytes() + suffix.encode())
        paths[target] = path
        verifier.PROMOTION_INDEX_PATHS = paths

    def test_current_six_promotion_indexes_pass(self):
        results = verifier.Results(); verifier.check_promotion_indexes(results)
        self.assertEqual([], results.failures)

    def test_each_appended_false_positive_claim_is_rejected_in_each_index(self):
        claims = (
            "PR55122 profile was PUSHED.",
            "PR55122 profile RELEASED.",
            "PR55122 is now the production default.",
            "PR55122 is active on the production service.",
            "PR55122 service activation is authorized.",
            "PR55122 MTP1 c8 is supported.",
        )
        for target in verifier.PROMOTION_INDEX_PATHS:
            for claim in claims:
                with self.subTest(target=target, claim=claim):
                    self._patch_one_exact_index(target, f"\n{claim}\n")
                    results = verifier.Results(); verifier.check_promotion_indexes(results)
                    self.assertTrue(any(
                        target in f and "full-file SHA256" in f
                        for f in results.failures
                    ))
                    verifier.PROMOTION_INDEX_PATHS = self.saved["PROMOTION_INDEX_PATHS"]

    def test_missing_expected_index_entry_is_rejected(self):
        target = "docs/README.md"
        self._patch_indexes(lambda name, lines: lines[1:] if name == target else lines)
        results = verifier.Results(); verifier.check_promotion_indexes(results)
        self.assertTrue(any(target in f and "exactly once" in f for f in results.failures))

    def test_duplicated_expected_index_entry_is_rejected(self):
        target = "patches/README.md"
        self._patch_indexes(lambda name, lines: lines + [lines[0]] if name == target else lines)
        results = verifier.Results(); verifier.check_promotion_indexes(results)
        self.assertTrue(any(target in f and "exactly once" in f for f in results.failures))

    def test_index_scope_creep_is_rejected(self):
        target = "README.md"
        self._patch_indexes(lambda name, lines: lines + ["PR55122 MTP1 c2 authorized and RELEASED"] if name == target else lines)
        results = verifier.Results(); verifier.check_promotion_indexes(results)
        self.assertTrue(any(target in f and "scope creep" in f for f in results.failures))

class TestPostCommitStateContract(unittest.TestCase):
    def test_all_eight_promoted_content_files_require_exact_post_commit_statement(self):
        self.assertEqual(8, len(PROMOTED_CONTENT_PATHS))
        for path in PROMOTED_CONTENT_PATHS:
            with self.subTest(path=path):
                self.assertIn(AUTHORIZATION_STATEMENT, path.read_text())

    def test_stale_current_no_commit_claims_are_absent(self):
        stale_claims = (
            "commit remains HOLD",
            "Commit, push, release",
            "No commit/push/release",
            "no commit/push/release",
            "commit, push, release, image tag/publication, build, launch, or service activation",
        )
        for path in PROMOTED_CONTENT_PATHS:
            lines = path.read_text().splitlines()
            current_lines = [line for line in lines if "it did not authorize commit" not in line]
            current_text = "\n".join(current_lines)
            for claim in stale_claims:
                with self.subTest(path=path, claim=claim):
                    self.assertNotIn(claim, current_text)

    def test_historical_gate5_non_authorization_is_immediately_corrected(self):
        gate5_history = (
            "it did not authorize commit, push, release, image tag/publication, build, launch, or "
            "service activation. " + AUTHORIZATION_STATEMENT
        )
        text = verifier.DOC_PATH.read_text()
        self.assertIn(gate5_history, text)
        self.assertEqual(1, text.count("it did not authorize commit"))

class TestAmbientClosure(unittest.TestCase):
    def run_verifier(self, overrides):
        env = os.environ.copy()
        for key in (*verifier.parse_env(verifier.PRESET_PATH), "COMPOSE_PROJECT_NAME", verifier.LOCAL_IMAGE_GATE, verifier.IMAGE_HOSTS_ENV):
            env.pop(key, None)
        env.update(overrides)
        return subprocess.run(
            [sys.executable, str(VERIFIER_PATH)],
            cwd=verifier.REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_three_ambient_attacker_values_fail_real_verifier_and_render_checks(self):
        cp = self.run_verifier({
            "VLLM_IMAGE": "sha256:aaaa",
            "COMPOSE_PROJECT_NAME": "production-collision",
            "ENTRYPOINT_FILE": "/tmp/attacker",
        })
        output = cp.stdout + cp.stderr
        self.assertEqual(1, cp.returncode, output)
        for needle in (
            "ambient preset drift VLLM_IMAGE",
            "ambient preset drift ENTRYPOINT_FILE",
            "ambient COMPOSE_PROJECT_NAME",
            "rendered project name",
            "rendered image",
            "rendered /entrypoint.sh bind",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, output)

    def test_generic_differing_preset_key_fails_real_verifier(self):
        cp = self.run_verifier({"TP_SIZE": "999"})
        output = cp.stdout + cp.stderr
        self.assertEqual(1, cp.returncode, output)
        self.assertIn("ambient preset drift TP_SIZE", output)

class TestRenderedAndImageIdentity(MutationCase):
    def test_real_render_resists_poisoned_ambient_values(self):
        if not verifier.docker_compose_available(): self.skipTest("docker compose unavailable")
        poison = {"VLLM_QSA_DET_TOPK":"0", "VLLM_QSA_DET_LIB":"/tmp/evil.so", "VLLM_ALL2ALL_BACKEND":"wrong", "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}
        with mock.patch.dict(os.environ, poison, clear=True):
            results = verifier.Results(); verifier.check_compose_config(results)
        self.assertEqual([], results.failures)
    def test_forced_local_image_id_mismatch_fails_closed(self):
        with mock.patch.dict(os.environ, {verifier.LOCAL_IMAGE_GATE:"1"}, clear=True), mock.patch.object(verifier.subprocess, "run", return_value=verifier.subprocess.CompletedProcess(args=[], returncode=0, stdout=verifier.json.dumps({"Id":"sha256:not-the-candidate","Config":{"Labels":verifier.EXPECTED_LABELS}})+"\n", stderr="")):
            results = verifier.Results(); verifier.check_local_image_identity(results)
        self.assertTrue(any("image ID" in f for f in results.failures))
    def test_forced_image_with_wrong_provenance_label_is_rejected(self):
        payload = {"Id": verifier.EXPECTED_IMAGE_ID, "Config": {"Labels": {**verifier.EXPECTED_LABELS, "org.vllm-spark.candidate.upstream-pr-head": "wrong"}}}
        env = {verifier.LOCAL_IMAGE_GATE:"1", verifier.IMAGE_HOSTS_ENV:"spark01,spark02"}
        cp = verifier.subprocess.CompletedProcess(args=[], returncode=0, stdout=verifier.json.dumps(payload)+"\n", stderr="")
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(verifier.subprocess, "run", return_value=cp):
            results = verifier.Results(); verifier.check_local_image_identity(results)
        self.assertTrue(any("spark01" in f and "provenance labels" in f for f in results.failures))

    def test_forced_image_allows_unrelated_inherited_base_labels(self):
        labels = {**verifier.EXPECTED_LABELS, "org.opencontainers.image.source": "base"}
        payload = {"Id": verifier.EXPECTED_IMAGE_ID, "Config": {"Labels": labels}}
        env = {verifier.LOCAL_IMAGE_GATE:"1", verifier.IMAGE_HOSTS_ENV:"spark01,spark02"}
        cp = verifier.subprocess.CompletedProcess(args=[], returncode=0, stdout=verifier.json.dumps(payload)+"\n", stderr="")
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(verifier.subprocess, "run", return_value=cp):
            results = verifier.Results(); verifier.check_local_image_identity(results)
        self.assertEqual([], results.failures)

    def test_forced_remote_inspect_avoids_ssh_unsafe_format_template(self):
        payload = {"Id": verifier.EXPECTED_IMAGE_ID, "Config": {"Labels": verifier.EXPECTED_LABELS}}
        env = {verifier.LOCAL_IMAGE_GATE:"1", verifier.IMAGE_HOSTS_ENV:"spark01,spark02"}
        cp = verifier.subprocess.CompletedProcess(args=[], returncode=0, stdout=verifier.json.dumps(payload)+"\n", stderr="")
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(verifier.subprocess, "run", return_value=cp) as run:
            results = verifier.Results(); verifier.check_local_image_identity(results)
        self.assertNotIn("--format", run.call_args.args[0])
        self.assertEqual([], results.failures)

    def test_forced_remote_hosts_reject_malicious_or_unexpected_values_before_subprocess(self):
        bad_values = (
            "-oProxyCommand=/bin/false",
            "spark01",
            "spark02,spark01",
            "spark01,spark01",
            "spark01, spark02",
            "spark01,,spark02",
            "spark01;touch/tmp/pwn,spark02",
            "spark01,spark03",
            "",
        )
        for value in bad_values:
            with self.subTest(value=value), mock.patch.dict(
                os.environ,
                {verifier.LOCAL_IMAGE_GATE: "1", verifier.IMAGE_HOSTS_ENV: value},
                clear=True,
            ), mock.patch.object(verifier.subprocess, "run") as run:
                results = verifier.Results(); verifier.check_local_image_identity(results)
                run.assert_not_called()
                self.assertTrue(any("forced remote hosts" in failure for failure in results.failures))

    def test_forced_target_node_identity_checks_every_named_host(self):
        completed = [
            verifier.subprocess.CompletedProcess(args=[], returncode=0, stdout=verifier.json.dumps({"Id":verifier.EXPECTED_IMAGE_ID,"Config":{"Labels":verifier.EXPECTED_LABELS}})+"\n", stderr=""),
            verifier.subprocess.CompletedProcess(args=[], returncode=0, stdout=verifier.json.dumps({"Id":"sha256:wrong","Config":{"Labels":verifier.EXPECTED_LABELS}})+"\n", stderr=""),
        ]
        env = {verifier.LOCAL_IMAGE_GATE:"1", verifier.IMAGE_HOSTS_ENV:"spark01,spark02"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(verifier.subprocess, "run", side_effect=completed) as run:
            results = verifier.Results(); verifier.check_local_image_identity(results)
        self.assertEqual(2, run.call_count)
        self.assertTrue(any("spark02" in f and "image ID" in f for f in results.failures))
if __name__ == "__main__": unittest.main(verbosity=2)
