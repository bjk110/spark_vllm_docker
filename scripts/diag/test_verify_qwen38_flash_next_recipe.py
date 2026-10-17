#!/usr/bin/env python3
import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "scripts/diag/verify_qwen38_flash_next_recipe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qwen38_flash_next_recipe_verifier", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FollowupC2PresetScopeTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.presets = Path(self.tempdir.name) / "presets"
        self.presets.mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_rejects_only_c2_presets_in_the_base_flash_next_fp8_family(self):
        forbidden = self.presets / "qwen3.8-flash-next-fp8-tp2-c2.env"
        separate_variant = self.presets / "qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env"
        near_match = self.presets / "qwen3.8-flash-next-fp8-tp2-c20.env"
        forbidden.touch()
        separate_variant.touch()
        near_match.touch()

        self.assertEqual(
            [forbidden.name],
            self.module.find_unexpected_followup_c2_presets(self.presets),
        )

    def test_ignores_non_c2_base_flash_next_fp8_preset(self):
        (self.presets / "qwen3.8-flash-next-fp8-tp2-candidate.env").touch()

        self.assertEqual(
            [],
            self.module.find_unexpected_followup_c2_presets(self.presets),
        )


if __name__ == "__main__":
    unittest.main()
