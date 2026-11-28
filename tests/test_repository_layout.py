"""Integration contracts for the installed src layout and migrated entry points."""

import importlib.util
from pathlib import Path
import unittest

from acr.provenance import config_path

ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayout(unittest.TestCase):
    def test_legacy_config_metadata_can_still_be_read(self):
        old = "configs/p2-guitarset-cqt-player.json"
        self.assertEqual(config_path(old), Path("configs/p2/p2-guitarset-cqt-player.json"))
        canonical = Path("configs/p2/p2-guitarset-cqt-player.json")
        self.assertEqual(config_path(canonical), canonical)

    def test_notebook_commands_configs_and_documentation_links(self):
        spec = importlib.util.spec_from_file_location(
            "verify_structure", ROOT / "tools/verify_structure.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.check_static()
        self.assertIn(
            "notebooks/p2/guitarset/02k_guitarset_transposition_repeats.ipynb", report["notebooks"]
        )
        self.assertGreater(len(report["notebook_cli_modules"]), 10)
        self.assertGreater(report["local_documentation_links"], 30)

    def test_data_and_models_do_not_import_training_modules(self):
        import ast

        for folder in ("data", "models"):
            for path in (ROOT / "src/acr" / folder).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        self.assertFalse(node.module.startswith("acr.training"), str(path))
