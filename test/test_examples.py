"""Smoke tests for all migrated example scripts."""

import io
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"

EXPECTED_FILES = {
    "_utils.py",
    "00_list_components.py",
    "01_image_io.py",
    "02_traditional_enhance.py",
    "03_deep_enhance.py",
    "04_evaluate.py",
    "05_train_tiny.py",
    "06_model_forward.py",
}

RUNNABLE_FILES = sorted(EXPECTED_FILES - {"_utils.py"})


class ExampleInventoryTests(unittest.TestCase):
    def test_every_source_example_was_migrated(self):
        migrated = {path.name for path in EXAMPLES_DIR.glob("*.py")}

        self.assertEqual(migrated, EXPECTED_FILES)

    def test_examples_do_not_reference_removed_names_or_absolute_drives(self):
        forbidden = (
            "import libllie",
            "from libllie",
            "LLIEModel",
            "LLIEnhancer",
        )

        for path in EXAMPLES_DIR.glob("*.py"):
            content = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                for text in forbidden:
                    self.assertNotIn(text, content)
                self.assertNotRegex(content, r"[A-Za-z]:\\")


class ExampleExecutionTests(unittest.TestCase):
    def test_all_runnable_examples_complete_with_synthetic_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "example-output"
            input_root = Path(temp_dir) / "example-input"
            input_root.mkdir()
            original_path = list(sys.path)
            sys.path.insert(0, str(EXAMPLES_DIR))
            sys.modules.pop("_utils", None)

            try:
                with patch.dict(
                    os.environ,
                    {
                        "OPENLLV_EXAMPLES_INPUTS": str(input_root),
                        "OPENLLV_EXAMPLES_OUTPUT": str(output_root),
                    },
                ):
                    for filename in RUNNABLE_FILES:
                        with self.subTest(filename=filename):
                            stream = io.StringIO()
                            with redirect_stdout(stream), redirect_stderr(stream):
                                runpy.run_path(
                                    str(EXAMPLES_DIR / filename),
                                    run_name="__main__",
                                )
                            self.assertTrue(stream.getvalue().strip())
            finally:
                sys.path[:] = original_path
                sys.modules.pop("_utils", None)

            self.assertTrue((output_root / "result" / "example_input.png").is_file())
            self.assertTrue((output_root / "io" / "copied.png").is_file())
            self.assertTrue(
                (output_root / "traditional" / "gamma_single.png").is_file()
            )
            self.assertTrue((output_root / "deep" / "zerodce_single.png").is_file())
            self.assertTrue(
                (output_root / "evaluation" / "evaluation.json").is_file()
            )
            self.assertTrue(
                (
                    output_root
                    / "checkpoints"
                    / "ZeroDCE_tiny"
                    / "checkpoints"
                    / "last.pt"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
