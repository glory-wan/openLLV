"""Tests for :mod:`openLLV.cli`."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from openLLV import cli


def sample_image(value=64):
    """Return a small uint8 RGB image."""
    return np.full((6, 8, 3), value, dtype=np.uint8)


class CLIParserTests(unittest.TestCase):
    def test_root_parser_uses_openllv_identity_and_requires_a_command(self):
        parser = cli._build_parser()

        self.assertEqual(parser.prog, "openllv")
        self.assertIn("openLLV", parser.description)
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_predict_command_parses_backend_and_runtime_options(self):
        args = cli._build_parser().parse_args(
            [
                "predict",
                "ZeroDCE",
                "input.png",
                "--output",
                "output.png",
                "--backend",
                "deep",
                "--device",
                "cpu",
                "--batch-size",
                "2",
                "--num-workers",
                "3",
                "--no-progress",
                "--no-save",
                "--output-name",
                "renamed.png",
                "--output-ext",
                "jpg",
                "--kwargs",
                "input_channels=3",
                "clamp_output=true",
            ]
        )

        self.assertEqual(args.command, "predict")
        self.assertEqual(args.target, "ZeroDCE")
        self.assertEqual(args.source, "input.png")
        self.assertEqual(args.output, "output.png")
        self.assertEqual(args.backend, "deep")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.num_workers, 3)
        self.assertTrue(args.no_progress)
        self.assertTrue(args.no_save)

    def test_eval_aliases_accept_short_and_long_directory_options(self):
        cases = (
            (
                [
                    "eval",
                    "--en",
                    "enhanced",
                    "--ref",
                    "reference",
                    "--metrics",
                    "PSNR",
                    "SSIM",
                ],
                "eval",
            ),
            (
                [
                    "evaluate",
                    "--en-img-dir",
                    "enhanced",
                    "--ref-img-dir",
                    "reference",
                    "--metrics",
                    "PSNR",
                ],
                "evaluate",
            ),
        )

        for arguments, command in cases:
            with self.subTest(command=command):
                args = cli._build_parser().parse_args(arguments)
                self.assertEqual(args.command, command)
                self.assertEqual(args.en_img_dir, "enhanced")
                self.assertEqual(args.ref_img_dir, "reference")
                self.assertIn("PSNR", args.metrics)

    def test_train_accepts_builtin_config_name_and_imwrite_options(self):
        train_args = cli._build_parser().parse_args(
            ["train", "ZeroDCE", "--kwargs", "epochs=1"]
        )
        write_args = cli._build_parser().parse_args(
            [
                "imwrite",
                "input.png",
                "--output",
                "results",
                "--save-format",
                "jpg",
                "--output-name",
                "converted.jpg",
            ]
        )

        self.assertEqual(train_args.config, "ZeroDCE")
        self.assertEqual(train_args.kwargs, ["epochs=1"])
        self.assertEqual(write_args.image, "input.png")
        self.assertEqual(write_args.save_format, "jpg")
        self.assertEqual(write_args.output_name, "converted.jpg")


class CLICommandDispatchTests(unittest.TestCase):
    def test_predict_command_maps_options_to_top_level_api(self):
        args = cli._build_parser().parse_args(
            [
                "predict",
                "gamma",
                "input.png",
                "-o",
                "output.png",
                "--no-progress",
                "--no-save",
                "--kwargs",
                "gamma=0.8",
            ]
        )
        expected = object()

        with patch.object(cli.llv, "predict", return_value=expected) as predict:
            result = cli._cmd_predict(args)

        self.assertIs(result, expected)
        predict.assert_called_once_with(
            "gamma",
            "input.png",
            output="output.png",
            gamma=0.8,
            backend="auto",
            batch_size=1,
            num_workers=0,
            progress_bar=False,
            save=False,
        )

    def test_train_evaluate_imwrite_and_list_commands_dispatch(self):
        parser = cli._build_parser()
        train_args = parser.parse_args(
            ["train", "ZeroDCE", "--kwargs", "epochs=2", "device=cpu"]
        )
        eval_args = parser.parse_args(
            ["eval", "--en", "enhanced", "--kwargs", "batch_size=2"]
        )
        write_args = parser.parse_args(
            ["imwrite", "input.png", "-o", "output.png"]
        )
        list_args = parser.parse_args(["list"])

        with patch.object(cli.llv, "train", return_value="trained") as train:
            self.assertEqual(cli._cmd_train(train_args), "trained")
            train.assert_called_once_with(
                "ZeroDCE",
                epochs=2,
                device="cpu",
            )

        with patch.object(cli.llv, "evaluate", return_value="evaluated") as evaluate:
            self.assertEqual(cli._cmd_evaluate(eval_args), "evaluated")
            evaluate.assert_called_once_with(
                en="enhanced",
                ref=None,
                metrics=None,
                save_path=None,
                return_evaluator=False,
                batch_size=2,
            )

        with patch.object(cli.llv, "imwrite", return_value="written") as imwrite:
            self.assertEqual(cli._cmd_imwrite(write_args), "written")
            imwrite.assert_called_once_with(
                "input.png",
                output="output.png",
                save_format=None,
                output_name=None,
            )

        available = {"models": []}
        with patch.object(cli.llv, "list_available", return_value=available):
            self.assertIs(cli._cmd_list_available(list_args), available)


class CLIValueParsingTests(unittest.TestCase):
    def test_key_value_parser_supports_common_literal_types(self):
        parsed = cli._parse_key_value_args(
            [
                "integer=2",
                "ratio=0.5",
                "enabled=true",
                "disabled=False",
                "missing=None",
                "items=[1, 2]",
                "mapping={'x': 1}",
                "text=cpu",
                "expression=a=b",
            ]
        )

        self.assertEqual(parsed["integer"], 2)
        self.assertEqual(parsed["ratio"], 0.5)
        self.assertIs(parsed["enabled"], True)
        self.assertIs(parsed["disabled"], False)
        self.assertIsNone(parsed["missing"])
        self.assertEqual(parsed["items"], [1, 2])
        self.assertEqual(parsed["mapping"], {"x": 1})
        self.assertEqual(parsed["text"], "cpu")
        self.assertEqual(parsed["expression"], "a=b")

    def test_invalid_key_value_arguments_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "KEY=VALUE"):
            cli._parse_key_value_args(["invalid"])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            cli._parse_key_value_args([" =value"])

    def test_drop_none_retains_false_and_zero(self):
        self.assertEqual(
            cli._drop_none({"none": None, "false": False, "zero": 0}),
            {"false": False, "zero": 0},
        )


class CLIOutputFormattingTests(unittest.TestCase):
    def test_component_tables_include_names_aliases_and_empty_rows(self):
        table = cli._format_component_table(
            "models",
            [
                {"name": "ModelA", "aliases": ["a", "model-a"]},
                {"name": "ModelB", "aliases": []},
            ],
        )
        empty_table = cli._format_component_table("empty", [])

        self.assertIn("models", table)
        self.assertIn("ModelA", table)
        self.assertIn("a, model-a", table)
        self.assertIn("ModelB", table)
        self.assertIn("| name", empty_table)

    def test_component_listing_detection_requires_detailed_rows(self):
        self.assertTrue(
            cli._is_component_listing(
                {"models": [{"name": "Model", "aliases": []}]}
            )
        )
        self.assertFalse(cli._is_component_listing({"models": ["Model"]}))
        self.assertFalse(cli._is_component_listing({}))

    def test_print_result_formats_tables_paths_and_json_results(self):
        outputs = []
        results = (
            {"models": [{"name": "Model", "aliases": ["alias"]}]},
            (Image.fromarray(sample_image()), Path("result.png")),
            (Image.fromarray(sample_image()), None),
            {"path": Path("output.json"), "values": [1, 2]},
        )

        for result in results:
            stream = io.StringIO()
            with redirect_stdout(stream):
                cli._print_result(result)
            outputs.append(stream.getvalue())

        self.assertIn("Available components", outputs[0])
        self.assertEqual(outputs[1].strip(), "result.png")
        no_save = json.loads(outputs[2])
        self.assertEqual(no_save["image"]["type"], "Image")
        self.assertIsNone(no_save["saved_path"])
        self.assertEqual(json.loads(outputs[3])["path"], "output.json")

    def test_json_safe_summarizes_arrays_without_dumping_pixels(self):
        summary = cli._json_safe(sample_image())

        self.assertEqual(summary["type"], "ndarray")
        self.assertEqual(summary["shape"], [6, 8, 3])
        self.assertEqual(summary["dtype"], "uint8")


class CLIMainIntegrationTests(unittest.TestCase):
    def test_main_lists_components_and_returns_success(self):
        available = {
            "models": [{"name": "Model", "aliases": ["alias"]}],
        }
        stream = io.StringIO()

        with patch.object(cli.llv, "list_available", return_value=available):
            with redirect_stdout(stream):
                exit_code = cli.main(["list"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Available components", stream.getvalue())

    def test_main_runs_real_imwrite_and_predict_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.png"
            output_path = root / "converted.jpg"
            Image.fromarray(sample_image()).save(input_path)

            write_stream = io.StringIO()
            with redirect_stdout(write_stream):
                write_code = cli.main(
                    [
                        "imwrite",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                )

            predict_stream = io.StringIO()
            with redirect_stdout(predict_stream):
                predict_code = cli.main(
                    [
                        "predict",
                        "gamma",
                        str(input_path),
                        "--no-save",
                        "--kwargs",
                        "gamma=1.0",
                    ]
                )

            self.assertEqual(write_code, 0)
            self.assertEqual(predict_code, 0)
            self.assertTrue(output_path.is_file())
            self.assertEqual(write_stream.getvalue().strip(), str(output_path))
            prediction_summary = json.loads(predict_stream.getvalue())
            self.assertEqual(prediction_summary["image"]["type"], "ndarray")
            self.assertIsNone(prediction_summary["saved_path"])


if __name__ == "__main__":
    unittest.main()
