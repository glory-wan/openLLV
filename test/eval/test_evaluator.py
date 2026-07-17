"""Tests for :class:`openLLV.evaluation.Evaluator`."""

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from openLLV.evaluation import BaseMetric, Evaluator


class SilentProgress:
    """Minimal tqdm replacement for deterministic unit-test output."""

    def __init__(self, *args, **kwargs):
        self.updated = 0
        self.postfix = None

    def update(self, value):
        self.updated += value

    def set_postfix(self, value):
        self.postfix = value

    def close(self):
        return None


class SampleDataset:
    def __init__(self, samples, ref_dict=None, paired_files=None):
        self.samples = samples
        self.ref_dict = {} if ref_dict is None else ref_dict
        self.paired_files = (
            [sample[2] for sample in samples]
            if paired_files is None
            else paired_files
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class StaticMetric:
    def __init__(self, value=1.0, requires_reference=True, higher_is_better=True):
        self.value = value
        self.requires_reference = requires_reference
        self.higher_is_better = higher_is_better
        self.calls = []

    def compute(self, enhanced, reference=None):
        self.calls.append((enhanced, reference))
        return self.value


class EvaluatorInitializationTests(unittest.TestCase):
    def test_constructor_uses_default_metrics_and_stores_eval_result(self):
        expected = {"metrics": {}, "filenames": [], "statistics": {}}

        with patch.object(Evaluator, "eval", return_value=expected) as eval_mock:
            with redirect_stdout(io.StringIO()):
                evaluator = Evaluator(
                    en_img_dir="enhanced",
                    device="cpu",
                    num_workers=0,
                )

        self.assertEqual(evaluator.device, torch.device("cpu"))
        self.assertEqual(evaluator.metric_order, ["PSNR", "SSIM"])
        self.assertEqual(evaluator.results, expected)
        eval_mock.assert_called_once_with(
            en_img_dir="enhanced",
            ref_img_dir=None,
            save_path=None,
            num_workers=0,
            batch_size=1,
        )

    def test_constructor_accepts_case_insensitive_string_metric(self):
        with patch.object(Evaluator, "eval", return_value={"ok": True}):
            with redirect_stdout(io.StringIO()):
                evaluator = Evaluator(
                    "enhanced",
                    metrics="psnr",
                    device="cpu",
                    num_workers=0,
                )

        self.assertEqual(evaluator.metric_order, ["PSNR"])
        self.assertIn("PSNR", evaluator.metric_instances)

    def test_constructor_preserves_requested_metric_order(self):
        with patch.object(Evaluator, "eval", return_value={}):
            with redirect_stdout(io.StringIO()):
                evaluator = Evaluator(
                    "enhanced",
                    metrics=["mae", "psnr", "mse"],
                    device="cpu",
                    num_workers=0,
                )

        self.assertEqual(evaluator.metric_order, ["MAE", "PSNR", "MSE"])

    def test_constructor_rejects_invalid_metrics_type(self):
        with self.assertRaisesRegex(TypeError, "Invalid type"):
            Evaluator("enhanced", metrics=("PSNR",), device="cpu")

    def test_constructor_warns_and_skips_unknown_metric(self):
        with patch.object(Evaluator, "eval", return_value={}):
            with self.assertWarnsRegex(UserWarning, "does not exist"):
                with redirect_stdout(io.StringIO()):
                    evaluator = Evaluator(
                        "enhanced",
                        metrics="unknown",
                        device="cpu",
                        num_workers=0,
                    )

        self.assertEqual(evaluator.metric_order, [])
        self.assertEqual(evaluator.metric_instances, {})

    def test_constructor_warns_and_skips_metric_creation_failure(self):
        with patch.object(
            BaseMetric,
            "create_metric",
            side_effect=RuntimeError("creation failed"),
        ):
            with patch.object(Evaluator, "eval", return_value={}):
                with self.assertWarnsRegex(UserWarning, "creation failed"):
                    with redirect_stdout(io.StringIO()):
                        evaluator = Evaluator(
                            "enhanced",
                            metrics="PSNR",
                            device="cpu",
                            num_workers=0,
                        )

        self.assertEqual(evaluator.metric_order, [])


class EvaluatorWorkflowTests(unittest.TestCase):
    def _bare_evaluator(self):
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.device = torch.device("cpu")
        evaluator.metric_instances = {}
        evaluator.metric_order = []
        return evaluator

    def test_eval_builds_dataset_evaluates_and_saves(self):
        evaluator = self._bare_evaluator()
        dataset = SimpleNamespace(
            en_files=["enhanced/a.png"],
            ref_files=["reference/a.png"],
            paired_files=["enhanced/a.png"],
            ref_dict={"a": "reference/a.png"},
        )
        expected = {
            "metrics": {"PSNR": {"a.png": 1.0}},
            "filenames": ["a.png"],
            "statistics": {},
        }

        with patch(
            "openLLV.evaluation.evaluator.EvaluateDataset",
            return_value=dataset,
        ) as dataset_mock:
            with patch.object(
                evaluator,
                "evaluate_dataset",
                return_value=expected,
            ) as evaluate_mock:
                with patch.object(evaluator, "save_results") as save_mock:
                    with redirect_stdout(io.StringIO()):
                        result = evaluator.eval(
                            "enhanced",
                            ref_img_dir="reference",
                            save_path="result.json",
                            batch_size=2,
                            num_workers=0,
                        )

        self.assertIs(result, expected)
        dataset_mock.assert_called_once_with(
            en_img_dir="enhanced",
            ref_img_dir="reference",
        )
        evaluate_mock.assert_called_once_with(
            dataset=dataset,
            batch_size=2,
            num_workers=0,
        )
        save_mock.assert_called_once_with(expected, save_path="result.json")

    def test_get_dataset_filenames_uses_paired_file_basenames(self):
        dataset = SimpleNamespace(
            paired_files=["folder/a.png", Path("other") / "b.jpg"]
        )

        names = Evaluator._get_dataset_filenames(dataset)

        self.assertEqual(names, ["a.png", "b.jpg"])

    def test_get_dataset_filenames_falls_back_to_indices(self):
        class LengthOnlyDataset:
            def __len__(self):
                return 3

        self.assertEqual(
            Evaluator._get_dataset_filenames(LengthOnlyDataset()),
            ["0", "1", "2"],
        )

    def test_collate_fn_stacks_enhanced_and_reference_images(self):
        batch = [
            (torch.zeros(3, 2, 2), torch.ones(3, 2, 2), "a.png"),
            (torch.ones(3, 2, 2), torch.zeros(3, 2, 2), "b.png"),
        ]

        enhanced, reference, names = Evaluator.collate_fn(batch)

        self.assertEqual(enhanced.shape, (2, 3, 2, 2))
        self.assertEqual(reference.shape, (2, 3, 2, 2))
        self.assertEqual(names, ["a.png", "b.png"])

    def test_collate_fn_keeps_reference_batch_none(self):
        batch = [
            (torch.zeros(3, 2, 2), None, "a.png"),
            (torch.ones(3, 2, 2), None, "b.png"),
        ]

        enhanced, reference, names = Evaluator.collate_fn(batch)

        self.assertEqual(enhanced.shape, (2, 3, 2, 2))
        self.assertIsNone(reference)
        self.assertEqual(names, ["a.png", "b.png"])

    def test_compute_metric_for_dataset_handles_success_and_failure(self):
        evaluator = self._bare_evaluator()
        dataset = SampleDataset(
            [
                (torch.zeros(3, 2, 2), None, "ok.png"),
                (torch.ones(3, 2, 2), None, 123),
            ]
        )

        def compute(enhanced, reference):
            if enhanced.mean().item() > 0.5:
                raise RuntimeError("sample failed")
            return 4.0

        metric = Mock()
        metric.compute.side_effect = compute
        evaluator.metric_instances = {"FAKE": metric}

        with patch("openLLV.evaluation.evaluator.tqdm", SilentProgress):
            with self.assertWarnsRegex(UserWarning, "sample failed"):
                values = evaluator._compute_metric_for_dataset(
                    dataset,
                    "FAKE",
                    batch_size=2,
                    num_workers=0,
                )

        self.assertEqual(values["ok.png"], 4.0)
        self.assertTrue(math.isnan(values["123"]))
        self.assertEqual(metric.compute.call_count, 2)

    def test_evaluate_dataset_skips_missing_reference_and_runs_no_ref_metric(self):
        evaluator = self._bare_evaluator()
        evaluator.metric_order = ["REF", "NOREF"]
        evaluator.metric_instances = {
            "REF": StaticMetric(requires_reference=True, higher_is_better=True),
            "NOREF": StaticMetric(requires_reference=False, higher_is_better=False),
        }
        dataset = SampleDataset(
            [(torch.zeros(3, 2, 2), None, "a.png")],
            ref_dict={},
            paired_files=["folder/a.png"],
        )

        with patch.object(
            evaluator,
            "_compute_metric_for_dataset",
            return_value={"a.png": 2.0},
        ) as compute_mock:
            with patch.object(evaluator, "print_final_summary") as summary_mock:
                with redirect_stdout(io.StringIO()):
                    results = evaluator.evaluate_dataset(
                        dataset,
                        batch_size=1,
                        num_workers=0,
                    )

        self.assertTrue(math.isnan(results["metrics"]["REF"]["a.png"]))
        self.assertEqual(results["statistics"]["REF"]["valid_count"], 0)
        self.assertEqual(results["metrics"]["NOREF"], {"a.png": 2.0})
        self.assertEqual(results["statistics"]["NOREF"]["better"], "↓")
        compute_mock.assert_called_once_with(
            dataset=dataset,
            metric_name="NOREF",
            batch_size=1,
            num_workers=0,
        )
        summary_mock.assert_called_once_with(results=results)

    def test_compute_metric_statistics_ignores_nan(self):
        stats = Evaluator._compute_metric_statistics(
            {"a": 1.0, "b": 3.0, "bad": float("nan")},
            "↑",
        )

        self.assertEqual(stats["mean"], 2.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertEqual(stats["std"], 1.0)
        self.assertEqual(stats["valid_count"], 2)
        self.assertEqual(stats["total_count"], 3)
        self.assertEqual(stats["better"], "↑")

    def test_compute_metric_statistics_handles_no_valid_values(self):
        stats = Evaluator._compute_metric_statistics(
            {"bad": float("nan")},
            "↓",
        )

        for key in ("mean", "min", "max", "std"):
            self.assertTrue(math.isnan(stats[key]))
        self.assertEqual(stats["valid_count"], 0)
        self.assertEqual(stats["total_count"], 1)

    def test_print_final_summary_includes_metric_name(self):
        evaluator = self._bare_evaluator()
        results = {
            "statistics": {
                "PSNR": {
                    "better": "↑",
                    "mean": 10.0,
                    "std": 1.0,
                    "min": 9.0,
                    "max": 11.0,
                }
            }
        }

        output = io.StringIO()
        with redirect_stdout(output):
            evaluator.print_final_summary(results)

        self.assertIn("Evaluation completed", output.getvalue())
        self.assertIn("PSNR", output.getvalue())

    def test_make_serializable_handles_nested_and_special_values(self):
        evaluator = self._bare_evaluator()

        value = evaluator._make_serializable(
            {
                "device": torch.device("cpu"),
                "nested": [1, True, None, torch.tensor([1, 2])],
            }
        )

        self.assertEqual(value["device"], "cpu")
        self.assertEqual(value["nested"][:3], [1, True, None])
        self.assertEqual(value["nested"][3], "tensor([1, 2])")

    def test_save_results_writes_expected_json_schema(self):
        evaluator = self._bare_evaluator()
        evaluator.metric_instances = {"PSNR": StaticMetric()}
        results = {
            "filenames": ["a.png"],
            "metrics": {"PSNR": {"a.png": 10.0}},
            "statistics": {"PSNR": {"mean": 10.0}},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "nested" / "result.json"
            with redirect_stdout(io.StringIO()):
                evaluator.save_results(results, save_path)

            payload = json.loads(save_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["metadata"]["device"], "cpu")
        self.assertEqual(payload["metadata"]["metrics"], ["PSNR"])
        self.assertEqual(payload["metadata"]["total_images"], 1)
        self.assertEqual(payload["filenames"], ["a.png"])
        self.assertEqual(payload["values"]["PSNR"]["a.png"], 10.0)

    def test_list_available_metrics_matches_registry(self):
        self.assertEqual(
            Evaluator.list_available_metrics(),
            BaseMetric.list_available_metrics(simple_names=True),
        )


class EvaluatorIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_image(directory, name, value):
        directory.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (value, value, value)).save(directory / name)

    def test_folder_evaluation_computes_metrics_and_saves_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            enhanced_dir = root / "enhanced"
            reference_dir = root / "reference"
            save_path = root / "results" / "metrics.json"
            self._write_image(enhanced_dir, "a.png", 64)
            self._write_image(reference_dir, "a.png", 64)
            self._write_image(enhanced_dir, "b.png", 128)
            self._write_image(reference_dir, "b.png", 128)

            with patch(
                "openLLV.data.utils.tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ):
                with patch("openLLV.evaluation.evaluator.tqdm", SilentProgress):
                    with redirect_stdout(io.StringIO()):
                        evaluator = Evaluator(
                            en_img_dir=str(enhanced_dir),
                            ref_img_dir=str(reference_dir),
                            metrics=["PSNR", "SSIM"],
                            save_path=save_path,
                            device="cpu",
                            batch_size=2,
                            num_workers=0,
                        )

            payload = json.loads(save_path.read_text(encoding="utf-8"))

        self.assertEqual(set(evaluator.results["filenames"]), {"a.png", "b.png"})
        self.assertEqual(set(evaluator.results["metrics"]), {"PSNR", "SSIM"})
        self.assertEqual(evaluator.results["statistics"]["PSNR"]["valid_count"], 2)
        self.assertTrue(
            all(math.isinf(value) for value in evaluator.results["metrics"]["PSNR"].values())
        )
        self.assertEqual(payload["metadata"]["total_images"], 2)

    def test_reference_metric_produces_nan_without_reference_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            enhanced_dir = root / "enhanced"
            save_path = root / "metrics.json"
            self._write_image(enhanced_dir, "sample.png", 80)

            with patch(
                "openLLV.data.utils.tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ):
                with patch("openLLV.evaluation.evaluator.tqdm", SilentProgress):
                    with redirect_stdout(io.StringIO()):
                        evaluator = Evaluator(
                            en_img_dir=str(enhanced_dir),
                            metrics="PSNR",
                            save_path=save_path,
                            device="cpu",
                            num_workers=0,
                        )

        value = evaluator.results["metrics"]["PSNR"]["sample.png"]
        self.assertTrue(math.isnan(value))
        self.assertEqual(evaluator.results["statistics"]["PSNR"]["valid_count"], 0)


if __name__ == "__main__":
    unittest.main()
