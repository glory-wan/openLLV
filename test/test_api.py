"""Tests for the public :mod:`openLLV.api` convenience functions."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import openLLV as llv
from openLLV import api
from openLLV.deepLearning.models import LLVModel


class APIIdentityModel(LLVModel):
    """Small registered model used by top-level prediction tests."""

    task = "api-test"
    aliases = ["api-identity"]

    def _init_model(self):
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, **kwargs):
        return self._format_output(x * self.scale)


def sample_image(value=64):
    """Return a small uint8 RGB image array."""
    return np.full((6, 8, 3), value, dtype=np.uint8)


class TopLevelExportTests(unittest.TestCase):
    def test_core_functions_are_exported_from_package_root(self):
        expected = {
            "predict",
            "enhance",
            "train",
            "evaluate",
            "eval",
            "imread",
            "imwrite",
            "read_image",
            "write_image",
            "list_models",
            "list_algorithms",
            "list_metrics",
            "list_losses",
            "list_datasets",
            "list_available",
        }

        self.assertTrue(expected.issubset(set(llv.__all__)))
        for name in expected:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(llv, name)))

        self.assertIs(llv.read_image, llv.imread)
        self.assertIs(llv.write_image, llv.imwrite)
        self.assertIs(llv.LLVModel, LLVModel)

    def test_predict_keyword_split_includes_current_runtime_options(self):
        predictor_kwargs, call_kwargs = api._split_predict_kwargs(
            {
                "device": "cpu",
                "config": {"width": 8},
                "gamma": 0.8,
                "save": False,
                "output_ext": "png",
                "model_kwargs": {"temperature": 0.5},
                "timeout": 2,
            }
        )

        self.assertEqual(
            predictor_kwargs,
            {
                "device": "cpu",
                "config": {"width": 8},
                "gamma": 0.8,
            },
        )
        self.assertEqual(
            call_kwargs,
            {
                "save": False,
                "output_ext": "png",
                "model_kwargs": {"temperature": 0.5},
                "timeout": 2,
            },
        )


class PredictionAPITests(unittest.TestCase):
    def test_predict_runs_traditional_and_deep_backends(self):
        traditional_output, traditional_path = llv.predict(
            "gamma",
            sample_image(),
            gamma=1.0,
            save=False,
        )
        deep_output, deep_path = llv.predict(
            "api-identity",
            sample_image(),
            device="cpu",
            save=False,
        )

        self.assertIsInstance(traditional_output, np.ndarray)
        self.assertIsNone(traditional_path)
        self.assertIsInstance(deep_output, Image.Image)
        self.assertIsNone(deep_path)

    def test_enhance_delegates_to_predict(self):
        expected = object()

        with patch("openLLV.api.predict", return_value=expected) as predict_mock:
            result = api.enhance(
                "gamma",
                "input.png",
                output="output.png",
                save=False,
            )

        self.assertIs(result, expected)
        predict_mock.assert_called_once_with(
            "gamma",
            "input.png",
            output="output.png",
            save=False,
        )


class TrainingAndEvaluationAPITests(unittest.TestCase):
    def test_train_constructs_trainer_and_returns_training_result(self):
        expected = {"history": [], "best_val_loss": 0.1}

        with patch("openLLV.deepLearning.Trainer") as trainer_class:
            trainer_class.return_value.train.return_value = expected
            result = api.train(
                "ZeroDCE",
                root_dir="dataset",
                epochs=1,
            )

        self.assertIs(result, expected)
        trainer_class.assert_called_once_with(
            "ZeroDCE",
            root_dir="dataset",
            epochs=1,
        )
        trainer_class.return_value.train.assert_called_once_with()

    def test_evaluate_supports_primary_arguments_and_aliases(self):
        evaluator = SimpleNamespace(results={"statistics": {}})

        with patch(
            "openLLV.evaluation.Evaluator",
            return_value=evaluator,
        ) as evaluator_class:
            results = api.evaluate(
                en=Path("enhanced"),
                ref=Path("reference"),
                metrics="PSNR",
                batch_size=2,
            )

        self.assertIs(results, evaluator.results)
        evaluator_class.assert_called_once_with(
            en_img_dir="enhanced",
            ref_img_dir="reference",
            metrics="PSNR",
            save_path=None,
            batch_size=2,
        )

    def test_evaluate_can_return_evaluator_and_eval_is_an_alias(self):
        evaluator = SimpleNamespace(results={"metrics": {}})

        with patch("openLLV.evaluation.Evaluator", return_value=evaluator):
            returned = api.evaluate(
                "enhanced",
                return_evaluator=True,
            )
        self.assertIs(returned, evaluator)

        expected = object()
        with patch("openLLV.api.evaluate", return_value=expected) as evaluate_mock:
            result = api.eval("enhanced", metrics=["NIQE"])
        self.assertIs(result, expected)
        evaluate_mock.assert_called_once_with("enhanced", metrics=["NIQE"])

    def test_evaluate_rejects_missing_or_duplicate_alias_arguments(self):
        with self.assertRaisesRegex(TypeError, "missing required argument"):
            api.evaluate()
        with self.assertRaisesRegex(TypeError, "both 'en_img_dir'"):
            api.evaluate("enhanced", en="other")
        with self.assertRaisesRegex(TypeError, "both 'ref_img_dir'"):
            api.evaluate("enhanced", "reference", ref="other")


class ImageIOAPITests(unittest.TestCase):
    def test_top_level_image_read_and_write_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.png"
            Image.fromarray(sample_image()).save(input_path)

            pil_image = llv.imread(input_path, output_format="pil")
            numpy_image = llv.read_image(input_path, output_format="numpy")
            saved_path = llv.imwrite(
                pil_image,
                output=root / "written",
                output_name="roundtrip.jpg",
            )

            self.assertIsInstance(pil_image, Image.Image)
            self.assertIsInstance(numpy_image, np.ndarray)
            self.assertEqual(pil_image.size, (8, 6))
            self.assertEqual(numpy_image.shape, (6, 8, 3))
            self.assertEqual(saved_path.name, "roundtrip.jpg")
            self.assertTrue(saved_path.is_file())


class ComponentListingAPITests(unittest.TestCase):
    def test_flat_lists_contain_current_registered_components(self):
        self.assertIn("zerodce", llv.list_models())
        self.assertIn("api-identity", llv.list_models())
        self.assertIn("gamma", llv.list_algorithms())
        self.assertIn("psnr", {name.lower() for name in llv.list_metrics()})
        self.assertIn("l1", llv.list_losses())
        self.assertIn("commondataset", llv.list_datasets())

    def test_grouped_listing_deduplicates_classes_and_preserves_aliases(self):
        available = llv.list_available()

        self.assertEqual(
            set(available),
            {"models", "algorithms", "metrics", "losses", "datasets"},
        )
        for category, rows in available.items():
            with self.subTest(category=category):
                names = [row["name"] for row in rows]
                self.assertEqual(len(names), len(set(names)))
                self.assertTrue(
                    all(set(row) == {"name", "aliases"} for row in rows)
                )

        api_model = next(
            row
            for row in available["models"]
            if row["name"] == "APIIdentityModel"
        )
        self.assertEqual(api_model["aliases"], ["api-identity"])


if __name__ == "__main__":
    unittest.main()
