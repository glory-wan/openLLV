"""Tests for :mod:`openLLV.predictor`."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import openLLV
from openLLV.deepLearning.models import LLVModel
from openLLV.deepLearning.predictor import Predictor as DeepLearningPredictor
from openLLV.predictor import Predictor
from openLLV.tradition.algorithms import Gamma, LLVEnhancer
from openLLV.tradition.predictor import Predictor as TraditionalPredictor


class UnifiedIdentityModel(LLVModel):
    """Small registered model used to exercise unified deep prediction."""

    task = "unified-predictor-test"
    aliases = ["unified-identity"]

    def _get_default_config(self):
        config = super()._get_default_config()
        config["marker"] = "default"
        return config

    def _init_model(self):
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, **kwargs):
        return self._format_output(x * self.scale)


def sample_image(value=64):
    """Create a small RGB-like uint8 image array."""
    return np.full((6, 8, 3), value, dtype=np.uint8)


def write_image(path, value=64):
    """Write a small image and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sample_image(value)).save(path)
    return path


class UnifiedPredictorInitializationTests(unittest.TestCase):
    def test_top_level_package_exports_unified_predictor(self):
        self.assertIs(openLLV.Predictor, Predictor)

    def test_auto_routes_registered_model_and_algorithm_names(self):
        deep = Predictor("UNIFIED-IDENTITY", device="cpu")
        traditional = Predictor("GaMmA")

        self.assertEqual(deep.backend, "deep")
        self.assertIsInstance(deep.predictor, DeepLearningPredictor)
        self.assertIsInstance(deep.predictor.model, UnifiedIdentityModel)
        self.assertEqual(traditional.backend, "traditional")
        self.assertIsInstance(traditional.predictor, TraditionalPredictor)
        self.assertIsInstance(traditional.predictor.enhancer, Gamma)

    def test_auto_routes_model_and_enhancer_instances(self):
        model = UnifiedIdentityModel()
        enhancer = Gamma()

        deep = Predictor(model, device="cpu")
        traditional = Predictor(enhancer)

        self.assertIs(deep.predictor.model, model)
        self.assertIs(traditional.predictor.enhancer, enhancer)

    def test_explicit_model_method_and_backend_aliases(self):
        deep = Predictor(model="unified-identity", backend="DL", device="cpu")
        traditional = Predictor(method="gamma", backend="tradition")

        self.assertEqual(deep.backend, "deep")
        self.assertEqual(traditional.backend, "traditional")

    def test_configuration_is_copied_and_keyword_values_take_precedence(self):
        deep_config = {"marker": "config"}
        traditional_config = {"gamma": 0.8}

        deep = Predictor(
            "unified-identity",
            config=deep_config,
            marker="keyword",
            device="cpu",
        )
        traditional = Predictor(
            "gamma",
            config=traditional_config,
            gamma=0.5,
        )

        self.assertEqual(deep.predictor.model.config["marker"], "keyword")
        self.assertEqual(traditional.predictor.enhancer.gamma, 0.5)
        self.assertEqual(deep_config, {"marker": "config"})
        self.assertEqual(traditional_config, {"gamma": 0.8})

    def test_checkpoint_suffix_is_routed_to_deep_backend(self):
        model = UnifiedIdentityModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            with redirect_stdout(io.StringIO()):
                checkpoint = model.save_model(temp_dir)
                predictor = Predictor(checkpoint, device="cpu")

        self.assertEqual(predictor.backend, "deep")
        self.assertIsInstance(predictor.predictor.model, UnifiedIdentityModel)

    def test_available_items_and_parameters_are_reported(self):
        predictor = Predictor("gamma", output_dir="unified-output")

        available = Predictor.list_available()
        params = predictor.get_params()

        self.assertIn("unified-identity", available["models"])
        self.assertIn("gamma", available["algorithms"])
        self.assertEqual(params["backend"], "traditional")
        self.assertEqual(params["predictor"]["method"], "gamma")
        self.assertEqual(params["predictor"]["output_dir"], "unified-output")

    def test_conflicting_and_missing_selectors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "only one"):
            Predictor(model="unified-identity", method="gamma")
        with self.assertRaisesRegex(ValueError, "either positional"):
            Predictor("gamma", method="gamma")
        with self.assertRaisesRegex(ValueError, "required"):
            Predictor()

    def test_invalid_backend_config_and_unknown_target_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "config must"):
            Predictor("gamma", config="invalid")
        with self.assertRaisesRegex(TypeError, "backend must"):
            Predictor("gamma", backend=None)
        with self.assertRaisesRegex(ValueError, "Unsupported backend"):
            Predictor("gamma", backend="other")
        with self.assertRaisesRegex(ValueError, "Cannot infer"):
            Predictor("not-a-model-or-method")

    def test_incompatible_instances_are_rejected_by_explicit_backend(self):
        with self.assertRaisesRegex(TypeError, "traditional backend"):
            Predictor(Gamma(), backend="deep", device="cpu")
        with self.assertRaisesRegex(TypeError, "deep backend"):
            Predictor(UnifiedIdentityModel(), backend="traditional")

    def test_ambiguous_registered_name_requires_an_explicit_backend(self):
        with patch.object(
            DeepLearningPredictor,
            "list_available_models",
            return_value=["collision"],
        ), patch.object(
            TraditionalPredictor,
            "list_available_methods",
            return_value=["collision"],
        ):
            with self.assertRaisesRegex(ValueError, "both a deep-learning"):
                Predictor("collision")


class UnifiedPredictorInferenceTests(unittest.TestCase):
    def test_single_image_prediction_uses_each_backend_contract(self):
        deep = Predictor("unified-identity", device="cpu")
        traditional = Predictor("gamma")

        deep_output, deep_path = deep.predict_single(sample_image(), save=False)
        traditional_output, traditional_path = traditional.predict_single(
            sample_image(),
            save=False,
        )

        self.assertIsInstance(deep_output, Image.Image)
        self.assertIsNone(deep_path)
        self.assertIsInstance(traditional_output, np.ndarray)
        self.assertIsNone(traditional_path)

    def test_predict_and_call_delegate_to_selected_backend(self):
        predictor = Predictor("gamma")

        via_predict, _ = predictor.predict(sample_image(), save=False)
        via_call, _ = predictor(sample_image(), save=False)

        np.testing.assert_array_equal(via_predict, via_call)

    def test_directory_prediction_works_for_both_backends(self):
        deep = Predictor("unified-identity", device="cpu")
        traditional = Predictor("gamma")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            write_image(input_dir / "a.png", 32)
            write_image(input_dir / "nested" / "b.jpg", 96)

            deep_paths = deep.predict_batch(
                input_dir,
                root / "deep-output",
                progress_bar=False,
            )
            traditional_paths = traditional.predict_batch(
                input_dir,
                root / "traditional-output",
                progress_bar=False,
            )

            self.assertEqual(len(deep_paths), 2)
            self.assertEqual(len(traditional_paths), 2)
            self.assertTrue(all(path.is_file() for path in deep_paths))
            self.assertTrue(all(path.is_file() for path in traditional_paths))


class UnifiedPredictorTypeTests(unittest.TestCase):
    def test_public_base_types_remain_the_backend_contract(self):
        self.assertTrue(issubclass(UnifiedIdentityModel, LLVModel))
        self.assertTrue(issubclass(Gamma, LLVEnhancer))


if __name__ == "__main__":
    unittest.main()
