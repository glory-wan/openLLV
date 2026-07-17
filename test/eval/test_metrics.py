"""Tests for all metrics exposed by :mod:`openLLV.evaluation.metrics`."""

import math
import sys
import types
import unittest
import warnings
from unittest.mock import Mock, patch

import torch

import openLLV.evaluation as evaluation
from openLLV.evaluation import metrics as metrics_module
from openLLV.evaluation.metrics import (
    LOEMetric,
    LPIPSMetric,
    MAEMetric,
    MSEMetric,
    MUSIQMetric,
    NIQEMetric,
    PIMetric,
    PSNRMetric,
    SSIMMetric,
)


class RecordingModel:
    """Small callable used instead of downloading pyiqa model weights."""

    def __init__(self, score=2.5, error=None):
        self.score = score
        self.error = error
        self.calls = []

    def __call__(self, *inputs):
        self.calls.append(inputs)
        if self.error is not None:
            raise self.error
        return self.score


def fake_pyiqa(create_metric):
    module = types.ModuleType("pyiqa")
    module.create_metric = create_metric
    return module


class MetricHelperTests(unittest.TestCase):
    def test_score_to_float_supports_common_pyiqa_outputs(self):
        cases = [
            (torch.tensor([1.0, 3.0]), 2.0),
            ({"score": torch.tensor([4.0])}, 4.0),
            ({"quality": 5.0}, 5.0),
            ({"custom": 6.0}, 6.0),
            ((torch.tensor(7.0), "ignored"), 7.0),
            ([8.0, 9.0], 8.0),
            (10, 10.0),
        ]

        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(metrics_module._score_to_float(score), expected)

    def test_prepare_pyiqa_input_batches_scales_and_clamps(self):
        image = torch.tensor([[[-255.0, 0.0, 255.0, 510.0]]])

        prepared = metrics_module._prepare_pyiqa_input(
            image,
            data_range=255.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(prepared.shape, (1, 1, 1, 4))
        torch.testing.assert_close(
            prepared,
            torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]]]),
        )

    def test_compute_pyiqa_score_returns_float(self):
        model = RecordingModel(score={"mos": torch.tensor([2.0, 4.0])})

        value = metrics_module._compute_pyiqa_score(
            model,
            torch.full((3, 4, 4), 255.0),
            data_range=255.0,
            device=torch.device("cpu"),
            display_name="FAKE",
        )

        self.assertEqual(value, 3.0)
        self.assertEqual(model.calls[0][0].shape, (1, 3, 4, 4))
        self.assertEqual(model.calls[0][0].max().item(), 1.0)

    def test_compute_pyiqa_score_warns_and_returns_nan_on_backend_error(self):
        model = RecordingModel(error=RuntimeError("backend failed"))

        with self.assertWarnsRegex(UserWarning, "backend failed"):
            value = metrics_module._compute_pyiqa_score(
                model,
                torch.zeros(1, 3, 4, 4),
                data_range=1.0,
                device=torch.device("cpu"),
                display_name="FAKE",
            )

        self.assertTrue(math.isnan(value))

    def test_create_pyiqa_metric_uses_requested_name_and_device(self):
        model = RecordingModel()
        create_metric = Mock(return_value=model)

        with patch.dict(sys.modules, {"pyiqa": fake_pyiqa(create_metric)}):
            result = metrics_module._create_pyiqa_metric(
                "niqe",
                torch.device("cpu"),
                "NIQE",
            )

        self.assertIs(result, model)
        create_metric.assert_called_once_with("niqe", device=torch.device("cpu"))

    def test_create_pyiqa_metric_has_clear_missing_dependency_error(self):
        with patch.dict(sys.modules, {"pyiqa": None}):
            with self.assertRaisesRegex(ImportError, "pyiqa"):
                metrics_module._create_pyiqa_metric(
                    "niqe",
                    torch.device("cpu"),
                    "NIQE",
                )

    def test_create_pyiqa_metric_wraps_backend_creation_error(self):
        create_metric = Mock(side_effect=ValueError("invalid backend"))

        with patch.dict(sys.modules, {"pyiqa": fake_pyiqa(create_metric)}):
            with self.assertRaisesRegex(RuntimeError, "invalid backend"):
                metrics_module._create_pyiqa_metric(
                    "niqe",
                    torch.device("cpu"),
                    "NIQE",
                )


class FullReferenceMetricTests(unittest.TestCase):
    def test_identical_images_have_ideal_reference_scores(self):
        image = torch.rand(1, 3, 16, 16)

        self.assertTrue(math.isinf(PSNRMetric(device="cpu").compute(image, image)))
        self.assertAlmostEqual(
            SSIMMetric(device="cpu").compute(image, image),
            1.0,
            places=5,
        )
        self.assertEqual(MSEMetric(device="cpu").compute(image, image), 0.0)
        self.assertEqual(MAEMetric(device="cpu").compute(image, image), 0.0)

    def test_reference_metrics_match_known_values(self):
        enhanced = torch.zeros(3, 8, 8)
        reference = torch.full((3, 8, 8), 0.5)

        self.assertAlmostEqual(
            PSNRMetric(device="cpu").compute(enhanced, reference),
            6.0205999,
            places=5,
        )
        self.assertAlmostEqual(
            MSEMetric(device="cpu").compute(enhanced, reference),
            0.25,
            places=7,
        )
        self.assertAlmostEqual(
            MAEMetric(device="cpu").compute(enhanced, reference),
            0.5,
            places=7,
        )

    def test_ssim_is_lower_for_different_images(self):
        dark = torch.zeros(1, 3, 16, 16)
        bright = torch.ones(1, 3, 16, 16)

        value = SSIMMetric(device="cpu").compute(dark, bright)

        self.assertLess(value, 1.0)
        self.assertGreaterEqual(value, 0.0)

    def test_metric_directions(self):
        self.assertTrue(PSNRMetric(device="cpu").higher_is_better)
        self.assertTrue(SSIMMetric(device="cpu").higher_is_better)
        self.assertFalse(MSEMetric(device="cpu").higher_is_better)
        self.assertFalse(MAEMetric(device="cpu").higher_is_better)


class LOEMetricTests(unittest.TestCase):
    def test_identical_lightness_order_has_zero_error(self):
        image = torch.linspace(0, 1, steps=16).reshape(1, 1, 4, 4)

        value = LOEMetric(device="cpu", patch_size=1).compute(image, image)

        self.assertEqual(value, 0.0)

    def test_reversed_two_pixel_order_has_expected_error(self):
        reference = torch.tensor([[[[0.0, 1.0]]]])
        enhanced = torch.tensor([[[[1.0, 0.0]]]])

        value = LOEMetric(device="cpu", patch_size=1).compute(
            enhanced,
            reference,
        )

        self.assertEqual(value, 0.5)

    def test_rgb_to_gray_uses_maximum_rgb_channel(self):
        metric = LOEMetric(device="cpu")
        image = torch.tensor([[[[0.1]], [[0.8]], [[0.3]]]])

        gray = metric._rgb_to_gray(image)

        self.assertEqual(gray.shape, (1, 1, 1, 1))
        self.assertAlmostEqual(gray.item(), 0.8, places=6)

    def test_extract_patches_uses_requested_patch_scale(self):
        metric = LOEMetric(device="cpu", patch_size=4)

        patches = metric._extract_patches(torch.ones(1, 1, 9, 10))

        self.assertEqual(patches.shape, (1, 9))

    def test_reference_is_required(self):
        with self.assertRaisesRegex(ValueError, "LOE"):
            LOEMetric(device="cpu").compute(torch.ones(1, 3, 4, 4))

    def test_lower_loe_is_better(self):
        self.assertFalse(LOEMetric(device="cpu").higher_is_better)


class LPIPSMetricTests(unittest.TestCase):
    def test_lpips_uses_fake_backend_and_normalizes_inputs(self):
        model = RecordingModel(score=torch.tensor([0.25]))
        create_metric = Mock(return_value=model)

        with patch.dict(sys.modules, {"pyiqa": fake_pyiqa(create_metric)}):
            metric = LPIPSMetric(
                device="cpu",
                data_range=255.0,
                net="vgg",
            )
            value = metric.compute(
                torch.full((3, 4, 4), 510.0),
                torch.zeros(3, 4, 4),
            )

        self.assertEqual(value, 0.25)
        create_metric.assert_called_once_with(
            "lpips",
            device=torch.device("cpu"),
            net="vgg",
        )
        enhanced_input, reference_input = model.calls[0]
        self.assertEqual(enhanced_input.max().item(), 1.0)
        self.assertEqual(reference_input.min().item(), 0.0)
        self.assertFalse(metric.higher_is_better)

    def test_lpips_falls_back_when_backend_rejects_net_argument(self):
        model = RecordingModel()
        create_metric = Mock(side_effect=[TypeError("no net"), model])

        with patch.dict(sys.modules, {"pyiqa": fake_pyiqa(create_metric)}):
            metric = LPIPSMetric(device="cpu", net="alex")

        self.assertIs(metric._metric_model, model)
        self.assertEqual(create_metric.call_count, 2)
        create_metric.assert_called_with("lpips", device=torch.device("cpu"))

    def test_lpips_requires_reference(self):
        model = RecordingModel()
        create_metric = Mock(return_value=model)

        with patch.dict(sys.modules, {"pyiqa": fake_pyiqa(create_metric)}):
            metric = LPIPSMetric(device="cpu")
            with self.assertRaisesRegex(ValueError, "LPIPS"):
                metric.compute(torch.zeros(1, 3, 4, 4))

    def test_lpips_backend_error_warns_and_returns_nan(self):
        model = RecordingModel(error=RuntimeError("lpips failed"))
        create_metric = Mock(return_value=model)

        with patch.dict(sys.modules, {"pyiqa": fake_pyiqa(create_metric)}):
            metric = LPIPSMetric(device="cpu")
            with self.assertWarnsRegex(UserWarning, "lpips failed"):
                value = metric.compute(
                    torch.zeros(1, 3, 4, 4),
                    torch.zeros(1, 3, 4, 4),
                )

        self.assertTrue(math.isnan(value))


class NoReferenceMetricTests(unittest.TestCase):
    def test_no_reference_metrics_select_backend_and_direction(self):
        cases = [
            (NIQEMetric, "niqe", False),
            (MUSIQMetric, "musiq", True),
            (PIMetric, "pi", False),
        ]

        for metric_class, backend_name, higher_is_better in cases:
            with self.subTest(metric=metric_class.__name__):
                model = RecordingModel(score=torch.tensor([3.5]))
                create_metric = Mock(return_value=model)
                module = fake_pyiqa(create_metric)

                with patch.dict(sys.modules, {"pyiqa": module}):
                    metric = metric_class(device="cpu", data_range=255.0)
                    value = metric.compute(torch.full((3, 4, 4), 255.0))

                self.assertEqual(value, 3.5)
                create_metric.assert_called_once_with(
                    backend_name,
                    device=torch.device("cpu"),
                )
                self.assertFalse(metric.requires_reference)
                self.assertEqual(metric.higher_is_better, higher_is_better)
                self.assertEqual(model.calls[0][0].shape, (1, 3, 4, 4))
                self.assertEqual(model.calls[0][0].max().item(), 1.0)

    def test_niqe_recreates_backend_if_model_was_cleared(self):
        first_model = RecordingModel(score=1.0)
        second_model = RecordingModel(score=2.0)
        create_metric = Mock(side_effect=[first_model, second_model])

        with patch.dict(sys.modules, {"pyiqa": fake_pyiqa(create_metric)}):
            metric = NIQEMetric(device="cpu")
            metric._metric_model = None
            value = metric.compute(torch.zeros(1, 3, 4, 4))

        self.assertEqual(value, 2.0)
        self.assertEqual(create_metric.call_count, 2)

    def test_public_api_exports_all_metric_classes(self):
        expected_names = {
            "PSNRMetric",
            "SSIMMetric",
            "MSEMetric",
            "MAEMetric",
            "LPIPSMetric",
            "LOEMetric",
            "NIQEMetric",
            "MUSIQMetric",
            "PIMetric",
        }

        for name in expected_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(evaluation, name))

        self.assertTrue(expected_names.issubset(set(metrics_module.__all__)))


if __name__ == "__main__":
    unittest.main()
