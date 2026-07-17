"""Tests for :mod:`openLLV.evaluation.baseMetric`."""

import unittest

import torch

from openLLV.evaluation import BaseMetric
from openLLV.evaluation.metrics import PSNRMetric


class BaseMetricTests(unittest.TestCase):
    """Exercise input preparation, registration, and metric construction."""

    def setUp(self):
        self._registry_snapshot = dict(BaseMetric._metric_registry)

    def tearDown(self):
        BaseMetric._metric_registry.clear()
        BaseMetric._metric_registry.update(self._registry_snapshot)

    @staticmethod
    def _recording_metric():
        class RecordingMetric(BaseMetric):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.received = None

            def _compute_impl(self, enImg, Refer):
                self.received = (enImg, Refer)
                return 12.5

        return RecordingMetric

    def test_abstract_base_class_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            BaseMetric(device="cpu")

    def test_initialization_sets_device_name_and_config(self):
        class ExampleMetric(BaseMetric):
            def _compute_impl(self, enImg, Refer):
                return 0.0

        metric = ExampleMetric(device="cpu", alpha=3)

        self.assertEqual(metric.device, torch.device("cpu"))
        self.assertEqual(metric.name, "Example")
        self.assertEqual(metric.config, {"alpha": 3})
        self.assertTrue(metric.requires_reference)
        self.assertTrue(metric.higher_is_better)

    def test_compute_adds_batch_dimension_and_delegates(self):
        metric = self._recording_metric()(device="cpu")
        enhanced = torch.ones(3, 4, 5)
        reference = torch.zeros(3, 4, 5)

        value = metric.compute(enhanced, reference)

        self.assertEqual(value, 12.5)
        self.assertEqual(metric.received[0].shape, (1, 3, 4, 5))
        self.assertEqual(metric.received[1].shape, (1, 3, 4, 5))
        self.assertEqual(metric.received[0].device.type, "cpu")

    def test_compute_accepts_no_reference(self):
        metric = self._recording_metric()(device="cpu")

        metric.compute(torch.ones(1, 3, 4, 4))

        self.assertIsNone(metric.received[1])

    def test_prepare_inputs_resizes_enhanced_image_to_reference_size(self):
        metric = self._recording_metric()(device="cpu")
        enhanced = torch.ones(1, 3, 4, 6)
        reference = torch.zeros(1, 3, 8, 10)

        metric.compute(enhanced, reference)

        self.assertEqual(metric.received[0].shape, reference.shape)
        self.assertEqual(metric.received[1].shape, reference.shape)

    def test_prepare_inputs_rejects_non_tensor_enhanced_image(self):
        metric = self._recording_metric()(device="cpu")

        with self.assertRaisesRegex(TypeError, "enImg must be torch.Tensor"):
            metric.compute([1, 2, 3])

    def test_prepare_inputs_rejects_non_tensor_reference(self):
        metric = self._recording_metric()(device="cpu")

        with self.assertRaisesRegex(TypeError, "Refer must be torch.Tensor or None"):
            metric.compute(torch.ones(3, 4, 4), "reference")

    def test_prepare_inputs_rejects_channel_mismatch_after_resize(self):
        metric = self._recording_metric()(device="cpu")

        with self.assertRaisesRegex(ValueError, "shapes still mismatch"):
            metric.compute(
                torch.ones(1, 1, 4, 4),
                torch.ones(1, 3, 8, 8),
            )

    def test_prepare_inputs_rejects_batch_mismatch(self):
        metric = self._recording_metric()(device="cpu")

        with self.assertRaisesRegex(ValueError, "shapes still mismatch"):
            metric.compute(
                torch.ones(2, 3, 4, 4),
                torch.ones(1, 3, 4, 4),
            )

    def test_subclasses_are_registered_automatically(self):
        class AutomaticallyRegisteredMetric(BaseMetric):
            def _compute_impl(self, enImg, Refer):
                return 0.0

        self.assertIs(
            BaseMetric._metric_registry["AutomaticallyRegisteredMetric"],
            AutomaticallyRegisteredMetric,
        )

    def test_register_supports_manual_registration(self):
        class ManuallyRegisteredMetric:
            def __init__(self, marker=None, **kwargs):
                self.marker = marker

        registered = BaseMetric.register(ManuallyRegisteredMetric)
        instance = BaseMetric.create_metric("manuallyregistered", marker="ok")

        self.assertIs(registered, ManuallyRegisteredMetric)
        self.assertIsInstance(instance, ManuallyRegisteredMetric)
        self.assertEqual(instance.marker, "ok")

    def test_list_available_metrics_supports_simple_and_class_names(self):
        simple_names = BaseMetric.list_available_metrics()
        class_names = BaseMetric.list_available_metrics(simple_names=False)

        self.assertIn("PSNR", simple_names)
        self.assertIn("PSNRMetric", class_names)
        self.assertNotIn("PSNRMetric", simple_names)

    def test_create_metric_is_case_insensitive_and_suffix_is_optional(self):
        without_suffix = BaseMetric.create_metric("psnr", device="cpu")
        with_suffix = BaseMetric.create_metric("PsNrMeTrIc", device="cpu")

        self.assertIsInstance(without_suffix, PSNRMetric)
        self.assertIsInstance(with_suffix, PSNRMetric)

    def test_create_metric_reports_available_metrics_and_suggestion(self):
        with self.assertRaises(ValueError) as context:
            BaseMetric.create_metric("psn")

        message = str(context.exception)
        self.assertIn("does not exist", message)
        self.assertIn("Available metrics", message)
        self.assertIn("Did you mean", message)
        self.assertIn("psnr", message.lower())


if __name__ == "__main__":
    unittest.main()
