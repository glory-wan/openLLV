"""Tests for :mod:`openLLV.deepLearning.loss.BaseLoss`."""

import unittest

import torch
import torch.nn as nn

import openLLV.deepLearning as deep_learning
from openLLV.deepLearning.loss import BaseLoss


class ExampleSupervisedLoss(BaseLoss):
    name = "example_supervised"
    aliases = ["example", "example_l1"]

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, prediction, target):
        return (prediction - target).abs().mean() * self.scale


class ExampleReferenceFreeLoss(BaseLoss):
    name = "example_reference_free"
    aliases = ["example_nr"]
    requires_target = False

    def forward(self, input_tensor, model_output):
        return (input_tensor - model_output["raw"]).square().mean()


class BaseLossRegistryTests(unittest.TestCase):
    def setUp(self):
        self._registry_snapshot = dict(BaseLoss._loss_registry)

    def tearDown(self):
        BaseLoss._loss_registry.clear()
        BaseLoss._loss_registry.update(self._registry_snapshot)

    def test_base_loss_is_torch_module(self):
        self.assertTrue(issubclass(BaseLoss, nn.Module))

    def test_subclasses_register_class_name_declared_name_and_aliases(self):
        self.assertIs(
            BaseLoss._loss_registry["examplesupervisedloss"],
            ExampleSupervisedLoss,
        )
        self.assertIs(
            BaseLoss._loss_registry["example_supervised"],
            ExampleSupervisedLoss,
        )
        self.assertIs(BaseLoss._loss_registry["example"], ExampleSupervisedLoss)

    def test_registry_key_normalization(self):
        self.assertEqual(BaseLoss._normalize_key("  EXAMPLE  "), "example")

    def test_create_loss_is_case_insensitive_and_forwards_kwargs(self):
        by_name = BaseLoss.create_loss("EXAMPLE_SUPERVISED", scale=2.0)
        by_alias = BaseLoss.create_loss(" Example_L1 ", scale=3.0)

        self.assertIsInstance(by_name, ExampleSupervisedLoss)
        self.assertEqual(by_name.scale, 2.0)
        self.assertIsInstance(by_alias, ExampleSupervisedLoss)
        self.assertEqual(by_alias.scale, 3.0)

    def test_list_registered_losses_is_sorted(self):
        registered = BaseLoss.list_registered_losses()

        self.assertEqual(registered, sorted(registered))
        self.assertIn("example_reference_free", registered)
        self.assertIn("example_supervised", registered)

    def test_manual_registration_returns_and_registers_class(self):
        class ManualLoss(BaseLoss):
            name = "manual"

            def forward(self, prediction, target):
                return torch.tensor(0.0)

        BaseLoss._loss_registry.pop("manual", None)
        BaseLoss._loss_registry.pop("manualloss", None)

        registered = BaseLoss.register(ManualLoss)

        self.assertIs(registered, ManualLoss)
        self.assertIs(BaseLoss._loss_registry["manual"], ManualLoss)

    def test_manual_registration_rejects_non_loss_class(self):
        with self.assertRaisesRegex(TypeError, "inherit BaseLoss"):
            BaseLoss.register(str)

    def test_empty_and_unknown_names_have_clear_errors(self):
        for name in (None, "", "   "):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    BaseLoss.create_loss(name)

        with self.assertRaises(ValueError) as context:
            BaseLoss.create_loss("exampl_supervised")

        message = str(context.exception)
        self.assertIn("Available losses", message)
        self.assertIn("Did you mean", message)
        self.assertIn("example_supervised", message)

    def test_similar_name_helper_supports_match_and_fallback(self):
        self.assertEqual(
            BaseLoss._get_similar_loss_name("exampl", ["example", "other"]),
            "example",
        )
        self.assertEqual(
            BaseLoss._get_similar_loss_name("xyz", ["example"]),
            "No similar losses found",
        )

    def test_public_packages_export_base_and_task_losses(self):
        self.assertIs(deep_learning.BaseLoss, BaseLoss)
        loss_module = __import__(
            "openLLV.deepLearning.loss",
            fromlist=["__all__"],
        )
        self.assertEqual(
            loss_module.__all__,
            ["BaseLoss", "LLIELoss", *loss_module.LLIELoss.__all__],
        )


class BaseLossComputeTests(unittest.TestCase):
    def test_supervised_compute_uses_tensor_output_directly(self):
        loss_fn = ExampleSupervisedLoss(scale=2.0)
        prediction = torch.tensor([1.0, 3.0])
        target = torch.tensor([0.0, 1.0])

        loss, used_prediction = loss_fn.compute(
            input_tensor=torch.zeros(2),
            model_output=prediction,
            target=target,
        )

        self.assertEqual(loss.item(), 3.0)
        self.assertIs(used_prediction, prediction)

    def test_supervised_compute_requires_target(self):
        with self.assertRaisesRegex(ValueError, "requires a target"):
            ExampleSupervisedLoss().compute(
                input_tensor=torch.zeros(1),
                model_output=torch.zeros(1),
            )

    def test_structured_supervised_output_requires_extractor(self):
        with self.assertRaisesRegex(TypeError, "extract_prediction is required"):
            ExampleSupervisedLoss().compute(
                input_tensor=torch.zeros(1),
                model_output={"pred": torch.zeros(1)},
                target=torch.zeros(1),
            )

    def test_supervised_compute_extracts_and_aligns_prediction(self):
        loss_fn = ExampleSupervisedLoss()
        raw_prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0])
        extractor_calls = []
        align_calls = []

        def extract(model_output, comparison):
            extractor_calls.append((model_output, comparison))
            return model_output["pred"]

        def align(prediction, comparison):
            align_calls.append((prediction, comparison))
            return prediction[: comparison.numel()]

        loss, prediction = loss_fn.compute(
            input_tensor=torch.zeros(1),
            model_output={"pred": raw_prediction},
            target=target,
            extract_prediction=extract,
            align_prediction=align,
        )

        self.assertEqual(loss.item(), 0.0)
        torch.testing.assert_close(prediction, target)
        self.assertIs(extractor_calls[0][1], target)
        self.assertIs(align_calls[0][1], target)

    def test_reference_free_compute_uses_input_and_raw_output(self):
        loss_fn = ExampleReferenceFreeLoss()
        input_tensor = torch.tensor([1.0, 3.0])
        model_output = {"raw": torch.tensor([0.0, 1.0])}

        loss, prediction = loss_fn.compute(
            input_tensor=input_tensor,
            model_output=model_output,
        )

        self.assertEqual(loss.item(), 2.5)
        self.assertIsNone(prediction)

    def test_reference_free_compute_can_extract_and_align_prediction(self):
        input_tensor = torch.tensor([1.0, 2.0])
        model_output = {
            "raw": torch.tensor([1.0, 2.0]),
            "pred": torch.tensor([1.0, 2.0, 3.0]),
        }

        loss, prediction = ExampleReferenceFreeLoss().compute(
            input_tensor=input_tensor,
            model_output=model_output,
            extract_prediction=lambda output, comparison: output["pred"],
            align_prediction=lambda value, comparison: value[: comparison.numel()],
        )

        self.assertEqual(loss.item(), 0.0)
        torch.testing.assert_close(prediction, input_tensor)

    def test_reference_free_prediction_extraction_errors_are_non_fatal(self):
        loss, prediction = ExampleReferenceFreeLoss().compute(
            input_tensor=torch.zeros(1),
            model_output={"raw": torch.zeros(1)},
            extract_prediction=lambda output, comparison: 1 / 0,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNone(prediction)


if __name__ == "__main__":
    unittest.main()
