"""Tests for the migrated low-light enhancement loss package."""

from __future__ import annotations

import inspect
import unittest

import torch

from openLLV.deepLearning import loss as loss_package
from openLLV.deepLearning.loss import BaseLoss, LLIELoss


LOSS_FACTORIES = {
    "l1": {},
    "mse": {},
    "smooth_l1": {},
    "charbonnier": {},
    "zerodce": {},
    "zerodce_extension": {},
    "sci": {},
    "ruas": {},
    "lednet": {"use_perceptual": False},
    "darkir": {"use_perceptual": False, "use_edge": False},
    "zeroig": {},
    "uretinex": {},
    "retinexformer": {},
    "llnet": {},
    "kind": {},
    "kindplusplus": {},
    "enlightengan": {},
    "llflow": {},
    "cidnet": {"use_perceptual": False},
    "pairlie": {},
    "llformer": {},
}


class LLIELossPackageTests(unittest.TestCase):
    def test_all_public_names_are_available_from_both_packages(self):
        self.assertEqual(len(LLIELoss.__all__), 43)
        for name in LLIELoss.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(LLIELoss, name))
                self.assertIs(
                    getattr(loss_package, name),
                    getattr(LLIELoss, name),
                )

    def test_compatibility_aliases_reference_the_same_classes(self):
        alias_pairs = {
            "CIDNetLoss": "CIDNet_Loss",
            "DarkIRLoss": "DarkIR_Loss",
            "EnlightenGANLoss": "EnlightenGAN_Loss",
            "KinDLoss": "KinD_Loss",
            "KinDPlusPlusLoss": "KinDPlusPlus_Loss",
            "LEDNetLoss": "LEDNet_Loss",
            "LLFlowLoss": "LLFlow_Loss",
            "LLFormerLoss": "LLFormer_Loss",
            "LLNetLoss": "LLNet_Loss",
            "PairLIELoss": "PairLIE_Loss",
            "RetinexFormerLoss": "RetinexFormer_Loss",
            "URetinexLoss": "URetinex_Loss",
            "ZeroIGLoss": "ZeroIG_Loss",
        }
        for alias, canonical in alias_pairs.items():
            with self.subTest(alias=alias):
                self.assertIs(
                    getattr(LLIELoss, alias),
                    getattr(LLIELoss, canonical),
                )

    def test_all_concrete_losses_are_registered_and_constructible(self):
        exported_loss_classes = {
            value
            for name in LLIELoss.__all__
            if inspect.isclass(value := getattr(LLIELoss, name))
            and issubclass(value, BaseLoss)
        }
        self.assertEqual(len(exported_loss_classes), len(LOSS_FACTORIES))

        for loss_name, kwargs in LOSS_FACTORIES.items():
            with self.subTest(loss_name=loss_name):
                instance = BaseLoss.create_loss(loss_name, **kwargs)
                self.assertIsInstance(instance, BaseLoss)
                self.assertIn(instance.__class__, exported_loss_classes)

    def test_registry_supports_migrated_aliases_case_insensitively(self):
        cases = {
            "MAE": LLIELoss.L1Loss,
            "HVI-CIDNET-LOSS": LLIELoss.CIDNet_Loss,
            "KIND++": LLIELoss.KinDPlusPlus_Loss,
            "ZERO-IG": LLIELoss.ZeroIG_Loss,
            "ZERODCE++": LLIELoss.ZeroDCE_extension_Loss,
        }
        for alias, expected_class in cases.items():
            kwargs = {"use_perceptual": False} if alias == "HVI-CIDNET-LOSS" else {}
            with self.subTest(alias=alias):
                self.assertIsInstance(
                    BaseLoss.create_loss(alias, **kwargs),
                    expected_class,
                )


class CommonLossTests(unittest.TestCase):
    def setUp(self):
        self.prediction = torch.ones(1, 3, 8, 8)
        self.target = torch.zeros_like(self.prediction)

    def test_common_losses_match_expected_values(self):
        cases = (
            (LLIELoss.L1Loss(), 1.0),
            (LLIELoss.MSELoss(), 1.0),
            (LLIELoss.SmoothL1Loss(), 0.5),
            (LLIELoss.CharbonnierLoss(eps=1e-3), (1.0 + 1e-6) ** 0.5),
        )
        for loss_function, expected in cases:
            with self.subTest(loss=loss_function.__class__.__name__):
                value = loss_function(self.prediction, self.target)
                self.assertEqual(value.ndim, 0)
                self.assertAlmostEqual(value.item(), expected, places=6)

    def test_common_loss_uses_base_compute_contract(self):
        loss, prediction = LLIELoss.L1Loss().compute(
            input_tensor=self.target,
            model_output={"pred": self.prediction},
            target=self.target,
            extract_prediction=lambda output, _: output["pred"],
        )
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertTrue(torch.equal(prediction, self.prediction))

    def test_charbonnier_rejects_non_positive_epsilon(self):
        with self.assertRaisesRegex(ValueError, "eps must be positive"):
            LLIELoss.CharbonnierLoss(eps=0)


class ModelLossComputeTests(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.zeros(1, 3, 16, 16)
        self.target = torch.zeros_like(self.input_tensor)
        self.prediction = torch.ones_like(self.input_tensor)

    def assert_scalar_finite(self, value: torch.Tensor) -> None:
        self.assertEqual(value.ndim, 0)
        self.assertTrue(torch.isfinite(value).item())

    def test_simple_supervised_losses_use_unified_compute_contract(self):
        cases = (
            (LLIELoss.LLFormer_Loss(), 0.5),
            (
                LLIELoss.CIDNet_Loss(
                    ssim_weight=0,
                    edge_weight=0,
                    perceptual_weight=0,
                    hvi_weight=0,
                    use_perceptual=False,
                ),
                1.0,
            ),
            (
                LLIELoss.DarkIR_Loss(
                    perceptual_weight=0,
                    edge_weight=0,
                    use_perceptual=False,
                    use_edge=False,
                    use_lol_loss=False,
                ),
                1.0,
            ),
            (
                LLIELoss.LEDNet_Loss(
                    perceptual_weight=0,
                    use_perceptual=False,
                    use_side_loss=False,
                ),
                1.0,
            ),
            (LLIELoss.RetinexFormer_Loss(illumination_weight=0), 1.0),
            (
                LLIELoss.LLNet_Loss(
                    sparsity_weight=0,
                    weight_decay=0,
                ),
                1.0,
            ),
            (
                LLIELoss.URetinex_Loss(
                    adjustment_reconstruction_weight=1,
                    adjustment_ssim_weight=0,
                ),
                1.0,
            ),
        )

        for loss_function, expected in cases:
            with self.subTest(loss=loss_function.__class__.__name__):
                loss, prediction = loss_function.compute(
                    input_tensor=self.input_tensor,
                    model_output=self.prediction,
                    target=self.target,
                )
                self.assert_scalar_finite(loss)
                self.assertAlmostEqual(loss.item(), expected, places=5)
                self.assertIs(prediction, self.prediction)

    def test_pairlie_structured_output(self):
        ones = torch.ones_like(self.input_tensor)
        illumination = torch.ones(1, 1, 16, 16)
        output = {
            "pred": ones,
            "aux": {
                "illumination": illumination,
                "reflectance": ones,
                "denoised": ones,
                "paired_reflectance": ones,
            },
        }
        loss, prediction = LLIELoss.PairLIE_Loss().compute(
            input_tensor=ones,
            model_output=output,
            target=ones,
        )
        self.assert_scalar_finite(loss)
        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertIs(prediction, ones)

    def test_llflow_structured_output(self):
        output = {
            "pred": self.prediction,
            "aux": {
                "condition": self.input_tensor,
                "flow_forward": lambda target, condition: (
                    torch.zeros_like(target),
                    target.new_zeros(target.shape[0]),
                ),
            },
        }
        loss_function = LLIELoss.LLFlow_Loss(
            nll_weight=0,
            reconstruction_weight=1,
            color_weight=0,
            tv_weight=0,
        )
        loss, prediction = loss_function.compute(
            input_tensor=self.input_tensor,
            model_output=output,
            target=self.target,
        )
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertTrue(torch.equal(prediction, self.prediction))

    def test_kind_family_structured_outputs(self):
        ones = torch.ones_like(self.input_tensor)
        illumination = torch.ones(1, 1, 16, 16)
        output = {
            "pred": ones,
            "aux": {
                "decompose_fn": lambda _: (ones, illumination),
                "low_reflectance": ones,
                "low_illumination": illumination,
                "restored_reflectance": ones,
                "adjusted_illumination": illumination,
            },
        }
        kind = LLIELoss.KinD_Loss(
            restoration_ssim_weight=0,
            adjustment_gradient_weight=0,
            final_ssim_weight=0,
        )
        kind_pp = LLIELoss.KinDPlusPlus_Loss(
            restoration_mse_weight=0,
            restoration_ssim_weight=0,
            adjustment_mse_weight=0,
            adjustment_gradient_weight=0,
            final_reconstruction_weight=0,
            final_ssim_weight=0,
        )
        for loss_function in (kind, kind_pp):
            with self.subTest(loss=loss_function.__class__.__name__):
                loss, prediction = loss_function.compute(
                    input_tensor=ones,
                    model_output=output,
                    target=ones,
                )
                self.assert_scalar_finite(loss)
                self.assertIs(prediction, ones)

    def test_enlightengan_zero_weight_structured_output(self):
        output = {
            "pred": self.prediction,
            "aux": {
                "global_discriminator": torch.nn.Identity(),
                "local_discriminator": torch.nn.Identity(),
                "local_box": (0, 0, 8, 8),
            },
        }
        loss_function = LLIELoss.EnlightenGAN_Loss(
            adversarial_weight=0,
            local_adversarial_weight=0,
            discriminator_weight=0,
            self_regularization_weight=0,
            exposure_weight=0,
            tv_weight=0,
        )
        loss, prediction = loss_function.compute(
            input_tensor=self.input_tensor,
            model_output=output,
            target=self.target,
        )
        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(prediction, self.prediction))

    def test_reference_free_losses_return_finite_scalars(self):
        constant = torch.full_like(self.input_tensor, 0.25)
        cases = (
            (
                LLIELoss.Sci_Loss(),
                {"pred": constant, "aux": {"enhanced": constant}},
            ),
            (
                LLIELoss.RUAS_Loss(denoise_weight=0),
                {"aux": {"u_list": [constant], "t_list": [constant]}},
            ),
            (
                LLIELoss.ZeroDCE_Loss(),
                {
                    "pred": constant,
                    "aux": {
                        "enhanced": constant,
                        "r": torch.zeros(1, 24, 16, 16),
                    },
                },
            ),
            (
                LLIELoss.ZeroDCE_extension_Loss(),
                {
                    "pred": constant,
                    "aux": {
                        "enhanced": constant,
                        "r": torch.zeros(1, 3, 16, 16),
                    },
                },
            ),
        )
        for loss_function, model_output in cases:
            with self.subTest(loss=loss_function.__class__.__name__):
                loss, _ = loss_function.compute(
                    input_tensor=constant,
                    model_output=model_output,
                )
                self.assert_scalar_finite(loss)

    def test_legacy_loss_inputs_field_remains_supported(self):
        constant = torch.full_like(self.input_tensor, 0.25)
        loss, prediction = LLIELoss.Sci_Loss().compute(
            input_tensor=constant,
            model_output={
                "pred": constant,
                "loss_inputs": {"enhanced": constant},
            },
            extract_prediction=lambda output, _: output["pred"],
        )
        self.assert_scalar_finite(loss)
        self.assertIs(prediction, constant)


if __name__ == "__main__":
    unittest.main()
