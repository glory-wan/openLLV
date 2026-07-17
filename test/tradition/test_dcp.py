"""Tests for the migrated Dark Channel Prior enhancer."""

from __future__ import annotations

import unittest

import numpy as np

from openLLV.tradition.algorithms import LLVEnhancer
from openLLV.tradition.algorithms.Dehazing import DCP as dcp_module
from openLLV.tradition.algorithms.Dehazing import DarkChannel


class DarkChannelMetadataTests(unittest.TestCase):
    def test_required_paper_metadata_is_the_module_docstring(self):
        self.assertEqual(
            dcp_module.__doc__.strip(),
            "Dark Channel Prior-based low-light enhancement.\n\n"
            "Original paper: https://ieeexplore.ieee.org/document/5206515",
        )

    def test_class_inherits_base_and_registers_name_and_alias(self):
        self.assertTrue(issubclass(DarkChannel, LLVEnhancer))
        self.assertIs(
            LLVEnhancer._enhancer_registry["darkchannel"],
            DarkChannel,
        )
        self.assertIs(LLVEnhancer._enhancer_registry["dcp"], DarkChannel)

        instance = LLVEnhancer.create_enhancer(
            "DCP",
            size=3,
            guided_radius=3,
        )
        self.assertIsInstance(instance, DarkChannel)


class DarkChannelParameterTests(unittest.TestCase):
    def test_default_and_custom_parameters_are_stored(self):
        default = DarkChannel()
        self.assertEqual(default.size, 15)
        self.assertEqual(default.omega, 0.95)
        self.assertEqual(default.t_min, 0.1)
        self.assertEqual(default.guided_radius, 60)
        self.assertEqual(default.guided_eps, 1e-4)

        custom = DarkChannel(
            size=5,
            omega=0.8,
            t_min=0.2,
            guided_radius=7,
            guided_eps=1e-3,
        )
        self.assertEqual(custom.size, 5)
        self.assertEqual(custom.omega, 0.8)
        self.assertEqual(custom.t_min, 0.2)
        self.assertEqual(custom.guided_radius, 7)
        self.assertEqual(custom.guided_eps, 1e-3)

    def test_invalid_algorithm_parameters_are_rejected(self):
        cases = (
            ({"size": 0}, "size must be > 0"),
            ({"omega": 0}, "omega must be in"),
            ({"omega": 1.01}, "omega must be in"),
            ({"t_min": 0}, "t_min must be in"),
            ({"t_min": 1}, "t_min must be in"),
            ({"guided_radius": 0}, "guided_radius must be > 0"),
            ({"guided_eps": 0}, "guided_eps must be > 0"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    DarkChannel(**kwargs)

    def test_base_parameters_are_forwarded(self):
        enhancer = DarkChannel(
            output_type="pil",
            keep_dtype=False,
            clip_output=False,
        )
        self.assertEqual(enhancer.output_type, "pil")
        self.assertFalse(enhancer.keep_dtype)
        self.assertFalse(enhancer.clip_output)


class DarkChannelComponentTests(unittest.TestCase):
    def test_dark_channel_with_unit_kernel_is_channel_minimum(self):
        image = np.array(
            [
                [[0.2, 0.5, 0.7], [0.8, 0.3, 0.6]],
                [[0.4, 0.9, 0.1], [0.5, 0.6, 0.7]],
            ],
            dtype=np.float64,
        )
        dark = DarkChannel(size=1)._dark_channel(image)
        np.testing.assert_allclose(
            dark,
            np.array([[0.2, 0.3], [0.1, 0.5]]),
        )

    def test_atmospheric_light_uses_brightest_dark_channel_pixel(self):
        image = np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9], [0.2, 0.3, 0.4]],
            ],
            dtype=np.float64,
        )
        dark = np.array([[0.1, 0.2], [0.9, 0.3]], dtype=np.float64)

        atmospheric_light = DarkChannel()._atm_light(image, dark)

        self.assertEqual(atmospheric_light.shape, (1, 3))
        np.testing.assert_allclose(
            atmospheric_light,
            np.array([[0.7, 0.8, 0.9]]),
        )

    def test_transmission_estimate_matches_dark_channel_formula(self):
        image = np.array(
            [[[[0.2, 0.4, 0.6], [0.5, 0.3, 0.9]]]],
            dtype=np.float64,
        ).reshape(1, 2, 3)
        enhancer = DarkChannel(size=1, omega=0.8)

        transmission = enhancer._transmission_estimate(
            image,
            np.ones((1, 3), dtype=np.float64),
        )

        expected = 1.0 - 0.8 * np.min(image, axis=2)
        np.testing.assert_allclose(transmission, expected)

    def test_guided_filter_preserves_constant_input(self):
        guidance = np.full((7, 7), 0.4, dtype=np.float64)
        transmission = np.full((7, 7), 0.7, dtype=np.float64)

        refined = DarkChannel(
            guided_radius=3,
            guided_eps=1e-4,
        )._guided_filter(guidance, transmission)

        np.testing.assert_allclose(refined, transmission, atol=1e-12)

    def test_recover_applies_minimum_transmission(self):
        image = np.array([[[0.4, 0.5, 0.6]]], dtype=np.float64)
        transmission = np.array([[0.01]], dtype=np.float64)
        atmospheric_light = np.array([[0.8, 0.8, 0.8]], dtype=np.float64)
        enhancer = DarkChannel(t_min=0.2)

        recovered = enhancer._recover(
            image,
            transmission,
            atmospheric_light,
        )

        expected = (image - atmospheric_light.reshape(1, 1, 3)) / 0.2
        expected = expected + atmospheric_light.reshape(1, 1, 3)
        np.testing.assert_allclose(recovered, expected)


class DarkChannelExecutionTests(unittest.TestCase):
    def test_end_to_end_enhancement_preserves_shape_dtype_and_range(self):
        image = np.random.default_rng(7).integers(
            0,
            256,
            size=(16, 16, 3),
            dtype=np.uint8,
        )
        output = DarkChannel(
            size=3,
            guided_radius=3,
        ).enhance(image)

        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, image.dtype)
        self.assertTrue(np.isfinite(output).all())
        self.assertGreaterEqual(int(output.min()), 0)
        self.assertLessEqual(int(output.max()), 255)

    def test_call_and_enhance_return_the_same_result(self):
        image = np.random.default_rng(11).integers(
            0,
            256,
            size=(12, 12, 3),
            dtype=np.uint8,
        )
        enhancer = DarkChannel(size=3, guided_radius=3)

        np.testing.assert_array_equal(
            enhancer(image),
            enhancer.enhance(image),
        )


if __name__ == "__main__":
    unittest.main()
