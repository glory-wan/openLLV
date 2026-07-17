"""Tests for traditional low-light image enhancement algorithms."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import openLLV.tradition.algorithms as algorithms_package
import openLLV.tradition.algorithms.LLIE as llie_package
from openLLV.tradition import Predictor
from openLLV.tradition.algorithms import (
    BIMEF,
    GCP,
    LIME,
    MSR,
    MSRCR,
    NPE,
    SSR,
    Gamma,
    LLVEnhancer,
)


LLIE_CLASSES = (BIMEF, Gamma, GCP, LIME, NPE, SSR, MSR, MSRCR)
LLIE_EXPORTS = [
    "BIMEF",
    "Gamma",
    "GCP",
    "LIME",
    "NPE",
    "SSR",
    "MSR",
    "MSRCR",
]


def rgb_sample(height: int = 24, width: int = 30) -> np.ndarray:
    """Create a deterministic low-light RGB test image."""
    y, x = np.indices((height, width))
    return np.stack(
        (
            5 + (x * 2 + y) % 70,
            8 + (x + y * 3) % 75,
            3 + (x * 3 + y * 2) % 65,
        ),
        axis=2,
    ).astype(np.uint8)


def gray_sample(height: int = 20, width: int = 26) -> np.ndarray:
    """Create a deterministic low-light grayscale image."""
    y, x = np.indices((height, width))
    return (4 + (x * 3 + y * 2) % 80).astype(np.uint8)


class LLIERegistrationTests(unittest.TestCase):
    def test_classes_inherit_llv_enhancer_and_are_registered(self):
        for algorithm in LLIE_CLASSES:
            name = algorithm.__name__.lower()
            with self.subTest(algorithm=algorithm.__name__):
                self.assertTrue(issubclass(algorithm, LLVEnhancer))
                self.assertIsInstance(
                    LLVEnhancer.create_enhancer(name.upper()),
                    algorithm,
                )
                self.assertIn(name, LLVEnhancer.list_registered_enhancers())

        self.assertIsInstance(
            LLVEnhancer.create_enhancer("GCP-MS"),
            GCP,
        )
        self.assertNotIn("_retinexbase", LLVEnhancer.list_registered_enhancers())

    def test_llie_and_algorithms_packages_export_migrated_classes(self):
        self.assertEqual(llie_package.__all__, LLIE_EXPORTS)
        for name, algorithm in zip(LLIE_EXPORTS, LLIE_CLASSES):
            with self.subTest(name=name):
                self.assertIs(getattr(llie_package, name), algorithm)
                self.assertIs(getattr(algorithms_package, name), algorithm)
                self.assertIn(name, algorithms_package.__all__)

    def test_predictor_lists_every_migrated_llie_method(self):
        methods = Predictor.list_available_methods()
        for name in ("bimef", "gamma", "gcp", "gcp-ms", "lime", "npe"):
            with self.subTest(name=name):
                self.assertIn(name, methods)
        for name in ("ssr", "msr", "msrcr"):
            with self.subTest(name=name):
                self.assertIn(name, methods)


class LLIEParameterTests(unittest.TestCase):
    def test_default_parameters_are_available_through_get_params(self):
        cases = (
            (BIMEF(), {"target_mean": 0.55, "max_ratio": 5.0}),
            (Gamma(), {"gamma": 0.6}),
            (GCP(), {"gamma_max": 6.0, "erosion_window": 15}),
            (LIME(), {"gamma": 0.8, "guided_radius": 15}),
            (NPE(), {"sigma": 15.0, "naturalness": 0.35}),
            (SSR(), {"sigma": 80.0, "low_clip": 1.0}),
            (MSR(), {"scales": (15.0, 80.0, 250.0)}),
            (MSRCR(), {"alpha": 125.0, "beta": 46.0, "gain": 1.0}),
        )
        for enhancer, expected in cases:
            with self.subTest(algorithm=enhancer.__class__.__name__):
                params = enhancer.get_params()
                for name, value in expected.items():
                    self.assertEqual(params[name], value)
                self.assertEqual(params["output_type"], "numpy")

    def test_set_params_updates_values_and_revalidates(self):
        cases = (
            (BIMEF(), {"target_mean": 0.6}),
            (Gamma(), {"gamma": 0.7}),
            (GCP(), {"gamma_max": 4.0}),
            (LIME(), {"exposure": 1.2}),
            (NPE(), {"naturalness": 0.5}),
            (SSR(), {"sigma": 25.0}),
            (MSR(), {"scales": [10, 30]}),
            (MSRCR(), {"alpha": 100.0}),
        )
        for enhancer, params in cases:
            with self.subTest(algorithm=enhancer.__class__.__name__):
                self.assertIs(enhancer.set_params(**params), enhancer)
                for name, value in params.items():
                    expected = tuple(float(item) for item in value) if (
                        name == "scales"
                    ) else value
                    self.assertEqual(getattr(enhancer, name), expected)

    def test_bimef_rejects_invalid_parameters(self):
        invalid = (
            {"exposure_ratio": 0},
            {"target_mean": 1},
            {"max_ratio": 0.5},
            {"well_exposed_sigma": 0},
            {"contrast_weight": -1},
        )
        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    BIMEF(**params)

    def test_gamma_and_gcp_reject_invalid_parameters(self):
        with self.assertRaises(TypeError):
            Gamma(gamma="0.5")
        with self.assertRaises(ValueError):
            Gamma(gamma=0)

        invalid_gcp = (
            {"gamma_max": 0.5},
            {"erosion_window": 0},
            {"atmospheric_bins": 0},
            {"atmospheric_percentile": 1},
            {"t_min": 0},
            {"blur_ksize": 4},
            {"low_percentile": 80, "high_percentile": 20},
            {"eps": 0},
        )
        for params in invalid_gcp:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    GCP(**params)

    def test_lime_and_npe_reject_invalid_parameters(self):
        invalid_lime = (
            {"gamma": 0},
            {"guided_radius": 0},
            {"guided_eps": 0},
            {"illumination_floor": 0},
            {"exposure": 0},
        )
        invalid_npe = (
            {"sigma": 0},
            {"illumination_floor": 0},
            {"enhancement_strength": 0},
            {"naturalness": 2},
            {"detail_weight": -1},
        )
        for algorithm, cases in ((LIME, invalid_lime), (NPE, invalid_npe)):
            for params in cases:
                with self.subTest(algorithm=algorithm.__name__, params=params):
                    with self.assertRaises(ValueError):
                        algorithm(**params)

    def test_retinex_variants_reject_invalid_parameters(self):
        for algorithm in (SSR, MSR, MSRCR):
            with self.subTest(algorithm=algorithm.__name__, value="clips"):
                with self.assertRaises(ValueError):
                    algorithm(low_clip=99, high_clip=1)
            with self.subTest(algorithm=algorithm.__name__, value="eps"):
                with self.assertRaises(ValueError):
                    algorithm(eps=0)

        with self.assertRaises(ValueError):
            SSR(sigma=0)
        with self.assertRaises(ValueError):
            MSR(scales=[])
        with self.assertRaises(TypeError):
            MSR(scales="15,80")
        for params in ({"alpha": 0}, {"beta": 0}, {"gain": 0}):
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    MSRCR(**params)


class LLIEEnhancementTests(unittest.TestCase):
    def test_all_algorithms_preserve_color_shape_dtype_and_range(self):
        image = rgb_sample()
        for algorithm in LLIE_CLASSES:
            with self.subTest(algorithm=algorithm.__name__):
                output = algorithm().enhance(image)
                self.assertEqual(output.shape, image.shape)
                self.assertEqual(output.dtype, np.uint8)
                self.assertGreaterEqual(int(output.min()), 0)
                self.assertLessEqual(int(output.max()), 255)

    def test_all_algorithms_support_grayscale_working_images(self):
        image = gray_sample()
        for algorithm in LLIE_CLASSES:
            with self.subTest(algorithm=algorithm.__name__):
                output = algorithm()._enhance(image)
                self.assertEqual(output.shape, image.shape)
                self.assertTrue(np.all(np.isfinite(output)))

    def test_all_algorithms_support_unit_float_input(self):
        image = rgb_sample().astype(np.float32) / 255.0
        for algorithm in LLIE_CLASSES:
            with self.subTest(algorithm=algorithm.__name__):
                output = algorithm().enhance(image)
                self.assertEqual(output.shape, image.shape)
                self.assertEqual(output.dtype, np.float32)
                self.assertGreaterEqual(float(output.min()), 0.0)
                self.assertLessEqual(float(output.max()), 1.0)
                self.assertTrue(np.all(np.isfinite(output)))

    def test_gamma_matches_power_law_definition(self):
        image = np.array([[0, 64, 128, 255]], dtype=np.uint8)
        expected = np.rint(
            np.power(image.astype(np.float32) / 255.0, 0.5) * 255.0
        ).astype(np.uint8)
        np.testing.assert_array_equal(Gamma(gamma=0.5)._enhance(image), expected)

    def test_runtime_overrides_do_not_mutate_configured_parameters(self):
        image = gray_sample()
        gamma = Gamma(gamma=0.8)
        default_output = gamma._enhance(image)
        override_output = gamma._enhance(image, gamma=0.4)
        self.assertEqual(gamma.gamma, 0.8)
        self.assertFalse(np.array_equal(default_output, override_output))

        lime = LIME(exposure=1.0)
        lime._enhance(image, exposure=1.5)
        self.assertEqual(lime.exposure, 1.0)

    def test_algorithms_with_alpha_support_preserve_alpha_channel(self):
        color = cv2.cvtColor(rgb_sample(16, 18), cv2.COLOR_RGB2BGR)
        alpha = np.arange(16 * 18, dtype=np.uint16).reshape(16, 18) % 256
        alpha = alpha.astype(np.uint8)
        image = np.dstack((color, alpha))
        for algorithm in (BIMEF, GCP, LIME, NPE, SSR, MSR, MSRCR):
            with self.subTest(algorithm=algorithm.__name__):
                output = algorithm()._enhance(image)
                self.assertEqual(output.shape, image.shape)
                np.testing.assert_allclose(output[:, :, 3], alpha, atol=1e-4)

    def test_retinex_constant_image_uses_normalization_fallback(self):
        image = np.full((16, 18, 3), 32, dtype=np.uint8)
        for algorithm in (SSR, MSR, MSRCR):
            with self.subTest(algorithm=algorithm.__name__):
                output = algorithm()._enhance(image)
                self.assertEqual(output.shape, image.shape)
                self.assertTrue(np.all(np.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
