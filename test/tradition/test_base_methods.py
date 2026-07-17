"""Tests for histogram-based traditional enhancement algorithms."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

import openLLV.tradition.algorithms as algorithms_package
import openLLV.tradition.algorithms.BaseMethods as base_methods_package
from openLLV.tradition import Predictor
from openLLV.tradition.algorithms import AHE, CLAHE, HE, RCLAHE, LLVEnhancer


ALGORITHMS = (AHE, HE, CLAHE, RCLAHE)
COLOR_SPACES = ("rgb", "hsv", "hls", "yuv", "lab")


def rgb_sample(height: int = 32, width: int = 40) -> np.ndarray:
    """Create a deterministic RGB image with non-uniform histograms."""
    y, x = np.indices((height, width))
    return np.stack(
        (
            (x * 7 + y * 3) % 256,
            (x * 2 + y * 11 + 17) % 256,
            (x * 13 + y * 5 + 31) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def gray_sample(height: int = 24, width: int = 28) -> np.ndarray:
    """Create a deterministic grayscale image."""
    y, x = np.indices((height, width))
    return ((x * 9 + y * 5) % 256).astype(np.uint8)


class BaseMethodRegistrationTests(unittest.TestCase):
    def test_algorithms_inherit_register_and_create_through_llv_enhancer(self):
        for name, algorithm in (
            ("ahe", AHE),
            ("he", HE),
            ("clahe", CLAHE),
            ("rclahe", RCLAHE),
        ):
            with self.subTest(name=name):
                self.assertTrue(issubclass(algorithm, LLVEnhancer))
                self.assertIsInstance(
                    LLVEnhancer.create_enhancer(name.upper()),
                    algorithm,
                )
                self.assertIn(name, LLVEnhancer.list_registered_enhancers())

    def test_base_methods_and_algorithms_packages_export_all_four_classes(self):
        self.assertEqual(
            base_methods_package.__all__,
            ["AHE", "HE", "CLAHE", "RCLAHE"],
        )
        for name, algorithm in zip(base_methods_package.__all__, ALGORITHMS):
            with self.subTest(name=name):
                self.assertIs(getattr(base_methods_package, name), algorithm)
                self.assertIs(getattr(algorithms_package, name), algorithm)
                self.assertIn(name, algorithms_package.__all__)

    def test_predictor_default_method_now_resolves_to_he(self):
        predictor = Predictor()
        self.assertIsInstance(predictor.enhancer, HE)
        self.assertEqual(predictor.method_name, "he")


class BaseMethodConfigurationTests(unittest.TestCase):
    def test_color_space_aliases_are_normalized(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm.__name__, alias="bgr"):
                self.assertEqual(algorithm(color_space=" BGR ").color_space, "rgb")
            with self.subTest(algorithm=algorithm.__name__, alias="ycbcr"):
                self.assertEqual(
                    algorithm(color_space="YCbCr").color_space,
                    "yuv",
                )

    def test_invalid_color_spaces_are_rejected_consistently(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm.__name__, value=None):
                with self.assertRaisesRegex(TypeError, "color_space must be str"):
                    algorithm(color_space=None)
            with self.subTest(algorithm=algorithm.__name__, value="xyz"):
                with self.assertRaisesRegex(ValueError, "Unsupported color_space"):
                    algorithm(color_space="xyz")

    def test_adaptive_algorithm_parameters_are_validated(self):
        for algorithm in (AHE, CLAHE, RCLAHE):
            for value in ((0, 8), (8,), (8, 2.5), "8,8"):
                with self.subTest(algorithm=algorithm.__name__, grid=value):
                    with self.assertRaisesRegex(ValueError, "tile_grid_size"):
                        algorithm(tile_grid_size=value)

        for algorithm in (CLAHE, RCLAHE):
            for value in (0, -1, float("inf"), "2"):
                with self.subTest(algorithm=algorithm.__name__, clip=value):
                    with self.assertRaisesRegex(ValueError, "clip_limit"):
                        algorithm(clip_limit=value)

        for value in (0, -1, 1.5, True):
            with self.subTest(iterations=value):
                with self.assertRaisesRegex(ValueError, "iterations"):
                    RCLAHE(iterations=value)

    def test_algorithm_parameters_and_base_options_are_reported(self):
        cases = (
            (
                AHE(color_space="lab", tile_grid_size=[4, 6], keep_dtype=False),
                {"color_space": "lab", "tile_grid_size": (4, 6)},
            ),
            (HE(color_space="hsv"), {"color_space": "hsv"}),
            (
                CLAHE(color_space="hls", clip_limit=3, tile_grid_size=(2, 4)),
                {
                    "color_space": "hls",
                    "clip_limit": 3.0,
                    "tile_grid_size": (2, 4),
                },
            ),
            (
                RCLAHE(
                    color_space="rgb",
                    clip_limit=4,
                    tile_grid_size=(4, 2),
                    iterations=2,
                ),
                {
                    "color_space": "rgb",
                    "clip_limit": 4.0,
                    "tile_grid_size": (4, 2),
                    "iterations": 2,
                },
            ),
        )
        for enhancer, expected in cases:
            with self.subTest(algorithm=enhancer.__class__.__name__):
                params = enhancer.get_params()
                for key, value in expected.items():
                    self.assertEqual(params[key], value)
                self.assertEqual(params["output_type"], "numpy")


class BaseMethodEnhancementTests(unittest.TestCase):
    def test_all_algorithms_support_every_migrated_color_space(self):
        image = rgb_sample()
        for algorithm in ALGORITHMS:
            for color_space in COLOR_SPACES:
                with self.subTest(
                    algorithm=algorithm.__name__,
                    color_space=color_space,
                ):
                    output = algorithm(color_space=color_space).enhance(image)
                    self.assertEqual(output.shape, image.shape)
                    self.assertEqual(output.dtype, np.uint8)
                    self.assertGreaterEqual(int(output.min()), 0)
                    self.assertLessEqual(int(output.max()), 255)

    def test_he_rgb_matches_channel_wise_opencv_equalization(self):
        image = rgb_sample()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        expected = cv2.merge(
            [cv2.equalizeHist(channel) for channel in cv2.split(bgr)]
        )
        np.testing.assert_array_equal(HE(color_space="rgb").enhance(image), expected)

    def test_grayscale_implementations_match_opencv(self):
        image = gray_sample()
        cases = (
            (HE(), cv2.equalizeHist(image)),
            (
                AHE(tile_grid_size=(4, 4)),
                cv2.createCLAHE(255.0, (4, 4)).apply(image),
            ),
            (
                CLAHE(clip_limit=3.0, tile_grid_size=(4, 4)),
                cv2.createCLAHE(3.0, (4, 4)).apply(image),
            ),
        )
        for enhancer, expected in cases:
            with self.subTest(algorithm=enhancer.__class__.__name__):
                np.testing.assert_array_equal(enhancer._enhance(image), expected)

    def test_rclahe_applies_the_requested_number_of_iterations(self):
        image = gray_sample()
        enhancer = RCLAHE(iterations=3, tile_grid_size=(4, 4))
        with patch.object(
            enhancer,
            "_apply_once",
            wraps=enhancer._apply_once,
        ) as apply_once:
            output = enhancer._enhance(image)

        self.assertEqual(apply_once.call_count, 3)
        expected = image
        reference = cv2.createCLAHE(2.0, (4, 4))
        for _ in range(3):
            expected = reference.apply(expected)
        np.testing.assert_array_equal(output, expected)

    def test_float_inputs_are_converted_to_uint8_before_equalization(self):
        image = gray_sample().astype(np.float32) / 255.0
        expected = cv2.equalizeHist((image * 255).astype(np.uint8))
        output = HE()._enhance(image)
        self.assertEqual(output.dtype, np.uint8)
        np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    unittest.main()
