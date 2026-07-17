"""Tests for the traditional image enhancement base class."""

from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import openLLV.tradition as tradition_package
from openLLV.tradition import algorithms as algorithms_package
from openLLV.tradition.algorithms.BaseModel import LLVEnhancer


class ExampleEnhancer(LLVEnhancer):
    """Small deterministic enhancer used to exercise the base contract."""

    name = "example-traditional"
    aliases = ["example-alias", "demo-enhancer"]

    def __init__(self, offset: float = 0.0, **kwargs):
        self.offset = float(offset)
        super().__init__(**kwargs)

    def _enhance(self, image: np.ndarray, **kwargs) -> np.ndarray:
        offset = float(kwargs.get("offset", self.offset))
        return image.astype(np.float32) + offset


class BadReturnEnhancer(LLVEnhancer):
    def _enhance(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return [image]


class AbstractEnhancer(LLVEnhancer):
    pass


class _PrivateEnhancer(LLVEnhancer):
    def _enhance(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return image


def rgb_sample(dtype=np.uint8):
    return np.array(
        [
            [[10, 20, 30], [250, 1, 100]],
            [[0, 128, 255], [40, 50, 60]],
        ],
        dtype=dtype,
    )


class EnhancerRegistryTests(unittest.TestCase):
    def test_concrete_subclass_registers_class_name_name_and_aliases(self):
        registry = LLVEnhancer._enhancer_registry
        for key in (
            "exampleenhancer",
            "example-traditional",
            "example-alias",
            "demo-enhancer",
        ):
            with self.subTest(key=key):
                self.assertIs(registry[key], ExampleEnhancer)

    def test_abstract_and_private_subclasses_are_not_registered(self):
        registered = LLVEnhancer.list_registered_enhancers()
        self.assertNotIn("abstractenhancer", registered)
        self.assertNotIn("_privateenhancer", registered)

    def test_factory_is_case_insensitive_and_forwards_arguments(self):
        enhancer = LLVEnhancer.create_enhancer(
            "  EXAMPLE-ALIAS  ",
            offset=3,
            keep_dtype=False,
        )
        self.assertIsInstance(enhancer, ExampleEnhancer)
        self.assertEqual(enhancer.offset, 3.0)
        self.assertFalse(enhancer.keep_dtype)

    def test_manual_registration_and_type_validation(self):
        class ManualEnhancer(LLVEnhancer):
            name = "manual-example"

            def _enhance(self, image: np.ndarray, **kwargs) -> np.ndarray:
                return image

        for key in ("manualenhancer", "manual-example"):
            LLVEnhancer._enhancer_registry.pop(key, None)

        self.assertIs(LLVEnhancer.register(ManualEnhancer), ManualEnhancer)
        self.assertIs(
            LLVEnhancer.create_enhancer("manual-example").__class__,
            ManualEnhancer,
        )
        with self.assertRaisesRegex(TypeError, "subclass of LLVEnhancer"):
            LLVEnhancer.register(object)

    def test_factory_rejects_empty_and_unknown_names_with_suggestion(self):
        for name in ("", "   ", None):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    LLVEnhancer.create_enhancer(name)

        with self.assertRaises(ValueError) as context:
            LLVEnhancer.create_enhancer("example-traditiona")
        message = str(context.exception)
        self.assertIn("not registered", message)
        self.assertIn("example-traditional", message)

    def test_similar_name_helper_supports_match_and_fallback(self):
        self.assertEqual(
            LLVEnhancer._get_similar_enhancer_name(
                "exampl",
                ["example", "other"],
            ),
            "example",
        )
        self.assertEqual(
            LLVEnhancer._get_similar_enhancer_name("xyz", ["example"]),
            "No similar enhancers found",
        )


class EnhancerParameterTests(unittest.TestCase):
    def test_default_parameters_get_params_and_repr(self):
        enhancer = ExampleEnhancer()
        self.assertEqual(
            enhancer.get_params(),
            {
                "name": "example-traditional",
                "output_type": "numpy",
                "keep_dtype": True,
                "clip_output": True,
            },
        )
        self.assertEqual(
            repr(enhancer),
            "ExampleEnhancer(output_type='numpy', keep_dtype=True, "
            "clip_output=True)",
        )

    def test_constructor_validates_output_type_and_boolean_options(self):
        with self.assertRaisesRegex(ValueError, "output_type must be one of"):
            ExampleEnhancer(output_type="tensor")
        with self.assertRaisesRegex(TypeError, "keep_dtype must be bool"):
            ExampleEnhancer(keep_dtype=1)
        with self.assertRaisesRegex(TypeError, "clip_output must be bool"):
            ExampleEnhancer(clip_output="yes")

    def test_set_params_updates_values_returns_self_and_revalidates(self):
        enhancer = ExampleEnhancer()
        returned = enhancer.set_params(
            output_type="pil",
            keep_dtype=False,
            clip_output=False,
        )
        self.assertIs(returned, enhancer)
        self.assertEqual(enhancer.output_type, "pil")
        self.assertFalse(enhancer.keep_dtype)
        self.assertFalse(enhancer.clip_output)

        with self.assertRaisesRegex(ValueError, "has no parameter"):
            enhancer.set_params(missing=True)

    def test_dtype_range_clip_and_cast_helpers(self):
        self.assertEqual(LLVEnhancer._dtype_range(np.uint8), (0.0, 255.0))
        self.assertEqual(LLVEnhancer._dtype_range(np.int16), (-32768.0, 32767.0))
        self.assertEqual(LLVEnhancer._dtype_range(np.float32), (0.0, 1.0))
        self.assertEqual(LLVEnhancer._dtype_range(None), (0.0, 255.0))

        clipped = LLVEnhancer._clip_to_valid_range(
            np.array([-2.0, 1.4, 300.0]),
            np.uint8,
        )
        np.testing.assert_array_equal(clipped, np.array([0.0, 1.4, 255.0]))

        cast = LLVEnhancer._cast_to_dtype(
            np.array([-1.0, 1.4, 300.0]),
            np.uint8,
        )
        np.testing.assert_array_equal(cast, np.array([0, 1, 255], dtype=np.uint8))

        with self.assertRaisesRegex(TypeError, "Unsupported dtype"):
            LLVEnhancer._dtype_range(np.bool_)


class EnhancerExecutionTests(unittest.TestCase):
    def test_numpy_input_is_enhanced_clipped_and_cast_to_original_dtype(self):
        image = rgb_sample()
        result = ExampleEnhancer(offset=10)(image)
        expected = np.clip(
            image[:, :, ::-1].astype(np.float32) + 10,
            0,
            255,
        ).astype(np.uint8)

        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_runtime_enhancement_kwargs_override_subclass_defaults(self):
        image = rgb_sample()
        enhancer = ExampleEnhancer(offset=1)
        default = enhancer.enhance(image)
        overridden = enhancer.enhance(image, offset=5)

        bgr = image[:, :, ::-1].astype(np.float32)
        expected_difference = (
            np.clip(bgr + 5, 0, 255).astype(np.uint8).astype(np.int16)
            - np.clip(bgr + 1, 0, 255).astype(np.uint8).astype(np.int16)
        )
        np.testing.assert_array_equal(
            overridden.astype(np.int16) - default.astype(np.int16),
            expected_difference,
        )

    def test_keep_dtype_false_preserves_enhancer_float_output(self):
        output = ExampleEnhancer(
            offset=0.5,
            keep_dtype=False,
        ).enhance(rgb_sample())

        self.assertEqual(output.dtype, np.float32)
        self.assertAlmostEqual(float(output[0, 0, 0]), 30.5)

    def test_float_input_is_clipped_to_unit_range(self):
        image = np.array(
            [[[0.0, 0.5, 1.0], [0.9, 0.2, 0.1]]],
            dtype=np.float32,
        )
        output = ExampleEnhancer(offset=0.25).enhance(image)

        self.assertEqual(output.dtype, np.float32)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_path_input_is_loaded_through_image_reader(self):
        image = rgb_sample()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            Image.fromarray(image, mode="RGB").save(path)
            output = ExampleEnhancer().enhance(path)

        self.assertEqual(output.dtype, np.uint8)
        np.testing.assert_array_equal(output, image[:, :, ::-1])

    def test_non_numpy_output_formats_are_supported(self):
        image = rgb_sample()

        pil_output = ExampleEnhancer(output_type="pil").enhance(image)
        byte_output = ExampleEnhancer(output_type="bytes").enhance(
            image,
            output_ext="png",
        )
        base64_output = ExampleEnhancer(output_type="base64").enhance(
            image,
            output_ext="png",
        )
        file_output = Path(
            ExampleEnhancer(output_type="file").enhance(
                image,
                output_ext="png",
            )
        )

        try:
            self.assertIsInstance(pil_output, Image.Image)
            self.assertEqual(pil_output.size, (2, 2))
            self.assertTrue(byte_output.startswith(b"\x89PNG"))
            self.assertTrue(base64.b64decode(base64_output).startswith(b"\x89PNG"))
            self.assertTrue(file_output.is_file())
            with Image.open(io.BytesIO(byte_output)) as decoded:
                self.assertEqual(decoded.size, (2, 2))
        finally:
            file_output.unlink(missing_ok=True)

    def test_invalid_enhancer_return_type_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "must return np.ndarray"):
            BadReturnEnhancer().enhance(rgb_sample())

    def test_image_validation_rejects_invalid_shapes_sizes_and_dtypes(self):
        invalid_cases = (
            (np.zeros((2,), dtype=np.uint8), ValueError),
            (np.zeros((2, 2, 2), dtype=np.uint8), ValueError),
            (np.zeros((0, 2), dtype=np.uint8), ValueError),
            (np.zeros((2, 2), dtype=np.bool_), TypeError),
        )
        for image, error in invalid_cases:
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaises(error):
                    LLVEnhancer._validate_image(image)

        with self.assertRaisesRegex(TypeError, "must be np.ndarray"):
            LLVEnhancer._validate_image([[1]])


class EnhancerExportTests(unittest.TestCase):
    def test_tradition_packages_export_the_base_interface(self):
        base_exports = [
            "LLVEnhancer",
            "ImageInput",
            "OutputType",
            "EnhanceOutput",
        ]
        self.assertIs(algorithms_package.LLVEnhancer, LLVEnhancer)
        self.assertEqual(
            algorithms_package.__all__,
            [
                *base_exports,
                "AHE",
                "HE",
                "CLAHE",
                "RCLAHE",
                "DarkChannel",
                "BIMEF",
                "Gamma",
                "GCP",
                "LIME",
                "NPE",
                "SSR",
                "MSR",
                "MSRCR",
            ],
        )
        self.assertTrue(hasattr(algorithms_package, "DarkChannel"))

        self.assertIs(tradition_package.LLVEnhancer, LLVEnhancer)
        self.assertEqual(
            tradition_package.__all__,
            [*base_exports, "Predictor"],
        )


if __name__ == "__main__":
    unittest.main()
