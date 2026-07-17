"""Tests for the migrated traditional-algorithm Predictor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import openLLV.tradition as tradition_package
from openLLV.tradition import Predictor
from openLLV.tradition.algorithms import LLVEnhancer


class PredictorTestEnhancer(LLVEnhancer):
    name = "predictor-test"
    aliases = ["predictor-alias"]

    def __init__(self, offset: float = 0.0, **kwargs):
        self.offset = float(offset)
        super().__init__(**kwargs)

    def _enhance(self, image: np.ndarray, **kwargs) -> np.ndarray:
        offset = float(kwargs.get("offset", self.offset))
        return image.astype(np.float32) + offset

    def get_params(self):
        params = super().get_params()
        params["offset"] = self.offset
        return params


class NonNumpyEnhancer(LLVEnhancer):
    name = "predictor-non-numpy"

    def _enhance(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return image

    def __call__(self, image, **kwargs):
        return Image.new("RGB", (2, 2))


def rgb_sample() -> np.ndarray:
    return np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
        ],
        dtype=np.uint8,
    )


def save_rgb(path: Path, image: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_sample() if image is None else image, mode="RGB").save(path)


class PredictorInitializationTests(unittest.TestCase):
    def test_registered_name_loads_enhancer_and_kwargs_override_config(self):
        predictor = Predictor(
            method="PREDICTOR-ALIAS",
            config={"offset": 2, "clip_output": False},
            offset=5,
        )

        self.assertIsInstance(predictor.enhancer, PredictorTestEnhancer)
        self.assertEqual(predictor.enhancer.offset, 5.0)
        self.assertFalse(predictor.enhancer.clip_output)
        self.assertEqual(predictor.enhancer.output_type, "numpy")
        self.assertEqual(predictor.method_name, "predictor-test")
        self.assertEqual(
            predictor.output_dir,
            Path("results") / "predictor-test",
        )

    def test_custom_output_directory_is_normalized_to_path(self):
        predictor = Predictor(
            method="predictor-test",
            output_dir="custom-results",
        )
        self.assertEqual(predictor.output_dir, Path("custom-results"))

    def test_existing_enhancer_is_reused_configured_and_forced_to_numpy(self):
        enhancer = PredictorTestEnhancer(offset=1, output_type="pil")
        predictor = Predictor(
            method=enhancer,
            config={"offset": 3},
            keep_dtype=False,
        )

        self.assertIs(predictor.enhancer, enhancer)
        self.assertEqual(enhancer.offset, 3)
        self.assertFalse(enhancer.keep_dtype)
        self.assertEqual(enhancer.output_type, "numpy")

    def test_invalid_method_type_and_unknown_name_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "method must be str or LLVEnhancer"):
            Predictor(method=object())
        with self.assertRaisesRegex(ValueError, "not registered"):
            Predictor(method="missing-traditional-method")

    def test_available_methods_come_from_enhancer_registry(self):
        methods = Predictor.list_available_methods()
        self.assertIn("predictor-test", methods)
        self.assertIn("predictor-alias", methods)
        self.assertIn("dcp", methods)

    def test_get_params_combines_predictor_and_enhancer_metadata(self):
        predictor = Predictor(
            method="predictor-test",
            output_dir="outputs",
            offset=4,
        )

        self.assertEqual(
            predictor.get_params(),
            {
                "method": "predictor-test",
                "output_dir": "outputs",
                "enhancer": {
                    "name": "predictor-test",
                    "output_type": "numpy",
                    "keep_dtype": True,
                    "clip_output": True,
                    "offset": 4.0,
                },
            },
        )


class PredictorSingleImageTests(unittest.TestCase):
    def test_predict_single_without_saving_returns_numpy_output(self):
        predictor = Predictor(method="predictor-test")
        enhanced, saved = predictor.predict_single(rgb_sample(), save=False)

        self.assertIsNone(saved)
        self.assertEqual(enhanced.dtype, np.uint8)
        np.testing.assert_array_equal(enhanced, rgb_sample()[:, :, ::-1])

    def test_runtime_kwargs_are_forwarded_to_enhancer(self):
        enhanced, _ = Predictor(method="predictor-test").predict_single(
            rgb_sample(),
            save=False,
            offset=5,
        )
        expected = np.clip(
            rgb_sample()[:, :, ::-1].astype(np.float32) + 5,
            0,
            255,
        ).astype(np.uint8)
        np.testing.assert_array_equal(enhanced, expected)

    def test_default_output_directory_and_custom_name_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "default"
            predictor = Predictor(
                method="predictor-test",
                output_dir=output_dir,
            )

            enhanced, saved = predictor.predict_single(
                rgb_sample(),
                output_name="renamed.jpg",
                output_ext="png",
            )

            self.assertEqual(saved, output_dir / "renamed.png")
            self.assertTrue(saved.is_file())
            with Image.open(saved) as image:
                self.assertEqual(image.size, (3, 2))
            self.assertEqual(enhanced.shape, (2, 3, 3))

    def test_source_name_is_preserved_for_directory_and_explicit_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            save_rgb(source)
            predictor = Predictor(method="predictor-test")

            _, directory_saved = predictor.predict_single(
                source,
                save_path=root / "directory-output",
            )
            explicit = root / "explicit" / "custom.bmp"
            _, explicit_saved = predictor.predict_single(
                source,
                save_path=explicit,
            )

            self.assertEqual(
                directory_saved,
                root / "directory-output" / "source.png",
            )
            self.assertEqual(explicit_saved, explicit)
            self.assertTrue(directory_saved.is_file())
            self.assertTrue(explicit_saved.is_file())

    def test_call_and_predict_route_single_images(self):
        predictor = Predictor(method="predictor-test")

        called, called_path = predictor(
            rgb_sample(),
            save=False,
        )
        predicted, predicted_path = predictor.predict(
            rgb_sample(),
            save=False,
        )

        self.assertIsNone(called_path)
        self.assertIsNone(predicted_path)
        np.testing.assert_array_equal(called, predicted)

    def test_non_numpy_enhancer_output_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "expects numpy output"):
            Predictor(method=NonNumpyEnhancer()).predict_single(
                rgb_sample(),
                save=False,
            )


class PredictorBatchTests(unittest.TestCase):
    def test_recursive_batch_preserves_relative_paths_and_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "inputs"
            output_dir = root / "outputs"
            save_rgb(input_dir / "z.png")
            save_rgb(input_dir / "nested" / "A.jpg")
            (input_dir / "ignore.txt").write_text("not an image", encoding="utf-8")

            saved = Predictor(method="predictor-test").predict_batch(
                input_dir,
                output_dir=output_dir,
                progress_bar=False,
            )

            self.assertEqual(
                saved,
                [output_dir / "nested" / "A.jpg", output_dir / "z.png"],
            )
            self.assertTrue(all(path.is_file() for path in saved))
            for path in saved:
                with Image.open(path) as image:
                    self.assertEqual(image.size, (3, 2))

    def test_directory_call_routes_to_batch_predictor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "inputs"
            output_dir = root / "outputs"
            save_rgb(input_dir / "sample.png")

            saved = Predictor(method="predictor-test")(
                input_dir,
                output=output_dir,
                progress_bar=False,
            )

            self.assertEqual(saved, [output_dir / "sample.png"])

    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                Predictor(method="predictor-test").predict_batch(
                    temp_dir,
                    progress_bar=False,
                ),
                [],
            )

    def test_non_directory_batch_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sample.png"
            save_rgb(file_path)
            with self.assertRaisesRegex(NotADirectoryError, "must be a directory"):
                Predictor(method="predictor-test").predict_batch(file_path)


class PredictorHelperTests(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor(
            method="predictor-test",
            output_dir="results",
        )

    def test_list_images_is_recursive_filtered_and_case_insensitive_sorted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_rgb(root / "z.PNG")
            save_rgb(root / "nested" / "A.JpG")
            (root / "x.txt").write_text("x", encoding="utf-8")

            self.assertEqual(
                Predictor._list_images(root),
                [root / "nested" / "A.JpG", root / "z.PNG"],
            )

    def test_directory_detection_only_accepts_existing_pathlike_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            file_path = root / "file.png"
            save_rgb(file_path)

            self.assertTrue(Predictor._is_directory_source(root))
            self.assertTrue(Predictor._is_directory_source(str(root)))
            self.assertFalse(Predictor._is_directory_source(file_path))
            self.assertFalse(Predictor._is_directory_source(rgb_sample()))

    def test_source_name_inference_supports_paths_urls_and_fallback(self):
        self.assertEqual(
            Predictor._infer_source_name(Path("folder/sample.png")),
            "sample.png",
        )
        self.assertEqual(
            Predictor._infer_source_name(
                "https://example.com/images/a%20b.webp?download=1"
            ),
            "a b.webp",
        )
        self.assertEqual(
            Predictor._infer_source_name("relative/image.bmp"),
            "image.bmp",
        )
        self.assertEqual(Predictor._infer_source_name(rgb_sample()), "image.jpg")

    def test_output_path_resolution_handles_default_directory_and_file_paths(self):
        self.assertEqual(
            self.predictor._resolve_single_output_path(
                image=rgb_sample(),
                save_path=None,
                output_name=None,
                output_ext="png",
            ),
            Path("results") / "image.png",
        )
        self.assertEqual(
            self.predictor._resolve_single_output_path(
                image=Path("source.jpg"),
                save_path=Path("custom") / "result.bmp",
                output_name=None,
                output_ext=None,
            ),
            Path("custom") / "result.bmp",
        )
        self.assertEqual(
            self.predictor._resolve_single_output_path(
                image=Path("source.jpg"),
                save_path=Path("directory"),
                output_name=None,
                output_ext=None,
            ),
            Path("directory") / "source.jpg",
        )

    def test_suffix_and_file_path_helpers(self):
        self.assertEqual(Predictor._normalize_suffix("png"), ".png")
        self.assertEqual(Predictor._normalize_suffix(".JPG"), ".JPG")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            Predictor._normalize_suffix("  ")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = root / "folder.with.dot"
            directory.mkdir()
            file_path = root / "file"
            file_path.write_bytes(b"")
            self.assertFalse(Predictor._looks_like_file_path(directory))
            self.assertTrue(Predictor._looks_like_file_path(file_path))
            self.assertTrue(Predictor._looks_like_file_path(root / "new.png"))
            self.assertFalse(Predictor._looks_like_file_path(root / "new-dir"))

    def test_ensure_uint8_scales_unit_float_and_clips_other_ranges(self):
        original = np.array([[1]], dtype=np.uint8)
        self.assertIs(Predictor._ensure_uint8(original), original)
        np.testing.assert_array_equal(
            Predictor._ensure_uint8(
                np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
            ),
            np.array([[0, 127, 255]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            Predictor._ensure_uint8(
                np.array([[-1.0, 128.9, 300.0]], dtype=np.float32)
            ),
            np.array([[0, 128, 255]], dtype=np.uint8),
        )
        empty = Predictor._ensure_uint8(np.array([], dtype=np.float32))
        self.assertEqual(empty.dtype, np.uint8)
        self.assertEqual(empty.size, 0)

    def test_save_numpy_image_creates_parent_and_reports_opencv_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "nested" / "image.png"
            Predictor._save_numpy_image(rgb_sample()[:, :, ::-1], target)
            self.assertTrue(target.is_file())

            failed = Path(temp_dir) / "failed" / "image.png"
            with patch(
                "openLLV.tradition.predictor.cv2.imwrite",
                return_value=False,
            ):
                with self.assertRaisesRegex(ValueError, "Failed to save image"):
                    Predictor._save_numpy_image(rgb_sample(), failed)
            self.assertTrue(failed.parent.is_dir())

    def test_tradition_package_exports_predictor(self):
        self.assertIs(tradition_package.Predictor, Predictor)
        self.assertIn("Predictor", tradition_package.__all__)


if __name__ == "__main__":
    unittest.main()
