"""Tests for :mod:`openLLV.data.utils`."""

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from openLLV.data.utils import ConvertFormat, get_img_from_folder


class ImageDiscoveryTests(unittest.TestCase):
    def test_get_img_from_folder_is_recursive_filtered_and_sorted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "nested").mkdir()
            for relative_path in (
                "z.JPG",
                "a.png",
                "nested/b.webp",
                "nested/c.HEIC",
                "nested/not_image.txt",
            ):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"content")

            with patch(
                "openLLV.data.utils.tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ):
                images = get_img_from_folder(str(root))

        expected = sorted(
            [
                str(root / "z.JPG"),
                str(root / "a.png"),
                str(root / "nested/b.webp"),
                str(root / "nested/c.HEIC"),
            ]
        )
        self.assertEqual(images, expected)

    def test_get_img_from_missing_folder_returns_empty_list(self):
        with patch(
            "openLLV.data.utils.tqdm",
            side_effect=lambda iterable, **kwargs: iterable,
        ):
            images = get_img_from_folder("definitely-missing-directory")

        self.assertEqual(images, [])


class ConvertFormatTests(unittest.TestCase):
    @staticmethod
    def _rgb_image():
        return np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [10, 20, 30]],
            ],
            dtype=np.uint8,
        )

    def test_png_bytes_round_trip_preserves_rgb_values(self):
        image = self._rgb_image()
        converter = ConvertFormat()

        encoded = converter(image, "img2bytes", ext=".PNG")
        decoded = converter(encoded, "bytes2img", ext="png")

        self.assertIsInstance(encoded, bytes)
        np.testing.assert_array_equal(decoded, image)

    def test_base64_round_trip_and_data_uri_are_supported(self):
        image = self._rgb_image()
        converter = ConvertFormat()

        encoded = converter(image, "img2base64", ext="png")
        decoded = converter(
            f"data:image/png;base64,{encoded}",
            "base642img",
            ext="png",
        )

        self.assertIsInstance(encoded, str)
        base64.b64decode(encoded, validate=True)
        np.testing.assert_array_equal(decoded, image)

    def test_bytes_decoder_accepts_bytearray(self):
        converter = ConvertFormat()
        encoded = converter(self._rgb_image(), "img2bytes", ext="png")

        decoded = converter(bytearray(encoded), "bytes2img", ext="png")

        np.testing.assert_array_equal(decoded, self._rgb_image())

    def test_grayscale_input_is_not_channel_swapped(self):
        image = np.array([[0, 127], [200, 255]], dtype=np.uint8)
        converter = ConvertFormat()

        encoded = converter(image, "img2bytes", ext="png")
        decoded_bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_GRAYSCALE)

        np.testing.assert_array_equal(decoded_bgr, image)

    def test_call_updates_conversion_state(self):
        converter = ConvertFormat()
        image = self._rgb_image()

        result = converter(image, "img2bytes", ext=".png")

        self.assertEqual(converter.ext, "png")
        self.assertEqual(converter.convert_way, "img2bytes")
        self.assertIs(converter.data, result)

    def test_call_rejects_missing_data(self):
        with self.assertRaisesRegex(ValueError, "No data stream"):
            ConvertFormat()(None, "img2bytes")

    def test_call_rejects_unknown_conversion(self):
        with self.assertRaisesRegex(ValueError, "Unsupported convert_way"):
            ConvertFormat()(self._rgb_image(), "unknown")

    def test_bytes_to_image_rejects_wrong_type_and_invalid_payload(self):
        converter = ConvertFormat()

        with self.assertRaisesRegex(TypeError, "bytes2img expects"):
            converter("not bytes", "bytes2img")
        with self.assertRaisesRegex(ValueError, "Failed to decode"):
            converter(b"not an encoded image", "bytes2img")

    def test_image_to_bytes_rejects_wrong_type_and_encode_failure(self):
        converter = ConvertFormat()

        with self.assertRaisesRegex(TypeError, "img2bytes expects"):
            converter("not an array", "img2bytes")

        with patch("openLLV.data.utils.cv2.imencode", return_value=(False, None)):
            with self.assertRaisesRegex(ValueError, "Failed to encode"):
                converter(self._rgb_image(), "img2bytes", ext="png")

    def test_base64_to_img_rejects_wrong_type_and_invalid_payload(self):
        converter = ConvertFormat()

        with self.assertRaisesRegex(TypeError, "base642img expects"):
            converter(123, "base642img")
        with self.assertRaisesRegex(ValueError, "Failed to decode"):
            converter("not-base64!", "base642img")
        with self.assertRaisesRegex(ValueError, "Failed to decode image"):
            converter(base64.b64encode(b"not an image"), "base642img")

    def test_img_to_base64_rejects_wrong_type_and_encode_failure(self):
        converter = ConvertFormat()

        with self.assertRaisesRegex(TypeError, "img2base64 expects"):
            converter([], "img2base64")

        with patch("openLLV.data.utils.cv2.imencode", return_value=(False, None)):
            with self.assertRaisesRegex(ValueError, "Failed to encode"):
                converter(self._rgb_image(), "img2base64", ext="png")

    def test_rgb_to_bgr_helper_swaps_only_three_channel_images(self):
        rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
        gray = np.array([[1]], dtype=np.uint8)

        bgr = ConvertFormat._rgb_to_bgr_if_needed(rgb)
        unchanged = ConvertFormat._rgb_to_bgr_if_needed(gray)

        np.testing.assert_array_equal(bgr, np.array([[[3, 2, 1]]], dtype=np.uint8))
        self.assertIs(unchanged, gray)


if __name__ == "__main__":
    unittest.main()
