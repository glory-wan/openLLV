"""Tests for :mod:`openLLV.data.image_io`."""

import base64
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from PIL import Image

import openLLV.data as data_package
from openLLV.data.image_io import (
    ImageFormat,
    ImageReader,
    ImageWriter,
    InputType,
    _BytesReader,
    _ensure_uint8_from_float,
    read_image,
    write_image,
)


def rgb_image():
    return np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [10, 20, 30]],
        ],
        dtype=np.uint8,
    )


def png_bytes(image=None):
    buffer = io.BytesIO()
    Image.fromarray(rgb_image() if image is None else image).save(buffer, format="PNG")
    return buffer.getvalue()


class ImageReaderTests(unittest.TestCase):
    def test_format_and_input_type_enums_have_expected_values(self):
        self.assertEqual(ImageFormat.NUMPY.value, "numpy")
        self.assertEqual(ImageFormat.FILE.value, "file")
        self.assertEqual(InputType.FILE_PATH.value, "file_path")
        self.assertEqual(InputType.TENSOR.value, "tensor")
        self.assertEqual(InputType.UNKNOWN.value, "unknown")

    def test_numpy_rgb_input_produces_bgr_numpy_output(self):
        output = ImageReader()(rgb_image(), output_format="numpy")

        np.testing.assert_array_equal(output, rgb_image()[:, :, ::-1])

    def test_numpy_input_round_trips_to_rgb_pil(self):
        output = ImageReader()(rgb_image(), output_format=ImageFormat.PIL)

        self.assertIsInstance(output, Image.Image)
        np.testing.assert_array_equal(np.asarray(output), rgb_image())

    def test_pil_input_produces_bgr_numpy_output(self):
        image = Image.fromarray(rgb_image(), mode="RGB")

        output = ImageReader()(image, output_format="numpy")

        np.testing.assert_array_equal(output, rgb_image()[:, :, ::-1])

    def test_encoded_bytes_and_bytearray_are_supported(self):
        reader = ImageReader()

        for payload in (png_bytes(), bytearray(png_bytes())):
            with self.subTest(payload_type=type(payload).__name__):
                output = reader(payload, output_format="numpy")
                np.testing.assert_array_equal(output, rgb_image()[:, :, ::-1])
                self.assertEqual(reader.input_type, InputType.BYTES)
                self.assertEqual(reader.ext, "png")

    def test_base64_and_data_uri_are_supported(self):
        encoded = base64.b64encode(png_bytes()).decode("ascii")
        reader = ImageReader()

        plain = reader(encoded, output_format="numpy")
        data_uri = reader(
            f"data:image/png;base64,{encoded}",
            output_format="numpy",
        )

        np.testing.assert_array_equal(plain, rgb_image()[:, :, ::-1])
        np.testing.assert_array_equal(data_uri, plain)
        self.assertEqual(reader.input_type, InputType.BASE64)

    def test_string_and_path_file_inputs_are_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            Image.fromarray(rgb_image()).save(path)
            reader = ImageReader()

            from_string = reader(str(path), output_format="numpy")
            self.assertEqual(reader.input_type, InputType.FILE_PATH)
            from_path = reader(path, output_format="numpy")

        np.testing.assert_array_equal(from_string, rgb_image()[:, :, ::-1])
        np.testing.assert_array_equal(from_path, from_string)

    def test_file_url_input_uses_local_file_loader(self):
        reader = ImageReader()
        with patch.object(reader, "_load_from_file", return_value=png_bytes()) as load_mock:
            output = reader("file:///temporary/sample.png", output_format="numpy")

        np.testing.assert_array_equal(output, rgb_image()[:, :, ::-1])
        self.assertEqual(reader.input_type, InputType.URL)
        self.assertEqual(reader.ext, "png")
        load_mock.assert_called_once()

    def test_tensor_inputs_support_chw_and_single_batch(self):
        tensor = torch.from_numpy(rgb_image()).permute(2, 0, 1).float() / 255.0
        reader = ImageReader()

        chw = reader(tensor, output_format="numpy")
        batched = reader(tensor.unsqueeze(0), output_format="numpy")

        np.testing.assert_array_equal(chw, rgb_image()[:, :, ::-1])
        np.testing.assert_array_equal(batched, chw)

    def test_tensor_grayscale_and_rgba_inputs_are_supported(self):
        gray = torch.tensor([[0.0, 1.0]])
        rgba = torch.tensor([[[1.0]], [[0.0]], [[0.0]], [[0.5]]])
        reader = ImageReader()

        gray_output = reader(gray, output_format="numpy")
        rgba_output = reader(rgba, output_format="numpy")

        np.testing.assert_array_equal(gray_output, np.array([[0, 255]], dtype=np.uint8))
        np.testing.assert_array_equal(rgba_output, np.array([[[0, 0, 255]]], dtype=np.uint8))

    def test_tensor_rejects_multi_image_batch_and_invalid_rank(self):
        reader = ImageReader()

        with self.assertRaisesRegex(ValueError, "batch size > 1"):
            reader(torch.zeros(2, 3, 4, 4), output_format="numpy")
        with self.assertRaisesRegex(ValueError, "Expected tensor shape"):
            reader(torch.zeros(5), output_format="numpy")

    def test_output_bytes_and_base64_are_valid_encoded_images(self):
        reader = ImageReader()

        encoded_bytes = reader(rgb_image(), output_format="bytes", ext="png")
        encoded_base64 = reader(rgb_image(), output_format="base64", ext="png")
        decoded_bytes = np.asarray(Image.open(io.BytesIO(encoded_bytes)).convert("RGB"))
        decoded_base64 = np.asarray(
            Image.open(io.BytesIO(base64.b64decode(encoded_base64))).convert("RGB")
        )

        np.testing.assert_array_equal(decoded_bytes, rgb_image())
        np.testing.assert_array_equal(decoded_base64, rgb_image())

    def test_output_file_creates_readable_temporary_image(self):
        path = Path(ImageReader()(rgb_image(), output_format="file", ext="png"))
        try:
            self.assertTrue(path.exists())
            np.testing.assert_array_equal(
                np.asarray(Image.open(path).convert("RGB")),
                rgb_image(),
            )
        finally:
            path.unlink(missing_ok=True)

    def test_read_alias_delegates_to_call(self):
        reader = ImageReader()
        output = reader.read(rgb_image(), "pil")

        self.assertIsInstance(output, Image.Image)
        np.testing.assert_array_equal(np.asarray(output), rgb_image())

    def test_get_info_reports_metadata_and_empty_state(self):
        reader = ImageReader()
        self.assertEqual(reader.get_info(), {"error": "No image data provided"})

        info = reader.get_info(rgb_image(), ext="png")

        self.assertEqual(info["input_type"], "numpy")
        self.assertEqual(info["extension"], "png")
        self.assertEqual(info["shape"], (2, 2, 3))
        self.assertEqual(info["height"], 2)
        self.assertEqual(info["width"], 2)
        self.assertEqual(info["channels"], 3)
        self.assertEqual(info["dtype"], "uint8")
        self.assertGreater(info["size_kb"], 0)

    def test_get_info_returns_error_instead_of_raising(self):
        info = ImageReader().get_info(Path("missing-image.png"))

        self.assertEqual(info["input_type"], "unknown")
        self.assertIn("does not exist", info["error"])

    def test_runtime_options_are_applied(self):
        reader = ImageReader()

        reader(
            rgb_image(),
            output_format="numpy",
            timeout=3,
            headers={"X-Test": "yes"},
            verify_ssl=False,
        )

        self.assertEqual(reader.timeout, 3)
        self.assertEqual(reader.headers["X-Test"], "yes")
        self.assertFalse(reader.verify_ssl)

    def test_invalid_output_format_is_rejected(self):
        with self.assertRaises(ValueError):
            ImageReader()(rgb_image(), output_format="unsupported")

    def test_missing_path_and_unknown_input_are_rejected(self):
        reader = ImageReader()

        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            reader(Path("missing.png"), output_format="numpy")
        with self.assertRaisesRegex(ValueError, "Cannot convert"):
            reader(object(), output_format="numpy")

    def test_url_and_base64_detection_helpers(self):
        self.assertTrue(ImageReader._is_url("https://example.com/image.png"))
        self.assertTrue(ImageReader._is_url("file:///tmp/image.png"))
        self.assertFalse(ImageReader._is_url("plain text"))
        encoded = base64.b64encode(png_bytes()).decode("ascii")
        self.assertTrue(ImageReader._is_base64(encoded))
        self.assertTrue(ImageReader._is_base64(f"data:image/png;base64,{encoded}"))
        self.assertFalse(ImageReader._is_base64("not base64!"))

    def test_extension_detection_for_file_url_bytes_base64_and_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.PNG"
            path.write_bytes(png_bytes())
            reader = ImageReader()

            reader._set_input(path)
            self.assertEqual(reader.ext, "png")
            reader._set_input("https://example.com/sample.webp")
            self.assertEqual(reader.ext, "webp")
            reader._set_input(png_bytes())
            self.assertEqual(reader.ext, "png")
            reader._set_input(base64.b64encode(png_bytes()).decode("ascii"))
            self.assertEqual(reader.ext, "png")
            reader._set_input(np.zeros((1, 1, 3), dtype=np.uint8))
            self.assertEqual(reader.ext, "jpg")
            reader._set_input(np.zeros((1, 1, 3), dtype=np.uint8), ext=".TIFF")
            self.assertEqual(reader.ext, "tiff")

    def test_http_download_uses_requests_options(self):
        response = Mock()
        response.content = b"payload"
        response.raise_for_status = Mock()
        requests_module = types.ModuleType("requests")
        requests_module.get = Mock(return_value=response)
        reader = ImageReader()
        reader.timeout = 4
        reader.verify_ssl = False

        with patch.dict(sys.modules, {"requests": requests_module}):
            result = reader._download_from_url("https://example.com/image.png")

        self.assertEqual(result, b"payload")
        requests_module.get.assert_called_once_with(
            "https://example.com/image.png",
            headers=reader.headers,
            timeout=4,
            verify=False,
        )
        response.raise_for_status.assert_called_once_with()

    def test_http_download_wraps_requests_error(self):
        requests_module = types.ModuleType("requests")
        requests_module.get = Mock(side_effect=RuntimeError("network failed"))

        with patch.dict(sys.modules, {"requests": requests_module}):
            with self.assertRaisesRegex(ValueError, "network failed"):
                ImageReader()._download_from_url("https://example.com/image.png")

    def test_download_falls_back_to_urllib_without_requests(self):
        response = Mock()
        response.read.return_value = b"urllib payload"
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)

        with patch.dict(sys.modules, {"requests": None}):
            with patch("openLLV.data.image_io.urllib.request.urlopen", return_value=context) as open_mock:
                result = ImageReader()._download_from_url("https://example.com/image.png")

        self.assertEqual(result, b"urllib payload")
        open_mock.assert_called_once()

    def test_load_from_file_returns_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "payload.bin"
            path.write_bytes(b"abc")

            self.assertEqual(ImageReader._load_from_file(str(path)), b"abc")

    def test_numpy_normalization_supports_gray_rgb_and_rgba(self):
        gray = np.array([[5]], dtype=np.uint8)
        rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
        rgba = np.array([[[1, 2, 3, 255]]], dtype=np.uint8)

        np.testing.assert_array_equal(
            ImageReader._normalize_numpy_input(gray),
            np.array([[[5, 5, 5]]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            ImageReader._normalize_numpy_input(rgb),
            np.array([[[3, 2, 1]]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            ImageReader._normalize_numpy_input(rgba),
            np.array([[[3, 2, 1]]], dtype=np.uint8),
        )

    def test_uint8_and_color_helpers_cover_channel_variants(self):
        clipped = ImageReader._ensure_uint8(
            np.array([[-1.0, 128.9, 300.0]], dtype=np.float32)
        )
        np.testing.assert_array_equal(clipped, np.array([[0, 128, 255]], dtype=np.uint8))

        rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
        bgr = ImageReader._rgb_to_bgr(rgb)
        np.testing.assert_array_equal(bgr, np.array([[[3, 2, 1]]], dtype=np.uint8))
        np.testing.assert_array_equal(ImageReader._bgr_to_rgb(bgr), rgb)


class ImageWriterTests(unittest.TestCase):
    def test_writer_uses_default_directory_and_fallback_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ImageWriter(Path(temp_dir) / "results")

            saved = writer(rgb_image(), save_format="png")

            self.assertEqual(saved.name, "image.png")
            self.assertTrue(saved.exists())
            np.testing.assert_array_equal(np.asarray(Image.open(saved).convert("RGB")), rgb_image())

    def test_writer_accepts_explicit_file_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "nested" / "custom.png"

            saved = ImageWriter()(rgb_image(), output=target)

            self.assertEqual(saved, target)
            self.assertTrue(target.exists())

    def test_writer_accepts_output_directory_and_output_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "created-directory"

            saved = ImageWriter()(rgb_image(), output=output_dir, output_name="named.png")

            self.assertEqual(saved, output_dir / "named.png")
            self.assertTrue(saved.exists())

    def test_writer_preserves_source_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            destination = root / "destination"
            Image.fromarray(rgb_image()).save(source)

            saved = ImageWriter()(source, output=destination)

            self.assertEqual(saved, destination / "source.png")

    def test_save_format_overrides_inferred_and_explicit_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            writer = ImageWriter()

            explicit = writer(
                rgb_image(),
                output=root / "output.png",
                save_format="bmp",
            )
            directory = writer(
                rgb_image(),
                output=root / "directory",
                output_name="name.jpg",
                save_format="png",
            )

            self.assertEqual(explicit.suffix, ".bmp")
            self.assertEqual(directory.name, "name.png")

    def test_write_alias_delegates_to_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ImageWriter()
            saved = writer.write(rgb_image(), Path(temp_dir) / "image.png")

            self.assertTrue(saved.exists())
            np.testing.assert_array_equal(
                np.asarray(Image.open(saved).convert("RGB")),
                rgb_image(),
            )

    def test_tensor_to_pil_supports_gray_rgb_rgba_and_single_batch(self):
        gray = ImageWriter._tensor_to_pil(torch.tensor([[0.0, 1.0]]))
        rgb = ImageWriter._tensor_to_pil(torch.ones(3, 2, 2))
        rgba = ImageWriter._tensor_to_pil(torch.ones(4, 2, 2))
        batched = ImageWriter._tensor_to_pil(torch.ones(1, 3, 2, 2))

        self.assertEqual(gray.mode, "L")
        self.assertEqual(rgb.mode, "RGB")
        self.assertEqual(rgba.mode, "RGBA")
        self.assertEqual(batched.mode, "RGB")

    def test_tensor_to_pil_rejects_batch_and_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "batch size > 1"):
            ImageWriter._tensor_to_pil(torch.zeros(2, 3, 4, 4))
        with self.assertRaisesRegex(ValueError, "Expected tensor shape"):
            ImageWriter._tensor_to_pil(torch.zeros(5))

    def test_rgba_tensor_can_be_saved_as_jpeg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "rgba.jpg"

            saved = ImageWriter()(torch.ones(4, 2, 2), output=target)

            self.assertTrue(saved.exists())
            with Image.open(saved) as image:
                self.assertEqual(image.mode, "RGB")

    def test_output_path_helpers_cover_url_path_directory_and_suffix(self):
        self.assertEqual(
            ImageWriter._infer_source_name("https://example.com/a%20b.png"),
            "a b.png",
        )
        self.assertEqual(ImageWriter._infer_source_name(object(), "jpg"), "image.jpg")
        self.assertEqual(ImageWriter._normalize_suffix("png"), ".png")
        self.assertEqual(ImageWriter._normalize_suffix(".bmp"), ".bmp")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            ImageWriter._normalize_suffix("   ")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing_dir = root / "directory.with.dot"
            existing_dir.mkdir()
            existing_file = root / "file"
            existing_file.write_bytes(b"")

            self.assertFalse(ImageWriter._looks_like_file_path(existing_dir))
            self.assertTrue(ImageWriter._looks_like_file_path(existing_file))
            self.assertTrue(ImageWriter._looks_like_file_path(root / "new.png"))
            self.assertFalse(ImageWriter._looks_like_file_path(root / "new-directory"))

    def test_public_read_and_write_helpers_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "roundtrip.png"

            saved = write_image(rgb_image(), target)
            loaded = read_image(saved, output_format="pil")

            self.assertEqual(saved, target)
            np.testing.assert_array_equal(np.asarray(loaded), rgb_image())


class ImageIOHelperTests(unittest.TestCase):
    def test_bytes_reader_returns_seekable_stream(self):
        stream = _BytesReader(b"abc")
        self.assertEqual(stream.read(), b"abc")
        stream.seek(0)
        self.assertEqual(stream.read(1), b"a")

    def test_ensure_uint8_from_float_scales_unit_range_and_clips_other_values(self):
        unit = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        wider = np.array([[-1.0, 128.9, 300.0]], dtype=np.float32)
        already = np.array([[1]], dtype=np.uint8)

        np.testing.assert_array_equal(
            _ensure_uint8_from_float(unit),
            np.array([[0, 127, 255]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            _ensure_uint8_from_float(wider),
            np.array([[0, 128, 255]], dtype=np.uint8),
        )
        self.assertIs(_ensure_uint8_from_float(already), already)

    def test_data_package_exports_public_image_io_api(self):
        for name in (
            "ImageReader",
            "ImageWriter",
            "read_image",
            "write_image",
            "ImageFormat",
            "InputType",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(data_package, name))


if __name__ == "__main__":
    unittest.main()
