"""Image reading and writing utilities for_teach multiple input formats."""

import base64
import tempfile
import urllib.parse
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from openLLV.data.utils import ConvertFormat

__all__ = [
    "ImageReader",
    "ImageWriter",
    "read_image",
    "write_image",
    "ImageFormat",
    "InputType",
]


class ImageFormat(Enum):
    """Supported output image formats."""

    NUMPY = "numpy"
    PIL = "pil"
    BYTES = "bytes"
    BASE64 = "base64"
    FILE = "file"


class InputType(Enum):
    """Supported input data types."""

    URL = "url"
    FILE_PATH = "file_path"
    BASE64 = "base64"
    BYTES = "bytes"
    NUMPY = "numpy"
    PIL = "pil"
    TENSOR = "tensor"
    UNKNOWN = "unknown"


class ImageReader:
    """Unified image reader for_teach common image input formats.

    The reader accepts local files, URLs, base64 strings, bytes, PIL images,
    numpy arrays, and torch tensors. For backward compatibility, numpy output
    uses OpenCV-style BGR channel order. PIL output uses RGB channel order.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
        ".webp", ".gif", ".ppm", ".pgm", ".pbm", ".sr", ".ras",
    }

    def __init__(self):
        """Initialize an image reader with default network options."""
        self.original_data = None
        self.data = None
        self.ext = None
        self.input_type = None
        self.metadata = {}

        self.timeout = 10
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.verify_ssl = True

        self.convertor = ConvertFormat()

    def __call__(self, input_data: Any, output_format: Union[str, ImageFormat] = ImageFormat.PIL, **kwargs) -> Any:
        """Read and convert image data to the requested format.

        Args:
            input_data: Image input data. Supported types include file path,
                URL, base64 string, bytes, numpy array, PIL image, and tensor.
            output_format: Output format name or ``ImageFormat`` enum value.
            **kwargs: Reader options such as ``ext``, ``timeout``, ``headers``,
                and ``verify_ssl``.

        Returns:
            Converted image data in the requested output format.

        Raises:
            ValueError: If the output format is unsupported or the input cannot
                be converted to an image.
        """
        self._apply_options(kwargs)
        self._set_input(input_data, kwargs.get("ext"))

        if isinstance(output_format, str):
            output_format = ImageFormat(output_format.lower())

        numpy_image = self._to_numpy()

        if output_format == ImageFormat.NUMPY:
            return numpy_image
        if output_format == ImageFormat.PIL:
            return self._numpy_to_pil(numpy_image)
        if output_format == ImageFormat.BYTES:
            return self._numpy_to_bytes(numpy_image, self.ext)
        if output_format == ImageFormat.BASE64:
            return self._numpy_to_base64(numpy_image, self.ext)
        if output_format == ImageFormat.FILE:
            temp_file = tempfile.NamedTemporaryFile(suffix=f".{self.ext}", delete=False)
            temp_file.close()
            cv2.imwrite(temp_file.name, self._ensure_uint8(numpy_image))
            return temp_file.name

        raise ValueError(f"Unsupported output format: {output_format}")

    def read(self, input_data: Any, output_format: Union[str, ImageFormat] = ImageFormat.PIL, **kwargs) -> Any:
        """Read and convert image data.

        Args:
            input_data: Image input data.
            output_format: Output format name or ``ImageFormat`` enum value.
            **kwargs: Reader options passed to ``__call__``.

        Returns:
            Converted image data in the requested output format.
        """
        return self(input_data, output_format=output_format, **kwargs)

    def get_info(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
        """Get metadata for_teach image input.

        Args:
            input_data: Optional image input data. If omitted, the current
                reader state is used.
            **kwargs: Reader options such as ``ext``, ``timeout``, ``headers``,
                and ``verify_ssl``.

        Returns:
            Dictionary containing detected input type, extension, shape, dtype,
            image size information, or an error message.
        """
        if input_data is not None:
            self._apply_options(kwargs)
            self._set_input(input_data, kwargs.get("ext"))
        elif self.data is None:
            return {"error": "No image data provided"}

        try:
            numpy_image = self._to_numpy()
            return {
                "input_type": self.input_type.value if self.input_type else "unknown",
                "extension": self.ext,
                "shape": numpy_image.shape,
                "height": numpy_image.shape[0],
                "width": numpy_image.shape[1],
                "channels": numpy_image.shape[2] if numpy_image.ndim == 3 else 1,
                "dtype": str(numpy_image.dtype),
                "size_kb": numpy_image.nbytes / 1024 if hasattr(numpy_image, "nbytes") else 0,
            }
        except Exception as exc:
            return {
                "input_type": self.input_type.value if self.input_type else "unknown",
                "extension": self.ext,
                "error": str(exc) or "Could not read image",
            }

    def _apply_options(self, kwargs: Dict[str, Any]) -> None:
        """Apply runtime reader options.

        Args:
            kwargs: Option dictionary containing optional network settings.
        """
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
        if "headers" in kwargs:
            self.headers.update(kwargs["headers"])
        if "verify_ssl" in kwargs:
            self.verify_ssl = kwargs["verify_ssl"]

    def _set_input(self, input_data: Any, ext=None) -> None:
        """Set current input data and detect its type and extension.

        Args:
            input_data: Image input data.
            ext: Optional explicit image extension.
        """
        self.original_data = input_data
        self.data = input_data
        self.ext = ext.lower().lstrip(".") if isinstance(ext, str) else ext
        self.input_type = self._detect_input_type()
        if self.ext is None:
            self.ext = self._detect_extension()

    def _detect_input_type(self) -> InputType:
        """Detect the current input data type.

        Returns:
            Detected ``InputType`` value.
        """
        if isinstance(self.data, Path):
            if self.data.exists():
                return InputType.FILE_PATH
            return InputType.UNKNOWN
        if isinstance(self.data, str):
            if self._is_url(self.data):
                return InputType.URL
            if Path(self.data).exists():
                return InputType.FILE_PATH
            if self._is_base64(self.data):
                return InputType.BASE64
            return InputType.UNKNOWN
        if isinstance(self.data, (bytes, bytearray)):
            return InputType.BYTES
        if isinstance(self.data, np.ndarray):
            return InputType.NUMPY
        if isinstance(self.data, Image.Image):
            return InputType.PIL
        if torch.is_tensor(self.data):
            return InputType.TENSOR
        return InputType.UNKNOWN

    @staticmethod
    def _is_url(text: str) -> bool:
        """Check whether text looks like a supported URL.

        Args:
            text: Text to inspect.

        Returns:
            True if the text has a supported URL scheme.
        """
        try:
            result = urllib.parse.urlparse(text)
            return result.scheme in {"http", "https", "ftp", "file"} and bool(result.netloc or result.path)
        except Exception:
            return False

    @staticmethod
    def _is_base64(text: str) -> bool:
        """Check whether text is valid base64 image data.

        Args:
            text: Text or data URI to inspect.

        Returns:
            True if the text can be decoded as base64.
        """
        if text.startswith("data:image") and "," in text:
            text = text.split(",", 1)[1]
        try:
            base64.b64decode(text, validate=True)
            return True
        except Exception:
            return False

    def _detect_extension(self) -> str:
        """Infer an image extension for_teach the current input.

        Returns:
            Lowercase extension without a leading dot. Defaults to ``"jpg"``
            when the extension cannot be inferred.
        """
        if self.input_type == InputType.FILE_PATH:
            suffix = Path(self.data).suffix.lower()
            if suffix in self.SUPPORTED_EXTENSIONS:
                return suffix.lstrip(".")

        if self.input_type == InputType.URL:
            path = urllib.parse.urlparse(self.data).path.lower()
            for ext in self.SUPPORTED_EXTENSIONS:
                if path.endswith(ext):
                    return ext.lstrip(".")

        if self.input_type in {InputType.BYTES, InputType.BASE64}:
            try:
                raw = self.data
                if self.input_type == InputType.BASE64:
                    raw = raw.split(",", 1)[1] if isinstance(raw, str) and "," in raw else raw
                    raw = base64.b64decode(raw)
                with Image.open(_BytesReader(raw)) as img:
                    if img.format:
                        return img.format.lower()
            except Exception:
                pass

        return "jpg"

    def _download_from_url(self, url: str) -> bytes:
        """Download image bytes from a URL.

        Args:
            url: Image URL or ``file://`` URL.

        Returns:
            Raw image bytes.

        Raises:
            ValueError: If downloading fails.
        """
        parsed = urllib.parse.urlparse(url)

        if parsed.scheme == "file":
            file_path = urllib.request.url2pathname(parsed.path)
            return self._load_from_file(file_path)

        try:
            import requests

            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            return response.content
        except ImportError:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.read()
        except Exception as exc:
            raise ValueError(f"Failed to download image from URL: {exc}") from exc

    @staticmethod
    def _load_from_file(file_path: str) -> bytes:
        """Load raw bytes from a local file.

        Args:
            file_path: Local file path.

        Returns:
            File contents as bytes.
        """
        with open(file_path, "rb") as file:
            return file.read()

    def _to_numpy(self) -> np.ndarray:
        """Convert current input data to a BGR numpy image.

        Returns:
            Image as a numpy array.

        Raises:
            ValueError: If the input type cannot be converted.
        """
        if self.input_type == InputType.NUMPY:
            return self._normalize_numpy_input(self.data)
        if self.input_type == InputType.PIL:
            return self._pil_to_numpy(self.data)
        if self.input_type == InputType.TENSOR:
            return self._tensor_to_numpy(self.data)
        if self.input_type == InputType.URL:
            return self._bytes_to_numpy(self._download_from_url(self.data))
        if self.input_type == InputType.FILE_PATH:
            return self._bytes_to_numpy(self._load_from_file(self.data))
        if self.input_type == InputType.BASE64:
            return self._base64_to_numpy(self.data)
        if self.input_type == InputType.BYTES:
            return self._bytes_to_numpy(self.data)
        if self._is_missing_file_path(self.data):
            raise FileNotFoundError(f"Image path does not exist: {self.data}")
        raise ValueError(f"Cannot convert input type {self.input_type} to numpy")

    def _is_missing_file_path(self, value: Any) -> bool:
        """Check whether an unknown input is a missing local image path."""
        if isinstance(value, Path):
            return not value.exists()

        if not isinstance(value, str) or self._is_url(value) or self._is_base64(value):
            return False

        path = Path(value)
        if path.exists():
            return False

        if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
            return True

        return any(sep in value for sep in ("/", "\\"))

    def _bytes_to_numpy(self, bytes_data: Union[bytes, bytearray]) -> np.ndarray:
        """Decode image bytes into a BGR numpy image.

        Args:
            bytes_data: Encoded image bytes.

        Returns:
            Decoded BGR image array.
        """
        rgb_image = self.convertor(convert_way="bytes2img", ext=self.ext, data=bytes_data)
        return self._rgb_to_bgr(rgb_image)

    def _base64_to_numpy(self, base64_str: Union[str, bytes, bytearray]) -> np.ndarray:
        """Decode base64 image data into a BGR numpy image.

        Args:
            base64_str: Base64 string, data URI, or bytes.

        Returns:
            Decoded BGR image array.
        """
        rgb_image = self.convertor(convert_way="base642img", ext=self.ext, data=base64_str)
        return self._rgb_to_bgr(rgb_image)

    def _numpy_to_bytes(self, numpy_image: np.ndarray, ext: str = "jpg") -> bytes:
        """Encode a BGR numpy image to image bytes.

        Args:
            numpy_image: BGR numpy image.
            ext: Output image extension.

        Returns:
            Encoded image bytes.
        """
        rgb_image = self._bgr_to_rgb(self._ensure_uint8(numpy_image))
        return self.convertor(convert_way="img2bytes", ext=ext, data=rgb_image)

    def _numpy_to_base64(self, numpy_image: np.ndarray, ext: str = "jpg") -> str:
        """Encode a BGR numpy image to base64 image data.

        Args:
            numpy_image: BGR numpy image.
            ext: Output image extension.

        Returns:
            Base64-encoded image string.
        """
        rgb_image = self._bgr_to_rgb(self._ensure_uint8(numpy_image))
        return self.convertor(convert_way="img2base64", ext=ext, data=rgb_image)

    def _pil_to_numpy(self, pil_image: Image.Image) -> np.ndarray:
        """Convert a PIL image to a BGR numpy image.

        Args:
            pil_image: PIL image.

        Returns:
            BGR numpy image.
        """
        rgb_image = np.array(pil_image.convert("RGB"))
        return self._rgb_to_bgr(rgb_image)

    def _numpy_to_pil(self, numpy_image: np.ndarray) -> Image.Image:
        """Convert a BGR numpy image to a RGB PIL image.

        Args:
            numpy_image: BGR numpy image.

        Returns:
            RGB PIL image.
        """
        rgb_image = self._bgr_to_rgb(self._ensure_uint8(numpy_image))
        return Image.fromarray(rgb_image)

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor image to a BGR numpy image.

        Args:
            tensor: Tensor with shape ``[H, W]``, ``[C, H, W]``, or
                ``[1, C, H, W]``.

        Returns:
            BGR numpy image with dtype ``uint8``.

        Raises:
            ValueError: If the tensor shape is unsupported or contains more
                than one batched image.
        """
        tensor = tensor.detach().cpu()
        if tensor.dim() == 4:
            if tensor.size(0) != 1:
                raise ValueError("Only single-image tensors are supported; got batch size > 1.")
            tensor = tensor[0]
        if tensor.dim() == 2:
            array = tensor.float().numpy()
            return _ensure_uint8_from_float(array)
        if tensor.dim() != 3:
            raise ValueError(f"Expected tensor shape [H,W], [C,H,W], or [1,C,H,W], got {tuple(tensor.shape)}.")

        array = tensor.float().numpy()
        if array.shape[0] in {1, 3, 4}:
            array = np.transpose(array, (1, 2, 0))
        array = _ensure_uint8_from_float(array)

        if array.ndim == 2:
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        if array.shape[2] == 1:
            return cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)
        if array.shape[2] == 4:
            return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR)

    @staticmethod
    def _normalize_numpy_input(image: np.ndarray) -> np.ndarray:
        """Normalize numpy image channel layout to BGR.

        Args:
            image: Input numpy image.

        Returns:
            BGR numpy image when conversion is needed; otherwise the original
            image array.
        """
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    @staticmethod
    def _ensure_uint8(image: np.ndarray) -> np.ndarray:
        """Convert an image array to ``uint8``.

        Args:
            image: Input image array.

        Returns:
            ``uint8`` image clipped to ``[0, 255]``.
        """
        if image.dtype == np.uint8:
            return image
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert a RGB image array to BGR when applicable.

        Args:
            image: Input image array.

        Returns:
            BGR image array, or the original image if conversion is not needed.
        """
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    @staticmethod
    def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert a BGR image array to RGB when applicable.

        Args:
            image: Input image array.

        Returns:
            RGB image array, or the original image if conversion is not needed.
        """
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return image


class ImageWriter:
    """Unified image writer for_teach ImageReader-compatible inputs and tensors.

    If ``output`` is a directory or None, the source filename is preserved when
    it can be inferred. If ``output`` is a file path, that exact path is used.
    The ``save_format`` argument can override the output suffix.
    """

    def __init__(self, output_dir: Union[str, Path] = "results"):
        """Initialize an image writer.

        Args:
            output_dir: Default directory used when no output path is provided.
        """
        self.output_dir = Path(output_dir)
        self.reader = ImageReader()

    def __call__(
        self,
        image: Any,
        output: Optional[Union[str, Path]] = None,
        *,
        save_format: Optional[str] = None,
        output_name: Optional[str] = None,
        **reader_kwargs: Any,
    ) -> Path:
        """Write image data to disk.

        Args:
            image: Image input data supported by ``ImageReader`` or a tensor.
            output: Optional output file path or output directory. If omitted,
                ``output_dir`` is used.
            save_format: Optional output image format or suffix.
            output_name: Optional output filename used when ``output`` is a
                directory or omitted.
            **reader_kwargs: Extra arguments passed to ``ImageReader``.

        Returns:
            Path to the saved image.
        """
        pil_image = self._to_pil(image, **reader_kwargs)
        target_path = self._resolve_output_path(
            image=image,
            output=output,
            save_format=save_format,
            output_name=output_name,
        )
        self._save_pil_image(pil_image, target_path)
        return target_path

    def write(
        self,
        image: Any,
        output: Optional[Union[str, Path]] = None,
        *,
        save_format: Optional[str] = None,
        output_name: Optional[str] = None,
        **reader_kwargs: Any,
    ) -> Path:
        """Write image data to disk.

        Args:
            image: Image input data supported by ``ImageReader`` or a tensor.
            output: Optional output file path or output directory.
            save_format: Optional output image format or suffix.
            output_name: Optional output filename.
            **reader_kwargs: Extra arguments passed to ``ImageReader``.

        Returns:
            Path to the saved image.
        """
        return self(
            image,
            output=output,
            save_format=save_format,
            output_name=output_name,
            **reader_kwargs,
        )

    def _to_pil(self, image: Any, **reader_kwargs: Any) -> Image.Image:
        """Convert supported image input to a PIL image.

        Args:
            image: Image input data.
            **reader_kwargs: Extra arguments passed to ``ImageReader``.

        Returns:
            PIL image.
        """
        if torch.is_tensor(image):
            return self._tensor_to_pil(image)
        return self.reader(image, output_format=ImageFormat.PIL, **reader_kwargs)

    def _resolve_output_path(
        self,
        *,
        image: Any,
        output: Optional[Union[str, Path]],
        save_format: Optional[str],
        output_name: Optional[str],
    ) -> Path:
        """Resolve the final output path for_teach an image.

        Args:
            image: Original image input.
            output: Optional output file path or directory.
            save_format: Optional output image format or suffix.
            output_name: Optional output filename.

        Returns:
            Resolved output path.
        """
        inferred_name = output_name or self._infer_source_name(image, save_format=save_format)
        inferred_path = Path(inferred_name)

        if save_format is not None:
            inferred_path = inferred_path.with_suffix(self._normalize_suffix(save_format))

        if output is None:
            return self.output_dir / inferred_path.name

        output_path = Path(output)
        if self._looks_like_file_path(output_path):
            if save_format is not None:
                output_path = output_path.with_suffix(self._normalize_suffix(save_format))
            return output_path

        return output_path / inferred_path.name

    @classmethod
    def _infer_source_name(cls, source: Any, save_format: Optional[str] = None) -> str:
        """Infer an output filename from the source image.

        Args:
            source: Original image input.
            save_format: Optional output image format or suffix.

        Returns:
            Inferred filename with suffix.
        """
        fallback_suffix = cls._normalize_suffix(save_format or "png")

        if isinstance(source, Path):
            return source.name if source.suffix else f"image{fallback_suffix}"

        if isinstance(source, str):
            parsed = urllib.parse.urlparse(source)
            if parsed.scheme in {"http", "https", "ftp", "file"}:
                name = Path(urllib.parse.unquote(parsed.path)).name
                if name:
                    return name

            path = Path(source)
            if path.name and path.suffix:
                return path.name

        return f"image{fallback_suffix}"

    @classmethod
    def _looks_like_file_path(cls, path: Path) -> bool:
        """Check whether a path should be treated as a file path.

        Args:
            path: Candidate path.

        Returns:
            True if the path exists as a file or has a file suffix.
        """
        if path.exists():
            return path.is_file()
        return bool(path.suffix)

    @staticmethod
    def _normalize_suffix(ext: str) -> str:
        """Normalize an image suffix.

        Args:
            ext: Image suffix or format name.

        Returns:
            Suffix with a leading dot.

        Raises:
            ValueError: If ``ext`` is empty.
        """
        ext = ext.strip()
        if not ext:
            raise ValueError("save_format must not be empty.")
        return ext if ext.startswith(".") else f".{ext}"

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert a tensor image to a PIL image.

        Args:
            tensor: Tensor with shape ``[H, W]``, ``[C, H, W]``, or
                ``[1, C, H, W]``.

        Returns:
            PIL image.

        Raises:
            ValueError: If the tensor shape is unsupported or contains more
                than one batched image.
        """
        tensor = tensor.detach().cpu()
        if tensor.dim() == 4:
            if tensor.size(0) != 1:
                raise ValueError("Only single-image tensors are supported; got batch size > 1.")
            tensor = tensor[0]
        if tensor.dim() == 2:
            array = _ensure_uint8_from_float(tensor.float().numpy())
            return Image.fromarray(array, mode="L")
        if tensor.dim() != 3:
            raise ValueError(f"Expected tensor shape [H,W], [C,H,W], or [1,C,H,W], got {tuple(tensor.shape)}.")

        array = tensor.float().numpy()
        if array.shape[0] in {1, 3, 4}:
            array = np.transpose(array, (1, 2, 0))
        array = _ensure_uint8_from_float(array)

        if array.ndim == 2:
            return Image.fromarray(array, mode="L")
        if array.shape[2] == 1:
            return Image.fromarray(array[:, :, 0], mode="L")
        if array.shape[2] == 4:
            return Image.fromarray(array, mode="RGBA")
        return Image.fromarray(array[:, :, :3], mode="RGB")

    @staticmethod
    def _save_pil_image(image: Image.Image, save_path: Path) -> None:
        """Save a PIL image to disk.

        Args:
            image: PIL image to save.
            save_path: Output image path.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_path.suffix.lower()
        output_image = image
        if suffix in {".jpg", ".jpeg"} and image.mode not in {"RGB", "L"}:
            output_image = image.convert("RGB")
        output_image.save(save_path)


def read_image(source: Any, output_format: str = "pil", **kwargs) -> Any:
    """Read image data using ``ImageReader``.

    Args:
        source: Image input data supported by ``ImageReader``.
        output_format: Output format name.
        **kwargs: Extra reader options.

    Returns:
        Converted image data.
    """
    reader = ImageReader()
    return reader(source, output_format, **kwargs)


def write_image(
    image: Any,
    output: Optional[Union[str, Path]] = None,
    *,
    save_format: Optional[str] = None,
    output_name: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """Write image data using ``ImageWriter``.

    Args:
        image: Image input data supported by ``ImageWriter``.
        output: Optional output file path or output directory.
        save_format: Optional output image format or suffix.
        output_name: Optional output filename.
        **kwargs: Extra reader options.

    Returns:
        Path to the saved image.
    """
    writer = ImageWriter()
    return writer(
        image,
        output=output,
        save_format=save_format,
        output_name=output_name,
        **kwargs,
    )


def _BytesReader(data: Union[bytes, bytearray]):
    """Create a byte stream for_teach PIL image probing.

    Args:
        data: Raw bytes.

    Returns:
        ``BytesIO`` stream.
    """
    from io import BytesIO

    return BytesIO(data)


def _ensure_uint8_from_float(image: np.ndarray) -> np.ndarray:
    """Convert a float or integer image array to ``uint8``.

    Args:
        image: Input image array. Floating point images in ``[0, 1]`` are
            scaled to ``[0, 255]`` before conversion.

    Returns:
        ``uint8`` image clipped to ``[0, 255]``.
    """
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)

    if image.size > 0 and image.max() <= 1.0:
        image = image * 255.0

    return np.clip(image, 0, 255).astype(np.uint8)
