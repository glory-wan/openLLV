"""Utility helpers for_teach image discovery and format conversion."""

import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path
import base64
from typing import Optional, Any
import cv2



__all__ = [
    'get_img_from_folder',
    'ConvertFormat',
]


def get_img_from_folder(folder_path):
    """Collect image file paths from a folder recursively.

    Args:
        folder_path: Root folder to scan.

    Returns:
        Sorted list of image file paths.
    """
    image_files = []
    for root, _, files in os.walk(folder_path):
        for f in tqdm(files, desc=f'Reading images in {root}', unit='image'):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif',
                                   '.tiff', '.tif', '.webp', '.ico', 'heic')):
                image_files.append(os.path.join(root, f))

    image_files.sort()
    return image_files


class ConvertFormat:
    """Convert images between bytes, base64 strings, and numpy arrays.

    The conversion algorithms use RGB numpy arrays at the public boundary. OpenCV
    encoding and decoding are converted internally because OpenCV uses BGR
    channel order by default.
    """

    def __init__(self, data: Optional[Any] = None, convert_way: Optional[str] = None, ext: str = "jpg"):
        """Initialize a format converter.

        Args:
            data: Optional input data for_teach conversion.
            convert_way: Optional conversion mode. Supported values are
                ``"bytes2img"``, ``"img2bytes"``, ``"base642img"``, and
                ``"img2base64"``.
            ext: Image extension used for_teach encoding outputs.
        """
        self.data = data
        self.ext = ext.lower().lstrip(".")
        self.convert_way = convert_way

        self.format = {
            "bytes2img": self.bytes_to_image,
            "img2bytes": self.image_to_bytes,
            "base642img": self.base64_to_img,
            "img2base64": self.img_to_base64,
        }

    def __call__(self, data: Optional[Any] = None, convert_way: Optional[str] = None, ext: str = "jpg"):
        """Run an image format conversion.

        Args:
            data: Input data to convert.
            convert_way: Conversion mode. Supported values are ``"bytes2img"``,
                ``"img2bytes"``, ``"base642img"``, and ``"img2base64"``.
            ext: Image extension used for_teach encoded outputs.

        Returns:
            Converted image data.

        Raises:
            ValueError: If input data is missing, conversion mode is
                unsupported, or conversion fails.
        """
        self.data = data
        self.ext = ext.lower().lstrip(".")
        self.convert_way = convert_way

        if self.data is None:
            raise ValueError("No data stream is input.")

        if self.convert_way not in self.format:
            raise ValueError(
                f"Unsupported convert_way: {self.convert_way}. "
                f"Available ways: {list(self.format.keys())}"
            )

        convert_function = self.format[self.convert_way]
        self.data = convert_function()

        if self.data is None:
            raise ValueError("Cannot convert format of image.")

        return self.data

    def bytes_to_image(self):
        """Decode image bytes to a RGB numpy array.

        Returns:
            RGB image array.

        Raises:
            TypeError: If input data is not bytes-like.
            ValueError: If OpenCV fails to decode the image.
        """
        if not isinstance(self.data, (bytes, bytearray)):
            raise TypeError(
                f"bytes2img expects bytes or bytearray, but got {type(self.data)}."
            )

        img_array = np.frombuffer(self.data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image from bytes.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def image_to_bytes(self):
        """Encode a RGB numpy image to image bytes.

        Returns:
            Encoded image bytes.

        Raises:
            TypeError: If input data is not a numpy array.
            ValueError: If OpenCV fails to encode the image.
        """
        if not isinstance(self.data, np.ndarray):
            raise TypeError(
                f"img2bytes expects np.ndarray, but got {type(self.data)}."
            )

        img = self._rgb_to_bgr_if_needed(self.data)

        success, encoded_img = cv2.imencode(f".{self.ext}", img)

        if not success:
            raise ValueError("Failed to encode image to bytes.")

        return encoded_img.tobytes()

    def base64_to_img(self):
        """Decode base64 image data to a RGB numpy array.

        Returns:
            RGB image array.

        Raises:
            TypeError: If input data is not str or bytes-like.
            ValueError: If base64 or image decoding fails.
        """
        if not isinstance(self.data, (str, bytes, bytearray)):
            raise TypeError(
                f"base642img expects str, bytes, or bytearray, but got {type(self.data)}."
            )

        data = self.data

        if isinstance(data, str):
            if "," in data:
                data = data.split(",", 1)[1]
            data = data.encode("utf-8")

        try:
            image_bytes = base64.b64decode(data)
        except Exception as exc:
            raise ValueError("Failed to decode base64 data.") from exc

        image_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image from base64 data.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def img_to_base64(self):
        """Encode a RGB numpy image to a base64 string.

        Returns:
            Base64-encoded image string.

        Raises:
            TypeError: If input data is not a numpy array.
            ValueError: If OpenCV fails to encode the image.
        """
        if not isinstance(self.data, np.ndarray):
            raise TypeError(
                f"img2base64 expects np.ndarray, but got {type(self.data)}."
            )

        img = self._rgb_to_bgr_if_needed(self.data)

        success, encoded_img = cv2.imencode(f".{self.ext}", img)

        if not success:
            raise ValueError("Failed to encode image to base64.")

        base64_bytes = base64.b64encode(encoded_img.tobytes())
        return base64_bytes.decode("utf-8")

    @staticmethod
    def _rgb_to_bgr_if_needed(img: np.ndarray) -> np.ndarray:
        """Convert a RGB image to BGR when it has three channels.

        Args:
            img: Input image array.

        Returns:
            BGR image array for_teach three-channel inputs, otherwise the original
            image array.
        """
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img



