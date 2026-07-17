"""Histogram equalization enhancer."""

from typing import Any, Dict

import cv2
import numpy as np

from ..BaseModel import LLVEnhancer

__all__ = ["HE"]


class HE(LLVEnhancer):
    """Histogram Equalization enhancer.

    Supports per-channel equalization and luminance-channel equalization in
    HSV, HLS, YUV/YCbCr, and LAB color spaces.
    """

    name = "he"

    _COLOR_SPACE_ALIASES = {
        "rgb": "rgb",
        "bgr": "rgb",
        "hsv": "hsv",
        "hls": "hls",
        "yuv": "yuv",
        "ycbcr": "yuv",
        "lab": "lab",
    }

    def __init__(
        self,
        *,
        color_space: str = "rgb",
        **kwargs: Any,
    ) -> None:
        """Initialize HE enhancer.

        Args:
            color_space: Color space where equalization is applied.
            **kwargs: Base enhancer parameters.
        """
        super().__init__(**kwargs)

        self.color_space = self._normalize_color_space(color_space)

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply histogram equalization.

        Args:
            image: Input BGR or grayscale image array.
            **kwargs: Unused method-specific parameters.

        Returns:
            Equalized image array.

        Raises:
            ValueError: If ``color_space`` is unsupported.
        """
        img = self._ensure_uint8(image)

        if img.ndim == 2 or img.shape[2] == 1:
            gray = img if img.ndim == 2 else img[:, :, 0]
            return self._he_gray(gray)

        if self.color_space == "rgb":
            return self._he_rgb(img)

        elif self.color_space == "hsv":
            return self._he_hsv(img)

        elif self.color_space == "hls":
            return self._he_hls(img)

        elif self.color_space == "yuv":
            return self._he_yuv(img)

        elif self.color_space == "lab":
            return self._he_lab(img)

        else:
            raise ValueError(f"Unsupported color space: {self.color_space}")

    def _he_gray(self, img: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to a grayscale image.

        Args:
            img: Grayscale image array.

        Returns:
            Equalized grayscale image.
        """
        return cv2.equalizeHist(img)

    def _he_rgb(self, img: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to each BGR channel.

        Args:
            img: BGR image array.

        Returns:
            Equalized BGR image array.
        """
        channels = cv2.split(img)
        eq_channels = [cv2.equalizeHist(c) for c in channels]
        return cv2.merge(eq_channels)

    def _he_hsv(self, img: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to the HSV value channel.

        Args:
            img: BGR image array.

        Returns:
            Equalized BGR image array.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    def _he_hls(self, img: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to the HLS lightness channel.

        Args:
            img: BGR image array.

        Returns:
            Equalized BGR image array.
        """
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        l = cv2.equalizeHist(l)
        return cv2.cvtColor(cv2.merge([h, l, s]), cv2.COLOR_HLS2BGR)

    def _he_yuv(self, img: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to the YUV luminance channel.

        Args:
            img: BGR image array.

        Returns:
            Equalized BGR image array.
        """
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        y = cv2.equalizeHist(y)
        return cv2.cvtColor(cv2.merge([y, u, v]), cv2.COLOR_YUV2BGR)

    def _he_lab(self, img: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to the LAB lightness channel.

        Args:
            img: BGR image array.

        Returns:
            Equalized BGR image array.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _normalize_color_space(self, cs: str) -> str:
        """Normalize a color-space name.

        Args:
            cs: Color-space name or alias.

        Returns:
            Canonical color-space name.

        Raises:
            TypeError: If ``cs`` is not a string.
            ValueError: If the color space is unsupported.
        """
        if not isinstance(cs, str):
            raise TypeError(f"color_space must be str, got {type(cs)}")

        key = cs.strip().lower()

        if key not in self._COLOR_SPACE_ALIASES:
            raise ValueError(
                f"Unsupported color_space: {cs}. "
                f"Supported: {list(self._COLOR_SPACE_ALIASES.keys())}"
            )

        return self._COLOR_SPACE_ALIASES[key]

    def get_params(self) -> Dict[str, Any]:
        """Return base parameters together with HE configuration."""
        params = super().get_params()
        params["color_space"] = self.color_space
        return params

    @staticmethod
    def _ensure_uint8(image: np.ndarray) -> np.ndarray:
        """Convert image to uint8 because OpenCV HE requires uint8 input.

        Args:
            image: Input image array.

        Returns:
            Uint8 image array.
        """
        if image.dtype == np.uint8:
            return image

        image = image.astype(np.float32)

        if image.max() <= 1.0:
            image = image * 255.0

        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
