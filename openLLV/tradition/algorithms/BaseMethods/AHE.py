"""Adaptive histogram equalization enhancer."""

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from ..BaseModel import LLVEnhancer

__all__ = ["AHE"]


class AHE(LLVEnhancer):
    """Adaptive Histogram Equalization enhancer.

    This implementation approximates AHE by using CLAHE with a very large
    clip limit.
    """

    name = "ahe"

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
        color_space: str = "yuv",
        tile_grid_size: Tuple[int, int] = (8, 8),
        **kwargs: Any,
    ) -> None:
        """Initialize AHE enhancer.

        Args:
            color_space: Color space where equalization is applied.
            tile_grid_size: Tile grid size used by CLAHE.
            **kwargs: Base enhancer parameters.
        """
        super().__init__(**kwargs)

        self.color_space = self._normalize_color_space(color_space)
        self.tile_grid_size = self._validate_tile_grid_size(tile_grid_size)

        self._clahe = cv2.createCLAHE(
            clipLimit=255.0,
            tileGridSize=self.tile_grid_size,
        )

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply adaptive histogram equalization.

        Args:
            image: Input BGR or grayscale image array.
            **kwargs: Unused method-specific parameters.

        Returns:
            Enhanced image array.

        Raises:
            ValueError: If ``color_space`` is unsupported.
        """
        img = self._ensure_uint8(image)

        if img.ndim == 2 or img.shape[2] == 1:
            gray = img if img.ndim == 2 else img[:, :, 0]
            return self._clahe.apply(gray)

        if self.color_space == "rgb":
            return self._process_rgb(img)
        elif self.color_space == "hsv":
            return self._process_hsv(img)
        elif self.color_space == "hls":
            return self._process_hls(img)
        elif self.color_space == "yuv":
            return self._process_yuv(img)
        elif self.color_space == "lab":
            return self._process_lab(img)

        raise ValueError(f"Unsupported color space: {self.color_space}")

    def _process_rgb(self, img):
        """Apply equalization to each BGR channel.

        Args:
            img: BGR image array.

        Returns:
            Equalized BGR image array.
        """
        ch = cv2.split(img)
        ch = [self._clahe.apply(c) for c in ch]
        return cv2.merge(ch)

    def _process_hsv(self, img):
        """Apply equalization to the HSV value channel.

        Args:
            img: BGR image array.

        Returns:
            Equalized BGR image array.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = self._clahe.apply(v)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    def _process_hls(self, img):
        """Apply equalization to the HLS lightness channel.

        Args:
            img: BGR image array.

        Returns:
            Equalized BGR image array.
        """
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([h, l, s]), cv2.COLOR_HLS2BGR)

    def _process_yuv(self, img):
        """Apply equalization to the YUV luminance channel.

        Args:
            img: BGR image array.

        Returns:
            Equalized BGR image array.
        """
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        y = self._clahe.apply(y)
        return cv2.cvtColor(cv2.merge([y, u, v]), cv2.COLOR_YUV2BGR)

    def _process_lab(self, img):
        """Apply equalization to the LAB lightness channel.

        Args:
            img: BGR image array.

        Returns:
            Equalized BGR image array.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
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
            raise TypeError(f"color_space must be str, got {type(cs)!r}")

        key = cs.strip().lower()
        if key not in self._COLOR_SPACE_ALIASES:
            raise ValueError(
                f"Unsupported color_space: {cs!r}. "
                f"Supported: {list(self._COLOR_SPACE_ALIASES)}"
            )
        return self._COLOR_SPACE_ALIASES[key]

    @staticmethod
    def _validate_tile_grid_size(
        tile_grid_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Validate and normalize the CLAHE tile-grid size."""
        if (
            not isinstance(tile_grid_size, (tuple, list))
            or len(tile_grid_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value <= 0
                for value in tile_grid_size
            )
        ):
            raise ValueError("tile_grid_size must contain two positive integers")
        return int(tile_grid_size[0]), int(tile_grid_size[1])

    def get_params(self) -> Dict[str, Any]:
        """Return base parameters together with AHE configuration."""
        params = super().get_params()
        params.update(
            color_space=self.color_space,
            tile_grid_size=self.tile_grid_size,
        )
        return params

    @staticmethod
    def _ensure_uint8(img):
        """Convert image to uint8.

        Args:
            img: Input image array.

        Returns:
            Uint8 image array.
        """
        if img.dtype == np.uint8:
            return img
        img = img.astype(np.float32)
        if img.max() <= 1:
            img *= 255
        return np.clip(img, 0, 255).astype(np.uint8)
