"""Contrast-limited adaptive histogram equalization enhancer."""

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from ..BaseModel import LLVEnhancer

__all__ = ["CLAHE"]


class CLAHE(LLVEnhancer):
    """Contrast Limited Adaptive Histogram Equalization enhancer.

    Original paper: https://ieeexplore.ieee.org/document/109340

    """

    name = "clahe"

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
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        **kwargs: Any,
    ) -> None:
        """Initialize CLAHE enhancer.

        Args:
            color_space: Color space where equalization is applied.
            clip_limit: CLAHE contrast clipping limit.
            tile_grid_size: CLAHE tile grid size.
            **kwargs: Base enhancer parameters.
        """
        super().__init__(**kwargs)

        self.color_space = self._normalize_color_space(color_space)
        self.clip_limit = self._validate_clip_limit(clip_limit)
        self.tile_grid_size = self._validate_tile_grid_size(tile_grid_size)

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply CLAHE enhancement.

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
            return self._rgb(img)
        elif self.color_space == "hsv":
            return self._hsv(img)
        elif self.color_space == "hls":
            return self._hls(img)
        elif self.color_space == "yuv":
            return self._yuv(img)
        elif self.color_space == "lab":
            return self._lab(img)

        raise ValueError(f"Unsupported color space: {self.color_space}")

    def _rgb(self, img):
        """Apply CLAHE to each BGR channel.

        Args:
            img: BGR image array.

        Returns:
            Enhanced BGR image array.
        """
        return cv2.merge([self._clahe.apply(c) for c in cv2.split(img)])

    def _hsv(self, img):
        """Apply CLAHE to the HSV value channel.

        Args:
            img: BGR image array.

        Returns:
            Enhanced BGR image array.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = self._clahe.apply(v)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    def _hls(self, img):
        """Apply CLAHE to the HLS lightness channel.

        Args:
            img: BGR image array.

        Returns:
            Enhanced BGR image array.
        """
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([h, l, s]), cv2.COLOR_HLS2BGR)

    def _yuv(self, img):
        """Apply CLAHE to the YUV luminance channel.

        Args:
            img: BGR image array.

        Returns:
            Enhanced BGR image array.
        """
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        y = self._clahe.apply(y)
        return cv2.cvtColor(cv2.merge([y, u, v]), cv2.COLOR_YUV2BGR)

    def _lab(self, img):
        """Apply CLAHE to the LAB lightness channel.

        Args:
            img: BGR image array.

        Returns:
            Enhanced BGR image array.
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
    def _validate_clip_limit(clip_limit: float) -> float:
        """Validate and normalize the CLAHE clipping limit."""
        if (
            isinstance(clip_limit, bool)
            or not isinstance(clip_limit, (int, float, np.number))
            or not np.isfinite(clip_limit)
            or clip_limit <= 0
        ):
            raise ValueError("clip_limit must be a finite number greater than 0")
        return float(clip_limit)

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
        """Return base parameters together with CLAHE configuration."""
        params = super().get_params()
        params.update(
            color_space=self.color_space,
            clip_limit=self.clip_limit,
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
