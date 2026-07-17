"""Recursive CLAHE enhancer."""

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from ..BaseModel import LLVEnhancer

__all__ = ["RCLAHE"]


class RCLAHE(LLVEnhancer):
    """Recursive CLAHE enhancer.

    The method applies CLAHE multiple times to progressively enhance local
    contrast.
    """

    name = "rclahe"

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
        iterations: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize recursive CLAHE enhancer.

        Args:
            color_space: Color space where CLAHE is applied.
            clip_limit: CLAHE contrast clipping limit.
            tile_grid_size: CLAHE tile grid size.
            iterations: Number of recursive CLAHE applications.
            **kwargs: Base enhancer parameters.
        """
        super().__init__(**kwargs)

        self.color_space = self._normalize_color_space(color_space)
        self.clip_limit = self._validate_clip_limit(clip_limit)
        self.tile_grid_size = self._validate_tile_grid_size(tile_grid_size)
        self.iterations = self._validate_iterations(iterations)

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply recursive CLAHE enhancement.

        Args:
            image: Input BGR or grayscale image array.
            **kwargs: Unused method-specific parameters.

        Returns:
            Enhanced image array.
        """
        img = self._ensure_uint8(image)

        for _ in range(self.iterations):
            img = self._apply_once(img)

        return img

    def _apply_once(self, img):
        """Apply one CLAHE iteration.

        Args:
            img: BGR or grayscale image array.

        Returns:
            Enhanced image array.

        Raises:
            ValueError: If ``color_space`` is unsupported.
        """
        if img.ndim == 2 or img.shape[2] == 1:
            gray = img if img.ndim == 2 else img[:, :, 0]
            return self._clahe.apply(gray)

        if self.color_space == "rgb":
            return cv2.merge([self._clahe.apply(c) for c in cv2.split(img)])

        elif self.color_space == "hsv":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = self._clahe.apply(v)
            return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

        elif self.color_space == "hls":
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            h, l, s = cv2.split(hls)
            l = self._clahe.apply(l)
            return cv2.cvtColor(cv2.merge([h, l, s]), cv2.COLOR_HLS2BGR)

        elif self.color_space == "yuv":
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            y = self._clahe.apply(y)
            return cv2.cvtColor(cv2.merge([y, u, v]), cv2.COLOR_YUV2BGR)

        elif self.color_space == "lab":
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        raise ValueError(f"Unsupported color space: {self.color_space}")

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

    @staticmethod
    def _validate_iterations(iterations: int) -> int:
        """Validate the number of recursive enhancement passes."""
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, (int, np.integer))
            or iterations <= 0
        ):
            raise ValueError("iterations must be a positive integer")
        return int(iterations)

    def get_params(self) -> Dict[str, Any]:
        """Return base parameters together with recursive CLAHE configuration."""
        params = super().get_params()
        params.update(
            color_space=self.color_space,
            clip_limit=self.clip_limit,
            tile_grid_size=self.tile_grid_size,
            iterations=self.iterations,
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
