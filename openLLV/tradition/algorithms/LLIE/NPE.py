"""NPE low-light enhancement algorithm."""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..BaseModel import LLVEnhancer

__all__ = ["NPE"]


class NPE(LLVEnhancer):
    """Naturalness Preserved Enhancement for non-uniform illumination images.

    Original paper: https://doi.org/10.1109/TIP.2013.2261309
    Official source code: None
    """

    name = "NPE"
    aliases = []

    def __init__(
        self,
        *,
        sigma: float = 15.0,
        illumination_floor: float = 0.05,
        enhancement_strength: float = 4.0,
        naturalness: float = 0.35,
        detail_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize NPE enhancer.

        Args:
            sigma: Gaussian scale used by the bright-pass illumination filter.
            illumination_floor: Lower bound for illumination estimation.
            enhancement_strength: Bi-log illumination mapping strength.
            naturalness: Blend weight for preserving original naturalness.
            detail_weight: Weight applied to reflectance detail restoration.
            **kwargs: Base enhancer parameters.

        Raises:
            ValueError: If any NPE parameter is invalid.
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.illumination_floor = illumination_floor
        self.enhancement_strength = enhancement_strength
        self.naturalness = naturalness
        self.detail_weight = detail_weight
        self._validate_params()

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply NPE enhancement.

        Args:
            image: Input BGR, grayscale, or BGRA image array.
            **kwargs: Optional runtime parameter overrides.

        Returns:
            NPE-enhanced image array.
        """
        sigma = kwargs.get("sigma", self.sigma)
        illumination_floor = kwargs.get("illumination_floor", self.illumination_floor)
        enhancement_strength = kwargs.get(
            "enhancement_strength",
            self.enhancement_strength,
        )
        naturalness = kwargs.get("naturalness", self.naturalness)
        detail_weight = kwargs.get("detail_weight", self.detail_weight)
        self._validate_values(
            sigma=sigma,
            illumination_floor=illumination_floor,
            enhancement_strength=enhancement_strength,
            naturalness=naturalness,
            detail_weight=detail_weight,
        )

        original_dtype = image.dtype
        image_float = self._to_float01(image)
        work_image, alpha, was_grayscale = self._split_image(image_float)

        illumination = self._bright_pass_illumination(
            work_image,
            sigma=float(sigma),
            floor=float(illumination_floor),
        )
        reflectance = np.clip(work_image / illumination[:, :, None], 0.0, 1.0)
        mapped_illumination = self._bi_log_transform(
            illumination,
            strength=float(enhancement_strength),
        )
        enhanced = np.clip(reflectance * mapped_illumination[:, :, None], 0.0, 1.0)
        detail_enhanced = np.clip(
            work_image + float(detail_weight) * (enhanced - work_image),
            0.0,
            1.0,
        )
        output = np.clip(
            float(naturalness) * work_image
            + (1.0 - float(naturalness)) * detail_enhanced,
            0.0,
            1.0,
        )
        output = self._merge_image(output, alpha=alpha, was_grayscale=was_grayscale)
        return self._restore_dtype_range(output, original_dtype)

    def get_params(self) -> Dict[str, Any]:
        """Get enhancer parameters.

        Returns:
            Dictionary containing base and NPE parameters.
        """
        params = super().get_params()
        params.update(
            {
                "sigma": self.sigma,
                "illumination_floor": self.illumination_floor,
                "enhancement_strength": self.enhancement_strength,
                "naturalness": self.naturalness,
                "detail_weight": self.detail_weight,
            }
        )
        return params

    def set_params(self, **params: Any) -> "NPE":
        """Set enhancer parameters.

        Args:
            **params: Parameter names and values.

        Returns:
            The enhancer itself.
        """
        super().set_params(**params)
        self._validate_params()
        return self

    @staticmethod
    def _bright_pass_illumination(
        image: np.ndarray,
        *,
        sigma: float,
        floor: float,
    ) -> np.ndarray:
        """Estimate illumination with a bright-pass filter.

        Args:
            image: Normalized working image.
            sigma: Gaussian smoothing scale.
            floor: Minimum illumination value.

        Returns:
            Smoothed illumination map.
        """
        bright_pass = np.max(image, axis=2)
        illumination = cv2.GaussianBlur(
            bright_pass,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        return np.maximum(illumination, floor)

    @staticmethod
    def _bi_log_transform(illumination: np.ndarray, *, strength: float) -> np.ndarray:
        """Apply bi-log style illumination mapping.

        Args:
            illumination: Illumination map in ``[0, 1]``.
            strength: Mapping strength.

        Returns:
            Mapped illumination image.
        """
        brightened = np.log1p(strength * illumination) / np.log1p(strength)
        preserved = 1.0 - np.log1p(strength * (1.0 - illumination)) / np.log1p(strength)
        weight = 1.0 - illumination
        return np.clip(weight * brightened + (1.0 - weight) * preserved, 0.0, 1.0)

    @staticmethod
    def _to_float01(image: np.ndarray) -> np.ndarray:
        """Convert an image to float32 in ``[0, 1]``."""
        image_float = image.astype(np.float32)
        if np.issubdtype(image.dtype, np.integer):
            return image_float / np.iinfo(image.dtype).max
        return np.clip(image_float, 0.0, 1.0)

    @staticmethod
    def _restore_dtype_range(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Restore image values to the source dtype range."""
        if np.issubdtype(dtype, np.integer):
            return image * np.iinfo(dtype).max
        return image

    @staticmethod
    def _split_image(
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """Split image into working image and optional alpha channel."""
        if image.ndim == 2:
            return image[:, :, None], None, True
        if image.shape[2] == 1:
            return image, None, True
        if image.shape[2] == 4:
            return image[:, :, :3], image[:, :, 3:4], False
        return image, None, False

    @staticmethod
    def _merge_image(
        image: np.ndarray,
        *,
        alpha: Optional[np.ndarray],
        was_grayscale: bool,
    ) -> np.ndarray:
        """Merge enhanced image back to source channel layout."""
        if was_grayscale:
            return image[:, :, 0]
        if alpha is not None:
            return np.concatenate([image, alpha], axis=2)
        return image

    def _validate_params(self) -> None:
        """Validate current NPE parameters."""
        self._validate_values(
            sigma=self.sigma,
            illumination_floor=self.illumination_floor,
            enhancement_strength=self.enhancement_strength,
            naturalness=self.naturalness,
            detail_weight=self.detail_weight,
        )

    @staticmethod
    def _validate_values(
        *,
        sigma: float,
        illumination_floor: float,
        enhancement_strength: float,
        naturalness: float,
        detail_weight: float,
    ) -> None:
        """Validate NPE parameter values."""
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
        if not (0 < illumination_floor <= 1):
            raise ValueError("illumination_floor must be in (0, 1].")
        if enhancement_strength <= 0:
            raise ValueError("enhancement_strength must be > 0.")
        if not (0 <= naturalness <= 1):
            raise ValueError("naturalness must be in [0, 1].")
        if detail_weight < 0:
            raise ValueError("detail_weight must be >= 0.")
