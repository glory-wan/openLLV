"""LIME low-light enhancement algorithm."""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..BaseModel import LLVEnhancer

__all__ = ["LIME"]


class LIME(LLVEnhancer):
    """Low-light Image Enhancement via Illumination Map Estimation.

    Original paper: https://doi.org/10.1109/TIP.2016.2639450
    Official source code: None
    """

    name = "LIME"
    aliases = []

    def __init__(
        self,
        *,
        gamma: float = 0.8,
        guided_radius: int = 15,
        guided_eps: float = 1e-3,
        illumination_floor: float = 0.05,
        exposure: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize LIME enhancer.

        Args:
            gamma: Illumination gamma. Smaller values preserve more darkness;
                larger values brighten more aggressively.
            guided_radius: Radius used by guided filtering for illumination
                refinement.
            guided_eps: Guided filter regularization term.
            illumination_floor: Lower bound for refined illumination.
            exposure: Global exposure multiplier applied after enhancement.
            **kwargs: Base enhancer parameters.

        Raises:
            ValueError: If any LIME parameter is invalid.
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        self.illumination_floor = illumination_floor
        self.exposure = exposure
        self._validate_params()

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply LIME enhancement.

        Args:
            image: Input BGR, grayscale, or BGRA image array.
            **kwargs: Optional runtime parameter overrides.

        Returns:
            LIME-enhanced image array.
        """
        gamma = kwargs.get("gamma", self.gamma)
        guided_radius = kwargs.get("guided_radius", self.guided_radius)
        guided_eps = kwargs.get("guided_eps", self.guided_eps)
        illumination_floor = kwargs.get("illumination_floor", self.illumination_floor)
        exposure = kwargs.get("exposure", self.exposure)
        self._validate_values(
            gamma=gamma,
            guided_radius=guided_radius,
            guided_eps=guided_eps,
            illumination_floor=illumination_floor,
            exposure=exposure,
        )

        original_dtype = image.dtype
        image_float = self._to_float01(image)
        work_image, alpha, was_grayscale = self._split_image(image_float)

        initial_illumination = np.max(work_image, axis=2)
        refined_illumination = self._guided_filter(
            guide=initial_illumination,
            source=initial_illumination,
            radius=int(guided_radius),
            eps=float(guided_eps),
        )
        refined_illumination = np.maximum(
            refined_illumination,
            float(illumination_floor),
        )
        illumination = np.power(refined_illumination, float(gamma))
        enhanced = work_image / illumination[:, :, None]
        enhanced = np.clip(enhanced * float(exposure), 0.0, 1.0)
        enhanced = self._merge_image(
            enhanced,
            alpha=alpha,
            was_grayscale=was_grayscale,
        )
        return self._restore_dtype_range(enhanced, original_dtype)

    def get_params(self) -> Dict[str, Any]:
        """Get enhancer parameters.

        Returns:
            Dictionary containing base and LIME parameters.
        """
        params = super().get_params()
        params.update(
            {
                "gamma": self.gamma,
                "guided_radius": self.guided_radius,
                "guided_eps": self.guided_eps,
                "illumination_floor": self.illumination_floor,
                "exposure": self.exposure,
            }
        )
        return params

    def set_params(self, **params: Any) -> "LIME":
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
    def _to_float01(image: np.ndarray) -> np.ndarray:
        """Convert an image to float32 in ``[0, 1]``.

        Args:
            image: Input image array.

        Returns:
            Float image array.
        """
        image_float = image.astype(np.float32)
        if np.issubdtype(image.dtype, np.integer):
            return image_float / np.iinfo(image.dtype).max
        return np.clip(image_float, 0.0, 1.0)

    @staticmethod
    def _restore_dtype_range(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Restore image values to the source dtype range.

        Args:
            image: Enhanced image in ``[0, 1]``.
            dtype: Source image dtype.

        Returns:
            Image in the value range expected by ``dtype``.
        """
        if np.issubdtype(dtype, np.integer):
            return image * np.iinfo(dtype).max
        return image

    @staticmethod
    def _split_image(
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """Split image into working color image and optional alpha channel.

        Args:
            image: Normalized source image.

        Returns:
            Tuple of working image, alpha channel, and grayscale flag.
        """
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
        """Merge enhanced image back to the source channel layout.

        Args:
            image: Enhanced working image.
            alpha: Optional alpha channel.
            was_grayscale: Whether source image was grayscale.

        Returns:
            Enhanced image in source channel layout.
        """
        if was_grayscale:
            return image[:, :, 0]
        if alpha is not None:
            return np.concatenate([image, alpha], axis=2)
        return image

    @staticmethod
    def _guided_filter(
        guide: np.ndarray,
        source: np.ndarray,
        *,
        radius: int,
        eps: float,
    ) -> np.ndarray:
        """Apply grayscale guided filtering.

        Args:
            guide: Guidance image.
            source: Filtering source image.
            radius: Box-filter radius.
            eps: Regularization term.

        Returns:
            Refined source image.
        """
        kernel = (radius, radius)
        mean_guide = cv2.boxFilter(guide, cv2.CV_32F, kernel)
        mean_source = cv2.boxFilter(source, cv2.CV_32F, kernel)
        corr_guide = cv2.boxFilter(guide * guide, cv2.CV_32F, kernel)
        corr_source = cv2.boxFilter(guide * source, cv2.CV_32F, kernel)

        var_guide = corr_guide - mean_guide * mean_guide
        cov_source = corr_source - mean_guide * mean_source
        a = cov_source / (var_guide + eps)
        b = mean_source - a * mean_guide

        mean_a = cv2.boxFilter(a, cv2.CV_32F, kernel)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, kernel)
        return mean_a * guide + mean_b

    def _validate_params(self) -> None:
        """Validate current LIME parameters."""
        self._validate_values(
            gamma=self.gamma,
            guided_radius=self.guided_radius,
            guided_eps=self.guided_eps,
            illumination_floor=self.illumination_floor,
            exposure=self.exposure,
        )

    @staticmethod
    def _validate_values(
        *,
        gamma: float,
        guided_radius: int,
        guided_eps: float,
        illumination_floor: float,
        exposure: float,
    ) -> None:
        """Validate LIME parameter values.

        Args:
            gamma: Illumination gamma.
            guided_radius: Guided filter radius.
            guided_eps: Guided filter epsilon.
            illumination_floor: Minimum illumination.
            exposure: Global exposure multiplier.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if guided_radius <= 0:
            raise ValueError("guided_radius must be > 0.")
        if guided_eps <= 0:
            raise ValueError("guided_eps must be > 0.")
        if not (0 < illumination_floor <= 1):
            raise ValueError("illumination_floor must be in (0, 1].")
        if exposure <= 0:
            raise ValueError("exposure must be > 0.")
