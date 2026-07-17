"""BIMEF low-light enhancement algorithm."""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..BaseModel import LLVEnhancer

__all__ = ["BIMEF"]


class BIMEF(LLVEnhancer):
    """Bio-inspired multi-exposure fusion enhancer.

    Original paper: https://doi.org/10.48550/arXiv.1711.00591
    Official source code: None
    """

    name = "BIMEF"
    aliases = []

    def __init__(
        self,
        *,
        exposure_ratio: Optional[float] = None,
        target_mean: float = 0.55,
        max_ratio: float = 5.0,
        well_exposed_sigma: float = 0.2,
        contrast_weight: float = 1.0,
        saturation_weight: float = 1.0,
        well_exposed_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize BIMEF enhancer.

        Args:
            exposure_ratio: Optional manual exposure ratio. If ``None``, the
                ratio is estimated from image luminance.
            target_mean: Target luminance mean used for automatic exposure.
            max_ratio: Maximum automatic exposure ratio.
            well_exposed_sigma: Sigma for well-exposedness weighting.
            contrast_weight: Exponent for contrast weight.
            saturation_weight: Exponent for saturation weight.
            well_exposed_weight: Exponent for well-exposedness weight.
            **kwargs: Base enhancer parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        super().__init__(**kwargs)
        self.exposure_ratio = exposure_ratio
        self.target_mean = target_mean
        self.max_ratio = max_ratio
        self.well_exposed_sigma = well_exposed_sigma
        self.contrast_weight = contrast_weight
        self.saturation_weight = saturation_weight
        self.well_exposed_weight = well_exposed_weight
        self._validate_params()

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply BIMEF enhancement.

        Args:
            image: Input BGR, grayscale, or BGRA image array.
            **kwargs: Optional runtime parameter overrides.

        Returns:
            BIMEF-enhanced image array.
        """
        exposure_ratio = kwargs.get("exposure_ratio", self.exposure_ratio)
        target_mean = kwargs.get("target_mean", self.target_mean)
        max_ratio = kwargs.get("max_ratio", self.max_ratio)
        well_exposed_sigma = kwargs.get("well_exposed_sigma", self.well_exposed_sigma)
        contrast_weight = kwargs.get("contrast_weight", self.contrast_weight)
        saturation_weight = kwargs.get("saturation_weight", self.saturation_weight)
        well_exposed_weight = kwargs.get(
            "well_exposed_weight",
            self.well_exposed_weight,
        )
        self._validate_values(
            exposure_ratio=exposure_ratio,
            target_mean=target_mean,
            max_ratio=max_ratio,
            well_exposed_sigma=well_exposed_sigma,
            contrast_weight=contrast_weight,
            saturation_weight=saturation_weight,
            well_exposed_weight=well_exposed_weight,
        )

        original_dtype = image.dtype
        image_float = self._to_float01(image)
        work_image, alpha, was_grayscale = self._split_image(image_float)

        ratio = self._estimate_exposure_ratio(
            work_image,
            exposure_ratio=exposure_ratio,
            target_mean=float(target_mean),
            max_ratio=float(max_ratio),
        )
        exposed = np.clip(work_image * ratio, 0.0, 1.0)
        fused = self._fuse_exposures(
            work_image,
            exposed,
            well_exposed_sigma=float(well_exposed_sigma),
            contrast_weight=float(contrast_weight),
            saturation_weight=float(saturation_weight),
            well_exposed_weight=float(well_exposed_weight),
        )
        fused = self._merge_image(fused, alpha=alpha, was_grayscale=was_grayscale)
        return self._restore_dtype_range(fused, original_dtype)

    def get_params(self) -> Dict[str, Any]:
        """Get enhancer parameters.

        Returns:
            Dictionary containing base and BIMEF parameters.
        """
        params = super().get_params()
        params.update(
            {
                "exposure_ratio": self.exposure_ratio,
                "target_mean": self.target_mean,
                "max_ratio": self.max_ratio,
                "well_exposed_sigma": self.well_exposed_sigma,
                "contrast_weight": self.contrast_weight,
                "saturation_weight": self.saturation_weight,
                "well_exposed_weight": self.well_exposed_weight,
            }
        )
        return params

    def set_params(self, **params: Any) -> "BIMEF":
        """Set enhancer parameters.

        Args:
            **params: Parameter names and values.

        Returns:
            The enhancer itself.
        """
        super().set_params(**params)
        self._validate_params()
        return self

    @classmethod
    def _fuse_exposures(
        cls,
        image: np.ndarray,
        exposed: np.ndarray,
        *,
        well_exposed_sigma: float,
        contrast_weight: float,
        saturation_weight: float,
        well_exposed_weight: float,
    ) -> np.ndarray:
        """Fuse original and exposed images with exposure-fusion weights.

        Args:
            image: Original normalized image.
            exposed: Exposure-adjusted normalized image.
            well_exposed_sigma: Sigma for well-exposedness.
            contrast_weight: Contrast weight exponent.
            saturation_weight: Saturation weight exponent.
            well_exposed_weight: Well-exposedness weight exponent.

        Returns:
            Fused image.
        """
        weight_original = cls._fusion_weight(
            image,
            sigma=well_exposed_sigma,
            contrast_weight=contrast_weight,
            saturation_weight=saturation_weight,
            well_exposed_weight=well_exposed_weight,
        )
        weight_exposed = cls._fusion_weight(
            exposed,
            sigma=well_exposed_sigma,
            contrast_weight=contrast_weight,
            saturation_weight=saturation_weight,
            well_exposed_weight=well_exposed_weight,
        )
        total = weight_original + weight_exposed + 1e-6
        weight_original = weight_original / total
        weight_exposed = weight_exposed / total
        return np.clip(
            image * weight_original[:, :, None] + exposed * weight_exposed[:, :, None],
            0.0,
            1.0,
        )

    @classmethod
    def _fusion_weight(
        cls,
        image: np.ndarray,
        *,
        sigma: float,
        contrast_weight: float,
        saturation_weight: float,
        well_exposed_weight: float,
    ) -> np.ndarray:
        """Compute exposure fusion weight map.

        Args:
            image: Normalized image.
            sigma: Sigma for well-exposedness.
            contrast_weight: Contrast exponent.
            saturation_weight: Saturation exponent.
            well_exposed_weight: Well-exposedness exponent.

        Returns:
            Weight map.
        """
        luminance = cls._luminance(image)
        contrast = np.abs(cv2.Laplacian(luminance.astype(np.float32), cv2.CV_32F))
        saturation = (
            np.std(image, axis=2)
            if image.shape[2] > 1
            else np.ones_like(luminance)
        )
        well_exposed = np.exp(-((luminance - 0.5) ** 2) / (2 * sigma * sigma))

        weight = np.ones_like(luminance, dtype=np.float32)
        weight *= np.power(contrast + 1e-6, contrast_weight)
        weight *= np.power(saturation + 1e-6, saturation_weight)
        weight *= np.power(well_exposed + 1e-6, well_exposed_weight)
        return weight

    @classmethod
    def _estimate_exposure_ratio(
        cls,
        image: np.ndarray,
        *,
        exposure_ratio: Optional[float],
        target_mean: float,
        max_ratio: float,
    ) -> float:
        """Estimate automatic exposure ratio.

        Args:
            image: Normalized image.
            exposure_ratio: Optional manual exposure ratio.
            target_mean: Target luminance mean.
            max_ratio: Maximum ratio.

        Returns:
            Exposure ratio.
        """
        if exposure_ratio is not None:
            return float(exposure_ratio)

        luminance = cls._luminance(image)
        mean_luminance = float(np.mean(luminance))
        return float(np.clip(target_mean / max(mean_luminance, 1e-6), 1.0, max_ratio))

    @staticmethod
    def _luminance(image: np.ndarray) -> np.ndarray:
        """Compute luminance from BGR or grayscale working image.

        Args:
            image: Normalized working image.

        Returns:
            Luminance map.
        """
        if image.shape[2] == 1:
            return image[:, :, 0]
        return 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]

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
        """Validate current BIMEF parameters."""
        self._validate_values(
            exposure_ratio=self.exposure_ratio,
            target_mean=self.target_mean,
            max_ratio=self.max_ratio,
            well_exposed_sigma=self.well_exposed_sigma,
            contrast_weight=self.contrast_weight,
            saturation_weight=self.saturation_weight,
            well_exposed_weight=self.well_exposed_weight,
        )

    @staticmethod
    def _validate_values(
        *,
        exposure_ratio: Optional[float],
        target_mean: float,
        max_ratio: float,
        well_exposed_sigma: float,
        contrast_weight: float,
        saturation_weight: float,
        well_exposed_weight: float,
    ) -> None:
        """Validate BIMEF parameter values."""
        if exposure_ratio is not None and exposure_ratio <= 0:
            raise ValueError("exposure_ratio must be > 0.")
        if not (0 < target_mean < 1):
            raise ValueError("target_mean must be in (0, 1).")
        if max_ratio < 1:
            raise ValueError("max_ratio must be >= 1.")
        if well_exposed_sigma <= 0:
            raise ValueError("well_exposed_sigma must be > 0.")
        if min(contrast_weight, saturation_weight, well_exposed_weight) < 0:
            raise ValueError("Fusion weight exponents must be >= 0.")
