"""Retinex-based traditional low-light enhancement algorithms."""

from abc import ABC
from typing import Any, Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np

from ..BaseModel import LLVEnhancer

__all__ = ["SSR", "MSR", "MSRCR"]


class _RetinexBase(LLVEnhancer, ABC):
    """Base implementation shared by Retinex algorithms."""

    aliases = []

    def __init__(
        self,
        *,
        low_clip: float = 1.0,
        high_clip: float = 99.0,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """Initialize common Retinex parameters.

        Args:
            low_clip: Lower percentile for display-range normalization.
            high_clip: Upper percentile for display-range normalization.
            eps: Small value used to avoid logarithm and division instability.
            **kwargs: Base enhancer parameters.

        Raises:
            ValueError: If any common Retinex parameter is invalid.
        """
        super().__init__(**kwargs)
        self.low_clip = low_clip
        self.high_clip = high_clip
        self.eps = eps
        self._validate_common_params()

    def _to_float01(self, image: np.ndarray) -> np.ndarray:
        """Convert an input image to float32 in ``[0, 1]``.

        Args:
            image: Input image array.

        Returns:
            Float image array.
        """
        image_float = image.astype(np.float32)

        if np.issubdtype(image.dtype, np.integer):
            return image_float / np.iinfo(image.dtype).max

        return np.clip(image_float, 0.0, 1.0)

    def _restore_dtype_range(
        self,
        image: np.ndarray,
        original_dtype: np.dtype,
    ) -> np.ndarray:
        """Restore enhanced image to the source dtype value range.

        Args:
            image: Enhanced image in ``[0, 1]``.
            original_dtype: Source image dtype.

        Returns:
            Image in the value range expected by the source dtype.
        """
        if np.issubdtype(original_dtype, np.integer):
            return image * np.iinfo(original_dtype).max

        return image

    def _split_image(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Split source image into Retinex input and optional alpha channel.

        Args:
            image: Normalized source image.

        Returns:
            Tuple of working image, optional alpha channel, and whether the
            original image was two-dimensional.
        """
        if image.ndim == 2:
            return image[:, :, None], None, True

        if image.shape[2] == 4:
            return image[:, :, :3], image[:, :, 3:4], False

        return image, None, False

    def _merge_image(
        self,
        image: np.ndarray,
        *,
        alpha: np.ndarray,
        was_grayscale: bool,
    ) -> np.ndarray:
        """Merge enhanced Retinex output back to the source channel layout.

        Args:
            image: Enhanced working image in ``[0, 1]``.
            alpha: Optional alpha channel.
            was_grayscale: Whether the source image was two-dimensional.

        Returns:
            Enhanced image with the original channel layout.
        """
        if was_grayscale:
            return image[:, :, 0]

        if alpha is not None:
            return np.concatenate([image, alpha], axis=2)

        return image

    def _single_scale_retinex(
        self,
        image: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Compute single-scale Retinex response.

        Args:
            image: Normalized image in ``[0, 1]``.
            sigma: Gaussian surround scale.

        Returns:
            Retinex response array.
        """
        surround = cv2.GaussianBlur(
            image,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        if surround.ndim == 2 and image.ndim == 3:
            surround = surround[:, :, None]
        return np.log(image + self.eps) - np.log(surround + self.eps)

    def _multi_scale_retinex(
        self,
        image: np.ndarray,
        scales: Sequence[float],
    ) -> np.ndarray:
        """Compute multi-scale Retinex response.

        Args:
            image: Normalized image in ``[0, 1]``.
            scales: Gaussian surround scales.

        Returns:
            Average Retinex response across scales.
        """
        retinex = np.zeros_like(image, dtype=np.float32)
        for sigma in scales:
            retinex += self._single_scale_retinex(image, sigma)
        return retinex / len(scales)

    def _normalize_for_display(
        self,
        image: np.ndarray,
        *,
        fallback: np.ndarray,
    ) -> np.ndarray:
        """Normalize Retinex response to displayable ``[0, 1]`` range.

        Args:
            image: Retinex response.
            fallback: Source image used when a channel has degenerate range.

        Returns:
            Display-normalized image.
        """
        normalized = np.empty_like(image, dtype=np.float32)

        for channel in range(image.shape[2]):
            channel_data = image[:, :, channel]
            low = np.percentile(channel_data, self.low_clip)
            high = np.percentile(channel_data, self.high_clip)

            if high - low <= self.eps:
                normalized[:, :, channel] = fallback[:, :, channel]
                continue

            clipped = np.clip(channel_data, low, high)
            normalized[:, :, channel] = (clipped - low) / (high - low)

        return np.clip(normalized, 0.0, 1.0)

    def get_params(self) -> Dict[str, Any]:
        """Get enhancer parameters.

        Returns:
            Dictionary containing base parameters and common Retinex parameters.
        """
        params = super().get_params()
        params.update(
            {
                "low_clip": self.low_clip,
                "high_clip": self.high_clip,
                "eps": self.eps,
            }
        )
        return params

    def set_params(self, **params: Any) -> "_RetinexBase":
        """Set enhancer parameters.

        Args:
            **params: Parameter names and values.

        Returns:
            The enhancer itself.
        """
        super().set_params(**params)
        self._validate_common_params()
        return self

    def _validate_common_params(self) -> None:
        """Validate common Retinex parameters.

        Raises:
            ValueError: If any common parameter is invalid.
        """
        if not (0 <= self.low_clip < self.high_clip <= 100):
            raise ValueError(
                "Percentiles must satisfy "
                "0 <= low_clip < high_clip <= 100."
            )

        if self.eps <= 0:
            raise ValueError("eps must be > 0.")

    @staticmethod
    def _validate_sigma(sigma: float) -> None:
        """Validate a Gaussian surround scale.

        Args:
            sigma: Gaussian sigma.

        Raises:
            TypeError: If ``sigma`` is not numeric.
            ValueError: If ``sigma`` is not positive.
        """
        if not isinstance(sigma, (int, float)):
            raise TypeError(f"sigma must be int or float, got {type(sigma)!r}.")

        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma!r}.")

    @classmethod
    def _validate_scales(cls, scales: Iterable[float]) -> Tuple[float, ...]:
        """Validate Gaussian surround scales.

        Args:
            scales: Iterable of Gaussian sigma values.

        Returns:
            Tuple of validated scales.

        Raises:
            TypeError: If ``scales`` is not iterable.
            ValueError: If ``scales`` is empty or contains invalid values.
        """
        if isinstance(scales, (str, bytes)):
            raise TypeError("scales must be an iterable of positive numbers.")

        try:
            validated = tuple(float(scale) for scale in scales)
        except TypeError as exc:
            raise TypeError(
                "scales must be an iterable of positive numbers."
            ) from exc

        if not validated:
            raise ValueError("scales must not be empty.")

        for scale in validated:
            cls._validate_sigma(scale)

        return validated


class SSR(_RetinexBase):
    """Single Scale Retinex enhancer.

    Original paper: https://doi.org/10.1109/83.557356
    """

    name = "SSR"
    aliases = []

    def __init__(
        self,
        *,
        sigma: float = 80.0,
        **kwargs: Any,
    ) -> None:
        """Initialize SSR enhancer.

        Args:
            sigma: Gaussian surround scale.
            **kwargs: Base and common Retinex parameters.

        Raises:
            ValueError: If ``sigma`` is not positive.
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self._validate_sigma(self.sigma)

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Single Scale Retinex.

        Args:
            image: Input image array.
            **kwargs: Optional ``sigma`` override.

        Returns:
            SSR-enhanced image array.
        """
        sigma = kwargs.get("sigma", self.sigma)
        self._validate_sigma(sigma)

        original_dtype = image.dtype
        image_float = self._to_float01(image)
        work_image, alpha, was_grayscale = self._split_image(image_float)

        retinex = self._single_scale_retinex(work_image, float(sigma))
        enhanced = self._normalize_for_display(retinex, fallback=work_image)
        enhanced = self._merge_image(
            enhanced,
            alpha=alpha,
            was_grayscale=was_grayscale,
        )

        return self._restore_dtype_range(enhanced, original_dtype)

    def get_params(self) -> Dict[str, Any]:
        """Get SSR parameters.

        Returns:
            Dictionary containing base, common Retinex, and SSR parameters.
        """
        params = super().get_params()
        params.update({"sigma": self.sigma})
        return params

    def set_params(self, **params: Any) -> "SSR":
        """Set SSR parameters.

        Args:
            **params: Parameter names and values.

        Returns:
            The enhancer itself.
        """
        super().set_params(**params)
        self._validate_sigma(self.sigma)
        return self


class MSR(_RetinexBase):
    """Multi Scale Retinex enhancer.

    Original paper: https://doi.org/10.1109/83.597272
    """

    name = "MSR"
    aliases = []

    def __init__(
        self,
        *,
        scales: Sequence[float] = (15.0, 80.0, 250.0),
        **kwargs: Any,
    ) -> None:
        """Initialize MSR enhancer.

        Args:
            scales: Gaussian surround scales.
            **kwargs: Base and common Retinex parameters.

        Raises:
            ValueError: If ``scales`` is empty or contains non-positive values.
        """
        super().__init__(**kwargs)
        self.scales = self._validate_scales(scales)

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Multi Scale Retinex.

        Args:
            image: Input image array.
            **kwargs: Optional ``scales`` override.

        Returns:
            MSR-enhanced image array.
        """
        scales = self._validate_scales(kwargs.get("scales", self.scales))

        original_dtype = image.dtype
        image_float = self._to_float01(image)
        work_image, alpha, was_grayscale = self._split_image(image_float)

        retinex = self._multi_scale_retinex(work_image, scales)
        enhanced = self._normalize_for_display(retinex, fallback=work_image)
        enhanced = self._merge_image(
            enhanced,
            alpha=alpha,
            was_grayscale=was_grayscale,
        )

        return self._restore_dtype_range(enhanced, original_dtype)

    def get_params(self) -> Dict[str, Any]:
        """Get MSR parameters.

        Returns:
            Dictionary containing base, common Retinex, and MSR parameters.
        """
        params = super().get_params()
        params.update({"scales": self.scales})
        return params

    def set_params(self, **params: Any) -> "MSR":
        """Set MSR parameters.

        Args:
            **params: Parameter names and values.

        Returns:
            The enhancer itself.
        """
        super().set_params(**params)
        self.scales = self._validate_scales(self.scales)
        return self


class MSRCR(MSR):
    """Multi Scale Retinex with Color Restoration enhancer.

    Original paper: https://doi.org/10.1109/83.597272
    """

    name = "MSRCR"
    aliases = []

    def __init__(
        self,
        *,
        scales: Sequence[float] = (15.0, 80.0, 250.0),
        alpha: float = 125.0,
        beta: float = 46.0,
        gain: float = 1.0,
        offset: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize MSRCR enhancer.

        Args:
            scales: Gaussian surround scales.
            alpha: Color-restoration intensity gain.
            beta: Color-restoration log gain.
            gain: Global gain applied to the restored Retinex response.
            offset: Global offset applied before display normalization.
            **kwargs: Base and common Retinex parameters.

        Raises:
            ValueError: If color restoration parameters are invalid.
        """
        super().__init__(scales=scales, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gain = gain
        self.offset = offset
        self._validate_msrcr_params()

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Multi Scale Retinex with Color Restoration.

        Args:
            image: Input image array.
            **kwargs: Optional parameter overrides.

        Returns:
            MSRCR-enhanced image array.
        """
        scales = self._validate_scales(kwargs.get("scales", self.scales))
        alpha = kwargs.get("alpha", self.alpha)
        beta = kwargs.get("beta", self.beta)
        gain = kwargs.get("gain", self.gain)
        offset = kwargs.get("offset", self.offset)
        self._validate_msrcr_values(alpha, beta, gain, offset)

        original_dtype = image.dtype
        image_float = self._to_float01(image)
        work_image, alpha_channel, was_grayscale = self._split_image(image_float)

        retinex = self._multi_scale_retinex(work_image, scales)
        restored = self._color_restore(
            work_image,
            retinex,
            alpha=float(alpha),
            beta=float(beta),
            gain=float(gain),
            offset=float(offset),
        )
        enhanced = self._normalize_for_display(restored, fallback=work_image)
        enhanced = self._merge_image(
            enhanced,
            alpha=alpha_channel,
            was_grayscale=was_grayscale,
        )

        return self._restore_dtype_range(enhanced, original_dtype)

    def _color_restore(
        self,
        image: np.ndarray,
        retinex: np.ndarray,
        *,
        alpha: float,
        beta: float,
        gain: float,
        offset: float,
    ) -> np.ndarray:
        """Apply MSRCR color restoration.

        Args:
            image: Normalized image in ``[0, 1]``.
            retinex: Multi-scale Retinex response.
            alpha: Color-restoration intensity gain.
            beta: Color-restoration log gain.
            gain: Global gain.
            offset: Global offset.

        Returns:
            Color-restored Retinex response.
        """
        if image.shape[2] == 1:
            return gain * (retinex + offset)

        intensity = np.sum(image, axis=2, keepdims=True)
        color_restoration = beta * (
            np.log(alpha * image + self.eps) - np.log(intensity + self.eps)
        )
        return gain * (color_restoration * retinex + offset)

    def get_params(self) -> Dict[str, Any]:
        """Get MSRCR parameters.

        Returns:
            Dictionary containing base, common Retinex, MSR, and MSRCR
            parameters.
        """
        params = super().get_params()
        params.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "gain": self.gain,
                "offset": self.offset,
            }
        )
        return params

    def set_params(self, **params: Any) -> "MSRCR":
        """Set MSRCR parameters.

        Args:
            **params: Parameter names and values.

        Returns:
            The enhancer itself.
        """
        super().set_params(**params)
        self._validate_msrcr_params()
        return self

    def _validate_msrcr_params(self) -> None:
        """Validate instance MSRCR parameters.

        Raises:
            TypeError: If a parameter has invalid type.
            ValueError: If a positive parameter is non-positive.
        """
        self._validate_msrcr_values(
            self.alpha,
            self.beta,
            self.gain,
            self.offset,
        )

    @staticmethod
    def _validate_msrcr_values(
        alpha: float,
        beta: float,
        gain: float,
        offset: float,
    ) -> None:
        """Validate MSRCR color-restoration values.

        Args:
            alpha: Color-restoration intensity gain.
            beta: Color-restoration log gain.
            gain: Global gain.
            offset: Global offset.

        Raises:
            TypeError: If any parameter is not numeric.
            ValueError: If ``alpha``, ``beta``, or ``gain`` is non-positive.
        """
        for name, value in {
            "alpha": alpha,
            "beta": beta,
            "gain": gain,
            "offset": offset,
        }.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be int or float, got {type(value)!r}.")

        if alpha <= 0:
            raise ValueError("alpha must be > 0.")

        if beta <= 0:
            raise ValueError("beta must be > 0.")

        if gain <= 0:
            raise ValueError("gain must be > 0.")
