"""Gamma Correction Prior enhancer in mixed color spaces."""

from typing import Any, Dict

import cv2
import numpy as np

from ..BaseModel import LLVEnhancer

__all__ = ["GCP"]


class GCP(LLVEnhancer):
    """Gamma Correction Prior low-light image enhancer.

    Original paper: https://www.sciencedirect.com/science/article/abs/pii/S0031320323006994
    Official source code: https://github.com/TripleJ2543/Low_Light_Pattern_Recognition_2023
    """

    name = "GCP"
    aliases = ["gcp", "gcp-ms"]

    def __init__(
        self,
        *,
        gamma_max: float = 6.0,
        erosion_window: int = 15,
        atmospheric_bins: int = 200,
        atmospheric_percentile: float = 0.99,
        t_min: float = 0.1,
        blur_ksize: int = 7,
        high_percentile: float = 99.5,
        low_percentile: float = 0.5,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """Initialize the GCP enhancer.

        Args:
            gamma_max: Maximum adaptive gamma value.
            erosion_window: Dark-channel erosion kernel size.
            atmospheric_bins: Number of histogram bins used for atmospheric
                light estimation.
            atmospheric_percentile: Dark-channel percentile used to select
                atmospheric-light candidate pixels.
            t_min: Minimum transmission value.
            blur_ksize: Gaussian blur kernel size.
            high_percentile: Upper percentile used for final range adjustment.
            low_percentile: Lower percentile used for final range adjustment.
            eps: Small value used to avoid division by zero.
            **kwargs: Base enhancer parameters.

        Raises:
            ValueError: If any GCP parameter is outside its valid range.
        """
        super().__init__(**kwargs)

        self.gamma_max = gamma_max
        self.erosion_window = erosion_window
        self.atmospheric_bins = atmospheric_bins
        self.atmospheric_percentile = atmospheric_percentile
        self.t_min = t_min
        self.blur_ksize = blur_ksize
        self.high_percentile = high_percentile
        self.low_percentile = low_percentile
        self.eps = eps

        self._validate_params()

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Run the Gamma Correction Prior enhancement pipeline.

        Args:
            image: Input BGR image array.
            **kwargs: Optional parameter overrides for this call.

        Returns:
            Enhanced image array.
        """
        params = self.get_params()
        params.update(kwargs)
        self._validate_params_dict(params)

        original_dtype = image.dtype
        image_3c, alpha, grayscale_input = self._prepare_color_image(image)
        float_image = self._to_float01(image_3c)

        denoised_inverted = 1.0 - cv2.GaussianBlur(
            float_image,
            (int(params["blur_ksize"]), int(params["blur_ksize"])),
            0,
        )

        atmospheric_light = self._compute_atmospheric_light(
            denoised_inverted,
            erosion_window=int(params["erosion_window"]),
            n_bins=int(params["atmospheric_bins"]),
            percentile=float(params["atmospheric_percentile"]),
            eps=float(params["eps"]),
        )

        normalized = denoised_inverted / atmospheric_light.reshape(1, 1, 3)
        normalized = self._normalize_channels(normalized, eps=float(params["eps"]))

        transmission = self._estimate_transmission(
            float_image,
            normalized,
            gamma_max=float(params["gamma_max"]),
            t_min=float(params["t_min"]),
            eps=float(params["eps"]),
        )

        recovered = self._recover(
            float_image,
            transmission,
            atmospheric_light,
        )
        adjusted = self._adjust_range(
            recovered,
            high_percentile=float(params["high_percentile"]),
            low_percentile=float(params["low_percentile"]),
            eps=float(params["eps"]),
        )

        enhanced = self._restore_image_shape(
            adjusted,
            alpha=alpha,
            grayscale_input=grayscale_input,
        )

        if np.issubdtype(original_dtype, np.integer):
            return enhanced * np.iinfo(original_dtype).max

        return enhanced

    def get_params(self) -> Dict[str, Any]:
        """Get enhancer parameters.

        Returns:
            Dictionary containing base parameters and GCP-specific parameters.
        """
        params = super().get_params()
        params.update(
            {
                "gamma_max": self.gamma_max,
                "erosion_window": self.erosion_window,
                "atmospheric_bins": self.atmospheric_bins,
                "atmospheric_percentile": self.atmospheric_percentile,
                "t_min": self.t_min,
                "blur_ksize": self.blur_ksize,
                "high_percentile": self.high_percentile,
                "low_percentile": self.low_percentile,
                "eps": self.eps,
            }
        )
        return params

    def set_params(self, **params: Any) -> "GCP":
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
    def _prepare_color_image(image: np.ndarray):
        """Convert an input image to a three-channel working image.

        Args:
            image: Input image array.

        Returns:
            Tuple containing the three-channel image, optional alpha channel,
            and a flag indicating whether the input was grayscale.
        """
        if image.ndim == 2:
            return np.repeat(image[:, :, None], 3, axis=2), None, True

        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=2), None, True

        if image.shape[2] == 4:
            return image[:, :, :3], image[:, :, 3:4], False

        return image, None, False

    @staticmethod
    def _restore_image_shape(
        image: np.ndarray,
        *,
        alpha: np.ndarray,
        grayscale_input: bool,
    ) -> np.ndarray:
        """Restore the enhanced image to the original channel layout.

        Args:
            image: Enhanced three-channel image in ``[0, 1]``.
            alpha: Optional alpha channel from the source image.
            grayscale_input: Whether the source image was grayscale.

        Returns:
            Image with the source channel layout restored.
        """
        if grayscale_input:
            return image[:, :, 0]

        if alpha is not None:
            alpha_float = alpha.astype(np.float32)
            if alpha_float.size > 0 and alpha_float.max() > 1.0:
                alpha_float = alpha_float / 255.0
            return np.concatenate([image, np.clip(alpha_float, 0.0, 1.0)], axis=2)

        return image

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
    def _compute_atmospheric_light(
        image: np.ndarray,
        *,
        erosion_window: int,
        n_bins: int,
        percentile: float,
        eps: float,
    ) -> np.ndarray:
        """Estimate atmospheric light from the dark channel.

        Args:
            image: Normalized BGR image array.
            erosion_window: Dark-channel erosion kernel size.
            n_bins: Number of histogram bins.
            percentile: Candidate-pixel dark-channel percentile.
            eps: Small value used to avoid division by zero.

        Returns:
            Atmospheric light vector with shape ``(3,)``.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (erosion_window, erosion_window),
        )
        dark = cv2.erode(np.min(image, axis=2), kernel)
        hist, edges = np.histogram(dark, n_bins, (0.0, 1.0))

        threshold_count = image.shape[0] * image.shape[1] * percentile
        cumulative = np.cumsum(hist)
        indices = np.nonzero(cumulative > threshold_count)[0]
        threshold = edges[indices[0]] if indices.size else np.max(dark)

        mask = dark >= threshold
        if not np.any(mask):
            mask = dark >= np.max(dark)

        atmospheric_light = np.median(image[mask], axis=0)
        return np.maximum(atmospheric_light.astype(np.float32), eps)

    @staticmethod
    def _get_intensity(image: np.ndarray) -> np.ndarray:
        """Compute average channel intensity.

        Args:
            image: Input BGR image array.

        Returns:
            Intensity map.
        """
        return (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3.0

    @staticmethod
    def _get_max_channel(image: np.ndarray) -> np.ndarray:
        """Compute the maximum BGR channel value per pixel.

        Args:
            image: Input BGR image array.

        Returns:
            Maximum-channel map.
        """
        return np.max(image, axis=2)

    @staticmethod
    def _normalize_channels(image: np.ndarray, *, eps: float) -> np.ndarray:
        """Normalize each channel independently to ``[0, 1]``.

        Args:
            image: Input BGR image array.
            eps: Small value used to avoid division by zero.

        Returns:
            Channel-normalized image.
        """
        normalized = np.empty_like(image, dtype=np.float32)
        for channel in range(3):
            channel_data = image[:, :, channel]
            min_value = np.min(channel_data)
            max_value = np.max(channel_data)
            normalized[:, :, channel] = (channel_data - min_value) / max(
                max_value - min_value,
                eps,
            )

        return np.clip(normalized, 0.0, 1.0)

    @classmethod
    def _pixel_adaptive_gamma(
        cls,
        input_image: np.ndarray,
        normalized_image: np.ndarray,
        *,
        gamma_max: float,
    ) -> np.ndarray:
        """Apply pixel-adaptive gamma correction.

        Args:
            input_image: Normalized source image in ``[0, 1]``.
            normalized_image: Atmospheric-light-normalized image.
            gamma_max: Maximum adaptive gamma value.

        Returns:
            Gamma-corrected image.
        """
        gamma_min = 1.0
        x_min = 0.0
        x_max = 1.0
        image_max = cls._get_max_channel(input_image)

        scale = (gamma_max - gamma_min) / (np.exp(-x_min) - np.exp(-x_max))
        offset = gamma_max - scale * np.exp(-x_min)
        gamma_curve = scale * np.exp(-image_max) + offset
        gamma = np.where(image_max < x_min, gamma_max, gamma_curve)
        gamma = np.where(image_max > x_max, gamma_min, gamma)

        return normalized_image ** gamma[:, :, None]

    @classmethod
    def _estimate_transmission(
        cls,
        input_image: np.ndarray,
        normalized_image: np.ndarray,
        *,
        gamma_max: float,
        t_min: float,
        eps: float,
    ) -> np.ndarray:
        """Estimate transmission map.

        Args:
            input_image: Normalized source image in ``[0, 1]``.
            normalized_image: Atmospheric-light-normalized image.
            gamma_max: Maximum adaptive gamma value.
            t_min: Minimum transmission value.
            eps: Small value used to avoid division by zero.

        Returns:
            Transmission map clipped to ``[t_min, 1]``.
        """
        input_intensity = cls._get_intensity(normalized_image)
        input_max = cls._get_max_channel(normalized_image)
        gamma_corrected = cls._pixel_adaptive_gamma(
            input_image,
            normalized_image,
            gamma_max=gamma_max,
        )
        corrected_intensity = cls._get_intensity(gamma_corrected)
        corrected_max = cls._get_max_channel(gamma_corrected)

        numerator = np.maximum(
            corrected_max * input_intensity - input_max * corrected_intensity,
            eps,
        )
        denominator = np.maximum(
            (corrected_max - corrected_intensity) * input_intensity,
            eps,
        )
        transmission = 1.0 - input_intensity * (numerator / denominator)
        return np.clip(transmission, t_min, 1.0)

    @staticmethod
    def _recover(
        image: np.ndarray,
        transmission: np.ndarray,
        atmospheric_light: np.ndarray,
    ) -> np.ndarray:
        """Recover an enhanced image with the estimated transmission map.

        Args:
            image: Normalized source image in ``[0, 1]``.
            transmission: Estimated transmission map.
            atmospheric_light: Atmospheric light vector.

        Returns:
            Recovered image clipped to ``[0, 1]``.
        """
        recovered = (
            image - 1.0 + atmospheric_light.reshape(1, 1, 3)
        ) / transmission[:, :, None] + 1.0 - atmospheric_light.reshape(1, 1, 3)
        return np.clip(recovered, 0.0, 1.0)

    @staticmethod
    def _adjust_range(
        image: np.ndarray,
        *,
        high_percentile: float,
        low_percentile: float,
        eps: float,
    ) -> np.ndarray:
        """Adjust the image dynamic range using global percentiles.

        Args:
            image: Recovered image in ``[0, 1]``.
            high_percentile: Upper percentile.
            low_percentile: Lower percentile.
            eps: Small value used to avoid division by zero.

        Returns:
            Percentile-adjusted image.
        """
        high_value = np.percentile(image, high_percentile)
        low_value = np.percentile(image, low_percentile)
        adjusted = (image - low_value) / max(high_value - low_value, eps)
        return np.clip(adjusted, 0.0, 1.0)

    def _validate_params(self) -> None:
        """Validate instance parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        self._validate_params_dict(self.get_params())

    @staticmethod
    def _validate_params_dict(params: Dict[str, Any]) -> None:
        """Validate GCP parameter values.

        Args:
            params: Parameter dictionary.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if params["gamma_max"] < 1:
            raise ValueError("gamma_max must be >= 1.")

        if params["erosion_window"] <= 0:
            raise ValueError("erosion_window must be > 0.")

        if params["atmospheric_bins"] <= 0:
            raise ValueError("atmospheric_bins must be > 0.")

        if not (0 < params["atmospheric_percentile"] < 1):
            raise ValueError("atmospheric_percentile must be in (0, 1).")

        if not (0 < params["t_min"] <= 1):
            raise ValueError("t_min must be in (0, 1].")

        if params["blur_ksize"] <= 0 or params["blur_ksize"] % 2 == 0:
            raise ValueError("blur_ksize must be a positive odd integer.")

        if not (0 <= params["low_percentile"] < params["high_percentile"] <= 100):
            raise ValueError(
                "Percentiles must satisfy 0 <= low_percentile < "
                "high_percentile <= 100."
            )

        if params["eps"] <= 0:
            raise ValueError("eps must be > 0.")
