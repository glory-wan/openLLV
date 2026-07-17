"""Gamma correction enhancer."""

from typing import Any, Dict, Optional

import numpy as np

from ..BaseModel import LLVEnhancer

__all__ = ["Gamma"]


class Gamma(LLVEnhancer):
    """Gamma correction enhancer."""

    name = "Gamma"
    aliases = []

    def __init__(
        self,
        gamma: float = 0.6,
        **kwargs: Any,
    ) -> None:
        """Initialize gamma enhancer.

        Args:
            gamma: Gamma exponent. Values below 1 brighten images.
            **kwargs: Base enhancer parameters.

        Raises:
            TypeError: If ``gamma`` is not numeric.
            ValueError: If ``gamma`` is not positive.
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self._validate_gamma(self.gamma)

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply gamma correction.

        Args:
            image: Input image array.
            **kwargs: Optional ``gamma`` override.

        Returns:
            Gamma-corrected image array.
        """
        gamma = kwargs.get("gamma", self.gamma)
        self._validate_gamma(gamma)

        original_dtype = image.dtype

        image_float = image.astype(np.float32)

        if np.issubdtype(image.dtype, np.integer):
            max_value = np.iinfo(image.dtype).max
            image_float = image_float / max_value
        else:
            image_float = np.clip(image_float, 0.0, 1.0)

        enhanced = np.power(image_float, gamma)

        if np.issubdtype(original_dtype, np.integer):
            max_value = np.iinfo(original_dtype).max
            enhanced = enhanced * max_value
            enhanced = np.rint(enhanced)

        return enhanced.astype(original_dtype, copy=False)

    def get_params(self) -> Dict[str, Any]:
        """Get enhancer parameters.

        Returns:
            Dictionary containing base parameters and ``gamma``.
        """
        params = super().get_params()
        params.update(
            {
                "gamma": self.gamma,
            }
        )
        return params

    def set_params(self, **params: Any) -> "Gamma":
        """Set enhancer parameters.

        Args:
            **params: Parameter names and values.

        Returns:
            The enhancer itself.
        """
        super().set_params(**params)

        if "gamma" in params:
            self._validate_gamma(self.gamma)

        return self

    @staticmethod
    def _validate_gamma(gamma: float) -> None:
        """Validate gamma value.

        Args:
            gamma: Gamma value.

        Raises:
            TypeError: If ``gamma`` is not numeric.
            ValueError: If ``gamma`` is not positive.
        """
        if not isinstance(gamma, (int, float)):
            raise TypeError(f"gamma must be int or float, got {type(gamma)!r}.")

        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma!r}.")
