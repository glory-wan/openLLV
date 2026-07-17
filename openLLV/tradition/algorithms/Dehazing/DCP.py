"""Dark Channel Prior-based low-light enhancement.

Original paper: https://ieeexplore.ieee.org/document/5206515
"""

from typing import Any

import cv2
import numpy as np

from ..BaseModel import LLVEnhancer

__all__ = ["DarkChannel"]


class DarkChannel(LLVEnhancer):
    """Dark Channel Prior-based low-light enhancement.

    Original paper: https://ieeexplore.ieee.org/document/5206515
    """

    name = "DarkChannel"
    aliases = ["dcp"]

    def __init__(
        self,
        *,
        size: int = 15,
        omega: float = 0.95,
        t_min: float = 0.1,
        guided_radius: int = 60,
        guided_eps: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """Initialize dark-channel enhancer.

        Args:
            size: Dark-channel erosion kernel size.
            omega: Transmission estimate weight.
            t_min: Minimum transmission value.
            guided_radius: Guided filter radius.
            guided_eps: Guided filter regularization epsilon.
            **kwargs: Base enhancer parameters.

        Raises:
            ValueError: If any algorithm parameter is invalid.
        """
        super().__init__(**kwargs)

        self.size = size
        self.omega = omega
        self.t_min = t_min
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps

        self._validate_params()

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Run the dark-channel enhancement pipeline.

        Args:
            image: Input BGR image array.
            **kwargs: Unused method-specific parameters.

        Returns:
            Enhanced BGR image array.
        """

        img = image.astype(np.float64) / 255.0

        inv = 1.0 - img

        dark = self._dark_channel(inv)
        A = self._atm_light(inv, dark)

        te = self._transmission_estimate(inv, A)
        t = self._transmission_refine(image, te)

        J = self._recover(inv, t, A)

        result = 1.0 - J

        return result * 255.0

    def _dark_channel(self, im: np.ndarray) -> np.ndarray:
        """Compute dark channel.

        Args:
            im: Normalized BGR image array.

        Returns:
            Dark-channel map.
        """
        min_channel = np.min(im, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.size, self.size)
        )
        dark = cv2.erode(min_channel, kernel)
        return dark

    def _atm_light(self, im: np.ndarray, dark: np.ndarray) -> np.ndarray:
        """Estimate atmospheric light.

        Args:
            im: Normalized inverted image array.
            dark: Dark-channel map.

        Returns:
            Atmospheric light vector with shape ``[1, 3]``.
        """
        h, w = im.shape[:2]
        imsz = h * w
        numpx = max(imsz // 1000, 1)

        darkvec = dark.reshape(imsz)
        imvec = im.reshape(imsz, 3)

        indices = np.argsort(darkvec)[-numpx:]

        A = np.mean(imvec[indices], axis=0, keepdims=True)
        return A

    def _transmission_estimate(
        self,
        im: np.ndarray,
        A: np.ndarray,
    ) -> np.ndarray:
        """Estimate transmission map.

        Args:
            im: Normalized inverted image array.
            A: Atmospheric light vector.

        Returns:
            Estimated transmission map.
        """
        norm_im = im / A.reshape(1, 1, 3)
        transmission = 1 - self.omega * self._dark_channel(norm_im)
        return transmission

    def _guided_filter(
        self,
        I: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Apply guided filtering.

        Args:
            I: Guidance image.
            p: Filtering input map.

        Returns:
            Filtered map.
        """
        r = self.guided_radius
        eps = self.guided_eps

        mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))

        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        return mean_a * I + mean_b

    def _transmission_refine(
        self,
        im: np.ndarray,
        t_est: np.ndarray,
    ) -> np.ndarray:
        """Refine transmission map with guided filtering.

        Args:
            im: Original BGR image array.
            t_est: Estimated transmission map.

        Returns:
            Refined transmission map.
        """
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float64) / 255.0

        t = self._guided_filter(gray, t_est)
        return t

    def _recover(
        self,
        im: np.ndarray,
        t: np.ndarray,
        A: np.ndarray,
    ) -> np.ndarray:
        """Recover image using atmospheric scattering model.

        Args:
            im: Normalized inverted image array.
            t: Transmission map.
            A: Atmospheric light vector.

        Returns:
            Recovered image array.
        """
        t = np.maximum(t, self.t_min)

        J = (im - A.reshape(1, 1, 3)) / t[..., None] + A.reshape(1, 1, 3)
        return J

    def _validate_params(self) -> None:
        """Validate dark-channel parameters.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if self.size <= 0:
            raise ValueError("size must be > 0")

        if not (0 < self.omega <= 1):
            raise ValueError("omega must be in (0, 1]")

        if not (0 < self.t_min < 1):
            raise ValueError("t_min must be in (0, 1)")

        if self.guided_radius <= 0:
            raise ValueError("guided_radius must be > 0")

        if self.guided_eps <= 0:
            raise ValueError("guided_eps must be > 0")
