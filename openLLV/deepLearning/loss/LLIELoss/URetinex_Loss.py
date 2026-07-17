"""URetinex-Net supervised loss functions."""

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs, has_loss_inputs


class URetinexTVLoss(nn.Module):
    """Total variation loss for URetinex illumination maps."""

    def forward(self, illumination: torch.Tensor) -> torch.Tensor:
        """Compute illumination total variation.

        Args:
            illumination: Illumination tensor with shape ``[B, 1, H, W]``.

        Returns:
            Scalar total variation loss tensor.
        """
        batch, _, height, width = illumination.shape
        count_h = max(1, (height - 1) * width)
        count_w = max(1, height * (width - 1))
        h_tv = torch.pow(illumination[:, :, 1:, :] - illumination[:, :, : height - 1, :], 2).sum()
        w_tv = torch.pow(illumination[:, :, :, 1:] - illumination[:, :, :, : width - 1], 2).sum()
        return (h_tv / count_h + w_tv / count_w) / batch


class URetinexGradientConsistencyLoss(nn.Module):
    """Gradient consistency loss for URetinex illumination adjustment."""

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute absolute-gradient consistency loss.

        Args:
            source: Source image or illumination tensor.
            target: Target image or illumination tensor.

        Returns:
            Scalar gradient consistency loss tensor.
        """
        source_y, source_x = self._gradient(source)
        target_y, target_x = self._gradient(target)
        return F.l1_loss(source_y, target_y) + F.l1_loss(source_x, target_x)

    @staticmethod
    def _gradient(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute absolute vertical and horizontal gradients.

        Args:
            image: Input tensor with shape ``[B, C, H, W]``.

        Returns:
            A tuple containing vertical and horizontal gradient tensors.
        """
        grad_y = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        grad_x = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
        return grad_y, grad_x


class URetinexSSIMLoss(nn.Module):
    """Differentiable ``1 - SSIM`` loss."""

    def __init__(self, window_size: int = 11, data_range: float = 1.0) -> None:
        """Initialize SSIM loss.

        Args:
            window_size: Gaussian window size.
            data_range: Dynamic range of image values.
        """
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.register_buffer("window", self._create_window(window_size))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ``1 - SSIM`` loss.

        Args:
            prediction: Predicted image tensor.
            target: Target image tensor.

        Returns:
            Scalar SSIM loss tensor.
        """
        channels = prediction.shape[1]
        window = self.window.to(prediction.device, prediction.dtype).expand(
            channels,
            1,
            self.window_size,
            self.window_size,
        )

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        padding = self.window_size // 2

        mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
        mu_y = F.conv2d(target, window, padding=padding, groups=channels)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_xy

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        )
        return 1.0 - ssim_map.mean()

    @staticmethod
    def _create_window(window_size: int) -> torch.Tensor:
        """Create a 2D Gaussian SSIM window.

        Args:
            window_size: Gaussian window size.

        Returns:
            Window tensor with shape ``[1, 1, window_size, window_size]``.
        """
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window = gauss[:, None] @ gauss[None, :]
        return window.view(1, 1, window_size, window_size)


class URetinex_Loss(BaseLoss):
    """URetinex-Net loss adapted to openLLV's current URetinexNet output contract.

    The paper trains three modules separately:
    - initialization decomposition loss: ||I - R*L||_1 + mu ||L - max_c(I)||_2^2
    - unfolding loss: P/R and Q/L constraints, TV(L), reflectance MSE/SSIM/VGG
    - illumination adjustment loss: ||grad(L_hat)-grad(L)||_1
      + ||R*L_hat - I_high||_2^2 + (1 - SSIM(R*L_hat, I_high))

    Current openLLV URetinexNet exposes final R, L and adjusted illumination
    High_L. This loss implements all terms available from those tensors.
    """

    name = "uretinex"
    aliases = ["uretinex_loss", "uretinexnet", "uretinexnet_loss", "URetinex_Loss"]
    requires_target = True

    def __init__(
        self,
        *,
        low_reconstruction_weight: float = 1.0,
        illumination_prior_weight: float = 0.1,
        illumination_tv_weight: float = 0.01,
        reflectance_weight: float = 0.1,
        adjustment_reconstruction_weight: float = 1.0,
        adjustment_ssim_weight: float = 1.0,
        adjustment_gradient_weight: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        """Initialize URetinex loss.

        Args:
            low_reconstruction_weight: Weight for ``R * L`` low-light
                reconstruction loss.
            illumination_prior_weight: Weight for illumination prior loss.
            illumination_tv_weight: Weight for illumination total variation.
            reflectance_weight: Weight for reflectance supervision.
            adjustment_reconstruction_weight: Weight for prediction-target
                reconstruction loss.
            adjustment_ssim_weight: Weight for SSIM loss.
            adjustment_gradient_weight: Weight for illumination gradient
                consistency loss.
            eps: Small constant used for numerical stability.
        """
        super().__init__()
        self.low_reconstruction_weight = float(low_reconstruction_weight)
        self.illumination_prior_weight = float(illumination_prior_weight)
        self.illumination_tv_weight = float(illumination_tv_weight)
        self.reflectance_weight = float(reflectance_weight)
        self.adjustment_reconstruction_weight = float(adjustment_reconstruction_weight)
        self.adjustment_ssim_weight = float(adjustment_ssim_weight)
        self.adjustment_gradient_weight = float(adjustment_gradient_weight)
        self.eps = float(eps)

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.tv_loss = URetinexTVLoss()
        self.ssim_loss = URetinexSSIMLoss()
        self.gradient_loss = URetinexGradientConsistencyLoss()

    def forward(
        self,
        low_image: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
        *,
        reflectance: torch.Tensor,
        illumination: torch.Tensor,
        adjusted_illumination: torch.Tensor,
    ) -> torch.Tensor:
        """Compute URetinex supervised loss.

        Args:
            low_image: Low-light input tensor.
            target: Paired normal-light target tensor.
            prediction: Enhanced prediction tensor.
            reflectance: Estimated reflectance tensor.
            illumination: Estimated low-light illumination tensor.
            adjusted_illumination: Adjusted illumination tensor.

        Returns:
            Scalar URetinex loss tensor.
        """
        low_image = low_image.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        prediction = prediction.clamp(0.0, 1.0)

        retinex_low = (reflectance * illumination).clamp(0.0, 1.0)
        illumination_init = torch.max(low_image, dim=1, keepdim=True).values

        loss = low_image.new_tensor(0.0)
        loss = loss + self.low_reconstruction_weight * self.l1(retinex_low, low_image)
        loss = loss + self.illumination_prior_weight * self.mse(illumination, illumination_init)
        loss = loss + self.illumination_tv_weight * self.tv_loss(illumination)

        if self.reflectance_weight > 0:
            target_illumination = torch.max(target, dim=1, keepdim=True).values.clamp_min(self.eps)
            target_reflectance = (target / target_illumination).clamp(0.0, 1.0)
            loss = loss + self.reflectance_weight * self.mse(reflectance.clamp(0.0, 1.0), target_reflectance)

        loss = loss + self.adjustment_reconstruction_weight * self.mse(prediction, target)
        loss = loss + self.adjustment_ssim_weight * self.ssim_loss(prediction, target)
        loss = loss + self.adjustment_gradient_weight * self.gradient_loss(adjusted_illumination, illumination)
        return loss

    def compute(
        self,
        *,
        input_tensor: torch.Tensor,
        model_output: Any,
        target: Optional[torch.Tensor] = None,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        align_prediction: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute URetinex loss through the Trainer interface.

        Args:
            input_tensor: Low-light input tensor.
            model_output: Raw URetinex model output.
            target: Paired normal-light target tensor.
            extract_prediction: Optional callback for extracting prediction
                from non-standard outputs.
            align_prediction: Optional callback for aligning prediction shape
                with target.

        Returns:
            A tuple containing the scalar loss tensor and prediction tensor.

        Raises:
            ValueError: If ``target`` is missing.
            TypeError: If prediction or loss inputs cannot be extracted.
            KeyError: If required URetinex ``loss_inputs`` are missing.
        """
        if target is None:
            raise ValueError("URetinex_Loss requires a paired normal-light target tensor.")

        prediction = self._extract_prediction(model_output, target, extract_prediction)
        if align_prediction is not None:
            prediction = align_prediction(prediction, target)

        if not self._has_loss_inputs(model_output):
            loss = target.new_tensor(0.0)
            loss = loss + self.adjustment_reconstruction_weight * self.mse(prediction.clamp(0.0, 1.0), target.clamp(0.0, 1.0))
            loss = loss + self.adjustment_ssim_weight * self.ssim_loss(prediction.clamp(0.0, 1.0), target.clamp(0.0, 1.0))
            return loss, prediction

        loss_inputs = self._extract_loss_inputs(model_output)
        reflectance = loss_inputs["reflectance"]
        illumination = loss_inputs["illumination"]
        adjusted_illumination = loss_inputs["adjusted_illumination"]

        loss = self(
            input_tensor,
            target,
            prediction,
            reflectance=reflectance,
            illumination=illumination,
            adjusted_illumination=adjusted_illumination,
        )
        return loss, prediction

    @staticmethod
    def _extract_prediction(
        model_output: Any,
        target: torch.Tensor,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        """Extract prediction tensor from URetinex model output.

        Args:
            model_output: Raw model output.
            target: Target tensor passed to an optional extractor.
            extract_prediction: Optional fallback extractor callback.

        Returns:
            Prediction tensor.

        Raises:
            TypeError: If prediction cannot be extracted.
        """
        if torch.is_tensor(model_output):
            return model_output
        if isinstance(model_output, dict) and torch.is_tensor(model_output.get("pred")):
            return model_output["pred"]
        if extract_prediction is not None:
            return extract_prediction(model_output, target)
        raise TypeError(f"Cannot extract URetinex prediction from {type(model_output).__name__}.")

    @staticmethod
    def _extract_loss_inputs(model_output: Any):
        """Extract URetinex intermediate tensors from model output.

        Args:
            model_output: Raw model output dictionary.

        Returns:
            Dictionary containing ``reflectance``, ``illumination``, and
            ``adjusted_illumination`` tensors.

        Raises:
            TypeError: If ``model_output`` is not a dictionary.
            KeyError: If required intermediate tensors are missing.
        """
        if not isinstance(model_output, dict):
            raise TypeError("URetinex_Loss requires model_output dict with loss_inputs.")

        loss_inputs = get_loss_inputs(model_output)
        required = ("reflectance", "illumination", "adjusted_illumination")
        missing = [name for name in required if name not in loss_inputs]
        if missing:
            raise KeyError(f"URetinex model output missing loss_inputs: {missing}")
        return loss_inputs

    @staticmethod
    def _has_loss_inputs(model_output: Any) -> bool:
        """Check whether model output contains structured loss inputs.

        Args:
            model_output: Raw model output.

        Returns:
            True if ``model_output`` contains a ``loss_inputs`` dictionary.
        """
        return has_loss_inputs(model_output)


URetinexLoss = URetinex_Loss
