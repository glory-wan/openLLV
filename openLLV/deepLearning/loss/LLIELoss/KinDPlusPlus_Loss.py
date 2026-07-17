"""KinD++ supervised loss functions."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class KinDPPSSIMLoss(nn.Module):
    """Differentiable ``1 - SSIM`` loss."""

    def __init__(self, window_size: int = 11, data_range: float = 1.0) -> None:
        """Initialize SSIM loss.

        Args:
            window_size: Gaussian window size.
            data_range: Dynamic range of image values.
        """
        super().__init__()
        self.window_size = int(window_size)
        self.data_range = float(data_range)
        self.register_buffer("window", self._create_window(self.window_size))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ``1 - SSIM``.

        Args:
            prediction: Predicted image tensor.
            target: Target image tensor.

        Returns:
            Scalar SSIM loss.
        """
        channels = prediction.shape[1]
        window = self.window.to(prediction.device, prediction.dtype).expand(
            channels,
            1,
            self.window_size,
            self.window_size,
        )
        padding = self.window_size // 2
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
        mu_y = F.conv2d(target, window, padding=padding, groups=channels)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_xy

        ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        )
        return 1.0 - ssim.mean()

    @staticmethod
    def _create_window(window_size: int) -> torch.Tensor:
        """Create a Gaussian SSIM window.

        Args:
            window_size: Window size.

        Returns:
            Window tensor.
        """
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window = gauss[:, None] @ gauss[None, :]
        return window.view(1, 1, window_size, window_size)


class KinDPlusPlus_Loss(BaseLoss):
    """KinD++ loss adapted from the official staged training objectives."""

    name = "kindplusplus"
    aliases = [
        "kind++",
        "kind_plus_plus",
        "kindplusplus_loss",
        "kind++_loss",
        "KinDPlusPlus_Loss",
    ]
    requires_target = True

    def __init__(
        self,
        *,
        decomposition_weight: float = 1.0,
        equal_reflectance_weight: float = 0.009,
        mutual_illumination_weight: float = 0.2,
        input_illumination_weight: float = 0.15,
        restoration_mse_weight: float = 1.0,
        restoration_ssim_weight: float = 1.0,
        adjustment_mse_weight: float = 1.0,
        adjustment_gradient_weight: float = 1.0,
        final_reconstruction_weight: float = 1.0,
        final_ssim_weight: float = 0.1,
        eps: float = 1e-4,
    ) -> None:
        """Initialize KinD++ loss.

        Args:
            decomposition_weight: Weight for decomposition-stage losses.
            equal_reflectance_weight: Weight for low/high reflectance equality.
            mutual_illumination_weight: Weight for mutual illumination loss.
            input_illumination_weight: Weight for input-aware illumination loss.
            restoration_mse_weight: Weight for reflectance MSE restoration.
            restoration_ssim_weight: Weight for reflectance SSIM restoration.
            adjustment_mse_weight: Weight for illumination adjustment MSE.
            adjustment_gradient_weight: Weight for illumination gradient loss.
            final_reconstruction_weight: Weight for final enhanced image L1 loss.
            final_ssim_weight: Weight for final enhanced image SSIM loss.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.decomposition_weight = float(decomposition_weight)
        self.equal_reflectance_weight = float(equal_reflectance_weight)
        self.mutual_illumination_weight = float(mutual_illumination_weight)
        self.input_illumination_weight = float(input_illumination_weight)
        self.restoration_mse_weight = float(restoration_mse_weight)
        self.restoration_ssim_weight = float(restoration_ssim_weight)
        self.adjustment_mse_weight = float(adjustment_mse_weight)
        self.adjustment_gradient_weight = float(adjustment_gradient_weight)
        self.final_reconstruction_weight = float(final_reconstruction_weight)
        self.final_ssim_weight = float(final_ssim_weight)
        self.eps = float(eps)
        self.ssim_loss = KinDPPSSIMLoss()

    def compute(
        self,
        *,
        input_tensor: torch.Tensor,
        model_output: Any,
        target: Optional[torch.Tensor] = None,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        align_prediction: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute KinD++ loss through the Trainer interface.

        Args:
            input_tensor: Low-light input tensor.
            model_output: Raw KinD++ model output.
            target: Paired normal-light target tensor.
            extract_prediction: Optional prediction extractor.
            align_prediction: Optional prediction alignment callback.

        Returns:
            Tuple containing scalar loss and prediction tensor.

        Raises:
            ValueError: If required target or model outputs are missing.
        """
        if target is None:
            raise ValueError("KinDPlusPlus_Loss requires a paired normal-light target tensor.")
        if not isinstance(model_output, dict):
            raise ValueError("KinDPlusPlus_Loss expects KinD++ training output dictionary.")

        prediction = self._extract_prediction(model_output, target, extract_prediction)
        if align_prediction is not None:
            prediction = align_prediction(prediction, target)

        loss_inputs = get_loss_inputs(model_output)
        decompose_fn = loss_inputs.get("decompose_fn")
        if decompose_fn is None:
            raise ValueError("KinD++ model output must provide decompose_fn.")

        high_reflectance, high_illumination = decompose_fn(target)
        low_reflectance = loss_inputs["low_reflectance"]
        low_illumination = loss_inputs["low_illumination"]
        restored_reflectance = loss_inputs["restored_reflectance"]
        adjusted_illumination = loss_inputs["adjusted_illumination"]

        loss = prediction.new_tensor(0.0)
        loss = loss + self.decomposition_weight * self._decomposition_loss(
            input_tensor=input_tensor,
            target=target,
            low_reflectance=low_reflectance,
            low_illumination=low_illumination,
            high_reflectance=high_reflectance,
            high_illumination=high_illumination,
        )

        high_reflectance_target = torch.clamp((high_reflectance * 0.99).pow(1.2), 0.0, 1.0)
        if self.restoration_mse_weight > 0:
            loss = loss + self.restoration_mse_weight * F.mse_loss(restored_reflectance, high_reflectance_target)
        if self.restoration_ssim_weight > 0:
            loss = loss + self.restoration_ssim_weight * self.ssim_loss(restored_reflectance, high_reflectance_target)

        if self.adjustment_mse_weight > 0:
            loss = loss + self.adjustment_mse_weight * F.mse_loss(adjusted_illumination, high_illumination)
        if self.adjustment_gradient_weight > 0:
            loss = loss + self.adjustment_gradient_weight * self._gradient_mse(
                adjusted_illumination,
                high_illumination,
            )

        target_clamped = target.clamp(0.0, 1.0)
        prediction_clamped = prediction.clamp(0.0, 1.0)
        if self.final_reconstruction_weight > 0:
            loss = loss + self.final_reconstruction_weight * F.l1_loss(prediction_clamped, target_clamped)
        if self.final_ssim_weight > 0:
            loss = loss + self.final_ssim_weight * self.ssim_loss(prediction_clamped, target_clamped)

        return loss, prediction

    def _decomposition_loss(
        self,
        *,
        input_tensor: torch.Tensor,
        target: torch.Tensor,
        low_reflectance: torch.Tensor,
        low_illumination: torch.Tensor,
        high_reflectance: torch.Tensor,
        high_illumination: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KinD++ decomposition losses.

        Args:
            input_tensor: Low-light input image.
            target: Normal-light target image.
            low_reflectance: Low-light reflectance.
            low_illumination: Low-light illumination.
            high_reflectance: Normal-light reflectance.
            high_illumination: Normal-light illumination.

        Returns:
            Scalar decomposition loss.
        """
        input_clamped = input_tensor.clamp(0.0, 1.0)
        target_clamped = target.clamp(0.0, 1.0)
        loss = F.l1_loss(low_reflectance * low_illumination, input_clamped)
        loss = loss + F.l1_loss(high_reflectance * high_illumination, target_clamped)
        loss = loss + self.equal_reflectance_weight * F.l1_loss(low_reflectance, high_reflectance)
        loss = loss + self.mutual_illumination_weight * self._mutual_illumination_loss(
            low_illumination,
            high_illumination,
        )
        loss = loss + self.input_illumination_weight * self._input_illumination_loss(
            low_illumination,
            input_clamped,
        )
        loss = loss + self.input_illumination_weight * self._input_illumination_loss(
            high_illumination,
            target_clamped,
        )
        return loss

    @staticmethod
    def _gradient_abs(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute absolute gradients.

        Args:
            image: Input tensor.

        Returns:
            Tuple containing x and y absolute gradients.
        """
        grad_x = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
        grad_y = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        return grad_x, grad_y

    @staticmethod
    def _gradient_signed(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute signed gradients.

        Args:
            image: Input tensor.

        Returns:
            Tuple containing x and y gradients.
        """
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
        return grad_x, grad_y

    def _mutual_illumination_loss(
        self,
        low_illumination: torch.Tensor,
        high_illumination: torch.Tensor,
    ) -> torch.Tensor:
        """Compute official mutual illumination smoothness term.

        Args:
            low_illumination: Low-light illumination.
            high_illumination: Normal-light illumination.

        Returns:
            Scalar mutual illumination loss.
        """
        low_x, low_y = self._gradient_abs(low_illumination)
        high_x, high_y = self._gradient_abs(high_illumination)
        x_sum = low_x + high_x
        y_sum = low_y + high_y
        return torch.mean(x_sum * torch.exp(-10.0 * x_sum)) + torch.mean(y_sum * torch.exp(-10.0 * y_sum))

    def _input_illumination_loss(self, illumination: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Compute input-aware illumination smoothness.

        Args:
            illumination: Illumination tensor.
            image: RGB image tensor.

        Returns:
            Scalar input-aware illumination loss.
        """
        gray = image.mean(dim=1, keepdim=True)
        illum_x, illum_y = self._gradient_abs(illumination)
        gray_x, gray_y = self._gradient_abs(gray)
        x_loss = torch.abs(illum_x / torch.maximum(gray_x, gray_x.new_tensor(0.01)))
        y_loss = torch.abs(illum_y / torch.maximum(gray_y, gray_y.new_tensor(0.01)))
        return torch.mean(x_loss) + torch.mean(y_loss)

    def _gradient_mse(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute illumination gradient MSE.

        Args:
            prediction: Adjusted illumination tensor.
            target: Target illumination tensor.

        Returns:
            Scalar gradient MSE loss.
        """
        pred_x, pred_y = self._gradient_signed(prediction)
        target_x, target_y = self._gradient_signed(target)
        return F.mse_loss(pred_x, target_x) + F.mse_loss(pred_y, target_y)

    @staticmethod
    def _extract_prediction(
        model_output: Dict[str, Any],
        target: torch.Tensor,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        """Extract prediction tensor from KinD++ output.

        Args:
            model_output: Raw model output.
            target: Target tensor used by fallback extractor.
            extract_prediction: Optional fallback extractor.

        Returns:
            Prediction tensor.

        Raises:
            TypeError: If prediction cannot be extracted.
        """
        if torch.is_tensor(model_output.get("pred")):
            return model_output["pred"]
        if extract_prediction is not None:
            return extract_prediction(model_output, target)
        raise TypeError("Cannot extract KinD++ prediction tensor.")


KinDPlusPlusLoss = KinDPlusPlus_Loss
