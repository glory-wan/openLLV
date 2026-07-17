"""KinD supervised loss functions."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class KinDSSIMLoss(nn.Module):
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
        """Compute ``1 - SSIM`` loss.

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

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        )
        return 1.0 - ssim_map.mean()

    @staticmethod
    def _create_window(window_size: int) -> torch.Tensor:
        """Create a 2D Gaussian SSIM window.

        Args:
            window_size: Window size.

        Returns:
            Window tensor with shape ``[1, 1, window_size, window_size]``.
        """
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window = gauss[:, None] @ gauss[None, :]
        return window.view(1, 1, window_size, window_size)


class KinDGradientLoss(nn.Module):
    """Absolute-gradient consistency loss."""

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient consistency.

        Args:
            source: Source tensor.
            target: Target tensor.

        Returns:
            Scalar gradient loss.
        """
        source_y, source_x = self._gradient(source)
        target_y, target_x = self._gradient(target)
        return F.l1_loss(source_y, target_y) + F.l1_loss(source_x, target_x)

    @staticmethod
    def _gradient(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute vertical and horizontal gradients.

        Args:
            image: Input tensor.

        Returns:
            Tuple containing vertical and horizontal gradients.
        """
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        return grad_y, grad_x


class KinD_Loss(BaseLoss):
    """KinD loss adapted from the official staged training objectives.

    The original KinD implementation trains decomposition, reflectance
    restoration, and illumination adjustment networks with separate objectives.
    This loss combines the available terms into a single openLLV training
    objective while preserving the main Retinex constraints.
    """

    name = "kind"
    aliases = ["kind_loss", "KinD_Loss", "kindling_the_darkness"]
    requires_target = True

    def __init__(
        self,
        *,
        decomposition_weight: float = 1.0,
        mutual_reconstruction_weight: float = 0.001,
        reflectance_consistency_weight: float = 0.01,
        illumination_smoothness_weight: float = 0.1,
        restoration_weight: float = 1.0,
        restoration_ssim_weight: float = 0.1,
        adjustment_weight: float = 1.0,
        adjustment_gradient_weight: float = 0.1,
        final_weight: float = 1.0,
        final_ssim_weight: float = 0.1,
    ) -> None:
        """Initialize KinD loss.

        Args:
            decomposition_weight: Weight for same-image Retinex reconstruction.
            mutual_reconstruction_weight: Weight for cross reconstruction.
            reflectance_consistency_weight: Weight for low/high reflectance
                consistency.
            illumination_smoothness_weight: Weight for illumination smoothness.
            restoration_weight: Weight for reflectance restoration supervision.
            restoration_ssim_weight: Weight for reflectance SSIM loss.
            adjustment_weight: Weight for illumination adjustment supervision.
            adjustment_gradient_weight: Weight for illumination gradient loss.
            final_weight: Weight for final enhanced-image reconstruction.
            final_ssim_weight: Weight for final enhanced-image SSIM loss.
        """
        super().__init__()
        self.decomposition_weight = float(decomposition_weight)
        self.mutual_reconstruction_weight = float(mutual_reconstruction_weight)
        self.reflectance_consistency_weight = float(reflectance_consistency_weight)
        self.illumination_smoothness_weight = float(illumination_smoothness_weight)
        self.restoration_weight = float(restoration_weight)
        self.restoration_ssim_weight = float(restoration_ssim_weight)
        self.adjustment_weight = float(adjustment_weight)
        self.adjustment_gradient_weight = float(adjustment_gradient_weight)
        self.final_weight = float(final_weight)
        self.final_ssim_weight = float(final_ssim_weight)
        self.ssim_loss = KinDSSIMLoss()
        self.gradient_loss = KinDGradientLoss()

    def compute(
        self,
        *,
        input_tensor: torch.Tensor,
        model_output: Any,
        target: Optional[torch.Tensor] = None,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        align_prediction: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute KinD loss through the Trainer interface.

        Args:
            input_tensor: Low-light input tensor.
            model_output: Raw KinD model output.
            target: Paired normal-light target tensor.
            extract_prediction: Optional prediction extractor.
            align_prediction: Optional prediction alignment callback.

        Returns:
            Tuple containing scalar loss and prediction tensor.

        Raises:
            ValueError: If target or KinD loss inputs are missing.
        """
        if target is None:
            raise ValueError("KinD_Loss requires a paired normal-light target tensor.")
        if not isinstance(model_output, dict):
            raise ValueError("KinD_Loss expects KinD training output dictionary.")

        prediction = self._extract_prediction(model_output, target, extract_prediction)
        if align_prediction is not None:
            prediction = align_prediction(prediction, target)

        loss_inputs = get_loss_inputs(model_output)
        decompose_fn = loss_inputs.get("decompose_fn")
        if decompose_fn is None:
            raise ValueError("KinD model output must provide a decompose_fn for target decomposition.")

        high_reflectance, high_illumination = decompose_fn(target)
        low_reflectance = loss_inputs["low_reflectance"]
        low_illumination = loss_inputs["low_illumination"]
        restored_reflectance = loss_inputs["restored_reflectance"]
        adjusted_illumination = loss_inputs["adjusted_illumination"]

        high_reflectance = high_reflectance.clamp(0.0, 1.0)
        high_illumination = high_illumination.clamp(0.0, 1.0)

        loss = prediction.new_tensor(0.0)
        loss = loss + self.decomposition_weight * self._decomposition_loss(
            input_tensor=input_tensor,
            target=target,
            low_reflectance=low_reflectance,
            low_illumination=low_illumination,
            high_reflectance=high_reflectance,
            high_illumination=high_illumination,
        )
        loss = loss + self.restoration_weight * F.l1_loss(restored_reflectance, high_reflectance)
        if self.restoration_ssim_weight > 0:
            loss = loss + self.restoration_ssim_weight * self.ssim_loss(restored_reflectance, high_reflectance)

        loss = loss + self.adjustment_weight * F.l1_loss(adjusted_illumination, high_illumination)
        if self.adjustment_gradient_weight > 0:
            loss = loss + self.adjustment_gradient_weight * self.gradient_loss(
                adjusted_illumination,
                high_illumination,
            )

        loss = loss + self.final_weight * F.l1_loss(prediction.clamp(0.0, 1.0), target.clamp(0.0, 1.0))
        if self.final_ssim_weight > 0:
            loss = loss + self.final_ssim_weight * self.ssim_loss(prediction.clamp(0.0, 1.0), target.clamp(0.0, 1.0))

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
        """Compute KinD decomposition losses.

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
        low_reconstruction = low_reflectance * low_illumination
        high_reconstruction = high_reflectance * high_illumination
        loss = F.l1_loss(low_reconstruction, input_tensor.clamp(0.0, 1.0))
        loss = loss + F.l1_loss(high_reconstruction, target.clamp(0.0, 1.0))

        if self.mutual_reconstruction_weight > 0:
            loss = loss + self.mutual_reconstruction_weight * F.l1_loss(
                high_reflectance * low_illumination,
                input_tensor.clamp(0.0, 1.0),
            )
            loss = loss + self.mutual_reconstruction_weight * F.l1_loss(
                low_reflectance * high_illumination,
                target.clamp(0.0, 1.0),
            )

        if self.reflectance_consistency_weight > 0:
            loss = loss + self.reflectance_consistency_weight * F.l1_loss(low_reflectance, high_reflectance)

        if self.illumination_smoothness_weight > 0:
            smooth = self._illumination_smoothness(low_illumination, low_reflectance)
            smooth = smooth + self._illumination_smoothness(high_illumination, high_reflectance)
            loss = loss + self.illumination_smoothness_weight * smooth

        return loss

    @staticmethod
    def _illumination_smoothness(illumination: torch.Tensor, reflectance: torch.Tensor) -> torch.Tensor:
        """Compute edge-aware illumination smoothness.

        Args:
            illumination: Illumination tensor.
            reflectance: Reflectance tensor.

        Returns:
            Scalar smoothness loss.
        """
        illum_y = illumination[:, :, 1:, :] - illumination[:, :, :-1, :]
        illum_x = illumination[:, :, :, 1:] - illumination[:, :, :, :-1]
        reflectance_y = torch.mean(
            torch.abs(reflectance[:, :, 1:, :] - reflectance[:, :, :-1, :]),
            dim=1,
            keepdim=True,
        )
        reflectance_x = torch.mean(
            torch.abs(reflectance[:, :, :, 1:] - reflectance[:, :, :, :-1]),
            dim=1,
            keepdim=True,
        )
        weight_y = torch.exp(-10.0 * reflectance_y)
        weight_x = torch.exp(-10.0 * reflectance_x)
        return torch.mean(torch.abs(illum_y) * weight_y) + torch.mean(torch.abs(illum_x) * weight_x)

    @staticmethod
    def _extract_prediction(
        model_output: Dict[str, Any],
        target: torch.Tensor,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        """Extract prediction tensor from KinD output.

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
        raise TypeError("Cannot extract KinD prediction tensor.")


KinDLoss = KinD_Loss
