"""ZERO-IG zero-shot loss functions."""

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


def _pair_downsampler(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Downsample an image with two complementary pair filters.

    Args:
        image: Input tensor with shape ``[B, C, H, W]``.

    Returns:
        A tuple containing two downsampled tensors.
    """
    channels = image.shape[1]
    filter1 = image.new_tensor([[[[0.0, 0.5], [0.5, 0.0]]]]).repeat(channels, 1, 1, 1)
    filter2 = image.new_tensor([[[[0.5, 0.0], [0.0, 0.5]]]]).repeat(channels, 1, 1, 1)
    output1 = F.conv2d(image, filter1, stride=2, groups=channels)
    output2 = F.conv2d(image, filter2, stride=2, groups=channels)
    return output1, output2


def _calculate_local_variance(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate local patch variance.

    Args:
        image: Input tensor with shape ``[B, C, H, W]``.
        patch_size: Local patch size.

    Returns:
        Local variance tensor.
    """
    padding = patch_size // 2
    image_mean = F.avg_pool2d(image, kernel_size=patch_size, stride=1, padding=padding)
    image_pad = F.pad(image, (padding, padding, padding, padding), mode="constant", value=0)
    mean_pad = F.pad(image_mean, (padding, padding, padding, padding), mode="constant", value=0)

    image_patches = image_pad.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean_patches = mean_pad.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    return torch.mean((image_patches - mean_patches) ** 2, dim=(-1, -2))


class ZeroIGLocalMean(nn.Module):
    """Local mean filter used by ZERO-IG interaction loss."""

    def __init__(self, patch_size: int = 5) -> None:
        """Initialize local mean filter.

        Args:
            patch_size: Local patch size.
        """
        super().__init__()
        self.patch_size = patch_size
        self.padding = patch_size // 2

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute local patch mean.

        Args:
            image: Input tensor with shape ``[B, C, H, W]``.

        Returns:
            Local mean tensor.
        """
        image = F.pad(
            image,
            (self.padding, self.padding, self.padding, self.padding),
            mode="reflect",
        )
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        return patches.mean(dim=(-1, -2))


class ZeroIGTVLoss(nn.Module):
    """Total variation loss used by ZERO-IG."""

    def __init__(self, loss_weight: float = 1.0) -> None:
        """Initialize total variation loss.

        Args:
            loss_weight: Multiplicative loss weight.
        """
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss.

        Args:
            image: Tensor with shape ``[B, C, H, W]``.

        Returns:
            Scalar total variation loss tensor.
        """
        batch_size = image.size(0)
        h_x = image.size(2)
        w_x = image.size(3)
        count_h = max(1, (h_x - 1) * w_x)
        count_w = max(1, h_x * (w_x - 1))
        h_tv = torch.pow(image[:, :, 1:, :] - image[:, :, : h_x - 1, :], 2).sum()
        w_tv = torch.pow(image[:, :, :, 1:] - image[:, :, :, : w_x - 1], 2).sum()
        return self.loss_weight * 2.0 * (h_tv / count_h + w_tv / count_w) / batch_size


class ZeroIGSmoothLoss(nn.Module):
    """Illumination smoothness loss from the official ZERO-IG code."""

    _OFFSETS = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (2, 0),
        (-2, 0),
        (0, 2),
        (0, -2),
        (2, 1),
        (-2, -1),
        (-2, 1),
        (2, -1),
        (1, 2),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (2, 2),
        (-2, -2),
        (-2, 2),
        (2, -2),
    )

    def __init__(self, sigma: float = 10.0) -> None:
        """Initialize ZERO-IG smoothness loss.

        Args:
            sigma: Color sensitivity for neighborhood weights.
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, input_image: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        """Compute weighted illumination smoothness loss.

        Args:
            input_image: Low-light RGB input tensor.
            illumination: Estimated illumination tensor.

        Returns:
            Scalar smoothness loss tensor.
        """
        ycbcr = self._rgb_to_ycbcr(input_image)
        sigma_color = -1.0 / (2.0 * self.sigma * self.sigma)
        loss = illumination.new_tensor(0.0)

        for dy, dx in self._OFFSETS:
            input_a, input_b = self._shift_pair(ycbcr, dy, dx)
            out_a, out_b = self._shift_pair(illumination, dy, dx)
            weight = torch.exp(torch.sum((input_a - input_b) ** 2, dim=1, keepdim=True) * sigma_color)
            loss = loss + torch.mean(weight * torch.norm(out_a - out_b, p=1, dim=1, keepdim=True))

        return loss

    @staticmethod
    def _rgb_to_ycbcr(input_image: torch.Tensor) -> torch.Tensor:
        """Convert RGB tensor to YCbCr.

        Args:
            input_image: RGB image tensor with shape ``[B, 3, H, W]``.

        Returns:
            YCbCr tensor with shape ``[B, 3, H, W]``.

        Raises:
            ValueError: If ``input_image`` is not an RGB NCHW tensor.
        """
        if input_image.dim() != 4 or input_image.size(1) != 3:
            raise ValueError(f"Expected RGB NCHW tensor, got {tuple(input_image.shape)}.")

        r = input_image[:, 0:1, :, :]
        g = input_image[:, 1:2, :, :]
        b = input_image[:, 2:3, :, :]
        y = 0.257 * r + 0.564 * g + 0.098 * b + 16.0 / 255.0
        cb = -0.148 * r - 0.291 * g + 0.439 * b + 128.0 / 255.0
        cr = 0.439 * r - 0.368 * g - 0.071 * b + 128.0 / 255.0
        return torch.cat((y, cb, cr), dim=1)

    @staticmethod
    def _shift_pair(image: torch.Tensor, dy: int, dx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create shifted tensor pairs for finite-difference penalties.

        Args:
            image: Input tensor with shape ``[B, C, H, W]``.
            dy: Vertical offset.
            dx: Horizontal offset.

        Returns:
            A tuple containing aligned source and shifted target tensors.
        """
        if dy >= 0:
            y_src = slice(dy, None)
            y_dst = slice(None, -dy if dy else None)
        else:
            y_src = slice(None, dy)
            y_dst = slice(-dy, None)

        if dx >= 0:
            x_src = slice(dx, None)
            x_dst = slice(None, -dx if dx else None)
        else:
            x_src = slice(None, dx)
            x_dst = slice(-dx, None)

        return image[:, :, y_src, x_src], image[:, :, y_dst, x_dst]


class ZeroIG_Loss(BaseLoss):
    """ZERO-IG zero-shot self-supervised loss.

    This is a direct openLLV adaptation of the official ZERO-IG LossFunction:
    LD-Net residual/consistency losses, IE-Net illumination losses, and RD-Net
    residual, consistency, illumination, interaction, variance, and color losses.
    It does not require paired normal-light targets.
    """

    name = "zeroig"
    aliases = ["zeroig_loss", "zero_ig", "zero-ig", "ZeroIG_Loss"]
    requires_target = False

    def __init__(
        self,
        *,
        enhance_brightness_weight: float = 700.0,
        enhance_normalized_weight: float = 1000.0,
        smooth_weight: float = 5.0,
        tv_weight: float = 1600.0,
        ld_res_weight: float = 1000.0,
        ld_cons_weight: float = 1000.0,
        rd_res_weight: float = 1000.0,
        rd_cons_weight: float = 1000.0,
        color_weight: float = 10000.0,
        illumination_weight: float = 1000.0,
        interaction_weight: float = 10000.0,
        variance_weight: float = 1000.0,
        target_luminance: float = 0.5,
        max_enhancement_factor: float = 25.0,
        eps: float = 1e-9,
    ) -> None:
        """Initialize ZERO-IG loss.

        Args:
            enhance_brightness_weight: Weight for brightness enhancement loss.
            enhance_normalized_weight: Weight for normalized low-light loss.
            smooth_weight: Weight for illumination smoothness loss.
            tv_weight: Weight for illumination total variation loss.
            ld_res_weight: Weight for LD-Net residual loss.
            ld_cons_weight: Weight for LD-Net consistency loss.
            rd_res_weight: Weight for RD-Net residual loss.
            rd_cons_weight: Weight for RD-Net consistency loss.
            color_weight: Weight for color consistency loss.
            illumination_weight: Weight for illumination consistency loss.
            interaction_weight: Weight for local interaction loss.
            variance_weight: Weight for noise variance loss.
            target_luminance: Target luminance used by illumination estimation.
            max_enhancement_factor: Maximum enhancement factor.
            eps: Small constant used for numerical stability.
        """
        super().__init__()
        self.enhance_brightness_weight = float(enhance_brightness_weight)
        self.enhance_normalized_weight = float(enhance_normalized_weight)
        self.smooth_weight = float(smooth_weight)
        self.tv_weight = float(tv_weight)
        self.ld_res_weight = float(ld_res_weight)
        self.ld_cons_weight = float(ld_cons_weight)
        self.rd_res_weight = float(rd_res_weight)
        self.rd_cons_weight = float(rd_cons_weight)
        self.color_weight = float(color_weight)
        self.illumination_weight = float(illumination_weight)
        self.interaction_weight = float(interaction_weight)
        self.variance_weight = float(variance_weight)
        self.target_luminance = float(target_luminance)
        self.max_enhancement_factor = float(max_enhancement_factor)
        self.eps = float(eps)

        self.mse = nn.MSELoss()
        self.smooth_loss = ZeroIGSmoothLoss()
        self.local_mean = ZeroIGLocalMean(patch_size=5)
        self.tv_loss = ZeroIGTVLoss()

    def forward(self, input_tensor: torch.Tensor, zeroig_result: Any) -> torch.Tensor:
        """Compute ZERO-IG zero-shot training loss.

        Args:
            input_tensor: Low-light input tensor.
            zeroig_result: ZERO-IG model output containing the 21 training
                tensors required by the official loss.

        Returns:
            Scalar ZERO-IG loss tensor.
        """
        outputs = self._extract_outputs(zeroig_result)
        (
            L_pred1,
            L_pred2,
            L2,
            s2,
            s21,
            s22,
            H2,
            H11,
            H12,
            H13,
            s13,
            H14,
            s14,
            H3,
            s3,
            H3_pred,
            H4_pred,
            L_pred1_L_pred2_diff,
            H3_denoised1_H3_denoised2_diff,
            H2_blur,
            H3_blur,
        ) = outputs

        input_tensor = input_tensor + self.eps
        loss = input_tensor.new_tensor(0.0)

        loss = loss + self._illumination_estimation_loss(L2, s2)
        loss = loss + self._ld_denoising_loss(input_tensor, L_pred1, L_pred2, L2)
        loss = loss + self._rd_denoising_loss(
            s21=s21,
            s22=s22,
            H2=H2,
            H11=H11,
            H12=H12,
            H3=H3,
            s3=s3,
            H3_pred=H3_pred,
            H4_pred=H4_pred,
            H3_denoised1_H3_denoised2_diff=H3_denoised1_H3_denoised2_diff,
            H2_blur=H2_blur,
            H3_blur=H3_blur,
            s2=s2,
        )

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
        """Compute ZERO-IG loss through the Trainer interface.

        Args:
            input_tensor: Low-light input tensor.
            model_output: Raw ZERO-IG model output.
            target: Optional target tensor. It is unused because ZERO-IG is
                reference-free.
            extract_prediction: Optional prediction extractor. It is unused
                because prediction is read directly from dict output when
                available.
            align_prediction: Optional alignment callback. It is unused here.

        Returns:
            A tuple containing the scalar loss tensor and optional prediction.
        """
        loss = self(input_tensor, model_output)
        prediction = None
        if isinstance(model_output, dict) and torch.is_tensor(model_output.get("pred")):
            prediction = model_output["pred"]
        return loss, prediction

    def _illumination_estimation_loss(self, L2: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """Compute ZERO-IG illumination-estimation loss.

        Args:
            L2: Low-light layer tensor.
            s2: Estimated illumination tensor.

        Returns:
            Scalar illumination-estimation loss tensor.
        """
        input_y = 0.299 * L2.detach()[:, 2, :, :] + 0.587 * L2.detach()[:, 1, :, :] + 0.144 * L2.detach()[:, 0, :, :]
        y_mean = torch.mean(input_y, dim=(1, 2)).clamp_min(self.eps)
        enhancement_factor = self.target_luminance / y_mean
        enhancement_factor = enhancement_factor.view(-1, 1, 1, 1).clamp(1.0, self.max_enhancement_factor)

        adjustment_ratio = torch.pow(L2.new_tensor(0.7), -enhancement_factor) / enhancement_factor
        adjustment_ratio = adjustment_ratio.repeat(1, 3, 1, 1)

        normalized_low_light_layer = torch.clamp(L2.detach() / s2, self.eps, 0.8)
        enhanced_brightness = torch.pow(L2.detach() * enhancement_factor, enhancement_factor)
        clamped_enhanced_brightness = torch.clamp(enhanced_brightness * adjustment_ratio, self.eps, 1.0)
        clamped_adjusted_low_light = torch.clamp(L2.detach() * enhancement_factor, self.eps, 1.0)

        loss = self.mse(s2, clamped_enhanced_brightness) * self.enhance_brightness_weight
        loss = loss + self.mse(normalized_low_light_layer, clamped_adjusted_low_light) * self.enhance_normalized_weight
        loss = loss + self.smooth_loss(L2.detach(), s2) * self.smooth_weight
        loss = loss + self.tv_loss(s2) * self.tv_weight
        return loss

    def _ld_denoising_loss(
        self,
        input_tensor: torch.Tensor,
        L_pred1: torch.Tensor,
        L_pred2: torch.Tensor,
        L2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LD-Net denoising loss.

        Args:
            input_tensor: Low-light input tensor.
            L_pred1: First low-light denoising prediction.
            L_pred2: Second low-light denoising prediction.
            L2: Low-light layer tensor.

        Returns:
            Scalar LD-Net loss tensor.
        """
        L11, L12 = _pair_downsampler(input_tensor)
        denoised1, denoised2 = _pair_downsampler(L2)

        loss = self.mse(L11, L_pred2) * self.ld_res_weight
        loss = loss + self.mse(L12, L_pred1) * self.ld_res_weight
        loss = loss + self.mse(L_pred1, denoised1) * self.ld_cons_weight
        loss = loss + self.mse(L_pred2, denoised2) * self.ld_cons_weight
        return loss

    def _rd_denoising_loss(
        self,
        *,
        s21: torch.Tensor,
        s22: torch.Tensor,
        H2: torch.Tensor,
        H11: torch.Tensor,
        H12: torch.Tensor,
        H3: torch.Tensor,
        s3: torch.Tensor,
        H3_pred: torch.Tensor,
        H4_pred: torch.Tensor,
        H3_denoised1_H3_denoised2_diff: torch.Tensor,
        H2_blur: torch.Tensor,
        H3_blur: torch.Tensor,
        s2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RD-Net denoising and consistency loss.

        Args:
            s21: First downsampled illumination tensor.
            s22: Second downsampled illumination tensor.
            H2: Intermediate high-light tensor.
            H11: First downsampled high-light tensor.
            H12: Second downsampled high-light tensor.
            H3: Denoised high-light tensor.
            s3: Refined illumination tensor.
            H3_pred: First RD-Net prediction.
            H4_pred: Second RD-Net prediction.
            H3_denoised1_H3_denoised2_diff: Local interaction weight tensor.
            H2_blur: Blurred intermediate high-light tensor.
            H3_blur: Blurred denoised high-light tensor.
            s2: Estimated illumination tensor.

        Returns:
            Scalar RD-Net loss tensor.
        """
        H3_denoised1, H3_denoised2 = _pair_downsampler(H3)

        loss = self.mse(H3_pred, torch.cat([H12.detach(), s22.detach()], dim=1)) * self.rd_res_weight
        loss = loss + self.mse(H4_pred, torch.cat([H11.detach(), s21.detach()], dim=1)) * self.rd_res_weight
        loss = loss + self.mse(H3_pred[:, 0:3, :, :], H3_denoised1) * self.rd_cons_weight
        loss = loss + self.mse(H4_pred[:, 0:3, :, :], H3_denoised2) * self.rd_cons_weight

        loss = loss + self.mse(H2_blur.detach(), H3_blur) * self.color_weight
        loss = loss + self.mse(s2.detach(), s3) * self.illumination_weight

        local_mean1 = self.local_mean(H3_denoised1)
        local_mean2 = self.local_mean(H3_denoised2)
        weighted_diff1 = (
            (1.0 - H3_denoised1_H3_denoised2_diff) * local_mean1
            + H3_denoised1 * H3_denoised1_H3_denoised2_diff
        )
        weighted_diff2 = (
            (1.0 - H3_denoised1_H3_denoised2_diff) * local_mean2
            + H3_denoised1 * H3_denoised1_H3_denoised2_diff
        )
        loss = loss + self.mse(H3_denoised1, weighted_diff1) * self.interaction_weight
        loss = loss + self.mse(H3_denoised2, weighted_diff2) * self.interaction_weight

        noise_var = _calculate_local_variance(H3 - H2)
        H2_var = _calculate_local_variance(H2)
        loss = loss + self.mse(H2_var, noise_var) * self.variance_weight
        return loss

    @staticmethod
    def _extract_outputs(zeroig_result: Any) -> Tuple[torch.Tensor, ...]:
        """Extract ZERO-IG training tensors.

        Args:
            zeroig_result: Raw ZERO-IG model output.

        Returns:
            Tuple containing the first 21 tensors required by the loss.

        Raises:
            TypeError: If the output structure is unsupported or incomplete.
        """
        if isinstance(zeroig_result, dict):
            loss_inputs = get_loss_inputs(zeroig_result)
            if isinstance(loss_inputs, dict) and "outputs" in loss_inputs:
                zeroig_result = loss_inputs["outputs"]
            elif "outputs" in zeroig_result:
                zeroig_result = zeroig_result["outputs"]

        if not isinstance(zeroig_result, (tuple, list)) or len(zeroig_result) < 21:
            raise TypeError(
                "ZeroIG_Loss expects ZeroIG training outputs with 21 tensors. "
                f"Got {type(zeroig_result).__name__}."
            )

        return tuple(zeroig_result[:21])


ZeroIGLoss = ZeroIG_Loss
