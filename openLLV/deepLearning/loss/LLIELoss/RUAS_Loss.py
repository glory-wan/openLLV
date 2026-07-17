"""RUAS reference-free loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class RUAS_Loss(BaseLoss):
    """Reference-free RUAS training loss.

    The official RUAS implementation optimizes the illumination estimation
    module with a fidelity term plus RTV-style smoothness, and optimizes the
    denoise module with a tiny fidelity term plus total variation.
    """
    name = "ruas"
    aliases = []
    requires_target = False

    def __init__(
            self,
            enhance_fidelity_weight: float = 0.5,
            denoise_fidelity_weight: float = 1e-7,
            denoise_weight: float = 1.0,
            tv_weight: float = 1.0,
            sigma: float = 0.1,
    ):
        """Initialize RUAS loss.

        Args:
            enhance_fidelity_weight: Weight for illumination fidelity loss.
            denoise_fidelity_weight: Weight for denoising fidelity loss.
            denoise_weight: Weight for the denoising loss branch.
            tv_weight: Weight for denoising total variation loss.
            sigma: Color sensitivity used by the RTV-style smoothness term.
        """
        super(RUAS_Loss, self).__init__()
        self.enhance_loss = RUASEnhanceLoss(
            fidelity_weight=enhance_fidelity_weight,
            sigma=sigma,
        )
        self.denoise_loss = RUASDenoiseLoss(
            fidelity_weight=denoise_fidelity_weight,
            tv_weight=tv_weight,
        )
        self.denoise_weight = denoise_weight

    def forward(self, img_lowlight, ruas_result):
        """Compute RUAS training loss.

        Args:
            img_lowlight: Low-light input tensor.
            ruas_result: RUAS model output containing enhanced and illumination
                sequences.

        Returns:
            Scalar reference-free RUAS loss tensor.
        """
        u_list, t_list = _unpack_ruas_output(ruas_result)

        loss_enhance = self.enhance_loss(img_lowlight, u_list, t_list)
        loss_denoise = img_lowlight.new_tensor(0.0)
        if len(u_list) >= 2:
            loss_denoise = self.denoise_loss(u_list[-1], u_list[-2])

        return loss_enhance + self.denoise_weight * loss_denoise


class RUASEnhanceLoss(nn.Module):
    """IEM loss: 0.5 * MSE(t_K, y) + RTV(t_K)."""

    def __init__(self, fidelity_weight: float = 0.5, sigma: float = 0.1):
        """Initialize RUAS enhancement loss.

        Args:
            fidelity_weight: Weight for the illumination fidelity term.
            sigma: Color sensitivity for the smoothness term.
        """
        super(RUASEnhanceLoss, self).__init__()
        self.fidelity_weight = fidelity_weight
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss(sigma=sigma)

    def forward(self, input_img, u_list, t_list=None):
        """Compute RUAS illumination-estimation loss.

        Args:
            input_img: Low-light input tensor.
            u_list: Enhanced image sequence, output tensor, or structured RUAS
                output when ``t_list`` is omitted.
            t_list: Optional illumination sequence.

        Returns:
            Scalar enhancement loss tensor.

        Raises:
            ValueError: If no illumination tensor can be extracted.
        """
        if t_list is None:
            if torch.is_tensor(u_list):
                illumination = u_list
            else:
                _, t_list = _unpack_ruas_output(u_list)
                if not t_list:
                    raise ValueError("RUASEnhanceLoss requires a non-empty t_list.")
                illumination = t_list[-1]
        elif not t_list:
            raise ValueError("RUASEnhanceLoss requires a non-empty t_list.")
        else:
            illumination = t_list[-1]

        fidelity_loss = self.l2_loss(illumination, input_img)
        smooth_loss = self.smooth_loss(input_img, illumination)
        return self.fidelity_weight * fidelity_loss + smooth_loss


class RUASDenoiseLoss(nn.Module):
    """NRM loss: epsilon * MSE(denoised, enhanced) + TV(denoised)."""

    def __init__(self, fidelity_weight: float = 1e-7, tv_weight: float = 1.0):
        """Initialize RUAS denoising loss.

        Args:
            fidelity_weight: Weight for denoising fidelity loss.
            tv_weight: Weight for total variation loss.
        """
        super(RUASDenoiseLoss, self).__init__()
        self.fidelity_weight = fidelity_weight
        self.l2_loss = nn.MSELoss()
        self.tv_loss = TVLoss(TVLoss_weight=tv_weight)

    def forward(self, output, target):
        """Compute RUAS denoising loss.

        Args:
            output: Denoised output tensor.
            target: Enhanced target tensor.

        Returns:
            Scalar denoising loss tensor.
        """
        return self.fidelity_weight * self.l2_loss(output, target) + self.tv_loss(output)


class TVLoss(nn.Module):
    """Total variation loss for RUAS tensors."""

    def __init__(self, TVLoss_weight=1):
        """Initialize total variation loss.

        Args:
            TVLoss_weight: Multiplicative loss weight.
        """
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        """Compute total variation loss.

        Args:
            x: Tensor with shape ``[B, C, H, W]``.

        Returns:
            Scalar total variation loss tensor.

        Raises:
            ValueError: If ``x`` is not a 4D tensor.
        """
        if x.dim() != 4:
            raise ValueError(f"TVLoss expects a 4D NCHW tensor, got shape {tuple(x.shape)}.")

        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = x[:, :, 1:, :].numel() / batch_size
        count_w = x[:, :, :, 1:].numel() / batch_size

        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class SmoothLoss(nn.Module):
    """RTV-style smoothness term used by RUAS.

    This is the batched/device-safe version of the official implementation's
    24-neighborhood weighted gradient penalty.
    """

    def __init__(self, sigma: float = 0.1):
        """Initialize RUAS smoothness loss.

        Args:
            sigma: Color sensitivity for neighborhood weights.
        """
        super(SmoothLoss, self).__init__()
        self.sigma = sigma

    def forward(self, input_img, output):
        """Compute RTV-style weighted smoothness loss.

        Args:
            input_img: Low-light RGB input tensor.
            output: Illumination or enhanced output tensor to regularize.

        Returns:
            Scalar smoothness loss tensor.
        """
        input_ycbcr = self.rgb2ycbcr(input_img)
        sigma_color = -1.0 / (2.0 * self.sigma * self.sigma)

        loss = output.new_tensor(0.0)
        for dy, dx in _RUAS_OFFSETS:
            input_a, input_b = _shift_pair(input_ycbcr, dy, dx)
            output_a, output_b = _shift_pair(output, dy, dx)

            weight = torch.exp(
                torch.sum(torch.pow(input_a - input_b, 2), dim=1, keepdim=True) * sigma_color
            )
            pixel_grad = weight * torch.norm(output_a - output_b, p=1, dim=1, keepdim=True)
            loss = loss + torch.mean(pixel_grad)

        return loss

    @staticmethod
    def rgb2ycbcr(input_img):
        """Convert RGB tensor to YCbCr.

        Args:
            input_img: RGB image tensor with shape ``[B, 3, H, W]``.

        Returns:
            YCbCr tensor with shape ``[B, 3, H, W]``.

        Raises:
            ValueError: If ``input_img`` is not an RGB NCHW tensor.
        """
        if input_img.dim() != 4 or input_img.size(1) != 3:
            raise ValueError(
                f"SmoothLoss expects an RGB NCHW tensor, got shape {tuple(input_img.shape)}."
            )

        r = input_img[:, 0:1, :, :]
        g = input_img[:, 1:2, :, :]
        b = input_img[:, 2:3, :, :]

        y = 0.257 * r + 0.564 * g + 0.098 * b + 16.0 / 255.0
        cb = -0.148 * r - 0.291 * g + 0.439 * b + 128.0 / 255.0
        cr = 0.439 * r - 0.368 * g - 0.071 * b + 128.0 / 255.0
        return torch.cat((y, cb, cr), dim=1)


_RUAS_OFFSETS = (
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


def _shift_pair(x, dy: int, dx: int):
    """Create shifted tensor pairs for finite-difference penalties.

    Args:
        x: Input tensor with shape ``[B, C, H, W]``.
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

    return x[:, :, y_src, x_src], x[:, :, y_dst, x_dst]


def _unpack_ruas_output(ruas_result):
    """Extract RUAS enhanced and illumination sequences.

    Args:
        ruas_result: RUAS model output as a tuple/list or dictionary.

    Returns:
        A tuple ``(u_list, t_list)``.

    Raises:
        TypeError: If the output structure is unsupported.
    """
    if isinstance(ruas_result, dict):
        loss_inputs = get_loss_inputs(ruas_result)
        if isinstance(loss_inputs, dict) and "u_list" in loss_inputs and "t_list" in loss_inputs:
            return loss_inputs["u_list"], loss_inputs["t_list"]
        if "u_list" in ruas_result and "t_list" in ruas_result:
            return ruas_result["u_list"], ruas_result["t_list"]
        if "enhanced_list" in ruas_result and "illumination_list" in ruas_result:
            return ruas_result["enhanced_list"], ruas_result["illumination_list"]

    if isinstance(ruas_result, (tuple, list)) and len(ruas_result) == 2:
        return ruas_result[0], ruas_result[1]

    raise TypeError(
        "RUAS_Loss expects RUAS output as (u_list, t_list) or a dict with "
        "'u_list'/'t_list'."
    )


LossFunction = RUASEnhanceLoss
DenoiseLossFunction = RUASDenoiseLoss
