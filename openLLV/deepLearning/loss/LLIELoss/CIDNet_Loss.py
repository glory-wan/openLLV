"""Loss functions for HVI-CIDNet."""

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


def _rgb_to_hvi(image: torch.Tensor, density_k: torch.Tensor) -> torch.Tensor:
    """Loss-side HVI conversion using the model's learned density value."""
    eps = 1e-8
    value, max_indices = image.max(dim=1)
    image_min = image.min(dim=1).values
    delta = value - image_min
    red, green, blue = image[:, 0], image[:, 1], image[:, 2]
    safe_delta = delta + eps
    hue = torch.where(
        max_indices == 0,
        torch.remainder((green - blue) / safe_delta, 6.0),
        torch.where(
            max_indices == 1,
            2.0 + (blue - red) / safe_delta,
            4.0 + (red - green) / safe_delta,
        ),
    )
    hue = torch.where(delta == 0, torch.zeros_like(hue), hue).unsqueeze(1) / 6.0
    saturation = delta / (value + eps)
    saturation = torch.where(value == 0, torch.zeros_like(saturation), saturation).unsqueeze(1)
    intensity = value.unsqueeze(1)
    color_sensitive = torch.pow(
        torch.sin(intensity * (0.5 * math.pi)) + eps,
        density_k.to(device=image.device, dtype=image.dtype),
    )
    angle = 2.0 * math.pi * hue
    return torch.cat(
        (
            color_sensitive * saturation * torch.cos(angle),
            color_sensitive * saturation * torch.sin(angle),
            intensity,
        ),
        dim=1,
    )


class CIDNetSSIMLoss(nn.Module):
    """Differentiable ``1 - SSIM`` loss used by the official training code."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        coordinates = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gaussian = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
        gaussian = gaussian / gaussian.sum()
        window = torch.outer(gaussian, gaussian).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", window)
        self.window_size = int(window_size)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = prediction.shape[1]
        window = self.window.to(device=prediction.device, dtype=prediction.dtype).repeat(channels, 1, 1, 1)
        padding = self.window_size // 2
        mean_prediction = F.conv2d(prediction, window, padding=padding, groups=channels)
        mean_target = F.conv2d(target, window, padding=padding, groups=channels)
        mean_prediction_sq = mean_prediction.square()
        mean_target_sq = mean_target.square()
        mean_product = mean_prediction * mean_target
        variance_prediction = (
            F.conv2d(prediction.square(), window, padding=padding, groups=channels)
            - mean_prediction_sq
        )
        variance_target = (
            F.conv2d(target.square(), window, padding=padding, groups=channels)
            - mean_target_sq
        )
        covariance = (
            F.conv2d(prediction * target, window, padding=padding, groups=channels)
            - mean_product
        )
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2.0 * mean_product + c1) * (2.0 * covariance + c2)) / (
            (mean_prediction_sq + mean_target_sq + c1)
            * (variance_prediction + variance_target + c2)
        )
        return 1.0 - ssim_map.mean()


class CIDNetEdgeLoss(nn.Module):
    """Laplacian-pyramid edge loss from the official CIDNet objective."""

    def __init__(self) -> None:
        super().__init__()
        kernel_1d = torch.tensor([[0.05, 0.25, 0.4, 0.25, 0.05]], dtype=torch.float32)
        self.register_buffer("kernel", torch.outer(kernel_1d.flatten(), kernel_1d.flatten())[None, None])

    def _gaussian(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.shape[1]
        kernel = self.kernel.to(device=image.device, dtype=image.dtype).repeat(channels, 1, 1, 1)
        image = F.pad(image, (2, 2, 2, 2), mode="replicate")
        return F.conv2d(image, kernel, groups=channels)

    def _laplacian(self, image: torch.Tensor) -> torch.Tensor:
        filtered = self._gaussian(image)
        downsampled = filtered[:, :, ::2, ::2]
        upsampled = torch.zeros_like(filtered)
        upsampled[:, :, ::2, ::2] = downsampled * 4.0
        return image - self._gaussian(upsampled)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self._laplacian(prediction), self._laplacian(target))


class CIDNetPerceptualLoss(nn.Module):
    """VGG19 feature MSE used by CIDNet at four pre-activation layers."""

    DEFAULT_LAYER_WEIGHTS = {
        "conv1_2": 1.0,
        "conv2_2": 1.0,
        "conv3_4": 1.0,
        "conv4_4": 1.0,
    }
    LAYER_TO_INDEX = {
        "conv1_2": 2,
        "conv2_2": 7,
        "conv3_4": 16,
        "conv4_4": 25,
    }

    def __init__(
        self,
        layer_weights: Optional[Dict[str, float]] = None,
        pretrained: bool = True,
        use_input_norm: bool = True,
        range_norm: bool = True,
    ) -> None:
        super().__init__()
        self.layer_weights = dict(layer_weights or self.DEFAULT_LAYER_WEIGHTS)
        unknown = set(self.layer_weights) - set(self.LAYER_TO_INDEX)
        if unknown:
            raise ValueError("Unsupported VGG layer(s): {}".format(sorted(unknown)))
        self.use_input_norm = bool(use_input_norm)
        self.range_norm = bool(range_norm)
        max_index = max(self.LAYER_TO_INDEX[name] for name in self.layer_weights)
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        self.vgg = models.vgg19(weights=weights).features[: max_index + 1].eval()
        for module in self.vgg.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        for parameter in self.vgg.parameters():
            parameter.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if self.range_norm:
            image = (image + 1.0) / 2.0
        if self.use_input_norm:
            image = (image - self.mean) / self.std
        return image

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = self._preprocess(prediction)
        target = self._preprocess(target.detach())
        index_to_name = {index: name for name, index in self.LAYER_TO_INDEX.items()}
        loss = prediction.new_tensor(0.0)
        for index, layer in enumerate(self.vgg):
            prediction = layer(prediction)
            target = layer(target)
            name = index_to_name.get(index)
            if name in self.layer_weights:
                loss = loss + F.mse_loss(prediction, target) * self.layer_weights[name]
        return loss


class CIDNet_Loss(BaseLoss):
    """Official CIDNet objective evaluated jointly in RGB and HVI spaces.

    ``L = L_rgb + hvi_weight * L_hvi``, where each domain combines L1,
    SSIM, Laplacian edge, and optional VGG19 perceptual losses.
    """

    name = "cidnet"
    aliases = ["cidnet_loss", "HVI-CIDNet-Loss"]
    requires_target = True

    def __init__(
        self,
        pixel_weight: float = 1.0,
        ssim_weight: float = 0.5,
        edge_weight: float = 50.0,
        perceptual_weight: float = 0.01,
        hvi_weight: float = 1.0,
        use_perceptual: bool = True,
        pretrained_vgg: bool = True,
        perceptual_layer_weights: Optional[Dict[str, float]] = None,
        perceptual_range_norm: bool = True,
        use_input_norm: bool = True,
        density_k: float = 0.2,
    ) -> None:
        super().__init__()
        self.pixel_weight = float(pixel_weight)
        self.ssim_weight = float(ssim_weight)
        self.edge_weight = float(edge_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.hvi_weight = float(hvi_weight)
        self.ssim_loss = CIDNetSSIMLoss()
        self.edge_loss = CIDNetEdgeLoss()
        self.perceptual_loss = None
        if use_perceptual and self.perceptual_weight > 0:
            self.perceptual_loss = CIDNetPerceptualLoss(
                layer_weights=perceptual_layer_weights,
                pretrained=pretrained_vgg,
                range_norm=perceptual_range_norm,
                use_input_norm=use_input_norm,
            )
        self.register_buffer("default_density_k", torch.tensor([float(density_k)]))

    def _domain_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.pixel_weight * F.l1_loss(prediction, target)
        if self.ssim_weight:
            loss = loss + self.ssim_weight * self.ssim_loss(prediction, target)
        if self.edge_weight:
            loss = loss + self.edge_weight * self.edge_loss(prediction, target)
        if self.perceptual_loss is not None:
            loss = loss + self.perceptual_weight * self.perceptual_loss(prediction, target)
        return loss

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        prediction_hvi: Optional[torch.Tensor] = None,
        density_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        density_k = self.default_density_k if density_k is None else density_k
        loss = self._domain_loss(prediction, target)
        if self.hvi_weight:
            if prediction_hvi is None or prediction_hvi.shape[-2:] != prediction.shape[-2:]:
                prediction_hvi = _rgb_to_hvi(prediction, density_k)
            target_hvi = _rgb_to_hvi(target, density_k)
            loss = loss + self.hvi_weight * self._domain_loss(prediction_hvi, target_hvi)
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
        if target is None:
            raise ValueError("CIDNet_Loss requires a paired target tensor.")
        if extract_prediction is None:
            if not torch.is_tensor(model_output):
                raise TypeError("extract_prediction is required for structured CIDNet output.")
            prediction = model_output
        else:
            prediction = extract_prediction(model_output, target)
        if align_prediction is not None:
            prediction = align_prediction(prediction, target)

        loss_inputs = get_loss_inputs(model_output)
        prediction_hvi = loss_inputs.get("prediction_hvi") if isinstance(loss_inputs, dict) else None
        density_k = loss_inputs.get("density_k") if isinstance(loss_inputs, dict) else None
        return self(
            prediction,
            target,
            prediction_hvi=prediction_hvi,
            density_k=density_k,
        ), prediction


CIDNetLoss = CIDNet_Loss
