"""HVI-CIDNet low-light image enhancement model.

Original paper: HVI: A New Color Space for Low-light Image Enhancement
Paper link: https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.pdf
Official source code: https://github.com/Fediory/HVI-CIDNet
"""


import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseModel import LLVModel


def rgb_to_hvi(image: torch.Tensor, density_k: torch.Tensor) -> torch.Tensor:
    """Convert an RGB tensor in ``[0, 1]`` to the HVI color space."""
    eps = 1e-8
    value, max_indices = image.max(dim=1)
    image_min = image.min(dim=1).values
    delta = value - image_min

    red, green, blue = image[:, 0], image[:, 1], image[:, 2]
    safe_delta = delta + eps
    hue_red = torch.remainder((green - blue) / safe_delta, 6.0)
    hue_green = 2.0 + (blue - red) / safe_delta
    hue_blue = 4.0 + (red - green) / safe_delta
    hue = torch.where(
        max_indices == 0,
        hue_red,
        torch.where(max_indices == 1, hue_green, hue_blue),
    )
    hue = torch.where(delta == 0, torch.zeros_like(hue), hue) / 6.0

    saturation = delta / (value + eps)
    saturation = torch.where(value == 0, torch.zeros_like(saturation), saturation)

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    intensity = value.unsqueeze(1)
    color_sensitive = torch.pow(
        torch.sin(intensity * (0.5 * math.pi)) + eps,
        density_k.to(device=image.device, dtype=image.dtype),
    )
    angle = 2.0 * math.pi * hue
    horizontal = color_sensitive * saturation * torch.cos(angle)
    vertical = color_sensitive * saturation * torch.sin(angle)
    return torch.cat((horizontal, vertical, intensity), dim=1)


def hvi_to_rgb(
    image: torch.Tensor,
    density_k: torch.Tensor,
    saturation_scale: float = 1.0,
    intensity_scale: float = 1.0,
) -> torch.Tensor:
    """Convert an HVI tensor to RGB."""
    eps = 1e-8
    horizontal = image[:, 0].clamp(-1.0, 1.0)
    vertical = image[:, 1].clamp(-1.0, 1.0)
    value = image[:, 2].clamp(0.0, 1.0)

    color_sensitive = torch.pow(
        torch.sin(value * (0.5 * math.pi)) + eps,
        density_k.detach().to(device=image.device, dtype=image.dtype),
    )
    horizontal = (horizontal / (color_sensitive + eps)).clamp(-1.0, 1.0)
    vertical = (vertical / (color_sensitive + eps)).clamp(-1.0, 1.0)
    hue = torch.remainder(torch.atan2(vertical + eps, horizontal + eps) / (2.0 * math.pi), 1.0)
    saturation = torch.sqrt(horizontal.square() + vertical.square() + eps)
    saturation = (saturation * float(saturation_scale)).clamp(0.0, 1.0)

    sector = torch.floor(hue * 6.0).to(torch.long).remainder(6)
    fraction = hue * 6.0 - sector.to(hue.dtype)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)

    red = torch.where(
        (sector == 0) | (sector == 5),
        value,
        torch.where((sector == 1), q, torch.where((sector == 4), t, p)),
    )
    green = torch.where(
        (sector == 1) | (sector == 2),
        value,
        torch.where((sector == 0), t, torch.where((sector == 3), q, p)),
    )
    blue = torch.where(
        (sector == 3) | (sector == 4),
        value,
        torch.where((sector == 2), t, torch.where((sector == 5), q, p)),
    )
    return torch.stack((red, green, blue), dim=1) * float(intensity_scale)


class HVITransform(nn.Module):
    """Learnable RGB/HVI conversion used by CIDNet."""

    def __init__(self, density_k: float = 0.2) -> None:
        super().__init__()
        self.density_k = nn.Parameter(torch.tensor([float(density_k)]))

    def HVIT(self, image: torch.Tensor) -> torch.Tensor:
        """Convert RGB to HVI using the learned density parameter."""
        return rgb_to_hvi(image, self.density_k)

    def PHVIT(
        self,
        image: torch.Tensor,
        saturation_scale: float = 1.0,
        intensity_scale: float = 1.0,
    ) -> torch.Tensor:
        """Convert HVI to RGB using the learned density parameter."""
        return hvi_to_rgb(
            image,
            self.density_k,
            saturation_scale=saturation_scale,
            intensity_scale=intensity_scale,
        )


class LayerNorm2d(nn.Module):
    """Layer normalization over the channel dimension."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mean = image.mean(dim=1, keepdim=True)
        variance = (image - mean).square().mean(dim=1, keepdim=True)
        image = (image - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * image + self.bias[:, None, None]


class NormDownsample(nn.Module):
    """CIDNet convolutional downsampling block."""

    def __init__(self, input_channels: int, output_channels: int, use_norm: bool = False) -> None:
        super().__init__()
        self.use_norm = use_norm
        self.norm = LayerNorm2d(output_channels) if use_norm else nn.Identity()
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=0.5),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.norm(self.prelu(self.down(image)))


class NormUpsample(nn.Module):
    """CIDNet convolutional upsampling and skip-fusion block."""

    def __init__(self, input_channels: int, output_channels: int, use_norm: bool = False) -> None:
        super().__init__()
        self.use_norm = use_norm
        self.norm = LayerNorm2d(output_channels) if use_norm else nn.Identity()
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=2.0),
        )
        self.up = nn.Conv2d(output_channels * 2, output_channels, 1, bias=False)

    def forward(self, image: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        image = self.up_scale(image)
        if image.shape[-2:] != skip.shape[-2:]:
            image = F.interpolate(image, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        image = self.prelu(self.up(torch.cat((image, skip), dim=1)))
        return self.norm(image)


class CrossAttention(nn.Module):
    """Channel-wise cross-attention block from CIDNet."""

    def __init__(self, channels: int, heads: int, bias: bool = False) -> None:
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.q = nn.Conv2d(channels, channels, 1, bias=bias)
        self.q_dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)
        self.kv = nn.Conv2d(channels, channels * 2, 1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            channels * 2,
            channels * 2,
            3,
            padding=1,
            groups=channels * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, 1, bias=bias)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = tensor.shape
        return tensor.reshape(batch, self.heads, channels // self.heads, height * width)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = query.shape
        query = self._split_heads(self.q_dwconv(self.q(query)))
        key, value = self.kv_dwconv(self.kv(context)).chunk(2, dim=1)
        key = self._split_heads(key)
        value = self._split_heads(value)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = torch.softmax((query @ key.transpose(-2, -1)) * self.temperature, dim=-1)
        output = attention @ value
        output = output.reshape(batch, channels, height, width)
        return self.project_out(output)


class IntensityEnhancementLayer(nn.Module):
    """Intensity enhancement/feed-forward layer used in LCA."""

    def __init__(self, channels: int, expansion: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            3,
            padding=1,
            groups=hidden_channels * 2,
            bias=bias,
        )
        self.dwconv1 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=bias)
        self.project_out = nn.Conv2d(hidden_channels, channels, 1, bias=bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        first, second = self.dwconv(self.project_in(image)).chunk(2, dim=1)
        first = torch.tanh(self.dwconv1(first)) + first
        second = torch.tanh(self.dwconv2(second)) + second
        return self.project_out(first * second)


class HVLightenCrossAttention(nn.Module):
    """LCA block for the chromatic H/V branch."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.gdfn = IntensityEnhancementLayer(channels)
        self.norm = LayerNorm2d(channels)
        self.ffn = CrossAttention(channels, heads)

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        image = image + self.ffn(self.norm(image), self.norm(context))
        return self.gdfn(self.norm(image))


class ILightenCrossAttention(nn.Module):
    """LCA block for the intensity branch."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.gdfn = IntensityEnhancementLayer(channels)
        self.ffn = CrossAttention(channels, heads)

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        image = image + self.ffn(self.norm(image), self.norm(context))
        return image + self.gdfn(self.norm(image))


class CIDNet(LLVModel):

    task = "llie"
    aliases = ["HVI-CIDNet"]

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        config = super()._get_default_config()
        config.update({
            "channels": [36, 36, 72, 144],
            "heads": [1, 2, 4, 8],
            "norm": False,
            "density_k": 0.2,
            "input_gamma": 1.0,
            "saturation_scale": 1.0,
            "intensity_scale": 1.0,
            "clamp_output": True,
            "mode": "inference",
        })
        return config

    def _validate_config(self) -> None:
        super()._validate_config()
        if self.config["input_channels"] != 3:
            raise ValueError("CIDNet requires exactly three RGB input channels.")
        channels = self.config["channels"]
        heads = self.config["heads"]
        if len(channels) != 4 or len(heads) != 4:
            raise ValueError("'channels' and 'heads' must each contain four values.")
        if any(int(value) <= 0 for value in channels + heads):
            raise ValueError("All CIDNet channel and head counts must be positive.")
        if any(int(channel) % int(head) != 0 for channel, head in zip(channels, heads)):
            raise ValueError("Every CIDNet channel count must be divisible by its head count.")
        if float(self.config["input_gamma"]) <= 0:
            raise ValueError("'input_gamma' must be positive.")
        if self.config["mode"] not in {"train", "inference"}:
            raise ValueError("'mode' must be 'train' or 'inference'.")

    def _init_model(self) -> None:
        ch1, ch2, ch3, ch4 = [int(value) for value in self.config["channels"]]
        _, head2, head3, head4 = [int(value) for value in self.config["heads"]]
        use_norm = bool(self.config["norm"])

        self.HVE_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(3, ch1, 3, bias=False))
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm)
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm)
        self.HVD_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(ch1, 2, 3, bias=False))

        self.IE_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(1, ch1, 3, bias=False))
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm)
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm)
        self.ID_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(ch1, 1, 3, bias=False))

        self.HV_LCA1 = HVLightenCrossAttention(ch2, head2)
        self.HV_LCA2 = HVLightenCrossAttention(ch3, head3)
        self.HV_LCA3 = HVLightenCrossAttention(ch4, head4)
        self.HV_LCA4 = HVLightenCrossAttention(ch4, head4)
        self.HV_LCA5 = HVLightenCrossAttention(ch3, head3)
        self.HV_LCA6 = HVLightenCrossAttention(ch2, head2)
        self.I_LCA1 = ILightenCrossAttention(ch2, head2)
        self.I_LCA2 = ILightenCrossAttention(ch3, head3)
        self.I_LCA3 = ILightenCrossAttention(ch4, head4)
        self.I_LCA4 = ILightenCrossAttention(ch4, head4)
        self.I_LCA5 = ILightenCrossAttention(ch3, head3)
        self.I_LCA6 = ILightenCrossAttention(ch2, head2)
        self.trans = HVITransform(float(self.config["density_k"]))

    def forward(self, image: torch.Tensor, **kwargs: Any) -> Union[torch.Tensor, Dict[str, Any]]:
        """Enhance an RGB image and expose HVI tensors during training."""
        original_height, original_width = image.shape[-2:]
        gamma = float(kwargs.get("input_gamma", self.config["input_gamma"]))
        if gamma != 1.0:
            image = image.clamp_min(0.0).pow(gamma)
        image = self._pad_to_multiple(image, 8)
        hvi = self.trans.HVIT(image)
        intensity = hvi[:, 2:3].to(image.dtype)

        i_enc0 = self.IE_block0(intensity)
        i_enc1 = self.IE_block1(i_enc0)
        hv_enc0 = self.HVE_block0(hvi)
        hv_enc1 = self.HVE_block1(hv_enc0)

        i_enc2 = self.I_LCA1(i_enc1, hv_enc1)
        hv_enc2 = self.HV_LCA1(hv_enc1, i_enc1)
        i_skip1, hv_skip1 = i_enc2, hv_enc2
        i_enc2 = self.IE_block2(i_enc2)
        hv_enc2 = self.HVE_block2(hv_enc2)

        i_enc3 = self.I_LCA2(i_enc2, hv_enc2)
        hv_enc3 = self.HV_LCA2(hv_enc2, i_enc2)
        i_skip2, hv_skip2 = i_enc3, hv_enc3
        i_enc3 = self.IE_block3(i_enc3)
        hv_enc3 = self.HVE_block3(hv_enc3)

        i_enc4 = self.I_LCA3(i_enc3, hv_enc3)
        hv_enc4 = self.HV_LCA3(hv_enc3, i_enc3)
        i_dec4 = self.I_LCA4(i_enc4, hv_enc4)
        hv_dec4 = self.HV_LCA4(hv_enc4, i_enc4)

        hv_dec3 = self.HVD_block3(hv_dec4, hv_skip2)
        i_dec3 = self.ID_block3(i_dec4, i_skip2)
        i_dec2 = self.I_LCA5(i_dec3, hv_dec3)
        hv_dec2 = self.HV_LCA5(hv_dec3, i_dec3)
        hv_dec2 = self.HVD_block2(hv_dec2, hv_skip1)
        i_dec2 = self.ID_block2(i_dec3, i_skip1)

        i_dec1 = self.I_LCA6(i_dec2, hv_dec2)
        hv_dec1 = self.HV_LCA6(hv_dec2, i_dec2)
        i_dec1 = self.ID_block1(i_dec1, i_enc0)
        intensity_delta = self.ID_block0(i_dec1)
        hv_dec1 = self.HVD_block1(hv_dec1, hv_enc0)
        hv_delta = self.HVD_block0(hv_dec1)

        output_hvi = torch.cat((hv_delta, intensity_delta), dim=1) + hvi
        prediction = self.trans.PHVIT(
            output_hvi,
            saturation_scale=float(self.config["saturation_scale"]),
            intensity_scale=float(self.config["intensity_scale"]),
        )
        prediction = prediction[:, :, :original_height, :original_width]
        if self.config["clamp_output"]:
            prediction = prediction.clamp(0.0, 1.0)

        if self.config["mode"] == "train":
            prediction_hvi = self.trans.HVIT(prediction)
            return self._format_output(
                prediction,
                aux={
                    "prediction_hvi": prediction_hvi,
                    "density_k": self.trans.density_k,
                },
                meta={"color_space": "HVI"},
            )
        return prediction

    def HVIT(self, image: torch.Tensor) -> torch.Tensor:
        """Compatibility helper matching the official CIDNet API."""
        return self.trans.HVIT(image)

    @staticmethod
    def _pad_to_multiple(image: torch.Tensor, multiple: int) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple
        if pad_height == 0 and pad_width == 0:
            return image
        return F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")
