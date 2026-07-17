"""Transformer-based UHD low-light enhancement model.

    Original paper: Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method
    Paper link: https://arxiv.org/abs/2212.11548
    Official source code: https://github.com/TaoWangzj/LLFormer
    Official project url: https://taowangzj.github.io/projects/LLFormer/
    """

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseModel import LLVModel


class BiasFreeLayerNorm(nn.Module):
    """Bias-free channel layer normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        variance = tensor.var(dim=-1, keepdim=True, unbiased=False)
        return tensor / torch.sqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    """Channel layer normalization with learned scale and bias."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=-1, keepdim=True)
        variance = tensor.var(dim=-1, keepdim=True, unbiased=False)
        return (tensor - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Apply token-wise normalization to a BCHW feature map."""

    def __init__(self, channels: int, layernorm_type: str) -> None:
        super().__init__()
        self.body = (
            BiasFreeLayerNorm(channels)
            if layernorm_type == "BiasFree"
            else WithBiasLayerNorm(channels)
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = tensor.shape
        tokens = tensor.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        tokens = self.body(tokens)
        return tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class AxialAttentionImpl(nn.Module):
    """One-direction multi-head self-attention used by LLFormer."""

    def __init__(self, channels: int, heads: int, bias: bool = True) -> None:
        super().__init__()
        self.num_dims = channels
        self.num_heads = heads
        self.q1 = nn.Conv2d(channels, channels * 3, 1, bias=bias)
        self.q2 = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.q3 = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(channels, channels, 1, bias=bias)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = tensor.shape
        head_channels = channels // self.num_heads
        return (
            tensor.reshape(batch, self.num_heads, head_channels, height, width)
            .permute(0, 1, 3, 4, 2)
            .reshape(batch * self.num_heads * height, width, head_channels)
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = tensor.shape
        query, key, value = map(
            self._reshape_heads,
            self.q3(self.q2(self.q1(tensor))).chunk(3, dim=1),
        )
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = torch.softmax((query @ key.transpose(-2, -1)) * self.fac, dim=-1)
        output = attention @ value
        head_channels = channels // self.num_heads
        output = (
            output.reshape(batch, self.num_heads, height, width, head_channels)
            .permute(0, 1, 4, 2, 3)
            .reshape(batch, channels, height, width)
        )
        return self.fin(output)


class AxialAttention(nn.Module):
    """Sequential row and column axial attention."""

    def __init__(self, channels: int, heads: int = 1, bias: bool = True) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("Attention channels must be divisible by heads.")
        self.num_dims = channels
        self.num_heads = heads
        self.row_att = AxialAttentionImpl(channels, heads, bias)
        self.col_att = AxialAttentionImpl(channels, heads, bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.row_att(tensor)
        tensor = self.col_att(tensor.transpose(-2, -1)).transpose(-2, -1)
        return tensor


class DualGatedFeedForward(nn.Module):
    """Dual-gated depthwise feed-forward network."""

    def __init__(self, channels: int, expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            3,
            padding=1,
            groups=hidden_channels * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, 1, bias=bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        first, second = self.dwconv(self.project_in(tensor)).chunk(2, dim=1)
        tensor = F.gelu(second) * first + F.gelu(first) * second
        return self.project_out(tensor)


class TransformerBlock(nn.Module):
    """LLFormer axial-attention transformer block."""

    def __init__(
        self,
        channels: int,
        heads: int,
        expansion_factor: float,
        bias: bool,
        layernorm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(channels, layernorm_type)
        # The official model always enables bias inside axial attention.
        self.attn = AxialAttention(channels, heads, bias=True)
        self.norm2 = LayerNorm(channels, layernorm_type)
        self.ffn = DualGatedFeedForward(channels, expansion_factor, bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor + self.attn(self.norm1(tensor))
        return tensor + self.ffn(self.norm2(tensor))


class OverlapPatchEmbed(nn.Module):
    """Overlapping 3x3 convolutional patch embedding."""

    def __init__(self, input_channels: int, embed_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(input_channels, embed_dim, 3, padding=1, bias=bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.proj(tensor)


class Downsample(nn.Module):
    """Pixel-unshuffle downsampling block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.body(tensor)


class Upsample(nn.Module):
    """Pixel-shuffle upsampling block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.body(tensor)


class LayerAttentionFusion(nn.Module):
    """Cross-layer attention fusion module (LAM)."""

    def __init__(self, channels: int, bias: bool = True) -> None:
        super().__init__()
        self.chanel_in = channels
        self.temperature = nn.Parameter(torch.ones(1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, layers, channels, height, width = tensor.shape
        flat = tensor.reshape(batch, layers * channels, height, width)
        query, key, value = self.qkv_dwconv(self.qkv(flat)).chunk(3, dim=1)
        query = F.normalize(query.reshape(batch, layers, -1), dim=-1)
        key = F.normalize(key.reshape(batch, layers, -1), dim=-1)
        value = value.reshape(batch, layers, -1)
        attention = torch.softmax((query @ key.transpose(-2, -1)) * self.temperature, dim=-1)
        output = (attention @ value).reshape(batch, layers * channels, height, width)
        output = self.project_out(output).reshape(batch, layers, channels, height, width)
        return (output + tensor).reshape(batch, layers * channels, height, width)


def _blocks(
    channels: int,
    count: int,
    heads: int,
    expansion_factor: float,
    bias: bool,
    layernorm_type: str,
) -> nn.Sequential:
    return nn.Sequential(*[
        TransformerBlock(channels, heads, expansion_factor, bias, layernorm_type)
        for _ in range(count)
    ])


class LLFormer(LLVModel):

    task = "llie"
    aliases = ["LL-Former"]

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        config = super()._get_default_config()
        config.update({
            "output_channels": 3,
            "dim": 16,
            "num_blocks": [2, 4, 8, 16],
            "num_refinement_blocks": 2,
            "heads": [1, 2, 4, 8],
            "ffn_expansion_factor": 2.66,
            "bias": False,
            "layernorm_type": "WithBias",
            "attention": True,
            "skip": False,
            "pad_input": True,
            "clamp_output": True,
            "tile_size": None,
            "tile_overlap": 0,
            "mode": "inference",
        })
        return config

    def _validate_config(self) -> None:
        super()._validate_config()
        if int(self.config["input_channels"]) <= 0 or int(self.config["output_channels"]) <= 0:
            raise ValueError("Input and output channels must be positive.")
        dim = int(self.config["dim"])
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("LLFormer 'dim' must be a positive even integer.")
        num_blocks = list(self.config["num_blocks"])
        heads = list(self.config["heads"])
        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError("'num_blocks' and 'heads' must contain four values.")
        if any(int(value) <= 0 for value in num_blocks + heads):
            raise ValueError("All block and head counts must be positive.")
        channels = [dim, dim * 2, dim * 4, dim * 8]
        if any(channel % int(head) != 0 for channel, head in zip(channels, heads)):
            raise ValueError("Every LLFormer level width must be divisible by its head count.")
        if int(self.config["num_refinement_blocks"]) <= 0:
            raise ValueError("'num_refinement_blocks' must be positive.")
        if float(self.config["ffn_expansion_factor"]) <= 0:
            raise ValueError("'ffn_expansion_factor' must be positive.")
        if self.config["layernorm_type"] not in {"BiasFree", "WithBias"}:
            raise ValueError("'layernorm_type' must be 'BiasFree' or 'WithBias'.")
        if self.config["skip"] and self.config["input_channels"] != self.config["output_channels"]:
            raise ValueError("Residual 'skip' requires matching input and output channels.")
        if self.config["mode"] not in {"train", "inference"}:
            raise ValueError("'mode' must be 'train' or 'inference'.")
        tile_size = self._normalize_spatial_value(self.config.get("tile_size"), "tile_size", allow_none=True)
        if tile_size is not None:
            if any(value % 16 != 0 for value in tile_size):
                raise ValueError("LLFormer tile dimensions must be divisible by 16.")
            overlap = self._normalize_spatial_value(
                self.config.get("tile_overlap", 0),
                "tile_overlap",
                allow_zero=True,
            )
            if any(overlap_value >= tile_value for overlap_value, tile_value in zip(overlap, tile_size)):
                raise ValueError("Each tile_overlap value must be smaller than tile_size.")

    def _init_model(self) -> None:
        dim = int(self.config["dim"])
        num_blocks = [int(value) for value in self.config["num_blocks"]]
        heads = [int(value) for value in self.config["heads"]]
        refinement_blocks = int(self.config["num_refinement_blocks"])
        expansion = float(self.config["ffn_expansion_factor"])
        bias = bool(self.config["bias"])
        norm_type = self.config["layernorm_type"]
        attention = bool(self.config["attention"])

        self.coefficient = nn.Parameter(torch.ones(4, 2, dim * 8), requires_grad=attention)
        self.patch_embed = OverlapPatchEmbed(int(self.config["input_channels"]), dim, bias=False)
        self.encoder_1 = _blocks(dim, num_blocks[0], heads[0], expansion, bias, norm_type)
        self.encoder_2 = _blocks(dim, num_blocks[0], heads[0], expansion, bias, norm_type)
        self.encoder_3 = _blocks(dim, num_blocks[0], heads[0], expansion, bias, norm_type)
        self.layer_fussion = LayerAttentionFusion(dim * 3)
        self.conv_fuss = nn.Conv2d(dim * 3, dim, 1, bias=bias)

        self.down_1 = Downsample(dim)
        self.decoder_level1_0 = _blocks(dim * 2, num_blocks[0], heads[1], expansion, bias, norm_type)
        self.down_2 = Downsample(dim * 2)
        self.decoder_level2_0 = _blocks(dim * 4, num_blocks[1], heads[2], expansion, bias, norm_type)
        self.down_3 = Downsample(dim * 4)
        self.decoder_level3_0 = _blocks(dim * 8, num_blocks[2], heads[3], expansion, bias, norm_type)
        self.down_4 = Downsample(dim * 8)
        self.decoder_level4 = _blocks(dim * 16, num_blocks[3], heads[3], expansion, bias, norm_type)

        self.up4_3 = Upsample(dim * 16)
        self.decoder_level3_1 = _blocks(dim * 8, num_blocks[2], heads[3], expansion, bias, norm_type)
        self.up3_2 = Upsample(dim * 8)
        self.decoder_level2_1 = _blocks(dim * 4, num_blocks[1], heads[2], expansion, bias, norm_type)
        self.up2_1 = Upsample(dim * 4)
        self.decoder_level1_1 = _blocks(dim * 2, num_blocks[0], heads[1], expansion, bias, norm_type)
        self.up2_0 = Upsample(dim * 2)

        self.coefficient_4_3 = nn.Parameter(torch.ones(2, dim * 8), requires_grad=attention)
        self.coefficient_3_2 = nn.Parameter(torch.ones(2, dim * 4), requires_grad=attention)
        self.coefficient_2_1 = nn.Parameter(torch.ones(2, dim * 2), requires_grad=attention)
        self.coefficient_1_0 = nn.Parameter(torch.ones(2, dim), requires_grad=attention)
        self.skip_4_3 = nn.Conv2d(dim * 8, dim * 8, 1, bias=bias)
        self.skip_3_2 = nn.Conv2d(dim * 4, dim * 4, 1, bias=bias)
        self.skip_2_1 = nn.Conv2d(dim * 2, dim * 2, 1, bias=bias)
        self.skip_1_0 = nn.Conv2d(dim * 2, dim * 2, 1, bias=bias)

        self.latent = _blocks(dim, num_blocks[0], heads[0], expansion, bias, norm_type)
        self.refinement_1 = _blocks(dim, refinement_blocks, heads[0], expansion, bias, norm_type)
        self.refinement_2 = _blocks(dim, refinement_blocks, heads[0], expansion, bias, norm_type)
        self.refinement_3 = _blocks(dim, refinement_blocks, heads[0], expansion, bias, norm_type)
        self.layer_fussion_2 = LayerAttentionFusion(dim * 3)
        self.conv_fuss_2 = nn.Conv2d(dim * 3, dim, 1, bias=bias)
        self.output = nn.Conv2d(dim, int(self.config["output_channels"]), 3, padding=1, bias=bias)
        self.skip = bool(self.config["skip"])

    @staticmethod
    def _weighted_skip(first: torch.Tensor, second: torch.Tensor, coefficient: torch.Tensor) -> torch.Tensor:
        return (
            coefficient[0][None, :, None, None] * first
            + coefficient[1][None, :, None, None] * second
        )

    def _forward_core(self, image: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder_1(self.patch_embed(image))
        encoder2 = self.encoder_2(encoder1)
        encoder3 = self.encoder_3(encoder2)
        fusion = torch.stack((encoder1, encoder2, encoder3), dim=1)
        fusion = self.conv_fuss(self.layer_fussion(fusion))

        level1_0 = self.decoder_level1_0(self.down_1(fusion))
        level2_0 = self.decoder_level2_0(self.down_2(level1_0))
        level3_0 = self.decoder_level3_0(self.down_3(level2_0))
        level4 = self.decoder_level4(self.down_4(level3_0))

        level4 = self.up4_3(level4)
        level3_1 = self.skip_4_3(self._weighted_skip(level3_0, level4, self.coefficient_4_3))
        level3_1 = self.up3_2(self.decoder_level3_1(level3_1))
        level2_1 = self.skip_3_2(self._weighted_skip(level2_0, level3_1, self.coefficient_3_2))
        level2_1 = self.up2_1(self.decoder_level2_1(level2_1))
        level1_1 = self.skip_1_0(self._weighted_skip(level1_0, level2_1, self.coefficient_2_1))
        level1_1 = self.up2_0(self.decoder_level1_1(level1_1))

        fusion = self.latent(fusion)
        output = self._weighted_skip(fusion, level1_1, self.coefficient_1_0)
        refinement1 = self.refinement_1(output)
        refinement2 = self.refinement_2(refinement1)
        refinement3 = self.refinement_3(refinement2)
        output = self.conv_fuss_2(
            self.layer_fussion_2(torch.stack((refinement1, refinement2, refinement3), dim=1))
        )
        return self.output(output) + image if self.skip else self.output(output)

    def forward(self, image: torch.Tensor, **kwargs: Any) -> Union[torch.Tensor, Dict[str, Any]]:
        """Enhance an image, padding to the architecture's 16-pixel stride."""
        height, width = image.shape[-2:]
        padded = self._pad_to_multiple(image, 16) if self.config["pad_input"] else image
        tile_size = self._normalize_spatial_value(
            kwargs.get("tile_size", self.config.get("tile_size")),
            "tile_size",
            allow_none=True,
        )
        if self.config["mode"] == "inference" and tile_size is not None:
            tile_overlap = self._normalize_spatial_value(
                kwargs.get("tile_overlap", self.config.get("tile_overlap", 0)),
                "tile_overlap",
                allow_zero=True,
            )
            prediction = self._forward_tiled(padded, tile_size, tile_overlap)
        else:
            prediction = self._forward_core(padded)
        prediction = prediction[:, :, :height, :width]

        if self.config["mode"] == "train":
            return self._format_output(prediction, meta={"padded_size": tuple(padded.shape[-2:])})
        if self.config["clamp_output"]:
            prediction = prediction.clamp(0.0, 1.0)
        return prediction

    def _forward_tiled(
        self,
        image: torch.Tensor,
        tile_size,
        tile_overlap,
    ) -> torch.Tensor:
        """Enhance large images with overlap-and-average tiled inference."""
        tile_height = min(tile_size[0], image.shape[-2])
        tile_width = min(tile_size[1], image.shape[-1])
        if tile_height == image.shape[-2] and tile_width == image.shape[-1]:
            return self._forward_core(image)
        stride_height = max(1, tile_height - tile_overlap[0])
        stride_width = max(1, tile_width - tile_overlap[1])
        top_positions = self._tile_positions(image.shape[-2], tile_height, stride_height)
        left_positions = self._tile_positions(image.shape[-1], tile_width, stride_width)
        output = image.new_zeros(
            image.shape[0],
            int(self.config["output_channels"]),
            image.shape[-2],
            image.shape[-1],
        )
        counts = image.new_zeros(1, 1, image.shape[-2], image.shape[-1])
        for top in top_positions:
            for left in left_positions:
                patch = image[:, :, top:top + tile_height, left:left + tile_width]
                restored = self._forward_core(patch)
                output[:, :, top:top + tile_height, left:left + tile_width] += restored
                counts[:, :, top:top + tile_height, left:left + tile_width] += 1
        return output / counts.clamp_min(1)

    @staticmethod
    def _tile_positions(length: int, tile: int, stride: int):
        positions = list(range(0, max(1, length - tile + 1), stride))
        final = length - tile
        if positions[-1] != final:
            positions.append(final)
        return positions

    @staticmethod
    def _normalize_spatial_value(value, name: str, allow_none: bool = False, allow_zero: bool = False):
        if value is None and allow_none:
            return None
        if isinstance(value, int):
            values = (value, value)
        else:
            try:
                values = tuple(int(item) for item in value)
            except (TypeError, ValueError):
                raise ValueError("{} must be an int or two integers.".format(name))
        if len(values) != 2:
            raise ValueError("{} must contain two values.".format(name))
        minimum = 0 if allow_zero else 1
        if any(item < minimum for item in values):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError("{} values must be {}.".format(name, qualifier))
        return values

    @staticmethod
    def _pad_to_multiple(image: torch.Tensor, multiple: int) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple
        if pad_height == 0 and pad_width == 0:
            return image
        mode = "reflect" if pad_height < height and pad_width < width else "replicate"
        return F.pad(image, (0, pad_width, 0, pad_height), mode=mode)
