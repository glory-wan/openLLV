"""
RetinexFormer model for one-stage Retinex-based low-light enhancement.

Original paper: Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
Paper link: https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_Retinexformer_One-stage_Retinex-based_Transformer_for_Low-light_Image_Enhancement_ICCV_2023_paper.pdf
Official source code: https://github.com/caiyuanhao1998/Retinexformer
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseModel import LLVModel


class RetinexFormerGELU(nn.Module):
    """GELU activation wrapper used by RetinexFormer blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        return F.gelu(x)


class RetinexFormerPreNorm(nn.Module):
    """LayerNorm wrapper for channel-last RetinexFormer blocks."""

    def __init__(self, dim: int, module: nn.Module) -> None:
        """Initialize the pre-normalization wrapper.

        Args:
            dim: Feature dimension.
            module: Module applied after normalization.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Normalize and forward input.

        Args:
            x: Channel-last input tensor.
            *args: Extra positional arguments forwarded to the wrapped module.
            **kwargs: Extra keyword arguments forwarded to the wrapped module.

        Returns:
            Output tensor returned by the wrapped module.
        """
        return self.module(self.norm(x), *args, **kwargs)


class IlluminationEstimator(nn.Module):
    """Estimate illumination features and illumination map."""

    def __init__(
        self,
        feature_channels: int,
        in_channels: int = 3,
        out_channels: int = 3,
    ) -> None:
        """Initialize the illumination estimator.

        Args:
            feature_channels: Number of intermediate illumination features.
            in_channels: Number of image input channels.
            out_channels: Number of output illumination map channels.
        """
        super().__init__()
        estimator_in_channels = in_channels + 1
        groups = estimator_in_channels if feature_channels % estimator_in_channels == 0 else 1

        self.conv1 = nn.Conv2d(
            estimator_in_channels,
            feature_channels,
            kernel_size=1,
            bias=True,
        )
        self.depth_conv = nn.Conv2d(
            feature_channels,
            feature_channels,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=groups,
        )
        self.conv2 = nn.Conv2d(
            feature_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate illumination information from an image.

        Args:
            image: Input image tensor with shape ``[B, C, H, W]``.

        Returns:
            Tuple containing illumination features and illumination map.
        """
        mean_channel = image.mean(dim=1, keepdim=True)
        estimator_input = torch.cat([image, mean_channel], dim=1)
        features = self.conv1(estimator_input)
        illumination_features = self.depth_conv(features)
        illumination_map = self.conv2(illumination_features)
        return illumination_features, illumination_map


class IlluminationGuidedMSA(nn.Module):
    """Illumination-guided multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
    ) -> None:
        """Initialize illumination-guided attention.

        Args:
            dim: Input feature dimension.
            dim_head: Dimension per attention head.
            heads: Number of attention heads.
        """
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(hidden_dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            RetinexFormerGELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        illumination_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply illumination-guided attention.

        Args:
            x: Channel-last feature tensor with shape ``[B, H, W, C]``.
            illumination_features: Channel-last illumination features with the
                same spatial size and channel count as ``x``.

        Returns:
            Channel-last output tensor.
        """
        batch, height, width, channels = x.shape
        num_tokens = height * width
        flat_x = x.reshape(batch, num_tokens, channels)
        flat_illumination = illumination_features.reshape(batch, num_tokens, channels)

        q = self._reshape_heads(self.to_q(flat_x))
        k = self._reshape_heads(self.to_k(flat_x))
        v = self._reshape_heads(self.to_v(flat_x))
        illumination = self._reshape_heads(flat_illumination)

        v = v * illumination
        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        v = v.transpose(-2, -1)

        attention = (k @ q.transpose(-2, -1)) * self.rescale
        attention = attention.softmax(dim=-1)
        output = attention @ v
        output = output.permute(0, 3, 1, 2).reshape(batch, num_tokens, -1)
        output = self.proj(output).view(batch, height, width, channels)

        pos = self.pos_emb(
            flat_x.view(batch, height, width, channels).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        return output + pos

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape token features into attention heads.

        Args:
            tensor: Tensor with shape ``[B, N, heads * dim_head]``.

        Returns:
            Tensor with shape ``[B, heads, N, dim_head]``.
        """
        batch, num_tokens, _ = tensor.shape
        return tensor.view(batch, num_tokens, self.heads, self.dim_head).permute(0, 2, 1, 3)


class RetinexFormerFeedForward(nn.Module):
    """Depthwise convolutional feed-forward network."""

    def __init__(self, dim: int, mult: int = 4) -> None:
        """Initialize the feed-forward network.

        Args:
            dim: Input feature dimension.
            mult: Hidden channel multiplier.
        """
        super().__init__()
        hidden_dim = dim * mult
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, bias=False),
            RetinexFormerGELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False, groups=hidden_dim),
            RetinexFormerGELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward layers.

        Args:
            x: Channel-last feature tensor with shape ``[B, H, W, C]``.

        Returns:
            Channel-last output tensor.
        """
        output = self.net(x.permute(0, 3, 1, 2).contiguous())
        return output.permute(0, 2, 3, 1)


class IlluminationGuidedAttentionBlock(nn.Module):
    """Stack of illumination-guided attention and feed-forward blocks."""

    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        num_blocks: int,
    ) -> None:
        """Initialize an illumination-guided attention block.

        Args:
            dim: Feature dimension.
            dim_head: Dimension per attention head.
            heads: Number of attention heads.
            num_blocks: Number of repeated attention blocks.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        IlluminationGuidedMSA(dim=dim, dim_head=dim_head, heads=heads),
                        RetinexFormerPreNorm(
                            dim,
                            RetinexFormerFeedForward(dim=dim),
                        ),
                    ]
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        illumination_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention blocks to feature tensors.

        Args:
            x: Feature tensor with shape ``[B, C, H, W]``.
            illumination_features: Illumination feature tensor with shape
                ``[B, C, H, W]``.

        Returns:
            Output feature tensor with shape ``[B, C, H, W]``.
        """
        x = x.permute(0, 2, 3, 1)
        illumination = illumination_features.permute(0, 2, 3, 1)

        for attention, feed_forward in self.blocks:
            x = attention(x, illumination) + x
            x = feed_forward(x) + x

        return x.permute(0, 3, 1, 2)


class RetinexFormerDenoiser(nn.Module):
    """U-Net style illumination-guided denoiser."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 32,
        levels: int = 2,
        num_blocks: Sequence[int] = (1, 1, 1),
    ) -> None:
        """Initialize the RetinexFormer denoiser.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output image channels.
            dim: Base feature dimension.
            levels: Number of encoder-decoder levels.
            num_blocks: Number of attention blocks per level and bottleneck.
        """
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.num_blocks = self._normalize_num_blocks(num_blocks, levels)

        self.embedding = nn.Conv2d(in_channels, dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList()
        current_dim = dim
        for level in range(levels):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        IlluminationGuidedAttentionBlock(
                            dim=current_dim,
                            dim_head=dim,
                            heads=max(1, current_dim // dim),
                            num_blocks=self.num_blocks[level],
                        ),
                        nn.Conv2d(current_dim, current_dim * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(current_dim, current_dim * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            current_dim *= 2

        self.bottleneck = IlluminationGuidedAttentionBlock(
            dim=current_dim,
            dim_head=dim,
            heads=max(1, current_dim // dim),
            num_blocks=self.num_blocks[-1],
        )

        self.decoder_layers = nn.ModuleList()
        for level in range(levels):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            current_dim,
                            current_dim // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        nn.Conv2d(current_dim, current_dim // 2, 1, 1, bias=False),
                        IlluminationGuidedAttentionBlock(
                            dim=current_dim // 2,
                            dim_head=dim,
                            heads=max(1, (current_dim // 2) // dim),
                            num_blocks=self.num_blocks[levels - 1 - level],
                        ),
                    ]
                )
            )
            current_dim //= 2

        self.mapping = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=False)
        self.apply(self._init_weights)

    def forward(
        self,
        image: torch.Tensor,
        illumination_features: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise an illumination-enhanced image.

        Args:
            image: Image tensor with shape ``[B, C, H, W]``.
            illumination_features: Illumination feature tensor.

        Returns:
            Restored image tensor.
        """
        features = self.embedding(image)
        encoder_features: List[torch.Tensor] = []
        illumination_pyramid: List[torch.Tensor] = []

        for attention_block, feature_downsample, illumination_downsample in self.encoder_layers:
            features = attention_block(features, illumination_features)
            encoder_features.append(features)
            illumination_pyramid.append(illumination_features)
            features = feature_downsample(features)
            illumination_features = illumination_downsample(illumination_features)

        features = self.bottleneck(features, illumination_features)

        for index, (upsample, fusion, attention_block) in enumerate(self.decoder_layers):
            features = upsample(features)
            skip = encoder_features[self.levels - 1 - index]
            features = self._match_spatial_size(features, skip)
            features = fusion(torch.cat([features, skip], dim=1))
            illumination_features = illumination_pyramid[self.levels - 1 - index]
            features = attention_block(features, illumination_features)

        return self.mapping(features) + image

    @staticmethod
    def _match_spatial_size(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Match source spatial size to target.

        Args:
            source: Source tensor.
            target: Target tensor.

        Returns:
            Spatially matched source tensor.
        """
        if source.shape[-2:] == target.shape[-2:]:
            return source
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _normalize_num_blocks(
        num_blocks: Sequence[int],
        levels: int,
    ) -> Tuple[int, ...]:
        """Normalize the number of blocks for all levels.

        Args:
            num_blocks: User-provided block counts.
            levels: Number of encoder-decoder levels.

        Returns:
            Tuple with ``levels + 1`` block counts.
        """
        blocks = tuple(int(value) for value in num_blocks)
        if len(blocks) < levels + 1:
            blocks = blocks + (blocks[-1],) * (levels + 1 - len(blocks))
        return blocks[: levels + 1]

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize selected module weights.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class RetinexFormerSingleStage(nn.Module):
    """Single RetinexFormer enhancement stage."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        feature_channels: int = 32,
        levels: int = 2,
        num_blocks: Sequence[int] = (1, 1, 1),
    ) -> None:
        """Initialize a single RetinexFormer stage.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output image channels.
            feature_channels: Base feature channels.
            levels: Number of encoder-decoder levels.
            num_blocks: Number of attention blocks per level and bottleneck.
        """
        super().__init__()
        self.estimator = IlluminationEstimator(
            feature_channels=feature_channels,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.denoiser = RetinexFormerDenoiser(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=feature_channels,
            levels=levels,
            num_blocks=num_blocks,
        )

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run one RetinexFormer stage.

        Args:
            image: Input image tensor.

        Returns:
            Tuple containing the restored image and stage intermediate tensors.
        """
        illumination_features, illumination_map = self.estimator(image)
        enhanced_input = image * illumination_map + image
        output = self.denoiser(enhanced_input, illumination_features)
        return output, {
            "illumination_features": illumination_features,
            "illumination_map": illumination_map,
            "enhanced_input": enhanced_input,
        }


class RetinexFormer(LLVModel):

    task = "llie"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize RetinexFormer.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default RetinexFormer configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update(
            {
                "output_channels": 3,
                "feature_channels": 32,
                "stage": 1,
                "levels": 2,
                "num_blocks": [1, 1, 1],
                "mode": "inference",
                "clamp_output": True,
            }
        )
        return default_config

    def _validate_config(self) -> None:
        """Validate RetinexFormer configuration.

        Raises:
            ValueError: If a configuration parameter is invalid.
        """
        super()._validate_config()

        if self.config["input_channels"] <= 0:
            raise ValueError("'input_channels' must be positive.")
        if self.config["output_channels"] <= 0:
            raise ValueError("'output_channels' must be positive.")
        if self.config["feature_channels"] <= 0:
            raise ValueError("'feature_channels' must be positive.")
        if self.config["stage"] <= 0:
            raise ValueError("'stage' must be positive.")
        if self.config["levels"] <= 0:
            raise ValueError("'levels' must be positive.")
        if self.config["mode"] not in {"train", "inference"}:
            raise ValueError("'mode' must be 'train' or 'inference'.")

    def _init_model(self) -> None:
        """Initialize RetinexFormer stages."""
        self.stage = int(self.config["stage"])
        self.pad_factor = 2 ** int(self.config["levels"])
        self.body = nn.ModuleList(
            [
                RetinexFormerSingleStage(
                    in_channels=int(self.config["input_channels"]),
                    out_channels=int(self.config["output_channels"]),
                    feature_channels=int(self.config["feature_channels"]),
                    levels=int(self.config["levels"]),
                    num_blocks=self.config["num_blocks"],
                )
                for _ in range(self.stage)
            ]
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a RetinexFormer forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.
            **kwargs: Optional forward parameters.

        Returns:
            Training mode: standardized output dictionary containing stage
            intermediates. Inference mode: enhanced image tensor.
        """
        original_size = x.shape[-2:]
        padded = self._pad_to_factor(x, self.pad_factor)
        stage_outputs = []
        output = padded

        for stage in self.body:
            output, intermediates = stage(output)
            stage_outputs.append(intermediates)

        output = output[..., : original_size[0], : original_size[1]]
        if self.config.get("clamp_output", True):
            output = output.clamp(0.0, 1.0)

        if self.config["mode"] == "train":
            return self._format_output(
                pred=output,
                aux={
                    "stage_outputs": stage_outputs,
                },
                meta={
                    "mode": self.config["mode"],
                    "stage": self.stage,
                },
            )

        return output

    @staticmethod
    def _pad_to_factor(x: torch.Tensor, factor: int) -> torch.Tensor:
        """Pad a tensor so spatial dimensions are divisible by ``factor``.

        Args:
            x: Input tensor.
            factor: Divisibility factor.

        Returns:
            Padded tensor.
        """
        if factor <= 1:
            return x

        height, width = x.shape[-2:]
        pad_h = (factor - height % factor) % factor
        pad_w = (factor - width % factor) % factor
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
