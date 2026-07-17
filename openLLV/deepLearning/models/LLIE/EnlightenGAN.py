"""
EnlightenGAN model for unpaired low-light enhancement.

Original paper: EnlightenGAN: Deep Light Enhancement Without Paired Supervision
Paper link: https://doi.org/10.1109/TIP.2021.3051462
Official source code: https://github.com/VITA-Group/EnlightenGAN
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseModel import LLVModel


class EnlightenGANConvBlock(nn.Module):
    """Convolution, optional normalization, and activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        normalize: bool = True,
        activation: str = "leaky_relu",
    ) -> None:
        """Initialize a convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            normalize: Whether to use instance normalization.
            activation: Activation type, either ``"leaky_relu"`` or ``"relu"``.
        """
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not normalize,
            )
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.block(x)


class EnlightenGANUpBlock(nn.Module):
    """Upsampling block used by the EnlightenGAN generator."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize an upsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply upsampling.

        Args:
            x: Input tensor.

        Returns:
            Upsampled feature tensor.
        """
        return self.block(x)


class EnlightenGANGenerator(nn.Module):
    """Attention-guided U-Net generator for EnlightenGAN."""

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        attention: bool = True,
    ) -> None:
        """Initialize the generator.

        Args:
            image_channels: Number of image channels.
            base_channels: Number of base feature channels.
            attention: Whether to concatenate an attention map to the input.
        """
        super().__init__()
        self.attention = attention
        in_channels = image_channels + (1 if attention else 0)

        self.down1 = EnlightenGANConvBlock(in_channels, base_channels, normalize=False)
        self.down2 = EnlightenGANConvBlock(base_channels, base_channels * 2)
        self.down3 = EnlightenGANConvBlock(base_channels * 2, base_channels * 4)
        self.down4 = EnlightenGANConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = EnlightenGANUpBlock(base_channels * 8, base_channels * 4)
        self.up2 = EnlightenGANUpBlock(base_channels * 8, base_channels * 2)
        self.up1 = EnlightenGANUpBlock(base_channels * 4, base_channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, image_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, attention_map: torch.Tensor) -> torch.Tensor:
        """Generate an enhanced image.

        Args:
            image: Low-light input image.
            attention_map: Attention map estimated from the input image.

        Returns:
            Enhanced image tensor.
        """
        if self.attention:
            x = torch.cat([image, attention_map], dim=1)
        else:
            x = image

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u3 = self._resize(self.up3(d4), d3)
        u3 = torch.cat([u3, d3], dim=1)
        u2 = self._resize(self.up2(u3), d2)
        u2 = torch.cat([u2, d2], dim=1)
        u1 = self._resize(self.up1(u2), d1)
        u1 = torch.cat([u1, d1], dim=1)

        output = self.out_conv(u1)
        return self._resize(output, image)

    @staticmethod
    def _resize(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Resize source to target spatial size.

        Args:
            source: Source tensor.
            target: Target tensor.

        Returns:
            Resized tensor.
        """
        if source.shape[-2:] == target.shape[-2:]:
            return source
        return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)


class EnlightenGANDiscriminator(nn.Module):
    """PatchGAN discriminator used by EnlightenGAN."""

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        num_layers: int = 3,
    ) -> None:
        """Initialize a PatchGAN discriminator.

        Args:
            image_channels: Number of image channels.
            base_channels: Number of base feature channels.
            num_layers: Number of downsampling layers.
        """
        super().__init__()
        layers = [
            nn.Conv2d(image_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base_channels
        for _ in range(1, num_layers):
            next_channels = min(channels * 2, base_channels * 8)
            layers.extend(
                [
                    nn.Conv2d(next_channels // 2, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(next_channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = next_channels
        layers.append(nn.Conv2d(channels, 1, kernel_size=4, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Classify an image or patch.

        Args:
            image: Input image tensor.

        Returns:
            Patch-level discriminator logits.
        """
        return self.net(image)


class EnlightenGAN(LLVModel):

    task = "llie"
    aliases = ["EnlightenGAN"]

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize EnlightenGAN.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default EnlightenGAN configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update(
            {
                "generator_channels": 32,
                "discriminator_channels": 32,
                "discriminator_layers": 3,
                "use_attention": True,
                "local_patch_ratio": 0.5,
                "mode": "inference",
            }
        )
        return default_config

    def _validate_config(self) -> None:
        """Validate EnlightenGAN configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()
        for key in ("generator_channels", "discriminator_channels", "discriminator_layers"):
            if int(self.config[key]) <= 0:
                raise ValueError(f"'{key}' must be positive.")
        patch_ratio = float(self.config["local_patch_ratio"])
        if not (0 < patch_ratio <= 1):
            raise ValueError("'local_patch_ratio' must be in (0, 1].")
        if self.config["mode"] not in {"train", "inference"}:
            raise ValueError("'mode' must be 'train' or 'inference'.")

    def _init_model(self) -> None:
        """Initialize EnlightenGAN generator and discriminators."""
        image_channels = int(self.config["input_channels"])
        self.generator = EnlightenGANGenerator(
            image_channels=image_channels,
            base_channels=int(self.config["generator_channels"]),
            attention=bool(self.config["use_attention"]),
        )
        self.global_discriminator = EnlightenGANDiscriminator(
            image_channels=image_channels,
            base_channels=int(self.config["discriminator_channels"]),
            num_layers=int(self.config["discriminator_layers"]),
        )
        self.local_discriminator = EnlightenGANDiscriminator(
            image_channels=image_channels,
            base_channels=int(self.config["discriminator_channels"]),
            num_layers=int(self.config["discriminator_layers"]),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights following common GAN practice."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.normal_(module.weight, mean=1.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run an EnlightenGAN forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.

        Returns:
            Training mode: standardized output dict with generator output and
            discriminator modules. Inference mode: enhanced image tensor.
        """
        attention_map = self._attention_map(x)
        enhanced = self.generator(x, attention_map)

        if self.config["mode"] == "train":
            fake_local, local_box = self._local_patch(enhanced)
            return self._format_output(
                pred=enhanced,
                aux={
                    "enhanced": enhanced,
                    "attention_map": attention_map,
                    "global_discriminator": self.global_discriminator,
                    "local_discriminator": self.local_discriminator,
                    "fake_local": fake_local,
                    "local_box": local_box,
                },
                meta={"mode": self.config["mode"]},
            )

        return enhanced

    @staticmethod
    def _attention_map(image: torch.Tensor) -> torch.Tensor:
        """Estimate an attention map from input darkness.

        Args:
            image: Input image tensor.

        Returns:
            Single-channel attention tensor.
        """
        gray = image.mean(dim=1, keepdim=True)
        return (1.0 - gray).clamp(0.0, 1.0)

    def _local_patch(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Crop a deterministic center local patch.

        Args:
            image: Input image tensor.

        Returns:
            Tuple containing local patch and ``(top, left, height, width)``.
        """
        _, _, height, width = image.shape
        ratio = float(self.config["local_patch_ratio"])
        patch_h = max(16, int(height * ratio))
        patch_w = max(16, int(width * ratio))
        patch_h = min(height, patch_h)
        patch_w = min(width, patch_w)
        top = max(0, (height - patch_h) // 2)
        left = max(0, (width - patch_w) // 2)
        return image[:, :, top : top + patch_h, left : left + patch_w], (top, left, patch_h, patch_w)
