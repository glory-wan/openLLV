"""
KinD model for practical low-light image enhancement.

Original paper: Kindling the Darkness: A Practical Low-light Image Enhancer
Paper link: https://doi.org/10.1145/3343031.3350926
Official source code: https://github.com/zhangyhuaee/KinD
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseModel import LLVModel


class KinDConvBlock(nn.Module):
    """Convolution block used by KinD subnetworks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize a convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and activation.

        Args:
            x: Input tensor.

        Returns:
            Output feature tensor.
        """
        return self.block(x)


class KinDDecompositionNet(nn.Module):
    """KinD decomposition network for reflectance and illumination."""

    def __init__(
        self,
        image_channels: int = 3,
        feature_channels: int = 64,
        num_layers: int = 5,
    ) -> None:
        """Initialize the decomposition network.

        Args:
            image_channels: Number of image channels.
            feature_channels: Number of intermediate feature channels.
            num_layers: Number of intermediate convolution layers.
        """
        super().__init__()
        self.image_channels = image_channels
        self.shallow = KinDConvBlock(image_channels + 1, feature_channels)
        self.body = nn.Sequential(
            *[
                KinDConvBlock(feature_channels, feature_channels)
                for _ in range(num_layers)
            ]
        )
        self.out_conv = nn.Conv2d(
            feature_channels,
            image_channels + 1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose an image into reflectance and illumination.

        Args:
            image: Input image tensor with shape ``[B, C, H, W]``.

        Returns:
            Tuple containing reflectance ``R`` and illumination ``I``.
        """
        max_channel = torch.max(image, dim=1, keepdim=True).values
        features = self.shallow(torch.cat([image, max_channel], dim=1))
        features = self.body(features)
        output = torch.sigmoid(self.out_conv(features))
        reflectance = output[:, : self.image_channels, :, :]
        illumination = output[:, self.image_channels : self.image_channels + 1, :, :]
        return reflectance, illumination


class KinDRestorationNet(nn.Module):
    """KinD reflectance restoration network."""

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        """Initialize the restoration network.

        Args:
            image_channels: Number of image channels.
            base_channels: Number of base feature channels.
        """
        super().__init__()
        in_channels = image_channels + 1
        self.enc1 = KinDConvBlock(in_channels, base_channels)
        self.enc2 = KinDConvBlock(base_channels, base_channels * 2)
        self.enc3 = KinDConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = KinDConvBlock(base_channels * 4, base_channels * 8)

        self.dec3 = KinDConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = KinDConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = KinDConvBlock(base_channels * 2 + base_channels, base_channels)
        self.out_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, reflectance: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        """Restore reflectance using illumination guidance.

        Args:
            reflectance: Low-light reflectance tensor.
            illumination: Low-light illumination tensor.

        Returns:
            Restored reflectance tensor.
        """
        x = torch.cat([reflectance, illumination], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2, stride=2))
        e3 = self.enc3(F.avg_pool2d(e2, kernel_size=2, stride=2))
        e4 = self.enc4(F.avg_pool2d(e3, kernel_size=2, stride=2))

        d3 = self._resize(e4, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self._resize(d3, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._resize(d2, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        residual = torch.tanh(self.out_conv(d1))
        return torch.clamp(reflectance + residual, 0.0, 1.0)

    @staticmethod
    def _resize(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Resize a feature map to match a target tensor.

        Args:
            source: Source feature tensor.
            target: Target feature tensor.

        Returns:
            Resized source tensor.
        """
        return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)


class KinDIlluminationAdjustmentNet(nn.Module):
    """KinD illumination adjustment network."""

    def __init__(self, feature_channels: int = 32, num_layers: int = 3) -> None:
        """Initialize the illumination adjustment network.

        Args:
            feature_channels: Number of intermediate feature channels.
            num_layers: Number of intermediate convolution layers.
        """
        super().__init__()
        self.in_conv = KinDConvBlock(2, feature_channels)
        self.body = nn.Sequential(
            *[
                KinDConvBlock(feature_channels, feature_channels)
                for _ in range(num_layers)
            ]
        )
        self.out_conv = nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1)

    def forward(self, illumination: torch.Tensor, ratio_map: torch.Tensor) -> torch.Tensor:
        """Adjust illumination with an exposure-ratio map.

        Args:
            illumination: Input illumination tensor.
            ratio_map: Desired exposure-ratio tensor.

        Returns:
            Adjusted illumination tensor.
        """
        features = self.in_conv(torch.cat([illumination, ratio_map], dim=1))
        features = self.body(features)
        residual = torch.tanh(self.out_conv(features))
        return torch.clamp(illumination * ratio_map + residual, 0.0, 1.0)


class KinD(LLVModel):

    task = "llie"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize KinD.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default KinD configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update(
            {
                "decomposition_channels": 64,
                "decomposition_layers": 5,
                "restoration_channels": 32,
                "adjustment_channels": 32,
                "adjustment_layers": 3,
                "illumination_ratio": 5.0,
                "mode": "inference",
            }
        )
        return default_config

    def _validate_config(self) -> None:
        """Validate KinD configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()
        positive_keys = [
            "decomposition_channels",
            "decomposition_layers",
            "restoration_channels",
            "adjustment_channels",
            "adjustment_layers",
            "illumination_ratio",
        ]
        for key in positive_keys:
            if float(self.config[key]) <= 0:
                raise ValueError(f"'{key}' must be positive.")
        if self.config["mode"] not in {"train", "inference"}:
            raise ValueError("'mode' must be 'train' or 'inference'.")

    def _init_model(self) -> None:
        """Initialize KinD subnetworks."""
        image_channels = int(self.config["input_channels"])
        self.decomposition = KinDDecompositionNet(
            image_channels=image_channels,
            feature_channels=int(self.config["decomposition_channels"]),
            num_layers=int(self.config["decomposition_layers"]),
        )
        self.restoration = KinDRestorationNet(
            image_channels=image_channels,
            base_channels=int(self.config["restoration_channels"]),
        )
        self.adjustment = KinDIlluminationAdjustmentNet(
            feature_channels=int(self.config["adjustment_channels"]),
            num_layers=int(self.config["adjustment_layers"]),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a KinD forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.

        Returns:
            Training mode: standardized output dict with Retinex components.
            Inference mode: enhanced image tensor.
        """
        low_reflectance, low_illumination = self.decompose(x)
        ratio_map = torch.ones_like(low_illumination) * float(self.config["illumination_ratio"])
        restored_reflectance = self.restoration(low_reflectance, low_illumination)
        adjusted_illumination = self.adjustment(low_illumination, ratio_map)
        enhanced = torch.clamp(restored_reflectance * adjusted_illumination, 0.0, 1.0)

        if self.config["mode"] == "train":
            return self._format_output(
                pred=enhanced,
                aux={
                    "low_reflectance": low_reflectance,
                    "low_illumination": low_illumination,
                    "restored_reflectance": restored_reflectance,
                    "adjusted_illumination": adjusted_illumination,
                    "ratio_map": ratio_map,
                    "decompose_fn": self.decompose,
                },
                meta={"mode": self.config["mode"]},
            )

        return enhanced

    def decompose(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose an image with the shared decomposition network.

        Args:
            image: Input image tensor.

        Returns:
            Tuple containing reflectance and illumination tensors.
        """
        return self.decomposition(image)
