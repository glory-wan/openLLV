"""

KinD++ model for low-light enhancement beyond brightening.

Original paper: Beyond Brightening Low-light Images
Paper link: https://doi.org/10.1007/s11263-020-01407-x
Official source code: https://github.com/zhangyhuaee/KinD_plus

"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseModel import LLVModel


class KinDPPConvBlock(nn.Module):
    """Convolution, normalization, and activation block used by KinD++."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_norm: bool = False,
    ) -> None:
        """Initialize a KinD++ convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            use_norm: Whether to use batch normalization.
        """
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.block(x)


class KinDPPDecompositionNet(nn.Module):
    """KinD++ decomposition network."""

    def __init__(self, image_channels: int = 3, base_channels: int = 32) -> None:
        """Initialize decomposition network.

        Args:
            image_channels: Number of image channels.
            base_channels: Base number of feature channels.
        """
        super().__init__()
        self.image_channels = image_channels
        self.conv1 = KinDPPConvBlock(image_channels, base_channels)
        self.conv2 = KinDPPConvBlock(base_channels, base_channels * 2)
        self.conv3 = KinDPPConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = KinDPPConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up1 = KinDPPConvBlock(base_channels * 2 + base_channels, base_channels)
        self.reflectance_out = nn.Conv2d(base_channels, image_channels, kernel_size=1)
        self.illumination_conv = KinDPPConvBlock(base_channels, base_channels)
        self.illumination_out = nn.Conv2d(base_channels * 2, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose an image into reflectance and illumination.

        Args:
            image: Input image tensor with shape ``[B, C, H, W]``.

        Returns:
            Tuple containing reflectance and illumination tensors.
        """
        conv1 = self.conv1(image)
        conv2 = self.conv2(F.max_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.max_pool2d(conv2, kernel_size=2, stride=2))

        up2 = F.interpolate(conv3, size=conv2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.up2(torch.cat([up2, conv2], dim=1))
        up1 = F.interpolate(up2, size=conv1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up1(torch.cat([up1, conv1], dim=1))

        reflectance = torch.sigmoid(self.reflectance_out(up1))
        illumination_feature = self.illumination_conv(conv1)
        illumination = torch.sigmoid(self.illumination_out(torch.cat([illumination_feature, up1], dim=1)))
        return reflectance, illumination


class KinDPPMSIA(nn.Module):
    """Multi-scale illumination attention module from KinD++."""

    def __init__(self, channels: int, *, use_norm: bool = True) -> None:
        """Initialize MSIA.

        Args:
            channels: Number of feature channels.
            use_norm: Whether multi-scale branches use batch normalization.
        """
        super().__init__()
        self.attention = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.scale1 = self._scale_branch(channels, channels, kernel_size=3, use_norm=use_norm)
        self.scale2 = self._scale_branch(channels, channels, kernel_size=3, use_norm=use_norm)
        self.scale4 = self._scale_branch(channels, channels, kernel_size=1, use_norm=use_norm)
        self.fusion = nn.Conv2d(channels * 4, channels, kernel_size=1)

    def forward(self, feature: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale illumination attention.

        Args:
            feature: Input feature tensor.
            illumination: Illumination tensor with shape ``[B, 1, H, W]``.

        Returns:
            Attention-refined multi-scale feature tensor.
        """
        if illumination.shape[-2:] != feature.shape[-2:]:
            illumination = F.interpolate(
                illumination,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        attended = feature * torch.sigmoid(self.attention(illumination))
        scale1 = self.scale1(attended)
        scale2 = self._pool_process_upsample(attended, self.scale2, level=2)
        scale4 = self._pool_process_upsample(attended, self.scale4, level=4)
        return self.fusion(torch.cat([attended, scale1, scale2, scale4], dim=1))

    @staticmethod
    def _scale_branch(
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        use_norm: bool,
    ) -> nn.Sequential:
        """Create a scale-processing branch.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            use_norm: Whether to use batch normalization.

        Returns:
            Sequential branch module.
        """
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def _pool_process_upsample(
        feature: torch.Tensor,
        branch: nn.Module,
        *,
        level: int,
    ) -> torch.Tensor:
        """Pool, process, and upsample one MSIA branch.

        Args:
            feature: Input feature tensor.
            branch: Branch module.
            level: Pooling level.

        Returns:
            Branch output resized to the source feature size.
        """
        pooled = F.max_pool2d(feature, kernel_size=level, stride=level, ceil_mode=True)
        processed = branch(pooled)
        return F.interpolate(processed, size=feature.shape[-2:], mode="bilinear", align_corners=False)


class KinDPPRestorationNet(nn.Module):
    """KinD++ reflectance restoration network with MSIA blocks."""

    def __init__(self, image_channels: int = 3, base_channels: int = 32) -> None:
        """Initialize restoration network.

        Args:
            image_channels: Number of image channels.
            base_channels: Base number of feature channels.
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            KinDPPConvBlock(image_channels, base_channels),
            KinDPPConvBlock(base_channels, base_channels * 2),
        )
        self.msia1 = KinDPPMSIA(base_channels * 2)
        self.conv2 = nn.Sequential(
            KinDPPConvBlock(base_channels * 2, base_channels * 4),
            KinDPPConvBlock(base_channels * 4, base_channels * 8),
        )
        self.msia2 = KinDPPMSIA(base_channels * 8)
        self.conv3 = nn.Sequential(
            KinDPPConvBlock(base_channels * 8, base_channels * 16),
            KinDPPConvBlock(base_channels * 16, base_channels * 8),
        )
        self.msia3 = KinDPPMSIA(base_channels * 8)
        self.conv4 = nn.Sequential(
            KinDPPConvBlock(base_channels * 8, base_channels * 4),
            KinDPPConvBlock(base_channels * 4, base_channels * 2),
        )
        self.msia4 = KinDPPMSIA(base_channels * 2)
        self.conv5 = KinDPPConvBlock(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, reflectance: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        """Restore reflectance under illumination guidance.

        Args:
            reflectance: Low-light reflectance tensor.
            illumination: Low-light illumination tensor.

        Returns:
            Restored reflectance tensor.
        """
        out = self.msia1(self.conv1(reflectance), illumination)
        out = self.msia2(self.conv2(out), illumination)
        out = self.msia3(self.conv3(out), illumination)
        out = self.msia4(self.conv4(out), illumination)
        out = self.conv5(out)
        return torch.sigmoid(self.out_conv(out))


class KinDPPIlluminationAdjustmentNet(nn.Module):
    """KinD++ illumination adjustment network."""

    def __init__(self, feature_channels: int = 32) -> None:
        """Initialize illumination adjustment network.

        Args:
            feature_channels: Number of intermediate feature channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            KinDPPConvBlock(2, feature_channels),
            KinDPPConvBlock(feature_channels, feature_channels),
            KinDPPConvBlock(feature_channels, feature_channels),
            nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, illumination: torch.Tensor, ratio_map: torch.Tensor) -> torch.Tensor:
        """Adjust an illumination map.

        Args:
            illumination: Source illumination tensor.
            ratio_map: Exposure-ratio tensor.

        Returns:
            Adjusted illumination tensor.
        """
        return self.net(torch.cat([illumination, ratio_map], dim=1))


class KinDPlusPlus(LLVModel):

    task = "llie"
    aliases = ['Kind++']

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize KinD++.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default KinD++ configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update(
            {
                "decomposition_channels": 32,
                "restoration_channels": 32,
                "adjustment_channels": 32,
                "illumination_ratio": 5.0,
                "mode": "inference",
            }
        )
        return default_config

    def _validate_config(self) -> None:
        """Validate KinD++ configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()
        for key in (
            "decomposition_channels",
            "restoration_channels",
            "adjustment_channels",
            "illumination_ratio",
        ):
            if float(self.config[key]) <= 0:
                raise ValueError(f"'{key}' must be positive.")
        if self.config["mode"] not in {"train", "inference"}:
            raise ValueError("'mode' must be 'train' or 'inference'.")

    def _init_model(self) -> None:
        """Initialize KinD++ subnetworks."""
        image_channels = int(self.config["input_channels"])
        self.decomposition = KinDPPDecompositionNet(
            image_channels=image_channels,
            base_channels=int(self.config["decomposition_channels"]),
        )
        self.restoration = KinDPPRestorationNet(
            image_channels=image_channels,
            base_channels=int(self.config["restoration_channels"]),
        )
        self.adjustment = KinDPPIlluminationAdjustmentNet(
            feature_channels=int(self.config["adjustment_channels"]),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution and normalization layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a KinD++ forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.

        Returns:
            Training mode: standardized output dict with KinD++ intermediate
            tensors. Inference mode: enhanced image tensor.
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
        """Decompose an image with KinD++ DecomNet.

        Args:
            image: Input image tensor.

        Returns:
            Tuple containing reflectance and illumination tensors.
        """
        return self.decomposition(image)
