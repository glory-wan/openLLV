"""
LLFlow model for low-light enhancement with normalizing flow.

Original paper: Low-Light Image Enhancement with Normalizing Flow
Paper link: https://doi.org/10.1609/aaai.v36i3.20162
Official source code: https://github.com/wyf0912/LLFlow
Official project url: https://wyf0912.github.io/LLFlow/
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseModel import LLVModel


class LLFlowConditionEncoder(nn.Module):
    """Conditional encoder for low-light image features."""

    def __init__(
        self,
        input_channels: int = 3,
        condition_channels: int = 32,
        num_blocks: int = 4,
    ) -> None:
        """Initialize the conditional encoder.

        Args:
            input_channels: Number of input image channels.
            condition_channels: Number of conditional feature channels.
            num_blocks: Number of residual convolution blocks.
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, condition_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(condition_channels, condition_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(condition_channels, condition_channels, kernel_size=3, padding=1),
                )
                for _ in range(num_blocks)
            ]
        )
        self.tail = nn.Sequential(
            nn.Conv2d(condition_channels, condition_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract conditional features.

        Args:
            image: Low-light input image tensor.

        Returns:
            Conditional feature tensor.
        """
        features = self.head(image)
        for block in self.blocks:
            features = features + block(features)
        return self.tail(features)


class LLFlowAffineCoupling(nn.Module):
    """Conditional affine coupling layer."""

    def __init__(
        self,
        channels: int,
        condition_channels: int,
        hidden_channels: int,
        *,
        flip: bool = False,
        scale_clamp: float = 2.0,
    ) -> None:
        """Initialize an affine coupling layer.

        Args:
            channels: Number of image channels transformed by the flow.
            condition_channels: Number of condition feature channels.
            hidden_channels: Number of hidden coupling network channels.
            flip: Whether to flip channel order before and after coupling.
            scale_clamp: Clamp value for log-scale prediction.
        """
        super().__init__()
        self.channels = channels
        self.split_channels = channels // 2
        self.remaining_channels = channels - self.split_channels
        self.flip = flip
        self.scale_clamp = float(scale_clamp)
        self.net = nn.Sequential(
            nn.Conv2d(self.split_channels + condition_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, self.remaining_channels * 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward or reverse affine coupling.

        Args:
            value: Flow tensor.
            condition: Conditional feature tensor.
            reverse: Whether to run the inverse transform.

        Returns:
            Tuple containing transformed tensor and log determinant.
        """
        if self.flip:
            value = torch.flip(value, dims=[1])

        first, second = value[:, : self.split_channels], value[:, self.split_channels :]
        scale_shift = self.net(torch.cat([first, condition], dim=1))
        shift, log_scale = torch.chunk(scale_shift, chunks=2, dim=1)
        log_scale = self.scale_clamp * torch.tanh(log_scale / self.scale_clamp)

        if reverse:
            second = (second - shift) * torch.exp(-log_scale)
            logdet = -self._sum_logdet(log_scale)
        else:
            second = second * torch.exp(log_scale) + shift
            logdet = self._sum_logdet(log_scale)

        output = torch.cat([first, second], dim=1)
        if self.flip:
            output = torch.flip(output, dims=[1])
        return output, logdet

    @staticmethod
    def _sum_logdet(log_scale: torch.Tensor) -> torch.Tensor:
        """Sum log-scale values per sample.

        Args:
            log_scale: Log-scale tensor.

        Returns:
            Per-sample log determinant tensor.
        """
        return log_scale.flatten(start_dim=1).sum(dim=1)


class LLFlow(LLVModel):

    task = "llie"
    aliases = []

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize LLFlow.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default LLFlow configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update(
            {
                "condition_channels": 32,
                "condition_blocks": 4,
                "flow_layers": 8,
                "flow_hidden_channels": 64,
                "scale_clamp": 2.0,
                "sample_temperature": 0.0,
                "mode": "inference",
            }
        )
        return default_config

    def _validate_config(self) -> None:
        """Validate LLFlow configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()
        if int(self.config["input_channels"]) < 2:
            raise ValueError("'input_channels' must be at least 2 for affine coupling.")
        for key in ("condition_channels", "condition_blocks", "flow_layers", "flow_hidden_channels"):
            if int(self.config[key]) <= 0:
                raise ValueError(f"'{key}' must be positive.")
        if float(self.config["scale_clamp"]) <= 0:
            raise ValueError("'scale_clamp' must be positive.")
        if float(self.config["sample_temperature"]) < 0:
            raise ValueError("'sample_temperature' must be non-negative.")
        if self.config["mode"] not in {"train", "inference"}:
            raise ValueError("'mode' must be 'train' or 'inference'.")

    def _init_model(self) -> None:
        """Initialize LLFlow encoder and flow layers."""
        channels = int(self.config["input_channels"])
        condition_channels = int(self.config["condition_channels"])
        self.condition_encoder = LLFlowConditionEncoder(
            input_channels=channels,
            condition_channels=condition_channels,
            num_blocks=int(self.config["condition_blocks"]),
        )
        self.flow_layers = nn.ModuleList(
            [
                LLFlowAffineCoupling(
                    channels=channels,
                    condition_channels=condition_channels,
                    hidden_channels=int(self.config["flow_hidden_channels"]),
                    flip=bool(index % 2),
                    scale_clamp=float(self.config["scale_clamp"]),
                )
                for index in range(int(self.config["flow_layers"]))
            ]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.weight.abs().sum() > 0:
                    nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run an LLFlow forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.

        Returns:
            Training mode: standardized output dict with flow helpers.
            Inference mode: enhanced image tensor.
        """
        condition = self.condition_encoder(x)
        latent = self._sample_latent(x)
        enhanced = self.flow_reverse(latent, condition)

        if self.config["mode"] == "train":
            return self._format_output(
                pred=enhanced,
                aux={
                    "condition": condition,
                    "flow_forward": self.flow_forward,
                    "flow_reverse": self.flow_reverse,
                    "latent": latent,
                },
                meta={"mode": self.config["mode"]},
            )

        return enhanced

    def flow_forward(self, image: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map a normal-light image to latent space.

        Args:
            image: Normal-light image tensor in ``[0, 1]``.
            condition: Conditional features extracted from the low-light image.

        Returns:
            Tuple containing latent tensor and per-sample log determinant.
        """
        value = self._logit(image)
        total_logdet = value.new_zeros(value.shape[0])
        for layer in self.flow_layers:
            value, logdet = layer(value, condition, reverse=False)
            total_logdet = total_logdet + logdet
        return value, total_logdet

    def flow_reverse(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Map latent variables to enhanced image space.

        Args:
            latent: Latent tensor.
            condition: Conditional low-light features.

        Returns:
            Enhanced image tensor in ``[0, 1]``.
        """
        value = latent
        for layer in reversed(self.flow_layers):
            value, _ = layer(value, condition, reverse=True)
        return torch.sigmoid(value)

    def _sample_latent(self, reference: torch.Tensor) -> torch.Tensor:
        """Sample or create latent tensor for inference.

        Args:
            reference: Reference image tensor.

        Returns:
            Latent tensor with the same shape as ``reference``.
        """
        temperature = float(self.config["sample_temperature"])
        if temperature == 0:
            return torch.zeros_like(reference)
        return torch.randn_like(reference) * temperature

    @staticmethod
    def _logit(image: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Convert image values from ``[0, 1]`` to logit space.

        Args:
            image: Image tensor.
            eps: Clamp value for numerical stability.

        Returns:
            Logit-space tensor.
        """
        image = image.clamp(eps, 1.0 - eps)
        return torch.log(image) - torch.log1p(-image)
