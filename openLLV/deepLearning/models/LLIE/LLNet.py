"""LLNet patch autoencoder for low-light enhancement and denoising.

    Original paper: LLNet: A Deep Autoencoder Approach to Natural Low-light Image Enhancement
    Paper link: https://doi.org/10.1016/j.patcog.2016.06.008
    Official source code: https://github.com/kglore/llnet_color

    """

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseModel import LLVModel


class LLNet(LLVModel):

    task = "llie"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize LLNet.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default LLNet configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update(
            {
                "patch_size": 17,
                "patch_stride": 3,
                "hidden_dims": [2000, 1600, 1200],
                "activation": "sigmoid",
                "output_activation": "sigmoid",
                "mode": "inference",
            }
        )
        return default_config

    def _validate_config(self) -> None:
        """Validate LLNet configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()

        if int(self.config["patch_size"]) <= 0:
            raise ValueError("'patch_size' must be positive.")
        if int(self.config["patch_stride"]) <= 0:
            raise ValueError("'patch_stride' must be positive.")
        if int(self.config["patch_size"]) % 2 == 0:
            raise ValueError("'patch_size' must be odd for symmetric image padding.")

        hidden_dims = self.config.get("hidden_dims")
        if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
            raise ValueError("'hidden_dims' must be a non-empty list or tuple.")
        if any(int(dim) <= 0 for dim in hidden_dims):
            raise ValueError("All LLNet hidden dimensions must be positive.")

        if self.config["activation"] not in {"sigmoid", "relu", "tanh"}:
            raise ValueError("'activation' must be 'sigmoid', 'relu', or 'tanh'.")
        if self.config["output_activation"] not in {"sigmoid", "clamp", "none"}:
            raise ValueError("'output_activation' must be 'sigmoid', 'clamp', or 'none'.")
        if self.config["mode"] not in {"train", "inference"}:
            raise ValueError("'mode' must be 'train' or 'inference'.")

    def _init_model(self) -> None:
        """Initialize LLNet encoder and decoder layers."""
        input_dim = (
            int(self.config["input_channels"])
            * int(self.config["patch_size"])
            * int(self.config["patch_size"])
        )
        hidden_dims = [int(dim) for dim in self.config["hidden_dims"]]
        encoder_dims = [input_dim, *hidden_dims]
        decoder_dims = [*reversed(hidden_dims), input_dim]

        self.encoder = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(encoder_dims[:-1], encoder_dims[1:])
        )
        self.decoder = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(decoder_dims[:-1], decoder_dims[1:])
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run an LLNet forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.

        Returns:
            Training mode: standardized output dict containing the enhanced
            image and hidden activations. Inference mode: enhanced image tensor.
        """
        enhanced, hidden_activations = self._enhance_image(x)

        if self.config["mode"] == "train":
            return self._format_output(
                pred=enhanced,
                aux={
                    "enhanced": enhanced,
                    "hidden_activations": hidden_activations,
                    "weight_tensors": self._regularized_weights(),
                },
                meta={
                    "mode": self.config["mode"],
                    "patch_size": int(self.config["patch_size"]),
                    "patch_stride": int(self.config["patch_stride"]),
                },
            )

        return enhanced

    def _enhance_image(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Enhance an image tensor by processing overlapping patches.

        Args:
            x: Input tensor with shape ``[B, C, H, W]``.

        Returns:
            Tuple containing the reconstructed image tensor and encoder hidden
            activations.
        """
        batch_size, channels, height, width = x.shape
        patch_size = int(self.config["patch_size"])
        stride = int(self.config["patch_stride"])
        pad = patch_size // 2

        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        padded_size = (height + 2 * pad, width + 2 * pad)

        patches = F.unfold(padded, kernel_size=patch_size, stride=stride)
        patches = patches.transpose(1, 2)
        patch_shape = patches.shape
        patches = patches.reshape(-1, channels * patch_size * patch_size)

        reconstructed, hidden_activations = self._forward_patches(patches)
        reconstructed = reconstructed.reshape(patch_shape).transpose(1, 2)

        output = F.fold(
            reconstructed,
            output_size=padded_size,
            kernel_size=patch_size,
            stride=stride,
        )
        normalizer = F.fold(
            torch.ones_like(reconstructed),
            output_size=padded_size,
            kernel_size=patch_size,
            stride=stride,
        ).clamp_min(1e-6)
        output = output / normalizer
        output = output[:, :, pad : pad + height, pad : pad + width]
        return output, hidden_activations

    def _forward_patches(self, patches: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the autoencoder on vectorized patches.

        Args:
            patches: Flattened patches with shape ``[N, C * P * P]``.

        Returns:
            Tuple containing reconstructed patches and encoder hidden
            activations.
        """
        hidden_activations: List[torch.Tensor] = []
        out = patches
        for layer in self.encoder:
            out = self._activate(layer(out))
            hidden_activations.append(out)

        for index, layer in enumerate(self.decoder):
            out = layer(out)
            if index < len(self.decoder) - 1:
                out = self._activate(out)

        out = self._activate_output(out)
        return out, hidden_activations

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the configured hidden activation.

        Args:
            value: Input activation tensor.

        Returns:
            Activated tensor.
        """
        activation = self.config["activation"]
        if activation == "sigmoid":
            return torch.sigmoid(value)
        if activation == "relu":
            return F.relu(value, inplace=False)
        return torch.tanh(value)

    def _activate_output(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the configured output activation.

        Args:
            value: Output patch tensor.

        Returns:
            Activated or clipped output patch tensor.
        """
        activation = self.config["output_activation"]
        if activation == "sigmoid":
            return torch.sigmoid(value)
        if activation == "clamp":
            return value.clamp(0.0, 1.0)
        return value

    def _regularized_weights(self) -> List[torch.Tensor]:
        """Collect weights used by LLNet regularization losses.

        Returns:
            List of encoder and decoder weight tensors.
        """
        weights: List[torch.Tensor] = []
        for layer in self.encoder:
            weights.append(layer.weight)
        for layer in self.decoder:
            weights.append(layer.weight)
        return weights
