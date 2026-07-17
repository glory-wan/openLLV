"""

PairLIE decomposition and enhancement network.

    Original paper: Learning a Simple Low-light Image Enhancer from Paired Low-light Instances
    Paper link: https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.pdf
    Official source code: https://github.com/zhenqifu/PairLIE

"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..BaseModel import LLVModel


def _estimator_layers(
    input_channels: int,
    output_channels: int,
    feature_channels: int,
) -> nn.Sequential:
    """Build the five-layer estimator shared by PairLIE's three branches."""
    layers = []
    channels = input_channels
    for _ in range(4):
        layers.extend([
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, feature_channels, 3, stride=1, padding=0),
            nn.ReLU(),
        ])
        channels = feature_channels
    layers.extend([
        nn.ReflectionPad2d(1),
        nn.Conv2d(feature_channels, output_channels, 3, stride=1, padding=0),
    ])
    return nn.Sequential(*layers)


class IlluminationEstimator(nn.Module):
    """Estimate the single-channel illumination map."""

    def __init__(self, feature_channels: int = 64) -> None:
        super().__init__()
        # Keep the official module name for raw state-dict compatibility.
        self.L_net = _estimator_layers(3, 1, feature_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.L_net(image))


class ReflectanceEstimator(nn.Module):
    """Estimate the three-channel reflectance map."""

    def __init__(self, feature_channels: int = 64) -> None:
        super().__init__()
        self.R_net = _estimator_layers(3, 3, feature_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.R_net(image))


class NoiseEstimator(nn.Module):
    """Estimate the denoised low-light image used for decomposition."""

    def __init__(self, feature_channels: int = 64) -> None:
        super().__init__()
        self.N_net = _estimator_layers(3, 3, feature_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.N_net(image))


class PairLIE(LLVModel):

    task = "llie"
    aliases = ["Pair-LIE"]
    requires_paired_forward = True

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        config = super()._get_default_config()
        config.update({
            "feature_channels": 64,
            "enhancement_gamma": 0.2,
            "clamp_output": True,
            "mode": "inference",
        })
        return config

    def _validate_config(self) -> None:
        super()._validate_config()
        if int(self.config["input_channels"]) != 3:
            raise ValueError("PairLIE requires exactly three RGB input channels.")
        if int(self.config["feature_channels"]) <= 0:
            raise ValueError("'feature_channels' must be positive.")
        if float(self.config["enhancement_gamma"]) <= 0:
            raise ValueError("'enhancement_gamma' must be positive.")
        if self.config["mode"] not in {"train", "inference"}:
            raise ValueError("'mode' must be 'train' or 'inference'.")

    def _init_model(self) -> None:
        feature_channels = int(self.config["feature_channels"])
        self.L_net = IlluminationEstimator(feature_channels)
        self.R_net = ReflectanceEstimator(feature_channels)
        self.N_net = NoiseEstimator(feature_channels)

    def decompose(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return illumination, reflectance, and denoised representations."""
        denoised = self.N_net(image)
        illumination = self.L_net(denoised)
        reflectance = self.R_net(denoised)
        return illumination, reflectance, denoised

    def enhance(self, illumination: torch.Tensor, reflectance: torch.Tensor) -> torch.Tensor:
        """Compose the enhanced image from estimated components."""
        prediction = illumination.pow(float(self.config["enhancement_gamma"])) * reflectance
        if self.config["clamp_output"]:
            prediction = prediction.clamp(0.0, 1.0)
        return prediction

    def forward(
        self,
        image: torch.Tensor,
        paired_image: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run single-image inference or paired low-light training forward."""
        illumination, reflectance, denoised = self.decompose(image)
        prediction = self.enhance(illumination, reflectance)

        if self.config["mode"] != "train":
            return prediction

        aux = {
            "illumination": illumination,
            "reflectance": reflectance,
            "denoised": denoised,
            "noise": image - denoised,
        }
        if paired_image is not None:
            paired_illumination, paired_reflectance, paired_denoised = self.decompose(paired_image)
            aux.update({
                "paired_illumination": paired_illumination,
                "paired_reflectance": paired_reflectance,
                "paired_denoised": paired_denoised,
            })

        return self._format_output(
            prediction,
            aux=aux,
            meta={"mode": "train", "paired": paired_image is not None},
        )
