"""PairLIE training objective."""

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class PairLIETotalVariationLoss(nn.Module):
    """Two-pixel finite-difference illumination regularizer."""

    def forward(self, illumination: torch.Tensor) -> torch.Tensor:
        if illumination.shape[-2] < 3 or illumination.shape[-1] < 3:
            raise ValueError("PairLIE TV loss requires spatial dimensions of at least 3x3.")
        vertical = (illumination[:, :, 2:, :] - illumination[:, :, :-2, :]).abs()
        horizontal = (illumination[:, :, :, 2:] - illumination[:, :, :, :-2]).abs()
        return vertical.mean() + horizontal.mean()


class PairLIE_Loss(BaseLoss):
    """Official PairLIE loss for two low-light instances of one scene.

    The objective combines reflectance consistency, Retinex reconstruction and
    illumination regularization, plus a strong input-preservation term for the
    learned denoised representation.
    """

    name = "pairlie"
    aliases = ["pairlie_loss", "PairLIE-Loss"]
    requires_target = True

    def __init__(
        self,
        consistency_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        preservation_weight: float = 500.0,
        division_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.consistency_weight = float(consistency_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.preservation_weight = float(preservation_weight)
        self.division_eps = float(division_eps)
        self.tv_loss = PairLIETotalVariationLoss()

    def forward(
        self,
        input_tensor: torch.Tensor,
        illumination: torch.Tensor,
        reflectance: torch.Tensor,
        denoised: torch.Tensor,
        paired_reflectance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PairLIE's three weighted loss groups."""
        consistency = F.mse_loss(reflectance, paired_reflectance)

        max_rgb = input_tensor.max(dim=1, keepdim=True).values
        reconstruction = (
            F.mse_loss(illumination * reflectance, denoised)
            + F.mse_loss(
                reflectance,
                denoised / illumination.detach().clamp_min(self.division_eps),
            )
            + F.mse_loss(illumination, max_rgb)
            + self.tv_loss(illumination)
        )
        preservation = F.mse_loss(input_tensor, denoised)
        return (
            self.consistency_weight * consistency
            + self.reconstruction_weight * reconstruction
            + self.preservation_weight * preservation
        )

    def compute(
        self,
        *,
        input_tensor: torch.Tensor,
        model_output: Any,
        target: Optional[torch.Tensor] = None,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        align_prediction: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute loss from PairLIE's standardized paired training output."""
        if target is None:
            raise ValueError("PairLIE_Loss requires a second low-light instance.")
        if not isinstance(model_output, dict):
            raise TypeError("PairLIE_Loss requires PairLIE's structured training output.")

        loss_inputs = get_loss_inputs(model_output)
        required = {"illumination", "reflectance", "denoised", "paired_reflectance"}
        missing = required - set(loss_inputs)
        if missing:
            raise KeyError("PairLIE training output is missing: {}".format(sorted(missing)))

        loss = self(
            input_tensor,
            loss_inputs["illumination"],
            loss_inputs["reflectance"],
            loss_inputs["denoised"],
            loss_inputs["paired_reflectance"],
        )

        prediction = model_output.get("pred")
        if not torch.is_tensor(prediction) and extract_prediction is not None:
            prediction = extract_prediction(model_output, target)
        if torch.is_tensor(prediction) and align_prediction is not None:
            prediction = align_prediction(prediction, target)
        return loss, prediction


PairLIELoss = PairLIE_Loss
