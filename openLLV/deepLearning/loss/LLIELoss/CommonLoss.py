"""Common reconstruction losses used by low-light enhancement models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..BaseLoss import BaseLoss


class L1Loss(BaseLoss):
    """Registered wrapper around :class:`torch.nn.L1Loss`."""

    name = "l1"
    aliases = ["mae", "l1_loss"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.loss = nn.L1Loss(**kwargs)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the L1 reconstruction loss."""
        return self.loss(prediction, target)


class MSELoss(BaseLoss):
    """Registered wrapper around :class:`torch.nn.MSELoss`."""

    name = "mse"
    aliases = ["l2", "mse_loss"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.loss = nn.MSELoss(**kwargs)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean squared reconstruction loss."""
        return self.loss(prediction, target)


class SmoothL1Loss(BaseLoss):
    """Registered wrapper around :class:`torch.nn.SmoothL1Loss`."""

    name = "smooth_l1"
    aliases = ["smoothl1", "huber", "smooth_l1_loss"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.loss = nn.SmoothL1Loss(**kwargs)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the smooth L1 reconstruction loss."""
        return self.loss(prediction, target)


class CharbonnierLoss(BaseLoss):
    """Charbonnier loss for paired image restoration."""

    name = "charbonnier"
    aliases = ["charbonnier_loss"]

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.eps = float(eps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Charbonnier reconstruction loss."""
        return torch.mean(
            torch.sqrt((prediction - target) ** 2 + self.eps ** 2)
        )


__all__ = ["L1Loss", "MSELoss", "SmoothL1Loss", "CharbonnierLoss"]
