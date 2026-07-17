"""LLFormer supervised training loss."""

import torch
import torch.nn as nn

from ..BaseLoss import BaseLoss


class LLFormer_Loss(BaseLoss):
    """Smooth L1 loss used by the official LLFormer training entrypoint."""

    name = "llformer"
    aliases = ["llformer_loss", "LLFormer-Loss"]
    requires_target = True

    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.SmoothL1Loss(beta=float(beta), reduction=reduction)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(prediction, target)


LLFormerLoss = LLFormer_Loss

