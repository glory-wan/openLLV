"""Official LLFlow conditional Gaussian negative log-likelihood adapter."""

import torch

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class LLFlow_Loss(BaseLoss):
    """Mean official NLL; no reconstruction, color, or TV penalties."""

    name = "llflow"
    aliases = ["llflow_loss", "low_light_flow", "normalizing_flow_loss"]
    requires_target = True

    def forward(self, model_output, target):
        """Reduce the per-image NLL produced by a paired LLFlow forward."""
        if target is None:
            raise ValueError("LLFlow_Loss requires a paired normal-light target tensor.")
        nll = get_loss_inputs(model_output).get("nll")
        if not torch.is_tensor(nll) or nll.ndim != 1 or nll.shape[0] != target.shape[0]:
            raise ValueError("LLFlow training output must contain one aux.nll value per sample.")
        return nll.mean()

    def compute(self, *, input_tensor, model_output, target=None,
                extract_prediction=None, align_prediction=None):
        return self(model_output, target), model_output.get("pred")


LLFlowLoss = LLFlow_Loss
