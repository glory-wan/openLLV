"""LLFlow loss functions."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class LLFlow_Loss(BaseLoss):
    """LLFlow negative log-likelihood loss.

    LLFlow models the conditional distribution of normally exposed images with
    an invertible network. The main training objective is the flow negative
    log-likelihood of the normal-light target conditioned on the low-light
    image. Optional reconstruction, color, and smoothness terms are included to
    stabilize openLLV's unified single-stage training interface.
    """

    name = "llflow"
    aliases = ["llflow_loss", "low_light_flow", "normalizing_flow_loss"]
    requires_target = True

    def __init__(
        self,
        *,
        nll_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        color_weight: float = 0.1,
        tv_weight: float = 0.01,
        include_constant: bool = False,
    ) -> None:
        """Initialize LLFlow loss.

        Args:
            nll_weight: Weight for flow negative log-likelihood.
            reconstruction_weight: Weight for deterministic zero-latent
                reconstruction loss.
            color_weight: Weight for mean-color consistency.
            tv_weight: Weight for total variation smoothness.
            include_constant: Whether to include the Gaussian constant term in
                the NLL value.
        """
        super().__init__()
        self.nll_weight = float(nll_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.color_weight = float(color_weight)
        self.tv_weight = float(tv_weight)
        self.include_constant = bool(include_constant)

    def compute(
        self,
        *,
        input_tensor: torch.Tensor,
        model_output: Any,
        target: Optional[torch.Tensor] = None,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        align_prediction: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute LLFlow loss through the Trainer interface.

        Args:
            input_tensor: Low-light input tensor.
            model_output: Raw LLFlow model output.
            target: Paired normal-light target tensor.
            extract_prediction: Optional prediction extractor.
            align_prediction: Optional prediction alignment callback.

        Returns:
            Tuple containing scalar loss and prediction tensor.

        Raises:
            ValueError: If target or flow helpers are missing.
        """
        if target is None:
            raise ValueError("LLFlow_Loss requires a paired normal-light target tensor.")
        if not isinstance(model_output, dict):
            raise ValueError("LLFlow_Loss expects LLFlow training output dictionary.")

        prediction = self._extract_prediction(model_output, target, extract_prediction)
        if align_prediction is not None:
            prediction = align_prediction(prediction, target)

        loss_inputs = get_loss_inputs(model_output)
        condition = loss_inputs.get("condition")
        flow_forward = loss_inputs.get("flow_forward")
        if condition is None or flow_forward is None:
            raise ValueError("LLFlow model output must provide condition and flow_forward.")

        target = target.clamp(0.0, 1.0)
        prediction = prediction.clamp(0.0, 1.0)
        latent, logdet = flow_forward(target, condition)

        loss = prediction.new_tensor(0.0)
        if self.nll_weight > 0:
            loss = loss + self.nll_weight * self._negative_log_likelihood(latent, logdet)
        if self.reconstruction_weight > 0:
            loss = loss + self.reconstruction_weight * F.l1_loss(prediction, target)
        if self.color_weight > 0:
            loss = loss + self.color_weight * self._color_loss(prediction, target)
        if self.tv_weight > 0:
            loss = loss + self.tv_weight * self._tv_loss(prediction)

        return loss, prediction

    def _negative_log_likelihood(self, latent: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel Gaussian negative log-likelihood.

        Args:
            latent: Latent tensor.
            logdet: Per-sample log determinant.

        Returns:
            Scalar NLL loss.
        """
        num_dims = latent[0].numel()
        nll = 0.5 * latent.flatten(start_dim=1).pow(2).sum(dim=1) - logdet
        if self.include_constant:
            constant = 0.5 * num_dims * torch.log(latent.new_tensor(2.0 * 3.141592653589793))
            nll = nll + constant
        return (nll / max(1, num_dims)).mean()

    @staticmethod
    def _color_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean-color consistency loss.

        Args:
            prediction: Enhanced image tensor.
            target: Target image tensor.

        Returns:
            Scalar color loss.
        """
        pred_mean = prediction.mean(dim=(-2, -1))
        target_mean = target.mean(dim=(-2, -1))
        return F.l1_loss(pred_mean, target_mean)

    @staticmethod
    def _tv_loss(image: torch.Tensor) -> torch.Tensor:
        """Compute total variation smoothness loss.

        Args:
            image: Input image tensor.

        Returns:
            Scalar TV loss.
        """
        loss = image.new_tensor(0.0)
        if image.shape[-2] > 1:
            loss = loss + torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
        if image.shape[-1] > 1:
            loss = loss + torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
        return loss

    @staticmethod
    def _extract_prediction(
        model_output: Dict[str, Any],
        target: torch.Tensor,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        """Extract prediction tensor from LLFlow output.

        Args:
            model_output: Raw model output.
            target: Target tensor used by fallback extractor.
            extract_prediction: Optional fallback extractor.

        Returns:
            Prediction tensor.

        Raises:
            TypeError: If prediction cannot be extracted.
        """
        if torch.is_tensor(model_output.get("pred")):
            return model_output["pred"]
        if extract_prediction is not None:
            return extract_prediction(model_output, target)
        raise TypeError("Cannot extract LLFlow prediction tensor.")


LLFlowLoss = LLFlow_Loss
