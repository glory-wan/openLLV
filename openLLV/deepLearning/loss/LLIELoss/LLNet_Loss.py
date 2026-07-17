"""LLNet supervised loss functions."""

from typing import Any, Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class LLNet_Loss(BaseLoss):
    """LLNet reconstruction loss with sparse autoencoder regularization.

    The LLNet paper trains denoising autoencoders with reconstruction error,
    KL-divergence sparsity regularization, and L2 weight regularization, then
    fine-tunes the stacked model with reconstruction error and weight decay. This
    implementation exposes those terms in one openLLV-compatible supervised
    loss.
    """

    name = "llnet"
    aliases = ["llnet_loss", "LLNet_Loss", "sparse_denoising_autoencoder"]
    requires_target = True

    def __init__(
        self,
        *,
        reconstruction_weight: float = 1.0,
        sparsity_weight: float = 0.001,
        weight_decay: float = 1e-5,
        target_sparsity: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        """Initialize LLNet loss.

        Args:
            reconstruction_weight: Weight for supervised MSE reconstruction.
            sparsity_weight: Weight for KL sparsity regularization.
            weight_decay: Weight for L2 regularization over autoencoder weights.
            target_sparsity: Target average hidden activation.
            eps: Small value used to stabilize logarithms.
        """
        super().__init__()
        self.reconstruction_weight = float(reconstruction_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.weight_decay = float(weight_decay)
        self.target_sparsity = float(target_sparsity)
        self.eps = float(eps)
        self.reconstruction_loss = nn.MSELoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        hidden_activations: Optional[Iterable[torch.Tensor]] = None,
        weight_tensors: Optional[Iterable[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute LLNet loss.

        Args:
            prediction: Enhanced image tensor.
            target: Paired normal-light target tensor.
            hidden_activations: Optional encoder activations for sparsity loss.
            weight_tensors: Optional model weights for L2 regularization.

        Returns:
            Scalar loss tensor.
        """
        prediction = prediction.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        loss = self.reconstruction_weight * self.reconstruction_loss(prediction, target)

        if hidden_activations is not None and self.sparsity_weight > 0:
            loss = loss + self.sparsity_weight * self._kl_sparsity(hidden_activations, prediction)

        if weight_tensors is not None and self.weight_decay > 0:
            loss = loss + self.weight_decay * self._weight_l2(weight_tensors, prediction)

        return loss

    def compute(
        self,
        *,
        input_tensor: torch.Tensor,
        model_output: Any,
        target: Optional[torch.Tensor] = None,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        align_prediction: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute LLNet loss through the Trainer interface.

        Args:
            input_tensor: Low-light input tensor.
            model_output: Raw LLNet model output.
            target: Paired normal-light target tensor.
            extract_prediction: Optional callback for extracting prediction.
            align_prediction: Optional callback for aligning prediction shape
                with target.

        Returns:
            Tuple containing scalar loss and prediction tensor.

        Raises:
            ValueError: If target is missing.
            TypeError: If prediction cannot be extracted.
        """
        if target is None:
            raise ValueError("LLNet_Loss requires a paired normal-light target tensor.")

        prediction = self._extract_prediction(model_output, target, extract_prediction)
        if align_prediction is not None:
            prediction = align_prediction(prediction, target)

        loss_inputs = get_loss_inputs(model_output)
        hidden_activations = loss_inputs.get("hidden_activations")
        weight_tensors = loss_inputs.get("weight_tensors")

        return (
            self(
                prediction,
                target,
                hidden_activations=hidden_activations,
                weight_tensors=weight_tensors,
            ),
            prediction,
        )

    def _kl_sparsity(
        self,
        hidden_activations: Iterable[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL sparsity regularization over hidden activations.

        Args:
            hidden_activations: Encoder activation tensors.
            reference: Tensor used for device and dtype placement.

        Returns:
            Scalar sparsity loss.
        """
        losses: List[torch.Tensor] = []
        rho = reference.new_tensor(self.target_sparsity).clamp(self.eps, 1.0 - self.eps)
        one = reference.new_tensor(1.0)

        for activation in hidden_activations:
            if not torch.is_tensor(activation):
                continue
            activation = activation.to(device=reference.device, dtype=reference.dtype)
            dims = tuple(range(activation.dim() - 1))
            rho_hat = activation.mean(dim=dims).clamp(self.eps, 1.0 - self.eps)
            kl = rho * torch.log(rho / rho_hat) + (one - rho) * torch.log(
                (one - rho) / (one - rho_hat)
            )
            losses.append(kl.mean())

        if not losses:
            return reference.new_tensor(0.0)
        return sum(losses) / len(losses)

    @staticmethod
    def _weight_l2(
        weight_tensors: Iterable[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean L2 regularization over model weights.

        Args:
            weight_tensors: Iterable of model weight tensors.
            reference: Tensor used for device and dtype placement.

        Returns:
            Scalar L2 regularization loss.
        """
        losses = [
            torch.mean(weight.to(device=reference.device, dtype=reference.dtype) ** 2)
            for weight in weight_tensors
            if torch.is_tensor(weight)
        ]
        if not losses:
            return reference.new_tensor(0.0)
        return sum(losses) / len(losses)

    @staticmethod
    def _extract_prediction(
        model_output: Any,
        target: torch.Tensor,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        """Extract prediction tensor from LLNet output.

        Args:
            model_output: Raw model output.
            target: Target tensor used by the fallback extractor.
            extract_prediction: Optional fallback extractor callback.

        Returns:
            Prediction tensor.

        Raises:
            TypeError: If prediction cannot be extracted.
        """
        if torch.is_tensor(model_output):
            return model_output
        if isinstance(model_output, dict) and torch.is_tensor(model_output.get("pred")):
            return model_output["pred"]
        if extract_prediction is not None:
            return extract_prediction(model_output, target)
        raise TypeError(f"Cannot extract LLNet prediction from {type(model_output).__name__}.")


LLNetLoss = LLNet_Loss
