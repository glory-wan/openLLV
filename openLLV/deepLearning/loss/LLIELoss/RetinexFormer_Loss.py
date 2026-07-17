"""RetinexFormer supervised loss functions."""

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class RetinexFormerIlluminationConsistencyLoss(nn.Module):
    """Regularize RetinexFormer illumination maps across stages."""

    def forward(self, stage_outputs: Any) -> torch.Tensor:
        """Compute illumination consistency loss.

        Args:
            stage_outputs: List of RetinexFormer stage intermediate outputs.

        Returns:
            Scalar illumination consistency loss.
        """
        if not stage_outputs:
            return torch.tensor(0.0)

        losses = []
        for stage_output in stage_outputs:
            if not isinstance(stage_output, dict):
                continue
            illumination_map = stage_output.get("illumination_map")
            enhanced_input = stage_output.get("enhanced_input")
            if torch.is_tensor(illumination_map):
                losses.append(self._smoothness(illumination_map))
            if torch.is_tensor(illumination_map) and torch.is_tensor(enhanced_input):
                losses.append(F.l1_loss(torch.sigmoid(illumination_map), enhanced_input.clamp(0.0, 1.0)))

        if not losses:
            reference = self._find_reference_tensor(stage_outputs)
            return reference.new_tensor(0.0) if reference is not None else torch.tensor(0.0)

        return sum(losses) / len(losses)

    @staticmethod
    def _smoothness(image: torch.Tensor) -> torch.Tensor:
        """Compute simple total variation smoothness.

        Args:
            image: Input image or feature tensor.

        Returns:
            Scalar smoothness loss.
        """
        loss = image.new_tensor(0.0)
        if image.shape[-2] > 1:
            loss = loss + torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
        if image.shape[-1] > 1:
            loss = loss + torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
        return loss

    @staticmethod
    def _find_reference_tensor(stage_outputs: Any) -> Optional[torch.Tensor]:
        """Find a tensor from nested stage outputs.

        Args:
            stage_outputs: Nested stage outputs.

        Returns:
            First tensor found, or ``None``.
        """
        if torch.is_tensor(stage_outputs):
            return stage_outputs
        if isinstance(stage_outputs, dict):
            for value in stage_outputs.values():
                found = RetinexFormerIlluminationConsistencyLoss._find_reference_tensor(value)
                if found is not None:
                    return found
        if isinstance(stage_outputs, (list, tuple)):
            for value in stage_outputs:
                found = RetinexFormerIlluminationConsistencyLoss._find_reference_tensor(value)
                if found is not None:
                    return found
        return None


class RetinexFormer_Loss(BaseLoss):
    """RetinexFormer supervised training loss.

    The official RetinexFormer training configuration uses pixel-level L1 loss.
    This implementation keeps that behavior by default and adds an optional
    illumination consistency term for model outputs that expose stage
    intermediates.
    """

    name = "retinexformer"
    aliases = [
        "retinexformer_loss",
        "RetinexFormer_Loss",
        "retinexformer_l1",
    ]
    requires_target = True

    def __init__(
        self,
        *,
        pixel_weight: float = 1.0,
        illumination_weight: float = 0.0,
    ) -> None:
        """Initialize RetinexFormer loss.

        Args:
            pixel_weight: Weight for the supervised pixel L1 loss.
            illumination_weight: Weight for optional illumination consistency
                regularization.
        """
        super().__init__()
        self.pixel_weight = float(pixel_weight)
        self.illumination_weight = float(illumination_weight)
        self.pixel_loss = nn.L1Loss()
        self.illumination_loss = RetinexFormerIlluminationConsistencyLoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        stage_outputs: Any = None,
    ) -> torch.Tensor:
        """Compute RetinexFormer loss.

        Args:
            prediction: Enhanced image tensor.
            target: Paired normal-light target tensor.
            stage_outputs: Optional RetinexFormer stage intermediate outputs.

        Returns:
            Scalar loss tensor.
        """
        prediction = prediction.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        loss = self.pixel_weight * self.pixel_loss(prediction, target)
        if self.illumination_weight > 0 and stage_outputs is not None:
            illumination_loss = self.illumination_loss(stage_outputs).to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
            loss = loss + self.illumination_weight * illumination_loss
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
        """Compute RetinexFormer loss through the Trainer interface.

        Args:
            input_tensor: Low-light input tensor.
            model_output: Raw RetinexFormer model output.
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
            raise ValueError("RetinexFormer_Loss requires a paired normal-light target tensor.")

        prediction = self._extract_prediction(model_output, target, extract_prediction)
        if align_prediction is not None:
            prediction = align_prediction(prediction, target)

        stage_outputs = None
        if isinstance(model_output, dict):
            stage_outputs = get_loss_inputs(model_output).get("stage_outputs")

        return self(prediction, target, stage_outputs=stage_outputs), prediction

    @staticmethod
    def _extract_prediction(
        model_output: Any,
        target: torch.Tensor,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        """Extract prediction tensor from RetinexFormer output.

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
        raise TypeError(
            f"Cannot extract RetinexFormer prediction from {type(model_output).__name__}."
        )


RetinexFormerLoss = RetinexFormer_Loss
