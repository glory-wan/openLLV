"""LEDNet supervised loss functions."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class VGG19PerceptualLoss(nn.Module):
    """VGG19 perceptual loss used by LEDNet.

    The original LEDNet BasicSR config uses VGG19 features before relu at
    conv1_2, conv2_2, conv3_4 and conv4_4, with L1 distance and weight 0.01.
    """

    DEFAULT_LAYER_WEIGHTS = {
        "conv1_2": 1.0,
        "conv2_2": 1.0,
        "conv3_4": 1.0,
        "conv4_4": 1.0,
    }

    _LAYER_TO_INDEX = {
        "conv1_2": 2,
        "conv2_2": 7,
        "conv3_4": 16,
        "conv4_4": 25,
    }

    def __init__(
        self,
        layer_weights: Optional[Dict[str, float]] = None,
        *,
        perceptual_weight: float = 0.01,
        pretrained: bool = True,
        use_input_norm: bool = True,
        range_norm: bool = False,
    ) -> None:
        """Initialize VGG19 perceptual loss.

        Args:
            layer_weights: Optional mapping from VGG layer names to weights.
            perceptual_weight: Global weight applied to the perceptual term.
            pretrained: Whether to load ImageNet-pretrained VGG19 weights.
            use_input_norm: Whether to normalize inputs with ImageNet mean and
                standard deviation.
            range_norm: Whether to map inputs from ``[-1, 1]`` to ``[0, 1]``.
        """
        super().__init__()
        self.layer_weights = dict(layer_weights or self.DEFAULT_LAYER_WEIGHTS)
        self.perceptual_weight = float(perceptual_weight)
        self.use_input_norm = bool(use_input_norm)
        self.range_norm = bool(range_norm)
        self.max_index = max(self._LAYER_TO_INDEX[name] for name in self.layer_weights)

        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg19(weights=weights).features[: self.max_index + 1].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute VGG19 perceptual loss.

        Args:
            prediction: Predicted image tensor.
            target: Target image tensor.

        Returns:
            Scalar perceptual loss tensor.
        """
        prediction = self._preprocess(prediction)
        target = self._preprocess(target.detach())

        loss = prediction.new_tensor(0.0)
        pred_feature = prediction
        target_feature = target

        index_to_name = {index: name for name, index in self._LAYER_TO_INDEX.items()}
        for index, layer in enumerate(self.vgg):
            pred_feature = layer(pred_feature)
            target_feature = layer(target_feature)
            layer_name = index_to_name.get(index)
            if layer_name in self.layer_weights:
                loss = loss + F.l1_loss(pred_feature, target_feature) * self.layer_weights[layer_name]

        return loss * self.perceptual_weight

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare image tensor for VGG19 features.

        Args:
            image: Image tensor with shape ``[B, C, H, W]``.

        Returns:
            Three-channel normalized image tensor.
        """
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] != 3:
            image = image[:, :3]

        if self.range_norm:
            image = (image + 1.0) / 2.0
        image = image.clamp(0.0, 1.0)

        if self.use_input_norm:
            image = (image - self.mean) / self.std
        return image


class LEDNet_Loss(BaseLoss):
    """LEDNet loss from "LEDNet: Joint Low-light Enhancement and Deblurring in the Dark".

    Original form:
        L = L_deb + side_loss_weight * L_en

    where each term is:
        L1(output, gt) + perceptual_weight * VGG19Perceptual(output, gt)

    In the original released BasicSR config, perceptual_weight=0.01,
    side_loss_weight=0.8, and VGG layers are conv1_2, conv2_2, conv3_4,
    conv4_4.
    """

    name = "lednet"
    aliases = ["lednet_loss", "LEDNet_Loss"]
    requires_target = True

    def __init__(
        self,
        *,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.01,
        side_loss_weight: float = 0.8,
        use_side_loss: bool = True,
        use_perceptual: bool = True,
        pretrained_vgg: bool = True,
        layer_weights: Optional[Dict[str, float]] = None,
        range_norm: bool = False,
        use_input_norm: bool = True,
    ) -> None:
        """Initialize LEDNet loss.

        Args:
            pixel_weight: Weight for the L1 pixel loss.
            perceptual_weight: Weight for the VGG perceptual loss.
            side_loss_weight: Weight for side-output supervision.
            use_side_loss: Whether to supervise LEDNet side output when present.
            use_perceptual: Whether to enable VGG perceptual loss.
            pretrained_vgg: Whether to load ImageNet-pretrained VGG19 weights.
            layer_weights: Optional VGG layer weights.
            range_norm: Whether to map perceptual inputs from ``[-1, 1]`` to
                ``[0, 1]``.
            use_input_norm: Whether to apply ImageNet normalization before VGG.
        """
        super().__init__()
        self.pixel_weight = float(pixel_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.side_loss_weight = float(side_loss_weight)
        self.use_side_loss = bool(use_side_loss)
        self.use_perceptual = bool(use_perceptual) and self.perceptual_weight > 0

        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = None
        if self.use_perceptual:
            self.perceptual_loss = VGG19PerceptualLoss(
                layer_weights=layer_weights,
                perceptual_weight=self.perceptual_weight,
                pretrained=pretrained_vgg,
                use_input_norm=use_input_norm,
                range_norm=range_norm,
            )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        side_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute LEDNet supervised loss.

        Args:
            prediction: Main prediction tensor.
            target: Ground-truth normal-light tensor.
            side_output: Optional side output tensor.

        Returns:
            Scalar LEDNet loss tensor.
        """
        total_loss = self._single_output_loss(prediction, target)

        if self.use_side_loss and side_output is not None:
            side_target = F.interpolate(
                target,
                size=side_output.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
            total_loss = total_loss + self.side_loss_weight * self._single_output_loss(
                side_output,
                side_target,
            )

        return total_loss

    def compute(
        self,
        *,
        input_tensor: torch.Tensor,
        model_output: Any,
        target: Optional[torch.Tensor] = None,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        align_prediction: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute LEDNet loss through the Trainer interface.

        Args:
            input_tensor: Low-light input tensor. It is unused by this
                supervised loss but kept for the ``BaseLoss`` interface.
            model_output: Raw LEDNet model output.
            target: Ground-truth normal-light tensor.
            extract_prediction: Optional callback for extracting predictions
                from non-standard outputs.
            align_prediction: Optional callback for aligning prediction shape
                with the target.

        Returns:
            A tuple containing the scalar loss tensor and prediction tensor.

        Raises:
            ValueError: If ``target`` is missing.
            TypeError: If prediction cannot be extracted.
        """
        if target is None:
            raise ValueError("LEDNet_Loss requires a target tensor.")

        prediction = self._extract_prediction(model_output, target, extract_prediction)
        if align_prediction is not None:
            prediction = align_prediction(prediction, target)

        side_output = self._extract_side_output(model_output)
        loss = self(prediction, target, side_output=side_output)
        return loss, prediction

    def _single_output_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss for one LEDNet output tensor.

        Args:
            prediction: Prediction tensor.
            target: Target tensor.

        Returns:
            Scalar loss tensor.
        """
        loss = self.pixel_weight * self.pixel_loss(prediction, target)
        if self.perceptual_loss is not None:
            loss = loss + self.perceptual_loss(prediction, target)
        return loss

    @staticmethod
    def _extract_prediction(
        model_output: Any,
        target: torch.Tensor,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        """Extract prediction tensor from LEDNet model output.

        Args:
            model_output: Raw model output.
            target: Target tensor passed to an optional extractor.
            extract_prediction: Optional fallback extractor callback.

        Returns:
            Prediction tensor.

        Raises:
            TypeError: If prediction cannot be extracted.
        """
        if torch.is_tensor(model_output):
            return model_output

        if isinstance(model_output, dict):
            prediction = model_output.get("pred")
            if torch.is_tensor(prediction):
                return prediction

        if extract_prediction is not None:
            return extract_prediction(model_output, target)

        raise TypeError(f"Cannot extract LEDNet prediction from {type(model_output).__name__}.")

    @staticmethod
    def _extract_side_output(model_output: Any) -> Optional[torch.Tensor]:
        """Extract optional LEDNet side output.

        Args:
            model_output: Raw model output.

        Returns:
            Side output tensor, or None if unavailable.
        """
        if not isinstance(model_output, dict):
            return None

        loss_inputs = get_loss_inputs(model_output)
        if isinstance(loss_inputs, dict):
            side_output = loss_inputs.get("side_output")
            if torch.is_tensor(side_output):
                return side_output

        side_output = model_output.get("side_output")
        if torch.is_tensor(side_output):
            return side_output

        return None


LEDNetLoss = LEDNet_Loss
