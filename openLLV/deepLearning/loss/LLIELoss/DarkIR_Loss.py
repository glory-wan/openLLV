"""DarkIR supervised loss functions."""

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class DarkIRVGGFeatureLoss(nn.Module):
    """VGG19 feature loss used by the released DarkIR training code.

    The original repository computes VGG19 feature distances from five slices
    with weights [1/32, 1/16, 1/8, 1/4, 1]. The paper describes this term as
    the perceptual/LPIPS component.
    """

    FEATURE_WEIGHTS = (1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0)
    SLICE_RANGES = ((0, 2), (2, 7), (7, 12), (12, 21), (21, 30))

    def __init__(
        self,
        *,
        loss_weight: float = 0.01,
        criterion: str = "l1",
        pretrained: bool = True,
        use_input_norm: bool = True,
    ) -> None:
        """Initialize DarkIR VGG feature loss.

        Args:
            loss_weight: Global weight for the feature loss.
            criterion: Feature distance criterion, either ``"l1"`` or
                ``"l2"``/``"mse"``.
            pretrained: Whether to load ImageNet-pretrained VGG19 weights.
            use_input_norm: Whether to normalize inputs with ImageNet mean and
                standard deviation.

        Raises:
            ValueError: If ``criterion`` is unsupported.
        """
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.use_input_norm = bool(use_input_norm)

        if criterion == "l1":
            self.criterion = nn.L1Loss()
        elif criterion in {"l2", "mse"}:
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported VGG criterion: {criterion!r}.")

        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        features = models.vgg19(weights=weights).features.eval()
        self.slices = nn.ModuleList()
        for start, end in self.SLICE_RANGES:
            block = nn.Sequential(*[features[index] for index in range(start, end)])
            block.eval()
            for param in block.parameters():
                param.requires_grad = False
            self.slices.append(block)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute VGG feature loss.

        Args:
            prediction: Predicted image tensor.
            target: Target image tensor.

        Returns:
            Scalar feature loss tensor.
        """
        prediction = self._preprocess(prediction)
        target = self._preprocess(target.detach())

        loss = prediction.new_tensor(0.0)
        pred_feature = prediction
        target_feature = target
        for weight, block in zip(self.FEATURE_WEIGHTS, self.slices):
            pred_feature = block(pred_feature)
            target_feature = block(target_feature)
            loss = loss + weight * self.criterion(pred_feature, target_feature.detach())

        return self.loss_weight * loss

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare image tensor for VGG19 feature extraction.

        Args:
            image: Image tensor with shape ``[B, C, H, W]``.

        Returns:
            Three-channel normalized image tensor.
        """
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] != 3:
            image = image[:, :3]

        image = image.clamp(0.0, 1.0)
        if self.use_input_norm:
            image = (image - self.mean) / self.std
        return image


class DarkIREdgeLoss(nn.Module):
    """Laplacian edge loss from the DarkIR source code."""

    def __init__(self, *, loss_weight: float = 50.0, criterion: str = "l2") -> None:
        """Initialize DarkIR edge loss.

        Args:
            loss_weight: Global weight for the edge loss.
            criterion: Edge distance criterion, either ``"l1"`` or
                ``"l2"``/``"mse"``.

        Raises:
            ValueError: If ``criterion`` is unsupported.
        """
        super().__init__()
        self.loss_weight = float(loss_weight)
        if criterion == "l1":
            self.criterion = nn.L1Loss()
        elif criterion in {"l2", "mse"}:
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported edge criterion: {criterion!r}.")

        kernel_1d = torch.tensor([[0.05, 0.25, 0.4, 0.25, 0.05]], dtype=torch.float32)
        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel_2d)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian edge loss.

        Args:
            prediction: Predicted image tensor.
            target: Target image tensor.

        Returns:
            Scalar edge loss tensor.
        """
        return self.loss_weight * self.criterion(
            self._laplacian_kernel(prediction),
            self._laplacian_kernel(target),
        )

    def _conv_gauss(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian filtering used by the Laplacian pyramid.

        Args:
            image: Image tensor with shape ``[B, C, H, W]``.

        Returns:
            Filtered image tensor.
        """
        channels = image.shape[1]
        kernel = self.kernel.repeat(channels, 1, 1, 1)
        _, _, kernel_h, kernel_w = kernel.shape
        image = F.pad(
            image,
            (kernel_w // 2, kernel_w // 2, kernel_h // 2, kernel_h // 2),
            mode="replicate",
        )
        return F.conv2d(image, kernel, groups=channels)

    def _laplacian_kernel(self, image: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian pyramid residual.

        Args:
            image: Image tensor.

        Returns:
            Laplacian residual tensor.
        """
        filtered = self._conv_gauss(image)
        down = filtered[:, :, ::2, ::2]
        up = torch.zeros_like(filtered)
        up[:, :, ::2, ::2] = down * 4.0
        filtered = self._conv_gauss(up)
        return image - filtered


class DarkIR_Loss(BaseLoss):
    """DarkIR loss from "DarkIR: Robust Low-Light Image Restoration".

    Paper form:
        L = lambda_pixel * L1(x_hat, x)
            + lambda_perceptual * LPIPS/VGG(x_hat, x)
            + lambda_edge * L_edge(x_hat, x)
            + L_lol

    L_lol supervises the low-resolution encoder side output after upsampling:
        L_lol = L1(side_up, x) + perceptual(side_up, x)

    The source code implements the perceptual term with a VGG19 feature loss
    and the edge term with a Laplacian pyramid kernel.
    """

    name = "darkir"
    aliases = ["darkir_loss", "DarkIR_Loss"]
    requires_target = True

    def __init__(
        self,
        *,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.01,
        edge_weight: float = 50.0,
        lol_weight: float = 1.0,
        use_perceptual: bool = True,
        use_edge: bool = True,
        use_lol_loss: bool = True,
        pretrained_vgg: bool = True,
        perceptual_criterion: str = "l1",
        edge_criterion: str = "l2",
        use_input_norm: bool = True,
        side_upsample_mode: str = "nearest",
    ) -> None:
        """Initialize DarkIR loss.

        Args:
            pixel_weight: Weight for the L1 pixel loss.
            perceptual_weight: Weight for VGG feature perceptual loss.
            edge_weight: Weight for Laplacian edge loss.
            lol_weight: Weight for side-output LOL loss.
            use_perceptual: Whether to enable perceptual loss.
            use_edge: Whether to enable edge loss.
            use_lol_loss: Whether to supervise side output when available.
            pretrained_vgg: Whether to load ImageNet-pretrained VGG19 weights.
            perceptual_criterion: Feature distance criterion.
            edge_criterion: Edge distance criterion.
            use_input_norm: Whether to apply ImageNet normalization before VGG.
            side_upsample_mode: Interpolation mode for side-output upsampling.
        """
        super().__init__()
        self.pixel_weight = float(pixel_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.edge_weight = float(edge_weight)
        self.lol_weight = float(lol_weight)
        self.use_lol_loss = bool(use_lol_loss)
        self.side_upsample_mode = side_upsample_mode

        self.pixel_loss = nn.L1Loss()

        self.perceptual_loss = None
        if use_perceptual and self.perceptual_weight > 0:
            self.perceptual_loss = DarkIRVGGFeatureLoss(
                loss_weight=self.perceptual_weight,
                criterion=perceptual_criterion,
                pretrained=pretrained_vgg,
                use_input_norm=use_input_norm,
            )

        self.edge_loss = None
        if use_edge and self.edge_weight > 0:
            self.edge_loss = DarkIREdgeLoss(
                loss_weight=self.edge_weight,
                criterion=edge_criterion,
            )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        side_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute DarkIR supervised loss.

        Args:
            prediction: Main prediction tensor.
            target: Ground-truth normal-light tensor.
            side_output: Optional low-resolution side output tensor.

        Returns:
            Scalar DarkIR loss tensor.
        """
        loss = self.pixel_weight * self.pixel_loss(prediction, target)

        if self.perceptual_loss is not None:
            loss = loss + self.perceptual_loss(prediction, target)

        if self.edge_loss is not None:
            loss = loss + self.edge_loss(prediction, target)

        if self.use_lol_loss and side_output is not None:
            side_prediction = self._upsample_side_output(side_output, target)
            loss = loss + self.lol_weight * self._lol_loss(side_prediction, target)

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
        """Compute DarkIR loss through the Trainer interface.

        Args:
            input_tensor: Low-light input tensor. It is unused by this
                supervised loss but kept for the ``BaseLoss`` interface.
            model_output: Raw DarkIR model output.
            target: Ground-truth normal-light tensor.
            extract_prediction: Optional callback for extracting prediction
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
            raise ValueError("DarkIR_Loss requires a target tensor.")

        prediction = self._extract_prediction(model_output, target, extract_prediction)
        if align_prediction is not None:
            prediction = align_prediction(prediction, target)

        side_output = self._extract_side_output(model_output)
        loss = self(prediction, target, side_output=side_output)
        return loss, prediction

    def _lol_loss(self, side_prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute DarkIR side-output LOL loss.

        Args:
            side_prediction: Upsampled side-output prediction.
            target: Ground-truth target tensor.

        Returns:
            Scalar side-output loss tensor.
        """
        loss = self.pixel_loss(side_prediction, target)
        if self.perceptual_loss is not None:
            loss = loss + self.perceptual_loss(side_prediction, target)
        return loss

    def _upsample_side_output(self, side_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Upsample side output to target spatial size.

        Args:
            side_output: Side-output tensor.
            target: Target tensor used for spatial size.

        Returns:
            Upsampled side-output tensor.
        """
        kwargs = {"size": target.shape[-2:], "mode": self.side_upsample_mode}
        if self.side_upsample_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        return F.interpolate(side_output, **kwargs)

    @staticmethod
    def _extract_prediction(
        model_output: Any,
        target: torch.Tensor,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        """Extract prediction tensor from DarkIR model output.

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

        raise TypeError(f"Cannot extract DarkIR prediction from {type(model_output).__name__}.")

    @staticmethod
    def _extract_side_output(model_output: Any) -> Optional[torch.Tensor]:
        """Extract optional DarkIR side output.

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


DarkIRLoss = DarkIR_Loss
