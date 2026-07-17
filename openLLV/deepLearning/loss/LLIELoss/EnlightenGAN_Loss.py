"""EnlightenGAN loss functions."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class EnlightenGAN_Loss(BaseLoss):
    """EnlightenGAN objective adapted for openLLV's unified trainer.

    The original method uses global and local adversarial losses, self feature
    preserving regularization, attention guidance, and patch-level training. The
    current implementation computes generator and discriminator objectives in a
    single Trainer-compatible loss while preserving the global/local GAN terms.
    """

    name = "enlightengan"
    aliases = ["enlightengan_loss", "enlighten_gan", "enlighten_gan_loss"]
    requires_target = True

    def __init__(
        self,
        *,
        adversarial_weight: float = 1.0,
        local_adversarial_weight: float = 0.5,
        discriminator_weight: float = 0.5,
        self_regularization_weight: float = 10.0,
        exposure_weight: float = 1.0,
        tv_weight: float = 0.1,
        target_mean: float = 0.6,
    ) -> None:
        """Initialize EnlightenGAN loss.

        Args:
            adversarial_weight: Weight for global generator adversarial loss.
            local_adversarial_weight: Weight for local generator adversarial
                loss.
            discriminator_weight: Weight for discriminator LSGAN losses.
            self_regularization_weight: Weight for self feature preservation.
            exposure_weight: Weight for brightness exposure regularization.
            tv_weight: Weight for total variation smoothness.
            target_mean: Target mean brightness for exposure regularization.
        """
        super().__init__()
        self.adversarial_weight = float(adversarial_weight)
        self.local_adversarial_weight = float(local_adversarial_weight)
        self.discriminator_weight = float(discriminator_weight)
        self.self_regularization_weight = float(self_regularization_weight)
        self.exposure_weight = float(exposure_weight)
        self.tv_weight = float(tv_weight)
        self.target_mean = float(target_mean)
        self.gan_loss = nn.MSELoss()

    def compute(
        self,
        *,
        input_tensor: torch.Tensor,
        model_output: Any,
        target: Optional[torch.Tensor] = None,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        align_prediction: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute EnlightenGAN loss through the Trainer interface.

        Args:
            input_tensor: Low-light input tensor.
            model_output: Raw EnlightenGAN model output.
            target: Normal-light target tensor used as the real domain.
            extract_prediction: Optional prediction extractor.
            align_prediction: Optional prediction alignment callback.

        Returns:
            Tuple containing scalar loss and prediction tensor.

        Raises:
            ValueError: If target or required model outputs are missing.
        """
        if target is None:
            raise ValueError("EnlightenGAN_Loss requires target images as the real normal-light domain.")
        if not isinstance(model_output, dict):
            raise ValueError("EnlightenGAN_Loss expects EnlightenGAN training output dictionary.")

        prediction = self._extract_prediction(model_output, target, extract_prediction)
        if align_prediction is not None:
            prediction = align_prediction(prediction, target)

        loss_inputs = get_loss_inputs(model_output)
        global_discriminator = loss_inputs["global_discriminator"]
        local_discriminator = loss_inputs["local_discriminator"]
        local_box = loss_inputs["local_box"]

        target = target.clamp(0.0, 1.0)
        input_tensor = input_tensor.clamp(0.0, 1.0)
        prediction = prediction.clamp(0.0, 1.0)

        loss = prediction.new_tensor(0.0)
        if self.adversarial_weight > 0:
            loss = loss + self.adversarial_weight * self._generator_gan_loss(
                global_discriminator,
                prediction,
            )
        if self.local_adversarial_weight > 0:
            fake_local = self._crop_by_box(prediction, local_box)
            loss = loss + self.local_adversarial_weight * self._generator_gan_loss(
                local_discriminator,
                fake_local,
            )
        if self.discriminator_weight > 0:
            real_local = self._crop_by_box(target, local_box)
            fake_local = self._crop_by_box(prediction.detach(), local_box)
            loss = loss + self.discriminator_weight * self._discriminator_loss(
                global_discriminator,
                real=target,
                fake=prediction.detach(),
            )
            loss = loss + self.discriminator_weight * self._discriminator_loss(
                local_discriminator,
                real=real_local,
                fake=fake_local,
            )

        if self.self_regularization_weight > 0:
            loss = loss + self.self_regularization_weight * self._self_regularization(
                prediction,
                input_tensor,
            )
        if self.exposure_weight > 0:
            loss = loss + self.exposure_weight * self._exposure_loss(prediction)
        if self.tv_weight > 0:
            loss = loss + self.tv_weight * self._tv_loss(prediction)

        return loss, prediction

    def _generator_gan_loss(self, discriminator: nn.Module, fake: torch.Tensor) -> torch.Tensor:
        """Compute generator LSGAN loss with frozen discriminator weights.

        Args:
            discriminator: Discriminator module.
            fake: Generated image tensor.

        Returns:
            Scalar generator adversarial loss.
        """
        previous = [param.requires_grad for param in discriminator.parameters()]
        try:
            for param in discriminator.parameters():
                param.requires_grad_(False)
            pred_fake = discriminator(fake)
            return self.gan_loss(pred_fake, torch.ones_like(pred_fake))
        finally:
            for param, requires_grad in zip(discriminator.parameters(), previous):
                param.requires_grad_(requires_grad)

    def _discriminator_loss(
        self,
        discriminator: nn.Module,
        *,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discriminator LSGAN loss.

        Args:
            discriminator: Discriminator module.
            real: Real normal-light image tensor.
            fake: Generated image tensor detached from the generator.

        Returns:
            Scalar discriminator loss.
        """
        pred_real = discriminator(real)
        pred_fake = discriminator(fake)
        real_loss = self.gan_loss(pred_real, torch.ones_like(pred_real))
        fake_loss = self.gan_loss(pred_fake, torch.zeros_like(pred_fake))
        return 0.5 * (real_loss + fake_loss)

    @staticmethod
    def _self_regularization(prediction: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Compute self feature-preserving regularization.

        Args:
            prediction: Enhanced image tensor.
            input_tensor: Low-light input tensor.

        Returns:
            Scalar regularization loss.
        """
        pred_gray = prediction.mean(dim=1, keepdim=True)
        input_gray = input_tensor.mean(dim=1, keepdim=True)
        color_loss = F.l1_loss(prediction - pred_gray, input_tensor - input_gray)
        gradient_loss = EnlightenGAN_Loss._gradient_l1(prediction, input_tensor)
        return color_loss + gradient_loss

    def _exposure_loss(self, prediction: torch.Tensor) -> torch.Tensor:
        """Compute exposure regularization.

        Args:
            prediction: Enhanced image tensor.

        Returns:
            Scalar exposure loss.
        """
        pooled = F.avg_pool2d(prediction.mean(dim=1, keepdim=True), kernel_size=16, stride=16)
        return torch.mean(torch.abs(pooled - self.target_mean))

    @staticmethod
    def _tv_loss(image: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss.

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
    def _gradient_l1(prediction: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss between image gradients.

        Args:
            prediction: Enhanced image tensor.
            input_tensor: Low-light input tensor.

        Returns:
            Scalar gradient loss.
        """
        pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        pred_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        input_x = input_tensor[:, :, :, 1:] - input_tensor[:, :, :, :-1]
        input_y = input_tensor[:, :, 1:, :] - input_tensor[:, :, :-1, :]
        return F.l1_loss(pred_x, input_x) + F.l1_loss(pred_y, input_y)

    @staticmethod
    def _crop_by_box(image: torch.Tensor, box: Tuple[int, int, int, int]) -> torch.Tensor:
        """Crop image by a local box.

        Args:
            image: Input tensor.
            box: Box tuple ``(top, left, height, width)``.

        Returns:
            Cropped tensor.
        """
        top, left, height, width = box
        return image[:, :, top : top + height, left : left + width]

    @staticmethod
    def _extract_prediction(
        model_output: Dict[str, Any],
        target: torch.Tensor,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        """Extract prediction tensor from EnlightenGAN output.

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
        raise TypeError("Cannot extract EnlightenGAN prediction tensor.")


EnlightenGANLoss = EnlightenGAN_Loss
