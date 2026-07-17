"""Image quality metrics for_teach low-light enhancement evaluation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import math
import warnings
import cv2

from .baseMetric import BaseMetric

CV2_AVAILABLE = True


def _score_to_float(score: Any) -> float:
    """Convert pyiqa metric output to a Python float.

    Args:
        score: Metric output returned by pyiqa. Supported formats include
            tensors, dictionaries, tuples, lists, and scalar numeric values.

    Returns:
        Mean score converted to ``float``.
    """
    if isinstance(score, dict):
        for key in ('score', 'quality', 'overall', 'mos'):
            if key in score:
                score = score[key]
                break
        else:
            score = next(iter(score.values()))

    if isinstance(score, (tuple, list)):
        score = score[0]

    if isinstance(score, torch.Tensor):
        return score.detach().float().mean().item()

    return float(score)


def _create_pyiqa_metric(metric_name: str, device: torch.device, display_name: str):
    """Create a pyiqa metric instance.

    Args:
        metric_name: Internal metric name used by pyiqa.
        device: Torch device used to run the metric.
        display_name: Human-readable metric name used in error messages.

    Returns:
        The metric object created by pyiqa.

    Raises:
        ImportError: If pyiqa is not installed.
        RuntimeError: If pyiqa fails to create the requested metric.
    """
    try:
        import pyiqa
    except ImportError:
        raise ImportError(f"计算 {display_name} 需要安装 'pyiqa' 库。请运行: pip install pyiqa")

    try:
        return pyiqa.create_metric(metric_name, device=device)
    except Exception as e:
        raise RuntimeError(f"无法创建 {display_name} 指标: {e}")


def _prepare_pyiqa_input(img: torch.Tensor, data_range: float, device: torch.device) -> torch.Tensor:
    """Prepare an image tensor for_teach pyiqa metrics.

    Args:
        img: Input image tensor with shape ``[C, H, W]`` or
            ``[B, C, H, W]``.
        data_range: Numeric range of the input tensor.
        device: Torch device where the metric runs.

    Returns:
        Batched image tensor clipped to ``[0, 1]`` and moved to ``device``.
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)

    img_input = img / data_range if data_range != 1.0 else img
    return torch.clamp(img_input, 0.0, 1.0).to(device)


def _compute_pyiqa_score(
        metric_model,
        enImg: torch.Tensor,
        data_range: float,
        device: torch.device,
        display_name: str
) -> float:
    """Compute a no-reference pyiqa score.

    Args:
        metric_model: Initialized pyiqa metric object.
        enImg: Enhanced image tensor.
        data_range: Numeric range of ``enImg``.
        device: Torch device where the metric runs.
        display_name: Human-readable metric name used in warnings.

    Returns:
        Metric score as a float, or ``nan`` if pyiqa raises an exception.
    """
    img_input = _prepare_pyiqa_input(enImg, data_range, device)

    try:
        with torch.no_grad():
            score = metric_model(img_input)
        return _score_to_float(score)
    except Exception as e:
        warnings.warn(f"{display_name} 计算内部异常: {e}")
        return float('nan')


__all__ = [
    # Full-reference metrics.
    'PSNRMetric',
    'SSIMMetric',
    'MSEMetric',
    'MAEMetric',
    'LPIPSMetric',
    'LOEMetric',

    # No-reference metrics.
    'NIQEMetric',
    'MUSIQMetric',
    'PIMetric',
]


class PSNRMetric(BaseMetric):
    """Peak Signal-to-Noise Ratio metric."""

    def __init__(self, data_range: float = 1.0, **kwargs):
        """Initialize the PSNR metric.

        Args:
            data_range: Maximum value of the image data range.
            **kwargs: Additional arguments forwarded to ``BaseMetric``.
        """
        super().__init__(**kwargs)
        self.data_range = data_range

    def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor) -> float:
        """Compute PSNR between enhanced and reference images.

        Args:
            enImg: Enhanced image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.
            Refer: Reference image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.

        Returns:
            PSNR value in decibels.
        """
        if enImg.dim() == 3:
            enImg = enImg.unsqueeze(0)
            Refer = Refer.unsqueeze(0)

        mse = F.mse_loss(enImg, Refer, reduction='mean')
        if mse == 0:
            return float('inf')

        psnr = 20 * torch.log10(self.data_range / torch.sqrt(mse))
        return psnr.item()

    @property
    def higher_is_better(self) -> bool:
        """Whether larger PSNR values indicate better quality."""
        return True


class SSIMMetric(BaseMetric):
    """Structural Similarity Index metric."""

    def __init__(self, data_range: float = 1.0, window_size: int = 11,
                 sigma: float = 1.5, **kwargs):
        """Initialize the SSIM metric.

        Args:
            data_range: Maximum value of the image data range.
            window_size: Size of the Gaussian window.
            sigma: Standard deviation of the Gaussian window.
            **kwargs: Additional arguments forwarded to ``BaseMetric``.
        """
        super().__init__(**kwargs)
        self.data_range = data_range
        self.window_size = window_size
        self.sigma = sigma
        self._create_window()

    def _create_window(self):
        """Create the base Gaussian window used by SSIM."""
        gauss = torch.Tensor([
            math.exp(-(x - self.window_size // 2) ** 2 / float(2 * self.sigma ** 2))
            for x in range(self.window_size)
        ])
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size, window_size]
        window = window / window.sum()
        self.window = window  # Expanded dynamically by channel count during computation.

    def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor) -> float:
        """Compute SSIM between enhanced and reference images.

        Args:
            enImg: Enhanced image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.
            Refer: Reference image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.

        Returns:
            Mean SSIM value.
        """
        if enImg.dim() == 3:
            enImg = enImg.unsqueeze(0)
        if Refer.dim() == 3:
            Refer = Refer.unsqueeze(0)

        batch_size, channels, height, width = enImg.shape

        window = self.window.to(enImg.device)
        window = window.expand(channels, 1, self.window_size, self.window_size)  # [C, 1, window_size, window_size]

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        mu1 = F.conv2d(enImg, window, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(Refer, window, padding=self.window_size // 2, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(enImg * enImg, window, padding=self.window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(Refer * Refer, window, padding=self.window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(enImg * Refer, window, padding=self.window_size // 2, groups=channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    @property
    def higher_is_better(self) -> bool:
        """Whether larger SSIM values indicate better quality."""
        return True


class MSEMetric(BaseMetric):
    """Mean Squared Error metric."""

    def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor) -> float:
        """Compute MSE between enhanced and reference images.

        Args:
            enImg: Enhanced image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.
            Refer: Reference image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.

        Returns:
            Mean squared error value.
        """
        if enImg.dim() == 3:
            enImg = enImg.unsqueeze(0)
            Refer = Refer.unsqueeze(0)

        mse = F.mse_loss(enImg, Refer, reduction='mean')
        return mse.item()

    @property
    def higher_is_better(self) -> bool:
        """Whether larger MSE values indicate better quality."""
        return False


class MAEMetric(BaseMetric):
    """Mean Absolute Error metric."""

    def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor) -> float:
        """Compute MAE between enhanced and reference images.

        Args:
            enImg: Enhanced image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.
            Refer: Reference image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.

        Returns:
            Mean absolute error value.
        """
        if enImg.dim() == 3:
            enImg = enImg.unsqueeze(0)
            Refer = Refer.unsqueeze(0)

        mae = F.l1_loss(enImg, Refer, reduction='mean')
        return mae.item()

    @property
    def higher_is_better(self) -> bool:
        """Whether larger MAE values indicate better quality."""
        return False


class LPIPSMetric(BaseMetric):
    """Learned Perceptual Image Patch Similarity metric."""

    def __init__(self, device: str = None, data_range: float = 1.0,
                 net: str = 'alex', **kwargs):
        """Initialize the LPIPS metric.

        Args:
            device: Device used to run pyiqa.
            data_range: Maximum value of the image data range.
            net: Backbone network name used by the LPIPS implementation.
            **kwargs: Additional arguments forwarded to ``BaseMetric``.
        """
        super().__init__(device, **kwargs)
        self.data_range = data_range
        self.net = net
        self._metric_model = None
        self._load_vgg_features()

    def _load_vgg_features(self):
        """Load the pyiqa LPIPS backend.

        Raises:
            ImportError: If pyiqa is not installed.
            RuntimeError: If pyiqa fails to create the LPIPS metric.
        """
        try:
            import pyiqa
        except ImportError:
            raise ImportError("计算 LPIPS 需要安装 'pyiqa' 库。请运行: pip install pyiqa")

        try:
            self._metric_model = pyiqa.create_metric('lpips', device=self.device, net=self.net)
        except TypeError:
            self._metric_model = pyiqa.create_metric('lpips', device=self.device)
        except Exception as e:
            raise RuntimeError(f"无法创建 LPIPS 指标: {e}")

    def _prepare_pyiqa_input(self, img: torch.Tensor) -> torch.Tensor:
        """Prepare an image tensor for_teach LPIPS.

        Args:
            img: Input image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.

        Returns:
            Batched image tensor clipped to ``[0, 1]`` and moved to the metric
            device.
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)

        img_input = img / self.data_range if self.data_range != 1.0 else img
        return torch.clamp(img_input, 0.0, 1.0).to(self.device)

    def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor) -> float:
        """Compute LPIPS between enhanced and reference images.

        Args:
            enImg: Enhanced image tensor.
            Refer: Reference image tensor.

        Returns:
            LPIPS value, or ``nan`` if pyiqa raises an exception.

        Raises:
            ValueError: If ``Refer`` is ``None``.
        """
        if Refer is None:
            raise ValueError("LPIPS 需要参考图像 Refer")

        if self._metric_model is None:
            self._load_vgg_features()

        en_input = self._prepare_pyiqa_input(enImg)
        ref_input = self._prepare_pyiqa_input(Refer)

        try:
            with torch.no_grad():
                score = self._metric_model(en_input, ref_input)
            return _score_to_float(score)
        except Exception as e:
            warnings.warn(f"LPIPS 计算内部异常: {e}")
            return float('nan')

    @property
    def higher_is_better(self) -> bool:
        """Whether larger LPIPS values indicate better quality."""
        return False


class LOEMetric(BaseMetric):
    """Lightness Order Error metric."""

    def __init__(self, patch_size: int = 50, **kwargs):
        """Initialize the LOE metric.

        Args:
            patch_size: Patch size used when approximating lightness order.
            **kwargs: Additional arguments forwarded to ``BaseMetric``.
        """
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def _rgb_to_gray(self, img: torch.Tensor) -> torch.Tensor:
        """Convert RGB image tensors to lightness tensors.

        Args:
            img: Input image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.

        Returns:
            Single-channel lightness tensor.
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)

        if img.size(1) == 1:
            return img

        # LOE uses the maximum RGB channel value as lightness.
        return img[:, :3].max(dim=1, keepdim=True).values

    def _extract_patches(self, img: torch.Tensor) -> torch.Tensor:
        """Extract pooled lightness patches.

        Args:
            img: Input lightness tensor.

        Returns:
            Flattened adaptive-average-pooled patch tensor.
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)

        _, _, height, width = img.shape
        out_h = max(1, math.ceil(height / self.patch_size))
        out_w = max(1, math.ceil(width / self.patch_size))

        patches = F.adaptive_avg_pool2d(img, output_size=(out_h, out_w))
        return patches.flatten(start_dim=1)

    def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor = None) -> float:
        """Compute LOE between enhanced and reference images.

        Args:
            enImg: Enhanced image tensor.
            Refer: Original low-light image tensor used as the lightness-order
                reference.

        Returns:
            Mean LOE value.

        Raises:
            ValueError: If ``Refer`` is ``None``.
        """
        if Refer is None:
            raise ValueError("LOE 需要原始低照度图像作为 Refer")

        en_light = self._rgb_to_gray(enImg)
        ref_light = self._rgb_to_gray(Refer)

        en_patches = self._extract_patches(en_light)
        ref_patches = self._extract_patches(ref_light)

        en_order = en_patches.unsqueeze(2) >= en_patches.unsqueeze(1)
        ref_order = ref_patches.unsqueeze(2) >= ref_patches.unsqueeze(1)

        loe = torch.logical_xor(en_order, ref_order).float().mean(dim=(1, 2))
        return loe.mean().item()

    @property
    def higher_is_better(self) -> bool:
        """Whether larger LOE values indicate better quality."""
        return False


class NIQEMetric(BaseMetric):
    """Natural Image Quality Evaluator metric.

    This no-reference metric is backed by pyiqa and can run on GPU devices
    supported by the configured torch environment.
    """

    pyiqa_metric_name = 'niqe'

    def __init__(self, data_range: float = 1.0, **kwargs):
        """Initialize the NIQE metric.

        Args:
            data_range: Maximum value of the image data range.
            **kwargs: Additional arguments forwarded to ``BaseMetric``.
        """
        super().__init__(**kwargs)
        self.data_range = data_range
        self._metric_model = None
        self._import_pyiqa()

    def _import_pyiqa(self):
        """Load the pyiqa backend for_teach the configured metric name."""
        self._metric_model = _create_pyiqa_metric(self.pyiqa_metric_name, self.device, self.name)

    def _prepare_pyiqa_input(self, img: torch.Tensor) -> torch.Tensor:
        """Prepare an image tensor for_teach the pyiqa backend.

        Args:
            img: Input image tensor.

        Returns:
            Batched image tensor clipped to ``[0, 1]`` and moved to the metric
            device.
        """
        return _prepare_pyiqa_input(img, self.data_range, self.device)

    def _compute_pyiqa_score(self, enImg: torch.Tensor) -> float:
        """Compute a pyiqa-backed no-reference score.

        Args:
            enImg: Enhanced image tensor.

        Returns:
            Metric score as a float.
        """
        if self._metric_model is None:
            self._import_pyiqa()

        return _compute_pyiqa_score(self._metric_model, enImg, self.data_range, self.device, self.name)

    def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor = None) -> float:
        """Compute NIQE for_teach an enhanced image.

        Args:
            enImg: Enhanced image tensor.
            Refer: Unused reference tensor kept for_teach interface compatibility.

        Returns:
            NIQE score.
        """
        return self._compute_pyiqa_score(enImg)

    @property
    def requires_reference(self) -> bool:
        """Whether this metric requires a reference image."""
        return False

    @property
    def higher_is_better(self) -> bool:
        """Whether larger NIQE values indicate better quality."""
        # Lower NIQE values indicate closer natural-image statistics.
        return False



class MUSIQMetric(NIQEMetric):
    """Multi-scale Image Quality Transformer metric."""

    pyiqa_metric_name = 'musiq'

    def __init__(self, scales: List[float] = None, **kwargs):
        """Initialize the MUSIQ metric.

        Args:
            scales: Optional image scales used by callers.
            **kwargs: Additional arguments forwarded to ``NIQEMetric``.
        """
        self.scales = scales
        super().__init__(**kwargs)

    def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor = None) -> float:
        """Compute MUSIQ for_teach an enhanced image.

        Args:
            enImg: Enhanced image tensor.
            Refer: Unused reference tensor kept for_teach interface compatibility.

        Returns:
            MUSIQ score.
        """
        return self._compute_pyiqa_score(enImg)

    @property
    def requires_reference(self) -> bool:
        """Whether this metric requires a reference image."""
        return False

    @property
    def higher_is_better(self) -> bool:
        """Whether larger MUSIQ values indicate better quality."""
        return True


class PIMetric(BaseMetric):
    """Perceptual Index metric."""

    pyiqa_metric_name = 'pi'

    def __init__(self, data_range: float = 1.0, **kwargs):
        """Initialize the PI metric.

        Args:
            data_range: Maximum value of the image data range.
            **kwargs: Additional arguments forwarded to ``BaseMetric``.
        """
        super().__init__(**kwargs)
        self.data_range = data_range
        self._metric_model = _create_pyiqa_metric(self.pyiqa_metric_name, self.device, self.name)

    def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor = None) -> float:
        """Compute PI for_teach an enhanced image.

        Args:
            enImg: Enhanced image tensor.
            Refer: Unused reference tensor kept for_teach interface compatibility.

        Returns:
            PI score.
        """
        if self._metric_model is None:
            self._metric_model = _create_pyiqa_metric(self.pyiqa_metric_name, self.device, self.name)

        return _compute_pyiqa_score(self._metric_model, enImg, self.data_range, self.device, self.name)

    @property
    def requires_reference(self) -> bool:
        """Whether this metric requires a reference image."""
        return False

    @property
    def higher_is_better(self) -> bool:
        """Whether larger PI values indicate better quality."""
        return False
