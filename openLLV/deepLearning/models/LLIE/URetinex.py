"""URetinex-Net model for Retinex-based low-light enhancement.

    Original paper: URetinex-Net: Retinex-Based Deep Unfolding Network for Low-Light Image Enhancement
    Paper link: https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf
    Official source code: https://github.com/AndersonYong/URetinex-Net
    """

from ..BaseModel import LLVModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import math


class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer"""

    def __init__(self, channel, reduction=16):
        """Initialize squeeze-and-excitation layer.

        Args:
            channel: Number of input channels.
            reduction: Channel reduction ratio.
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply squeeze-and-excitation weighting.

        Args:
            x: Input feature tensor.

        Returns:
            Channel-reweighted feature tensor.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DecompositionModule(nn.Module):
    """Decomposition module for separating reflectance and illumination"""

    def __init__(self):
        """Initialize Retinex decomposition module."""
        super().__init__()
        self.decom = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 4, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        """Decompose image into reflectance and illumination.

        Args:
            x: Input image tensor with shape ``[B, 3, H, W]``.

        Returns:
            A tuple ``(R, L)`` containing reflectance and illumination tensors.
        """
        output = self.decom(x)
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]
        return R, L


class PModule(nn.Module):
    """P update module for the URetinex unfolding process."""

    def __init__(self):
        """Initialize P update module."""
        super().__init__()

    def forward(self, I, Q, R, gamma):
        """Update reflectance proxy ``P``.

        Args:
            I: Input image tensor.
            Q: Current illumination proxy tensor.
            R: Current reflectance tensor.
            gamma: Reflectance penalty weight.

        Returns:
            Updated ``P`` tensor.
        """
        return ((I * Q + gamma * R) / (gamma + Q * Q))


class QModule(nn.Module):
    """Q update module for the URetinex unfolding process."""

    def __init__(self):
        """Initialize Q update module."""
        super().__init__()

    def forward(self, I, P, L, lamda):
        """Update illumination proxy ``Q``.

        Args:
            I: Input image tensor.
            P: Current reflectance proxy tensor.
            L: Current illumination tensor.
            lamda: Illumination penalty weight.

        Returns:
            Updated ``Q`` tensor.
        """
        IR = I[:, 0:1, :, :]
        IG = I[:, 1:2, :, :]
        IB = I[:, 2:3, :, :]

        PR = P[:, 0:1, :, :]
        PG = P[:, 1:2, :, :]
        PB = P[:, 2:3, :, :]

        return (IR * PR + IG * PG + IB * PB + lamda * L) / ((PR * PR + PG * PG + PB * PB) + lamda)


class RestorationModule(nn.Module):
    """Restoration module for refining reflectance"""

    def __init__(self, use_concat_l=True):
        """Initialize reflectance restoration module.

        Args:
            use_concat_l: Whether to concatenate illumination features with
                reflectance features.
        """
        super().__init__()
        self.use_concat_l = use_concat_l

        if self.use_concat_l:
            self.conv1_r = nn.Conv2d(3, 32, 3, 1, 1)
            self.relu1_r = nn.ReLU(inplace=True)
            self.conv1_l = nn.Conv2d(1, 32, 3, 1, 1)
            self.relu1_l = nn.ReLU(inplace=True)
            in_channels = 64
        else:
            self.conv1_r = nn.Conv2d(3, 64, 3, 1, 1)
            self.relu1_r = nn.ReLU(inplace=True)
            in_channels = 64

        self.se_layer = SELayer(channel=in_channels)

        self.conv3 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, r, l):
        """Refine reflectance using reflectance and illumination estimates.

        Args:
            r: Reflectance tensor.
            l: Illumination tensor.

        Returns:
            Restored reflectance tensor.
        """
        if self.use_concat_l:
            r_fs = self.relu1_r(self.conv1_r(r))
            l_fs = self.relu1_l(self.conv1_l(l))
            features = torch.cat([r_fs, l_fs], dim=1)
            se_features = self.se_layer(features)
        else:
            r_fs = self.relu1_r(self.conv1_r(r))
            se_features = self.se_layer(r_fs)

        x1 = self.relu3(self.conv3(se_features))
        x2 = self.relu4(self.conv4(x1))
        x3 = self.relu5(self.conv5(x2))
        x4 = self.relu6(self.conv6(x3))
        x5 = self.relu7(self.conv7(x4))
        noise = self.conv8(x5)
        r_restore = r + noise
        return r_restore


class IlluminationEnhancementModule(nn.Module):
    """Illumination enhancement module"""

    def __init__(self):
        """Initialize illumination enhancement module."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv4 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv5 = nn.Conv2d(32, 1, 1, 1, 0)

        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU()

    def forward(self, l):
        """Enhance illumination estimate.

        Args:
            l: Input illumination tensor.

        Returns:
            Enhanced illumination tensor.
        """
        x = l
        x1 = self.leaky_relu1(self.conv1(x))
        x2 = self.leaky_relu2(self.conv2(x1))
        x3 = self.leaky_relu3(self.conv3(x2))
        x4 = self.leaky_relu4(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        return x5


class IlluminationAdjustmentModule(nn.Module):
    """Illumination adjustment module"""

    def __init__(self):
        """Initialize illumination adjustment module."""
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv4 = nn.Conv2d(32, 1, 5, 1, 2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, l, alpha):
        """Adjust illumination with a ratio tensor or scalar tensor.

        Args:
            l: Input illumination tensor.
            alpha: Adjustment ratio tensor.

        Returns:
            Adjusted illumination tensor.
        """
        if alpha.dim() == 1 or (alpha.dim() == 4 and alpha.shape[1] == 1):
            alpha = alpha.view(alpha.shape[0], 1, 1, 1).expand_as(l)

        input_tensor = torch.cat([l, alpha], dim=1)
        x = self.conv1(input_tensor)
        x = self.conv2(self.leaky_relu(x))
        x = self.conv3(self.leaky_relu(x))
        x = self.conv4(self.leaky_relu(x))
        x = self.relu(x)
        return x


class URetinexNet(LLVModel):

    task = "llie"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize URetinex-Net.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default URetinex-Net configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update({
            'unfolding_rounds': 3,  # Number of unfolding iterations
            'gamma': 0.01,  # Weight for P module
            'lambda': 0.01,  # Weight for Q module
            'gamma_offset': 0.01,  # Offset for gamma per iteration
            'lambda_offset': 0.01,  # Offset for lambda per iteration
            'use_concat_l': True,  # Whether to concatenate L in restoration
            'mode': 'inference',  # Mode: 'train' or 'inference'
            'adjustment_ratio': 5.0,  # Default illumination adjustment ratio
            'adaptive_ratio': False,  # Whether to use adaptive ratio calculation
        })
        return default_config

    def _validate_config(self):
        """Validate URetinex-Net configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()

        if self.config['unfolding_rounds'] <= 0:
            raise ValueError("'unfolding_rounds' must be positive")
        if self.config['gamma'] <= 0:
            raise ValueError("'gamma' must be positive")
        if self.config['lambda'] <= 0:
            raise ValueError("'lambda' must be positive")
        if self.config['mode'] not in ['train', 'inference']:
            raise ValueError("'mode' must be 'train' or 'inference'")

    def _init_model(self):
        """Initialize URetinex-Net components."""
        self.decom_low = DecompositionModule()
        self.decom_high = DecompositionModule() if self.config.get('adaptive_ratio', False) else None

        self.P_module = PModule()
        self.Q_module = QModule()
        self.restoration_module = RestorationModule(use_concat_l=self.config['use_concat_l'])
        self.illumination_module = IlluminationEnhancementModule()

        self.adjustment_module = IlluminationAdjustmentModule()

    def get_ratio(self, high_l, low_l):
        """Calculate adjustment ratio between high and low illumination.

        Args:
            high_l: High-light illumination tensor.
            low_l: Low-light illumination tensor.

        Returns:
            Adjustment ratio tensor.
        """
        ratio = (low_l / (high_l + 1e-6)).mean()
        low_ratio = torch.ones(high_l.shape, device=high_l.device) * (1 / (ratio + 1e-6))
        return low_ratio

    def unfolding_process(self, input_low_img):
        """Estimate reflectance and illumination with unfolding iterations.

        Args:
            input_low_img: Low-light input tensor with shape ``[B, 3, H, W]``.

        Returns:
            A tuple ``(R, L)`` containing estimated reflectance and
            illumination tensors.
        """
        R, L = None, None
        gamma = self.config['gamma']
        lambda_val = self.config['lambda']

        for t in range(self.config['unfolding_rounds']):
            if t == 0:  # Initialize R0, L0
                P, Q = self.decom_low(input_low_img)
            else:  # Update P and Q
                w_p = gamma + self.config['gamma_offset'] * t
                w_q = lambda_val + self.config['lambda_offset'] * t
                P = self.P_module(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q_module(I=input_low_img, P=P, L=L, lamda=w_q)

            # Update R and L
            R = self.restoration_module(r=P, l=Q)
            L = self.illumination_module(l=Q)

        return R, L

    def forward(self, x: torch.Tensor, high_img: Optional[torch.Tensor] = None,
                ratio: Optional[float] = None) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a URetinex-Net forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.
            high_img: Optional high-light image for adaptive ratio calculation.
            ratio: Optional manual adjustment ratio.

        Returns:
            Training mode: standardized output dict containing Retinex
            intermediate tensors. Inference mode: enhanced image tensor.
        """

        with torch.set_grad_enabled(self.config['mode'] == 'train'):
            R, L = self.unfolding_process(x)

            if ratio is not None:
                adjustment_ratio = ratio
            elif self.config.get('adaptive_ratio', False) and high_img is not None:
                _, high_L = self.decom_high(high_img)
                adjustment_ratio = self.get_ratio(high_L, L)
            else:
                adjustment_ratio = self.config['adjustment_ratio']

            if isinstance(adjustment_ratio, torch.Tensor):
                High_L = self.adjustment_module(l=L, alpha=adjustment_ratio)
            else:
                ratio_tensor = torch.tensor([adjustment_ratio],
                                            device=x.device,
                                            dtype=x.dtype)
                High_L = self.adjustment_module(l=L, alpha=ratio_tensor)

            enhanced_image = High_L * R

            aux = {
                'reflectance': R,
                'illumination': L,
                'adjusted_illumination': High_L,
                'adjustment_ratio': adjustment_ratio,
            }
            if self.config['mode'] == 'train':
                return self._format_output(
                    pred=enhanced_image,
                aux=aux,
                    meta={'mode': self.config['mode']},
                )

            return enhanced_image
