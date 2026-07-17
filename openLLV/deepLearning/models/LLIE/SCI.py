"""SCI model for fast low-light image enhancement.

    Original paper: Toward Fast, Flexible, and Robust Low-Light Image Enhancement
    Paper link: https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf
    Official source code: https://github.com/vis-opt-group/SCI
    """

from ..BaseModel import LLVModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List


class EnhanceNetwork(nn.Module):
    """SCI illumination enhancement subnetwork."""

    def __init__(self, layers, channels):
        """Initialize the enhancement subnetwork.

        Args:
            layers: Number of residual convolution blocks.
            channels: Number of feature channels.
        """
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Estimate illumination from an input image.

        Args:
            x: Input tensor with shape ``[B, 3, H, W]``.

        Returns:
            Estimated illumination tensor.
        """
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + x
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class CalibrateNetwork(nn.Module):
    """SCI calibration subnetwork."""

    def __init__(self, layers, channels):
        """Initialize the calibration subnetwork.

        Args:
            layers: Number of residual convolution blocks.
            channels: Number of feature channels.
        """
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Estimate calibration residual.

        Args:
            x: Input tensor with shape ``[B, 3, H, W]``.

        Returns:
            Calibration residual tensor.
        """
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = x - fea

        return delta


class SCI(LLVModel):

    task = "llie"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize SCI.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default SCI configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update({
            'stage': 3,
            'enhance_layers': 1,
            'enhance_channels': 3,
            'calibrate_layers': 3,
            'calibrate_channels': 16,
            'mode': 'inference',
        })
        return default_config

    def _validate_config(self):
        """Validate SCI configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()

        if self.config['stage'] <= 0:
            raise ValueError("'stage' must be positive")
        if self.config['enhance_layers'] <= 0:
            raise ValueError("'enhance_layers' must be positive")
        if self.config['calibrate_layers'] <= 0:
            raise ValueError("'calibrate_layers' must be positive")
        if self.config['mode'] not in ['train', 'inference']:
            raise ValueError("'mode' must be 'train' or 'inference'")

    def _init_model(self):
        """Initialize SCI network components."""
        self.stage = self.config['stage']

        self.enhance = EnhanceNetwork(
            layers=self.config['enhance_layers'],
            channels=self.config['enhance_channels']
        )

        self.calibrate = CalibrateNetwork(
            layers=self.config['calibrate_layers'],
            channels=self.config['calibrate_channels']
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize convolution and batch-normalization weights."""
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1., 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.enhance.in_conv.apply(weights_init)
        for block in self.enhance.blocks:
            block.apply(weights_init)
        self.enhance.out_conv.apply(weights_init)

        self.calibrate.in_conv.apply(weights_init)
        for block in self.calibrate.blocks:
            block.apply(weights_init)
        self.calibrate.out_conv.apply(weights_init)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a SCI forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.

        Returns:
            Training mode: standardized output dict containing stage outputs.
            Inference mode: final enhanced image tensor.
        """
        if self.config['mode'] == 'train':
            ilist, rlist, inlist, attlist = self._forward_train(x)
            pred = rlist[-1]
            return self._format_output(
                pred=pred,
                aux={
                    'enhanced': pred,
                    'ilist': ilist,
                    'rlist': rlist,
                    'inlist': inlist,
                    'attlist': attlist,
                },
                meta={'mode': self.config['mode'], 'stage': self.stage},
            )
        else:
            pred = self._forward_inference(x)
            return pred

    def _forward_train(self, x: torch.Tensor) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ]:
        """Run SCI staged training forward pass.

        Args:
            x: Low-light input tensor.

        Returns:
            Tuple containing illumination, reflectance, stage input, and
            attention-residual lists.
        """
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = x.clone()

        for i in range(self.stage):
            inlist.append(input_op)
            i_current = self.enhance(input_op)
            r_current = x / i_current
            r_current = torch.clamp(r_current, 0, 1)
            att = self.calibrate(r_current)
            input_op = x + att

            ilist.append(i_current)
            rlist.append(r_current)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    def _forward_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run SCI inference forward pass.

        Args:
            x: Low-light input tensor.

        Returns:
            Final enhanced image tensor.
        """
        i = self.enhance(x)
        enImg = x / i
        enImg = torch.clamp(enImg, 0, 1)
        return enImg
