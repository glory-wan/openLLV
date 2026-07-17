"""Zero-DCE++ model for zero-reference low-light enhancement.

    Original paper: Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation
    Paper link: https://ieeexplore.ieee.org/document/9369102/
    Official source code: https://github.com/Li-Chongyi/Zero-DCE_extension
    Official project url: https://li-chongyi.github.io/Proj_Zero-DCE++.html
    """

from ..BaseModel import LLVModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union


class ZeroDCEPlusPlus(LLVModel):

    task = "llie"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize Zero-DCE++.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Zero-DCE++ configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update({
            'number_f': 32,
            'scale_factor': 1,
            'mode': 'inference'
        })
        return default_config

    def _validate_config(self):
        """Validate Zero-DCE++ configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()

        if self.config['number_f'] <= 0:
            raise ValueError("'number_f' must be positive")
        if self.config['scale_factor'] <= 0:
            raise ValueError("'scale_factor' must be positive")
        if self.config['mode'] not in ['train', 'inference']:
            raise ValueError("'mode' must be 'train' or 'inference'")

    class CSDN_Tem(nn.Module):
        """Depthwise separable convolution block."""

        def __init__(self, in_ch, out_ch):
            """Initialize the convolution block.

            Args:
                in_ch: Number of input channels.
                out_ch: Number of output channels.
            """
            super().__init__()
            self.depth_conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_ch
            )
            self.point_conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1
            )

        def forward(self, input):
            """Apply depthwise separable convolution.

            Args:
                input: Input tensor.

            Returns:
                Output tensor.
            """
            out = self.depth_conv(input)
            out = self.point_conv(out)
            return out

    def _init_model(self):
        """Initialize Zero-DCE++ network layers."""
        number_f = self.config['number_f']
        scale_factor = self.config['scale_factor']

        self.e_conv1 = self.CSDN_Tem(self.config['input_channels'], number_f)
        self.e_conv2 = self.CSDN_Tem(number_f, number_f)
        self.e_conv3 = self.CSDN_Tem(number_f, number_f)
        self.e_conv4 = self.CSDN_Tem(number_f, number_f)
        self.e_conv5 = self.CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = self.CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = self.CSDN_Tem(number_f * 2, 3)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        if self.config['mode'] == 'inference' and scale_factor > 1:
            self.downsample = nn.UpsamplingBilinear2d(scale_factor=1 / scale_factor)
        else:
            self.downsample = None

    def _enhance(self, x: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
        """Apply curve estimation to enhance an image.

        Args:
            x: Input image tensor.
            x_r: Curve parameter tensor.

        Returns:
            Enhanced image tensor.
        """
        for _ in range(4):
            x = x + x_r * (torch.pow(x, 2) - x)

        enhance_image_1 = x

        for _ in range(4):
            if _ == 0:
                x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
            else:
                x = x + x_r * (torch.pow(x, 2) - x)

        enhance_image = x
        return enhance_image

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a Zero-DCE++ forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.

        Returns:
            Training mode: standardized output dict with enhanced image and
            curve parameters. Inference mode: enhanced image tensor.
        """
        scale_factor = self.config['scale_factor']

        if scale_factor > 1 and self.downsample is not None and self.config['mode'] == 'inference':
            x_down = self.downsample(x)
        else:
            x_down = x

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if scale_factor > 1 and scale_factor != 1:
            x_r = self.upsample(x_r)

        enhance_image = self._enhance(x, x_r)

        if self.config['mode'] == 'train':
            return self._format_output(
                pred=enhance_image,
                aux={
                    'enhanced': enhance_image,
                    'r': x_r,
                    'curve_params': x_r,
                },
                meta={'mode': self.config['mode'], 'scale_factor': scale_factor},
            )
        else:
            return enhance_image

    def train_mode(self):
        """Switch the model to training mode.

        Returns:
            The model itself.
        """
        self.config['mode'] = 'train'
        self.train()
        if hasattr(self, 'downsample'):
            self.downsample = None
        return self

    def eval_mode(self):
        """Switch the model to evaluation/inference mode.

        Returns:
            The model itself.
        """
        self.config['mode'] = 'inference'
        self.eval()
        if self.config['scale_factor'] > 1:
            self.downsample = nn.UpsamplingBilinear2d(
                scale_factor=1 / self.config['scale_factor']
            )
        return self
