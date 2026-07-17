"""Zero-DCE model for zero-reference low-light enhancement.

    Original paper: Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
    Paper link: https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf
    Official source code: https://github.com/Li-Chongyi/Zero-DCE
    Official project url: https://li-chongyi.github.io/Proj_Zero-DCE.html
    """

from ..BaseModel import LLVModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union


class ZeroDCE(LLVModel):

    task = "llie"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize Zero-DCE.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Zero-DCE configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update({
            'number_f': 32,  # Number of feature channels
            'num_iterations': 8,  # Number of iterations (curve parameters)
            'mode': 'inference'  # Mode: 'train' or 'inference'
        })
        return default_config

    def _validate_config(self):
        """Validate Zero-DCE configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()

        if self.config['number_f'] <= 0:
            raise ValueError("'number_f' must be positive")
        if self.config['num_iterations'] <= 0:
            raise ValueError("'num_iterations' must be positive")
        if self.config['mode'] not in ['train', 'inference']:
            raise ValueError("'mode' must be 'train' or 'inference'")

    def _init_model(self):
        """Initialize Zero-DCE network layers."""
        nf = self.config['number_f']

        self.conv1 = nn.Conv2d(self.config['input_channels'], nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(nf * 2, self.config['num_iterations'] * 3, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a Zero-DCE forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.

        Returns:
            Training mode: standardized output dict with enhanced image and
            curve parameters. Inference mode: enhanced image tensor.
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhanced = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)

        if self.config['mode'] == 'train':
            return self._format_output(
                pred=enhanced,
                aux={
                    'enhanced': enhanced,
                    'r': r,
                },
                meta={'mode': self.config['mode']},
            )

        return enhanced

