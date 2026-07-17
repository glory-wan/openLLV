"""RUAS model for unsupervised low-light enhancement.

    Original paper: Retinex-inspired Unrolling with Cooperative Prior Architecture Search for Low-light Image Enhancement
    Paper link: https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf
    Official source code: https://github.com/KarelZhang/RUAS
    """

from ..BaseModel import LLVModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List
from collections import namedtuple


PRIMITIVES = [
    'skip_connect',
    'conv_1x1',
    'conv_3x3',
    'dilconv_3x3',
    'resconv_1x1',
    'resconv_3x3',
    'resdilconv_3x3',
]

OPS = {
    'skip_connect': lambda C_in, C_out: Identity(),
    'conv_1x1': lambda C_in, C_out: ConvBlock(C_in, C_out, 1),
    'conv_3x3': lambda C_in, C_out: ConvBlock(C_in, C_out, 3),
    'dilconv_3x3': lambda C_in, C_out: ConvBlock(C_in, C_out, 3, dilation=2),
    'resconv_1x1': lambda C_in, C_out: ResBlock(C_in, C_out, 1),
    'resconv_3x3': lambda C_in, C_out: ResBlock(C_in, C_out, 3),
    'resdilconv_3x3': lambda C_in, C_out: ResBlock(C_in, C_out, 3, dilation=2),
}


class ConvBlock(nn.Module):
    """Single convolution block used by RUAS search cells."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, dilation=1, groups=1):
        """Initialize convolution block.

        Args:
            C_in: Number of input channels.
            C_out: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            dilation: Convolution dilation.
            groups: Number of convolution groups.
        """
        super(ConvBlock, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=padding,
                            bias=True, dilation=dilation, groups=groups)

    def forward(self, x):
        """Apply convolution block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.op(x)


class ResBlock(nn.Module):
    """Residual convolution block used by RUAS search cells."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, dilation=1, groups=1):
        """Initialize residual block.

        Args:
            C_in: Number of input channels.
            C_out: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            dilation: Convolution dilation.
            groups: Number of convolution groups.
        """
        super(ResBlock, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=padding,
                            bias=True, dilation=dilation, groups=groups)

    def forward(self, x):
        """Apply residual convolution block.

        Args:
            x: Input tensor.

        Returns:
            Residual output tensor.
        """
        return self.op(x) + x


class Identity(nn.Module):
    """Identity operation used by RUAS search cells."""

    def __init__(self):
        """Initialize identity operation."""
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input tensor unchanged.

        Args:
            x: Input tensor.

        Returns:
            The unchanged input tensor.
        """
        return x


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

IEM_Genotype = Genotype(
    normal=[('skip_connect', 0), ('resconv_1x1', 1), ('resdilconv_3x3', 2),
            ('conv_3x3', 3), ('conv_3x3', 4), ('skip_connect', 5), ('conv_3x3', 6)],
    normal_concat=None, reduce=None, reduce_concat=None
)

NRM_Genotype = Genotype(
    normal=[('resconv_1x1', 0), ('resconv_1x1', 1), ('resdilconv_3x3', 2),
            ('skip_connect', 3), ('resconv_1x1', 4), ('resconv_1x1', 5), ('skip_connect', 6)],
    normal_concat=None, reduce=None, reduce_concat=None
)


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    """Create a padded convolution layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        dilation: Convolution dilation.
        groups: Number of convolution groups.

    Returns:
        Configured ``nn.Conv2d`` layer.
    """
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding,
                     bias=True, dilation=dilation, groups=groups)


class SearchBlock(nn.Module):
    """RUAS searched feature block."""

    def __init__(self, channel, genotype):
        """Initialize a searched block from a genotype.

        Args:
            channel: Number of feature channels.
            genotype: RUAS genotype describing operations.
        """
        super(SearchBlock, self).__init__()
        self.stride = 1
        self.channel = channel

        op_names, indices = zip(*genotype.normal)

        self.dc = self.distilled_channels = self.channel
        self.rc = self.remaining_channels = self.channel
        self.c1_d = OPS[op_names[0]](self.channel, self.dc)
        self.c1_r = OPS[op_names[1]](self.channel, self.rc)
        self.c2_d = OPS[op_names[2]](self.channel, self.dc)
        self.c2_r = OPS[op_names[3]](self.channel, self.rc)
        self.c3_d = OPS[op_names[4]](self.channel, self.dc)
        self.c3_r = OPS[op_names[5]](self.channel, self.rc)
        self.c4 = OPS[op_names[6]](self.channel, self.dc)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=False)
        self.c5 = conv_layer(self.dc * 4, self.channel, 1)

    def forward(self, input):
        """Run the searched block.

        Args:
            input: Input feature tensor.

        Returns:
            Output feature tensor.
        """
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.c5(out)

        return out_fused


class IEM(nn.Module):
    """RUAS illumination estimation module."""

    def __init__(self, channel, genetype):
        """Initialize illumination estimation module.

        Args:
            channel: Number of feature channels.
            genetype: RUAS genotype for the search block.
        """
        super(IEM, self).__init__()
        self.channel = channel
        self.genetype = genetype

        self.cell = SearchBlock(self.channel, self.genetype)
        self.activate = nn.Sigmoid()

    def max_operation(self, x):
        """Apply local maximum operation.

        Args:
            x: Input tensor.

        Returns:
            Local maximum tensor.
        """
        pad = nn.ConstantPad2d(1, 0)
        x = pad(x)[:, :, 1:, 1:]
        x = torch.max(x[:, :, :-1, :], x[:, :, 1:, :])
        x = torch.max(x[:, :, :, :-1], x[:, :, :, 1:])
        return x

    def forward(self, input_y, input_u, k):
        """Estimate enhanced image and transmission map.

        Args:
            input_y: Original low-light input tensor.
            input_u: Previous enhanced tensor.
            k: Iteration index.

        Returns:
            A tuple ``(u, t)`` containing enhanced image and transmission map.
        """
        if k == 0:
            t_hat = self.max_operation(input_y)
        else:
            t_hat = self.max_operation(input_u) - 0.5 * (input_u - input_y)
        t = t_hat
        t = self.cell(t)
        t = self.activate(t)
        t = torch.clamp(t, 0.001, 1.0)
        u = torch.clamp(input_y / t, 0.0, 1.0)

        return u, t


class EnhanceNetwork(nn.Module):
    """RUAS enhancement network composed of IEM stages."""

    def __init__(self, iteration, channel, genotype):
        """Initialize enhancement network.

        Args:
            iteration: Number of IEM iterations.
            channel: Number of feature channels.
            genotype: RUAS enhancement genotype.
        """
        super(EnhanceNetwork, self).__init__()
        self.iem_nums = iteration
        self.channel = channel
        self.genotype = genotype

        self.iems = nn.ModuleList()
        for i in range(self.iem_nums):
            self.iems.append(IEM(self.channel, self.genotype))

    def max_operation(self, x):
        """Apply local maximum operation.

        Args:
            x: Input tensor.

        Returns:
            Local maximum tensor.
        """
        pad = nn.ConstantPad2d(1, 0)
        x = pad(x)[:, :, 1:, 1:]
        x = torch.max(x[:, :, :-1, :], x[:, :, 1:, :])
        x = torch.max(x[:, :, :, :-1], x[:, :, :, 1:])
        return x

    def forward(self, input):
        """Run RUAS enhancement stages.

        Args:
            input: Low-light input tensor.

        Returns:
            A tuple containing enhanced image list and transmission map list.
        """
        t_list = []
        u_list = []
        u = torch.ones_like(input)
        for i in range(self.iem_nums):
            u, t = self.iems[i](input, u, i)
            u_list.append(u)
            t_list.append(t)
        return u_list, t_list


class DenoiseNetwork(nn.Module):
    """RUAS denoising network."""

    def __init__(self, layers, channel, genotype):
        """Initialize denoising network.

        Args:
            layers: Number of denoising blocks.
            channel: Number of feature channels.
            genotype: RUAS denoising genotype.
        """
        super(DenoiseNetwork, self).__init__()
        self.nrm_nums = layers
        self.channel = channel
        self.genotype = genotype
        self.stem = conv_layer(3, self.channel, 3)
        self.nrms = nn.ModuleList()
        for i in range(self.nrm_nums):
            self.nrms.append(SearchBlock(self.channel, genotype))
        self.activate = nn.Sequential(conv_layer(self.channel, 3, 3))

    def forward(self, input):
        """Run RUAS denoising network.

        Args:
            input: Enhanced image tensor.

        Returns:
            A tuple containing denoised image and estimated noise.
        """
        feat = self.stem(input)
        for i in range(self.nrm_nums):
            feat = self.nrms[i](feat)
        n = self.activate(feat)
        output = input - n
        return output, n


class RUAS(LLVModel):

    task = "llie"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize RUAS.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default RUAS configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update({
            'iem_nums': 3,  # Number of IEM iterations
            'nrm_nums': 3,  # Number of NRM layers
            'enhance_channel': 3,  # Channel number for enhance network
            'denoise_channel': 6,  # Channel number for denoise network
            'mode': 'inference',  # 'train' or 'inference'
            'pretrained_denoise_path': None,  # Path to pre-trained denoise model
        })
        return default_config

    def _validate_config(self):
        """Validate RUAS configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()

        if self.config['iem_nums'] <= 0:
            raise ValueError("'iem_nums' must be positive")
        if self.config['nrm_nums'] <= 0:
            raise ValueError("'nrm_nums' must be positive")
        if self.config['mode'] not in ['train', 'inference']:
            raise ValueError("'mode' must be 'train' or 'inference'")

    def _init_model(self):
        """Initialize RUAS enhancement and denoising networks."""
        enhance_genotype = IEM_Genotype
        denoise_genotype = NRM_Genotype

        self.enhance_net = EnhanceNetwork(
            iteration=self.config['iem_nums'],
            channel=self.config['enhance_channel'],
            genotype=enhance_genotype
        )

        self.denoise_net = DenoiseNetwork(
            layers=self.config['nrm_nums'],
            channel=self.config['denoise_channel'],
            genotype=denoise_genotype
        )

        if self.config.get('pretrained_denoise_path'):
            self._load_pretrained_denoise()

    def _load_pretrained_denoise(self):
        """Load pre-trained denoise model weights if configured."""
        try:
            model_dict = torch.load(
                self.config['pretrained_denoise_path'],
                map_location="cpu",
            )
            self.denoise_net.load_state_dict(model_dict)
            print(f"✅ Pre-trained denoise model loaded from: {self.config['pretrained_denoise_path']}")
        except Exception as e:
            print(f"⚠️  Failed to load pre-trained denoise model: {e}")

    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a RUAS forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.
            **kwargs: Additional forward-pass parameters.

        Returns:
            Training mode: standardized output dict containing enhanced image
            and intermediate RUAS lists. Inference mode: final enhanced image.
        """
        u_list, t_list = self.enhance_net(x)
        u_d, noise = self.denoise_net(u_list[-1])
        u_list.append(u_d)

        if self.config['mode'] == 'train':
            return self._format_output(
                pred=u_list[-1],
                aux={
                    'u_list': u_list,
                    't_list': t_list,
                    'noise': noise,
                },
                meta={'mode': self.config['mode']},
            )

        return u_list[-1]

    def get_optimizers(self):
        """Create RUAS optimizers matching the original implementation.

        Returns:
            Dictionary containing enhancement and denoise optimizers.
        """
        return {
            'enhancement_optimizer': torch.optim.SGD(
                self.enhance_net.parameters(),
                lr=0.015,
                momentum=0.9,
                weight_decay=3e-4
            ),
            'denoise_optimizer': torch.optim.SGD(
                self.denoise_net.parameters(),
                lr=0.001,
                momentum=0.9,
                weight_decay=3e-4
            )
        }
