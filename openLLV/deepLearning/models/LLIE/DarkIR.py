"""
DarkIR model implementation.

Original paper: DarkIR: Robust Low-Light Image Restoration
Paper link: https://openaccess.thecvf.com/content/CVPR2025/papers/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.pdf
Official source code: https://github.com/cidautai/DarkIR
"""

from ..BaseModel import LLVModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List


class LayerNorm2d(nn.Module):
    """Layer normalization over channel dimension for 2D feature maps."""

    def __init__(self, channels, eps=1e-6):
        """Initialize 2D layer normalization.

        Args:
            channels: Number of channels.
            eps: Numerical stability constant.
        """
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        """Apply 2D layer normalization.

        Args:
            x: Input tensor with shape ``[B, C, H, W]``.

        Returns:
            Normalized tensor.
        """
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        y = self.weight.view(1, C, 1, 1) * y + self.bias.view(1, C, 1, 1)
        return y


class SimpleGate(nn.Module):
    """Channel split-and-multiply gate."""

    def forward(self, x):
        """Apply simple gate operation.

        Args:
            x: Input tensor whose channel dimension is split in half.

        Returns:
            Elementwise product of the two channel chunks.
        """
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FreMLP(nn.Module):
    """Frequency-domain MLP block used by DarkIR."""

    def __init__(self, nc, expand=2):
        """Initialize frequency-domain MLP.

        Args:
            nc: Number of channels.
            expand: Channel expansion factor.
        """
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        """Process features in the Fourier domain.

        Args:
            x: Input feature tensor.

        Returns:
            Reconstructed spatial-domain feature tensor.
        """
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class Branch(nn.Module):
    """Depthwise convolution branch used by DarkIR blocks."""

    def __init__(self, c, DW_Expand, dilation=1):
        """Initialize branch.

        Args:
            c: Base channel count.
            DW_Expand: Depthwise expansion factor.
            dilation: Depthwise convolution dilation.
        """
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel,
                      kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation)
        )

    def forward(self, input):
        """Run branch convolution.

        Args:
            input: Input feature tensor.

        Returns:
            Output feature tensor.
        """
        return self.branch(input)


class EBlock(nn.Module):
    """DarkIR encoder block."""

    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        """Initialize encoder block.

        Args:
            c: Number of channels.
            DW_Expand: Depthwise expansion factor.
            dilations: Dilation values for branch convolutions.
            extra_depth_wise: Whether to use an extra depthwise convolution.
        """
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True,
                                    dilation=1) if extra_depth_wise else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc=c, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        """Run encoder block.

        Args:
            inp: Input feature tensor.

        Returns:
            Output feature tensor.
        """
        y = inp
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        x_step2 = self.norm2(y)
        x_freq = self.freq(x_step2)
        x = y * x_freq
        x = y + x * self.gamma
        return x


class DBlock(nn.Module):
    """DarkIR decoder block."""

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations=[1], extra_depth_wise=False):
        """Initialize decoder block.

        Args:
            c: Number of channels.
            DW_Expand: Depthwise expansion factor.
            FFN_Expand: Feed-forward expansion factor.
            dilations: Dilation values for branch convolutions.
            extra_depth_wise: Whether to use an extra depthwise convolution.
        """
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c,
                                    bias=True, dilation=1) if extra_depth_wise else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand=1, dilation=dilation))

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        """Run decoder block.

        Args:
            inp: Input feature tensor.

        Returns:
            Output feature tensor.
        """
        x = self.norm1(inp)
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x)
        x = y + x * self.gamma
        return x


class CustomSequential(nn.Module):
    """Sequential container backed by ``nn.ModuleList``."""

    def __init__(self, *args):
        """Initialize custom sequential container.

        Args:
            *args: Modules executed in order.
        """
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x):
        """Run modules sequentially.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after all modules.
        """
        for module in self.modules_list:
            x = module(x)
        return x


class DarkIR(LLVModel):

    task = "llie"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize DarkIR.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default DarkIR configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update({
            'width': 32,
            'middle_blk_num_enc': 2,
            'middle_blk_num_dec': 2,
            'enc_blk_nums': [1, 2, 3],
            'dec_blk_nums': [3, 1, 1],
            'dilations': [1, 4, 9],
            'extra_depth_wise': True,
            'mode': 'inference',  # 'train' or 'inference'
            'side_loss': False,  # Whether to compute side loss for training
        })
        return default_config

    def _validate_config(self):
        """Validate DarkIR configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()

        if self.config['width'] <= 0:
            raise ValueError("'width' must be positive")
        if self.config['middle_blk_num_enc'] <= 0:
            raise ValueError("'middle_blk_num_enc' must be positive")
        if self.config['middle_blk_num_dec'] <= 0:
            raise ValueError("'middle_blk_num_dec' must be positive")
        if self.config['mode'] not in ['train', 'inference']:
            raise ValueError("'mode' must be 'train' or 'inference'")

    def _init_model(self):
        """Initialize DarkIR encoder, middle, decoder, and side-output layers."""
        img_channel = self.config['input_channels']
        width = self.config['width']
        middle_blk_num_enc = self.config['middle_blk_num_enc']
        middle_blk_num_dec = self.config['middle_blk_num_dec']
        enc_blk_nums = self.config['enc_blk_nums']
        dec_blk_nums = self.config['dec_blk_nums']
        dilations = self.config['dilations']
        extra_depth_wise = self.config['extra_depth_wise']

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3,
                               padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3,
                                padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks_enc = None
        self.middle_blks_dec = None
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Build encoder blocks
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                CustomSequential(
                    *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        # Build middle blocks
        self.middle_blks_enc = CustomSequential(
            *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
        )
        self.middle_blks_dec = CustomSequential(
            *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)]
        )

        # Build decoder blocks
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                CustomSequential(
                    *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        # Side output for middle loss
        self.side_out = nn.Conv2d(in_channels=width * 2 ** len(self.encoders),
                                  out_channels=img_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, side_loss: bool = False, **kwargs) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a DarkIR forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.
            side_loss: Whether to enable side-output supervision.
            **kwargs: Additional forward-pass parameters.

        Returns:
            Training mode: standardized output dict. Inference mode: restored
            image tensor.
        """
        if side_loss:
            self.config['side_loss'] = True

        _, _, H, W = x.shape
        input_tensor = self.check_image_size(x)

        # Encoder path
        x = self.intro(input_tensor)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # Middle blocks
        x_light = self.middle_blks_enc(x)

        # Side output for loss computation
        if self.config['side_loss']:
            out_side = self.side_out(x_light)

        # Decoder path
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = x + input_tensor
        out_main = x[:, :, :H, :W]

        aux = {}
        if self.config['mode'] == 'train' and self.config['side_loss']:
            aux['side_output'] = out_side

        if self.config['mode'] == 'train':
            return self._format_output(
                pred=out_main,
                aux=aux,
                meta={'mode': self.config['mode'], 'side_loss': self.config['side_loss']},
            )

        return out_main

    def check_image_size(self, x):
        """Pad input so spatial dimensions are divisible by ``padder_size``.

        Args:
            x: Input tensor with shape ``[B, C, H, W]``.

        Returns:
            Padded input tensor.
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value=0)
        return x


    def enable_side_loss(self, enable: bool = True):
        """Enable or disable side-output loss computation.

        Args:
            enable: Whether to enable side-output loss.

        Returns:
            The model itself.
        """
        self.config['side_loss'] = enable
        return self

    def get_model_info(self):
        """Get DarkIR architecture information.

        Returns:
            Dictionary containing architecture configuration values.
        """
        info = {
            'width': self.config['width'],
            'middle_blk_num_enc': self.config['middle_blk_num_enc'],
            'middle_blk_num_dec': self.config['middle_blk_num_dec'],
            'enc_blk_nums': self.config['enc_blk_nums'],
            'dec_blk_nums': self.config['dec_blk_nums'],
            'dilations': self.config['dilations'],
            'extra_depth_wise': self.config['extra_depth_wise'],
        }
        return info
