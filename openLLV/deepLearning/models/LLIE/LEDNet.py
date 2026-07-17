"""
LEDNet model for low-light enhancement and deblurring.

Original paper: LEDNet: Joint Low-light Enhancement and Deblurring in the Dark
Paper link: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660562.pdf
Official source code: https://github.com/sczhou/LEDNet
Official project url: https://shangchenzhou.com/projects/LEDNet/
"""

from ..BaseModel import LLVModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


class KernelConv2D(nn.Module):
    """Dynamic 2D kernel convolution used by LEDNet."""

    def __init__(self, ksize=5, act=True):
        """Initialize dynamic kernel convolution.

        Args:
            ksize: Dynamic kernel size.
            act: Whether to apply LeakyReLU after convolution.
        """
        super(KernelConv2D, self).__init__()
        self.ksize = ksize
        self.act = act

    def forward(self, feat_in, kernel):
        """Apply spatially varying kernel convolution.

        Args:
            feat_in: Input feature tensor.
            kernel: Dynamic kernel tensor.

        Returns:
            Filtered feature tensor.
        """
        channels = feat_in.size(1)
        N, kernels, H, W = kernel.size()
        pad = (self.ksize - 1) // 2

        feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
        feat_in = feat_in.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        feat_in = feat_in.permute(0, 2, 3, 1, 4, 5).contiguous()
        feat_in = feat_in.reshape(N, H, W, channels, -1)

        kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, -1)
        feat_out = torch.sum(feat_in * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        if self.act:
            feat_out = F.leaky_relu(feat_out, negative_slope=0.2, inplace=True)
        return feat_out


def get_pad_layer(pad_type):
    """Get a padding layer class by name.

    Args:
        pad_type: Padding type, such as ``"reflect"``, ``"replicate"``, or
            ``"zero"``.

    Returns:
        Padding layer class.
    """
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
        PadLayer = nn.ReflectionPad2d
    return PadLayer


class Downsample(nn.Module):
    """Anti-aliased downsampling layer."""

    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        """Initialize downsampling layer.

        Args:
            pad_type: Padding type.
            filt_size: Low-pass filter size.
            stride: Downsampling stride.
            channels: Number of input channels.
            pad_off: Extra padding offset.
        """
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2)),
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2))
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        if self.filt_size == 1:
            a = np.array([1., ])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        """Downsample an input tensor.

        Args:
            inp: Input tensor.

        Returns:
            Downsampled tensor.
        """
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class PPM(nn.Module):
    """Pyramid Pooling Module"""

    def __init__(self, in_dim, reduction_dim, bins):
        """Initialize pyramid pooling module.

        Args:
            in_dim: Number of input channels.
            reduction_dim: Number of channels in pooled branches.
            bins: Pooling bin sizes.
        """
        super(PPM, self).__init__()
        self.features = nn.ModuleList()
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim + reduction_dim * len(bins), in_dim, kernel_size=3, padding=1, bias=False),
            nn.PReLU()
        )

    def forward(self, x):
        """Apply pyramid pooling and feature fusion.

        Args:
            x: Input feature tensor.

        Returns:
            Fused feature tensor.
        """
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat


class ResidualDownSample(nn.Module):
    """Residual Downsample Block"""

    def __init__(self, in_channels, out_channels, bias=False):
        """Initialize residual downsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bias: Whether convolution layers use bias.
        """
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
            nn.PReLU(),
            Downsample(channels=in_channels, filt_size=3, stride=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias)
        )

        self.bot = nn.Sequential(
            Downsample(channels=in_channels, filt_size=3, stride=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        """Apply residual downsampling.

        Args:
            x: Input feature tensor.

        Returns:
            Downsampled feature tensor.
        """
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class ResidualUpSample(nn.Module):
    """Residual Upsample Block"""

    def __init__(self, in_channels, out_channels, bias=False):
        """Initialize residual upsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bias: Whether convolution layers use bias.
        """
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias)
        )

        self.bot = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        """Apply residual upsampling.

        Args:
            x: Input feature tensor.

        Returns:
            Upsampled feature tensor.
        """
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class BasicBlock_E(nn.Module):
    """Encoder Basic Block"""

    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        """Initialize LEDNet encoder block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            mode: Optional reshape mode. ``"down"`` enables downsampling.
            bias: Whether convolution layers use bias.
        """
        super(BasicBlock_E, self).__init__()
        self.mode = mode

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.PReLU()
        )
        if mode == 'down':
            self.reshape_conv = ResidualDownSample(in_channels, out_channels)

    def forward(self, x):
        """Run encoder block.

        Args:
            x: Input feature tensor.

        Returns:
            Output feature tensor.
        """
        res = self.body1(x)
        out = res + x
        out = self.body2(out)
        if self.mode is not None:
            out = self.reshape_conv(out)
        return out


class BasicBlock_D_2Res(nn.Module):
    """Decoder Basic Block with 2 Residuals"""

    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        """Initialize LEDNet decoder block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            mode: Optional reshape mode. ``"up"`` enables upsampling.
            bias: Whether convolution layers use bias.
        """
        super(BasicBlock_D_2Res, self).__init__()
        self.mode = mode
        if mode == 'up':
            self.reshape_conv = ResidualUpSample(in_channels, out_channels)

        self.body1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias)
        )

    def forward(self, x):
        """Run decoder block.

        Args:
            x: Input feature tensor.

        Returns:
            Output feature tensor.
        """
        if self.mode is not None:
            x = self.reshape_conv(x)
        res1 = self.body1(x)
        out1 = res1 + x
        res2 = self.body2(out1)
        out2 = res2 + out1
        return out2


class CurveCALayer(nn.Module):
    """Curve-based Channel Attention Layer"""

    def __init__(self, channel, n_curve):
        """Initialize curve-based channel attention.

        Args:
            channel: Number of feature channels.
            n_curve: Number of curve iterations.
        """
        super(CurveCALayer, self).__init__()
        self.n_curve = n_curve
        self.relu = nn.ReLU(inplace=False)
        self.predict_a = nn.Sequential(
            nn.Conv2d(channel, channel, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, n_curve, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply curve-based channel attention.

        Args:
            x: Input feature tensor.

        Returns:
            Attention-enhanced feature tensor.
        """
        a = self.predict_a(x)
        x = self.relu(x) - self.relu(x - 1)
        for i in range(self.n_curve):
            x = x + a[:, i:i + 1] * x * (1 - x)
        return x


class LEDNet(LLVModel):

    task = "llie"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize LEDNet.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default LEDNet configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update({
            'channels': [32, 64, 128, 128],  # Channel configuration [ch1, ch2, ch3, ch4]
            'connection': False,  # Whether to use skip connections
            'use_side_loss': False,  # Whether to use side supervision for training
            'mode': 'inference',  # Mode: 'train' or 'inference'
            'kernel_size': 5,  # Kernel size for dynamic convolution
            'curve_n': 3,  # Number of curve iterations in attention
            'ppm_bins': (1, 2, 3, 6),  # Bin sizes for Pyramid Pooling Module
        })
        return default_config

    def _validate_config(self):
        """Validate LEDNet configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()

        if len(self.config['channels']) != 4:
            raise ValueError("'channels' must be a list of 4 integers")
        if any(c <= 0 for c in self.config['channels']):
            raise ValueError("All channel values must be positive")
        if self.config['kernel_size'] % 2 == 0:
            raise ValueError("'kernel_size' must be odd")
        if self.config['curve_n'] <= 0:
            raise ValueError("'curve_n' must be positive")
        if self.config['mode'] not in ['train', 'inference']:
            raise ValueError("'mode' must be 'train' or 'inference'")

    def _init_model(self):
        """Initialize LEDNet architecture based on configuration."""
        ch1, ch2, ch3, ch4 = self.config['channels']
        connection = self.config['connection']
        ks_2d = self.config['kernel_size']
        curve_n = self.config['curve_n']

        # Encoder blocks
        self.E_block1 = nn.Sequential(
            nn.Conv2d(self.config['input_channels'], ch1, 3, stride=1, padding=1),
            nn.PReLU(),
            BasicBlock_E(ch1, ch2, mode='down')
        )
        self.E_block2 = BasicBlock_E(ch2, ch3, mode='down')
        self.E_block3 = BasicBlock_E(ch3, ch4, mode='down')

        # Side output for auxiliary supervision (used in training)
        self.side_out = nn.Conv2d(ch4, self.config['input_channels'], 3, stride=1, padding=1)

        # Middle blocks
        self.M_block1 = BasicBlock_E(ch4, ch4)
        self.M_block2 = BasicBlock_E(ch4, ch4)

        # Dynamic filter generation modules
        self.conv_fac_k3 = nn.Sequential(
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch4, ch4 * ks_2d ** 2, 1, stride=1)
        )

        self.conv_fac_k2 = nn.Sequential(
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch3, ch3 * ks_2d ** 2, 1, stride=1)
        )

        self.conv_fac_k1 = nn.Sequential(
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch2, ch2 * ks_2d ** 2, 1, stride=1)
        )

        # Dynamic kernel convolution
        self.kconv_deblur = KernelConv2D(ksize=ks_2d, act=True)

        # Curve-based attention layers
        self.conv_1c = CurveCALayer(ch2, curve_n)
        self.conv_2c = CurveCALayer(ch3, curve_n)
        self.conv_3c = CurveCALayer(ch4, curve_n)

        # Pyramid Pooling Modules
        self.PPM1 = PPM(ch2, ch2 // 4, bins=self.config['ppm_bins'])
        self.PPM2 = PPM(ch3, ch3 // 4, bins=self.config['ppm_bins'])
        self.PPM3 = PPM(ch4, ch4 // 4, bins=self.config['ppm_bins'])

        # Decoder blocks
        self.D_block3 = BasicBlock_D_2Res(ch4, ch4)
        self.D_block2 = BasicBlock_D_2Res(ch4, ch3, mode='up')
        self.D_block1 = BasicBlock_D_2Res(ch3, ch2, mode='up')
        self.D_block0 = nn.Sequential(
            BasicBlock_D_2Res(ch2, ch1, mode='up'),
            nn.Conv2d(ch1, self.config['input_channels'], 3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor, side_loss: Optional[bool] = None) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a LEDNet forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.
            side_loss: Whether to compute side supervision. If None, the value
                is inferred from model configuration.

        Returns:
            Training mode: standardized output dict. Inference mode: enhanced
            image tensor.
        """
        if side_loss is None:
            side_loss = self.config.get('use_side_loss', False) and self.config['mode'] == 'train'

        # Encoder
        e_feat1 = self.E_block1(x)  # ch2, 1/2 resolution
        e_feat1 = self.PPM1(e_feat1)
        e_feat1 = self.conv_1c(e_feat1)

        e_feat2 = self.E_block2(e_feat1)  # ch3, 1/4 resolution
        e_feat2 = self.PPM2(e_feat2)
        e_feat2 = self.conv_2c(e_feat2)

        e_feat3 = self.E_block3(e_feat2)  # ch4, 1/8 resolution
        e_feat3 = self.PPM3(e_feat3)
        e_feat3 = self.conv_3c(e_feat3)

        # Side output for auxiliary loss (if enabled)
        if side_loss:
            out_side = self.side_out(e_feat3)

        # Middle processing
        m_feat = self.M_block1(e_feat3)
        m_feat = self.M_block2(m_feat)

        # Decoder with dynamic kernel convolution
        d_feat3 = self.D_block3(m_feat)  # ch4, 1/8 resolution
        kernel_3 = self.conv_fac_k3(e_feat3)
        d_feat3 = self.kconv_deblur(d_feat3, kernel_3)
        if self.config['connection']:
            d_feat3 = d_feat3 + e_feat3

        d_feat2 = self.D_block2(d_feat3)  # ch3, 1/4 resolution
        kernel_2 = self.conv_fac_k2(e_feat2)
        d_feat2 = self.kconv_deblur(d_feat2, kernel_2)
        if self.config['connection']:
            d_feat2 = d_feat2 + e_feat2

        d_feat1 = self.D_block1(d_feat2)  # ch2, 1/2 resolution
        kernel_1 = self.conv_fac_k1(e_feat1)
        d_feat1 = self.kconv_deblur(d_feat1, kernel_1)
        if self.config['connection']:
            d_feat1 = d_feat1 + e_feat1

        # Final output
        out = self.D_block0(d_feat1)

        aux = {}
        if side_loss:
            aux['side_output'] = out_side

        if self.config['mode'] == 'train':
            return self._format_output(
                pred=out,
                aux=aux,
                meta={'mode': self.config['mode'], 'side_loss': side_loss},
            )

        return out
