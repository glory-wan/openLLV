"""Official standard LLFlow (LOL-pc) adapted to the LLVModel interface.

Upstream: https://github.com/wyf0912/LLFlow
Commit: 115da161a96de868d67494a32db848e31f85bbc1
Copyright (c) 2021 Yufei Wang. CC BY-NC-SA 4.0; see LLFlow.LICENSE.
Official condition encoder, flow layers and helpers are consolidated here.
Parameter names, architecture and conditional Gaussian NLL are preserved.
"""

import functools
import math
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from ..BaseModel import LLVModel
from openLLV.data.datasets.LLFlowDataset import prepare_llflow_input, reflect_pad16


def opt_get(options, keys, default=None):
    for key in keys:
        if not isinstance(options, dict) or key not in options:
            return default
        options = options[key]
    return options


def standard_options(config):
    """Build the architecture options of official LOL-pc.yml."""
    return {
        "scale": 1,
        "cond_encoder": "ConEncoder1",
        "concat_histeq": True,
        "concat_color_map": False,
        "gray_map": False,
        "le_curve": False,
        "sigmoid_output": False,
        "datasets": {"train": {"GT_size": 160, "quant": config["quant"]}},
        "network_G": {"flow": {
            "K": config["K"], "L": 3,
            "coupling": "CondAffineSeparatedAndCond",
            "additionalFlowNoAffine": 2,
            "split": {"enable": False}, "fea_up0": True,
            "stackRRDB": {"blocks": [1, 3, 5, 7], "concat": True},
        }},
    }


def _sum(tensor, dim=None, keepdim=False):
    if dim is None:
        # sum up all dim
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


def _mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


def _split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def _cat_feature(tensor_a, tensor_b):
    return torch.cat((tensor_a, tensor_b), dim=1)


def _pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def f_conv2d_bias(in_channels, out_channels):
    def padding_same(kernel, stride):
        return [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)]

    padding = padding_same([3, 3], [1, 1])
    assert padding == [1, 1], padding
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=[3, 3], stride=1, padding=1,
                  bias=True))


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        # Preserve loaded statistics even when a valid learned bias is zero.
        # Keep this flag non-persistent to retain official state-dict keys.
        if prefix + "bias" in state_dict and prefix + "logs" in state_dict:
            self.inited = True

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        if (self.bias != 0).any():
            self.inited = True
            return
        assert input.device == self.bias.device, (input.device, self.bias.device)
        with torch.no_grad():
            bias = _mean(input.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = _mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, reverse=False, offset=None):
        bias = self.bias

        if offset is not None:
            bias = bias + offset

        if not reverse:
            return input + bias
        else:
            return input - bias

    def _scale(self, input, logdet=None, reverse=False, offset=None):
        logs = self.logs

        if offset is not None:
            logs = logs + offset

        if not reverse:
            input = input * torch.exp(logs) # should have shape batchsize, n_channels, 1, 1
            # input = input * torch.exp(logs+logs_offset)
        else:
            input = input * torch.exp(-logs)
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            dlogdet = _sum(logs) * _pixels(input)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, reverse=False, offset_mask=None, logs_offset=None, bias_offset=None):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)

        if offset_mask is not None:
            logs_offset *= offset_mask
            bias_offset *= offset_mask
        # no need to permute dims as old version
        if not reverse:
            # center and scale

            # self.input = input
            input = self._center(input, reverse, bias_offset)
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
            input = self._center(input, reverse, bias_offset)
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


class MaskedActNorm2d(ActNorm2d):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def forward(self, input, mask, logdet=None, reverse=False):

        assert mask.dtype == torch.bool
        output, logdet_out = super().forward(input, logdet, reverse)

        input[mask] = output[mask]
        logdet[mask] = logdet_out[mask]

        return input, logdet


class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", do_actnorm=True, weight_std=0.05):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", logscale_factor=3):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        if mean is None and logs is None:
            return -0.5 * (x ** 2 + GaussianDiag.Log2PI)
        else:
            return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = 0
        if isinstance(x, (list, tuple)):
            for x_ in x: likelihood += _sum(GaussianDiag.likelihood(mean, logs, x_), dim=[1, 2, 3])
        else:
            likelihood = _sum(GaussianDiag.likelihood(mean, logs, x), dim=[1, 2, 3])
        return likelihood
        # likelihood = GaussianDiag.likelihood(mean, logs, x)
        # return _sum(likelihood, dim=[1, 2, 3])

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def sample_eps(shape, eps_std, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        eps = torch.normal(mean=torch.zeros(shape),
                           std=torch.ones(shape) * eps_std)
        return eps


def squeeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor2) == 0, "{}".format(C)
    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)
    return x


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = squeeze2d(input, self.factor)  # Squeeze in forward
            return output, logdet
        else:
            output = unsqueeze2d(input, self.factor)
            return output, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        pixels = _pixels(input)
        # Preserve the exact transform and Jacobian. Upstream's retry loop can
        # hang forever on singular weights; random perturbations also change the
        # transform without changing the trained parameter.
        dlogdet = torch.linalg.slogdet(self.weight.float())[1] * pixels
        if not torch.isfinite(dlogdet):
            raise RuntimeError("LLFlow invertible convolution has a singular or non-finite weight.")
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.linalg.inv(self.weight.double()).to(self.weight.dtype) \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight, dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = opt_get(opt, ['network_G', 'flow', 'conditionInFeaDim'], 320)
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)
        self.opt = opt
        self.le_curve = opt['le_curve'] if opt['le_curve'] is not None else False
        if self.le_curve:
            self.fCurve = self.F(in_channels=self.in_channels_rrdb,
                                 out_channels=self.in_channels,
                                 hidden_channels=self.hidden_channels,
                                 kernel_hidden=self.kernel_hidden,
                                 n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Curve conditional
            if self.le_curve:
                # logdet = logdet + _sum(torch.log(torch.sigmoid(z) * (1 - torch.sigmoid(z))), dim=[1, 2, 3])
                # z = torch.sigmoid(z)
                # alpha = self.fCurve(ft)
                # alpha = (torch.tanh(alpha + 2.) + self.affine_eps)
                # logdet = logdet + _sum(torch.log((1 + alpha - 2 * z * alpha).abs()), dim=[1, 2, 3])
                # z = z + alpha * z * (1 - z)

                alpha = self.fCurve(ft)
                # alpha = (torch.sigmoid(alpha + 2.) + self.affine_eps)
                alpha = torch.relu(alpha) + self.affine_eps
                logdet = logdet + _sum(torch.log(alpha * torch.pow(z.abs(), alpha - 1)) + self.affine_eps)
                z = torch.pow(z.abs(), alpha) * z.sign()

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = _cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = _cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            # Curve conditional
            if self.le_curve:
                # alpha = self.fCurve(ft)
                # alpha = (torch.sigmoid(alpha + 2.) + self.affine_eps)
                # z = (1 + alpha) / alpha - (
                #             alpha + torch.pow(2 * alpha - 4 * alpha * z + torch.pow(alpha, 2) + 1, 0.5) + 1) / (
                #             2 * alpha)
                # z = torch.log((z / (1 - z)).clamp(1 / 1000, 1000))

                alpha = self.fCurve(ft)
                alpha = torch.relu(alpha) + self.affine_eps
                # alpha = (torch.sigmoid(alpha + 2.) + self.affine_eps)
                z = torch.pow(z.abs(), 1 / alpha) * z.sign()

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return _sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = _split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_aff(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        h = f(z)
        shift, scale = _split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


def getConditional(rrdbResults, position):
    img_ft = rrdbResults if isinstance(rrdbResults, torch.Tensor) else rrdbResults[position]
    return img_ft


class FlowStep(nn.Module):
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "squeeze_invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_alternating_2_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlign": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1SubblocksShuf": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder4": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive",
                 LU_decomposed=False, opt=None, image_injector=None, idx=None, acOpt=None, normOpt=None, in_shape=None,
                 position=None):
        # check configures
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.image_injector = image_injector

        self.norm_type = normOpt['type'] if normOpt else 'ActNorm2d'
        self.position = normOpt['position'] if normOpt else None

        self.in_shape = in_shape
        self.position = position
        self.acOpt = acOpt

        # 1. actnorm
        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)

        # 3. coupling
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = CondAffineSeparatedAndCond(in_channels=in_channels,
                                                                                                opt=opt)
        elif flow_coupling == "noCoupling":
            pass
        else:
            raise RuntimeError("coupling not Found:", flow_coupling)

    def forward(self, input, logdet=None, reverse=False, rrdbResults=None):
        if not reverse:
            return self.normal_flow(input, logdet, rrdbResults)
        else:
            return self.reverse_flow(input, logdet, rrdbResults)

    def normal_flow(self, z, logdet, rrdbResults=None):
        if self.flow_coupling == "bentIdentityPreAct":
            z, logdet = self.bentIdentPar(z, logdet, reverse=False)

        # 1. actnorm
        if self.norm_type == "ConditionalActNormImageInjector":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.actnorm(z, img_ft=img_ft, logdet=logdet, reverse=False)
        elif self.norm_type == "noNorm":
            pass
        else:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)

        need_features = self.affine_need_features()

        # 3. coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=False, ft=img_ft)
        return z, logdet

    def reverse_flow(self, z, logdet, rrdbResults=None):

        need_features = self.affine_need_features()

        # 1.coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=True, ft=img_ft)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features


class Split2d(nn.Module):
    def __init__(self, num_channels, logs_eps=0, cond_channels=0, position=None, consume_ratio=0.5, opt=None):
        super().__init__()

        self.num_channels_consume = int(round(num_channels * consume_ratio))
        self.num_channels_pass = num_channels - self.num_channels_consume

        self.conv = Conv2dZeros(in_channels=self.num_channels_pass + cond_channels,
                                out_channels=self.num_channels_consume * 2)
        self.logs_eps = logs_eps
        self.position = position
        self.opt = opt

    def split2d_prior(self, z, ft):
        if ft is not None:
            z = torch.cat([z, ft], dim=1)
        h = self.conv(z)
        return _split_feature(h, "cross")

    def exp_eps(self, logs):
        return torch.exp(logs) + self.logs_eps

    def forward(self, input, logdet=0., reverse=False, eps_std=None, eps=None, ft=None, y_onehot=None):
        if not reverse:
            # self.input = input
            z1, z2 = self.split_ratio(input)
            mean, logs = self.split2d_prior(z1, ft)

            eps = (z2 - mean) / self.exp_eps(logs)

            logdet = logdet + self.get_logdet(logs, mean, z2)

            # print(logs.shape, mean.shape, z2.shape)
            # self.eps = eps
            # print('split, enc eps:', eps)
            return z1, logdet, eps
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1, ft)

            if eps is None:
                #print("WARNING: eps is None, generating eps untested functionality!")
                eps = GaussianDiag.sample_eps(mean.shape, eps_std)

            eps = eps.to(mean.device)
            z2 = mean + self.exp_eps(logs) * eps

            z = _cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(logs, mean, z2)

            return z, logdet
            # return z, logdet, eps

    def get_logdet(self, logs, mean, z2):
        logdet_diff = GaussianDiag.logp(mean, logs, z2)
        # print("Split2D: logdet diff", logdet_diff.item())
        return logdet_diff

    def split_ratio(self, input):
        z1, z2 = input[:, :self.num_channels_pass, ...], input[:, self.num_channels_pass:, ...]
        return z1, z2


class FlowUpsamplerNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L=None,
                 actnorm_scale=1.0,
                 flow_permutation=None,
                 flow_coupling="affine",
                 LU_decomposed=False, opt=None):

        super().__init__()
        self.hr_size = opt['datasets']['train']['GT_size']
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.sigmoid_output = opt['sigmoid_output'] if opt['sigmoid_output'] is not None else False
        self.L = opt_get(opt, ['network_G', 'flow', 'L'])
        self.K = opt_get(opt, ['network_G', 'flow', 'K'])
        if isinstance(self.K, int):
            self.K = [K for K in [K, ] * (self.L + 1)]

        self.opt = opt
        H, W, self.C = image_shape
        self.check_image_shape()

        if opt['scale'] == 16:
            self.levelToName = {
                0: 'fea_up16',
                1: 'fea_up8',
                2: 'fea_up4',
                3: 'fea_up2',
                4: 'fea_up1',
            }

        if opt['scale'] == 8:
            self.levelToName = {
                0: 'fea_up8',
                1: 'fea_up4',
                2: 'fea_up2',
                3: 'fea_up1',
                4: 'fea_up0'
            }

        elif opt['scale'] == 4:
            self.levelToName = {
                0: 'fea_up4',
                1: 'fea_up2',
                2: 'fea_up1',
                3: 'fea_up0',
                4: 'fea_up-1'
            }
        elif opt['scale'] == 1:
            self.levelToName = {
                # 0: 'fea_up4',
                1: 'fea_up2',
                2: 'fea_up1',
                3: 'fea_up0',
                # 4: 'fea_up-1'
            }

        affineInCh = self.get_affineInCh(opt_get)
        flow_permutation = self.get_flow_permutation(flow_permutation, opt)

        normOpt = opt_get(opt, ['network_G', 'flow', 'norm'])

        conditional_channels = {}
        n_rrdb = self.get_n_rrdb_channels(opt, opt_get)
        n_bypass_channels = opt_get(opt, ['network_G', 'flow', 'levelConditional', 'n_channels'])
        conditional_channels[0] = n_rrdb
        for level in range(1, self.L + 1):
            # Level 1 gets conditionals from 2, 3, 4 => L - level
            # Level 2 gets conditionals from 3, 4
            # Level 3 gets conditionals from 4
            # Level 4 gets conditionals from None
            n_bypass = 0 if n_bypass_channels is None else (self.L - level) * n_bypass_channels
            conditional_channels[level] = n_rrdb + n_bypass

        # Upsampler
        for level in range(1, self.L + 1):
            # 1. Squeeze
            H, W = self.arch_squeeze(H, W)

            # 2. K FlowStep
            self.arch_additionalFlowAffine(H, LU_decomposed, W, actnorm_scale, hidden_channels, opt)
            self.arch_FlowStep(H, self.K[level], LU_decomposed, W, actnorm_scale, affineInCh, flow_coupling,
                               flow_permutation,
                               hidden_channels, normOpt, opt, opt_get,
                               n_conditinal_channels=conditional_channels[level])
            # Split
            self.arch_split(H, W, level, self.L, opt, opt_get)

        if opt_get(opt, ['network_G', 'flow', 'split', 'enable']):
            self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64 // 2 // 2)
        else:
            self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64)

        self.H = H
        self.W = W
        self.scaleH = opt['datasets']['train']['GT_size'] / H
        self.scaleW = opt['datasets']['train']['GT_size'] / W

    def get_n_rrdb_channels(self, opt, opt_get):
        blocks = opt_get(opt, ['network_G', 'flow', 'stackRRDB', 'blocks'])
        n_rrdb = 64 if blocks is None else (len(blocks) + 1) * 64
        return n_rrdb

    def arch_FlowStep(self, H, K, LU_decomposed, W, actnorm_scale, affineInCh, flow_coupling, flow_permutation,
                      hidden_channels, normOpt, opt, opt_get, n_conditinal_channels=None):
        condAff = self.get_condAffSetting(opt, opt_get)
        if condAff is not None:
            condAff['in_channels_rrdb'] = n_conditinal_channels

        for k in range(K):
            position_name = get_position_name(H, self.opt['scale'], opt)
            if normOpt: normOpt['position'] = position_name

            self.layers.append(
                FlowStep(in_channels=self.C,
                         hidden_channels=hidden_channels,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         acOpt=condAff,
                         position=position_name,
                         LU_decomposed=LU_decomposed, opt=opt, idx=k, normOpt=normOpt))
            self.output_shapes.append(
                [-1, self.C, H, W])

    def get_condAffSetting(self, opt, opt_get):
        condAff = opt_get(opt, ['network_G', 'flow', 'condAff']) or None
        condAff = opt_get(opt, ['network_G', 'flow', 'condFtAffine']) or condAff
        return condAff

    def arch_split(self, H, W, L, levels, opt, opt_get):
        correct_splits = opt_get(opt, ['network_G', 'flow', 'split', 'correct_splits'], False)
        correction = 0 if correct_splits else 1
        if opt_get(opt, ['network_G', 'flow', 'split', 'enable']) and L < levels - correction:
            logs_eps = opt_get(opt, ['network_G', 'flow', 'split', 'logs_eps']) or 0
            consume_ratio = opt_get(opt, ['network_G', 'flow', 'split', 'consume_ratio']) or 0.5
            position_name = get_position_name(H, self.opt['scale'], opt)
            position = position_name if opt_get(opt, ['network_G', 'flow', 'split', 'conditional']) else None
            cond_channels = opt_get(opt, ['network_G', 'flow', 'split', 'cond_channels'])
            cond_channels = 0 if cond_channels is None else cond_channels

            t = opt_get(opt, ['network_G', 'flow', 'split', 'type'], 'Split2d')

            if t == 'Split2d':
                split = Split2d(num_channels=self.C, logs_eps=logs_eps, position=position,
                                                     cond_channels=cond_channels, consume_ratio=consume_ratio, opt=opt)
            self.layers.append(split)
            self.output_shapes.append([-1, split.num_channels_pass, H, W])
            self.C = split.num_channels_pass

    def arch_additionalFlowAffine(self, H, LU_decomposed, W, actnorm_scale, hidden_channels, opt):
        if 'additionalFlowNoAffine' in opt['network_G']['flow']:
            n_additionalFlowNoAffine = int(opt['network_G']['flow']['additionalFlowNoAffine'])
            for _ in range(n_additionalFlowNoAffine):
                self.layers.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation='invconv',
                             flow_coupling='noCoupling',
                             LU_decomposed=LU_decomposed, opt=opt))
                self.output_shapes.append(
                    [-1, self.C, H, W])

    def arch_squeeze(self, H, W):
        self.C, H, W = self.C * 4, H // 2, W // 2
        self.layers.append(SqueezeLayer(factor=2))
        self.output_shapes.append([-1, self.C, H, W])
        return H, W

    def get_flow_permutation(self, flow_permutation, opt):
        flow_permutation = opt['network_G']['flow'].get('flow_permutation', 'invconv')
        return flow_permutation

    def get_affineInCh(self, opt_get):
        affineInCh = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        affineInCh = (len(affineInCh) + 1) * 64
        return affineInCh

    def check_image_shape(self):
        assert self.C == 1 or self.C == 3, ("image_shape should be HWC, like (64, 64, 3)"
                                            "self.C == 1 or self.C == 3")

    def forward(self, gt=None, rrdbResults=None, z=None, epses=None, logdet=0., reverse=False, eps_std=None,
                y_onehot=None):

        if reverse:
            epses_copy = [eps for eps in epses] if isinstance(epses, list) else epses

            sr, logdet = self.decode(rrdbResults, z, eps_std, epses=epses_copy, logdet=logdet, y_onehot=y_onehot)
            if self.sigmoid_output:
                sr = torch.sigmoid(sr)
            return sr, logdet
        else:
            assert gt is not None
            # assert rrdbResults is not None
            if self.sigmoid_output:
                gt = torch.log((gt / (1 - gt)).clamp(1 / 1000, 1000))
            z, logdet = self.encode(gt, rrdbResults, logdet=logdet, epses=epses, y_onehot=y_onehot)

            return z, logdet

    def encode(self, gt, rrdbResults, logdet=0.0, epses=None, y_onehot=None):
        fl_fea = gt
        reverse = False
        level_conditionals = {}
        bypasses = {}

        L = opt_get(self.opt, ['network_G', 'flow', 'L'])

        for level in range(1, L + 1):
            bypasses[level] = torch.nn.functional.interpolate(gt, scale_factor=2 ** -level, mode='bilinear',
                                                              align_corners=False)

        for layer, shape in zip(self.layers, self.output_shapes):
            size = shape[2]
            level = int(np.log(self.hr_size / size) / np.log(2))
            if level > 0 and level not in level_conditionals.keys():
                if rrdbResults is None:
                    level_conditionals[level] = None
                else:
                    level_conditionals[level] = rrdbResults[self.levelToName[level]]

            if isinstance(layer, FlowStep):
                fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse, rrdbResults=level_conditionals[level])
            elif isinstance(layer, Split2d):
                fl_fea, logdet = self.forward_split2d(epses, fl_fea, layer, logdet, reverse, level_conditionals[level],
                                                      y_onehot=y_onehot)
            else:
                fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse)

        z = fl_fea

        if not isinstance(epses, list):
            return z, logdet

        epses.append(z)
        return epses, logdet

    def forward_preFlow(self, fl_fea, logdet, reverse):
        if hasattr(self, 'preFlow'):
            for l in self.preFlow:
                fl_fea, logdet = l(fl_fea, logdet, reverse=reverse)
        return fl_fea, logdet

    def forward_split2d(self, epses, fl_fea, layer, logdet, reverse, rrdbResults, y_onehot=None):
        ft = None if layer.position is None else rrdbResults[layer.position]
        fl_fea, logdet, eps = layer(fl_fea, logdet, reverse=reverse, eps=epses, ft=ft, y_onehot=y_onehot)

        if isinstance(epses, list):
            epses.append(eps)
        return fl_fea, logdet

    def decode(self, rrdbResults, z, eps_std=None, epses=None, logdet=0.0, y_onehot=None):
        z = epses.pop() if isinstance(epses, list) else z

        fl_fea = z
        # debug.imwrite("fl_fea", fl_fea)
        bypasses = {}
        level_conditionals = {}
        if not opt_get(self.opt, ['network_G', 'flow', 'levelConditional', 'conditional']) == True:
            for level in range(self.L + 1):
                if level not in self.levelToName.keys():
                    level_conditionals[level] = None
                else:
                    level_conditionals[level] = rrdbResults[self.levelToName[level]] if rrdbResults else None

        for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):
            size = shape[2]
            level = int(np.log(self.hr_size / size) / np.log(2))
            # size = fl_fea.shape[2]
            # level = int(np.log(160 / size) / np.log(2))

            if isinstance(layer, Split2d):
                fl_fea, logdet = self.forward_split2d_reverse(eps_std, epses, fl_fea, layer,
                                                              rrdbResults[self.levelToName[level]], logdet=logdet,
                                                              y_onehot=y_onehot)
            elif isinstance(layer, FlowStep):
                fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True, rrdbResults=level_conditionals[level])
            else:
                fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True)

        sr = fl_fea

        assert sr.shape[1] == 3
        return sr, logdet

    def forward_split2d_reverse(self, eps_std, epses, fl_fea, layer, rrdbResults, logdet, y_onehot=None):
        ft = None if layer.position is None else rrdbResults[layer.position]
        fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True,
                               eps=epses.pop() if isinstance(epses, list) else None,
                               eps_std=eps_std, ft=ft, y_onehot=y_onehot)
        return fl_fea, logdet


def get_position_name(H, scale, opt):
    downscale_factor = opt['datasets']['train']['GT_size'] // H
    position_name = 'fea_up{}'.format(scale / downscale_factor)
    return position_name


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ConEncoder1(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, opt=None):
        self.opt = opt
        self.gray_map_bool = False
        self.concat_color_map = False
        if opt['concat_histeq']:
            in_nc = in_nc + 3
        if opt['concat_color_map']:
            in_nc = in_nc + 3
            self.concat_color_map = True
        if opt['gray_map']:
            in_nc = in_nc + 1
            self.gray_map_bool = True
        in_nc = in_nc + 6
        super(ConEncoder1, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_second = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### downsampling
        self.downconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.downconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.awb_para = nn.Linear(nf, 3)
        self.fine_tune_color_map = nn.Sequential(nn.Conv2d(nf, 3, 1, 1),nn.Sigmoid())

    def forward(self, x, get_steps=False):
        if self.gray_map_bool:
            x = torch.cat([x, 1 - x.mean(dim=1, keepdim=True)], dim=1)
        if self.concat_color_map:
            x = torch.cat([x, x / (x.sum(dim=1, keepdim=True) + 1e-4)], dim=1)

        raw_low_input = x[:, 0:3].exp()
        # fea_for_awb = F.adaptive_avg_pool2d(fea_down8, 1).view(-1, 64)
        awb_weight = 1  # (1 + self.awb_para(fea_for_awb).unsqueeze(2).unsqueeze(3))
        low_after_awb = raw_low_input * awb_weight
        # import pdb
        # pdb.set_trace()
        color_map = low_after_awb / (low_after_awb.sum(dim=1, keepdims=True) + 1e-4)
        dx, dy = self.gradient(color_map)
        noise_map = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]
        # color_map = self.fine_tune_color_map(torch.cat([color_map, noise_map], dim=1))

        fea = self.conv_first(torch.cat([x, color_map, noise_map], dim=1))
        fea = self.lrelu(fea)
        fea = self.conv_second(fea)
        fea_head = F.max_pool2d(fea, 2)

        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        block_results = {}
        fea = fea_head
        for idx, m in enumerate(self.RRDB_trunk.children()):
            fea = m(fea)
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea
        trunk = self.trunk_conv(fea)
        # fea = F.max_pool2d(fea, 2)
        fea_down2 = fea_head + trunk

        fea_down4 = self.downconv1(F.interpolate(fea_down2, scale_factor=1 / 2, mode='bilinear', align_corners=False,
                                                 recompute_scale_factor=True))
        fea = self.lrelu(fea_down4)

        fea_down8 = self.downconv2(
            F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        # fea = self.lrelu(fea_down8)

        # fea_down16 = self.downconv3(
        #     F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        # fea = self.lrelu(fea_down16)

        results = {'fea_up0': fea_down8,
                   'fea_up1': fea_down4,
                   'fea_up2': fea_down2,
                   'fea_up4': fea_head,
                   'last_lr_fea': fea_down4,
                   'color_map': self.fine_tune_color_map(F.interpolate(fea_down2, scale_factor=2))
                   }

        # 'color_map': color_map}  # raw

        if get_steps:
            for k, v in block_results.items():
                results[k] = v
            return results
        else:
            return None

    def gradient(self, x):
        def sub_gradient(x):
            left_shift_x, right_shift_x, grad = torch.zeros_like(
                x), torch.zeros_like(x), torch.zeros_like(x)
            left_shift_x[:, :, 0:-1] = x[:, :, 1:]
            right_shift_x[:, :, 1:] = x[:, :, 0:-1]
            grad = 0.5 * (left_shift_x - right_shift_x)
            return grad

        return sub_gradient(x), sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)


class LLFlow(LLVModel):
    """Standard 64-channel, 24-RRDB, three-level conditional flow.

    Input is RGB in [0, 1], or the six-channel prepared LLFlowDataset input.
    Trainer supplies paired_image through its existing paired-forward hook.
    """

    task = "llie"
    aliases = []
    requires_paired_forward = True

    def _get_default_config(self):
        config = super()._get_default_config()
        config.update(nf=64, nb=24, K=12, L=3, train_gt_ratio=0.2,
                      quant=32, inference_padding="reflect16", mode="inference")
        return config

    def _validate_config(self):
        super()._validate_config()
        removed = {"condition_channels", "condition_blocks", "flow_layers",
                   "flow_hidden_channels", "scale_clamp", "sample_temperature"}
        legacy = removed.intersection(self.config)
        if legacy:
            raise ValueError("Removed simplified LLFlow configuration: " + ", ".join(sorted(legacy)))
        if self.config["input_channels"] != 3 or self.config["nf"] != 64 or self.config["L"] != 3:
            raise ValueError("Standard LLFlow requires input_channels=3, nf=64 and L=3.")
        for key, minimum in (("nb", 8), ("K", 1)):
            value = self.config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}.")
        ratio = float(self.config["train_gt_ratio"])
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("train_gt_ratio must be in [0, 1].")
        if not math.isfinite(float(self.config["quant"])) or float(self.config["quant"]) <= 0:
            raise ValueError("quant must be positive and finite.")
        if self.config["inference_padding"] not in {"reflect16", "none"}:
            raise ValueError("inference_padding must be 'reflect16' or 'none'.")
        if self.config["mode"] not in {"train", "inference"}:
            raise ValueError("mode must be 'train' or 'inference'.")

    def _init_model(self):
        self.opt = standard_options(self.config)
        self.RRDB = ConEncoder1(3, 3, 64, self.config["nb"], 32, 1, self.opt)
        self.flowUpsamplerNet = FlowUpsamplerNet(
            (160, 160, 3), 64, self.config["K"],
            flow_coupling="CondAffineSeparatedAndCond", opt=self.opt,
        )

    def rrdbPreprocessing(self, low):
        """Extract and concatenate the official multi-scale RRDB features."""
        features = self.RRDB(low, get_steps=True)
        concat = torch.cat([features[f"block_{i}"] for i in (1, 3, 5, 7)], dim=1)
        for key in ("last_lr_fea", "fea_up1", "fea_up2", "fea_up4", "fea_up0"):
            features[key] = torch.cat([
                features[key], F.interpolate(concat, features[key].shape[-2:])
            ], dim=1)
        return features

    def encode(self, target, condition, *, add_gt_noise=False):
        """Return (latent, per-sample NLL, logdet) with the official prior.

        condition is the dictionary returned by rrdbPreprocessing. The GT-prior
        choice is one Python random draw per batch, as in upstream normal_flow.
        """
        pixels = target.shape[-2] * target.shape[-1]
        logdet = torch.zeros_like(target[:, 0, 0, 0])
        value = target
        if add_gt_noise:
            quant = float(self.config["quant"])
            value = value + (torch.rand_like(value) - 0.5) / quant
            logdet = logdet - math.log(quant) * pixels
        latent, logdet = self.flowUpsamplerNet(
            rrdbResults=condition, gt=value, logdet=logdet, reverse=False,
        )
        if random.random() > float(self.config["train_gt_ratio"]):
            mean = squeeze2d(condition["color_map"], 8)
        else:
            mean = squeeze2d(target / (target.sum(dim=1, keepdim=True) + 1e-4), 8)
        objective = logdet.clone() + GaussianDiag.logp(mean, latent.new_tensor(0.0), latent)
        return latent, -objective / (math.log(2.0) * pixels), logdet

    def decode(self, condition):
        """Decode the predicted color map, not a zero/random latent tensor."""
        latent = squeeze2d(condition["color_map"], 8)
        image, _ = self.flowUpsamplerNet(
            rrdbResults=condition, z=latent, logdet=latent.new_zeros(latent.shape[0]),
            reverse=True, eps_std=0,
        )
        return image

    def forward(self, x, paired_image=None, *, add_gt_noise=False):
        """Enhance RGB, or encode paired GT for Trainer's NLL loss.

        Paired training/validation requires matching spatial dimensions divisible
        by eight. Inference reflect16 padding follows official test_unpaired.py;
        set inference_padding='none' for the unpadded core computation.
        """
        if x.ndim != 4 or x.shape[1] not in (3, 6) or not x.is_floating_point():
            raise ValueError("LLFlow expects floating BCHW RGB or prepared six-channel input.")
        if min(x.shape[-2:]) < 1:
            raise ValueError("Input spatial dimensions must be positive.")
        original_h, original_w = x.shape[-2:]
        top = left = 0
        if paired_image is not None:
            if paired_image.shape != (x.shape[0], 3, original_h, original_w):
                raise ValueError("Paired GT must be RGB and match the input batch and spatial size.")
        elif self.config["mode"] == "train":
            raise ValueError("LLFlow training requires paired_image (normal-light GT).")
        elif self.config["inference_padding"] == "reflect16":
            x, (top, _, left, _) = reflect_pad16(x)
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError("LLFlow core requires height and width divisible by 8.")
        low = prepare_llflow_input(x) if x.shape[1] == 3 else x
        condition = self.rrdbPreprocessing(low)
        if paired_image is not None:
            latent, nll, logdet = self.encode(paired_image, condition, add_gt_noise=add_gt_noise)
            # Encode first so ActNorm initializes from GT, never generated images.
            # No reverse loss is added; this only preserves the image-output contract.
            with torch.no_grad():
                prediction = self.decode(condition)
            return self._format_output(
                prediction, aux={"nll": nll, "latent": latent, "logdet": logdet},
                meta={"mode": self.config["mode"]},
            )
        prediction = self.decode(condition)
        return prediction[:, :, top:top + original_h, left:left + original_w]

    def load_official_weights(self, path):
        """Strictly load a local official *_G.pth state dictionary; return self.

        A leading DataParallel 'module.' is removed. Use save_model afterwards
        to create a checkpoint accepted by the unchanged Predictor.
        """
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError("Expected a pure official LLFlow state dictionary.")
        state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
        self.load_state_dict(state, strict=True)
        return self
