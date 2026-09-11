# Adapted from wyf0912/LLFlow, commit 115da161a96de868d67494a32db848e31f85bbc1.
# Copyright (c) 2021 Yufei Wang. CC BY-NC-SA 4.0; see licenses/.
# Changes: package-local imports; standard-model subset where noted.

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from . import thops


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
        pixels = thops.pixels(input)
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
