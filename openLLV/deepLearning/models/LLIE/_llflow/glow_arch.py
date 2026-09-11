# Adapted from wyf0912/LLFlow, commit 115da161a96de868d67494a32db848e31f85bbc1.
# Copyright (c) 2021 Yufei Wang. CC BY-NC-SA 4.0; see licenses/.
# Changes: package-local imports; standard-model subset where noted.

import torch.nn as nn


def f_conv2d_bias(in_channels, out_channels):
    def padding_same(kernel, stride):
        return [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)]

    padding = padding_same([3, 3], [1, 1])
    assert padding == [1, 1], padding
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=[3, 3], stride=1, padding=1,
                  bias=True))
