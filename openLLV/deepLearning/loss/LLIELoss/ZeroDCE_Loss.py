"""Zero-DCE and Zero-DCE++ reference-free loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from openLLV.deepLearning.loss.BaseLoss import BaseLoss
from ._utils import get_loss_inputs


class ZeroDCE_Loss(BaseLoss):
    """Reference-free Zero-DCE loss."""

    name = "zerodce"
    aliases = []
    requires_target = False

    def __init__(self):
        """Initialize Zero-DCE loss components."""
        super(ZeroDCE_Loss, self).__init__()
        self.L_spa = L_spa()
        self.L_exp = L_exp(16, 0.6)
        self.L_color = L_color()
        self.L_TV = L_TV()

    def forward(self, img_lowlight, zerodce_result):
        """Compute Zero-DCE training loss.

        Args:
            img_lowlight: Low-light input tensor.
            zerodce_result: Model output dictionary containing ``pred`` and
                ``loss_inputs`` with enhancement curves.

        Returns:
            Scalar reference-free loss tensor.
        """
        loss_inputs = get_loss_inputs(zerodce_result) or zerodce_result
        enhanced_image = loss_inputs.get('enhanced', zerodce_result.get('pred'))
        A = loss_inputs['r']
        loss_TV = self.L_TV(A) * 200
        loss_spa = torch.mean(self.L_spa(enhanced_image, img_lowlight))
        loss_col = torch.mean(self.L_color(enhanced_image)) * 5
        loss_exp = torch.mean(self.L_exp(img_lowlight)) * 10

        loss = loss_TV + loss_spa + loss_col + loss_exp

        return loss


class ZeroDCE_extension_Loss(BaseLoss):
    """Reference-free Zero-DCE++ loss."""

    name = "zerodce_extension"
    aliases = ["zerodceplusplus","zerodce++"]
    requires_target = False

    def __init__(self):
        """Initialize Zero-DCE++ loss components."""
        super(ZeroDCE_extension_Loss, self).__init__()
        self.L_spa = L_spa()
        self.L_exp = L_exp(16, 0.6)
        self.L_color = L_color()
        self.L_TV = L_TV()

    def forward(self, img_lowlight, zerodce_result):
        """Compute Zero-DCE++ training loss.

        Args:
            img_lowlight: Low-light input tensor.
            zerodce_result: Model output dictionary containing ``pred`` and
                ``loss_inputs`` with enhancement curves.

        Returns:
            Scalar reference-free loss tensor.
        """
        loss_inputs = get_loss_inputs(zerodce_result) or zerodce_result
        enhanced_image = loss_inputs.get('enhanced', zerodce_result.get('pred'))
        A = loss_inputs['r']
        loss_TV = self.L_TV(A) * 1600
        loss_spa = torch.mean(self.L_spa(enhanced_image, img_lowlight))
        loss_col = torch.mean(self.L_color(enhanced_image)) * 5
        loss_exp = torch.mean(self.L_exp(img_lowlight)) * 10

        loss = loss_TV + loss_spa + loss_col + loss_exp

        return loss



class L_color(nn.Module):
    """Color constancy loss used by Zero-DCE."""

    def __init__(self):
        """Initialize color constancy loss."""
        super(L_color, self).__init__()

    def forward(self, x):
        """Compute color-channel consistency loss.

        Args:
            x: Enhanced image tensor with shape ``[B, C, H, W]``.

        Returns:
            Per-image color constancy loss tensor.
        """
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):
    """Spatial consistency loss used by Zero-DCE."""

    def __init__(self):
        """Initialize spatial consistency kernels."""
        super(L_spa, self).__init__()
        kernel_left = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("weight_left", kernel_left)
        self.register_buffer("weight_right", kernel_right)
        self.register_buffer("weight_up", kernel_up)
        self.register_buffer("weight_down", kernel_down)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        """Compute spatial consistency loss.

        Args:
            org: Original low-light image tensor.
            enhance: Enhanced image tensor.

        Returns:
            Spatial consistency loss map.
        """
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        one = org_pool.new_tensor(1.0)
        zero = org_pool.new_tensor(0.0)
        half = org_pool.new_tensor(0.5)
        threshold = org_pool.new_tensor(0.3)
        weight_diff = torch.max(one + 10000 * torch.min(org_pool - threshold, zero), half)
        E_1 = torch.mul(torch.sign(enhance_pool - half), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)

        return E


class L_exp(nn.Module):
    """Exposure control loss used by Zero-DCE."""

    def __init__(self, patch_size, mean_val):
        """Initialize exposure control loss.

        Args:
            patch_size: Average pooling patch size.
            mean_val: Target exposure value.
        """
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        """Compute exposure control loss.

        Args:
            x: Image tensor with shape ``[B, C, H, W]``.

        Returns:
            Scalar exposure loss tensor.
        """
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - mean.new_tensor(self.mean_val), 2))
        return d


class L_TV(nn.Module):
    """Total variation loss for enhancement curves."""

    def __init__(self, TVLoss_weight=1):
        """Initialize total variation loss.

        Args:
            TVLoss_weight: Multiplicative loss weight.
        """
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        """Compute total variation loss.

        Args:
            x: Tensor with shape ``[B, C, H, W]``.

        Returns:
            Scalar total variation loss tensor.
        """
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):
    """Spatial color deviation loss."""

    def __init__(self):
        """Initialize spatial color deviation loss."""
        super(Sa_Loss, self).__init__()

    def forward(self, x):
        """Compute spatial color deviation loss.

        Args:
            x: Image tensor with shape ``[B, C, H, W]``.

        Returns:
            Scalar spatial color deviation loss tensor.
        """
        b, c, h, w = x.shape
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)

        k = torch.mean(k)
        return k


