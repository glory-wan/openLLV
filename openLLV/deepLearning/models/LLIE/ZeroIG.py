"""Zero-IG model for zero-shot low-light enhancement and denoising.

    Original paper: ZERO-IG: Zero-Shot Illumination-Guided Joint Denoising and Adaptive Enhancement for Low-Light Images
    Paper link: https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZERO-IG_Zero-Shot_Illumination-Guided_Joint_Denoising_and_Adaptive_Enhancement_for_Low-Light_CVPR_2024_paper.pdf
    Official source code: https://github.com/Doyle59217/ZeroIG
    """

from ..BaseModel import LLVModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np


class ZeroIG(LLVModel):

    task = "llie"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize Zero-IG.

        Args:
            config: Optional model configuration dictionary.
            **kwargs: Configuration overrides.
        """
        super().__init__(config, **kwargs)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Zero-IG configuration.

        Returns:
            Default configuration dictionary.
        """
        default_config = super()._get_default_config()
        default_config.update({
            'enhance_layers': 3,  # Number of layers in enhancer
            'enhance_channels': 64,  # Channel number for enhancer
            'denoise1_channels': 48,  # Channel number for first denoiser
            'denoise2_channels': 48,  # Channel number for second denoiser
            'mode': 'inference',  # 'train', 'inference', or 'finetune'
            'pretrained_weights': None,  # Path to pre-trained weights for finetune mode
        })
        return default_config

    def _validate_config(self):
        """Validate Zero-IG configuration.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        super()._validate_config()

        if self.config['enhance_layers'] <= 0:
            raise ValueError("'enhance_layers' must be positive")
        if self.config['mode'] not in ['train', 'inference', 'finetune']:
            raise ValueError("'mode' must be 'train', 'inference', or 'finetune'")

    def _init_model(self):
        """Initialize Zero-IG enhancement and denoising subnetworks."""
        self.enhance = Enhancer(layers=self.config['enhance_layers'],
                                channels=self.config['enhance_channels'])
        self.denoise_1 = Denoise_1(chan_embed=self.config['denoise1_channels'])
        self.denoise_2 = Denoise_2(chan_embed=self.config['denoise2_channels'])
        self.TextureDifference = TextureDifference()
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        if self.config['mode'] == 'train':
            self._init_weights()

        elif self.config['mode'] == 'finetune' and self.config.get('pretrained_weights'):
            self._load_finetune_weights()

    def _init_weights(self):
        """Initialize model weights following the original implementation."""
        self.enhance.in_conv.apply(self.enhance_weights_init)
        self.enhance.conv.apply(self.enhance_weights_init)
        self.enhance.out_conv.apply(self.enhance_weights_init)

        self.denoise_1.apply(self.denoise_weights_init)
        self.denoise_2.apply(self.denoise_weights_init)

    def enhance_weights_init(self, m):
        """Initialize enhancer module weights.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def denoise_weights_init(self, m):
        """Initialize denoiser module weights.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def _load_finetune_weights(self):
        """Load pre-trained weights for finetune mode."""
        try:
            base_weights = torch.load(
                self.config['pretrained_weights'],
                map_location="cpu",
            )
            model_dict = self.state_dict()

            pretrained_dict = {k: v for k, v in base_weights.items() if k in model_dict}

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

            print(f"✅ Pre-trained weights loaded from: {self.config['pretrained_weights']}")
        except Exception as e:
            print(f"⚠️  Failed to load pre-trained weights: {e}")

    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a Zero-IG forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.
            **kwargs: Additional forward-pass parameters.

        Returns:
            Training mode: standardized output dict containing all loss
            intermediates. Inference mode: final enhanced and denoised image.
            Finetune mode: standardized output dict containing ``H2`` and
            ``H3``.
        """
        mode = self.config['mode']

        if mode == 'finetune':
            H2, H3 = self._finetune_forward(x)
            return self._format_output(
                pred=H3,
                aux={'H2': H2, 'H3': H3},
                meta={'mode': mode},
            )
        elif mode == 'train':
            outputs = self._train_forward(x)
            return self._format_output(
                pred=outputs[13],
            aux={
                    'outputs': outputs,
                    'H3': outputs[13],
                },
                meta={'mode': mode},
            )
        else:
            H3 = self._inference_forward(x)
            return H3

    def _finetune_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Zero-IG finetune forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.

        Returns:
            A tuple ``(H2, H3)`` containing enhanced and denoised images.
        """
        eps = 1e-4
        input_tensor = x + eps

        L2 = input_tensor - self.denoise_1(input_tensor)
        L2 = torch.clamp(L2, eps, 1)
        s2 = self.enhance(L2)
        H2 = input_tensor / s2
        H2 = torch.clamp(H2, eps, 1)
        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]

        return H2, H3

    def _train_forward(self, x: torch.Tensor) -> Tuple:
        """Run Zero-IG training forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.

        Returns:
            Tuple of intermediate outputs used by ``ZeroIG_Loss``.
        """
        eps = 1e-4
        input_tensor = x + eps

        L11, L12 = pair_downsampler(input_tensor)
        L_pred1 = L11 - self.denoise_1(L11)
        L_pred2 = L12 - self.denoise_1(L12)
        L2 = input_tensor - self.denoise_1(input_tensor)
        L2 = torch.clamp(L2, eps, 1)

        s2 = self.enhance(L2.detach())
        s21, s22 = pair_downsampler(s2)
        H2 = input_tensor / s2
        H2 = torch.clamp(H2, eps, 1)

        H11 = L11 / s21
        H11 = torch.clamp(H11, eps, 1)

        H12 = L12 / s22
        H12 = torch.clamp(H12, eps, 1)

        H3_pred = torch.cat([H11, s21], 1).detach() - self.denoise_2(torch.cat([H11, s21], 1))
        H3_pred = torch.clamp(H3_pred, eps, 1)
        H13 = H3_pred[:, :3, :, :]
        s13 = H3_pred[:, 3:, :, :]

        H4_pred = torch.cat([H12, s22], 1).detach() - self.denoise_2(torch.cat([H12, s22], 1))
        H4_pred = torch.clamp(H4_pred, eps, 1)
        H14 = H4_pred[:, :3, :, :]
        s14 = H4_pred[:, 3:, :, :]

        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]
        s3 = H5_pred[:, 3:, :, :]

        L_pred1_L_pred2_diff = self.TextureDifference(L_pred1, L_pred2)
        H3_denoised1, H3_denoised2 = pair_downsampler(H3)
        H3_denoised1_H3_denoised2_diff = self.TextureDifference(H3_denoised1, H3_denoised2)

        H1 = L2 / s2
        H1 = torch.clamp(H1, 0, 1)
        H2_blur = blur(H1)
        H3_blur = blur(H3)

        return (L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13,
                H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff,
                H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur)

    def _inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Zero-IG inference forward pass.

        Args:
            x: Low-light input tensor with shape ``[B, C, H, W]``.

        Returns:
            Final enhanced and denoised image tensor.
        """
        eps = 1e-4
        input_tensor = x + eps

        L2 = input_tensor - self.denoise_1(input_tensor)
        L2 = torch.clamp(L2, eps, 1)
        s2 = self.enhance(L2.detach())
        H2 = input_tensor / s2
        H2 = torch.clamp(H2, eps, 1)
        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]

        return H3

    def set_mode(self, mode: str):
        """Set Zero-IG mode.

        Args:
            mode: One of ``"train"``, ``"inference"``, or ``"finetune"``.

        Returns:
            The model itself.

        Raises:
            ValueError: If ``mode`` is unsupported.
        """
        if mode not in ['train', 'inference', 'finetune']:
            raise ValueError("mode must be 'train', 'inference', or 'finetune'")
        self.config['mode'] = mode
        if mode == 'train':
            self.train()
        else:
            self.eval()
        return self


class Denoise_1(nn.Module):
    """First denoising subnetwork used by Zero-IG."""

    def __init__(self, chan_embed=48):
        """Initialize first denoising subnetwork.

        Args:
            chan_embed: Number of hidden channels.
        """
        super(Denoise_1, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(3, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 3, 1)

    def forward(self, x):
        """Run first denoising subnetwork.

        Args:
            x: Input tensor with shape ``[B, 3, H, W]``.

        Returns:
            Denoising residual tensor.
        """
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Denoise_2(nn.Module):
    """Second denoising subnetwork used by Zero-IG."""

    def __init__(self, chan_embed=96):
        """Initialize second denoising subnetwork.

        Args:
            chan_embed: Number of hidden channels.
        """
        super(Denoise_2, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(6, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 6, 1)

    def forward(self, x):
        """Run second denoising subnetwork.

        Args:
            x: Input tensor with shape ``[B, 6, H, W]``.

        Returns:
            Denoising residual tensor.
        """
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Enhancer(nn.Module):
    """Zero-IG illumination enhancement subnetwork."""

    def __init__(self, layers, channels):
        """Initialize enhancer.

        Args:
            layers: Number of residual convolution blocks.
            channels: Number of feature channels.
        """
        super(Enhancer, self).__init__()
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

    def forward(self, input):
        """Estimate illumination map.

        Args:
            input: Input image tensor.

        Returns:
            Illumination tensor.
        """
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        fea = torch.clamp(fea, 0.0001, 1)
        return fea


class TextureDifference(nn.Module):
    """Local texture-difference mask used by Zero-IG."""

    def __init__(self, patch_size=5, constant_C=1e-5, threshold=0.975):
        """Initialize texture-difference module.

        Args:
            patch_size: Local patch size.
            constant_C: Stability constant.
            threshold: Threshold used to binarize the texture similarity map.
        """
        super(TextureDifference, self).__init__()
        self.patch_size = patch_size
        self.constant_C = constant_C
        self.threshold = threshold

    def forward(self, image1, image2):
        """Compute binary texture similarity mask.

        Args:
            image1: First image tensor.
            image2: Second image tensor.

        Returns:
            Binary tensor indicating locally similar texture regions.
        """
        image1 = self.rgb_to_gray(image1)
        image2 = self.rgb_to_gray(image2)

        stddev1 = self.local_stddev(image1)
        stddev2 = self.local_stddev(image2)
        numerator = 2 * stddev1 * stddev2
        denominator = stddev1 ** 2 + stddev2 ** 2 + self.constant_C
        diff = numerator / denominator

        binary_diff = torch.where(diff > self.threshold,
                                  torch.tensor(1.0, device=diff.device),
                                  torch.tensor(0.0, device=diff.device))
        return binary_diff

    def local_stddev(self, image):
        """Compute local standard deviation.

        Args:
            image: Grayscale image tensor.

        Returns:
            Local standard-deviation tensor.
        """
        padding = self.patch_size // 2
        image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        mean = patches.mean(dim=(4, 5), keepdim=True)
        squared_diff = (patches - mean) ** 2
        local_variance = squared_diff.mean(dim=(4, 5))
        local_stddev = torch.sqrt(local_variance + 1e-9)
        return local_stddev

    def rgb_to_gray(self, image):
        """Convert RGB tensor to grayscale.

        Args:
            image: RGB tensor with shape ``[B, 3, H, W]``.

        Returns:
            Grayscale tensor with shape ``[B, 1, H, W]``.
        """
        gray_image = 0.144 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.299 * image[:, 2, :, :]
        return gray_image.unsqueeze(1)


def pair_downsampler(img):
    """Downsample image with two complementary pair filters.

    Args:
        img: Input tensor with shape ``[B, C, H, W]``.

    Returns:
        A tuple containing two downsampled tensors.
    """
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = torch.nn.functional.conv2d(img, filter1, stride=2, groups=c)
    output2 = torch.nn.functional.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2


def blur(x):
    """Apply Gaussian blur.

    Args:
        x: Input tensor with shape ``[B, C, H, W]``.

    Returns:
        Blurred tensor.
    """
    kernel_size = 21
    padding = kernel_size // 2

    interval = (2 * 1 + 1.) / kernel_size
    x_coord = torch.linspace(-1 - interval / 2., 1 + interval / 2., kernel_size + 1, device=x.device)

    def gauss_cdf(x):
        return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2., device=x.device))))

    kern1d = torch.diff(gauss_cdf(x_coord))
    kernel_raw = torch.sqrt(torch.outer(kern1d, kern1d))
    kernel = kernel_raw / torch.sum(kernel_raw)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(x.size(1), 1, 1, 1)

    x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    return F.conv2d(x_padded, kernel, padding=0, groups=x.size(1))
