"""Paired LOL data preparation for the official standard LLFlow model.

Adapted from wyf0912/LLFlow (115da161), CC BY-NC-SA 4.0.
See deepLearning/models/LLIE/LLFlow.LICENSE.
"""

import cv2
import numpy as np
import torch

from .CommonDataset import CommonDataset


def equalize_rgb(image):
    """Equalize each uint8 RGB channel independently, as in official LLFlow."""
    return cv2.merge([cv2.equalizeHist(channel) for channel in cv2.split(image)])


def prepare_llflow_input(image, equalized=None):
    """Convert BCHW RGB [0,1] to [log(low+1e-3), equalized RGB].

    Optional equalized carries the full-image histogram result through paired
    cropping. Only the histogram branch quantizes to uint8 and runs on CPU.
    """
    if equalized is None:
        arrays = image.detach().float().clamp(0, 1).mul(255).round().byte().cpu()
        arrays = arrays.permute(0, 2, 3, 1).numpy()
        equalized = torch.from_numpy(np.stack([equalize_rgb(a) for a in arrays]))
        equalized = equalized.permute(0, 3, 1, 2).to(device=image.device, dtype=image.dtype) / 255
    return torch.cat([torch.log(torch.clamp(image + 1e-3, min=1e-3)), equalized], dim=1)


def reflect_pad16(image):
    """Match test_unpaired.py's BORDER_REFLECT, including a full extra 16.

    PyTorch's reflection pad uses different edge semantics, so use symmetric
    index mapping. This also supports images smaller than the padding width.
    """
    height, width = image.shape[-2:]
    pad_h, pad_w = 16 - height % 16, 16 - width % 16
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    def indices(length, before, after):
        idx = torch.arange(-before, length + after, device=image.device) % (2 * length)
        return torch.where(idx < length, idx, 2 * length - 1 - idx)

    padded = image.index_select(-2, indices(height, top, bottom))
    padded = padded.index_select(-1, indices(width, left, right))
    return padded, (top, bottom, left, right)


class LLFlowDataset(CommonDataset):
    """Equalize the full image before synchronized crop and horizontal flip.

    Supports official LOL our485/eval15 low/high folders and CommonDataset's
    input/target layouts. Outputs six prepared low-light channels and RGB GT.
    """

    aliases = []

    def __init__(self, root_dir, split="train", input_dir=None, target_dir=None,
                 return_filename=True, crop_size=None, use_flip=False,
                 transform_input=None, transform_target=None,
                 common_transform=None, resize=None):
        if any(value is not None for value in
               (transform_input, transform_target, common_transform, resize)):
            raise ValueError("LLFlowDataset uses fixed official preprocessing; custom transforms/resize are unsupported.")
        super().__init__(root_dir=root_dir, split=split, input_dir=input_dir,
                         target_dir=target_dir, return_filename=return_filename,
                         crop_size=crop_size, use_flip=use_flip)
        if self.target_dir is None:
            raise ValueError("LLFlowDataset requires paired normal-light targets.")

    def _resolve_pair_dirs(self, input_dir, target_dir):
        if input_dir is None:
            folder = "our485" if self.split.lower() in ("train", "training") else "eval15"
            low, high = self.root_dir / folder / "low", self.root_dir / folder / "high"
            if low.is_dir() and high.is_dir():
                return low, high
        return super()._resolve_pair_dirs(input_dir, target_dir)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]
        low = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        high = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if low is None or high is None:
            raise ValueError(f"Cannot read LLFlow pair: {input_path}, {target_path}")
        low, high = cv2.cvtColor(low, cv2.COLOR_BGR2RGB), cv2.cvtColor(high, cv2.COLOR_BGR2RGB)
        if low.shape != high.shape:
            raise ValueError("LLFlow low/GT images must have matching RGB shapes.")
        equalized = equalize_rgb(low)
        images = [low, high, equalized]
        if self.crop_size is not None:
            size = self.crop_size
            height, width = low.shape[:2]
            top = np.random.randint(0, height - size + 1) if height > size else 0
            left = np.random.randint(0, width - size + 1) if width > size else 0
            images = [im[top:top + size, left:left + size] for im in images]
        if self.use_flip and not np.random.choice([True, False]):
            images = [np.flip(im, 1).copy() for im in images]
        low, high, equalized = [
            torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float() / 255
            for im in images
        ]
        if low.shape[-2] % 8 or low.shape[-1] % 8:
            raise ValueError("LLFlow training/validation dimensions must be divisible by 8.")
        prepared = prepare_llflow_input(low.unsqueeze(0), equalized.unsqueeze(0))[0]
        result = (prepared, high)
        return (*result, input_path.name) if self.return_filename else result
