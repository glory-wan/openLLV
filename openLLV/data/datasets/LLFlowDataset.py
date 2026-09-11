"""Paired LOL data preparation for the official standard LLFlow model.

Adapted from wyf0912/LLFlow (115da161), CC BY-NC-SA 4.0.
See deepLearning/models/LLIE/_llflow/licenses/.
"""

import cv2
import numpy as np
import torch

from .CommonDataset import CommonDataset
from ..llflow_preprocessing import equalize_rgb, prepare_llflow_input


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
