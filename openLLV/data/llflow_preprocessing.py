"""LLFlow's per-channel OpenCV equalization and log-low preprocessing.

Adapted from wyf0912/LLFlow (115da161), CC BY-NC-SA 4.0.
See deepLearning/models/LLIE/_llflow/licenses/.
"""

import cv2
import numpy as np
import torch


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
