"""Shared helpers for the openLLV example scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


EXAMPLES_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLES_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RESULTS_DIR = Path(
    os.environ.get("OPENLLV_EXAMPLES_OUTPUT", EXAMPLES_DIR / "outputs")
).expanduser()
INPUTS_DIR = Path(
    os.environ.get("OPENLLV_EXAMPLES_INPUTS", ROOT / "Inputs")
).expanduser()

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def ensure_results_dir(*parts: str) -> Path:
    """Create and return a directory below the example output root."""
    path = RESULTS_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_save_image(
    input_path: Path,
    output_path: Path,
    scale: float,
) -> Path:
    """Resize an image by ``scale``, save it, and return its output path."""
    if scale <= 0:
        raise ValueError("scale must be greater than zero.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        new_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image.resize(new_size).save(output_path)
    return output_path


def find_example_image() -> Path:
    """Find a user-provided input image or create a synthetic fallback."""
    if INPUTS_DIR.is_dir():
        for path in sorted(INPUTS_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                return path

    fallback_dir = ensure_results_dir("result")
    fallback_path = fallback_dir / "example_input.png"
    if not fallback_path.exists():
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        image[..., 0] = 30
        image[..., 1] = np.linspace(
            20,
            120,
            image.shape[1],
            dtype=np.uint8,
        )
        image[..., 2] = np.linspace(
            40,
            180,
            image.shape[0],
            dtype=np.uint8,
        )[:, None]
        Image.fromarray(image, mode="RGB").save(fallback_path)
    return fallback_path


def find_example_folder() -> Path:
    """Find an image directory or return the synthetic fallback directory."""
    if INPUTS_DIR.is_dir() and any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in INPUTS_DIR.iterdir()
    ):
        return INPUTS_DIR

    return find_example_image().parent


def create_tiny_common_dataset(
    root: Path,
    image_size: Tuple[int, int] = (32, 32),
    count: int = 2,
) -> Path:
    """Create a tiny paired dataset using the current CommonDataset layout."""
    for split in ("train", "val"):
        input_dir = root / split / "input"
        target_dir = root / split / "target"
        input_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        for index in range(count):
            low_value = 20 + index * 10
            high_value = 100 + index * 10
            Image.new(
                "RGB",
                image_size,
                (low_value, low_value, low_value),
            ).save(input_dir / f"{index:03d}.png")
            Image.new(
                "RGB",
                image_size,
                (high_value, high_value, high_value),
            ).save(target_dir / f"{index:03d}.png")

    return root
