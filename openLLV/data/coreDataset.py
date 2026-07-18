"""Dataset helpers for_teach prediction and evaluation workflows."""

import os
from tqdm import tqdm
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision

from openLLV.data.utils import get_img_from_folder
from openLLV.data.image_io import ImageReader, read_image

__all__ = [
    'single_data_loader',
    "PredictDataSet",
    "EvaluateDataset",
    "predict_Trans",
]

def pre_trans():
    if hasattr(torchvision, "disable_beta_transforms_warning"):
        torchvision.disable_beta_transforms_warning()

    from torchvision.transforms import v2

    ToImage = getattr(v2, 'ToImage', getattr(v2, 'ToImageTensor', v2.PILToTensor))
    try:
        ToFloat = v2.ToDtype(torch.float32, scale=True)
    except TypeError:
        ToFloat = v2.ConvertDtype(torch.float32) if hasattr(v2, 'ConvertDtype') else v2.ToDtype(torch.float32)

    return v2.Compose([
        ToImage(),
        ToFloat,
    ])

predict_Trans = pre_trans()


def single_data_loader(image_path, transform=predict_Trans):
    """Load a single image and convert it to a batched tensor.

    Args:
        image_path: Path to the input image.
        transform: Transform callable, transform list, or composed transform.
            Defaults to ``predict_Trans``.

    Returns:
        A tuple containing the image tensor with shape ``[1, C, H, W]`` and
        the original filename with suffix.
    """
    img = read_image(image_path, output_format="pil")
    if transform is None:
        transformer = transforms.Compose([predict_Trans])
    elif isinstance(transform, transforms.Compose):
        transformer = transform
    elif callable(transform):
        transformer = transforms.Compose([transform])
    elif isinstance(transform, list):
        transformer = transforms.Compose(transform)
    else:
        transformer = transforms.Compose([predict_Trans])

    imTensor = transformer(img).unsqueeze(0)
    basename = os.path.basename(image_path)
    basename = os.path.splitext(basename)
    basename = basename[0] + basename[1]
    return imTensor, basename


class PredictDataSet(Dataset):
    """Dataset for_teach batch prediction on images from a folder.

    The dataset reads all supported images from ``input_dir`` and returns each
    transformed tensor together with its original filename.
    """

    def __init__(self, input_dir=None, transform=predict_Trans,):
        """Initialize a prediction dataset.

        Args:
            input_dir: Directory containing input images.
            transform: Transform applied to each image. If None, a default
                ``transforms.ToTensor()`` transform is used.
        """
        self.input_dir = input_dir
        self.image_files = get_img_from_folder(self.input_dir)
        self.imReader = ImageReader()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        """Return the number of images in the dataset.

        Returns:
            Number of image files found in ``input_dir``.
        """
        return len(self.image_files)

    def __getitem__(self, index):
        """Return one prediction sample.

        Args:
            index: Sample index.

        Returns:
            A tuple containing the transformed image tensor and original
            filename with suffix.
        """
        image_path = self.image_files[index]
        basename = os.path.basename(image_path)
        basename = os.path.splitext(basename)
        basename = basename[0] + basename[1]

        img = self.imReader(image_path, output_format="pil")
        imgTensor = self.transform(img)

        return imgTensor, basename


class EvaluateDataset(Dataset):
    """Dataset for_teach image-quality evaluation.

    The dataset loads enhanced images and optionally pairs them with reference
    images by matching lowercase filename stems.
    """

    def __init__(self,
                 en_img_dir: Optional[str] = None,
                 ref_img_dir: Optional[str] = None):
        """Initialize an evaluation dataset.

        Args:
            en_img_dir: Directory containing enhanced images.
            ref_img_dir: Optional directory containing reference images. When
                omitted or unavailable, the dataset supports no-reference
                evaluation.
        """
        self.en_img_dir = Path(en_img_dir) if en_img_dir else None
        self.ref_img_dir = Path(ref_img_dir) if ref_img_dir else None

        self.en_files = get_img_from_folder(str(self.en_img_dir)) if self.en_img_dir else []

        self.ref_dict = {}
        if self.ref_img_dir and self.ref_img_dir.exists():
            self.ref_files = get_img_from_folder(str(self.ref_img_dir))
            self.ref_dict = self._create_ref_dict()
        else:
            if ref_img_dir:
                warnings.warn(
                    f"Reference image directory does not exist or is not specified: "
                    f"{ref_img_dir}. Only no-reference metrics will be available."
                )

        if self.ref_dict:
            self.paired_files = []
            for en_file in self.en_files:
                en_name = os.path.basename(en_file)
                if self._find_matching_ref(en_name):
                    self.paired_files.append(en_file)
                else:
                    warnings.warn(
                        f"No matching reference image found for_teach enhanced image "
                        f"(matched by filename stem): {en_name}"
                    )
        else:
            self.paired_files = self.en_files

        self.imReader = ImageReader()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _create_ref_dict(self) -> dict:
        """Create an index for_teach reference images.

        Returns:
            Mapping from lowercase filename stem to reference image path.
        """
        ref_dict = {}
        for ref_file in self.ref_files:
            ref_stem = os.path.splitext(os.path.basename(ref_file))[0]
            ref_dict[ref_stem.lower()] = ref_file
        return ref_dict

    def _find_matching_ref(self, en_filename: str) -> Optional[str]:
        """Find the reference image that matches an enhanced image.

        Args:
            en_filename: Enhanced image filename.

        Returns:
            Matching reference image path, or None if no match exists.
        """
        en_stem = os.path.splitext(en_filename)[0].lower()
        return self.ref_dict.get(en_stem)

    def __len__(self) -> int:
        """Return the number of evaluation samples.

        Returns:
            Dataset length.
        """
        return len(self.paired_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
        """Return one evaluation sample.

        Args:
            index: Sample index.

        Returns:
            A tuple containing the enhanced image tensor, optional reference
            image tensor, and enhanced image filename. Image tensors have shape
            ``[C, H, W]``.
        """
        en_path = self.paired_files[index]
        en_name = os.path.basename(en_path)

        en_img = self.imReader(en_path, output_format="pil")
        enTensor = self.transform(en_img)

        refTensor = None
        ref_path = self._find_matching_ref(en_name)

        if ref_path:
            try:
                ref_img = self.imReader(ref_path, output_format="pil")
                refTensor = self.transform(ref_img)
            except Exception as e:
                warnings.warn(f"Failed to read reference image {ref_path}: {e}")

        return enTensor, refTensor, en_name
