"""Base dataset abstractions for datasets."""

import os
import math
import random
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from openLLV.data.image_io import ImageReader

PathLike = Union[str, Path]
ImagePair = Tuple[Path, Optional[Path]]
ResizeInput = Optional[Union[int, Tuple[int, int], List[int]]]

__all__ = [
    "BaseDataset",
]


class BaseDataset(Dataset, ABC):
    """Base class for paired image-to-image datasets.

    This class implements the common workflow used by restoration datasets:
    resolving split directories, collecting input/target pairs, reading
    images, preprocessing them, and returning tensors with filenames.

    Subclasses normally only need to implement `_resolve_pair_dirs`. If a
    dataset uses a special pairing rule, override `_build_pairs` as well.
    """

    supported_extensions = ImageReader.SUPPORTED_EXTENSIONS
    name = "BaseDataset"
    aliases: List[str] = []
    _dataset_registry: Dict[str, Type["BaseDataset"]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register dataset subclasses automatically.

        Args:
            **kwargs: Keyword arguments forwarded to ``Dataset.__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "BaseDataset":
            return
        BaseDataset._register_dataset(cls)

    @classmethod
    def _normalize_registry_key(cls, name: str) -> str:
        """Normalize a dataset registry key.

        Args:
            name: Dataset name or alias.

        Returns:
            Lowercase registry key without leading or trailing whitespace.
        """
        return name.strip().lower()

    @classmethod
    def _register_dataset(
        cls,
        dataset_class: Type["BaseDataset"],
    ) -> Type["BaseDataset"]:
        """Register a dataset class and its aliases.

        Args:
            dataset_class: Dataset class to register.

        Returns:
            The registered dataset class.

        Raises:
            TypeError: If ``dataset_class`` is not a ``BaseDataset`` subclass.
        """
        if not issubclass(dataset_class, BaseDataset):
            raise TypeError(
                "dataset_class must be a subclass of BaseDataset, got "
                f"{dataset_class!r}."
            )

        candidate_names = [
            dataset_class.__name__,
            getattr(dataset_class, "name", dataset_class.__name__),
            *getattr(dataset_class, "aliases", []),
        ]
        for name in candidate_names:
            if isinstance(name, str) and name.strip():
                registry_key = cls._normalize_registry_key(name)
                cls._dataset_registry[registry_key] = dataset_class

        return dataset_class

    @classmethod
    def register(cls, dataset_class: Type["BaseDataset"]) -> Type["BaseDataset"]:
        """Register a dataset class manually.

        Args:
            dataset_class: Dataset class to register.

        Returns:
            The registered dataset class.
        """
        return cls._register_dataset(dataset_class)

    @classmethod
    def create_dataset(cls, dataset_name: str, **kwargs: Any) -> "BaseDataset":
        """Create a registered dataset instance.

        Args:
            dataset_name: Registered dataset name or alias.
            **kwargs: Keyword arguments passed to the dataset constructor.

        Returns:
            Instantiated dataset object.

        Raises:
            ValueError: If ``dataset_name`` is not registered.
        """
        dataset_class = cls.get_dataset_class(dataset_name)
        return dataset_class(**kwargs)

    @classmethod
    def get_dataset_class(cls, dataset_name: str) -> Type["BaseDataset"]:
        """Get a registered dataset class by name.

        Args:
            dataset_name: Registered dataset name or alias.

        Returns:
            Dataset class associated with ``dataset_name``.

        Raises:
            ValueError: If ``dataset_name`` is empty or not registered.
        """
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError("dataset_name must be a non-empty string.")

        key = cls._normalize_registry_key(dataset_name)
        dataset_class = cls._dataset_registry.get(key)
        if dataset_class is None:
            available = cls.list_registered_datasets()
            suggestion = cls._get_similar_dataset_name(key, available)
            raise ValueError(
                f"Dataset '{dataset_name}' is not registered.\n"
                f"Available datasets: {available}\n"
                f"Did you mean: {suggestion}"
            )
        return dataset_class

    @classmethod
    def list_registered_datasets(cls) -> List[str]:
        """List all registered dataset names and aliases.

        Returns:
            Sorted registry keys.
        """
        return sorted(cls._dataset_registry.keys())

    @staticmethod
    def _get_similar_dataset_name(
        dataset_name: str,
        available_datasets: List[str],
    ) -> str:
        """Find close dataset-name suggestions.

        Args:
            dataset_name: Requested dataset name.
            available_datasets: Registered dataset names and aliases.

        Returns:
            Comma-separated suggestion string, or a fallback message.
        """
        from difflib import get_close_matches

        suggestions = get_close_matches(
            dataset_name,
            available_datasets,
            n=3,
            cutoff=0.4,
        )
        return ", ".join(suggestions) if suggestions else "No similar datasets found"

    def __init__(
            self,
            root_dir: PathLike,
            split: str = "train",
            input_dir: Optional[PathLike] = None,
            target_dir: Optional[PathLike] = None,
            transform_input: Optional[Callable] = None,
            transform_target: Optional[Callable] = None,
            common_transform: Optional[Callable] = None,
            resize: ResizeInput = None,
            return_filename: bool = True,
            strict_pairing: bool = True,
            image_extensions: Optional[Sequence[str]] = None,
            mean: Optional[Sequence[float]] = None,
            std: Optional[Sequence[float]] = None,
            crop_size: Optional[int] = None,
            use_flip: bool = False,
            use_rot: bool = False,
    ):
        """Initialize a paired image dataset.

        Args:
            root_dir: Dataset root directory.
            split: Dataset split name, for example "train", "_test", or "val".
            input_dir: Optional explicit input image directory.
            target_dir: Optional explicit target image directory.
            transform_input: Optional transform applied to input images after
                resizing. ``ToTensor`` is used when this is ``None``.
            transform_target: Optional transform applied to target images after
                resizing. ``ToTensor`` is used when this is ``None``.
            common_transform: Transform applied before separate transforms.
            resize: Output image size. An integer produces a square ``(size,
                size)`` image; a two-item sequence is interpreted as
                ``(height, width)``. ``None`` preserves the original size.
            return_filename: Whether ``__getitem__`` returns the input filename.
            strict_pairing: Raise an error when no pairs are found.
            image_extensions: Optional supported image suffixes.
            mean: Optional per-channel mean applied to both image tensors.
            std: Positive per-channel standard deviation; required with mean.
            crop_size: Optional positive square size for paired random cropping.
            use_flip: Apply a shared horizontal flip with probability 0.5.
            use_rot: Apply shared vertical flip and transpose, each with
                probability 0.5, after cropping and horizontal flipping.

        Raises:
            FileNotFoundError: If the dataset root or image directories do not
                exist.
            RuntimeError: If no image pairs are found and ``strict_pairing`` is True.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.return_filename = return_filename
        self.strict_pairing = strict_pairing
        self.common_transform = common_transform
        if crop_size is not None and (
            isinstance(crop_size, bool) or not isinstance(crop_size, int) or crop_size <= 0
        ):
            raise ValueError("crop_size must be a positive integer or None.")
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be provided together.")
        if mean is not None and (
            not mean or len(mean) != len(std)
            or not all(math.isfinite(v) for v in mean)
            or not all(math.isfinite(v) and v > 0 for v in std)
        ):
            raise ValueError("mean/std must have matching non-empty lengths and finite values; std must be positive.")
        self.crop_size = crop_size
        self.use_flip = use_flip
        self.use_rot = use_rot
        self.resize_size = self.normalize_resize_size(resize)
        self.transform_input = self._build_image_transform(
            self.resize_size,
            transform_input,
            mean, std,
        )
        self.transform_target = self._build_image_transform(
            self.resize_size,
            transform_target,
            mean, std,
        )
        self.image_reader = ImageReader()

        if image_extensions is not None:
            self.supported_extensions = {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in image_extensions
            }

        self.input_dir, self.target_dir = self._resolve_pair_dirs(
            input_dir=Path(input_dir) if input_dir is not None else None,
            target_dir=Path(target_dir) if target_dir is not None else None,
        )

        self._validate_dirs()
        self.pairs = self._build_pairs()

        if not self.pairs:
            message = (
                f"No image pairs found for {self.__class__.__name__}: "
                f"input_dir={self.input_dir}, target_dir={self.target_dir}"
            )
            if strict_pairing:
                raise RuntimeError(message)
            warnings.warn(message)

    @abstractmethod
    def _resolve_pair_dirs(
            self,
            input_dir: Optional[Path],
            target_dir: Optional[Path],
    ) -> Tuple[Path, Optional[Path]]:
        """Resolve input and target directories for this dataset.

        Subclasses can respect explicit ``input_dir`` and ``target_dir``, then
        fall back to dataset-specific conventions based on ``root_dir`` and
        ``split``.

        Args:
            input_dir: Optional explicit input image directory.
            target_dir: Optional explicit target image directory.

        Returns:
            Resolved input directory and optional target directory.
        """
        raise NotImplementedError

    def _validate_dirs(self) -> None:
        """Validate dataset root and resolved image directories.

        Raises:
            FileNotFoundError: If any required directory does not exist.
        """
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root directory does not exist: {self.root_dir}"
            )

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input image directory does not exist: {self.input_dir}"
            )

        if self.target_dir is not None and not self.target_dir.exists():
            raise FileNotFoundError(
                f"Target image directory does not exist: {self.target_dir}"
            )

    def _build_pairs(self) -> List[ImagePair]:
        """Build image pairs by matching filename stems case-insensitively.

        Returns:
            A list of ``(input_path, target_path)`` tuples. ``target_path`` may
            be ``None`` for unpaired datasets.
        """
        input_files = self._list_images(self.input_dir)

        if self.target_dir is None:
            return [(input_path, None) for input_path in input_files]

        target_map = self._index_images_by_stem(self.target_dir)
        pairs = []

        for input_path in input_files:
            target_path = target_map.get(input_path.stem.lower())
            if target_path is None:
                warnings.warn(
                    f"No matching target image for: {input_path.name}"
                )
                continue
            pairs.append((input_path, target_path))

        return pairs

    def _list_images(self, directory: Path) -> List[Path]:
        """List supported image files in a directory.

        Args:
            directory: Directory to scan.

        Returns:
            Sorted image paths with supported suffixes.
        """
        files = [
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in self.supported_extensions
        ]
        return sorted(files, key=lambda path: path.name.lower())

    def _index_images_by_stem(self, directory: Path) -> Dict[str, Path]:
        """Index image paths by lowercase filename stem.

        Args:
            directory: Directory containing target images.

        Returns:
            Mapping from lowercase filename stem to image path.
        """
        image_map = {}
        for path in self._list_images(directory):
            stem = path.stem.lower()
            if stem in image_map:
                warnings.warn(
                    f"Duplicate image stem '{stem}' in {directory}; "
                    "keeping first match."
                )
                continue
            image_map[stem] = path
        return image_map

    def _read_image(self, path: Path):
        """Read an image as a PIL image.

        Args:
            path: Image path to read.

        Returns:
            PIL image returned by ``ImageReader``.
        """
        return self.image_reader(str(path), output_format="pil")

    def _paired_spatial_transform(self, input_image, target_image):
        """Crop and augment PIL images using one set of shared random choices."""
        if self.crop_size is None and not self.use_flip and not self.use_rot:
            return input_image, target_image
        if target_image is not None and input_image.size != target_image.size:
            raise ValueError("Paired spatial transforms require matching input/target sizes.")
        images = [input_image, target_image]
        if self.crop_size is not None:
            width, height = input_image.size
            size = self.crop_size
            if min(width, height) < size:
                raise ValueError(f"Image size {input_image.size} is smaller than crop_size={size}.")
            top = random.randint(0, height - size)
            left = random.randint(0, width - size)
            box = (left, top, left + size, top + size)
            images = [im.crop(box) if im is not None else None for im in images]
        operations = (
            (self.use_flip and random.random() < 0.5, Image.Transpose.FLIP_LEFT_RIGHT),
            (self.use_rot and random.random() < 0.5, Image.Transpose.FLIP_TOP_BOTTOM),
            (self.use_rot and random.random() < 0.5, Image.Transpose.TRANSPOSE),
        )
        for enabled, operation in operations:
            if enabled:
                images = [im.transpose(operation) if im is not None else None for im in images]
        return tuple(images)

    def _apply_common_transform(self, input_image, target_image):
        """Apply a transform shared by input and target images.

        The transform may accept ``(input_image, target_image)`` as separate
        positional arguments, a single image pair tuple, or individual images.

        Args:
            input_image: Input image object.
            target_image: Optional target image object.

        Returns:
            Transformed ``(input_image, target_image)`` pair.
        """
        if self.common_transform is None:
            return input_image, target_image

        if target_image is None:
            return self.common_transform(input_image), None

        try:
            result = self.common_transform(input_image, target_image)
            if isinstance(result, tuple) and len(result) == 2:
                return result
        except TypeError:
            pass

        try:
            result = self.common_transform((input_image, target_image))
            if isinstance(result, tuple) and len(result) == 2:
                return result
        except TypeError:
            pass

        return (
            self.common_transform(input_image),
            self.common_transform(target_image),
        )

    @staticmethod
    def normalize_resize_size(resize: ResizeInput) -> Optional[Tuple[int, int]]:
        """Normalize a resize value to an explicit ``(height, width)`` pair."""
        if resize is None:
            return None

        if isinstance(resize, bool):
            raise TypeError("resize cannot be bool.")

        if isinstance(resize, int):
            if resize <= 0:
                raise ValueError("resize must be greater than 0.")
            return resize, resize

        if isinstance(resize, (tuple, list)):
            if len(resize) != 2:
                raise ValueError("resize must contain exactly (height, width).")

            height, width = resize

            if (
                isinstance(height, bool)
                or isinstance(width, bool)
                or not isinstance(height, int)
                or not isinstance(width, int)
            ):
                raise TypeError("resize height and width must be integers.")

            if height <= 0 or width <= 0:
                raise ValueError(
                    "resize height and width must be greater than 0."
                )

            return height, width

        raise TypeError(
            "resize must be None, a positive integer, or a (height, width) pair."
        )

    @staticmethod
    def _build_image_transform(
            resize_size: Optional[Tuple[int, int]],
            image_transform: Optional[Callable],
            mean: Optional[Sequence[float]] = None,
            std: Optional[Sequence[float]] = None,
    ) -> Callable:
        """Combine optional resizing with an image transform.

        The default final transform is ``ToTensor``. A caller-provided image
        transform replaces ``ToTensor`` while retaining the configured resize.
        Optional mean/std normalization is applied last to both branches.
        """
        operations = []
        if resize_size is not None:
            operations.append(
                transforms.Resize(
                    resize_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            )
        operations.append(
            image_transform
            if image_transform is not None
            else transforms.ToTensor()
        )
        if mean is not None:
            operations.append(transforms.Normalize(mean, std))
        return transforms.Compose(operations)

    def _apply_transform(self, image, transform: Optional[Callable]):
        """Apply an optional single-image transform.

        Args:
            image: Image object or tensor to transform.
            transform: Transform callable.

        Returns:
            Transformed image, original image, or None.
        """
        if image is None:
            return None
        if transform is None:
            return image
        return transform(image)

    def __len__(self) -> int:
        """Return the number of available image pairs.

        Returns:
            Dataset length.
        """
        return len(self.pairs)

    def __getitem__(self, index: int):
        """Return one input/target sample.

        Args:
            index: Sample index.

        Returns:
            ``(input_tensor, target_tensor, filename)`` when
            ``return_filename`` is true; otherwise ``(input_tensor,
            target_tensor)``. ``target_tensor`` may be ``None`` for unpaired
            datasets.
        """
        input_path, target_path = self.pairs[index]

        input_image = self._read_image(input_path)
        target_image = (
            self._read_image(target_path) if target_path is not None else None
        )

        input_image, target_image = self._paired_spatial_transform(input_image, target_image)
        input_image, target_image = self._apply_common_transform(
            input_image,
            target_image,
        )
        input_tensor = self._apply_transform(
            input_image,
            self.transform_input,
        )
        target_tensor = self._apply_transform(
            target_image,
            self.transform_target,
        )

        if self.return_filename:
            return input_tensor, target_tensor, os.path.basename(input_path)
        return input_tensor, target_tensor

    def get_stats(self) -> Dict[str, Any]:
        """Get basic dataset statistics.

        Returns:
            Dictionary containing dataset name, directories, split, and number
            of image pairs.
        """
        return {
            "dataset": self.__class__.__name__,
            "root_dir": str(self.root_dir),
            "split": self.split,
            "input_dir": str(self.input_dir),
            "target_dir": (
                str(self.target_dir) if self.target_dir is not None else ""
            ),
            "resize": self.resize_size,
            "num_pairs": len(self.pairs),
        }
