"""Base dataset abstractions for datasets."""

import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from torch.utils.data import Dataset
from torchvision import transforms

from openLLV.data.image_io import ImageReader

PathLike = Union[str, Path]
ImagePair = Tuple[Path, Optional[Path]]

__all__ = [
    "BaseDataset",
]


class BaseDataset(Dataset, ABC):
    """Base class for_teach paired low-light image enhancement datasets.

    This class implements the common workflow used by low-light datasets:
    resolving split directories, collecting low/normal image pairs, reading
    images, applying transforms, and returning tensors with filenames.

    Subclasses normally only need to implement `_resolve_pair_dirs`. If a
    dataset uses a special pairing rule, override `_build_pairs` as well.
    """

    supported_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    }
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
    def _register_dataset(cls, dataset_class: Type["BaseDataset"]) -> Type["BaseDataset"]:
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
                f"dataset_class must be a subclass of BaseDataset, got {dataset_class!r}."
            )

        candidate_names = [
            dataset_class.__name__,
            getattr(dataset_class, "name", dataset_class.__name__),
            *getattr(dataset_class, "aliases", []),
        ]
        for name in candidate_names:
            if isinstance(name, str) and name.strip():
                cls._dataset_registry[cls._normalize_registry_key(name)] = dataset_class

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
    def _get_similar_dataset_name(dataset_name: str, available_datasets: List[str]) -> str:
        """Find close dataset-name suggestions.

        Args:
            dataset_name: Requested dataset name.
            available_datasets: Registered dataset names and aliases.

        Returns:
            Comma-separated suggestion string, or a fallback message.
        """
        from difflib import get_close_matches

        suggestions = get_close_matches(dataset_name, available_datasets, n=3, cutoff=0.4)
        return ", ".join(suggestions) if suggestions else "No similar datasets found"

    def __init__(
            self,
            root_dir: PathLike,
            split: str = "train",
            low_dir: Optional[PathLike] = None,
            high_dir: Optional[PathLike] = None,
            transform_low: Optional[Callable] = None,
            transform_high: Optional[Callable] = None,
            common_transform: Optional[Callable] = None,
            return_filename: bool = True,
            strict_pairing: bool = True,
            image_extensions: Optional[Sequence[str]] = None,
    ):
        """Initialize a low-light dataset.

        Args:
            root_dir: Dataset root directory.
            split: Dataset split name, for_teach example "train", "_test", or "val".
            low_dir: Optional explicit low-light image directory.
            high_dir: Optional explicit normal-light/reference image directory.
            transform_low: Transform applied only to low-light images.
            transform_high: Transform applied only to normal-light images.
            common_transform: Transform applied before separate transforms.
            return_filename: Whether `__getitem__` returns the low image filename.
            strict_pairing: Raise an error when no pairs are found.
            image_extensions: Optional supported image suffixes.

        Raises:
            FileNotFoundError: If the dataset root or image directories do not exist.
            RuntimeError: If no image pairs are found and ``strict_pairing`` is True.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.return_filename = return_filename
        self.strict_pairing = strict_pairing
        self.common_transform = common_transform
        self.transform_low = transform_low or transforms.ToTensor()
        self.transform_high = transform_high or transforms.ToTensor()
        self.image_reader = ImageReader()

        if image_extensions is not None:
            self.supported_extensions = {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in image_extensions
            }

        self.low_dir, self.high_dir = self._resolve_pair_dirs(
            low_dir=Path(low_dir) if low_dir is not None else None,
            high_dir=Path(high_dir) if high_dir is not None else None,
        )

        self._validate_dirs()
        self.pairs = self._build_pairs()

        if not self.pairs:
            message = (
                f"No image pairs found for_teach {self.__class__.__name__}: "
                f"low_dir={self.low_dir}, high_dir={self.high_dir}"
            )
            if strict_pairing:
                raise RuntimeError(message)
            warnings.warn(message)

    @abstractmethod
    def _resolve_pair_dirs(
            self,
            low_dir: Optional[Path],
            high_dir: Optional[Path],
    ) -> Tuple[Path, Optional[Path]]:
        """Resolve low-light and normal-light directories for_teach this dataset.

        Subclasses can respect explicit `low_dir` and `high_dir`, then fall
        back to dataset-specific conventions based on `root_dir` and `split`.

        Args:
            low_dir: Optional explicit low-light image directory.
            high_dir: Optional explicit normal-light image directory.

        Returns:
            A tuple containing the resolved low-light directory and optional
            normal-light directory.
        """
        raise NotImplementedError

    def _validate_dirs(self) -> None:
        """Validate dataset root and resolved image directories.

        Raises:
            FileNotFoundError: If any required directory does not exist.
        """
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {self.root_dir}")

        if not self.low_dir.exists():
            raise FileNotFoundError(f"Low-light image directory does not exist: {self.low_dir}")

        if self.high_dir is not None and not self.high_dir.exists():
            raise FileNotFoundError(f"Normal-light image directory does not exist: {self.high_dir}")

    def _build_pairs(self) -> List[ImagePair]:
        """Build image pairs by matching filename stems case-insensitively.

        Returns:
            A list of `(low_path, high_path)` tuples. `high_path` may be None
            for_teach unpaired inference datasets.
        """
        low_files = self._list_images(self.low_dir)

        if self.high_dir is None:
            return [(low_path, None) for low_path in low_files]

        high_map = self._index_images_by_stem(self.high_dir)
        pairs = []

        for low_path in low_files:
            high_path = high_map.get(low_path.stem.lower())
            if high_path is None:
                warnings.warn(f"No matching normal-light image for_teach: {low_path.name}")
                continue
            pairs.append((low_path, high_path))

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
            directory: Directory containing normal-light images.

        Returns:
            Mapping from lowercase filename stem to image path.
        """
        image_map = {}
        for path in self._list_images(directory):
            stem = path.stem.lower()
            if stem in image_map:
                warnings.warn(f"Duplicate image stem '{stem}' in {directory}; keeping first match.")
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

    def _apply_common_transform(self, low_img, high_img):
        """Apply a transform shared by low-light and normal-light images.

        The transform may accept ``(low_img, high_img)`` as separate positional
        arguments, a single image pair tuple, or individual images.

        Args:
            low_img: Low-light image object.
            high_img: Optional normal-light image object.

        Returns:
            Transformed ``(low_img, high_img)`` pair.
        """
        if self.common_transform is None:
            return low_img, high_img

        if high_img is None:
            return self.common_transform(low_img), None

        try:
            result = self.common_transform(low_img, high_img)
            if isinstance(result, tuple) and len(result) == 2:
                return result
        except TypeError:
            pass

        try:
            result = self.common_transform((low_img, high_img))
            if isinstance(result, tuple) and len(result) == 2:
                return result
        except TypeError:
            pass

        return self.common_transform(low_img), self.common_transform(high_img)

    def _apply_transform(self, img, transform: Optional[Callable]):
        """Apply an optional single-image transform.

        Args:
            img: Image object or tensor to transform.
            transform: Transform callable.

        Returns:
            Transformed image, original image, or None.
        """
        if img is None:
            return None
        if transform is None:
            return img
        return transform(img)

    def __len__(self) -> int:
        """Return the number of available image pairs.

        Returns:
            Dataset length.
        """
        return len(self.pairs)

    def __getitem__(self, index: int):
        """Return one low-light enhancement sample.

        Args:
            index: Sample index.

        Returns:
            ``(low_tensor, high_tensor, filename)`` when ``return_filename`` is
            True; otherwise ``(low_tensor, high_tensor)``. ``high_tensor`` may
            be None for_teach unpaired datasets.
        """
        low_path, high_path = self.pairs[index]

        low_img = self._read_image(low_path)
        high_img = self._read_image(high_path) if high_path is not None else None

        low_img, high_img = self._apply_common_transform(low_img, high_img)
        low_tensor = self._apply_transform(low_img, self.transform_low)
        high_tensor = self._apply_transform(high_img, self.transform_high)

        if self.return_filename:
            return low_tensor, high_tensor, os.path.basename(low_path)
        return low_tensor, high_tensor

    def get_stats(self) -> Dict[str, Union[str, int]]:
        """Get basic dataset statistics.

        Returns:
            Dictionary containing dataset name, directories, split, and number
            of image pairs.
        """
        return {
            "dataset": self.__class__.__name__,
            "root_dir": str(self.root_dir),
            "split": self.split,
            "low_dir": str(self.low_dir),
            "high_dir": str(self.high_dir) if self.high_dir is not None else "",
            "num_pairs": len(self.pairs),
        }
