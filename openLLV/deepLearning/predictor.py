"""Unified predictor for learned low-level vision models."""

from __future__ import annotations

import urllib.parse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

if hasattr(torchvision, "disable_beta_transforms_warning"):
    torchvision.disable_beta_transforms_warning()

from torchvision.transforms import v2
from tqdm import tqdm

from openLLV.data.coreDataset import predict_Trans
from openLLV.data.image_io import ImageReader
from openLLV.deepLearning.models import LLVModel
from openLLV.utils import log_info_env
from openLLV.utils.cancel import CancelSignal
from openLLV.utils.errors import TaskCancelled


ImageInput = Union[
    str,
    Path,
    bytes,
    bytearray,
    np.ndarray,
    Image.Image,
    torch.Tensor,
]
ModelInput = Union[str, Path, LLVModel]
ResizeInput = Optional[Union[int, Tuple[int, int], List[int]]]

__all__ = ["Predictor", "ImageInput", "ModelInput", "ResizeInput"]


def _prepare_image_tensor(
    image: Image.Image,
    transform: Any,
    resize_size: Optional[Tuple[int, int]],
) -> torch.Tensor:
    """Apply optional resizing and return one transformed ``[C,H,W]`` tensor."""
    if resize_size is not None:
        image = v2.functional.resize(image, resize_size, antialias=True)

    tensor = transform(image)
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"transform must return a torch.Tensor, got {type(tensor)!r}."
        )
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                "Single-image prediction requires transform batch size 1, "
                f"got {tensor.shape[0]}."
            )
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(
            "Expected transformed image shape [C,H,W] or [N,C,H,W], "
            f"got {tuple(tensor.shape)}."
        )
    return tensor


class _DirectoryPredictionDataset(Dataset):
    """Load and transform directory images in DataLoader workers."""

    def __init__(
        self,
        image_files: List[Path],
        transform: Any,
        resize_size: Optional[Tuple[int, int]],
        reader_kwargs: Mapping[str, Any],
    ) -> None:
        self.image_files = image_files
        self.transform = transform
        self.resize_size = resize_size
        self.reader_kwargs = dict(reader_kwargs)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Tuple[int, Path, torch.Tensor]:
        image_path = self.image_files[index]
        try:
            image = ImageReader()(
                image_path,
                output_format="pil",
                **self.reader_kwargs,
            )
            tensor = _prepare_image_tensor(
                image,
                self.transform,
                self.resize_size,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to prepare directory prediction image: {image_path}"
            ) from error
        return index, image_path, tensor


class _SizeGroupedBatchSampler(Sampler[List[int]]):
    """Yield full same-size batches and singleton remainder batches."""

    def __init__(
        self,
        image_files: List[Path],
        batch_size: int,
        resize_size: Optional[Tuple[int, int]],
    ) -> None:
        self.batch_size = batch_size
        groups: "OrderedDict[Tuple[int, int], List[int]]" = OrderedDict()

        for index, image_path in enumerate(image_files):
            if resize_size is not None:
                size_key = resize_size
            else:
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                except Exception as error:
                    raise RuntimeError(
                        f"Failed to inspect directory prediction image: {image_path}"
                    ) from error
                size_key = (height, width)
            groups.setdefault(size_key, []).append(index)

        self.batches: List[List[int]] = []
        for indices in groups.values():
            full_batch_end = len(indices) - (len(indices) % batch_size)
            for start in range(0, full_batch_end, batch_size):
                self.batches.append(indices[start : start + batch_size])
            self.batches.extend(
                [index]
                for index in indices[full_batch_end:]
            )

    def __iter__(self) -> Iterable[List[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def _return_samples_as_list(samples: List[Any]) -> List[Any]:
    """Keep variable-shaped samples unstacked until the predictor validates them."""
    return samples


class Predictor:
    """Run inference with any concrete :class:`LLVModel` descendant.

    The predictor owns runtime device placement. Model classes remain free of
    device-management state, as required by the ``LLVModel`` contract. A model
    can be supplied as a registered name, an ``LLVModel`` instance, or a
    checkpoint created by ``LLVModel.save_model``.

    Single-image inputs support every format accepted by
    :class:`openLLV.data.ImageReader`. Directory inputs are processed
    recursively while preserving their relative directory structure.
    """

    SUPPORTED_EXTENSIONS = ImageReader.SUPPORTED_EXTENSIONS

    CHECKPOINT_EXTENSIONS = {".pt", ".pth"}

    def __init__(
        self,
        model: ModelInput,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        transform: Optional[Any] = None,
        resize: ResizeInput = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        """Initialize a low-level vision predictor.

        Args:
            model: Registered model name, checkpoint path, or ``LLVModel``
                instance.
            output_dir: Default prediction directory. When omitted, outputs
                are written to ``results/<model class name>``.
            config: Optional model configuration overrides. For a checkpoint,
                these values override the saved configuration.
            device: Runtime device. CUDA is selected automatically when
                available; otherwise CPU is used.
            transform: Optional callable or list of torchvision v2 transforms.
            resize: Optional input size. A positive integer produces a square;
                a two-item tuple/list is interpreted as ``(height, width)``.
                ``None`` preserves every source image's original dimensions.
            batch_size: Number of same-size directory images per model call.
                Incomplete groups fall back to single-image calls.
            num_workers: DataLoader workers used for directory image reading and
                preprocessing.

        Raises:
            TypeError: If ``resize`` has an unsupported type or element type.
            ValueError: If ``resize`` is invalid, ``batch_size`` is not
                positive, or ``num_workers`` is negative.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer.")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self._load_model(model, config=config)
        self.model.to(self.device)
        self.model.eval_mode()

        self.model_name = self._get_model_name(self.model)
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path("results") / self.model_name
        )
        self.transform = self._build_transform(transform)
        self.resize_size = self._normalize_resize_size(resize)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_reader = ImageReader()

    def __call__(
        self,
        source: ImageInput,
        output: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        """Predict a single image or every image below a directory.

        Args:
            source: Image input or directory path.
            output: Optional output file or directory.
            **kwargs: Arguments forwarded to :meth:`predict_single` or
                :meth:`predict_batch`.

        Returns:
            ``(PIL.Image, saved_path)`` for a single image. Directory input
            returns saved paths when ``save=True`` and PIL images when
            ``save=False``.
        """
        log_info_env(self.device)
        if self._is_directory_source(source):
            return self.predict_batch(source, output_dir=output, **kwargs)
        return self.predict_single(source, save_path=output, **kwargs)

    def predict(
        self,
        source: ImageInput,
        output: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        """Alias for :meth:`__call__`."""
        return self(source, output=output, **kwargs)

    def predict_single(
        self,
        image: ImageInput,
        save_path: Optional[Union[str, Path]] = None,
        *,
        output_name: Optional[str] = None,
        output_ext: Optional[str] = None,
        save: bool = True,
        transform: Optional[Any] = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        **reader_kwargs: Any,
    ) -> Tuple[Image.Image, Optional[Path]]:
        """Run inference for one image.

        Args:
            image: Input supported by ``ImageReader``.
            save_path: Optional output file or directory.
            output_name: Optional output filename when saving to a directory.
            output_ext: Optional output suffix or format.
            save: Whether to save the prediction.
            transform: Optional per-call transform override.
            model_kwargs: Optional keyword arguments forwarded to the model's
                ``forward`` method. Tensor values, including nested values, are
                moved to the predictor device automatically.
            **reader_kwargs: Additional ``ImageReader`` arguments.

        Returns:
            Enhanced/restored PIL image and its optional saved path.
        """
        tensor = self._load_image_tensor(
            image,
            transform=transform,
            **reader_kwargs,
        )
        prediction = self._predict_tensor(tensor, model_kwargs=model_kwargs)
        output_image = self._tensor_to_pil(prediction)

        if not save:
            return output_image, None

        target_path = self._resolve_single_output_path(
            image=image,
            save_path=save_path,
            output_name=output_name,
            output_ext=output_ext,
        )
        self._save_pil_image(output_image, target_path)
        return output_image, target_path

    def predict_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        *,
        progress_bar: bool = True,
        output_name: Optional[str] = None,
        output_ext: Optional[str] = None,
        save: bool = True,
        transform: Optional[Any] = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        **reader_kwargs: Any,
    ) -> Union[List[Path], List[Image.Image]]:
        """Run inference recursively for all supported images in a directory.

        Relative subdirectories and source filenames are preserved exactly
        when ``output_ext`` is omitted, including extension letter case.
        Images are grouped by input size. Only complete groups of
        ``self.batch_size`` are stacked; incomplete groups run one image at a
        time. No padding is applied.

        Args:
            input_dir: Source image directory.
            output_dir: Optional output root directory.
            progress_bar: Whether to display a tqdm progress bar.
            output_name: Unsupported for directory prediction. Pass ``None``
                to preserve each source filename.
            output_ext: Optional suffix applied to every saved output. When
                omitted, each source suffix is preserved exactly.
            save: Whether to save predictions. If ``False``, no output files
                or directories are created.
            transform: Optional per-call transform override.
            model_kwargs: Optional keyword arguments forwarded to each model
                call.
            **reader_kwargs: Additional ``ImageReader`` arguments.

        Returns:
            Saved output paths when ``save`` is ``True``; otherwise enhanced
            PIL images. Both lists follow deterministic source-path order.

        Raises:
            NotADirectoryError: If ``input_dir`` is not a directory.
            ValueError: If ``output_name`` is supplied or ``output_ext`` is
                empty.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(
                f"input_dir must be a directory, got {input_dir}."
            )

        if output_name is not None:
            raise ValueError(
                "output_name is only supported for single-image prediction."
            )

        cancel = reader_kwargs.pop("cancel", None)
        if cancel is not None and not isinstance(cancel, CancelSignal):
            raise TypeError(
                "cancel must be a CancelSignal or None, got "
                f"{type(cancel).__name__}."
            )

        suffix = (
            self._normalize_suffix(output_ext)
            if output_ext is not None
            else None
        )

        image_files = self._list_images(input_dir)
        if not image_files:
            return []

        output_root = (
            Path(output_dir) if output_dir is not None else self.output_dir
        )
        saved_paths: List[Optional[Path]] = [None] * len(image_files)
        output_images: List[Optional[Image.Image]] = [None] * len(image_files)
        transformer = (
            self._build_transform(transform)
            if transform is not None
            else self.transform
        )
        dataset = _DirectoryPredictionDataset(
            image_files=image_files,
            transform=transformer,
            resize_size=self.resize_size,
            reader_kwargs=reader_kwargs,
        )
        batch_sampler = _SizeGroupedBatchSampler(
            image_files=image_files,
            batch_size=self.batch_size,
            resize_size=self.resize_size,
        )
        data_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=_return_samples_as_list,
        )
        iterator = (
            tqdm(data_loader, desc=f"Predicting with {self.model_name}")
            if progress_bar
            else data_loader
        )

        for samples in iterator:
            if cancel is not None and cancel.is_cancelled():
                raise TaskCancelled
            sample_predictions: List[Tuple[Any, torch.Tensor]] = []
            same_shape = len({tuple(sample[2].shape) for sample in samples}) == 1
            if (
                self.batch_size > 1
                and len(samples) == self.batch_size
                and same_shape
            ):
                batch_tensor = torch.stack(
                    [sample[2] for sample in samples],
                    dim=0,
                ).to(self.device)
                prediction = self._predict_tensor(
                    batch_tensor,
                    model_kwargs=model_kwargs,
                )
                predictions = self._split_batch_prediction(
                    prediction,
                    expected_batch_size=len(samples),
                )
                sample_predictions.extend(zip(samples, predictions))
            else:
                for sample in samples:
                    prediction = self._predict_tensor(
                        sample[2].unsqueeze(0).to(self.device),
                        model_kwargs=model_kwargs,
                    )
                    sample_predictions.append((sample, prediction))

            for sample, prediction in sample_predictions:
                index, image_path, _ = sample
                output_image = self._tensor_to_pil(prediction)
                if save:
                    relative_path = image_path.relative_to(input_dir)
                    if suffix is not None:
                        relative_path = relative_path.with_suffix(suffix)
                    target_path = output_root / relative_path
                    self._save_pil_image(output_image, target_path)
                    saved_paths[index] = target_path
                else:
                    output_images[index] = output_image

            if progress_bar:
                iterator.set_postfix({"current": samples[-1][1].name})

        results = saved_paths if save else output_images
        if any(result is None for result in results):
            raise RuntimeError("Directory prediction did not produce every output.")
        return [result for result in results if result is not None]

    def get_params(self) -> Dict[str, Any]:
        """Return predictor runtime parameters and model configuration."""
        return {
            "model": self.model_name,
            "task": getattr(self.model, "task", None),
            "device": str(self.device),
            "output_dir": str(self.output_dir),
            "resize": self.resize_size,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "config": dict(getattr(self.model, "config", {})),
        }

    @staticmethod
    def list_available_models() -> List[str]:
        """List every registered concrete ``LLVModel`` name and alias."""
        return LLVModel.list_registered_models()

    @staticmethod
    def _get_model_name(model: LLVModel) -> str:
        """Return the concrete model class name."""
        return model.__class__.__name__

    def _load_model(
        self,
        model: ModelInput,
        config: Optional[Dict[str, Any]] = None,
    ) -> LLVModel:
        """Load a model instance, registered model name, or checkpoint."""
        if config is not None and not isinstance(config, dict):
            raise TypeError(
                f"config must be a dictionary or None, got {type(config).__name__}."
            )

        if isinstance(model, LLVModel):
            if config:
                model.config.update(config)
                model._validate_config()
            return model

        if isinstance(model, Path):
            model = str(model)

        if not isinstance(model, str):
            raise TypeError(
                "model must be a registered name, checkpoint path, or "
                f"LLVModel instance, got {type(model)!r}."
            )

        model_path = Path(model)
        if model_path.suffix.lower() in self.CHECKPOINT_EXTENSIONS:
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint file does not exist: {model_path}"
                )
            return self._load_checkpoint_model(model_path, config=config)

        model_name = self._resolve_model_name(model)
        return LLVModel.create_model(model_name, config=dict(config or {}))

    def _load_checkpoint_model(
        self,
        checkpoint_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> LLVModel:
        """Reconstruct a registered model from an ``LLVModel`` checkpoint."""
        return LLVModel.load_model(
            checkpoint_path=checkpoint_path,
            config_overrides=dict(config or {}),
            strict=True,
        )

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        """Resolve a registered model name or alias case-insensitively."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        registered = LLVModel.list_registered_models()
        lookup = {name.lower(): name for name in registered}
        key = model_name.strip().lower()
        if key in lookup:
            return lookup[key]
        raise ValueError(
            f"Model '{model_name}' is not registered. "
            f"Available models: {registered}"
        )

    def _load_image_tensor(
        self,
        image: ImageInput,
        transform: Optional[Any] = None,
        **reader_kwargs: Any,
    ) -> torch.Tensor:
        """Read an image and return a batched tensor on the runtime device."""
        pil_image = self.image_reader(
            image,
            output_format="pil",
            **reader_kwargs,
        )
        transformer = (
            self._build_transform(transform)
            if transform is not None
            else self.transform
        )
        tensor = _prepare_image_tensor(
            pil_image,
            transformer,
            self.resize_size,
        )
        return tensor.unsqueeze(0).to(self.device)

    def _predict_tensor(
        self,
        tensor: torch.Tensor,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Run model inference and extract the standardized prediction."""
        if model_kwargs is None:
            resolved_kwargs: Dict[str, Any] = {}
        elif isinstance(model_kwargs, Mapping):
            resolved_kwargs = {
                key: self._move_to_device(value)
                for key, value in model_kwargs.items()
            }
        else:
            raise TypeError(
                "model_kwargs must be a mapping or None, "
                f"got {type(model_kwargs)!r}."
            )

        with torch.inference_mode():
            output = self.model(tensor, **resolved_kwargs)

        prediction = self._extract_prediction(output)
        if not torch.is_tensor(prediction):
            raise TypeError(
                "Model prediction must be a torch.Tensor, "
                f"got {type(prediction)!r}."
            )
        return prediction

    @staticmethod
    def _split_batch_prediction(
        prediction: torch.Tensor,
        expected_batch_size: int,
    ) -> List[torch.Tensor]:
        """Validate and split a model prediction along its batch dimension."""
        if prediction.ndim != 4:
            raise ValueError(
                "Batched directory prediction requires model output shape "
                f"[N,C,H,W], got {tuple(prediction.shape)}."
            )
        if prediction.shape[0] != expected_batch_size:
            raise ValueError(
                "Model output batch size must match the input batch size: "
                f"expected {expected_batch_size}, got {prediction.shape[0]}."
            )
        return list(prediction.unbind(dim=0))

    def _move_to_device(self, value: Any) -> Any:
        """Move tensor values in nested model arguments to the runtime device."""
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {
                key: self._move_to_device(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        return value

    @classmethod
    def _extract_prediction(cls, output: Any) -> torch.Tensor:
        """Extract a prediction tensor from supported model output layouts."""
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            if "pred" in output:
                return output["pred"]
            tensors = cls._flatten_tensors(output)
            if tensors:
                return tensors[-1]
            raise KeyError(
                "Model output dictionary does not contain a tensor prediction."
            )
        if isinstance(output, (tuple, list)):
            tensors = cls._flatten_tensors(output)
            if tensors:
                return tensors[-1]
        raise TypeError(
            "Cannot extract prediction tensor from model output type: "
            f"{type(output).__name__}."
        )

    @classmethod
    def _flatten_tensors(cls, value: Any) -> List[torch.Tensor]:
        """Collect tensors recursively from dictionaries, tuples, and lists."""
        if torch.is_tensor(value):
            return [value]

        tensors: List[torch.Tensor] = []
        if isinstance(value, dict):
            for item in value.values():
                tensors.extend(cls._flatten_tensors(item))
        elif isinstance(value, (tuple, list)):
            for item in value:
                tensors.extend(cls._flatten_tensors(item))
        return tensors

    @staticmethod
    def _build_transform(transform: Optional[Any]):
        """Build a callable image transform."""
        if transform is None:
            return predict_Trans
        if isinstance(transform, list):
            if not all(callable(item) for item in transform):
                raise TypeError("Every item in transform list must be callable.")
            return v2.Compose(transform)
        if callable(transform):
            return transform
        raise TypeError(
            "transform must be callable, a list of callables, or None, "
            f"got {type(transform)!r}."
        )

    @staticmethod
    def _normalize_resize_size(
        resize: ResizeInput,
    ) -> Optional[Tuple[int, int]]:
        """Normalize a resize value to ``(height, width)``."""
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

    @classmethod
    def _list_images(cls, directory: Path) -> List[Path]:
        """List supported images recursively in deterministic order."""
        return sorted(
            (
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() in cls.SUPPORTED_EXTENSIONS
            ),
            key=lambda path: str(path).lower(),
        )

    @staticmethod
    def _is_directory_source(source: ImageInput) -> bool:
        """Return whether ``source`` is an existing directory path."""
        return isinstance(source, (str, Path)) and Path(source).is_dir()

    def _resolve_single_output_path(
        self,
        *,
        image: ImageInput,
        save_path: Optional[Union[str, Path]],
        output_name: Optional[str],
        output_ext: Optional[str],
    ) -> Path:
        """Resolve the final output path for a single prediction."""
        original_path = Path(output_name or self._infer_source_name(image))
        suffix = (
            self._normalize_suffix(output_ext)
            if output_ext is not None
            else None
        )
        if suffix is not None:
            original_path = original_path.with_suffix(suffix)

        if save_path is None:
            return self.output_dir / original_path.name

        save_path = Path(save_path)
        if self._looks_like_file_path(save_path):
            return save_path.with_suffix(suffix) if suffix is not None else save_path
        return save_path / original_path.name

    @staticmethod
    def _looks_like_file_path(path: Path) -> bool:
        """Return whether a path denotes or resembles a file path."""
        if path.exists():
            return path.is_file()
        return bool(path.suffix)

    @staticmethod
    def _infer_source_name(source: ImageInput) -> str:
        """Infer an output filename from an image source."""
        if isinstance(source, Path):
            return source.name

        if isinstance(source, str):
            parsed = urllib.parse.urlparse(source)
            if parsed.scheme in {"http", "https", "ftp", "file"}:
                name = Path(urllib.parse.unquote(parsed.path)).name
                if name:
                    return name

            path = Path(source)
            if path.name and path.suffix:
                return path.name

        return "image.jpg"

    @staticmethod
    def _normalize_suffix(ext: str) -> str:
        """Normalize an output suffix to include a leading dot."""
        ext = ext.strip()
        if not ext:
            raise ValueError("output_ext must not be empty.")
        return ext if ext.startswith(".") else f".{ext}"

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert one prediction tensor in ``[0, 1]`` to a PIL image."""
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "Single-image prediction requires output batch size 1, "
                    f"got {tensor.shape[0]}."
                )
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(
                "Expected prediction tensor shape [C,H,W] or [1,C,H,W], "
                f"got {tuple(tensor.shape)}."
            )

        tensor = tensor.detach().cpu().float().clamp(0.0, 1.0)
        channels = tensor.shape[0]
        array = tensor.numpy()

        if channels == 1:
            image = np.clip(array[0] * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(image, mode="L")
        if channels not in {3, 4}:
            raise ValueError(
                "Prediction must contain 1, 3, or 4 image channels, "
                f"got {channels}."
            )

        image = np.transpose(array, (1, 2, 0))
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(image, mode="RGB" if channels == 3 else "RGBA")

    @staticmethod
    def _save_pil_image(image: Image.Image, save_path: Path) -> None:
        """Save a PIL prediction and create parent directories as needed."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        output_image = image
        if (
            save_path.suffix.lower() in {".jpg", ".jpeg"}
            and image.mode not in {"RGB", "L"}
        ):
            output_image = image.convert("RGB")
        output_image.save(save_path)
