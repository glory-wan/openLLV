"""Base class and registry utilities for traditional enhancement algorithms."""

from abc import ABC, abstractmethod
import inspect
from pathlib import Path
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
import warnings

import cv2
import numpy as np
from PIL import Image

from openLLV.data.image_io import ImageReader


ImageInput = Union[str, Path, bytes, bytearray, np.ndarray, Image.Image]
OutputType = Literal["numpy", "pil", "bytes", "base64", "file"]
ValueRangeName = Literal["auto", "unit", "byte"]
ValueRange = Union[ValueRangeName, Tuple[float, float], List[float]]
WorkingRange = Literal["unit", "byte"]
EnhanceOutput = Union[np.ndarray, Image.Image, bytes, str]

__all__ = ["LLVEnhancer", "ImageInput", "OutputType", "EnhanceOutput"]


class LLVEnhancer(ABC):
    """Base class for traditional low-level vision enhancement algorithms.

    The base class provides unified image reading through ``ImageReader``,
    output conversion, dtype clipping/casting, automatic subclass registration,
    and factory creation by enhancer name or alias. Public three-channel inputs
    and outputs are RGB; subclasses receive and return internal BGR arrays in
    the canonical range declared by ``working_range``.
    """

    name: str = "LLVEnhancer"
    _enhancer_registry: Dict[str, Type["LLVEnhancer"]] = {}
    aliases: List[str] = []
    working_range: WorkingRange = "byte"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register enhancer subclasses automatically.

        Args:
            **kwargs: Keyword arguments forwarded to superclass initialization.
        """
        super().__init_subclass__(**kwargs)

        if cls.__name__ == "LLVEnhancer":
            return

        LLVEnhancer._register_algorithm(cls)

    @classmethod
    def _normalize_registry_key(cls, name: str) -> str:
        """Normalize enhancer name for case-insensitive lookup.

        Args:
            name: Enhancer name or alias.

        Returns:
            Lowercase registry key without leading or trailing whitespace.
        """
        return name.strip().lower()

    @classmethod
    def _register_algorithm(
        cls,
        enhancer_class: Type["LLVEnhancer"],
    ) -> Type["LLVEnhancer"]:
        """Register an enhancer class and its aliases.

        Args:
            enhancer_class: Enhancer class to register.

        Returns:
            The registered enhancer class.

        Raises:
            TypeError: If ``enhancer_class`` does not inherit
                ``LLVEnhancer``.
        """
        if not issubclass(enhancer_class, LLVEnhancer):
            raise TypeError(
                f"enhancer_class must be a subclass of LLVEnhancer, "
                f"got {enhancer_class!r}."
            )

        if enhancer_class.__name__.startswith("_") or inspect.isabstract(enhancer_class):
            return enhancer_class

        candidate_names = [enhancer_class.__name__]
        if "name" in enhancer_class.__dict__:
            candidate_names.append(getattr(enhancer_class, "name"))
        if "aliases" in enhancer_class.__dict__:
            candidate_names.extend(getattr(enhancer_class, "aliases"))

        for name in candidate_names:
            if not isinstance(name, str) or not name.strip():
                continue

            key = cls._normalize_registry_key(name)
            cls._enhancer_registry[key] = enhancer_class

        return enhancer_class

    @classmethod
    def register(
        cls,
        enhancer_class: Type["LLVEnhancer"],
    ) -> Type["LLVEnhancer"]:
        """Register an enhancer class manually.

        Args:
            enhancer_class: Enhancer class to register.

        Returns:
            The registered enhancer class.
        """
        return cls._register_algorithm(enhancer_class)

    @classmethod
    def list_registered_enhancers(cls) -> List[str]:
        """List registered enhancer names and aliases.

        Returns:
            Sorted registry keys.
        """
        return sorted(cls._enhancer_registry.keys())

    @classmethod
    def create_enhancer(
        cls,
        enhancer_name: str,
        **kwargs: Any,
    ) -> "LLVEnhancer":
        """Create an enhancer instance by name.

        Args:
            enhancer_name: Registered enhancer name, class name, or alias.
            **kwargs: Keyword arguments intended for the enhancer constructor.
                Parameters unsupported by the selected enhancer are ignored
                with a warning.

        Returns:
            Instantiated enhancer.

        Raises:
            ValueError: If ``enhancer_name`` is empty or not registered.
        """
        if not isinstance(enhancer_name, str) or not enhancer_name.strip():
            raise ValueError("enhancer_name must be a non-empty string.")

        enhancer_name = cls._normalize_registry_key(enhancer_name)
        enhancer_class = cls._enhancer_registry.get(enhancer_name)

        if enhancer_class is None:
            available = cls.list_registered_enhancers()
            suggestion = cls._get_similar_enhancer_name(enhancer_name, available)

            raise ValueError(
                f"Enhancer '{enhancer_name}' is not registered.\n"
                f"Available enhancers: {available}\n"
                f"Did you mean: {suggestion}"
            )

        constructor_kwargs, unused_kwargs = cls._filter_constructor_kwargs(
            enhancer_class,
            kwargs,
        )
        enhancer = enhancer_class(**constructor_kwargs)
        cls._print_enhancer_params(enhancer)

        if unused_kwargs:
            unused_text = ", ".join(
                f"{name}={value!r}"
                for name, value in unused_kwargs.items()
            )
            warnings.warn(
                f"Algorithm '{enhancer_class.__name__}' does not use the "
                f"following parameter(s): {unused_text}. These parameters "
                "were ignored.",
                UserWarning,
                stacklevel=2,
            )

        return enhancer

    @classmethod
    def _get_constructor_parameter_names(
        cls,
        enhancer_class: Type["LLVEnhancer"],
    ) -> List[str]:
        """Collect explicit constructor parameters from an enhancer hierarchy.

        Concrete algorithms commonly accept ``**kwargs`` only to forward the
        base options to ``LLVEnhancer``. Variadic parameters are therefore not
        treated as permission to accept arbitrary user keywords.

        Args:
            enhancer_class: Concrete enhancer class selected by the factory.

        Returns:
            Sorted constructor parameter names accepted by the concrete class
            and its ``LLVEnhancer`` ancestors.
        """
        parameter_names = set()

        for current_class in enhancer_class.__mro__:
            if not issubclass(current_class, LLVEnhancer):
                continue

            constructor = current_class.__dict__.get("__init__")
            if constructor is not None:
                try:
                    signature = inspect.signature(constructor)
                except (TypeError, ValueError):
                    signature = None

                if signature is not None:
                    for name, parameter in signature.parameters.items():
                        if name == "self":
                            continue
                        if parameter.kind in {
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        }:
                            parameter_names.add(name)

            if current_class is LLVEnhancer:
                break

        return sorted(parameter_names)

    @classmethod
    def _filter_constructor_kwargs(
        cls,
        enhancer_class: Type["LLVEnhancer"],
        kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split constructor keywords into supported and unused mappings."""
        supported_names = set(
            cls._get_constructor_parameter_names(enhancer_class)
        )
        supported = {
            name: value
            for name, value in kwargs.items()
            if name in supported_names
        }
        unused = {
            name: value
            for name, value in kwargs.items()
            if name not in supported_names
        }
        return supported, unused

    @classmethod
    def _print_enhancer_params(cls, enhancer: "LLVEnhancer") -> None:
        """Print the selected algorithm and its effective parameter values."""
        params = dict(enhancer.get_params())
        for name in cls._get_constructor_parameter_names(
            enhancer.__class__
        ):
            if name not in params and hasattr(enhancer, name):
                params[name] = getattr(enhancer, name)

        print(f"Algorithm: {enhancer.__class__.__name__}", file=sys.stderr)
        print("Parameters:", file=sys.stderr)
        if not params:
            print("  (none)", file=sys.stderr, flush=True)
            return

        param_items = list(params.items())
        for index, (name, value) in enumerate(param_items):
            print(
                f"  {name}: {value!r}",
                file=sys.stderr,
                flush=index == len(param_items) - 1,
            )

    @classmethod
    def _get_similar_enhancer_name(
        cls,
        enhancer_name: str,
        available_enhancers: List[str],
        max_suggestions: int = 3,
    ) -> str:
        """Find close enhancer-name suggestions.

        Args:
            enhancer_name: Requested enhancer name.
            available_enhancers: Registered enhancer names and aliases.
            max_suggestions: Maximum number of suggestions.

        Returns:
            Comma-separated suggestion string, or a fallback message.
        """
        from difflib import get_close_matches

        suggestions = get_close_matches(
            enhancer_name,
            available_enhancers,
            n=max_suggestions,
            cutoff=0.4,
        )

        if suggestions:
            return ", ".join(suggestions)

        return "No similar enhancers found"

    def __init__(
        self,
        *,
        output_type: OutputType = "numpy",
        keep_dtype: bool = True,
        clip_output: bool = True,
        value_range: ValueRange = "auto",
    ) -> None:
        """Initialize a traditional enhancer.

        Args:
            output_type: Output format returned by ``enhance``.
            keep_dtype: Whether to cast output back to the input dtype.
            clip_output: Whether to clip output to the resolved input value
                range.
            value_range: Input value-range convention. ``"auto"`` infers
                ``[0, 1]`` or ``[0, 255]`` for floating arrays; ``"unit"``
                and ``"byte"`` select those ranges explicitly. A two-value
                tuple or list defines a custom ``(minimum, maximum)`` range.

        Raises:
            TypeError: If boolean options or ``value_range`` have invalid
                types.
            ValueError: If ``output_type`` or ``value_range`` is invalid.
        """
        self.output_type = output_type
        self.keep_dtype = keep_dtype
        self.clip_output = clip_output
        self.value_range = value_range
        self.image_reader = ImageReader()

        self._validate_base_params()

    def __call__(
        self,
        image: ImageInput,
        *,
        output_ext: Optional[str] = None,
        value_range: Optional[ValueRange] = None,
        **kwargs: Any,
    ) -> EnhanceOutput:
        """Enhance a single image.

        Args:
            image: Image input supported by ``ImageReader``.
            output_ext: Optional output extension for encoded outputs.
            value_range: Optional input value-range override for this call.
            **kwargs: Method-specific enhancement parameters.

        Returns:
            Enhanced image in the configured output type. Three-channel NumPy
            output uses public RGB order.
        """
        return self.enhance(
            image,
            output_ext=output_ext,
            value_range=value_range,
            **kwargs,
        )

    def enhance(
        self,
        image: ImageInput,
        *,
        output_ext: Optional[str] = None,
        value_range: Optional[ValueRange] = None,
        **kwargs: Any,
    ) -> EnhanceOutput:
        """Enhance a single image.

        Args:
            image: Image input supported by ``ImageReader``.
            output_ext: Optional output extension for encoded outputs.
            value_range: Optional input value-range override for this call.
            **kwargs: Method-specific enhancement parameters.

        Returns:
            Enhanced image in the configured output type.

        Raises:
            TypeError: If subclass ``_enhance`` does not return a numpy array.
            ValueError: If input values do not fit the selected value range.
        """
        original_dtype = self._get_original_dtype(image)
        img = self._load_image(image)

        self._validate_image(img)
        source_range = self._resolve_value_range(
            img,
            value_range=value_range,
        )
        working_image = self._convert_to_working_range(
            img,
            source_range=source_range,
        )

        enhanced = self._enhance(working_image, **kwargs)

        if not isinstance(enhanced, np.ndarray):
            raise TypeError(
                f"{self.__class__.__name__}._enhance() must return np.ndarray, "
                f"got {type(enhanced)!r}."
            )

        enhanced = self._postprocess(
            enhanced,
            original_dtype=original_dtype,
            source_range=source_range,
        )
        enhanced = self._bgr_to_rgb(enhanced)

        if self.output_type != "numpy":
            output_kwargs = {}
            if output_ext is not None:
                output_kwargs["ext"] = output_ext

            return self.image_reader(
                self._convert_source_range_to_uint8(
                    enhanced,
                    source_range=source_range,
                ),
                output_format=self.output_type,
                **output_kwargs,
            )

        return enhanced

    @abstractmethod
    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Implement method-specific enhancement.

        Image input is a numpy array loaded by ``ImageReader``. Three-channel
        numpy inputs follow OpenCV-style BGR channel order. Values are uint8
        in ``[0, 255]`` when ``working_range == "byte"`` and float32 in
        ``[0, 1]`` when ``working_range == "unit"``. Return the same range.

        Args:
            image: Input image array.
            **kwargs: Method-specific enhancement parameters.

        Returns:
            Enhanced image array.
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        """Get enhancer parameters.

        Returns:
            Dictionary containing base enhancer parameters.
        """
        return {
            "name": self.name,
            "output_type": self.output_type,
            "keep_dtype": self.keep_dtype,
            "clip_output": self.clip_output,
            "value_range": self.value_range,
        }

    def set_params(self, **params: Any) -> "LLVEnhancer":
        """Set enhancer parameters.

        Args:
            **params: Parameter names and values.

        Returns:
            The enhancer itself.

        Raises:
            ValueError: If a parameter name does not exist.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"{self.__class__.__name__} has no parameter {key!r}."
                )
            setattr(self, key, value)

        self._validate_base_params()
        return self

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            String representation of the enhancer and its parameters.
        """
        params = self.get_params()
        params_str = ", ".join(
            f"{k}={v!r}" for k, v in params.items() if k != "name"
        )
        return f"{self.__class__.__name__}({params_str})"

    def _load_image(self, image: ImageInput) -> np.ndarray:
        """Load an image input as a numpy array.

        ``ImageReader`` returns public RGB arrays. This method converts them to
        OpenCV-style BGR before invoking a traditional algorithm.

        Args:
            image: Image input supported by ``ImageReader``.

        Returns:
            Image as a numpy array.

        Raises:
            TypeError: If ``ImageReader`` does not return a numpy array.
        """
        image_data = str(image) if isinstance(image, Path) else image

        img = self.image_reader(image_data, output_format="numpy")

        if not isinstance(img, np.ndarray):
            raise TypeError(
                f"ImageReader must return np.ndarray when output_format='numpy', "
                f"got {type(img)!r}."
            )

        return self._rgb_to_bgr(img)

    @staticmethod
    def _rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert a public RGB/RGBA array to internal BGR/BGRA order."""
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        return image

    @staticmethod
    def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert an internal BGR/BGRA array to public RGB/RGBA order."""
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return image

    @staticmethod
    def _get_original_dtype(image: ImageInput) -> Optional[np.dtype]:
        """Infer original image dtype.

        Args:
            image: Original image input.

        Returns:
            Original numpy dtype when available, otherwise ``uint8``.
        """
        if isinstance(image, np.ndarray):
            return image.dtype

        return np.dtype(np.uint8)

    def _postprocess(
        self,
        image: np.ndarray,
        *,
        original_dtype: Optional[np.dtype],
        source_range: Tuple[float, float],
    ) -> np.ndarray:
        """Clip and cast enhanced output.

        Args:
            image: Enhanced image array.
            original_dtype: Original input dtype.
            source_range: Resolved semantic range of the input image.

        Returns:
            Postprocessed image array.
        """
        image = self._restore_source_range(
            image,
            source_range=source_range,
        )

        if self.clip_output:
            image = np.clip(image, source_range[0], source_range[1])

        if self.keep_dtype and original_dtype is not None:
            image = self._cast_to_dtype(image, original_dtype)

        return image

    def _validate_base_params(self) -> None:
        """Validate base enhancer parameters.

        Raises:
            TypeError: If boolean parameters or ``value_range`` have invalid
                types.
            ValueError: If output, working, or input value-range options are
                unsupported.
        """
        valid_output_types = {"numpy", "pil", "bytes", "base64", "file"}
        valid_working_ranges = {"unit", "byte"}

        if self.output_type not in valid_output_types:
            raise ValueError(
                f"output_type must be one of {valid_output_types}, "
                f"got {self.output_type!r}."
            )

        if not isinstance(self.keep_dtype, bool):
            raise TypeError(
                f"keep_dtype must be bool, got {type(self.keep_dtype)!r}."
            )

        if not isinstance(self.clip_output, bool):
            raise TypeError(
                f"clip_output must be bool, got {type(self.clip_output)!r}."
            )

        self.value_range = self._normalize_value_range(self.value_range)

        if self.working_range not in valid_working_ranges:
            raise ValueError(
                f"working_range must be one of {valid_working_ranges}, "
                f"got {self.working_range!r}."
            )

    @staticmethod
    def _normalize_value_range(value_range: ValueRange) -> ValueRange:
        """Validate and normalize an input value-range specification."""
        if isinstance(value_range, str):
            if value_range not in {"auto", "unit", "byte"}:
                raise ValueError(
                    "value_range must be 'auto', 'unit', 'byte', or a "
                    f"two-value range, got {value_range!r}."
                )
            return value_range

        if not isinstance(value_range, (tuple, list)) or len(value_range) != 2:
            raise TypeError(
                "value_range must be 'auto', 'unit', 'byte', or a "
                "two-value tuple/list."
            )

        minimum, maximum = value_range
        if (
            isinstance(minimum, (bool, np.bool_))
            or isinstance(maximum, (bool, np.bool_))
            or not isinstance(minimum, (int, float, np.integer, np.floating))
            or not isinstance(maximum, (int, float, np.integer, np.floating))
        ):
            raise TypeError("Custom value_range bounds must be numeric.")

        minimum = float(minimum)
        maximum = float(maximum)
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError("Custom value_range bounds must be finite.")
        if minimum >= maximum:
            raise ValueError(
                "Custom value_range minimum must be less than maximum."
            )

        return minimum, maximum

    def _resolve_value_range(
        self,
        image: np.ndarray,
        *,
        value_range: Optional[ValueRange] = None,
    ) -> Tuple[float, float]:
        """Resolve and validate the semantic value range of an image."""
        requested = self.value_range if value_range is None else value_range
        requested = self._normalize_value_range(requested)

        if np.issubdtype(image.dtype, np.floating) and not np.all(
            np.isfinite(image)
        ):
            raise ValueError("Floating image values must all be finite.")

        image_min = float(np.min(image))
        image_max = float(np.max(image))

        if requested == "auto":
            if image_min < 0.0:
                raise ValueError(
                    "value_range='auto' does not accept negative image "
                    "values; pass an explicit (minimum, maximum) range."
                )

            if np.issubdtype(image.dtype, np.floating):
                if image_max <= 1.0:
                    resolved = (0.0, 1.0)
                elif image_max <= 255.0:
                    resolved = (0.0, 255.0)
                else:
                    raise ValueError(
                        "value_range='auto' only infers floating images in "
                        "[0, 1] or [0, 255]; pass an explicit "
                        "(minimum, maximum) range."
                    )
            else:
                dtype = np.dtype(image.dtype)
                if dtype == np.dtype(np.uint8):
                    resolved = (0.0, 255.0)
                else:
                    resolved = (0.0, float(np.iinfo(dtype).max))
        elif requested == "unit":
            resolved = (0.0, 1.0)
        elif requested == "byte":
            resolved = (0.0, 255.0)
        else:
            resolved = requested

        if image_min < resolved[0] or image_max > resolved[1]:
            raise ValueError(
                f"Image values [{image_min}, {image_max}] fall outside "
                f"value_range={resolved}."
            )

        return resolved

    def _convert_to_working_range(
        self,
        image: np.ndarray,
        *,
        source_range: Tuple[float, float],
    ) -> np.ndarray:
        """Convert an input image to the algorithm's canonical range."""
        minimum, maximum = source_range
        normalized = (
            image.astype(np.float32) - minimum
        ) / (maximum - minimum)

        if self.working_range == "unit":
            return normalized.astype(np.float32, copy=False)

        return np.rint(normalized * 255.0).astype(np.uint8)

    def _restore_source_range(
        self,
        image: np.ndarray,
        *,
        source_range: Tuple[float, float],
    ) -> np.ndarray:
        """Map algorithm output from its working range to the source range."""
        normalized = image.astype(np.float32)
        if self.working_range == "byte":
            normalized = normalized / 255.0

        minimum, maximum = source_range
        return normalized * (maximum - minimum) + minimum

    @staticmethod
    def _convert_source_range_to_uint8(
        image: np.ndarray,
        *,
        source_range: Tuple[float, float],
    ) -> np.ndarray:
        """Map a source-range image to uint8 for PIL/file encoding."""
        minimum, maximum = source_range
        normalized = (
            image.astype(np.float32) - minimum
        ) / (maximum - minimum)
        normalized = np.clip(normalized, 0.0, 1.0)
        return np.rint(normalized * 255.0).astype(np.uint8)

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Validate image array structure and dtype.

        Args:
            image: Image array to validate.

        Raises:
            TypeError: If input is not a supported numeric numpy array.
            ValueError: If shape or size is invalid.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be np.ndarray, got {type(image)!r}.")

        if image.ndim not in (2, 3):
            raise ValueError(
                f"image must have shape (H, W) or (H, W, C), got {image.shape}."
            )

        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError(
                "For color images, channel dimension must be 1, 3, or 4, "
                f"got shape {image.shape}."
            )

        if image.size == 0:
            raise ValueError("image must not be empty.")

        if not np.issubdtype(image.dtype, np.integer) and not np.issubdtype(
            image.dtype, np.floating
        ):
            raise TypeError(
                f"image dtype must be integer or floating point, got {image.dtype}."
            )

    @staticmethod
    def _dtype_range(dtype: Optional[np.dtype]) -> Tuple[float, float]:
        """Get valid value range for a dtype.

        Args:
            dtype: Numpy dtype.

        Returns:
            Tuple containing minimum and maximum valid values.

        Raises:
            TypeError: If dtype is unsupported.
        """
        if dtype is None:
            return 0.0, 255.0

        dtype = np.dtype(dtype)

        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return float(info.min), float(info.max)

        if np.issubdtype(dtype, np.floating):
            info = np.finfo(dtype)
            return float(info.min), float(info.max)

        raise TypeError(f"Unsupported dtype: {dtype}")

    @classmethod
    def _clip_to_valid_range(
        cls,
        image: np.ndarray,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """Clip image values to a dtype-valid range.

        Args:
            image: Input image array.
            dtype: Optional dtype used to determine valid range.

        Returns:
            Clipped image array.
        """
        min_value, max_value = cls._dtype_range(dtype or image.dtype)
        return np.clip(image, min_value, max_value)

    @classmethod
    def _cast_to_dtype(cls, image: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Cast image array to a target dtype.

        Args:
            image: Input image array.
            dtype: Target dtype.

        Returns:
            Cast image array.
        """
        dtype = np.dtype(dtype)

        if np.issubdtype(dtype, np.integer):
            image = np.rint(image)
            image = cls._clip_to_valid_range(image, dtype)

        elif np.issubdtype(dtype, np.floating):
            image = cls._clip_to_valid_range(image, dtype)

        return image.astype(dtype, copy=False)
