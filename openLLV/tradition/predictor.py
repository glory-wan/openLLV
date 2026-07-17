"""Prediction interface for traditional low-light enhancement algorithms."""

import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from .algorithms import ImageInput, LLVEnhancer

__all__ = ["Predictor"]


class Predictor:
    """Predictor for traditional low-light image enhancement algorithms.

    It wraps the enhancers registered under openLLV.tradition.algorithms
    and supports every single-image input accepted by ImageReader. Directory
    input is also supported and will process all image files recursively.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
        ".webp", ".gif", ".ppm", ".pgm", ".pbm", ".sr", ".ras",
    }

    def __init__(
        self,
        method: Union[str, LLVEnhancer] = "he",
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a traditional-method predictor.

        Args:
            method: Registered method name or an existing ``LLVEnhancer``
                instance.
            output_dir: Default output directory used when no output path is
                provided.
            config: Optional enhancer configuration dictionary.
            **kwargs: Additional keyword arguments forwarded to the enhancer.
        """
        self.enhancer = self._load_enhancer(method, config=config, **kwargs)
        self.method_name = self._get_method_name(self.enhancer)
        self.output_dir = Path(output_dir) if output_dir is not None else Path("results") / self.method_name

    def __call__(
        self,
        source: ImageInput,
        output: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        """Enhance a source image or every image under a directory.

        Directory input is routed to predict_batch(). All other ImageReader
        compatible inputs are routed to predict_single().

        Args:
            source: Image source or directory path.
            output: Optional output file path or directory path.
            **kwargs: Additional keyword arguments forwarded to the enhancer.

        Returns:
            ``(enhanced_image, saved_path)`` for a single image, or a list of
            saved paths for a directory input.
        """
        if self._is_directory_source(source):
            return self.predict_batch(source, output_dir=output, **kwargs)
        return self.predict_single(source, save_path=output, **kwargs)

    def predict(self, source, output: Optional[Union[str, Path]] = None, **kwargs: Any):
        """Enhance a source through the unified call interface.

        Args:
            source: Image source or directory path.
            output: Optional output file path or directory path.
            **kwargs: Additional keyword arguments forwarded to ``__call__``.

        Returns:
            Prediction result returned by ``__call__``.
        """
        return self(source, output=output, **kwargs)

    def predict_single(
        self,
        image: ImageInput,
        save_path: Optional[Union[str, Path]] = None,
        *,
        output_name: Optional[str] = None,
        output_ext: Optional[str] = None,
        save: bool = True,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Optional[Path]]:
        """Enhance one image.

        If save_path is a directory, the original image name is preserved. If it
        is a file path, that exact file path is used; its suffix controls the
        encoded output format.

        Args:
            image: Image input accepted by the configured enhancer.
            save_path: Optional output file path or directory path.
            output_name: Optional output filename used when saving to a
                directory.
            output_ext: Optional output suffix override.
            save: Whether to save the enhanced image.
            **kwargs: Additional keyword arguments forwarded to the enhancer.

        Returns:
            Tuple of enhanced image array and saved path. The saved path is
            ``None`` when ``save`` is ``False``.

        Raises:
            TypeError: If the enhancer does not return a NumPy array.
        """
        enhanced = self.enhancer(image, **kwargs)
        if not isinstance(enhanced, np.ndarray):
            raise TypeError(
                f"Traditional predictor expects numpy output from enhancer, got {type(enhanced)!r}."
            )

        if not save:
            return enhanced, None

        target_path = self._resolve_single_output_path(
            image=image,
            save_path=save_path,
            output_name=output_name,
            output_ext=output_ext,
        )
        self._save_numpy_image(enhanced, target_path)
        return enhanced, target_path

    def predict_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        *,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> List[Path]:
        """Enhance all image files under ``input_dir`` recursively.

        Output filenames and suffixes are kept identical to the source images.
        Relative subdirectories are preserved to avoid filename collisions.

        Args:
            input_dir: Directory containing input images.
            output_dir: Optional directory where enhanced images are saved.
            progress_bar: Whether to display a tqdm progress bar.
            **kwargs: Additional keyword arguments forwarded to the enhancer.

        Returns:
            List of saved image paths.

        Raises:
            NotADirectoryError: If ``input_dir`` is not a directory.
            TypeError: If the enhancer does not return a NumPy array.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"input_dir must be a directory, got {input_dir}.")

        image_files = self._list_images(input_dir)
        if not image_files:
            return []

        output_root = Path(output_dir) if output_dir is not None else self.output_dir
        saved_paths: List[Path] = []
        iterator = tqdm(image_files, desc=f"Enhancing with {self.method_name}") if progress_bar else image_files

        for image_path in iterator:
            relative_path = image_path.relative_to(input_dir)
            target_path = output_root / relative_path
            enhanced = self.enhancer(str(image_path), **kwargs)

            if not isinstance(enhanced, np.ndarray):
                raise TypeError(
                    f"Traditional predictor expects numpy output from enhancer, got {type(enhanced)!r}."
                )

            self._save_numpy_image(enhanced, target_path)
            saved_paths.append(target_path)

            if progress_bar:
                iterator.set_postfix({"current": image_path.name})

        return saved_paths

    def get_params(self) -> Dict[str, Any]:
        """Get predictor and enhancer parameters.

        Returns:
            Dictionary containing method name, output directory, and enhancer
            parameters.
        """
        return {
            "method": self.method_name,
            "output_dir": str(self.output_dir),
            "enhancer": self.enhancer.get_params(),
        }

    @staticmethod
    def list_available_methods() -> List[str]:
        """List registered traditional enhancement algorithms.

        Returns:
            List of registered enhancer names and aliases.
        """
        return LLVEnhancer.list_registered_enhancers()

    @staticmethod
    def _get_method_name(enhancer: LLVEnhancer) -> str:
        """Get a normalized method name from an enhancer.

        Args:
            enhancer: Enhancer instance.

        Returns:
            Lowercase method name.
        """
        name = getattr(enhancer, "name", enhancer.__class__.__name__)
        return str(name).strip().lower() or enhancer.__class__.__name__.lower()

    @staticmethod
    def _load_enhancer(
        method: Union[str, LLVEnhancer],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> LLVEnhancer:
        """Create or configure an enhancer instance.

        Args:
            method: Registered method name or existing enhancer instance.
            config: Optional enhancer configuration dictionary.
            **kwargs: Additional keyword arguments forwarded to the enhancer.

        Returns:
            Configured enhancer instance with NumPy output enabled.

        Raises:
            TypeError: If ``method`` is neither a string nor an ``LLVEnhancer``.
        """
        if isinstance(method, LLVEnhancer):
            enhancer = method
            if config:
                enhancer.set_params(**config)
            if kwargs:
                enhancer.set_params(**kwargs)
            enhancer.set_params(output_type="numpy")
            return enhancer

        if not isinstance(method, str):
            raise TypeError(f"method must be str or LLVEnhancer, got {type(method)!r}.")

        params = {}
        if config:
            params.update(config)
        params.update(kwargs)
        params["output_type"] = "numpy"
        return LLVEnhancer.create_enhancer(method, **params)

    @classmethod
    def _list_images(cls, directory: Path) -> List[Path]:
        """List supported image files under a directory recursively.

        Args:
            directory: Root directory to scan.

        Returns:
            Sorted list of image file paths.
        """
        return sorted(
            [
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in cls.SUPPORTED_EXTENSIONS
            ],
            key=lambda path: str(path).lower(),
        )

    @staticmethod
    def _is_directory_source(source: ImageInput) -> bool:
        """Check whether a source points to an existing directory.

        Args:
            source: Candidate image source.

        Returns:
            ``True`` if ``source`` is a path-like directory, otherwise
            ``False``.
        """
        return isinstance(source, (str, Path)) and Path(source).is_dir()

    def _resolve_single_output_path(
        self,
        *,
        image: ImageInput,
        save_path: Optional[Union[str, Path]],
        output_name: Optional[str],
        output_ext: Optional[str],
    ) -> Path:
        """Resolve the output path for a single-image prediction.

        Args:
            image: Original image source.
            save_path: Optional output file path or directory path.
            output_name: Optional output filename override.
            output_ext: Optional output suffix override.

        Returns:
            Resolved output file path.
        """
        original_name = output_name or self._infer_source_name(image)
        original_path = Path(original_name)

        if output_ext is not None:
            suffix = self._normalize_suffix(output_ext)
            original_path = original_path.with_suffix(suffix)

        if save_path is None:
            return self.output_dir / original_path.name

        save_path = Path(save_path)

        if self._looks_like_file_path(save_path):
            return save_path

        return save_path / original_path.name

    @classmethod
    def _looks_like_file_path(cls, path: Path) -> bool:
        """Determine whether a path should be treated as a file path.

        Args:
            path: Candidate output path.

        Returns:
            ``True`` if the path exists as a file or has a file suffix.
        """
        if path.exists():
            return path.is_file()
        return bool(path.suffix)

    @classmethod
    def _infer_source_name(cls, source: ImageInput) -> str:
        """Infer a filename from an image source.

        Args:
            source: Image source accepted by the predictor.

        Returns:
            Source filename when it can be inferred, otherwise ``"image.jpg"``.
        """
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
        """Normalize an output extension.

        Args:
            ext: Extension with or without a leading dot.

        Returns:
            Extension with a leading dot.

        Raises:
            ValueError: If ``ext`` is empty after stripping whitespace.
        """
        ext = ext.strip()
        if not ext:
            raise ValueError("output_ext must not be empty.")
        return ext if ext.startswith(".") else f".{ext}"

    @staticmethod
    def _save_numpy_image(image: np.ndarray, save_path: Path) -> None:
        """Save a NumPy image with OpenCV.

        Args:
            image: Image array to save.
            save_path: Destination file path.

        Raises:
            ValueError: If OpenCV fails to save the image.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image = Predictor._ensure_uint8(image)
        ok = cv2.imwrite(str(save_path), image)
        if not ok:
            raise ValueError(f"Failed to save image to {save_path}.")

    @staticmethod
    def _ensure_uint8(image: np.ndarray) -> np.ndarray:
        """Convert an image array to uint8 for OpenCV writing.

        Args:
            image: Input image array.

        Returns:
            Uint8 image array clipped to ``[0, 255]``.
        """
        if image.dtype == np.uint8:
            return image

        image = image.astype(np.float32)
        if image.size > 0 and image.max() <= 1.0:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)
