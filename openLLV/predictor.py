"""Unified prediction interface for openLLV."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .deepLearning.models import LLVModel
from .deepLearning.predictor import Predictor as DeepLearningPredictor
from .tradition.algorithms import LLVEnhancer
from .tradition.predictor import Predictor as TraditionalPredictor


TargetInput = Union[str, Path, LLVModel, LLVEnhancer]

__all__ = ["Predictor"]


class Predictor:
    """Route prediction to a deep-learning model or traditional algorithm.

    The selected backend can be provided explicitly or inferred from a model
    name, checkpoint path, algorithm name, or backend instance. Once created,
    this class exposes the same call, single-image, and batch entry points as
    the backend predictor.
    """

    DEEP_BACKENDS = {
        "deep",
        "deeplearning",
        "deep_learning",
        "dl",
        "model",
    }
    TRADITIONAL_BACKENDS = {
        "tradition",
        "traditional",
        "traditionalalgorithm",
        "traditional_algorithm",
        "ta",
        "method",
        "algorithm",
    }

    def __init__(
        self,
        target: Optional[TargetInput] = None,
        *,
        model: Optional[Union[str, Path, LLVModel]] = None,
        method: Optional[Union[str, LLVEnhancer]] = None,
        backend: str = "auto",
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Any] = None,
        transform: Optional[Any] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a unified predictor.

        Args:
            target: Model name, checkpoint path, algorithm name, model
                instance, or enhancer instance.
            model: Explicit deep-learning model selector.
            method: Explicit traditional-algorithm selector.
            backend: ``"auto"``, a deep-learning backend alias, or a
                traditional backend alias.
            output_dir: Default directory used for saved predictions.
            config: Model or algorithm configuration.
            device: Device used by the deep-learning predictor.
            transform: Input transform used by the deep-learning predictor.
            batch_size: Deep-learning batch prediction size.
            num_workers: Deep-learning data-loader worker count.
            **kwargs: Model configuration overrides for the deep backend, or
                algorithm parameter overrides for the traditional backend.

        Raises:
            TypeError: If ``config`` or a backend instance has an invalid type.
            ValueError: If selectors conflict or the backend cannot be resolved.
        """
        if config is not None and not isinstance(config, dict):
            raise TypeError(
                f"config must be a dictionary or None, got "
                f"{type(config).__name__}."
            )

        self.backend = self._resolve_backend(
            target=target,
            model=model,
            method=method,
            backend=backend,
        )

        if self.backend == "deep":
            model_input = self._resolve_model_input(
                target=target,
                model=model,
                method=method,
            )
            model_config = dict(config or {})
            model_config.update(kwargs)
            self.predictor = DeepLearningPredictor(
                model=model_input,
                output_dir=output_dir,
                config=model_config or None,
                device=device,
                transform=transform,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        else:
            method_input = self._resolve_method_input(
                target=target,
                model=model,
                method=method,
            )
            self.predictor = TraditionalPredictor(
                method=method_input,
                output_dir=output_dir,
                config=dict(config) if config is not None else None,
                **kwargs,
            )

    def __call__(
        self,
        source: Any,
        output: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Any:
        """Predict an image or directory through the selected backend."""
        return self.predictor(source, output=output, **kwargs)

    def predict(
        self,
        source: Any,
        output: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Any:
        """Predict an image or directory through the unified call interface."""
        return self(source, output=output, **kwargs)

    def predict_single(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate single-image prediction to the selected backend."""
        return self.predictor.predict_single(*args, **kwargs)

    def predict_batch(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate batch or directory prediction to the selected backend."""
        return self.predictor.predict_batch(*args, **kwargs)

    def get_params(self) -> Dict[str, Any]:
        """Return the selected backend and its predictor parameters."""
        return {
            "backend": self.backend,
            "predictor": self.predictor.get_params(),
        }

    @classmethod
    def list_available_models(cls) -> List[str]:
        """Return registered deep-learning model names and aliases."""
        return DeepLearningPredictor.list_available_models()

    @classmethod
    def list_available_methods(cls) -> List[str]:
        """Return registered traditional algorithm names and aliases."""
        return TraditionalPredictor.list_available_methods()

    @classmethod
    def list_available(cls) -> Dict[str, List[str]]:
        """Return all models and traditional algorithms by category."""
        return {
            "models": cls.list_available_models(),
            "algorithms": cls.list_available_methods(),
        }

    @classmethod
    def _resolve_backend(
        cls,
        *,
        target: Optional[TargetInput],
        model: Optional[Union[str, Path, LLVModel]],
        method: Optional[Union[str, LLVEnhancer]],
        backend: str,
    ) -> str:
        """Resolve and normalize the requested predictor backend."""
        if model is not None and method is not None:
            raise ValueError("Pass only one of 'model' or 'method'.")
        if target is not None and (model is not None or method is not None):
            raise ValueError(
                "Pass either positional 'target' or keyword 'model'/'method', "
                "not both."
            )
        if not isinstance(backend, str):
            raise TypeError(
                f"backend must be a string, got {type(backend).__name__}."
            )

        normalized_backend = backend.strip().lower()
        if normalized_backend in cls.DEEP_BACKENDS:
            return "deep"
        if normalized_backend in cls.TRADITIONAL_BACKENDS:
            return "traditional"
        if normalized_backend != "auto":
            raise ValueError(
                f"Unsupported backend '{backend}'. Use 'auto', 'deep', or "
                "'traditional'."
            )

        candidate = (
            model
            if model is not None
            else method
            if method is not None
            else target
        )
        if isinstance(candidate, LLVModel):
            return "deep"
        if isinstance(candidate, LLVEnhancer):
            return "traditional"
        if method is not None:
            return "traditional"
        if model is not None:
            return "deep"
        if candidate is None:
            raise ValueError("A model, method, or target name is required.")

        candidate_name = str(candidate) if isinstance(candidate, Path) else candidate
        if isinstance(candidate_name, str):
            if Path(candidate_name).suffix.lower() in {".pt", ".pth"}:
                return "deep"

            deep_lookup = {
                name.strip().lower()
                for name in DeepLearningPredictor.list_available_models()
            }
            method_lookup = {
                name.strip().lower()
                for name in TraditionalPredictor.list_available_methods()
            }
            key = candidate_name.strip().lower()

            if key in deep_lookup and key in method_lookup:
                raise ValueError(
                    f"'{candidate_name}' is both a deep-learning model and a "
                    "traditional algorithm. Pass backend='deep' or "
                    "backend='traditional'."
                )
            if key in deep_lookup:
                return "deep"
            if key in method_lookup:
                return "traditional"

        raise ValueError(
            f"Cannot infer predictor backend from {candidate!r}.\n"
            f"Available models: {DeepLearningPredictor.list_available_models()};\n"
            "available algorithms: "
            f"{TraditionalPredictor.list_available_methods()}."
        )

    @staticmethod
    def _resolve_model_input(
        *,
        target: Optional[TargetInput],
        model: Optional[Union[str, Path, LLVModel]],
        method: Optional[Union[str, LLVEnhancer]],
    ) -> Union[str, Path, LLVModel]:
        """Resolve the selector passed to the deep-learning predictor."""
        model_input = model if model is not None else target
        if model_input is None or method is not None:
            raise ValueError(
                "A deep-learning model name, checkpoint path, or LLVModel "
                "instance is required."
            )
        if isinstance(model_input, LLVEnhancer):
            raise TypeError(
                "LLVEnhancer instances must use the traditional backend."
            )
        if not isinstance(model_input, (str, Path, LLVModel)):
            raise TypeError(
                "Deep-learning model input must be a name, checkpoint path, "
                f"or LLVModel instance, got {type(model_input).__name__}."
            )
        return model_input

    @staticmethod
    def _resolve_method_input(
        *,
        target: Optional[TargetInput],
        model: Optional[Union[str, Path, LLVModel]],
        method: Optional[Union[str, LLVEnhancer]],
    ) -> Union[str, LLVEnhancer]:
        """Resolve the selector passed to the traditional predictor."""
        method_input = method if method is not None else target
        if method_input is None or model is not None:
            raise ValueError(
                "A traditional algorithm name or LLVEnhancer instance is "
                "required."
            )
        if isinstance(method_input, LLVModel):
            raise TypeError("LLVModel instances must use the deep backend.")
        if isinstance(method_input, Path):
            method_input = str(method_input)
        if not isinstance(method_input, (str, LLVEnhancer)):
            raise TypeError(
                "Traditional algorithm input must be a name or LLVEnhancer "
                f"instance, got {type(method_input).__name__}."
            )
        return method_input
