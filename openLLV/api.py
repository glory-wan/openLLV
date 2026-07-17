"""Top-level convenience API for openLLV."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union


_PREDICT_CALL_KWARGS = {
    "progress_bar",
    "output_name",
    "output_ext",
    "save",
    "model_kwargs",
    "ext",
    "timeout",
    "headers",
    "verify_ssl",
}

_MISSING = object()


def _split_predict_kwargs(
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split predictor construction options from prediction-call options."""
    predictor_kwargs: Dict[str, Any] = {}
    call_kwargs: Dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in _PREDICT_CALL_KWARGS:
            call_kwargs[key] = value
        else:
            predictor_kwargs[key] = value

    return predictor_kwargs, call_kwargs


def predict(
    method: Any,
    source: Any,
    output: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Any:
    """Process an image or directory with a model or traditional algorithm.

    Args:
        method: Registered model/algorithm name, checkpoint path, ``LLVModel``
            instance, or ``LLVEnhancer`` instance.
        source: Image source or directory accepted by the selected predictor.
        output: Optional output file or directory.
        **kwargs: Predictor construction options, model/algorithm parameters,
            or prediction-call options.

    Returns:
        Result returned by the selected backend predictor.

    Examples:
        ``openLLV.predict("ZeroDCE", "input.jpg", output="out.png")``
        ``openLLV.predict("he", "images", output="results/he")``
    """
    from .predictor import Predictor

    predictor_kwargs, call_kwargs = _split_predict_kwargs(kwargs)
    predictor = Predictor(method, **predictor_kwargs)
    return predictor(source, output=output, **call_kwargs)


def enhance(
    method: Any,
    source: Any,
    output: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Any:
    """Alias for :func:`predict` retained for enhancement workflows."""
    return predict(method, source, output=output, **kwargs)


def train(
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Construct a Trainer and run its complete training loop.

    Args:
        config: Built-in config name, YAML path, configuration dictionary, or
            ``None``.
        **kwargs: Configuration overrides forwarded to ``Trainer``.

    Returns:
        Training result returned by ``Trainer.train()``.
    """
    from .deepLearning import Trainer

    trainer = Trainer(config, **kwargs)
    return trainer.train()


def evaluate(
    en_img_dir: Any = _MISSING,
    ref_img_dir: Any = _MISSING,
    metrics: Optional[Union[str, List[str]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    return_evaluator: bool = False,
    *,
    en: Any = _MISSING,
    ref: Any = _MISSING,
    **kwargs: Any,
) -> Any:
    """Evaluate a directory of processed images.

    Args:
        en_img_dir: Directory containing processed images.
        ref_img_dir: Optional reference-image directory.
        metrics: Metric name or list of metric names.
        save_path: Optional path used to save JSON results.
        return_evaluator: Return the ``Evaluator`` instead of its results.
        en: Backward-compatible alias for ``en_img_dir``.
        ref: Backward-compatible alias for ``ref_img_dir``.
        **kwargs: Options forwarded to ``Evaluator``.

    Returns:
        Evaluation results, or the ``Evaluator`` instance when requested.
    """
    from .evaluation import Evaluator
    from .evaluation import metrics as _metrics  # noqa: F401

    if en is not _MISSING:
        if en_img_dir is not _MISSING:
            raise TypeError(
                "evaluate() received both 'en_img_dir' and its alias 'en'"
            )
        en_img_dir = en

    if ref is not _MISSING:
        if ref_img_dir is not _MISSING:
            raise TypeError(
                "evaluate() received both 'ref_img_dir' and its alias 'ref'"
            )
        ref_img_dir = ref

    if en_img_dir is _MISSING:
        raise TypeError("evaluate() missing required argument: 'en_img_dir'")
    if ref_img_dir is _MISSING:
        ref_img_dir = None

    evaluator = Evaluator(
        en_img_dir=str(en_img_dir),
        ref_img_dir=str(ref_img_dir) if ref_img_dir is not None else None,
        metrics=metrics,
        save_path=save_path,
        **kwargs,
    )
    return evaluator if return_evaluator else evaluator.results


def eval(*args: Any, **kwargs: Any) -> Any:
    """Alias for :func:`evaluate`."""
    return evaluate(*args, **kwargs)


def imread(
    source: Any,
    output_format: str = "pil",
    **kwargs: Any,
) -> Any:
    """Read an image in any format supported by ``ImageReader``."""
    from .data.image_io import read_image as _read_image

    return _read_image(source, output_format=output_format, **kwargs)


def imwrite(
    image: Any,
    output: Optional[Union[str, Path]] = None,
    *,
    save_format: Optional[str] = None,
    output_name: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """Write an image in any format supported by ``ImageWriter``."""
    from .data.image_io import write_image as _write_image

    return _write_image(
        image,
        output=output,
        save_format=save_format,
        output_name=output_name,
        **kwargs,
    )


read_image = imread
write_image = imwrite


def list_models() -> List[str]:
    """Return all registered model lookup names and aliases."""
    from .predictor import Predictor

    return Predictor.list_available_models()


def list_algorithms() -> List[str]:
    """Return all registered traditional algorithm names and aliases."""
    from .predictor import Predictor

    return Predictor.list_available_methods()


def list_metrics() -> List[str]:
    """Return all registered evaluation metric names."""
    from .evaluation import Evaluator
    from .evaluation import metrics as _metrics  # noqa: F401

    return Evaluator.list_available_metrics()


def list_losses() -> List[str]:
    """Return all registered deep-learning loss names and aliases."""
    from .deepLearning.loss import BaseLoss

    return BaseLoss.list_registered_losses()


def list_datasets() -> List[str]:
    """Return all registered dataset names and aliases."""
    from .data.datasets import BaseDataset

    return BaseDataset.list_registered_datasets()


def _component_rows(
    registry: Mapping[str, Type[Any]],
) -> List[Dict[str, Any]]:
    """Build one deduplicated display row per registered component class."""
    component_classes = set(registry.values())
    rows: List[Dict[str, Any]] = []

    for component_class in sorted(
        component_classes,
        key=lambda value: value.__name__.casefold(),
    ):
        aliases = getattr(component_class, "aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        else:
            aliases = list(aliases)

        rows.append(
            {
                "name": component_class.__name__,
                "aliases": aliases,
            }
        )

    return rows


def list_available() -> Dict[str, List[Dict[str, Any]]]:
    """Return public component classes grouped by registry category.

    Each implementation class appears once and retains its declared aliases,
    unlike the flat ``list_*`` helpers that return every accepted lookup key.
    """
    from .data.datasets import BaseDataset
    from .deepLearning.loss import BaseLoss
    from .deepLearning.models import LLVModel
    from .evaluation import BaseMetric
    from .evaluation import metrics as _metrics  # noqa: F401
    from .tradition.algorithms import LLVEnhancer

    return {
        "models": _component_rows(LLVModel._model_registry),
        "algorithms": _component_rows(LLVEnhancer._enhancer_registry),
        "metrics": _component_rows(BaseMetric._metric_registry),
        "losses": _component_rows(BaseLoss._loss_registry),
        "datasets": _component_rows(BaseDataset._dataset_registry),
    }


__all__ = [
    "enhance",
    "eval",
    "evaluate",
    "imread",
    "imwrite",
    "list_algorithms",
    "list_available",
    "list_datasets",
    "list_losses",
    "list_metrics",
    "list_models",
    "predict",
    "read_image",
    "train",
    "write_image",
]
