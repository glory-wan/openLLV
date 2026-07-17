"""Base metric class and registry utilities."""

import torch
from typing import Dict, List
from abc import ABC, abstractmethod
import torch.nn.functional as F


class BaseMetric(ABC):
    """Base class for_teach image quality metrics.

    Subclasses are registered automatically and must implement
    ``_compute_impl``. The public ``compute`` method normalizes input tensor
    dimensions, moves data to the configured device, and aligns spatial sizes
    before metric computation.
    """

    _metric_registry = {}

    def __init_subclass__(cls, **kwargs):
        """Register metric subclasses automatically.

        Args:
            **kwargs: Keyword arguments forwarded to superclass initialization.
        """
        super().__init_subclass__(**kwargs)
        BaseMetric._metric_registry[cls.__name__] = cls

    def __init__(self, device=None, **kwargs):
        """Initialize a metric.

        Args:
            device: Device used for_teach metric computation. If None, CUDA is used
                when available.
            **kwargs: Metric-specific configuration parameters.
        """
        self.device = torch.device(device if device else
                                  ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.name = self.__class__.__name__.replace('Metric', '')
        self.config = kwargs

    @classmethod
    def register(cls, metic_class):
        """Register a metric class manually.

        Args:
            metic_class: Metric class to register.

        Returns:
            The registered metric class.
        """
        cls._metric_registry[metic_class.__name__] = metic_class
        return metic_class

    def _prepare_inputs(self, enImg: torch.Tensor, Refer: torch.Tensor = None) -> tuple:
        """Prepare input tensors for_teach metric computation.

        Args:
            enImg: Enhanced image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.
            Refer: Optional reference image tensor with shape ``[C, H, W]`` or
                ``[B, C, H, W]``.

        Returns:
            Tuple containing prepared enhanced and reference tensors.

        Raises:
            TypeError: If inputs are not tensors.
            ValueError: If tensors cannot be aligned to the same shape.
        """
        if not isinstance(enImg, torch.Tensor):
            raise TypeError(f"enImg must be torch.Tensor, got {type(enImg)}")

        if enImg.dim() == 3:
            enImg = enImg.unsqueeze(0)

        enImg = enImg.to(self.device)

        if Refer is not None:
            if not isinstance(Refer, torch.Tensor):
                raise TypeError(f"Refer must be torch.Tensor or None, got {type(Refer)}")

            if Refer.dim() == 3:
                Refer = Refer.unsqueeze(0)

            Refer = Refer.to(self.device)

            if enImg.shape != Refer.shape:
                if enImg.shape[2:] != Refer.shape[2:]:
                    enImg = F.interpolate(
                        enImg,
                        size=Refer.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                if enImg.shape != Refer.shape:
                    raise ValueError(
                        f"After alignment, shapes still mismatch. enImg: {enImg.shape}, Refer: {Refer.shape}"
                    )

        return enImg, Refer

    @abstractmethod
    def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor) -> float:
        """Compute a metric value from prepared tensors.

        Args:
            enImg: Prepared enhanced image tensor on the metric device.
            Refer: Optional prepared reference image tensor on the metric
                device.

        Returns:
            Metric value.
        """
        pass

    def compute(self, enImg: torch.Tensor, Refer: torch.Tensor = None) -> float:
        """Compute a metric value.

        Args:
            enImg: Enhanced image tensor.
            Refer: Optional reference image tensor. No-reference metrics may
                accept None.

        Returns:
            Metric value.
        """
        enImg, Refer = self._prepare_inputs(enImg, Refer)

        return self._compute_impl(enImg, Refer)

    @property
    def requires_reference(self) -> bool:
        """Whether the metric requires a reference image.

        Returns:
            True if a reference image is required.
        """
        return True

    @property
    def higher_is_better(self):
        """Whether larger metric values indicate better quality.

        Returns:
            True if higher values are better.
        """
        return True

    @classmethod
    def list_available_metrics(cls, simple_names: bool = True) -> List[str]:
        """List available metric names.

        Args:
            simple_names: Whether to remove the ``Metric`` suffix.

        Returns:
            List of available metric names.
        """
        if simple_names:
            return [name.replace('Metric', '') for name in cls._metric_registry.keys()]
        return list(cls._metric_registry.keys())

    @classmethod
    def create_metric(cls, metric_name: str, **kwargs) -> 'BaseMetric':
        """Create a registered metric instance.

        Args:
            metric_name: Metric name. The ``Metric`` suffix is optional and
                matching is case-insensitive.
            **kwargs: Keyword arguments passed to the metric constructor.

        Returns:
            Metric instance.

        Raises:
            ValueError: If ``metric_name`` is not registered.
        """
        if not metric_name.lower().endswith('metric'):
            metric_name = f"{metric_name}Metric"

        metric_name_lower = metric_name.lower()
        metric_mapping = {name.lower(): name for name in cls._metric_registry.keys()}

        if metric_name_lower in metric_mapping:
            actual_name = metric_mapping[metric_name_lower]
            return cls._metric_registry[actual_name](**kwargs)
        else:
            available = cls.list_available_metrics()

            from difflib import get_close_matches
            suggestions = get_close_matches(
                metric_name_lower,
                metric_mapping.keys(),
                n=3,
                cutoff=0.3
            )

            if suggestions:
                suggested_names = [name.replace('metric', '') for name in suggestions]
                error_msg = (f"Metric '{metric_name}' does not exist.\n"
                             f"Available metrics: {available}\n"
                             f"Did you mean: {', '.join(suggested_names)}")
            else:
                error_msg = (f"Metric '{metric_name}' does not exist.\n"
                             f"Available metrics: {available}")

            raise ValueError(error_msg)
