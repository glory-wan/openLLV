"""Base abstraction and registry for deep-learning loss functions."""

from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

__all__ = ["BaseLoss"]


class BaseLoss(nn.Module, ABC):
    """Base class for openLLV training losses.

    The base class provides automatic subclass registration,
    case-insensitive factory construction, and a unified ``compute`` interface
    for trainers.

    Supervised losses use the standard ``forward(prediction, target)``
    signature. Reference-free losses should set ``requires_target = False``
    and implement ``forward(input_tensor, model_output)``.
    """

    name: str = "baseloss"
    aliases: List[str] = []
    requires_target: bool = True

    _loss_registry: Dict[str, Type["BaseLoss"]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register loss subclasses automatically.

        Args:
            **kwargs: Keyword arguments forwarded to ``nn.Module`` subclass
                initialization.
        """
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "BaseLoss":
            return
        BaseLoss._register_loss(cls)

    @classmethod
    def _normalize_key(cls, name: str) -> str:
        """Return a lowercase registry key without surrounding whitespace."""
        return name.strip().lower()

    @classmethod
    def _register_loss(
        cls,
        loss_class: Type["BaseLoss"],
    ) -> Type["BaseLoss"]:
        """Register a loss class using its class name, name, and aliases.

        Args:
            loss_class: ``BaseLoss`` subclass to register.

        Returns:
            The registered loss class.

        Raises:
            TypeError: If ``loss_class`` does not inherit ``BaseLoss``.
        """
        if not issubclass(loss_class, BaseLoss):
            raise TypeError(
                f"loss_class must inherit BaseLoss, got {loss_class!r}."
            )

        candidate_names = [
            loss_class.__name__,
            getattr(loss_class, "name", loss_class.__name__),
            *getattr(loss_class, "aliases", []),
        ]

        for candidate in candidate_names:
            if isinstance(candidate, str) and candidate.strip():
                cls._loss_registry[cls._normalize_key(candidate)] = loss_class

        return loss_class

    @classmethod
    def register(
        cls,
        loss_class: Type["BaseLoss"],
    ) -> Type["BaseLoss"]:
        """Register a loss class manually and return it unchanged."""
        return cls._register_loss(loss_class)

    @classmethod
    def create_loss(cls, loss_name: str, **kwargs: Any) -> "BaseLoss":
        """Create a registered loss instance.

        Args:
            loss_name: Registered loss class name, declared name, or alias.
            **kwargs: Keyword arguments forwarded to the loss constructor.

        Returns:
            Instantiated loss object.

        Raises:
            ValueError: If ``loss_name`` is empty or is not registered.
        """
        if not isinstance(loss_name, str) or not loss_name.strip():
            raise ValueError("loss_name must be a non-empty string.")

        key = cls._normalize_key(loss_name)
        loss_class = cls._loss_registry.get(key)
        if loss_class is None:
            available = cls.list_registered_losses()
            suggestion = cls._get_similar_loss_name(key, available)
            raise ValueError(
                f"Loss '{loss_name}' is not registered.\n"
                f"Available losses: {available}\n"
                f"Did you mean: {suggestion}"
            )

        return loss_class(**kwargs)

    @classmethod
    def list_registered_losses(cls) -> List[str]:
        """Return sorted registered loss names and aliases."""
        return sorted(cls._loss_registry)

    @staticmethod
    def _get_similar_loss_name(
        loss_name: str,
        available_losses: List[str],
    ) -> str:
        """Return close registered-name suggestions for an unknown loss."""
        from difflib import get_close_matches

        suggestions = get_close_matches(
            loss_name,
            available_losses,
            n=3,
            cutoff=0.4,
        )
        return (
            ", ".join(suggestions)
            if suggestions
            else "No similar losses found"
        )

    def compute(
        self,
        *,
        input_tensor: torch.Tensor,
        model_output: Any,
        target: Optional[torch.Tensor] = None,
        extract_prediction: Optional[
            Callable[[Any, torch.Tensor], torch.Tensor]
        ] = None,
        align_prediction: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute a loss through the unified trainer-facing interface.

        Args:
            input_tensor: Model input tensor.
            model_output: Raw model output.
            target: Optional paired target tensor.
            extract_prediction: Optional callback that extracts a prediction
                tensor from a structured model output.
            align_prediction: Optional callback that aligns the prediction to
                the comparison tensor.

        Returns:
            Scalar loss tensor and the optional prediction used to compute it.

        Raises:
            ValueError: If a supervised loss is called without ``target``.
            TypeError: If a structured supervised output is provided without
                ``extract_prediction``.
        """
        if self.requires_target:
            if target is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires a target tensor."
                )

            if extract_prediction is None:
                if not torch.is_tensor(model_output):
                    raise TypeError(
                        "extract_prediction is required when model_output is "
                        "not a tensor."
                    )
                prediction = model_output
            else:
                prediction = extract_prediction(model_output, target)

            if align_prediction is not None:
                prediction = align_prediction(prediction, target)
            return self(prediction, target), prediction

        loss = self(input_tensor, model_output)
        prediction = None
        if extract_prediction is not None:
            try:
                prediction = extract_prediction(model_output, input_tensor)
                if align_prediction is not None:
                    prediction = align_prediction(prediction, input_tensor)
            except Exception:
                prediction = None
        return loss, prediction
