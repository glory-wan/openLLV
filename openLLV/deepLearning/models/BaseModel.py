"""Base model class and registry utilities for low-level vision models."""

from __future__ import annotations

import abc
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import torch
import torch.nn as nn
import yaml


ModelType = TypeVar("ModelType", bound="LLVModel")


class LLVModel(nn.Module, abc.ABC):
    """Base class for learned low-level vision models.

    The class provides automatic registration, configuration merging, model
    construction, checkpoint serialization, checkpoint loading, and a common
    output contract. Concrete models should inherit directly from this class
    and declare their low-level vision domain through ``task``.
    """

    aliases: List[str] = []
    task: str

    _model_registry: Dict[str, Type["LLVModel"]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register concrete model subclasses automatically.

        Abstract subclasses are intentionally not registered.

        Args:
            **kwargs: Keyword arguments forwarded to ``nn.Module`` subclass
                initialization.
        """
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            LLVModel._register_model(cls)

    @classmethod
    def _normalize_key(cls, name: str) -> str:
        """Normalize a registry key for case-insensitive lookup."""
        return name.strip().lower()

    @classmethod
    def _register_model(
        cls,
        model_class: Type[ModelType],
    ) -> Type[ModelType]:
        """Register a concrete model class and its aliases.

        Args:
            model_class: Model class to register.

        Returns:
            The registered model class.

        Raises:
            TypeError: If ``model_class`` does not inherit ``LLVModel``.
            ValueError: If a registry key belongs to another model class.
        """
        if not issubclass(model_class, LLVModel):
            raise TypeError(
                f"model_class must inherit LLVModel, got {model_class!r}."
            )

        if inspect.isabstract(model_class):
            return model_class

        aliases = model_class.__dict__.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        candidate_names = [model_class.__name__, *aliases]
        for candidate in candidate_names:
            if not isinstance(candidate, str) or not candidate.strip():
                continue

            key = cls._normalize_key(candidate)
            registered_class = cls._model_registry.get(key)
            if registered_class is not None and registered_class is not model_class:
                raise ValueError(
                    f"Model registry key '{candidate}' is already registered by "
                    f"{registered_class.__name__}."
                )
            cls._model_registry[key] = model_class

        return model_class

    @classmethod
    def register(cls, model_class: Type[ModelType]) -> Type[ModelType]:
        """Register a model class manually."""
        return cls._register_model(model_class)

    @classmethod
    def list_registered_models(cls) -> List[str]:
        """Return sorted registered model names and aliases."""
        return sorted(cls._model_registry)

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a low-level vision model.

        Args:
            config: Configuration dictionary containing model parameters.
            **kwargs: Parameters that override same-named values in ``config``.
        """
        super().__init__()
        self.config = self._merge_configs(config, kwargs)
        self._validate_config()
        self._init_model()

    @abc.abstractmethod
    def _init_model(self) -> None:
        """Initialize the model architecture."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a forward pass and return a standardized output."""
        raise NotImplementedError

    def _format_output(
        self,
        pred: torch.Tensor,
        aux: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format a model output without training-component coupling.

        Args:
            pred: Final restored or enhanced image tensor.
            aux: Optional model-specific intermediate outputs.
            meta: Optional runtime or debugging metadata.

        Returns:
            Dictionary with ``pred``, ``aux``, and ``meta`` fields.
        """
        return {
            "pred": pred,
            "aux": aux or {},
            "meta": meta or {},
        }

    def _merge_configs(
        self,
        config: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge default, explicit, and keyword configuration values."""
        if config is not None and not isinstance(config, dict):
            raise TypeError(
                f"config must be a dictionary or None, got {type(config).__name__}."
            )

        merged_config = self._get_default_config().copy()
        if config:
            merged_config.update(config)
        merged_config.update(kwargs)
        return merged_config

    def _get_default_config(self) -> Dict[str, Any]:
        """Return configuration shared by low-level vision models."""
        return {
            "model_name": self.__class__.__name__,
            "input_channels": 3,
            "save_dir": f"./checkpoints/{self.task}/{self.__class__.__name__}",
        }

    def _validate_config(self) -> None:
        """Validate configuration shared by low-level vision models."""
        input_channels = self.config.get("input_channels")
        if not isinstance(input_channels, int) or input_channels <= 0:
            raise ValueError("input_channels must be a positive integer.")

    def save_model(
        self,
        save_path: Optional[Union[str, Path]] = None,
        save_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model weights and optional training state as a checkpoint."""
        output_dir = Path(save_path or self.config["save_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_state_dict = {
            key: value
            for key, value in self.state_dict().items()
            if key.rsplit(".", 1)[-1] not in {"total_ops", "total_params"}
        }
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "config": dict(self.config),
            "model_class": self.__class__.__name__,
            "model_module": self.__class__.__module__,
        }

        if save_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if metrics is not None:
            checkpoint["metrics"] = metrics

        checkpoint_path = output_dir / f"{self.__class__.__name__}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to: {checkpoint_path}")
        return checkpoint_path

    def save_config(
        self,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save the current model configuration as UTF-8 YAML."""
        output_dir = Path(save_path or self.config["save_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / f"{self.__class__.__name__}.yaml"
        with config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(
                self.config,
                config_file,
                allow_unicode=True,
                sort_keys=False,
            )
        return config_path

    @classmethod
    def load_model(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> "LLVModel":
        """Load a checkpoint on CPU and reconstruct its registered model class.

        Runtime placement is intentionally left to the trainer or caller.
        """
        if checkpoint_path is None:
            raise FileNotFoundError("checkpoint_path is required.")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file does not exist: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must contain a dictionary.")

        model_class_name = checkpoint.get("model_class")
        if not isinstance(model_class_name, str) or not model_class_name:
            raise ValueError("Checkpoint does not contain a valid model_class.")

        model_class = cls._model_registry.get(cls._normalize_key(model_class_name))
        if model_class is None:
            available_models = cls.list_registered_models()
            raise ValueError(
                f"Model class '{model_class_name}' is not registered. "
                f"Available models: {available_models}"
            )

        config = checkpoint.get("config", {})
        if not isinstance(config, dict):
            raise TypeError("Checkpoint config must be a dictionary.")
        config = dict(config)
        if config_overrides:
            config.update(config_overrides)

        model = model_class(config=config)
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain model_state_dict.")
        model.load_state_dict(state_dict, strict=strict)

        print(f"{model_class.__name__} loaded from {checkpoint_path}")
        return model

    @classmethod
    def create_model(
        cls,
        model_name: str,
        config: Any = None,
        **kwargs: Any,
    ) -> "LLVModel":
        """Create a registered model instance by class name or alias."""
        model_key = cls._normalize_key(model_name)
        model_class = cls._model_registry.get(model_key)
        if model_class is None:
            available_models = cls.list_registered_models()
            raise ValueError(
                f"Model '{model_name}' is not registered.\n"
                f"Available models: {available_models}\n"
                f"Did you mean: "
                f"{cls._get_similar_model_name(model_name, available_models)}"
            )

        if config is None:
            resolved_config: Dict[str, Any] = {}
        elif isinstance(config, dict):
            resolved_config = dict(config)
        elif isinstance(config, (str, Path)):
            resolved_config = cls.load_config(config)
        else:
            raise TypeError(
                f"Invalid config type: {type(config).__name__}. Expected None, "
                "dict, str, or Path."
            )

        return model_class(config=resolved_config, **kwargs)

    @classmethod
    def _get_similar_model_name(
        cls,
        model_name: str,
        available_models: List[str],
        max_suggestions: int = 3,
    ) -> str:
        """Return close registered-name suggestions."""
        from difflib import get_close_matches

        suggestions = get_close_matches(
            model_name,
            available_models,
            n=max_suggestions,
            cutoff=0.4,
        )
        return ", ".join(suggestions) if suggestions else "No similar models found"

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate a YAML model configuration."""
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")

        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        if config is None:
            return {}
        if not isinstance(config, dict):
            raise ValueError("Model config must be a YAML mapping.")
        return config

    def summary(self) -> None:
        """Print model, parameter, mode, and configuration details."""
        params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        lines = [
            "=" * 70,
            f"Model Name: {self.__class__.__name__}",
            "-" * 70,
            f"Mode: {'train' if self.training else 'inference'}",
            "",
            f"Total Parameters: {params:,}",
            f"Parameters (K): {params / 1e3:.2f}",
            f"Parameters (M): {params / 1e6:.2f}",
            f"Trainable Parameters: {trainable_params:,}",
            "",
            "Configuration:",
        ]

        config_items = []
        for key, value in self.config.items():
            if key in {"mode", "input_channels"}:
                continue
            if isinstance(value, (int, float, str, bool)):
                config_items.append(f"- {key}: {value}")
            elif isinstance(value, (list, tuple)) and len(str(value)) < 50:
                config_items.append(f"- {key}: {value}")

        lines.extend(config_items or ["- No additional configuration parameters"])
        lines.append("=" * 70)
        print("\n".join(lines))

    def train_mode(self) -> "LLVModel":
        """Switch the model to training mode and return it."""
        self.config["mode"] = "train"
        self.train()
        return self

    def eval_mode(self) -> "LLVModel":
        """Switch the model to evaluation mode and return it."""
        self.config["mode"] = "inference"
        self.eval()
        return self
