"""Training utilities for descendants of :class:`LLVModel`."""

from __future__ import annotations

import importlib
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from openLLV.data import datasets as dataset_module
from openLLV.data.datasets import BaseDataset
from openLLV.deepLearning.config import (
    deep_update,
    get_config_path,
    get_default_device,
    get_default_train_config,
    load_config as load_train_config,
)
from openLLV.deepLearning.loss import BaseLoss
from openLLV.deepLearning.models import LLVModel
from openLLV.utils import device_display_name, log_info_env

__all__ = ["Trainer"]


def _collate_optional(batch: list[Any]) -> Any:
    """Collate nested batches while allowing fields that are entirely None.

    ``BaseDataset`` uses ``None`` for missing reference images. PyTorch's
    default collator rejects those values, even though reference-free losses
    do not require a target. This module-level function remains picklable when
    dataloaders use worker processes on Windows.
    """
    if not batch:
        return batch
    if all(item is None for item in batch):
        return None
    if any(item is None for item in batch):
        raise ValueError("A batch field cannot mix None and non-None values.")

    first = batch[0]
    if isinstance(first, Mapping):
        return {
            key: _collate_optional([item[key] for item in batch])
            for key in first
        }
    if isinstance(first, tuple):
        return tuple(
            _collate_optional(list(items))
            for items in zip(*batch)
        )
    if isinstance(first, list):
        return [
            _collate_optional(list(items))
            for items in zip(*batch)
        ]
    return default_collate(batch)


def _create_grad_scaler(enabled: bool):
    """Create a CUDA gradient scaler across supported PyTorch versions."""
    amp_module = getattr(torch, "amp", None)
    grad_scaler = getattr(amp_module, "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    """Configuration-driven trainer for every concrete ``LLVModel``.

    The trainer owns device placement, constructs registered models, datasets,
    losses, optimizers, and schedulers, and supports paired and reference-free
    objectives. Defaults are loaded from :mod:`openLLV.deepLearning.config`.
    """

    def __init__(
        self,
        config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a trainer.

        Args:
            config: Optional YAML path, built-in model config name, or
                configuration dictionary.
            **kwargs: Direct configuration overrides. Supported flat arguments
                are mapped into model, data, loss, optimizer, scheduler, and
                train sections.

        Raises:
            FileNotFoundError: If a config path does not exist.
            ValueError: If required model or dataset settings are missing.
            TypeError: If unsupported keyword arguments are provided.
        """
        config_input = config
        self.config_path = None
        if isinstance(config, (str, Path)):
            self.config_path = get_config_path(config)
            config_input = self.config_path

        self.user_config = deep_update({}, self._load_config(config_input))
        if kwargs:
            direct_config = self._kwargs_to_config(kwargs)
            self.user_config = deep_update(self.user_config, direct_config)
        self.config = self._with_defaults(self.user_config)
        self._validate_required_config()

        self.device = self._resolve_device(
            self.config["train"].get("device") or get_default_device()
        )

        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.history: list[Dict[str, Any]] = []
        self.training_started_at: Optional[str] = None
        self.training_ended_at: Optional[str] = None

        self._set_seed(self.config["train"].get("seed"))

        self.model = self._build_model()
        self._resolve_output_dir()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model.config["save_dir"] = str(self.output_dir)
        self.train_loader, self.val_loader = self._build_dataloaders()
        self.criterion = self._build_loss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.amp_enabled = (
            bool(self.config["train"].get("amp", False))
            and self.device.type == "cuda"
        )
        self.scaler = _create_grad_scaler(self.amp_enabled)
        self._save_training_config()

        resume = self.config["train"].get("resume")
        if resume:
            self.load_checkpoint(resume)
        log_info_env(device=self.device)
        self.print_training_info()

    @staticmethod
    def _load_config(
        config: Optional[Union[str, Path, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Load training configuration.

        Args:
            config: None, configuration dictionary, YAML path, or built-in
                model configuration name.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If a config path does not exist.
            ValueError: If the YAML file does not contain a mapping.
        """
        if config is None:
            return {}

        if isinstance(config, dict):
            return dict(config)

        if not isinstance(config, (str, Path)):
            raise TypeError(
                "config must be None, a dictionary, or a YAML path, "
                f"got {type(config)!r}."
            )

        return load_train_config(config)

    @staticmethod
    def _with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with default trainer configuration.

        Args:
            config: User configuration dictionary.

        Returns:
            Merged configuration dictionary.
        """
        merged = get_default_train_config()
        return deep_update(merged, dict(config))

    @staticmethod
    def _kwargs_to_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat Trainer keyword arguments to nested configuration.

        Args:
            kwargs: Flat keyword arguments passed to ``Trainer``.

        Returns:
            Nested configuration dictionary.

        Raises:
            TypeError: If an unsupported keyword argument is provided.
        """
        config: Dict[str, Any] = {}
        flat_map = {
            "model": ("model", "name"),
            "model_name": ("model", "name"),
            "model_params": ("model", "params"),
            "dataset": ("data", "dataset"),
            "dataset_name": ("data", "dataset"),
            "root_dir": ("data", "root_dir"),
            "batch_size": ("data", "batch_size"),
            "num_workers": ("data", "num_workers"),
            "pin_memory": ("data", "pin_memory"),
            "shuffle": ("data", "shuffle"),
            "drop_last": ("data", "drop_last"),
            "train_split": ("data", "train_split"),
            "val_split": ("data", "val_split"),
            "return_filename": ("data", "return_filename"),
            "resize": ("data", "resize"),
            "train_input_dir": ("data", "train_input_dir"),
            "train_target_dir": ("data", "train_target_dir"),
            "val_input_dir": ("data", "val_input_dir"),
            "val_target_dir": ("data", "val_target_dir"),
            "data_params": ("data", "params"),
            "train_params": ("data", "train_params"),
            "val_params": ("data", "val_params"),
            "loss": ("loss", "name"),
            "loss_name": ("loss", "name"),
            "loss_params": ("loss", "params"),
            "output_index": ("loss", "output_index"),
            "output_key": ("loss", "output_key"),
            "optimizer": ("optimizer", "name"),
            "optimizer_name": ("optimizer", "name"),
            "lr": ("optimizer", "lr"),
            "optimizer_params": ("optimizer", "params"),
            "scheduler": ("scheduler", "name"),
            "scheduler_name": ("scheduler", "name"),
            "scheduler_params": ("scheduler", "params"),
            "epochs": ("train", "epochs"),
            "output_dir": ("train", "output_dir"),
            "save_every": ("train", "save_every"),
            "validate_every": ("train", "validate_every"),
            "log_every": ("train", "log_every"),
            "grad_clip": ("train", "grad_clip"),
            "amp": ("train", "amp"),
            "resume": ("train", "resume"),
            "resume_path": ("train", "resume"),
            "strict_resume": ("train", "strict_resume"),
            "seed": ("train", "seed"),
            "device": ("train", "device"),
            "progress_bar": ("train", "progress_bar"),
        }

        for key, value in kwargs.items():
            if key in {"data", "model_config"}:
                target = "data" if key == "data" else "model"
                if not isinstance(value, dict):
                    raise TypeError(f"{key} must be a dictionary.")
                deep_update(config.setdefault(target, {}), dict(value))
                continue
            if (
                key in {"model", "loss", "optimizer", "scheduler", "train"}
                and isinstance(value, dict)
            ):
                deep_update(config.setdefault(key, {}), dict(value))
                continue
            if key not in flat_map:
                raise TypeError(f"Unsupported Trainer argument: {key}")

            section, option = flat_map[key]
            config.setdefault(section, {})[option] = value

        return config

    def _validate_required_config(self) -> None:
        """Validate required fields and common trainer hyperparameters.

        Raises:
            ValueError: If required model or dataset root settings are missing.
        """
        for section in ("model", "data", "loss", "optimizer", "scheduler", "train"):
            if not isinstance(self.config.get(section), dict):
                raise TypeError(f"config.{section} must be a dictionary.")

        model_cfg = self.config["model"]
        model_input = model_cfg.get("name")
        if model_input is None:
            raise ValueError(
                "A model is required. Pass model='ZeroDCE' or set "
                "model.name in config."
            )
        if not isinstance(model_cfg.get("params"), dict):
            raise TypeError("config.model.params must be a dictionary.")

        if isinstance(model_input, type) and not issubclass(model_input, LLVModel):
            raise TypeError("A model class must inherit LLVModel.")
        if not isinstance(model_input, (str, Path, LLVModel, type)):
            raise TypeError(
                "model.name must be a registered name, checkpoint path, "
                "LLVModel class, or LLVModel instance."
            )

        data_cfg = self.config["data"]
        dataset_input = data_cfg.get("dataset")
        if dataset_input is None:
            raise ValueError("config.data.dataset is required.")
        if (
            data_cfg.get("root_dir") is None
            and not isinstance(dataset_input, BaseDataset)
        ):
            raise ValueError(
                "A dataset root directory is required. Pass root_dir=... or "
                "set data.root_dir in config."
            )

        self._validate_positive_int(
            "data.batch_size",
            data_cfg.get("batch_size"),
        )
        self._validate_non_negative_int(
            "data.num_workers",
            data_cfg.get("num_workers"),
        )
        for key in ("params", "train_params", "val_params"):
            if not isinstance(data_cfg.get(key), dict):
                raise TypeError(f"config.data.{key} must be a dictionary.")

        train_cfg = self.config["train"]
        for key in ("epochs", "save_every", "validate_every", "log_every"):
            self._validate_positive_int(f"train.{key}", train_cfg.get(key))
        grad_clip = train_cfg.get("grad_clip")
        if grad_clip is not None and float(grad_clip) <= 0:
            raise ValueError("config.train.grad_clip must be greater than 0.")

        lr = self.config["optimizer"].get("lr")
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("config.optimizer.lr must be greater than 0.")

    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> None:
        """Validate a positive integer configuration value."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"config.{name} must be a positive integer.")

    @staticmethod
    def _validate_non_negative_int(name: str, value: Any) -> None:
        """Validate a non-negative integer configuration value."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"config.{name} must be a non-negative integer.")

    @staticmethod
    def _resolve_device(device: Union[str, torch.device]) -> torch.device:
        """Resolve and validate the trainer-owned runtime device."""
        try:
            resolved = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Invalid training device: {device!r}.") from exc

        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if resolved.type == "mps":
            backend = getattr(torch.backends, "mps", None)
            if backend is None or not backend.is_available():
                raise RuntimeError("MPS was requested but is not available.")
        return resolved

    @staticmethod
    def _safe_name(value: Any) -> str:
        """Convert a value to a filesystem-safe run name.

        Args:
            value: Value used to infer a name.

        Returns:
            Filesystem-safe name string.
        """
        if isinstance(value, (LLVModel, BaseDataset, nn.Module)):
            text = value.__class__.__name__
        elif isinstance(value, type):
            text = value.__name__
        else:
            text = str(value)
        if any(separator in text for separator in ("/", "\\")):
            text = Path(text).stem
        safe = "".join(
            character
            if character.isalnum() or character in {"-", "_"}
            else "_"
            for character in text
        )
        return safe.strip("_") or "run"

    def _resolve_output_dir(self) -> None:
        """Resolve and store the trainer output directory."""
        output_dir = self.config["train"].get("output_dir")
        if output_dir is None:
            model_name = self.model.__class__.__name__
            dataset_name = self._safe_name(self.config["data"]["dataset"])
            output_dir = Path("checkpoints") / f"{model_name}_{dataset_name}"
            self.config["train"]["output_dir"] = str(output_dir)

        self.output_dir = Path(output_dir)

    @staticmethod
    def _set_seed(seed: Optional[int] = 42) -> None:
        """Set random seeds for reproducible training.

        Args:
            seed: Optional random seed. If None, no seed is set.
        """
        if seed is None:
            return
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer or None.")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def print_training_info(self) -> None:
        """Print a compact summary of the trainer setup."""
        data_cfg = self.config["data"]

        print("=" * 70)
        print("Trainer initialized")
        print("-" * 70)
        print(f"{'Model':<20}: {self.model.__class__.__name__:<20}")
        print(f"{'Loss':<20}: {self.criterion.__class__.__name__:<20}")
        print(f"{'Training device':<20}: {device_display_name(device=self.device):<20}")
        print(f"{'Optimizer':<20}: {self.optimizer.__class__.__name__:<20} ")
        print(f"{'lr':<20}: {self.optimizer.param_groups[0]['lr']:<20}")
        scheduler_name = (
            self.scheduler.__class__.__name__
            if self.scheduler is not None
            else "None"
        )
        print(f"{'Scheduler':<20}: {scheduler_name:<20}")
        print(f"{'Dataset name':<20}: {str(data_cfg.get('dataset')):<20}")
        print(f"{'Batch size':<20}: {str(data_cfg.get('batch_size')):<20}")
        print(f"{'image resize':<20}: {str(data_cfg.get('resize')):<20}")
        print(f"{'Root directory':<20}: {str(data_cfg.get('root_dir')):<20}")
        print(f"{'Output directory':<20}: {str(self.output_dir):<20}")
        print("=" * 70)

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        """Resolve a registered model name case-insensitively.

        Args:
            model_name: Requested model name.

        Returns:
            Canonical registered model name.

        Raises:
            ValueError: If the model is not registered.
        """
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

    def _build_model(self) -> LLVModel:
        """Build or load the training model.

        Returns:
            Model instance moved to the training device.

        Raises:
            ValueError: If a model name cannot be resolved.
        """
        model_cfg = self.config["model"]
        model_input = model_cfg["name"]
        model_params = dict(model_cfg.get("params") or {})
        model_params["mode"] = "train"

        if isinstance(model_input, LLVModel):
            model = model_input
            model.config.update(model_params)
            model._validate_config()
        elif isinstance(model_input, type):
            if not issubclass(model_input, LLVModel):
                raise TypeError("The configured model class must inherit LLVModel.")
            model = model_input(config=model_params)
        else:
            model_str = str(model_input)
            if Path(model_str).is_file():
                model = LLVModel.load_model(
                    checkpoint_path=model_str,
                    config_overrides=model_params,
                    strict=bool(model_cfg.get("strict", True)),
                )
            else:
                model_name = self._resolve_model_name(model_str)
                model = LLVModel.create_model(model_name, config=model_params)

        model.to(self.device)
        model.train_mode()
        return model

    @staticmethod
    def _resolve_dataset_class(
        dataset_name: Union[str, Type[BaseDataset]],
    ) -> Type[BaseDataset]:
        """Resolve a dataset class from a registered name or import path.

        Args:
            dataset_name: Registered dataset name, class name, or dotted import
                path.

        Returns:
            Dataset class.

        Raises:
            TypeError: If a dotted dataset class does not inherit
                ``BaseDataset``.
            ValueError: If the dataset cannot be resolved.
        """
        if isinstance(dataset_name, type):
            if not issubclass(dataset_name, BaseDataset):
                raise TypeError(
                    "The configured dataset class must inherit BaseDataset."
                )
            return dataset_name
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError("dataset_name must be a non-empty string.")

        dataset_name = dataset_name.strip()
        if "." in dataset_name:
            module_name, class_name = dataset_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            dataset_cls = getattr(module, class_name)
            if not isinstance(dataset_cls, type) or not issubclass(
                dataset_cls,
                BaseDataset,
            ):
                raise TypeError(f"Dataset '{dataset_name}' must inherit BaseDataset.")
            return dataset_cls

        try:
            return BaseDataset.get_dataset_class(dataset_name)
        except ValueError:
            pass

        if hasattr(dataset_module, dataset_name):
            dataset_cls = getattr(dataset_module, dataset_name)
            if isinstance(dataset_cls, type) and issubclass(dataset_cls, BaseDataset):
                return dataset_cls

        for name in dir(dataset_module):
            if name.lower() == dataset_name.lower():
                dataset_cls = getattr(dataset_module, name)
                if isinstance(dataset_cls, type) and issubclass(
                    dataset_cls,
                    BaseDataset,
                ):
                    return dataset_cls

        available = BaseDataset.list_registered_datasets()
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {available}"
        )

    def _build_dataset(
        self,
        split_key: str,
        split_value: str,
        input_dir_key: str,
        target_dir_key: str,
    ) -> BaseDataset:
        """Build one dataset split.

        Args:
            split_key: Split prefix used for split-specific params.
            split_value: Dataset split name.
            input_dir_key: Config key for an explicit input directory.
            target_dir_key: Config key for an explicit target directory.

        Returns:
            Dataset instance.

        Raises:
            ValueError: If ``data.root_dir`` is missing.
        """
        data_cfg = self.config["data"]
        dataset_input = data_cfg["dataset"]
        if isinstance(dataset_input, BaseDataset):
            if split_key != "train":
                raise ValueError(
                    "A dataset instance can only be used as the training split. "
                    "Set data.val_split=None to disable validation."
                )
            return dataset_input

        dataset_cls = self._resolve_dataset_class(dataset_input)
        root_dir = data_cfg.get("root_dir")
        if root_dir is None:
            raise ValueError("data.root_dir is required in the training config.")

        dataset_params = dict(data_cfg.get("params", {}))
        split_params = dict(data_cfg.get(f"{split_key}_params", {}))
        dataset_kwargs = {**dataset_params, **split_params}
        dataset_kwargs.update(
            {
                "root_dir": root_dir,
                "split": split_value,
                "return_filename": data_cfg.get("return_filename", True),
            }
        )

        input_dir = data_cfg.get(input_dir_key)
        target_dir = data_cfg.get(target_dir_key)
        if input_dir is not None:
            dataset_kwargs["input_dir"] = input_dir
        if target_dir is not None:
            dataset_kwargs["target_dir"] = target_dir

        if "resize" not in dataset_kwargs and data_cfg.get("resize") is not None:
            dataset_kwargs["resize"] = data_cfg["resize"]

        dataset_kwargs.setdefault("transform_input", None)
        dataset_kwargs.setdefault("transform_target", None)
        dataset_kwargs.setdefault("common_transform", None)

        return dataset_cls(**dataset_kwargs)

    def _build_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Build train and optional validation dataloaders.

        Returns:
            Tuple containing train dataloader and optional validation dataloader.
        """
        data_cfg = self.config["data"]

        train_dataset = self._build_dataset(
            split_key="train",
            split_value=data_cfg.get("train_split", "train"),
            input_dir_key="train_input_dir",
            target_dir_key="train_target_dir",
        )

        loader_kwargs = {
            "batch_size": data_cfg["batch_size"],
            "num_workers": data_cfg["num_workers"],
            "pin_memory": bool(data_cfg.get("pin_memory", True))
            and self.device.type == "cuda",
            "collate_fn": _collate_optional,
        }

        val_loader = None
        use_val = not isinstance(data_cfg["dataset"], BaseDataset) and bool(
            data_cfg.get("val_split") or data_cfg.get("val_input_dir")
        )
        if use_val:
            val_dataset = self._build_dataset(
                split_key="val",
                split_value=data_cfg.get("val_split", "_test"),
                input_dir_key="val_input_dir",
                target_dir_key="val_target_dir",
            )
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                **loader_kwargs,
            )

        train_loader = DataLoader(
            train_dataset,
            shuffle=bool(data_cfg.get("shuffle", True)),
            drop_last=bool(data_cfg.get("drop_last", False)),
            **loader_kwargs,
        )

        return train_loader, val_loader

    def _build_loss(self) -> nn.Module:
        """Build the configured loss function.

        Returns:
            Loss module moved to the training device when it is a ``BaseLoss``.

        Raises:
            AttributeError: If a dotted loss path cannot be resolved.
            ValueError: If a registered loss name cannot be resolved.
        """
        loss_cfg = self.config["loss"]
        raw_name = loss_cfg.get("name")
        if isinstance(raw_name, nn.Module):
            return raw_name.to(self.device)
        if isinstance(raw_name, type):
            if not issubclass(raw_name, nn.Module):
                raise TypeError("The configured loss class must inherit nn.Module.")
            return raw_name(**dict(loss_cfg.get("params", {}))).to(self.device)
        if raw_name is None or str(raw_name).strip() == "":
            raw_name = self._default_loss_name_for_model()
            loss_cfg["name"] = raw_name

        if not isinstance(raw_name, str):
            raise TypeError(
                "loss.name must be a registered name, nn.Module class, or "
                "nn.Module instance."
            )

        name = raw_name.strip()
        params = dict(loss_cfg.get("params", {}))

        if "." in name:
            module_name, class_name = name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            criterion = getattr(module, class_name)(**params)
            if not isinstance(criterion, nn.Module):
                raise TypeError("The configured loss must inherit nn.Module.")
            return criterion.to(self.device)

        return BaseLoss.create_loss(name.lower(), **params).to(self.device)

    def _default_loss_name_for_model(self) -> str:
        """Infer a default loss name from the model class.

        Returns:
            Registered loss name when matched, otherwise ``"charbonnier"``.
        """
        model_name = self.model.__class__.__name__.strip()
        registered_losses = BaseLoss.list_registered_losses()
        registered_lookup = {name.lower(): name for name in registered_losses}

        candidates = [
            model_name,
            f"{model_name}_loss",
            f"{model_name}Loss",
        ]
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in registered_lookup:
                return registered_lookup[key]

        compact_model_name = "".join(
            character
            for character in model_name.lower()
            if character.isalnum()
        )
        for loss_name in registered_losses:
            compact_loss_name = "".join(
                character
                for character in loss_name.lower()
                if character.isalnum()
            )
            if compact_loss_name in {
                compact_model_name,
                f"{compact_model_name}loss",
            }:
                return loss_name

        return "charbonnier"

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the configured optimizer for trainable model parameters."""
        optimizer_cfg = self.config["optimizer"]
        optimizer_input = optimizer_cfg.get("name", "adam")
        trainable_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_params:
            raise ValueError("The model has no trainable parameters.")

        if isinstance(optimizer_input, torch.optim.Optimizer):
            return optimizer_input

        params = dict(optimizer_cfg.get("params", {}))
        params.setdefault("lr", optimizer_cfg.get("lr", 1e-4))
        if isinstance(optimizer_input, type):
            if not issubclass(optimizer_input, torch.optim.Optimizer):
                raise TypeError(
                    "The configured optimizer class must inherit Optimizer."
                )
            return optimizer_input(trainable_params, **params)
        if not isinstance(optimizer_input, str):
            raise TypeError(
                "optimizer.name must be a supported name, Optimizer class, "
                "or Optimizer instance."
            )

        optimizer_classes = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        name = optimizer_input.strip().lower()
        optimizer_class = optimizer_classes.get(name)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer '{optimizer_input}'.")
        return optimizer_class(trainable_params, **params)

    def _build_scheduler(self) -> Optional[Any]:
        """Build the configured learning-rate scheduler, when enabled."""
        sched_cfg = self.config["scheduler"]
        name = sched_cfg.get("name")
        if name is None or name == "":
            return None

        if not isinstance(name, str):
            if hasattr(name, "step") and hasattr(name, "state_dict"):
                return name
            raise TypeError(
                "scheduler.name must be a supported name or scheduler instance."
            )

        params = dict(sched_cfg.get("params", {}))
        name = name.strip().lower()

        if name == "steplr":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, **params)
        if name == "multisteplr":
            return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **params)
        if name == "cosineannealinglr":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **params)
        if name == "reducelronplateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **params)

        raise ValueError(f"Unsupported scheduler '{sched_cfg.get('name')}'.")

    def _move_batch_to_device(
        self,
        batch: Any,
        require_target: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Any]:
        """Move a dataloader batch to the training device.

        Args:
            batch: Batch returned by a dataset.
            require_target: Whether a paired target is required.

        Returns:
            Tuple containing input tensor, optional target tensor, and
            optional filenames.

        Raises:
            ValueError: If the batch format is unsupported or a required target
                is missing.
        """
        if isinstance(batch, Mapping):
            input_tensor = self._first_mapping_value(
                batch,
                ("input", "image", "source"),
            )
            target_tensor = self._first_mapping_value(
                batch,
                ("target", "reference", "gt"),
                required=False,
            )
            filenames = self._first_mapping_value(
                batch,
                ("filename", "filenames", "name", "path"),
                required=False,
            )
        elif torch.is_tensor(batch):
            input_tensor, target_tensor, filenames = batch, None, None
        elif isinstance(batch, (tuple, list)) and len(batch) == 3:
            input_tensor, target_tensor, filenames = batch
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            input_tensor, second = batch
            if torch.is_tensor(second) or second is None:
                target_tensor, filenames = second, None
            else:
                target_tensor, filenames = None, second
        elif isinstance(batch, (tuple, list)) and len(batch) == 1:
            input_tensor, target_tensor, filenames = batch[0], None, None
        else:
            raise ValueError(
                f"Unsupported batch format: {type(batch).__name__}."
            )

        if require_target and target_tensor is None:
            raise ValueError(
                "Supervised training requires paired target images."
            )

        if not torch.is_tensor(input_tensor):
            raise TypeError(
                f"Batch input must be a tensor, got {type(input_tensor)!r}."
            )
        if target_tensor is not None and not torch.is_tensor(target_tensor):
            raise TypeError(
                "Batch target must be a tensor or None, got "
                f"{type(target_tensor)!r}."
            )
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        if target_tensor is not None:
            target_tensor = target_tensor.to(
                self.device,
                non_blocking=True,
            )

        return input_tensor, target_tensor, filenames

    @staticmethod
    def _first_mapping_value(
        mapping: Mapping[str, Any],
        keys: Tuple[str, ...],
        *,
        required: bool = True,
    ) -> Any:
        """Return the first present batch value from a list of key aliases."""
        for key in keys:
            if key in mapping:
                return mapping[key]
        if required:
            raise KeyError(
                f"Batch mapping is missing one of the required keys: {keys}."
            )
        return None

    def _extract_prediction(self, output: Any, target: torch.Tensor) -> torch.Tensor:
        """Extract a prediction tensor from a model output.

        Args:
            output: Raw model output.
            target: Target tensor used to select a compatible prediction.

        Returns:
            Prediction tensor.

        Raises:
            KeyError: If a configured ``loss.output_key`` is missing.
            TypeError: If no prediction tensor can be extracted.
        """
        loss_cfg = self.config["loss"]
        output_key = loss_cfg.get("output_key")
        output_index = loss_cfg.get("output_index")

        if output_key is not None and isinstance(output, dict):
            if output_key not in output:
                raise KeyError(
                    f"Configured loss.output_key '{output_key}' was not found "
                    "in the model output."
                )
            selected = output[output_key]
            if torch.is_tensor(selected):
                return selected
            tensors = self._flatten_tensors(selected)
            if tensors:
                return self._select_tensor(tensors, target)

        if output_index is not None and isinstance(output, (tuple, list)):
            selected = output[int(output_index)]
            if torch.is_tensor(selected):
                return selected
            tensors = self._flatten_tensors(selected)
            if tensors:
                return self._select_tensor(tensors, target)

        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            for key in (
                "pred",
                "enhanced",
                "output",
                "prediction",
                "result",
                "image",
            ):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
            tensors = [value for value in output.values() if torch.is_tensor(value)]
            if tensors:
                return self._select_tensor(tensors, target)

        if isinstance(output, (tuple, list)):
            if self._looks_like_grouped_training_output(output):
                tensors = self._flatten_tensors(output[1])
                if tensors:
                    return self._select_tensor(tensors, target)

            tensors = self._flatten_tensors(output)
            if tensors:
                return self._select_tensor(tensors, target)

        raise TypeError(
            "Cannot extract a prediction tensor from model output type: "
            f"{type(output).__name__}."
        )

    @staticmethod
    def _looks_like_grouped_training_output(output: Any) -> bool:
        """Check whether output looks like grouped training output.

        Args:
            output: Raw model output.

        Returns:
            True if the output matches the grouped tuple/list pattern.
        """
        return (
            isinstance(output, (tuple, list))
            and len(output) >= 2
            and isinstance(output[0], (tuple, list))
            and isinstance(output[1], (tuple, list))
            and any(torch.is_tensor(item) for item in output[1])
        )

    def _flatten_tensors(self, value: Any):
        """Collect tensors from nested model output structures.

        Args:
            value: Tensor, dictionary, tuple, list, or arbitrary object.

        Returns:
            Flat list of tensors found in ``value``.
        """
        tensors = []
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, dict):
            for item in value.values():
                tensors.extend(self._flatten_tensors(item))
        elif isinstance(value, (tuple, list)):
            for item in value:
                tensors.extend(self._flatten_tensors(item))
        return tensors

    @staticmethod
    def _select_tensor(tensors, target: torch.Tensor) -> torch.Tensor:
        """Select the prediction tensor most compatible with a target.

        Args:
            tensors: Candidate tensors.
            target: Target tensor.

        Returns:
            Selected tensor.
        """
        for tensor in reversed(tensors):
            if (
                tensor.dim() == target.dim()
                and tensor.shape[:2] == target.shape[:2]
            ):
                return tensor
        for tensor in reversed(tensors):
            if tensor.dim() == target.dim():
                return tensor
        return tensors[-1]

    def _align_prediction(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Align prediction tensor spatial size and channel count to target.

        Args:
            prediction: Prediction tensor.
            target: Target tensor.

        Returns:
            Aligned prediction tensor.

        Raises:
            ValueError: If prediction and target channel counts differ.
        """
        if prediction.ndim != target.ndim:
            raise ValueError(
                "Prediction and target dimensions do not match: "
                f"{prediction.ndim} vs {target.ndim}."
            )
        if prediction.ndim != 4:
            raise ValueError(
                "Image prediction and target tensors must have shape "
                "[N, C, H, W]."
            )
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if prediction.shape[1] != target.shape[1]:
            raise ValueError(
                f"Prediction and target channel counts do not match: "
                f"{prediction.shape[1]} vs {target.shape[1]}"
            )
        return prediction

    def _compute_loss(
        self,
        output: Any,
        target: Optional[torch.Tensor],
        input_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute loss from a raw model output.

        Args:
            output: Raw model output.
            target: Optional paired target tensor.
            input_tensor: Input tensor required by ``BaseLoss``.

        Returns:
            Tuple containing scalar loss tensor and optional prediction tensor.

        Raises:
            ValueError: If required input or target tensors are missing.
        """
        if isinstance(self.criterion, BaseLoss):
            if input_tensor is None:
                raise ValueError("input_tensor is required when using BaseLoss.")

            return self.criterion.compute(
                input_tensor=input_tensor,
                model_output=output,
                target=target,
                extract_prediction=self._extract_prediction,
                align_prediction=self._align_prediction,
            )

        if target is None:
            raise ValueError(
                f"{self.criterion.__class__.__name__} requires a target tensor."
            )
        prediction = self._extract_prediction(output, target)
        prediction = self._align_prediction(prediction, target)
        return self.criterion(prediction, target), prediction

    def _compute_batch_loss(
        self,
        batch: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one batch forward pass and compute its loss.

        Args:
            batch: Batch returned by the dataloader.

        Returns:
            Tuple containing scalar loss tensor and optional prediction tensor.
        """
        requires_target = (
            not isinstance(self.criterion, BaseLoss)
            or self.criterion.requires_target
        )
        input_tensor, target_tensor, _ = self._move_batch_to_device(
            batch,
            require_target=requires_target,
        )

        original_mode = None
        had_mode = False
        paired_forward = bool(getattr(self.model, "requires_paired_forward", False))
        needs_training_output = isinstance(self.criterion, BaseLoss)
        if needs_training_output and hasattr(self.model, "config"):
            had_mode = "mode" in self.model.config
            original_mode = self.model.config.get("mode")
            self.model.config["mode"] = "train"

        try:
            if paired_forward:
                output = self.model(
                    input_tensor,
                    paired_image=target_tensor,
                )
            else:
                output = self.model(input_tensor)
        finally:
            if had_mode:
                self.model.config["mode"] = original_mode
            elif needs_training_output and hasattr(self.model, "config"):
                self.model.config.pop("mode", None)

        loss, prediction = self._compute_loss(
            output,
            target_tensor,
            input_tensor=input_tensor,
        )
        if not torch.is_tensor(loss):
            raise TypeError(
                f"Loss must be a torch.Tensor, got {type(loss)!r}."
            )
        if loss.numel() != 1:
            loss = loss.mean()
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError("The computed loss is NaN or infinite.")
        return loss, prediction

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary containing training history, best validation loss, and
            checkpoint directory.
        """
        epochs = int(self.config["train"]["epochs"])
        if self.training_started_at is None:
            self.training_started_at = datetime.now().isoformat(timespec="seconds")
            self.training_ended_at = None
            self._save_training_config()
            print(f"begin Training at {self.training_started_at}")

        try:
            for epoch in range(self.start_epoch, epochs + 1):
                print('')
                epoch_start = time.time()
                train_loss = self.train_one_epoch(epoch)

                val_loss = None
                if (
                    self.val_loader is not None
                    and epoch
                    % int(self.config["train"]["validate_every"])
                    == 0
                ):
                    val_loss = self.validate(epoch)

                self._step_scheduler(val_loss, train_loss)

                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "seconds": time.time() - epoch_start,
                }
                self.history.append(record)
                self._save_history()

                is_best = val_loss is not None and val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                if epoch % int(self.config["train"]["save_every"]) == 0:
                    self.save_checkpoint("last.pt", epoch, val_loss)

                if is_best:
                    self.save_checkpoint("best.pt", epoch, val_loss)

                print(
                    f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} "
                    f"| val_loss={val_loss:.6f}" if val_loss is not None
                    else f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f}"
                )
        finally:
            self.training_ended_at = datetime.now().isoformat(timespec="seconds")
            self._save_training_config()

        print(
            "\n"
            f"Finished training at {self.training_ended_at}.\n"
            f"Training results saved at {self.output_dir}\n"
        )

        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "checkpoint_dir": str(self.checkpoint_dir),
        }

    def train_one_epoch(self, epoch: int) -> float:
        """Train the model for one epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train_mode()
        total_loss = 0.0
        total_samples = 0
        log_every = int(self.config["train"]["log_every"])
        grad_clip = self.config["train"].get("grad_clip")

        pbar = tqdm(
            self.train_loader,
            desc=f"Train {epoch}",
            unit="batch",
            disable=not bool(self.config["train"].get("progress_bar", True)),
        )
        for step, batch in enumerate(pbar, start=1):
            batch_size = self._batch_size(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                enabled=self.amp_enabled,
            ):
                loss, _ = self._compute_batch_loss(batch)

            self.scaler.scale(loss).backward()

            if grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(grad_clip),
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if step % log_every == 0 or step == 1:
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
                pbar.set_postfix(postfix)

        if total_samples == 0:
            raise RuntimeError(
                "The training dataloader produced no samples. Check "
                "drop_last, batch_size, and dataset contents."
            )
        return total_loss / total_samples

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Run validation for one epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Mean validation loss.
        """
        self.model.eval_mode()
        total_loss = 0.0
        total_samples = 0

        if self.val_loader is None:
            raise RuntimeError("Validation was requested without a val_loader.")

        pbar = tqdm(
            self.val_loader,
            desc=f"Val {epoch}",
            unit="batch",
            disable=not bool(self.config["train"].get("progress_bar", True)),
        )
        for step, batch in enumerate(pbar, start=1):
            loss, _ = self._compute_batch_loss(batch)

            batch_size = self._batch_size(batch)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            postfix = {"loss": f"{loss.item():.6f}"}
            pbar.set_postfix(postfix)

        if total_samples == 0:
            raise RuntimeError("The validation dataloader produced no samples.")
        return total_loss / total_samples

    def _batch_size(self, batch: Any) -> int:
        """Infer batch size from any supported batch representation."""
        if isinstance(batch, Mapping):
            tensor = self._first_mapping_value(
                batch,
                ("input", "image", "source"),
            )
        elif torch.is_tensor(batch):
            tensor = batch
        elif isinstance(batch, (tuple, list)) and batch:
            tensor = batch[0]
        else:
            raise ValueError(
                f"Cannot infer batch size from {type(batch).__name__}."
            )
        if not torch.is_tensor(tensor) or tensor.ndim == 0:
            raise TypeError("The batch input must be a non-scalar tensor.")
        return int(tensor.shape[0])

    def _step_scheduler(self, val_loss: Optional[float], train_loss: float) -> None:
        """Advance the configured learning-rate scheduler.

        Args:
            val_loss: Optional validation loss.
            train_loss: Training loss for the current epoch.
        """
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric = val_loss if val_loss is not None else train_loss
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_loss: Optional[float] = None,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            filename: Checkpoint filename.
            epoch: Current epoch index.
            val_loss: Optional validation loss.

        Returns:
            Path to the saved checkpoint.
        """
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string.")
        save_path = self.checkpoint_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": {
                key: value
                for key, value in self.model.state_dict().items()
                if key.rsplit(".", 1)[-1]
                not in {"total_ops", "total_params"}
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": dict(self.model.config),
            "trainer_config": self._to_yaml_safe(self.config),
            "model_class": self.model.__class__.__name__,
            "model_module": self.model.__class__.__module__,
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "history": list(self.history),
            "training_started_at": self.training_started_at,
            "training_ended_at": self.training_ended_at,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler.is_enabled():
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        self.model.save_config(save_path=self.output_dir)
        torch.save(checkpoint, save_path)
        return save_path

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load a training checkpoint.

        Args:
            checkpoint_path: Checkpoint file path.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Training checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise TypeError("Training checkpoint must contain a dictionary.")

        checkpoint_model = checkpoint.get("model_class")
        if (
            checkpoint_model is not None
            and checkpoint_model != self.model.__class__.__name__
        ):
            raise ValueError(
                f"Checkpoint model {checkpoint_model!r} does not match "
                f"{self.model.__class__.__name__!r}."
            )
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain model_state_dict.")
        self.model.load_state_dict(
            state_dict,
            strict=bool(self.config["train"].get("strict_resume", True)),
        )

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler.is_enabled() and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        history = checkpoint.get("history", [])
        if isinstance(history, list):
            self.history = history
        self.training_started_at = checkpoint.get(
            "training_started_at",
            self.training_started_at,
        )
        print(
            f"Resumed training from {checkpoint_path}, "
            f"start_epoch={self.start_epoch}"
        )

    def _save_history(self) -> None:
        """Save training history as JSON."""
        history_path = self.log_dir / "history.json"
        with history_path.open("w", encoding="utf-8") as history_file:
            json.dump(
                self.history,
                history_file,
                indent=2,
                ensure_ascii=False,
            )

    def _save_training_config(self) -> Path:
        """Save the current training configuration as YAML.

        Returns:
            Path to the saved configuration file.
        """
        model_name = (
            self.model.__class__.__name__
            if hasattr(self, "model")
            else self._safe_name(self.config["model"].get("name", "model"))
        )
        config_path = self.output_dir / f"{model_name}.yaml"
        payload = {
            "model_name": model_name,
            "dataset_name": self._safe_name(
                self.config["data"].get("dataset")
            ),
            "training_started_at": self.training_started_at,
            "training_ended_at": self.training_ended_at,
            "config": self._to_yaml_safe(self.config),
        }
        with config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(
                payload,
                config_file,
                allow_unicode=True,
                sort_keys=False,
            )
        return config_path

    @classmethod
    def _to_yaml_safe(cls, value: Any) -> Any:
        """Convert values to YAML-serializable objects.

        Args:
            value: Value to convert.

        Returns:
            YAML-safe representation of ``value``.
        """
        if isinstance(value, Mapping):
            return {str(k): cls._to_yaml_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_yaml_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, (LLVModel, BaseDataset, nn.Module)):
            return value.__class__.__name__
        return str(value)
