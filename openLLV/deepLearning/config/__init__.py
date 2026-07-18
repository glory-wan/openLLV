"""Built-in training configurations and configuration helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import yaml


CONFIG_DIR = Path(__file__).resolve().parent


def get_default_device() -> str:
    """Return the best available default training device."""
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "model": {
        "name": None,
        "params": {},
    },
    "data": {
        "dataset": "CommonDataset",
        "root_dir": None,
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": True,
        "shuffle": True,
        "drop_last": False,
        "train_split": "train",
        "val_split": "val",
        "return_filename": True,
        "resize": None,
        "params": {},
        "train_params": {},
        "val_params": {},
    },
    "loss": {
        "name": None,
        "params": {},
        "output_index": None,
        "output_key": None,
    },
    "optimizer": {
        "name": "adam",
        "lr": 1e-4,
        "params": {},
    },
    "scheduler": {
        "name": None,
        "params": {},
    },
    "train": {
        "epochs": 100,
        "output_dir": None,
        "save_every": 1,
        "validate_every": 1,
        "log_every": 10,
        "grad_clip": None,
        "amp": False,
        "resume": None,
        "strict_resume": True,
        "seed": 42,
        "device": get_default_device(),
        "progress_bar": True,
    },
}


def get_default_train_config() -> Dict[str, Any]:
    """Return an independent copy of the default trainer configuration."""
    return copy.deepcopy(DEFAULT_TRAIN_CONFIG)


def deep_update(
    base: Dict[str, Any],
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return ``base``."""
    if not isinstance(base, dict) or not isinstance(updates, dict):
        raise TypeError("base and updates must both be dictionaries.")

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = _copy_config_value(value)
    return base


def list_available_configs() -> List[str]:
    """Return the names of all built-in YAML training configurations."""
    return sorted(
        (path.stem for path in CONFIG_DIR.glob("*.yaml")),
        key=str.casefold,
    )


def get_config_path(config: Union[str, Path]) -> Path:
    """Resolve a filesystem path or built-in model configuration name.

    Built-in configurations can be selected by YAML filename, filename stem,
    or the ``model.name`` value declared inside the YAML file. Matching is
    case-insensitive and ignores punctuation, so names such as
    ``"ZeroDCEPlusPlus"`` resolve ``ZeroDCE++.yaml``.

    Args:
        config: Existing YAML path or built-in configuration name.

    Returns:
        Resolved YAML path.

    Raises:
        TypeError: If ``config`` is not a string or path.
        ValueError: If ``config`` is empty.
        FileNotFoundError: If no existing or built-in config matches.
    """
    if not isinstance(config, (str, Path)):
        raise TypeError(
            f"config must be a string or Path, got {type(config).__name__}."
        )

    config_text = str(config).strip()
    if not config_text:
        raise ValueError("config must not be empty.")

    candidate = Path(config_text).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    if candidate.parent != Path("."):
        raise FileNotFoundError(
            f"Training config file does not exist: {candidate}"
        )

    requested_name = (
        candidate.stem
        if candidate.suffix.lower() in {".yaml", ".yml"}
        else candidate.name
    )
    requested_key = _normalize_config_name(requested_name)

    matches = []
    for config_path in CONFIG_DIR.glob("*.yaml"):
        names = {config_path.stem}
        loaded = _load_yaml_path(config_path)
        model_section = loaded.get("model", {})
        if isinstance(model_section, dict):
            model_name = model_section.get("name")
            if isinstance(model_name, str) and model_name.strip():
                names.add(model_name)

        if requested_key in {_normalize_config_name(name) for name in names}:
            matches.append(config_path)

    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) == 1:
        return unique_matches[0].resolve()
    if len(unique_matches) > 1:
        raise ValueError(
            f"Built-in config name '{config}' is ambiguous: "
            f"{[path.name for path in unique_matches]}"
        )

    raise FileNotFoundError(
        f"Training config file does not exist and no built-in config matches "
        f"'{config}'. Available configs: {list_available_configs()}"
    )


def load_config(config: Union[str, Path]) -> Dict[str, Any]:
    """Load an existing or built-in YAML training configuration."""
    return _load_yaml_path(get_config_path(config))


def _load_yaml_path(config_path: Path) -> Dict[str, Any]:
    """Load one YAML path and require a mapping at its root."""
    with config_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Training config must be a YAML mapping: {config_path}"
        )
    return loaded


def _normalize_config_name(name: str) -> str:
    """Normalize config and model names for punctuation-insensitive lookup."""
    normalized = name.casefold().replace("++", "plusplus").replace("+", "plus")
    return "".join(
        character for character in normalized if character.isalnum()
    )


def _copy_config_value(value: Any) -> Any:
    """Copy configuration containers while preserving runtime objects."""
    if isinstance(value, dict):
        return {
            key: _copy_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_config_value(item) for item in value)
    if isinstance(value, set):
        return {_copy_config_value(item) for item in value}
    return value


__all__ = [
    "CONFIG_DIR",
    "DEFAULT_TRAIN_CONFIG",
    "deep_update",
    "get_config_path",
    "get_default_device",
    "get_default_train_config",
    "list_available_configs",
    "load_config",
]
