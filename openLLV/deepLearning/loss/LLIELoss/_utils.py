"""Shared helpers for low-light model loss adapters."""

from __future__ import annotations

from typing import Any, Dict


def get_loss_inputs(model_output: Any) -> Dict[str, Any]:
    """Return structured training data from current or legacy model output.

    openLLV stores model-specific intermediate values under ``aux``. The
    original LibLLIE trainer used ``loss_inputs`` instead, so migrated losses
    accept both layouts while preferring the legacy field when both exist.
    """
    if not isinstance(model_output, dict):
        return {}

    for key in ("loss_inputs", "aux"):
        value = model_output.get(key)
        if isinstance(value, dict):
            return value
    return {}


def has_loss_inputs(model_output: Any) -> bool:
    """Return whether a model output contains a structured training mapping."""
    return isinstance(model_output, dict) and any(
        isinstance(model_output.get(key), dict)
        for key in ("loss_inputs", "aux")
    )


__all__ = ["get_loss_inputs", "has_loss_inputs"]
