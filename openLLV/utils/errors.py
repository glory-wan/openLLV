"""Exceptions shared across openLLV tasks."""

from __future__ import annotations

from typing import Dict, Optional

__all__ = ["TaskCancelled", "EvaluateCancelled"]


class TaskCancelled(Exception):
    """Generic exception raised when a task is cancelled externally."""


class EvaluateCancelled(TaskCancelled):
    """Raised when evaluation is cancelled externally.

    Args:
        partial: Mapping of filename to value for the images of the current
            metric that were already evaluated before cancellation.
    """

    def __init__(self, *args: object, partial: Optional[Dict] = None) -> None:
        super().__init__(*args)
        self.partial = partial
