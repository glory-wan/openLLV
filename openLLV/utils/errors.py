"""Exceptions shared across openLLV tasks."""

from __future__ import annotations

__all__ = ["TaskCancelled"]


class TaskCancelled(Exception):
    """Generic exception raised when a task is cancelled externally."""
