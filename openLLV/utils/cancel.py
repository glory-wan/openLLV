"""External cancellation signal for long-running tasks."""

from __future__ import annotations

import threading

__all__ = ["CancelSignal"]


class CancelSignal:
    """Thread-safe flag to request a graceful stop of a long-running task.

    Tasks poll :meth:`is_cancelled` at their natural boundaries; once set, the
    flag stays set.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation. Safe to call from any thread."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    def __bool__(self) -> bool:
        return self._event.is_set()
