"""Per-task packages live under ``earnings_analyst.tasks.<name>``."""

from __future__ import annotations

from .exceptions import TaskNotImplementedError
# Import the registry from ``earnings_analyst.tasks.registry`` (not from this package).


__all__ = [
    "TaskNotImplementedError",
]

