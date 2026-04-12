"""
Central registry of all tasks.

Add a new task:
1. Create ``tasks/<folder>/`` with ``spec.py``, ``grader.py``, and ``__init__.py``
   exporting ``SPEC`` and ``grade``.
2. Import the package below and append ``(SPEC, grade)`` to ``_TASK_ENTRIES``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from . import (
    get_figures,
    sentiment_label,
    one_day_move,
    thirty_day_move,
    next_quarter_move,
)
from .types import TaskSpec

GradingFn = Callable[[str, str, list[str]], float]

_TASK_ENTRIES: list[tuple[TaskSpec, GradingFn]] = [
    (get_figures.SPEC, get_figures.grade),
    (sentiment_label.SPEC, sentiment_label.grade),
    (one_day_move.SPEC, one_day_move.grade),
    (thirty_day_move.SPEC, thirty_day_move.grade),
    (next_quarter_move.SPEC, next_quarter_move.grade),
]


TASKS: dict[str, TaskSpec] = {spec["task_id"]: spec for spec, _ in _TASK_ENTRIES}
GRADERS: dict[str, GradingFn] = {spec["task_id"]: fn for spec, fn in _TASK_ENTRIES}

DEFAULT_TASK: Final[str] = "sentiment_label"

TASK_IDS: tuple[str, ...] = tuple(sorted(TASKS.keys()))


def get_task_spec(task_id: str) -> TaskSpec:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id={task_id!r}. Valid: {list(TASKS.keys())}")
    return TASKS[task_id]


def get_grader(task_id: str) -> GradingFn:
    if task_id not in GRADERS:
        raise KeyError(f"Unknown task_id={task_id!r}. Valid: {list(GRADERS.keys())}")
    return GRADERS[task_id]
