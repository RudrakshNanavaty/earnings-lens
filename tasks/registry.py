"""
Central registry of all tasks.

Add a new task:
1. Create ``tasks/<folder>/`` with ``spec.py``, ``grader.py``, and ``__init__.py``
   exporting ``SPEC`` and ``grade``. If the folder name is not a valid Python
   identifier (e.g. ``1_day_move``), register it via ``load_task_subpackage`` in
   this file.
2. Append ``(SPEC, grade)`` to ``_TASK_ENTRIES``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from . import next_quarter_move, sentiment_label
from .loader import load_task_subpackage
from .types import TaskSpec

GradingFn = Callable[[str, str, list[str]], float]

_pkg_1_day_move = load_task_subpackage(
    "1_day_move",
    "earnings_analyst.tasks._pkg_1_day_move",
)
_pkg_30_day_move = load_task_subpackage(
    "30_day_move",
    "earnings_analyst.tasks._pkg_30_day_move",
)

_TASK_ENTRIES: list[tuple[TaskSpec, GradingFn]] = [
    (sentiment_label.SPEC, sentiment_label.grade),
    (_pkg_1_day_move.SPEC, _pkg_1_day_move.grade),
    (_pkg_30_day_move.SPEC, _pkg_30_day_move.grade),
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
