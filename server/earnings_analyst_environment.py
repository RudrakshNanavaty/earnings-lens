"""
Earnings Analyst Environment Implementation.

Samples rows from the Hugging Face earnings-call dataset and exposes task-specific
observations from ``tasks.registry.TASKS``.
"""

from __future__ import annotations

import math
import os
import json
import random
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from earnings_analyst.environment_config import DEFAULT_TASK, TASKS
from earnings_analyst.models import EarningsAnalystAction, EarningsAnalystObservation
from earnings_analyst.tasks.exceptions import TaskNotImplementedError
from earnings_analyst.tasks.registry import get_grader

from .dataset_loader import dataset


def _resolve_task_id(explicit: str | None) -> str:
    return (
        explicit or os.environ.get("EARNINGS_ANALYST_TASK_ID") or DEFAULT_TASK
    ).strip()


def _non_empty_text(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    return bool(s)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


class EarningsAnalystEnvironment(Environment):
    """
    RL environment over earnings-call rows: reset samples a row and returns
    text_context, numerical_context, and task_instruction per the active task.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str | None = None) -> None:
        self._task_id = _resolve_task_id(task_id)
        if self._task_id not in TASKS:
            raise KeyError(
                f"Unknown task_id={self._task_id!r}. Valid: {sorted(TASKS.keys())}"
            )
        self._cfg = TASKS[self._task_id]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_row: dict[str, Any] | None = None

    def reset(self) -> EarningsAnalystObservation:
        """Sample one dataset row and return the agent-visible observation bundle."""
        if not self._cfg["implemented"]:
            raise TaskNotImplementedError(
                f"Task {self._task_id!r} is not implemented yet. "
                f"Set implemented=True and fill spec/grader under tasks/ when ready."
            )

        self._state = State(episode_id=str(uuid4()), step_count=0)
        idx = random.randrange(len(dataset))
        row = dataset[idx]
        # Normalize to a plain dict for grading and column access
        self._current_row = dict(row)

        text_context = {
            col: str(self._current_row[col]).strip()
            for col in self._cfg["text_cols"]
            if _non_empty_text(self._current_row.get(col))
        }
        numerical_context: dict[str, float] = {}
        for col in self._cfg["numerical_cols"]:
            v = _finite_float(self._current_row.get(col))
            if v is not None:
                numerical_context[col] = v

        return EarningsAnalystObservation(
            text_context=text_context,
            numerical_context=numerical_context,
            task_instruction=self._cfg["task_instruction"],
            done=False,
            reward=0.0,
        )

    def step(self, action: EarningsAnalystAction) -> EarningsAnalystObservation:  # type: ignore[override]
        """
        Score the agent's string prediction against the sampled row (task-specific grader).

        Args:
            action: Agent action with ``prediction`` string.

        Returns:
            Terminal observation with reward and metadata including ground truth.
        """
        self._state.step_count += 1
        label_col = self._cfg.get("label_col", "symbol")
        label_values = list(self._cfg.get("label_values", []))
        row = self._current_row or {}

        # Handle composite ground truth if multiple columns are specified (e.g. for get_figures)
        if "xbrl_columns" in self._cfg:
            gt_data = {col: row.get(col) for col in self._cfg["xbrl_columns"]}
            ground_truth = json.dumps(gt_data)
        else:
            ground_truth = str(row.get(label_col, "")).strip()

        grade_fn = get_grader(self._task_id)
        reward = float(
            grade_fn(
                action.prediction,
                ground_truth,
                label_values,
            )
        )


        return EarningsAnalystObservation(
            text_context={},
            numerical_context={},
            task_instruction=self._cfg["task_instruction"],
            done=True,
            reward=reward,
            ground_truth=ground_truth,
            metadata={
                "task_id": self._task_id,
                "predicted": action.prediction,
            },
        )


    @property
    def state(self) -> State:
        """Current environment state."""
        return self._state
