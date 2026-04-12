"""
Earnings Analyst Environment Implementation.

Samples rows from the Hugging Face earnings-call dataset and exposes task-specific
observations from environment_config.TASKS.
"""

from __future__ import annotations

import math
import random
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..environment_config import DEFAULT_TASK, TASKS
    from ..models import EarningsAnalystAction, EarningsAnalystObservation
except ImportError:
    from environment_config import DEFAULT_TASK, TASKS
    from models import EarningsAnalystAction, EarningsAnalystObservation

try:
    from .dataset_loader import dataset
except ImportError:
    from server.dataset_loader import dataset


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

    def __init__(self, task_id: str = DEFAULT_TASK) -> None:
        if task_id not in TASKS:
            raise KeyError(
                f"Unknown task_id={task_id!r}. Valid: {sorted(TASKS.keys())}"
            )
        self._cfg = TASKS[task_id]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_row: dict[str, Any] | None = None

    def reset(self) -> EarningsAnalystObservation:
        """Sample one dataset row and return the agent-visible observation bundle."""
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
        Execute one step (stub). Scoring against ``sentiment_label`` is a follow-up.

        Args:
            action: Agent action with predicted ``sentiment``.

        Returns:
            Terminal observation placeholder; reward grading not implemented yet.
        """
        self._state.step_count += 1
        return EarningsAnalystObservation(
            text_context={},
            numerical_context={},
            task_instruction=self._cfg["task_instruction"],
            done=True,
            reward=0.0,
            metadata={"predicted_sentiment": action.sentiment},
        )

    @property
    def state(self) -> State:
        """Current environment state."""
        return self._state
