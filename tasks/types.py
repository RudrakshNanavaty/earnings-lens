"""Shared task specification types."""

from __future__ import annotations

from typing import Literal, TypedDict


class TaskSpec(TypedDict):
    """Configuration for one earnings-analyst task (per episode)."""

    task_id: str
    implemented: bool
    text_cols: list[str]
    numerical_cols: list[str]
    label_col: str
    label_values: list[str]
    task_instruction: str
    kind: Literal["classification", "regression", "other"]
