"""Grading logic for ``1_day_move``."""

from __future__ import annotations
from ..grading import grade_smart_move


def grade(predicted: str, ground_truth: str, label_values: list[str]) -> float:
    """
    Score the agent's prediction using the smart reward (Directional + Numerical).
    This logic is handled centrally in grading.py to ensure consistency across movement tasks.
    """
    return grade_smart_move(predicted, ground_truth, label_values)
