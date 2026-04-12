"""Ordinal sentiment grading: exact=1.0, adjacent label=0.5, else 0.0."""

from __future__ import annotations

from ..grading import grade_ordinal


def grade(predicted: str, ground_truth: str, label_values: list[str]) -> float:
    return grade_ordinal(predicted, ground_truth, label_values)
