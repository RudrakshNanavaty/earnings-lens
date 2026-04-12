"""Shared grading helpers for task modules."""

from __future__ import annotations


def _normalize_text(text: str) -> str:
    return str(text).strip().lower()


def _index_of_label(label: str, label_values: list[str]) -> int | None:
    normalized_target = _normalize_text(label)
    for index, candidate in enumerate(label_values):
        if _normalize_text(candidate) == normalized_target:
            return index
    return None


def grade_ordinal(
    predicted: str,
    ground_truth: str,
    label_values: list[str],
) -> float:
    """Ordinal distance reward: exact=1.0, one step=0.5, else 0.0."""
    predicted_index = _index_of_label(predicted, label_values)
    ground_truth_index = _index_of_label(ground_truth, label_values)
    if predicted_index is None or ground_truth_index is None:
        return 0.0
    ordinal_distance = abs(predicted_index - ground_truth_index)
    if ordinal_distance == 0:
        return 1.0
    if ordinal_distance == 1:
        return 0.5
    return 0.0


def grade_exact(
    predicted: str,
    ground_truth: str,
    label_values: list[str],
) -> float:
    _ = label_values
    if _normalize_text(predicted) == _normalize_text(ground_truth):
        return 1.0
    return 0.0
