"""Shared grading helpers for task modules."""

from __future__ import annotations
import json
import math


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


def grade_smart_move(
    predicted: str,
    ground_truth: str,
    label_values: list[str],
) -> float:
    """
    Smarter reward for price movement tasks:
    - 40%: Directional Accuracy (Did you get the sign right?)
    - 60%: Numerical Proximity (How close is the percentage?)
    """
    try:
        # 0. Parsing
        actual_move = float(ground_truth)
        
        # Try to parse as JSON first
        predicted_percent = 0.0
        predicted_label = predicted
        if predicted.strip().startswith("{"):
            data = json.loads(predicted)
            predicted_percent = float(data.get("percentage_move", 0.0))
            predicted_label = str(data.get("label", "neutral"))
        else:
            # Fallback: try to extract a float from the string if it's not JSON
            try:
                predicted_percent = float(predicted)
            except ValueError:
                predicted_percent = 0.0
        
        # 1. Directional Accuracy (40%)
        # sign(x) is 1 for positive, -1 for negative, 0 for zero
        actual_sign = 0 if abs(actual_move) < 1e-4 else (1 if actual_move > 0 else -1)
        # For simplicity, we compare signs of the numeric percentage if available
        predicted_sign = 0 if abs(predicted_percent) < 1e-4 else (1 if predicted_percent > 0 else -1)
        
        directional_reward = 1.0 if actual_sign == predicted_sign else (0.5 if actual_sign == 0 or predicted_sign == 0 else 0.0)
        
        # 2. Numerical Proximity (60%)
        # Using exponential decay: exp(-k * error)
        # Scale k: error of 10% (0.1) results in exp(-1.0) ~ 0.36
        k = 10.0
        abs_error = abs(predicted_percent - actual_move)
        numerical_reward = math.exp(-k * abs_error)
        
        # 3. Weighted Combination
        return 0.4 * directional_reward + 0.6 * numerical_reward

    except (json.JSONDecodeError, ValueError, TypeError):
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


def grade_regression(
    predicted: str,
    ground_truth: str,
    scale: float = 0.1,
) -> float:
    """
    Score a numerical prediction: exp(-abs(pred - gt) / scale).
    Returns 1.0 for exact, decaying towards 0.0.
    """
    import math

    try:
        # Ground truth is passed as str(float) from the environment
        gt_val = float(ground_truth)
    except (ValueError, TypeError):
        return 0.0

    # Try to parse predicted as a pure number if it's not JSON
    # (Though usually the task asks for JSON)
    try:
        pred_val = float(predicted)
    except (ValueError, TypeError):
        # Fallback: try to find a number in the string or just return 0
        return 0.0

    error = abs(pred_val - gt_val)
    return math.exp(-error / scale)
