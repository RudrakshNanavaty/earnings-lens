"""Grading for ``next_quarter_move`` (regression)."""

from __future__ import annotations
import json
import re

from ..grading import grade_regression


def grade(predicted: str, ground_truth: str, label_values: list[str]) -> float:
    """
    Parses predicted string for a 'move' key or a numeric value,
    then grades against ground_truth via exponential decay.
    """
    _ = label_values
    
    # Try to extract number from JSON if possible
    pred_val_str = predicted
    try:
        data = json.loads(predicted)
        if isinstance(data, dict) and "move" in data:
            pred_val_str = str(data["move"])
    except (json.JSONDecodeError, TypeError):
        # Fallback: find the first float-like thing in the string
        match = re.search(r"[-+]?\d*\.\d+|\d+", predicted)
        if match:
            pred_val_str = match.group()

    return grade_regression(pred_val_str, ground_truth, scale=0.1)
