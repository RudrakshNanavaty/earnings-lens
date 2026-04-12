"""Grading logic for ``get_figures``."""

from __future__ import annotations
import json
import math
from typing import Any


def _safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _get_score(pred: float | None, target: float | None, tolerance: float = 0.01) -> float:
    """Compare pred to target with relative error tolerance."""
    if target is None:
        # If ground truth is null, reward 1.0 if prediction is also null, else 0.0
        return 1.0 if pred is None else 0.0
    
    if pred is None:
        return 0.0
    
    if abs(target) < 1e-9:
        return 1.0 if abs(pred) < 1e-9 else 0.0
    
    relative_error = abs(pred - target) / abs(target)
    return 1.0 if relative_error <= tolerance else 0.0


def _flatten_metrics(data: dict[str, Any]) -> dict[str, float | None]:
    """Helper to flatten the nested metrics JSON provided by the agent."""
    flat = {}
    for section in ["income_statement", "balance_sheet", "cash_flow"]:
        if section in data and isinstance(data[section], dict):
            for key, val in data[section].items():
                flat[key] = _safe_float(val)
    return flat


def grade(predicted: str, ground_truth: str, label_values: list[str]) -> float:
    """
    Score the agent's extraction performance across multiple financial metrics.
    
    Args:
        predicted: Agent's response string (expected JSON).
        ground_truth: Environment's packed JSON string of XBRL values.
        label_values: Unused.
    
    Returns:
        Average score (0.0 to 1.0) across all Metrics.
    """
    try:
        pred_data = json.loads(predicted)
        target_data = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    # Flatten the agent's nested response
    pred_metrics = _flatten_metrics(pred_data)
    
    # Environment's target_data is already flat (mapping column_name -> value)
    # We need to map the canonical keys (revenue, etc.) to the column values.
    from .spec import METRIC_TO_COLUMN
    
    scores = []
    for metric_key, col_name in METRIC_TO_COLUMN.items():
        pred_val = pred_metrics.get(metric_key)
        target_val = _safe_float(target_data.get(col_name))
        
        scores.append(_get_score(pred_val, target_val))
        
    if not scores:
        return 0.0
        
    return sum(scores) / len(scores)
