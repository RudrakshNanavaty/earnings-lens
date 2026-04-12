"""
Project-level dataset location.

Task definitions live in ``tasks/<name>/`` and are registered in ``tasks.registry``.
"""

from __future__ import annotations

try:
    from earnings_analyst.tasks.registry import DEFAULT_TASK, TASKS
except ImportError:
    from tasks.registry import DEFAULT_TASK, TASKS

DATASET_ID = "RudrakshNanavaty/earnings-call-data"
DATASET_FILE = "episodes_press_release_8k.parquet"

__all__ = ["DATASET_ID", "DATASET_FILE", "DEFAULT_TASK", "TASKS"]
