"""
Project-level dataset location.

Task definitions live in ``tasks/<name>/`` and are registered in ``earnings_analyst.tasks.registry``.
"""

from __future__ import annotations

from earnings_analyst.tasks.registry import DEFAULT_TASK, TASKS

DATASET_ID = "RudrakshNanavaty/earnings-call-data"
DATASET_FILE = "episodes_press_release_8k.parquet"

# Columns for Gradio episode selection (must exist in the parquet).
DATASET_EPISODE_ID_COLUMN = "episode_id"
DATASET_SYMBOL_COLUMN = "symbol"
DATASET_COMPANY_NAME_COLUMN = "company_name"
DATASET_YEAR_COLUMN = "year"
DATASET_QUARTER_COLUMN = "quarter"

__all__ = [
    "DATASET_ID",
    "DATASET_FILE",
    "DATASET_EPISODE_ID_COLUMN",
    "DATASET_SYMBOL_COLUMN",
    "DATASET_COMPANY_NAME_COLUMN",
    "DATASET_YEAR_COLUMN",
    "DATASET_QUARTER_COLUMN",
    "DEFAULT_TASK",
    "TASKS",
]
