"""Task specification for ``30_day_move`` — fill in when implementing."""

from __future__ import annotations

from ..types import TaskSpec

CANONICAL_TASK_ID = "30_day_move"

SPEC: TaskSpec = {
    "task_id": CANONICAL_TASK_ID,
    "implemented": False,
    "text_cols": [],
    "numerical_cols": [],
    "label_col": "",
    "label_values": [],
    "task_instruction": "",
    "kind": "regression",
}
