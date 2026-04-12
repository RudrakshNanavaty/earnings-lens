"""Task specification for ``get_figures`` (Financial figure extraction)."""

from __future__ import annotations
from ..types import TaskSpec

CANONICAL_TASK_ID = "get_figures"

SPEC: TaskSpec = {
    "task_id": CANONICAL_TASK_ID,
    "implemented": False,  # Safety gate: set to True once ground truth column is confirmed.
    "text_cols": [
        "earnings_transcript",
        "press_release_8k_body",
        "press_release_ex991",
        "press_release_ex992",
        "press_release_sources",
    ],
    "numerical_cols": [],
    "label_col": "symbol",  # Placeholder
    "label_values": [],
    "task_instruction": (
        "Extract key financial figures from the provided earnings call materials.\n\n"
        "Return a JSON object matching this exact schema:\n"
        '{"revenue": <float>, "net_income": <float>, "eps": <float>}\n\n'
        "Use the currency specified in the documents. If a figure is not found, use null.\n"
        "Do not include any other keys or explanation."
    ),
    "kind": "other",
}
