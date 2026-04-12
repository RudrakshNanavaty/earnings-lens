"""Task specification for ``next_quarter_move`` (predicting return until next qtr earnings)."""

from __future__ import annotations

from ..types import TaskSpec

CANONICAL_TASK_ID = "next_quarter_move"

SPEC: TaskSpec = {
    "task_id": CANONICAL_TASK_ID,
    "implemented": True,
    "text_cols": [
        "earnings_transcript",
        "press_release_8k_body",
        "press_release_ex991",
        "press_release_ex992",
    ],
    "numerical_cols": [
        "price_momentum_30d",
        "price_momentum_90d",
        "pct_from_52w_high_pt",
        "avg_volume_20d",
        "d_minus_1_close",
    ],
    "label_col": "move_next_qtr",
    "label_values": [],  # Regression tasks don't use categorical labels
    "task_instruction": (
        "Analyse the provided earnings call materials and predict the stock price movement "
        "from this quarter's earnings date until the day before the next quarter's earnings date.\n\n"
        "Returns a JSON object matching this exact schema:\n"
        '{"move": <predicted float, e.g. 0.05 for 5% gain or -0.02 for 2% loss>}\n\n'
        "Do not include any other keys or explanation."
    ),
    "kind": "regression",
}
