"""Task specification for ``1_day_move`` — fill in when implementing."""

from __future__ import annotations

from ..types import TaskSpec

CANONICAL_TASK_ID = "1_day_move"

SPEC: TaskSpec = {
    "task_id": CANONICAL_TASK_ID,
    "implemented": True,
    "text_cols": [
        "earnings_transcript",
        "press_release_8k_body",
        "press_release_ex991",
        "press_release_ex992",
        "press_release_sources",
    ],
    "numerical_cols": [
        "price_momentum_30d",
        "price_momentum_90d",
        "pct_from_52w_high_pt",
        "avg_volume_20d",
        "d_minus_1_close",
    ],
    "label_col": "move_1d",
    "label_values": [
        "very bearish",
        "bearish",
        "neutral",
        "bullish",
        "very bullish",
    ],
    "task_instruction": (
        "Analyse the provided earnings call materials and market data to predict the stock price movement after 1 day.\n\n"
        "Return a JSON object matching this exact schema:\n"
        '{"percentage_move": <float>, "label": "<one of: very bearish | bearish | neutral | bullish | very bullish>"}\n\n'
        "Brackets:\n"
        "- More than 7% negative: very bearish\n"
        "- 1-7% negative: bearish\n"
        "- -1% to +1%: neutral\n"
        "- 1-7% positive: bullish\n"
        "- More than 7% positive: very bullish\n\n"
        "Do not include any other keys or explanation."
    ),
    "kind": "regression",
}

