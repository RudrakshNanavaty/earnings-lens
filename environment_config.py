"""
Central configuration for all earnings analyst tasks.

Plain data only — no imports from env or dataset loader. Add new tasks here.
"""

DATASET_ID = "RudrakshNanavaty/earnings-call-data"
DATASET_FILE = "episodes.parquet"

TASKS = {
    "sentiment_v1": {
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
        "label_col": "sentiment_label",
        "label_values": [
            "very bearish",
            "bearish",
            "neutral",
            "bullish",
            "very bullish",
        ],
        "task_instruction": (
            "Analyse the provided earnings call materials and classify the overall market sentiment.\n\n"
            "Return a JSON object matching this exact schema:\n"
            '{"sentiment": "<one of: very bearish | bearish | neutral | bullish | very bullish>"}\n\n'
            "Do not include any other keys or explanation."
        ),
    },
}

DEFAULT_TASK = "sentiment_v1"
