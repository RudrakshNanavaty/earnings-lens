"""Task specification for ``get_figures`` (Financial figure extraction)."""

from __future__ import annotations
from ..types import TaskSpec

CANONICAL_TASK_ID = "get_figures"

# Mapping from JSON metric keys to dataset XBRL columns
METRIC_TO_COLUMN: dict[str, str] = {
    "revenue": "xbrl_revenue",
    "cost_of_revenue": "xbrl_cost_of_revenue",
    "gross_profit": "xbrl_gross_profit",
    "operating_income": "xbrl_operating_income",
    "net_income": "xbrl_net_income",
    "eps_basic": "xbrl_eps_basic",
    "eps_diluted": "xbrl_eps_diluted",
    "cash_and_cash_equivalents": "xbrl_cash_and_cash_equivalents",
    "total_assets": "xbrl_total_assets",
    "total_liabilities": "xbrl_total_liabilities",
    "net_cash_operating_activities": "xbrl_net_cash_operating_activities",
    "capital_expenditures": "xbrl_capital_expenditures",
}

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
        "avg_volume_20d",
    ],
    "label_col": "xbrl_revenue",  # Primary ground truth column
    "label_values": [],
    "task_instruction": (
        "Extract key financial figures from the provided earnings call materials.\n\n"
        "Return a JSON object matching this exact US-GAAP taxonomy schema:\n"
        "{\n"
        '  "taxonomy_version": "us-gaap-2024",\n'
        '  "income_statement": {\n'
        '    "revenue": <float>,\n'
        '    "cost_of_revenue": <float>,\n'
        '    "gross_profit": <float>,\n'
        '    "operating_income": <float>,\n'
        '    "net_income": <float>,\n'
        '    "eps_basic": <float>,\n'
        '    "eps_diluted": <float>\n'
        "  },\n"
        '  "balance_sheet": {\n'
        '    "cash_and_cash_equivalents": <float>,\n'
        '    "total_assets": <float>,\n'
        '    "total_liabilities": <float>\n'
        "  },\n"
        '  "cash_flow": {\n'
        '    "net_cash_operating_activities": <float>,\n'
        '    "capital_expenditures": <float>\n'
        "  }\n"
        "}\n\n"
        "Values should ideally be in USD. If a figure is not found or not mentioned, use null.\n"
        "Do not include any other keys, explanations, or markdown blocks."
    ),
    "kind": "extraction",
    # Metadata for the environment to pack all ground truth figures
    "xbrl_columns": list(METRIC_TO_COLUMN.values()),
}
