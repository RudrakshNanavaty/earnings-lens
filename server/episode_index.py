"""
Build lookups over the loaded Hugging Face split for Gradio (company / year / quarter).

Rows are keyed by ``(symbol, year, quarter)`` and must match ``episode_id`` when present
(``symbol_year_Qquarter``, e.g. ``A_2006_Q1``).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

from earnings_analyst.environment_config import (
    DATASET_COMPANY_NAME_COLUMN,
    DATASET_EPISODE_ID_COLUMN,
    DATASET_QUARTER_COLUMN,
    DATASET_SYMBOL_COLUMN,
    DATASET_YEAR_COLUMN,
)

# First sidebar option: sample a random row (same behavior as HTTP clients).
RANDOM_EPISODE_LABEL = "— Random row —"


def _normalize_cell(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        x = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return x


def format_episode_id(symbol: str, year: int, quarter: int) -> str:
    """Canonical ``episode_id`` pattern: ``{symbol}_{year}_Q{quarter}``."""
    return f"{symbol}_{year}_Q{quarter}"


class EpisodeIndex:
    """Maps ticker symbol + fiscal year + quarter to a dataset row index."""

    _REQUIRED_COLUMNS: tuple[str, ...] = (
        DATASET_EPISODE_ID_COLUMN,
        DATASET_SYMBOL_COLUMN,
        DATASET_COMPANY_NAME_COLUMN,
        DATASET_YEAR_COLUMN,
        DATASET_QUARTER_COLUMN,
    )

    def __init__(
        self,
        dataset: Any,
        episode_id_col: str = DATASET_EPISODE_ID_COLUMN,
        symbol_col: str = DATASET_SYMBOL_COLUMN,
        company_name_col: str = DATASET_COMPANY_NAME_COLUMN,
        year_col: str = DATASET_YEAR_COLUMN,
        quarter_col: str = DATASET_QUARTER_COLUMN,
    ) -> None:
        names = dataset.column_names
        missing = [c for c in self._REQUIRED_COLUMNS if c not in names]
        if missing:
            raise ValueError(
                f"Dataset is missing column(s) {missing}. "
                f"Check environment_config.py. Available columns: {names!r}."
            )

        self._symbol_to_display: dict[str, str] = {}
        self._display_to_symbol: dict[str, str] = {}
        self._symbol_to_years: dict[str, set[int]] = defaultdict(set)
        self._symbol_year_to_quarters: dict[tuple[str, int], set[int]] = defaultdict(
            set
        )
        self._triple_to_index: dict[tuple[str, int, int], int] = {}
        duplicate_triples: set[tuple[str, int, int]] = set()

        n = len(dataset)
        for i in range(n):
            row = dataset[i]
            sym = _normalize_cell(row.get(symbol_col))
            cname = _normalize_cell(row.get(company_name_col))
            yr = _as_int(row.get(year_col))
            qn = _as_int(row.get(quarter_col))
            if not sym or yr is None or qn is None:
                continue

            if sym not in self._symbol_to_display:
                label = cname or sym
                display = f"{label} ({sym})"
                self._symbol_to_display[sym] = display
                self._display_to_symbol[display] = sym

            triple = (sym, yr, qn)
            self._symbol_to_years[sym].add(yr)
            self._symbol_year_to_quarters[(sym, yr)].add(qn)

            if triple in self._triple_to_index:
                duplicate_triples.add(triple)
            else:
                self._triple_to_index[triple] = i
            # episode_id_col is validated to exist; row lookup uses (symbol, year, quarter).

        if duplicate_triples:
            warnings.warn(
                "Duplicate (symbol, year, quarter) rows; using the first index for each.",
                UserWarning,
                stacklevel=1,
            )

    def sorted_company_displays(self) -> list[str]:
        return sorted(self._display_to_symbol.keys())

    def symbol_for_display(self, display: str) -> str:
        if display not in self._display_to_symbol:
            raise KeyError(f"Unknown company selection: {display!r}")
        return self._display_to_symbol[display]

    def years_for_symbol(self, symbol: str) -> list[int]:
        return sorted(self._symbol_to_years.get(symbol, []))

    def quarters_for(self, symbol: str, year: int) -> list[int]:
        return sorted(self._symbol_year_to_quarters.get((symbol, year), []))

    def row_index(self, symbol: str, year: int, quarter: int) -> int:
        key = (symbol, year, quarter)
        if key not in self._triple_to_index:
            raise KeyError(
                f"No row for symbol={symbol!r}, year={year!r}, quarter={quarter!r}. "
                "Pick a valid combination from the dropdowns."
            )
        return self._triple_to_index[key]


_instance: EpisodeIndex | None = None


def get_episode_index() -> EpisodeIndex:
    """Lazily build the index over ``dataset_loader.dataset`` (single pass)."""
    global _instance
    if _instance is None:
        from earnings_analyst.server.dataset_loader import dataset

        _instance = EpisodeIndex(dataset)
    return _instance


__all__ = [
    "EpisodeIndex",
    "format_episode_id",
    "get_episode_index",
    "RANDOM_EPISODE_LABEL",
]
