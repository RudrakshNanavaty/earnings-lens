"""Unit tests for :mod:`earnings_analyst.server.episode_index`."""

from __future__ import annotations

import pytest
from datasets import Dataset

from earnings_analyst.server.episode_index import (
    EpisodeIndex,
    RANDOM_EPISODE_LABEL,
    format_episode_id,
)


def test_format_episode_id() -> None:
    assert format_episode_id("A", 2006, 1) == "A_2006_Q1"


def test_episode_index_lookup_and_cascades() -> None:
    ds = Dataset.from_dict(
        {
            "episode_id": ["AAA_2024_Q1", "AAA_2024_Q2", "BBB_2024_Q1"],
            "symbol": ["AAA", "AAA", "BBB"],
            "company_name": ["Alpha Co", "Alpha Co", "Beta Co"],
            "year": [2024, 2024, 2024],
            "quarter": [1, 2, 1],
        }
    )
    idx = EpisodeIndex(ds)
    d_aaa = "Alpha Co (AAA)"
    d_bbb = "Beta Co (BBB)"
    assert idx.sorted_company_displays() == [d_aaa, d_bbb]
    assert idx.symbol_for_display(d_aaa) == "AAA"
    assert idx.years_for_symbol("AAA") == [2024]
    assert idx.quarters_for("AAA", 2024) == [1, 2]
    assert idx.row_index("AAA", 2024, 1) == 0
    assert idx.row_index("AAA", 2024, 2) == 1
    assert idx.row_index("BBB", 2024, 1) == 2


def test_episode_index_duplicate_triple_warns() -> None:
    ds = Dataset.from_dict(
        {
            "episode_id": ["X_2024_Q1", "X_2024_Q1_dup"],
            "symbol": ["X", "X"],
            "company_name": ["X", "X"],
            "year": [2024, 2024],
            "quarter": [1, 1],
        }
    )
    with pytest.warns(UserWarning, match="Duplicate"):
        idx = EpisodeIndex(ds)
    assert idx.row_index("X", 2024, 1) == 0


def test_episode_index_skips_incomplete_rows() -> None:
    ds = Dataset.from_dict(
        {
            "episode_id": ["_", "Z_2024_Q1"],
            "symbol": ["", "ZZZ"],
            "company_name": ["", "Zed"],
            "year": [2024, 2024],
            "quarter": [1, 1],
        }
    )
    idx = EpisodeIndex(ds)
    assert idx.sorted_company_displays() == ["Zed (ZZZ)"]
    assert idx.row_index("ZZZ", 2024, 1) == 1


def test_episode_index_missing_column_raises() -> None:
    ds = Dataset.from_dict({"a": [1]})
    with pytest.raises(ValueError, match="missing column"):
        EpisodeIndex(ds)


def test_random_label_constant() -> None:
    assert "Random" in RANDOM_EPISODE_LABEL
