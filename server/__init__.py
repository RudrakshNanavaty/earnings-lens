"""Earnings Analyst environment server components.

``EarningsAnalystEnvironment`` is loaded lazily so importing sibling modules
(e.g. ``episode_index``) does not pull in the Hugging Face dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .earnings_analyst_environment import EarningsAnalystEnvironment

__all__ = ["EarningsAnalystEnvironment"]


def __getattr__(name: str) -> Any:
    if name == "EarningsAnalystEnvironment":
        from .earnings_analyst_environment import EarningsAnalystEnvironment as _Env

        return _Env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
