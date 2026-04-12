"""
Data models for the Earnings Analyst Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class EarningsAnalystAction(Action):
    """Action for sentiment classification (and future tasks)."""

    sentiment: str = Field(
        ...,
        description=(
            "Predicted sentiment: one of very bearish, bearish, neutral, bullish, very bullish"
        ),
    )


class EarningsAnalystObservation(Observation):
    """Observation bundle: text context, numerical context, and task instruction."""

    text_context: dict[str, str] = Field(
        default_factory=dict,
        description="Non-null text fields for the active task (column name -> text)",
    )
    numerical_context: dict[str, float] = Field(
        default_factory=dict,
        description="Market / numerical features for the active task (column name -> value)",
    )
    task_instruction: str = Field(
        default="",
        description="Natural language instruction and JSON schema for the agent",
    )
