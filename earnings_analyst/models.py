"""
Data models for the Earnings Analyst Environment.
"""

from typing import Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class EarningsAnalystAction(Action):
    """Agent response: a single string prediction (format depends on the active task)."""

    prediction: str = Field(
        ...,
        description=(
            "Model prediction as a string (e.g. label, or stringified number), "
            "per the active task's instruction schema."
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
    ground_truth: str = Field(
        default="",
        description="Actual value for the task (populated on terminal step)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for debugging, rewards, or logging",
    )
