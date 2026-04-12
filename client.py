"""Earnings Analyst Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EarningsAnalystAction, EarningsAnalystObservation


class EarningsAnalystEnv(
    EnvClient[EarningsAnalystAction, EarningsAnalystObservation, State]
):
    """
    Client for the Earnings Analyst Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> async with EarningsAnalystEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset()
        ...     print(result.observation.task_instruction)
        ...
        ...     result = await client.step(EarningsAnalystAction(prediction="neutral"))
        ...     print(result.observation.metadata)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = await EarningsAnalystEnv.from_docker_image("earnings_analyst-env:latest")
        >>> try:
        ...     result = await client.reset()
        ...     result = await client.step(EarningsAnalystAction(prediction="0.01"))
        ... finally:
        ...     await client.close()
    """

    def _step_payload(self, action: EarningsAnalystAction) -> Dict:
        """
        Convert EarningsAnalystAction to JSON payload for step message.

        Args:
            action: EarningsAnalystAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "prediction": action.prediction,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EarningsAnalystObservation]:
        """
        Parse server response into StepResult[EarningsAnalystObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with EarningsAnalystObservation
        """
        obs_data = payload.get("observation", {})
        observation = EarningsAnalystObservation(
            text_context=obs_data.get("text_context") or {},
            numerical_context=obs_data.get("numerical_context") or {},
            task_instruction=obs_data.get("task_instruction", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
