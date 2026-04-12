"""
Single-episode inference: reset env, call OpenAI, step with prediction string.

Requires a running OpenEnv server (e.g. `uv run server`).

The server must use the same task as your prompt expects (``EARNINGS_ANALYST_TASK_ID``).

Usage:
    uv run python inference.py
    # or
    python inference.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

try:
    from earnings_analyst.client import EarningsAnalystEnv
    from earnings_analyst.models import (
        EarningsAnalystAction,
        EarningsAnalystObservation,
    )
except ImportError:
    from client import EarningsAnalystEnv
    from models import EarningsAnalystAction, EarningsAnalystObservation

load_dotenv()

DEFAULT_LABELS = [
    "very bearish",
    "bearish",
    "neutral",
    "bullish",
    "very bullish",
]


@dataclass
class EpisodeResult:
    reward: float | None
    predicted: str
    ground_truth: str
    done: bool
    model_response_text: str | None = None


def _normalize_prediction(model_text: str, valid: list[str] | None = None) -> str:
    """Map model output to a canonical label or return as is for regression."""
    if not valid:
        return model_text.strip()
    
    labels = valid
    normalized_model_text = str(model_text).strip().lower()
    for canonical_label in labels:
        if normalized_model_text == canonical_label.lower():
            return canonical_label
    for canonical_label in labels:
        canonical_lower = canonical_label.lower()
        if (
            canonical_lower in normalized_model_text
            or normalized_model_text in canonical_lower
        ):
            return canonical_label
    return "neutral"


def build_user_content(obs: EarningsAnalystObservation) -> str:
    parts: list[str] = [obs.task_instruction]
    if obs.text_context:
        parts.append("## Text context (by field name)")
        for name, text in sorted(obs.text_context.items()):
            parts.append(f"### {name}\n{text}")
    if obs.numerical_context:
        parts.append("## Numerical context (JSON)")
        parts.append(json.dumps(obs.numerical_context, indent=2))
    return "\n\n".join(parts)


async def predict_with_openai(
    obs: EarningsAnalystObservation,
    *,
    client: AsyncOpenAI,
    model: str,
    valid_labels: list[str] | None = None,
) -> tuple[str, str]:
    """
    Example Chat Completions call returning a JSON object.
    """
    user_content = build_user_content(obs)
    system_prompt = (
        "You are a financial analyst assistant. "
        "Your task is to analyze the provided financial data and respond "
        "EXACTLY as instructed in the Task Instruction. "
        "Reply with a single JSON object only, no markdown or extra text."
    )
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    response_text = (completion.choices[0].message.content or "").strip()
    
    # Try to extract the primary value based on common keys
    predicted = response_text
    try:
        parsed: dict[str, Any] = json.loads(response_text)
        if isinstance(parsed, dict):
            # Check for common return keys
            for key in ["sentiment", "move", "label", "prediction"]:
                if key in parsed:
                    if valid_labels:
                        predicted = _normalize_prediction(str(parsed[key]), valid_labels)
                    else:
                        predicted = str(parsed[key])
                    break
    except (json.JSONDecodeError, TypeError, ValueError):
        if valid_labels:
            predicted = _normalize_prediction(response_text, valid_labels)
    
    return predicted, response_text


async def run_episode(
    *,
    base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    model: str | None = None,
    verbose: bool = True,
) -> EpisodeResult:
    """
    One reset → OpenAI prediction → step. Returns reward and metadata from the server.
    """
    environment_base_url = base_url or os.environ.get(
        "ENV_SERVER_URL", "http://localhost:8000"
    )
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment or .env")

    resolved_openai_base_url = openai_base_url or os.environ.get("OPENAI_BASE_URL")
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o")

    openai_client_options: dict[str, Any] = {"api_key": api_key}
    if resolved_openai_base_url:
        openai_client_options["base_url"] = resolved_openai_base_url
    
    if verbose:
        print(f"DEBUG: Using base_url={resolved_openai_base_url or 'default'} model={model_name}")
    
    client = AsyncOpenAI(**openai_client_options)



    async with EarningsAnalystEnv(base_url=environment_base_url) as env:
        reset_out = await env.reset()
        observation = reset_out.observation
        # We pass valid_labels if they exist in the observation/registry
        # This implementation assumes the client can fetch labels or we hardcode.
        # For simplicity, we'll try to use labels from metadata if available on reset
        # Or just use None for regression.
        valid_labels = getattr(observation, "label_values", None)
        
        predicted, response_text = await predict_with_openai(
            observation, client=client, model=model_name, valid_labels=valid_labels
        )
        step_out = await env.step(EarningsAnalystAction(prediction=predicted))
        step_observation = step_out.observation
        ground_truth_label = str(getattr(step_observation, "ground_truth", ""))

        reward = step_out.reward
        if verbose:
            print(
                f"predicted={predicted!r} "
                f"ground_truth={ground_truth_label!r} reward={reward}"
            )
            if response_text and len(response_text) < 2000:
                print(f"model_response_json={response_text!r}")
        truncated_response = (
            response_text if len(response_text) < 4000 else response_text[:4000] + "..."
        )
        return EpisodeResult(
            reward=reward,
            predicted=predicted,
            ground_truth=ground_truth_label,
            done=step_out.done,
            model_response_text=truncated_response,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one episode via OpenAI + env server (example inference script)"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ENV_SERVER_URL", "http://localhost:8000"),
        help="OpenEnv HTTP base URL",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()
    try:
        asyncio.run(
            run_episode(
                base_url=args.base_url,
                model=args.model,
                verbose=not args.quiet,
            )
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
