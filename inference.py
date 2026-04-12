"""
Inference Script — Earnings Analyst (OpenEnv)
=============================================
Mandatory environment variables (injected by the hackathon evaluator):
    HF_TOKEN        Hugging Face / API key for the LLM.
    API_BASE_URL    The LLM API endpoint (OpenAI-compatible).
    MODEL_NAME      The model identifier.
    IMAGE_NAME      Docker image for the environment (triggers from_docker_image).

Local-dev fallbacks (set in .env):
    OPENAI_API_KEY  Alternative API key when HF_TOKEN is absent.
    OPENAI_MODEL    Alternative model name when MODEL_NAME is absent.
    ENV_SERVER_URL  HTTP base URL used when IMAGE_NAME is not set.

STDOUT FORMAT
    [START] task=<task> env=earnings_analyst model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from earnings_analyst.client import EarningsAnalystEnv
from earnings_analyst.models import EarningsAnalystAction, EarningsAnalystObservation

load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv(
    "OPENAI_MODEL", "Qwen/Qwen2.5-72B-Instruct"
)
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000")
TASK_NAME = os.getenv("EARNINGS_ANALYST_TASK_ID", "sentiment_label")
BENCHMARK = "earnings_analyst"
SUCCESS_SCORE_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Logging helpers — mandatory STDOUT protocol
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: str | None
) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def _normalize_prediction(model_text: str, valid: list[str] | None = None) -> str:
    """Map model output to a canonical label, or return as-is for regression."""
    if not valid:
        return model_text.strip()
    normalized = str(model_text).strip().lower()
    for label in valid:
        if normalized == label.lower():
            return label
    for label in valid:
        label_lower = label.lower()
        if label_lower in normalized or normalized in label_lower:
            return label
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


def get_prediction(
    client: OpenAI,
    obs: EarningsAnalystObservation,
    valid_labels: list[str] | None = None,
) -> tuple[str, str | None]:
    """Call the LLM and return (prediction_string, last_action_error_or_None)."""
    user_content = build_user_content(obs)
    system_prompt = (
        "You are a financial analyst assistant. "
        "Your task is to analyze the provided financial data and respond "
        "EXACTLY as instructed in the Task Instruction. "
        "Reply with a single JSON object only, no markdown or extra text."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        response_text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        return "neutral", str(exc)

    predicted = response_text
    try:
        parsed: dict[str, Any] = json.loads(response_text)
        if isinstance(parsed, dict):
            for key in ["sentiment", "label", "move", "prediction"]:
                if key in parsed:
                    if valid_labels:
                        predicted = _normalize_prediction(
                            str(parsed[key]), valid_labels
                        )
                    else:
                        predicted = str(parsed[key])
                    break
    except (json.JSONDecodeError, TypeError, ValueError):
        if valid_labels:
            predicted = _normalize_prediction(response_text, valid_labels)

    return predicted, None


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------


async def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "Set HF_TOKEN (or OPENAI_API_KEY) in the environment or .env"
        )

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    if IMAGE_NAME:
        env = await EarningsAnalystEnv.from_docker_image(IMAGE_NAME)
    else:
        env = EarningsAnalystEnv(base_url=ENV_SERVER_URL)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_out = await env.reset()
        obs = reset_out.observation
        valid_labels: list[str] | None = getattr(obs, "label_values", None) or None

        done = reset_out.done

        step = 1
        while not done:
            prediction, error = get_prediction(client, obs, valid_labels)
            step_out = await env.step(EarningsAnalystAction(prediction=prediction))

            reward = step_out.reward or 0.0
            done = step_out.done
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step, action=prediction, reward=reward, done=done, error=error
            )

            obs = step_out.observation
            step += 1

        score = float(rewards[-1]) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as exc:
            pass  # don't let cleanup failure suppress the [END] line
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
