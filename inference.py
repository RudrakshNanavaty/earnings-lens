"""
EarningsLens Mandatory Inference Script
=======================================
Follows the required STDOUT format for hackathon submission.
Supports all 5 tasks: sentiment_label, 1_day_move, 30_day_move, next_quarter_move, get_figures.
"""

import asyncio
import json
import os
from typing import List, Optional, Any

from openai import OpenAI

from earnings_analyst.client import EarningsAnalystEnv
from earnings_analyst.models import EarningsAnalystAction
from dotenv import load_dotenv

load_dotenv()


# Environment Variables
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
ENV_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

# Task Configuration
TASK_ID = os.getenv("EARNINGS_ANALYST_TASK_ID", "sentiment_label")
BENCHMARK = "earnings-lens"
MAX_STEPS = 1
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_TOTAL_REWARD = 1.0  # Max reward for single-step tasks is 1.0


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Masking newline in action for clean STDOUT if it's JSON
    action_clean = action.replace("\n", " ")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_system_prompt(task_id: str) -> str:
    """Returns a highly specific system prompt for the financial analyst agent."""
    prompts = {
        "sentiment_label": (
            "You are a financial analyst. Analyze the sentiment of the provided earnings call context. "
            'Respond with a single JSON object: {"sentiment": "<label>"}. '
            "Valid labels: very bearish, bearish, neutral, bullish, very bullish."
        ),
        "1_day_move": (
            "Predict the 1-day stock price movement from the earnings call. "
            'Respond with a single JSON object: {"percentage_move": <float>, "label": "<direction>"}. '
            "Label should be 'up' or 'down'."
        ),
        "30_day_move": (
            "Predict the 30-day stock price movement from the earnings call. "
            'Respond with a single JSON object: {"percentage_move": <float>, "label": "<direction>"}. '
            "Label should be 'up' or 'down'."
        ),
        "next_quarter_move": (
            "Predict the stock price movement for the next quarter. "
            'Respond with a single JSON object: {"percentage_move": <float>, "label": "<direction>"}. '
            "Label should be 'up' or 'down'."
        ),
        "get_figures": (
            "Extract key US-GAAP financial figures. Respond with the requested nested JSON schema "
            "containing income_statement, balance_sheet, and cash_flow metrics."
        ),
    }
    return prompts.get(
        task_id, "You are a financial analyst. Respond exactly as instructed."
    )


def build_user_prompt(obs: Any) -> str:
    parts = [obs.task_instruction]
    if obs.text_context:
        parts.append("\nText Context:")
        for k, v in obs.text_context.items():
            parts.append(f"{k}: {v[:1000]}...")  # Truncate for prompt efficiency
    if obs.numerical_context:
        parts.append("\nNumerical Context:")
        parts.append(json.dumps(obs.numerical_context))
    return "\n".join(parts)


def parse_prediction(text: str, task_id: str) -> str:
    """Extract the primary prediction string for the env.step() call."""
    try:
        data = json.loads(text)
        if task_id == "sentiment_label":
            return str(data.get("sentiment", "neutral"))
        if task_id == "get_figures":
            return text  # Grader expects the full JSON
        # For movement tasks, we return the percentage or a combined string
        # Our specific graders handle either. I'll pass the full JSON to be safe.
        return text
    except Exception:
        return text


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Environment initialization
    if IMAGE_NAME:
        env = await EarningsAnalystEnv.from_docker_image(IMAGE_NAME)
    else:
        env = EarningsAnalystEnv(base_url=ENV_URL)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)

    try:
        # If EarningsAnalystEnv is used with 'async with', it connects.
        # Here I simulate the template behavior.
        async with env:
            result = await env.reset()
            obs = result.observation

            system_prompt = build_system_prompt(TASK_ID)
            user_prompt = build_user_prompt(obs)

            # Model Call
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"}
                if "JSON" in system_prompt or "figures" in TASK_ID
                else None,
            )
            response_text = (completion.choices[0].message.content or "").strip()

            prediction = parse_prediction(response_text, TASK_ID)

            # Step
            result = await env.step(EarningsAnalystAction(prediction=prediction))

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = 1

            log_step(step=1, action=prediction, reward=reward, done=done, error=None)

            score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Inference failed: {exc}", flush=True)
    finally:
        try:
            await env.close()
        except:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
