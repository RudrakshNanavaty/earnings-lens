"""
Evaluate inference over N random episodes (default 100).

Requires OpenEnv server and OPENAI_API_KEY. The server's active task (``EARNINGS_ANALYST_TASK_ID``)
must match what you are measuring; ``--task`` here selects which task spec is used for report labels.

Usage:
    uv run python evaluate.py
    uv run python evaluate.py --samples 50 --quiet
    uv run python evaluate.py --samples 10 -o results.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from collections import defaultdict

from dotenv import load_dotenv

from earnings_analyst.environment_config import DEFAULT_TASK, TASKS
from inference import run_episode

load_dotenv()


def _normalize_label(label_text: str) -> str:
    return str(label_text).strip().lower()


def exact_match(predicted_label: str, ground_truth_label: str) -> bool:
    return _normalize_label(predicted_label) == _normalize_label(ground_truth_label)


def confusion_key(predicted_label: str, ground_truth_label: str) -> tuple[str, str]:
    return (
        _normalize_label(predicted_label),
        _normalize_label(ground_truth_label),
    )


_CSV_FIELDNAMES = (
    "sample_index",
    "task_id",
    "model",
    "predicted",
    "ground_truth",
    "exact_match",
    "reward",
    "done",
    "model_response",
)


async def run_evaluation(
    *,
    samples: int,
    base_url: str | None,
    model: str | None,
    task_id: str,
    quiet: bool,
    output_path: str | None,
) -> None:
    spec = TASKS.get(task_id) or TASKS[DEFAULT_TASK]
    label_values = list(spec["label_values"])
    resolved_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")

    rewards: list[float] = []
    exact_match_count = 0
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    per_ground_truth_label: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0}
    )
    csv_rows: list[dict[str, str | int | float | bool]] = []

    for episode_index in range(samples):
        if not quiet:
            print(f"episode {episode_index + 1}/{samples} ...", flush=True)
        episode_result = await run_episode(
            base_url=base_url,
            model=model,
            verbose=True,

        )
        episode_reward = float(
            episode_result.reward if episode_result.reward is not None else 0.0
        )
        rewards.append(episode_reward)
        ground_truth_label = episode_result.ground_truth
        predicted_label = episode_result.predicted
        is_exact = exact_match(predicted_label, ground_truth_label)
        if is_exact:
            exact_match_count += 1
        confusion[confusion_key(predicted_label, ground_truth_label)] += 1
        normalized_ground_truth = _normalize_label(ground_truth_label)
        per_ground_truth_label[normalized_ground_truth]["n"] += 1
        if is_exact:
            per_ground_truth_label[normalized_ground_truth]["correct"] += 1

        # --- VERBOSE PRINTING BLOCK (Safe to remove) ---
        # NOTE: This block is exclusively for result visibility in the console.
        if not quiet:
            print(f"\n--- Episode {episode_index + 1}/{samples} Summary ---")
            print(f"Reward: {episode_reward:.4f}")
            print(f"Predicted: {predicted_label}")
            print(f"Ground Truth: {ground_truth_label}")
            print(f"Model Response: {episode_result.model_response_text}")
            print("-" * 40)
        # ------------------------------------------------

        csv_rows.append(
            {
                "sample_index": episode_index + 1,
                "task_id": task_id,
                "model": resolved_model,
                "predicted": predicted_label,
                "ground_truth": ground_truth_label,
                "exact_match": is_exact,
                "reward": episode_reward,
                "done": episode_result.done,
                "model_response": episode_result.model_response_text or "",
            }
        )

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    exact_accuracy = exact_match_count / samples if samples else 0.0

    print("\n=== Evaluation summary ===")
    print(f"samples: {samples}")
    print(f"mean_reward: {mean_reward:.4f}")
    print(f"exact_accuracy: {exact_accuracy:.4f} ({exact_match_count}/{samples})")
    if label_values:
        print("\nPer ground-truth label (exact match rate):")
        for lab in label_values:
            normalized_key = _normalize_label(lab)
            row = per_ground_truth_label.get(normalized_key, {"n": 0, "correct": 0})
            total_count, correct_count = row["n"], row["correct"]
            rate = (correct_count / total_count) if total_count else 0.0
            print(f"  {lab!r}: {rate:.4f} ({correct_count}/{total_count})")
    else:
        print(
            "\n(No label_values in selected task spec — add them when the task is implemented.)"
        )

    print("\nConfusion (predicted -> counts by ground_truth):")
    counts_by_truth: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (
        predicted_normalized,
        truth_normalized,
    ), occurrence_count in sorted(confusion.items()):
        counts_by_truth[truth_normalized].append(
            (predicted_normalized, occurrence_count)
        )
    for truth_normalized in sorted(counts_by_truth.keys()):
        parts = ", ".join(
            f"{predicted_normalized!r}:{occurrence_count}"
            for predicted_normalized, occurrence_count in sorted(
                counts_by_truth[truth_normalized]
            )
        )
        print(f"  truth={truth_normalized!r}: {parts}")

    if output_path:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote {len(csv_rows)} row(s) to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate over N episodes (align server EARNINGS_ANALYST_TASK_ID with --task)"
    )
    parser.add_argument("--samples", type=int, default=100, help="Number of episodes")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ENV_SERVER_URL", "http://localhost:8000"),
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Task id for label list in report (must match server task)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-episode lines"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="PATH",
        help="Write one row per episode to this CSV file (UTF-8)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(
            run_evaluation(
                samples=args.samples,
                base_url=args.base_url,
                model=args.model,
                task_id=args.task,
                quiet=args.quiet,
                output_path=args.output,
            )
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
