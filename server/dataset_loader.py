"""
Load the Hugging Face dataset once as a module-level singleton.

Task-specific columns are declared per task under ``tasks/<name>/spec.py``.
"""

from datasets import load_dataset

try:
    from ..environment_config import DATASET_FILE, DATASET_ID
except ImportError:
    from environment_config import DATASET_FILE, DATASET_ID

# Loaded once on first import; all resets share this object.
# Pin Hub parquet so we never pick up features.parquet / raw_*.parquet from the same repo.
dataset = load_dataset(
    DATASET_ID,
    data_files={"train": DATASET_FILE},
    split="train",
)
