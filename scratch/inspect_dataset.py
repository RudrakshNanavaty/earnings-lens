from datasets import load_dataset
import os

DATASET_ID = "RudrakshNanavaty/earnings-call-data"
DATASET_FILE = "episodes_press_release_8k.parquet"

print(f"Loading dataset {DATASET_ID} file {DATASET_FILE}...")
dataset = load_dataset(
    DATASET_ID,
    data_files={"train": DATASET_FILE},
    split="train",
)

print("\nColumns:")
print(dataset.column_names)

print("\nFirst row summary:")
row = dataset[0]
for k, v in row.items():
    if v is not None:
        val_str = str(v)
        if len(val_str) > 100:
            val_str = val_str[:100] + "..."
        print(f"{k}: {val_str}")
    else:
        print(f"{k}: None")
