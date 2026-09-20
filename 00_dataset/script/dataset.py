import os

from datasets import load_dataset  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]


def main() -> None:
    print(
        "Loading dataset from Hugging Face Hub: tanaos/synthetic-intent-classifier-dataset-v1"
    )
    dataset = load_dataset("tanaos/synthetic-intent-classifier-dataset-v1")

    if "train" in dataset:
        df = dataset["train"].to_pandas()
    else:
        split_name = next(iter(dataset.keys()))
        df = dataset[split_name].to_pandas()

    print(f"Loaded dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, shuffle=True
    )

    output_dir = os.path.join("00_dataset", "output")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"Saved train set ({len(train_df)} rows) to {train_path}")
    print(f"Saved test set ({len(test_df)} rows) to {test_path}")

    print("\nTrain label distribution:")
    for col in ["label", "labels", "intent"]:
        if col in train_df.columns:
            print(train_df[col].value_counts())
            break


if __name__ == "__main__":
    main()
