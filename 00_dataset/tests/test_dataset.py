import os

import pandas as pd


def test_dataset_files_exist() -> None:
    train_path = os.path.join("00_dataset", "output", "train.csv")
    test_path = os.path.join("00_dataset", "output", "test.csv")

    assert os.path.exists(train_path), f"Train dataset not found at {train_path}"
    assert os.path.exists(test_path), f"Test dataset not found at {test_path}"


def test_dataset_split_ratio() -> None:
    train_path = os.path.join("00_dataset", "output", "train.csv")
    test_path = os.path.join("00_dataset", "output", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    total = len(train_df) + len(test_df)
    train_ratio = len(train_df) / total
    test_ratio = len(test_df) / total

    assert len(train_df) > 0, "Train dataset is empty"
    assert len(test_df) > 0, "Test dataset is empty"
    assert abs(train_ratio - 0.7) < 0.05, (
        f"Train ratio {train_ratio} is not close to 0.7"
    )
    assert abs(test_ratio - 0.3) < 0.05, f"Test ratio {test_ratio} is not close to 0.3"


def test_dataset_columns() -> None:
    train_path = os.path.join("00_dataset", "output", "train.csv")
    train_df = pd.read_csv(train_path)

    assert "text" in train_df.columns, "Column 'text' missing in train dataset"
    has_label = (
        "label" in train_df.columns
        or "labels" in train_df.columns
        or "intent" in train_df.columns
    )
    assert has_label, "Label column missing in train dataset"
