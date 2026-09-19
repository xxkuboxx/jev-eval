import time
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def main() -> None:
    input_path = Path("01_lightgbm/input/train.csv")
    output_dir = Path("01_lightgbm/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {input_path}")
    df = pd.read_csv(input_path)

    # データのカラム: text, labels
    X = df["text"].astype(str).values
    y = df["labels"].values

    print("Training TF-IDF + LightGBM pipeline...")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ("clf", LGBMClassifier(random_state=42, n_estimators=100, n_jobs=-1)),
        ]
    )

    start_time = time.time()
    pipeline.fit(X, y)
    train_duration = time.time() - start_time
    print(f"Training completed in {train_duration:.2f} seconds.")

    model_path = output_dir / "model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
