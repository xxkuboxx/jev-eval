import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def main() -> None:
    input_path = Path("01_lightgbm/input/test.csv")
    model_path = Path("01_lightgbm/output/model.pkl")
    output_dir = Path("01_lightgbm/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}")
    pipeline = joblib.load(model_path)

    print(f"Loading test data from {input_path}")
    df = pd.read_csv(input_path)
    X = df["text"].astype(str).values
    y_true = df["labels"].values

    print("Running inference and measuring latency...")
    latencies = []
    probabilities_list = []

    for text in X:
        t0 = time.perf_counter()
        proba = pipeline.predict_proba([text])[0]
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        probabilities_list.append(proba)

    probabilities = np.array(probabilities_list)
    avg_latency_ms = float(np.mean(latencies))

    classes = pipeline.named_steps["clf"].classes_

    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    roc_auc_macro = float(
        roc_auc_score(y_true_bin, probabilities, average="macro", multi_class="ovr")
    )

    print(f"Average Latency: {avg_latency_ms:.2f} ms/sample")
    print(f"ROC-AUC (Macro-average): {roc_auc_macro:.4f}")

    pred_df = df.copy()
    for i, cls in enumerate(classes):
        pred_df[f"prob_{cls}"] = probabilities[:, i]
    pred_df["latency_ms"] = latencies
    predictions_path = output_dir / "predictions.csv"
    pred_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    metrics = {
        "model": "lightgbm",
        "roc_auc_macro": roc_auc_macro,
        "avg_latency_ms": avg_latency_ms,
        "total_samples": len(df),
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
