import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    input_path = Path("02_distilbert/input/test.csv")
    model_dir = Path("02_distilbert/output/model_weights")
    output_dir = Path("02_distilbert/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ラベルマッピングの読み込み (int 型に正規化)
    with open(output_dir / "label_mapping.json", "r", encoding="utf-8") as f:
        classes = [int(c) for c in json.load(f)]

    print(f"Loading model and tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    print(f"Loading test data from {input_path}")
    df = pd.read_csv(input_path)
    texts = df["text"].astype(str).tolist()
    y_true = df["labels"].astype(int).values

    print("Running inference and measuring latency...")
    latencies = []
    probabilities_list = []

    # 1件ずつ推論して厳密なレイテンシ（ms/sample）を計測
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)
        probabilities_list.append(probs)

    probabilities = np.array(probabilities_list)
    if np.isnan(probabilities).any():
        raise ValueError(
            "Model inference produced NaN values. Please verify model training stability and weights."
        )
    avg_latency_ms = float(np.mean(latencies))

    # ROC-AUC (Macro-average) の算出
    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    roc_auc_macro = float(
        roc_auc_score(y_true_bin, probabilities, average="macro", multi_class="ovr")
    )

    print(f"Average Latency: {avg_latency_ms:.2f} ms/sample")
    print(f"ROC-AUC (Macro-average): {roc_auc_macro:.4f}")

    # 予測結果の保存
    pred_df = df.copy()
    for i, cls in enumerate(classes):
        pred_df[f"prob_{cls}"] = probabilities[:, i]
    pred_df["latency_ms"] = latencies
    predictions_path = output_dir / "predictions.csv"
    pred_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # メトリクスの保存
    metrics = {
        "model": "distilbert-base-uncased",
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
