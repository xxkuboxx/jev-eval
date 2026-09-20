import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

load_dotenv()

LABEL_MAPPING = {
    0: "挨拶・日常会話 (Greeting / Small Talk)",
    1: "挨拶・別れ (Goodbye / Farewell)",
    2: "感謝・お礼 (Expression of Gratitude)",
    3: "同意・肯定 (Agreement / Affirmation)",
    4: "否定・不満 (Disagreement / Discontent)",
    5: "話題の提案・日常の雑談 (Topic Suggestion / Small Talk)",
    6: "機能や仕組みに関する質問 (System Capabilities Inquiry)",
    7: "称賛・ポジティブなフィードバック (Praise / Positive Feedback)",
    8: "不満・不平の表明 (Complaint / Dissatisfaction)",
    9: "助けや説明の要求 (Help / Clarification Request)",
    10: "改善提案・機能追加の要望 (Feature Suggestion / Constructive Feedback)",
    11: "言語・設定の変更要求 (Language / Settings Change Request)",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Gemini Flash Lite on intent classification test data."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test mode on the first 10 samples only (does not save output files).",
    )
    args = parser.parse_args()

    input_path = Path("03_gemini_flash_lite/input/test.csv")
    output_dir = Path("03_gemini_flash_lite/output")
    tmp_dir = Path("03_gemini_flash_lite/tmp")
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading test data from {input_path}")
    df = pd.read_csv(input_path)

    if args.smoke:
        print("Smoke test mode enabled: limiting evaluation to the first 10 samples.")
        df = df.iloc[:10].copy()

    X = df["text"].astype(str).values
    y_true = df["labels"].values

    unique_classes = sorted(df["labels"].unique().tolist())
    print(f"Target classes ({len(unique_classes)}): {unique_classes}")

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            api_version="v1",
            retry_options=types.HttpRetryOptions(
                attempts=10,  # Default: 5
                initial_delay=10,  # Default: 1.0
                max_delay=100,  # Default: 60.0
                exp_base=1.5,  # Default: 2
                jitter=0.5,  # Default: 1
                http_status_codes=[429],  # Default: [408, 429, 500, 502, 503, 504]
            ),
        ),
    )
    model_name = "gemini-3.5-flash-lite"

    class IntentClassification(BaseModel):
        predicted_label: int = Field(
            description=(
                "Predicted intent category integer label (must be one of the exact integer label values "
                f"from {unique_classes})"
            )
        )
        confidence: float = Field(
            description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0
        )

    print(f"Running inference with {model_name} (parallel max_workers=10)...")

    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}

    mapping_str = "\n".join(
        [
            f"  - Label {label}: {desc}"
            for label, desc in LABEL_MAPPING.items()
            if label in unique_classes
        ]
    )
    system_instruction = (
        "You are an expert intent classification system.\n"
        "The dataset uses integer labels mapped to specific intent categories as follows:\n"
        f"{mapping_str}\n\n"
        f"The available intent category integer labels are: {unique_classes}.\n"
        "You must output the exact integer label value from this set in the predicted_label field, "
        "strictly according to the requested schema."
    )

    # Note: Using client.chats.create for chat session (send_message) to avoid AFC warnings.
    chat = client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=IntentClassification,
            temperature=0.0,
        ),
    )

    def process_sample(item: tuple[int, str]) -> tuple[int, str, float, float]:
        idx, text = item
        tmp_file = tmp_dir / f"sample_{idx}.json"
        if not args.smoke and tmp_file.exists():
            try:
                with open(tmp_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return (
                        data["idx"],
                        data["pred_label_str"],
                        data["latency_ms"],
                        data["conf"],
                    )
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
                print(
                    f"Warning: failed to load intermediate result for sample {idx}: {e}"
                )

        prompt = (
            "Classify the intent of the following text into one of the category integer labels "
            f"from {unique_classes} based on the category descriptions.\nText: {text}"
        )
        t0 = time.perf_counter()
        try:
            response = chat.send_message(message=prompt)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            if response.text is None:
                raise ValueError("Response text is None")
            result_data = json.loads(response.text)
            pred_label = int(result_data.get("predicted_label", unique_classes[0]))
            conf = float(result_data.get("confidence", 1.0))
            res = (idx, str(pred_label), latency_ms, conf)
        except (ValueError, RuntimeError, TypeError, APIError) as e:
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            print(f"Error at sample {idx}: {e}")
            res = (idx, str(unique_classes[0]), latency_ms, 0.0)

        if not args.smoke:
            try:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "idx": res[0],
                            "pred_label_str": res[1],
                            "latency_ms": res[2],
                            "conf": res[3],
                        },
                        f,
                        ensure_ascii=False,
                    )
            except OSError as e:
                print(
                    f"Warning: failed to save intermediate result for sample {idx}: {e}"
                )

        return res

    latencies = [0.0] * len(X)
    predictions = [""] * len(X)
    probabilities_list = [np.zeros(len(unique_classes))] * len(X)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_sample, (i, t)): i for i, t in enumerate(X)}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(X),
            desc="Parallel inference progress",
        ):
            idx, pred_label_str, latency_ms, conf = future.result()
            pred_label = int(pred_label_str)
            latencies[idx] = latency_ms
            predictions[idx] = pred_label_str

            proba = np.zeros(len(unique_classes))
            if pred_label in class_to_idx:
                c_idx = class_to_idx[pred_label]
                proba[c_idx] = conf
                remaining = (1.0 - conf) / max(1, (len(unique_classes) - 1))
                for i in range(len(unique_classes)):
                    if i != c_idx:
                        proba[i] = remaining
            else:
                proba[:] = 1.0 / len(unique_classes)
            probabilities_list[idx] = proba

    probabilities = np.array(probabilities_list)
    avg_latency_ms = float(np.mean(latencies))

    try:
        y_true_bin = label_binarize(y_true, classes=unique_classes)
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        roc_auc_macro = float(
            roc_auc_score(y_true_bin, probabilities, average="macro", multi_class="ovr")
        )
    except (ValueError, RuntimeError) as e:
        print(
            f"Warning: ROC-AUC calculation failed (possibly due to class subset): {e}"
        )
        roc_auc_macro = 0.0

    print(f"Average Latency: {avg_latency_ms:.2f} ms/sample")
    print(f"ROC-AUC (Macro-average): {roc_auc_macro:.4f}")

    if args.smoke:
        print("Smoke test mode completed successfully. Skipping result file saving.")
        return

    pred_df = df.copy()
    pred_df["predicted_label"] = [int(p) for p in predictions]
    for i, cls in enumerate(unique_classes):
        pred_df[f"prob_{cls}"] = probabilities[:, i]
    pred_df["latency_ms"] = latencies
    predictions_path = output_dir / "predictions.csv"
    pred_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    metrics = {
        "model": model_name,
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
