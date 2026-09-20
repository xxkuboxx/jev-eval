import argparse
import concurrent.futures
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from typesafe_sdk import Choice, TypeSafeClient

MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 32.0

CHOICE_CRITERIA = {
    "0": {
        "what": "会話の開始、挨拶、体調や機嫌を尋ねる日常会話 (Greeting / Small Talk)",
        "not_for": "別れの挨拶、感謝の表現、不満や苦情",
        "examples": ["こんにちは", "おはようございます", "元気ですか？"],
    },
    "1": {
        "what": "会話の終了、退席、別れの挨拶 (Goodbye / Farewell)",
        "not_for": "会話の開始、感謝のみの表現",
        "examples": ["さようなら", "またね", "おやすみなさい", "これで失礼します"],
    },
    "2": {
        "what": "相手への感謝やお礼、助かったことへの謝意 (Expression of Gratitude)",
        "not_for": "挨拶のみ、同意のみ、新たな質問",
        "examples": [
            "ありがとう",
            "感謝します",
            "助かりました",
            "ありがとうございます",
        ],
    },
    "3": {
        "what": "相手の発言への肯定、同意、承諾、了解 (Agreement / Affirmation)",
        "not_for": "感謝、否定や不満、質問",
        "examples": ["はい", "了解しました", "その通りです", "承知いたしました"],
    },
    "4": {
        "what": "相手の発言や提案に対する拒否、否定、不同意 (Disagreement / Discontent)",
        "not_for": "単なる別れ、建設的な機能改善提案",
        "examples": ["いいえ", "違います", "納得できません", "そうではありません"],
    },
    "5": {
        "what": "新しい雑談テーマの提案、趣味や世間話の投げかけ (Topic Suggestion / Small Talk)",
        "not_for": "挨拶のみ、機能や仕組みへの質問、業務要求",
        "examples": ["最近面白い映画ある？", "今日の天気はどうかな", "何か雑談しよう"],
    },
    "6": {
        "what": "システムやAIの機能、仕様、できることに関する質問 (System Capabilities Inquiry)",
        "not_for": "単なる雑談、助けの要請、改善提案",
        "examples": ["あなたは何ができますか？", "どんな形式のファイルを扱えますか？"],
    },
    "7": {
        "what": "回答や成果に対する褒め言葉、高い評価 (Praise / Positive Feedback)",
        "not_for": "単なる了解や同意、挨拶",
        "examples": ["素晴らしい回答です", "とても優秀ですね", "完璧です"],
    },
    "8": {
        "what": "回答の質やバグ、動作に対する苦情・強い不満 (Complaint / Dissatisfaction)",
        "not_for": "建設的な改善要望、単なる否定回答",
        "examples": [
            "回答が的外れで役に立たない",
            "バグだらけで動かない",
            "ひどい品質だ",
        ],
    },
    "9": {
        "what": "詳細な解説、使い方、ヘルプ、やり直しの要求 (Help / Clarification Request)",
        "not_for": "システムの仕様質問、機能改善の要望",
        "examples": [
            "もう少し詳しく説明して",
            "具体例を教えて",
            "やり方を助けてほしい",
        ],
    },
    "10": {
        "what": "システムの改良案、新機能やUIの追加要望 (Feature Suggestion / Constructive Feedback)",
        "not_for": "単なる不満や愚痴、質問",
        "examples": ["この機能を追加してほしい", "UIをもっとシンプルにしてほしい"],
    },
    "11": {
        "what": "出力言語の切り替え、口調や設定の変更要求 (Language / Settings Change Request)",
        "not_for": "質問、感謝、挨拶",
        "examples": ["英語で答えてください", "敬語をやめて", "設定をリセットして"],
    },
}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Evaluate Jev (jev-1.13) on intent classification test data."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test mode on the first 10 samples only (does not save output files).",
    )
    args = parser.parse_args()

    if not os.environ.get("TYPESAFE_API_KEY"):
        raise OSError(
            "Error: TYPESAFE_API_KEY environment variable is not set. Please configure it in .env or your environment."
        )

    input_path = Path("04_jev/input/test.csv")
    output_dir = Path("04_jev/output")
    tmp_dir = Path("04_jev/tmp")
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

    criteria_for_classes = {
        str(cls): CHOICE_CRITERIA[str(cls)]
        for cls in unique_classes
        if str(cls) in CHOICE_CRITERIA
    }

    choice_question = Choice(
        instructions="Which intent category best classifies the intent expressed in `text`?",
        criteria=criteria_for_classes,
    )

    model_name = "jev-1.13"
    print(f"Running inference with {model_name} (parallel max_workers=10)...")

    def process_sample(
        item: tuple[int, str],
    ) -> tuple[int, str, float, float, dict[str, float]]:
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
                        data["probabilities"],
                    )
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
                print(
                    f"Warning: failed to load intermediate result for sample {idx}: {e}"
                )

        last_exception: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.perf_counter()
                with TypeSafeClient() as client:
                    response = client.system_one(
                        state={"text": text},
                        questions={"intent": choice_question},
                    )
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0

                if "intent" not in response.answers:
                    raise KeyError(
                        f"Missing 'intent' key in response.answers for sample {idx}"
                    )
                answer = response.answers["intent"]
                if not hasattr(answer, "choice") or answer.choice is None:
                    raise AttributeError(
                        f"Missing 'choice' attribute in answer for sample {idx}"
                    )
                pred_choice = str(answer.choice)
                conf = float(getattr(answer, "confidence", 1.0))
                raw_probs = getattr(answer, "probabilities", {})
                if not raw_probs:
                    raise ValueError(
                        f"Empty 'probabilities' in answer for sample {idx}"
                    )
                probs = {str(k): float(v) for k, v in raw_probs.items()}
                res = (idx, pred_choice, latency_ms, conf, probs)
                break
            except Exception as e:
                last_exception = e
                if attempt == MAX_RETRIES:
                    print(
                        f"Fatal error: sample {idx} failed after {MAX_RETRIES} attempts: {e}"
                    )
                    raise
                delay = min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
                jitter = random.uniform(0.0, 0.5 * delay)
                sleep_time = delay + jitter
                print(
                    f"Warning: sample {idx} attempt {attempt} failed ({e}). Retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)
        else:
            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"Sample {idx} failed unexpectedly.")

        if not args.smoke:
            try:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "idx": res[0],
                            "pred_label_str": res[1],
                            "latency_ms": res[2],
                            "conf": res[3],
                            "probabilities": res[4],
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
    class_to_idx = {str(cls): i for i, cls in enumerate(unique_classes)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_sample, (i, t)): i for i, t in enumerate(X)}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(X),
            desc="Jev inference progress",
        ):
            idx, pred_label_str, latency_ms, _conf, probs = future.result()
            latencies[idx] = latency_ms
            predictions[idx] = pred_label_str

            proba = np.zeros(len(unique_classes))
            for cls_str, p_val in probs.items():
                if cls_str in class_to_idx:
                    proba[class_to_idx[cls_str]] = p_val
            p_sum = proba.sum()
            if p_sum <= 0:
                raise ValueError(
                    f"Sum of probabilities for sample {idx} is non-positive ({p_sum})"
                )
            proba /= p_sum
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
