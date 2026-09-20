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
        "what": "Opening greetings, saying hello, or starting a conversation (Greeting)",
        "not_for": "Saying goodbye, casual chit-chat about one's day without greeting, expressing gratitude, asking for help",
        "examples": ["Hello", "Hi there!", "Good morning", "Hi, nice to meet you"],
    },
    "1": {
        "what": "Ending the conversation, taking leave, parting ways, or saying goodbye (Goodbye / Farewell)",
        "not_for": "Initial greetings, expressing gratitude alone, casual chat",
        "examples": ["Goodbye!", "See you later", "Bye for now", "Take care, farewell"],
    },
    "2": {
        "what": "Expressing appreciation, thanks, or gratitude for assistance received (Expression of Gratitude)",
        "not_for": "Mere agreement, parting without thanks, general praise without thanking",
        "examples": [
            "Thank you so much",
            "Thanks for your help",
            "I really appreciate your assistance",
            "Thanks a million",
        ],
    },
    "3": {
        "what": "Confirming, agreeing, accepting, acknowledging, or giving affirmative responses (Agreement / Affirmation)",
        "not_for": "Expressing disagreement, asking a question, expressing gratitude",
        "examples": [
            "Yes, that sounds good",
            "That's correct",
            "I agree with that",
            "Sounds good to me",
        ],
    },
    "4": {
        "what": "Disagreeing with a statement, declining, saying no, or stating something is incorrect (Disagreement / Discontent)",
        "not_for": "Saying goodbye, constructive feature suggestions, system bugs or product complaints",
        "examples": [
            "No, I disagree",
            "That is not correct",
            "That's not what I meant",
            "I cannot agree with that",
        ],
    },
    "5": {
        "what": "Casual conversation, small talk, asking how someone is doing, commenting on day/weather/hobbies (Small Talk / Casual Chat)",
        "not_for": "Formal opening greetings alone, technical questions about system features, complaints",
        "examples": [
            "How is your day going?",
            "Did you catch the game last night?",
            "Nice weather we're having",
            "What have you been up to lately?",
        ],
    },
    "6": {
        "what": "Inquiring about what the AI/system can do, its general features, capabilities, or limitations (System Capabilities Inquiry)",
        "not_for": "Inquiries specifically about language translation or supported languages, asking for detailed help on a topic",
        "examples": [
            "What are your capabilities?",
            "What can you do?",
            "What features do you support?",
            "Are you able to integrate with calendar apps?",
        ],
    },
    "7": {
        "what": "Complimenting, praising, giving high ratings, or expressing admiration and positive feedback (Praise / Positive Feedback)",
        "not_for": "Simple confirmation or agreement, basic thank-you without praise",
        "examples": [
            "You did a fantastic job!",
            "Your customer service is top-notch",
            "I am really impressed with the service",
            "Excellent work!",
        ],
    },
    "8": {
        "what": "Expressing frustration, dissatisfaction, disappointment, poor quality reports, or negative feedback (Complaint / Dissatisfaction)",
        "not_for": "General disagreement with an opinion, purely constructive feature suggestions without expressing a problem",
        "examples": [
            "This service is terrible",
            "I am very disappointed with the quality",
            "The response was frustratingly slow",
            "I have some negative feedback",
        ],
    },
    "9": {
        "what": "Requesting assistance, clarification, more detailed explanations, elaboration, or asking to repeat (Help / Clarification Request)",
        "not_for": "High-level questions about system capabilities, language switching requests, complaints",
        "examples": [
            "Could you explain that in more detail?",
            "Can you clarify what you mean?",
            "Could you provide more information on this?",
            "Please explain how this works",
        ],
    },
    "10": {
        "what": "Suggesting new features, enhancements, user interface improvements, or constructive recommendations (Feature Suggestion / Constructive Feedback)",
        "not_for": "Complaining without proposing an enhancement, asking general questions, simple disagreement",
        "examples": [
            "You should add a dark mode option",
            "I suggest adding a feature to save preferences",
            "It would be great if you could support image uploads",
            "Here is a suggestion to improve the UI",
        ],
    },
    "11": {
        "what": "Inquiring about supported languages, requesting to change conversational language, or adjusting language settings (Language / Settings)",
        "not_for": "General system capability questions unrelated to language, general greetings",
        "examples": [
            "Can you speak Spanish?",
            "What languages do you support?",
            "Could we switch to French?",
            "Please respond in Japanese",
        ],
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
