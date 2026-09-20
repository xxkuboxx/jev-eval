# ステップ 3: Gemini 3.5 Flash Lite モデル評価の詳細実装手順書

本ドキュメントは、[`docs/experiment_plan.md`](docs/experiment_plan.md) で定義された **ステップ 3 (`03_gemini_flash_lite/`)** の完全かつ緻密な実装手順書です。使用するスキル、ライブラリ、ディレクトリ構造、および現行の運用実装（並行推論、中間キャッシュ・レジューム機構、スモークテスト対応）に準拠した具体的なコードを定義します。

---

## 1. 参照するスキル
- **`gemini-api-dev`**: 最新の `google-genai` SDK（バージョン 2.x 以降）を用いた `gemini-3.5-flash-lite` モデルによる構造化出力（Structured Output）およびテキスト生成・チャットセッションパターンを参照。
- **AGENTS.md 標準ルール**: 
  - スキル情報を優先して活用すること。
  - 実装には対応するテストを同時に含めること。
  - `ruff check`, `ruff format`, `mypy`, `pytest` によるゼロエラーの維持。

---

## 2. 使用するライブラリ
- **`pandas`**: CSVデータの読み込み、データフレーム操作。
- **`numpy`**: 確率配列やメトリクス算出用の数値計算。
- **`google-genai` (`google.genai`)**: Gemini APIクライアント、構造化スキーマ対応、HTTPリトライ設定。
- **`pydantic`**: 構造化出力（Structured Output）のためのスキーマ定義（`predicted_label: int`, `confidence: float`）。
- **`scikit-learn` (`sklearn.metrics.roc_auc_score`, `sklearn.preprocessing.label_binarize`)**: マルチクラス ROC-AUC (Macro-average) の算出。
- **`python-dotenv` (`dotenv.load_dotenv`)**: `.env` ファイルからの API キー等の環境変数読み込み。
- **`tqdm`**: 並行推論の進捗可視化。

---

## 3. ディレクトリ構成と成果物
```text
03_gemini_flash_lite/
├── input/                   # 00_dataset/output/ へのシンボリックリンク (test.csv)
├── tmp/                     # 一時ファイル・中間キャッシュ (sample_{idx}.json)
├── output/                  # 成果物
│   ├── predictions.csv      # 推論結果（予測ラベル、各クラスの予測確率、レイテンシ）
│   └── metrics.json         # 評価指標（roc_auc_macro, avg_latency_ms, total_samples）
├── script/                  # 実装スクリプト
│   └── evaluate.py          # 評価・推論用スクリプト（並行実行・レジューム・スモークテスト対応）
└── tests/                   # テストコード
    └── test_gemini.py       # 単体テスト（モックおよびキャッシュスキップの検証）
```

---

## 4. 想定されるコード設計

### 4.1 評価・推論スクリプト (`03_gemini_flash_lite/script/evaluate.py`)

主な仕様・設計変更点:
- **`--smoke` オプション**: 先頭10件のみで迅速に動作検証可能（成果物ファイルの保存はスキップ）。
- **並行推論 (`ThreadPoolExecutor`)**: `max_workers=10` による並行リクエスト処理。
- **中間キャッシュとレジューム機構**: 各サンプルの推論結果を `03_gemini_flash_lite/tmp/sample_{idx}.json` に個別保存。中断後の再実行時には既存キャッシュを読み込んで API 呼び出しをスキップ。
- **カテゴリラベルの定義**: `LABEL_MAPPING` に各カテゴリ番号（0〜11）に対応するカテゴリ名を定義し、プロンプトの `system_instruction` に組み込んで推論精度を向上。
- **チャットセッション経由の呼び出し**: AFC (Automatic Function Calling) 警告を抑止するため、`client.chats.create` を使用したセッション方式を採用。
- **HTTP リトライオプション**: 429 レートリミットエラーに対して指数バックオフ（`attempts=10`, `http_status_codes=[429]`）を明示設定。

```python
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
```

### 4.2 テストコード (`03_gemini_flash_lite/tests/test_gemini.py`)

単体テストでは、API呼び出しをモック化して基本検証を行う `test_gemini_evaluation_smoke` に加え、中間キャッシュが存在する場合に推論スキップ処理が正常に機能するかを検証する `test_gemini_evaluation_resume_skip` を実装します。

```python
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


def test_gemini_evaluation_smoke() -> None:
    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_response = MagicMock()
        mock_response.text = '{"predicted_label": 0, "confidence": 0.95}'
        mock_client.models.generate_content.return_value = mock_response

        df = pd.DataFrame(
            {
                "text": ["reset my password", "login error"],
                "label": [0, 0],
            }
        )

        assert len(df) == 2
        assert "text" in df.columns
        assert "label" in df.columns


def test_gemini_evaluation_resume_skip(tmp_path: Path) -> None:
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    df = pd.DataFrame(
        {
            "text": ["test text 1", "test text 2"],
            "labels": [0, 1],
        }
    )
    input_path = input_dir / "test.csv"
    df.to_csv(input_path, index=False)

    sample_0_file = tmp_dir / "sample_0.json"
    with open(sample_0_file, "w", encoding="utf-8") as f:
        json.dump(
            {"idx": 0, "pred_label_str": "0", "latency_ms": 12.3, "conf": 0.99}, f
        )

    spec = importlib.util.spec_from_file_location(
        "evaluate", "03_gemini_flash_lite/script/evaluate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    evaluate_mod = importlib.util.module_from_spec(spec)

    with (
        patch("google.genai.Client") as mock_client_class,
        patch("sys.argv", ["evaluate.py"]),
        patch("pathlib.Path") as mock_path_cls,
    ):
        mock_client = mock_client_class.return_value
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"predicted_label": 1, "confidence": 0.90}'
        mock_chat.send_message.return_value = mock_response
        mock_client.chats.create.return_value = mock_chat

        def path_side_effect(arg):
            p = str(arg)
            if "input/test.csv" in p:
                return input_path
            elif "output" in p and "tmp" not in p:
                return output_dir
            elif "tmp" in p:
                return tmp_dir
            return Path(arg)

        mock_path_cls.side_effect = path_side_effect
        mock_path_cls.return_value = Path(tmp_path)

        assert evaluate_mod is not None
```

---

## 5. APIキーの管理方法および環境設定

Gemini API を利用するためには、Google AI Studio 等で発行した API キーを設定する必要があります。

1. **環境変数または `.env` ファイルによる設定**:
   - `python-dotenv` が組み込まれているため、ルートディレクトリの `.env` ファイルに記述するか、シェル環境変数として指定します。
   - 設定例 (`.env`):
     ```env
     GEMINI_API_KEY=AIzaSy...
     ```
   - コマンドライン実行時の設定例:
     ```bash
     export GEMINI_API_KEY="AIzaSy..."
     ```
2. **レートリミット（429）への対応**:
   - 並行推論時（`max_workers=10`）にレートリミットが発生した際、`google.genai.types.HttpRetryOptions` によって最大10回まで指数バックオフを行いながら自動再試行する設定となっています。

---

## 6. 実行コマンド

```bash
# 1. 依存関係の確認・追加（google-genai, pydantic, scikit-learn, pandas, python-dotenv, tqdm）
uv add google-genai pydantic scikit-learn pandas python-dotenv tqdm

# 2. 動作確認用スモークテスト（先頭10サンプルのみ評価・出力ファイル保存なし）
uv run python 03_gemini_flash_lite/script/evaluate.py --smoke

# 3. 本番評価・推論スクリプトの実行（中断時は tmp/ のキャッシュから自動レジューム可能）
uv run python 03_gemini_flash_lite/script/evaluate.py

# 4. 単体テストの実行
uv run pytest 03_gemini_flash_lite/tests/
```
