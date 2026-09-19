# ステップ 1: LightGBM モデル評価の詳細実装手順書

本ドキュメントは、[`docs/experiment_plan.md`](docs/experiment_plan.md) で定義された **ステップ 1 (`01_lightgbm/`)** の完全かつ緻密な実装手順書です。使用するスキル、ライブラリ、ディレクトリ構造、および具体的なコード例を定義します。

---

## 1. 参照するスキル
- **`huggingface-community-evals`**: 再現性のあるPythonスクリプト実行やローカル環境における `uv run` を活用した依存関係・実行管理パターンを参照。
- **AGENTS.md 標準ルール**: 
  - スキル情報を優先して活用すること。
  - 実装には対応するテストを同時に含めること。
  - `ruff check`, `ruff format`, `mypy`, `pytest` によるゼロエラーの維持。

---

## 2. 使用するライブラリ
- **`pandas`**: CSVデータの読み込み、データフレーム操作。
- **`scikit-learn` (`sklearn.feature_extraction.text.TfidfVectorizer`)**: テキストの特徴量化（TF-IDF）。
- **`scikit-learn` (`sklearn.metrics.roc_auc_score`)**: マルチクラス ROC-AUC (Macro-average) の算出。
- **`lightgbm` (`lightgbm.LGBMClassifier`)**: マルチクラス勾配ブースティング分類器。
- **`joblib`**: 学習済みモデルのシリアライズ・保存・ロード。

---

## 3. ディレクトリ構成と成果物
```text
01_lightgbm/
├── input/                   # 00_dataset/output/ へのシンボリックリンク (train.csv, test.csv)
├── tmp/                     # 一時ファイル
├── output/                  # 成果物
│   ├── model.pkl            # 学習済みモデル（TF-IDF + LightGBM パイプライン）
│   ├── predictions.csv      # 推論結果（予測確率、予測ラベル、正解ラベル、レイテンシ）
│   └── metrics.json         # 評価指標（roc_auc_macro, avg_latency_ms）
├── script/                  # 実装スクリプト
│   ├── train.py             # モデル学習用スクリプト
│   └── evaluate.py          # 推論・評価用スクリプト
└── tests/                   # テストコード
    └── test_lightgbm.py     # 単体テスト・統合テスト
```

---

## 4. 想定されるコード設計

### 4.1 学習スクリプト (`01_lightgbm/script/train.py`)
```python
from pathlib import Path
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text.TfidfVectorizer import TfidfVectorizer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


def main():
    input_path = Path("01_lightgbm/input/train.csv")
    output_dir = Path("01_lightgbm/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {input_path}")
    df = pd.read_csv(input_path)

    # データの前提カラム: text, label
    X = df["text"].astype(str).values
    y = df["label"].values

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
```

### 4.2 評価・推論スクリプト (`01_lightgbm/script/evaluate.py`)
```python
from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    input_path = Path("01_lightgbm/input/test.csv")
    model_path = Path("01_lightgbm/output/model.pkl")
    output_dir = Path("01_lightgbm/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}")
    pipeline = joblib.load(model_path)

    print(f"Loading test data from {input_path}")
    df = pd.read_csv(input_path)
    X = df["text"].astype(str).values
    y_true = df["label"].values

    print("Running inference and measuring latency...")
    latencies = []
    probabilities_list = []

    # 1件ずつあるいはバッチで推論しつつレイテンシを計測
    for text in X:
        t0 = time.perf_counter()
        proba = pipeline.predict_proba([text])[0]
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ミリ秒
        probabilities_list.append(proba)

    probabilities = np.array(probabilities_list)
    avg_latency_ms = float(np.mean(latencies))

    # クラス順序の取得
    classes = pipeline.named_steps["clf"].classes_

    # ROC-AUC (Macro-average) の算出
    # y_true が文字列や数値の場合、label_binarize またはそのまま classes を指定
    from sklearn.preprocessing import label_binarize

    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1:
        # 2クラスの場合の処理（必要に応じて）
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
```

### 4.3 テストコード (`01_lightgbm/tests/test_lightgbm.py`)
```python
from pathlib import Path
import pandas as pd
import joblib
from sklearn.feature_extraction.text.TfidfVectorizer import TfidfVectorizer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


def test_lightgbm_pipeline_smoke(tmp_path):
    # ダミーデータの作成
    train_df = pd.DataFrame(
        {
            "text": [
                "account login issue",
                "billing charge problem",
                "password reset request",
                "payment failure error",
            ],
            "label": ["auth", "billing", "auth", "billing"],
        }
    )

    X = train_df["text"].values
    y = train_df["label"].values

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=100)),
            ("clf", LGBMClassifier(random_state=42, n_estimators=2)),
        ]
    )
    pipeline.fit(X, y)

    # 推論テスト
    probas = pipeline.predict_proba(["login problem"])
    assert probas.shape[0] == 1
    assert probas.shape[1] == 2
```

---

## 5. 実行コマンド
```bash
# 1. 依存関係の追加（必要に応じて）
uv add lightgbm scikit-learn joblib pandas

# 2. トレーニング実行
uv run python 01_lightgbm/script/train.py

# 3. 評価・推論実行
uv run python 01_lightgbm/script/evaluate.py

# 4. テスト実行
uv run pytest 01_lightgbm/tests/
```
