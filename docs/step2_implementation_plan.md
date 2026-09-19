# ステップ 2: DistilBERT モデル評価の詳細実装手順書

本ドキュメントは、[`docs/experiment_plan.md`](docs/experiment_plan.md) で定義された **ステップ 2 (`02_distilbert/`)** の完全かつ緻密な実装手順書です。使用するスキル、ライブラリ、ディレクトリ構造、および具体的なコード例を定義します。

---

## 1. 参照するスキル
- **`huggingface-community-evals`**: Hugging Face Hub モデル・Transformers を活用したローカル GPU ファインチューニングおよび推論パイプライン構築のパターンを参照。
- **AGENTS.md 標準ルール**:
  - スキル情報を優先して活用すること。
  - 実装には対応するテストを同時に含めること。
  - `ruff check`, `ruff format`, `mypy`, `pytest` によるゼロエラーの維持。

---

## 2. 使用するライブラリ
- **`pandas`**: CSVデータの読み込み、データフレーム操作。
- **`torch`**: PyTorchによるテンソル演算およびGPUトレーニング。
- **`transformers` (`AutoModelForSequenceClassification`, `AutoTokenizer`, `DataCollatorWithPadding`, `Trainer`, `TrainingArguments`)**: `distilbert/distilbert-base-uncased` のファインチューニングおよび推論。
- **`datasets` (`Dataset`)**: pandas DataFrame から Hugging Face Dataset への変換。
- **`scikit-learn` (`sklearn.metrics.roc_auc_score`, `sklearn.preprocessing.label_binarize`)**: マルチクラス ROC-AUC (Macro-average) の算出。

---

## 3. ディレクトリ構成と成果物
```text
02_distilbert/
├── input/                   # 00_dataset/output/ へのシンボリックリンク (train.csv, test.csv)
├── tmp/                     # 一時ファイル（チェックポイント等）
├── output/                  # 成果物
│   ├── model_weights/       # ファインチューニング済みモデル重みおよびトークナイザー設定
│   ├── label_mapping.json   # 整数クラスリストマッピング
│   ├── predictions.csv      # 推論結果（予測確率、正解ラベル、レイテンシ）
│   └── metrics.json         # 評価指標（roc_auc_macro, avg_latency_ms）
├── script/                  # 実装スクリプト
│   ├── train.py             # モデルファインチューニング用スクリプト
│   └── evaluate.py          # 推論・評価用スクリプト
└── tests/                   # テストコード
    └── test_distilbert.py   # 単体テスト・統合テスト
```

---

## 4. 想定されるコード設計

### 4.1 学習スクリプト (`02_distilbert/script/train.py`)
```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy}


def main():
    input_path = Path("02_distilbert/input/train.csv")
    output_dir = Path("02_distilbert/output")
    model_output_dir = output_dir / "model_weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    classes = sorted([int(c) for c in df["labels"].unique()])
    label_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    df["label_idx"] = df["labels"].map(label_to_idx)
    num_labels = len(classes)

    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    dataset = Dataset.from_pandas(
        df[["text", "label_idx"]].rename(columns={"label_idx": "labels"})
    )
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_train = split_dataset["train"].map(tokenize_function, batched=True)
    tokenized_eval = split_dataset["test"].map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir="02_distilbert/tmp",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=4e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_steps=100,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        label_smoothing_factor=0.05,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        save_total_limit=1,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
```

### 4.2 評価・推論スクリプト (`02_distilbert/script/evaluate.py`)
```python
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    input_path = Path("02_distilbert/input/test.csv")
    model_dir = Path("02_distilbert/output/model_weights")
    output_dir = Path("02_distilbert/output")

    with open(output_dir / "label_mapping.json", "r", encoding="utf-8") as f:
        classes = [int(c) for c in json.load(f)]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    df = pd.read_csv(input_path)
    texts = df["text"].astype(str).tolist()
    y_true = df["labels"].astype(int).values

    latencies = []
    probabilities_list = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        probabilities_list.append(probs)

    probabilities = np.array(probabilities_list)
    avg_latency_ms = float(np.mean(latencies))

    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    roc_auc_macro = float(
        roc_auc_score(y_true_bin, probabilities, average="macro", multi_class="ovr")
    )

    pred_df = df.copy()
    for i, cls in enumerate(classes):
        pred_df[f"prob_{cls}"] = probabilities[:, i]
    pred_df["latency_ms"] = latencies
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    metrics = {
        "model": "distilbert-base-uncased",
        "roc_auc_macro": roc_auc_macro,
        "avg_latency_ms": avg_latency_ms,
        "total_samples": len(df),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
```

---

## 5. 実行コマンドと README 要件

### 5.1 実行コマンド
```bash
# 1. 依存関係の同期
uv sync

# 2. ファインチューニング実行（GPU環境推奨）
uv run python 02_distilbert/script/train.py

# 3. 評価・推論実行
uv run python 02_distilbert/script/evaluate.py

# 4. テスト実行
uv run pytest 02_distilbert/tests/
```
