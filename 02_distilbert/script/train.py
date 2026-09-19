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


def main() -> None:
    input_path = Path("02_distilbert/input/train.csv")
    output_dir = Path("02_distilbert/output")
    model_output_dir = output_dir / "model_weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {input_path}")
    df = pd.read_csv(input_path)

    # ラベルマッピングの作成（整数型のまま昇順ソートして保存）
    classes = sorted([int(c) for c in df["labels"].unique()])
    label_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    df["label_idx"] = df["labels"].map(label_to_idx)
    num_labels = len(classes)

    # ラベルマッピングの保存（推論時に使用するため）
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    # Hugging Face Dataset へ変換 (Trainer が認識する標準カラム名 'labels' に設定)
    dataset = Dataset.from_pandas(
        df[["text", "label_idx"]].rename(columns={"label_idx": "labels"})
    )

    # 訓練/検証データ分割 (90% / 10%)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    model_name = "distilbert/distilbert-base-uncased"
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_train = split_dataset["train"].map(tokenize_function, batched=True)
    tokenized_eval = split_dataset["test"].map(tokenize_function, batched=True)

    # 動的パディング用 DataCollator (各ミニバッチ内の最長に合わせてパディング)
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

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Starting DistilBERT fine-tuning on: {device_name}...")
    trainer.train()

    print(f"Saving model weights to {model_output_dir}")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)


if __name__ == "__main__":
    main()
