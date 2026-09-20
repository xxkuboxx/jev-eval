import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


def test_distilbert_inference_smoke() -> None:
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer is not None

    encoded = tokenizer("test intent classification", return_tensors="pt")
    assert "input_ids" in encoded


def test_distilbert_dynamic_padding() -> None:
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    batch_raw = [
        tokenizer("Short text", truncation=True),
        tokenizer(
            "This is a noticeably longer sentence for padding test",
            truncation=True,
        ),
    ]
    batch_collated = collator(batch_raw)
    assert "input_ids" in batch_collated
    max_len = max(len(x["input_ids"]) for x in batch_raw)
    assert batch_collated["input_ids"].shape[1] == max_len


def test_distilbert_trained_weights_no_nan() -> None:
    model_dir = Path("02_distilbert/output/model_weights")
    if model_dir.exists():
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
        assert not has_nan, "Trained model weights contain NaN values."

    metrics_path = Path("02_distilbert/output/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        assert not torch.isnan(torch.tensor(metrics["roc_auc_macro"])).item()
        assert metrics["roc_auc_macro"] > 0.0


def test_distilbert_label_mapping_format() -> None:
    label_path = Path("02_distilbert/output/label_mapping.json")
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        assert all(isinstance(c, int) for c in classes), "classes must be integers"
        assert classes == sorted(classes), "classes must be sorted ascending"
