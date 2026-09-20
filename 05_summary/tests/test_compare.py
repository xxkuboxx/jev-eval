import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def test_calculate_pareto_frontier() -> None:
    spec = importlib.util.spec_from_file_location(
        "compare", "05_summary/script/compare.py"
    )
    assert spec is not None
    assert spec.loader is not None
    compare_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare_mod)

    dummy_models = [
        {"model_key": "A", "roc_auc_macro": 0.99, "avg_latency_ms": 5.0},
        {"model_key": "B", "roc_auc_macro": 0.98, "avg_latency_ms": 4.0},
        {"model_key": "C", "roc_auc_macro": 0.97, "avg_latency_ms": 10.0},
        {"model_key": "D", "roc_auc_macro": 0.995, "avg_latency_ms": 15.0},
    ]

    pareto_keys = compare_mod.calculate_pareto_frontier(dummy_models)
    assert set(pareto_keys) == {"A", "B", "D"}
    assert "C" not in pareto_keys


def test_cost_catalogs() -> None:
    spec = importlib.util.spec_from_file_location(
        "compare", "05_summary/script/compare.py"
    )
    assert spec is not None
    assert spec.loader is not None
    compare_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare_mod)

    # API コストカタログの検証
    api_catalog = compare_mod.API_COST_CATALOG
    assert "03_gemini_flash_lite" in api_catalog
    assert "04_jev" in api_catalog

    gemini_cost = api_catalog["03_gemini_flash_lite"]
    assert gemini_cost["input_cost_per_1m"] == 0.30
    assert gemini_cost["output_cost_per_1m"] == 2.50

    jev_cost = api_catalog["04_jev"]
    assert jev_cost["input_cost_per_1m"] == 0.042
    assert jev_cost["output_cost_per_1m"] == 0.00

    # サーバーデプロイ型モデルカタログの検証
    server_catalog = compare_mod.SERVER_DEPLOYED_CATALOG
    assert "01_lightgbm" in server_catalog
    assert "02_distilbert" in server_catalog


def test_load_model_data_success(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "compare", "05_summary/script/compare.py"
    )
    assert spec is not None
    assert spec.loader is not None
    compare_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare_mod)

    model_dir = tmp_path / "04_jev"
    model_dir.mkdir()

    metrics_file = model_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "jev-1.13",
                "roc_auc_macro": 0.97,
                "avg_latency_ms": 500.0,
                "total_samples": 5,
            },
            f,
        )

    predictions_file = model_dir / "predictions.csv"
    df = pd.DataFrame(
        {
            "labels": [0, 1, 0, 1, 0],
            "predicted_label": [0, 1, 0, 0, 0],
            "latency_ms": [480.0, 490.0, 500.0, 510.0, 520.0],
        }
    )
    df.to_csv(predictions_file, index=False)

    data = compare_mod.load_model_data(
        model_key="04_jev",
        display_name="Jev (jev-1.13)",
        category="判断特化 System One API",
        base_dir=tmp_path,
    )

    assert data["model_key"] == "04_jev"
    assert data["roc_auc_macro"] == 0.97
    assert data["accuracy"] == 0.8
    assert data["min_latency_ms"] == 480.0
    assert data["p25_latency_ms"] == 490.0
    assert data["p50_latency_ms"] == 500.0
    assert data["p75_latency_ms"] == 510.0
    assert data["input_cost_per_1m"] == 0.042
    assert data["output_cost_per_1m"] == 0.00


def test_fail_first_missing_files(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "compare", "05_summary/script/compare.py"
    )
    assert spec is not None
    compare_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(compare_mod)

    # フォルダが存在しない場合
    with pytest.raises(FileNotFoundError):
        compare_mod.load_model_data("non_existent", "None", "None", tmp_path)


def test_fail_first_missing_keys(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "compare", "05_summary/script/compare.py"
    )
    assert spec is not None
    compare_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(compare_mod)

    model_dir = tmp_path / "01_lightgbm"
    model_dir.mkdir()

    # roc_auc_macro が欠落している metrics.json
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "lightgbm",
                "avg_latency_ms": 4.0,
                "total_samples": 5,
            },
            f,
        )

    df = pd.DataFrame(
        {
            "labels": [0, 1],
            "predicted_label": [0, 1],
            "latency_ms": [3.0, 4.0],
        }
    )
    df.to_csv(model_dir / "predictions.csv", index=False)

    with pytest.raises(KeyError):
        compare_mod.load_model_data("01_lightgbm", "LGBM", "ML", tmp_path)


def test_fail_first_nan_detection(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "compare", "05_summary/script/compare.py"
    )
    assert spec is not None
    compare_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(compare_mod)

    model_dir = tmp_path / "01_lightgbm"
    model_dir.mkdir()

    # predictions.csv に NaN が混入
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "lightgbm",
                "roc_auc_macro": 0.99,
                "avg_latency_ms": 4.0,
                "total_samples": 2,
            },
            f,
        )

    df = pd.DataFrame(
        {
            "labels": [0, 1],
            "predicted_label": [0, None],  # NaN
            "latency_ms": [3.0, 4.0],
        }
    )
    df.to_csv(model_dir / "predictions.csv", index=False)

    with pytest.raises(ValueError):
        compare_mod.load_model_data("01_lightgbm", "LGBM", "ML", tmp_path)


def test_fail_first_row_count_mismatch(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "compare", "05_summary/script/compare.py"
    )
    assert spec is not None
    compare_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(compare_mod)

    model_dir = tmp_path / "01_lightgbm"
    model_dir.mkdir()

    # total_samples=5 に対して CSV が 2 行のみ
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "lightgbm",
                "roc_auc_macro": 0.99,
                "avg_latency_ms": 4.0,
                "total_samples": 5,
            },
            f,
        )

    df = pd.DataFrame(
        {
            "labels": [0, 1],
            "predicted_label": [0, 1],
            "latency_ms": [3.0, 4.0],
        }
    )
    df.to_csv(model_dir / "predictions.csv", index=False)

    with pytest.raises(ValueError):
        compare_mod.load_model_data("01_lightgbm", "LGBM", "ML", tmp_path)


def test_compare_main_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    model_keys = ["01_lightgbm", "02_distilbert", "03_gemini_flash_lite", "04_jev"]
    for i, k in enumerate(model_keys):
        m_dir = input_dir / k
        m_dir.mkdir()
        with open(m_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": f"model_{i}",
                    "roc_auc_macro": 0.90 + i * 0.02,
                    "avg_latency_ms": (i + 1) * 10.0,
                    "total_samples": 10,
                },
                f,
            )
        df = pd.DataFrame(
            {
                "labels": [0, 1] * 5,
                "predicted_label": [0, 1] * 5,
                "latency_ms": [(i + 1) * 10.0] * 10,
            }
        )
        df.to_csv(m_dir / "predictions.csv", index=False)

    spec = importlib.util.spec_from_file_location(
        "compare", "05_summary/script/compare.py"
    )
    assert spec is not None
    assert spec.loader is not None
    compare_mod = importlib.util.module_from_spec(spec)

    test_args = [
        "compare.py",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    spec.loader.exec_module(compare_mod)
    compare_mod.main()

    report_file = output_dir / "comparison_report.md"
    summary_file = output_dir / "summary_metrics.json"

    assert report_file.exists()
    assert summary_file.exists()

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_json = json.load(f)
        assert len(summary_json["models"]) == 4
        assert len(summary_json["pareto_optimal_keys"]) >= 1
        assert "api_cost_catalog" in summary_json
        assert "server_deployed_catalog" in summary_json
        for m in summary_json["models"]:
            assert "min_latency_ms" in m
            assert "p25_latency_ms" in m
            assert "p75_latency_ms" in m

    report_text = report_file.read_text(encoding="utf-8")
    assert "LightGBM" in report_text
    assert "Jev" in report_text
    assert "Gemini 3.5 Flash Lite" in report_text
    assert "$0.042" in report_text
    assert "$0.00" in report_text
    assert "パレートフロンティア" in report_text
    assert "最小値 (ms)" in report_text
    assert "p25 (ms)" in report_text
    assert "p75 (ms)" in report_text
