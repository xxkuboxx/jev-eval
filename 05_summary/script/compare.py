import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# API モデルのトークン課金単価定義 (/1M tokens)
API_COST_CATALOG: dict[str, dict[str, Any]] = {
    "03_gemini_flash_lite": {
        "model_name": "Gemini 3.5 Flash-Lite",
        "input_cost_per_1m": 0.30,
        "output_cost_per_1m": 2.50,
        "pricing_type": "API 従量課金",
        "role": "汎用LLM（テキスト生成、要約、マルチモーダル処理）",
    },
    "04_jev": {
        "model_name": "TypeSafe Jev",
        "input_cost_per_1m": 0.042,
        "output_cost_per_1m": 0.00,
        "pricing_type": "API 従量課金",
        "role": "判定特化（Yes/No、複数選択肢の分類、スコアリング）",
    },
}

# サーバーデプロイ型モデルのコスト定義
SERVER_DEPLOYED_CATALOG: dict[str, dict[str, Any]] = {
    "01_lightgbm": {
        "model_name": "LightGBM",
        "pricing_type": "サーバー固定費（CPUインスタンス）",
        "role": "従来型機械学習（TF-IDF + 勾配ブースティング）",
    },
    "02_distilbert": {
        "model_name": "DistilBERT",
        "pricing_type": "サーバー固定費（GPUインスタンス）",
        "role": "Transformer ローカル推論",
    },
}


def validate_finite_number(
    val: Any, name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """数値が有限の実数であり、指定された範囲内にあることを厳格に検証する (Fail First)。"""
    if not isinstance(val, (int, float)):
        raise TypeError(
            f"Validation failed: '{name}' must be numeric, got {type(val)} ({val})"
        )
    f_val = float(val)
    if math.isnan(f_val) or math.isinf(f_val):
        raise ValueError(f"Validation failed: '{name}' is NaN or Inf ({f_val})")
    if min_val is not None and f_val < min_val:
        raise ValueError(f"Validation failed: '{name}' ({f_val}) must be >= {min_val}")
    if max_val is not None and f_val > max_val:
        raise ValueError(f"Validation failed: '{name}' ({f_val}) must be <= {max_val}")
    return f_val


def calculate_pareto_frontier(
    models_data: list[dict[str, Any]],
) -> list[str]:
    """ROC-AUC (最大化) と 平均レイテンシ (最小化) のパレート最適解 (非劣モデル) を特定する。"""
    if not models_data:
        raise ValueError("Cannot calculate Pareto frontier on empty models data.")

    for m in models_data:
        validate_finite_number(
            m["roc_auc_macro"], f"{m['model_key']} roc_auc_macro", 0.0, 1.0
        )
        validate_finite_number(
            m["avg_latency_ms"], f"{m['model_key']} avg_latency_ms", 0.0
        )

    pareto_models = []
    for candidate in models_data:
        dominated = False
        for other in models_data:
            if candidate["model_key"] == other["model_key"]:
                continue
            better_or_equal_auc = other["roc_auc_macro"] >= candidate["roc_auc_macro"]
            better_or_equal_lat = other["avg_latency_ms"] <= candidate["avg_latency_ms"]
            strictly_better = (
                other["roc_auc_macro"] > candidate["roc_auc_macro"]
                or other["avg_latency_ms"] < candidate["avg_latency_ms"]
            )
            if better_or_equal_auc and better_or_equal_lat and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_models.append(candidate["model_key"])

    if not pareto_models:
        raise RuntimeError(
            "Pareto frontier calculation unexpectedly returned no optimal models."
        )
    return pareto_models


def load_model_data(
    model_key: str, display_name: str, category: str, base_dir: Path
) -> dict[str, Any]:
    """単一モデルの metrics.json および predictions.csv から統計量を抽出・厳格バリデーションする (Fail First)。"""
    model_dir = base_dir / model_key
    metrics_file = model_dir / "metrics.json"
    predictions_file = model_dir / "predictions.csv"

    if not metrics_file.exists():
        # Fallback to direct path if symlink/dir fails in windows python env
        alt_metrics_file = Path(model_key) / "output" / "metrics.json"
        alt_predictions_file = Path(model_key) / "output" / "predictions.csv"
        if alt_metrics_file.exists() and alt_predictions_file.exists():
            metrics_file = alt_metrics_file
            predictions_file = alt_predictions_file
        else:
            raise FileNotFoundError(
                f"Fail First: Missing metrics.json for '{model_key}' at {metrics_file}"
            )
    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Fail First: Missing predictions.csv for '{model_key}' at {predictions_file}"
        )

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 必須キーの完全性検証 (Fail First)
    required_metric_keys = [
        "model",
        "roc_auc_macro",
        "avg_latency_ms",
        "total_samples",
    ]
    for k in required_metric_keys:
        if k not in metrics:
            raise KeyError(f"Fail First: Missing required key '{k}' in {metrics_file}")

    roc_auc_macro = validate_finite_number(
        metrics["roc_auc_macro"], f"{model_key} roc_auc_macro", 0.0, 1.0
    )
    avg_latency_ms = validate_finite_number(
        metrics["avg_latency_ms"], f"{model_key} avg_latency_ms", 0.0
    )
    total_samples = int(metrics["total_samples"])
    if total_samples <= 0:
        raise ValueError(
            f"Fail First: total_samples must be positive, got {total_samples}"
        )

    # predictions.csv の厳格なバリデーション (Fail First)
    df = pd.read_csv(predictions_file)
    if len(df) != total_samples:
        raise ValueError(
            f"Fail First: Row count mismatch in {predictions_file}. "
            f"Expected {total_samples} rows from metrics.json, but got {len(df)} rows."
        )

    required_columns = ["labels", "latency_ms"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(
                f"Fail First: Missing required column '{col}' in {predictions_file}"
            )

    # NaN / Null の完全排除 (Fail First)
    for col in required_columns:
        if df[col].isna().any():
            raise ValueError(
                f"Fail First: Column '{col}' in {predictions_file} contains NaN or null values."
            )

    # 予測ラベルの取得 (predicted_label または prob_* から argmax)
    predicted_labels: Any
    if "predicted_label" in df.columns:
        if df["predicted_label"].isna().any():
            raise ValueError(
                f"Fail First: Column 'predicted_label' in {predictions_file} contains NaN or null values."
            )
        predicted_labels = df["predicted_label"]
    else:
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if not prob_cols:
            raise KeyError(
                f"Fail First: Neither 'predicted_label' nor 'prob_*' columns found in {predictions_file}"
            )
        if df[prob_cols].isna().any().any():
            raise ValueError(
                f"Fail First: Probability columns in {predictions_file} contain NaN or null values."
            )
        predicted_labels = df[prob_cols].values.argmax(axis=1)

    accuracy = float(accuracy_score(df["labels"], predicted_labels))
    validate_finite_number(accuracy, f"{model_key} accuracy", 0.0, 1.0)

    macro_f1 = float(
        f1_score(
            df["labels"],
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )
    validate_finite_number(macro_f1, f"{model_key} macro_f1", 0.0, 1.0)

    latencies = np.array(df["latency_ms"].tolist(), dtype=float)
    if (
        np.any(np.isnan(latencies))
        or np.any(np.isinf(latencies))
        or np.any(latencies <= 0.0)
    ):
        raise ValueError(
            f"Fail First: latency_ms in {predictions_file} contains non-positive, NaN, or Inf values."
        )

    min_latency = float(np.min(latencies))
    p25_latency = float(np.percentile(latencies, 25))
    p50_latency = float(np.percentile(latencies, 50))
    p75_latency = float(np.percentile(latencies, 75))
    p90_latency = float(np.percentile(latencies, 90))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))
    std_latency = float(np.std(latencies))

    validate_finite_number(min_latency, f"{model_key} min_latency", 0.0)
    validate_finite_number(p25_latency, f"{model_key} p25_latency", 0.0)
    validate_finite_number(p50_latency, f"{model_key} p50_latency", 0.0)
    validate_finite_number(p75_latency, f"{model_key} p75_latency", 0.0)
    validate_finite_number(p90_latency, f"{model_key} p90_latency", 0.0)
    validate_finite_number(p95_latency, f"{model_key} p95_latency", 0.0)
    validate_finite_number(p99_latency, f"{model_key} p99_latency", 0.0)
    validate_finite_number(std_latency, f"{model_key} std_latency", 0.0)

    result: dict[str, Any] = {
        "model_key": model_key,
        "display_name": display_name,
        "category": category,
        "model_name": str(metrics["model"]),
        "roc_auc_macro": roc_auc_macro,
        "avg_latency_ms": avg_latency_ms,
        "total_samples": total_samples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "min_latency_ms": min_latency,
        "p25_latency_ms": p25_latency,
        "p50_latency_ms": p50_latency,
        "p75_latency_ms": p75_latency,
        "p90_latency_ms": p90_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "std_latency_ms": std_latency,
    }

    # コスト情報の紐付けと検証
    if model_key in API_COST_CATALOG:
        cost_info = API_COST_CATALOG[model_key]
        result["cost_type"] = cost_info["pricing_type"]
        result["input_cost_per_1m"] = validate_finite_number(
            cost_info["input_cost_per_1m"],
            f"{model_key} input_cost_per_1m",
            0.0,
        )
        result["output_cost_per_1m"] = validate_finite_number(
            cost_info["output_cost_per_1m"],
            f"{model_key} output_cost_per_1m",
            0.0,
        )
        result["role"] = cost_info["role"]
    elif model_key in SERVER_DEPLOYED_CATALOG:
        cost_info = SERVER_DEPLOYED_CATALOG[model_key]
        result["cost_type"] = cost_info["pricing_type"]
        result["input_cost_per_1m"] = None
        result["output_cost_per_1m"] = None
        result["role"] = cost_info["role"]
    else:
        raise KeyError(
            f"Fail First: Unknown cost catalog mapping for model '{model_key}'."
        )

    return result


def generate_markdown_report(
    models_data: list[dict[str, Any]], pareto_keys: list[str]
) -> str:
    """統合比較結果を Markdown 形式のレポートにレンダリングする。"""
    if not models_data:
        raise ValueError("Fail First: Cannot generate report with empty models data.")

    lines = []
    lines.append("# 意図分類モデル 統合比較・トレードオフ分析レポート\n")
    lines.append(
        "本レポートは、`tanaos/synthetic-intent-classfier-dataset`（3,447件のテストデータ）に対して実施された4つのモデル（**LightGBM**、**DistilBERT**、**Gemini 3.5 Flash Lite**、**Jev**）の分類性能、レスポンス性能、および推論コストを集約・比較・考察した最終実験報告書です。\n"
    )

    lines.append("## 1. 総合評価サマリーテーブル\n")
    lines.append(
        "| モデル名 | 分類アプローチ | ROC-AUC (Macro) | Accuracy | Macro-F1 | 平均レイテンシ (ms) | 最小値 (ms) | p25 (ms) | p50 (ms) | p75 (ms) | p90 (ms) | p99 (ms) | パレート最適 |"
    )
    lines.append(
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for d in models_data:
        is_pareto = "★ 最適 (Frontier)" if d["model_key"] in pareto_keys else "-"
        lines.append(
            f"| **{d['display_name']}** (`{d['model_name']}`) | {d['category']} | "
            f"**{d['roc_auc_macro']:.4f}** | {d['accuracy']:.4f} | {d['macro_f1']:.4f} | "
            f"{d['avg_latency_ms']:.2f} | {d['min_latency_ms']:.2f} | {d['p25_latency_ms']:.2f} | "
            f"{d['p50_latency_ms']:.2f} | {d['p75_latency_ms']:.2f} | {d['p90_latency_ms']:.2f} | {d['p99_latency_ms']:.2f} | {is_pareto} |"
        )
    lines.append("")

    lines.append("## 2. 推論コストと課金モデルの比較\n")
    lines.append("### 2.1 API 従量課金モデルのコスト比較（100万トークンあたり）\n")
    lines.append("| モデル | 入力（/1M tokens） | 出力（/1M tokens） | 主な役割 |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(
        "| **Gemini 3.5 Flash-Lite** | $0.30 | $2.50 | 汎用LLM（テキスト生成、要約、マルチモーダル処理） |"
    )
    lines.append(
        "| **TypeSafe Jev** | **$0.042** | **$0.00** | 判定特化（Yes/No、複数選択肢の分類、スコアリング） |\n"
    )

    lines.append("#### コスト考察（Jev vs Gemini Flash Lite）:")
    lines.append(
        "- **入力コスト比 約 1/7**: Jev の入力単価は $0.042/1M tokens であり、Gemini 3.5 Flash-Lite（$0.30）に比べて約 **86% 安価** です。"
    )
    lines.append(
        "- **出力コスト $0.00**: Jev は生成を行わず確率分布を直接出力する System One モデルのため、**出力トークン課金が完全に $0.00** です。Gemini は JSON 構造化出力テキストの生成に伴い高単価（$2.50/1M tokens）な出力トークン費用が発生するため、意図分類における総リクエストコストは Jev が 1 桁以上圧倒的に低コストとなります。\n"
    )

    lines.append("### 2.2 サーバーデプロイ型モデル（固定費モデル）")
    lines.append(
        "- **LightGBM / DistilBERT**: 自社サーバーやクラウド VM（CPU / GPU インスタンス）に常時デプロイして稼働させるモデルです。"
    )
    lines.append(
        "- リクエスト単位のトークン課金は発生せず「インスタンス固定費」となるため、上記のリクエスト単価比較には含まれません。数百万件規模の極めて大規模かつ定常的なトラフィックがある場合、固定費モデルとしてのコスト効率が高まります。\n"
    )

    lines.append("## 3. パレートフロンティア（精度 vs 速度）分析\n")
    lines.append(
        "精度（ROC-AUC Macro）の最大化と、推論時間（平均レイテンシ）の最小化におけるトレードオフを分析した結果、以下のパレート最適解が特定されました：\n"
    )
    for d in models_data:
        if d["model_key"] in pareto_keys:
            lines.append(
                f"- **{d['display_name']}**: ROC-AUC `{d['roc_auc_macro']:.4f}`, 平均レイテンシ `{d['avg_latency_ms']:.2f} ms`"
            )
    lines.append("")

    lines.append("## 4. 各モデルの詳細特性・トレードオフ考察\n")
    lines.append("### 4.1 LightGBM (TF-IDF + 勾配ブースティング)")
    lines.append(
        "- **強み**: 圧倒的な推論速度（平均約 4ms）と低リソース消費。CPU のみで動作し、インフラ維持コストが極めて低い。大規模アノテーションデータが存在する環境下で ROC-AUC 0.994 と非常に高い分類精度を発揮。"
    )
    lines.append(
        "- **課題**: テキストの語順や深い意味理解（長文文脈・言い換え）を捉えにくく、未知の表現や語彙に対する汎化性能はドメイン学習データに強く依存する。\n"
    )

    lines.append("### 4.2 DistilBERT (ローカル Transformer ファインチューニング)")
    lines.append(
        "- **強み**: 評価対象中トップの ROC-AUC (0.997) を達成。BERT の双方向言語理解とファインチューニングの組み合わせにより、極めて高精度な境界判別が可能。ローカル GPU 上で平均約 8ms と非常に高速。"
    )
    lines.append(
        "- **課題**: 学習・推論に NVIDIA GPU が必須であり、コンテナサイズやメモリフットプリントが大きい。また、モデルのファインチューニングと再学習運用の体制が必要。\n"
    )

    lines.append("### 4.3 Gemini 3.5 Flash Lite (クラウド生成型LLM API)")
    lines.append(
        "- **強み**: 学習データ不要のゼロショット分類が可能。プロンプトへのカテゴリ指示のみで即座に導入可能。"
    )
    lines.append(
        "- **課題**: 平均レイテンシが約 7.7 秒と著しく遅く、出力トークン課金（$2.50/1M tokens）が発生するため分類タスクとしては割高。また、確率分布を完全には返却しないため、信頼度スコアからの疑似復元が必要となる。\n"
    )

    lines.append("### 4.4 Jev (TypeSafe System One ジャッジメントモデル)")
    lines.append(
        "- **強み**: 生成を行わない高速な判断特化型モデルであり、平均レイテンシ約 560ms とクラウド API として極めて高速（Gemini Flash Lite の約 1/14）。入力単価 $0.042/1M tokens・出力単価 $0.00 と驚異的なコストパフォーマンスを実現。`Choice` プリミティブにより正規化された直接確率分布（`probabilities`）が返却され、ROC-AUC 0.971 とゼロショット分類として高精度。"
    )
    lines.append(
        "- **課題**: ローカルモデル（LightGBM / DistilBERT）に比べるとネットワーク遅延（百ミリ秒台）が発生する。\n"
    )

    lines.append("## 5. ユースケース別推奨アーキテクチャ\n")
    lines.append("| ユースケース / 要件 | 推奨モデル | 採用理由 |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(
        "| **リアルタイム対話・音声ボット** (SLA < 50ms) | **LightGBM** / **DistilBERT** | 10ms 未満の超低レイテンシ推論が必須なため（インフラ固定費）。 |"
    )
    lines.append(
        "| **最高精度・オフラインバッチ処理** | **DistilBERT** | 最高の ROC-AUC (0.997) を誇り、確定的な推論が可能。 |"
    )
    lines.append(
        "| **学習データ無しの即時プロトタイプ・超低コスト API 導入** | **Jev** | アノテーション工数ゼロ、Gemini に比べ圧倒的に低遅延（560ms）、かつ出力トークン $0 で極めて安価。 |"
    )
    lines.append(
        "| **高信頼性スコアリング・信頼度ルーティング** | **Jev** | 正確な確率分布（`probabilities`）が得られるため、閾値判定・フォールバック制御が確実。 |"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results across all models and generate a comprehensive comparison report (Fail First)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="05_summary/input",
        help="Directory containing subdirectories or symlinks for each model's output.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="05_summary/output",
        help="Directory to save the comparison report and summary json.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_config = [
        {
            "key": "01_lightgbm",
            "display": "LightGBM",
            "category": "TF-IDF + GBDT (ローカル)",
        },
        {
            "key": "02_distilbert",
            "display": "DistilBERT",
            "category": "Transformer Fine-tuning (ローカル GPU)",
        },
        {
            "key": "03_gemini_flash_lite",
            "display": "Gemini 3.5 Flash Lite",
            "category": "生成型 LLM API (ゼロショット)",
        },
        {
            "key": "04_jev",
            "display": "Jev (jev-1.13)",
            "category": "判断特化 System One API (ゼロショット)",
        },
    ]

    print(f"Aggregating evaluation data from {input_dir} (Fail First Mode)...")
    models_data = []
    expected_samples: int | None = None

    for cfg in models_config:
        data = load_model_data(
            model_key=cfg["key"],
            display_name=cfg["display"],
            category=cfg["category"],
            base_dir=input_dir,
        )

        # 全モデル間でテストサンプル数が完全に一致しているかを検証 (Fail First)
        if expected_samples is None:
            expected_samples = data["total_samples"]
        elif data["total_samples"] != expected_samples:
            raise ValueError(
                f"Fail First: Sample count mismatch detected! "
                f"Model '{cfg['key']}' has {data['total_samples']} samples, "
                f"but previous models had {expected_samples} samples."
            )

        models_data.append(data)
        print(
            f"Loaded {cfg['display']}: ROC-AUC={data['roc_auc_macro']:.4f}, Latency={data['avg_latency_ms']:.2f}ms"
        )

    # パレート最適解の抽出
    pareto_keys = calculate_pareto_frontier(models_data)
    print(f"Identified Pareto optimal models: {pareto_keys}")

    # レポート生成
    report_content = generate_markdown_report(models_data, pareto_keys)
    report_path = output_dir / "comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Comparison report saved to {report_path}")

    # サマリーJSONの保存
    summary_path = output_dir / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": models_data,
                "pareto_optimal_keys": pareto_keys,
                "api_cost_catalog": API_COST_CATALOG,
                "server_deployed_catalog": SERVER_DEPLOYED_CATALOG,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Summary metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
