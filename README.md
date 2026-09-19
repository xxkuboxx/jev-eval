# JEV-Eval: Intent Classification & Benchmarking Project

マルチクラス意図分類タスクにおいて、LightGBM、DeBERTa、Gemini Flash Lite、および Jev 判定モデルの分類性能（ROC-AUC Macro-average）および推論レイテンシを比較・評価するためのベンチマーク検証リポジトリです。

## ディレクトリ構成

```text
jev-eval/
├── .github/workflows/ci.yml       # GitHub Actions CI (Ruff, Mypy, Pytest)
├── 00_dataset/                  # ステップ0: 共通データセットの取得と分割
│   ├── output/                  # train.csv (70%), test.csv (30%)
│   ├── script/                  # dataset.py (データ取得・分割スクリプト)
│   └── tests/                   # test_dataset.py (単体テスト)
├── 01_lightgbm/                 # ステップ1: LightGBMモデル実験 (TF-IDF + LightGBM)
│   ├── input/                   # train.csv, test.csv
│   ├── output/                  # model.pkl, predictions.csv, metrics.json
│   ├── script/                  # train.py, evaluate.py
│   └── tests/                   # test_lightgbm.py
├── 02_deberta/                  # ステップ2: DeBERTaモデル実験
├── 03_gemini_flash_lite/        # ステップ3: Gemini Flash Liteモデル実験
├── 04_jev/                      # ステップ4: Jevモデル実験
├── 05_summary/                  # ステップ5: 統合比較・トレードオフ分析
├── docs/                        # 実験計画・実装計画書
└── tests/                       # 共通テスト
```

## 実行方法

1. **データセットの取得と分割**
   Hugging Face Hub から共通データセットを取得し、訓練用（70%）および評価用（30%）のCSVファイルとして出力します。
   ```bash
   uv run python 00_dataset/script/dataset.py
   ```

2. **ステップ 1: LightGBM 実験の実行**
   ```bash
   # 学習
   uv run python 01_lightgbm/script/train.py

   # 推論・評価
   uv run python 01_lightgbm/script/evaluate.py
   ```

3. **テストおよび品質チェックの実行**
   ```bash
   uv run pytest
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy .
   ```
