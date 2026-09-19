# JEV-Eval: Intent Classification & Benchmarking Project

マルチクラス意図分類タスクにおいて、LightGBM、DeBERTa、Gemini Flash Lite、および Jev 判定モデルの分類性能（ROC-AUC Macro-average）および推論レイテンシを比較・評価するためのベンチマーク検証リポジトリです。

## ディレクトリ構成

```text
jev-eval/
├── 00_dataset/                  # ステップ0: 共通データセットの取得と分割
│   ├── output/                  # train.csv (70%), test.csv (30%)
│   ├── script/                  # dataset.py (データ取得・分割スクリプト)
│   └── tests/                   # test_dataset.py (単体テスト)
├── 01_lightgbm/                 # ステップ1: LightGBMモデル実験
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

2. **テストの実行**
   データセットの品質および形式の検証を行います。
   ```bash
   uv run pytest 00_dataset/tests/test_dataset.py
   ```
