# JEV-Eval: Intent Classification & Benchmarking Project

マルチクラス意図分類タスクにおいて、LightGBM、DistilBERT、Gemini Flash Lite、および Jev 判定モデルの分類性能（ROC-AUC Macro-average）および推論レイテンシを比較・評価するためのベンチマーク検証リポジトリです。

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
├── 02_distilbert/               # ステップ2: DistilBERTモデル実験
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

3. **ステップ 2: DistilBERT 実験の実行 (GPU/CUDA 対応)**
   ```bash
   # 学習 (CUDA GPU が検出された場合は自動的に GPU を利用します)
   uv run python 02_distilbert/script/train.py

   # 推論・評価
   uv run python 02_distilbert/script/evaluate.py
   ```
   *※ NVIDIA GPU (GTX 1050 等の Pascal 世代) の場合、FP16 演算による性能劣化や NaN を避けるため FP32 で動作し、動的パディングと実効バッチサイズ 32（バッチサイズ 16 × 勾配累積ステップ 2）に最適化されています。*
   *※ PyTorch は PyTorch 公式 CUDA 12.4 インデックス (`https://download.pytorch.org/whl/cu124`) より管理されています。*

4. **ステップ 3: Gemini Flash Lite 実験の実行**
   ```bash
   # .env.sample を基に .env ファイルを作成し、APIキーを設定するか、環境変数を設定してください
   cp .env.sample .env
   # または export GEMINI_API_KEY="your-api-key"

   # 動作確認用スモークテスト（先頭10サンプルのみ・結果保存なし）
   uv run python 03_gemini_flash_lite/script/evaluate.py --smoke

   # 本番推論・評価 (HTTPリトライ機構対応: 最大10回試行、初期遅延10秒、最大遅延100秒、指数バックオフ base=1.5、ジッター0.5)
   uv run python 03_gemini_flash_lite/script/evaluate.py
   ```

3. **テストおよび品質チェックの実行**
   ```bash
   uv run pytest
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy .
   ```
