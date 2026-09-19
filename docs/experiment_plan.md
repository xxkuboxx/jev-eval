# 意図分類モデル評価 実験計画書

本実験は、`tanaos/synthetic-intent-classfier-dataset` データセットを使用し、4つのモデル（**LightGBM**、**DeBERTa**、**gemini-3.5-flash-lite**、**jev**）の分類性能（マルチクラス ROC-AUC Macro-average）およびレスポンス性能（平均レイテンシ）を評価・比較するための実験計画書です。

実験環境および依存関係の管理には、Pythonの **`uv`** を使用して高速かつ再現性高く整備します。

---

## ディレクトリ構造設計

各実験フェーズ（ステップ）において、入力データ、一時ファイル、スクリプト、および最終出力を完全に独立して管理するため、以下のディレクトリ構造を採用します。各ステップの `input/` フォルダには、前段のステップの `output/` から必要な成果物へのシンボリックリンクを配置します。

```text
jev-eval/
├── pyproject.toml               # uvによるプロジェクト設定・依存関係定義
├── uv.lock                      # uvロックファイル
├── 00_dataset/                  # ステップ0: 共通データセットの取得と分割
│   ├── input/                   # （外部データセットのため空またはHugging Faceキャッシュ）
│   ├── tmp/                     # 一時ファイル
│   ├── output/                  # train.csv (70%), test.csv (30%)
│   └── script/                  # dataset.py
├── 01_lightgbm/                 # ステップ1: LightGBMモデル実験
│   ├── input/                   # 00_dataset/output/ へのシンボリックリンク (train.csv, test.csv)
│   ├── tmp/                     # 一時ファイル（チェックポイント等）
│   ├── output/                  # model.pkl, predictions.csv, metrics.json
│   └── script/                  # train.py, evaluate.py
├── 02_deberta/                  # ステップ2: DeBERTaモデル実験
│   ├── input/                   # 00_dataset/output/ へのシンボリックリンク (train.csv, test.csv)
│   ├── tmp/                     # 一時ファイル（DeBERTa中間チェックポイント等）
│   ├── output/                  # model_weights/, predictions.csv, metrics.json
│   └── script/                  # train.py, evaluate.py
├── 03_gemini_flash_lite/        # ステップ3: Gemini Flash Liteモデル実験
│   ├── input/                   # 00_dataset/output/ へのシンボリックリンク (test.csv)
│   ├── tmp/                     # 一時ファイル
│   ├── output/                  # predictions.csv, metrics.json
│   └── script/                  # evaluate.py
├── 04_jev/                      # ステップ4: Jevモデル実験
│   ├── input/                   # 00_dataset/output/ へのシンボリックリンク (test.csv)
│   ├── tmp/                     # 一時ファイル
│   ├── output/                  # predictions.csv, metrics.json
│   └── script/                  # evaluate.py
└── 05_summary/                  # ステップ5: 統合比較・トレードオフ分析
    ├── input/                   # 各モデルの output/metrics.json へのシンボリックリンク
    ├── tmp/                     # 一時ファイル
    ├── output/                  # comparison_report.md
    └── script/                  # compare.py
```

---

## 事前準備: Python環境と `uv` のセットアップ

実験を開始する前に、Pythonおよび `uv` を用して環境を整備します。

1. プロジェクトのルートディレクトリで `uv init` を行い、`pyproject.toml` を準備する。
2. 必要なライブラリ（`scikit-learn`, `lightgbm`, `transformers`, `datasets`, `torch`, `google-genai` 等）を `uv add` でインストールする。

---

## ステップ 0: データセットの取得と分割 (`00_dataset/`)

すべての実験の土台となるデータを生成します。

- **`00_dataset/script/dataset.py` の実行**:
  - Hugging Face Datasetsライブラリから `tanaos/synthetic-intent-classfier-dataset` をロードする。
  - データをシャッフルし、70%を `00_dataset/output/train.csv`、30%を `00_dataset/output/test.csv` として出力・保存する。

---

## 評価指標に関する設計方針（各ステップでの完全な自己完結）

共通の共通モジュール（common.py 等）は作成しません。すべての評価指標（マルチクラス ROC-AUC Macro-average、平均レイテンシ）は、**各モデルのステップ内（`script/evaluate.py` 内）で、そのモデルおよび実験で使用する最適なライブラリを用いて完全に独立して計算・算出**します。ただし、結果の整合性を保つため、算出する指標の定義（マルチクラス ROC-AUC の Macro-average、および1件あたりの平均処理時間）はすべてのステップで統一します。

---

## ステップ 1: LightGBM モデルの評価 (`01_lightgbm/`)

- **目的**: 従来型機械学習（TF-IDF + LightGBM）による分類性能とレイテンシの計測。

### 実行手順:
1. `01_lightgbm/input/` 内のシンボリックリンク経由で `train.csv` を読み込み、テキストを `TfidfVectorizer` で特徴量化して LightGBM のトレーニングを行う（`script/train.py`）。
2. 学習済みモデルを `01_lightgbm/output/model.pkl` として保存する。
3. `input/` 内の `test.csv` を読み込み、推論を実行する（`script/evaluate.py`）。
4. 推論結果（予測確率・正解ラベル・推論時間）を `output/predictions.csv` として保存する。
5. `script/evaluate.py` 内において、使用するライブラリ（`scikit-learn` 等）を用いてマルチクラス **ROC-AUC（Macro-average）** および平均レイテンシを独自に計算し、結果を `output/metrics.json` に保存する。

---

## ステップ 2: DeBERTa モデルの評価 (`02_deberta/`)

- **目的**: Transformerベース（`microsoft/deberta-v3-small`）のローカルGPUファインチューニングによる性能検証。

### 実行手順:
1. ローカルNVIDIA GPU環境下で、`input/` 内の `train.csv` を用いて `microsoft/deberta-v3-small` のファインチューニングを実施する（`script/train.py`）。
2. モデル重みを `02_deberta/output/model_weights/` に保存する。
3. `input/` 内の `test.csv` を読み込み、推論を実行する（`script/evaluate.py`）。
4. 推論結果を `output/predictions.csv` として保存する。
5. `script/evaluate.py` 内において、使用するライブラリ（`scikit-learn` または `PyTorch` / `transformers` 関連関数）を用いてマルチクラス **ROC-AUC（Macro-average）** および平均レイテンシを独自に計算し、結果を `output/metrics.json` に保存する。

---

## ステップ 3: Gemini 3.5 Flash Lite モデルの評価 (`03_gemini_flash_lite/`)

- **目的**: 超高速LLM（API利用）を用いたゼロショット意図分類の性能とレイテンシの検証。

### 実行手順:
1. Gemini APIクライアントを初期化し、システムプロンプトおよび構造化出力スキーマを定義する。
2. `input/` 内の `test.csv` を読み込み、全件に対してAPIリクエストを送信する（`script/evaluate.py`）。
3. 各リクエストの往復時間、予測確率、正解ラベルを `output/predictions.csv` として保存する。
4. `script/evaluate.py` 内において、使用するライブラリ（`scikit-learn` 等）を用いてマルチクラス **ROC-AUC（Macro-average）** および平均レイテンシを独自に計算し、結果を `output/metrics.json` に保存する。

---

## ステップ 4: Jev モデルの評価 (`04_jev/`)

- **目的**: TypeSafe System Oneモデルの `Choice` プリミティブを活用したジャッジメントモデルによる意図分類の検証。

### 実行手順:
1. TypeSafe SDK クライアントを初期化し、12クラス分類用プロンプト・選択肢定義（`what`, `not_for`, `examples`）を記述する。
2. `input/` 内の `test.csv` を読み込み、全件に対してJev APIリクエストを送信する（`script/evaluate.py`）。
3. 各リクエストの往復時間、`probabilities`、正解ラベルを `output/predictions.csv` として保存する。
4. `script/evaluate.py` 内において、使用するライブラリ（`scikit-learn` 等）を用いてマルチクラス **ROC-AUC（Macro-average）** および平均レイテンシを独自に計算し、結果を `output/metrics.json` に保存する。

---

## ステップ 5: 統合比較・トレードオフ分析 (`05_summary/`)

- **目的**: 各ステップの成果物を集約し、全モデルの性能比較と考察を行う。

### 実行手順:
1. `input/` 内のシンボリックリンク経由で各モデルの `output/metrics.json` を読み込む（`script/compare.py`）。
2. ROC-AUC スコアと平均レイテンシの一覧テーブルを作成する。
3. 精度と速度のトレードオフを分析し、最終レポートを `output/comparison_report.md` として出力する。
