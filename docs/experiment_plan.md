# 意図分類モデル評価 実験計画書

本実験は、`tanaos/synthetic-intent-classfier-dataset` データセットを使用し、4つのモデル（**LightGBM**、**DistilBERT**、**gemini-3.5-flash-lite**、**jev**）の分類性能（マルチクラス ROC-AUC Macro-average）およびレスポンス性能（平均レイテンシ）を評価・比較するための実験計画書です。

実験環境および依存関係の管理には、Pythonの **`uv`** を使用して高速かつ再現性高く整備します。

---

## ディレクトリ構造設計

各実験フェーズ（ステップ）において、入力データ、一時ファイル、スクリプト、単体テスト、および最終出力を完全に独立して管理するため、以下のディレクトリ構造を採用します。各ステップの `input/` フォルダには、前段のステップの `output/` から必要な成果物へのシンボリックリンクを配置します。

```text
jev-eval/
├── pyproject.toml               # uvによるプロジェクト設定・依存関係定義
├── uv.lock                      # uvロックファイル
├── 00_dataset/                  # ステップ0: 共通データセットの取得と分割
│   ├── input/                   # （外部データセットのため空またはHugging Faceキャッシュ）
│   ├── tmp/                     # 一時ファイル
│   ├── output/                  # train.csv (70%), test.csv (30%)
│   ├── script/                  # dataset.py
│   └── tests/                   # test_dataset.py
├── 01_lightgbm/                 # ステップ1: LightGBMモデル実験
│   ├── input/                   # 00_dataset/output/ へのシンボリックリンク (train.csv, test.csv)
│   ├── tmp/                     # 一時ファイル（チェックポイント等）
│   ├── output/                  # model.pkl, predictions.csv, metrics.json
│   ├── script/                  # train.py, evaluate.py
│   └── tests/                   # test_lightgbm.py
├── 02_distilbert/               # ステップ2: DistilBERTモデル実験
│   ├── input/                   # 00_dataset/output/ へのシンボリックリンク (train.csv, test.csv)
│   ├── tmp/                     # 一時ファイル（DistilBERT中間チェックポイント等）
│   ├── output/                  # model_weights/, predictions.csv, metrics.json
│   ├── script/                  # train.py, evaluate.py
│   └── tests/                   # test_distilbert.py
├── 03_gemini_flash_lite/        # ステップ3: Gemini Flash Liteモデル実験
│   ├── input/                   # 00_dataset/output/ へのシンボリックリンク (test.csv)
│   ├── tmp/                     # 一時ファイル・中間キャッシュ (sample_{idx}.json)
│   ├── output/                  # predictions.csv, metrics.json
│   ├── script/                  # evaluate.py (並行処理・レジューム・スモークテスト対応)
│   └── tests/                   # test_gemini.py (モックテストおよびキャッシュスキップ検証)
├── 04_jev/                      # ステップ4: Jevモデル実験
│   ├── input/                   # 00_dataset/output/ へのシンボリックリンク (test.csv)
│   ├── tmp/                     # 一時ファイル・中間キャッシュ (sample_{idx}.json)
│   ├── output/                  # predictions.csv, metrics.json
│   ├── script/                  # evaluate.py (Choiceプリミティブ・並行処理・レジューム・スモークテスト対応)
│   └── tests/                   # test_jev.py (モックテストおよびキャッシュスキップ検証)
└── 05_summary/                  # ステップ5: 統合比較・トレードオフ分析
    ├── input/                   # 各モデルの output/metrics.json へのシンボリックリンク
    ├── tmp/                     # 一時ファイル
    ├── output/                  # comparison_report.md
    ├── script/                  # compare.py
    └── tests/                   # test_compare.py
```

---

## 事前準備: Python環境と `uv` のセットアップ

実験を開始する前に、Pythonおよび `uv` を用いて環境を整備します。

1. プロジェクトのルートディレクトリで `uv init` を行い、`pyproject.toml` を準備する。
2. 必要なライブラリ（`scikit-learn`, `lightgbm`, `transformers`, `datasets`, `torch`, `google-genai`, `typesafe-sdk`, `pydantic`, `python-dotenv`, `tqdm` 等）を `uv add` でインストールする。
3. APIキーなどの認証情報はルートディレクトリの `.env` ファイルに定義し、`python-dotenv` 経由で安全に読み込む（リポジトリにはコミットしない）。

---

## ステップ 0: データセットの取得と分割 (`00_dataset/`)

すべての実験の土台となるデータを生成します。

- **`00_dataset/script/dataset.py` の実行**:
  - Hugging Face Datasetsライブラリから `tanaos/synthetic-intent-classfier-dataset` をロードする。
  - データをシャッフルし、70%を `00_dataset/output/train.csv`、30%を `00_dataset/output/test.csv` として出力・保存する。
  - カラム構造として `text`（入力文）および `labels`（0〜11の整数カテゴリラベル）を保証する。

---

## 評価指標に関する設計方針（各ステップでの完全な自己完結）

共通の共通モジュール（common.py 等）は作成しません。すべての評価指標（マルチクラス ROC-AUC Macro-average、平均レイテンシ）は、**各モデルのステップ内（`script/evaluate.py` 内）で、そのモデルおよび実験で使用する最適なライブラリを用いて完全に独立して計算・算出**します。ただし、結果の整合性を保つため、算出する指標の定義（マルチクラス ROC-AUC の Macro-average、および1件あたりの平均処理時間）はすべてのステップで統一します。

---

## モデル実験ステップの共通ルール（README.md の記載事項・実装規約）

モデル実験の各ステップ（`01_lightgbm/`, `02_distilbert/`, `03_gemini_flash_lite/`, `04_jev/` 等）では、実装完了後に必ず各ステップのトップ階層に `README.md` を作成または最新の状態に更新してください。`README.md` には以下の内容を必ず記載することとします：

1. **モデル構成**: 使用したアルゴリズム、ハイパーパラメータ、モデル名、プロンプト/スキーマ設計、Jev質問設計などの詳細。
2. **学習データ数 / テストデータ数**: 使用したデータセットのサンプル数および入力パス。
3. **トレーニング・推論実績**: 所要時間、学習ログ、レイテンシ分布、中間キャッシュの実績（APIモデルの場合）。
4. **評価指標**: マルチクラス ROC-AUC (Macro-average) および平均推論レイテンシ（ms/sample）。
5. **画面やコマンドからの確認方法**: スモークテスト（`--smoke`）や本番推論、単体テスト（`pytest`）の具体的な実行コマンド例。

---

## ステップ 1: LightGBM モデルの評価 (`01_lightgbm/`)

- **目的**: 従来型機械学習（TF-IDF + LightGBM）による分類性能とレイテンシの計測。

### 実行手順:
1. `01_lightgbm/input/` 内のシンボリックリンク経由で `train.csv` を読み込み、テキストを `TfidfVectorizer` で特徴量化して LightGBM のトレーニングを行う（`script/train.py`）。
2. 学習済みモデルを `01_lightgbm/output/model.pkl` として保存する。
3. `input/` 内の `test.csv` を読み込み、推論を実行する（`script/evaluate.py`）。
4. 推論結果（予測確率・正解ラベル・推論時間）を `output/predictions.csv` として保存する。
5. `script/evaluate.py` 内において、`scikit-learn` を用いてマルチクラス **ROC-AUC（Macro-average）** および平均レイテンシを計算し、結果を `output/metrics.json` に保存する。

---

## ステップ 2: DistilBERT モデルの評価 (`02_distilbert/`)

- **目的**: Transformerベース（`distilbert/distilbert-base-uncased`）のローカルGPUファインチューニングによる性能検証。

### 実行手順:
1. ローカルNVIDIA GPU環境下で、`input/` 内の `train.csv` を用いて `distilbert/distilbert-base-uncased` のファインチューニングを実施する（`script/train.py`）。
2. モデル重みを `02_distilbert/output/model_weights/` に保存する。
3. `input/` 内の `test.csv` を読み込み、推論を実行する（`script/evaluate.py`）。
4. 推論結果を `output/predictions.csv` として保存する。
5. `script/evaluate.py` 内において、`scikit-learn` または `PyTorch` / `transformers` 関連関数を用いてマルチクラス **ROC-AUC（Macro-average）** および平均レイテンシを計算し、結果を `output/metrics.json` に保存する。

---

## ステップ 3: Gemini 3.5 Flash Lite モデルの評価 (`03_gemini_flash_lite/`)

- **目的**: 超高速LLM（API利用）を用いたゼロショット意図分類の性能とレイテンシの検証。

### 実装・運用設計（[`docs/step3_implementation_plan.md`](step3_implementation_plan.md) 準拠）:
1. **APIクライアント初期化とリトライ設定**:
   - `google-genai` SDK を用い、429（レートリミット）対策として `types.HttpRetryOptions(attempts=10, initial_delay=10, max_delay=100, exp_base=1.5, jitter=0.5, http_status_codes=[429])` を設定。
   - `python-dotenv` により環境変数 `GEMINI_API_KEY` を自動ロード。
2. **システムプロンプトと構造化出力スキーマ**:
   - Pydantic によるスキーマ定義: `predicted_label: int`, `confidence: float` (0.0〜1.0)。
   - 0〜11のカテゴリ番号と意図の詳細説明マッピング（`LABEL_MAPPING`）を `system_instruction` に組み込み。
   - AFC (Automatic Function Calling) 警告を抑止するため、`client.chats.create` によるチャットセッションを作成して `chat.send_message` を利用。
3. **並行推論と中間キャッシュ・レジューム機構**:
   - `ThreadPoolExecutor(max_workers=10)` により並行リクエスト送信。
   - 各サンプルの推論結果を `03_gemini_flash_lite/tmp/sample_{idx}.json` に個別保存。中断後の再実行時には既存キャッシュを読み込んで API 呼び出しをスキップする自動レジュームを実装。
4. **スモークテスト対応**:
   - コマンドライン引数 `--smoke` により、先頭10件のみで迅速に動作確認可能（成果物ファイル保存はスキップ）。
5. **評価指標算出と成果物保存**:
   - 予測ラベルおよび信頼度から確率分布配列を擬似復元し、`sklearn.metrics.roc_auc_score` でマルチクラス **ROC-AUC（Macro-average）** を算出。
   - `output/predictions.csv`（予測ラベル、各クラスの予測確率、レイテンシ）および `output/metrics.json`（`roc_auc_macro`, `avg_latency_ms`, `total_samples`）に保存。
6. **テスト整備**:
   - `03_gemini_flash_lite/tests/test_gemini.py` にて、API呼び出しモックテストおよび中間キャッシュによるスキップ動作テストを実装。

---

## ステップ 4: Jev モデルの評価 (`04_jev/`)

- **目的**: TypeSafe System One モデル（`jev-1.13`）の `Choice` プリミティブを活用したジャッジメントモデルによる意図分類の性能・レイテンシの検証。

### 実装・運用設計（ステップ3の運用パターンを踏襲）:
1. **TypeSafe SDK クライアント初期化と環境設定**:
   - `typesafe-sdk` の `TypeSafeClient` を使用し、環境変数 `TYPESAFE_API_KEY`（`.env` 経由）から認証情報をロード。
   - APIリトライやレートリミット対策を組み込み、安定したリクエスト処理を実現。
2. **Jev 質問・クライテリア設計 (`Choice` プリミティブ)**:
   - Jev はテキスト生成を行わず、定義された回答肢に対する確率分布（`probabilities`）を直接返すモデルであるため、意図分類には `Choice` プリミティブを採用。
   - `state={"text": text}` を入力とし、`instructions="Which intent category best classifies the user text?"` を指定。
   - 12カテゴリの選択肢（0〜11）それぞれに対して、`what`（該当する意図の説明）、`not_for`（近接するが該当しない意図）、`examples`（典型的な具体例）を定義し、精度の高い分類境界を構築。
3. **並行推論と中間キャッシュ・レジューム機構**:
   - ステップ3と同様に `ThreadPoolExecutor`（例: `max_workers=10`）による並行リクエスト処理。
   - 各サンプルの判定結果・確率分布・レイテンシを `04_jev/tmp/sample_{idx}.json` に個別保存。
   - 中断時やエラー発生時も、次回実行時に既存キャッシュを自動検知して未処理サンプルのみを継続処理するレジューム機構を実装。
4. **スモークテスト対応**:
   - コマンドライン引数 `--smoke` により、先頭10サンプルのみで動作確認を実施可能とする。
5. **評価指標算出と成果物保存**:
   - `Choice` から返却される正規化された確率分布（`probabilities`）をそのままマルチクラス予測確率として利用。
   - `sklearn.metrics.roc_auc_score` を用いてマルチクラス **ROC-AUC（Macro-average）** および平均レイテンシ（ms/sample）を算出。
   - 推論結果を `04_jev/output/predictions.csv`、評価指標を `04_jev/output/metrics.json` に保存。
6. **テストコードの整備**:
   - `04_jev/tests/test_jev.py` を作成し、モックを用いたスモークテストおよび中間キャッシュによるスキップ動作テストを網羅。

---

## ステップ 5: 統合比較・トレードオフ分析 (`05_summary/`)

- **目的**: 各ステップの成果物を集約し、全モデル（LightGBM, DistilBERT, Gemini 3.5 Flash Lite, Jev）の性能比較と考察を行う。

### 実行手順:
1. `input/` 内のシンボリックリンク経由で各モデルの `output/metrics.json` を読み込む（`script/compare.py`）。
2. ROC-AUC スコアと平均レイテンシの一覧テーブルを作成する。
3. 精度と速度のトレードオフ（パレートフロンティア分析）、および各モデルの長所・短所（コスト、ローカル実行要件、並行処理性能など）を分析し、最終レポートを `output/comparison_report.md` として出力する。
