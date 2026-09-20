# JEV-Eval: Intent Classification & Benchmarking Project

## 概要

マルチクラス意図分類（Intent Classification）タスクにおいて、LightGBM、DistilBERT、Gemini Flash Lite、および Jev 判定モデルの分類性能（ROC-AUC Macro-average 等）、推論レイテンシ、および推論コストを比較・評価するためのベンチマーク検証プロジェクトです。

---

## 実験条件

### 利用データセット

#### 1. 基本情報
- **Hugging Face Hub リポジトリ**: `tanaos/synthetic-intent-classifier-dataset-v1`
- **総レコード数**: 11,489 件
  - **訓練セット (Train)**: 8,042 件 (70%)
  - **テストセット (Test)**: 3,447 件 (30%)
- **フォーマット**: CSV / Parquet (`data/data.csv`)

#### 2. スキーマ構成
- **`text`** (`string`): 入力テキスト（発話やメッセージ）
- **`labels`** (`int64`): 0 から 11 までの数値で表現されたインテントカテゴリの整数ラベル

#### 3. ラベルID（整数）とカテゴリの対応表
| ラベルID | カテゴリ名 | 英語表記 |
| :---: | :--- | :--- |
| `0` | 挨拶・日常会話 | Greeting / Small Talk |
| `1` | 感謝・お礼 | Expression of Gratitude |
| `2` | 称賛・ポジティブなフィードバック | Praise / Positive Feedback |
| `3` | 同意・肯定 | Agreement / Affirmation |
| `4` | 否定・不満 | Disagreement / Discontent |
| `5` | 話題の提案・リクエスト | Topic Suggestion |
| `6` | 機能や仕組みに関する質問 | System Capabilities Inquiry |
| `7` | 改善提案・建設的フィードバック | Constructive Feedback |
| `8` | 不満・不平の表明 | Complaint / Dissatisfaction |
| `9` | 助けや説明の要求 | Help / Clarification Request |
| `10` | 提案への賛同・評価 | Evaluation / Agreement with Suggestion |
| `11` | 言語・設定の変更要求 | Language / Settings Change Request |

---

### 比較モデル

#### LightGBM

##### 1. モデル構成・特徴量
- **アルゴリズム**: TF-IDF 特徴量抽出 + 勾配ブースティング決定木 (`LGBMClassifier`)
- **ベクトル化**: `TfidfVectorizer(max_features=10000, ngram_range=(1, 2))`
- **推論器**: `LGBMClassifier(random_state=42, n_estimators=100, n_jobs=-1)`

##### 2. 学習設定・データ規模
- **訓練データ**: 8,042 サンプル
- **テストデータ**: 3,447 サンプル

##### 3. 学習実績および実行時の様子
- **所要時間**: 約 5.74 秒
- **使用特徴量数**: 677、Total Bins: 16,384
- **学習挙動**: 全12クラスの初期対数尤度スコアは約 -2.36 〜 -2.64 で安定収束

---

#### DistilBERT

##### 1. モデル構成
- **ベースモデル**: `distilbert/distilbert-base-uncased`
- **アーキテクチャ**: `AutoModelForSequenceClassification`（12クラス分類ヘッド）

##### 2. ハイパーパラメータ
- **エポック数**: 5 (`num_train_epochs=5`)
- **学習率**: `4e-5` (Cosine decay, `warmup_steps=100`)
- **最適化アルゴリズム**: AdamW (`weight_decay=0.01`, `adam_epsilon=1e-8`, `max_grad_norm=1.0`)
- **正則化**: ラベルスムージング `0.05`（過剰な確信を抑制し、確率キャリブレーションを向上）
- **バッチサイズ**: 16 (GPU per device) × 勾配蓄積 2 ステップ = 実効バッチサイズ 32
- **パディング**: 動的パディング (`DataCollatorWithPadding`, 最大許容長: 128)
- **演算精度**: FP32 (`fp16=False`, Pascal 世代 GPU における数値安定性確保)

##### 3. トレーニング実績および実行時の様子
- **ハードウェア環境**: CUDA 12.4, NVIDIA GeForce GTX 1050 with Max-Q Design 4GB
- **所要時間**: 約 4 分 53 秒 (293.1 秒)
- **総ステップ数**: 1,135 ステップ
- **検証スループット**: 最大 443.0 samples/sec (動的パディング効果)
- **エポック推移**:
  - Epoch 1: Val Loss `0.5095`, Val Accuracy `91.30%`
  - Epoch 2: Val Loss `0.4644`, Val Accuracy `94.29%` (★Best Checkpoint 採用)
  - Epoch 3: Val Loss `0.4716`, Val Accuracy `93.29%`
  - Epoch 4: Val Loss `0.4557`, Val Accuracy `93.54%`
  - Epoch 5: Val Loss `0.4628`, Val Accuracy `93.54%`

---

#### Gemini Flash Lite

##### 1. モデル構成
- **バックエンドモデル**: `gemini-3.5-flash-lite` (Google GenAI SDK)
- **アプローチ**: Pydantic スキーマ制約（Structured Outputs）によるゼロショット分類

##### 2. 堅牢性・実行時設計
- **並行実行**: `ThreadPoolExecutor(max_workers=10)`
- **リトライ制御**: 429 レートリミット等のエラーに対する指数バックオフ付き自動再試行
- **中間キャッシュ・レジューム機構**: 正常完了サンプルを `tmp/sample_{idx}.json` に即座に永続化し、中断時も未処理分から自動再開

##### 3. 実行時の様子および挙動分析
- **処理規模**: 3,447 件完全処理
- **リクエスト挙動**: 並行実行時のレートリミット（429）およびサーバー側キューイング遅延による指数バックオフ待機が発生
- **予測確信度傾向**: 多くのサンプルで `0.98`〜`1.0` の高い確信度を出力

---

#### Jev (`jev-1.13`)

##### 1. モデル構成およびプロンプト設計
- **バックエンドモデル**: `jev-1.13` (TypeSafe System One API)
- **プリミティブ**: `Choice` プリミティブによるゼロショット分類
- **クライテリア設計**: 全12クラスの意図に対して排他的かつ対照的な `criteria`（`what`, `not_for`, `examples`）を定義し、厳密な正規化確率分布を出力

##### 2. 堅牢性・実行時設計
- **ジッター付き指数バックオフ**: 一時的なネットワーク揺らぎやレートリミットに対して最大5回試行（1s, 2s, 4s, 8s, 16s + jitter）で自動再試行
- **Fail-Fast 原則**: リトライ上限超過や属性欠損に対してフォールバック値を捏造せず、即座に例外停止してデータの整合性を担保
- **安全な永続化**: 正常成功したサンプルのみを `tmp/sample_{idx}.json` に保存

##### 3. 実行時の様子および推論挙動分析
- **総所要時間**: 193.38 秒（約 3 分 13 秒）
- **実効スループット**: 約 17.83 samples/sec（毎分約 1,070 件）
- **セッション安定性**: 最大ギャップわずか 1.73 秒。クラッシュや再起動なしで単一セッション完走
- **スパイク挙動と自動復旧**: 3 秒を超過したサンプルは全 3,447 件中わずか 10 件（進捗 45% 付近に集中）。指数バックオフが自律的に稼働し、Fail-Fast 停止を起こさず自動復旧
- **モデル出力の確信度**: 平均確信度 0.95（中央値 1.00）。確信度 0.50 未満はわずか 55 件（1.6%）

---

## 実験結果

3,447 件の共通テストデータに対する全4モデルの性能・速度・コストのベンチマーク測定結果です。

### 1. 総合ベンチマーク測定結果テーブル

| モデル名 | 分類アプローチ | ROC-AUC (Macro) | Accuracy | Macro-F1 | 平均 (ms) | 最小値 (ms) | p25 (ms) | p50 (ms) | p75 (ms) | p90 (ms) | p99 (ms) | パレート最適 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | TF-IDF + GBDT (ローカル) | **0.9945** | 0.9066 | 0.9081 | 7.76 | 3.44 | 5.00 | 6.16 | 9.33 | 13.43 | 21.55 | ★ 最適 (Frontier) |
| **DistilBERT** | Transformer Fine-tuning (ローカル GPU) | **0.9970** | 0.9315 | 0.9333 | 8.03 | 5.54 | 6.74 | 7.35 | 8.75 | 10.00 | 17.31 | ★ 最適 (Frontier) |
| **Gemini 3.5 Flash Lite** | 生成型 LLM API (ゼロショット) | **0.9690** | 0.8840 | 0.8838 | 7715.26 | 713.28 | 2766.05 | 5686.80 | 11999.15 | 17346.87 | 22961.09 | - |
| **Jev (jev-1.13)** | 判断特化 System One API (ゼロショット) | **0.9708** | 0.8645 | 0.8605 | 560.21 | 441.24 | 502.83 | 527.97 | 559.11 | 599.30 | 875.41 | - |

### 2. 推論コスト比較（API 従量課金）

- **TypeSafe Jev (`jev-1.13`)**:
  - 入力単価: **`$0.042 / 1M tokens`**
  - 出力単価: **`$0.00 / 1M tokens`**（判定特化・テキスト生成を行わないため出力コスト完全ゼロ）
  - 3,447 件評価の総コスト: **約 $0.006**（1 セント未満）
- **Gemini Flash Lite (`gemini-3.5-flash-lite`)**:
  - 入力単価: `$0.30 / 1M tokens`
  - 出力単価: `$2.50 / 1M tokens`
  - 3,447 件評価の総コスト: 約 $0.158
- **コスト対比実績**:
  - Jev は Gemini Flash Lite と比較して入力単価が **約 1/7（86% 削減）**、出力単価は完全無料であり、総コスト比で **約 1/26（96% 削減）**。

---

## 実行手順

### 1. 環境構築

Python 3.12 および `uv` を使用します。

```bash
# 依存関係の同期（CUDA 12.4 PyTorch 対応）
uv sync

# API キーの設定
cp .env.sample .env
# .env を編集して GEMINI_API_KEY, JEV_API_KEY を設定
```

### 2. データセットの取得と分割

Hugging Face Hub からデータセットを取得し、訓練用（70%）および評価用（30%）の CSV を生成します。

```bash
uv run python 00_dataset/script/dataset.py
```

### 3. LightGBM モデルの訓練と評価

```bash
uv run python 01_lightgbm/script/train.py
uv run python 01_lightgbm/script/evaluate.py
```

### 4. DistilBERT モデルの訓練と評価

```bash
uv run python 02_distilbert/script/train.py
uv run python 02_distilbert/script/evaluate.py
```

### 5. Gemini Flash Lite モデルの評価

```bash
# スモークテスト（10サンプル）
uv run python 03_gemini_flash_lite/script/evaluate.py --smoke

# 本番推論（レジューム機能付き）
uv run python 03_gemini_flash_lite/script/evaluate.py
```

### 6. Jev モデルの評価

```bash
# スモークテスト（10サンプル）
uv run python 04_jev/script/evaluate.py --smoke

# 本番推論（並行推論・レジューム機能付き）
uv run python 04_jev/script/evaluate.py
```

### 7. 統合比較とレポート生成

全モデルの出力を検証・集約し、比較レポート（Markdown）とメトリクスサマリ（JSON）を自動生成します。

```bash
uv run python 05_summary/script/compare.py
```

### 8. テストおよび品質チェック

```bash
# 単体テスト全件実行
uv run pytest

# リントおよびフォーマットチェック
uv run ruff check .
uv run ruff format --check .

# 静的型チェック
uv run mypy .
```

---

## ディレクトリ構成

```text
jev-eval/
├── .github/workflows/ci.yml       # GitHub Actions CI (Ruff, Mypy, Pytest)
├── 00_dataset/                  # 共通データセットの取得と分割
│   ├── output/                  # train.csv (70%), test.csv (30%)
│   ├── script/                  # dataset.py (データ取得・分割スクリプト)
│   └── tests/                   # test_dataset.py (単体テスト)
├── 01_lightgbm/                 # LightGBM モデル実験 (TF-IDF + LightGBM)
│   ├── input/                   # train.csv, test.csv
│   ├── output/                  # model.joblib, predictions.csv, metrics.json
│   ├── script/                  # train.py, evaluate.py
│   └── tests/                   # test_lightgbm.py
├── 02_distilbert/               # DistilBERT モデル実験 (ファインチューニング)
│   ├── input/                   # train.csv, test.csv
│   ├── output/                  # model_weights/, label_mapping.json, predictions.csv, metrics.json
│   ├── script/                  # train.py, evaluate.py
│   └── tests/                   # test_distilbert.py
├── 03_gemini_flash_lite/        # Gemini Flash Lite モデル実験 (ゼロショットAPI)
│   ├── input/                   # test.csv
│   ├── tmp/                     # 個別サンプルの中間キャッシュ (sample_{idx}.json)
│   ├── output/                  # predictions.csv, metrics.json
│   ├── script/                  # evaluate.py
│   └── tests/                   # test_gemini.py
├── 04_jev/                      # Jev モデル実験 (判断特化 System One API)
│   ├── input/                   # test.csv
│   ├── tmp/                     # 個別サンプルの中間キャッシュ (sample_{idx}.json)
│   ├── output/                  # predictions.csv, metrics.json
│   ├── script/                  # evaluate.py
│   └── tests/                   # test_jev.py
├── 05_summary/                  # 統合比較・トレードオフ分析（コスト・パレート分析）
│   ├── output/                  # comparison_report.md, summary_metrics.json
│   ├── script/                  # compare.py (統合比較・レポート自動生成スクリプト)
│   └── tests/                   # test_compare.py (単体テスト・Fail First 異常系検証)
└── docs/                        # 実験計画・各ステップ実装計画書
```
