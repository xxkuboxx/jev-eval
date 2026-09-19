# データセット詳細仕様書 (`00_dataset/`)

本ドキュメントは、本プロジェクトで使用する共通データセット [`tanaos/synthetic-intent-classifier-dataset-v1`](https://huggingface.co/datasets/tanaos/synthetic-intent-classifier-dataset-v1) に関する詳細な情報およびデータ仕様をまとめたものです。

## 1. データセットの基本情報
- **Hugging Face Hub リポジトリ**: `tanaos/synthetic-intent-classifier-dataset-v1`
- **総レコード数**: 11,489 件
- **フォーマット**: CSV / Parquet (`data/data.csv`)

## 2. スキーマ構成とラベルマッピング
データセットの各レコードは以下のカラムを持ちます：

- **`text`** (`string`): 入力テキスト（発話やメッセージ）
- **`labels`** (`int64`): 0 から 11 までの数値で表現されたインテントカテゴリの整数ラベル。

### ラベルID（整数）とカテゴリの対応表
Hugging Face上のデータセット仕様に基づく、各整数ラベルに対応するインテントの内容は以下の通りです：
- **`0`**: 挨拶・日常会話 (Greeting / Small Talk)
- **`1`**: 感謝・お礼 (Expression of Gratitude)
- **`2`**: 称賛・ポジティブなフィードバック (Praise / Positive Feedback)
- **`3`**: 同意・肯定 (Agreement / Affirmation)
- **`4`**: 否定・不満 (Disagreement / Discontent)
- **`5`**: 話題の提案・リクエスト (Topic Suggestion)
- **`6`**: 機能や仕組みに関する質問 (System Capabilities Inquiry)
- **`7`**: 改善提案・建設的フィードバック (Constructive Feedback)
- **`8`**: 不満・不平の表明 (Complaint / Dissatisfaction)
- **`9`**: 助けや説明の要求 (Help / Clarification Request)
- **`10`**: 提案への賛同・評価 (Evaluation / Agreement with Suggestion)
- **`11`**: 言語・設定の変更要求 (Language / Settings Change Request)
