# 02_distilbert: DistilBERT モデル評価

`distilbert/distilbert-base-uncased` を用いたテキスト分類モデルのファインチューニングおよび評価結果を記載します。

## 実験結果サマリ

- **モデル構成**: `distilbert/distilbert-base-uncased` (`AutoModelForSequenceClassification`)
- **ハイパーパラメータ**:
  - エポック数: 5 (`num_train_epochs=5`)
  - 学習率: `4e-5` (Cosine decay, `lr_scheduler_type="cosine"`)
  - ウォームアップステップ数: `100` (`warmup_steps=100`)
  - 最適化アルゴリズム: AdamW (`adam_epsilon=1e-8`, `weight_decay=0.01`)
  - 勾配クリッピング: `max_grad_norm=1.0`
  - ラベルスムージング: `0.05` (`label_smoothing_factor=0.05`)
  - バッチサイズ: 16 (`per_device_train_batch_size=16`, `gradient_accumulation_steps=2`, 実効バッチサイズ 32)
  - パディング方式: 動的パディング (`DataCollatorWithPadding`, 最大許容系列長: 128)
  - 演算精度: FP32 (`fp16=False` ※Pascal世代GPUにおける数値安定性・NaN防止のため)
- **学習データ数**: 8,042 サンプル (`02_distilbert/input/train.csv`, 内部分割: 訓練 7,237 / 検証 805)
- **テストデータ数**: 3,447 サンプル (`02_distilbert/input/test.csv`)

### トレーニング実績
- **トレーニング所要時間**: 約 4 分 53 秒 (293.1 秒, CUDA 12.4, NVIDIA GeForce GTX 1050 with Max-Q Design 4GB 環境)
- **総ステップ数**: 1,135 ステップ (5 エポック)
- **学習時のログ・特徴**:
  - 動的パディング（`DataCollatorWithPadding`）の導入により、検証時の評価スループットは最大 **443.0 samples/sec** を記録。
  - 各エポックごとの検証メトリクス推移:
    - **Epoch 1**: Validation Loss: `0.5095`, Validation Accuracy: `91.30%` (Runtime: 1.82s)
    - **Epoch 2**: Validation Loss: `0.4644`, Validation Accuracy: **`94.29%`** (★Best Checkpoint)
    - **Epoch 3**: Validation Loss: `0.4716`, Validation Accuracy: `93.29%`
    - **Epoch 4**: Validation Loss: `0.4557`, Validation Accuracy: `93.54%`
    - **Epoch 5**: Validation Loss: `0.4628`, Validation Accuracy: `93.54%`
  - `load_best_model_at_end=True` により、検証精度（Accuracy）が最大（94.29%）となった Epoch 2 時点の重みが自動的に最終モデルとして採用・保存されました。
  - ラベルスムージング（`0.05`）の導入により、モデルの過剰な確信（Overconfidence）が抑制され、確率予測のキャリブレーションと汎化性能が向上しました。

### 評価指標
- **ROC-AUC (Macro-average)**: **`0.9970`**
- **平均推論レイテンシ**: **`8.03 ms/sample`**
- **テストサンプル総数**: 3,447 サンプル

---

## ディレクトリ構成
- **`script/train.py`**: データセット（`02_distilbert/input/train.csv`）を読み込み、DistilBERTモデルをファインチューニングしてモデル重み（`02_distilbert/output/model_weights/`）およびラベルマッピング（`02_distilbert/output/label_mapping.json`）を保存します。
- **`script/evaluate.py`**: テストデータ（`02_distilbert/input/test.csv`）に対して1件ずつ推論を行い、予測確率・厳密なレイテンシ（ms/sample）を計測し、`metrics.json` および `predictions.csv` を出力します。
- **`tests/test_distilbert.py`**: トークナイザーの動作、動的パディング、推論スモークテスト、および保存されたモデル重み・メトリクスの非NaN検証ユニットテスト。

## 実行手順

### 1. 依存関係のインストール
`pyproject.toml` に設定された PyTorch CUDA 12.4 インデックスより環境が構築されます。
```bash
uv sync
```

### 2. トレーニング実行
```bash
uv run python 02_distilbert/script/train.py
```

### 3. 評価・推論実行
```bash
uv run python 02_distilbert/script/evaluate.py
```

### 4. テスト実行
```bash
uv run pytest 02_distilbert/tests/
```
