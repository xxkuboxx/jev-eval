# 01_lightgbm: LightGBM モデル評価

TF-IDF 特徴量と LightGBM (`LGBMClassifier`) を用いたマルチクラス意図分類モデルの実験・評価結果を記載します。

## 実験結果サマリ

- **モデル構成**: `TfidfVectorizer(max_features=10000, ngram_range=(1, 2))` + `LGBMClassifier(random_state=42, n_estimators=100, n_jobs=-1)`
- **学習データ数**: 8,042 サンプル (`01_lightgbm/input/train.csv`)
- **テストデータ数**: 3,447 サンプル (`01_lightgbm/input/test.csv`)

### トレーニング実績
- **トレーニング所要時間**: 約 5.74 秒
- **学習時のログ・特徴**:
  - Auto-choosing row-wise multi-threading
  - Total Bins: 16,384, 使用特徴量数: 677
  - マルチクラス分類（全12クラス）の初期対数尤度スコア（Start training from score）は約 -2.36 から -2.64 の範囲で安定して学習を開始。

### 評価指標
- **ROC-AUC (Macro-average)**: `0.9945`
- **平均推論レイテンシ**: `3.98 ms/sample`
