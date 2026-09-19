## Summary
ステップ 1 (`01_lightgbm/`) のマルチクラス意図分類モデル評価パイプラインを実装し、トレーニング・推論・評価・テストをすべて完了しました。

## Changes
1. **LightGBM モデル学習スクリプト (`01_lightgbm/script/train.py`)**:
   - `01_lightgbm/input/train.csv` の読み込み
   - `TfidfVectorizer` (max_features=10000, ngram_range=(1, 2)) と `LGBMClassifier` によるパイプライン構築・学習
   - 学習済みモデルの `01_lightgbm/output/model.pkl` への保存
2. **推論・評価スクリプト (`01_lightgbm/script/evaluate.py`)**:
   - `01_lightgbm/input/test.csv` を用いた推論と1件あたりの平均レイテンシ計測 (`3.98 ms/sample`)
   - Macro-average ROC-AUC の算出 (`0.9945`)
   - 予測結果 (`predictions.csv`) および評価指標 (`metrics.json`) の出力
3. **単体テスト (`01_lightgbm/tests/test_lightgbm.py`)**:
   - パイプラインのスモークテスト実装 (`pytest` で全テスト通過確認済み)
4. **ドキュメント更新**:
   - トップ階層 [`README.md`](README.md:1) および [`01_lightgbm/README.md`](01_lightgbm/README.md:1) にトレーニング実績や実行手順を詳細に記載

## Verification Results
- **PyTest**: 全4件のテストが正常に成功 (3.06s)
- **Ruff & Mypy**: リント・型チェックともにゼロエラー確認済み
