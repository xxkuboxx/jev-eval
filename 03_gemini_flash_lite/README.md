# Gemini Flash Lite Model Evaluation (`03_gemini_flash_lite/`)

本ディレクトリは、[`docs/experiment_plan.md`](../docs/experiment_plan.md) で定義された **ステップ 3 (`03_gemini_flash_lite/`)** の実験環境であり、Google の `gemini-2.5-flash` モデルを用いた構造化出力によるマルチクラス意図分類タスクの推論および評価を行います。

## ディレクトリ構成

```text
03_gemini_flash_lite/
├── input/                   # テストデータ (test.csv)
├── output/                  # 成果物 (predictions.csv, metrics.json)
├── script/                  # 実装スクリプト
│   └── evaluate.py          # 推論・評価用スクリプト
└── tests/                   # テストコード
    └── test_gemini.py       # 単体テスト
```

## 実行方法

1. **APIキーの設定**
   環境変数 `GEMINI_API_KEY` に Google AI Studio 等で発行された API キーを設定してください。
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```

2. **評価・推論スクリプトの実行**
   ```bash
   # 動作確認用スモークテスト（先頭10サンプルのみ・結果保存なし）
   uv run python 03_gemini_flash_lite/script/evaluate.py --smoke

   # 本番推論・評価（エラー等で中断した場合、`03_gemini_flash_lite/tmp/` に保存された途中結果を自動で再利用し、処理済みサンプルをスキップしてリトライ実行されます）
   uv run python 03_gemini_flash_lite/script/evaluate.py
   ```

3. **単体テストの実行**
   ```bash
   uv run pytest 03_gemini_flash_lite/tests/
   ```

## 実行時の実績・傾向分析 (`03_gemini_flash_lite/tmp/` の実測データに基づく知見)

全 3,447 件のテストサンプルに対する `03_gemini_flash_lite/tmp/` 内のキャッシュファイル群（各サンプルの予測結果、レイテンシ、信頼度）の実測データを分析したところ、以下の具体的な実行傾向が確認されました：

- **レイテンシの傾向とリトライ挙動**:
  - 平均レイテンシは約 **7,715 ms**（約 7.7 秒）、最小レイテンシは **713 ms**、最大レイテンシは **30,771 ms** でした。
  - 数秒を大きく上回る高レイテンシのサンプルが一部存在しており、これはAPIのレートリミット（429）等に起因する指数バックオフ自動リトライ、あるいは並行度（`max_workers=10`）に起因するサーバー側のキューイング遅延が発生したことを示しています。
- **モデルの予測確信度 (Confidence)**:
  - 構造化出力による予測結果の確信度は大半が極めて高く（多くが `0.98`〜`1.0`）、モデルが意図分類に対して非常に高い確実性を持って判定を下していることが分かります。
- **キャッシュ/レジュームの実績**:
  - 各サンプルの実行結果が `sample_{idx}.json` として個別に一時保存されているため、大規模なデータセット（3,447件）の評価中に中断や例外が生じても、次回実行時に処理済みサンプルを完全にスキップして未処理分から効率よく再開できる運用実績が確認されました。


