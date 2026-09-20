## Summary
TypeSafe System One (Jev) モデルを活用した評価モジュールの実装および Gemini Flash Lite 評価モジュールの統合・ドキュメント更新を行いました。

## Changes
1. **Jev Evaluation Module (`04_jev/`)**:
   - `04_jev/script/evaluate.py`: Jev APIを用いた選択肢問題（Choice）、スコア問題（Score）、数値問題（Noul）等の評価スクリプトを実装。
   - `04_jev/tests/test_jev.py`: Jev評価スクリプトに対する単体テストを実装。
   - `04_jev/README.md`: Jev評価の使い方や実行手順に関するドキュメントを整備。
2. **Gemini Flash Lite & Overall Integration (`03_gemini_flash_lite/`, root)**:
   - `03_gemini_flash_lite/` 実装の完了および単体テスト・評価スクリプトの整備。
   - `pyproject.toml`, `uv.lock`: 依存関係の追加・更新。
   - トップレベル [`README.md`](README.md): 全体の進捗・評価モジュールの最新状態を反映。

## Verification Results
- `uv run ruff check`, `uv run ruff format --check`, `uv run mypy .`: エラー・警告ゼロを確認。
- `uv run pytest`: 全15テストケース (dataset, lightgbm, distilbert, gemini, jev) すべて成功 (`15 passed in 20.85s`)。
