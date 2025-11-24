# 2025-11-24 VectorEngineQuery → VectorEngineSearchRequest リネーム計画

## 目的

- VectorEngine で用いる検索リクエスト型を `VectorEngineSearchResults` と対になる名前に揃え、API の責務をより明示する。
- ハイブリッド検索やサンプルコードを含め、`VectorEngineQuery` 参照をすべて `VectorEngineSearchRequest` に更新する（後方互換は保持しない）。

## 関連ファイル・ディレクトリ

- `src/vector/engine.rs`（型定義、検索実装、テスト用 helper）。
- `src/hybrid/engine.rs`, `src/hybrid/search/searcher.rs`（VectorEngineQuery を受け渡す箇所）。
- `examples/vector_search.rs`, `tests/vector_engine_scenarios.rs`（API 利用例）。
- ドキュメント類: `README.md`, `docs/vector_engine.md`, `docs/reports/2025-11-22-vector-collection-rename.md`（用語説明）。

## 実装ステップ

1. **型定義のリネーム** ✅ 2025-11-24 完了
   - `rg -l "VectorEngineQuery" | xargs sed -i 's/VectorEngineQuery/VectorEngineSearchRequest/g'` でコア構造体/impl と関連コメントを一括置換し、`VectorEngine::search` 引数や `FieldSelector` 等の型注釈も更新した。
2. **内部参照の一括更新** ✅ 2025-11-24 完了
   - `rg -l "vector_engine_query" | xargs sed -i 's/vector_engine_query/vector_engine_search_request/g'` を基に `src/hybrid/**`, `examples`, `tests` の参照を整理し、関連メソッド (`with_vector_engine_search_request`) まで整合させた。
3. **ドキュメントとサンプル表記の更新** ✅ 2025-11-24 完了
   - README・`docs/vector_engine.md`・サンプル/レポートの記述を新名称に差し替え、`examples/vector_search.rs` のチュートリアル文面も更新。
4. **ビルド検証** ✅ 2025-11-24 完了
   - `cargo fmt`, `cargo clippy`, `cargo test --all` を順番に実行し、エラー無く完了（テスト 628 件成功）。
5. **CHANGELOG 追記** ✅ 2025-11-24 完了
   - Unreleased セクションに `VectorEngineQuery` → `VectorEngineSearchRequest` の破壊的 rename を追記。

## リスク・考慮点

- 参照箇所が広範囲（ハイブリッド検索、テスト、サンプル）にわたるため、部分的な rename 漏れがビルド失敗に直結する。
- ドキュメントの画面コピーやテキスト説明に旧名が残ると混乱を招くため、`grep` ベースでの全文チェックが必要。
- 将来的にさらに命名整理を行う場合、この変更が別名との整合を左右するため、後続方針と齟齬がないか事前確認する。

## 推奨アクション

- `rg "VectorEngineSearchRequest" -n` で全面検索し、該当ファイルを上記ステップ順に処理する。
- リネーム後すぐに `cargo check` を走らせ、コンパイルエラー箇所を即時洗い出してからテストへ進む。
- CHANGELOG とドキュメント更新が完了した段階で semantic commit (`feat(vector): rename search request type` 等) を予定し、レビュー依頼時に破壊的変更である旨を明記する。
