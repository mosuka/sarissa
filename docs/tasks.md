# 2025-11-23 VectorCollection → VectorEngine リネーム計画（破壊的変更前提）

## 目的

- Doc-centric ベクトル実装を唯一の VectorEngine として再定義し、`VectorCollection*` という命名とモジュールを廃止する。
- コード／ドキュメント／テスト／リソースのすべてを新名称へ統一し、旧 API は完全削除する。

## 関連ファイル・ディレクトリ

- `src/vector/collection.rs` → 新しい `src/vector/engine.rs` 本体として統合。
- `src/vector/engine.rs`（旧実装）、`src/vector/index/*.rs`, `src/vector/core/*.rs`。
- `src/hybrid/**`, `examples/vector_search.rs`, `tests/vector_engine_scenarios.rs`。
- ドキュメント類: `README.md`, `docs/vector_engine.md`, `docs/reports/**`。
- リソース: `resources/vector_engine_sample.json`。

## 作業ステップ

1. **モジュール再構成** ✅ 2025-11-23 完了
   - `src/vector/collection.rs` の中身を `src/vector/engine.rs` に移動し、`mod collection;` を削除済み。
   - `pub use` などで `platypus::vector::engine::*` のみを公開 API にする。

2. **型と構造体のリネーム** ✅ 2025-11-23 完了
   - `VectorCollection` → `VectorEngine`、`VectorCollectionConfig` → `VectorEngineConfig`、`VectorCollectionQuery` → `VectorEngineQuery` をコード全体で確認済み。
   - `VectorCollectionFilter`, `VectorCollectionSearchResults`, `VectorCollectionHit`, `VectorCollectionStats` など派生型も未使用であることを `rg "VectorCollection" -n` で検証。

3. **周辺コードの更新** ✅ 2025-11-23 完了
   - `rg "platypus::vector::collection" -n` / `rg "crate::vector::collection" -n` でソース内の旧モジュール参照が存在しないことを確認。
   - `src/hybrid/**`, `examples`, `tests`, `src/vector/*.rs` はすべて `vector::engine` API へ切り替わっており、互換メソッドも削除済み。

4. **リソース／テスト／ドキュメント名の変更** ✅ 2025-11-23 完了
   - `ls resources` / `ls tests` で `vector_engine_*` 系ファイルを確認し、旧 `vector_collection_*` ファイルが存在しないことを確認。
   - `docs/vector_engine.md`、`README.md` のサンプル/導線が `resources/vector_engine_sample.json` と `tests/vector_engine_scenarios.rs` を参照していることを目視確認し、`git grep -n "vector_collection"` で残存参照が無いことを確認。

5. **検証** ✅ 2025-11-23 完了
   - `cargo fmt`, `cargo clippy`, `cargo test --all` を順番に実行し、いずれもエラーなく終了したことをターミナルで確認（テストは 600+ 件のユニットテストがすべて成功）。
   - 破壊的変更を明示するため `CHANGELOG.md` を新規作成し、「Doc-centric VectorEngine へ全面置換」を Breaking Changes として追記。
