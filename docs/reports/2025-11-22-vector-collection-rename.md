# 2025-11-22 VectorEngine → VectorEngine リネーム調査

## 現状の構成

- doc-centric 実装は `src/vector/collection.rs` の `VectorEngine*` 系型に集約されている。
- 旧来の `VectorEngine` (`src/vector/engine.rs`) はレガシー VectorIndex API へのブリッジとして存続し、内部で `VectorEngineSearchRequest` を直接組み立てている。
- 公開 API/ドキュメントでは `platypus::vector::engine::*` を参照する箇所が多数存在（`README.md`, `docs/vector_collection.md`, `examples/vector_search.rs`, `tests/vector_collection_scenarios.rs` 等）。
- ハイブリッド系 (`src/hybrid/*.rs`) はすでに `VectorEngineSearchRequest`/`VectorEngineSearchResults` を直接扱っており、旧 `VectorEngine` には依存していない。

## リネーム対象と影響範囲

- 型/モジュール名: `VectorEngine`, `VectorEngineConfig`, `VectorEngineSearchRequest`, `VectorEngineFilter`, `VectorEngineSearchResults`, `VectorEngineHit`, `VectorEngineStats` など十数個。
- モジュールパス: `platypus::vector::engine` -> `platypus::vector::engine`（もしくは `vector::collection` を削除して `vector::engine` に実装を移動）。
- 周辺アセット: `docs/vector_collection.md`, `resources/vector_collection_sample.json`, `tests/vector_collection_scenarios.rs`, `examples/vector_search.rs` の命名・記述。
- Downstream 互換性: crate 利用者が `use platypus::vector::engine::VectorEngine;` のように呼んでいるため、単純な rename は破壊的変更になる。

## 実現シナリオ

1. **互換フェーズ**
   - 旧 `VectorEngine` 型を `deprecated` に設定し、内部実装を `VectorEngine` へ全面委譲。
   - 新 API として `pub use vector::collection::VectorEngine as DocVectorEngine;` 等を提供し、`HybridEngine` などはこの新別名を使う。
   - ドキュメントでは「将来 `VectorEngine` が `VectorEngine` に置き換わる」旨を周知。
2. **完全置換フェーズ**
   - `src/vector/engine.rs` のレガシー実装を削除。
   - `VectorEngine` を `VectorEngine` へリネーム（ファイル名/モジュール名も `collection` から `engine` へ移動）。
   - 旧名との互換性確保のため `pub use crate::vector::engine::{VectorEngine as VectorEngine, ...};` といった type alias を少なくとも 1 リリース維持。
   - JSON やドキュメント資産（サンプルファイル/テスト名）も「collection」→「engine」へ追従。
3. **クリーンアップフェーズ**
   - `docs/vector_collection.md` を `docs/vector_engine.md` に改称し、README/タスク文書から旧語を撤去。
   - alias/非推奨 API を削除。

## リスク・考慮点

- **公開 API 破壊**: rename は crate の semver メジャーアップデートを伴う。互換 alias を挟まないと利用者側で大規模修正が発生する。
- **ドキュメント乖離**: 一斉改名時に docs / README / tests / サンプルを揃えないと、利用者が新旧名称の混在で混乱する。
- **概念のずれ**: `VectorEngine` という名前は「doc-centric で複数フィールドを抱える集合」という性質を説明している。一方 `VectorEngine` は旧 API からの継承であり、将来的に doc-centric 以外のストア（例: sharded collections）を追加する場合でも名称がぶれないか要検討。

## 推奨アクション

1. まず `VectorEngine` を正式 API として前面に出し、`VectorEngine`（旧）を非推奨化 + `VectorEngine` への type alias に置き換える（例: `pub type VectorEngine = VectorEngine;`）。これで利用者は新構造を意識しつつコード修正なしで移行可能。
2. alias 期間中にドキュメント・サンプルを `vector::engine::VectorEngine`（実体は `VectorEngine`）へ差し替え、`vector::collection` モジュールは re-export のみにする。
3. 次のメジャーリリースで `vector::collection` モジュールを削除し、ファイル自体を `src/vector/engine.rs` に移す（または `collection.rs` を `engine/doc_centric.rs` などへ分割し、`mod` 再構成）。
4. サンプル/テスト/ドキュメント資産のファイル名（`vector_collection_*`）を新名称へ揃えつつ、必要ならリダイレクト的な注意書きを残す。

## 破壊的変更を許容した場合のリネーム案

- ゴール: `VectorEngine*` という命名を完全に排し、すべて `VectorEngine*` に統一する。モジュールパスも `platypus::vector::engine` のみとし、`vector::collection` 自体を廃止する。

### 実施ステップ

1. **モジュール再構成**
   - `src/vector/collection.rs` を `src/vector/engine.rs` に直接統合（既存ファイルを置換）。
   - `mod collection;` を削除し、`pub mod engine;` のみ残す。`vector.rs` から `pub mod collection;` を除去。
2. **型/関数リネーム**
   - `VectorEngine` → `VectorEngine`、`VectorEngineConfig` → `VectorEngineConfig`、`VectorEngineSearchRequest` → `VectorSearchQuery`（など最終命名に合わせて一括変換）。
   - 派生型（`VectorEngineFilter`, `VectorEngineSearchResults`, `VectorEngineHit`, `VectorEngineStats`…）も同様に `VectorEngine` 起点の名前へ変更。
   - `FieldSelector`, `MetadataFilter` など独立概念は名称維持でよい。
3. **旧 VectorEngine の削除**
   - `src/vector/engine.rs` にあったレガシー実装を完全削除（必要なら `src/vector/legacy_engine.rs` などに退避して `#[deprecated]` 付きで re-export するが、破壊的変更を許容するなら廃止で良い）。
4. **use 宣言・API 更新**
   - `use platypus::vector::engine::...` を全リポジトリ検索で `vector::engine` に置換。
   - `HybridEngine`, `HybridSearchRequest`, `examples`, `tests`, `docs` に出現する `VectorEngine*` を新名称へ書き換え。
5. **リソース/ドキュメント改名**
   - `docs/vector_collection.md` → `docs/vector_engine.md`。
   - `resources/vector_collection_sample.json` → `resources/vector_engine_sample.json` 等。
   - `tests/vector_collection_scenarios.rs` → `tests/vector_engine_scenarios.rs`。
   - README の「Doc-centric VectorEngine」セクションを「Doc-centric VectorEngine」に改称、コードスニペットの `use` も更新。
6. **Crate 公開 API の調整**
   - `pub use` で新しい型をエクスポートし、不要になった `vector::collection` モジュールを完全削除。
   - Cargo feature やドキュメント内のリンクをすべて新名称に揃える。

### 補足

- 破壊的変更を許容するため、互換 alias や `deprecated` アノテーションは不要。代わりに CHANGELOG やリリースノートで rename を明記する。
- リネームは `git mv` を使って履歴を保ちつつ、`cargo fmt`/`cargo clippy` で最終確認する。

この手順なら、一度に巨大な rename を行うよりも安全に全面置換が可能で、`VectorEngine` → `VectorEngine` リネーム後も利用者影響を最小化できます。
