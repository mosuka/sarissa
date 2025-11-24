# 2025-11-21 ドキュメントセントリック検索 調査報告

## 背景

VectorEngine を正規の検索 API として使うために `VectorEngine` / `HybridEngine` を含む周辺レイヤーを移行中。現状は互換層で旧 `VectorSearchRequest` に変換しているため、doc-centric なクエリ機能を完全には活かせていない。残りの作業範囲を特定するため、関連モジュールとドキュメントを確認した。

調査対象: `src/vector/engine.rs`, `src/hybrid/engine.rs`, `src/hybrid/search/merger.rs`, `docs/vector_engine.md`, `docs/tasks.md` (2025-11-21 時点)。

## 調査結果

### 1. VectorEngine 互換層の制約

- `VectorEngine::build_legacy_request` (`src/vector/engine.rs`) は以下の制限を持つ:
  - クエリベクターは 1 件のみ許容 (`query.query_vectors.len() > 1` でエラー)。
  - フィールド指定も `FieldSelector::Exact` 1 件のみ対応。`Prefix`/`Role` ベースの選択や複数フィールドは未サポート。
  - `VectorScoreMode::LateInteraction` と `MetadataFilter` は即座にエラーとして弾かれる。
  - `overfetch` は `top_k` 拡張にしか活用されず、フィールド別の overfetch や weight 変換は未実装。
- そのため VectorEngine 側で追加された `FieldSelector`, `MetadataFilter`, 複数ベクター／ロールといった doc-centric 機能は互換層で潰れてしまう。

### 2. HybridEngine からの呼び出し

- `HybridEngine::build_vector_engine_search_request` (`src/hybrid/engine.rs`) は生の `Vector` 1 本からデフォルト `VectorEngineSearchRequest` を生成するだけで、`HybridSearchRequest` からの `FieldSelector` やメタデータ条件を受け取る経路が無い。
- Hybrid 検索時に得られる `VectorEngineSearchResults` の `field_hits` やメタデータは `ResultMerger` (`src/hybrid/search/merger.rs`) で利用されておらず、単に doc-id 単位のスコアに落とし込まれる。
- その結果、doc-centric 検索で意図した「フィールド別ウェイト」「ドキュメントメタデータ連動」等が Hybrid 経路からは観測できない。

### 3. サンプル / E2E テスト不足

- `vector::engine` のユニットテストは単一フィールド・単一クエリでの動作検証のみで、`FieldSelector` やメタデータフィルタを扱うケースが無い。
- `hybrid::engine` には構造テストしかなく、実際に VectorEngineSearchRequest による doc-centric 検索を通す統合テストが不足している。
- `docs/vector_engine.md` にも Hybrid 経路での利用例や `MetadataFilter` の活用例が未掲載。

## 実現までに必要な主な作業

1. **VectorEngine の互換層強化**
   - 複数 `QueryVector` と `FieldSelector` (Exact/Prefix/Role) を解析し、フィールドごとの `VectorSearchRequest` を組み立てる仕組みを追加。
   - `MetadataFilter` を `VectorSearcher` に中継できるよう、検索前にレジストリ経由で doc/field を絞り込むか、暫定的に CPU フィルタを挟む。
   - `VectorScoreMode` ごとのスコア合成ロジックを doc-centric 仕様に合わせて拡張（特に WeightedSum でのフィールドウェイト・クエリウェイトの反映）。

2. **HybridEngine / ResultMerger の doc-centric 化**
   - `HybridSearchRequest` に VectorEngine 向けパラメータ（フィールド指定、メタデータフィルタ、score_mode 等）を追加し、`HybridEngine::build_vector_engine_search_request` で反映。
   - `ResultMerger` で `field_hits` とメタデータを使った可視化・加重スコアリング（例: フィールド単位で別重みを掛けて lexical スコアと融合）を実装。
   - Vector 結果のみでも `document_store` にメタデータを渡せるよう、`VectorEngineSearchResults` から補完する導線を追加。

3. **E2E テストとドキュメントの更新**
   - 複数フィールド・複数ロールを含む `DocumentVectors` サンプルを追加し、`VectorEngine`/`HybridEngine` 双方で doc-centric クエリを検証する統合テストを新設。
   - `docs/vector_engine.md` と README に Hybrid からの呼び出し例、`MetadataFilter` のチュートリアル、score_mode の使い分けを追記。

以上より、doc-centric 検索を実用レベルにするには「互換層の機能拡張」「Hybrid 経路の対応」「E2E テスト + ドキュメント整備」の 3 軸での追加実装が必要。現状は互換層での制約が大きく、VectorEngine の機能を直接 expose できていないため、まず VectorEngine の変換ロジックを拡張し、その上で Hybrid 経路とテストを整備する順序を推奨する。

## 2025-11-22 追加メモ

- `resources/vector_engine_sample.json` と `tests/vector_engine_scenarios.rs` を追加し、MemoryStorage 上で doc-centric クエリ（`VectorScoreMode` / `MetadataFilter` / `FieldSelector`）を検証する統合テストを整備済み。`cargo test --test vector_engine_scenarios` で再現可能。
- `examples/vector_search.rs` を同サンプルデータを読み込む doc-centric デモへ刷新。`VectorEngineFilter` や `VectorScoreMode::MaxSim/WeightedSum` の具体例を CLI なしで確認できる。
- README / `docs/vector_engine.md` にサンプルデータとテスト導線を追記済み。今後は Hybrid 経路の手順や CLI での呼び出し例を追補する予定。
- 依然として `VectorEngine` 互換層と `HybridEngine`/`ResultMerger` の doc-centric 対応が未完了のため、次フェーズで重点的に実装する。
