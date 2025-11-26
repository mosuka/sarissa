# Tasks

## VectorEngine SegmentPayload DSL 移行計画 (作成日: 2025-11-26)

### 関連ファイル・ディレクトリ

- `src/vector/core/document.rs` (FieldPayload, SegmentPayload, StoredVector 定義)
- `src/vector/engine.rs` / `src/hybrid/engine.rs` (ingestion と検索フロー)
- `src/vector/core/document/` 配下 (segment/field ヘルパー群)
- `docs/reports/vectorengine_multimodal_report_20251125.md` (要件・背景)
- `examples/*.rs`, `tests/vector_engine_scenarios.rs` (サンプル・回帰テスト)

### 実装ステップ

1. **FieldPayload DSL 下準備**

    - [ ] `FieldPayload` に `segments: Vec<SegmentPayload>` を追加し、既存 `text_segments` をラップする互換メソッドを実装。
    - [ ] `DocumentPayload` 生成まわり (`document::field`, `add_text_field` など) を SegmentPayload API 経由に移行。

2. **PayloadSource / VectorType 対応**

    - [ ] `PayloadSource` enum と `SegmentPayload` ストラクチャを定義し、`VectorRole` 呼称からのマイグレーション層を用意。
    - [ ] `EmbedderRegistry` に `supports(source: &PayloadSource)` を追加し、対応 embedder へ委譲。

3. **VectorEngine ingest リファクタリング**

    - [ ] `VectorEngine::embed_field_payload` を SegmentPayload 単位の map/reduce パイプラインへ置き換え。
    - [ ] feature flag `embeddings-multimodal` ガード下で画像/外部ベクトルを処理するフローを追加。

4. **Query / Hybrid 側対応**

    - [ ] `VectorEngineSearchRequest`・`hybrid::engine` リクエスト DSL を SegmentPayload 互換形式に揃える。
    - [ ] 既存 JSON API（examples, tests）のフィールド名を `vector_type`, `PayloadSource` ベースに更新。

5. **移行サポートと検証**

    - [ ] 旧 API 呼び出し (text_segments) から新 DSL への deprecation ガイドとログを追加。
    - [ ] `cargo test`, `cargo clippy`, `cargo fmt` を feature on/off 両方で実行し、回帰テストを拡充。

### リスク・考慮点

- 互換層: 既存 `FieldPayload::add_text_segment` をすぐ廃止できないため、二重管理期間のバグ混入に注意。
- Embedder 能力差: 画像非対応の embedder でマルチモーダルセグメントを渡した際のエラーハンドリング仕様を明確化する必要がある。
- メモリ/性能: `PayloadSource::Bytes` で大容量を扱う際のコピーコスト、スレッドプール設計の見直しが必須。
- API 互換性: 外部クライアントが利用する JSON DSL の変更は段階的ロールアウトと十分なドキュメント整備が必要。

### 推奨アクション

1. 上記ステップ 1-2 を feature flag 背景で先行実装し、既存機能を壊さない形で SegmentPayload を内部的に導入する。
2. ステップ 3 以降の大規模変更はブランチ分割し、性能検証と混在期間のサポート方針 (deprecation 期間) を docs に追記する。
3. ハイブリッド/検索 DSL の整合性を確認するため、`examples/` と `tests/` にマルチモーダルシナリオを追加し、CI に feature flag on/off matrix を組み込む。

