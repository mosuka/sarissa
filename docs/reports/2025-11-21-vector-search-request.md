# 2025-11-21 VectorSearchRequest 呼び出し調査

## 背景

`VectorCollection` 側で互換アダプタ (`search_with_legacy_request`) を廃止したため、既存モジュールから直接 `VectorCollectionQuery` を構築する必要が出てきた。まずは現在の `VectorSearchRequest` 依存箇所を全量把握し、どのモジュールから移行作業を進めるべきかを整理する。

## 依存箇所サマリ

| ファイル | 用途 | 移行上の論点 |
| --- | --- | --- |
| `src/vector/search/searcher.rs` | `VectorSearchRequest`/`VectorSearchResults` の定義と `VectorSearcher` トレイト | 旧 API の中心。`VectorCollection` に統一するなら本モジュール自体を段階的に縮退させる必要がある。|
| `src/vector/engine.rs` | エンドユーザ向けの検索エンジンが `VectorSearchRequest` を直接受け取って `VectorSearcher` に委譲 | `VectorCollectionQuery` を生成するラッパ関数を追加し、検索/カウントで `VectorCollection` を呼ぶように差し替える。|
| `src/hybrid/engine.rs` | `HybridSearchRequest` から `VectorSearchRequest` を組み立て、`VectorEngine` へ渡す | ハイブリッド経路でも `VectorCollectionQuery` を直接構築できるよう、ベクター側のサブセット (top_k, min_similarity 等) を `VectorCollectionQuery::builder()` に写像する必要がある。|
| `src/hybrid/search/searcher.rs` | `HybridSearchRequest` 内部で `VectorSearchParams` を保持 | API ドキュメントや `HybridSearchParams` の意味付けを `VectorCollectionQuery` ベースに更新する必要あり。|
| `src/vector/index/field.rs` | Legacy フィールドアダプタが `VectorSearchRequest` を生成し、`VectorSearcher` で実行 | Doc-centric パスに収束させるには、ここを `VectorFieldReader` から `VectorCollection::search` へルーティングする別構造に書き換える必要がある。|
| `README.md` | ユーザ向けコードサンプルが `VectorSearchRequest` を推奨 | ドキュメント更新タスク (#4) で `VectorCollectionQuery` サンプルへ差し替える必要がある。|

## 詳細メモ

- `VectorSearchRequest` は `query`, `params`, `field_name` のシンプルな構造で、`VectorCollectionQuery` が持つ `FieldSelector`, `MetadataFilter`, `score_mode` 等を表現できない。
- `VectorEngine`/`HybridEngine` の検索エントリーポイントはすべて `VectorSearchRequest` 型を公開 API として露出しているため、段階的な互換レイヤーを用意するか、メジャーバージョンで破壊的変更として差し替えるかを決める必要がある。
- `vector/index/field.rs` 配下のレガシーアダプタは doc-centric レイヤーの `VectorFieldReader` と `VectorSearcher` を橋渡ししている。ここを更新しない限り、既存の index 実装 (Flat/HNSW/IVF) は `VectorSearchRequest` を必要とし続ける。

## 次のアクション案

1. `VectorEngine` に `VectorCollection` (doc-centric) を注入する経路を設計し、`VectorSearchRequest` → `VectorCollectionQuery` 変換ヘルパを追加する。
2. `HybridEngine` は上記ヘルパを利用して `VectorCollection` 検索を呼び出し、旧 `VectorSearcher` への依存を段階的に削減する。
3. `vector/index/field.rs` のレガシー読み取り経路を doc-centric フィールドリーダーへ置換し、最終的に `VectorSearchRequest` 型を廃止する。
4. README / `docs/vector_collection.md` / `docs/tasks.md` を `VectorCollectionQuery` 中心の記述に揃える。
