# 実装計画: vector_searchのPerFieldEmbedder化

## 目的

- `examples/vector_search.rs` において `PerFieldEmbedder` を利用し、フィールドごとに異なる `TextEmbedder` を切り替えるサンプルを提供する。

## 背景

- `examples/lexical_search.rs` では `PerFieldAnalyzer` を使い分ける事例があり、概念対応としてベクトル検索例でも per-field の仕組みを示すと理解が進む。
- 現状の `vector_search.rs` は単一の `DemoTextEmbedder` を直接登録しており、複数 Embedder を差し替えるパターンが伝わりづらい。

## スコープ

- `examples/vector_search.rs` 内の `VectorEngine` 構築ロジックを `PerFieldEmbedder` ベースに変更。
- タイトル用・本文用に別々の `DemoTextEmbedder`（またはパラメータ違い）を用意し、`PerFieldEmbedder` に登録。
- ドキュメント投入・検索フローは現状を維持しつつ、Embedder 差し替えのログ・コメントを追加。

## ステークホルダー

- プロジェクトメンテナ (mosuka)
- Platypus 利用者

## スケジュール

| フェーズ | 作業内容 | 予定 |
| --- | --- | --- |
| 設計 | PerFieldEmbedder 適用方法の検討 | 当日 |
| 実装 | `vector_search.rs` 書き換え | 当日 |
| 検証 | `cargo run --example vector_search` | 当日 |

## リスク管理

- **Embedder の切り替え漏れ**: `PerFieldEmbedder` にフィールド名を正しく登録しないと panic/無効ベクトルになる → テスト実行で確認。
- **コード複雑化**: サンプルが冗長になる → コメントで意図を説明し、余分な抽象化は避ける。

## 成果物

- 更新された `examples/vector_search.rs`
- 実装レポート (docs/implementations)
