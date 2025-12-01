# 実装計画: lexical_search例のDocument::builder移行

## 目的

- `examples/lexical_search.rs` が `Document::builder()` を用いてインデクシングするようリファクタリングし、`JsonlDocumentConverter` への依存を解消する。

## 背景

- `boolean_query` 例では `Document::builder()` を使用しており、学習コストが低い。
- `lexical_search` 例も同様のスタイルに統一することで、利用者が Document API を理解しやすくする。

## スコープ

- `examples/lexical_search.rs` のドキュメント投入処理の差し替え。
- JSONL 相当のサンプルデータをコード内に保持する `sample_documents()` ヘルパーの実装。
- 動作検証のための `cargo run --example lexical_search`。

## ステークホルダー

- 開発者: プロジェクトメンテナ (mosuka)
- ドキュメント利用者: Platypus 利用開発者

## スケジュール

| フェーズ | 作業内容 | 予定 |
| --- | --- | --- |
| 調査 | 既存例と `Document::builder` API の確認 | 当日 |
| 実装 | 例の差し替えとヘルパー追加 | 当日 |
| 検証 | `cargo run --example lexical_search` | 当日 |

## リスク管理

- **データ不整合**: JSONL とコード内データの乖離 → JSONL を参照し、同値のフィールドを作成。
- **ビルド失敗**: 新規関数や import ミス → `cargo fmt`/`cargo run` で検証。

## 成果物

- 更新済み `examples/lexical_search.rs`
- 実装レポート (docs/implementations)
