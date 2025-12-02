# 実装計画: vector_searchでCandleTextEmbedder使用

## 目的

- `examples/vector_search.rs` を `CandleTextEmbedder` ベースのサンプルに更新し、実際の Sentence Transformers モデルを使った自動埋め込みフローを示す。

## 背景

- 現状はデモ用 Embedder のみで、実運用を想定したモデルの使い方が伝えづらい。
- ユーザーから Candle 版を例として見たい要望があった。

## スコープ

- `PerFieldEmbedder` への登録部分を `CandleTextEmbedder` に差し替え。
- モデル名や必要 feature (`embeddings-candle`) に関するコメントを追加。
- fallback/エラーハンドルを整え、ドキュメント更新（実装レポート）を行う。

## ステークホルダー

- mosuka (メンテナ)
- Platypus 利用者

## スケジュール

| フェーズ | 作業内容 | 予定 |
| --- | --- | --- |
| 設計 | Candle 版の初期化・PerFieldEmbedder 組み合わせ検討 | 当日 |
| 実装 | `vector_search.rs` の書き換え | 当日 |
| 検証 | `cargo run --example vector_search --features embeddings-candle` | 当日 |

## リスク管理

- **モデルダウンロード失敗**: ネットワーク依存 → エラーを `expect` ではなく `?` で伝播し、メッセージを明示。
- **feature 未有効によるビルド失敗**: README 的なコメントで `--features embeddings-candle` を案内。
- **実行時間増大**: モデル初期化に時間がかかる → サンプル冒頭に注意書きを追加。

## 成果物

- 更新済み `examples/vector_search.rs`
- 実装レポート (docs/implementations)
