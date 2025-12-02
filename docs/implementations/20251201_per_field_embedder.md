# 実装レポート: vector_searchのPerFieldEmbedder化

## 実装内容

- `examples/vector_search.rs` に `PerFieldEmbedder` を導入し、タイトルと本文で異なる `TextEmbedder` を使い分ける例へ刷新。
- `DemoTextEmbedder` をフィールドごとに `name` と `scale` を持つ構造体へ拡張し、`PerFieldEmbedder` に登録できるよう変更。
- VectorEngine 構築時にデフォルト (本文) 用 Embedder を設定し、`title_embedding` にはスコアを強調する別インスタンスを割り当て。

## 手順

1. `PerFieldEmbedder` を import し、`VectorEngine` 登録部で `DemoTextEmbedder` を 2 種類生成。
2. 本文用 Embedder をデフォルトとして `PerFieldEmbedder::new` に渡し、タイトル用 Embedder を `add_embedder` で追加。
3. `DemoTextEmbedder` に `name`/`scale` プロパティを追加し、`embed()` の出力へスケール因子を反映。
4. `cargo fmt` で整形後、`cargo run --example vector_search` で動作を確認。

## 動作確認結果

- `cargo run --example vector_search` を実行し、ドキュメント挿入・検索結果が正常に出力されることを確認。field hit にはタイトル/本文の両方が表示され、PerFieldEmbedder による差し替えが反映されている。

## 課題と対策

- **課題:** サンプル Embedder では scale 値の違いが概念的であり、実モデルとの差を完全には再現できない。
  - **対策:** コメントでフィールドごとにモデルを切り替える用途を明記済み。必要に応じて本物の TextEmbedder 実装へ差し替えられる構成にしている。
