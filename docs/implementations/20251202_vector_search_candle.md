# 実装レポート: vector_searchのCandle対応

## 実装内容

- `examples/vector_search.rs` をDemoTextEmbedder依存からCandleTextEmbedderへの実装に置き換え。
- `PerFieldEmbedder` にCandleベースの埋め込み器を登録し、タイトルと本文で同一モデルを共有しつつ重み付けを継承。
- `embeddings-candle` フィーチャの有効化が必須であることをコードコメントおよび実行時メッセージで明示。
- フィーチャ未有効時には実行できない旨を早期に通知するガードを追加。

## 手順

1. 既存のDemoTextEmbedder定義と関連インポートを削除。
2. `cfg` 属性を用いて `embeddings-candle` フィーチャが無効な場合のフォールバック `main` を追加。
3. CandleTextEmbedderを読み込み、`Arc<dyn TextEmbedder>` として `PerFieldEmbedder` に登録する処理を実装。
4. ベクターフィールド設定において実際の次元値をCandleインスタンスから取得するよう更新。
5. `HashMap::insert` ベースで埋め込み器レジストリを再構築し、`VectorEngine` 初期化フローを維持。
6. `cargo fmt` によりコードスタイルを整形。

## 動作確認結果

- コマンド: `cargo run --example vector_search --features embeddings-candle`
- 結果: 3件のドキュメントが挿入され、検索結果としてdoc3およびdoc1が得られることを確認。`PerFieldEmbedder` を介してCandleによる自動埋め込みが正しく実行された。

## 課題と対策

- HuggingFaceモデルダウンロードおよびCandle初回実行に時間がかかるため、実行時メッセージでダウンロード発生の可能性を通知。
- `embeddings-candle` フィーチャが無効なケースでのビルド失敗を避けるため、`cfg` ガードで依存コードを分離し、ユーザーに再実行手順を案内。
