# 実装レポート: lexical_searchのDocument::builder移行

## 実装内容

- `examples/lexical_search.rs` のインデクシング処理を `JsonlDocumentConverter` から `Document::builder()` ベースにリプレース。
- JSONL と同等の10件サンプルを返す `sample_documents()` を実装し、`TextOption`/`IntegerOption`/`GeoOption` を用いて各フィールドを構築。
- `TextOption::stored(false)` の設定により body フィールドのフットプリントを最小化。

## 手順

1. `examples/lexical_search.rs` から JSONL コンバータ関連の import と処理を削除。
2. `sample_documents()` を追加し、`Document::builder()` で ID・title・author・category・body・tags・location・year・rating を設定。
3. メイン処理で `sample_documents()` の結果をインデクシングするループに差し替え。
4. `cargo fmt` でコード整形。

## 動作確認結果

- `cargo run --example lexical_search` を実行し、全 19 ステップの検索・ソートデモが従来どおり成功することを確認。

## 課題と対策

- **課題:** データソースがコード内にハードコードされるため、将来的な更新漏れリスクがある。
  - **対策:** `sample_documents()` 冒頭にコメントを追加済み。今後 JSONL を更新する場合は本関数も同期させる運用ドキュメントを検討。
