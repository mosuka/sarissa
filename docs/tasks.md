# 2025-11-24 RawFieldPayload → FieldPayload リネーム計画

## 目的

- フィールド単位の未埋め込みデータ表現を `FieldVectors` と対で理解しやすくするため、`RawFieldPayload` を `FieldPayload` に改称して命名規則を統一する。
- まだリリース前の API を整えることで、後日の破壊的変更リスクをなくし、ドキュメント・サンプルで一貫した名前を提示できるようにする。

## 関連ファイル・ディレクトリ

- `src/vector/core/document.rs`（型定義と serde 連携）。
- `src/vector/engine.rs`, `src/vector/field.rs`, `src/vector/index/**`（エンジン内部での payload 受け渡し）。
- `src/hybrid/engine.rs`, `src/hybrid/search/searcher.rs`（クエリ側で payload を扱う箇所）。
- `examples/*.rs`, `tests/vector_engine_scenarios.rs` などのサンプル／E2E テスト。
- ドキュメント: `README.md`, `docs/vector_engine.md`, `docs/reports/**`。

## 実装ステップ

1. **型とモジュールのリネーム** ✅ 2025-11-24 完了
   - `RawFieldPayload` 構造体・関連関数（`add_metadata` など）を `FieldPayload` へリネームし、`use` 句や `pub` エクスポートも更新する。
2. **依存箇所の一括置換** ✅ 2025-11-24 完了
   - `rg -l "RawFieldPayload"` で参照ファイルを抽出し、構造体名／モジュールパス／ドキュコメントを `FieldPayload` へ置換する。テスト・サンプル・ハイブリッド層も含む。
3. **serde / API 互換性確認** ✅ 2025-11-24 完了
   - Serde 名（`RawFieldPayload` を明示 rename していないか）を確認し、必要なら `#[serde(rename = "raw_field_payload")]` を保つかどうか判断。まだ互換性要件がないなら単純 rename で問題ないかレビューする。
4. **ドキュメント更新** ✅ 2025-11-24 完了
   - README や `docs/vector_engine.md` に記載された `RawFieldPayload` を `FieldPayload` へ書き換え、文脈上 “raw” を説明する文章を調整する。
5. **ビルド・テスト検証** ✅ 2025-11-24 完了
   - `cargo fmt` → `cargo test --tests vector_engine_scenarios` を実行し、コンパイル／テストが通ることを確認（必要に応じて examples も実行）。
6. **変更履歴の整理** ✅ 2025-11-24 完了
   - `CHANGELOG.md` に「命名統一のため FieldPayload へ改称した」旨を追記した。

## リスク・考慮点

- serde で永続化しているデータが既にある場合、フィールド名の変更でデシリアライズできなくなる恐れがある。現時点でストレージや WAL に `RawFieldPayload` 名が残らないかを調べる。
- 命名だけの差分だが、`rg` 置換でスペルミスが起きるとコンパイルエラーになりやすい。段階的に置換 → `cargo check` で検証する。
- ドキュメントや README を更新し忘れると、ユーザが新旧名称を混同する可能性がある。

## 推奨アクション

- VSCode の rename 機能や `sed` を使う場合でも、小分けにコミットできるよう段階を分ける。
- `FieldPayload` への改称と同時に `type RawFieldPayload = FieldPayload;` を短期的に残すか検討し、必要なければ削除しておく。
- リネーム完了後に `rg "RawField"` を走らせ、不要な表記が残っていないことを確認する。
