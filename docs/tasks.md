# Tasks

## MLモジュールの削除とアーキテクチャ見直し

プロジェクトをシンプルにするため、`ml` モジュールとその関連機能を削除する。

### 計画

- [x] ファイル削除
  - [x] `src/ml.rs`
  - [x] `src/ml/` ディレクトリ
  - [x] `resources/ml/` ディレクトリ
  - [x] `examples/ml_based_intent_classifier.rs`
  - [x] `examples/keyword_based_intent_classifier.rs`
- [x] コード修正
  - [x] `src/lib.rs`: `pub mod ml;` の削除
  - [x] `src/error.rs`: `crate::ml::MLError` への依存を削除
  - [x] `Cargo.toml`: `smartcore` 依存関係の削除
- [x] 確認
  - [x] `cargo build` が通ること
  - [x] `cargo test` が通ること
