# documentモジュール移動 実装計画

## 目的

`src/document` モジュールを `src/lexical/document` 以下に移動し、アーキテクチャの対称性と保守性を向上させる。

## 背景

- 計画案（20251207_move_document_to_lexical.md）に基づく。

## スコープ

- `src/document/` 配下の全ファイル・ディレクトリ（analyzed.rs, converter.rs, document.rs, field.rs, parser.rs, converter/）
- 既存importパス・テスト・ドキュメントの修正
- CI・ビルド確認

## 実施手順

1. 影響範囲調査（semantic search/grepで `document` 参照箇所を洗い出し）
2. `src/document` → `src/lexical/document` へ物理移動
3. 参照箇所のimportパス修正（src, tests, benches, examples, docs等）
4. テスト・CI実行で動作確認
5. ドキュメント・設計図の更新
6. 実装レポート作成

## スケジュール

- 1日目: 調査・移動・import修正
- 2日目: テスト・CI・ドキュメント更新・レポート作成

## リスク管理

- 依存関係の見落とし → 事前grep/semantic searchで網羅的に調査
- 他エンジンとの共通化設計が必要な場合は追加検討
- CI失敗時は即座にロールバック可能なようにコミット管理

## 成果物

- `src/document` → `src/lexical/document` への移動完了
- 影響箇所のimportパス・テスト修正
- docs/配下の設計・実装レポート更新
