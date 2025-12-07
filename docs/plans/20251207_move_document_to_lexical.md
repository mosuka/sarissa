# documentモジュール移動計画

## 目的

`src/document` モジュールを `src/lexical/document` 以下に移動し、`src/vector` との構造的対称性を高め、アーキテクチャの整理・保守性向上を図る。

## 背景

- 現状、`document` モジュールは `src/` 直下に存在し、`lexical`/`vector` エンジン配下の責務分離が不明瞭。
- `src/vector` では関連モジュールが一箇所に集約されているため、`lexical` 側も同様の構造に揃えることで設計意図が明確になる。

## スコープ

- `src/document` 配下の全ファイル・ディレクトリ（analyzed.rs, converter.rs, document.rs, field.rs, parser.rs, converter/）
- 既存のimportパス・テスト・ドキュメントの修正
- CI・ビルド確認

## ステークホルダー

- プロジェクトメンテナ
- 開発チーム
- ドキュメント管理担当

## スケジュール

1. 計画案作成・レビュー（2025/12/07）
2. 承認後、実装着手（1日）
3. importパス・テスト修正（1日）
4. 動作確認・CI通過（即日）
5. ドキュメント更新（即日）

## リスク管理

- 依存関係の見落としによるビルドエラー → 事前grep/semantic searchで影響範囲を洗い出し
- 他エンジンとの共通化設計が必要な場合は追加検討

## 成果物

- `src/document` → `src/lexical/document` への移動完了
- 影響箇所のimportパス・テスト修正
- docs/配下の設計・実装レポート更新
