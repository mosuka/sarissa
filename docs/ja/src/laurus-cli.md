# CLI 概要

Laurus はコマンドラインツール `laurus` を提供しており、コードを書かずにインデックスの作成、ドキュメントの管理、検索クエリの実行が可能です。

## 機能

- **インデックス管理** -- TOML スキーマファイルからインデックスを作成・検査。対話式スキーマジェネレーター付き
- **ドキュメント CRUD** -- JSON によるドキュメントの追加、取得、削除
- **検索** -- [Query DSL](concepts/query_dsl.md) を使用したクエリ実行
- **デュアル出力** -- 人間が読みやすいテーブル形式または機械処理向け JSON 形式
- **対話型 REPL** -- ライブセッションでインデックスを操作
- **gRPC サーバー** -- `laurus serve` で [gRPC サーバー](laurus-server.md)を起動

## はじめに

```bash
# インストール
cargo install laurus-cli

# スキーマを対話的に生成
laurus create schema

# スキーマからインデックスを作成
laurus --index-dir ./my_index create index --schema schema.toml

# ドキュメントを追加
laurus --index-dir ./my_index add doc --id doc1 --data '{"fields":{"title":"Hello","body":"World"}}'

# 変更をコミット
laurus --index-dir ./my_index commit

# 検索
laurus --index-dir ./my_index search "body:world"
```

詳細はサブセクションを参照してください:

- [インストール](laurus-cli/installation.md) -- CLI のインストール方法
- [コマンドリファレンス](laurus-cli/commands.md) -- 全コマンドの詳細
- [スキーマフォーマット](laurus-cli/schema_format.md) -- スキーマ TOML フォーマットのリファレンス
- [REPL](laurus-cli/repl.md) -- 対話モード
