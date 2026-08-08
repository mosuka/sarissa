# laurus-cli

[![Crates.io](https://img.shields.io/crates/v/laurus-cli.svg)](https://crates.io/crates/laurus-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Laurus](https://github.com/mosuka/laurus) 検索エンジンのコマンドラインインターフェースです。

## 機能

- **インデックス管理** -- TOML スキーマファイルからインデックスを作成・検査。対話式スキーマジェネレーター付き
- **ドキュメント CRUD** -- JSON によるドキュメントの追加、上書き（upsert）、取得、削除
- **検索** -- Laurus Query DSL を使用したクエリ実行
- **デュアル出力** -- 人間が読みやすいテーブル形式または機械処理向け JSON 形式（`--format json`）
- **対話型 REPL** -- コマンド履歴付きのライブセッションでインデックスを操作
- **サーバー連携** -- CLI から直接 gRPC サーバーや MCP サーバーを起動

## インストール

ビルド済みバイナリ（Alpine/scratch 向けの完全静的リンク musl ビルドを含む）は
各 [GitHub リリース](https://github.com/mosuka/laurus/releases)に添付されています。
全ターゲットの一覧とダウンロード例は
[インストール](https://mosuka.github.io/laurus/ja/laurus-cli/installation.html)
を参照してください。

```bash
cargo install laurus-cli
```

## クイックスタート

```bash
# スキーマファイルからインデックスを作成
laurus --index-dir ./my_index create index --schema schema.toml

# ドキュメントを追加
laurus --index-dir ./my_index add doc \
  --id doc1 --data '{"title":"Hello","body":"World"}'

# ドキュメントを上書き（upsert）
laurus --index-dir ./my_index put doc \
  --id doc1 --data '{"title":"Updated","body":"Content"}'

# 変更をコミット
laurus --index-dir ./my_index commit

# 検索
laurus --index-dir ./my_index search "body:world"

# ID でドキュメントを取得
laurus --index-dir ./my_index get docs --id doc1

# ID でドキュメントを削除
laurus --index-dir ./my_index delete docs --id doc1

# 対話型 REPL を起動
laurus --index-dir ./my_index repl
```

## コマンド一覧

| コマンド | 説明 |
| :--- | :--- |
| `create index [--schema <FILE>]` | インデックスを作成（スキーマ未指定時は対話型ウィザード） |
| `create schema [--output <FILE>]` | 対話型スキーマ生成ウィザード |
| `get stats` | インデックスの統計情報を表示 |
| `get schema` | 現在のスキーマを JSON で表示 |
| `get docs --id <ID>` | ID で全ドキュメント（チャンクを含む）を取得 |
| `add doc --id <ID> --data <JSON>` | ドキュメントを新しいチャンクとして追加（追記） |
| `add field --name <NAME> --field-option <JSON>` | インデックスにフィールドを動的に追加 |
| `put doc --id <ID> --data <JSON>` | ドキュメントを上書き（既存を置換） |
| `delete docs --id <ID>` | ID で全ドキュメント（チャンクを含む）を削除 |
| `delete field --name <NAME>` | スキーマからフィールドを削除 |
| `commit` | 保留中の変更をディスクにコミット |
| `search <QUERY> [--limit N] [--offset N]` | 検索クエリを実行 |
| `repl` | 対話型 REPL セッションを開始 |
| `serve [OPTIONS]` | gRPC サーバーを起動 |
| `mcp [--endpoint <URL>]` | MCP サーバーを stdio で起動 |

## ドキュメント

- [CLI ガイド](https://mosuka.github.io/laurus/ja/laurus-cli.html)
- [コマンドリファレンス](https://mosuka.github.io/laurus/ja/laurus-cli/commands.html)
- [REPL](https://mosuka.github.io/laurus/ja/laurus-cli/repl.html)
- [スキーマフォーマット](https://mosuka.github.io/laurus/ja/laurus-cli/schema_format.html)
- [チュートリアル](https://mosuka.github.io/laurus/ja/laurus-cli/tutorial.html)

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](../LICENSE) ファイルを参照してください。
