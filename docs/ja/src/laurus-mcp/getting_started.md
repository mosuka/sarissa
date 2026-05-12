# laurus-mcp をはじめる

## 前提条件

- `laurus` CLI バイナリがインストール済み（`cargo install laurus-cli`）
- 実行中の `laurus-server` インスタンス（[laurus-server はじめに](../laurus-server/getting_started.md)を参照）
- MCP をサポートする AI クライアント（Claude Desktop、Claude Code など）

## 設定

### ステップ 1: laurus-server を起動

```bash
laurus serve --port 50051
```

### ステップ 2: MCP クライアントの設定

#### Claude Code

CLI コマンドで追加する方法（推奨）：

```bash
claude mcp add laurus -- laurus mcp --endpoint http://localhost:50051
```

または `~/.claude/settings.json` を直接編集：

```json
{
  "mcpServers": {
    "laurus": {
      "command": "laurus",
      "args": ["mcp", "--endpoint", "http://localhost:50051"]
    }
  }
}
```

#### Claude Desktop

以下の設定ファイルを編集：

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "laurus": {
      "command": "laurus",
      "args": ["mcp", "--endpoint", "http://localhost:50051"]
    }
  }
}
```

## 使用ワークフロー

### ワークフロー 1: 既存のインデックスを使用する

CLI でインデックスを事前に作成してから MCP サーバーで検索します：

```bash
# ステップ 1: スキーマファイルを作成
cat > schema.toml << 'EOF'
[fields.title]
Text = { indexed = true, stored = true }

[fields.body]
Text = { indexed = true, stored = true }
EOF

# ステップ 2: サーバーを起動してインデックスを作成
laurus serve --port 50051 &
laurus create index --schema schema.toml

# ステップ 3: MCP サーバーを Claude Code に登録
claude mcp add laurus -- laurus mcp --endpoint http://localhost:50051
```

### ワークフロー 2: AI 主導のインデックス作成

laurus-server を起動してから MCP サーバーを登録し、AI にインデックスを作成させます：

```bash
# ステップ 1: laurus-server を起動（インデックス不要）
laurus serve --port 50051

# ステップ 2: MCP サーバーを Claude Code に登録
claude mcp add laurus -- laurus mcp --endpoint http://localhost:50051
```

次に Claude に依頼します：

> 「ブログ記事用の検索インデックスを作成してください。タイトルと本文テキストで検索できるようにして、著者と公開日も保存したいです。」

Claude はスキーマを設計して `create_index` を自動的に呼び出します。

### ワークフロー 3: 実行時に接続する

エンドポイントを指定せずに MCP サーバーを登録します：

```bash
claude mcp add laurus -- laurus mcp
```

または設定ファイルを直接編集：

```json
{
  "mcpServers": {
    "laurus": {
      "command": "laurus",
      "args": ["mcp"]
    }
  }
}
```

次に Claude に接続を依頼します：

> 「`http://localhost:50051` の laurus サーバーに接続してください」

Claude は他のツールを使用する前に `connect` を呼び出します。

## 環境変数

`--endpoint` フラグを省略し、代わりに `LAURUS_ENDPOINT` を渡すこともできます。エンドポイントがマシンごとに固定で、各 MCP クライアントの設定にハードコードしたくない場合に便利です：

```bash
export LAURUS_ENDPOINT=http://localhost:50051
claude mcp add laurus -- laurus mcp
```

クライアント設定内で指定する場合：

```json
{
  "mcpServers": {
    "laurus": {
      "command": "laurus",
      "args": ["mcp"],
      "env": {
        "LAURUS_ENDPOINT": "http://localhost:50051"
      }
    }
  }
}
```

両方が指定された場合は、明示的な `--endpoint` フラグが優先されます（clap の `#[arg(long, env = "...")]` の標準動作）。

## MCP サーバーの削除

Claude Code から登録済みの MCP サーバーを削除するには：

```bash
claude mcp remove laurus
```

Claude Desktop の場合は、設定ファイルから `laurus` エントリを削除してアプリケーションを再起動してください。

## ライフサイクル

```text
laurus-server 起動（別プロセス）
  └─ gRPC ポート 50051 でリッスン

Claude 起動
  └─ 起動: laurus mcp --endpoint http://localhost:50051
       └─ stdio イベントループに入る
            ├─ stdin 経由でツール呼び出しを受信
            ├─ gRPC 経由で laurus-server にプロキシ
            └─ stdout 経由で結果を送信
Claude 終了
  └─ laurus-mcp プロセスが終了
  └─ laurus-server は継続して動作
```
