# 設定

laurus-server は CLI 引数、環境変数、TOML 設定ファイルで設定できます。

## 設定の優先順位

サーバーとインデックスの設定は以下の順序で解決されます（優先度が高い順）。

```text
CLI 引数 > 環境変数 > 設定ファイル > デフォルト値
```

ログの詳細度は `RUST_LOG` 環境変数でのみ制御します（デフォルト: `info`）。

例:

```bash
# CLI 引数が環境変数と設定ファイルより優先される
LAURUS_PORT=4567 laurus serve --config config.toml --port 1234
# -> ポート 1234 でリッスン

# 環境変数が設定ファイルより優先される
LAURUS_PORT=4567 laurus serve --config config.toml
# -> ポート 4567 でリッスン

# CLI 引数も環境変数も未設定の場合、設定ファイルの値が使用される
laurus serve --config config.toml
# -> config.toml のポートを使用（未設定の場合はデフォルト 50051）
```

## TOML 設定ファイル

### フォーマット

```toml
[server]
host = "0.0.0.0"
port = 50051
http_port = 8080  # オプション: HTTP ゲートウェイを有効化

[index]
data_dir = "./laurus_data"
```

ログの詳細度は設定ファイルではなく、`RUST_LOG` 環境変数で制御します（デフォルト: `info`）。

### フィールドリファレンス

#### `[server]` セクション

| フィールド | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `host` | String | `"0.0.0.0"` | gRPC サーバーのリッスンアドレス |
| `port` | Integer | `50051` | gRPC サーバーのリッスンポート |
| `http_port` | Integer | -- | HTTP ゲートウェイポート。設定すると gRPC と並行して HTTP/JSON ゲートウェイが起動 |

#### `[index]` セクション

| フィールド | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `data_dir` | String | `"./laurus_data"` | インデックスデータディレクトリのパス |

## 環境変数

| 変数 | 対応する設定 | 説明 |
| :--- | :--- | :--- |
| `LAURUS_HOST` | `server.host` | リッスンアドレス |
| `LAURUS_PORT` | `server.port` | gRPC リッスンポート |
| `LAURUS_HTTP_PORT` | `server.http_port` | HTTP ゲートウェイポート |
| `LAURUS_INDEX_DIR` | `index.data_dir` | インデックスデータディレクトリ |
| `RUST_LOG` | -- | ログフィルタディレクティブ（例: `info`, `debug`, `laurus=debug,tonic=warn`） |
| `LAURUS_CONFIG` | -- | TOML 設定ファイルのパス |

## CLI 引数

| オプション | 短縮形 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `--config <PATH>` | `-c` | -- | TOML 設定ファイルのパス |
| `--host <HOST>` | `-H` | `0.0.0.0` | リッスンアドレス |
| `--port <PORT>` | `-p` | `50051` | gRPC リッスンポート |
| `--http-port <PORT>` | -- | -- | HTTP ゲートウェイポート |
| `--index-dir <PATH>` | -- | `./laurus_index` | インデックスデータディレクトリ（グローバルオプション） |

## よくある設定例

### 開発環境（gRPC のみ）

```toml
[server]
host = "127.0.0.1"
port = 50051

[index]
data_dir = "./dev_data"
```

```bash
RUST_LOG=debug laurus serve --config config.toml
```

### 本番環境（gRPC + HTTP ゲートウェイ）

```toml
[server]
host = "0.0.0.0"
port = 50051
http_port = 8080

[index]
data_dir = "/var/lib/laurus/data"
```

### 最小構成（環境変数のみ）

```bash
export LAURUS_INDEX_DIR=/var/lib/laurus/data
export LAURUS_PORT=50051
export LAURUS_HTTP_PORT=8080
export RUST_LOG=info
laurus serve
```
