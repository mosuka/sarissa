# HTTP ゲートウェイ

HTTP ゲートウェイは Laurus 検索エンジンへの RESTful HTTP/JSON インターフェースを提供します。gRPC サーバーと並行して動作し、リクエストを内部的にプロキシします。

```text
Client (HTTP/JSON) --> HTTP Gateway (axum) --> gRPC Server (tonic) --> Engine
```

## HTTP ゲートウェイの有効化

`http_port` を設定するとゲートウェイが起動します。

```bash
# CLI 引数で指定
laurus serve --http-port 8080

# 環境変数で指定
LAURUS_HTTP_PORT=8080 laurus serve

# 設定ファイルで指定
laurus serve --config config.toml
# （[server] セクションで http_port を設定）
```

`http_port` が未設定の場合、gRPC サーバーのみが起動します。

## エンドポイント

| メソッド | パス | gRPC メソッド | 説明 |
| :--- | :--- | :--- | :--- |
| GET | `/v1/health` | `HealthService/Check` | ヘルスチェック |
| POST | `/v1/index` | `IndexService/CreateIndex` | 新しいインデックスを作成 |
| GET | `/v1/index` | `IndexService/GetIndex` | インデックスの統計情報を取得 |
| GET | `/v1/schema` | `IndexService/GetSchema` | インデックスのスキーマを取得 |
| PUT | `/v1/documents/:id` | `DocumentService/PutDocument` | ドキュメントの Upsert |
| POST | `/v1/documents/:id` | `DocumentService/AddDocument` | ドキュメントの追加（チャンク） |
| GET | `/v1/documents/:id` | `DocumentService/GetDocuments` | ID でドキュメントを取得 |
| DELETE | `/v1/documents/:id` | `DocumentService/DeleteDocuments` | ID でドキュメントを削除 |
| POST | `/v1/commit` | `DocumentService/Commit` | 保留中の変更をコミット |
| POST | `/v1/schema/fields` | `IndexService/AddField` | フィールドの追加 |
| DELETE | `/v1/schema/fields/:name` | `IndexService/DeleteField` | フィールドの削除 |
| POST | `/v1/search` | `SearchService/Search` | 検索（単発） |
| POST | `/v1/search/stream` | `SearchService/SearchStream` | 検索（Server-Sent Events） |

## API の使用例

### ヘルスチェック

```bash
curl http://localhost:8080/v1/health
```

### インデックスの作成

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{
    "schema": {
      "dynamic_field_policy": "dynamic",
      "fields": {
        "title": {"text": {"indexed": true, "stored": true, "term_vectors": true}},
        "body": {"text": {"indexed": true, "stored": true, "term_vectors": true}}
      },
      "default_fields": ["title", "body"]
    }
  }'
```

`dynamic_field_policy` は省略可能なキーで、スキーマに宣言されていないフィールドの扱いを制御します。指定できる値は `"strict"` / `"dynamic"`（デフォルト）/ `"ignore"` の 3 種類です。詳細および `"dynamic"` での情報損失に関する警告は [スキーマとフィールド](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。

### インデックス統計情報の取得

```bash
curl http://localhost:8080/v1/index
```

### スキーマの取得

```bash
curl http://localhost:8080/v1/schema
```

### ドキュメントの Upsert（PUT）

ドキュメントが既に存在する場合は置換します。

```bash
curl -X PUT http://localhost:8080/v1/documents/doc1 \
  -H 'Content-Type: application/json' \
  -d '{
    "document": {
      "fields": {
        "title": "Hello World",
        "body": "This is a test document."
      }
    }
  }'
```

### ドキュメントの追加（POST）

同じ ID の既存ドキュメントを置換せずに新しいチャンクを追加します。

```bash
curl -X POST http://localhost:8080/v1/documents/doc1 \
  -H 'Content-Type: application/json' \
  -d '{
    "document": {
      "fields": {
        "title": "Hello World",
        "body": "This is a test document."
      }
    }
  }'
```

### ドキュメントの取得

```bash
curl http://localhost:8080/v1/documents/doc1
```

### ドキュメントの削除

```bash
curl -X DELETE http://localhost:8080/v1/documents/doc1
```

### コミット

```bash
curl -X POST http://localhost:8080/v1/commit
```

### 検索

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "body:test", "limit": 10}'
```

#### フィールドブースト付き検索

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "rust programming",
    "limit": 10,
    "field_boosts": {"title": 2.0}
  }'
```

#### ハイブリッド検索

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "body:rust",
    "query_vectors": [{"vector": [0.1, 0.2, 0.3], "weight": 1.0}],
    "limit": 10,
    "fusion": {"rrf": {"k": 60}}
  }'
```

### ストリーミング検索（SSE）

`/v1/search/stream` エンドポイントは Server-Sent Events（SSE）として結果を返します。各結果は個別のイベントとして送信されます。

```bash
curl -N -X POST http://localhost:8080/v1/search/stream \
  -H 'Content-Type: application/json' \
  -d '{"query": "body:test", "limit": 10}'
```

レスポンスは SSE イベントのストリームです。

```text
data: {"id":"doc1","score":0.8532,"document":{...}}

data: {"id":"doc2","score":0.4210,"document":{...}}
```

## リクエスト/レスポンス形式

すべてのリクエストおよびレスポンスボディは JSON を使用します。JSON の構造は gRPC の protobuf メッセージに対応しています。メッセージ定義の詳細は [gRPC API リファレンス](grpc_api.md)を参照してください。
