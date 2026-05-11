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
| POST | `/v1/schema/fields` | `IndexService/AddField` | フィールドを動的に追加 |
| DELETE | `/v1/schema/fields/{name}` | `IndexService/DeleteField` | スキーマからフィールドを削除 |
| PUT | `/v1/documents/{id}` | `DocumentService/PutDocument` | ドキュメントの Upsert |
| POST | `/v1/documents/{id}` | `DocumentService/AddDocument` | ドキュメントの追加（チャンク） |
| GET | `/v1/documents/{id}` | `DocumentService/GetDocuments` | ID でドキュメントを取得 |
| DELETE | `/v1/documents/{id}` | `DocumentService/DeleteDocuments` | ID でドキュメントを削除 |
| POST | `/v1/commit` | `DocumentService/Commit` | 保留中の変更をコミット |
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

### フィールドの動的追加

稼働中のインデックスにフィールドを追加します。リクエストボディは `POST /v1/index` と同じ `FieldOption` JSON 形式を使います:

```bash
curl -X POST http://localhost:8080/v1/schema/fields \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "category",
    "field_option": {"text": {"indexed": true, "stored": true}}
  }'
```

レスポンスでは更新後のスキーマが返されます。

### フィールドの削除

スキーマからフィールドを削除します。フィールド名はパスで指定します:

```bash
curl -X DELETE http://localhost:8080/v1/schema/fields/category
```

既にインデックスされたデータはストレージに残りますが、アクセスできなくなります。フィールド固有のアナライザとエンベッダーは解除されます。

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

## JSON フィールド値の型推論

ドキュメント投入リクエスト（`PUT /v1/documents/{id}` または
`POST /v1/documents/{id}`）のボディに含まれる `document.fields` の各値は、
スキーマレス取り込みと同じ推論ルールでエンジンの
[`DataValue`](../concepts/schema_and_fields.md) に変換されます。これにより
HTTP 経路と gRPC 経路で挙動が一致します。

| JSON 値 | 推論されるフィールド型 | 備考 |
| :--- | :--- | :--- |
| `null` | (スキップ) | `NullValue` として送出され、取り込み時に破棄されます。 |
| `true` / `false` | `boolean` | |
| 整数（`i64` に収まる） | `integer` | |
| 浮動小数点 / 巨大整数 | `float` | |
| `"text"` | `text` | |
| `[1, 2, 3]`（全要素 integer） | `integer`（`multi_valued: true`） | 多値数値フィールド。 |
| `[1.0, 2.5]`（非整数を含む数値配列） | `float`（`multi_valued: true`） | |
| `[]`（空配列） | (スキップ) | 要素型を決定できないためフィールドはスキップされます。 |
| `{"latitude": ..., "longitude": ...}` | `geo` | |
| `{"lat": ..., "lon": ...}` / `{"lat": ..., "lng": ...}` | `geo` | latitude / longitude の短縮別名を受け付けます。 |
| `{"x": ..., "y": ..., "z": ...}` | `geo3d` | 3 キーすべて必須、有限な数値、ECEF メートル単位。`lat`/`lon` キーとの混在は拒否されます。 |

以下の場合、ゲートウェイは HTTP 400（`Bad Request`）を返します:

- 配列が混在型もしくは非数値要素を含む（例: `[1, "x"]`）
- オブジェクトが有効な地理座標ではない（2D は latitude/longitude キーが、
  3D は `x`/`y`/`z` のいずれかが欠けている）
- 緯度が `[-90, 90]` の範囲外、または経度が `[-180, 180]` の範囲外
- 3D ECEF の座標が有限値でない（`NaN` / `Inf`）
- 同一オブジェクトに 2D（`lat`/`lon`）と 3D（`x`/`y`/`z`）のキーが混在

ベクトルおよびバイト列フィールドは JSON だけからは推論できないため、
スキーマで明示的に宣言する必要があります。宣言済みのベクトルフィールドに
数値配列が送られた場合は自動的に `f32` ベクトルへキャストされるので、
REST クライアントは埋め込みベクトルを通常の JSON 配列として送信できます。

### 3D 地理クエリ

3D ECEF クエリは `query` に渡す Lexical DSL 文字列をそのまま再利用します。ゲートウェイは文字列を変更せずエンジンへ転送するため、gRPC 経由と同じ DSL 形式が HTTP 経由でも動作します:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "position:geo3d_distance(-3955182, 3350553, 3700276, 5000)",
    "limit": 10
  }'
```

`geo3d_bbox` および `geo3d_nearest` の構文は [Query DSL → 3D 地理クエリ](../concepts/query_dsl.md#3d-geographic-queries-geo3d_) を参照してください。

## リクエスト/レスポンス形式

すべてのリクエストおよびレスポンスボディは JSON を使用します。JSON の構造は gRPC の protobuf メッセージに対応しています。メッセージ定義の詳細は [gRPC API リファレンス](grpc_api.md)を参照してください。
