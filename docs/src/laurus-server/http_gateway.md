# HTTP Gateway

The HTTP Gateway provides a RESTful HTTP/JSON interface to the Laurus search engine. It runs alongside the gRPC server and proxies requests internally:

```text
Client (HTTP/JSON) --> HTTP Gateway (axum) --> gRPC Server (tonic) --> Engine
```

## Enabling the HTTP Gateway

The gateway starts when `http_port` is configured:

```bash
# Via CLI argument
laurus serve --http-port 8080

# Via environment variable
LAURUS_HTTP_PORT=8080 laurus serve

# Via config file
laurus serve --config config.toml
# (set http_port in [server] section)
```

If `http_port` is not set, only the gRPC server starts.

## Endpoints

| Method | Path | gRPC Method | Description |
| :--- | :--- | :--- | :--- |
| GET | `/v1/health` | `HealthService/Check` | Health check |
| POST | `/v1/index` | `IndexService/CreateIndex` | Create a new index |
| GET | `/v1/index` | `IndexService/GetIndex` | Get index statistics |
| GET | `/v1/schema` | `IndexService/GetSchema` | Get the index schema |
| PUT | `/v1/documents/:id` | `DocumentService/PutDocument` | Upsert a document |
| POST | `/v1/documents/:id` | `DocumentService/AddDocument` | Add a document (chunk) |
| GET | `/v1/documents/:id` | `DocumentService/GetDocuments` | Get documents by ID |
| DELETE | `/v1/documents/:id` | `DocumentService/DeleteDocuments` | Delete documents by ID |
| POST | `/v1/commit` | `DocumentService/Commit` | Commit pending changes |
| POST | `/v1/search` | `SearchService/Search` | Search (unary) |
| POST | `/v1/search/stream` | `SearchService/SearchStream` | Search (Server-Sent Events) |

## API Examples

### Health Check

```bash
curl http://localhost:8080/v1/health
```

### Create an Index

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

The `dynamic_field_policy` key is optional. It controls how fields absent
from the schema are handled at ingest time. Accepted values: `"strict"`,
`"dynamic"` (default), `"ignore"`. See
[Schema & Fields](../concepts/schema_and_fields.md#dynamic-schema) for the
full semantics and the warning about silent truncation under `"dynamic"`.

### Get Index Statistics

```bash
curl http://localhost:8080/v1/index
```

### Get Schema

```bash
curl http://localhost:8080/v1/schema
```

### Upsert a Document (PUT)

Replaces the document if it already exists:

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

### Add a Document (POST)

Adds a new chunk without replacing existing documents with the same ID:

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

### Get Documents

```bash
curl http://localhost:8080/v1/documents/doc1
```

### Delete Documents

```bash
curl -X DELETE http://localhost:8080/v1/documents/doc1
```

### Commit

```bash
curl -X POST http://localhost:8080/v1/commit
```

### Search

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "body:test", "limit": 10}'
```

#### Search with Field Boosts

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "rust programming",
    "limit": 10,
    "field_boosts": {"title": 2.0}
  }'
```

#### Hybrid Search

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

### Streaming Search (SSE)

The `/v1/search/stream` endpoint returns results as Server-Sent Events (SSE). Each result is sent as a separate event:

```bash
curl -N -X POST http://localhost:8080/v1/search/stream \
  -H 'Content-Type: application/json' \
  -d '{"query": "body:test", "limit": 10}'
```

The response is a stream of SSE events:

```text
data: {"id":"doc1","score":0.8532,"document":{...}}

data: {"id":"doc2","score":0.4210,"document":{...}}
```

## Request/Response Format

All request and response bodies use JSON. The JSON structure mirrors the gRPC protobuf messages. See [gRPC API Reference](grpc_api.md) for the full message definitions.
