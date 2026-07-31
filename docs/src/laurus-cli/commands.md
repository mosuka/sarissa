# Command Reference

## Global Options

Every command accepts these options:

| Option | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--index-dir <PATH>` | `LAURUS_INDEX_DIR` | `./laurus_index` | Path to the index data directory |
| `--format <FORMAT>` | — | `table` | Output format: `table` or `json` |

```bash
# Example: use JSON output with a custom data directory
laurus --index-dir /var/data/my_index --format json search "title:rust"
```

---

## `create` — Create a Resource

### `create index`

Create a new index. If `--schema` is given, uses that TOML file; otherwise launches the interactive schema wizard.

```bash
laurus create index [--schema <FILE>]
```

**Arguments:**

| Flag | Required | Description |
| :--- | :--- | :--- |
| `--schema <FILE>` | No | Path to a TOML file defining the index schema. When omitted, the command checks if a `schema.toml` already exists in the index directory and uses it; otherwise the interactive wizard is launched. |

**Schema file format:**

The schema file follows the same structure as the `Schema` type in the Laurus library. See [Schema Format Reference](schema_format.md) for full details. Example:

```toml
default_fields = ["title", "body"]

[fields.title.Text]
stored = true
indexed = true

[fields.body.Text]
stored = true
indexed = true

[fields.category.Text]
stored = true
indexed = true
```

**Examples:**

```bash
# From a schema file
laurus --index-dir ./my_index create index --schema schema.toml
# Index created at ./my_index.

# Interactive wizard (no --schema flag)
laurus --index-dir ./my_index create index
# === Laurus Schema Generator ===
# Field name: title
# ...
# Index created at ./my_index.
```

> **Note:** If both `schema.toml` and `store/` already exist, an error is returned. Delete the index directory to recreate. If only `schema.toml` exists (e.g. after an interrupted creation), running `create index` without `--schema` recovers the index by creating the missing storage from the existing schema.

### `create schema`

Interactively generate a schema TOML file through a guided wizard.

```bash
laurus create schema [--output <FILE>]
```

**Arguments:**

| Flag | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `--output <FILE>` | No | `schema.toml` | Output file path for the generated schema |

The wizard guides you through:

1. **Field definition** — Enter a field name, select the type, and configure type-specific options
2. **Repeat** — Add as many fields as needed
3. **Default fields** — Select which lexical fields to use as default search fields
4. **Preview** — Review the generated TOML before saving
5. **Save** — Write the schema file

**Supported field types:**

| Type | Category | Options |
| :--- | :--- | :--- |
| `Text` | Lexical | `indexed`, `stored`, `term_vectors` |
| `Integer` | Lexical | `indexed`, `stored` |
| `Float` | Lexical | `indexed`, `stored` |
| `Boolean` | Lexical | `indexed`, `stored` |
| `DateTime` | Lexical | `indexed`, `stored` |
| `Geo` | Lexical | `indexed`, `stored` |
| `Geo3d` | Lexical | `indexed`, `stored` |
| `Bytes` | Lexical | `stored` |
| `Hnsw` | Vector | `dimension`, `distance`, `m`, `ef_construction` |
| `Flat` | Vector | `dimension`, `distance` |
| `Ivf` | Vector | `dimension`, `distance`, `n_clusters`, `n_probe` |

**Example:**

```bash
# Generate schema.toml interactively
laurus create schema

# Specify output path
laurus create schema --output my_schema.toml

# Then create an index from the generated schema
laurus create index --schema schema.toml
```

---

## `get` — Get a Resource

### `get stats`

Display statistics about the index.

```bash
laurus get stats
```

**Table output example:**

```text
Document count: 42

Vector fields:
╭──────────┬─────────┬───────────╮
│ Field    │ Vectors │ Dimension │
├──────────┼─────────┼───────────┤
│ text_vec │ 42      │ 384       │
╰──────────┴─────────┴───────────╯
```

**JSON output example:**

```bash
laurus --format json get stats
```

```json
{
  "document_count": 42,
  "fields": {
    "text_vec": {
      "vector_count": 42,
      "dimension": 384
    }
  }
}
```

### `get schema`

Display the current index schema as JSON.

```bash
laurus get schema
```

**Example:**

```bash
laurus get schema
# {
#   "fields": { ... },
#   "default_fields": ["title", "body"],
#   ...
# }
```

### `get docs`

Retrieve all documents (including chunks) by external ID.

```bash
laurus get docs --id <ID>
```

**Table output example:**

```text
╭──────┬─────────────────────────────────────────╮
│ ID   │ Fields                                  │
├──────┼─────────────────────────────────────────┤
│ doc1 │ body: This is a test, title: Hello World │
╰──────┴─────────────────────────────────────────╯
```

**JSON output example:**

```bash
laurus --format json get docs --id doc1
```

```json
[
  {
    "id": "doc1",
    "document": {
      "title": "Hello World",
      "body": "This is a test document."
    }
  }
]
```

---

## `add` — Add a Resource

### `add doc`

Add a document to the index. Documents are not searchable until `commit` is called.

```bash
laurus add doc --id <ID> --data <JSON>
```

**Arguments:**

| Flag | Required | Description |
| :--- | :--- | :--- |
| `--id <ID>` | Yes | External document ID (string) |
| `--data <JSON>` | Yes | Document fields as a JSON string |

The JSON is the document's serde shape: a `fields` object mapping each field name to an externally-tagged value (`Text`, `Int64`, `Float64`, `Bool`, `VectorValue`, ...):

```json
{
  "fields": {
    "title": {"Text": "Introduction to Rust"},
    "body": {"Text": "Rust is a systems programming language."},
    "year": {"Int64": 2024}
  }
}
```

**Example:**

```bash
laurus add doc --id doc1 --data '{"fields":{"title":{"Text":"Hello World"},"body":{"Text":"This is a test document."}}}'
# Document 'doc1' added. Run 'commit' to persist changes.
```

> **Tip:** Multiple documents can share the same external ID (chunking pattern). Use `add doc` for each chunk.

### `add docs`

Bulk-add document chunks from a JSONL file — one `{"id": "...", "document": {"fields": {...}}}` entry per line, where `document` uses the same JSON shape as `add doc --data`. Entries are applied through the engine's batch API (one WAL fsync per batch) and, unlike `add doc`, the command **commits automatically**: every `--commit-every` applied documents and once at the end.

```bash
laurus add docs --file <JSONL> [--batch-size 1000] [--commit-every 0]
```

**Arguments:**

| Flag | Required | Description |
| :--- | :--- | :--- |
| `--file <JSONL>` | Yes | Path to the JSONL file to ingest |
| `--batch-size <N>` | No | Documents per engine batch call (default `1000`) |
| `--commit-every <N>` | No | Commit every N applied documents; `0` = only the final commit (default) |

Repeated IDs accumulate as chunks. On a mid-file failure the error names the offending line, the applied prefix is committed, and re-running the remaining lines continues the ingest.

---

## `put` — Put (Upsert) a Resource

### `put doc`

Put (upsert) a document into the index. If a document with the same ID already exists, all its chunks are deleted before the new document is indexed. Documents are not searchable until `commit` is called.

```bash
laurus put doc --id <ID> --data <JSON>
```

**Arguments:**

| Flag | Required | Description |
| :--- | :--- | :--- |
| `--id <ID>` | Yes | External document ID (string) |
| `--data <JSON>` | Yes | Document fields as a JSON string |

**Example:**

```bash
laurus put doc --id doc1 --data '{"fields":{"title":{"Text":"Updated Title"},"body":{"Text":"This replaces the existing document."}}}'
# Document 'doc1' put (upserted). Run 'commit' to persist changes.
```

> **Note:** Unlike `add doc`, `put doc` replaces all existing chunks for the given ID. Use `add doc` when you want to append chunks, and `put doc` when you want to replace the entire document.

### `put docs`

Bulk-upsert documents from a JSONL file — one `{"id": "...", "document": {"fields": {...}}}` entry per line, applied through the engine's batch API (one WAL fsync per batch). Duplicate IDs dedup in order (the last occurrence wins). Like `add docs`, the command **commits automatically**.

```bash
laurus put docs --file <JSONL> [--batch-size 1000] [--commit-every 0]
```

Arguments are the same as `add docs`. On a mid-file failure the error names the offending line and the applied prefix is committed; because puts are idempotent, re-running the whole file (or its remaining suffix) is safe.

**Example:**

```bash
cat > docs.jsonl <<'JSONL'
{"id": "doc1", "document": {"fields": {"title": {"Text": "Hello"}}}}
{"id": "doc2", "document": {"fields": {"title": {"Text": "World"}}}}
JSONL
laurus put docs --file docs.jsonl
# 2 documents put (upserted) and committed.
```

---

### `add field`

Dynamically add a new field to an existing index.

```bash
laurus add field --index-dir ./data \
    --name category \
    --field-option '{"Text": {"indexed": true, "stored": true}}'
```

The `--field-option` argument accepts a JSON string using the same
externally-tagged format as the schema file. The schema is automatically
persisted after the field is added.

---

## `delete` — Delete a Resource

### `delete docs`

Delete all documents (including chunks) by external ID.

```bash
laurus delete docs --id <ID>
```

**Example:**

```bash
laurus delete docs --id doc1
# Documents 'doc1' deleted. Run 'commit' to persist changes.
```

### `delete field`

Remove a field from the index schema.

```bash
laurus delete field --name <FIELD_NAME>
```

**Example:**

```bash
laurus delete field --name category
# Field 'category' deleted.
```

Existing indexed data for the field remains in storage but becomes
inaccessible. Per-field analyzers and embedders are unregistered.

---

## `commit`

Commit pending changes (additions and deletions) to the index. Until committed, changes are not visible to search.

```bash
laurus commit
```

**Example:**

```bash
laurus --index-dir ./my_index commit
# Changes committed successfully.
```

---

## `train`

### `train pq-codebook`

Train a **shared PQ codebook** for an HNSW vector field (Issue #631).
The codebook is trained once on a representative sample and then reused
by every subsequent commit and merge, instead of re-training k-means
per segment — commits on PQ fields get dramatically faster, and even
tiny per-commit segments stay on PQ.

```bash
laurus train pq-codebook --field <FIELD> --input <JSONL> \
    [--sample-size <N>] [--output <NAME>] [--update-schema]
```

| Argument | Description |
| :--- | :--- |
| `--field` | The HNSW vector field to train for. Must be configured with a `ProductQuantization` quantizer. |
| `--input` | JSONL training file — the same `{"id": "...", "document": {"fields": {...}}}` shape as `put docs` / `add docs`. The field value must be a pre-computed `Vector` (embedder-generated input is not supported). |
| `--sample-size` | Use only the first N vectors of the file (deterministic). Omit to use all of them; thousands of representative vectors are enough. |
| `--output` | Storage-relative codebook file name. Defaults to the field's configured `pq_codebook_path`, else `{field}.pqcb`. Use to train a v2 codebook alongside a live one. |
| `--update-schema` | Rewrite `schema.toml` so the field's `pq_codebook_path` names the trained file. |

Commits use the codebook only when the schema's `pq_codebook_path`
names it (see [Schema Format](schema_format.md#product-quantization-hnsw-only)) —
pass `--update-schema` to set it as part of training. A commit made
while `pq_codebook_path` is set but the codebook has not been trained
yet fails with an error naming this command; there is no silent
fallback to per-segment training. The codebook is picked up when the
index is opened, so train **before** the ingesting `add` / `put` /
`commit` invocation (each CLI invocation opens the index fresh, so any
subsequent command sees it).

**Example:**

```bash
cat > train.jsonl <<'JSONL'
{"id": "t1", "document": {"fields": {"embedding": {"Vector": [0.1, 0.2, 0.3, 0.4]}}}}
{"id": "t2", "document": {"fields": {"embedding": {"Vector": [0.5, 0.6, 0.7, 0.8]}}}}
JSONL
laurus train pq-codebook --field embedding --input train.jsonl --update-schema
# Training PQ codebook for field 'embedding' on 2 vectors...
# Trained codebook 'embedding.pqcb' (m = 2, k = 256, sub_dim = 2, dimension = 4) from 2 vectors.
# Updated schema.toml: embedding.pq_codebook_path = "embedding.pqcb".
```

---

## `search`

Execute a search query using the [Query DSL](../concepts/query_dsl.md).

```bash
laurus search <QUERY> [--limit <N>] [--offset <N>]
```

**Arguments:**

| Argument / Flag | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `<QUERY>` | Yes | — | Query string in Laurus Query DSL |
| `--limit <N>` | No | `10` | Maximum number of results |
| `--offset <N>` | No | `0` | Number of results to skip |

**Query syntax examples:**

```bash
# Term query
laurus search "body:rust"

# Phrase query
laurus search 'body:"machine learning"'

# Boolean query
laurus search "+body:programming -body:python"

# Fuzzy query (typo tolerance)
laurus search "body:programing~2"

# Wildcard query
laurus search "title:intro*"

# Range query
laurus search "price:[10 TO 50]"

# 3D geographic queries (sphere / bounding box / k-NN)
laurus search "position:geo3d_distance(-3955182, 3350553, 3700276, 5000)"
laurus search "position:geo3d_bbox(-4000000, 3300000, 3650000, -3900000, 3400000, 3750000)"
laurus search "position:geo3d_nearest(-3955182, 3350553, 3700276, 10)"
```

**Table output example:**

```text
╭──────┬────────┬─────────────────────────────────────────╮
│ ID   │ Score  │ Fields                                  │
├──────┼────────┼─────────────────────────────────────────┤
│ doc1 │ 0.8532 │ body: Rust is a systems..., title: Intr │
│ doc3 │ 0.4210 │ body: JavaScript powers..., title: Web  │
╰──────┴────────┴─────────────────────────────────────────╯
```

**JSON output example:**

```bash
laurus --format json search "body:rust" --limit 5
```

```json
[
  {
    "id": "doc1",
    "score": 0.8532,
    "document": {
      "title": "Introduction to Rust",
      "body": "Rust is a systems programming language."
    }
  }
]
```

---

## `repl`

Start an interactive REPL session. See [REPL](repl.md) for details.

```bash
laurus repl
```

---

## `serve`

Start the gRPC server (and optionally the HTTP Gateway).

```bash
laurus serve [OPTIONS]
```

For startup options, configuration, and usage examples, see the [laurus-server documentation](../laurus-server.md):

- [Getting Started](../laurus-server/getting_started.md) — startup options and gRPC connection examples
- [Configuration](../laurus-server/configuration.md) — TOML config file, environment variables, and priority rules
- [Hands-on Tutorial](../laurus-server/tutorial.md) — step-by-step walkthrough

---

## `mcp`

Start the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server on stdio. The MCP server lets AI assistants such as Claude Code or Claude Desktop drive a running laurus-server through a standard set of tools (`create_index`, `add_document`, `search`, etc.).

```bash
laurus mcp [--endpoint <URL>]
```

**Arguments:**

| Flag | Environment Variable | Required | Description |
| :--- | :--- | :--- | :--- |
| `--endpoint <URL>` | `LAURUS_ENDPOINT` | No | gRPC endpoint of a running laurus-server (e.g. `http://localhost:50051`). If omitted, the server starts without a connection; clients can call the `connect` MCP tool later to attach. |

**Examples:**

```bash
# Start the MCP server pre-connected to a local laurus-server
laurus mcp --endpoint http://localhost:50051

# Start the MCP server without a connection; clients call `connect` first
laurus mcp
```

For the full list of MCP tools exposed by this server and how to wire it into Claude Code or Claude Desktop, see the [laurus-mcp documentation](../laurus-mcp.md).
