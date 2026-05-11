# Architecture

This page explains how Laurus is structured internally. Understanding the architecture will help you make better decisions about schema design, analyzer selection, and search strategies.

## Project Structure

Laurus is organized as a Cargo workspace with five crates:

```mermaid
graph TB
    CLI["laurus-cli\n(Binary Crate)\nCLI + REPL"]
    SRV["laurus-server\n(Library + Binary)\ngRPC Server + HTTP Gateway"]
    MCP["laurus-mcp\n(Library + Binary)\nMCP Server"]
    PY["laurus-python\n(cdylib)\nPython Bindings"]
    LIB["laurus\n(Library Crate)\nCore Search Engine"]

    CLI --> LIB
    CLI --> SRV
    CLI --> MCP
    SRV --> LIB
    MCP --> SRV
    MCP --> LIB
    PY --> LIB
```

| Crate | Type | Description |
| :--- | :--- | :--- |
| **laurus** | Library | Core search engine -- lexical, vector, and hybrid search |
| **laurus-cli** | Binary | Command-line interface for index management and search |
| **laurus-server** | Library + Binary | gRPC server with optional HTTP/JSON gateway |
| **laurus-mcp** | Binary | MCP (Model Context Protocol) stdio server that proxies to laurus-server |
| **laurus-python** | cdylib | Python bindings via PyO3 / Maturin |
| **laurus-nodejs** | cdylib | Node.js bindings via NAPI-RS |
| **laurus-wasm** | WebAssembly | Browser / edge bindings via wasm-bindgen |
| **laurus-ruby** | cdylib | Ruby bindings via magnus / rb-sys |
| **laurus-php** | PHP extension | PHP bindings via ext-php-rs (excluded from the workspace) |

For details on each crate, see:

- [Library Overview](laurus.md)
- [CLI Overview](laurus-cli.md)
- [Server Overview](laurus-server.md)
- [MCP Server Overview](laurus-mcp.md)
- [Python Bindings Overview](laurus-python.md)
- [Node.js Bindings Overview](laurus-nodejs.md)
- [WASM Bindings Overview](laurus-wasm.md)
- [Ruby Bindings Overview](laurus-ruby.md)
- [PHP Bindings Overview](laurus-php.md)

## High-Level Overview

Laurus is organized around a single `Engine` that coordinates four internal components:

```mermaid
graph TB
    subgraph Engine
        SCH["Schema"]
        LS["LexicalStore\n(Inverted Index)"]
        VS["VectorStore\n(HNSW / Flat / IVF)"]
        DL["DocumentLog\n(WAL + Document Storage)"]
    end

    Storage["Storage (trait)\nMemory / File / File+Mmap"]

    LS --- Storage
    VS --- Storage
    DL --- Storage
```

| Component | Responsibility |
| :--- | :--- |
| **Schema** | Declares fields and their types; determines how each field is routed |
| **LexicalStore** | Inverted index for keyword search (BM25 scoring) |
| **VectorStore** | Vector index for similarity search (Flat, HNSW, or IVF) |
| **DocumentLog** | Write-ahead log (WAL) for durability + raw document storage |

All three stores share a single `Storage` backend, isolated by key prefixes (`lexical/`, `vector/`, `documents/`).

## Engine Lifecycle

### Building an Engine

The `EngineBuilder` assembles the engine from its parts:

```rust
let engine = Engine::builder(storage, schema)
    .analyzer(analyzer)      // optional: for text fields
    .embedder(embedder)      // optional: for vector fields
    .build()
    .await?;
```

```mermaid
sequenceDiagram
    participant User
    participant EngineBuilder
    participant Engine

    User->>EngineBuilder: new(storage, schema)
    User->>EngineBuilder: .analyzer(analyzer)
    User->>EngineBuilder: .embedder(embedder)
    User->>EngineBuilder: .build().await
    EngineBuilder->>EngineBuilder: split_schema()
    Note over EngineBuilder: Separate fields into\nLexicalIndexConfig\n+ VectorIndexConfig
    EngineBuilder->>Engine: Create LexicalStore
    EngineBuilder->>Engine: Create VectorStore
    EngineBuilder->>Engine: Create DocumentLog
    EngineBuilder->>Engine: Recover from WAL
    EngineBuilder-->>User: Engine ready
```

During `build()`, the engine:

1. **Splits the schema** — lexical fields go to `LexicalIndexConfig`, vector fields go to `VectorIndexConfig`
2. **Creates prefixed storage** — each component gets an isolated namespace (`lexical/`, `vector/`, `documents/`)
3. **Initializes stores** — `LexicalStore` and `VectorStore` are constructed with their configs
4. **Recovers from WAL** — replays any uncommitted operations from a previous session

### Schema Splitting

The `Schema` contains both lexical and vector fields. At build time, `split_schema()` separates them:

```mermaid
graph LR
    S["Schema\ntitle: Text\nbody: Text\ncategory: Text\npage: Integer\ncontent_vec: HNSW"]

    S --> LC["LexicalIndexConfig\ntitle: TextOption\nbody: TextOption\ncategory: TextOption\npage: IntegerOption\n_id: KeywordAnalyzer"]

    S --> VC["VectorIndexConfig\ncontent_vec: HnswOption\n(dim=384, m=16, ef=200)"]
```

Key details:

- The reserved `_id` field is always added to the lexical config with `KeywordAnalyzer` (exact match)
- A `PerFieldAnalyzer` wraps per-field analyzer settings; if you pass a simple `StandardAnalyzer`, it becomes the default for all text fields
- A `PerFieldEmbedder` works the same way for vector fields

## Indexing Data Flow

When you call `engine.add_document(id, doc)`:

```mermaid
sequenceDiagram
    participant User
    participant Engine
    participant WAL as DocumentLog (WAL)
    participant Lexical as LexicalStore
    participant Vector as VectorStore

    User->>Engine: add_document("doc-1", doc)
    Engine->>WAL: Append to WAL
    Engine->>Engine: Assign internal ID (u64)

    loop For each field in document
        alt Lexical field (text, integer, etc.)
            Engine->>Lexical: Analyze + index field
        else Vector field
            Engine->>Vector: Embed + index field
        end
    end

    Note over Engine: Document is buffered\nbut NOT yet searchable

    User->>Engine: commit()
    Engine->>Lexical: Flush segments to storage
    Engine->>Vector: Flush segments to storage
    Engine->>WAL: Truncate WAL
    Note over Engine: Documents are\nnow searchable
```

Key points:

- **WAL-first**: every write is logged before modifying in-memory structures
- **Dual indexing**: each field is routed to either the lexical or vector store based on the schema
- **Commit required**: documents become searchable only after `commit()`

## Search Data Flow

When you call `engine.search(request)`:

```mermaid
sequenceDiagram
    participant User
    participant Engine
    participant Lexical as LexicalStore
    participant Vector as VectorStore
    participant Fusion

    User->>Engine: search(request)

    opt Filter query present
        Engine->>Lexical: Execute filter query
        Lexical-->>Engine: Allowed document IDs
    end

    par Lexical search
        Engine->>Lexical: Execute lexical query
        Lexical-->>Engine: Ranked hits (BM25)
    and Vector search
        Engine->>Vector: Execute vector query
        Vector-->>Engine: Ranked hits (similarity)
    end

    alt Both lexical and vector results
        Engine->>Fusion: Fuse results (RRF or WeightedSum)
        Fusion-->>Engine: Merged ranked list
    end

    Engine->>Engine: Apply offset + limit
    Engine-->>User: Vec of SearchResult
```

The search pipeline has three stages:

1. **Filter** (optional) — execute a filter query on the lexical index to get a set of allowed document IDs
2. **Search** — run lexical and/or vector queries in parallel
3. **Fusion** — if both query types are present, merge results using RRF (default, k=60) or WeightedSum

## Storage Architecture

All components share a single `Storage` trait implementation, but use key prefixes to isolate their data:

```mermaid
graph TB
    Engine --> PS1["PrefixedStorage\nprefix: 'lexical/'"]
    Engine --> PS2["PrefixedStorage\nprefix: 'vector/'"]
    Engine --> PS3["PrefixedStorage\nprefix: 'documents/'"]

    PS1 --> S["Storage Backend\n(Memory / File / File+Mmap)"]
    PS2 --> S
    PS3 --> S
```

| Backend | Description | Best For |
| :--- | :--- | :--- |
| `MemoryStorage` | All data in memory | Testing, small datasets, ephemeral use |
| `FileStorage` | Standard file I/O | General production use |
| `FileStorage` (mmap) | Memory-mapped files (`use_mmap = true`) | Large datasets, read-heavy workloads |

## Per-Field Dispatch

When a `PerFieldAnalyzer` is provided, the engine dispatches analysis to field-specific analyzers. The same pattern applies to `PerFieldEmbedder`.

```mermaid
graph LR
    PFA["PerFieldAnalyzer"]
    PFA -->|"title"| KA["KeywordAnalyzer"]
    PFA -->|"body"| SA["StandardAnalyzer"]
    PFA -->|"description"| JA["JapaneseAnalyzer"]
    PFA -->|"_id"| KA2["KeywordAnalyzer\n(always)"]
    PFA -->|other fields| DEF["Default Analyzer\n(StandardAnalyzer)"]
```

This allows different fields to use different analysis strategies within the same engine.

## Summary

| Aspect | Detail |
| :--- | :--- |
| **Core struct** | `Engine` — coordinates all operations |
| **Builder** | `EngineBuilder` — assembles Engine from Storage + Schema + Analyzer + Embedder |
| **Schema split** | Lexical fields → `LexicalIndexConfig`, Vector fields → `VectorIndexConfig` |
| **Write path** | WAL → in-memory buffers → `commit()` → persistent storage |
| **Read path** | Query → parallel lexical/vector search → fusion → ranked results |
| **Storage isolation** | `PrefixedStorage` with `lexical/`, `vector/`, `documents/` prefixes |
| **Per-field dispatch** | `PerFieldAnalyzer` and `PerFieldEmbedder` route to field-specific implementations |

## Next Steps

- Understand field types and schema design: [Schema & Fields](concepts/schema_and_fields.md)
- Learn about text analysis: [Text Analysis](concepts/analysis.md)
- Learn about embeddings: [Embeddings](concepts/embedding.md)
