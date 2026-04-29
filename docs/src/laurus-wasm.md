# WASM Binding Overview

The `laurus-wasm` package provides WebAssembly bindings for the
Laurus search engine. It enables lexical, vector, and hybrid search
directly in browsers and edge runtimes (Cloudflare Workers,
Vercel Edge Functions, Deno Deploy) without a server.

## Features

- **Lexical Search** -- Full-text search powered by an inverted
  index with BM25 scoring
- **Vector Search** -- Approximate nearest neighbor (ANN) search
  using Flat, HNSW, or IVF indexes
- **Hybrid Search** -- Combine lexical and vector results with
  fusion algorithms (RRF, WeightedSum)
- **Rich Query DSL** -- Term, Phrase, Fuzzy, Wildcard,
  NumericRange, Geo, Boolean, Span queries
- **Text Analysis** -- Tokenizers, filters, and synonym expansion
- **In-memory Storage** -- Fast ephemeral indexes
- **OPFS Persistence** -- Indexes survive page reloads via the
  Origin Private File System
- **TypeScript Types** -- Auto-generated `.d.ts` type definitions
- **Async API** -- All I/O operations return Promises

## Architecture

```mermaid
graph LR
    subgraph "laurus-wasm"
        WASM[wasm-bindgen API]
    end
    subgraph "laurus (core)"
        Engine
        MemoryStorage
    end
    subgraph "Browser"
        JS[JavaScript / TypeScript]
        OPFS[Origin Private File System]
    end
    JS --> WASM
    WASM --> Engine
    Engine --> MemoryStorage
    WASM -.->|persist| OPFS
```

## Embedding Strategies

On native platforms Laurus supports several built-in embedders (Candle BERT,
Candle CLIP, OpenAI API) that the engine can invoke automatically when a
document is indexed or when `searchVectorText("field", "query text")` is
called. These native embedders cannot run inside `wasm32-unknown-unknown` and
are therefore disabled in the WASM build:

| Embedder      | Dependency        | Why it cannot run in WASM                                   |
| ------------- | ----------------- | ----------------------------------------------------------- |
| `candle_bert` | candle (GPU/SIMD) | Requires native SIMD intrinsics and file system for models  |
| `candle_clip` | candle            | Same as above                                               |
| `openai`      | reqwest (HTTP)    | Requires a full async HTTP client (tokio + TLS)             |

(They are excluded from the WASM build via the `embeddings-candle` /
`embeddings-openai` feature flags, which depend on the `native` feature that
is disabled for `wasm32-unknown-unknown`.)

`laurus-wasm` exposes two `addEmbedder` types instead:

- **`"precomputed"`** — The caller supplies vectors directly via `putDocument()`
  and `searchVector()`. The engine performs no embedding.
- **`"callback"`** — Register a JavaScript callback
  `embed: (text) => Promise<number[]>` and the engine will invoke it during
  ingestion and from `searchVectorText()`. This enables in-engine
  auto-embedding using Transformers.js (or any other in-browser embedding
  library) so callers can use the same `searchVectorText("field", "query text")`
  pattern as on native platforms.

### Option A — Precomputed vectors

Compute embeddings on the **JavaScript side** and pass precomputed vectors
to `putDocument()` and `searchVector()`:

```javascript
// Using Transformers.js (all-MiniLM-L6-v2, 384-dim)
import { pipeline } from '@huggingface/transformers';

const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

async function embed(text) {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

// Index with precomputed embedding
const vec = await embed("Introduction to Rust");
await index.putDocument("doc1", { title: "Introduction to Rust", embedding: vec });
await index.commit();

// Search with precomputed query embedding
const queryVec = await embed("safe systems programming");
const results = await index.searchVector("embedding", queryVec);
```

This approach gives you real semantic search in the browser using the same
sentence-transformer models available on native platforms, with the embedding
computation handled by Transformers.js (ONNX Runtime Web) instead of candle.

### Option B — Callback embedder

Register the same Transformers.js pipeline as a `"callback"` embedder so that
the engine can call it automatically. After registration, ingestion and
`searchVectorText()` work transparently without the caller managing vectors:

```javascript
import { pipeline } from '@huggingface/transformers';

const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

schema.addEmbedder("transformers", {
  type: "callback",
  embed: async (text) => {
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
  },
});
schema.addHnswField("embedding", 384, "cosine", undefined, undefined, "transformers");
const index = await Index.create(schema);

await index.putDocument("doc1", { title: "Introduction to Rust" });
await index.commit();

const results = await index.searchVectorText("embedding", "safe systems programming");
```

Compared to Option A, the callback approach lets the engine cache embeddings
during ingestion and avoids duplicating embedding code between writers and
readers. The trade-off is that every `commit()` waits for the JS callback to
resolve, so heavy bulk ingestion may benefit from precomputing vectors.

## When to Use laurus-wasm vs laurus-nodejs

| Criterion   | `laurus-wasm`              | `laurus-nodejs`               |
| ----------- | -------------------------- | ----------------------------- |
| Environment | Browser, Edge Runtime      | Node.js server                |
| Performance | Good (single-threaded)     | Best (native, multi-threaded) |
| Storage     | In-memory + OPFS           | In-memory + File system       |
| Embedding   | Precomputed + JS callback  | Candle, OpenAI, Precomputed   |
| Package     | `npm install laurus-wasm`  | `npm install laurus-nodejs`   |
| Binary size | ~5-10 MB (WASM)            | Platform-native               |
