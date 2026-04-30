# laurus-wasm

WebAssembly bindings for the
[Laurus](https://github.com/mosuka/laurus) search library —
unified lexical, vector, and hybrid search in the browser.

## Features

- **Lexical search** — BM25 scoring with Term, Phrase, Fuzzy,
  Wildcard, Geo, Boolean, and Span queries
- **Vector search** — HNSW, Flat, and IVF indexes with multiple distance metrics
- **Hybrid search** — Combine lexical and vector search
  with RRF or Weighted Sum fusion
- **CJK support** — Japanese, Chinese, and Korean tokenization via [Lindera](https://github.com/lindera/lindera)
- **OPFS persistence** — Data survives page reloads using the
  browser's Origin Private File System
- **JS callback embedder** — Supply your own embedding function
  (e.g. Transformers.js) via a JavaScript callback

## Quick Start

```javascript
import init, { Index, Schema } from "./pkg/laurus_wasm.js";

await init();

// Define schema
const schema = new Schema();
schema.addTextField("title");
schema.addTextField("body");
schema.setDefaultFields(["title", "body"]);

// Create an OPFS-persistent index (survives page reloads)
const index = await Index.open("my-index", schema);

// Index documents
await index.putDocument("doc1", {
  title: "Rust Programming",
  body: "Safety and speed.",
});
await index.putDocument("doc2", {
  title: "Python Basics",
  body: "Versatile language.",
});
await index.commit();

// Search with DSL string
const results = await index.search("programming", 5);
for (const r of results) {
  console.log(r.id, r.score, r.document.title);
}
```

## API Overview

### Index

```javascript
// Create index (in-memory or OPFS-persistent)
const index = await Index.create(schema);              // in-memory (ephemeral)
const index = await Index.open("my-index", schema);    // OPFS (persistent)

// Document CRUD
await index.putDocument("id", { field: "value" });     // upsert
await index.addDocument("id", { field: "chunk" });     // append (RAG)
const docs = await index.getDocuments("id");
await index.deleteDocuments("id");
await index.commit();                                  // flush + persist to OPFS

// Search
const results = await index.search("query DSL", limit, offset);
const results = await index.searchTerm("field", "term", limit);
const results = await index.searchVector("field", [0.1, ...], limit);
const results = await index.searchVectorText("field", "text", limit);

// Stats
const stats = index.stats();
// { documentCount: 42, vectorFields: {
//     embedding: { count: 42, dimension: 384 }
// } }
```

### Schema

```javascript
const schema = new Schema();
schema.addTextField("title", true, true, false, "lindera-ipadic");
schema.addIntegerField("year");
schema.addFloatField("price");
schema.addBooleanField("active");
schema.addDatetimeField("created_at");
schema.addGeoField("location");
schema.addBytesField("thumbnail");
schema.addHnswField("embedding", 384, "cosine", 16, 200, "minilm");
schema.addFlatField("embedding", 384);
schema.addIvfField("embedding", 384, "cosine", 100, 1);
schema.addEmbedder("minilm", {
  type: "callback",
  embed: async (text) => {
    // Your embedding function here (e.g. Transformers.js)
    return [0.1, 0.2, ...];
  },
});
schema.setDefaultFields(["title", "body"]);
```

## Examples

See the [examples/](examples/) directory for a full demo with
Transformers.js embeddings and OPFS persistence.

## Building from Source

```bash
cd laurus-wasm

# Development build
wasm-pack build --target web --dev

# Release build
wasm-pack build --target web --release

# Bundle the OPFS helper (./opfs subpath) into pkg/
./scripts/postbuild.sh

# Serve the demo
python3 -m http.server 8080
# Open http://localhost:8080/examples/
```

The post-build script copies `js/opfs.js` and `js/opfs.d.ts` into `pkg/`
and patches `pkg/package.json` so consumers can
`import { downloadDictionary } from "laurus-wasm/opfs"`. Run it after
every `wasm-pack build` (it is idempotent).

## License

MIT
