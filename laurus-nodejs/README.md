# laurus-nodejs

Node.js/TypeScript bindings for the
[Laurus](https://github.com/mosuka/laurus) search library —
unified lexical, vector, and hybrid search.

## Features

- **Lexical search** — BM25 scoring with Term, Phrase, Fuzzy,
  Wildcard, Geo, Boolean, and Span queries
- **Vector search** — HNSW, Flat, and IVF indexes with multiple distance metrics
- **Hybrid search** — Combine lexical and vector search
  with RRF or Weighted Sum fusion
- **CJK support** — Japanese, Chinese, and Korean tokenization via [Lindera](https://github.com/lindera/lindera)
- **Native performance** — Rust core via [napi-rs](https://napi.rs), no C API overhead
- **TypeScript types** — Auto-generated `.d.ts` type definitions

## Installation

```bash
npm install laurus-nodejs
```

## Quick Start

```javascript
import { Index, Schema } from "laurus-nodejs";

// Define schema
const schema = new Schema();
schema.addTextField("title");
schema.addTextField("body");
schema.setDefaultFields(["title", "body"]);

// Create an in-memory index
const index = await Index.create(null, schema);

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
// Create index (in-memory or file-based)
const index = await Index.create();                    // in-memory
const index = await Index.create("./myindex", schema); // persistent

// Document CRUD
await index.putDocument("id", { field: "value" });     // upsert
await index.addDocument("id", { field: "chunk" });     // append (RAG)
const docs = await index.getDocuments("id");
await index.deleteDocuments("id");
await index.commit();

// Search
const results = await index.search("query DSL", limit, offset);
const results = await index.searchTerm("field", "term", limit);
const results = await index.searchVector("field", [0.1, ...], limit);
const results = await index.searchVectorText("field", "text", limit);
const results = await index.searchWithRequest(searchRequest);

// Stats
const stats = index.stats();
// { documentCount: 42, vectorFields: {
//     embedding: { count: 42, dimension: 384 }
// } }
```

### Durability / WAL

A persistent index writes every change to a write-ahead log (WAL). By default
the WAL is `fsync`-ed on every record, so each write is fully durable. Opt into
group commit to batch `fsync` for higher write throughput (a crash can lose up
to the last unsynced batch, like SQLite's `synchronous = NORMAL`):

```javascript
import { Index, WalSyncPolicy } from "laurus-nodejs";

// maxRecords, maxBytes, maxIntervalMs (all optional)
const policy = WalSyncPolicy.group(4096, undefined, 1000);
const index = await Index.create("./myindex", schema, policy);

await index.putDocument("doc1", { title: "Hello" });
await index.flushWal(); // force a durable barrier on demand
await index.commit();   // also flushes the WAL
```

Omit `walSyncPolicy` (or pass `WalSyncPolicy.perRecord()`) to keep the default
per-record durability.

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
schema.addHnswField("embedding", 384, "cosine", 16, 200, undefined, "bert");
schema.addFlatField("embedding", 384);
schema.addIvfField("embedding", 384, "cosine", 100, 1);
schema.addEmbedder("bert", {
  type: "candle_bert",
  model: "sentence-transformers/all-MiniLM-L6-v2",
});
schema.setDefaultFields(["title", "body"]);
```

### Search Request (Advanced)

```javascript
import { SearchRequest } from "laurus-nodejs";

const req = new SearchRequest(10, 0);  // limit, offset
req.setQueryDsl("title:hello");
req.setLexicalTermQuery("body", "programming");
req.setLexicalPhraseQuery("title", ["machine", "learning"]);
req.setVectorQuery("embedding", [0.1, 0.2, ...]);
req.setVectorTextQuery("embedding", "query text");
req.setFilterQuery("category", "tech");
req.setRrfFusion(60.0);
req.setWeightedSumFusion(0.3, 0.7);

const results = await index.searchWithRequest(req);
```

### Text Analysis

```javascript
import { WhitespaceTokenizer, SynonymDictionary, SynonymGraphFilter } from "laurus-nodejs";

const tokenizer = new WhitespaceTokenizer();
const tokens = tokenizer.tokenize("hello world");

const synDict = new SynonymDictionary();
synDict.addSynonymGroup(["ml", "machine learning"]);

const filter = new SynonymGraphFilter(synDict, true, 0.8);
const expanded = filter.apply(tokens);
```

## Data Types

| JavaScript | Laurus Field Type |
| --- | --- |
| `string` | Text |
| `number` (integer) | Int64 |
| `number` (float) | Float64 |
| `boolean` | Boolean |
| `null` | Null |
| `number[]` | Vector |
| `{ lat, lon }` | Geo |
| `Date` / ISO8601 string | DateTime |
| `Buffer` | Bytes |

## Examples

See the [examples/](examples/) directory:

- [quickstart.mjs](examples/quickstart.mjs) — Basic index, document, and search
- [lexical-search.mjs](examples/lexical-search.mjs) — All lexical query types
- [vector-search.mjs](examples/vector-search.mjs) — Vector search with HNSW
- [hybrid-search.mjs](examples/hybrid-search.mjs) — Hybrid search
  with RRF and WeightedSum fusion

## Building from Source

```bash
cd laurus-nodejs
npm install
npm run build        # release build
npm run build:debug  # debug build
npm test             # run tests
```

## License

MIT
