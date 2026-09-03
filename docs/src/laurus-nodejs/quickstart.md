# Quick Start

## 1. Create an index

```javascript
import { Index, Schema } from "laurus-nodejs";

// In-memory index (ephemeral, useful for prototyping)
const index = await Index.create();

// File-based index (persistent)
// Writes `./myindex/schema.toml` and `./myindex/store/` -- the same layout
// `laurus-cli create index --schema` uses, so this directory can also be
// opened with the CLI (or vice versa).
const schema = new Schema();
schema.addTextField("name");
schema.addTextField("description");
const persistentIndex = await Index.create("./myindex", schema);

// Reopening it later only needs the path -- passing `schema` again would
// throw, since the schema is already persisted.
const reopened = await Index.create("./myindex");
```

## 2. Index documents

```javascript
await index.putDocument("express", {
  name: "Express",
  description: "Fast minimalist web framework for Node.js.",
});
await index.putDocument("fastify", {
  name: "Fastify",
  description: "Fast and low overhead web framework.",
});
await index.commit();
```

## 3. Lexical search

```javascript
// DSL string
const results = await index.search("name:express", 5);

// Term query
const results2 = await index.searchTerm(
  "description", "framework", 5,
);

// Print results
for (const r of results) {
  console.log(`[${r.id}] score=${r.score.toFixed(4)}  ${r.document.name}`);
}
```

## 4. Vector search

Vector search requires a schema with a vector field
and pre-computed embeddings.

```javascript
import { Index, Schema } from "laurus-nodejs";

const schema = new Schema();
schema.addTextField("name");
schema.addHnswField("embedding", 4);

const index = await Index.create(null, schema);
await index.putDocument("express", {
  name: "Express",
  embedding: [0.1, 0.2, 0.3, 0.4],
});
await index.putDocument("pg", {
  name: "pg",
  embedding: [0.9, 0.8, 0.7, 0.6],
});
await index.commit();

const results = await index.searchVector(
  "embedding", [0.1, 0.2, 0.3, 0.4], 3,
);
```

## 5. Hybrid search

```javascript
import {
  Index,
  RRF,
  SearchRequest,
  TermQuery,
  VectorQuery,
} from "laurus-nodejs";

const req = new SearchRequest({ limit: 5 });
req.setLexicalTerm(new TermQuery("name", "express"));
req.setVectorQuery(new VectorQuery("embedding", [0.1, 0.2, 0.3, 0.4]));
req.setRrfFusion(new RRF(60.0));

const results = await index.searchWithRequest(req);
```

## 6. Update and delete

```javascript
// Update: putDocument replaces all existing versions
await index.putDocument("express", {
  name: "Express v5",
  description: "Updated content.",
});
await index.commit();

// Append a new version (RAG chunking pattern)
await index.addDocument("express", {
  name: "Express chunk 2",
  description: "Additional chunk.",
});
await index.commit();

// Retrieve all versions
const docs = await index.getDocuments("express");

// Delete
await index.deleteDocuments("express");
await index.commit();
```

## 7. Schema management

```javascript
const schema = new Schema();
schema.addTextField("name");
schema.addTextField("description");
schema.addIntegerField("stars");
schema.addFloatField("score");
schema.addBooleanField("published");
schema.addBytesField("thumbnail");
schema.addGeoField("location");
schema.addDatetimeField("createdAt");
schema.addHnswField("embedding", 384);
schema.addFlatField("smallVec", 64);
schema.addIvfField("ivfVec", 128, "cosine", 100, 1);
```

## 8. Index statistics

```javascript
const stats = index.stats();
console.log(stats.documentCount);
console.log(stats.vectorFields);
```
