# API Reference

## Index

The main entry point for creating and querying search indexes.

### Static Methods

#### `Index.create(schema?)`

Create a new in-memory (ephemeral) index.

- **Parameters:**
  - `schema` (Schema, optional) -- Schema definition.
- **Returns:** `Promise<Index>`

#### `Index.open(name, schema?)`

Open or create a persistent index backed by OPFS.

- **Parameters:**
  - `name` (string) -- Index name (OPFS subdirectory).
  - `schema` (Schema, optional) -- Schema definition.
- **Returns:** `Promise<Index>`

### Instance Methods

#### `putDocument(id, document)`

Replace a document (upsert).

- **Parameters:**
  - `id` (string) -- Document identifier.
  - `document` (object) -- Key-value pairs matching schema fields.
- **Returns:** `Promise<void>`

#### `addDocument(id, document)`

Append a document version (multi-version RAG pattern).

- **Parameters / Returns:** Same as `putDocument`.

#### `getDocuments(id)`

Retrieve all versions of a document.

- **Parameters:**
  - `id` (string)
- **Returns:** `Promise<object[]>`

#### `deleteDocuments(id)`

Delete all versions of a document.

- **Parameters:**
  - `id` (string)
- **Returns:** `Promise<void>`

#### `commit()`

Flush writes and make changes searchable. If opened with
`Index.open()`, data is also persisted to OPFS.

- **Returns:** `Promise<void>`

#### `search(query, limit?, offset?)`

Search using a DSL string query.

- **Parameters:**
  - `query` (string) -- Query DSL (e.g. `"title:hello"`).
  - `limit` (number, default 10)
  - `offset` (number, default 0)
- **Returns:** `Promise<SearchResult[]>`

#### `searchTerm(field, term, limit?, offset?)`

Search for an exact term.

- **Parameters:**
  - `field` (string) -- Field name.
  - `term` (string) -- Exact term.
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchVector(field, vector, limit?, offset?)`

Search by vector similarity.

- **Parameters:**
  - `field` (string) -- Vector field name.
  - `vector` (number[]) -- Query embedding.
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchVectorText(field, text, limit?, offset?)`

Search by text (embedded by the registered embedder).

- **Parameters:**
  - `field` (string) -- Vector field name.
  - `text` (string) -- Text to embed.
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `stats()`

Return index statistics.

- **Returns:** `{ documentCount: number, vectorFields: { [name]: { count, dimension } } }`

## Schema

Builder for defining index fields and embedders.

### Constructor

#### `new Schema()`

Create an empty schema.

### Methods

#### `addTextField(name, stored?, indexed?, termVectors?, analyzer?)`

Add a full-text field.

#### `addIntegerField(name, stored?, indexed?, multiValued?)`

Add a 64-bit integer field. Pass `multiValued: true` to accept arrays of
integers; range queries then match if any value satisfies the predicate
(Lucene-style "any match" with constant scoring).

#### `addFloatField(name, stored?, indexed?, multiValued?)`

Add a 64-bit float field. Pass `multiValued: true` to accept arrays of
floats; range queries then match if any value satisfies the predicate
(Lucene-style "any match" with constant scoring).

#### `addBooleanField(name, stored?, indexed?)`

Add a boolean field.

#### `addDateTimeField(name, stored?, indexed?)`

Add a date/time field.

#### `addGeoField(name, stored?, indexed?)`

Add a geographic coordinate field.

#### `addBytesField(name, stored?)`

Add a binary data field.

#### `addHnswField(name, dimension, distance?, m?, efConstruction?, embedder?)`

Add an HNSW vector index field.

- `distance`: `"cosine"` (default), `"euclidean"`, `"dot_product"`,
  `"manhattan"`, `"angular"`
- `m`: Branching factor (default 16)
- `efConstruction`: Build-time expansion (default 200)

#### `addFlatField(name, dimension, distance?, embedder?)`

Add a brute-force vector index field.

#### `addIvfField(name, dimension, distance?, nClusters?, nProbe?, embedder?)`

Add an IVF vector index field.

#### `addEmbedder(name, config)`

Register a named embedder. In WASM, only `"precomputed"` type is supported.

```javascript
schema.addEmbedder("my-embedder", { type: "precomputed" });
```

#### `setDefaultFields(fields)`

Set the default search fields.

#### `setDynamicFieldPolicy(policy)`

Set how the engine treats fields that appear in ingested documents but are
absent from the schema. `policy` is one of `"strict"`, `"dynamic"`
(default), or `"ignore"` (case-insensitive). Throws on an invalid value.

- `"strict"` — Reject the document.
- `"dynamic"` — Infer a type for each undeclared field and add it to the
  schema. **Warning**: integer fields silently truncate incoming float
  values (`3.14` → `3`).
- `"ignore"` — Silently drop the undeclared fields.

See [Schema & Fields](../concepts/schema_and_fields.md#dynamic-schema) for
the full behaviour matrix.

#### `dynamicFieldPolicy()`

Returns the current policy as a lowercase string.

#### `fieldNames()`

Returns an array of defined field names.

## SearchResult

```typescript
interface SearchResult {
  id: string;
  score: number;
  document: object | null;
}
```

## Analysis

### WhitespaceTokenizer

```javascript
const tokenizer = new WhitespaceTokenizer();
const tokens = tokenizer.tokenize("hello world");
// [{ text: "hello", position: 0, ... }, { text: "world", position: 1, ... }]
```

### SynonymDictionary

```javascript
const dict = new SynonymDictionary();
dict.addSynonymGroup(["ml", "machine learning"]);
```

### SynonymGraphFilter

```javascript
const filter = new SynonymGraphFilter(dict, true, 0.8);
const expanded = filter.apply(tokens);
```
