# API Reference

## Index

The primary entry point. Wraps the Laurus search engine.

```typescript
class Index {
  static create(
    path?: string | null,
    schema?: Schema,
  ): Promise<Index>;
}
```

### Factory method

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `string \| null` | `null` | Directory for persistent storage. `null` creates an in-memory index. |
| `schema` | `Schema` | empty | Schema definition. |

### Methods

| Method | Description |
| :--- | :--- |
| `putDocument(id, doc)` | Upsert a document. Replaces all existing versions. |
| `addDocument(id, doc)` | Append a document chunk without removing existing versions. |
| `getDocuments(id)` | Return all stored versions for the given ID. |
| `deleteDocuments(id)` | Delete all versions for the given ID. |
| `commit()` | Flush writes and make pending changes searchable. |
| `search(query, limit?, offset?)` | Search with a DSL string. |
| `searchTerm(field, term, limit?, offset?)` | Search with an exact term match. |
| `searchVector(field, vector, limit?, offset?)` | Search with a pre-computed vector. |
| `searchVectorText(field, text, limit?, offset?)` | Search with text (auto-embedded). |
| `searchWithRequest(request)` | Search with a `SearchRequest`. |
| `stats()` | Return index statistics. |

All document methods and search methods are async
and return Promises. `stats()` is synchronous.

---

## Schema

Defines the fields and index types for an `Index`.

```typescript
class Schema {
  constructor();
}
```

### Field methods

| Method | Description |
| :--- | :--- |
| `addTextField(name, stored?, indexed?, termVectors?, analyzer?)` | Full-text field (inverted index, BM25). |
| `addIntegerField(name, stored?, indexed?)` | 64-bit integer field. |
| `addFloatField(name, stored?, indexed?)` | 64-bit float field. |
| `addBooleanField(name, stored?, indexed?)` | Boolean field. |
| `addBytesField(name, stored?)` | Raw bytes field. |
| `addGeoField(name, stored?, indexed?)` | Geographic coordinate field. |
| `addDatetimeField(name, stored?, indexed?)` | UTC datetime field. |
| `addHnswField(name, dimension, distance?, m?, efConstruction?, embedder?)` | HNSW vector field. |
| `addFlatField(name, dimension, distance?, embedder?)` | Flat (brute-force) vector field. |
| `addIvfField(name, dimension, distance?, nClusters?, nProbe?, embedder?)` | IVF vector field. |
| `addEmbedder(name, config)` | Register a named embedder. |
| `setDefaultFields(fields)` | Set default search fields. |
| `setDynamicFieldPolicy(policy)` | Set how undeclared fields are handled. `policy` is `"strict"`, `"dynamic"` (default), or `"ignore"`. See notes below. |
| `dynamicFieldPolicy()` | Return the current policy as a lowercase string. |
| `fieldNames()` | Return all field names. |

#### Dynamic field policy

Controls what happens when a document is ingested with field names that are
not declared in the schema:

- `"strict"` — Reject the document.
- `"dynamic"` (default) — Infer a type for each undeclared field and add it
  to the schema. **Warning**: integer fields silently truncate incoming
  float values (`3.14` → `3`). Use `"strict"` if you need to reject such
  type mismatches.
- `"ignore"` — Silently drop the undeclared fields.

See [Schema & Fields](../concepts/schema_and_fields.md#dynamic-schema) for
the full behaviour matrix.

### Distance metrics

| Value | Description |
| :--- | :--- |
| `"cosine"` | Cosine similarity (default) |
| `"euclidean"` | Euclidean distance |
| `"dot_product"` | Dot product |
| `"manhattan"` | Manhattan distance |
| `"angular"` | Angular distance |

---

## Query classes

### TermQuery

```typescript
new TermQuery(field: string, term: string)
```

Matches documents containing the exact term in the given field.

### PhraseQuery

```typescript
new PhraseQuery(field: string, terms: string[])
```

Matches documents containing the terms in order.

### FuzzyQuery

```typescript
new FuzzyQuery(field: string, term: string, maxEdits?: number)
```

Approximate match allowing up to `maxEdits` edit-distance
errors (default 2).

### WildcardQuery

```typescript
new WildcardQuery(field: string, pattern: string)
```

Pattern match. `*` matches any sequence, `?` matches one
character.

### NumericRangeQuery

```typescript
new NumericRangeQuery(
  field: string,
  min?: number | null,
  max?: number | null,
  isFloat?: boolean,
)
```

Matches numeric values in `[min, max]`. Pass `null` for an
open bound.

### GeoQuery

```typescript
GeoQuery.withinRadius(
  field: string, lat: number, lon: number, distanceKm: number,
): GeoQuery

GeoQuery.withinBoundingBox(
  field: string,
  minLat: number, minLon: number,
  maxLat: number, maxLon: number,
): GeoQuery
```

Geographic search by radius or bounding box.

### BooleanQuery

```typescript
class BooleanQuery {
  constructor();
  mustTerm(field: string, term: string): void;
  shouldTerm(field: string, term: string): void;
  mustNotTerm(field: string, term: string): void;
}
```

Compound boolean query with MUST / SHOULD / MUST_NOT clauses.

### SpanQuery

```typescript
SpanQuery.term(field: string, term: string): SpanQuery
SpanQuery.near(
  field: string, terms: string[],
  slop?: number, ordered?: boolean,
): SpanQuery
SpanQuery.nearSpans(
  field: string, clauses: SpanQuery[],
  slop?: number, ordered?: boolean,
): SpanQuery
SpanQuery.containing(
  field: string, big: SpanQuery, little: SpanQuery,
): SpanQuery
SpanQuery.within(
  field: string,
  include: SpanQuery, exclude: SpanQuery, distance: number,
): SpanQuery
```

Positional/proximity span queries.

### VectorQuery

```typescript
new VectorQuery(field: string, vector: number[])
```

Nearest-neighbor search using a pre-computed embedding vector.

### VectorTextQuery

```typescript
new VectorTextQuery(field: string, text: string)
```

Converts `text` to an embedding at query time. Requires an
embedder configured on the index.

---

## SearchRequest

Full-featured search request for advanced control.

```typescript
class SearchRequest {
  constructor(limit?: number, offset?: number);
}
```

### Setter methods

| Method | Description |
| :--- | :--- |
| `setQueryDsl(dsl)` | Set a DSL string query. |
| `setLexicalTermQuery(field, term)` | Set a term-based lexical query. |
| `setLexicalPhraseQuery(field, terms)` | Set a phrase-based lexical query. |
| `setVectorQuery(field, vector)` | Set a pre-computed vector query. |
| `setVectorTextQuery(field, text)` | Set a text-based vector query. |
| `setFilterQuery(field, term)` | Set a post-scoring filter. |
| `setRrfFusion(k?)` | Use RRF fusion (default k=60). |
| `setWeightedSumFusion(lexicalWeight?, vectorWeight?)` | Use weighted sum fusion. |

---

## SearchResult

Returned by search methods as an array.

```typescript
interface SearchResult {
  id: string;        // External document identifier
  score: number;     // Relevance score
  document: object | null; // Retrieved fields, or null
}
```

---

## Fusion algorithms

### RRF

```typescript
new RRF(k?: number)  // default 60.0
```

Reciprocal Rank Fusion. Merges lexical and vector result lists
by rank position.

### WeightedSum

```typescript
new WeightedSum(
  lexicalWeight?: number,  // default 0.5
  vectorWeight?: number,   // default 0.5
)
```

Normalises both score lists independently, then combines them.

---

## Text analysis

### SynonymDictionary

```typescript
class SynonymDictionary {
  constructor();
  addSynonymGroup(terms: string[]): void;
}
```

### WhitespaceTokenizer

```typescript
class WhitespaceTokenizer {
  constructor();
  tokenize(text: string): Token[];
}
```

### SynonymGraphFilter

```typescript
class SynonymGraphFilter {
  constructor(
    dictionary: SynonymDictionary,
    keepOriginal?: boolean,  // default true
    boost?: number,          // default 1.0
  );
  apply(tokens: Token[]): Token[];
}
```

### Token

```typescript
interface Token {
  text: string;
  position: number;
  startOffset: number;
  endOffset: number;
  boost: number;
  stopped: boolean;
  positionIncrement: number;
  positionLength: number;
}
```

---

## Field value types

JavaScript values are automatically converted to Laurus
`DataValue` types:

| JavaScript type | Laurus type | Notes |
| :--- | :--- | :--- |
| `null` | `Null` | |
| `boolean` | `Bool` | |
| `number` (integer) | `Int64` | |
| `number` (float) | `Float64` | |
| `string` | `Text` | ISO8601 strings become `DateTime` |
| `number[]` | `Vector` | Coerced to `f32` |
| `{ lat, lon }` | `Geo` | Two `number` values |
| `Date` | `DateTime` | Via timestamp |
| `Buffer` | `Bytes` | |
