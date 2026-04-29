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
| `addIntegerField(name, stored?, indexed?, multiValued?)` | 64-bit integer field. Pass `multiValued: true` to accept arrays of integers (range queries match if any value satisfies the predicate). |
| `addFloatField(name, stored?, indexed?, multiValued?)` | 64-bit float field. Pass `multiValued: true` to accept arrays of floats (range queries match if any value satisfies the predicate). |
| `addBooleanField(name, stored?, indexed?)` | Boolean field. |
| `addBytesField(name, stored?)` | Raw bytes field. |
| `addGeoField(name, stored?, indexed?)` | Geographic coordinate field. |
| `addGeo3dField(name, stored?, indexed?)` | 3D ECEF Cartesian point field (x, y, z in metres). See [Geo3d concepts](../concepts/geo3d.md). |
| `addDatetimeField(name, stored?, indexed?)` | UTC datetime field. |
| `addHnswField(name, dimension, distance?, m?, efConstruction?, embedder?)` | HNSW vector field. |
| `addFlatField(name, dimension, distance?, embedder?)` | Flat (brute-force) vector field. |
| `addIvfField(name, dimension, distance?, nClusters?, nProbe?, embedder?)` | IVF vector field. |
| `addEmbedder(name, config)` | Register a named embedder. |
| `setDefaultFields(fields)` | Set default search fields. |
| `setDynamicFieldPolicy(policy)` | Set how undeclared fields are handled. `policy` is `"strict"`, `"dynamic"` (default), or `"ignore"`. See notes below. |
| `dynamicFieldPolicy()` | Return the current policy as a lowercase string. |
| `fieldNames()` | Return all field names. |
| `toString()` | Return a string representation of the schema (`"Schema(fields=[...])"`). |

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
  numericType?: "integer" | "float",
)
```

Matches numeric values in `[min, max]`. Pass `null` (or omit) for an
open bound. `numericType` selects the underlying range type
(`"integer"` (default) or `"float"`); other values throw.

### GeoDistanceQuery

```typescript
GeoDistanceQuery.withinRadius(
  field: string, lat: number, lon: number, distanceM: number,
): GeoDistanceQuery
```

Geographic distance (radius) search.

### GeoBoundingBoxQuery

```typescript
GeoBoundingBoxQuery.withinBoundingBox(
  field: string,
  minLat: number, minLon: number,
  maxLat: number, maxLon: number,
): GeoBoundingBoxQuery
```

Geographic bounding-box search.

### Geo3dDistanceQuery

```typescript
Geo3dDistanceQuery.withinSphere(
  field: string,
  x: number, y: number, z: number,
  distanceM: number,
): Geo3dDistanceQuery
```

Sphere search over a 3D ECEF point field. Returns documents whose `(x, y, z)`
coordinate is within `distanceM` metres of the centre. See
[Geo3d concepts](../concepts/geo3d.md) for ECEF theory.

### Geo3dBoundingBoxQuery

```typescript
Geo3dBoundingBoxQuery.withinBox(
  field: string,
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
): Geo3dBoundingBoxQuery
```

Axis-aligned 3D bounding-box search.

### Geo3dNearestQuery

```typescript
Geo3dNearestQuery.kNearest(
  field: string,
  x: number, y: number, z: number,
  k: number,
  initialRadiusM?: number,
  maxRadiusM?: number,
): Geo3dNearestQuery
```

k-nearest-neighbour search over a 3D ECEF point field. The optional
`initialRadiusM` and `maxRadiusM` parameters tune the iterative-expansion
search cone.

### BooleanQuery

```typescript
class BooleanQuery {
  constructor();
  // For each query type X in
  //   { Term, Phrase, Fuzzy, Wildcard, NumericRange,
  //     GeoDistance, GeoBoundingBox,
  //     Geo3dDistance, Geo3dBoundingBox, Geo3dNearest,
  //     Boolean, Span }:
  mustX(query: X): void;
  shouldX(query: X): void;
  mustNotX(query: X): void;
}
```

Compound boolean query with MUST / SHOULD / MUST_NOT clauses. Each clause
takes an instance of a specific query class — for example,
`mustTerm(new TermQuery("body", "rust"))` or
`shouldGeo3dNearest(Geo3dNearestQuery.kNearest(...))`.

The Node.js binding exposes 36 per-type methods (12 query types × 3
polarities) instead of a single polymorphic `must(query)` because of a
limitation in `napi-derive`'s validation of `Either<&T, ...>` arguments
for classes with `js_name` overrides.

`must` clauses all have to match; `mustNot` clauses must not match.
`should` clauses contribute to scoring; at least one of them must match if
there are no `must` clauses.

```javascript
const bq = new BooleanQuery();
bq.mustTerm(new TermQuery("body", "programming"));
bq.mustNotTerm(new TermQuery("title", "python"));
bq.shouldFuzzy(new FuzzyQuery("body", "data", 1));
```

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
interface SearchRequestOptions {
  queryDsl?: string;
  limit?: number;   // default 10
  offset?: number;  // default 0
}

class SearchRequest {
  constructor(options?: SearchRequestOptions);
}
```

Construct with primitive options first; attach polymorphic clauses with
the per-type setters below. As with `BooleanQuery`, the binding exposes
per-type setters because of `napi-derive`'s limitation on `Either<&T, ...>`
arguments.

### DSL and fusion setters

| Method | Description |
| :--- | :--- |
| `setQueryDsl(dsl: string)` | Set a DSL string query. |
| `setRrfFusion(rrf: RRF)` | Use RRF fusion. |
| `setWeightedSumFusion(ws: WeightedSum)` | Use weighted-sum fusion. |

### Vector setters

| Method | Description |
| :--- | :--- |
| `setVectorQuery(query: VectorQuery)` | Set a pre-computed vector query. |
| `setVectorTextQuery(query: VectorTextQuery)` | Set a text-based vector query (auto-embedded by the configured embedder). |

### Lexical setters (per type)

For each query type `X` in `{ Term, Phrase, Fuzzy, Wildcard, NumericRange,
GeoDistance, GeoBoundingBox, Geo3dDistance, Geo3dBoundingBox, Geo3dNearest,
Boolean, Span }`, the request exposes:

| Method | Description |
| :--- | :--- |
| `setLexicalX(query: X)` | Set the lexical component for an explicit hybrid request. |
| `setFilterX(query: X)` | Set the post-scoring filter component. |

That is, 24 per-type setters in total (12 lexical + 12 filter), in addition
to the DSL, vector, and fusion setters above.

```javascript
const req = new SearchRequest({ limit: 5 });
req.setLexicalTerm(new TermQuery("title", "rust"));
req.setVectorQuery(new VectorQuery("embedding", [0.1, 0.2, 0.3, 0.4]));
req.setRrfFusion(new RRF(60.0));
const results = await index.searchWithRequest(req);
```

---

## SearchResult

Returned by search methods as an array.

```typescript
interface SearchResult {
  id: string;        // External document identifier
  score: number;     // Relevance score
  document: object | null; // Retrieved fields, or null if not stored
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
| `string` | `Text` | ISO 8601 strings become `DateTime` |
| `number[]` | `Vector` | Coerced to `f32` |
| `{ lat, lon }` | `Geo` | Two `number` values |
| `{ x, y, z }` | `GeoEcef` | Three `number` values, meters (3D ECEF Cartesian) |
