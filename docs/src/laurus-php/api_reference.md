# API Reference

## Index

The primary entry point. Wraps the Laurus search engine.

```php
new \Laurus\Index(?string $path = null, ?Schema $schema = null)
```

### Constructor

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `$path` | `string\|null` | `null` | Directory path for persistent storage. `null` creates an in-memory index. |
| `$schema` | `Schema\|null` | `null` | Schema definition. An empty schema is used when omitted. |

### Methods

| Method | Description |
| :--- | :--- |
| `putDocument(string $id, array $doc): void` | Upsert a document. Replaces all existing versions with the same ID. |
| `addDocument(string $id, array $doc): void` | Append a document chunk without removing existing versions. |
| `getDocuments(string $id): array` | Return all stored versions for the given ID. |
| `deleteDocuments(string $id): void` | Delete all versions for the given ID. |
| `commit(): void` | Flush buffered writes and make all pending changes searchable. |
| `search(mixed $query, int $limit = 10, int $offset = 0): array` | Execute a search query. Returns an array of `SearchResult`. |
| `stats(): array` | Return index statistics (`"document_count"`, `"vector_fields"`). |

### `search` query argument

The `$query` parameter accepts any of the following:

- A **DSL string** (e.g. `"title:hello"`, `"embedding:\"memory safety\""`)
- A **lexical query object** (`TermQuery`, `PhraseQuery`, `BooleanQuery`, ...)
- A **vector query object** (`VectorQuery`, `VectorTextQuery`)
- A **`SearchRequest`** for full control

---

## Schema

Defines the fields and index types for an `Index`.

```php
new \Laurus\Schema()
```

### Field methods

| Method | Description |
| :--- | :--- |
| `addTextField(string $name, bool $stored = true, bool $indexed = true, bool $termVectors = false, ?string $analyzer = null): void` | Full-text field (inverted index, BM25). |
| `addIntegerField(string $name, bool $stored = true, bool $indexed = true, bool $multi_valued = false): void` | 64-bit integer field. Pass `$multi_valued = true` to accept arrays of integers (range queries match if any value satisfies the predicate). |
| `addFloatField(string $name, bool $stored = true, bool $indexed = true, bool $multi_valued = false): void` | 64-bit float field. Pass `$multi_valued = true` to accept arrays of floats (range queries match if any value satisfies the predicate). |
| `addBooleanField(string $name, bool $stored = true, bool $indexed = true): void` | Boolean field. |
| `addBytesField(string $name, bool $stored = true): void` | Raw bytes field. |
| `addGeoField(string $name, bool $stored = true, bool $indexed = true): void` | Geographic coordinate field (lat/lon). |
| `addGeo3dField(string $name, bool $stored = true, bool $indexed = true): void` | 3D ECEF Cartesian point field (x, y, z in metres). See [Geo3d concepts](../concepts/geo3d.md). |
| `addDatetimeField(string $name, bool $stored = true, bool $indexed = true): void` | UTC datetime field. |
| `addHnswField(string $name, int $dimension, ?string $distance = "cosine", int $m = 16, int $efConstruction = 200, ?string $embedder = null): void` | HNSW approximate nearest-neighbor vector field. |
| `addFlatField(string $name, int $dimension, ?string $distance = "cosine", ?string $embedder = null): void` | Flat (brute-force) vector field. |
| `addIvfField(string $name, int $dimension, ?string $distance = "cosine", int $nClusters = 100, int $nProbe = 1, ?string $embedder = null): void` | IVF approximate nearest-neighbor vector field. |

### Other methods

| Method | Description |
| :--- | :--- |
| `addEmbedder(string $name, array $config): void` | Register a named embedder definition. `$config` is an associative array with a `"type"` key (see below). |
| `setDefaultFields(array $fieldNames): void` | Set the default fields used when no field is specified in a query. `$fieldNames` is an array of strings. |
| `setDynamicFieldPolicy(string $policy): void` | Set how undeclared fields are handled. `$policy` is `"strict"`, `"dynamic"` (default), or `"ignore"`. See notes below. |
| `dynamicFieldPolicy(): string` | Return the current policy as a lowercase string. |
| `fieldNames(): array` | Return the list of field names defined in this schema. |

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

### Embedder types

| `"type"` | Required keys | Feature flag |
| :--- | :--- | :--- |
| `"precomputed"` | -- | (always available) |
| `"candle_bert"` | `"model"` | `embeddings-candle` |
| `"candle_clip"` | `"model"` | `embeddings-multimodal` |
| `"openai"` | `"model"` | `embeddings-openai` |

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

```php
new \Laurus\TermQuery(string $field, string $term)
```

Matches documents containing the exact term in the given field.

### PhraseQuery

```php
new \Laurus\PhraseQuery(string $field, array $terms)
```

Matches documents containing the terms in order. `$terms` is an array of strings.

### FuzzyQuery

```php
new \Laurus\FuzzyQuery(string $field, string $term, int $maxEdits = 2)
```

Approximate match allowing up to `$maxEdits` edit-distance errors.

### WildcardQuery

```php
new \Laurus\WildcardQuery(string $field, string $pattern)
```

Pattern match. `*` matches any sequence of characters, `?` matches any single character.

### NumericRangeQuery

```php
new \Laurus\NumericRangeQuery(string $field, mixed $min, mixed $max, ?string $numericType = "integer")
```

Matches numeric values in the range `[$min, $max]`. Pass `null` for an open bound. Set `$numericType` to `"integer"` or `"float"`.

### GeoQuery

```php
// Radius search
\Laurus\GeoQuery::withinRadius(string $field, float $lat, float $lon, float $distanceKm): GeoQuery

// Bounding box search
\Laurus\GeoQuery::withinBoundingBox(string $field, float $minLat, float $minLon, float $maxLat, float $maxLon): GeoQuery
```

`withinRadius` returns documents whose coordinate is within `$distanceKm` of the given point. `withinBoundingBox` returns documents within the specified bounding box.

### Geo3dDistanceQuery

```php
\Laurus\Geo3dDistanceQuery::withinSphere(
    string $field,
    float $x, float $y, float $z,
    float $radiusM,
): Geo3dDistanceQuery
```

Sphere search over a 3D ECEF point field. Returns documents whose `(x, y, z)`
coordinate is within `$radiusM` metres of the centre. See
[Geo3d concepts](../concepts/geo3d.md) for ECEF theory.

### Geo3dBoundingBoxQuery

```php
\Laurus\Geo3dBoundingBoxQuery::withinBox(
    string $field,
    float $minX, float $minY, float $minZ,
    float $maxX, float $maxY, float $maxZ,
): Geo3dBoundingBoxQuery
```

Axis-aligned 3D bounding-box search.

### Geo3dNearestQuery

```php
\Laurus\Geo3dNearestQuery::kNearest(
    string $field,
    float $x, float $y, float $z,
    int $k,
): Geo3dNearestQuery
```

k-nearest-neighbour search over a 3D ECEF point field. The `initial_radius_m` /
`max_radius_m` tuning parameters of the core query are not yet exposed in the
PHP binding — see [#344] for parity across bindings.

[#344]: https://github.com/mosuka/laurus/issues/344

### BooleanQuery

```php
$bq = new \Laurus\BooleanQuery();
$bq->must($query);
$bq->should($query);
$bq->mustNot($query);
```

Compound boolean query. `must` clauses all have to match; at least one `should` clause must match; `mustNot` clauses must not match.

### SpanQuery

```php
// Single term
\Laurus\SpanQuery::term(string $field, string $term): SpanQuery

// Near: terms within slop positions
\Laurus\SpanQuery::near(string $field, array $terms, int $slop = 0, bool $ordered = true): SpanQuery

// Containing: big span contains little span
\Laurus\SpanQuery::containing(string $field, SpanQuery $big, SpanQuery $little): SpanQuery

// Within: include span within exclude span at max distance
\Laurus\SpanQuery::within(string $field, SpanQuery $include, SpanQuery $exclude, int $distance): SpanQuery
```

Positional / proximity span queries. `near` takes an array of term strings.

### VectorQuery

```php
new \Laurus\VectorQuery(string $field, array $vector)
```

Approximate nearest-neighbor search using a pre-computed embedding vector. `$vector` is an array of floats.

### VectorTextQuery

```php
new \Laurus\VectorTextQuery(string $field, string $text)
```

Converts `$text` to an embedding at query time and runs vector search. Requires an embedder configured on the index.

---

## SearchRequest

Full-featured search request for advanced control.

```php
new \Laurus\SearchRequest(
    mixed $query = null,
    mixed $lexicalQuery = null,
    mixed $vectorQuery = null,
    mixed $filterQuery = null,
    mixed $fusion = null,
    int $limit = 10,
    int $offset = 0,
)
```

| Parameter | Description |
| :--- | :--- |
| `$query` | A DSL string or single query object. Mutually exclusive with `$lexicalQuery` / `$vectorQuery`. |
| `$lexicalQuery` | Lexical component for explicit hybrid search. |
| `$vectorQuery` | Vector component for explicit hybrid search. |
| `$filterQuery` | Lexical filter applied after scoring. |
| `$fusion` | Fusion algorithm (`RRF` or `WeightedSum`). Defaults to `RRF(k: 60)` when both components are set. |
| `$limit` | Maximum number of results (default 10). |
| `$offset` | Pagination offset (default 0). |

---

## SearchResult

Returned by `Index->search()`.

```php
$result->getId()        // string   -- External document identifier
$result->getScore()     // float    -- Relevance score
$result->getDocument()  // array|null -- Retrieved field values, or null if deleted
```

---

## Fusion algorithms

### RRF

```php
new \Laurus\RRF(float $k = 60.0)
```

Reciprocal Rank Fusion. Merges lexical and vector result lists by rank position. `$k` is a smoothing constant; higher values reduce the influence of top-ranked results.

### WeightedSum

```php
new \Laurus\WeightedSum(float $lexicalWeight = 0.5, float $vectorWeight = 0.5)
```

Normalises both score lists independently, then combines them as `$lexicalWeight * lexical_score + $vectorWeight * vector_score`.

---

## Text analysis

### SynonymDictionary

```php
$dict = new \Laurus\SynonymDictionary();
$dict->addSynonymGroup(["fast", "quick", "rapid"]);
```

A dictionary of synonym groups. All terms in a group are treated as synonyms of each other.

### WhitespaceTokenizer

```php
$tokenizer = new \Laurus\WhitespaceTokenizer();
$tokens = $tokenizer->tokenize("hello world");
```

Splits text on whitespace boundaries and returns an array of `Token` objects.

### SynonymGraphFilter

```php
$filter = new \Laurus\SynonymGraphFilter($dictionary, true, 1.0);
$expanded = $filter->apply($tokens);
```

Token filter that expands tokens with their synonyms from a `SynonymDictionary`.

### Token

```php
$token->getText()               // string  -- The token text
$token->getPosition()           // int     -- Position in the token stream
$token->getStartOffset()        // int     -- Character start offset in the original text
$token->getEndOffset()          // int     -- Character end offset in the original text
$token->getBoost()              // float   -- Score boost factor (1.0 = no adjustment)
$token->isStopped()             // bool    -- Whether removed by a stop filter
$token->getPositionIncrement()  // int     -- Difference from the previous token's position
$token->getPositionLength()     // int     -- Number of positions spanned
```

---

## Field value types

PHP values are automatically converted to Laurus `DataValue` types:

| PHP type | Laurus type | Notes |
| :--- | :--- | :--- |
| `null` | `Null` | |
| `true` / `false` | `Bool` | |
| `int` | `Int64` | |
| `float` | `Float64` | |
| `string` | `Text` | |
| `array` of numerics | `Vector` | Elements coerced to `f32` |
| `array` with `"lat"`, `"lon"` | `Geo` | Two `float` values |
| `string` (ISO 8601) | `DateTime` | Parsed from ISO 8601 format |
