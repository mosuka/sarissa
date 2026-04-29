# API Reference

## Index

The primary entry point. Wraps the Laurus search engine.

```ruby
Laurus::Index.new(path: nil, schema: nil)
```

### Constructor

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path:` | `String \| nil` | `nil` | Directory path for persistent storage. `nil` creates an in-memory index. |
| `schema:` | `Schema \| nil` | `nil` | Schema definition. An empty schema is used when omitted. |

### Methods

| Method | Description |
| :--- | :--- |
| `put_document(id, doc)` | Upsert a document. Replaces all existing versions with the same ID. |
| `add_document(id, doc)` | Append a document chunk without removing existing versions. |
| `get_documents(id) -> Array<Hash>` | Return all stored versions for the given ID. |
| `delete_documents(id)` | Delete all versions for the given ID. |
| `commit` | Flush buffered writes and make all pending changes searchable. |
| `search(query, limit: 10, offset: 0) -> Array<SearchResult>` | Execute a search query. |
| `stats -> Hash` | Return index statistics (`"document_count"`, `"vector_fields"`). |

### `search` query argument

The `query` parameter accepts any of the following:

- A **DSL string** (e.g. `"title:hello"`, `"embedding:\"memory safety\""`)
- A **lexical query object** (`TermQuery`, `PhraseQuery`, `BooleanQuery`, ...)
- A **vector query object** (`VectorQuery`, `VectorTextQuery`)
- A **`SearchRequest`** for full control

---

## Schema

Defines the fields and index types for an `Index`.

```ruby
Laurus::Schema.new
```

### Field methods

| Method | Description |
| :--- | :--- |
| `add_text_field(name, stored: true, indexed: true, term_vectors: false, analyzer: nil)` | Full-text field (inverted index, BM25). |
| `add_integer_field(name, stored: true, indexed: true, multi_valued: false)` | 64-bit integer field. Pass `multi_valued: true` to accept arrays of integers (range queries match if any value satisfies the predicate). |
| `add_float_field(name, stored: true, indexed: true, multi_valued: false)` | 64-bit float field. Pass `multi_valued: true` to accept arrays of floats (range queries match if any value satisfies the predicate). |
| `add_boolean_field(name, stored: true, indexed: true)` | Boolean field. |
| `add_bytes_field(name, stored: true)` | Raw bytes field. |
| `add_geo_field(name, stored: true, indexed: true)` | Geographic coordinate field (lat/lon). |
| `add_geo3d_field(name, stored: true, indexed: true)` | 3D ECEF Cartesian point field (x, y, z in metres). See [Geo3d concepts](../concepts/geo3d.md). |
| `add_datetime_field(name, stored: true, indexed: true)` | UTC datetime field. |
| `add_hnsw_field(name, dimension, distance: "cosine", m: 16, ef_construction: 200, embedder: nil)` | HNSW approximate nearest-neighbor vector field. |
| `add_flat_field(name, dimension, distance: "cosine", embedder: nil)` | Flat (brute-force) vector field. |
| `add_ivf_field(name, dimension, distance: "cosine", n_clusters: 100, n_probe: 1, embedder: nil)` | IVF approximate nearest-neighbor vector field. |

### Other methods

| Method | Description |
| :--- | :--- |
| `add_embedder(name, config)` | Register a named embedder definition. `config` is a Hash with a `"type"` key (see below). |
| `set_default_fields(fields)` | Set the default fields used when no field is specified in a query. `fields` is an Array of Strings. |
| `set_dynamic_field_policy(policy)` | Set how undeclared fields are handled. `policy` is `"strict"`, `"dynamic"` (default), or `"ignore"`. See notes below. |
| `dynamic_field_policy -> String` | Return the current policy as a lowercase string. |
| `field_names -> Array<String>` | Return the list of field names defined in this schema. |

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

```ruby
Laurus::TermQuery.new(field, term)
```

Matches documents containing the exact term in the given field.

### PhraseQuery

```ruby
Laurus::PhraseQuery.new(field, terms)
```

Matches documents containing the terms in order. `terms` is an Array of Strings.

### FuzzyQuery

```ruby
Laurus::FuzzyQuery.new(field, term, max_edits: 2)
```

Approximate match allowing up to `max_edits` edit-distance errors.

### WildcardQuery

```ruby
Laurus::WildcardQuery.new(field, pattern)
```

Pattern match. `*` matches any sequence of characters, `?` matches any single character.

### NumericRangeQuery

```ruby
Laurus::NumericRangeQuery.new(field, min: nil, max: nil)
```

Matches numeric values in the range `[min, max]`. Pass `nil` for an open bound. The type (integer or float) is inferred from the Ruby type of `min`/`max`.

### GeoDistanceQuery

```ruby
Laurus::GeoDistanceQuery.within_radius(field, lat, lon, distance_km)
```

Geo-distance (radius) search. Returns documents whose `(lat, lon)` coordinate
is within `distance_km` kilometres of the given point.

### GeoBoundingBoxQuery

```ruby
Laurus::GeoBoundingBoxQuery.within_bounding_box(
  field, min_lat, min_lon, max_lat, max_lon,
)
```

Geo bounding-box search. Returns documents whose `(lat, lon)` coordinate lies
inside the axis-aligned `[min_lat, max_lat] × [min_lon, max_lon]` rectangle.

### Geo3dDistanceQuery

```ruby
Laurus::Geo3dDistanceQuery.within_sphere(field, x, y, z, radius_m)
```

Sphere search over a 3D ECEF point field. Returns documents whose `(x, y, z)`
coordinate is within `radius_m` metres of the centre. See
[Geo3d concepts](../concepts/geo3d.md) for ECEF theory.

### Geo3dBoundingBoxQuery

```ruby
Laurus::Geo3dBoundingBoxQuery.within_box(
  field,
  min_x, min_y, min_z,
  max_x, max_y, max_z,
)
```

Axis-aligned 3D bounding-box search.

### Geo3dNearestQuery

```ruby
Laurus::Geo3dNearestQuery.k_nearest(
  field, x, y, z, k,
  initial_radius_m: nil,
  max_radius_m: nil,
)
```

k-nearest-neighbour search over a 3D ECEF point field. The optional
`initial_radius_m:` and `max_radius_m:` keyword arguments tune the
iterative-expansion search cone.

### BooleanQuery

```ruby
bq = Laurus::BooleanQuery.new
bq.must(query)
bq.should(query)
bq.must_not(query)
```

Compound boolean query. `must` clauses all have to match; `must_not` clauses must not match. `should` clauses contribute to scoring; at least one of them must match if there are no `must` clauses.

### SpanQuery

```ruby
# Single term
Laurus::SpanQuery.term(field, term)

# Near: terms within slop positions
Laurus::SpanQuery.near(field, terms, slop: 0, ordered: true)

# Near with nested SpanQuery clauses
Laurus::SpanQuery.near_spans(field, clauses, slop: 0, ordered: true)

# Containing: big span contains little span
Laurus::SpanQuery.containing(field, big, little)

# Within: include span within exclude span at max distance
Laurus::SpanQuery.within(field, include_span, exclude_span, distance)
```

Positional / proximity span queries. `near` takes an Array of term Strings, while `near_spans` takes an Array of `SpanQuery` objects for nested expressions.

### VectorQuery

```ruby
Laurus::VectorQuery.new(field, vector)
```

Approximate nearest-neighbor search using a pre-computed embedding vector. `vector` is an Array of Floats.

### VectorTextQuery

```ruby
Laurus::VectorTextQuery.new(field, text)
```

Converts `text` to an embedding at query time and runs vector search. Requires an embedder configured on the index.

---

## SearchRequest

Full-featured search request for advanced control.

```ruby
Laurus::SearchRequest.new(
  query: nil,
  lexical_query: nil,
  vector_query: nil,
  filter_query: nil,
  fusion: nil,
  limit: 10,
  offset: 0,
)
```

| Parameter | Description |
| :--- | :--- |
| `query:` | A DSL string or single query object. Mutually exclusive with `lexical_query:` / `vector_query:`. |
| `lexical_query:` | Lexical component for explicit hybrid search. |
| `vector_query:` | Vector component for explicit hybrid search. |
| `filter_query:` | Lexical filter applied after scoring. |
| `fusion:` | Fusion algorithm (`RRF` or `WeightedSum`). Defaults to `RRF(k: 60)` when both components are set. |
| `limit:` | Maximum number of results (default 10). |
| `offset:` | Pagination offset (default 0). |

---

## SearchResult

Returned by `Index#search`.

```ruby
result.id        # => String   -- External document identifier
result.score     # => Float    -- Relevance score
result.document  # => Hash|nil -- Retrieved field values, or nil if deleted
```

---

## Fusion algorithms

### RRF

```ruby
Laurus::RRF.new(k: 60.0)
```

Reciprocal Rank Fusion. Merges lexical and vector result lists by rank position. `k` is a smoothing constant; higher values reduce the influence of top-ranked results.

### WeightedSum

```ruby
Laurus::WeightedSum.new(lexical_weight: 0.5, vector_weight: 0.5)
```

Normalises both score lists independently, then combines them as `lexical_weight * lexical_score + vector_weight * vector_score`.

---

## Text analysis

### SynonymDictionary

```ruby
dict = Laurus::SynonymDictionary.new
dict.add_synonym_group(["fast", "quick", "rapid"])
```

A dictionary of synonym groups. All terms in a group are treated as synonyms of each other.

### WhitespaceTokenizer

```ruby
tokenizer = Laurus::WhitespaceTokenizer.new
tokens = tokenizer.tokenize("hello world")
```

Splits text on whitespace boundaries and returns an Array of `Token` objects.

### SynonymGraphFilter

```ruby
filter = Laurus::SynonymGraphFilter.new(dictionary, keep_original: true, boost: 1.0)
expanded = filter.apply(tokens)
```

Token filter that expands tokens with their synonyms from a `SynonymDictionary`.

### Token

```ruby
token.text                # => String  -- The token text
token.position            # => Integer -- Position in the token stream
token.start_offset        # => Integer -- Character start offset in the original text
token.end_offset          # => Integer -- Character end offset in the original text
token.boost               # => Float   -- Score boost factor (1.0 = no adjustment)
token.stopped             # => Boolean -- Whether removed by a stop filter
token.position_increment  # => Integer -- Difference from the previous token's position
token.position_length     # => Integer -- Number of positions spanned
```

---

## Field value types

Ruby values are automatically converted to Laurus `DataValue` types:

| Ruby type | Laurus type | Notes |
| :--- | :--- | :--- |
| `nil` | `Null` | |
| `true` / `false` | `Bool` | |
| `Integer` | `Int64` | |
| `Float` | `Float64` | |
| `String` | `Text` | |
| `Array` of numerics | `Vector` | Elements coerced to `f32` |
| `Hash` with `"lat"`, `"lon"` | `Geo` | Two `Float` values |
| `Hash` with `"x"`, `"y"`, `"z"` | `GeoEcef` | Three `Float` values, meters (3D ECEF Cartesian) |
| `Time` / `String` responding to `iso8601` | `DateTime` | Converted via `iso8601` |
