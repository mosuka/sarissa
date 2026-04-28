# API Reference

## Index

The primary entry point. Wraps the Laurus search engine.

```python
class Index:
    def __init__(self, path: str | None = None, schema: Schema | None = None) -> None: ...
```

### Constructor

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str \| None` | `None` | Directory path for persistent storage. `None` creates an in-memory index. |
| `schema` | `Schema \| None` | `None` | Schema definition. An empty schema is used when omitted. |

### Methods

| Method | Description |
| :--- | :--- |
| `put_document(id, doc)` | Upsert a document. Replaces all existing versions with the same ID. |
| `add_document(id, doc)` | Append a document chunk without removing existing versions. |
| `get_documents(id) -> list[dict]` | Return all stored versions for the given ID. |
| `delete_documents(id)` | Delete all versions for the given ID. |
| `commit()` | Flush buffered writes and make all pending changes searchable. |
| `search(query, *, limit=10, offset=0) -> list[SearchResult]` | Execute a search query. |
| `stats() -> dict` | Return index statistics (`document_count`, `vector_fields`). |

### `search` query argument

The `query` parameter accepts any of the following:

- A **DSL string** (e.g. `"title:hello"`, `"embedding:\"memory safety\""`)
- A **lexical query object** (`TermQuery`, `PhraseQuery`, `BooleanQuery`, …)
- A **vector query object** (`VectorQuery`, `VectorTextQuery`)
- A **`SearchRequest`** for full control

---

## Schema

Defines the fields and index types for an `Index`.

```python
class Schema:
    def __init__(self) -> None: ...
```

### Field methods

| Method | Description |
| :--- | :--- |
| `add_text_field(name)` | Full-text field (inverted index, BM25). |
| `add_integer_field(name, *, stored=True, indexed=True, multi_valued=False)` | 64-bit integer field. Set `multi_valued=True` to accept arrays of integers (range queries match if any value satisfies the predicate). |
| `add_float_field(name, *, stored=True, indexed=True, multi_valued=False)` | 64-bit float field. Set `multi_valued=True` to accept arrays of floats (range queries match if any value satisfies the predicate). |
| `add_bool_field(name)` | Boolean field. |
| `add_bytes_field(name)` | Raw bytes field. |
| `add_geo_field(name)` | Geographic coordinate field (lat/lon). |
| `add_geo3d_field(name, *, stored=True, indexed=True)` | 3D ECEF Cartesian point field (x, y, z in metres). See [Geo3d concepts](../concepts/geo3d.md). |
| `add_datetime_field(name)` | UTC datetime field. |
| `add_hnsw_field(name, dimension, *, distance="cosine", m=16, ef_construction=100)` | HNSW approximate nearest-neighbor vector field. |
| `add_flat_field(name, dimension, *, distance="cosine")` | Flat (brute-force) vector field. |
| `add_ivf_field(name, dimension, *, distance="cosine", n_clusters=100, n_probe=1)` | IVF approximate nearest-neighbor vector field. |
| `set_default_fields(fields)` | Set default search fields (list of strings). |
| `set_dynamic_field_policy(policy)` | Set how undeclared fields are handled. `policy` is `"strict"`, `"dynamic"` (default), or `"ignore"`. See notes below. |
| `dynamic_field_policy()` | Return the current policy as a lowercase string. |
| `field_names()` | Return all field names. |

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

---

## Query classes

### TermQuery

```python
TermQuery(field: str, term: str)
```

Matches documents containing the exact term in the given field.

### PhraseQuery

```python
PhraseQuery(field: str, terms: list[str])
```

Matches documents containing the terms in order.

### FuzzyQuery

```python
FuzzyQuery(field: str, term: str, max_edits: int = 1)
```

Approximate match allowing up to `max_edits` edit-distance errors.

### WildcardQuery

```python
WildcardQuery(field: str, pattern: str)
```

Pattern match. `*` matches any sequence of characters, `?` matches any single character.

### NumericRangeQuery

```python
NumericRangeQuery(field: str, min: int | float | None, max: int | float | None)
```

Matches numeric values in the range `[min, max]`. Pass `None` for an open bound.

### GeoDistanceQuery

```python
GeoDistanceQuery.within_radius(
    field: str, lat: float, lon: float, distance_km: float,
)
```

Geo-distance (radius) search. Returns documents whose `(lat, lon)` coordinate
is within `distance_km` kilometres of the given point.

### GeoBoundingBoxQuery

```python
GeoBoundingBoxQuery.within_bounding_box(
    field: str,
    min_lat: float, min_lon: float,
    max_lat: float, max_lon: float,
)
```

Geo bounding-box search. Returns documents whose `(lat, lon)` coordinate lies
inside the axis-aligned `[min_lat, max_lat] × [min_lon, max_lon]` rectangle.

### Geo3dDistanceQuery

```python
Geo3dDistanceQuery.within_sphere(
    field: str, x: float, y: float, z: float, radius_m: float,
)
```

Sphere search over a 3D ECEF point field. Returns documents whose `(x, y, z)`
coordinate is within `radius_m` metres of the centre. See
[Geo3d concepts](../concepts/geo3d.md) for ECEF theory.

### Geo3dBoundingBoxQuery

```python
Geo3dBoundingBoxQuery.within_box(
    field: str,
    min_x: float, min_y: float, min_z: float,
    max_x: float, max_y: float, max_z: float,
)
```

Axis-aligned 3D bounding-box search. Returns documents whose ECEF point lies
inside `[min_x, max_x] × [min_y, max_y] × [min_z, max_z]`.

### Geo3dNearestQuery

```python
Geo3dNearestQuery.k_nearest(
    field: str,
    x: float, y: float, z: float,
    k: int,
    *,
    initial_radius_m: float | None = None,
    max_radius_m: float | None = None,
)
```

k-nearest-neighbour search over a 3D ECEF point field. Returns the `k`
documents closest to `(x, y, z)`. The optional `initial_radius_m` and
`max_radius_m` tune the iterative-expansion search cone.

### BooleanQuery

```python
BooleanQuery(
    must: list[Query] | None = None,
    should: list[Query] | None = None,
    must_not: list[Query] | None = None,
)
```

Compound boolean query. `must` clauses all have to match; at least one `should` clause must match; `must_not` clauses must not match.

### SpanNearQuery

```python
SpanNearQuery(field: str, terms: list[str], slop: int = 0, in_order: bool = True)
```

Matches documents where the terms appear within `slop` positions of each other.

### VectorQuery

```python
VectorQuery(field: str, vector: list[float])
```

Approximate nearest-neighbor search using a pre-computed embedding vector.

### VectorTextQuery

```python
VectorTextQuery(field: str, text: str)
```

Converts `text` to an embedding at query time and runs vector search. Requires an embedder configured on the index.

---

## SearchRequest

Full-featured search request for advanced control.

```python
class SearchRequest:
    def __init__(
        self,
        *,
        query=None,
        lexical_query=None,
        vector_query=None,
        filter_query=None,
        fusion=None,
        limit: int = 10,
        offset: int = 0,
    ) -> None: ...
```

| Parameter | Description |
| :--- | :--- |
| `query` | A DSL string or single query object. Mutually exclusive with `lexical_query` / `vector_query`. |
| `lexical_query` | Lexical component for explicit hybrid search. |
| `vector_query` | Vector component for explicit hybrid search. |
| `filter_query` | Lexical filter applied after scoring. |
| `fusion` | Fusion algorithm (`RRF` or `WeightedSum`). Defaults to `RRF(k=60)` when both components are set. |
| `limit` | Maximum number of results (default 10). |
| `offset` | Pagination offset (default 0). |

---

## SearchResult

Returned by `Index.search()`.

```python
class SearchResult:
    id: str          # External document identifier
    score: float     # Relevance score
    document: dict | None  # Retrieved field values, or None if deleted
```

---

## Fusion algorithms

### RRF

```python
RRF(k: float = 60.0)
```

Reciprocal Rank Fusion. Merges lexical and vector result lists by rank position. `k` is a smoothing constant; higher values reduce the influence of top-ranked results.

### WeightedSum

```python
WeightedSum(lexical_weight: float = 0.5, vector_weight: float = 0.5)
```

Normalises both score lists independently, then combines them as `lexical_weight * lexical_score + vector_weight * vector_score`.

---

## Text analysis

### SynonymDictionary

```python
class SynonymDictionary:
    def __init__(self) -> None: ...
    def add_synonym_group(self, synonyms: list[str]) -> None: ...
```

### WhitespaceTokenizer

```python
class WhitespaceTokenizer:
    def __init__(self) -> None: ...
    def tokenize(self, text: str) -> list[Token]: ...
```

### SynonymGraphFilter

```python
class SynonymGraphFilter:
    def __init__(
        self,
        dictionary: SynonymDictionary,
        keep_original: bool = True,
        boost: float = 1.0,
    ) -> None: ...
    def apply(self, tokens: list[Token]) -> list[Token]: ...
```

### Token

```python
class Token:
    text: str
    position: int
    position_increment: int
    position_length: int
    boost: float
```

---

## Field value types

Python values are automatically converted to Laurus `DataValue` types:

| Python type | Laurus type | Notes |
| :--- | :--- | :--- |
| `None` | `Null` | |
| `bool` | `Bool` | Checked before `int` |
| `int` | `Int64` | |
| `float` | `Float64` | |
| `str` | `Text` | |
| `bytes` | `Bytes` | |
| `list[float]` | `Vector` | Elements coerced to `f32` |
| `(lat, lon)` tuple | `Geo` | Two `float` values |
| `datetime.datetime` | `DateTime` | Converted via `isoformat()` |
