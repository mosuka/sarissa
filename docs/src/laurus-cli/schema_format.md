# Schema Format Reference

The schema file defines the structure of your index — what fields exist, their types, and how they are indexed. Laurus uses TOML format for schema files.

## Overview

A schema consists of three top-level elements:

```toml
# Policy for fields not declared below. Optional — defaults to "dynamic".
dynamic_field_policy = "dynamic"

# Fields to search by default when a query does not specify a field.
default_fields = ["title", "body"]

# Field definitions. Each field has a name and a typed configuration.
[fields.<field_name>.<FieldType>]
# ... type-specific options
```

- **`dynamic_field_policy`** — How the engine treats fields present in an ingested document but **absent** from this schema. Accepted values: `"strict"`, `"dynamic"`, `"ignore"`. Defaults to `"dynamic"`. See [Dynamic Schema](../concepts/schema_and_fields.md#dynamic-schema) for the full semantics and the warning about silent truncation under `"dynamic"`.
- **`default_fields`** — A list of field names used as default search targets by the [Query DSL](../concepts/query_dsl.md). Only lexical fields (Text, Integer, Float, etc.) can be default fields. This key is optional and defaults to an empty list.
- **`fields`** — A map of field names to their typed configuration. Each field must specify exactly one field type.

## Field Naming

- Field names are arbitrary strings (e.g., `title`, `body_vec`, `created_at`).
- **Field names starting with `_` are reserved** for the engine. The only allow-listed name is `_id` (managed automatically). Attempting to declare any other `_`-prefixed field results in an error.
- Field names must be unique within a schema.

## Field Types

Fields fall into two categories: **Lexical** (for keyword/full-text search) and **Vector** (for similarity search). A single field cannot be both.

### Lexical Fields

#### Text

Full-text searchable field. Text is processed by the analysis pipeline (tokenization, normalization, stemming, etc.).

```toml
[fields.title.Text]
indexed = true       # Whether to index this field for search
stored = true        # Whether to store the original value for retrieval
term_vectors = false # Whether to store term positions (for phrase queries, highlighting)
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables searching this field |
| `stored` | `bool` | `true` | Stores the original value so it can be returned in results |
| `term_vectors` | `bool` | `true` | Stores term positions for phrase queries, highlighting, and more-like-this |

#### Integer

64-bit signed integer field. Supports range queries and exact match.

```toml
[fields.year.Integer]
indexed = true
stored = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables range and exact-match queries |
| `stored` | `bool` | `true` | Stores the original value |

#### Float

64-bit floating point field. Supports range queries.

```toml
[fields.rating.Float]
indexed = true
stored = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables range queries |
| `stored` | `bool` | `true` | Stores the original value |

#### Boolean

Boolean field (`true` / `false`).

```toml
[fields.published.Boolean]
indexed = true
stored = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables filtering by boolean value |
| `stored` | `bool` | `true` | Stores the original value |

#### DateTime

UTC timestamp field. Supports range queries.

```toml
[fields.created_at.DateTime]
indexed = true
stored = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables range queries on date/time |
| `stored` | `bool` | `true` | Stores the original value |

#### Geo

Geographic point field (latitude/longitude). Supports radius and bounding box queries.

```toml
[fields.location.Geo]
indexed = true
stored = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables geo queries (radius, bounding box) |
| `stored` | `bool` | `true` | Stores the original value |

#### Bytes

Raw binary data field. Not indexed — stored only.

```toml
[fields.thumbnail.Bytes]
stored = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `stored` | `bool` | `true` | Stores the binary data |

### Vector Fields

Vector fields are indexed for approximate nearest neighbor (ANN) search. They require a `dimension` (the length of each vector) and a `distance` metric.

#### Hnsw

Hierarchical Navigable Small World graph index. Best for most use cases — offers a good balance of speed and recall.

```toml
[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
m = 16
ef_construction = 200
base_weight = 1.0
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | `128` | Vector dimensionality (must match your embedding model) |
| `distance` | `string` | `"Cosine"` | Distance metric (see [Distance Metrics](#distance-metrics)) |
| `m` | `integer` | `16` | Max bi-directional connections per node. Higher = better recall, more memory |
| `ef_construction` | `integer` | `200` | Search width during index construction. Higher = better quality, slower build |
| `base_weight` | `float` | `1.0` | Scoring weight in hybrid search fusion |
| `quantizer` | `object` | *none* | Optional quantization method (see [Quantization](#quantization)) |

**Tuning guidelines:**

- `m`: 12–48 is typical. Use higher values for higher-dimensional vectors.
- `ef_construction`: 100–500. Higher values produce a better graph but increase build time.
- `dimension`: Must exactly match the output dimension of your embedding model (e.g., 384 for `all-MiniLM-L6-v2`, 768 for `BERT-base`, 1536 for `text-embedding-3-small`).

#### Flat

Brute-force linear scan index. Provides exact results with no approximation. Best for small datasets (< 10,000 vectors).

```toml
[fields.embedding.Flat]
dimension = 384
distance = "Cosine"
base_weight = 1.0
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | `128` | Vector dimensionality |
| `distance` | `string` | `"Cosine"` | Distance metric (see [Distance Metrics](#distance-metrics)) |
| `base_weight` | `float` | `1.0` | Scoring weight in hybrid search fusion |
| `quantizer` | `object` | *none* | Optional quantization method (see [Quantization](#quantization)) |

#### Ivf

Inverted File Index. Clusters vectors and searches only a subset of clusters. Suitable for very large datasets.

```toml
[fields.embedding.Ivf]
dimension = 384
distance = "Cosine"
n_clusters = 100
n_probe = 1
base_weight = 1.0
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | *(required)* | Vector dimensionality |
| `distance` | `string` | `"Cosine"` | Distance metric (see [Distance Metrics](#distance-metrics)) |
| `n_clusters` | `integer` | `100` | Number of clusters. More clusters = finer partitioning |
| `n_probe` | `integer` | `1` | Number of clusters to search at query time. Higher = better recall, slower |
| `base_weight` | `float` | `1.0` | Scoring weight in hybrid search fusion |
| `quantizer` | `object` | *none* | Optional quantization method (see [Quantization](#quantization)) |

> **Note:** Unlike Hnsw and Flat, the `dimension` field in Ivf is **required** and has no default value.

**Tuning guidelines:**

- `n_clusters`: A common heuristic is `sqrt(N)` where N is the total number of vectors.
- `n_probe`: Start with 1 and increase until recall is acceptable. Typical range is 1–20.

## Distance Metrics

The `distance` option for vector fields accepts the following values:

| Value | Description | Use When |
| :--- | :--- | :--- |
| `"Cosine"` | Cosine distance (1 - cosine similarity). Default. | Normalized text/image embeddings |
| `"Euclidean"` | L2 (Euclidean) distance | Spatial data, non-normalized vectors |
| `"Manhattan"` | L1 (Manhattan) distance | Sparse feature vectors |
| `"DotProduct"` | Dot product (higher = more similar) | Pre-normalized vectors where magnitude matters |
| `"Angular"` | Angular distance | Similar to cosine, but based on angle |

For most embedding models (BERT, Sentence Transformers, OpenAI, etc.), `"Cosine"` is the correct choice.

## Quantization

Vector fields optionally support quantization to reduce memory usage at the cost of some accuracy. Specify the `quantizer` option as a TOML table.

### None (default)

No quantization — full precision 32-bit floats.

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"
# quantizer is omitted (no quantization)
```

### Scalar 8-bit

Compresses each float32 component to uint8 (~4x memory reduction).

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"
quantizer = "Scalar8Bit"
```

### Product Quantization

Splits the vector into subvectors and quantizes each independently.

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"

[fields.embedding.Hnsw.quantizer.ProductQuantization]
subvector_count = 48
```

| Option | Type | Description |
| :--- | :--- | :--- |
| `subvector_count` | `integer` | Number of subvectors. Must evenly divide `dimension`. |

## Complete Examples

### Full-text search only

A simple blog post index with lexical search:

```toml
default_fields = ["title", "body"]

[fields.title.Text]
indexed = true
stored = true
term_vectors = false

[fields.body.Text]
indexed = true
stored = true
term_vectors = false

[fields.category.Text]
indexed = true
stored = true
term_vectors = false

[fields.published_at.DateTime]
indexed = true
stored = true
```

### Vector search only

A vector-only index for semantic similarity:

```toml
[fields.embedding.Hnsw]
dimension = 768
distance = "Cosine"
m = 16
ef_construction = 200
```

### Hybrid search (lexical + vector)

Combine lexical and vector search for best-of-both-worlds retrieval:

```toml
default_fields = ["title", "body"]

[fields.title.Text]
indexed = true
stored = true
term_vectors = false

[fields.body.Text]
indexed = true
stored = true
term_vectors = true

[fields.category.Text]
indexed = true
stored = true
term_vectors = false

[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
m = 16
ef_construction = 200
```

> **Tip:** A single field cannot be both lexical and vector. Use separate fields (e.g., `body` for text, `body_vec` for embedding) and map them both to the same source content.

### E-commerce product index

A more complex schema with mixed field types:

```toml
default_fields = ["name", "description"]

[fields.name.Text]
indexed = true
stored = true
term_vectors = false

[fields.description.Text]
indexed = true
stored = true
term_vectors = true

[fields.price.Float]
indexed = true
stored = true

[fields.in_stock.Boolean]
indexed = true
stored = true

[fields.created_at.DateTime]
indexed = true
stored = true

[fields.location.Geo]
indexed = true
stored = true

[fields.description_vec.Hnsw]
dimension = 384
distance = "Cosine"
```

## Generating a Schema

You can generate a schema TOML file interactively using the CLI:

```bash
laurus create schema
laurus create schema --output my_schema.toml
```

See [`create schema`](commands.md#create-schema) for details.

## Using a Schema

Once you have a schema file, create an index from it:

```bash
laurus create index --schema schema.toml
```

Or load it programmatically in Rust:

```rust
use laurus::Schema;

let toml_str = std::fs::read_to_string("schema.toml")?;
let schema: Schema = toml::from_str(&toml_str)?;
```
