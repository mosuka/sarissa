# Vector Indexing

Vector indexing powers similarity-based search. When a document's vector field is indexed, Laurus stores the embedding vector in a specialized index structure that enables fast approximate nearest neighbor (ANN) retrieval.

## How Vector Indexing Works

```mermaid
sequenceDiagram
    participant Doc as Document
    participant Embedder
    participant Normalize as Normalizer
    participant Index as Vector Index

    Doc->>Embedder: "Rust is a systems language"
    Embedder-->>Normalize: [0.12, -0.45, 0.78, ...]
    Normalize->>Normalize: L2 normalize
    Normalize-->>Index: [0.14, -0.52, 0.90, ...]
    Index->>Index: Insert into index structure
```

### Step by Step

1. **Embed**: The text (or image) is converted to a vector by the configured embedder
2. **Normalize**: The vector is L2-normalized (for cosine similarity)
3. **Index**: The vector is inserted into the configured index structure (Flat, HNSW, or IVF)
4. **Commit**: On `commit()`, the index is flushed to persistent storage

## Index Types

Laurus supports three vector index types, each with different performance characteristics:

### Comparison

| Property | Flat | HNSW | IVF |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 100% (exact) | ~95-99% (approximate) | ~90-98% (approximate) |
| **Search speed** | O(n) linear scan | O(log n) graph walk | O(n/k) cluster scan |
| **Memory usage** | Low | Higher (graph edges) | Moderate (centroids) |
| **Index build time** | Fast | Moderate | Slower (clustering) |
| **Best for** | < 10K vectors | 10K - 10M vectors | > 1M vectors |

### Flat Index

The simplest index. Compares the query vector against every stored vector (brute-force).

```rust
use laurus::vector::FlatOption;
use laurus::vector::core::distance::DistanceMetric;

let opt = FlatOption {
    dimension: 384,
    distance: DistanceMetric::Cosine,
    ..Default::default()
};
```

- **Pros**: 100% recall (exact results), simple, low memory
- **Cons**: Slow for large datasets (linear scan)
- **Use when**: You have fewer than ~10,000 vectors, or you need exact results

### HNSW Index

**Hierarchical Navigable Small World** graph. The default and most commonly used index type.

```mermaid
graph TB
    subgraph "Layer 2 (sparse)"
        A2["A"] --- C2["C"]
    end

    subgraph "Layer 1 (medium)"
        A1["A"] --- B1["B"]
        A1 --- C1["C"]
        B1 --- D1["D"]
        C1 --- D1
    end

    subgraph "Layer 0 (dense - all vectors)"
        A0["A"] --- B0["B"]
        A0 --- C0["C"]
        B0 --- D0["D"]
        B0 --- E0["E"]
        C0 --- D0
        C0 --- F0["F"]
        D0 --- E0
        E0 --- F0
    end

    A2 -.->|"entry point"| A1
    A1 -.-> A0
    C2 -.-> C1
    C1 -.-> C0
    B1 -.-> B0
    D1 -.-> D0
```

The HNSW algorithm searches from the top (sparse) layer down to the bottom (dense) layer, narrowing the search space at each level.

```rust
use laurus::vector::HnswOption;
use laurus::vector::core::distance::DistanceMetric;

let opt = HnswOption {
    dimension: 384,
    distance: DistanceMetric::Cosine,
    m: 16,                  // max connections per node per layer
    ef_construction: 200,   // search width during index building
    ..Default::default()
};
```

#### HNSW Parameters

| Parameter | Default | Description | Impact |
| :--- | :--- | :--- | :--- |
| `m` | 16 | Max bi-directional connections per layer | Higher = better recall, more memory |
| `ef_construction` | 200 | Search width during index building | Higher = better recall, slower build |
| `dimension` | 128 | Vector dimensions | Must match embedder output |
| `distance` | Cosine | Distance metric | See Distance Metrics below |

**Tuning tips:**

- Increase `m` (e.g., 32 or 64) for higher recall at the cost of memory
- Increase `ef_construction` (e.g., 400) for better index quality at the cost of build time
- At search time, the `ef_search` parameter (set in the search request) controls the search width

### IVF Index

**Inverted File Index**. Partitions vectors into clusters, then only searches relevant clusters.

```mermaid
graph TB
    Q["Query Vector"]
    Q --> C1["Cluster 1\n(centroid)"]
    Q --> C2["Cluster 2\n(centroid)"]

    C1 --> V1["vec_3"]
    C1 --> V2["vec_7"]
    C1 --> V3["vec_12"]

    C2 --> V4["vec_1"]
    C2 --> V5["vec_9"]
    C2 --> V6["vec_15"]

    style C1 fill:#f9f,stroke:#333
    style C2 fill:#f9f,stroke:#333
```

```rust
use laurus::vector::IvfOption;
use laurus::vector::core::distance::DistanceMetric;

let opt = IvfOption {
    dimension: 384,
    distance: DistanceMetric::Cosine,
    n_clusters: 100,   // number of clusters
    n_probe: 10,       // clusters to search at query time
    ..Default::default()
};
```

#### IVF Parameters

| Parameter | Default | Description | Impact |
| :--- | :--- | :--- | :--- |
| `n_clusters` | 100 | Number of Voronoi cells | More clusters = faster search, lower recall |
| `n_probe` | 1 | Clusters to search at query time | Higher = better recall, slower search |
| `dimension` | (required) | Vector dimensions | Must match embedder output |
| `distance` | Cosine | Distance metric | See Distance Metrics below |

**Tuning tips:**

- Set `n_clusters` to roughly `sqrt(n)` where `n` is the number of vectors
- Set `n_probe` to 5-20% of `n_clusters` for a good recall/speed trade-off
- IVF requires a training phase — initial indexing may be slower

## Distance Metrics

| Metric | Description | Range | Best For |
| :--- | :--- | :--- | :--- |
| `Cosine` | 1 - cosine similarity | [0, 2] | Text embeddings (most common) |
| `Euclidean` | L2 distance | [0, +inf) | Spatial data |
| `Manhattan` | L1 distance | [0, +inf) | Feature vectors |
| `DotProduct` | Negative inner product | (-inf, +inf) | Pre-normalized vectors |
| `Angular` | Angular distance | [0, pi] | Directional similarity |

```rust
use laurus::vector::core::distance::DistanceMetric;

let metric = DistanceMetric::Cosine;      // Default for text
let metric = DistanceMetric::Euclidean;    // For spatial data
let metric = DistanceMetric::Manhattan;    // L1 distance
let metric = DistanceMetric::DotProduct;   // For pre-normalized vectors
let metric = DistanceMetric::Angular;      // Angular distance
```

> **Note:** For cosine similarity, vectors are automatically L2-normalized before indexing. Lower distance = more similar.

## Quantization

Vectors are stored on disk as **8-bit scalar-quantized integers**
(Issue #481 Stage 1). Compared to the previous 32-bit float storage
this is **~4x smaller** with negligible recall loss in practice
(Recall@10 remains ≥ 0.95 against the f32 ground truth — see the
recall test at `laurus/tests/vector_recall_test.rs`).

| Method | Enum Variant | Description | Memory Reduction |
| :--- | :--- | :--- | :--- |
| **Scalar 8-bit** *(default)* | `Scalar8Bit` | Per-segment global affine quantization to `u8` | ~4x |
| **Product Quantization** *(reserved)* | `ProductQuantization { subvector_count }` | Stage 3 of #481 — currently `NotImplemented` | ~16-64x |

```rust
use laurus::vector::HnswOption;
use laurus::vector::core::quantization::QuantizationMethod;

// `quantizer` defaults to `Scalar8Bit`; the explicit form below is
// equivalent to `HnswOption { dimension: 384, ..Default::default() }`.
let opt = HnswOption {
    dimension: 384,
    quantizer: QuantizationMethod::Scalar8Bit,
    ..Default::default()
};
```

> **Breaking change (Issue #481 Stage 1):** the `quantizer` field is no
> longer `Option<QuantizationMethod>`; quantization is mandatory and
> defaults to `Scalar8Bit`. There is no longer an unquantized (f32)
> on-disk format. Existing pre-Stage-1 vector indexes are intentionally
> not readable by this version — rebuild the index from source data.

### How Scalar8Bit works

- Each segment trains a single **global** `(offset, scale)` pair from
  its f32 vectors at flush time (`offset = min`, `scale = (max - min) / 255`).
- Each `f32` element is encoded as `u8 = clamp(round((v - offset) / scale), 0, 255)`.
- Per-vector metadata (`sum_q: u32`, `norm_q: f32`) is precomputed and
  persisted alongside the int8 payload so the cosine search hot loop
  collapses to one int8 SIMD multiply-accumulate plus three scalar
  corrections — no per-element dequantization at search time.
- Segment files start with the `LVS1` magic + a 16-byte header so the
  reader can detect the format at load time.

### Two-stage rerank (Issue #481 Stage 2 — reserved)

`VectorIndexQueryParams.rerank_factor: Option<usize>` and the gRPC /
JSON gateway counterpart `VectorParams.rerank_factor` are reserved for
the upcoming two-stage rerank flow:

1. The quantized HNSW search returns `top_k * rerank_factor` candidates.
2. Those candidates are re-scored with the full f32 vectors before the
   final `top_k` is returned.

Setting `rerank_factor` today returns `LaurusError::NotImplemented` /
`tonic::Status::Unimplemented`. The API surface is reserved so existing
callers can opt in once Stage 2 lands without a further proto / binding
schema bump.

## Segment Files

Each vector index type stores its data in a single segment file:

| Index Type | File Extension | Contents |
| :--- | :--- | :--- |
| HNSW | `.hnsw` | Graph structure, vectors, and metadata |
| Flat | `.flat` | Raw vectors and metadata |
| IVF | `.ivf` | Cluster centroids, assigned vectors, and metadata |

## Code Example

```rust
use std::sync::Arc;
use laurus::{Document, Engine, Schema};
use laurus::lexical::TextOption;
use laurus::vector::HnswOption;
use laurus::vector::core::distance::DistanceMetric;
use laurus::storage::memory::MemoryStorage;

#[tokio::main]
async fn main() -> laurus::Result<()> {
    let storage = Arc::new(MemoryStorage::new(Default::default()));
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_hnsw_field("embedding", HnswOption {
            dimension: 384,
            distance: DistanceMetric::Cosine,
            m: 16,
            ef_construction: 200,
            ..Default::default()
        })
        .build();

    // With an embedder, text in vector fields is automatically embedded
    let engine = Engine::builder(storage, schema)
        .embedder(my_embedder)
        .build()
        .await?;

    // Add text to the vector field — it will be embedded automatically
    engine.add_document("doc-1", Document::builder()
        .add_text("title", "Rust Programming")
        .add_text("embedding", "Rust is a systems programming language.")
        .build()
    ).await?;

    engine.commit().await?;

    Ok(())
}
```

## Next Steps

- Search the vector index: [Vector Search](../search/vector_search.md)
- Combine with lexical search: [Hybrid Search](../search/hybrid_search.md)
