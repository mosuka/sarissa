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
| **Product Quantization** | `ProductQuantization { subvector_count }` | Stage 3 of #481 — per-segment codebook of `M × 256` centroids, each vector stored as `M` bytes | ~16-64x (HNSW-only) |

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

### Two-stage rerank (Issue #481 Stage 2)

Stage 1 stores vectors as int8 only. The graph search runs entirely
against int8 distances, which is fast but introduces a small
quantization error. Stage 2 adds an **optional** per-field f32
sidecar so the searcher can rescore the top candidates against the
original full-precision vectors:

1. The HNSW int8 graph search returns up to `ef_search` candidates
   ranked by quantized cosine distance.
2. The top `top_k * rerank_factor` candidates are rescored against
   the f32 vectors loaded from the
   [LRS1 sidecar](#lrs1-rerank-sidecar) (`*.hnsw.f32`).
3. The new ranking is truncated to `top_k` and returned.

Stage 2 is opt-in per field via
[`HnswOption.rerank_storage`](../../laurus-cli/schema_format.md#rerank-storage):

```rust
use laurus::vector::HnswOption;
use laurus::vector::core::rerank::RerankStorageKind;

let opt = HnswOption {
    rerank_storage: Some(RerankStorageKind::F32),
    ..HnswOption::default()
};
```

Queries pass the rerank factor through `VectorIndexQuery::rerank_factor`
(low-level), `SearchRequestBuilder::vector_rerank_factor` (engine), or
the gRPC / JSON `VectorParams.rerank_factor` field.

Fields without `rerank_storage` enabled silently fall back to the
Stage 1 int8 ranking even when `rerank_factor` is set — there is no
f32 information to recover from a Stage 1 segment.

#### LRS1 rerank sidecar

The sidecar is a separate file written next to the LVS1 segment when
`rerank_storage` is enabled:

```text
offset  size  field
------  ----  -------------------------------------------
     0     4  magic         ASCII "LRS1"
     4     2  version       u16 LE  (current = 1)
     6     2  storage_kind  u16 LE  (1 = F32; 0 reserved; 2.. future)
     8     8  reserved      zero-padded
    16     4  dim           u32 LE
    20     4  vector_count  u32 LE
    24     -  payload       vector_count * dim * bytes_per_element
   end     8  footer        magic "LRC1" u32 LE + CRC-32 u32 LE
                             over header + payload
```

Vectors are written in the same `(doc_id, field_name)` order as the
matching LVS1 segment, so a (sidecar position) → (LVS1 position)
mapping is the identity. The HNSW reader loads the sidecar into a
`RerankStoragePool` at init time when the storage loading mode is
Eager; Lazy mode skips the sidecar to honor its memory-savings
promise (Stage 2 segments opened in Lazy mode silently degrade to
Stage 1).

New sidecars end with an 8-byte CRC-32 footer over the header and
payload that is verified whenever the sidecar is read (both the
searcher load and the writer reload), so silent on-disk corruption is
rejected instead of skewing rerank scores. Because the header fully
determines the content length, the footer is detected by the bytes
remaining after the payload — sidecars written before the footer was
introduced have zero trailing bytes and still load, with verification
skipped.

#### Recall vs speed trade-off

`rerank_factor` lets you exchange a small per-query rerank cost
(~`top_k * rerank_factor` exact-distance calls — a few µs at dim
128) for higher Recall@10. The gain depends on the corpus and the
graph search budget (`ef_search`):

- Real clustered embedding data (text-embedding-3, BERT, etc.)
  reaches `Recall@10 ≥ 0.99` at low `ef_search`; rerank polishes
  the ranking with negligible latency overhead.
- Synthetic random unit-norm data (the worst case for HNSW recall
  recovery) needs a higher `ef_search` for the int8 graph to visit
  enough true top-10 candidates; rerank then re-orders the visited
  set but cannot retrieve candidates the graph never reached.

The recall acceptance is split into two CI gates so the rerank
kernel and the full HNSW pipeline can fail independently:

- `stage2_brute_force_rerank_recall_at_10_meets_kernel_gate` asserts
  `Recall@10 ≥ 0.99`. Bypasses the HNSW graph entirely (brute-force
  int8 over the corpus, widen to `top_k * rerank_factor`, rescore
  with f32) so any miss is a rerank-kernel regression.
- `hnsw_quantized_recall_at_10_with_rerank_meets_stage2_recall_gate`
  asserts `Recall@10 ≥ 0.98`. Adds the HNSW graph-construction
  non-determinism that an f32 HNSW baseline would also contribute;
  the looser gate matches the observed run-to-run variance band on
  this synthetic adversarial distribution. Real clustered embedding
  data and a stronger HNSW config (m=32, ef_construction=500) reach
  ≥ 0.99 on this path too — see the diagnostic sweep below.

The companion `stage2_recall_sweep_diagnostic` (opt-in via
`LAURUS_STAGE2_SWEEP=1`) sweeps `(ef_search, rerank_factor)` across
three corpus / query distributions and two HNSW configs so
production deployments can calibrate the budget for their actual
embedding distribution.

#### Real-data validation (Issue #498)

A third opt-in CI gate validates Stage 2 against a real ANN
benchmark dataset (SIFT1M from TEXMEX) so the synthetic-data gates
above do not become the only signal:

- `hnsw_quantized_recall_at_10_with_rerank_on_sift_meets_stage2_real_data_recall_gate`
  asserts `Recall@10 ≥ 0.99` on a 50 000-vector SIFT1M subsample at
  `(m=16, ef_construction=200, ef_search=200, rerank_factor=5)`.
- The companion bench `bench_hnsw_graph_search_rerank_real_data` (in
  `laurus/benches/vector_search_bench.rs`) measures end-to-end Stage 2
  latency on the same fixture. The accompanying example
  `laurus/examples/sift_rerank_probe.rs` runs a full
  `(ef_search × rerank_factor × HNSW config)` sweep with `(Recall,
  latency)` per cell so operators can pick the operating point for
  their own data shape.

Both are gated on `LAURUS_REAL_BENCHMARK=1` AND the presence of the
SIFT1M `.fvecs` files under `.cache/sift/sift/`. Default CI runs are
unchanged. To enable locally:

```sh
./scripts/fetch-sift.sh --large   # ~478MB
LAURUS_REAL_BENCHMARK=1 cargo test --release \
    --test vector_recall_test \
    hnsw_quantized_recall_at_10_with_rerank_on_sift_meets_stage2_real_data_recall_gate \
    -- --nocapture
LAURUS_REAL_BENCHMARK=1 cargo bench --bench vector_search_bench \
    -- "HNSW Graph Search Rerank Real"
```

The Issue #481 wording originally asked for "≥ 3× speedup vs the
pre-Stage-1 f32 baseline." Cross-branch Criterion measurements on
SIFT1M-50k (median over 30 samples, same `(m, ef_construction,
ef_search)`) gave 625 µs/query on the pre-Stage-1 f32 HNSW path
versus 323 µs/query on the Stage 2 int8 + rerank path — a **1.94×**
speedup. Issue #498 reduces the real-data gate to **≥ 1.5×**
accordingly, with the recall side held at the original 0.99. The
gap between the original 3× target and the measured 1.94× comes
from rerank only re-ordering candidates the int8 graph traversal
already visited — a lower `ef_search` does not pay back through
rerank when the candidate set itself becomes too narrow, which a
follow-up could address by widening the graph search budget
independently of `ef_search` (the Lucene 99 pattern).

### Product Quantization with rerank (Issue #481 Stage 3)

Stage 3 adds an opt-in **Product Quantization** path for the HNSW
index. Each segment trains a per-field codebook of `M` sub-vectors
× `K = 256` centroids using Lloyd k-means with k-means++
initialisation, and stores every vector as `M` bytes (one centroid
index per sub-vector). The search hot loop replaces the int8 SIMD
kernel with **asymmetric distance computation** (ADC): per-query
the searcher builds an `M × K` look-up table of squared distances
between the query's sub-vectors and the codebook entries, then
scores each candidate as `Σ_m lut[m][codes[m]]` — `M` table
lookups + `M − 1` adds per candidate.

PQ is enabled per field via `HnswOption.quantizer`:

```rust
use laurus::vector::HnswOption;
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::vector::core::rerank::RerankStorageKind;

let opt = HnswOption {
    dimension: 128,
    quantizer: QuantizationMethod::ProductQuantization { subvector_count: 32 },
    // PQ-only Recall@10 caps out around 0.78-0.92 on SIFT1M, so
    // production deployments should pair PQ with the LRS1 rerank
    // sidecar — same Stage 2 mechanism, just driven by PQ
    // candidate generation instead of int8.
    rerank_storage: Some(RerankStorageKind::F32),
    ..Default::default()
};
```

`subvector_count` must divide `dimension`. Common choices for
`dim = 128`: `M ∈ {8, 16, 32}` (sub_dim 16 / 8 / 4). Larger `M`
trades on-disk compression for higher recall — Issue #481 Stage 3
ships only the 8-bit (K = 256) variant; the on-disk format reserves
a 4-bit (K = 16) slot for a future PR.

#### On-disk format

PQ segments use the same `LVS1` header as Scalar8Bit (`quant_kind =
2`) and carry the codebook inside the per-segment metadata block:

```text
[ Fixed header           16 bytes ]
[ PQ params               8 bytes ]    m / k / sub_dim / padding (u16 × 4)
[ Codebook                m × k × sub_dim × 4 bytes ]
[ Per-vector codes        num_vectors × m bytes ]
```

For `dim = 128, M = 32, K = 256`: codebook = `32 × 256 × 4 × 4 =
131 072 bytes` (128 KB) per segment plus 32 bytes per vector.

#### Recall and speed gates (Issue #481 Stage 3)

- **Kernel-level test** — synthetic 5 000-vector / dim 128 / 100
  queries at `(m=16, ef_construction=200, ef_search=200,
  rerank_factor=10, M=32)`:
  `hnsw_pq_rerank_recall_at_10_meets_stage3_recall_gate` asserts
  Recall@10 ≥ 0.95. Measured 0.9660.
- **Real-data test** — SIFT1M-50k subsample (opt-in via
  `LAURUS_REAL_BENCHMARK=1`, same config):
  `hnsw_pq_rerank_recall_at_10_on_sift_meets_stage3_real_data_recall_gate`
  asserts Recall@10 ≥ 0.95. Measured 0.9965.
- **Real-data speed bench** —
  `bench_hnsw_graph_search_pq_rerank_real_data` (opt-in,
  Criterion). Cross-branch measurement at the same SIFT1M-50k
  config: pre-Stage-1 f32 HNSW = 625.21 µs/query (PR #500);
  Stage 3 PQ + rerank = **299.54 µs/query** = **2.09× speedup**.

Issue #481 originally asked for ≥ 5× speedup at Recall ≥ 0.95.
The PR established that target is not reachable on SIFT1M with the
current implementation — both Stage 2 (#500) and this Stage 3 PR
have measured speedups in the 1.9-2.1× band because rerank
dominates the wall-clock once the candidate set is wide enough to
recover recall. The gate was reduced to ≥ 1.5× accordingly. A
follow-up could pursue the Lucene 99 pattern (independent graph
search budget) and / or a 4-bit PQ variant to close the gap.

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
