# Examples

The `laurus/examples/` directory contains runnable examples demonstrating different features of the library.

## Running Examples

```bash
# Run an example without feature flags
cargo run --example <name>

# Run an example with a feature flag
cargo run --example <name> --features <flag>
```

## Available Examples

### quickstart

A minimal example showing the basic workflow: create storage, define a schema, build an engine, index documents, and search.

```bash
cargo run --example quickstart
```

Demonstrates: In-memory storage, `TextOption`, `TermQuery`, `LexicalSearchRequest`.

### lexical_search

Comprehensive example of all lexical query types, using both the Builder API and the QueryParser DSL.

```bash
cargo run --example lexical_search
```

Demonstrates: `TermQuery`, `PhraseQuery`, `FuzzyQuery`, `WildcardQuery`, `NumericRangeQuery`, `GeoDistanceQuery` / `GeoBoundingBoxQuery`, `BooleanQuery`, `SpanQuery`.

### vector_search

Vector search with a mock embedder, including filtered vector search and DSL syntax.

```bash
cargo run --example vector_search
```

Demonstrates: `PerFieldEmbedder`, `VectorSearchRequestBuilder`, filtered search, DSL syntax (`field:"query"`).

### hybrid_search

Combining lexical and vector search with different fusion algorithms.

```bash
cargo run --example hybrid_search
```

Demonstrates: Lexical-only, vector-only, and hybrid search. Both `RRF` and `WeightedSum` fusion algorithms. Builder API and DSL.

### geo3d_search

3D Earth-Centered Earth-Fixed (ECEF) geographic search. Indexes six landmarks (Tokyo Tower, Tokyo Skytree, Mt. Fuji summit, Statue of Liberty, Sydney Opera House, an ISS sample point) computed via `wgs84_to_ecef`.

```bash
cargo run --example geo3d_search
```

Demonstrates: `Geo3dDistanceQuery` (sphere), `Geo3dBoundingBoxQuery` (3D AABB), `Geo3dNearestQuery` (k-NN), and the `wgs84_to_ecef` conversion utility. The bounding-box query intentionally excludes the ISS sample to highlight that altitude is a first-class third axis.

### search_with_candle

Vector search using real BERT embeddings via Hugging Face Candle. The model is downloaded automatically on first run (~80 MB).

```bash
cargo run --example search_with_candle --features embeddings-candle
```

**Requires:** `embeddings-candle` feature flag.

Demonstrates: `CandleBertEmbedder` with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).

### search_with_openai

Vector search using the OpenAI Embeddings API.

```bash
export OPENAI_API_KEY=your-api-key
cargo run --example search_with_openai --features embeddings-openai
```

**Requires:** `embeddings-openai` feature flag, `OPENAI_API_KEY` environment variable.

Demonstrates: `OpenAIEmbedder` with `text-embedding-3-small` (1536 dimensions).

### multimodal_search

Multimodal (text + image) search using a CLIP model.

```bash
cargo run --example multimodal_search --features embeddings-multimodal
```

**Requires:** `embeddings-multimodal` feature flag.

Demonstrates: `CandleClipEmbedder`, indexing images from the filesystem, text-to-image and image-to-image queries.

### synonym_graph_filter

Demonstrates the `SynonymGraphFilter` for token expansion during analysis.

```bash
cargo run --example synonym_graph_filter
```

Demonstrates: Synonym dictionary creation, synonym-based token expansion, boost application, token position and position_length attributes.

## Helper Module: common.rs

The `common.rs` file provides shared utilities used by the examples:

- `memory_storage()` -- Create an in-memory storage instance
- `per_field_analyzer()` -- Create a `PerFieldAnalyzer` with `KeywordAnalyzer` for specific fields
- `MockEmbedder` -- A mock `Embedder` implementation for testing vector search without a real model
