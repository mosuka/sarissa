# Platypus

[![Crates.io](https://img.shields.io/crates/v/platypus.svg)](https://crates.io/crates/platypus)
[![Documentation](https://docs.rs/platypus/badge.svg)](https://docs.rs/platypus)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Platypus is a Rust-based hybrid search engine that unifies Keyword Search, Semantic Search, and Multimodal Search into a single, cohesive system.

The name comes from the platypus — one of the most remarkable real-world creatures, known for combining traits from mammals, birds, and reptiles into a single organism.
This unique fusion of distinct evolutionary features mirrors the three complementary forms of understanding in modern search:

Keyword Search — precise retrieval through lexical, symbolic, and linguistic matching.

Semantic Search — meaning-based retrieval powered by vector representations and embeddings.

Multimodal Search — bridging text, images, and other modalities through shared latent representations.

Together, these capabilities form a unified hybrid search architecture — much like the platypus itself, where diverse traits work in harmony to navigate complex environments.

Built in Rust for performance, safety, and extensibility, Platypus aims to provide a next-generation information retrieval platform that supports a broad range of use cases, from research exploration to production deployment.

## Features

### Core Search Capabilities

- **Pure Rust Implementation** - Memory-safe and fast performance with zero-cost abstractions
- **Keyword Search** - Full-text search with inverted index and BM25 scoring
- **Semantic Search** - HNSW-based approximate nearest neighbor search with multiple distance metrics (Cosine, Euclidean, Dot Product)
- **Multimodal Search** - Combined lexical and vector search with configurable score fusion strategies

### Text Analysis

- **Flexible Text Analysis Pipeline** - Configurable tokenization, stemming, and filtering
- **Multi-language Support** - Built-in support for Japanese, Korean, and Chinese via Lindera
- **Custom Analyzers** - Create custom analysis pipelines with pluggable tokenizers and filters
- **Synonym Support** - Synonym expansion for improved recall

### Advanced Query Types

- **Term Query** - Simple keyword matching
- **Phrase Query** - Exact phrase matching with positional information
- **Boolean Query** - Complex combinations with AND/OR/NOT logic
- **Range Query** - Numeric and date range queries
- **Fuzzy Query** - Approximate string matching with edit distance
- **Wildcard Query** - Pattern matching with * and ? wildcards
- **Geographic Query** - Location-based search with distance and bounding box queries

### Embedding & Semantic Search

- **Text Embeddings** - Generate semantic embeddings with Candle (local BERT models) or OpenAI API
- **Multimodal Search** - Cross-modal search with CLIP models for text-to-image and image-to-image similarity
- **Automatic Model Loading** - Models are automatically downloaded from HuggingFace on first use
- **GPU Acceleration** - Automatic GPU usage when available for embedding generation

### Storage & Performance

- **Multiple Storage Backends** - Filesystem, memory-mapped files, and in-memory storage
- **SIMD Acceleration** - Optimized vector operations for improved performance
- **Incremental Updates** - Real-time document addition without full reindexing

### Additional Features

- **Spell Correction** - Built-in spell checking and query suggestion system
- **Faceted Search** - Multi-dimensional search with facet aggregation and filtering
- **Schemaless Indexing** - Dynamic schema support for flexible document structures
- **Document Parsing** - Built-in support for various document formats

## Quick Start

Add Platypus to your `Cargo.toml`:

```toml
[dependencies]
platypus = "0.1"
```

### Basic Usage

```rust
use std::sync::Arc;

use tempfile::TempDir;
use platypus::analysis::analyzer::analyzer::Analyzer;
use platypus::analysis::analyzer::standard::StandardAnalyzer;
use platypus::lexical::document::document::Document;
use platypus::lexical::document::field::{IntegerOption, TextOption};
use platypus::error::Result;
use platypus::lexical::engine::LexicalEngine;
use platypus::lexical::index::config::{InvertedIndexConfig, LexicalIndexConfig};
use platypus::lexical::index::factory::LexicalIndexFactory;
use platypus::lexical::index::inverted::query::term::TermQuery;
use platypus::lexical::search::searcher::LexicalSearchRequest;
use platypus::storage::file::FileStorageConfig;
use platypus::storage::{StorageConfig, StorageFactory};

fn main() -> Result<()> {
    // Create storage in a temporary directory
    let temp_dir = TempDir::new().unwrap();
    let storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(
        temp_dir.path(),
    )))?;

    // Configure the inverted index with a StandardAnalyzer
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);
    let config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer: Arc::clone(&analyzer),
        ..InvertedIndexConfig::default()
    });
    let index = LexicalIndexFactory::create(storage, config)?;

    // Create a lexical search engine
    let mut engine = LexicalEngine::new(index)?;

    // Add documents with explicit field options
    let documents = vec![
        Document::builder()
            .add_text("title", "Rust Programming", TextOption::default())
            .add_text(
                "content",
                "Rust is a systems programming language",
                TextOption::default(),
            )
            .add_integer("year", 2024, IntegerOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Python Guide", TextOption::default())
            .add_text(
                "content",
                "Python is a versatile programming language",
                TextOption::default(),
            )
            .add_integer("year", 2023, IntegerOption::default())
            .build(),
    ];

    for doc in documents {
        engine.add_document(doc)?;
    }
    engine.commit()?;

    // Search documents
    let query = Box::new(TermQuery::new("content", "programming"));
    let request = LexicalSearchRequest::new(query)
        .load_documents(true)
        .max_docs(10);
    let results = engine.search(request)?;

    println!("Found {} matches", results.total_hits);
    for hit in results.hits {
        if let Some(doc) = hit.document {
            println!("Score: {:.2}, Doc ID: {}", hit.score, hit.doc_id);
            if let Some(title) = doc.get_text("title") {
                println!("  -> {}", title);
            }
        }
    }

    Ok(())
}
```

### Upsert / Hybrid Ingestion

- To replace an existing document with a specific ID, use `LexicalEngine::upsert_document(doc_id, doc)`. The `add_document` method only performs automatic ID assignment.
- In hybrid configurations, lexical and vector data are registered separately. First use `HybridEngine::add_document`/`upsert_document` to write lexical data, then use `HybridEngine::upsert_vector_document` for pre-embedded vectors, or `HybridEngine::upsert_vector_payload` to register raw text and embed it on the fly.

## Architecture

Platypus is built with a modular architecture:

### Core Components

- **Schema & Fields** - Define document structure with typed fields (text, numeric, boolean, geographic, vector)
- **Analysis Pipeline** - Configurable text processing with tokenizers, filters, and stemmers
- **Storage Layer** - Pluggable storage backends (filesystem, memory-mapped, in-memory) with transaction support
- **Lexical Index** - Inverted index with posting lists and term dictionaries for full-text search
- **Vector Index** - HNSW-based approximate nearest neighbor search for semantic similarity
- **Hybrid Search** - Combined lexical and vector search with configurable score fusion
- **Query Engine** - Flexible query system supporting multiple query types

### Engine Design Pattern

Both `LexicalEngine` and `VectorEngine` follow the same facade + factory pattern:

```rust
// LexicalEngine pattern
let index = LexicalIndexFactory::create(storage, config)?;
let engine = LexicalEngine::new(index)?;

// VectorEngine pattern
let collection = VectorCollectionFactory::create(config, storage, None)?;
let engine = VectorEngine::new(collection)?;
```

### Field Types

Platypus supports the following field value types through the `Document` builder API:

```rust
use chrono::Utc;
use platypus::lexical::document::document::Document;
use platypus::lexical::document::field::{
    BinaryOption, BooleanOption, DateTimeOption, FloatOption, GeoOption, IntegerOption,
    TextOption, VectorOption,
};

let doc = Document::builder()
    // Text field for full-text search
    .add_text("title", "Introduction to Rust", TextOption::default())

    // Integer field for numeric queries
    .add_integer("year", 2024, IntegerOption::default())

    // Float field for floating-point values
    .add_float("price", 49.99, FloatOption::default())

    // Boolean field for filtering
    .add_boolean("published", true, BooleanOption::default())

    // DateTime field for temporal queries
    .add_datetime("created_at", Utc::now(), DateTimeOption::default())

    // Geographic field for spatial queries
    .add_geo("location", 35.6762, 139.6503, GeoOption::default())

    // Binary field for arbitrary data
    .add_binary("thumbnail", vec![0u8, 1, 2, 3], BinaryOption::default())

    // Vector field storing text that will be embedded during indexing
    .add_vector("title_embedding", "Introduction to Rust", VectorOption::default())

    .build();
```

### Query Types

```rust
use platypus::lexical::index::inverted::query::boolean::BooleanQuery;
use platypus::lexical::index::inverted::query::fuzzy::FuzzyQuery;
use platypus::lexical::index::inverted::query::geo::{
    GeoBoundingBox, GeoBoundingBoxQuery, GeoDistanceQuery, GeoPoint,
};
use platypus::lexical::index::inverted::query::phrase::PhraseQuery;
use platypus::lexical::index::inverted::query::range::NumericRangeQuery;
use platypus::lexical::index::inverted::query::term::TermQuery;
use platypus::lexical::index::inverted::query::wildcard::WildcardQuery;

// Term query - simple keyword matching
let query = Box::new(TermQuery::new("field", "term"));

// Phrase query - exact phrase matching
let query = Box::new(PhraseQuery::new("field", vec!["hello", "world"]));

// Numeric range query
let query = Box::new(NumericRangeQuery::new_float("price", Some(100.0), Some(500.0)));

// Boolean query - combine multiple queries
let mut bool_query = BooleanQuery::new();
bool_query.add_must(Box::new(TermQuery::new("category", "book")));
bool_query.add_should(Box::new(TermQuery::new("author", "tolkien")));
let query = Box::new(bool_query);

// Fuzzy query - approximate string matching
let query = Box::new(FuzzyQuery::new("title", "progamming", 2)); // max edit distance: 2

// Wildcard query - pattern matching
let query = Box::new(WildcardQuery::new("filename", "*.pdf"));

// Geographic distance query
let query = Box::new(GeoDistanceQuery::new(
    "location",
    GeoPoint::new(40.7128, -74.0060), // NYC coordinates
    10.0, // 10km radius
));

// Geographic bounding box query
let query = Box::new(GeoBoundingBoxQuery::new(
    "location",
    GeoBoundingBox::new(
        GeoPoint::new(40.5, -74.5), // bottom-left
        GeoPoint::new(41.0, -73.5), // top-right
    ),
));
```

## Document Operations

Both `LexicalEngine` and `VectorEngine` provide consistent document operations:

| Operation | Method | Description |
|-----------|--------|-------------|
| Add | `add_document(doc) -> Result<u64>` | Add document with auto-assigned ID |
| Upsert | `upsert_document(doc_id, doc) -> Result<()>` | Insert or replace document with specific ID |
| Delete | `delete_document(doc_id) -> Result<()>` | Delete document by ID |
| Commit | `commit() -> Result<()>` | Persist pending changes |

## Advanced Features

### Vector Search with Text Embeddings

Platypus supports semantic search using text embeddings. You can use local BERT models via Candle or OpenAI's API.

#### Using Candle (Local BERT Models)

```toml
[dependencies]
platypus = { version = "0.1", features = ["embeddings-candle"] }
```

```rust
use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
use platypus::embedding::embedder::EmbedInput;
use platypus::embedding::noop::NoOpEmbedder;
use platypus::vector::engine::request::{QueryVector, VectorSearchRequest};
use platypus::vector::core::document::{DocumentVector, StoredVector, VectorType};
use platypus::vector::engine::VectorEngine;
use platypus::vector::engine::config::{VectorIndexConfig, VectorFieldConfig, VectorIndexKind};
use platypus::vector::DistanceMetric;
use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use std::sync::Arc;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> platypus::error::Result<()> {
    // Initialize embedder with a sentence-transformers model
    let embedder = CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?;

    // Create vector engine configuration (MiniLM-L6-v2 outputs 384-dim)
    let field_config = VectorFieldConfig {
        dimension: 384,
        distance: DistanceMetric::Cosine,
        index: VectorIndexKind::Flat,
        source_tag: "candle".into(),
        vector_type: VectorType::Text,
        base_weight: 1.0,
    };
    let config = VectorIndexConfig::builder()
        .embedder(NoOpEmbedder::new())
        .field("body", field_config)
        .default_field("body")
        .build()?;

    // Create storage and engine
    let storage: Arc<dyn platypus::storage::Storage> =
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let engine = VectorEngine::new(storage, config)?;

    // Generate embeddings for documents
    let documents = vec![
        "Rust is a systems programming language",
        "Python is great for data science",
        "Machine learning with neural networks",
    ];

    // Add documents with their embeddings (1 field = 1 vector)
    for text in &documents {
        let vector = embedder.embed(&EmbedInput::Text(text)).await?;
        let stored = StoredVector::new(
            Arc::from(vector.data.clone()),
            "candle".into(),
            VectorType::Text,
        );
        let mut doc = DocumentVector::new();
        doc.set_field("body", stored);
        engine.add_vectors(doc)?;
    }
    engine.commit()?;

    // Search with query embedding
    let query_vector = embedder.embed(&EmbedInput::Text("programming languages")).await?;
    let mut query = VectorSearchRequest::default();
    query.limit = 10;
    query.query_vectors.push(QueryVector {
        vector: StoredVector::new(
            Arc::from(query_vector.data.clone()),
            "candle".into(),
            VectorType::Text,
        ),
        weight: 1.0,
    });
    let results = engine.search(query)?;

    for hit in results.hits {
        println!("Doc ID: {}, Score: {:.4}", hit.doc_id, hit.score);
    }

    Ok(())
}
```

#### Using OpenAI Embeddings

```toml
[dependencies]
platypus = { version = "0.1", features = ["embeddings-openai"] }
```

```rust
use platypus::embedding::openai_text_embedder::OpenAITextEmbedder;
use platypus::embedding::text_embedder::TextEmbedder;

// Initialize with API key
let embedder = OpenAITextEmbedder::new(
    "your-api-key".to_string(),
    "text-embedding-3-small".to_string()
)?;

// Generate embeddings
let vector = embedder.embed("your text here").await?;
```

### Doc-centric VectorEngine

Platypus ships a document-centric vector flow where each `doc_id` owns multiple named vector fields and metadata. The full architecture is captured in `docs/vector_engine.md`, and two handy entry points are provided:

- `resources/vector_engine_sample.json` — three synthetic `DocumentVector` entries with field-level and document-level metadata for trying out `MetadataFilter` / `FieldSelector` scenarios.
- `cargo test --test vector_engine_scenarios` — spins up an in-memory engine, loads the sample, and verifies multi-field scoring plus metadata filters end to end.

You can also use the sample data in your own experiments:

```rust
use platypus::vector::engine::VectorEngine;
use platypus::vector::engine::config::VectorEngineConfig;
use platypus::vector::collection::factory::VectorCollectionFactory;
use platypus::vector::core::document::DocumentVector;

let config: VectorEngineConfig = load_vector_engine_config()?; // see docs/vector_engine.md
let sample_docs: Vec<DocumentVector> = serde_json::from_str(include_str!(
    "resources/vector_engine_sample.json"
))?;
let collection = VectorCollectionFactory::create(config, storage, None)?;
let engine = VectorEngine::new(collection)?;
for doc in sample_docs {
    engine.add_document(doc)?;
}
```

Once the engine is populated, build `VectorEngineSearchRequest` objects (see `examples/vector_search.rs`) to target specific fields, adjust `VectorScoreMode`, and apply metadata filters — exactly the same path used by the integration test above.

#### Automatic Embedding (Raw Payloads)

When you prefer not to precompute vectors yourself, hand the engine raw text payloads and let it run the configured embedder pipeline:

```rust
use platypus::vector::core::document::{DocumentPayload, Payload};
use platypus::vector::engine::request::{FieldSelector, QueryPayload, VectorSearchRequest};
use platypus::vector::engine::VectorEngine;

// Create a document with a text payload (1 field = 1 payload)
let mut payload = DocumentPayload::new();
payload.set_text("body_embedding", "Rust balances safety with performance");
engine.upsert_payloads(42, payload)?;

// Search with a text query payload
let mut query = VectorSearchRequest::default();
query.fields = Some(vec![FieldSelector::Exact("body_embedding".into())]);
query.query_payloads.push(QueryPayload::new(
    "body_embedding",
    Payload::text("systems programming"),
));
let hits = engine.search(query)?;
```

These helpers power `examples/vector_search.rs` and the `vector_engine_upserts_and_queries_raw_payloads` integration test, so you can follow the same pattern or wrap it through `HybridSearchRequest::with_vector_text` when composing hybrid searches.

**Note**: The flattened data model requires 1 field = 1 payload = 1 vector. For long texts that require chunking, create separate documents for each chunk and use metadata (e.g., `parent_doc_id`, `chunk_index`) to track relationships. See `examples/vector_search.rs` for a demonstration of this pattern.

#### Hybrid Search with VectorEngine

`HybridSearchRequest` now understands doc-centric overrides so you can keep using the lexical-first ergonomics while routing the vector side through `VectorEngine`:

```rust
use platypus::hybrid::search::searcher::{HybridSearchRequest, ScoreNormalization};
use platypus::vector::engine::request::{
    FieldSelector, QueryVector, VectorEngineSearchRequest, VectorScoreMode,
};
use platypus::vector::engine::filter::VectorEngineFilter;
use platypus::vector::core::document::StoredVector;
use platypus::vector::search::searcher::VectorSearchParams;

let my_vector = Vector::new(vec![0.1, 0.2, 0.3]);

let mut vector_query = VectorEngineSearchRequest::default();
vector_query.limit = 32;
vector_query.query_vectors.push(QueryVector {
    vector: StoredVector::from(my_vector),
    weight: 1.0,
});

let mut filter = VectorEngineFilter::default();
filter
    .document
    .equals
    .insert("lang".into(), "en".into());

let mut vector_params = VectorSearchParams::default();
vector_params.top_k = 32;

let request = HybridSearchRequest::new("rust programming")
    .with_vector_engine_search_request(vector_query)
    .vector_fields(vec![FieldSelector::Exact("body_embedding".into())])
    .vector_filter(filter)
    .vector_score_mode(VectorScoreMode::WeightedSum)
    .vector_overfetch(2.0)
    .vector_params(vector_params)
    .keyword_weight(0.5)
    .vector_weight(0.5)
    .min_vector_similarity(0.35)
    .normalization(ScoreNormalization::MinMax);
```

Key points:

- Vector overrides (`vector_fields`, `vector_filter`, `vector_score_mode`, `vector_overfetch`) are applied consistently whether you pass a raw `Vector` or a fully built `VectorEngineSearchRequest`.
- Document-level metadata filters (`VectorEngineFilter::document`) execute before any vector work, so hybrid queries can cheaply target subsets such as `lang = ja` or `tenant_id = acme`.
- Field-level hits from the vector side are preserved in `HybridSearchResult::vector_field_hits`, making it easy to explain which embeddings matched.
- The helper tests under `src/hybrid/engine.rs` and `src/hybrid/search/merger.rs` illustrate how the engine enforces `min_vector_similarity`, top-k limits, and metadata propagation end to end (`cargo test hybrid::engine` / `cargo test hybrid::search::merger`).

### Multimodal Search (Text + Images)

Platypus supports cross-modal search using CLIP (Contrastive Language-Image Pre-Training) models, enabling semantic search across text and images. This allows you to:

- **Text-to-Image Search**: Find images using natural language queries
- **Image-to-Image Search**: Find visually similar images using an image query
- **Semantic Understanding**: Search based on content meaning, not just keywords

#### Setup

Add the `embeddings-multimodal` feature to your `Cargo.toml`:

```toml
[dependencies]
platypus = { version = "0.1", features = ["embeddings-multimodal"] }
```

#### Text-to-Image Search Example

```rust
use platypus::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
use platypus::embedding::embedder::EmbedInput;
use platypus::embedding::noop::NoOpEmbedder;
use platypus::vector::engine::request::{QueryVector, VectorSearchRequest};
use platypus::vector::core::document::{DocumentVector, StoredVector, VectorType};
use platypus::vector::engine::VectorEngine;
use platypus::vector::engine::config::{VectorIndexConfig, VectorFieldConfig, VectorIndexKind};
use platypus::vector::DistanceMetric;
use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use std::sync::Arc;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> platypus::error::Result<()> {
    // Initialize CLIP embedder (automatically downloads model from HuggingFace)
    let embedder = CandleMultimodalEmbedder::new("openai/clip-vit-base-patch32")?;

    // Create vector engine configuration with CLIP's embedding dimension (512)
    let field_config = VectorFieldConfig {
        dimension: 512,
        distance: DistanceMetric::Cosine,
        index: VectorIndexKind::Hnsw,
        source_tag: "clip".into(),
        vector_type: VectorType::Image,
        base_weight: 1.0,
    };
    let config = VectorIndexConfig::builder()
        .embedder(NoOpEmbedder::new())
        .field("image", field_config)
        .default_field("image")
        .build()?;

    let storage: Arc<dyn platypus::storage::Storage> =
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let engine = VectorEngine::new(storage, config)?;

    // Index your image collection (1 field = 1 vector)
    let image_paths = vec!["image1.jpg", "image2.jpg", "image3.jpg"];
    for image_path in &image_paths {
        let vector = embedder.embed(&EmbedInput::ImagePath(image_path)).await?;
        let stored = StoredVector::new(
            Arc::from(vector.data.clone()),
            "clip".into(),
            VectorType::Image,
        );
        let mut doc = DocumentVector::new();
        doc.set_field("image", stored);
        engine.add_vectors(doc)?;
    }
    engine.commit()?;

    // Search images using natural language
    let query_vector = embedder
        .embed(&EmbedInput::Text("a photo of a cat playing"))
        .await?;
    let mut query = VectorSearchRequest::default();
    query.limit = 10;
    query.query_vectors.push(QueryVector {
        vector: StoredVector::new(
            Arc::from(query_vector.data.clone()),
            "clip".into(),
            VectorType::Text,
        ),
        weight: 1.0,
    });
    let results = engine.search(query)?;

    for hit in results.hits {
        println!("Image ID: {}, Score: {:.4}", hit.doc_id, hit.score);
    }

    Ok(())
}
```

#### Image-to-Image Search Example

```rust
// Find visually similar images using an image as query
let query_image_vector = embedder.embed_image("query.jpg").await?;
let mut query = VectorSearchRequest::default();
query.limit = 5;
query.query_vectors.push(QueryVector {
    vector: StoredVector::new(
        Arc::from(query_image_vector.as_slice()),
        "clip".into(),
        VectorType::Image,
    ),
    weight: 1.0,
});
let results = engine.search(query)?;

for hit in results.hits {
    println!("Similar Image ID: {}, Score: {:.4}", hit.doc_id, hit.score);
}
```

#### How It Works

1. **CLIP Model**: Uses OpenAI's CLIP model which maps both text and images into the same 512-dimensional vector space
2. **Automatic Download**: Models are automatically downloaded from HuggingFace on first use
3. **GPU Acceleration**: Automatically uses GPU if available (via Candle)
4. **Shared Embedding Space**: Text and image embeddings can be directly compared using cosine similarity

#### Supported Models

Currently supports CLIP ViT-Base-Patch32 architecture:

- Model: `openai/clip-vit-base-patch32`
- Embedding Dimension: 512
- Image Size: 224x224

#### Complete Examples

See working examples with detailed explanations:

- [examples/text_to_image_search.rs](examples/text_to_image_search.rs) - Full text-to-image search implementation
- [examples/image_to_image_search.rs](examples/image_to_image_search.rs) - Full image similarity search implementation

Run the examples:

```bash
# Text-to-image search
cargo run --example text_to_image_search --features embeddings-multimodal

# Image-to-image search
cargo run --example image_to_image_search --features embeddings-multimodal -- query.jpg
```

### Faceted Search

```rust
use platypus::lexical::index::inverted::query::term::TermQuery;
use platypus::lexical::search::facet::{FacetConfig, FacetedSearchEngine};
use platypus::lexical::search::searcher::LexicalSearchRequest;

// Create faceted search engine
let facet_config = FacetConfig {
    max_facet_count: 10,
    min_count: 1,
};

let mut faceted_engine = FacetedSearchEngine::new(
    engine,
    vec!["category".to_string(), "author".to_string()],
    facet_config,
)?;

// Perform faceted search
let query = Box::new(TermQuery::new("content", "programming"));
let request = LexicalSearchRequest::new(query).max_docs(10);
let results = faceted_engine.search(request)?;

// Access facet results
for facet in &results.facets {
    println!("Facet field: {}", facet.field);
    for count in &facet.counts {
        println!("  {}: {} documents", count.value, count.count);
    }
}
```

### Spell Correction

```rust
use platypus::spelling::corrector::{SpellingCorrector, CorrectorConfig};
use platypus::spelling::dictionary::Dictionary;

// Build a dictionary from your corpus
let mut dictionary = Dictionary::new();
dictionary.add_word("programming", 100);
dictionary.add_word("program", 80);
dictionary.add_word("programmer", 60);

// Create spell corrector with configuration
let config = CorrectorConfig {
    max_edit_distance: 2,
    min_word_frequency: 5,
    max_suggestions: 5,
};

let corrector = SpellingCorrector::new(dictionary, config);

// Check and suggest corrections
if let Some(correction) = corrector.correct("progamming") {
    println!("Did you mean: '{}'? (confidence: {:.2})",
        correction.suggestion, correction.confidence);
}
```

### Custom Analysis Pipeline

```rust
use platypus::analysis::analyzer::pipeline::PipelineAnalyzer;
use platypus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
use platypus::analysis::token_filter::lowercase::LowercaseFilter;
use platypus::analysis::token_filter::stop::StopWordFilter;

// Create custom analyzer with multiple filters
let mut analyzer = PipelineAnalyzer::new(Box::new(WhitespaceTokenizer));
analyzer.add_filter(Box::new(LowercaseFilter));
analyzer.add_filter(Box::new(StopWordFilter::english()));

// Analyze text
let text = "The Quick Brown Fox Jumps Over the Lazy Dog";
let tokens = analyzer.analyze(text)?;

for token in tokens {
    println!("Token: {}, Position: {}", token.text, token.position);
}
```

For language-specific tokenization (Japanese, Korean, Chinese):

```rust
use platypus::analysis::tokenizer::lindera::LinderaTokenizer;
use platypus::analysis::analyzer::pipeline::PipelineAnalyzer;

// Japanese tokenization with Lindera
let tokenizer = LinderaTokenizer::japanese()?;
let analyzer = PipelineAnalyzer::new(Box::new(tokenizer));

let text = "Tokyo is the capital of Japan";
let tokens = analyzer.analyze(text)?;
```

## Performance

Platypus is designed for high performance:

- **SIMD Acceleration** - Uses wide instruction sets for vector operations
- **Memory-Mapped I/O** - Efficient file access with minimal memory overhead
- **Incremental Updates** - Real-time document addition without full reindexing
- **Index Optimization** - Background merge operations for optimal search performance

## Development

### Building from Source

```bash
git clone https://github.com/mosuka/platypus.git
cd platypus
cargo build --release
```

### Running Tests

```bash
cargo test
```

### Running Benchmarks

```bash
cargo bench
```

### Checking Code Quality

```bash
cargo clippy
cargo fmt --check
```

## Examples

Platypus includes numerous examples demonstrating various features:

### Lexical Search Examples

- [term_query](examples/term_query.rs) - Basic term-based search
- [phrase_query](examples/phrase_query.rs) - Multi-word phrase matching
- [boolean_query](examples/boolean_query.rs) - Combining queries with AND/OR/NOT logic
- [fuzzy_query](examples/fuzzy_query.rs) - Fuzzy string matching with edit distance
- [wildcard_query](examples/wildcard_query.rs) - Pattern matching with wildcards
- [range_query](examples/range_query.rs) - Numeric and date range queries
- [geo_query](examples/geo_query.rs) - Geographic location-based search
- [field_specific_search](examples/field_specific_search.rs) - Search within specific fields
- [lexical_search](examples/lexical_search.rs) - Full lexical search example
- [query_parser](examples/query_parser.rs) - Parsing user query strings

### Vector Search Examples

- [vector_search](examples/vector_search.rs) - Doc-centric `VectorEngine` demo with sample metadata filters
- [embedding_with_candle](examples/embedding_with_candle.rs) - Local BERT model embeddings
- [embedding_with_openai](examples/embedding_with_openai.rs) - OpenAI API embeddings
- [dynamic_embedder_switching](examples/dynamic_embedder_switching.rs) - Switch between embedding providers
- [text_to_image_search](examples/text_to_image_search.rs) - Text-to-image search with CLIP
- [image_to_image_search](examples/image_to_image_search.rs) - Image similarity search

### Advanced Features

- [schemaless_indexing](examples/schemaless_indexing.rs) - Dynamic schema management
- [synonym_graph_filter](examples/synonym_graph_filter.rs) - Synonym expansion in queries
- [document_parser](examples/document_parser.rs) - Parsing various document formats
- [document_converter](examples/document_converter.rs) - Converting between document formats

Run any example with:

```bash
cargo run --example <example_name>

# For embedding examples, use feature flags:
# Doc-centric VectorEngine sample (uses built-in JSON fixtures)
cargo run --example vector_search
cargo run --example vector_search --features embeddings-candle
cargo run --example embedding_with_openai --features embeddings-openai
cargo run --example text_to_image_search --features embeddings-multimodal
cargo run --example image_to_image_search --features embeddings-multimodal
```

## Feature Flags

Platypus uses feature flags to enable optional functionality:

```toml
[dependencies]
# Default features only
platypus = "0.1"

# With Candle embeddings (local BERT models)
platypus = { version = "0.1", features = ["embeddings-candle"] }

# With OpenAI embeddings
platypus = { version = "0.1", features = ["embeddings-openai"] }

# With all embedding features
platypus = { version = "0.1", features = ["embeddings-all"] }
```

Available features:

- `embeddings-candle` - Local text embeddings using Candle and BERT models
- `embeddings-openai` - OpenAI API-based text embeddings
- `embeddings-multimodal` - Multimodal embeddings (text and images) using CLIP models
- `embeddings-all` - All embedding providers

## Documentation

- [API Documentation](https://docs.rs/platypus)
- [User Guide](https://github.com/mosuka/platypus/wiki)
- [Examples](./examples/)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under either of

- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
