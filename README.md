# Yatagarasu

[![Crates.io](https://img.shields.io/crates/v/yatagarasu.svg)](https://crates.io/crates/yatagarasu)
[![Documentation](https://docs.rs/yatagarasu/badge.svg)](https://docs.rs/yatagarasu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Yatagarasu (pronounced yah-tah-gah-rah-soo) is a Rust-based hybrid search engine that unifies Keyword Search, Semantic Search, and Multimodal Search into a single, extensible system.

The name comes from Yatagarasu (八咫烏) — the three-legged crow of Japanese mythology that guided emperors through unknown lands.
Each of its three legs symbolizes a distinct form of understanding:

🐦‍⬛ Keyword Search — precise retrieval through lexical and linguistic matching.

🐦‍⬛ Semantic Search — meaning-based retrieval powered by vector representations and embeddings.

🐦‍⬛ Multimodal Search — bridging text, image, and other modalities through shared representations.

Together, they form a unified search architecture that guides users toward knowledge hidden across all forms of data.

Built in Rust for performance, safety, and extensibility, Yatagarasu is designed for both research exploration and production-grade search applications.

## ✨ Features

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

## 🚀 Quick Start

Add Yatagarasu to your `Cargo.toml`:

```toml
[dependencies]
yatagarasu = "0.1"
```

### Basic Usage

```rust
use std::sync::Arc;

use tempfile::TempDir;
use yatagarasu::analysis::analyzer::analyzer::Analyzer;
use yatagarasu::analysis::analyzer::standard::StandardAnalyzer;
use yatagarasu::document::document::Document;
use yatagarasu::document::field::{IntegerOption, TextOption};
use yatagarasu::error::Result;
use yatagarasu::lexical::engine::LexicalEngine;
use yatagarasu::lexical::index::config::{InvertedIndexConfig, LexicalIndexConfig};
use yatagarasu::lexical::index::factory::LexicalIndexFactory;
use yatagarasu::lexical::index::inverted::query::term::TermQuery;
use yatagarasu::lexical::search::searcher::LexicalSearchRequest;
use yatagarasu::storage::file::FileStorageConfig;
use yatagarasu::storage::{StorageConfig, StorageFactory};

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

## 🏗️ Architecture

Yatagarasu is built with a modular architecture:

### Core Components

- **Schema & Fields** - Define document structure with typed fields (text, numeric, boolean, geographic, vector)
- **Analysis Pipeline** - Configurable text processing with tokenizers, filters, and stemmers
- **Storage Layer** - Pluggable storage backends (filesystem, memory-mapped, in-memory) with transaction support
- **Lexical Index** - Inverted index with posting lists and term dictionaries for full-text search
- **Vector Index** - HNSW-based approximate nearest neighbor search for semantic similarity
- **Hybrid Search** - Combined lexical and vector search with configurable score fusion
- **Query Engine** - Flexible query system supporting multiple query types

### Field Types

Yatagarasu supports the following field value types through the `Document` builder API:

```rust
use chrono::Utc;
use yatagarasu::document::document::Document;
use yatagarasu::document::field::{
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
use yatagarasu::lexical::index::inverted::query::boolean::BooleanQuery;
use yatagarasu::lexical::index::inverted::query::fuzzy::FuzzyQuery;
use yatagarasu::lexical::index::inverted::query::geo::{
    GeoBoundingBox, GeoBoundingBoxQuery, GeoDistanceQuery, GeoPoint,
};
use yatagarasu::lexical::index::inverted::query::phrase::PhraseQuery;
use yatagarasu::lexical::index::inverted::query::range::NumericRangeQuery;
use yatagarasu::lexical::index::inverted::query::term::TermQuery;
use yatagarasu::lexical::index::inverted::query::wildcard::WildcardQuery;

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

## 🎯 Advanced Features

### Vector Search with Text Embeddings

Yatagarasu supports semantic search using text embeddings. You can use local BERT models via Candle or OpenAI's API.

#### Using Candle (Local BERT Models)

```toml
[dependencies]
yatagarasu = { version = "0.1", features = ["embeddings-candle"] }
```

```rust
use yatagarasu::embedding::candle_text_embedder::CandleTextEmbedder;
use yatagarasu::embedding::text_embedder::TextEmbedder;
use yatagarasu::vector::DistanceMetric;
use yatagarasu::vector::engine::VectorEngine;
use yatagarasu::vector::index::{FlatIndexConfig, VectorIndexConfig, VectorIndexFactory};
use yatagarasu::vector::types::VectorSearchRequest;
use yatagarasu::storage::memory::{MemoryStorage, MemoryStorageConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> yatagarasu::error::Result<()> {
    // Initialize embedder with a sentence-transformers model
    let embedder = CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?;

    // Generate embeddings for documents
    let documents = vec![
        (1, "Rust is a systems programming language"),
        (2, "Python is great for data science"),
        (3, "Machine learning with neural networks"),
    ];

    // Create vector index configuration
    let vector_config = VectorIndexConfig::Flat(FlatIndexConfig {
        dimension: embedder.dimension(),
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: true,
        ..Default::default()
    });

    // Create storage and index
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = VectorIndexFactory::create(storage, vector_config)?;
    let mut engine = VectorEngine::new(index)?;

    // Add documents with their embeddings
    for (id, text) in &documents {
        let vector = embedder.embed(text).await?;
        engine.add_vector(*id, vector)?;
    }
    engine.commit()?;

    // Search with query embedding
    let query_vector = embedder.embed("programming languages").await?;
    let request = VectorSearchRequest::new(query_vector).top_k(10);
    let results = engine.search(request)?;

    for result in results.results {
        println!("Doc ID: {}, Similarity: {:.4}", result.doc_id, result.similarity);
    }

    Ok(())
}
```

#### Using OpenAI Embeddings

```toml
[dependencies]
yatagarasu = { version = "0.1", features = ["embeddings-openai"] }
```

```rust
use yatagarasu::embedding::openai_text_embedder::OpenAITextEmbedder;
use yatagarasu::embedding::text_embedder::TextEmbedder;

// Initialize with API key
let embedder = OpenAITextEmbedder::new(
    "your-api-key".to_string(),
    "text-embedding-3-small".to_string()
)?;

// Generate embeddings
let vector = embedder.embed("your text here").await?;
```

### Multimodal Search (Text + Images)

Yatagarasu supports cross-modal search using CLIP (Contrastive Language-Image Pre-Training) models, enabling semantic search across text and images. This allows you to:

- **Text-to-Image Search**: Find images using natural language queries
- **Image-to-Image Search**: Find visually similar images using an image query
- **Semantic Understanding**: Search based on content meaning, not just keywords

#### Setup

Add the `embeddings-multimodal` feature to your `Cargo.toml`:

```toml
[dependencies]
yatagarasu = { version = "0.1", features = ["embeddings-multimodal"] }
```

#### Text-to-Image Search Example

```rust
use yatagarasu::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
use yatagarasu::embedding::text_embedder::TextEmbedder;
use yatagarasu::embedding::image_embedder::ImageEmbedder;
use yatagarasu::vector::engine::VectorEngine;
use yatagarasu::vector::index::{HnswIndexConfig, VectorIndexConfig, VectorIndexFactory};
use yatagarasu::vector::types::VectorSearchRequest;
use yatagarasu::vector::DistanceMetric;
use yatagarasu::storage::memory::{MemoryStorage, MemoryStorageConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> yatagarasu::error::Result<()> {
    // Initialize CLIP embedder (automatically downloads model from HuggingFace)
    let embedder = CandleMultimodalEmbedder::new("openai/clip-vit-base-patch32")?;

    // Create vector index with CLIP's embedding dimension (512)
    let vector_config = VectorIndexConfig::Hnsw(HnswIndexConfig {
        dimension: ImageEmbedder::dimension(&embedder), // 512 for CLIP ViT-Base-Patch32
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    });

    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = VectorIndexFactory::create(storage, vector_config)?;
    let mut engine = VectorEngine::new(index)?;

    // Index your image collection
    let image_paths = vec!["image1.jpg", "image2.jpg", "image3.jpg"];
    for (id, image_path) in image_paths.iter().enumerate() {
        let vector = embedder.embed_image(image_path).await?;
        engine.add_vector(id as u64, vector)?;
    }
    engine.commit()?;

    // Search images using natural language
    let query_vector = embedder.embed("a photo of a cat playing").await?;
    let request = VectorSearchRequest::new(query_vector).top_k(10);
    let results = engine.search(request)?;

    for result in results.results {
        println!("Image ID: {}, Similarity: {:.4}", result.doc_id, result.similarity);
    }

    Ok(())
}
```

#### Image-to-Image Search Example

```rust
// Find visually similar images using an image as query
let query_image_vector = embedder.embed_image("query.jpg").await?;
let request = VectorSearchRequest::new(query_image_vector).top_k(5);
let results = engine.search(request)?;

for result in results.results {
    println!("Similar Image ID: {}, Similarity: {:.4}", result.doc_id, result.similarity);
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
use yatagarasu::lexical::index::inverted::query::term::TermQuery;
use yatagarasu::lexical::search::facet::{FacetConfig, FacetedSearchEngine};
use yatagarasu::lexical::types::LexicalSearchRequest;

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
use yatagarasu::spelling::corrector::{SpellingCorrector, CorrectorConfig};
use yatagarasu::spelling::dictionary::Dictionary;

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
use yatagarasu::analysis::analyzer::pipeline::PipelineAnalyzer;
use yatagarasu::analysis::tokenizer::whitespace::WhitespaceTokenizer;
use yatagarasu::analysis::token_filter::lowercase::LowercaseFilter;
use yatagarasu::analysis::token_filter::stop::StopWordFilter;

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
use yatagarasu::analysis::tokenizer::lindera::LinderaTokenizer;
use yatagarasu::analysis::analyzer::pipeline::PipelineAnalyzer;

// Japanese tokenization with Lindera
let tokenizer = LinderaTokenizer::japanese()?;
let analyzer = PipelineAnalyzer::new(Box::new(tokenizer));

let text = "東京は日本の首都です";
let tokens = analyzer.analyze(text)?;
```

## 📊 Performance

Yatagarasu is designed for high performance:

- **SIMD Acceleration** - Uses wide instruction sets for vector operations
- **Memory-Mapped I/O** - Efficient file access with minimal memory overhead
- **Incremental Updates** - Real-time document addition without full reindexing
- **Index Optimization** - Background merge operations for optimal search performance

## 🛠️ Development

### Building from Source

```bash
git clone https://github.com/mosuka/yatagarasu.git
cd yatagarasu
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

## 📖 Examples

Yatagarasu includes numerous examples demonstrating various features:

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

- [vector_search](examples/vector_search.rs) - Semantic text search using vector embeddings
- [embedding_with_candle](examples/embedding_with_candle.rs) - Local BERT model embeddings
- [embedding_with_openai](examples/embedding_with_openai.rs) - OpenAI API embeddings
- [dynamic_embedder_switching](examples/dynamic_embedder_switching.rs) - Switch between embedding providers
- [text_to_image_search](examples/text_to_image_search.rs) - Text-to-image search with CLIP
- [image_to_image_search](examples/image_to_image_search.rs) - Image similarity search

### Advanced Features

- [schemaless_indexing](examples/schemaless_indexing.rs) - Dynamic schema management
- [synonym_graph_filter](examples/synonym_graph_filter.rs) - Synonym expansion in queries
- [keyword_based_intent_classifier](examples/keyword_based_intent_classifier.rs) - Intent classification
- [ml_based_intent_classifier](examples/ml_based_intent_classifier.rs) - ML-powered intent detection
- [document_parser](examples/document_parser.rs) - Parsing various document formats
- [document_converter](examples/document_converter.rs) - Converting between document formats

Run any example with:

```bash
cargo run --example <example_name>

# For embedding examples, use feature flags:
cargo run --example vector_search --features embeddings-candle
cargo run --example embedding_with_openai --features embeddings-openai
cargo run --example text_to_image_search --features embeddings-multimodal
cargo run --example image_to_image_search --features embeddings-multimodal
```

## 🔧 Feature Flags

Yatagarasu uses feature flags to enable optional functionality:

```toml
[dependencies]
# Default features only
yatagarasu = "0.1"

# With Candle embeddings (local BERT models)
yatagarasu = { version = "0.1", features = ["embeddings-candle"] }

# With OpenAI embeddings
yatagarasu = { version = "0.1", features = ["embeddings-openai"] }

# With all embedding features
yatagarasu = { version = "0.1", features = ["embeddings-all"] }
```

Available features:

- `embeddings-candle` - Local text embeddings using Candle and BERT models
- `embeddings-openai` - OpenAI API-based text embeddings
- `embeddings-multimodal` - Multimodal embeddings (text and images) using CLIP models
- `embeddings-all` - All embedding providers

## 📚 Documentation

- [API Documentation](https://docs.rs/yatagarasu)
- [User Guide](https://github.com/mosuka/yatagarasu/wiki)
- [Examples](./examples/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under either of

- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
