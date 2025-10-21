# Sage

[![Crates.io](https://img.shields.io/crates/v/sage.svg)](https://crates.io/crates/sage)
[![Documentation](https://docs.rs/sage/badge.svg)](https://docs.rs/sage)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fast, featureful full-text search library for Rust, inspired by the [Lucene](https://github.com/apache/lucene) and Lucene alternatives.

## ✨ Features

- **Pure Rust Implementation** - Memory-safe and fast performance
- **Flexible Text Analysis** - Configurable tokenization, stemming, and filtering pipeline
- **Multiple Storage Backends** - File system, memory-mapped files, and in-memory storage
- **Advanced Query Types** - Term, phrase, range, boolean, fuzzy, wildcard, and geographic queries
- **Vector Search** - HNSW-based approximate nearest neighbor search with multiple distance metrics
- **Text Embeddings** - Built-in support for generating embeddings with Candle (local BERT models) and OpenAI
- **Multimodal Search** - Cross-modal search with CLIP models for text-to-image and image-to-image similarity
- **BM25 Scoring** - Industry-standard relevance scoring with customizable parameters
- **Spell Correction** - Built-in spell checking and query suggestion system
- **Faceted Search** - Multi-dimensional search with facet aggregation
- **Real-time Search** - Near real-time search with background index optimization
- **SIMD Acceleration** - Optimized vector operations for improved performance

## 🚀 Quick Start

Add Sage to your `Cargo.toml`:

```toml
[dependencies]
sage = "0.1"
```

### Basic Usage

```rust
use sage::prelude::*;
use tempfile::TempDir;

fn main() -> Result<()> {
    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    
    // Define a schema
    let mut schema = Schema::new();
    schema.add_field("title", Box::new(TextField::new().stored(true).indexed(true)))?;
    schema.add_field("content", Box::new(TextField::new().indexed(true)))?;
    
    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(
        temp_dir.path(), 
        schema, 
        IndexConfig::default()
    )?;
    
    // Add documents
    let documents = vec![
        Document::builder()
            .add_text("title", "Rust Programming")
            .add_text("content", "Rust is a systems programming language")
            .build(),
        Document::builder()
            .add_text("title", "Python Guide") 
            .add_text("content", "Python is a versatile programming language")
            .build(),
    ];
    
    engine.add_documents(documents)?;
    engine.commit()?;
    
    // Search documents
    let query = TermQuery::new("content".to_string(), "programming".to_string());
    let results = engine.search(&query, 10)?;
    
    println!("Found {} matches", results.total_hits);
    for hit in results.hits {
        println!("Score: {:.2}, Document: {:?}", hit.score, hit.document);
    }
    
    Ok(())
}
```

## 🏗️ Architecture

Sage is built with a modular architecture:

### Core Components

- **Schema & Fields** - Define document structure with typed fields (text, numeric, boolean, geographic)
- **Analysis Pipeline** - Configurable text processing with tokenizers, filters, and stemmers
- **Storage Layer** - Pluggable storage backends with transaction support
- **Index Structure** - Inverted index with posting lists and term dictionaries
- **Query Engine** - Flexible query system supporting multiple query types
- **Search Engine** - High-level interface combining indexing and search operations

### Field Types

```rust
// Text field for full-text search
TextField::new().stored(true).indexed(true)

// Numeric field for range queries
NumericField::new().indexed(true)

// Boolean field for filtering
BooleanField::new().indexed(true)

// ID field for exact matches
IdField::new()

// Geographic field for spatial queries  
GeoField::new().indexed(true)

// Vector field for similarity search
VectorField::new(128).indexed(true) // 128-dimensional vectors
```

### Query Types

```rust
// Term query
TermQuery::new("field".to_string(), "term".to_string())

// Phrase query
PhraseQuery::new("field".to_string(), vec!["hello".to_string(), "world".to_string()])

// Range query
RangeQuery::new("price".to_string(), Some(100.0), Some(500.0))

// Boolean query
BooleanQuery::new()
    .add_must(TermQuery::new("category".to_string(), "book".to_string()))
    .add_should(TermQuery::new("author".to_string(), "tolkien".to_string()))

// Fuzzy query
FuzzyQuery::new("title".to_string(), "progamming".to_string(), 2) // max edit distance: 2

// Wildcard query  
WildcardQuery::new("filename".to_string(), "*.pdf".to_string())

// Geographic query
GeoQuery::within_radius("location".to_string(), 40.7128, -74.0060, 10.0) // NYC, 10km radius
```

## 🎯 Advanced Features

### Vector Search with Text Embeddings

Sage supports semantic search using text embeddings. You can use local BERT models via Candle or OpenAI's API.

#### Using Candle (Local BERT Models)

```toml
[dependencies]
sage = { version = "0.1", features = ["embeddings-candle"] }
```

```rust
use sage::embedding::{CandleTextEmbedder, TextEmbedder};
use sage::vector::*;

// Initialize embedder with a sentence-transformers model
let embedder = CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?;

// Generate embeddings for documents
let documents = vec![
    "Rust is a systems programming language",
    "Python is great for data science",
    "Machine learning with neural networks",
];

let vectors = embedder.embed_batch(&documents).await?;

// Build vector index
let config = VectorIndexBuildConfig {
    dimension: embedder.dimension(),
    distance_metric: DistanceMetric::Cosine,
    index_type: VectorIndexType::Flat,
    normalize_vectors: true,
    ..Default::default()
};

let mut builder = VectorIndexBuilderFactory::create_builder(config)?;
let doc_vectors: Vec<(u64, Vector)> = documents
    .iter()
    .enumerate()
    .zip(vectors.iter())
    .map(|((idx, _), vec)| (idx as u64, vec.clone()))
    .collect();

builder.add_vectors(doc_vectors)?;
builder.finalize()?;

// Search with query embedding
let query_vector = embedder.embed("programming languages").await?;
// Perform similarity search...
```

#### Using OpenAI Embeddings

```toml
[dependencies]
sage = { version = "0.1", features = ["embeddings-openai"] }
```

```rust
use sage::embedding::{OpenAIEmbedder, TextEmbedder};

// Initialize with API key
let embedder = OpenAIEmbedder::new(
    "your-api-key",
    "text-embedding-3-small"
)?;

// Generate embeddings
let vector = embedder.embed("your text here").await?;
```

### Multimodal Search (Text + Images)

Sage supports cross-modal search using CLIP (Contrastive Language-Image Pre-Training) models, enabling semantic search across text and images. This allows you to:

- **Text-to-Image Search**: Find images using natural language queries
- **Image-to-Image Search**: Find visually similar images using an image query
- **Semantic Understanding**: Search based on content meaning, not just keywords

#### Setup

Add the `embeddings-multimodal` feature to your `Cargo.toml`:

```toml
[dependencies]
sage = { version = "0.1", features = ["embeddings-multimodal"] }
```

#### Text-to-Image Search Example

```rust
use sage::embedding::{CandleMultimodalEmbedder, TextEmbedder, ImageEmbedder};
use sage::vector::index::{VectorIndexBuildConfig, VectorIndexBuilderFactory};

// Initialize CLIP embedder (automatically downloads model from HuggingFace)
let embedder = CandleMultimodalEmbedder::new("openai/clip-vit-base-patch32")?;

// Create vector index with CLIP's embedding dimension (512)
let config = VectorIndexBuildConfig {
    dimension: embedder.dimension(), // 512 for CLIP ViT-Base-Patch32
    distance_metric: DistanceMetric::Cosine,
    index_type: VectorIndexType::HNSW,
    ..Default::default()
};
let mut builder = VectorIndexBuilderFactory::create_builder(config)?;

// Index your image collection
let mut image_vectors = Vec::new();
for (id, image_path) in image_paths.iter().enumerate() {
    let vector = embedder.embed_image(image_path).await?;
    image_vectors.push((id as u64, vector));
}
builder.add_vectors(image_vectors)?;
let index = builder.finalize()?;

// Search images using natural language
let query_vector = embedder.embed("a photo of a cat playing").await?;
let results = index.search(&query_vector, 10)?;
```

#### Image-to-Image Search Example

```rust
// Find visually similar images using an image as query
let query_image_vector = embedder.embed_image("query.jpg").await?;
let similar_images = index.search(&query_image_vector, 5)?;
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
use sage::search::facet::*;

// Configure faceted search
let mut search_request = SearchRequest::new(query)
    .add_facet("category".to_string(), FacetRequest::terms(10))
    .add_facet("price".to_string(), FacetRequest::range(vec![
        (0.0, 50.0),
        (50.0, 100.0), 
        (100.0, f64::INFINITY)
    ]));

let results = engine.search_with_facets(&search_request)?;

// Access facet results
for facet in results.facets {
    println!("Facet: {}", facet.field);
    for bucket in facet.buckets {
        println!("  {}: {} documents", bucket.label, bucket.count);
    }
}
```

### Spell Correction

```rust
use sage::spelling::*;

// Create spell corrector
let corrector = SpellCorrector::new()
    .max_edit_distance(2)
    .min_word_frequency(5);

// Check and suggest corrections
if let Some(suggestion) = corrector.suggest("progamming")? {
    println!("Did you mean: '{}'?", suggestion.suggestion);
}
```

### Custom Analysis Pipeline

```rust
use sage::analysis::*;

// Create custom analyzer
let analyzer = Analyzer::new()
    .tokenizer(Box::new(RegexTokenizer::new(r"\w+")?))
    .add_filter(Box::new(LowercaseFilter::new()))
    .add_filter(Box::new(StopWordFilter::english()))
    .add_filter(Box::new(PorterStemmer::new()));

// Use in field definition
let field = TextField::new()
    .analyzer(analyzer)
    .stored(true)
    .indexed(true);
```

## 📊 Performance

Sage is designed for high performance:

- **SIMD Acceleration** - Uses wide instruction sets for vector operations
- **Memory-Mapped I/O** - Efficient file access with minimal memory overhead
- **Parallel Processing** - Multi-threaded indexing and search operations
- **Incremental Updates** - Real-time document addition without full reindexing
- **Index Optimization** - Background merge operations for optimal search performance

### Benchmarks

On a modern machine with SSD storage:

- **Indexing**: ~50,000 documents/second
- **Search**: ~100,000 queries/second  
- **Memory Usage**: ~50MB per 1M documents
- **Index Size**: ~60% of original document size

## 🛠️ Development

### Building from Source

```bash
git clone https://github.com/mosuka/sage.git
cd sage
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

Sage includes numerous examples demonstrating various features:

### Lexical Search Examples

- `term_query` - Basic term-based search
- `phrase_query` - Multi-word phrase matching
- `boolean_query` - Combining queries with AND/OR/NOT logic
- `fuzzy_query` - Fuzzy string matching with edit distance
- `wildcard_query` - Pattern matching with wildcards
- `range_query` - Numeric and date range queries
- `geo_query` - Geographic location-based search
- `field_specific_search` - Search within specific fields
- `lexical_search` - Full lexical search example
- `query_parser` - Parsing user query strings

### Vector Search Examples

- `vector_search` - Semantic text search using CandleTextEmbedder
- `embedding_with_candle` - Local BERT model embeddings
- `embedding_with_openai` - OpenAI API embeddings
- `dynamic_embedder_switching` - Switch between embedding providers

### Advanced Features

- `parallel_search` - Parallel search execution
- `schemaless_indexing` - Dynamic schema management
- `synonym_graph_filter` - Synonym expansion in queries
- `keyword_based_intent_classifier` - Intent classification
- `ml_based_intent_classifier` - ML-powered intent detection
- `document_parser` - Parsing various document formats
- `document_converter` - Converting between document formats

Run any example with:

```bash
cargo run --example <example_name>

# For embedding examples, use feature flags:
cargo run --example vector_search --features embeddings-candle
cargo run --example embedding_with_openai --features embeddings-openai
```

## 🔧 Feature Flags

Sage uses feature flags to enable optional functionality:

```toml
[dependencies]
# Default features only
sage = "0.1"

# With Candle embeddings (local BERT models)
sage = { version = "0.1", features = ["embeddings-candle"] }

# With OpenAI embeddings
sage = { version = "0.1", features = ["embeddings-openai"] }

# With all embedding features
sage = { version = "0.1", features = ["embeddings-all"] }
```

Available features:

- `embeddings-candle` - Local text embeddings using Candle and BERT models
- `embeddings-openai` - OpenAI API-based text embeddings
- `embeddings-all` - All embedding providers

## 📚 Documentation

- [API Documentation](https://docs.rs/sage)
- [User Guide](https://github.com/mosuka/sage/wiki)
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

## 🙏 Acknowledgments

- Inspired by the [Lucene](https://github.com/apache/lucene) and Lucene alternatives.
- Built with the excellent Rust ecosystem

## 📧 Contact

- **Author**: [mosuka](https://github.com/mosuka)
- **Repository**: <https://github.com/mosuka/sage>
- **Issues**: <https://github.com/mosuka/sage/issues>

---

*Sage - Fast, featureful full-text search for Rust* 🦀
