//! Vector Search Example - Basic usage guide
//!
//! This example demonstrates the fundamental steps to use Sarissa for vector search:
//! 1. Setup storage
//! 2. Configure the vector index with an Embedder (CandleBertEmbedder via PerFieldEmbedder)
//! 3. Add documents with text content (vectors are generated automatically)
//! 4. Perform a nearest neighbor search (KNN) using text query
//!
//! To run this example:
//! ```bash
//! cargo run --example vector_search --features embeddings-candle
//! ```

#[cfg(feature = "embeddings-candle")]
use std::collections::HashMap;
#[cfg(feature = "embeddings-candle")]
use std::sync::Arc;

#[cfg(feature = "embeddings-candle")]
use sarissa::embedding::candle_bert_embedder::CandleBertEmbedder;
#[cfg(feature = "embeddings-candle")]
use sarissa::embedding::embedder::Embedder;
#[cfg(feature = "embeddings-candle")]
use sarissa::embedding::per_field::PerFieldEmbedder;
#[cfg(feature = "embeddings-candle")]
use sarissa::error::Result;
#[cfg(feature = "embeddings-candle")]
use sarissa::storage::file::FileStorageConfig;
#[cfg(feature = "embeddings-candle")]
use sarissa::storage::{StorageConfig, StorageFactory};
#[cfg(feature = "embeddings-candle")]
use sarissa::vector::DistanceMetric;
#[cfg(feature = "embeddings-candle")]
use sarissa::vector::core::document::DocumentPayload;
#[cfg(feature = "embeddings-candle")]
use sarissa::vector::engine::config::{VectorFieldConfig, VectorIndexConfig, VectorIndexKind};
#[cfg(feature = "embeddings-candle")]
use sarissa::vector::engine::{VectorEngine, VectorSearchRequestBuilder};
#[cfg(feature = "embeddings-candle")]
use tempfile::TempDir;

#[cfg(feature = "embeddings-candle")]
fn main() -> Result<()> {
    println!("=== Vector Search Example (Candle + PerFieldEmbedder) ===\n");

    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Embedder (CandleBertEmbedder wrapped in PerFieldEmbedder)
    // We use "sentence-transformers/all-MiniLM-L6-v2" which outputs 384-dimensional vectors.
    println!("Loading BERT model (this may take a while on first run)...");
    let candle_embedder = Arc::new(CandleBertEmbedder::new(
        "sentence-transformers/all-MiniLM-L6-v2",
    )?);

    // Create PerFieldEmbedder
    // We map both "title_vector" and "body_vector" to the same embedder instance.
    let mut per_field_embedder = PerFieldEmbedder::new(candle_embedder.clone());
    per_field_embedder.add_embedder("title_vector", candle_embedder.clone());
    per_field_embedder.add_embedder("body_vector", candle_embedder.clone());

    let embedder_arc: Arc<dyn Embedder> = Arc::new(per_field_embedder);

    // 3. Configure Index
    let field_config = VectorFieldConfig {
        dimension: 384, // Dimension for all-MiniLM-L6-v2
        distance: DistanceMetric::Cosine,
        index: VectorIndexKind::Flat,
        metadata: HashMap::new(),
        base_weight: 1.0,
    };

    let index_config = VectorIndexConfig::builder()
        .embedder_arc(embedder_arc)
        .field("title_vector", field_config.clone())
        .field("body_vector", field_config)
        .build()?;

    // 4. Create Engine
    let engine = VectorEngine::new(storage, index_config)?;

    // 5. Add Documents with Text
    // We use the same data as lexical_search.rs
    struct DocData<'a> {
        title: &'a str,
        body: &'a str,
        _category: &'a str,
    }

    let docs = vec![
        DocData {
            title: "The Rust Programming Language",
            body: "Rust is fast and memory efficient.",
            _category: "TECHNOLOGY",
        },
        DocData {
            title: "Learning Search Engines",
            body: "Search engines are complex but fascinating.",
            _category: "EDUCATION",
        },
        DocData {
            title: "Cooking with Rust (Iron Skillets)",
            body: "How to season your cast iron skillet.",
            _category: "LIFESTYLE",
        },
    ];

    println!("Indexing {} documents...", docs.len());
    for data in docs {
        let mut doc = DocumentPayload::new();
        // Use set_text to provide the raw text content for embedding
        doc.set_text("title_vector", data.title);
        doc.set_text("body_vector", data.body);

        let doc_id = engine.add_payloads(doc)?;
        println!("   Added Doc ID: {} -> Title: \"{}\"", doc_id, data.title);
    }
    engine.commit()?;

    // 6. Search
    // Demo 1: Search in 'title_vector'
    println!("\n--- Search 1: 'Rust' in 'title_vector' ---");
    let query_text = "Rust";

    let request = VectorSearchRequestBuilder::new()
        .add_text("title_vector", query_text)
        .limit(3)
        .build();

    let results = engine.search(request)?;

    println!("Found {} hits:", results.hits.len());
    for (i, hit) in results.hits.iter().enumerate() {
        println!("{}. Doc ID: {}, Score: {:.4}", i + 1, hit.doc_id, hit.score);
    }

    // Demo 2: Search in 'body_vector'
    println!("\n--- Search 2: 'season skillet' in 'body_vector' ---");
    let query_text_2 = "season skillet";

    let request_2 = VectorSearchRequestBuilder::new()
        .add_text("body_vector", query_text_2)
        .limit(3)
        .build();

    let results_2 = engine.search(request_2)?;

    println!("Found {} hits:", results_2.hits.len());
    for (i, hit) in results_2.hits.iter().enumerate() {
        println!("{}. Doc ID: {}, Score: {:.4}", i + 1, hit.doc_id, hit.score);
    }

    Ok(())
}

#[cfg(not(feature = "embeddings-candle"))]
fn main() {
    println!("This example requires the 'embeddings-candle' feature.");
    println!("Please run with: cargo run --example vector_search --features embeddings-candle");
}
