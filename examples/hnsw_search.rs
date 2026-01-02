//! HNSW Vector Search Example
//!
//! This example demonstrates how to use the HNSW (Hierarchical Navigable Small World) index
//! for fast approximate nearest neighbor search.
//!
//! The HNSW index offers significantly better search performance than Flat index for large datasets,
//! at the cost of slower indexing time and higher memory usage during construction.
//!
//! To run this example:
//! ```bash
//! cargo run --example hnsw_search --features embeddings-candle
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
use sarissa::vector::engine::VectorEngine;
#[cfg(feature = "embeddings-candle")]
use sarissa::vector::engine::config::{VectorFieldConfig, VectorIndexConfig, VectorIndexKind};
#[cfg(feature = "embeddings-candle")]
use sarissa::vector::engine::query::VectorSearchRequestBuilder;
#[cfg(feature = "embeddings-candle")]
use tempfile::TempDir;

#[cfg(feature = "embeddings-candle")]
fn main() -> Result<()> {
    println!("=== HNSW Vector Search Example ===\n");

    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Embedder
    println!("Loading BERT model...");
    let candle_embedder = Arc::new(CandleBertEmbedder::new(
        "sentence-transformers/all-MiniLM-L6-v2",
    )?);

    let mut per_field_embedder = PerFieldEmbedder::new(candle_embedder.clone());
    per_field_embedder.add_embedder("description_vector", candle_embedder.clone());

    let embedder_arc: Arc<dyn Embedder> = Arc::new(per_field_embedder);

    // 3. Configure Index with HNSW
    println!("Configuring HNSW index...");
    let field_config = VectorFieldConfig {
        dimension: 384,
        distance: DistanceMetric::Cosine,
        index: VectorIndexKind::Hnsw, // Use HNSW index
        metadata: HashMap::new(),
        base_weight: 1.0,
    };

    let index_config = VectorIndexConfig::builder()
        .embedder_arc(embedder_arc)
        .field("description_vector", field_config)
        .build()?;

    // 4. Create Engine
    let engine = VectorEngine::new(storage, index_config)?;

    // 5. Add Documents
    let docs = vec![
        "Artificial intelligence and machine learning are transforming industries.",
        "Deep learning models require significant computational power.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and understand the visual world.",
        "Reinforcement learning learns through trial and error interactions.",
    ];

    println!("Indexing {} documents with HNSW...", docs.len());
    for (_i, text) in docs.iter().enumerate() {
        let mut doc = DocumentPayload::new();
        doc.set_text("description_vector", *text);
        let doc_id = engine.add_payloads(doc)?;
        println!("   Indexed Doc ID: {}: \"{}\"", doc_id, text);
    }

    // HNSW graph is built upon finalization/commit
    engine.commit()?;

    // 6. Search
    println!("\n--- performing HNSW search for 'neural networks' ---");
    let query_text = "neural networks";

    let request = VectorSearchRequestBuilder::new()
        .add_text("description_vector", query_text)
        .limit(3)
        .build();

    let results = engine.search(request)?;

    println!("Found {} hits:", results.hits.len());
    for (i, hit) in results.hits.iter().enumerate() {
        println!("{}. Doc ID: {}, Score: {:.4}", i + 1, hit.doc_id, hit.score);
    }

    Ok(())
}

#[cfg(not(feature = "embeddings-candle"))]
fn main() {
    println!("This example requires the 'embeddings-candle' feature.");
    println!("Please run with: cargo run --example hnsw_search --features embeddings-candle");
}
