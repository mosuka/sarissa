//! Vector Search Example - Basic usage guide
//!
//! This example demonstrates the fundamental steps to use Sarissa for vector search:
//! 1. Setup storage
//! 2. Configure the vector index (using NoOpEmbedder for direct vector input)
//! 3. Add documents with pre-computed vectors using the `add_payloads` API
//! 4. Perform a nearest neighbor search (KNN)

use std::sync::Arc;

use sarissa::embedding::precomputed::PrecomputedEmbedder;
use sarissa::error::Result;
use sarissa::storage::file::FileStorageConfig;
use sarissa::storage::{StorageConfig, StorageFactory};
use sarissa::vector::DistanceMetric;
use sarissa::vector::core::document::{DocumentPayload, Payload, PayloadSource, VectorType};
use sarissa::vector::engine::config::{VectorFieldConfig, VectorIndexConfig, VectorIndexKind};
use sarissa::vector::engine::{VectorEngine, VectorSearchRequestBuilder};
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Vector Search Basic Example ===\n");

    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Index
    // We use NoOpEmbedder because we will provide pre-computed vectors directly.
    let field_config = VectorFieldConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
        index: VectorIndexKind::Flat,
        loading_mode: sarissa::vector::index::config::IndexLoadingMode::InMemory,
        vector_type: VectorType::Text,
        base_weight: 1.0,
    };

    let index_config = VectorIndexConfig::builder()
        .embedder(PrecomputedEmbedder::new()) // Configure NoOpEmbedder
        .field("vector_data", field_config)
        .build()?;

    // 3. Create Engine
    let engine = VectorEngine::new(storage, index_config)?;

    // 4. Add Documents with Vectors
    // Even though we have raw vectors, we use the `add_payloads` API.
    // This is the standard way to add data, allowing us to easily switch to
    // an actual Embedder (e.g., OpenAI, Candle) later if needed.
    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    println!("Indexing {} vectors...", vectors.len());
    for (_i, vec_data) in vectors.iter().enumerate() {
        let mut doc = DocumentPayload::new();

        // Use PayloadSource::Vector to provide the raw vector data
        doc.set_field(
            "vector_data",
            Payload {
                source: PayloadSource::Vector {
                    data: Arc::<[f32]>::from(vec_data.as_slice()),
                },
                vector_type: VectorType::Text,
            },
        );

        // Use add_payloads instead of add_vectors
        let doc_id = engine.add_payloads(doc)?;
        println!("   Added Doc ID: {} -> {:?}", doc_id, vec_data);
    }
    engine.commit()?;

    // 5. Search
    println!("\nSearching for nearest neighbor to [0.9, 0.1, 0.0]:");

    let query_vector = vec![0.9, 0.1, 0.0];

    // New simplified API using VectorSearchRequestBuilder
    let request = VectorSearchRequestBuilder::new()
        .add_vector("vector_data", query_vector)
        .limit(3)
        .build();

    let results = engine.search(request)?;

    println!("Found {} hits:", results.hits.len());
    for (i, hit) in results.hits.iter().enumerate() {
        println!("{}. Doc ID: {}, Score: {:.4}", i + 1, hit.doc_id, hit.score);
    }

    Ok(())
}
