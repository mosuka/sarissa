use std::collections::HashMap;
use std::sync::Arc;
use tempfile::Builder;

use sarissa::storage::file::FileStorageConfig;
use sarissa::storage::{StorageConfig, StorageFactory};
use sarissa::vector::collection::VectorCollection;
use sarissa::vector::core::distance::DistanceMetric;
use sarissa::vector::core::document::{DocumentVector, StoredVector};
use sarissa::vector::engine::config::{VectorFieldConfig, VectorIndexConfig, VectorIndexKind};
use sarissa::vector::engine::request::{VectorScoreMode, VectorSearchRequest};

#[test]
fn test_vacuum_reduces_file_size() {
    let dir = Builder::new().prefix("test_vacuum").tempdir().unwrap();
    let path = dir.path().to_path_buf();

    // 1. Create VectorCollection using FileStorage
    let field_config = VectorFieldConfig {
        dimension: 128,
        distance: DistanceMetric::Cosine,
        index: VectorIndexKind::Flat,
        metadata: HashMap::new(),
        base_weight: 1.0,
    };

    let config = VectorIndexConfig {
        fields: HashMap::from([("vectors".to_string(), field_config)]),
        default_fields: vec!["vectors".to_string()],
        metadata: HashMap::new(),
        default_distance: DistanceMetric::Cosine,
        default_dimension: None,
        default_index_kind: VectorIndexKind::Flat,
        default_base_weight: 1.0,
        implicit_schema: false,
        embedder: Arc::new(sarissa::embedding::precomputed::PrecomputedEmbedder::new()),
    };

    // Correctly construct FileStorageConfig
    // Note: FileStorageConfig::new(path) might be the way, or struct init.
    // Based on storage.rs doc example: let mut file_config = FileStorageConfig::new("/tmp/test_index");
    let file_config = FileStorageConfig::new(path.to_str().unwrap());

    let storage_config = StorageConfig::File(file_config);
    let storage = StorageFactory::create(storage_config).unwrap();

    let collection = VectorCollection::new(config, storage, None).unwrap();

    let dim = 128;
    let num_vectors = 200;

    // 2. Insert vectors
    println!("Inserting {} vectors...", num_vectors);
    for i in 0..num_vectors {
        let mut doc_vector = DocumentVector::new();
        doc_vector.set_field("vectors", StoredVector::new(Arc::from(vec![0.1f32; dim])));
        collection.upsert_document(i, doc_vector).unwrap();
    }

    println!("Flushing vectors to disk...");
    collection.flush_vectors().unwrap();
    collection.commit().unwrap();
    println!("committed.");

    // Check file size
    // Path: {root}/vector_fields/{field_name}/vectors.index.flat
    // VectorCollection uses "vector_fields/{sanitized_field_name}" as storage prefix.
    // FlatIndexWriter appends ".flat".
    let index_file_path = path
        .join("vector_fields")
        .join("vectors")
        .join("vectors.index.flat");

    assert!(
        index_file_path.exists(),
        "Index file should exist after commit: {:?}",
        index_file_path
    );
    let size_before = std::fs::metadata(&index_file_path).unwrap().len();
    println!("Size before deletion: {} bytes", size_before);

    // 3. Delete 100 vectors (even IDs)
    println!("Deleting {} vectors...", num_vectors / 2);
    for i in 0..num_vectors {
        if i % 2 == 0 {
            collection.delete_document(i).unwrap();
        }
    }
    collection.commit().unwrap();

    let size_intermediate = std::fs::metadata(&index_file_path).unwrap().len();
    println!(
        "Size after delete (before optimize): {} bytes",
        size_intermediate
    );

    // 4. Run Vacuum
    println!("Running optimize (Vacuum)...");
    collection.optimize().unwrap();

    let size_after = std::fs::metadata(&index_file_path).unwrap().len();
    println!("Size after optimize: {} bytes", size_after);

    assert!(
        size_after < size_before,
        "Size should decrease after vacuum. Before: {}, After: {}",
        size_before,
        size_after
    );
    assert!(
        size_after < (size_before as f64 * 0.7) as u64,
        "Size should be roughly half (allow some metadata overhead)"
    );

    // 5. Verify Search
    // Deleted (even) should not match. Odd should match.
    let request = VectorSearchRequest {
        query_vectors: vec![sarissa::vector::engine::request::QueryVector {
            vector: StoredVector::new(Arc::from(vec![0.1f32; dim])),
            weight: 1.0,
            fields: None,
        }],
        limit: num_vectors as usize,
        overfetch: 1.0,
        min_score: 0.0,
        score_mode: VectorScoreMode::MaxSim,
        filter: None,
        fields: None,
        query_payloads: vec![],
    };

    let searcher = collection.searcher().unwrap();
    let results = searcher.search(&request).unwrap();

    // Expect 100 hits
    assert_eq!(results.hits.len(), 100, "Should have 100 hits left");

    // Verify none are even
    for hit in results.hits {
        assert!(
            hit.doc_id % 2 != 0,
            "Deleted document {} found in search results",
            hit.doc_id
        );
    }
}
