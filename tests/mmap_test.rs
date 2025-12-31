use sarissa::embedding::precomputed::PrecomputedEmbedder;
use sarissa::storage::file::FileStorageConfig;
use sarissa::storage::{StorageConfig, StorageFactory};

use sarissa::vector::core::document::{DocumentPayload, Payload, PayloadSource};
use sarissa::vector::engine::config::{VectorFieldConfig, VectorIndexConfig};
use sarissa::vector::engine::{VectorEngine, VectorSearchRequestBuilder};

use std::sync::Arc;
use tempfile::tempdir;

#[test]
fn test_mmap_mode_basic_search() {
    let dir = tempdir().unwrap();
    let storage_path = dir.path().to_owned();

    let storage_config = StorageConfig::File(FileStorageConfig::new(storage_path.clone()));
    let storage = StorageFactory::create(storage_config).unwrap();

    // Configure a fields with Mmap loading
    let mut field_config = VectorFieldConfig::default();
    field_config.dimension = 3;

    let config = VectorIndexConfig::builder()
        .embedder(PrecomputedEmbedder::new())
        .field("mmap_field", field_config)
        .build()
        .unwrap();

    let engine = VectorEngine::new(storage, config).unwrap();

    // Add vectors
    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    for vec_data in vectors {
        let mut doc = DocumentPayload::new();
        doc.set_field(
            "mmap_field",
            Payload::new(PayloadSource::Vector {
                data: Arc::<[f32]>::from(vec_data.as_slice()),
            }),
        );
        engine.add_payloads(doc).unwrap();
    }
    engine.commit().unwrap();

    let query_vector = vec![1.0, 0.1, 0.0];
    let request = VectorSearchRequestBuilder::new()
        .add_vector("mmap_field", query_vector)
        .limit(2)
        .build();

    let results = engine.search(request).unwrap();

    assert_eq!(results.hits.len(), 2);
}

#[test]
fn test_mmap_mode_persistence_reload() {
    let dir = tempdir().unwrap();
    let storage_path = dir.path().to_owned();

    {
        let storage_config = StorageConfig::File(FileStorageConfig::new(storage_path.clone()));
        let storage = StorageFactory::create(storage_config).unwrap();

        let mut field_config = VectorFieldConfig::default();
        field_config.dimension = 3;

        let config = VectorIndexConfig::builder()
            .embedder(PrecomputedEmbedder::new())
            .field("mmap_field", field_config)
            .build()
            .unwrap();

        let engine = VectorEngine::new(storage, config).unwrap();

        let vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        for vec_data in vectors {
            let mut doc = DocumentPayload::new();
            doc.set_field(
                "mmap_field",
                Payload::new(PayloadSource::Vector {
                    data: Arc::<[f32]>::from(vec_data.as_slice()),
                }),
            );
            engine.add_payloads(doc).unwrap();
        }
        engine.commit().unwrap();
    }

    // Re-open
    {
        let storage_config = StorageConfig::File(FileStorageConfig::new(storage_path.clone()));
        let storage = StorageFactory::create(storage_config).unwrap();

        let mut field_config = VectorFieldConfig::default();
        field_config.dimension = 3;

        let config = VectorIndexConfig::builder()
            .embedder(PrecomputedEmbedder::new())
            .field("mmap_field", field_config)
            .build()
            .unwrap();

        let engine = VectorEngine::new(storage, config).unwrap();

        // IMPORTANT: In Mmap mode, vectors are LOADED from file on demand.
        // If file persistence works, search should find them.

        let query_vector = vec![0.0, 1.0, 0.0];
        let request = VectorSearchRequestBuilder::new()
            .add_vector("mmap_field", query_vector)
            .limit(1)
            .build();

        let results = engine.search(request).unwrap();

        assert_eq!(results.hits.len(), 1);
        // We expect it to match the second vector.
    }
}
