use laurus::PrecomputedEmbedder;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};

use laurus::vector::DistanceMetric;
use laurus::vector::VectorSearchRequestBuilder;
use laurus::vector::VectorStore;
use laurus::vector::{FieldOption, FlatOption};
use laurus::vector::{VectorFieldConfig, VectorIndexConfig};
use laurus::{DataValue, Document};

use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test(flavor = "multi_thread")]
async fn test_mmap_mode_basic_search() {
    let dir = tempdir().unwrap();
    let storage_path = dir.path().to_owned();

    let storage_config = StorageConfig::File(FileStorageConfig::new(storage_path.clone()));
    let storage = StorageFactory::create(storage_config).unwrap();

    // Configure a fields with Mmap loading
    let field_config = VectorFieldConfig {
        vector: Some(FieldOption::Flat(FlatOption {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            base_weight: 1.0,
            quantizer: Default::default(),
            rerank_storage: None,
            embedder: None,
        })),
        lexical: None,
    };

    let config = VectorIndexConfig::builder()
        .embedder(Arc::new(PrecomputedEmbedder::new()))
        .field("mmap_field", field_config)
        .build()
        .unwrap();

    let engine = VectorStore::new(storage, config).unwrap();

    // Add vectors
    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    for (i, vec_data) in vectors.into_iter().enumerate() {
        let doc = Document::builder()
            .add_field("mmap_field", DataValue::Vector(vec_data))
            .build();
        engine
            .upsert_document_by_internal_id((i + 1) as u64, doc)
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();

    let query_vector = vec![1.0, 0.1, 0.0];
    let request = VectorSearchRequestBuilder::new()
        .add_vector("mmap_field", query_vector)
        .limit(2)
        .build();

    let results = engine.search(request).unwrap();

    assert_eq!(results.hits.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mmap_mode_persistence_reload() {
    let dir = tempdir().unwrap();
    let storage_path = dir.path().to_owned();

    {
        let storage_config = StorageConfig::File(FileStorageConfig::new(storage_path.clone()));
        let storage = StorageFactory::create(storage_config).unwrap();

        let field_config = VectorFieldConfig {
            vector: Some(FieldOption::Flat(FlatOption {
                dimension: 3,
                distance: DistanceMetric::Cosine,
                base_weight: 1.0,
                quantizer: Default::default(),
                rerank_storage: None,
                embedder: None,
            })),
            lexical: None,
        };

        let config = VectorIndexConfig::builder()
            .embedder(Arc::new(PrecomputedEmbedder::new()))
            .field("mmap_field", field_config)
            .build()
            .unwrap();

        let engine = VectorStore::new(storage, config).unwrap();

        let vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        for (i, vec_data) in vectors.into_iter().enumerate() {
            let doc = Document::builder()
                .add_field("mmap_field", DataValue::Vector(vec_data))
                .build();
            engine
                .upsert_document_by_internal_id((i + 1) as u64, doc)
                .await
                .unwrap();
        }
        engine.commit().await.unwrap();
    }

    // Re-open
    {
        let storage_config = StorageConfig::File(FileStorageConfig::new(storage_path.clone()));
        let storage = StorageFactory::create(storage_config).unwrap();

        let field_config = VectorFieldConfig {
            vector: Some(FieldOption::Flat(FlatOption {
                dimension: 3,
                distance: DistanceMetric::Cosine,
                base_weight: 1.0,
                quantizer: Default::default(),
                rerank_storage: None,
                embedder: None,
            })),
            lexical: None,
        };

        let config = VectorIndexConfig::builder()
            .embedder(Arc::new(PrecomputedEmbedder::new()))
            .field("mmap_field", field_config)
            .build()
            .unwrap();

        let engine = VectorStore::new(storage, config).unwrap();

        // IMPORTANT: In Mmap mode, vectors are LOADED from file on demand.
        // If file persistence works, search should find them.

        let query_vector = vec![0.0, 1.0, 0.0];
        let request = VectorSearchRequestBuilder::new()
            .add_vector("mmap_field", query_vector)
            .limit(1)
            .build();

        let results = engine.search(request).unwrap();

        assert_eq!(results.hits.len(), 1);
        // We expect it to match the second vector (doc_id=2, vector [0,1,0]).
        assert_eq!(
            results.hits[0].doc_id, 2,
            "Top result should be doc_id=2 (exact match for [0,1,0])"
        );
    }
}
