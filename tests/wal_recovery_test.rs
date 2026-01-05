use sarissa::embedding::precomputed::PrecomputedEmbedder;
use sarissa::error::Result;
use sarissa::storage::Storage;
use sarissa::storage::memory::{MemoryStorage, MemoryStorageConfig};
use sarissa::vector::DistanceMetric;
use sarissa::vector::core::document::{DocumentVector, StoredVector};
use sarissa::vector::engine::VectorEngine;
use sarissa::vector::engine::config::{VectorFieldConfig, VectorIndexConfig, VectorIndexKind};
use sarissa::vector::engine::request::{FieldSelector, QueryVector, VectorSearchRequest};
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn test_wal_recovery_unflushed_data() -> Result<()> {
    // 1. Setup Storage (shared across "restarts")
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    // 2. "First Run": Create engine, add data, but DO NOT FLUSH
    {
        let engine = create_test_engine(storage.clone())?;

        // Add 3 documents
        for i in 1..=3 {
            let mut doc = DocumentVector::new();
            let vec_data = vec![i as f32, 0.0, 0.0, 0.0];
            doc.fields.insert(
                "test_field".to_string(),
                StoredVector::new(Arc::from(vec_data)),
            );
            engine.upsert_vectors(i, doc)?;
        }

        // Verify in-memory count
        let stats = engine.stats()?;
        assert_eq!(stats.document_count, 3, "Should have 3 docs in memory");

        // NO FLUSH (commit) here, just drop engine.
        // WAL should have recorded these 3 inserts.
    }

    // 3. "Restart": Re-create engine using SAME storage
    {
        // This `new` call should trigger WAL replay in VectorEngine
        let engine = create_test_engine(storage.clone())?;

        // 4. Verify Recovery
        let stats = engine.stats()?;
        assert_eq!(
            stats.document_count, 3,
            "Should have recovered 3 docs from WAL"
        );

        // Verify search
        let mut request = VectorSearchRequest::default();
        request.limit = 3;
        request.fields = Some(vec![FieldSelector::Exact("test_field".into())]);
        request.query_vectors.push(QueryVector {
            vector: StoredVector::new(Arc::from(vec![1.0, 0.0, 0.0, 0.0])),
            weight: 1.0,
            fields: None,
        });

        let results = engine.search(request)?;
        assert!(!results.hits.is_empty(), "Should find hits after recovery");
        assert_eq!(results.hits[0].doc_id, 1, "First doc should be ID 1");

        // 5. Add more data and COMMIT (flush)
        let mut doc = DocumentVector::new();
        doc.fields.insert(
            "test_field".to_string(),
            StoredVector::new(Arc::from(vec![4.0, 0.0, 0.0, 0.0])),
        );
        engine.upsert_vectors(4, doc)?;

        // Commit persists snapshot and truncates WAL
        engine.commit()?;

        // Verify updated count
        let stats = engine.stats()?;
        assert_eq!(stats.document_count, 4, "Total docs should be 4");
    }

    // 6. "Second Restart": Verify snapshot persistence and clean WAL
    {
        let engine = create_test_engine(storage.clone())?;

        let stats = engine.stats()?;
        assert_eq!(stats.document_count, 4, "Should load 4 persisted docs");

        // Verify that we can still search
        let mut request = VectorSearchRequest::default();
        request.limit = 1;
        request.fields = Some(vec![FieldSelector::Exact("test_field".into())]);
        request.query_vectors.push(QueryVector {
            vector: StoredVector::new(Arc::from(vec![4.0, 0.0, 0.0, 0.0])),
            weight: 1.0,
            fields: None,
        });
        let results = engine.search(request)?;
        assert_eq!(results.hits[0].doc_id, 4);
    }

    Ok(())
}

fn create_test_engine(storage: Arc<dyn Storage>) -> Result<VectorEngine> {
    let mut builder = VectorIndexConfig::builder()
        .default_fields(vec!["test_field".into()])
        .default_distance(DistanceMetric::Euclidean)
        .default_index_kind(VectorIndexKind::Hnsw)
        .embedder(PrecomputedEmbedder::new());

    builder = builder.field(
        "test_field",
        VectorFieldConfig {
            dimension: 4,
            distance: DistanceMetric::Euclidean,
            index: VectorIndexKind::Hnsw,
            metadata: HashMap::new(),
            base_weight: 1.0,
        },
    );

    let config = builder.build()?;
    VectorEngine::new(storage, config)
}
