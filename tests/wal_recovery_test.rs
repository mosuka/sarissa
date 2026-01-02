use sarissa::storage::memory::{MemoryStorage, MemoryStorageConfig};
use sarissa::vector::DistanceMetric;
use sarissa::vector::core::document::StoredVector;
use sarissa::vector::engine::{QueryVector, VectorFieldConfig, VectorIndexKind};
use sarissa::vector::field::{FieldSearchInput, VectorFieldReader, VectorFieldWriter};
use sarissa::vector::index::hnsw::segment::manager::{SegmentManager, SegmentManagerConfig};
use sarissa::vector::index::segmented_field::SegmentedVectorField;
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn test_wal_recovery_unflushed_data() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup Storage (shared across "restarts")
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let _wal_path = "wal_test_field.log";

    // 2. "First Run": Create field, add data, but DO NOT FLUSH
    {
        let manager_config = SegmentManagerConfig::default();
        let manager = Arc::new(SegmentManager::new(manager_config, storage.clone())?);

        let field_config = VectorFieldConfig {
            dimension: 4,
            distance: DistanceMetric::Euclidean,
            index: VectorIndexKind::Hnsw,
            metadata: HashMap::new(),
            base_weight: 1.0,
        };

        let field = SegmentedVectorField::create(
            "test_field", // Name determines WAL filename
            field_config,
            manager.clone(),
            storage.clone(),
        )?;

        // Add 3 vectors
        for i in 1..=3 {
            let vec_data = vec![i as f32, 0.0, 0.0, 0.0];
            field.add_stored_vector(
                i,
                &StoredVector::new(Arc::from(vec_data)),
                0, // timestamp
            )?;
        }

        // Verify in-memory count
        let stats = field.stats()?;
        assert_eq!(stats.vector_count, 3, "Should have 3 vectors in memory");

        // NO FLUSH here. We drop `field` and `manager`.
        // WAL should have recorded these 3 inserts.
    }

    // 3. "Restart": Re-create field using SAME storage
    {
        let manager_config = SegmentManagerConfig::default();
        let manager = Arc::new(SegmentManager::new(manager_config, storage.clone())?);

        let field_config = VectorFieldConfig {
            dimension: 4,
            distance: DistanceMetric::Euclidean,
            index: VectorIndexKind::Hnsw,
            metadata: HashMap::new(),
            base_weight: 1.0,
        };

        // This `create` call should trigger WAL replay
        let field = SegmentedVectorField::create(
            "test_field",
            field_config,
            manager.clone(),
            storage.clone(),
        )?;

        // 4. Verify Recovery
        let stats = field.stats()?;
        assert_eq!(
            stats.vector_count, 3,
            "Should have recovered 3 vectors from WAL"
        );

        // Verify search
        let query_vec = vec![1.0, 0.0, 0.0, 0.0];
        let qv = QueryVector {
            vector: StoredVector::new(Arc::from(query_vec)),
            weight: 1.0,
            fields: None,
        };

        let results = field.search(FieldSearchInput {
            field: "test_field".to_string(),
            query_vectors: vec![qv],
            limit: 3,
        })?;

        assert!(!results.hits.is_empty(), "Should find hits after recovery");
        assert_eq!(results.hits[0].doc_id, 1, "First doc should be ID 1");

        // 5. Add more data and FLUSH
        field.add_stored_vector(
            4,
            &StoredVector::new(Arc::from(vec![4.0, 0.0, 0.0, 0.0])),
            0,
        )?;

        // Flush (writes segment, truncates WAL)
        field.flush()?;

        // Verify updated count
        let stats = field.stats()?;
        assert_eq!(stats.vector_count, 4, "Total vectors should be 4");

        // Verify WAL is truncated (empty or minimal)
        // We can't easily check file size via `field.wal`, but we can verify subsequent restart.
    }

    // 6. "Second Restart": Verify flush persistence and clean WAL
    {
        let manager_config = SegmentManagerConfig::default();
        let manager = Arc::new(SegmentManager::new(manager_config, storage.clone())?);

        let field_config = VectorFieldConfig {
            dimension: 4,
            distance: DistanceMetric::Euclidean,
            index: VectorIndexKind::Hnsw,
            metadata: HashMap::new(),
            base_weight: 1.0,
        };

        // Should load persisted segment (3 vectors? no wait, flush writes ALL vectors from active segment?)
        // `SegmentedVectorField::flush` takes active segment, finalizes it, and registers it.
        // So after flush, active segment is None/Empty.
        // The persisted segment contains vectors 1, 2, 3, 4.

        let field = SegmentedVectorField::create(
            "test_field",
            field_config,
            manager.clone(),
            storage.clone(),
        )?;

        let stats = field.stats()?;
        assert_eq!(stats.vector_count, 4, "Should load 4 flushed vectors");
    }

    Ok(())
}
