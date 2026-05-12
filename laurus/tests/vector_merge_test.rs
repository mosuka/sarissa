use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::DistanceMetric;
use laurus::vector::StoredVector;
use laurus::vector::VectorFieldConfig;
use laurus::vector::index::field::{VectorFieldReader, VectorFieldWriter};
use laurus::vector::index::hnsw::segment::manager::{SegmentManager, SegmentManagerConfig};
use laurus::vector::index::segmented_field::SegmentedVectorField;
use laurus::vector::{FieldOption, HnswOption};
use std::sync::Arc;

#[tokio::test]
async fn test_segmented_field_manual_merge() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup Storage and Manager with small constraints
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    let manager_config = SegmentManagerConfig {
        max_segments: 2,            // Trigger merge when > 2
        merge_factor: 2,            // Merge 2 segments at a time
        min_vectors_per_segment: 1, // Allow small segments
        ..Default::default()
    };

    let manager = Arc::new(SegmentManager::new(manager_config, storage.clone())?);

    // 2. Setup Field
    let field_config = VectorFieldConfig {
        vector: Some(FieldOption::Hnsw(HnswOption {
            dimension: 4,
            distance: DistanceMetric::Euclidean,
            m: 16, // Default or specific to test? Using defaults or standard values
            ef_construction: 200, // Standard default
            base_weight: 1.0,
            quantizer: Default::default(),
            embedder: None,
        })),
        lexical: None,
    };

    let field = SegmentedVectorField::create(
        "test_field",
        field_config,
        manager.clone(),
        storage.clone(),
        None,
    )?;

    // 3. Add vectors and flush to create segments
    // Segment 1
    field
        .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    // Segment 2
    field
        .add_stored_vector(2, &StoredVector::new(vec![0.0, 1.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    // Segment 3
    field
        .add_stored_vector(3, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    // Check we have 3 segments
    let segments = manager.list_segments();
    assert_eq!(segments.len(), 3, "Should have 3 segments before merge");

    // 4. Trigger Merge
    // We expect candidates to be found because 3 > max_segments (2).
    // Policy: SimpleMergePolicy sorts by size (all same size 1). Picks 2 smallest (or first 2).

    field.perform_merge()?;

    // 5. Verify Results
    let segments_after = manager.list_segments();
    // merged 2 segments -> 1. Total: 1 (new) + 1 (remaining) = 2.
    assert_eq!(
        segments_after.len(),
        2,
        "Should have 2 segments after merge"
    );

    // Verify Stats
    let stats = field.stats()?; // Should be 3 vectors total
    assert_eq!(stats.vector_count, 3);

    Ok(())
}
