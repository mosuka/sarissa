use sarissa::storage::memory::{MemoryStorage, MemoryStorageConfig};
use sarissa::vector::DistanceMetric;
use sarissa::vector::core::document::StoredVector;
use sarissa::vector::engine::config::{VectorFieldConfig, VectorIndexKind};
use sarissa::vector::field::{VectorFieldReader, VectorFieldWriter};
use sarissa::vector::index::hnsw::segment::manager::{SegmentManager, SegmentManagerConfig};
use sarissa::vector::index::segmented_field::SegmentedVectorField;
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn test_segmented_field_manual_merge() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup Storage and Manager with small constraints
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    let mut manager_config = SegmentManagerConfig::default();
    manager_config.max_segments = 2; // Trigger merge when > 2
    manager_config.merge_factor = 2; // Merge 2 segments at a time
    manager_config.min_vectors_per_segment = 1; // Allow small segments

    let manager = Arc::new(SegmentManager::new(manager_config, storage.clone())?);

    // 2. Setup Field
    let field_config = VectorFieldConfig {
        dimension: 4,
        distance: DistanceMetric::Euclidean,
        index: VectorIndexKind::Hnsw,
        metadata: HashMap::new(),
        base_weight: 1.0, // Default weight
    };

    let field =
        SegmentedVectorField::create("test_field", field_config, manager.clone(), storage.clone())?;

    // 3. Add vectors and flush to create segments
    // Segment 1
    field.add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0].into()), 0)?;
    field.flush()?;

    // Segment 2
    field.add_stored_vector(2, &StoredVector::new(vec![0.0, 1.0, 0.0, 0.0].into()), 0)?;
    field.flush()?;

    // Segment 3
    field.add_stored_vector(3, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0].into()), 0)?;
    field.flush()?;

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
