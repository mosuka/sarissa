use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::DistanceMetric;
use laurus::vector::StoredVector;
use laurus::vector::Vector;
use laurus::vector::VectorFieldConfig;
use laurus::vector::index::field::{FieldSearchInput, VectorFieldReader, VectorFieldWriter};
use laurus::vector::index::hnsw::segment::manager::{SegmentManager, SegmentManagerConfig};
use laurus::vector::index::segmented_field::SegmentedVectorField;
use laurus::vector::store::request::QueryVector;
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
            default_ef_search: None,
            base_weight: 1.0,
            quantizer: Default::default(),
            rerank_storage: None,
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

/// Regression test for Issue [#660]: `SegmentedVectorField` must reuse
/// a cached `Arc<HnswIndexReader>` across queries against the same
/// segment, and must invalidate the cache entry when the segment is
/// removed via merge.
///
/// [#660]: https://github.com/mosuka/laurus/issues/660
#[tokio::test]
async fn segmented_field_reader_cache_reuses_and_invalidates()
-> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    let manager_config = SegmentManagerConfig {
        max_segments: 2,
        merge_factor: 2,
        min_vectors_per_segment: 1,
        ..Default::default()
    };
    let manager = Arc::new(SegmentManager::new(manager_config, storage.clone())?);

    let field_config = VectorFieldConfig {
        vector: Some(FieldOption::Hnsw(HnswOption {
            dimension: 4,
            distance: DistanceMetric::Euclidean,
            m: 16,
            ef_construction: 200,
            default_ef_search: None,
            base_weight: 1.0,
            quantizer: Default::default(),
            rerank_storage: None,
            embedder: None,
        })),
        lexical: None,
    };

    let field = SegmentedVectorField::create(
        "embedding",
        field_config,
        manager.clone(),
        storage.clone(),
        None,
    )?;

    // Flush three single-vector segments. After this the manager owns
    // 3 segments and the reader cache is still empty (no searches yet).
    field
        .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;
    field
        .add_stored_vector(2, &StoredVector::new(vec![0.0, 1.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;
    field
        .add_stored_vector(3, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    assert_eq!(manager.list_segments().len(), 3);
    assert!(
        field.reader_cache.is_empty(),
        "cache should be empty before the first search"
    );

    let segment_ids: Vec<String> = manager
        .list_segments()
        .iter()
        .map(|s| s.segment_id.clone())
        .collect();

    let query_input = || FieldSearchInput {
        field: "embedding".to_string(),
        query_vectors: vec![QueryVector {
            vector: Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            weight: 1.0,
            fields: None,
        }],
        limit: 3,
        allowed_ids: None,
    };

    // First search: populates the cache with one entry per managed segment.
    let _first = field.search(query_input())?;
    assert_eq!(
        field.reader_cache.len(),
        3,
        "cache should hold one reader per managed segment after the first search"
    );
    for id in &segment_ids {
        assert!(
            field.reader_cache.contains(id),
            "cache should contain entry for segment {id}"
        );
    }

    // Second search: cache should still have the same three entries; no
    // additional segments managed and no eviction.
    let _second = field.search(query_input())?;
    assert_eq!(
        field.reader_cache.len(),
        3,
        "cache size should be stable across repeat searches"
    );

    // Trigger a merge — `perform_merge` removes 2 source segments and
    // adds 1 merged segment, so the cache must drop the 2 source entries
    // and report a size of 1.
    field.perform_merge()?;

    let after = manager.list_segments();
    assert_eq!(after.len(), 2, "manager should hold 2 segments after merge");

    // Identify which source ids were merged away — those are the ones
    // whose cache entries should now be gone.
    let remaining: std::collections::HashSet<String> =
        after.iter().map(|s| s.segment_id.clone()).collect();
    let merged_away: Vec<&String> = segment_ids
        .iter()
        .filter(|id| !remaining.contains(*id))
        .collect();
    assert_eq!(
        merged_away.len(),
        2,
        "merge should consume 2 source segments"
    );

    for id in merged_away {
        assert!(
            !field.reader_cache.contains(id),
            "cache entry for merged-away segment {id} must be invalidated"
        );
    }

    assert!(
        field.reader_cache.len() <= 1,
        "cache should have at most the one survivor pre-search; got {}",
        field.reader_cache.len()
    );

    Ok(())
}
