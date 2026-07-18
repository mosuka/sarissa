//! Correctness tests for the segmented vector machinery (#634 PR-2 / #880).
//!
//! Same-doc_id upserts replayed from the WAL leave stale copies in older
//! segments. Pre-#880 the segmented search scored such a doc once per copy
//! (`score +=` across segments, letting stale data outrank live data) and
//! merges concatenated all copies into the merged segment. These tests pin
//! the fixed semantics:
//! - search: newest-generation-wins — a doc id is scored exactly once per
//!   query, from its newest copy;
//! - merge: cross-segment duplicates of a `(doc_id, field)` key collapse to
//!   the newest copy;
//! - quantization: a PQ-configured segment below the k-means training
//!   threshold is written as Scalar8Bit (self-describing header), instead of
//!   carrying a degenerate, oversized codebook.

use std::sync::Arc;

use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::HnswIndexConfig;
use laurus::vector::index::field::{FieldSearchInput, VectorFieldReader, VectorFieldWriter};
use laurus::vector::index::hnsw::HnswIndex;
use laurus::vector::index::hnsw::segment::manager::{SegmentManager, SegmentManagerConfig};
use laurus::vector::index::segmented_field::SegmentedVectorField;
use laurus::vector::index::storage::VectorStorage;
use laurus::vector::store::request::QueryVector;
use laurus::vector::{
    DistanceMetric, FieldOption, HnswOption, StoredVector, Vector, VectorFieldConfig,
};

fn field_config(dimension: usize) -> VectorFieldConfig {
    VectorFieldConfig {
        vector: Some(FieldOption::Hnsw(HnswOption {
            dimension,
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
    }
}

fn segmented_field(
    storage: Arc<MemoryStorage>,
) -> Result<(SegmentedVectorField, Arc<SegmentManager>), Box<dyn std::error::Error>> {
    let manager_config = SegmentManagerConfig {
        max_segments: 100, // No automatic merging unless forced
        merge_factor: 10,
        min_vectors_per_segment: 1,
        ..Default::default()
    };
    let manager = Arc::new(SegmentManager::new(manager_config, storage.clone())?);
    let field =
        SegmentedVectorField::create("embedding", field_config(4), manager.clone(), storage, None)?;
    Ok((field, manager))
}

fn query(vector: Vec<f32>, limit: usize) -> FieldSearchInput {
    FieldSearchInput {
        field: "embedding".to_string(),
        query_vectors: vec![QueryVector {
            vector: Vector::new(vector),
            weight: 1.0,
            fields: None,
        }],
        limit,
        allowed_ids: None,
    }
}

/// #880: a doc id upserted across segments is scored exactly once per query,
/// from its NEWEST copy. Pre-fix, the stale old copy in the older segment
/// both inflated the doc's score (`score +=`) and, being an exact match for
/// the old embedding, made the doc outrank genuinely closer live docs.
#[tokio::test]
async fn same_id_upsert_across_segments_scores_newest_copy_once()
-> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (field, manager) = segmented_field(storage)?;

    // Segment 1 (older generation): doc 1 v_old = [1,0,0,0], doc 2 near it.
    field
        .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
        .await?;
    field
        .add_stored_vector(2, &StoredVector::new(vec![0.9, 0.1, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    // Segment 2 (newer generation): doc 1 upserted to v_new = [0,0,1,0].
    field.delete_document(1, 0).await?;
    field
        .add_stored_vector(1, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0]), 0)
        .await?;
    field.flush().await?;
    assert_eq!(manager.list_segments().len(), 2);

    // Query the OLD embedding: doc 1's stale copy is an exact match, but it
    // must be shadowed by the newest copy — so doc 2 is the closest hit.
    let results = field.search(query(vec![1.0, 0.0, 0.0, 0.0], 2))?;
    assert_eq!(
        results.hits[0].doc_id, 2,
        "the stale old copy of doc 1 must not outrank doc 2 (#880): {:?}",
        results.hits
    );
    let doc1 = results.hits.iter().find(|h| h.doc_id == 1);
    if let Some(hit) = doc1 {
        assert!(
            hit.distance > 1.0,
            "doc 1 must be scored from its NEW copy (far from the old \
             embedding), got distance {}",
            hit.distance
        );
    }

    // Query the NEW embedding: doc 1's live copy is the exact match.
    let results = field.search(query(vec![0.0, 0.0, 1.0, 0.0], 1))?;
    assert_eq!(results.hits[0].doc_id, 1);
    assert!(
        results.hits[0].distance < 1e-3,
        "doc 1 must match exactly via its newest copy, got distance {}",
        results.hits[0].distance
    );
    Ok(())
}

/// #880: merging segments that contain copies of the same `(doc_id, field)`
/// collapses them to the newest copy. Pre-fix both copies were written into
/// the merged segment.
#[tokio::test]
async fn merge_dedups_same_id_across_source_segments() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (field, manager) = segmented_field(storage)?;

    // Older segment: doc 1 v_old.
    field
        .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;
    // Newer segment: doc 1 v_new (upsert), doc 2.
    field.delete_document(1, 0).await?;
    field
        .add_stored_vector(1, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0]), 0)
        .await?;
    field
        .add_stored_vector(2, &StoredVector::new(vec![0.0, 1.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    // Force-merge everything into one segment.
    field.optimize().await?;

    let segments = manager.list_segments();
    assert_eq!(segments.len(), 1, "force merge must leave one segment");
    assert_eq!(
        segments[0].vector_count, 2,
        "the merged segment must contain doc 1 exactly once (newest copy) \
         plus doc 2 — not the stale duplicate (#880)"
    );

    // The surviving copy of doc 1 is the newest one.
    let results = field.search(query(vec![0.0, 0.0, 1.0, 0.0], 1))?;
    assert_eq!(results.hits[0].doc_id, 1);
    assert!(results.hits[0].distance < 1e-3);
    Ok(())
}

/// #880: a PQ-configured segment below the k-means training threshold is
/// written as Scalar8Bit (the LVS1 header is self-describing) instead of
/// carrying a degenerate ~k×dim-float codebook for a handful of vectors.
#[test]
fn tiny_pq_segment_falls_back_to_scalar_quantization() -> Result<(), Box<dyn std::error::Error>> {
    use laurus::storage::{StorageConfig, StorageFactory};

    let dim = 8usize;
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = HnswIndexConfig {
        dimension: dim,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantization { subvector_count: 4 },
        ..Default::default()
    };
    let index = HnswIndex::create(storage.clone(), "tiny_pq", config)?;

    // 5 vectors ≪ the 256 centroids a PQ codebook trains.
    let vectors: Vec<(u64, String, Vector)> = (0..5u64)
        .map(|i| {
            let values: Vec<f32> = (0..dim).map(|d| (i as f32) + (d as f32) * 0.1).collect();
            (i, "v".to_string(), Vector::new(values))
        })
        .collect();
    {
        let mut writer = index.writer()?;
        writer.add_vectors(vectors)?;
        writer.finalize()?;
        writer.write()?;
    }

    let reader = index.reader()?;
    let hnsw_reader = reader
        .as_any()
        .downcast_ref::<laurus::vector::index::hnsw::reader::HnswIndexReader>()
        .expect("HnswIndexReader");
    assert!(
        matches!(hnsw_reader.vectors(), VectorStorage::OwnedQuantized(_)),
        "a tiny PQ-configured segment must fall back to Scalar8Bit (#880)"
    );
    Ok(())
}

/// #880 (adversarial-review fix): a PARTIAL merge must not invert
/// newest-wins. Pre-fix the merged segment was registered with generation 0
/// (the hand-built info discarded the engine's inherited max(sources)), so
/// an untouched OLDER segment sorted as newer: its stale copy shadowed the
/// merged newest copy at search, and the next merge physically dropped the
/// newest copy as a "duplicate" — silent data loss.
#[tokio::test]
async fn partial_merge_preserves_newest_wins_ordering() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let manager_config = SegmentManagerConfig {
        max_segments: 2, // three segments trigger a merge
        merge_factor: 2, // ...of the two smallest
        min_vectors_per_segment: 1,
        ..Default::default()
    };
    let manager = Arc::new(SegmentManager::new(manager_config, storage.clone())?);
    let field =
        SegmentedVectorField::create("embedding", field_config(4), manager.clone(), storage, None)?;

    // gen 1 (LARGER, so the smallest-first policy leaves it out): doc 1's
    // stale copy plus fillers.
    field
        .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
        .await?;
    for (id, v) in [(10u64, 0.6f32), (11, 0.5), (12, 0.4)] {
        field
            .add_stored_vector(id, &StoredVector::new(vec![v, 1.0 - v, 0.0, 0.0]), 0)
            .await?;
    }
    field.flush().await?;
    // gen 2 (small): doc 1 upserted to a new embedding.
    field.delete_document(1, 0).await?;
    field
        .add_stored_vector(1, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0]), 0)
        .await?;
    field.flush().await?;
    // gen 3 (small): doc 2.
    field
        .add_stored_vector(2, &StoredVector::new(vec![0.0, 1.0, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    // Partial merge: the two smallest (gens 2 and 3) merge; gen 1 survives.
    field.perform_merge()?;
    let segments = manager.list_segments();
    assert_eq!(segments.len(), 2, "partial merge leaves two segments");
    let merged_gen = segments
        .iter()
        .map(|s| s.generation)
        .max()
        .expect("segments exist");
    assert!(
        merged_gen >= 3,
        "the merged segment must inherit max(source generations), got \
         segments {:?}",
        segments
            .iter()
            .map(|s| (s.segment_id.clone(), s.generation))
            .collect::<Vec<_>>()
    );

    // The stale copy in the surviving OLD segment must stay shadowed.
    let results = field.search(query(vec![1.0, 0.0, 0.0, 0.0], 3))?;
    for hit in &results.hits {
        assert!(
            hit.doc_id != 1 || hit.distance > 1.0,
            "doc 1 must resolve from the merged (newest) copy, not the stale \
             one in the surviving old segment (#880): {:?}",
            results.hits
        );
    }

    // A subsequent full merge must NOT drop the newest copy as a duplicate.
    field.optimize().await?;
    let results = field.search(query(vec![0.0, 0.0, 1.0, 0.0], 1))?;
    assert_eq!(results.hits[0].doc_id, 1);
    assert!(
        results.hits[0].distance < 1e-3,
        "the newest copy must survive the follow-up merge (#880), got {:?}",
        results.hits
    );
    Ok(())
}

/// #880 (adversarial-review fix): the newest-wins mask must be based on
/// CONTAINMENT, not on which hits newer sources returned. After an upsert,
/// an old-embedding query rarely ranks the new copy into the newer
/// segment's top-k — pre-fix the stale exact-match copy slipped through the
/// returned-hits mask and ranked first.
#[tokio::test]
async fn stale_copy_masked_even_when_newest_copy_misses_topk()
-> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (field, _manager) = segmented_field(storage)?;

    // Older segment: doc 1's stale copy (exact match for the query below)
    // plus doc 9 nearby.
    field
        .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
        .await?;
    field
        .add_stored_vector(9, &StoredVector::new(vec![0.9, 0.1, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    // Newer segment: doc 1 upserted FAR from the query, plus two docs CLOSER
    // to the query than doc 1's new embedding — so the newer segment's
    // top-2 for the query does NOT include doc 1.
    field.delete_document(1, 0).await?;
    field
        .add_stored_vector(1, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0]), 0)
        .await?;
    field
        .add_stored_vector(3, &StoredVector::new(vec![0.8, 0.2, 0.0, 0.0]), 0)
        .await?;
    field
        .add_stored_vector(4, &StoredVector::new(vec![0.85, 0.15, 0.0, 0.0]), 0)
        .await?;
    field.flush().await?;

    let results = field.search(query(vec![1.0, 0.0, 0.0, 0.0], 2))?;
    for hit in &results.hits {
        assert!(
            hit.doc_id != 1 || hit.distance > 1.0,
            "the stale copy must be masked by CONTAINMENT in the newer \
             segment even though the new copy missed its top-k (#880): {:?}",
            results.hits
        );
    }
    Ok(())
}
