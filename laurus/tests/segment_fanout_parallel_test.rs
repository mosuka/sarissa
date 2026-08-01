//! Determinism test for the parallel multi-segment fan-out (Issue #926).
//!
//! `SegmentFanoutSearcher::search` runs its per-segment probes concurrently
//! on native targets. This test pins that repeated identical searches over
//! a multi-segment fixture (stale duplicates + a deletion) return
//! bit-identical results — parallel probing must not leak scheduling
//! nondeterminism into the merged order. Cross-segment score correctness
//! itself is pinned by `segment_score_comparability_test.rs` (#927), which
//! exercises the same fan-out (parallel on native) against brute force.

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::HnswIndexConfig;
use laurus::vector::index::hnsw::segmented::SegmentedHnswIndex;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryParams};
use laurus::vector::{DistanceMetric, Vector};

const DIM: usize = 8;
const TOP_K: usize = 5;

fn config() -> HnswIndexConfig {
    HnswIndexConfig {
        dimension: DIM,
        m: 8,
        ef_construction: 64,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Euclidean,
        segmented: true,
        ..Default::default()
    }
}

/// A vector whose components all equal `level * 10.0` — grid spacing keeps
/// every pairwise distance far outside quantization error.
fn grid(level: f32) -> Vector {
    Vector::new(vec![level * 10.0; DIM])
}

fn query(vector: &Vector, ef: usize) -> VectorIndexQuery {
    VectorIndexQuery {
        query: vector.clone(),
        params: VectorIndexQueryParams {
            top_k: TOP_K,
            ef_search: Some(ef),
            ..Default::default()
        },
        field_name: Some("v".to_string()),
        filter: None,
    }
}

/// Build a 4-segment index with cross-segment stale duplicates and one
/// deletion (the same shape as #927's regression fixture).
fn build_fixture() -> SegmentedHnswIndex {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index =
        SegmentedHnswIndex::open_or_create(storage as Arc<dyn Storage>, "vi", config()).unwrap();

    // 4 commits = 4 sealed segments (no merge trigger in this test).
    // Segment 1: docs 0..10 at levels 0..10.
    // Segment 2: docs 10..20 at levels 10..20.
    // Segment 3: docs 20..30 at levels 20..30, PLUS doc 3 re-added at
    //            level 25.5 (stale copy of doc 3 remains in segment 1).
    // Segment 4: doc 7 re-added at level 12.5 (stale copy in segment 1).
    let batches: Vec<Vec<(u64, String, Vector)>> = vec![
        (0..10u64)
            .map(|i| (i, "v".to_string(), grid(i as f32)))
            .collect(),
        (10..20u64)
            .map(|i| (i, "v".to_string(), grid(i as f32)))
            .collect(),
        (20..30u64)
            .map(|i| (i, "v".to_string(), grid(i as f32)))
            .chain(std::iter::once((3u64, "v".to_string(), grid(25.5))))
            .collect(),
        vec![(7u64, "v".to_string(), grid(12.5))],
    ];
    for batch in batches {
        let mut w = index.writer().unwrap();
        w.add_vectors(batch).unwrap();
        w.commit().unwrap();
    }

    // Delete doc 15 (level 15) — must never appear in results.
    index.soft_delete_document(15).unwrap();

    index
}

/// Repeated identical searches must return identical result lists —
/// guards against nondeterministic merge order under parallel probing.
#[test]
fn parallel_fanout_is_deterministic_across_runs() {
    let index = build_fixture();
    let searcher = index.searcher().unwrap();

    let reference: Vec<(u64, u32)> = searcher
        .search(&query(&grid(13.7), 256))
        .unwrap()
        .results
        .iter()
        .map(|r| (r.doc_id, r.similarity.to_bits()))
        .collect();
    assert!(!reference.is_empty());

    for run in 0..5 {
        let again: Vec<(u64, u32)> = searcher
            .search(&query(&grid(13.7), 256))
            .unwrap()
            .results
            .iter()
            .map(|r| (r.doc_id, r.similarity.to_bits()))
            .collect();
        assert_eq!(reference, again, "run {run}: results must be bit-identical");
    }
}
