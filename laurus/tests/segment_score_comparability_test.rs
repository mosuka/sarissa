//! Regression test for Issue #927: cross-segment score comparability.
//!
//! Per-segment similarities come from quantized kernels running against
//! per-segment affine params, with the query clamped into each segment's
//! value range. Before #927 the fan-out compared those segment-local
//! scores globally, so any segment whose value range excluded the query
//! reported its boundary doc as an exact match (`similarity = 1.0`) and
//! outranked truly closer docs from other segments — a single-doc
//! segment (one upsert commit) topped every out-of-range query.
//!
//! The fixture places vectors on a coarse grid (steps of 10.0 per
//! component) so the expected ordering is far outside quantization error,
//! and the per-segment `ef_search` exceeds every segment's size — making
//! per-segment search exact and the brute-force expectation
//! deterministic.

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

/// Build a 4-segment index whose segments cover disjoint value ranges,
/// with cross-segment stale duplicates and one deletion. Returns the
/// index and the live `(doc_id, level)` state (newest copy wins, deleted
/// doc excluded).
fn build_fixture() -> (SegmentedHnswIndex, Vec<(u64, f32)>) {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index =
        SegmentedHnswIndex::open_or_create(storage as Arc<dyn Storage>, "vi", config()).unwrap();

    // 4 commits = 4 sealed segments (no merge trigger in this test).
    // Segment 1: docs 0..10 at levels 0..10.
    // Segment 2: docs 10..20 at levels 10..20 (disjoint range).
    // Segment 3: docs 20..30 at levels 20..30, PLUS doc 3 re-added at
    //            level 25.5 (stale copy of doc 3 remains in segment 1).
    // Segment 4: doc 7 alone at level 12.5 — the degenerate-range,
    //            single-doc segment that used to top every query.
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

    let mut live: Vec<(u64, f32)> = (0..30u64)
        .map(|i| (i, i as f32))
        .filter(|(i, _)| *i != 15)
        .collect();
    for (id, level) in live.iter_mut() {
        if *id == 3 {
            *level = 25.5; // newest copy (segment 3)
        }
        if *id == 7 {
            *level = 12.5; // newest copy (segment 4)
        }
    }
    (index, live)
}

/// Expected top-k doc ids for a query at `q_level`, by exact distance over
/// the live state.
fn expected_top_k(live: &[(u64, f32)], q_level: f32) -> Vec<u64> {
    let mut d: Vec<(f32, u64)> = live
        .iter()
        .map(|(id, level)| ((level - q_level).abs(), *id))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.truncate(TOP_K);
    d.into_iter().map(|(_, id)| id).collect()
}

/// The #927 repro: queries inside, outside, and between the segments'
/// value ranges must return the brute-force expected top-k. Before the
/// fix, the query at level 0.2 returned `[7, 20, 10, 0, 1]` — the
/// out-of-range segments' boundary docs at similarity 1.0 — instead of
/// `[0, 1, 2, 4, 5]`. Newest-wins masking (stale docs 3 / 7) and the
/// deletion (doc 15) must survive the rescore unchanged.
#[test]
fn cross_segment_scores_match_bruteforce_on_disjoint_ranges() {
    let (index, live) = build_fixture();
    let searcher = index.searcher().unwrap();

    // Query levels chosen so no two live docs are equidistant: an exact
    // tie's rescored order would depend on each segment's reconstruction
    // error (dequantized basis), which is not what this test pins. The
    // smallest inter-hit distance gap (0.1 level = 2.8 raw L2) still
    // exceeds the worst-case reconstruction perturbation (~1.0 raw L2).
    for q_level in [0.2, 3.1, 7.1, 12.4, 15.3, 22.7, 25.4, 29.0] {
        let res = searcher.search(&query(&grid(q_level), 256)).unwrap();
        let got: Vec<u64> = res.results.iter().map(|r| r.doc_id).collect();
        let want = expected_top_k(&live, q_level);
        assert_eq!(
            got, want,
            "query at level {q_level}: cross-segment order must match brute force"
        );
    }
}

/// After the shared-basis rescore, `min_similarity` must filter on the
/// comparable score: a threshold that the degenerate segment's inflated
/// local score (1.0) would have passed must drop it once rescored.
#[test]
fn min_similarity_applies_to_the_shared_basis_score() {
    let (index, _) = build_fixture();
    let searcher = index.searcher().unwrap();

    // Query far outside segment 4's point range: doc 7's local score
    // used to saturate at 1.0. With Euclidean similarity = exp(-d), any
    // positive threshold excludes every far-away doc once rescored.
    let mut q = query(&grid(0.2), 256);
    q.params.min_similarity = 0.5;
    let res = searcher.search(&q).unwrap();
    for hit in &res.results {
        assert!(
            hit.similarity >= 0.5,
            "hit {} at similarity {} must respect min_similarity on the \
             shared basis",
            hit.doc_id,
            hit.similarity
        );
        assert_ne!(
            hit.doc_id, 7,
            "the degenerate segment's doc must not pass min_similarity \
             via its inflated local score"
        );
    }
}

/// Issue #672 regression: `SegmentFanoutSearcher::count` was rewritten
/// from per-segment `vector_ids()` full clones onto the Arc-backed
/// `doc_ids_for_field` path — its semantics must be unchanged: distinct
/// live `(doc_id, field)` keys, counting cross-segment duplicates once
/// (newest-wins), excluding soft-deleted docs, and honoring the field
/// filter.
#[test]
fn count_masks_duplicates_and_deletions() {
    use laurus::vector::search::searcher::VectorIndexQueryParams;

    let (index, live) = build_fixture();
    let searcher = index.searcher().unwrap();

    let count_for = |field: Option<&str>| {
        searcher
            .count(VectorIndexQuery {
                query: grid(0.0),
                params: VectorIndexQueryParams::default(),
                field_name: field.map(str::to_string),
                filter: None,
            })
            .unwrap()
    };

    // 30 docs, one deleted; the stale copies of docs 3 and 7 are the
    // same (doc, field) keys and must not be double-counted.
    assert_eq!(count_for(Some("v")), live.len() as u64);
    assert_eq!(count_for(None), live.len() as u64);
    assert_eq!(count_for(Some("missing")), 0);
}
