//! Frozen behavior oracle for the #650 PR-1 rerank refactor (#931).
//!
//! Captured against the pre-refactor inline HNSW Stage-2 implementation,
//! this pins the observable rerank contract — result ordering AND exact
//! score bits — across the rerank-on/off × single/multi-segment matrix,
//! so porting the logic onto `RerankPipeline` must not change a bit.
//!
//! The expectations are computed independently (brute force over the raw
//! f32 vectors + the public `DistanceMetric` conversion), not recorded as
//! goldens: Stage-2 rescoring is exact-f32 against the sidecar, so the
//! reranked results must equal the exact computation regardless of which
//! implementation produced them. Grid-spaced vectors keep every ordering
//! decision far outside int8 quantization error.

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::HnswIndexConfig;
use laurus::vector::index::hnsw::segmented::SegmentedHnswIndex;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryParams};
use laurus::vector::{DistanceMetric, Vector};

const DIM: usize = 8;
const TOP_K: usize = 5;
const RERANK_FACTOR: usize = 4;

fn config() -> HnswIndexConfig {
    HnswIndexConfig {
        dimension: DIM,
        m: 8,
        ef_construction: 64,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Euclidean,
        rerank_storage: Some(RerankStorageKind::F32),
        segmented: true,
        ..Default::default()
    }
}

/// Grid vector: all components `level * 10.0` — inter-doc distances dwarf
/// quantization error, so orderings are deterministic.
fn grid(level: f32) -> Vector {
    Vector::new(vec![level * 10.0; DIM])
}

fn query(vector: &Vector, rerank: Option<usize>) -> VectorIndexQuery {
    VectorIndexQuery {
        query: vector.clone(),
        params: VectorIndexQueryParams {
            top_k: TOP_K,
            ef_search: Some(256),
            rerank_factor: rerank,
            ..Default::default()
        },
        field_name: Some("v".to_string()),
        filter: None,
    }
}

/// Build an index from `batches` (one commit per batch). Returns the index
/// and the live `(doc_id, level)` state (newest copy wins).
fn build(batches: &[Vec<(u64, f32)>]) -> (SegmentedHnswIndex, Vec<(u64, f32)>) {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index =
        SegmentedHnswIndex::open_or_create(storage as Arc<dyn Storage>, "vi", config()).unwrap();
    let mut live: Vec<(u64, f32)> = Vec::new();
    for batch in batches {
        let docs: Vec<(u64, String, Vector)> = batch
            .iter()
            .map(|&(id, level)| (id, "v".to_string(), grid(level)))
            .collect();
        let mut w = index.writer().unwrap();
        w.add_vectors(docs).unwrap();
        w.commit().unwrap();
        for &(id, level) in batch {
            if let Some(entry) = live.iter_mut().find(|(d, _)| *d == id) {
                entry.1 = level;
            } else {
                live.push((id, level));
            }
        }
    }
    (index, live)
}

/// Exact expected `(doc_id, similarity_bits)` top-k for a query at
/// `q_level`, computed from raw f32 vectors through the public metric API —
/// the same arithmetic the exact-f32 sidecar rescoring performs.
fn expected(live: &[(u64, f32)], q_level: f32) -> Vec<(u64, u32)> {
    let metric = DistanceMetric::Euclidean;
    let q = grid(q_level);
    let mut d: Vec<(f32, u64)> = live
        .iter()
        .map(|&(id, level)| (metric.distance(&q.data, &grid(level).data).unwrap(), id))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.truncate(TOP_K);
    d.into_iter()
        .map(|(dist, id)| (id, metric.distance_to_similarity(dist).to_bits()))
        .collect()
}

fn run_case(batches: &[Vec<(u64, f32)>], q_levels: &[f32]) {
    let (index, live) = build(batches);
    let searcher = index.searcher().unwrap();

    for &q_level in q_levels {
        // Rerank ON: order and score bits must equal the exact-f32
        // computation (the sidecar holds the raw vectors).
        let reranked = searcher
            .search(&query(&grid(q_level), Some(RERANK_FACTOR)))
            .unwrap();
        let got: Vec<(u64, u32)> = reranked
            .results
            .iter()
            .map(|r| (r.doc_id, r.similarity.to_bits()))
            .collect();
        assert_eq!(
            got,
            expected(&live, q_level),
            "rerank-on at q={q_level} must match the exact f32 oracle bit-for-bit"
        );

        // Rerank OFF: same doc set (grid separation makes int8 ordering
        // agree with exact ordering), and the call must succeed — the
        // scores travel the quantized/shared-basis path, which the
        // refactor must leave untouched. Deterministic across runs.
        let plain = searcher.search(&query(&grid(q_level), None)).unwrap();
        let plain_ids: Vec<u64> = plain.results.iter().map(|r| r.doc_id).collect();
        let expected_ids: Vec<u64> = expected(&live, q_level).iter().map(|&(id, _)| id).collect();
        assert_eq!(
            plain_ids, expected_ids,
            "rerank-off at q={q_level}: grid separation must keep the quantized order exact"
        );
        let plain2 = searcher.search(&query(&grid(q_level), None)).unwrap();
        let bits1: Vec<u32> = plain
            .results
            .iter()
            .map(|r| r.similarity.to_bits())
            .collect();
        let bits2: Vec<u32> = plain2
            .results
            .iter()
            .map(|r| r.similarity.to_bits())
            .collect();
        assert_eq!(bits1, bits2, "rerank-off must be bit-deterministic");
    }
}

/// Single sealed segment: the per-segment searcher's Stage-2 arm runs with
/// the fan-out's serial single-reader branch.
#[test]
fn oracle_single_segment() {
    let batches = vec![(0..20u64).map(|i| (i, i as f32)).collect::<Vec<_>>()];
    run_case(&batches, &[0.2, 7.3, 13.6, 19.0]);
}

/// Three sealed segments with a cross-segment stale duplicate: exercises
/// the fan-out's newest-wins masking together with the exact-basis skip
/// (#927) under rerank.
#[test]
fn oracle_multi_segment_with_stale_duplicate() {
    let batches = vec![
        (0..10u64).map(|i| (i, i as f32)).collect::<Vec<_>>(),
        (10..20u64).map(|i| (i, i as f32)).collect::<Vec<_>>(),
        // Newer copy of doc 3 far from its stale position.
        vec![(20u64, 20.0), (21u64, 21.0), (3u64, 15.5)],
    ];
    run_case(&batches, &[0.2, 3.1, 15.4, 20.6]);
}
