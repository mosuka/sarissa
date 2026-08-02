//! Regression tests for Issue #933: per-searcher top-k must be selected by
//! distance, not by the underflow-prone similarity.
//!
//! `distance_to_similarity`'s `exp(-d)` (Euclidean/Manhattan) underflows to
//! `0.0f32` once raw distances get large, collapsing every distant candidate
//! into a tie — and the former similarity-descending sorts then decided
//! top-k **membership** by unstable-sort accident (the #931 oracle caught
//! `[0, 1, 2, 3, 10]` instead of `[0, 1, 2, 3, 4]`). The fan-out layer was
//! fixed by #927; these tests pin the same contract inside each per-segment
//! searcher (HNSW graph path, Flat scan, IVF probe).
//!
//! Grid vectors (components `level * 10.0`, dim 8) make raw distances large
//! enough to underflow while keeping every ordering decision far outside
//! int8 quantization error.

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
use laurus::vector::index::flat::segmented::SegmentedFlatIndex;
use laurus::vector::index::hnsw::segmented::SegmentedHnswIndex;
use laurus::vector::index::ivf::segmented::SegmentedIvfIndex;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryParams};
use laurus::vector::{DistanceMetric, Vector};

const DIM: usize = 8;
const N_DOCS: u64 = 20;
const TOP_K: usize = 5;

fn grid(level: f32) -> Vector {
    Vector::new(vec![level * 10.0; DIM])
}

fn corpus() -> Vec<(u64, String, Vector)> {
    (0..N_DOCS)
        .map(|i| (i, "v".to_string(), grid(i as f32)))
        .collect()
}

fn query(q_level: f32) -> VectorIndexQuery {
    VectorIndexQuery {
        query: grid(q_level),
        params: VectorIndexQueryParams {
            top_k: TOP_K,
            ef_search: Some(256),
            ..Default::default()
        },
        field_name: Some("v".to_string()),
        filter: None,
    }
}

/// Exact expected top-k for a query at `q_level` over the grid corpus.
fn expected(q_level: f32) -> Vec<u64> {
    let mut d: Vec<(f32, u64)> = (0..N_DOCS)
        .map(|i| ((i as f32 - q_level).abs(), i))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.truncate(TOP_K);
    d.into_iter().map(|(_, id)| id).collect()
}

fn assert_exact_order(index: &dyn VectorIndex, label: &str) {
    let searcher = index.searcher().unwrap();
    // Query levels at both ends and middle; 0.2 is the #931 discovery case
    // (docs beyond ~level 4 underflow to similarity 0.0).
    for q_level in [0.2, 9.4, 18.7] {
        let res = searcher.search(&query(q_level)).unwrap();
        let got: Vec<u64> = res.results.iter().map(|r| r.doc_id).collect();
        assert_eq!(
            got,
            expected(q_level),
            "{label}: top-k at q={q_level} must be selected by distance"
        );
    }
}

/// HNSW graph path (`final_results` sort): the #931 oracle's repro.
#[test]
fn hnsw_top_k_membership_is_distance_selected() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = SegmentedHnswIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vi",
        HnswIndexConfig {
            dimension: DIM,
            m: 8,
            ef_construction: 64,
            normalize_vectors: false,
            distance_metric: DistanceMetric::Euclidean,
            segmented: true,
            ..Default::default()
        },
    )
    .unwrap();
    let mut w = index.writer().unwrap();
    w.add_vectors(corpus()).unwrap();
    w.commit().unwrap();
    assert_exact_order(&index, "hnsw");
}

/// Flat scan paths (field-filtered and unfiltered sorts).
#[test]
fn flat_top_k_membership_is_distance_selected() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = SegmentedFlatIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vi",
        FlatIndexConfig {
            dimension: DIM,
            normalize_vectors: false,
            distance_metric: DistanceMetric::Euclidean,
            segmented: true,
            ..Default::default()
        },
    )
    .unwrap();
    let mut w = index.writer().unwrap();
    w.add_vectors(corpus()).unwrap();
    w.commit().unwrap();
    assert_exact_order(&index, "flat");
}

/// IVF probe path (candidate sort across probed clusters). `n_probe` covers
/// every cluster so the candidate set is complete and the expected order
/// exact.
#[test]
fn ivf_top_k_membership_is_distance_selected() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = SegmentedIvfIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vi",
        IvfIndexConfig {
            dimension: DIM,
            normalize_vectors: false,
            distance_metric: DistanceMetric::Euclidean,
            n_clusters: 4,
            n_probe: 4,
            segmented: true,
            ..Default::default()
        },
    )
    .unwrap();
    let mut w = index.writer().unwrap();
    w.add_vectors(corpus()).unwrap();
    w.commit().unwrap();
    assert_exact_order(&index, "ivf");
}
