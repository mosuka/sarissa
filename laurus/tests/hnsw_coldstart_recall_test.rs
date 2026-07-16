//! Regression tests for #872 — seed/incremental bulk-load recall parity.
//!
//! Appending a large batch onto a small HNSW base used to funnel every parallel
//! insert through the base's low-level entry point, fragmenting the graph:
//! reachability was still guaranteed (by the #868 connectivity repair), but the
//! appended nodes had far fewer/worse in-edges, so their search recall collapsed
//! to roughly half of a fresh full build of the same total size.
//!
//! The #872 fix has two regimes, and there is one test per regime:
//!   * base `< ef_construction` — the base is too shallow to navigate, so the
//!     append discards it and does a full rebuild (fresh quality). Covered by
//!     [`seed_then_bulk_load_recall_matches_fresh_build`] (base = 1).
//!   * base `>= ef_construction` — the incremental path is kept, but if the
//!     append promotes the entry point to a new top-level node, that node is
//!     inserted first and used as the build search start so inserts descend from
//!     the top layer instead of funneling through `old_ep`. Covered by
//!     [`incremental_append_recall_matches_fresh_build`] (base = 200).
//!
//! The gate is **self-recall@10**: query each node with its own vector and check
//! it appears in its own top-10. A well-built graph is ~1.0; a fragmented one is
//! much lower. Each test compares the appended nodes' self-recall against a fresh
//! full build of the same corpus — deterministic and independent of any fixed
//! absolute threshold (both builds see the identical vectors and query set).

use std::sync::Arc;

use laurus::storage::file::{FileStorage, FileStorageConfig};
use laurus::vector::index::hnsw::searcher::HnswSearcher;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};
use laurus::vector::{
    DistanceMetric, HnswIndexConfig, HnswIndexReader, HnswIndexWriter, Vector, VectorIndexWriter,
    VectorIndexWriterConfig,
};
use tempfile::tempdir;

fn doc_vec(i: u64) -> Vector {
    let mut v = vec![0.0f32; 16];
    let t = i as f32 * 0.001;
    v[0] = t.cos();
    v[1] = t.sin();
    v[2] = (t * 2.0).cos();
    v[3] = (t * 3.0).sin();
    Vector::new(v)
}
fn config() -> HnswIndexConfig {
    HnswIndexConfig {
        dimension: 16,
        m: 16,
        ef_construction: 100,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    }
}
fn writer_config() -> VectorIndexWriterConfig {
    VectorIndexWriterConfig {
        parallel_build: true,
        ..Default::default()
    }
}

/// Fraction of ids in `[lo, hi)` that appear in their own vector's top-10.
fn self_recall_at_10(path: &std::path::Path, name: &str, lo: u64, hi: u64) -> f32 {
    let storage = Arc::new(FileStorage::new(path, FileStorageConfig::new(path)).unwrap());
    let reader = HnswIndexReader::load(storage, name, DistanceMetric::Cosine).unwrap();
    let mut searcher = HnswSearcher::new(Arc::new(reader)).unwrap();
    searcher.set_ef_search(200);
    let mut hits = 0u64;
    for id in lo..hi {
        let req = VectorIndexQuery::new(doc_vec(id))
            .top_k(10)
            .field_name("v".to_string());
        if searcher
            .search(&req)
            .unwrap()
            .results
            .iter()
            .any(|r| r.doc_id == id)
        {
            hits += 1;
        }
    }
    hits as f32 / (hi - lo) as f32
}

/// Fresh full build of ids `0..n` in one shot.
fn fresh_build(path: &std::path::Path, name: &str, n: u64) {
    let storage = Arc::new(FileStorage::new(path, FileStorageConfig::new(path)).unwrap());
    let mut w = HnswIndexWriter::with_storage(config(), writer_config(), name, storage).unwrap();
    let v: Vec<_> = (0..n).map(|i| (i, "v".to_string(), doc_vec(i))).collect();
    w.add_vectors(v).unwrap();
    w.finalize().unwrap();
    w.write().unwrap();
}

/// Build `0..base` first, commit, then append `base..n` in a second commit.
fn seed_then_append(path: &std::path::Path, name: &str, base: u64, n: u64) {
    {
        let storage = Arc::new(FileStorage::new(path, FileStorageConfig::new(path)).unwrap());
        let mut w =
            HnswIndexWriter::with_storage(config(), writer_config(), name, storage).unwrap();
        let v: Vec<_> = (0..base)
            .map(|i| (i, "v".to_string(), doc_vec(i)))
            .collect();
        w.add_vectors(v).unwrap();
        w.finalize().unwrap();
        w.write().unwrap();
    }
    {
        let storage = Arc::new(FileStorage::new(path, FileStorageConfig::new(path)).unwrap());
        let mut w = HnswIndexWriter::load(config(), writer_config(), storage, name).unwrap();
        let v: Vec<_> = (base..n)
            .map(|i| (i, "v".to_string(), doc_vec(i)))
            .collect();
        w.add_vectors(v).unwrap();
        w.finalize().unwrap();
        w.write().unwrap();
    }
}

/// Assert the appended nodes' self-recall matches a fresh full build's over the
/// same id range, within a small tolerance (pre-#872 the append was ~0.5x).
fn assert_append_matches_fresh(base: u64, n: u64) {
    let fresh_dir = tempdir().unwrap();
    fresh_build(fresh_dir.path(), "fresh", n);
    let fresh = self_recall_at_10(fresh_dir.path(), "fresh", base, n);

    let cold_dir = tempdir().unwrap();
    seed_then_append(cold_dir.path(), "cold", base, n);
    let cold = self_recall_at_10(cold_dir.path(), "cold", base, n);

    // Sanity: the fresh build must itself be well-connected.
    assert!(
        fresh > 0.9,
        "fresh full-build self-recall@10 should be high, got {fresh:.4}"
    );
    // Absolute-difference tolerance so a marginally-higher cold value (it can
    // slightly exceed fresh) also passes.
    assert!(
        cold >= fresh - 0.05,
        "base={base} seed-then-bulk-load self-recall@10 ({cold:.4}) must match the \
         fresh build ({fresh:.4}) within tolerance (#872)",
    );
}

/// #872, base `< ef_construction`: a seed-1-then-bulk-append build is rebuilt
/// fresh, so it must reach the same self-recall as a fresh full build.
#[test]
fn seed_then_bulk_load_recall_matches_fresh_build() {
    assert_append_matches_fresh(1, 5000);
}

/// #872, base `>= ef_construction`: appending onto a real (non-trivial) base
/// keeps the incremental path; the promoted-entry-point-first fix must still
/// give the appended nodes fresh-build recall (pre-fix ~0.73 vs ~0.98).
#[test]
fn incremental_append_recall_matches_fresh_build() {
    assert_append_matches_fresh(200, 5000);
}
