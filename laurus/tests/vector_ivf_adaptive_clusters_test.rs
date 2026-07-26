//! Regression tests for Issue #889 PR-5: `IvfIndexWriter` used to
//! hard-error whenever the committed corpus was smaller than the
//! configured `n_clusters` (default 100) — and `add_vectors()` (the path
//! `VectorStore` actually calls in production) never clamped `n_clusters`
//! down the way `build()` did, so a fresh IVF index committing fewer than
//! ~100 documents failed outright on `main` before this fix.
//!
//! `train_centroids()` now clamps the *effective* cluster count to
//! `min(configured_n_clusters, vector_count).max(1)` instead of erroring,
//! and the configured ceiling itself is preserved (not silently
//! overwritten), so the effective count recovers as the corpus grows.

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::MemoryStorage;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::index::ivf::reader::IvfIndexReader;
use laurus::vector::index::ivf::writer::IvfIndexWriter;
use laurus::vector::reader::VectorIndexReader;
use laurus::vector::{IvfIndexConfig, Vector, VectorIndexWriter, VectorIndexWriterConfig};

const DIM: usize = 4;

fn doc_vec(i: u64) -> Vector {
    let t = i as f32 * 0.1;
    Vector::new(vec![t.cos(), t.sin(), (t * 2.0).cos(), (t * 2.0).sin()])
}

/// Read back `ivf_params()` (n_clusters, n_probe) from the on-disk file.
fn read_n_clusters(storage: Arc<dyn Storage>, name: &str) -> usize {
    let reader = IvfIndexReader::load(storage, name, DistanceMetric::Cosine).unwrap();
    reader.ivf_params().0
}

/// #889 PR-5: a fresh IVF index committing 50 documents via `add_vectors`
/// (not `build`) — the path `VectorStore` actually uses — must succeed and
/// every document must be searchable. This is the exact scenario that
/// hard-errored on `main` before this fix (default `n_clusters` is 100,
/// `add_vectors` never clamped it).
#[test]
fn add_vectors_below_default_n_clusters_commits_successfully() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    let config = IvfIndexConfig {
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        ..IvfIndexConfig::default()
    };
    assert_eq!(
        config.n_clusters, 100,
        "test assumes the documented default"
    );

    let mut writer = IvfIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "ivf_small",
        storage.clone(),
    )
    .unwrap();
    let vectors: Vec<_> = (0..50).map(|i| (i, "v".to_string(), doc_vec(i))).collect();
    writer.add_vectors(vectors).unwrap();
    writer.finalize().unwrap();
    writer.write().unwrap();

    let reader = IvfIndexReader::load(storage, "ivf_small", DistanceMetric::Cosine).unwrap();
    let mut ids: Vec<u64> = reader
        .vector_ids()
        .unwrap()
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..50).collect::<Vec<_>>());
}

/// A single-vector commit must degrade all the way to K=1 (brute-force
/// within that one "cluster") rather than erroring.
#[test]
fn single_vector_commit_degrades_to_one_cluster() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    let config = IvfIndexConfig {
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        ..IvfIndexConfig::default()
    };
    let mut writer = IvfIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "ivf_one",
        storage.clone(),
    )
    .unwrap();
    writer
        .add_vectors(vec![(0, "v".to_string(), doc_vec(0))])
        .unwrap();
    writer.finalize().unwrap();
    writer.write().unwrap();

    assert_eq!(read_n_clusters(storage, "ivf_one"), 1);
}

/// A corpus larger than the configured `n_clusters` must still honor it as
/// an upper bound — the adaptive clamp only kicks in below the ceiling.
#[test]
fn large_corpus_honors_configured_n_clusters_as_upper_bound() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    let config = IvfIndexConfig {
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        n_clusters: 5,
        ..IvfIndexConfig::default()
    };
    let mut writer = IvfIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "ivf_large",
        storage.clone(),
    )
    .unwrap();
    let vectors: Vec<_> = (0..200).map(|i| (i, "v".to_string(), doc_vec(i))).collect();
    writer.add_vectors(vectors).unwrap();
    writer.finalize().unwrap();
    writer.write().unwrap();

    assert_eq!(
        read_n_clusters(storage, "ivf_large"),
        5,
        "n_clusters must stay at the configured ceiling when the corpus exceeds it"
    );
}

/// The configured ceiling is not permanently overwritten by an initial
/// small commit: a later commit that grows the corpus back past the
/// ceiling recovers the full cluster count instead of staying clamped.
#[test]
fn ceiling_recovers_after_corpus_grows_past_it() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    let config = IvfIndexConfig {
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        n_clusters: 20,
        ..IvfIndexConfig::default()
    };

    // First commit: only 5 vectors, well below the ceiling of 20.
    {
        let mut writer = IvfIndexWriter::with_storage(
            config.clone(),
            VectorIndexWriterConfig::default(),
            "ivf_recover",
            storage.clone(),
        )
        .unwrap();
        let vectors: Vec<_> = (0..5).map(|i| (i, "v".to_string(), doc_vec(i))).collect();
        writer.add_vectors(vectors).unwrap();
        writer.finalize().unwrap();
        writer.write().unwrap();
    }
    assert_eq!(
        read_n_clusters(storage.clone(), "ivf_recover"),
        5,
        "clamped down to the small first commit's corpus size"
    );

    // Second commit (fresh writer, reloading the 5 committed docs): add 95
    // more, bringing the corpus to 100 -- well past the ceiling of 20.
    {
        let mut writer = IvfIndexWriter::with_storage(
            config,
            VectorIndexWriterConfig::default(),
            "ivf_recover",
            storage.clone(),
        )
        .unwrap();
        let vectors: Vec<_> = (5..100).map(|i| (i, "v".to_string(), doc_vec(i))).collect();
        writer.add_vectors(vectors).unwrap();
        writer.finalize().unwrap();
        writer.write().unwrap();
    }
    assert_eq!(
        read_n_clusters(storage, "ivf_recover"),
        20,
        "the ceiling must recover once the corpus grows past it again, not stay clamped at 5"
    );
}
