//! End-to-end vector search benchmarks for Flat, IVF, and HNSW indexes.
//!
//! Compares construction and search performance across all three index types
//! using the same dataset and query parameters.
//!
//! # Scope
//!
//! - End-to-end through `FlatVectorSearcher`, `IvfSearcher`, `HnswSearcher`.
//! - 1 000 / 5 000 vectors at dim=128, single field, top-10. Larger scale,
//!   `ef_search` sweep, and multi-field cases are tracked in #423.
//! - All inputs are deterministic via `common::DEFAULT_SEED` so two
//!   consecutive runs produce comparable numbers.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench vector_search_bench
//! ```
//!
//! Filter by group / case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench vector_search_bench -- "Flat Search"
//! cargo bench --bench vector_search_bench -- "HNSW Search"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench vector_search_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::vector::Vector;
use laurus::vector::index::ManagedVectorIndex;
use laurus::vector::index::config::{
    FlatIndexConfig, HnswIndexConfig, IvfIndexConfig, VectorIndexTypeConfig,
};
use laurus::vector::{
    FlatVectorSearcher, HnswSearcher, IvfSearcher, VectorIndexQuery, VectorIndexSearcher,
};

use common::{DEFAULT_SEED, SAMPLE_SIZE_SLOW, lcg_vec_unit};

/// Build a deterministic `Vector` of length `dim`. The caller-supplied
/// `state` advances per call, so successive vectors are different but the
/// whole sequence is reproducible.
fn generate_vector(state: &mut u64, dim: usize) -> Vector {
    Vector::new(lcg_vec_unit(state, dim))
}

fn generate_vectors(count: usize, dim: usize) -> Vec<(u64, String, Vector)> {
    let mut state = DEFAULT_SEED;
    (0..count)
        .map(|i| {
            (
                i as u64,
                "field".to_string(),
                generate_vector(&mut state, dim),
            )
        })
        .collect()
}

/// Build a deterministic query vector. Uses a different starting seed from
/// the corpus so the query is not byte-identical to a corpus member.
fn generate_query(dim: usize) -> Vector {
    let mut state = DEFAULT_SEED.wrapping_add(1);
    generate_vector(&mut state, dim)
}

fn create_storage() -> Arc<dyn Storage> {
    StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default())).unwrap()
}

// ---------------------------------------------------------------------------
// Construction benchmarks
// ---------------------------------------------------------------------------

fn bench_flat_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flat Construction");
    group.sample_size(SAMPLE_SIZE_SLOW); // slow construction path
    let dim = 128;

    for &count in &[1000, 5000] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let vectors = generate_vectors(count, dim);
            b.iter(|| {
                let storage = create_storage();
                let config = FlatIndexConfig {
                    dimension: dim,
                    distance_metric: DistanceMetric::Cosine,
                    ..Default::default()
                };
                let mut index =
                    ManagedVectorIndex::new(VectorIndexTypeConfig::Flat(config), storage, "bench")
                        .unwrap();
                index.add_vectors(vectors.clone()).unwrap();
                index.finalize().unwrap();
            });
        });
    }
    group.finish();
}

fn bench_ivf_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("IVF Construction");
    group.sample_size(SAMPLE_SIZE_SLOW); // slow construction path
    let dim = 128;

    for &count in &[1000, 5000] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let vectors = generate_vectors(count, dim);
            b.iter(|| {
                let storage = create_storage();
                let config = IvfIndexConfig {
                    dimension: dim,
                    distance_metric: DistanceMetric::Cosine,
                    n_clusters: 10,
                    n_probe: 3,
                    ..Default::default()
                };
                let mut index =
                    ManagedVectorIndex::new(VectorIndexTypeConfig::IVF(config), storage, "bench")
                        .unwrap();
                index.add_vectors(vectors.clone()).unwrap();
                index.finalize().unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Search benchmarks
// ---------------------------------------------------------------------------

fn bench_flat_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flat Search");
    let dim = 128;

    for &count in &[1000, 5000] {
        let vectors = generate_vectors(count, dim);
        let storage = create_storage();
        let config = FlatIndexConfig {
            dimension: dim,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let mut index =
            ManagedVectorIndex::new(VectorIndexTypeConfig::Flat(config), storage, "flat_bench")
                .unwrap();
        index.add_vectors(vectors).unwrap();
        index.finalize().unwrap();

        let reader = index.reader().unwrap();
        let searcher = FlatVectorSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: the probe must return top-10 hits before timing.
        let probe = searcher
            .search(&VectorIndexQuery::new(query.clone()).top_k(10))
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "flat top-10 probe must return at least one hit at count={count}"
        );

        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone()).top_k(10);
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_ivf_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("IVF Search");
    let dim = 128;

    for &count in &[1000, 5000] {
        let vectors = generate_vectors(count, dim);
        let storage = create_storage();
        let config = IvfIndexConfig {
            dimension: dim,
            distance_metric: DistanceMetric::Cosine,
            n_clusters: 10,
            n_probe: 3,
            ..Default::default()
        };
        let mut index =
            ManagedVectorIndex::new(VectorIndexTypeConfig::IVF(config), storage, "ivf_bench")
                .unwrap();
        index.add_vectors(vectors).unwrap();
        index.finalize().unwrap();
        index.write().unwrap();

        let reader = index.reader().unwrap();
        let searcher = IvfSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: the probe must return top-10 hits before timing.
        let probe = searcher
            .search(&VectorIndexQuery::new(query.clone()).top_k(10))
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "ivf top-10 probe must return at least one hit at count={count}"
        );

        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone()).top_k(10);
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_hnsw_search_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Search");
    let dim = 128;

    for &count in &[1000, 5000] {
        let vectors = generate_vectors(count, dim);
        let storage = create_storage();
        let config = HnswIndexConfig {
            dimension: dim,
            m: 16,
            ef_construction: 200,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let mut index =
            ManagedVectorIndex::new(VectorIndexTypeConfig::HNSW(config), storage, "hnsw_bench")
                .unwrap();
        index.add_vectors(vectors).unwrap();
        index.finalize().unwrap();
        index.write().unwrap();

        let reader = index.reader().unwrap();
        let searcher = HnswSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: the probe must return top-10 hits before timing.
        let probe = searcher
            .search(&VectorIndexQuery::new(query.clone()).top_k(10))
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "hnsw top-10 probe must return at least one hit at count={count}"
        );

        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone()).top_k(10);
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_flat_construction,
    bench_ivf_construction,
    bench_flat_search,
    bench_ivf_search,
    bench_hnsw_search_compare,
);
criterion_main!(benches);
