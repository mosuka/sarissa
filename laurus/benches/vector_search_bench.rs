//! End-to-end vector search benchmarks for Flat, IVF, and HNSW indexes.
//!
//! Compares construction and search performance across all three index types
//! using the same dataset and query parameters.
//!
//! # Scope
//!
//! - End-to-end through `FlatVectorSearcher`, `IvfSearcher`, `HnswSearcher`.
//! - Search corpus sweep: 1 000 / 5 000 by default; opt in to a 100 000
//!   vector case via `LAURUS_BENCH_LARGE=1` (see "Large-corpus gate" below).
//!   Construction benches stay at 1 000 / 5 000 to keep the default `cargo
//!   bench` runtime under five minutes.
//! - HNSW `ef_search` sweep: `bench_hnsw_ef_search_sweep` measures graph
//!   search at fixed corpus 5 000, dim 128, `ef_search ∈ {16, 64, 128, 256,
//!   512}`. Drives the visited-set hot path that #406 (replace
//!   `HashSet<u64>` with bitmap) targets.
//! - HNSW multi-field: `bench_hnsw_multi_field_search` builds 5 000 vectors
//!   distributed across `field_a` (30 %), `field_b` (30 %), `field_c`
//!   (40 %) and searches with `field_name("field_a")`. Drives the per-field
//!   `vector_ids` cache hot path that #405 targets — today every search
//!   linearly scans the full field list.
//! - All inputs are deterministic via `common::DEFAULT_SEED` so two
//!   consecutive runs produce comparable numbers (#427 hygiene).
//!
//! # Large-corpus gate
//!
//! Setting `LAURUS_BENCH_LARGE=1` adds a 100 000-vector case to
//! `bench_flat_search`, `bench_ivf_search`, `bench_hnsw_fallback_search`,
//! and `bench_hnsw_graph_search`. The 100 000-vector setup takes several
//! seconds and per-iter search at 100 k can be tens of milliseconds, so
//! this case is opt-in. Default runs (no env var) finish in well under
//! five minutes on a typical workstation.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench vector_search_bench                          # default sizes
//! LAURUS_BENCH_LARGE=1 cargo bench --bench vector_search_bench     # adds 100 k
//! ```
//!
//! Filter by group / case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench vector_search_bench -- "Flat Search"
//! cargo bench --bench vector_search_bench -- "ef_search/ef_64"
//! cargo bench --bench vector_search_bench -- "Multi-field"
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
use laurus::storage::Storage;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::vector::Vector;
use laurus::vector::index::ManagedVectorIndex;
use laurus::vector::index::config::{
    FlatIndexConfig, HnswIndexConfig, IvfIndexConfig, VectorIndexTypeConfig,
};
use laurus::vector::{
    FlatVectorSearcher, HnswSearcher, IvfSearcher, VectorIndexQuery, VectorIndexSearcher,
};

use common::{DEFAULT_SEED, SAMPLE_SIZE_SLOW, lcg_vec_unit, select_storage};

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

/// Names of the three fields used by the multi-field bench.
const MULTI_FIELD_NAMES: &[&str] = &["field_a", "field_b", "field_c"];

/// Build deterministic vectors split 30 / 30 / 40 % across `field_a`,
/// `field_b`, `field_c`. Used by `bench_hnsw_multi_field_search` to drive
/// the per-field-cache hot path that #405 targets — searching with
/// `field_name("field_a")` should only need to consider the 30 % of
/// vectors tagged `field_a`, but today the implementation scans every
/// (id, field_name) pair.
fn generate_multi_field_vectors(count: usize, dim: usize) -> Vec<(u64, String, Vector)> {
    let mut state = DEFAULT_SEED;
    (0..count)
        .map(|i| {
            // 30 / 30 / 40 split based on i % 10.
            let pct = i % 10;
            let field = if pct < 3 {
                MULTI_FIELD_NAMES[0] // field_a — 30 %
            } else if pct < 6 {
                MULTI_FIELD_NAMES[1] // field_b — 30 %
            } else {
                MULTI_FIELD_NAMES[2] // field_c — 40 %
            };
            (
                i as u64,
                field.to_string(),
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

/// Bench-storage handle. Delegates to `common::select_storage()` so
/// `LAURUS_BENCH_DISK=1` swaps the in-memory backend for a temp-dir
/// `FileStorage` without changing call sites.
fn create_storage() -> Arc<dyn Storage> {
    select_storage()
}

/// Default search-corpus sizes for Flat and IVF benches.
/// `LAURUS_BENCH_LARGE=1` appends a 100 000-vector case so #405 / #406
/// have a measurable target without ballooning the default `cargo bench`
/// runtime past five minutes.
fn search_corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![1000usize, 5000];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(100_000);
    }
    sizes
}

/// HNSW-specific search-corpus sizes. Adds 50 000 over `search_corpus_sizes`
/// because the fallback / graph search benches added that case in #421 to
/// demonstrate path-specific scaling. `LAURUS_BENCH_LARGE=1` further
/// appends 100 000.
fn hnsw_corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![1000usize, 5000, 50_000];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(100_000);
    }
    sizes
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

fn bench_hnsw_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Construction");
    group.sample_size(SAMPLE_SIZE_SLOW); // slow construction path
    let dim = 128;

    for &count in &[1000, 5000] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let vectors = generate_vectors(count, dim);
            b.iter(|| {
                let storage = create_storage();
                let config = HnswIndexConfig {
                    dimension: dim,
                    m: 16,
                    ef_construction: 200,
                    distance_metric: DistanceMetric::Cosine,
                    ..Default::default()
                };
                let mut index =
                    ManagedVectorIndex::new(VectorIndexTypeConfig::HNSW(config), storage, "bench")
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

    for &count in &search_corpus_sizes() {
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

    for &count in &search_corpus_sizes() {
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

/// Benchmark the **fallback (linear-scan) path** of `HnswSearcher::search`.
///
/// `VectorIndexQuery::new` leaves `field_name` as `None`. The graph branch
/// in `HnswSearcher::search` is gated on `Some(field_name)`, so a query
/// without a field name short-circuits to the linear-scan fallback. This is
/// the path #404 (drop the redundant `similarity()` + `distance()` pair)
/// targets.
fn bench_hnsw_fallback_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Fallback Search");
    let dim = 128;

    for &count in &hnsw_corpus_sizes() {
        let vectors = generate_vectors(count, dim);
        let storage = create_storage();
        let config = HnswIndexConfig {
            dimension: dim,
            m: 16,
            ef_construction: 200,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let mut index = ManagedVectorIndex::new(
            VectorIndexTypeConfig::HNSW(config),
            storage,
            "hnsw_fallback_bench",
        )
        .unwrap();
        index.add_vectors(vectors).unwrap();
        index.finalize().unwrap();
        index.write().unwrap();

        let reader = index.reader().unwrap();
        let searcher = HnswSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: the probe must return top-10 hits before timing.
        // No field_name set → fallback (linear scan) path.
        let probe = searcher
            .search(&VectorIndexQuery::new(query.clone()).top_k(10))
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "hnsw fallback top-10 probe must return at least one hit at count={count}"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone()).top_k(10);
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

/// Benchmark the **graph-traversal path** of `HnswSearcher::search`.
///
/// Setting `field_name` enables the graph branch (provided the reader has a
/// graph and is downcastable to `HnswIndexReader`). This is the path that
/// scales with `ef_search` rather than corpus size; #406 (replace the
/// `HashSet<u64>` visited set with a bitmap) and #405 (per-field
/// `vector_ids` cache) target this path.
fn bench_hnsw_graph_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Graph Search");
    let dim = 128;

    for &count in &hnsw_corpus_sizes() {
        let vectors = generate_vectors(count, dim);
        let storage = create_storage();
        let config = HnswIndexConfig {
            dimension: dim,
            m: 16,
            ef_construction: 200,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let mut index = ManagedVectorIndex::new(
            VectorIndexTypeConfig::HNSW(config),
            storage,
            "hnsw_graph_bench",
        )
        .unwrap();
        index.add_vectors(vectors).unwrap();
        index.finalize().unwrap();
        index.write().unwrap();

        let reader = index.reader().unwrap();
        let searcher = HnswSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: with field_name set, the graph branch must engage
        // and return at least one hit.
        let probe = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string()),
            )
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "hnsw graph top-10 probe must return at least one hit at count={count}"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string());
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

/// HNSW search latency across an `ef_search` sweep at a fixed corpus.
///
/// `ef_search` controls the size of the dynamic candidate list during
/// graph traversal — larger `ef` exhaustively explores more nodes per
/// query, trading latency for recall. The visited-set hot path inside the
/// inner loop scales linearly with the number of nodes touched, so this
/// sweep is what #406 (replace `HashSet<u64>` with bitmap) targets.
fn bench_hnsw_ef_search_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW ef_search");
    let dim = 128;
    let count = 5000usize;

    let vectors = generate_vectors(count, dim);
    let storage = create_storage();
    let config = HnswIndexConfig {
        dimension: dim,
        m: 16,
        ef_construction: 200,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let mut index = ManagedVectorIndex::new(
        VectorIndexTypeConfig::HNSW(config),
        storage,
        "hnsw_ef_bench",
    )
    .unwrap();
    index.add_vectors(vectors).unwrap();
    index.finalize().unwrap();
    index.write().unwrap();

    let reader = index.reader().unwrap();
    let query = generate_query(dim);

    for &ef in &[16usize, 64, 128, 256, 512] {
        let mut searcher = HnswSearcher::new(reader.clone()).unwrap();
        searcher.set_ef_search(ef);

        // Sanity: graph traversal at this ef must yield at least one hit.
        let probe = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string()),
            )
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "hnsw ef_search probe must return at least one hit at ef={ef}"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("ef_{ef}/top10")),
            &ef,
            |b, _| {
                b.iter(|| {
                    let request = VectorIndexQuery::new(query.clone())
                        .top_k(10)
                        .field_name("field".to_string());
                    searcher.search(&request).unwrap()
                });
            },
        );
    }
    group.finish();
}

/// HNSW search filtered to a single field on a multi-field corpus.
///
/// The corpus is split 30 / 30 / 40 % across `field_a`, `field_b`,
/// `field_c`. The query selects `field_a`, so only ~30 % of vectors are
/// candidates. Today every search calls `vector_ids()` to fetch all
/// `(id, field_name)` pairs and filters by string equality — wasted work
/// proportional to the non-target field count. #405 (per-field
/// `vector_ids` cache) targets this hot path.
fn bench_hnsw_multi_field_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Multi-field Search");
    let dim = 128;
    let count = 5000usize;

    let vectors = generate_multi_field_vectors(count, dim);
    let storage = create_storage();
    let config = HnswIndexConfig {
        dimension: dim,
        m: 16,
        ef_construction: 200,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let mut index = ManagedVectorIndex::new(
        VectorIndexTypeConfig::HNSW(config),
        storage,
        "hnsw_multi_field_bench",
    )
    .unwrap();
    index.add_vectors(vectors).unwrap();
    index.finalize().unwrap();
    index.write().unwrap();

    let reader = index.reader().unwrap();
    let searcher = HnswSearcher::new(reader).unwrap();
    let query = generate_query(dim);

    // Sanity: filtering to field_a (~30 % of corpus) must still hit.
    let probe = searcher
        .search(
            &VectorIndexQuery::new(query.clone())
                .top_k(10)
                .field_name("field_a".to_string()),
        )
        .unwrap();
    assert!(
        !probe.results.is_empty(),
        "hnsw multi-field probe must return at least one hit (field_a, count={count})"
    );

    group.throughput(Throughput::Elements(count as u64));
    group.bench_function("field_a_30pct/top10", |b| {
        b.iter(|| {
            let request = VectorIndexQuery::new(query.clone())
                .top_k(10)
                .field_name("field_a".to_string());
            searcher.search(&request).unwrap()
        });
    });

    group.finish();
}

/// Flat search filtered to a single field on a multi-field corpus.
///
/// The corpus is split 30 / 30 / 40 % across `field_a`, `field_b`,
/// `field_c`. The query selects `field_a`, so only ~30 % of vectors are
/// candidates. Today every search calls `vector_ids()` to materialise
/// every `(id, field_name)` pair and filters by string equality —
/// wasted work proportional to the non-target field count plus the
/// `Vec<(u64, String)>` clone.
///
/// #405 (per-field `vector_ids` cache) targets this hot path: with the
/// fix the searcher fetches a pre-built `Arc<[u64]>` for the target
/// field at O(1) cost.
fn bench_flat_multi_field_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flat Multi-field Search");
    let dim = 128;

    for &count in &search_corpus_sizes() {
        let vectors = generate_multi_field_vectors(count, dim);
        let storage = create_storage();
        let config = FlatIndexConfig {
            dimension: dim,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let mut index = ManagedVectorIndex::new(
            VectorIndexTypeConfig::Flat(config),
            storage,
            "flat_multi_field_bench",
        )
        .unwrap();
        index.add_vectors(vectors).unwrap();
        index.finalize().unwrap();

        let reader = index.reader().unwrap();
        let searcher = FlatVectorSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity: filtering to field_a (~30 % of corpus) must still hit.
        let probe = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field_a".to_string()),
            )
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "flat multi-field probe must return at least one hit (field_a, count={count})"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("field_a_30pct/{count}")),
            &count,
            |b, _| {
                b.iter(|| {
                    let request = VectorIndexQuery::new(query.clone())
                        .top_k(10)
                        .field_name("field_a".to_string());
                    searcher.search(&request).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_flat_construction,
    bench_ivf_construction,
    bench_hnsw_construction,
    bench_flat_search,
    bench_ivf_search,
    bench_hnsw_fallback_search,
    bench_hnsw_graph_search,
    bench_hnsw_ef_search_sweep,
    bench_hnsw_multi_field_search,
    bench_flat_multi_field_search,
);
criterion_main!(benches);
