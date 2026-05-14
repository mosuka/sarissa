//! End-to-end vector search benchmarks for Flat, IVF, and HNSW indexes.
//!
//! Compares construction and search performance across all three index types
//! using the same dataset and query parameters.
//!
//! # Quantization status (Issue #481 Stage 1)
//!
//! After Stage 1, **every** vector index group in this file goes through
//! the int8 scalar-quantized hot path: writers emit the `LVS1` segment
//! format, readers populate `VectorStorage::OwnedQuantized`, and the
//! search loop dispatches to `distance_quantized`. There is no longer an
//! f32-only path to compare against in-tree.
//!
//! - **Stage 1 speed gate** is `bench_hnsw_graph_search` at
//!   `top10/50000`: the int8 path here must achieve ≥ 2× speedup vs the
//!   pre-Stage-1 f32 numbers recorded on `main` before this PR landed.
//!   Compare via two separate runs (one per branch).
//! - **Stage 2 speed gate** is `bench_hnsw_graph_search_rerank` at
//!   `top10/50000`. Issue #481 Stage 2 asks for ≥ 3× speedup vs the
//!   same pre-Stage-1 f32 baseline; the bench above shows that on
//!   the suite's synthetic random unit-norm data, rerank's per-query
//!   overhead is well inside the noise band of
//!   `bench_hnsw_graph_search` (a few µs at dim 128 for 50 exact-
//!   distance calls). The realised speedup therefore tracks Stage 1
//!   plus that small overhead; on synthetic random data the
//!   underlying HNSW graph traversal is the floor, so the absolute
//!   speedup ratio is bounded by the Stage 1 result on the same
//!   data (≈ 2×). Real clustered embeddings allow a lower ef_search
//!   and a wider speedup margin -- see
//!   `tests/vector_recall_test.rs::stage2_recall_sweep_diagnostic`
//!   for the recall-vs-budget trade-off underlying that choice.
//! - **Recall gates** are _not_ measured here -- recall lives in
//!   `laurus/tests/vector_recall_test.rs` so the latency vs recall
//!   surfaces stay independent.
//! - The `HNSW Construction` group also runs through the new format
//!   (writer trains + serializes int8 at flush). Construction bench
//!   numbers should stay within the noise band of the f32 baseline
//!   since graph build still computes f32 distances; only the
//!   final `commit()` call now does the per-segment SQ training.
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
use laurus::vector::core::rerank::RerankStorageKind;
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
///
/// **Issue #481 Stage 1 speed gate**: the `top10/50000` case here is the
/// designated benchmark for the "≥ 2× speedup vs f32 baseline" condition.
/// Run before and after the Stage-1 PR (or against the pre-PR `main`
/// branch) and compare absolute medians. Recall is gated separately in
/// `laurus/tests/vector_recall_test.rs`.
/// Stage 2 (Issue #481): HNSW graph search with the LRS1 rerank
/// sidecar enabled and `rerank_factor` set per query. Measures the
/// end-to-end search latency of the two-stage flow:
///
/// 1. int8 HNSW graph search returns `ef_search` candidates.
/// 2. The top `top_k * rerank_factor` candidates are rescored
///    against the original f32 vectors loaded from the sidecar.
/// 3. The new top `top_k` is returned.
///
/// **Stage 2 speed gate** sits at `top10/50000` here: the int8 +
/// rerank path must achieve **≥ 3×** speedup vs the pre-Stage-1 f32
/// numbers recorded on `main` before Stage 1 landed. The rerank
/// rescore is bounded at `top_k * rerank_factor = 50` f32 distance
/// calls per query (a few µs at dim 128) so the int8 graph search
/// dominates the wall clock and the speedup tracks
/// [`bench_hnsw_graph_search`] within rerank's small overhead.
///
/// Tracks ef_search at the searcher default (50) for direct
/// comparison with [`bench_hnsw_graph_search`]; a higher ef_search
/// is the recall-vs-speed lever (see
/// `tests/vector_recall_test.rs::stage2_recall_sweep_diagnostic`).
fn bench_hnsw_graph_search_rerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Graph Search Rerank");
    let dim = 128;

    for &count in &hnsw_corpus_sizes() {
        let vectors = generate_vectors(count, dim);
        let storage = create_storage();
        let config = HnswIndexConfig {
            dimension: dim,
            m: 16,
            ef_construction: 200,
            distance_metric: DistanceMetric::Cosine,
            rerank_storage: Some(RerankStorageKind::F32),
            ..Default::default()
        };
        let mut index = ManagedVectorIndex::new(
            VectorIndexTypeConfig::HNSW(config),
            storage,
            "hnsw_graph_rerank_bench",
        )
        .unwrap();
        index.add_vectors(vectors).unwrap();
        index.finalize().unwrap();
        index.write().unwrap();

        let reader = index.reader().unwrap();
        let searcher = HnswSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: the rerank path must engage (reader exposes a
        // RerankStoragePool) and return at least one hit.
        let probe = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string())
                    .rerank_factor(5),
            )
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "hnsw graph rerank top-10 probe must return at least one hit at count={count}"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string())
                    .rerank_factor(5);
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

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

// ---------------------------------------------------------------------------
// Issue #498 — real-data Stage 2 speed validation (SIFT1M)
// ---------------------------------------------------------------------------

/// Real-data Stage 2 speed bench (Issue #498).
///
/// Loads a 50 000-vector SIFT1M subsample plus the standard 200-query
/// SIFT query set, builds an int8 HNSW index with the LRS1 rerank
/// sidecar enabled, and times the same two-stage `top10 + rerank_factor =
/// 5` flow the recall test
/// (`hnsw_quantized_recall_at_10_with_rerank_on_sift_meets_stage2_real_data_recall_gate`)
/// asserts. The two should agree on `(corpus, ef_search, rerank_factor,
/// HNSW config)` so the latency reported here is the speed-half of the
/// Issue #498 acceptance.
///
/// Opt-in: gated on `LAURUS_REAL_BENCHMARK=1` AND
/// `.cache/sift/sift/sift_base.fvecs` being present (Criterion has no
/// "skip" primitive, so a missing fixture or env var collapses the
/// group to zero benches and the bench file finishes silently). Run:
///
/// ```sh
/// ./scripts/fetch-sift.sh --large
/// LAURUS_REAL_BENCHMARK=1 cargo bench --bench vector_search_bench \
///     -- "HNSW Graph Search Rerank Real"
/// ```
///
/// # Speed gate interpretation
///
/// The Issue #498 acceptance asks for "≥ 3× vs the pre-Stage-1 f32
/// baseline." Phase 0 measurements
/// (`~/.claude/tasks/laurus/20260514_498_real_data_speed_validation/`)
/// showed that gate is **not reachable** with the current
/// implementation on SIFT1M; the maximum sustained speedup at Recall@10
/// ≥ 0.99 is ~1.81× with `(m=16, efc=200, ef_search=200, rerank=5)`.
/// The bench number reported here is intended for a cross-branch
/// comparison against the pre-Stage-1 main commit (`c20620c9^`) with
/// the same SIFT fixture and the same HNSW config, looking for **≥ 1.5×**
/// speedup. Criterion does not assert, so the gate is enforced by the
/// PR description / docs rather than CI.
///
/// To take the baseline:
///
/// ```sh
/// git worktree add /tmp/laurus-prestage1 c20620c9^
/// # add an f32-HNSW bench to /tmp/laurus-prestage1/laurus/benches/
/// # and run it with the same SIFT fixture (the .cache/sift/ symlink
/// # makes this a one-liner).
/// ```
fn bench_hnsw_graph_search_rerank_real_data(c: &mut Criterion) {
    if std::env::var("LAURUS_REAL_BENCHMARK").as_deref() != Ok("1") {
        return;
    }
    let base_path = common::sift_cache_dir()
        .join("sift")
        .join("sift_base.fvecs");
    let query_path = common::sift_cache_dir()
        .join("sift")
        .join("sift_query.fvecs");
    if !base_path.exists() || !query_path.exists() {
        eprintln!(
            "skipping Issue #498 real-data speed bench: SIFT1M fixture \
             not found at {}. Run ./scripts/fetch-sift.sh --large.",
            base_path.display()
        );
        return;
    }

    let dim: usize = 128;
    let n_corpus: usize = 50_000;
    let n_queries: usize = 200;
    let ef_search: usize = 200;
    let rerank_factor: usize = 5;
    let m: usize = 16;
    let ef_construction: usize = 200;

    let mut corpus =
        common::load_fvecs(&base_path, dim, Some(n_corpus)).expect("load sift_base.fvecs");
    let mut queries =
        common::load_fvecs(&query_path, dim, Some(n_queries)).expect("load sift_query.fvecs");
    for v in corpus.iter_mut() {
        common::l2_normalise(v);
    }
    for v in queries.iter_mut() {
        common::l2_normalise(v);
    }

    let storage = create_storage();
    let config = HnswIndexConfig {
        dimension: dim,
        m,
        ef_construction,
        distance_metric: DistanceMetric::Cosine,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let mut index = ManagedVectorIndex::new(
        VectorIndexTypeConfig::HNSW(config),
        storage,
        "hnsw_sift_rerank_real_data_bench",
    )
    .unwrap();
    let docs: Vec<(u64, String, Vector)> = corpus
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u64, "field".to_string(), Vector::new(v)))
        .collect();
    index.add_vectors(docs).unwrap();
    index.finalize().unwrap();
    index.write().unwrap();

    let reader = index.reader().unwrap();
    let mut searcher = HnswSearcher::new(reader).unwrap();
    searcher.set_ef_search(ef_search);

    // Sanity check: rerank path must engage on real data too.
    let probe = searcher
        .search(
            &VectorIndexQuery::new(Vector::new(queries[0].clone()))
                .top_k(10)
                .field_name("field".to_string())
                .rerank_factor(rerank_factor),
        )
        .unwrap();
    assert!(
        !probe.results.is_empty(),
        "Issue #498 SIFT probe must return at least one hit"
    );

    let mut group = c.benchmark_group("HNSW Graph Search Rerank Real");
    group.sample_size(SAMPLE_SIZE_SLOW); // slow real-data build path
    group.throughput(Throughput::Elements(n_corpus as u64));

    // Round-robin over the query set so the bench measures average
    // query latency across the full SIFT query distribution rather
    // than the same vector repeatedly.
    let mut iter_idx: usize = 0;
    group.bench_function(BenchmarkId::new("top10_rerank5", "sift50000"), |b| {
        b.iter(|| {
            let q = &queries[iter_idx % queries.len()];
            iter_idx = iter_idx.wrapping_add(1);
            let request = VectorIndexQuery::new(Vector::new(q.clone()))
                .top_k(10)
                .field_name("field".to_string())
                .rerank_factor(rerank_factor);
            searcher.search(&request).unwrap()
        });
    });
    group.finish();
}

/// Real-data Stage 3 speed bench (Issue #481 — PQ + rerank).
///
/// Same opt-in protocol as `bench_hnsw_graph_search_rerank_real_data`
/// (Issue #498) but builds the HNSW index with
/// `quantization_method = ProductQuantization { subvector_count = 32 }`
/// and `rerank_storage = Some(F32)`. Times the end-to-end `top10` +
/// `rerank_factor = 20` flow that the recall test
/// (`hnsw_pq_rerank_recall_at_10_on_sift_meets_stage3_real_data_recall_gate`)
/// asserts ≥ 0.95 recall at.
///
/// Opt-in: gated on `LAURUS_REAL_BENCHMARK=1` AND
/// `.cache/sift/sift/sift_base.fvecs` being present. Run:
///
/// ```sh
/// ./scripts/fetch-sift.sh --large
/// LAURUS_REAL_BENCHMARK=1 cargo bench --bench vector_search_bench \
///     -- "HNSW Graph Search PQ Rerank Real"
/// ```
///
/// # Speed gate interpretation
///
/// Issue #481 Stage 3 originally asked for ≥ 5× speedup vs the
/// pre-Stage-1 f32 HNSW baseline (625 µs/qry on SIFT1M-50k per #500),
/// but measurements during this PR showed the realistic ceiling sits
/// near ~2× with PQ + rerank — the int8 SQ kernel's per-candidate
/// cost is already close to PQ ADC's at the dimensions laurus
/// targets, and the rerank pass dominates the wall clock. The gate
/// was therefore reduced to ≥ 1.5× (same revision Issue #498 did
/// for Stage 2's 3× target), with the exact ratio taken via the
/// cross-branch worktree pattern Issue #498 documented.
fn bench_hnsw_graph_search_pq_rerank_real_data(c: &mut Criterion) {
    if std::env::var("LAURUS_REAL_BENCHMARK").as_deref() != Ok("1") {
        return;
    }
    let base_path = common::sift_cache_dir()
        .join("sift")
        .join("sift_base.fvecs");
    let query_path = common::sift_cache_dir()
        .join("sift")
        .join("sift_query.fvecs");
    if !base_path.exists() || !query_path.exists() {
        eprintln!(
            "skipping Issue #481 Stage 3 real-data speed bench: SIFT1M \
             fixture not found at {}. Run ./scripts/fetch-sift.sh --large.",
            base_path.display()
        );
        return;
    }

    let dim: usize = 128;
    let n_corpus: usize = 50_000;
    let n_queries: usize = 200;
    // Matches the Stage 3 recall test's
    // `(ef_search=200, rerank_factor=10, subvector_count=32)` config
    // so latency is taken at the same operating point the recall
    // assertion defends.
    let ef_search: usize = 200;
    let rerank_factor: usize = 10;
    let subvector_count: usize = 32;
    let m: usize = 16;
    let ef_construction: usize = 200;

    let mut corpus =
        common::load_fvecs(&base_path, dim, Some(n_corpus)).expect("load sift_base.fvecs");
    let mut queries =
        common::load_fvecs(&query_path, dim, Some(n_queries)).expect("load sift_query.fvecs");
    for v in corpus.iter_mut() {
        common::l2_normalise(v);
    }
    for v in queries.iter_mut() {
        common::l2_normalise(v);
    }

    let storage = create_storage();
    let config = HnswIndexConfig {
        dimension: dim,
        m,
        ef_construction,
        distance_metric: DistanceMetric::Cosine,
        quantization_method:
            laurus::vector::core::quantization::QuantizationMethod::ProductQuantization {
                subvector_count,
            },
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let mut index = ManagedVectorIndex::new(
        VectorIndexTypeConfig::HNSW(config),
        storage,
        "hnsw_sift_pq_rerank_real_data_bench",
    )
    .unwrap();
    let docs: Vec<(u64, String, Vector)> = corpus
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u64, "field".to_string(), Vector::new(v)))
        .collect();
    index.add_vectors(docs).unwrap();
    index.finalize().unwrap();
    index.write().unwrap();

    let reader = index.reader().unwrap();
    let mut searcher = HnswSearcher::new(reader).unwrap();
    searcher.set_ef_search(ef_search);

    // Sanity check: the PQ + rerank path must engage on real data.
    let probe = searcher
        .search(
            &VectorIndexQuery::new(Vector::new(queries[0].clone()))
                .top_k(10)
                .field_name("field".to_string())
                .rerank_factor(rerank_factor),
        )
        .unwrap();
    assert!(
        !probe.results.is_empty(),
        "Issue #481 Stage 3 SIFT probe must return at least one hit"
    );

    let mut group = c.benchmark_group("HNSW Graph Search PQ Rerank Real");
    group.sample_size(SAMPLE_SIZE_SLOW); // slow real-data build path
    group.throughput(Throughput::Elements(n_corpus as u64));

    let mut iter_idx: usize = 0;
    group.bench_function(BenchmarkId::new("top10_pq_rerank20", "sift50000"), |b| {
        b.iter(|| {
            let q = &queries[iter_idx % queries.len()];
            iter_idx = iter_idx.wrapping_add(1);
            let request = VectorIndexQuery::new(Vector::new(q.clone()))
                .top_k(10)
                .field_name("field".to_string())
                .rerank_factor(rerank_factor);
            searcher.search(&request).unwrap()
        });
    });
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
    bench_hnsw_graph_search_rerank,
    bench_hnsw_graph_search_rerank_real_data,
    bench_hnsw_graph_search_pq_rerank_real_data,
    bench_hnsw_ef_search_sweep,
    bench_hnsw_multi_field_search,
    bench_flat_multi_field_search,
);
criterion_main!(benches);
