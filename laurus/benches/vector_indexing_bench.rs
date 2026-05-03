//! Vector indexing throughput benchmarks.
//!
//! Targets the post-#429 coverage gap that vector indexing was only
//! partially measured. The existing `vector_search_bench::bench_*_construction`
//! lumps `add_vectors + finalize` into a single timed call at small scale
//! (1k / 5k); the `Engine::add_document` path with a vector field was not
//! benchmarked at all.
//!
//! # Scope
//!
//! Four measurement scenarios:
//!
//! 1. **`bench_add_vectors`** — `ManagedVectorIndex::add_vectors` × N for
//!    Flat / IVF / HNSW. Decomposed from `finalize` so each phase is
//!    reportable separately. Sweep N ∈ {1k, 5k}; LARGE adds 50k.
//! 2. **`bench_finalize`** — pre-buffer N vectors, then time
//!    `finalize()` alone. Different shape per index type (Flat is
//!    near-noop, IVF runs k-means, HNSW finalises the graph). N ∈ {1k, 5k}.
//! 3. **`bench_bulk_index`** — `add_vectors + finalize` for the three
//!    index types. Overlaps with the existing `bench_*_construction` in
//!    `vector_search_bench`, but reports `Throughput::Elements` for
//!    per-vector comparability and adds a 50k case under LARGE.
//! 4. **`bench_engine_add_document`** — `Engine::add_document` with a
//!    vector field, the high-level path that goes through schema
//!    dispatch and per-field vector storage. HNSW only (the path is the
//!    same for the other index types). N ∈ {1k, 10k}.
//!
//! Sweep dimension is fixed at 128. A standalone dimension sweep belongs
//! in `distance_bench` and was already added there in #424.
//!
//! # Mock vs real I/O
//!
//! All benches use `MemoryStorage`. Disk-backed variants are tracked
//! separately under #444.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench vector_indexing_bench                       # default
//! LAURUS_BENCH_LARGE=1 cargo bench --bench vector_indexing_bench  # +50k
//! ```
//!
//! Filter by case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench vector_indexing_bench -- "add_vectors/hnsw"
//! cargo bench --bench vector_indexing_bench -- "engine_add_document"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench vector_indexing_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokio::runtime::Runtime;

use common::{DEFAULT_SEED, SAMPLE_SIZE_SLOW, lcg_vec_unit};

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::core::vector::Vector;
use laurus::vector::index::ManagedVectorIndex;
use laurus::vector::index::config::{
    FlatIndexConfig, HnswIndexConfig, IvfIndexConfig, VectorIndexTypeConfig,
};
use laurus::{Document, Engine, Result, Schema};

const DIM: usize = 128;

/// Deterministic vector generation. Uses the suite-wide LCG so two runs
/// produce byte-identical inputs.
fn generate_vectors(count: usize) -> Vec<(u64, String, Vector)> {
    let mut state = DEFAULT_SEED;
    (0..count)
        .map(|i| {
            (
                i as u64,
                "field".to_string(),
                Vector::new(lcg_vec_unit(&mut state, DIM)),
            )
        })
        .collect()
}

fn create_storage() -> Arc<dyn Storage> {
    StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default())).unwrap()
}

/// Default sweep sizes for the low-level (`add_vectors` / `finalize` /
/// `bulk_index`) benches. `LAURUS_BENCH_LARGE=1` appends 50_000 so the
/// HNSW build cost at moderate scale is reachable without ballooning the
/// default `cargo bench` runtime.
fn ingest_corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![1000usize, 5000];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(50_000);
    }
    sizes
}

/// Sizes for the (heavier) Engine-API path. The Engine ingest involves
/// schema dispatch and per-doc bookkeeping on top of the raw vector
/// indexing, so the runtime budget is tighter.
fn engine_corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![1000usize, 10_000];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(50_000);
    }
    sizes
}

fn flat_type_config() -> VectorIndexTypeConfig {
    VectorIndexTypeConfig::Flat(FlatIndexConfig {
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    })
}

fn ivf_type_config() -> VectorIndexTypeConfig {
    VectorIndexTypeConfig::IVF(IvfIndexConfig {
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        n_clusters: 10,
        n_probe: 3,
        ..Default::default()
    })
}

fn hnsw_type_config() -> VectorIndexTypeConfig {
    VectorIndexTypeConfig::HNSW(HnswIndexConfig {
        dimension: DIM,
        m: 16,
        ef_construction: 200,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    })
}

/// Build a fresh `ManagedVectorIndex` for the given index type.
fn fresh_index(index_type: &VectorIndexTypeConfig, name: &'static str) -> ManagedVectorIndex {
    let storage = create_storage();
    ManagedVectorIndex::new(index_type.clone(), storage, name).unwrap()
}

// ---------------------------------------------------------------------------
// 1. add_vectors only
// ---------------------------------------------------------------------------

fn bench_add_vectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ingest/add_vectors");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &ingest_corpus_sizes() {
        let vectors = generate_vectors(n);

        for (label, type_config_factory) in [
            ("flat", flat_type_config as fn() -> VectorIndexTypeConfig),
            ("ivf", ivf_type_config as fn() -> VectorIndexTypeConfig),
            ("hnsw", hnsw_type_config as fn() -> VectorIndexTypeConfig),
        ] {
            // Sanity: a single add_vectors must succeed on a fresh index.
            {
                let mut index = fresh_index(&type_config_factory(), "vec_idx_probe");
                index
                    .add_vectors(vectors.clone())
                    .expect("add_vectors probe must not error");
            }

            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(label, n), &n, |b, _| {
                let vectors = vectors.clone();
                b.iter_batched(
                    || {
                        let index = fresh_index(&type_config_factory(), "vec_idx_add");
                        (index, vectors.clone())
                    },
                    |(mut index, vectors)| {
                        index.add_vectors(vectors).unwrap();
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. finalize only
// ---------------------------------------------------------------------------

fn bench_finalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ingest/finalize");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &[1000usize, 5000] {
        let vectors = generate_vectors(n);

        for (label, type_config_factory) in [
            ("flat", flat_type_config as fn() -> VectorIndexTypeConfig),
            ("ivf", ivf_type_config as fn() -> VectorIndexTypeConfig),
            ("hnsw", hnsw_type_config as fn() -> VectorIndexTypeConfig),
        ] {
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(label, n), &n, |b, _| {
                let vectors = vectors.clone();
                b.iter_batched(
                    || {
                        let mut index = fresh_index(&type_config_factory(), "vec_idx_fin");
                        index.add_vectors(vectors.clone()).unwrap();
                        index
                    },
                    |mut index| {
                        index.finalize().unwrap();
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Bulk: add_vectors + finalize
// ---------------------------------------------------------------------------

fn bench_bulk_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ingest/bulk");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &ingest_corpus_sizes() {
        let vectors = generate_vectors(n);

        for (label, type_config_factory) in [
            ("flat", flat_type_config as fn() -> VectorIndexTypeConfig),
            ("ivf", ivf_type_config as fn() -> VectorIndexTypeConfig),
            ("hnsw", hnsw_type_config as fn() -> VectorIndexTypeConfig),
        ] {
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(label, n), &n, |b, _| {
                let vectors = vectors.clone();
                b.iter_batched(
                    || {
                        let index = fresh_index(&type_config_factory(), "vec_idx_bulk");
                        (index, vectors.clone())
                    },
                    |(mut index, vectors)| {
                        index.add_vectors(vectors).unwrap();
                        index.finalize().unwrap();
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Engine API (Engine::add_document with vector field) — HNSW only
// ---------------------------------------------------------------------------

async fn build_vector_engine() -> Result<Engine> {
    let storage = create_storage();
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::default());

    let schema = Schema::builder()
        .add_hnsw_field(
            "embedding",
            HnswOption::new(DIM).distance(DistanceMetric::Cosine),
        )
        .build();

    Engine::builder(storage, schema)
        .analyzer(analyzer)
        .build()
        .await
}

/// Build `n` `(id, Document)` pairs, each carrying a deterministic
/// dim-128 vector in `embedding`.
fn build_engine_vector_documents(n: usize) -> Vec<(String, Document)> {
    let mut state = DEFAULT_SEED;
    (0..n)
        .map(|i| {
            let vec = lcg_vec_unit(&mut state, DIM);
            let doc = Document::builder().add_vector("embedding", vec).build();
            (i.to_string(), doc)
        })
        .collect()
}

fn bench_engine_add_document(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("vector_ingest/engine_add_document");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &engine_corpus_sizes() {
        // Sanity: a single add_document with a vector field must succeed
        // on a fresh engine.
        {
            let engine = rt.block_on(build_vector_engine()).unwrap();
            let docs = build_engine_vector_documents(1);
            rt.block_on(async {
                for (id, doc) in docs {
                    engine.add_document(&id, doc).await.unwrap();
                }
            });
        }

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let engine = rt.block_on(build_vector_engine()).unwrap();
                    let docs = build_engine_vector_documents(n);
                    (engine, docs)
                },
                |(engine, docs)| {
                    rt.block_on(async {
                        for (id, doc) in docs {
                            engine.add_document(&id, doc).await.unwrap();
                        }
                    });
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_add_vectors,
    bench_finalize,
    bench_bulk_index,
    bench_engine_add_document,
);
criterion_main!(benches);
