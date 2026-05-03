//! Hybrid (lexical + vector) search benchmarks.
//!
//! Targets the post-#429 coverage gap that `Engine::search` with both a
//! `lexical_query` and a `vector_query` set was never benchmarked. The
//! original audit (#416 item 8) flagged the fusion stage — particularly
//! the dedup + score-merge across two top-K candidate lists — as a
//! likely O(n²) hotspot. Without a hybrid surface, no fusion-side perf
//! change can post a credible before / after.
//!
//! # Scope
//!
//! Three measurement scenarios:
//!
//! 1. **`bench_hybrid_rrf`** — `lexical_query + vector_query` with
//!    `FusionAlgorithm::RRF { k: 60.0 }`, top-K = 10 at corpus 5 000.
//! 2. **`bench_hybrid_weighted_sum`** — same shape, but
//!    `FusionAlgorithm::WeightedSum { lexical_weight: 0.5,
//!    vector_weight: 0.5 }` so cost-difference between fusion strategies
//!    is measurable.
//! 3. **`bench_hybrid_top_k_sweep`** — RRF fusion at fixed corpus 5 000,
//!    K ∈ {10, 100, 500}. Larger K stresses the dedup / merge path the
//!    most.
//!
//! Optional: `LAURUS_BENCH_LARGE=1` adds a 100 000-doc case.
//!
//! # Vocabulary
//!
//! Mirrors the 3-tier Zipf vocabulary used by `lexical_search_bench` and
//! `lexical_indexing_bench`. Each document also carries a deterministic
//! 128-d vector in the `embedding` HNSW field. Inline copy with a
//! "keep in sync" comment; consolidation deferred.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench hybrid_search_bench
//! LAURUS_BENCH_LARGE=1 cargo bench --bench hybrid_search_bench
//! ```
//!
//! Filter by case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench hybrid_search_bench -- "rrf"
//! cargo bench --bench hybrid_search_bench -- "top_k/k_500"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench hybrid_search_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tokio::runtime::Runtime;

use common::{DEFAULT_SEED, SAMPLE_SIZE_FAST, lcg_vec_unit};

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::core::field::IntegerOption;
use laurus::lexical::{TermQuery, TextOption};
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::core::vector::Vector;
use laurus::vector::search::searcher::VectorSearchQuery;
use laurus::vector::store::request::QueryVector;
use laurus::{
    Document, Engine, FusionAlgorithm, LexicalSearchQuery, Result, Schema, SearchRequest,
    SearchRequestBuilder,
};

const DIM: usize = 128;
const VECTOR_FIELD: &str = "embedding";

const COMMON_TERMS: &str = "search engine system data query";

const TOPIC_PHRASES: &[&str] = &[
    "rust programming language systems safety concurrency memory ownership",
    "python data science machine learning artificial intelligence numpy",
    "javascript web development frontend backend node react framework",
    "database query optimization indexing performance search engine",
    "network protocol distributed systems cloud computing infrastructure",
    "security cryptography authentication authorization encryption",
    "algorithms data structures sorting searching graph traversal",
    "operating systems kernel processes threads scheduling memory",
];

const LONG_TAIL: &[&str] = &[
    "compaction",
    "histogram",
    "lattice",
    "registry",
    "compiler",
    "scheduler",
    "interpreter",
    "garbage",
    "collector",
    "allocator",
    "telemetry",
    "regression",
    "snapshot",
    "replication",
    "consensus",
    "quorum",
    "leader",
    "follower",
];

const LONG_TAIL_PER_DOC: usize = 5;

const CATEGORIES: &[&str] = &["programming", "data-science", "web", "database", "systems"];

fn build_body(i: usize) -> String {
    let topic = TOPIC_PHRASES[i % TOPIC_PHRASES.len()];

    let mut tail_words = Vec::with_capacity(LONG_TAIL_PER_DOC);
    for k in 0..LONG_TAIL_PER_DOC {
        let idx = (i.wrapping_mul(7) + k * 11) % LONG_TAIL.len();
        tail_words.push(LONG_TAIL[idx]);
    }
    let tail = tail_words.join(" ");

    format!("Document {i} {COMMON_TERMS} {topic} {tail} should match relevant terms")
}

fn memory_storage() -> Result<Arc<dyn Storage>> {
    StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))
}

async fn build_hybrid_engine(n: usize) -> Result<Engine> {
    let storage = memory_storage()?;
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::default());

    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .add_text_field("category", TextOption::default())
        .add_integer_field("year", IntegerOption::default())
        .add_hnsw_field(
            VECTOR_FIELD,
            HnswOption::new(DIM).distance(DistanceMetric::Cosine),
        )
        .add_default_field("body")
        .build();

    let engine = Engine::builder(storage, schema)
        .analyzer(analyzer)
        .build()
        .await?;

    let mut state = DEFAULT_SEED;
    for i in 0..n {
        let body = build_body(i);
        let vec = lcg_vec_unit(&mut state, DIM);
        let doc = Document::builder()
            .add_text("title", format!("Title for document {i}"))
            .add_text("body", &body)
            .add_text("category", CATEGORIES[i % CATEGORIES.len()])
            .add_integer("year", 2020 + (i % 5) as i64)
            .add_vector(VECTOR_FIELD, vec)
            .build();
        engine.add_document(&i.to_string(), doc).await?;
    }
    engine.commit().await?;
    Ok(engine)
}

fn corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![5_000usize];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(100_000);
    }
    sizes
}

/// Build a deterministic dim-128 query vector. Uses a separate seed
/// offset so the query is not byte-identical to a corpus member.
fn build_query_vector() -> Vector {
    let mut state = DEFAULT_SEED.wrapping_add(1);
    Vector::new(lcg_vec_unit(&mut state, DIM))
}

/// Build a `SearchRequest` carrying both a lexical term query and a
/// vector query, with the given fusion algorithm and top-K.
fn build_hybrid_request(fusion: FusionAlgorithm, limit: usize) -> SearchRequest {
    let lexical = LexicalSearchQuery::Obj(Box::new(TermQuery::new("body", "search")));
    let vector = VectorSearchQuery::Vectors(vec![QueryVector {
        vector: build_query_vector(),
        weight: 1.0,
        fields: Some(vec![VECTOR_FIELD.to_string()]),
    }]);

    SearchRequestBuilder::new()
        .lexical_query(lexical)
        .vector_query(vector)
        .fusion_algorithm(fusion)
        .limit(limit)
        .build()
}

fn bench_hybrid_rrf(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("hybrid/rrf");
    group.sample_size(SAMPLE_SIZE_FAST);

    for &n in &corpus_sizes() {
        let engine = rt.block_on(build_hybrid_engine(n)).unwrap();

        // Sanity: hybrid search must produce non-empty fused results.
        let probe = rt.block_on(async {
            let req = build_hybrid_request(FusionAlgorithm::RRF { k: 60.0 }, 10);
            engine.search(req).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "rrf hybrid probe must return at least one fused hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let req = build_hybrid_request(FusionAlgorithm::RRF { k: 60.0 }, 10);
                    black_box(engine.search(req).await.unwrap())
                }
            });
        });
    }

    group.finish();
}

fn bench_hybrid_weighted_sum(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("hybrid/weighted_sum");
    group.sample_size(SAMPLE_SIZE_FAST);

    for &n in &corpus_sizes() {
        let engine = rt.block_on(build_hybrid_engine(n)).unwrap();

        let probe = rt.block_on(async {
            let req = build_hybrid_request(
                FusionAlgorithm::WeightedSum {
                    lexical_weight: 0.5,
                    vector_weight: 0.5,
                },
                10,
            );
            engine.search(req).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "weighted_sum hybrid probe must return at least one fused hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let req = build_hybrid_request(
                        FusionAlgorithm::WeightedSum {
                            lexical_weight: 0.5,
                            vector_weight: 0.5,
                        },
                        10,
                    );
                    black_box(engine.search(req).await.unwrap())
                }
            });
        });
    }

    group.finish();
}

fn bench_hybrid_top_k_sweep(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("hybrid/top_k");
    group.sample_size(SAMPLE_SIZE_FAST);

    let n = 5_000usize;
    let engine = rt.block_on(build_hybrid_engine(n)).unwrap();

    let probe = rt.block_on(async {
        let req = build_hybrid_request(FusionAlgorithm::RRF { k: 60.0 }, 10);
        engine.search(req).await.unwrap()
    });
    assert!(
        !probe.is_empty(),
        "top_k hybrid probe must return at least one fused hit"
    );

    for &k in &[10usize, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("k_{k}")),
            &k,
            |b, &k| {
                b.to_async(&rt).iter(|| {
                    let engine = &engine;
                    async move {
                        let req = build_hybrid_request(FusionAlgorithm::RRF { k: 60.0 }, k);
                        black_box(engine.search(req).await.unwrap())
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hybrid_rrf,
    bench_hybrid_weighted_sum,
    bench_hybrid_top_k_sweep,
);
criterion_main!(benches);
