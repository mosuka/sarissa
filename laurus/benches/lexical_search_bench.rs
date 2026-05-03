//! End-to-end lexical search benchmarks.
//!
//! Measures full query execution time including matching, scoring, and
//! collection for various query types: `TermQuery`, `BooleanQuery`,
//! `PhraseQuery`, `FuzzyQuery`, and the DSL parser.
//!
//! # Scope
//!
//! - End-to-end through `Engine::search`, including async runtime overhead.
//! - Corpus sizes 100, 1 000, 5 000 documents by default; opt in to a
//!   100 000-document case via `LAURUS_BENCH_LARGE=1` (see "Large-corpus
//!   gate" below).
//! - Vocabulary follows a 3-tier Zipf-like distribution (see
//!   "Vocabulary").
//! - One-time correctness assert per bench function verifies the probe
//!   query returns at least one hit before the timed loop runs.
//!
//! # Vocabulary
//!
//! Document bodies are built from three tiers so term-frequency
//! distribution approximates Zipf's law (a few very common terms, several
//! medium-frequency topic phrases, and many rare long-tail words):
//!
//! - [`COMMON_TERMS`] (5 words: `search`, `system`, `data`, `engine`,
//!   `query`): present in **every** document. High document-frequency,
//!   low IDF — these are the WAND-friendly terms that #403 (top-K
//!   early-termination) targets.
//! - [`TOPIC_PHRASES`] (8 phrases × 8 words): each phrase appears in
//!   `1/8` of documents (≈ 12.5 %).
//! - [`LONG_TAIL`] (~80 domain words): each document picks 5 words from
//!   this pool by stride; each individual word appears in roughly 5 % of
//!   documents.
//!
//! `bench_phrase_query` queries the phrase `"search engine"`; both words
//! come from `COMMON_TERMS`, so the phrase is guaranteed to occur in every
//! document.
//!
//! # Large-corpus gate
//!
//! Setting the environment variable `LAURUS_BENCH_LARGE=1` adds a
//! 100 000-document case to `bench_term_query`,
//! `bench_term_query_varying_limit`, and `bench_boolean_query`. The
//! 100 000-document setup is several seconds, and per-iter search at
//! 100 k is in the ms range, so this large case uses the slow
//! `sample_size` (`SAMPLE_SIZE_SLOW`) automatically. Default runs (no env
//! var) finish in well under five minutes on a typical workstation.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench lexical_search_bench                            # default sizes
//! LAURUS_BENCH_LARGE=1 cargo bench --bench lexical_search_bench       # adds 100 k
//! ```
//!
//! Filter by group / case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench lexical_search_bench -- term_query
//! cargo bench --bench lexical_search_bench -- boolean_query/should_or_high_freq
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench lexical_search_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::hint::black_box;
use std::sync::Arc;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use tokio::runtime::Runtime;

use common::{SAMPLE_SIZE_FAST, SAMPLE_SIZE_SLOW, select_storage};

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::core::field::IntegerOption;
use laurus::lexical::{BooleanQuery, FuzzyQuery, PhraseQuery, TermQuery, TextOption};
use laurus::storage::Storage;
use laurus::{Document, Engine, LexicalSearchQuery, Result, Schema, SearchRequestBuilder};

/// Words present in **every** document. Drives the high-document-frequency
/// region of the term distribution and gives WAND / MaxScore early-
/// termination (#403) something to prune against. The pair `search engine`
/// always appears in this exact order so `bench_phrase_query` has a
/// guaranteed match.
const COMMON_TERMS: &str = "search engine system data query";

/// Topic phrases — each appears in `1/8` of documents (≈ 12.5 %).
/// Mid-frequency tier. Each phrase is internally diverse so individual
/// words inside it get further-spread frequency.
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

/// Long-tail vocabulary. Each document picks `LONG_TAIL_PER_DOC` words
/// from this pool by stride, so any individual word appears in roughly
/// `LONG_TAIL_PER_DOC / LONG_TAIL.len()` ≈ 6 % of documents. Provides the
/// rare-term tail of the Zipf distribution.
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
    "hashicorp",
    "kubernetes",
    "docker",
    "mesos",
    "borg",
    "raft",
    "paxos",
    "tantivy",
    "lucene",
    "solr",
    "elasticsearch",
    "qdrant",
    "weaviate",
    "milvus",
    "embedding",
    "transformer",
    "attention",
    "softmax",
    "gradient",
    "backprop",
    "tensor",
    "matrix",
    "vector",
    "scalar",
    "tokenizer",
    "lemmatizer",
    "stemmer",
    "synonym",
    "antonym",
    "wordnet",
    "thesaurus",
    "ontology",
    "taxonomy",
    "shingle",
    "ngram",
    "unigram",
    "bigram",
    "trigram",
    "skipgram",
    "rocksdb",
    "leveldb",
    "bigtable",
    "hbase",
    "cassandra",
    "scylla",
    "kafka",
    "pulsar",
    "rabbitmq",
    "zeromq",
    "nats",
    "redis",
    "prometheus",
    "grafana",
    "jaeger",
    "zipkin",
    "opentelemetry",
    "wasm",
    "webassembly",
    "wasi",
    "nodejs",
    "deno",
    "bun",
    "criterion",
    "tarpaulin",
    "miri",
    "loom",
    "fuzzer",
    "afl",
];

/// Number of long-tail words appended to each document body.
const LONG_TAIL_PER_DOC: usize = 5;

const CATEGORIES: &[&str] = &["programming", "data-science", "web", "database", "systems"];

/// Build a deterministic document body following the 3-tier distribution
/// described at the top of this file.
fn build_body(i: usize) -> String {
    let topic = TOPIC_PHRASES[i % TOPIC_PHRASES.len()];

    // Pick LONG_TAIL_PER_DOC distinct long-tail words by stride. Using a
    // co-prime stride (7) and starting offset (i) means consecutive
    // documents share few long-tail words but the distribution stays
    // even-ish across the whole corpus.
    let mut tail_words = Vec::with_capacity(LONG_TAIL_PER_DOC);
    for k in 0..LONG_TAIL_PER_DOC {
        let idx = (i.wrapping_mul(7) + k * 11) % LONG_TAIL.len();
        tail_words.push(LONG_TAIL[idx]);
    }
    let tail = tail_words.join(" ");

    format!("Document {i} {COMMON_TERMS} {topic} {tail} should match relevant terms")
}

/// Bench-storage handle. Delegates to `common::select_storage()` so
/// `LAURUS_BENCH_DISK=1` swaps the in-memory backend for a temp-dir
/// `FileStorage` without changing call sites. The function still returns
/// `Result` for compatibility with the existing async `build_engine`.
fn memory_storage() -> Result<Arc<dyn Storage>> {
    Ok(select_storage())
}

/// Build a pre-populated engine with `n` documents using the 3-tier
/// vocabulary described above.
async fn build_engine(n: usize) -> Result<Engine> {
    let storage = memory_storage()?;
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::default());

    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .add_text_field("category", TextOption::default())
        .add_integer_field("year", IntegerOption::default())
        .add_default_field("body")
        .build();

    let engine = Engine::builder(storage, schema)
        .analyzer(analyzer)
        .build()
        .await?;

    for i in 0..n {
        let body = build_body(i);
        let doc = Document::builder()
            .add_text("title", format!("Title for document {i}"))
            .add_text("body", &body)
            .add_text("category", CATEGORIES[i % CATEGORIES.len()])
            .add_integer("year", 2020 + (i % 5) as i64)
            .build();

        engine.add_document(&i.to_string(), doc).await?;
    }

    engine.commit().await?;
    Ok(engine)
}

/// Default corpus sizes for sweep cases. Setting `LAURUS_BENCH_LARGE=1`
/// in the environment appends a 100 000-document case so #403 (WAND /
/// MaxScore early-termination) has a measurable target without ballooning
/// the default `cargo bench` runtime.
fn corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![100usize, 1_000, 5_000];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(100_000);
    }
    sizes
}

/// Pick `sample_size` for a corpus of `n` documents. Large corpora use the
/// slow tier so wall time stays bounded.
fn sample_size_for(n: usize) -> usize {
    if n >= 50_000 {
        SAMPLE_SIZE_SLOW
    } else {
        SAMPLE_SIZE_FAST
    }
}

/// Apply the sample-size policy to a benchmark group. Picks the slowest
/// (= smallest) sample size required by any element of `sizes` so a single
/// group can host both small and large cases without losing precision on
/// the big end.
fn apply_sample_size(group: &mut BenchmarkGroup<'_, WallTime>, sizes: &[usize]) {
    let sample = sizes
        .iter()
        .copied()
        .map(sample_size_for)
        .min()
        .unwrap_or(SAMPLE_SIZE_FAST);
    group.sample_size(sample);
}

fn bench_term_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/term_query");
    let sizes = corpus_sizes();
    apply_sample_size(&mut group, &sizes);

    for &n in &sizes {
        let engine = rt.block_on(build_engine(n)).unwrap();

        // One-time sanity check: the probe query must return at least one
        // hit. If this fires, the corpus / query pair drifted out of sync.
        let probe = rt.block_on(async {
            let query = Box::new(TermQuery::new("body", "programming"));
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "term_query probe must return at least one hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::new("search", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let query = Box::new(TermQuery::new("body", "programming"));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(query))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });
    }
    group.finish();
}

fn bench_term_query_varying_limit(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    // Pick the largest corpus available (100 k under LAURUS_BENCH_LARGE,
    // 5 000 otherwise) — limit-sweep is most informative when there are
    // far more matching docs than the largest top-K asked for.
    let corpus_n = *corpus_sizes()
        .last()
        .expect("corpus_sizes() must be non-empty");
    let engine = rt.block_on(build_engine(corpus_n)).unwrap();

    // Sanity check: the largest limit must return more hits than the smallest.
    let probe_small = rt.block_on(async {
        let query = Box::new(TermQuery::new("body", "programming"));
        let request = SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Obj(query))
            .limit(10)
            .build();
        engine.search(request).await.unwrap()
    });
    assert!(
        !probe_small.is_empty(),
        "term_query_limit probe must return at least one hit (corpus={corpus_n})"
    );

    let mut group = c.benchmark_group("lexical/term_query_limit");
    apply_sample_size(&mut group, &[corpus_n]);

    for &limit in &[10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::new(format!("corpus_{corpus_n}/top"), limit),
            &limit,
            |b, &limit| {
                b.to_async(&rt).iter(|| {
                    let engine = &engine;
                    async move {
                        let query = Box::new(TermQuery::new("body", "programming"));
                        let request = SearchRequestBuilder::new()
                            .lexical_query(LexicalSearchQuery::Obj(query))
                            .limit(limit)
                            .build();
                        black_box(engine.search(request).await.unwrap())
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_boolean_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/boolean_query");
    let sizes = corpus_sizes();
    apply_sample_size(&mut group, &sizes);

    for &n in &sizes {
        let engine = rt.block_on(build_engine(n)).unwrap();

        // Sanity check: the OR probe over three known terms must hit.
        let probe = rt.block_on(async {
            let mut bq = BooleanQuery::new();
            bq.add_should(Box::new(TermQuery::new("body", "rust")));
            bq.add_should(Box::new(TermQuery::new("body", "python")));
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "boolean_query OR probe must return at least one hit at n={n}"
        );

        // MUST + MUST (AND)
        group.bench_with_input(BenchmarkId::new("must_and", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let mut bq = BooleanQuery::new();
                    bq.add_must(Box::new(TermQuery::new("body", "programming")));
                    bq.add_must(Box::new(TermQuery::new("body", "language")));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });

        // SHOULD + SHOULD (OR) — three topic terms (≈ 12.5 % each).
        group.bench_with_input(BenchmarkId::new("should_or", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let mut bq = BooleanQuery::new();
                    bq.add_should(Box::new(TermQuery::new("body", "rust")));
                    bq.add_should(Box::new(TermQuery::new("body", "python")));
                    bq.add_should(Box::new(TermQuery::new("body", "javascript")));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });

        // SHOULD + SHOULD (OR) — three high-frequency COMMON_TERMS that
        // appear in every document. This is the workload WAND / MaxScore
        // (#403) targets: an OR over high-DF terms where most candidates
        // can be skipped once the top-K min-score is established.
        group.bench_with_input(BenchmarkId::new("should_or_high_freq", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let mut bq = BooleanQuery::new();
                    bq.add_should(Box::new(TermQuery::new("body", "search")));
                    bq.add_should(Box::new(TermQuery::new("body", "system")));
                    bq.add_should(Box::new(TermQuery::new("body", "data")));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });

        // MUST + MUST_NOT (AND NOT)
        group.bench_with_input(BenchmarkId::new("must_not", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let mut bq = BooleanQuery::new();
                    bq.add_must(Box::new(TermQuery::new("body", "programming")));
                    bq.add_must_not(Box::new(TermQuery::new("body", "python")));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });
    }
    group.finish();
}

fn bench_phrase_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/phrase_query");
    // Phrase / fuzzy / DSL benches stay on the fixed small-corpus sweep —
    // the audit's scale-up requirement (#422) only targets term and
    // boolean cases. Sample size is set explicitly so we do not rely on
    // criterion's implicit default.
    group.sample_size(SAMPLE_SIZE_FAST);

    for &n in &[100, 1000, 5000] {
        let engine = rt.block_on(build_engine(n)).unwrap();

        // Sanity check: the two-term phrase probe must hit at least once.
        let probe = rt.block_on(async {
            let query = Box::new(PhraseQuery::new(
                "body",
                vec!["search".into(), "engine".into()],
            ));
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "phrase_query probe must return at least one hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::new("two_terms", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let query = Box::new(PhraseQuery::new(
                        "body",
                        vec!["search".into(), "engine".into()],
                    ));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(query))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("three_terms", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let query = Box::new(PhraseQuery::new(
                        "body",
                        vec!["practical".into(), "applications".into(), "in".into()],
                    ));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(query))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });
    }
    group.finish();
}

fn bench_fuzzy_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/fuzzy_query");
    group.sample_size(SAMPLE_SIZE_FAST);

    for &n in &[100, 1000, 5000] {
        let engine = rt.block_on(build_engine(n)).unwrap();

        // Sanity check: the edit1 probe must match `programming`.
        let probe = rt.block_on(async {
            let query = Box::new(FuzzyQuery::new("body", "programing").max_edits(1));
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "fuzzy_query probe must return at least one hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::new("edit1", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let query = Box::new(FuzzyQuery::new("body", "programing").max_edits(1));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(query))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("edit2", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let query = Box::new(FuzzyQuery::new("body", "progrming").max_edits(2));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(query))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });
    }
    group.finish();
}

fn bench_dsl_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(build_engine(5000)).unwrap();

    // Sanity check: the simple_term DSL probe must hit.
    let probe = rt.block_on(async {
        let request = SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Dsl("body:programming".to_string()))
            .limit(10)
            .build();
        engine.search(request).await.unwrap()
    });
    assert!(
        !probe.is_empty(),
        "dsl_query probe must return at least one hit"
    );

    let mut group = c.benchmark_group("lexical/dsl_query");
    group.sample_size(SAMPLE_SIZE_FAST);

    group.bench_function("simple_term", |b| {
        b.to_async(&rt).iter(|| {
            let engine = &engine;
            async move {
                let request = SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::Dsl("body:programming".to_string()))
                    .limit(10)
                    .build();
                black_box(engine.search(request).await.unwrap())
            }
        });
    });

    group.bench_function("boolean_and", |b| {
        b.to_async(&rt).iter(|| {
            let engine = &engine;
            async move {
                let request = SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::Dsl(
                        "body:programming AND body:language".to_string(),
                    ))
                    .limit(10)
                    .build();
                black_box(engine.search(request).await.unwrap())
            }
        });
    });

    group.bench_function("boolean_or", |b| {
        b.to_async(&rt).iter(|| {
            let engine = &engine;
            async move {
                let request = SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::Dsl(
                        "body:rust OR body:python OR body:javascript".to_string(),
                    ))
                    .limit(10)
                    .build();
                black_box(engine.search(request).await.unwrap())
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_term_query,
    bench_term_query_varying_limit,
    bench_boolean_query,
    bench_phrase_query,
    bench_fuzzy_query,
    bench_dsl_query,
);
criterion_main!(benches);
