//! Document mutation benchmarks: delete, update (put), delete-then-commit.
//!
//! Targets the post-#429 coverage gap that the mutation path
//! (`Engine::delete_documents`, `Engine::put_document`, post-delete
//! commit / compaction) was never benchmarked.
//!
//! # Scope
//!
//! Three measurement scenarios:
//!
//! 1. **`bench_delete_documents`** — pre-populate the engine with N
//!    docs + commit, then time `delete_documents(id) × M` (no commit).
//!    Reports per-delete latency. Sweep N ∈ {1k, 10k}, M = 100.
//! 2. **`bench_put_document`** — pre-populate the engine with N docs,
//!    then time `put_document(id, new_doc) × M`. `put_document` is the
//!    "upsert" path (delete existing then re-add), so this measures
//!    the combined cost. Sweep N ∈ {1k, 10k}, M = 100.
//! 3. **`bench_delete_then_commit`** — pre-populate with N docs,
//!    delete a fraction (10 %), then time the subsequent `commit()`.
//!    Crosses the `DeletionConfig::auto_compaction` threshold for
//!    larger N, so the cost shape includes any segment-rewrite work.
//!    Sweep N ∈ {1k, 10k}.
//!
//! Schema-driven reindex (e.g. adding a field and timing the rebuild)
//! is out of scope — no public API surfaces it directly today.
//!
//! # Vocabulary
//!
//! Mirrors the Zipf set from `lexical_search_bench` /
//! `lexical_indexing_bench`. Inline copy with a "keep in sync" header.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench mutation_bench
//! ```
//!
//! Filter by case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench mutation_bench -- "delete_documents/1000"
//! cargo bench --bench mutation_bench -- "delete_then_commit"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench mutation_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokio::runtime::Runtime;

use common::{SAMPLE_SIZE_SLOW, select_storage};

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::TextOption;
use laurus::lexical::core::field::IntegerOption;
use laurus::storage::Storage;
use laurus::{Document, Engine, Result, Schema};

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

fn build_document(i: usize) -> Document {
    let body = build_body(i);
    Document::builder()
        .add_text("title", format!("Title for document {i}"))
        .add_text("body", &body)
        .add_text("category", CATEGORIES[i % CATEGORIES.len()])
        .add_integer("year", 2020 + (i % 5) as i64)
        .build()
}

/// Bench-storage handle. Delegates to `common::select_storage()` so
/// `LAURUS_BENCH_DISK=1` swaps the in-memory backend for a temp-dir
/// `FileStorage` without changing call sites.
fn memory_storage() -> Result<Arc<dyn Storage>> {
    Ok(select_storage())
}

async fn build_populated_engine(n: usize) -> Result<Engine> {
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
        engine
            .add_document(&i.to_string(), build_document(i))
            .await?;
    }
    engine.commit().await?;
    Ok(engine)
}

/// Number of mutation operations per timed iteration. 100 keeps each
/// iteration's wall-time short enough that SAMPLE_SIZE_SLOW (10 samples)
/// finishes promptly while still amortising any per-iteration noise.
const M_OPS: usize = 100;

// ----------------------------------------------------------------------------
// 1. delete_documents
// ----------------------------------------------------------------------------

fn bench_delete_documents(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("mutation/delete_documents");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &[1000usize, 10_000] {
        // Sanity: a fresh populated engine + delete must succeed.
        {
            let engine = rt.block_on(build_populated_engine(n)).unwrap();
            rt.block_on(engine.delete_documents("0")).unwrap();
        }

        group.throughput(Throughput::Elements(M_OPS as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let engine = rt.block_on(build_populated_engine(n)).unwrap();
                    // Pick M deterministic IDs spread evenly across the
                    // corpus so every delete actually matches.
                    let ids: Vec<String> =
                        (0..M_OPS).map(|i| ((i * n) / M_OPS).to_string()).collect();
                    (engine, ids)
                },
                |(engine, ids)| {
                    rt.block_on(async {
                        for id in ids {
                            engine.delete_documents(&id).await.unwrap();
                        }
                    });
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ----------------------------------------------------------------------------
// 2. put_document (upsert = delete + add)
// ----------------------------------------------------------------------------

fn bench_put_document(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("mutation/put_document");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &[1000usize, 10_000] {
        // Sanity: put_document on an existing id replaces.
        {
            let engine = rt.block_on(build_populated_engine(n)).unwrap();
            let new_doc = build_document(0);
            rt.block_on(engine.put_document("0", new_doc)).unwrap();
        }

        group.throughput(Throughput::Elements(M_OPS as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let engine = rt.block_on(build_populated_engine(n)).unwrap();
                    let updates: Vec<(String, Document)> = (0..M_OPS)
                        .map(|i| {
                            let id = ((i * n) / M_OPS).to_string();
                            // Updated content: bump i so the new body
                            // differs from the original.
                            let new_doc = build_document(n + i);
                            (id, new_doc)
                        })
                        .collect();
                    (engine, updates)
                },
                |(engine, updates)| {
                    rt.block_on(async {
                        for (id, doc) in updates {
                            engine.put_document(&id, doc).await.unwrap();
                        }
                    });
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ----------------------------------------------------------------------------
// 3. delete-heavy + commit (may trigger auto-compaction)
// ----------------------------------------------------------------------------

fn bench_delete_then_commit(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("mutation/delete_then_commit");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &[1000usize, 10_000] {
        let to_delete = n / 10; // delete 10 % of the corpus

        // Sanity probe — sequence runs without error.
        {
            let engine = rt.block_on(build_populated_engine(n)).unwrap();
            rt.block_on(async {
                for i in 0..to_delete {
                    engine.delete_documents(&i.to_string()).await.unwrap();
                }
                engine.commit().await.unwrap();
            });
        }

        group.throughput(Throughput::Elements(to_delete as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let engine = rt.block_on(build_populated_engine(n)).unwrap();
                    // Apply the delete batch in setup; only the commit
                    // is timed below.
                    rt.block_on(async {
                        for i in 0..to_delete {
                            engine.delete_documents(&i.to_string()).await.unwrap();
                        }
                    });
                    engine
                },
                |engine| {
                    rt.block_on(engine.commit()).unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_delete_documents,
    bench_put_document,
    bench_delete_then_commit,
);
criterion_main!(benches);
