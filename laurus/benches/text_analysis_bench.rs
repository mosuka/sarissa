//! Criterion benchmarks for text analysis (tokenization).
//!
//! Measures the cost of running [`StandardAnalyzer`] over synthetic English
//! documents. Analysis is on the hot path for both ingestion (every indexed
//! document) and search (every query before posting-list traversal), so a
//! regression here shows up across the entire engine.
//!
//! # Scope
//!
//! - Single-document analysis (`analyze_single_document`).
//! - Batch analysis of 100 documents (`analyze_batch_documents`,
//!   `Throughput::Elements(100)`).
//! - Documents are deterministic — words are picked from a fixed 32-word
//!   vocabulary by a stride-based pseudo-random index, no external RNG.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench text_analysis_bench
//! ```
//!
//! Filter by case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench text_analysis_bench -- analyze_single_document
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench text_analysis_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;

/// Generate `count` deterministic test documents with variable length.
///
/// Each document is between 50 and 149 words long, drawn from a fixed
/// 32-word vocabulary using a stride-based index (no RNG required, fully
/// reproducible).
fn generate_test_documents(count: usize) -> Vec<String> {
    let words = [
        "search",
        "engine",
        "full",
        "text",
        "index",
        "query",
        "document",
        "field",
        "term",
        "phrase",
        "boolean",
        "vector",
        "similarity",
        "relevance",
        "score",
        "analysis",
        "tokenization",
        "stemming",
        "normalization",
        "clustering",
        "machine",
        "learning",
        "algorithm",
        "data",
        "structure",
        "performance",
        "optimization",
        "memory",
        "storage",
        "retrieval",
        "ranking",
        "filtering",
    ];

    let mut documents = Vec::with_capacity(count);
    for i in 0..count {
        let doc_length = 50 + (i % 100);
        let mut doc_words = Vec::with_capacity(doc_length);

        for j in 0..doc_length {
            let word_idx = (i * 7 + j * 13) % words.len();
            doc_words.push(words[word_idx]);
        }

        documents.push(doc_words.join(" "));
    }

    documents
}

fn bench_text_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_analysis");

    let analyzer = StandardAnalyzer::new().unwrap();
    let texts = generate_test_documents(1000);

    // One-time sanity check: the analyzer must produce at least one token
    // for the probe document. If this fails the bench premise is broken.
    let probe: Vec<_> = analyzer.analyze(&texts[0]).unwrap().collect();
    assert!(
        !probe.is_empty(),
        "analyzer probe must yield at least one token"
    );

    group.bench_function("analyze_single_document", |b| {
        b.iter(|| {
            let result = analyzer.analyze(black_box(&texts[0]));
            black_box(result)
        })
    });

    group.throughput(Throughput::Elements(100));
    group.bench_function("analyze_batch_documents", |b| {
        b.iter(|| {
            for text in texts.iter().take(100) {
                let result = analyzer.analyze(black_box(text));
                let _ = black_box(result);
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_text_analysis);
criterion_main!(benches);
