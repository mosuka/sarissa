//! Criterion benchmarks for [`BM25ScoringFunction::score`] from
//! `lexical::search::scoring::bm25`.
//!
//! This is the audit target tracked under #402 (precompute IDF per query and
//! switch term lookups to `TermId`). Note that this is a **different** code
//! path from `BM25Scorer` in `lexical::query::scorer`, which is benchmarked
//! by `search_perf.rs::bench_bm25_scoring`.
//!
//! # Scope
//!
//! - Direct micro-bench of `BM25ScoringFunction::score(query_terms, doc_stats,
//!   collection_stats, config)`.
//! - Sweep: `query_terms.len() ∈ {1, 5, 10}` × `candidate_count ∈ {100, 10k,
//!   100k}`.
//! - Each iteration runs `score()` once per candidate against the same query,
//!   so the timed work scales with `query_len * candidate_count` — exactly
//!   what #402 will reduce.
//! - Inputs are deterministic via `common::DEFAULT_SEED`; vocabulary is 100
//!   synthetic terms (`term_<i>`) shared between query, document term
//!   frequencies, and collection document frequencies.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench bm25_scoring_bench
//! ```
//!
//! Filter by case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench bm25_scoring_bench -- "qterms_5/100000"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench bm25_scoring_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use laurus::lexical::search::scoring::bm25::{
    BM25ScoringFunction, CollectionStats, DocumentStats, ScoringConfig, ScoringFunction,
};

use common::{DEFAULT_SEED, lcg_next_unit};

/// Vocabulary size for the synthetic corpus. Picked to be larger than the
/// largest `query_terms.len()` swept (10) by an order of magnitude so that
/// query terms span a non-trivial slice of the document-frequency map.
const VOCAB_SIZE: usize = 100;

/// Average document length used to populate `CollectionStats::avg_doc_length`
/// and the per-document `doc_length`. Mirrors a typical short-text field.
const AVG_DOC_LENGTH: f64 = 100.0;

/// Total document count reported in `CollectionStats::total_docs`. Affects
/// IDF magnitude but not the per-call scoring cost.
const TOTAL_DOCS: u64 = 1_000_000;

/// Build the static vocabulary `["term_0", "term_1", …, "term_VOCAB_SIZE-1"]`.
fn build_vocab() -> Vec<String> {
    (0..VOCAB_SIZE).map(|i| format!("term_{i}")).collect()
}

/// Pick `n` query terms from `vocab` using a deterministic LCG step.
/// The same seed produces the same query across runs so two `cargo bench`
/// invocations are directly comparable.
fn build_query(vocab: &[String], n: usize, state: &mut u64) -> Vec<String> {
    (0..n)
        .map(|_| {
            let pick = (lcg_next_unit(state) * VOCAB_SIZE as f32) as usize % VOCAB_SIZE;
            vocab[pick].clone()
        })
        .collect()
}

/// Build a collection-wide stats object whose `document_frequencies` is
/// populated for every term in `vocab`. Frequencies follow a smooth
/// pseudo-random distribution in `[1, total_docs)` so IDF varies per term.
fn build_collection_stats(vocab: &[String]) -> CollectionStats {
    let mut state = DEFAULT_SEED.wrapping_add(0xC011_EC71);
    let mut document_frequencies = HashMap::with_capacity(vocab.len());
    for term in vocab {
        let df = (lcg_next_unit(&mut state) * (TOTAL_DOCS as f32 - 1.0)) as u64 + 1;
        document_frequencies.insert(term.clone(), df);
    }
    CollectionStats {
        total_docs: TOTAL_DOCS,
        avg_doc_length: AVG_DOC_LENGTH,
        avg_field_lengths: HashMap::new(),
        document_frequencies,
        field_document_frequencies: HashMap::new(),
    }
}

/// Build `count` per-document stats objects. Each document's
/// `term_frequencies` covers the full vocabulary with a deterministic
/// pseudo-random TF in `[0, 10]`. Document length tracks `AVG_DOC_LENGTH`
/// with light variance so the BM25 length-normalisation factor varies.
fn build_doc_stats(vocab: &[String], count: usize) -> Vec<DocumentStats> {
    let mut state = DEFAULT_SEED.wrapping_add(0xD0C57A75);
    (0..count)
        .map(|i| {
            let mut term_frequencies = HashMap::with_capacity(vocab.len());
            for term in vocab {
                let tf = (lcg_next_unit(&mut state) * 10.0) as u64;
                term_frequencies.insert(term.clone(), tf);
            }
            let doc_length = AVG_DOC_LENGTH as u64
                + ((lcg_next_unit(&mut state) * 50.0) as u64).saturating_sub(25);
            DocumentStats {
                doc_id: i as u32,
                doc_length,
                field_lengths: HashMap::new(),
                term_frequencies,
                field_term_frequencies: HashMap::new(),
            }
        })
        .collect()
}

fn bench_bm25_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_scoring/score");

    let scorer = BM25ScoringFunction;
    let config = ScoringConfig::default();
    let vocab = build_vocab();
    let collection = build_collection_stats(&vocab);

    // Build the candidate document sets once per `n_candidates` so the
    // (expensive) HashMap-construction cost is not repeated for every
    // `n_qterms` value. The score() call itself is what we measure.
    for &n_candidates in &[100usize, 10_000, 100_000] {
        let docs = build_doc_stats(&vocab, n_candidates);

        for &n_qterms in &[1usize, 5, 10] {
            let mut state = DEFAULT_SEED.wrapping_add(n_qterms as u64);
            let query = build_query(&vocab, n_qterms, &mut state);

            // One-time sanity check: scoring at least one document must
            // yield a finite value. Note: BM25 IDF can be negative when a
            // term's document frequency exceeds half the collection, so a
            // negative score is legitimate; only NaN / Inf indicate broken
            // inputs.
            let probe = scorer
                .score(&query, &docs[0], &collection, &config)
                .expect("scoring probe must not error");
            assert!(
                probe.is_finite(),
                "BM25 probe score must be finite, got {probe}"
            );

            group.throughput(Throughput::Elements(n_candidates as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("qterms_{n_qterms}/{n_candidates}")),
                &(),
                |b, _| {
                    b.iter(|| {
                        for doc in &docs {
                            let score = scorer
                                .score(
                                    black_box(&query),
                                    black_box(doc),
                                    black_box(&collection),
                                    black_box(&config),
                                )
                                .unwrap();
                            black_box(score);
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_bm25_score);
criterion_main!(benches);
