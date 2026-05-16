//! Microbenchmarks for the SIMD batch BM25 kernel (#506).
//!
//! These benches isolate the SIMD speedup of
//! [`BM25Scorer::batch_score_into`] versus the per-doc scalar
//! [`BM25Scorer::score`] path. They deliberately avoid the corpus
//! build / matcher / collector overhead present in
//! `lexical_search_bench` so the measurement targets the scoring
//! kernel itself.
//!
//! Why microbenches: the end-to-end `bench_term_query` /
//! `bench_boolean_query` benches at 100 000 documents make the matcher
//! and collector dominate wall time; the SIMD-default loop's effect
//! (#506) only fires in the score step and is easy to drown in the
//! surrounding overhead. The microbench measures the kernel directly
//! so the f32x8 win shows up cleanly — and runs in seconds rather than
//! hours.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use laurus::lexical::query::scorer::{BM25Scorer, Scorer};

/// Construct a representative BM25 scorer. Parameters mirror a typical
/// indexed corpus: a frequent term (doc_freq high relative to
/// total_docs) and the default `(k1, b) = (1.2, 0.75)`.
fn make_scorer() -> BM25Scorer {
    BM25Scorer::new(
        50_000,  // doc_freq
        500_000, // total_term_freq
        100_000, // field_doc_count
        12.0,    // avg_field_length
        100_000, // total_docs
        1.0,     // boost
    )
}

/// Generate a deterministic, varied input pair so SIMD lanes are not
/// trivially predictable. `n` controls the batch size.
fn make_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let term_freqs: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32).collect();
    let field_lengths: Vec<f32> = (0..n).map(|i| 4.0 + (i % 19) as f32).collect();
    (term_freqs, field_lengths)
}

/// Scalar baseline: call `BM25Scorer::score` once per element.
fn bench_scalar(c: &mut Criterion) {
    let scorer = make_scorer();
    let mut group = c.benchmark_group("bm25_simd/scalar");
    for &n in &[8_usize, 64, 1024] {
        let (tf, fl) = make_inputs(n);
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut acc = 0.0_f32;
                for i in 0..n {
                    acc += scorer.score(i as u64, tf[i], Some(fl[i]));
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

/// SIMD batched: single `batch_score_into` call writes `n` results into
/// a reused output buffer. No heap allocation per iteration.
fn bench_simd_batch(c: &mut Criterion) {
    let scorer = make_scorer();
    let mut group = c.benchmark_group("bm25_simd/batch");
    for &n in &[8_usize, 64, 1024] {
        let (tf, fl) = make_inputs(n);
        let mut out = vec![0.0_f32; n];
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                scorer.batch_score_into(&tf, &fl, &mut out);
                black_box(&out);
            });
        });
    }
    group.finish();
}

/// Trait-dispatched batched call through `Box<dyn Scorer>` — mirrors
/// the real call shape inside the searcher default loop (#506) where
/// the scorer is a trait object.
fn bench_simd_batch_dyn(c: &mut Criterion) {
    let scorer: Box<dyn Scorer> = Box::new(make_scorer());
    let mut group = c.benchmark_group("bm25_simd/batch_dyn");
    for &n in &[8_usize, 64, 1024] {
        let (tf, fl) = make_inputs(n);
        let doc_ids: Vec<u64> = (0..n as u64).collect();
        let mut out = vec![0.0_f32; n];
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                scorer.batch_score(&doc_ids, &tf, &fl, &mut out);
                black_box(&out);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_scalar,
    bench_simd_batch,
    bench_simd_batch_dyn
);
criterion_main!(benches);
