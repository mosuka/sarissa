//! Criterion benchmarks for [`AdvancedSimilarityMetric::WeightedCosine`]
//! from `vector::search::scoring::similarity`.
//!
//! This is the audit target tracked under #414 (precompute the query-side
//! cosine norm once per search). Note that this is a **different** code path
//! from `DistanceMetric::Cosine` in `vector::core::distance`, which is
//! benchmarked by `distance_bench.rs`. The advanced metric supports
//! per-dimension weights, recomputes both norms per call, and clamps the
//! output to `[0, 1]`.
//!
//! # Scope
//!
//! - Direct micro-bench of `AdvancedSimilarityMetric::WeightedCosine.similarity(a, b, Some(weights))`.
//! - Sweep: `dim ∈ {64, 128, 384, 768, 1024}` × `candidate_count ∈ {1, 100, 10k}`.
//! - Each iteration runs `similarity()` once per candidate against the same
//!   query, so the timed work scales with `dim * candidate_count` — exactly
//!   what #414 will reduce by caching the query-side norm.
//! - Inputs are deterministic via `common::DEFAULT_SEED`. Components are in
//!   `[0, 1)`; weights are uniform 1.0 (the documented default-equivalent
//!   shape — perf-sensitive code path is identical to the unweighted case
//!   and #414 covers both).
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench cosine_similarity_bench
//! ```
//!
//! Filter by case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench cosine_similarity_bench -- "dim_768/10000"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench cosine_similarity_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use laurus::vector::core::vector::Vector;
use laurus::vector::search::scoring::similarity::AdvancedSimilarityMetric;

use common::{DEFAULT_SEED, lcg_vec_unit};

/// Build a deterministic query vector. Uses a separate seed offset so the
/// query is not byte-identical to a corpus member.
fn build_query(dim: usize) -> Vector {
    let mut state = DEFAULT_SEED.wrapping_add(0xC053_1EA1);
    Vector::new(lcg_vec_unit(&mut state, dim))
}

/// Build `count` deterministic candidate vectors of length `dim`.
fn build_candidates(dim: usize, count: usize) -> Vec<Vector> {
    let mut state = DEFAULT_SEED;
    (0..count)
        .map(|_| Vector::new(lcg_vec_unit(&mut state, dim)))
        .collect()
}

fn bench_weighted_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_cosine_similarity");

    let metric = AdvancedSimilarityMetric::WeightedCosine;

    for &dim in &[64usize, 128, 384, 768, 1024] {
        let weights: Vec<f32> = vec![1.0; dim];
        let query = build_query(dim);

        for &n_candidates in &[1usize, 100, 10_000] {
            let candidates = build_candidates(dim, n_candidates);

            // One-time sanity check: similarity must be in [0, 1] (the
            // metric clamps internally) and finite. Failure means inputs
            // are degenerate (zero vector).
            let probe = metric
                .similarity(&query, &candidates[0], Some(&weights))
                .expect("cosine probe must not error");
            assert!(
                probe.is_finite() && (0.0..=1.0).contains(&probe),
                "cosine probe must be finite and within [0, 1], got {probe}"
            );

            group.throughput(Throughput::Elements(n_candidates as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("dim_{dim}/{n_candidates}")),
                &(),
                |b, _| {
                    b.iter(|| {
                        for cand in &candidates {
                            let s = metric
                                .similarity(
                                    black_box(&query),
                                    black_box(cand),
                                    black_box(Some(&weights)),
                                )
                                .unwrap();
                            black_box(s);
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_weighted_cosine);
criterion_main!(benches);
