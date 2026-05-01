//! Criterion benchmarks for the [`DistanceMetric`] family.
//!
//! Compares Cosine, Euclidean, Manhattan, and DotProduct distances on a small
//! warm-set of vectors at a fixed dimension.
//!
//! # Scope
//!
//! - Microbench of [`DistanceMetric::distance`] for the four built-in
//!   metrics.
//! - Single dimension, single batch size, single query × 100 targets.
//! - Does **not** cover the SIMD-remainder path — that requires a
//!   non-multiple-of-8 dimension sweep (tracked separately in #424).
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench distance_bench
//! ```
//!
//! Filter by metric (criterion id matches the metric name):
//!
//! ```sh
//! cargo bench --bench distance_bench -- cosine
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench distance_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use laurus::vector::DistanceMetric;
use std::hint::black_box;

fn generate_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(count);
    for i in 0..count {
        let mut data = Vec::with_capacity(dimension);
        for j in 0..dimension {
            let value = ((i as f32 * 0.1 + j as f32 * 0.01).sin() * 0.5 + 0.5) * 2.0 - 1.0;
            data.push(value);
        }
        vectors.push(data);
    }
    vectors
}

fn bench_distances(c: &mut Criterion) {
    let dimension = 128;
    let vectors = generate_test_vectors(101, dimension);
    let query = &vectors[0];
    let targets = &vectors[1..101];

    // One-time sanity check: every metric returns a finite, non-NaN value
    // for the first target. Failing this means the bench premise is broken
    // before we even start measuring.
    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::DotProduct,
    ] {
        let probe = metric.distance(query, &targets[0]).unwrap();
        assert!(
            probe.is_finite(),
            "{}: distance must be finite for the probe pair, got {probe}",
            metric.name()
        );
    }

    let mut group = c.benchmark_group("distance_metrics");

    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::DotProduct,
    ] {
        group.bench_function(metric.name(), |b| {
            b.iter(|| {
                for target in targets {
                    let _ = black_box(
                        metric
                            .distance(black_box(query), black_box(target))
                            .unwrap(),
                    );
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_distances);
criterion_main!(benches);
