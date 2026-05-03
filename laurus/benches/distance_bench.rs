//! Criterion benchmarks for the [`DistanceMetric`] family.
//!
//! Compares Cosine, Euclidean, Manhattan, and DotProduct distances on a
//! warm-set of vectors across multiple embedding dimensions.
//!
//! # Scope
//!
//! - Microbench of [`DistanceMetric::distance`] for the four built-in
//!   metrics.
//! - **Dimension sweep**: 64, 100, 128, 384, 768, 1024.
//!   - The SIMD body uses `wide::f32x8` (8-lane), so any dimension that
//!     is a multiple of 8 has no scalar remainder.
//!   - Multiples of 8: 64, 128, 384, 768, 1024 — the SIMD body covers
//!     every element. Note that 384 (= 48 × 8) is a multiple of 8
//!     despite being an off-power-of-two value; it is retained because
//!     it matches a common embedding size (e.g. `all-MiniLM-L6-v2`).
//!   - Non-multiple of 8: **100** (= 12 × 8 + 4) is the only dimension
//!     in this sweep that exercises the SIMD remainder loop — the path
//!     that #415 (vectorize SIMD remainder in distance kernels) targets.
//!     If #415's effect needs to be measured at additional non-multiple
//!     dimensions, extend `DIMS` accordingly.
//! - Single batch size: 100 target vectors per query.
//! - Reports `Throughput::Elements(100)` so the per-distance cost is
//!   directly comparable across dimensions and metrics.
//! - Bench id encodes the dimension and metric as `dim_{N}/{metric}` so
//!   `cargo bench -- "dim_100"` filters all metrics for the non-8-multiple
//!   case and `cargo bench -- "cosine"` filters cosine across every
//!   dimension.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench distance_bench
//! ```
//!
//! Filter:
//!
//! ```sh
//! cargo bench --bench distance_bench -- "dim_384"
//! cargo bench --bench distance_bench -- "cosine"
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

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use laurus::vector::DistanceMetric;
use std::hint::black_box;

/// Embedding dimensions to sweep. Mix of 8-byte multiples and non-multiples
/// so the SIMD remainder path #415 targets has measurable cases.
const DIMS: &[usize] = &[64, 100, 128, 384, 768, 1024];

/// Number of target vectors compared against a single query per timed
/// iteration. Reflected in `Throughput::Elements(NUM_TARGETS)`.
const NUM_TARGETS: usize = 100;

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
    let mut group = c.benchmark_group("distance_metrics");
    group.throughput(Throughput::Elements(NUM_TARGETS as u64));

    let metrics = [
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::DotProduct,
    ];

    for &dim in DIMS {
        let vectors = generate_test_vectors(NUM_TARGETS + 1, dim);
        let query = vectors[0].clone();
        let targets: Vec<Vec<f32>> = vectors.into_iter().skip(1).collect();
        debug_assert_eq!(targets.len(), NUM_TARGETS);

        // One-time sanity check: every metric must return a finite,
        // non-NaN distance for the first target. Failing this means the
        // bench premise is broken before we start measuring.
        for metric in metrics {
            let probe = metric.distance(&query, &targets[0]).unwrap();
            assert!(
                probe.is_finite(),
                "{} (dim={dim}): distance must be finite for the probe pair, got {probe}",
                metric.name()
            );
        }

        for metric in metrics {
            let id = format!("dim_{dim}/{}", metric.name());
            group.bench_with_input(
                BenchmarkId::from_parameter(id),
                &(query.clone(), targets.clone()),
                |b, (query, targets)| {
                    b.iter(|| {
                        for target in targets {
                            let _ = black_box(
                                metric
                                    .distance(black_box(query), black_box(target))
                                    .unwrap(),
                            );
                        }
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_distances);
criterion_main!(benches);
