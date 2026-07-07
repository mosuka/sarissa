//! PQ codebook training benchmark (Issue #622).
//!
//! PQ is HNSW-only, and training runs inside the writer's `write()`
//! (not `finalize()`): `quantize_segment_pq` → `pq_train_codebook`
//! trains M per-sub-vector k-means codebooks (M = 16, K = 256,
//! `PQ_KMEANS_ITERATIONS` = 25). The bench builds and finalizes one
//! HNSW+PQ index per case in setup, then measures repeated `write()`
//! calls — each call retrains the codebook and re-serializes to
//! in-memory storage, so training dominates the measured op. This is
//! the designated A/B gate metric for #622's PQ-side changes
//! (parallel sub-vector training + SIMD L2 kernel).
//!
//! Setup is in-memory and deterministic via `common::DEFAULT_SEED`.

mod common;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use common::{DEFAULT_SEED, SAMPLE_SIZE_SLOW, lcg_vec_unit, select_storage};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::vector::index::ManagedVectorIndex;
use laurus::vector::index::config::{HnswIndexConfig, VectorIndexTypeConfig};

/// Vector dimension (matches the other vector benches; M = 16 gives
/// `sub_dim` = 8).
const DIM: usize = 128;

/// Deterministic corpus for one bench case.
fn generate_vectors(count: usize, dim: usize) -> Vec<(u64, String, Vector)> {
    let mut state = DEFAULT_SEED;
    (0..count)
        .map(|i| {
            (
                i as u64,
                "field".to_string(),
                Vector::new(lcg_vec_unit(&mut state, dim)),
            )
        })
        .collect()
}

/// Repeated `write()` on a finalized HNSW+PQ index: PQ codebook
/// training (M seeded k-means runs) plus the comparatively cheap
/// encode + serialization passes.
fn bench_pq_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("PQ Train");
    group.sample_size(SAMPLE_SIZE_SLOW); // slow training path

    for &count in &[2048usize, 8192] {
        // One-time setup: build + finalize the graph so the measured
        // op is write() = PQ training + serialization only.
        let config = HnswIndexConfig {
            dimension: DIM,
            m: 16,
            ef_construction: 200,
            distance_metric: DistanceMetric::Cosine,
            quantization_method: QuantizationMethod::ProductQuantization {
                subvector_count: 16,
            },
            ..Default::default()
        };
        let mut index = ManagedVectorIndex::new(
            VectorIndexTypeConfig::HNSW(config),
            select_storage(),
            "bench",
        )
        .unwrap();
        index.add_vectors(generate_vectors(count, DIM)).unwrap();
        index.finalize().unwrap();

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| index.write().unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pq_train);
criterion_main!(benches);
