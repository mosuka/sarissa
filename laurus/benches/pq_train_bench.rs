//! PQ codebook training benchmark (Issue #622), plus the Issue #631 A/B
//! counterpart that reuses a pre-trained shared codebook instead of
//! retraining.
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
//! The "PQ Encode Shared Codebook" group mirrors "PQ Train" at the same
//! corpus sizes but configures `HnswIndexConfig::pq_codebook_path` with a
//! codebook trained once up front, so the measured `write()` calls skip
//! k-means entirely (Issue #631) — the delta against "PQ Train" is the
//! actual training cost eliminated by reuse.
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
use laurus::vector::index::pq_codebook::train_and_write_pq_codebook;

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

/// Repeated `write()` on a finalized HNSW+PQ index configured with a
/// shared codebook (Issue #631): the codebook is trained once, up front,
/// outside the timed loop, and every measured `write()` call encodes
/// against it via `VectorQuantizer::from_pq_codebook` instead of running
/// `pq_train_codebook`.
fn bench_pq_encode_shared_codebook(c: &mut Criterion) {
    let mut group = c.benchmark_group("PQ Encode Shared Codebook");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &count in &[2048usize, 8192] {
        let storage = select_storage();
        let vectors = generate_vectors(count, DIM);

        // Train on the same corpus, normalized the same way `write()`
        // will see it (Cosine below implies `normalize_vectors: true`,
        // Issue #794's scale trap): `add_vectors`/`build` normalize in
        // place before the writer's buffer is populated, so the shared
        // codebook must be trained on normalized vectors too.
        let training_sample: Vec<Vector> = vectors.iter().map(|(_, _, v)| v.clone()).collect();
        let codebook_name = "field.pqcb".to_string();
        train_and_write_pq_codebook(
            storage.as_ref(),
            &codebook_name,
            DIM,
            16,
            256,
            true,
            &training_sample,
        )
        .unwrap();

        let config = HnswIndexConfig {
            dimension: DIM,
            m: 16,
            ef_construction: 200,
            distance_metric: DistanceMetric::Cosine,
            quantization_method: QuantizationMethod::ProductQuantization {
                subvector_count: 16,
            },
            pq_codebook_path: Some(codebook_name),
            ..Default::default()
        };
        let mut index =
            ManagedVectorIndex::new(VectorIndexTypeConfig::HNSW(config), storage, "bench").unwrap();
        index.add_vectors(vectors).unwrap();
        index.finalize().unwrap();

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| index.write().unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pq_train, bench_pq_encode_shared_codebook);
criterion_main!(benches);
