//! End-to-end writer → reader round-trip test for the PQ FastScan
//! HNSW integration introduced by Issue #701 (Part D Phase 2 of #695).
//!
//! Persists a small PQ-FastScan-encoded HNSW segment through the
//! `HnswIndexWriter`, reloads it via `HnswIndexReader`, and asserts
//! that the reader rebuilds an equivalent `PqFastScanPool` (same
//! `n_vectors` / `field_index` keys, same per-vector decoded codes,
//! identical `codebook` and `packed` buffers).
//!
//! The search-time integration (`QuantizedSearchCtx::PqFastScan`)
//! lands in [#702](https://github.com/mosuka/laurus/issues/702), so
//! this test exercises the persistence + load path only.

#![cfg(feature = "pq-fastscan")]

use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::vector::core::vector::Vector;
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::HnswIndexConfig;
use laurus::vector::index::hnsw::HnswIndex;
use laurus::vector::index::storage::VectorStorage;

#[test]
fn writer_reader_round_trip_preserves_pq_fastscan_codes() {
    // 32 vectors of dimension 8 = m=4 sub-quantisers × sub_dim=2.
    let dim = 8usize;
    let m = 4usize;
    let n = 32usize;

    let vectors: Vec<(u64, String, Vector)> = (0..n)
        .map(|i| {
            let values: Vec<f32> = (0..dim)
                .map(|d| ((i * 7 + d * 3) % 31) as f32 - 15.0)
                .collect();
            (i as u64, "v".to_string(), Vector::new(values))
        })
        .collect();

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))
        .expect("memory storage");
    let config = HnswIndexConfig {
        dimension: dim,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantizationFastScan { subvector_count: m },
        ..Default::default()
    };
    let index = HnswIndex::create(storage.clone(), "fastscan_round_trip", config)
        .expect("create HNSW index");

    {
        let mut writer = index.writer().expect("writer");
        writer.add_vectors(vectors.clone()).expect("add vectors");
        writer.finalize().expect("finalize");
        writer.write().expect("write");
    }

    // Reload through a fresh reader and verify the in-memory pool
    // matches the written codes.
    let reader = index.reader().expect("reader");
    let stats = reader.stats();
    assert_eq!(stats.vector_count, n);
    assert_eq!(stats.dimension, dim);

    // Cast to the concrete HNSW reader to inspect the VectorStorage
    // variant.
    let hnsw_reader = reader
        .as_any()
        .downcast_ref::<laurus::vector::index::hnsw::reader::HnswIndexReader>()
        .expect("HnswIndexReader");
    let vectors_storage = hnsw_reader.vectors();
    let pool = match vectors_storage {
        VectorStorage::OwnedPqFastScan(pool) => pool.clone(),
        other => panic!(
            "expected OwnedPqFastScan, got {:?}",
            std::mem::discriminant(other)
        ),
    };

    assert_eq!(pool.n_vectors, n, "n_vectors mismatch");
    assert_eq!(pool.params.m as usize, m);
    assert_eq!(pool.params.k, 16);
    assert_eq!(pool.codebook.len(), pool.params.codebook_len());

    let field_map = pool.field_index.get("v").expect("v field present in pool");
    for (doc_id, _, _) in &vectors {
        assert!(
            field_map.contains_key(doc_id),
            "doc_id {doc_id} missing from field index"
        );
    }

    // Verify the reader's `keys()` view agrees with the writer's input.
    let mut keys: Vec<(u64, String)> = vectors_storage.keys();
    keys.sort();
    let mut expected: Vec<(u64, String)> =
        vectors.iter().map(|(id, f, _)| (*id, f.clone())).collect();
    expected.sort();
    assert_eq!(keys, expected);
}

#[test]
fn writer_reader_round_trip_handles_partial_block() {
    // n = 21 < BLOCK_SIZE (32) exercises the trailing-block zero-padding
    // path that PqFastScanPool::build emits when fewer than 32 vectors
    // are written. The count sits above the FastScan min-train guard
    // (#880: segments with fewer than the K=16 centroids are written as
    // Scalar8Bit) while still leaving padded slots in the block.
    let dim = 4usize;
    let m = 2usize;
    let n = 21usize;

    let vectors: Vec<(u64, String, Vector)> = (0..n)
        .map(|i| {
            let values: Vec<f32> = (0..dim)
                .map(|d| ((i * 11 + d * 5) % 23) as f32 - 11.0)
                .collect();
            (i as u64, "v".to_string(), Vector::new(values))
        })
        .collect();

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))
        .expect("memory storage");
    let config = HnswIndexConfig {
        dimension: dim,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantizationFastScan { subvector_count: m },
        ..Default::default()
    };
    let index = HnswIndex::create(storage.clone(), "fastscan_partial_block", config)
        .expect("create HNSW index");

    {
        let mut writer = index.writer().expect("writer");
        writer.add_vectors(vectors.clone()).expect("add vectors");
        writer.finalize().expect("finalize");
        writer.write().expect("write");
    }

    let reader = index.reader().expect("reader");
    let hnsw_reader = reader
        .as_any()
        .downcast_ref::<laurus::vector::index::hnsw::reader::HnswIndexReader>()
        .expect("HnswIndexReader");
    let pool = match hnsw_reader.vectors() {
        VectorStorage::OwnedPqFastScan(pool) => pool.clone(),
        other => panic!(
            "expected OwnedPqFastScan, got {:?}",
            std::mem::discriminant(other)
        ),
    };
    assert_eq!(pool.n_vectors, n);
    assert_eq!(pool.block_count(), 1, "n=21 spans a single block");
    // Trailing padding positions (5..32) have all-zero codes — the
    // codebook lookups for those slots return centroid 0 of every
    // sub-quantiser, which is fine because the searcher masks them
    // by `n_vectors` at query time.
    for vec_idx in n..32 {
        let codes = pool.codes_at(vec_idx);
        assert!(
            codes.iter().all(|&c| c == 0),
            "padding slot {vec_idx} should be all-zero codes (got {codes:?})"
        );
    }
}
