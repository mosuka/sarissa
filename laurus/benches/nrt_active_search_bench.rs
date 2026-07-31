//! NRT active-segment (unflushed) vector search benchmark (Issue #640).
//!
//! Measures `SegmentedVectorField::search` when every vector still sits
//! in the active (unflushed) segment, so the whole query is the
//! brute-force scan in `search_active_segment`. This is the designated
//! A/B gate metric for #640 (prepared query + `parallel_scan` + partial
//! top-k selection in the active path).
//!
//! Corpus sizes straddle `PARALLEL_SCAN_THRESHOLD` (2048):
//! - `1000` — serial path (prepared-query + partial-select effect only)
//! - `10000` / `50000` — rayon path on top of the above
//!
//! Setup is pure in-memory (`MemoryStorage`, no disk cache, no flush),
//! deterministic via `common::DEFAULT_SEED`.

mod common;

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use common::{DEFAULT_SEED, SAMPLE_SIZE_FAST, lcg_vec_unit};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::index::field::{FieldSearchInput, VectorFieldReader, VectorFieldWriter};
use laurus::vector::index::hnsw;
use laurus::vector::index::segment::manager::{SegmentManager, SegmentManagerConfig};
use laurus::vector::index::segmented_field::SegmentedVectorField;
use laurus::vector::store::request::QueryVector;
use laurus::vector::{
    DistanceMetric, FieldOption, HnswOption, StoredVector, Vector, VectorFieldConfig,
};

/// Vector dimension for all cases (matches the other vector benches).
const DIM: usize = 128;

/// Build a Cosine `SegmentedVectorField` over in-memory storage and load
/// `count` deterministic vectors into its ACTIVE segment (no flush).
///
/// # Arguments
///
/// * `count` - Number of vectors buffered in the active segment.
///
/// # Returns
///
/// The field, ready for unflushed searches (the backing manager is kept
/// alive inside the field via `Arc`).
fn active_field_with(count: usize) -> SegmentedVectorField {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let manager = Arc::new(
        SegmentManager::new(
            SegmentManagerConfig::default(),
            storage.clone(),
            hnsw::segment::LAYOUT,
        )
        .expect("segment manager"),
    );
    let field_config = VectorFieldConfig {
        vector: Some(FieldOption::Hnsw(HnswOption {
            dimension: DIM,
            distance: DistanceMetric::Cosine,
            m: 16,
            ef_construction: 200,
            default_ef_search: None,
            base_weight: 1.0,
            quantizer: Default::default(),
            rerank_storage: None,
            embedder: None,
            pq_codebook_path: None,
        })),
        lexical: None,
    };
    let field = SegmentedVectorField::create("embedding", field_config, manager, storage, None)
        .expect("create segmented field");

    // `add_stored_vector` is async; drive it on a small runtime for setup.
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("tokio runtime");
    let mut state = DEFAULT_SEED;
    rt.block_on(async {
        for doc_id in 0..count as u64 {
            let v = StoredVector::new(lcg_vec_unit(&mut state, DIM));
            field
                .add_stored_vector(doc_id, &v, 0)
                .await
                .expect("buffer vector");
        }
    });
    field
}

/// Deterministic query distinct from every corpus vector.
fn query_input(limit: usize) -> FieldSearchInput {
    let mut state = DEFAULT_SEED.wrapping_add(1);
    FieldSearchInput {
        field: "embedding".to_string(),
        query_vectors: vec![QueryVector {
            vector: Vector::new(lcg_vec_unit(&mut state, DIM)),
            weight: 1.0,
            fields: None,
        }],
        limit,
        allowed_ids: None,
    }
}

/// Unflushed top-10 search across corpus sizes straddling the
/// parallel-scan threshold.
fn bench_nrt_active_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("NRT Active Search");
    group.sample_size(SAMPLE_SIZE_FAST);

    for &count in &[1_000usize, 10_000, 50_000] {
        let field = active_field_with(count);

        // Sanity: unflushed top-10 must return hits from the active scan.
        let probe = field.search(query_input(10)).expect("probe search");
        assert!(
            !probe.hits.is_empty(),
            "unflushed probe must hit the active segment at count={count}"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| field.search(query_input(10)).expect("search"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_nrt_active_search);
criterion_main!(benches);
