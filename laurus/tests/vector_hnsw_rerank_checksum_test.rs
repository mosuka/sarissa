//! Integration tests for CRC-32 integrity checking of the rerank
//! sidecar `.hnsw.f32` (issue #788).
//!
//! New sidecars carry a trailing `[magic u32][crc-32 u32]` footer over
//! header + payload, verified by `read_sidecar` during load. A flipped
//! byte in a committed sidecar must be rejected when the rerank path
//! loads it, while legacy footer-less sidecars (written before #788)
//! must still load and serve rerank-augmented searches.
//!
//! The tests drive the public `HnswIndex` API directly because that is
//! where `rerank_storage` takes effect today: the `VectorStore`-level
//! config conversion does not yet propagate `HnswOption::rerank_storage`
//! into `HnswIndexConfig` (issue #790), so a store-level commit never
//! emits a sidecar.

use std::io::{Read, Write};
use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::HnswIndexConfig;
use laurus::vector::index::hnsw::HnswIndex;
use laurus::vector::index::hnsw::reader::HnswIndexReader;
use laurus::vector::index::hnsw::searcher::HnswSearcher;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};
use laurus::{LaurusError, Result};

const DIM: usize = 16;
const N: u64 = 50;
const STEP: f32 = 0.01;
const INDEX_NAME: &str = "vector_index";
const SIDECAR_FILE: &str = "vector_index.hnsw.f32";
const FOOTER_SIZE: usize = 8;

fn doc_vec(i: u64) -> Vec<f32> {
    let theta = i as f32 * STEP;
    let mut v = vec![0.0; DIM];
    v[0] = theta.cos();
    v[1] = theta.sin();
    v
}

fn query_vec() -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[0] = 1.0;
    v
}

fn make_config() -> HnswIndexConfig {
    HnswIndexConfig {
        dimension: DIM,
        m: 16,
        ef_construction: 100,
        distance_metric: DistanceMetric::Cosine,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    }
}

/// Build a Stage-2 (rerank-enabled) HNSW index and commit it, emitting
/// `vector_index.hnsw` plus the `vector_index.hnsw.f32` sidecar.
fn build_committed(storage: Arc<dyn Storage>) {
    let index = HnswIndex::create(storage, INDEX_NAME, make_config()).unwrap();
    let mut writer = index.writer().unwrap();
    let vectors: Vec<(u64, String, Vector)> = (0..N)
        .map(|id| (id, "vec".to_string(), Vector::new(doc_vec(id))))
        .collect();
    writer.build(vectors).unwrap();
    writer.finalize().unwrap();
    writer.commit().unwrap();
}

fn read_all(storage: &Arc<dyn Storage>, name: &str) -> Vec<u8> {
    let mut input = storage.open_input(name).unwrap();
    let mut buf = Vec::new();
    input.read_to_end(&mut buf).unwrap();
    buf
}

fn write_all(storage: &Arc<dyn Storage>, name: &str, bytes: &[u8]) {
    let mut out = storage.create_output(name).unwrap();
    out.write_all(bytes).unwrap();
    out.close().unwrap();
}

/// Load the committed segment fresh (Eager mode reads the sidecar) and
/// run a rerank-augmented search over it.
///
/// Asserts that the rerank pool actually loaded: the searcher silently
/// falls back to Stage-1 ranking when the pool is absent, so without
/// this check a reader change that silently skipped the sidecar would
/// keep the legacy test green for the wrong reason.
fn search_with_rerank(storage: &Arc<dyn Storage>) -> Result<usize> {
    let reader = HnswIndexReader::load(Arc::clone(storage), INDEX_NAME, DistanceMetric::Cosine)?;
    assert!(
        reader.rerank_storage().is_some(),
        "the sidecar must be loaded into the rerank pool (not the silent Stage-1 fallback)"
    );
    let searcher = HnswSearcher::new(Arc::new(reader))?;
    let request = VectorIndexQuery::new(Vector::new(query_vec()))
        .top_k(10)
        .field_name("vec".to_string())
        .rerank_factor(3);
    let results = searcher.search(&request)?;
    Ok(results.results.len())
}

/// Assert that `result` failed specifically on the sidecar CRC check.
///
/// The load path has non-CRC error exits right after `read_sidecar`
/// (dim/count mismatch, plain I/O), and those must not satisfy the
/// corruption tests.
fn assert_checksum_error(result: Result<usize>) {
    match result {
        Err(LaurusError::Index(msg)) => {
            assert!(
                msg.contains("checksum mismatch"),
                "expected a checksum mismatch, got: {msg}"
            );
        }
        Err(other) => panic!("expected a checksum error, got {other:?}"),
        Ok(hits) => panic!("expected a checksum error, got Ok with {hits} hits"),
    }
}

#[test]
fn corrupted_rerank_sidecar_is_rejected() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    build_committed(storage.clone());
    assert!(
        storage.file_exists(SIDECAR_FILE),
        "the Stage-2 commit must emit {SIDECAR_FILE}"
    );

    // Flip a byte in the middle of the sidecar payload (24-byte header
    // + 50*16*4-byte payload + 8-byte footer, so len/2 lands in the
    // payload); the stored CRC stays the original, so verification
    // must fail when the rerank path loads the sidecar.
    let mut bytes = read_all(&storage, SIDECAR_FILE);
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    write_all(&storage, SIDECAR_FILE, &bytes);

    assert_checksum_error(search_with_rerank(&storage));
}

#[test]
fn corrupted_sidecar_footer_is_rejected() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    build_committed(storage.clone());

    // Flip a byte inside the trailing 8-byte footer itself (the stored
    // CRC half); this must be treated as corruption, not skipped.
    let mut bytes = read_all(&storage, SIDECAR_FILE);
    let pos = bytes.len() - FOOTER_SIZE / 2;
    bytes[pos] ^= 0xff;
    write_all(&storage, SIDECAR_FILE, &bytes);

    assert_checksum_error(search_with_rerank(&storage));
}

#[test]
fn legacy_rerank_sidecar_without_footer_still_loads() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    build_committed(storage.clone());

    // Drop the 8-byte CRC footer to mimic a pre-#788 sidecar with no
    // checksum. The remaining bytes are exactly the legacy layout, so
    // the rerank path must still load and serve results.
    let bytes = read_all(&storage, SIDECAR_FILE);
    let legacy = &bytes[..bytes.len() - FOOTER_SIZE];
    write_all(&storage, SIDECAR_FILE, legacy);

    let hits = search_with_rerank(&storage).unwrap();
    assert_eq!(
        hits, 10,
        "a footer-less (legacy) sidecar must still load and search"
    );
}
