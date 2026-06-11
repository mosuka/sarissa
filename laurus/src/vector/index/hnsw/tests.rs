use crate::error::Result;
use crate::storage::StorageConfig;
use crate::storage::StorageFactory;
use crate::storage::memory::MemoryStorageConfig;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::rerank::RerankStorageKind;
use crate::vector::core::vector::Vector;
use crate::vector::index::VectorIndex;
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::HnswIndex;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::index::rerank_sidecar::read_sidecar;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
use std::sync::Arc;

#[test]
fn test_hnsw_integration() -> Result<()> {
    let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    let storage = StorageFactory::create(storage_config)?;

    // HNSW Config
    let config = HnswIndexConfig {
        dimension: 3,
        m: 16,
        ef_construction: 100,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let index = HnswIndex::create(storage.clone(), "default_index", config.clone())?;
    let mut writer = index.writer()?;

    // Add vectors
    let vectors = vec![
        (1, "test".to_string(), Vector::new(vec![1.0, 0.0, 0.0])), // A
        (2, "test".to_string(), Vector::new(vec![0.0, 1.0, 0.0])), // B
        (3, "test".to_string(), Vector::new(vec![0.0, 0.0, 1.0])), // C
        (4, "test".to_string(), Vector::new(vec![0.707, 0.707, 0.0])), // Between A and B
    ];

    writer.build(vectors.clone())?;
    writer.finalize()?;
    // Note: commit is handled by VectorIndexWriter trait default which calls write("default_index")
    // Since we are using HnswIndexWriter directly via trait object or concrete?
    // index.writer() returns Box<dyn VectorIndexWriter>.
    writer.commit()?;

    // Read back
    let reader = index.reader()?;

    // Check graph loading
    use crate::vector::index::hnsw::reader::HnswIndexReader;
    let hnsw_reader = reader
        .as_any()
        .downcast_ref::<HnswIndexReader>()
        .expect("Should be HnswIndexReader");
    assert!(hnsw_reader.graph.is_some());

    // Search using Graph
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let searcher = HnswSearcher::new(reader.clone())?;

    // Query close to A (1,0,0)
    let query = Vector::new(vec![0.9, 0.1, 0.0]);
    let request = VectorIndexQuery::new(query)
        .top_k(1)
        .field_name("test".to_string());

    let results = searcher.search(&request)?;

    assert_eq!(results.results.len(), 1);
    assert_eq!(results.results[0].doc_id, 1);

    // Stage 2 (Issue #481): rerank_factor against a Stage 1 segment
    // (no sidecar) must silently degrade to Stage 1 ranking — there
    // is no f32 information to recover, so returning an error would
    // be a worse experience than just returning the int8 ranking.
    let rerank_request = VectorIndexQuery::new(Vector::new(vec![0.9, 0.1, 0.0]))
        .top_k(1)
        .field_name("test".to_string())
        .rerank_factor(2);
    let degraded = searcher.search(&rerank_request)?;
    assert_eq!(degraded.results.len(), 1);
    assert_eq!(degraded.results[0].doc_id, 1);

    Ok(())
}

#[test]
fn writer_omits_sidecar_when_rerank_storage_is_none() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = HnswIndexConfig {
        dimension: 3,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        rerank_storage: None,
        ..Default::default()
    };
    let mut writer = HnswIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "stage1_only",
        Arc::clone(&storage),
    )?;
    writer.add_vectors(vec![
        (1, "f".to_string(), Vector::new(vec![1.0, 0.0, 0.0])),
        (2, "f".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
    ])?;
    writer.finalize()?;
    writer.write()?;

    assert!(
        storage.file_exists("stage1_only.hnsw"),
        "main LVS1 segment must exist"
    );
    assert!(
        !storage.file_exists("stage1_only.hnsw.f32"),
        "no sidecar should be written when rerank_storage is None"
    );
    Ok(())
}

#[test]
fn writer_emits_sidecar_with_matching_header_when_rerank_storage_is_f32() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 3;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let mut writer = HnswIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "stage2_f32",
        Arc::clone(&storage),
    )?;

    let originals = vec![
        (1u64, "f".to_string(), Vector::new(vec![0.1, 0.2, 0.3])),
        (2u64, "f".to_string(), Vector::new(vec![-1.0, 0.5, 0.25])),
        (3u64, "f".to_string(), Vector::new(vec![0.7, -0.7, 0.0])),
    ];
    writer.add_vectors(originals.clone())?;
    writer.finalize()?;
    writer.write()?;

    assert!(
        storage.file_exists("stage2_f32.hnsw.f32"),
        "sidecar must be written when rerank_storage is Some(F32)"
    );
    let mut sidecar_in = storage.open_input("stage2_f32.hnsw.f32")?;
    let (header, payload) = read_sidecar(&mut sidecar_in)?;
    assert_eq!(header.dim as usize, dim);
    assert_eq!(header.vector_count as usize, originals.len());
    assert_eq!(header.storage_kind, RerankStorageKind::F32);
    assert_eq!(payload.len(), originals.len() * dim * 4);

    // Sidecar order matches the LVS1 sort-by-doc_id order, so the
    // first record must be doc_id 1 with values [0.1, 0.2, 0.3].
    let first_x = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let first_y = f32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
    let first_z = f32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
    assert!((first_x - 0.1).abs() < f32::EPSILON);
    assert!((first_y - 0.2).abs() < f32::EPSILON);
    assert!((first_z - 0.3).abs() < f32::EPSILON);
    Ok(())
}

#[test]
fn reader_loads_rerank_storage_when_sidecar_present() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 3;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let originals = vec![
        (1u64, "f".to_string(), Vector::new(vec![0.1, 0.2, 0.3])),
        (2u64, "f".to_string(), Vector::new(vec![-1.0, 0.5, 0.25])),
    ];
    let mut writer = HnswIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "stage2_reader",
        Arc::clone(&storage),
    )?;
    writer.add_vectors(originals.clone())?;
    writer.finalize()?;
    writer.write()?;

    let reader = HnswIndexReader::load(
        Arc::clone(&storage),
        "stage2_reader",
        DistanceMetric::Cosine,
    )?;
    let pool = reader
        .rerank_storage()
        .expect("rerank_storage must be Some when sidecar exists in Eager mode");
    assert_eq!(pool.dim, dim);
    assert_eq!(pool.vector_count, originals.len());
    assert_eq!(pool.kind, RerankStorageKind::F32);
    let v0 = pool
        .get_f32_slice(1, "f")
        .expect("doc 1 must be present in rerank pool");
    assert_eq!(v0, &[0.1, 0.2, 0.3]);
    let v1 = pool
        .get_f32_slice(2, "f")
        .expect("doc 2 must be present in rerank pool");
    assert_eq!(v1, &[-1.0, 0.5, 0.25]);
    Ok(())
}

#[test]
fn reader_rerank_storage_is_none_for_stage1_segment() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 3;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: None,
        ..Default::default()
    };
    let mut writer = HnswIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "stage1_reader",
        Arc::clone(&storage),
    )?;
    writer.add_vectors(vec![
        (1u64, "f".to_string(), Vector::new(vec![1.0, 0.0, 0.0])),
        (2u64, "f".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
    ])?;
    writer.finalize()?;
    writer.write()?;

    let reader = HnswIndexReader::load(
        Arc::clone(&storage),
        "stage1_reader",
        DistanceMetric::Cosine,
    )?;
    assert!(
        reader.rerank_storage().is_none(),
        "Stage 1 segment (no sidecar) must yield rerank_storage = None"
    );
    Ok(())
}

#[test]
fn searcher_returns_exact_f32_distance_when_rerank_storage_is_loaded() -> Result<()> {
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 4;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 32,
        distance_metric: DistanceMetric::Euclidean,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let originals = vec![
        (
            1u64,
            "f".to_string(),
            Vector::new(vec![0.123_456, 0.234_567, 0.345_678, 0.456_789]),
        ),
        (
            2u64,
            "f".to_string(),
            Vector::new(vec![0.987_654, 0.876_543, 0.765_432, 0.654_321]),
        ),
        (
            3u64,
            "f".to_string(),
            Vector::new(vec![-0.111_111, -0.222_222, -0.333_333, -0.444_444]),
        ),
    ];

    let index = HnswIndex::create(Arc::clone(&storage), "rerank_search", config)?;
    let mut writer = index.writer()?;
    writer.build(originals.clone())?;
    writer.finalize()?;
    writer.commit()?;

    let reader = index.reader()?;
    let mut searcher = HnswSearcher::new(reader)?;
    searcher.set_ef_search(50);

    // Query equal to doc 1's vector -> exact f32 distance must be 0.
    let request = VectorIndexQuery::new(Vector::new(vec![
        0.123_456, 0.234_567, 0.345_678, 0.456_789,
    ]))
    .top_k(1)
    .field_name("f".to_string())
    .rerank_factor(3);
    let results = searcher.search(&request)?;

    assert_eq!(results.results.len(), 1);
    assert_eq!(results.results[0].doc_id, 1, "doc 1 must be top match");
    // Stage 1 (int8) returns a small but non-zero approximation for
    // the self-distance because of quantization noise. Stage 2 with
    // rerank rescores against the original f32 vectors and must
    // recover the exact zero.
    assert_eq!(
        results.results[0].distance, 0.0,
        "rerank must restore the exact f32 self-distance, got {}",
        results.results[0].distance
    );
    Ok(())
}

#[test]
fn searcher_silently_falls_back_to_stage1_when_rerank_storage_absent() -> Result<()> {
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 3;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        rerank_storage: None,
        ..Default::default()
    };
    let index = HnswIndex::create(Arc::clone(&storage), "stage1_with_rerank_request", config)?;
    let mut writer = index.writer()?;
    writer.build(vec![
        (1u64, "f".to_string(), Vector::new(vec![1.0, 0.0, 0.0])),
        (2u64, "f".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
    ])?;
    writer.finalize()?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = HnswSearcher::new(reader)?;

    // Stage 1 segment + rerank_factor request must succeed (no
    // NotImplemented) and return the int8 ranking.
    let request = VectorIndexQuery::new(Vector::new(vec![1.0, 0.0, 0.0]))
        .top_k(1)
        .field_name("f".to_string())
        .rerank_factor(5);
    let results = searcher.search(&request)?;
    assert_eq!(results.results.len(), 1);
    assert_eq!(results.results[0].doc_id, 1);
    Ok(())
}

#[test]
fn writer_load_round_trips_byte_exact_via_sidecar() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 4;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let originals = vec![
        (
            10u64,
            "f".to_string(),
            Vector::new(vec![0.123_456, -0.987_654, 1.111_222, 0.000_001]),
        ),
        (
            20u64,
            "f".to_string(),
            Vector::new(vec![-1.5, 0.5, 0.25, -0.75]),
        ),
    ];

    {
        let mut writer = HnswIndexWriter::with_storage(
            config.clone(),
            VectorIndexWriterConfig::default(),
            "stage2_round_trip",
            Arc::clone(&storage),
        )?;
        writer.add_vectors(originals.clone())?;
        writer.finalize()?;
        writer.write()?;
    }

    let loaded = HnswIndexWriter::load(
        config,
        VectorIndexWriterConfig::default(),
        Arc::clone(&storage),
        "stage2_round_trip",
    )?;
    let loaded_vectors = loaded.vectors();
    assert_eq!(loaded_vectors.len(), originals.len());

    for (orig, got) in originals.iter().zip(loaded_vectors.iter()) {
        assert_eq!(orig.0, got.0, "doc_id");
        assert_eq!(orig.1, got.1, "field name");
        assert_eq!(
            orig.2.data, got.2.data,
            "f32 payload must round-trip byte-exact via the LRS1 sidecar"
        );
    }
    Ok(())
}

/// A corrupted sidecar must fail `HnswIndexWriter::load` (Issue #788).
///
/// This is the anti-laundering guarantee: if the writer-reload path
/// accepted a corrupted sidecar, the broken f32 values would enter the
/// writer's in-memory state and be re-emitted with a fresh, valid CRC
/// on the next commit — silently converting detectable corruption into
/// undetectable corruption.
#[test]
fn writer_load_rejects_corrupted_sidecar() -> Result<()> {
    use std::io::{Read, Write};

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = HnswIndexConfig {
        dimension: 4,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    {
        let mut writer = HnswIndexWriter::with_storage(
            config.clone(),
            VectorIndexWriterConfig::default(),
            "stage2_corrupt",
            Arc::clone(&storage),
        )?;
        writer.add_vectors(vec![
            (1u64, "f".to_string(), Vector::new(vec![1.0, 0.0, 0.0, 0.0])),
            (2u64, "f".to_string(), Vector::new(vec![0.0, 1.0, 0.0, 0.0])),
        ])?;
        writer.finalize()?;
        writer.write()?;
    }

    // Flip one byte in the middle of the sidecar payload.
    let sidecar_name = "stage2_corrupt.hnsw.f32";
    let mut bytes = Vec::new();
    storage
        .open_input(sidecar_name)?
        .read_to_end(&mut bytes)
        .map_err(crate::error::LaurusError::from)?;
    let payload_mid = crate::vector::index::rerank_sidecar::HEADER_SIZE
        + (bytes.len()
            - crate::vector::index::rerank_sidecar::HEADER_SIZE
            - crate::vector::index::rerank_sidecar::FOOTER_SIZE)
            / 2;
    bytes[payload_mid] ^= 0xff;
    let mut out = storage.create_output(sidecar_name)?;
    out.write_all(&bytes)
        .map_err(crate::error::LaurusError::from)?;
    out.close()?;

    let result = HnswIndexWriter::load(
        config,
        VectorIndexWriterConfig::default(),
        Arc::clone(&storage),
        "stage2_corrupt",
    );
    // Pin the failure to the CRC check: the load path also has
    // dim/count-mismatch (InvalidOperation) and Io exits right after
    // read_sidecar, and those must not satisfy this test.
    match result {
        Err(crate::error::LaurusError::Index(msg)) => {
            assert!(
                msg.contains("checksum mismatch"),
                "expected a checksum mismatch, got: {msg}"
            );
        }
        Err(other) => panic!(
            "a corrupted .hnsw.f32 must fail the writer reload path \
             with a checksum error, got {other:?}"
        ),
        Ok(_) => panic!(
            "a corrupted .hnsw.f32 must be rejected by the writer \
             reload path, got Ok"
        ),
    }
    Ok(())
}

#[test]
fn test_hnsw_pq_search_returns_corpus_neighbour() -> Result<()> {
    use crate::vector::core::quantization::QuantizationMethod;
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    let storage = StorageFactory::create(storage_config)?;

    // Two well-separated clusters in 4-D Euclidean space. M=2 → sub_dim=2.
    let config = HnswIndexConfig {
        dimension: 4,
        m: 8,
        ef_construction: 50,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantization { subvector_count: 2 },
        ..Default::default()
    };

    let index = HnswIndex::create(storage.clone(), "pq_round_trip", config.clone())?;
    let mut writer = index.writer()?;

    // Two widely-separated clusters with multiple points each. PQ trains a
    // per-sub-vector k-means codebook; a tiny corpus (e.g. 5 points) makes
    // the codebook degenerate, and platform-dependent f32 reduction order in
    // k-means can then flip a near/far quantisation code, occasionally
    // pulling a far-cluster doc into the top-k (issue #730 — flaked on
    // x86_64). Using a denser corpus and a much larger cluster separation
    // keeps the quantiser stable across platforms: the far cluster is so
    // distant that no quantisation error can place it among the near
    // neighbours.
    //
    // Near cluster (doc_ids 1..=8) sits around (10, 10, 20, 20); far cluster
    // (doc_ids 9..=16) sits around (-100, -100, -200, -200).
    let near_offsets = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.1],
        [-0.1, -0.1, -0.1, -0.1],
        [0.2, -0.2, 0.2, -0.2],
        [-0.2, 0.2, -0.2, 0.2],
        [0.05, 0.05, -0.05, -0.05],
        [-0.05, -0.05, 0.05, 0.05],
        [0.15, -0.1, 0.1, -0.15],
    ];
    let near_base = [10.0_f32, 10.0, 20.0, 20.0];
    let far_base = [-100.0_f32, -100.0, -200.0, -200.0];

    let mut vectors = Vec::with_capacity(16);
    for (i, off) in near_offsets.iter().enumerate() {
        let v: Vec<f32> = near_base.iter().zip(off).map(|(b, o)| b + o).collect();
        vectors.push(((i + 1) as u64, "embedding".to_string(), Vector::new(v)));
    }
    for (i, off) in near_offsets.iter().enumerate() {
        let v: Vec<f32> = far_base.iter().zip(off).map(|(b, o)| b + o).collect();
        vectors.push(((i + 9) as u64, "embedding".to_string(), Vector::new(v)));
    }

    writer.build(vectors.clone())?;
    writer.finalize()?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = HnswSearcher::new(reader)?;

    // Query at the near cluster centre — every top-3 result must come from
    // the near cluster (doc_ids 1..=8), never the far cluster (9..=16).
    // The exact ordering within the near cluster is not asserted because PQ
    // is approximate; only cluster membership is guaranteed by the large
    // separation.
    let query = Vector::new(vec![10.0, 10.0, 20.0, 20.0]);
    let request = VectorIndexQuery::new(query)
        .top_k(3)
        .field_name("embedding".to_string());
    let results = searcher.search(&request)?;
    assert_eq!(results.results.len(), 3, "expected top-3 results");
    let ids: std::collections::HashSet<u64> = results.results.iter().map(|r| r.doc_id).collect();
    for id in &ids {
        assert!(
            (1..=8).contains(id),
            "top-3 must all be near-cluster doc_ids (1..=8); got {ids:?}",
        );
    }
    Ok(())
}

/// Regression / parity test for Issue #644: HNSW `ef_search` must honour
/// the per-query override and the schema-level default, instead of being
/// permanently capped at the historical `50` constant.
#[test]
fn hnsw_searcher_honours_per_query_and_schema_ef_search() -> Result<()> {
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = HnswIndexConfig {
        dimension: 3,
        m: 16,
        ef_construction: 64,
        // Schema-level default lifts the searcher's fallback well above
        // the legacy hardcoded `50`.
        default_ef_search: Some(300),
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let index = HnswIndex::create(storage.clone(), "ef_test", config)?;
    let mut writer = index.writer()?;
    let vectors = (0..32u64)
        .map(|i| {
            let mut v = vec![0.0_f32; 3];
            v[(i as usize) % 3] = 1.0 + (i as f32) * 0.01;
            (i, "vec".to_string(), Vector::new(v))
        })
        .collect::<Vec<_>>();
    writer.build(vectors)?;
    writer.finalize()?;
    writer.commit()?;

    // The searcher built via `HnswIndex::searcher()` must pick up the
    // schema-level `default_ef_search` (Issue #644).
    let searcher = index.searcher()?;
    let request = VectorIndexQuery::new(Vector::new(vec![1.0, 0.0, 0.0]))
        .top_k(5)
        .field_name("vec".to_string());
    let results = searcher.search(&request)?;
    assert!(
        !results.results.is_empty(),
        "expected non-empty results from schema-default search path"
    );

    // Construct a direct HnswSearcher and exercise the per-query override.
    let reader = index.reader()?;
    let direct = HnswSearcher::new(reader.clone())?;
    let override_request = VectorIndexQuery::new(Vector::new(vec![1.0, 0.0, 0.0]))
        .top_k(5)
        .field_name("vec".to_string())
        .ef_search(400);
    let with_override = direct.search(&override_request)?;
    assert!(
        !with_override.results.is_empty(),
        "expected non-empty results from per-query override search path"
    );

    Ok(())
}
