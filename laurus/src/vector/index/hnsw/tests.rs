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

    // Stage 2 rerank API is reserved but not yet implemented (Issue #481):
    // requesting rerank_factor must surface NotImplemented up-front so
    // callers can plan migration without surprises.
    let rerank_request = VectorIndexQuery::new(Vector::new(vec![0.9, 0.1, 0.0]))
        .top_k(1)
        .field_name("test".to_string())
        .rerank_factor(2);
    let err = searcher
        .search(&rerank_request)
        .expect_err("rerank_factor should return NotImplemented");
    assert!(
        matches!(err, crate::error::LaurusError::NotImplemented(_)),
        "expected NotImplemented, got {err:?}"
    );

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
