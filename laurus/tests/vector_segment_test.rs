use async_trait::async_trait;
use laurus::lexical::LexicalIndexConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::DistanceMetric;
use laurus::vector::Vector;
use laurus::vector::{FieldOption, HnswOption};
use laurus::vector::{VectorFieldConfig, VectorIndexConfig};
use laurus::{DataValue, Document};
use laurus::{EmbedInput, EmbedInputType, Embedder};
use laurus::{LaurusError, Result};
use std::any::Any;
use std::sync::Arc;

#[derive(Debug)]
struct MockTextEmbedder {
    dimension: usize,
}

#[async_trait]
impl Embedder for MockTextEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(_) => Ok(Vector::new(vec![0.0; self.dimension])),
            _ => Err(LaurusError::invalid_argument(
                "this embedder only supports text input",
            )),
        }
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    fn name(&self) -> &str {
        "mock-text"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_vector_segment_integration() {
    // 1. Setup storage and config
    let storage_config = MemoryStorageConfig::default();
    let storage = Arc::new(MemoryStorage::new(storage_config));

    let mut field_configs = std::collections::HashMap::new();
    field_configs.insert(
        "vector_field".to_string(),
        VectorFieldConfig {
            vector: Some(FieldOption::Hnsw(HnswOption {
                dimension: 4,
                distance: DistanceMetric::Euclidean,
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
        },
    );

    let collection_config = VectorIndexConfig {
        fields: field_configs.clone(),
        embedder: Arc::new(MockTextEmbedder { dimension: 4 }),
        default_fields: vec!["vector_field".to_string()],
        metadata: std::collections::HashMap::new(),
        deletion_config: laurus::DeletionConfig::default(),
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    };

    // We construct engine manually to inject storage
    let engine =
        laurus::vector::VectorStore::new(storage.clone(), collection_config.clone()).unwrap();

    // 2. Insert vectors
    let vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];

    for (i, vec_data) in vectors.into_iter().enumerate() {
        let doc = Document::builder()
            .add_field("vector_field", DataValue::Vector(vec_data))
            .build();
        engine
            .upsert_document_by_internal_id((i + 1) as u64, doc)
            .await
            .unwrap();
    }

    // 3. Flush/Persist explicitly
    engine.commit().await.unwrap();

    // 4. Persistence check
    // We drop engine and recreates it.
    drop(engine);

    let engine_2 =
        laurus::vector::VectorStore::new(storage.clone(), collection_config.clone()).unwrap();

    // We verify stats.
    // Recovery should load segments.
    // The new VectorStore uses index.stats() which returns vector_count.
    // After commit, the documents should be persisted.

    let stats = engine_2.stats().unwrap();

    // We use assert!(stats.document_count > 0) to be safe against flush optimizations.
    // But given implementation, it should be 3.
    println!("Stats document count: {}", stats.document_count);
    assert_eq!(stats.document_count, 3);
}

/// End-to-end regression guard for Issue #798 (follow-up of #790).
///
/// Commits a dense, well-separated 16-document corpus through
/// `VectorStore` with `HnswOption::quantizer =
/// QuantizationMethod::ProductQuantization`, then reads the produced
/// on-disk LVS1 segment header back and asserts it reports
/// `QuantHeader::ProductQuantization` — a deterministic, PQ-specific
/// observable, not merely that search succeeds.
///
/// This is the only **behavioral** assertion that PQ is honored through
/// a store/engine commit: it exercises the `from_hnsw_option` converter
/// path (#790). If a regression dropped `quantizer` from that converter
/// (while keeping `rerank_storage`), the field would fall back to the
/// default `Scalar8Bit`, the segment header would report
/// `quant_kind = 1`, and this test would fail. The existing `pq_*` tests
/// build `HnswIndexConfig` directly and so bypass `from_hnsw_option`,
/// leaving this path uncovered until now.
#[tokio::test(flavor = "multi_thread")]
async fn test_pq_quantizer_honored_through_engine_commit() {
    use laurus::storage::Storage;
    use laurus::vector::core::quantization::QuantizationMethod;
    use laurus::vector::index::format::{QuantHeader, VectorSegmentHeader};
    use std::io::{Read, Seek, SeekFrom};

    // dim % subvector_count must be 0 (PqParams::from_dim_and_m).
    const DIM: usize = 4;
    const SUBVECTOR_COUNT: usize = 2;

    // 1. Storage + config carrying ProductQuantization on the HNSW field.
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    let mut field_configs = std::collections::HashMap::new();
    field_configs.insert(
        "vector_field".to_string(),
        VectorFieldConfig {
            vector: Some(FieldOption::Hnsw(HnswOption {
                dimension: DIM,
                distance: DistanceMetric::Euclidean,
                m: 8,
                ef_construction: 50,
                default_ef_search: None,
                base_weight: 1.0,
                quantizer: QuantizationMethod::ProductQuantization {
                    subvector_count: SUBVECTOR_COUNT,
                },
                rerank_storage: None,
                embedder: None,
                pq_codebook_path: None,
            })),
            lexical: None,
        },
    );

    let collection_config = VectorIndexConfig {
        fields: field_configs,
        embedder: Arc::new(MockTextEmbedder { dimension: DIM }),
        default_fields: vec!["vector_field".to_string()],
        metadata: std::collections::HashMap::new(),
        deletion_config: laurus::DeletionConfig::default(),
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    };

    let engine = laurus::vector::VectorStore::new(storage.clone(), collection_config).unwrap();

    // 2. Commit two widely-separated clusters of 128 points each — 256 in
    //    total, matching the PQ min-train threshold (#880: segments with
    //    fewer vectors than the 256 k-means centroids are written as
    //    Scalar8Bit instead of training a degenerate codebook, so this test
    //    must supply a trainable corpus to see the PQ header). The large
    //    cluster separation keeps the quantizer stable across platforms
    //    (platform-dependent f32 reduction order could otherwise flip a
    //    quantization code, issue #730).
    const POINTS_PER_CLUSTER: usize = 128;
    let near_base = [10.0_f32, 10.0, 20.0, 20.0];
    let far_base = [-100.0_f32, -100.0, -200.0, -200.0];

    let mut internal_id = 1u64;
    for base in [near_base, far_base] {
        for i in 0..POINTS_PER_CLUSTER {
            // Deterministic small offsets in [-0.32, 0.30] per component.
            let off = [
                ((i % 8) as f32) * 0.04 - 0.14,
                ((i / 8 % 8) as f32) * 0.04 - 0.14,
                ((i / 64 % 8) as f32) * 0.04 - 0.14,
                ((i % 16) as f32) * 0.04 - 0.32,
            ];
            let v: Vec<f32> = base.iter().zip(&off).map(|(b, o)| b + o).collect();
            let doc = Document::builder()
                .add_field("vector_field", DataValue::Vector(v))
                .build();
            engine
                .upsert_document_by_internal_id(internal_id, doc)
                .await
                .unwrap();
            internal_id += 1;
        }
    }

    engine.commit().await.unwrap();

    // 3. Locate the committed on-disk HNSW segment. The `.hnsw.tmp` is
    //    renamed away on success and the `.hnsw.f32` rerank sidecar is not
    //    written (rerank_storage is None), so exactly one `.hnsw` remains.
    let segment_name = storage
        .list_files()
        .unwrap()
        .into_iter()
        .find(|name| name.ends_with(".hnsw"))
        .expect("a committed .hnsw segment must exist");

    let mut input = storage.open_input(&segment_name).unwrap();

    // The `.hnsw` file starts with a 20-byte HNSW preamble
    // (num_vectors:u64 + dimension:u32 + m:u32 + ef_construction:u32)
    // written before the LVS1 `VectorSegmentHeader`. Read it to advance to
    // the header and sanity-check the committed vector count.
    let mut num_vectors_buf = [0u8; 8];
    input.read_exact(&mut num_vectors_buf).unwrap();
    assert_eq!(
        u64::from_le_bytes(num_vectors_buf),
        2 * POINTS_PER_CLUSTER as u64,
        "all committed vectors should land in the segment"
    );
    // Skip dimension / m / ef_construction (3 x u32) to reach the LVS1 header.
    input.seek(SeekFrom::Current(12)).unwrap();

    // 4. The header must report ProductQuantization (quant_kind = 2). The
    //    default Scalar8Bit would report quant_kind = 1 and fail here, so this
    //    catches a regression that drops `quantizer` from `from_hnsw_option`.
    let header = VectorSegmentHeader::read_from(&mut input).unwrap();
    match header.quant {
        QuantHeader::ProductQuantization { params, .. } => {
            assert_eq!(
                params.m as usize, SUBVECTOR_COUNT,
                "PQ header must record the configured subvector_count"
            );
            assert_eq!(
                params.sub_dim as usize,
                DIM / SUBVECTOR_COUNT,
                "PQ header sub_dim must be dim / subvector_count"
            );
        }
        other => panic!("expected a ProductQuantization segment header, got {other:?}"),
    }
}
