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
                base_weight: 1.0,
                quantizer: Default::default(),
                embedder: None,
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
