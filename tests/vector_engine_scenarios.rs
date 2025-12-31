use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use sarissa::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
use sarissa::embedding::per_field::PerFieldEmbedder;
use sarissa::embedding::precomputed::PrecomputedEmbedder;
use sarissa::error::{Result, SarissaError};
use sarissa::storage::Storage;
use sarissa::storage::memory::MemoryStorage;
use sarissa::vector::DistanceMetric;
use sarissa::vector::core::document::{
    DocumentPayload, DocumentVector, Payload, PayloadSource, StoredVector,
};
use sarissa::vector::core::vector::Vector;
use sarissa::vector::engine::{
    FieldSelector, MetadataFilter, QueryPayload, QueryVector, VectorEngine, VectorFieldConfig,
    VectorFilter, VectorIndexConfig, VectorIndexKind, VectorScoreMode, VectorSearchRequest,
};

#[test]
fn vector_engine_multi_field_search_prefers_relevant_documents() -> Result<()> {
    let engine = build_sample_engine()?;

    let mut query = VectorSearchRequest::default();
    query.limit = 2;
    query.score_mode = VectorScoreMode::WeightedSum;
    query.overfetch = 1.25;
    query.fields = Some(vec![
        FieldSelector::Exact("title_embedding".into()),
        FieldSelector::Exact("body_embedding".into()),
    ]);
    query.query_vectors.push(QueryVector {
        vector: stored_query_vector([0.9, 0.1, 0.0, 0.0]),
        weight: 1.0,
        fields: None,
    });

    let results = engine.search(query)?;
    assert_eq!(results.hits.len(), 2);
    assert_eq!(results.hits[0].doc_id, 1);
    assert!(results.hits[0].score >= results.hits[1].score);
    assert!(
        results.hits[0]
            .field_hits
            .iter()
            .any(|hit| hit.field == "title_embedding")
    );
    Ok(())
}

#[test]
fn vector_engine_respects_document_metadata_filters() -> Result<()> {
    let engine = build_sample_engine()?;

    let mut query = VectorSearchRequest::default();
    query.limit = 3;
    query.score_mode = VectorScoreMode::MaxSim;
    query.query_vectors.push(QueryVector {
        vector: stored_query_vector([0.2, 0.1, 0.9, 0.05]),
        weight: 1.0,
        fields: None,
    });

    let mut doc_filter = MetadataFilter::default();
    doc_filter.equals.insert("lang".into(), "ja".into());
    query.filter = Some(VectorFilter {
        document: doc_filter,
        field: MetadataFilter::default(),
    });

    let results = engine.search(query)?;
    assert_eq!(results.hits.len(), 1);
    assert_eq!(results.hits[0].doc_id, 3);
    Ok(())
}

#[test]
fn vector_engine_field_metadata_filters_limit_hits() -> Result<()> {
    let engine = build_sample_engine()?;

    let mut query = VectorSearchRequest::default();
    query.limit = 3;
    query.fields = Some(vec![
        FieldSelector::Exact("title_embedding".into()),
        FieldSelector::Exact("body_embedding".into()),
    ]);
    query.query_vectors.push(QueryVector {
        vector: stored_query_vector([0.8, 0.05, 0.05, 0.1]),
        weight: 1.0,
        fields: None,
    });

    let mut field_filter = MetadataFilter::default();
    field_filter.equals.insert("section".into(), "body".into());
    query.filter = Some(VectorFilter {
        document: MetadataFilter::default(),
        field: field_filter,
    });

    let results = engine.search(query)?;
    assert!(
        results
            .hits
            .iter()
            .flat_map(|hit| &hit.field_hits)
            .all(|hit| hit.field == "body_embedding")
    );
    assert!(results.hits.len() >= 1);
    Ok(())
}

#[test]
fn vector_engine_upserts_and_queries_raw_payloads() -> Result<()> {
    let engine = build_payload_engine()?;

    let mut payload = DocumentPayload::new();
    payload.set_metadata("lang", "en");
    payload.set_text("body_embedding", "rust embeddings");
    engine.upsert_payloads(42, payload)?;

    let mut query = VectorSearchRequest::default();
    query.limit = 1;
    query.fields = Some(vec![FieldSelector::Exact("body_embedding".into())]);
    query.query_payloads.push(QueryPayload::new(
        "body_embedding",
        Payload::text("embeddings overview"),
    ));

    let results = engine.search(query)?;
    assert_eq!(results.hits.len(), 1);
    assert_eq!(results.hits[0].doc_id, 42);
    Ok(())
}

#[test]
fn vector_engine_payload_accepts_image_bytes_segments() -> Result<()> {
    let engine = build_multimodal_payload_engine()?;

    let mut payload = DocumentPayload::new();
    payload.set_field("image_embedding", image_bytes_payload(&[1, 2, 3, 4]));
    engine.upsert_payloads(99, payload)?;

    let mut query = VectorSearchRequest::default();
    query.limit = 1;
    query.fields = Some(vec![FieldSelector::Exact("image_embedding".into())]);
    query.query_payloads.push(QueryPayload::new(
        "image_embedding",
        image_bytes_payload(&[4, 3, 2, 1]),
    ));

    let results = engine.search(query)?;
    assert_eq!(results.hits.len(), 1);
    assert_eq!(results.hits[0].doc_id, 99);
    Ok(())
}

fn build_sample_engine() -> Result<VectorEngine> {
    let config = sample_engine_config();
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    let engine = VectorEngine::new(storage, config)?;

    for (doc_id, document) in sample_documents() {
        engine.upsert_vectors(doc_id, document)?;
    }

    Ok(engine)
}

fn sample_engine_config() -> VectorIndexConfig {
    let mut fields: HashMap<String, VectorFieldConfig> = HashMap::new();
    fields.insert(
        "title_embedding".into(),
        VectorFieldConfig {
            dimension: 4,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,

            base_weight: 1.4,
        },
    );
    fields.insert(
        "body_embedding".into(),
        VectorFieldConfig {
            dimension: 4,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,

            base_weight: 1.0,
        },
    );

    let mut builder = VectorIndexConfig::builder()
        .default_fields(vec!["title_embedding".into(), "body_embedding".into()])
        .default_distance(DistanceMetric::Cosine)
        .default_index_kind(VectorIndexKind::Flat)
        .default_base_weight(1.0)
        .implicit_schema(false)
        .embedder(PrecomputedEmbedder::new());

    for (name, config) in fields {
        builder = builder.field(name, config);
    }

    builder.build().expect("build config")
}

fn stored_query_vector(data: [f32; 4]) -> StoredVector {
    StoredVector {
        data: Arc::from(data),
        weight: 1.0,
        attributes: HashMap::new(),
    }
}

fn sample_documents() -> Vec<(u64, DocumentVector)> {
    // Local struct to deserialize the legacy JSON format with multiple vectors per field.
    #[derive(serde::Deserialize)]
    struct LegacyFieldVectors {
        #[serde(default)]
        vectors: Vec<StoredVector>,
        #[serde(default)]
        metadata: HashMap<String, String>,
    }

    #[derive(serde::Deserialize)]
    struct LegacyDocumentVector {
        doc_id: u64,
        #[serde(default)]
        fields: HashMap<String, LegacyFieldVectors>,
        #[serde(default)]
        metadata: HashMap<String, String>,
    }

    let docs: Vec<LegacyDocumentVector> =
        serde_json::from_str(include_str!("../resources/vector_engine_sample.json"))
            .expect("vector_engine_sample.json parses");

    docs.into_iter()
        .map(|legacy| {
            // Convert LegacyFieldVectors to StoredVector (take first vector from each field)
            // Merge field-level metadata into StoredVector.attributes
            let converted_fields: HashMap<String, StoredVector> = legacy
                .fields
                .into_iter()
                .filter_map(|(field_name, field_vectors)| {
                    field_vectors.vectors.into_iter().next().map(|mut v| {
                        // Merge field-level metadata into vector attributes
                        for (k, val) in &field_vectors.metadata {
                            v.attributes.insert(k.clone(), val.clone());
                        }
                        (field_name, v)
                    })
                })
                .collect();

            (
                legacy.doc_id,
                DocumentVector {
                    fields: converted_fields,
                    metadata: legacy.metadata,
                },
            )
        })
        .collect()
}

fn image_bytes_payload(bytes: &[u8]) -> Payload {
    Payload::new(PayloadSource::bytes(
        bytes.to_vec(),
        Some("image/png".into()),
    ))
}

fn build_payload_engine() -> Result<VectorEngine> {
    // Create the embedder
    let embedder: Arc<dyn Embedder> = Arc::new(IntegrationTestEmbedder::new(4));

    // Create PerFieldEmbedder with the embedder as default
    let per_field_embedder = PerFieldEmbedder::new(embedder);

    // Build config using the new embedder field
    let config = VectorIndexConfig::builder()
        .field(
            "body_embedding",
            VectorFieldConfig {
                dimension: 4,
                distance: DistanceMetric::Cosine,
                index: VectorIndexKind::Flat,

                base_weight: 1.0,
            },
        )
        .default_field("body_embedding")
        .embedder(per_field_embedder)
        .build()?;

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    let engine = VectorEngine::new(storage, config)?;
    Ok(engine)
}

fn build_multimodal_payload_engine() -> Result<VectorEngine> {
    // Create the multimodal embedder (implements unified Embedder trait)
    let multimodal_embedder: Arc<dyn Embedder> = Arc::new(IntegrationMultimodalEmbedder::new(3));

    // Create PerFieldEmbedder with multimodal embedder as default
    let mut per_field_embedder = PerFieldEmbedder::new(multimodal_embedder.clone());
    per_field_embedder.add_embedder("image_embedding", multimodal_embedder);

    // Build config using the new embedder field
    let config = VectorIndexConfig::builder()
        .field(
            "image_embedding",
            VectorFieldConfig {
                dimension: 3,
                distance: DistanceMetric::Cosine,
                index: VectorIndexKind::Flat,

                base_weight: 1.0,
            },
        )
        .default_field("image_embedding")
        .embedder(per_field_embedder)
        .build()?;

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    let engine = VectorEngine::new(storage, config)?;
    Ok(engine)
}

#[derive(Debug)]
struct IntegrationTestEmbedder {
    dimension: usize,
}

impl IntegrationTestEmbedder {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl Embedder for IntegrationTestEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(text) => {
                let mut data = vec![0.0_f32; self.dimension];
                for (idx, byte) in text.bytes().enumerate() {
                    let bucket = idx % self.dimension;
                    data[bucket] += (byte as f32) / 255.0;
                }
                Ok(Vector::new(data))
            }
            _ => Err(SarissaError::invalid_argument(
                "IntegrationTestEmbedder only supports text input",
            )),
        }
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    fn name(&self) -> &str {
        "integration-test-embedder"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
struct IntegrationMultimodalEmbedder {
    dimension: usize,
}

impl IntegrationMultimodalEmbedder {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn vector_from_bytes(&self, bytes: &[u8]) -> Vector {
        if bytes.is_empty() {
            return Vector::new(vec![0.0; self.dimension]);
        }
        let sum = bytes.iter().map(|b| *b as f32).sum::<f32>();
        let avg = sum / (bytes.len() as f32);
        Vector::new(vec![avg; self.dimension])
    }
}

#[async_trait]
impl Embedder for IntegrationMultimodalEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(text) => Ok(self.vector_from_bytes(text.as_bytes())),
            EmbedInput::Bytes(bytes, _) => Ok(self.vector_from_bytes(bytes)),
        }
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text, EmbedInputType::Image]
    }

    fn name(&self) -> &str {
        "integration-multimodal"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
