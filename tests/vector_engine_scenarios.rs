use std::any::Any;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::sync::Arc;

use async_trait::async_trait;
use platypus::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
use platypus::embedding::noop::NoOpEmbedder;
use platypus::embedding::per_field::PerFieldEmbedder;
use platypus::error::{PlatypusError, Result};
use platypus::storage::Storage;
use platypus::storage::memory::MemoryStorage;
use platypus::vector::DistanceMetric;
use platypus::vector::core::document::{
    DocumentPayload, DocumentVector, FieldPayload, FieldVectors, PayloadSource, SegmentPayload,
    StoredVector, VectorType,
};
use platypus::vector::core::vector::Vector;
use platypus::vector::engine::{
    FieldSelector, MetadataFilter, QueryVector, VectorEngine, VectorFieldConfig, VectorFilter,
    VectorIndexConfig, VectorIndexKind, VectorScoreMode, VectorSearchRequest,
};
use tempfile::NamedTempFile;

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
    payload.metadata.insert("lang".into(), "en".into());
    payload.add_field("body_embedding", sample_payload("rust embeddings", "body"));
    engine.upsert_document_payload(42, payload)?;

    let mut query = VectorSearchRequest::default();
    query.limit = 1;
    query.fields = Some(vec![FieldSelector::Exact("body_embedding".into())]);
    query.query_vectors.extend(engine.embed_query_field_payload(
        "body_embedding",
        sample_payload("embeddings overview", "body"),
    )?);

    let results = engine.search(query)?;
    assert_eq!(results.hits.len(), 1);
    assert_eq!(results.hits[0].doc_id, 42);
    Ok(())
}

#[test]
fn vector_engine_payload_accepts_image_bytes_segments() -> Result<()> {
    let engine = build_multimodal_payload_engine()?;

    let mut payload = DocumentPayload::new();
    payload.add_field("image_embedding", image_bytes_payload(&[1, 2, 3, 4]));
    engine.upsert_document_payload(99, payload)?;

    let mut query = VectorSearchRequest::default();
    query.limit = 1;
    query.fields = Some(vec![FieldSelector::Exact("image_embedding".into())]);
    query.query_vectors.extend(
        engine.embed_query_field_payload("image_embedding", image_bytes_payload(&[4, 3, 2, 1]))?,
    );

    let results = engine.search(query)?;
    assert_eq!(results.hits.len(), 1);
    assert_eq!(results.hits[0].doc_id, 99);
    Ok(())
}

#[test]
fn vector_engine_payload_accepts_image_uri_segments() -> Result<()> {
    let engine = build_multimodal_payload_engine()?;

    let mut document_file = NamedTempFile::new()?;
    document_file.write_all(&[7, 6, 5, 4])?;
    let document_uri = document_file.path().to_string_lossy().to_string();

    let mut payload = DocumentPayload::new();
    payload.add_field("image_embedding", image_uri_payload(document_uri));
    engine.upsert_document_payload(314, payload)?;

    let mut query_file = NamedTempFile::new()?;
    query_file.write_all(&[4, 5, 6, 7])?;
    let query_uri = query_file.path().to_string_lossy().to_string();

    let mut query = VectorSearchRequest::default();
    query.limit = 1;
    query.fields = Some(vec![FieldSelector::Exact("image_embedding".into())]);
    query
        .query_vectors
        .extend(engine.embed_query_field_payload("image_embedding", image_uri_payload(query_uri))?);

    let results = engine.search(query)?;
    assert_eq!(results.hits.len(), 1);
    assert_eq!(results.hits[0].doc_id, 314);
    Ok(())
}

fn build_sample_engine() -> Result<VectorEngine> {
    let config = sample_engine_config();
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new_default());
    let engine = VectorEngine::new(storage, config)?;

    for (doc_id, document) in sample_documents() {
        engine.upsert_document(doc_id, document)?;
    }

    Ok(engine)
}

fn sample_engine_config() -> VectorIndexConfig {
    let mut fields = HashMap::new();
    fields.insert(
        "title_embedding".into(),
        VectorFieldConfig {
            dimension: 4,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "text-encoder-v1".into(),
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.4,
        },
    );
    fields.insert(
        "body_embedding".into(),
        VectorFieldConfig {
            dimension: 4,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "text-encoder-v1".into(),
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.0,
        },
    );

    VectorIndexConfig {
        fields,
        default_fields: vec!["title_embedding".into(), "body_embedding".into()],
        metadata: HashMap::new(),
        embedder: Arc::new(NoOpEmbedder::new()),
    }
}

fn stored_query_vector(data: [f32; 4]) -> StoredVector {
    StoredVector {
        data: Arc::from(data),
        embedder_id: "text-encoder-v1".into(),
        vector_type: VectorType::Text,
        weight: 1.0,
        attributes: HashMap::new(),
    }
}

fn sample_documents() -> Vec<(u64, DocumentVector)> {
    #[derive(serde::Deserialize)]
    struct LegacyDocumentVector {
        doc_id: u64,
        #[serde(default)]
        fields: HashMap<String, FieldVectors>,
        #[serde(default)]
        metadata: HashMap<String, String>,
    }

    let docs: Vec<LegacyDocumentVector> =
        serde_json::from_str(include_str!("../resources/vector_engine_sample.json"))
            .expect("vector_engine_sample.json parses");

    docs.into_iter()
        .map(|legacy| {
            (
                legacy.doc_id,
                DocumentVector {
                    fields: legacy.fields,
                    metadata: legacy.metadata,
                },
            )
        })
        .collect()
}

fn sample_payload(text: &str, section: &str) -> FieldPayload {
    let mut payload = FieldPayload::default();
    payload
        .metadata
        .insert("section".into(), section.to_string());
    payload.add_text_segment(text);
    payload
}

fn image_segment_payload(source: PayloadSource) -> FieldPayload {
    let mut payload = FieldPayload::default();
    payload.add_segment(SegmentPayload::new(source, VectorType::Image));
    payload
}

fn image_bytes_payload(bytes: &[u8]) -> FieldPayload {
    image_segment_payload(PayloadSource::Bytes {
        bytes: Arc::<[u8]>::from(bytes.to_vec()),
        mime: Some("image/png".into()),
    })
}

fn image_uri_payload(uri: String) -> FieldPayload {
    image_segment_payload(PayloadSource::Uri {
        uri,
        media_hint: Some("image/png".into()),
    })
}

fn build_payload_engine() -> Result<VectorEngine> {
    // Create the embedder
    let embedder: Arc<dyn Embedder> = Arc::new(IntegrationTestEmbedder::new(4));

    // Create PerFieldEmbedder with the embedder as default
    let per_field_embedder = PerFieldEmbedder::new(embedder);

    // Build config using the new embedder field
    // Note: `embedder` field is the lookup key that PerFieldEmbedder uses
    let config = VectorIndexConfig::builder()
        .field(
            "body_embedding",
            VectorFieldConfig {
                dimension: 4,
                distance: DistanceMetric::Cosine,
                index: VectorIndexKind::Flat,
                embedder_id: "payload-encoder".into(),
                vector_type: VectorType::Text,
                embedder: Some("body_embedding".into()),
                base_weight: 1.0,
            },
        )
        .default_field("body_embedding")
        .embedder(per_field_embedder)
        .build()?;

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new_default());
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
    // Note: `embedder` field is the lookup key that PerFieldEmbedder uses
    let config = VectorIndexConfig::builder()
        .field(
            "image_embedding",
            VectorFieldConfig {
                dimension: 3,
                distance: DistanceMetric::Cosine,
                index: VectorIndexKind::Flat,
                embedder_id: "multimodal-encoder".into(),
                vector_type: VectorType::Image,
                embedder: Some("image_embedding".into()),
                base_weight: 1.0,
            },
        )
        .default_field("image_embedding")
        .embedder(per_field_embedder)
        .build()?;

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new_default());
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
            _ => Err(PlatypusError::invalid_argument(
                "IntegrationTestEmbedder only supports text input",
            )),
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
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
            EmbedInput::ImagePath(path) => {
                let bytes = fs::read(path)?;
                Ok(self.vector_from_bytes(&bytes))
            }
            EmbedInput::ImageBytes(bytes, _) => Ok(self.vector_from_bytes(bytes)),
            EmbedInput::ImageUri(uri) => {
                // For test purposes, treat URI as file path
                let bytes = fs::read(uri)?;
                Ok(self.vector_from_bytes(&bytes))
            }
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
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
