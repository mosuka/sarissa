use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use platypus::embedding::text_embedder::TextEmbedder;
use platypus::error::Result;
use platypus::storage::Storage;
use platypus::storage::memory::MemoryStorage;
use platypus::vector::DistanceMetric;
use platypus::vector::core::document::{
    DocumentPayload, DocumentVectors, FieldPayload, RawTextSegment, StoredVector, VectorRole,
};
use platypus::vector::core::vector::Vector;
use platypus::vector::engine::{
    FieldSelector, MetadataFilter, QueryVector, VectorEmbedderConfig, VectorEmbedderProvider,
    VectorEngine, VectorEngineConfig, VectorEngineFilter, VectorEngineSearchRequest,
    VectorFieldConfig, VectorIndexKind, VectorScoreMode,
};

#[test]
fn vector_engine_multi_field_search_prefers_relevant_documents() -> Result<()> {
    let engine = build_sample_engine()?;

    let mut query = VectorEngineSearchRequest::default();
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

    let results = engine.search(&query)?;
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

    let mut query = VectorEngineSearchRequest::default();
    query.limit = 3;
    query.score_mode = VectorScoreMode::MaxSim;
    query.query_vectors.push(QueryVector {
        vector: stored_query_vector([0.2, 0.1, 0.9, 0.05]),
        weight: 1.0,
    });

    let mut doc_filter = MetadataFilter::default();
    doc_filter.equals.insert("lang".into(), "ja".into());
    query.filter = Some(VectorEngineFilter {
        document: doc_filter,
        field: MetadataFilter::default(),
    });

    let results = engine.search(&query)?;
    assert_eq!(results.hits.len(), 1);
    assert_eq!(results.hits[0].doc_id, 3);
    Ok(())
}

#[test]
fn vector_engine_field_metadata_filters_limit_hits() -> Result<()> {
    let engine = build_sample_engine()?;

    let mut query = VectorEngineSearchRequest::default();
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
    query.filter = Some(VectorEngineFilter {
        document: MetadataFilter::default(),
        field: field_filter,
    });

    let results = engine.search(&query)?;
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

    let mut payload = DocumentPayload::new(42);
    payload.metadata.insert("lang".into(), "en".into());
    payload.add_field("body_embedding", sample_payload("rust embeddings", "body"));
    let version = engine.upsert_document_payload(payload)?;
    assert!(version > 0);

    let mut query = VectorEngineSearchRequest::default();
    query.limit = 1;
    query.fields = Some(vec![FieldSelector::Exact("body_embedding".into())]);
    query.query_vectors.extend(engine.embed_query_field_payload(
        "body_embedding",
        sample_payload("embeddings overview", "body"),
    )?);

    let results = engine.search(&query)?;
    assert_eq!(results.hits.len(), 1);
    assert_eq!(results.hits[0].doc_id, 42);
    Ok(())
}

fn build_sample_engine() -> Result<VectorEngine> {
    let config = sample_engine_config();
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new_default());
    let engine = VectorEngine::new(config, storage, None)?;

    for document in sample_documents() {
        engine.upsert_document(document)?;
    }

    Ok(engine)
}

fn sample_engine_config() -> VectorEngineConfig {
    let mut fields = HashMap::new();
    fields.insert(
        "title_embedding".into(),
        VectorFieldConfig {
            dimension: 4,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "text-encoder-v1".into(),
            role: VectorRole::Text,
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
            role: VectorRole::Text,
            embedder: None,
            base_weight: 1.0,
        },
    );

    VectorEngineConfig {
        fields,
        embedders: HashMap::new(),
        default_fields: vec!["title_embedding".into(), "body_embedding".into()],
        metadata: HashMap::new(),
    }
}

fn stored_query_vector(data: [f32; 4]) -> StoredVector {
    StoredVector {
        data: Arc::from(data),
        embedder_id: "text-encoder-v1".into(),
        role: VectorRole::Text,
        weight: 1.0,
        attributes: HashMap::new(),
    }
}

fn sample_documents() -> Vec<DocumentVectors> {
    serde_json::from_str(include_str!("../resources/vector_engine_sample.json"))
        .expect("vector_engine_sample.json parses")
}

fn sample_payload(text: &str, section: &str) -> FieldPayload {
    let mut payload = FieldPayload::default();
    payload
        .metadata
        .insert("section".into(), section.to_string());
    payload.add_text_segment(RawTextSegment::new(text));
    payload
}

fn build_payload_engine() -> Result<VectorEngine> {
    let mut fields = HashMap::new();
    fields.insert(
        "body_embedding".into(),
        VectorFieldConfig {
            dimension: 4,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "payload-encoder".into(),
            role: VectorRole::Text,
            embedder: Some("integration_embedder".into()),
            base_weight: 1.0,
        },
    );

    let embedders = HashMap::from([(
        "integration_embedder".into(),
        VectorEmbedderConfig {
            provider: VectorEmbedderProvider::External,
            model: "integration-test".into(),
            options: HashMap::new(),
        },
    )]);

    let config = VectorEngineConfig {
        fields,
        embedders,
        default_fields: vec!["body_embedding".into()],
        metadata: HashMap::new(),
    };

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new_default());
    let engine = VectorEngine::new(config, storage, None)?;
    engine.register_embedder_instance(
        "integration_embedder",
        Arc::new(IntegrationTestEmbedder::new(4)),
    )?;
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
impl TextEmbedder for IntegrationTestEmbedder {
    async fn embed(&self, text: &str) -> Result<Vector> {
        let mut data = vec![0.0_f32; self.dimension];
        for (idx, byte) in text.bytes().enumerate() {
            let bucket = idx % self.dimension;
            data[bucket] += (byte as f32) / 255.0;
        }
        Ok(Vector::new(data))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "integration-test-embedder"
    }
}
