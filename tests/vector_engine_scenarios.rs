use std::collections::HashMap;
use std::sync::Arc;

use platypus::error::Result;
use platypus::storage::Storage;
use platypus::storage::memory::MemoryStorage;
use platypus::vector::DistanceMetric;
use platypus::vector::core::document::{DocumentVectors, StoredVector, VectorRole};
use platypus::vector::engine::{
    FieldSelector, MetadataFilter, QueryVector, VectorEngine, VectorEngineConfig,
    VectorEngineFilter, VectorEngineSearchRequest, VectorFieldConfig, VectorIndexKind,
    VectorScoreMode,
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
            base_weight: 1.0,
        },
    );

    VectorEngineConfig {
        fields,
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
