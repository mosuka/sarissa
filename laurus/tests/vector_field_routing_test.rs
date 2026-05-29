//! Integration tests for per-field vector search routing (issue #676).
//!
//! The store holds two HNSW vector fields, `img_vec` and `txt_vec`. Each
//! document carries a vector for exactly one field, so a query routed to a
//! field can only return that field's documents. The tests exercise both
//! routing inputs — `QueryVector.fields` (per-query) and
//! `VectorSearchParams.fields` (request-level `Exact` / `Prefix`) — and the
//! "no fields → search all" default.

use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::store::config::VectorFieldConfig;
use laurus::vector::store::request::{
    FieldSelector, QueryVector, VectorScoreMode, VectorSearchParams, VectorSearchRequest,
};
use laurus::vector::{FieldOption, VectorIndexConfig};
use laurus::{DataValue, Document};
use laurus::{EmbedInput, EmbedInputType, Embedder};
use laurus::{LaurusError, Result};

#[derive(Debug)]
struct MockEmbedder {
    dimension: usize,
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(_) => Ok(Vector::new(vec![0.0; self.dimension])),
            _ => Err(LaurusError::invalid_argument("text only")),
        }
    }
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }
    fn name(&self) -> &str {
        "mock"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn hnsw(dimension: usize) -> FieldOption {
    FieldOption::Hnsw(HnswOption {
        dimension,
        distance: DistanceMetric::Cosine,
        m: 16,
        ef_construction: 100,
        default_ef_search: None,
        base_weight: 1.0,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
    })
}

/// Store with two HNSW fields `img_vec` and `txt_vec`, each holding two docs.
/// doc1/doc2 → `img_vec`; doc3/doc4 → `txt_vec`.
async fn setup_store(dimension: usize) -> laurus::vector::VectorStore {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut field_configs = std::collections::HashMap::new();
    for name in ["img_vec", "txt_vec"] {
        field_configs.insert(
            name.to_string(),
            VectorFieldConfig {
                vector: Some(hnsw(dimension)),
                lexical: None,
            },
        );
    }
    let config = VectorIndexConfig {
        fields: field_configs,
        embedder: Arc::new(MockEmbedder { dimension }),
        default_fields: vec!["img_vec".to_string(), "txt_vec".to_string()],
        metadata: std::collections::HashMap::new(),
        deletion_config: laurus::DeletionConfig::default(),
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    };
    let store = laurus::vector::VectorStore::new(storage, config).unwrap();

    let docs = [
        (1u64, "img_vec", vec![1.0, 0.0, 0.0]),
        (2, "img_vec", vec![0.0, 1.0, 0.0]),
        (3, "txt_vec", vec![1.0, 0.0, 0.0]),
        (4, "txt_vec", vec![0.0, 1.0, 0.0]),
    ];
    for (id, field, v) in docs {
        let doc = Document::builder()
            .add_field(field, DataValue::Vector(v))
            .build();
        store.upsert_document_by_internal_id(id, doc).await.unwrap();
    }
    store.commit().await.unwrap();
    store
}

fn request(
    query_fields: Option<Vec<String>>,
    params_fields: Option<Vec<FieldSelector>>,
) -> VectorSearchRequest {
    VectorSearchRequest {
        query: laurus::vector::VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![1.0, 0.0, 0.0]),
            weight: 1.0,
            fields: query_fields,
        }]),
        params: VectorSearchParams {
            limit: 10,
            score_mode: VectorScoreMode::WeightedSum,
            fields: params_fields,
            ..Default::default()
        },
    }
}

fn doc_ids(results: &laurus::vector::VectorSearchResults) -> std::collections::HashSet<u64> {
    results.hits.iter().map(|h| h.doc_id).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_vector_fields_routes() {
    let store = setup_store(3).await;
    // Per-query field = img_vec → only img_vec docs (1, 2).
    let results = store
        .search(request(Some(vec!["img_vec".into()]), None))
        .unwrap();
    let ids = doc_ids(&results);
    assert!(
        ids.contains(&1) && ids.contains(&2),
        "img_vec docs present: {ids:?}"
    );
    assert!(
        !ids.contains(&3) && !ids.contains(&4),
        "txt_vec docs absent: {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_params_fields_exact() {
    let store = setup_store(3).await;
    // Request-level Exact(txt_vec) → only txt_vec docs (3, 4).
    let results = store
        .search(request(
            None,
            Some(vec![FieldSelector::Exact("txt_vec".into())]),
        ))
        .unwrap();
    let ids = doc_ids(&results);
    assert!(
        ids.contains(&3) && ids.contains(&4),
        "txt_vec docs present: {ids:?}"
    );
    assert!(
        !ids.contains(&1) && !ids.contains(&2),
        "img_vec docs absent: {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_params_fields_prefix() {
    let store = setup_store(3).await;
    // Prefix("img") → resolves to img_vec → only img_vec docs (1, 2).
    let results = store
        .search(request(
            None,
            Some(vec![FieldSelector::Prefix("img".into())]),
        ))
        .unwrap();
    let ids = doc_ids(&results);
    assert!(
        ids.contains(&1) && ids.contains(&2),
        "img_vec docs present: {ids:?}"
    );
    assert!(
        !ids.contains(&3) && !ids.contains(&4),
        "txt_vec docs absent: {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_no_fields_searches_all() {
    let store = setup_store(3).await;
    // No field selector anywhere → all fields searched (regression guard).
    let results = store.search(request(None, None)).unwrap();
    let ids = doc_ids(&results);
    for id in [1, 2, 3, 4] {
        assert!(
            ids.contains(&id),
            "doc {id} should be present with no field filter: {ids:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_per_query_overrides_params() {
    let store = setup_store(3).await;
    // Per-query fields=[img_vec] must win over request-level Exact(txt_vec).
    let results = store
        .search(request(
            Some(vec!["img_vec".into()]),
            Some(vec![FieldSelector::Exact("txt_vec".into())]),
        ))
        .unwrap();
    let ids = doc_ids(&results);
    assert!(
        ids.contains(&1) && ids.contains(&2),
        "per-query img_vec wins: {ids:?}"
    );
    assert!(
        !ids.contains(&3) && !ids.contains(&4),
        "txt_vec excluded: {ids:?}"
    );
}
