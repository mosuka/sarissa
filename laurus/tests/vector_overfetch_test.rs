//! Integration test for the `overfetch` factor (Issue #675).
//!
//! `VectorStore::search` previously ignored `VectorSearchParams.overfetch` and
//! always fetched `limit * 2` candidates per query vector. This test proves the
//! factor is now honoured end-to-end: a document that is only the *second*
//! nearest to each of two query vectors can only surface in the fused top-`k`
//! when `overfetch` widens the per-query candidate pool past `limit`.
//!
//! A Flat index with the Cosine metric is used so the ranking is exact and
//! deterministic (unlike the randomised HNSW graph).

use async_trait::async_trait;
use laurus::lexical::LexicalIndexConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::FlatOption;
use laurus::vector::store::config::VectorFieldConfig;
use laurus::vector::store::request::{
    QueryVector, VectorScoreMode, VectorSearchParams, VectorSearchRequest,
};
use laurus::vector::{FieldOption, VectorIndexConfig};
use laurus::{DataValue, Document};
use laurus::{EmbedInput, EmbedInputType, Embedder};
use laurus::{LaurusError, Result};
use std::any::Any;
use std::sync::Arc;

#[derive(Debug)]
struct MockEmbedder {
    dimension: usize,
}

#[async_trait]
impl Embedder for MockEmbedder {
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
        "mock"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Build a Flat / Cosine store holding three 2-D documents:
/// - doc 1 = A = (1, 0)
/// - doc 2 = B = (0, 1)
/// - doc 3 = D = (1, 1) (the diagonal — second-nearest to both queries below)
async fn setup_store() -> laurus::vector::VectorStore {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut field_configs = std::collections::HashMap::new();
    field_configs.insert(
        "v".to_string(),
        VectorFieldConfig {
            vector: Some(FieldOption::Flat(FlatOption {
                dimension: 2,
                distance: DistanceMetric::Cosine,
                ..Default::default()
            })),
            lexical: None,
        },
    );
    let config = VectorIndexConfig {
        fields: field_configs,
        embedder: Arc::new(MockEmbedder { dimension: 2 }),
        default_fields: vec!["v".to_string()],
        metadata: std::collections::HashMap::new(),
        deletion_config: laurus::DeletionConfig::default(),
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    };
    let store = laurus::vector::VectorStore::new(storage, config).unwrap();

    for (id, vec) in [
        (1u64, vec![1.0, 0.0]),
        (2, vec![0.0, 1.0]),
        (3, vec![1.0, 1.0]),
    ] {
        let doc = Document::builder()
            .add_field("v", DataValue::Vector(vec))
            .build();
        store.upsert_document_by_internal_id(id, doc).await.unwrap();
    }
    store.commit().await.unwrap();
    store
}

/// Two query vectors close to A and B respectively. The diagonal D is the
/// second-nearest to each, so it is excluded from a `top_k = 1` pool but
/// included from a `top_k = 2` pool.
fn two_queries() -> Vec<QueryVector> {
    // ~10 degrees off the x-axis (near A) and ~10 degrees off the y-axis
    // (near B). Cosine similarities to {A, D, B}: roughly {0.98, 0.82, 0.17}.
    let q1 = vec![(10f32).to_radians().cos(), (10f32).to_radians().sin()];
    let q2 = vec![(80f32).to_radians().cos(), (80f32).to_radians().sin()];
    vec![
        QueryVector {
            vector: Vector::new(q1),
            weight: 1.0,
            fields: None,
        },
        QueryVector {
            vector: Vector::new(q2),
            weight: 1.0,
            fields: None,
        },
    ]
}

fn request(overfetch: f32) -> VectorSearchRequest {
    VectorSearchRequest {
        query: laurus::vector::VectorSearchQuery::Vectors(two_queries()),
        params: VectorSearchParams {
            limit: 1,
            score_mode: VectorScoreMode::WeightedSum,
            overfetch,
            ..Default::default()
        },
    }
}

/// With `overfetch = 1.0` each query fetches only its single nearest doc
/// (A for q1, B for q2), so the diagonal D never enters the fused pool and the
/// top hit is the specialist A. With `overfetch = 2.0` each query also fetches
/// its second-nearest (D), whose summed score across both queries wins the top
/// slot. The differing winners prove the factor is honoured.
#[tokio::test]
async fn overfetch_controls_fused_candidate_pool() {
    let store = setup_store().await;

    let narrow = store.search(request(1.0)).unwrap();
    assert_eq!(narrow.hits.len(), 1, "limit = 1 must return one hit");
    assert_ne!(
        narrow.hits[0].doc_id, 3,
        "with overfetch = 1.0 the diagonal D (doc 3) is outside each query's \
         top_k = 1 pool and must not surface"
    );

    let wide = store.search(request(2.0)).unwrap();
    assert_eq!(wide.hits.len(), 1, "limit = 1 must return one hit");
    assert_eq!(
        wide.hits[0].doc_id, 3,
        "with overfetch = 2.0 the diagonal D (doc 3) enters both queries' \
         top_k = 2 pool and wins the fused top slot"
    );
}
