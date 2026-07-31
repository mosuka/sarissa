//! Integration tests for the multi-vector search path with rayon
//! parallelisation (issue [#710](https://github.com/mosuka/laurus/issues/710)
//! Phase 1 of [#648](https://github.com/mosuka/laurus/issues/648)).
//!
//! These tests verify that the parallel and serial code paths inside
//! [`laurus::vector::VectorStore::search_with_threshold`] produce identical
//! results for the same input. The boundary case (B = 3 below the
//! [`MULTI_QUERY_PARALLEL_THRESHOLD`] vs B = 4 at the threshold) is also
//! covered to make sure the dispatch logic is symmetric.

use async_trait::async_trait;
use laurus::lexical::LexicalIndexConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
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

async fn setup_store(dimension: usize) -> laurus::vector::VectorStore {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut field_configs = std::collections::HashMap::new();
    field_configs.insert(
        "vector_field".to_string(),
        VectorFieldConfig {
            vector: Some(FieldOption::Hnsw(HnswOption {
                dimension,
                distance: DistanceMetric::Cosine,
                m: 16,
                ef_construction: 100,
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
    let config = VectorIndexConfig {
        fields: field_configs,
        embedder: Arc::new(MockEmbedder { dimension }),
        default_fields: vec!["vector_field".to_string()],
        metadata: std::collections::HashMap::new(),
        deletion_config: laurus::DeletionConfig::default(),
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    };
    laurus::vector::VectorStore::new(storage, config).unwrap()
}

async fn build_dataset(store: &laurus::vector::VectorStore, n: usize, dimension: usize) {
    for i in 0..n {
        let mut v = vec![0.0_f32; dimension];
        v[i % dimension] = 1.0;
        v[(i + 1) % dimension] = 0.5;
        let doc = Document::builder()
            .add_field("vector_field", DataValue::Vector(v))
            .build();
        store
            .upsert_document_by_internal_id((i + 1) as u64, doc)
            .await
            .unwrap();
    }
    store.commit().await.unwrap();
}

fn build_queries(b: usize, dimension: usize) -> Vec<QueryVector> {
    (0..b)
        .map(|i| {
            let mut v = vec![0.0_f32; dimension];
            v[i % dimension] = 1.0;
            QueryVector {
                vector: Vector::new(v),
                weight: 1.0,
                fields: None,
            }
        })
        .collect()
}

fn build_request(
    query_vectors: Vec<QueryVector>,
    score_mode: VectorScoreMode,
) -> VectorSearchRequest {
    VectorSearchRequest {
        query: laurus::vector::VectorSearchQuery::Vectors(query_vectors),
        params: VectorSearchParams {
            limit: 10,
            score_mode,
            ..Default::default()
        },
    }
}

fn assert_hits_match(
    a: &laurus::vector::VectorSearchResults,
    b: &laurus::vector::VectorSearchResults,
) {
    assert_eq!(
        a.hits.len(),
        b.hits.len(),
        "result lengths differ: a={}, b={}",
        a.hits.len(),
        b.hits.len(),
    );
    let mut sa = a.hits.clone();
    let mut sb = b.hits.clone();
    sa.sort_by_key(|h| h.doc_id);
    sb.sort_by_key(|h| h.doc_id);
    for (x, y) in sa.iter().zip(sb.iter()) {
        assert_eq!(x.doc_id, y.doc_id, "doc_ids differ");
        assert!(
            (x.score - y.score).abs() < 1e-5,
            "scores differ for doc_id={}: a={}, b={}",
            x.doc_id,
            x.score,
            y.score,
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_multi_query_parallel_results_match_serial() {
    let dimension = 16;
    let store = setup_store(dimension).await;
    build_dataset(&store, 100, dimension).await;

    let queries = build_queries(8, dimension);
    let req_serial = build_request(queries.clone(), VectorScoreMode::WeightedSum);
    let req_parallel = build_request(queries, VectorScoreMode::WeightedSum);

    let serial = store.search_with_threshold(req_serial, usize::MAX).unwrap();
    let parallel = store.search_with_threshold(req_parallel, 0).unwrap();

    assert_hits_match(&serial, &parallel);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_multi_query_threshold_boundary_b3_b4() {
    let dimension = 16;
    let store = setup_store(dimension).await;
    build_dataset(&store, 100, dimension).await;

    // B=3 must go through the serial path at the default threshold (4).
    // Force-parallel (threshold=0) must produce the same results.
    let queries3 = build_queries(3, dimension);
    let req_default = build_request(queries3.clone(), VectorScoreMode::MaxSim);
    let req_forced_parallel = build_request(queries3, VectorScoreMode::MaxSim);

    let default_path = store.search_with_threshold(req_default, 4).unwrap();
    let forced_parallel = store.search_with_threshold(req_forced_parallel, 0).unwrap();
    assert_hits_match(&default_path, &forced_parallel);

    // B=4 must go through the parallel path at the default threshold (4).
    // Force-serial (threshold=usize::MAX) must produce the same results.
    let queries4 = build_queries(4, dimension);
    let req_default = build_request(queries4.clone(), VectorScoreMode::MaxSim);
    let req_forced_serial = build_request(queries4, VectorScoreMode::MaxSim);

    let default_path = store.search_with_threshold(req_default, 4).unwrap();
    let forced_serial = store
        .search_with_threshold(req_forced_serial, usize::MAX)
        .unwrap();
    assert_hits_match(&default_path, &forced_serial);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_multi_query_score_modes_match_under_parallel() {
    let dimension = 16;
    let store = setup_store(dimension).await;
    build_dataset(&store, 100, dimension).await;

    let queries = build_queries(6, dimension);

    for mode in [
        VectorScoreMode::WeightedSum,
        VectorScoreMode::MaxSim,
        VectorScoreMode::LateInteraction,
    ] {
        let req_s = build_request(queries.clone(), mode);
        let req_p = build_request(queries.clone(), mode);
        let serial = store.search_with_threshold(req_s, usize::MAX).unwrap();
        let parallel = store.search_with_threshold(req_p, 0).unwrap();
        assert_hits_match(&serial, &parallel);
    }
}
