//! Integration tests for filter-aware HNSW traversal (issue #645).
//!
//! The store holds one HNSW field `vec` whose documents lie on a 1-D angular
//! gradient: doc `i` points at angle `i * STEP` in the first two dimensions,
//! so cosine similarity to the query (angle 0) decreases monotonically with
//! `i`. The nearest-neighbour order is therefore exactly `0, 1, 2, ...`, which
//! makes every assertion deterministic.
//!
//! The headline case is `selective_filter_reaches_distant_match`: an allowed
//! doc placed far down the gradient (well outside the `ef_search` window a
//! plain post-filter would examine) must still be returned, because the
//! frontier expands through the intervening non-matching docs to reach it.

use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::store::config::VectorFieldConfig;
use laurus::vector::store::request::{
    QueryVector, VectorScoreMode, VectorSearchParams, VectorSearchRequest,
};
use laurus::vector::{FieldOption, VectorIndexConfig, VectorSearchQuery};
use laurus::{DataValue, Document};
use laurus::{EmbedInput, EmbedInputType, Embedder};
use laurus::{LaurusError, Result};

const DIM: usize = 16;
const N: u64 = 200;
/// Angular step between adjacent docs (~0.46°). `N * STEP < π` keeps cosine
/// similarity monotonically decreasing across the whole corpus.
const STEP: f32 = 0.008;

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

/// Vector for doc `i`: angle `i * STEP` in the first two dims, zero elsewhere.
fn doc_vec(i: u64) -> Vec<f32> {
    let theta = i as f32 * STEP;
    let mut v = vec![0.0; DIM];
    v[0] = theta.cos();
    v[1] = theta.sin();
    v
}

/// Query points at angle 0 (= doc 0's direction).
fn query_vec() -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[0] = 1.0;
    v
}

fn hnsw() -> FieldOption {
    FieldOption::Hnsw(HnswOption {
        dimension: DIM,
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

async fn setup_store() -> laurus::vector::VectorStore {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut field_configs = HashMap::new();
    field_configs.insert(
        "vec".to_string(),
        VectorFieldConfig {
            vector: Some(hnsw()),
            lexical: None,
        },
    );
    let config = VectorIndexConfig {
        fields: field_configs,
        embedder: Arc::new(MockEmbedder { dimension: DIM }),
        default_fields: vec!["vec".to_string()],
        metadata: HashMap::new(),
        deletion_config: laurus::DeletionConfig::default(),
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    };
    let store = laurus::vector::VectorStore::new(storage, config).unwrap();

    for id in 0..N {
        let doc = Document::builder()
            .add_field("vec", DataValue::Vector(doc_vec(id)))
            .build();
        store.upsert_document_by_internal_id(id, doc).await.unwrap();
    }
    store.commit().await.unwrap();
    store
}

/// Build a request routed to field `vec` (so the HNSW graph path is taken,
/// not the linear-scan fallback) with an optional allow-set.
fn request(limit: usize, allowed: Option<Vec<u64>>) -> VectorSearchRequest {
    VectorSearchRequest {
        query: VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(query_vec()),
            weight: 1.0,
            fields: Some(vec!["vec".into()]),
        }]),
        params: VectorSearchParams {
            limit,
            score_mode: VectorScoreMode::WeightedSum,
            fields: None,
            allowed_ids: allowed,
            ..Default::default()
        },
    }
}

fn hit_ids(results: &laurus::vector::VectorSearchResults) -> Vec<u64> {
    results.hits.iter().map(|h| h.doc_id).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn filter_none_returns_nearest() {
    let store = setup_store().await;
    // No filter: the unfiltered path is approximate (graph-dependent), so we
    // assert the robust invariants rather than an exact top-10 — it must
    // return `limit` hits and include the exact nearest neighbour (doc 0,
    // whose vector equals the query).
    let results = store.search(request(10, None)).unwrap();
    let ids = hit_ids(&results);
    assert_eq!(ids.len(), 10, "ten hits");
    assert!(ids.contains(&0), "exact nearest neighbour present: {ids:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn filter_excludes_non_matching() {
    let store = setup_store().await;
    // A selective allow-set never fills the result heap, so the frontier is
    // never pruned and the traversal is exhaustive (hence deterministic): the
    // result is exactly the allowed docs, nearest-first.
    let results = store.search(request(10, Some(vec![3u64, 7, 11]))).unwrap();
    assert_eq!(
        hit_ids(&results),
        vec![3, 7, 11],
        "exactly the allowed docs, nearest first"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn selective_filter_returns_matching_in_order() {
    let store = setup_store().await;
    // Sparse allow-set; nearest-first order is ascending id.
    let allowed = vec![5u64, 50, 150];
    let results = store.search(request(10, Some(allowed))).unwrap();
    assert_eq!(
        hit_ids(&results),
        vec![5, 50, 150],
        "all allowed docs returned, nearest first"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn selective_filter_reaches_distant_match() {
    let store = setup_store().await;
    // Doc 150 sits far down the gradient — outside the ~ef_search window a
    // plain post-filter would inspect. Filter-aware traversal must still
    // reach it by expanding the frontier through non-matching docs.
    let results = store.search(request(10, Some(vec![150u64]))).unwrap();
    assert_eq!(
        hit_ids(&results),
        vec![150],
        "distant lone match must be reached, not lost"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_filter_returns_empty() {
    let store = setup_store().await;
    // An empty allow-set excludes everything; this must be empty, not an error.
    let results = store.search(request(10, Some(vec![]))).unwrap();
    assert!(results.hits.is_empty(), "empty allow-set => no hits");
}

#[tokio::test(flavor = "multi_thread")]
async fn filter_all_returns_nearest() {
    let store = setup_store().await;
    // A non-selective allow-set (every doc) exercises the filtered branch with
    // the result heap filling and pruning, just like the unfiltered path. It
    // is therefore also approximate; assert the robust invariants.
    let all: Vec<u64> = (0..N).collect();
    let results = store.search(request(10, Some(all))).unwrap();
    let ids = hit_ids(&results);
    assert_eq!(ids.len(), 10, "ten hits");
    assert!(
        ids.contains(&0),
        "non-selective filter still returns the nearest: {ids:?}"
    );
}
