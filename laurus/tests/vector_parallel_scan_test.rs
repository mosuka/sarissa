//! Integration tests for the rayon-parallel brute-force scan in the Flat
//! and IVF searchers (issue #662).
//!
//! Each store holds a single vector field. Ten "near" documents (ids 1..=10)
//! carry a vector identical to the query direction (cosine similarity 1.0);
//! the remaining "far" documents carry an orthogonal vector (similarity 0.0).
//! A top-10 search must therefore return exactly the near set regardless of
//! corpus size, so the same assertion validates both the serial path (small
//! corpus) and the rayon path (corpus above `PARALLEL_SCAN_THRESHOLD`, which
//! is 2048; the parallel cases use 2100 candidates).
//!
//! The parallel/serial *equivalence* itself is proven by the
//! `parallel_scan` unit tests; these tests confirm the Flat / IVF searchers
//! are wired to it correctly and that ranking is unaffected.

use async_trait::async_trait;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::{FlatOption, IvfOption};
use laurus::vector::store::config::VectorFieldConfig;
use laurus::vector::store::request::{
    QueryVector, VectorScoreMode, VectorSearchParams, VectorSearchRequest,
};
use laurus::vector::{FieldOption, VectorIndexConfig, VectorSearchQuery};
use laurus::{DataValue, Document};
use laurus::{EmbedInput, EmbedInputType, Embedder};
use laurus::{LaurusError, Result};

const DIM: usize = 8;
/// Candidate count for the parallel cases. Must exceed the searcher's
/// internal `PARALLEL_SCAN_THRESHOLD` (2048) so the rayon path is taken.
const PARALLEL_N: u64 = 2100;

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

fn near_vec() -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[0] = 1.0;
    v
}

fn far_vec() -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[1] = 1.0;
    v
}

fn flat() -> FieldOption {
    FieldOption::Flat(FlatOption {
        dimension: DIM,
        distance: DistanceMetric::Cosine,
        base_weight: 1.0,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
    })
}

/// IVF with a single cluster so the (always `n_probe = 1`) searcher probes
/// every vector, putting the whole corpus through the parallel scan.
fn ivf_single_cluster() -> FieldOption {
    FieldOption::Ivf(IvfOption {
        dimension: DIM,
        distance: DistanceMetric::Cosine,
        n_clusters: 1,
        n_probe: 1,
        base_weight: 1.0,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
    })
}

/// Build a store with one vector field `vec` holding `n` documents:
/// ids `1..=10` are "near" (identical to the query), the rest are "far"
/// (orthogonal).
async fn setup_store(field: FieldOption, n: u64) -> laurus::vector::VectorStore {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut field_configs = HashMap::new();
    field_configs.insert(
        "vec".to_string(),
        VectorFieldConfig {
            vector: Some(field),
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

    for id in 1..=n {
        let v = if id <= 10 { near_vec() } else { far_vec() };
        let doc = Document::builder()
            .add_field("vec", DataValue::Vector(v))
            .build();
        store.upsert_document_by_internal_id(id, doc).await.unwrap();
    }
    store.commit().await.unwrap();
    store
}

fn query_request(query_fields: Option<Vec<String>>) -> VectorSearchRequest {
    VectorSearchRequest {
        query: VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(near_vec()),
            weight: 1.0,
            fields: query_fields,
        }]),
        params: VectorSearchParams {
            limit: 10,
            score_mode: VectorScoreMode::WeightedSum,
            fields: None,
            ..Default::default()
        },
    }
}

fn top_ids(results: &laurus::vector::VectorSearchResults) -> HashSet<u64> {
    results.hits.iter().map(|h| h.doc_id).collect()
}

fn near_set() -> HashSet<u64> {
    (1u64..=10).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn flat_unfiltered_parallel_scan() {
    let store = setup_store(flat(), PARALLEL_N).await;
    let results = store.search(query_request(None)).unwrap();
    assert_eq!(
        top_ids(&results),
        near_set(),
        "flat unfiltered parallel path"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn flat_filtered_parallel_scan() {
    let store = setup_store(flat(), PARALLEL_N).await;
    // Per-query field routing (#676) selects the field-filtered scan path.
    let results = store
        .search(query_request(Some(vec!["vec".into()])))
        .unwrap();
    assert_eq!(top_ids(&results), near_set(), "flat filtered parallel path");
}

#[tokio::test(flavor = "multi_thread")]
async fn flat_serial_small_scan() {
    let store = setup_store(flat(), 50).await;
    let results = store.search(query_request(None)).unwrap();
    assert_eq!(top_ids(&results), near_set(), "flat serial path");
}

#[tokio::test(flavor = "multi_thread")]
async fn ivf_parallel_scan() {
    let store = setup_store(ivf_single_cluster(), PARALLEL_N).await;
    let results = store.search(query_request(None)).unwrap();
    assert_eq!(top_ids(&results), near_set(), "ivf parallel path");
}
