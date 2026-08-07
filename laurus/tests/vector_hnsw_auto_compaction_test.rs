//! Integration tests for HNSW auto-compaction on commit (issue #782).
//!
//! #624 made HNSW deletion logical (mark a bitmap, filter at search) with
//! manual `optimize()` reclamation. #782 wires `DeletionConfig::auto_compaction`
//! / `compaction_threshold` into `commit()`: once the deletion ratio
//! (deleted / committed nodes) crosses the threshold, the commit triggers
//! compaction automatically so tombstones do not accumulate forever.
//!
//! These tests assert the trigger logic via its observable effect — the
//! `<name>.delmap` file is removed when compaction runs and kept when it does
//! not — plus that deleted documents never appear in either case.

use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::Storage;
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
const N: u64 = 100;
const STEP: f32 = 0.01;
/// Deletion bitmap for the "vec" field's sub-index (Issue #948:
/// `MultiFieldVectorIndex` gives every vector field its own
/// `PrefixedStorage` directory, and each field's sub-index is internally
/// named `"index"`, not the field name).
const DELMAP_FILE: &str = "vec/index.delmap";
const THRESHOLD: f64 = 0.3;

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

fn doc_vec(i: u64) -> Vec<f32> {
    let theta = i as f32 * STEP;
    let mut v = vec![0.0; DIM];
    v[0] = theta.cos();
    v[1] = theta.sin();
    v
}

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
        pq_codebook_path: None,
    })
}

/// Config with explicit auto-compaction policy.
fn make_config(auto_compaction: bool) -> VectorIndexConfig {
    let mut field_configs = HashMap::new();
    field_configs.insert(
        "vec".to_string(),
        VectorFieldConfig {
            vector: Some(hnsw()),
            lexical: None,
        },
    );
    VectorIndexConfig {
        fields: field_configs,
        embedder: Arc::new(MockEmbedder { dimension: DIM }),
        default_fields: vec!["vec".to_string()],
        metadata: HashMap::new(),
        deletion_config: laurus::DeletionConfig {
            auto_compaction,
            compaction_threshold: THRESHOLD,
            ..Default::default()
        },
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    }
}

async fn setup_store(auto_compaction: bool) -> (laurus::vector::VectorStore, Arc<dyn Storage>) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store =
        laurus::vector::VectorStore::new(storage.clone(), make_config(auto_compaction)).unwrap();
    for id in 0..N {
        let doc = Document::builder()
            .add_field("vec", DataValue::Vector(doc_vec(id)))
            .build();
        store.upsert_document_by_internal_id(id, doc).await.unwrap();
    }
    store.commit().await.unwrap();
    (store, storage)
}

async fn delete_and_commit(store: &laurus::vector::VectorStore, ids: &[u64]) {
    for &id in ids {
        store.delete_document_by_internal_id(id).await.unwrap();
    }
    store.commit().await.unwrap();
}

fn request(limit: usize) -> VectorSearchRequest {
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
            allowed_ids: None,
            ..Default::default()
        },
    }
}

fn hit_ids(results: &laurus::vector::VectorSearchResults) -> HashSet<u64> {
    results.hits.iter().map(|h| h.doc_id).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn auto_compaction_fires_when_ratio_crosses_threshold() {
    let (store, storage) = setup_store(true).await;
    // Delete 40 of 100 (40% >= 30%): the commit must auto-compact.
    let deleted: HashSet<u64> = (0..40).collect();
    delete_and_commit(&store, &deleted.iter().copied().collect::<Vec<_>>()).await;

    assert!(
        !storage.file_exists(DELMAP_FILE),
        "auto-compaction should have purged and removed the .delmap"
    );
    let ids = hit_ids(&store.search(request(10)).unwrap());
    assert!(
        ids.is_disjoint(&deleted),
        "deleted docs must not reappear after auto-compaction: {ids:?}"
    );
    assert_eq!(ids.len(), 10, "survivors must remain searchable: {ids:?}");
    assert!(ids.iter().all(|id| (40..N).contains(id)));
}

#[tokio::test(flavor = "multi_thread")]
async fn no_auto_compaction_below_threshold() {
    let (store, storage) = setup_store(true).await;
    // Delete 10 of 100 (10% < 30%): no compaction, deletions stay logical.
    let deleted: HashSet<u64> = (0..10).collect();
    delete_and_commit(&store, &deleted.iter().copied().collect::<Vec<_>>()).await;

    assert!(
        storage.file_exists(DELMAP_FILE),
        "below threshold the .delmap must remain (no compaction)"
    );
    let ids = hit_ids(&store.search(request(10)).unwrap());
    assert!(
        ids.is_disjoint(&deleted),
        "deleted docs must still be filtered out: {ids:?}"
    );
    assert_eq!(ids.len(), 10);
}

#[tokio::test(flavor = "multi_thread")]
async fn auto_compaction_disabled_never_fires() {
    let (store, storage) = setup_store(false).await;
    // Delete 40 of 100 (40% >= 30%) but with auto_compaction off: still logical.
    let deleted: HashSet<u64> = (0..40).collect();
    delete_and_commit(&store, &deleted.iter().copied().collect::<Vec<_>>()).await;

    assert!(
        storage.file_exists(DELMAP_FILE),
        "with auto_compaction disabled the .delmap must remain even above threshold"
    );
    let ids = hit_ids(&store.search(request(10)).unwrap());
    assert!(
        ids.is_disjoint(&deleted),
        "deleted docs must still be filtered out: {ids:?}"
    );
    assert_eq!(ids.len(), 10);
}
