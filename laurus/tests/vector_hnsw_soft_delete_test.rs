//! Integration tests for logical (soft) deletion of HNSW documents (issue
//! #624).
//!
//! Before #624 a deletion on the production `VectorStore -> HnswIndex` path
//! went through the writer, which nulled `self.graph` and forced the next
//! commit to rebuild the entire graph (O(N log N) distance evals) — a
//! reliability cliff for delete- and update-heavy workloads. #624 wires the
//! already-built deletion bitmap (issue #684) and deletion-aware traversal
//! (issue #665) into this path: a delete now just marks the bitmap (no
//! rebuild), search filters the document out, and `optimize()` physically
//! reclaims the dead nodes.
//!
//! HNSW graph construction is randomised, so these tests assert only
//! invariants that hold for any graph (deleted ids never returned, page stays
//! full from live docs, survivors remain searchable) plus the concrete
//! lifecycle effects unique to #624: the `.delmap` file is written on commit,
//! removed by `optimize()`, and the bitmap survives a store reopen.

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
const N: u64 = 120;
const STEP: f32 = 0.008;
/// The single-segment `HnswIndex` is always created under this name, so its
/// deletion bitmap lives at `vector_index.delmap`.
const DELMAP_FILE: &str = "vector_index.delmap";

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

/// Vector for doc `i`: angle `i * STEP` in the first two dims (monotonic
/// nearest-neighbour order in ascending `i` for a query at angle 0).
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
    })
}

fn make_config() -> VectorIndexConfig {
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
        // These tests exercise the manual soft-delete + explicit-optimize
        // lifecycle (#624), so disable auto-compaction (#782) — otherwise the
        // 33%-deletion commits would purge the bitmap before the assertions.
        deletion_config: laurus::DeletionConfig {
            auto_compaction: false,
            ..Default::default()
        },
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    }
}

/// Build a store over a fresh `MemoryStorage`, insert `N` gradient docs, and
/// return both the store and a handle to the storage so tests can inspect the
/// on-disk `.delmap` lifecycle.
async fn setup_store() -> (laurus::vector::VectorStore, Arc<dyn Storage>) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = laurus::vector::VectorStore::new(storage.clone(), make_config()).unwrap();

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
async fn soft_delete_writes_delmap_and_excludes_deleted() {
    let (store, storage) = setup_store().await;
    assert!(
        !storage.file_exists(DELMAP_FILE),
        "no deletions yet, so no .delmap should exist"
    );

    let deleted: HashSet<u64> = (0..40).collect();
    delete_and_commit(&store, &deleted.iter().copied().collect::<Vec<_>>()).await;

    // A logical deletion persists a deletion bitmap rather than rebuilding the
    // graph.
    assert!(
        storage.file_exists(DELMAP_FILE),
        "commit must persist the deletion bitmap (.delmap)"
    );

    let ids = hit_ids(&store.search(request(10)).unwrap());
    assert!(
        ids.is_disjoint(&deleted),
        "no soft-deleted doc may appear: deleted={deleted:?}, hits={ids:?}"
    );
    assert_eq!(ids.len(), 10, "page must stay full from live docs: {ids:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn optimize_reclaims_deleted_documents() {
    let (store, storage) = setup_store().await;
    let deleted: HashSet<u64> = (0..40).collect();
    delete_and_commit(&store, &deleted.iter().copied().collect::<Vec<_>>()).await;
    assert!(storage.file_exists(DELMAP_FILE));

    // Compaction physically rebuilds the graph without the deleted docs and
    // drops the bitmap file.
    store.optimize().unwrap();
    assert!(
        !storage.file_exists(DELMAP_FILE),
        "optimize() must remove the .delmap after purging"
    );

    let ids = hit_ids(&store.search(request(10)).unwrap());
    assert!(
        ids.is_disjoint(&deleted),
        "purged docs must not reappear after optimize: hits={ids:?}"
    );
    assert_eq!(
        ids.len(),
        10,
        "survivors must remain searchable after purge: {ids:?}"
    );
    assert!(
        ids.iter().all(|id| (40..N).contains(id)),
        "every hit must be a live survivor in [40, {N}): {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn soft_delete_survives_store_reopen() {
    let (store, storage) = setup_store().await;
    let deleted: HashSet<u64> = (0..40).collect();
    delete_and_commit(&store, &deleted.iter().copied().collect::<Vec<_>>()).await;
    drop(store);

    // Reopen the index over the same storage; the deletion bitmap must be
    // reloaded from `.delmap` so deleted docs stay excluded.
    let reopened = laurus::vector::VectorStore::new(storage.clone(), make_config()).unwrap();
    let ids = hit_ids(&reopened.search(request(10)).unwrap());
    assert!(
        ids.is_disjoint(&deleted),
        "deletions must persist across reopen: deleted={deleted:?}, hits={ids:?}"
    );
    assert_eq!(ids.len(), 10, "page must stay full after reopen: {ids:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn deleting_nearest_still_finds_next_nearest() {
    // Recall invariant: deleting the nearest docs must not collapse recall —
    // the walk fills the page from the next-nearest *live* docs rather than
    // returning deleted ones or a short page.
    let (store, storage) = setup_store().await;
    let deleted: HashSet<u64> = (0..40).collect();
    delete_and_commit(&store, &deleted.iter().copied().collect::<Vec<_>>()).await;

    let before = hit_ids(&store.search(request(10)).unwrap());
    assert_eq!(before.len(), 10);
    assert!(before.iter().all(|id| (40..N).contains(id)));

    // The same invariant must hold after compaction.
    store.optimize().unwrap();
    assert!(!storage.file_exists(DELMAP_FILE));
    let after = hit_ids(&store.search(request(10)).unwrap());
    assert_eq!(after.len(), 10);
    assert!(
        after.iter().all(|id| (40..N).contains(id)),
        "post-purge hits must all be live survivors: {after:?}"
    );
}
