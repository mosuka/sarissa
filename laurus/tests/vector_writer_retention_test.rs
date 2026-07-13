//! Integration tests for retaining the HNSW writer across commits
//! (Issue #864 / #572).
//!
//! Before the fix, `VectorStore::commit` `take()`d the cached writer, so the
//! first upsert after every commit reconstructed it via
//! `HnswIndexWriter::with_storage` — reloading and dequantizing the entire
//! `.hnsw` file. Retention keeps the committed writer (whose in-memory state
//! equals the file it just wrote) in the cache.
//!
//! The two safety valves are the corruption tests: compaction
//! (`maybe_auto_compact` inside commit) and `VectorStore::optimize` rewrite
//! the index through a **fresh** writer and clear the deletion bitmap, so a
//! stale retained writer would resurrect the physically reclaimed vectors on
//! its next commit. Both paths must invalidate the cache.
//!
//! The primary gate is deterministic: a `CountingStorage` decorator counts
//! `open_input(".hnsw")` calls, so the no-reload claim is asserted exactly.

use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{FileMetadata, LoadingMode, Storage, StorageInput, StorageOutput};
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
const STEP: f32 = 0.01;
const HNSW_FILE: &str = "vector_index.hnsw";
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

/// Decorator over [`MemoryStorage`] counting `open_input` / `create_output`
/// calls per file — the deterministic signals for "did this operation reload
/// the `.hnsw`?" and "did this commit rewrite the `.hnsw`?" — plus an
/// optional fault injector failing `create_output` for matching names, to
/// exercise the commit ladder's error paths.
#[derive(Debug)]
struct CountingStorage {
    inner: MemoryStorage,
    opens: std::sync::Mutex<HashMap<String, usize>>,
    creates: std::sync::Mutex<HashMap<String, usize>>,
    fail_create_matching: std::sync::Mutex<Option<String>>,
}

impl CountingStorage {
    fn new() -> Self {
        Self {
            inner: MemoryStorage::new(MemoryStorageConfig::default()),
            opens: std::sync::Mutex::new(HashMap::new()),
            creates: std::sync::Mutex::new(HashMap::new()),
            fail_create_matching: std::sync::Mutex::new(None),
        }
    }

    fn open_count(&self, name: &str) -> usize {
        self.opens.lock().unwrap().get(name).copied().unwrap_or(0)
    }

    /// Total `create_output` calls whose file name contains `pat`.
    fn create_count_matching(&self, pat: &str) -> usize {
        self.creates
            .lock()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.contains(pat))
            .map(|(_, n)| n)
            .sum()
    }

    /// Arm (`Some(substring)`) or disarm (`None`) the `create_output` fault.
    fn set_fail_create_matching(&self, pat: Option<&str>) {
        *self.fail_create_matching.lock().unwrap() = pat.map(String::from);
    }
}

impl Storage for CountingStorage {
    fn loading_mode(&self) -> LoadingMode {
        self.inner.loading_mode()
    }
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        *self
            .opens
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_insert(0) += 1;
        self.inner.open_input(name)
    }
    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        if let Some(pat) = self.fail_create_matching.lock().unwrap().as_deref()
            && name.contains(pat)
        {
            return Err(LaurusError::storage(format!(
                "injected create_output failure for '{name}'"
            )));
        }
        *self
            .creates
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_insert(0) += 1;
        self.inner.create_output(name)
    }
    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.inner.create_output_append(name)
    }
    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }
    fn delete_file(&self, name: &str) -> Result<()> {
        self.inner.delete_file(name)
    }
    fn list_files(&self) -> Result<Vec<String>> {
        self.inner.list_files()
    }
    fn file_size(&self, name: &str) -> Result<u64> {
        self.inner.file_size(name)
    }
    fn metadata(&self, name: &str) -> Result<FileMetadata> {
        self.inner.metadata(name)
    }
    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.inner.rename_file(old_name, new_name)
    }
    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }
    fn sync(&self) -> Result<()> {
        self.inner.sync()
    }
    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }
}

fn doc_vec(i: u64) -> Vec<f32> {
    let theta = i as f32 * STEP;
    let mut v = vec![0.0; DIM];
    v[0] = theta.cos();
    v[1] = theta.sin();
    v
}

fn vec_doc(i: u64) -> Document {
    Document::builder()
        .add_field("vec", DataValue::Vector(doc_vec(i)))
        .build()
}

fn hnsw() -> FieldOption {
    FieldOption::Hnsw(HnswOption {
        dimension: DIM,
        distance: DistanceMetric::Cosine,
        m: 16,
        ef_construction: 100,
        // Near-exhaustive search for the ≤101 vectors used here, so the
        // exact-count assertions are not subject to HNSW recall noise.
        default_ef_search: Some(400),
        base_weight: 1.0,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
    })
}

fn make_config(auto_compaction: bool, threshold: f64) -> VectorIndexConfig {
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
            compaction_threshold: threshold,
            ..Default::default()
        },
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    }
}

fn request(limit: usize) -> VectorSearchRequest {
    let mut query = vec![0.0; DIM];
    query[0] = 1.0;
    VectorSearchRequest {
        query: VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(query),
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

fn hit_ids(store: &laurus::vector::VectorStore, limit: usize) -> HashSet<u64> {
    store
        .search(request(limit))
        .unwrap()
        .hits
        .iter()
        .map(|h| h.doc_id)
        .collect()
}

/// #864: with retention, an upsert arriving after a commit must not re-open
/// the `.hnsw` file — pre-fix it reloaded (and dequantized) the whole index.
#[tokio::test(flavor = "multi_thread")]
async fn upsert_after_commit_does_not_reload_hnsw() {
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn Storage> = counting.clone();
    let store = laurus::vector::VectorStore::new(storage, make_config(false, 0.5)).unwrap();

    for id in 0..10u64 {
        store
            .upsert_document_by_internal_id(id, vec_doc(id))
            .await
            .unwrap();
    }
    store.commit().await.unwrap();

    let opens_after_commit = counting.open_count(HNSW_FILE);
    store
        .upsert_document_by_internal_id(10, vec_doc(10))
        .await
        .unwrap();
    assert_eq!(
        counting.open_count(HNSW_FILE),
        opens_after_commit,
        "the retained writer must serve the post-commit upsert without \
         re-opening the .hnsw (pre-#864: full reload per commit cycle)"
    );

    // And the follow-up commit persists correctly.
    store.commit().await.unwrap();
    assert_eq!(
        counting.open_count(HNSW_FILE),
        opens_after_commit,
        "the second commit writes from the retained writer; no reload either"
    );
}

/// #864 correctness: three upsert+commit cycles through the retained writer
/// preserve every record, both for the live store and after a cold reopen
/// (proving the retained writer kept writing complete, valid files).
///
/// Membership is asserted via `stats().document_count` (record-level, exact)
/// rather than exact search hit sets: incremental HNSW appends can
/// nondeterministically leave nodes unreachable on layer 0 — a pre-existing
/// bug reproduced on unmodified main, tracked as #868 — so search-based
/// full-recall assertions would flake for reasons unrelated to retention.
#[tokio::test(flavor = "multi_thread")]
async fn retained_writer_commit_cycles_preserve_all_vectors() {
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn Storage> = counting.clone();
    let store = laurus::vector::VectorStore::new(storage.clone(), make_config(false, 0.5)).unwrap();

    for cycle in 0..3u64 {
        for i in 0..10u64 {
            let id = cycle * 10 + i;
            store
                .upsert_document_by_internal_id(id, vec_doc(id))
                .await
                .unwrap();
        }
        store.commit().await.unwrap();
    }

    let expected: HashSet<u64> = (0..30).collect();
    assert_eq!(
        store.stats().unwrap().document_count,
        30,
        "all 30 records across 3 retained-writer commit cycles must persist \
         (no loss, no duplication)"
    );
    let ids = hit_ids(&store, 30);
    assert!(
        ids.is_subset(&expected) && !ids.is_empty(),
        "search must return only the ingested ids: {ids:?}"
    );

    // Cold reopen on the same storage: the files must be self-sufficient.
    drop(store);
    let reopened = laurus::vector::VectorStore::new(storage, make_config(false, 0.5)).unwrap();
    assert_eq!(
        reopened.stats().unwrap().document_count,
        30,
        "a cold reopen must see the same 30 records"
    );
    let ids = hit_ids(&reopened, 30);
    assert!(
        ids.is_subset(&expected) && !ids.is_empty(),
        "post-reopen search must return only the ingested ids: {ids:?}"
    );
}

/// #864 corruption test: auto-compaction inside commit rewrites the `.hnsw`
/// through a fresh writer and clears the delmap. The retained writer MUST be
/// invalidated when compaction ran — otherwise its next commit resurrects
/// the physically reclaimed vectors with no deletion bitmap marking them.
#[tokio::test(flavor = "multi_thread")]
async fn auto_compaction_invalidates_retained_writer_no_resurrection() {
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn Storage> = counting.clone();
    // 30% threshold, auto-compaction ON.
    let store = laurus::vector::VectorStore::new(storage.clone(), make_config(true, 0.3)).unwrap();

    for id in 0..100u64 {
        store
            .upsert_document_by_internal_id(id, vec_doc(id))
            .await
            .unwrap();
    }
    store.commit().await.unwrap();

    // Delete 40 of 100 (40% >= 30%): this commit runs compaction.
    let deleted: HashSet<u64> = (0..40).collect();
    for &id in &deleted {
        store.delete_document_by_internal_id(id).await.unwrap();
    }
    store.commit().await.unwrap();
    assert!(
        !storage.file_exists(DELMAP_FILE),
        "precondition: compaction must have run and removed the .delmap"
    );

    // The dangerous cycle: upsert + commit AFTER compaction. A stale
    // retained writer would write the 40 reclaimed vectors back.
    store
        .upsert_document_by_internal_id(100, vec_doc(100))
        .await
        .unwrap();
    store.commit().await.unwrap();

    // Deterministic resurrection gate: a stale writer would rewrite the 40
    // reclaimed records, bouncing the record count from 61 back to 101.
    // (Search hit sets are not asserted exactly because of the pre-existing
    // incremental-append reachability bug #868.)
    assert_eq!(
        store.stats().unwrap().document_count,
        61,
        "exactly the 60 survivors + the new doc may exist on disk — more \
         means the stale retained writer resurrected compacted-away records"
    );
    let ids = hit_ids(&store, 100);
    assert!(
        ids.is_disjoint(&deleted),
        "compacted-away docs must NOT resurface in search after a \
         post-compaction commit from the (invalidated) writer cache: {ids:?}"
    );
}

/// #864 corruption test, explicit-`optimize` variant: `VectorStore::optimize`
/// physically reclaims soft-deleted docs through a fresh writer and clears
/// the delmap; the retained writer cache must be dropped there too.
#[tokio::test(flavor = "multi_thread")]
async fn store_optimize_invalidates_retained_writer() {
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn Storage> = counting.clone();
    // Auto-compaction OFF so only the explicit optimize() reclaims.
    let store = laurus::vector::VectorStore::new(storage.clone(), make_config(false, 0.9)).unwrap();

    for id in 0..100u64 {
        store
            .upsert_document_by_internal_id(id, vec_doc(id))
            .await
            .unwrap();
    }
    store.commit().await.unwrap();

    let deleted: HashSet<u64> = (0..40).collect();
    for &id in &deleted {
        store.delete_document_by_internal_id(id).await.unwrap();
    }
    store.commit().await.unwrap();

    store.optimize().await.unwrap();
    assert!(
        !storage.file_exists(DELMAP_FILE),
        "precondition: optimize must have reclaimed and removed the .delmap"
    );

    store
        .upsert_document_by_internal_id(100, vec_doc(100))
        .await
        .unwrap();
    store.commit().await.unwrap();

    // Same deterministic gate as the auto-compaction variant (see #868 for
    // why exact search hit sets are not asserted).
    assert_eq!(
        store.stats().unwrap().document_count,
        61,
        "exactly the 60 survivors + the new doc may exist on disk — more \
         means the stale retained writer resurrected optimized-away records"
    );
    let ids = hit_ids(&store, 100);
    assert!(
        ids.is_disjoint(&deleted),
        "optimized-away docs must NOT resurface in search after a \
         post-optimize commit from the (invalidated) writer cache: {ids:?}"
    );
}

/// #864 review follow-up: a retained writer with **no pending changes** must
/// not rewrite the `.hnsw` on a no-op commit — `has_pending_changes()` gates
/// the flush, so back-to-back commits pay zero index writes.
#[tokio::test(flavor = "multi_thread")]
async fn noop_commit_does_not_rewrite_hnsw() {
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn Storage> = counting.clone();
    let store = laurus::vector::VectorStore::new(storage, make_config(false, 0.5)).unwrap();

    for id in 0..10u64 {
        store
            .upsert_document_by_internal_id(id, vec_doc(id))
            .await
            .unwrap();
    }
    store.commit().await.unwrap();

    let writes_after_commit = counting.create_count_matching(".hnsw");
    assert!(
        writes_after_commit > 0,
        "the first commit must write the index"
    );

    // Two commits with zero changes: the retained (finalized) writer must be
    // skipped, not re-committed into an identical full rewrite.
    store.commit().await.unwrap();
    store.commit().await.unwrap();
    assert_eq!(
        counting.create_count_matching(".hnsw"),
        writes_after_commit,
        "no-change commits must not rewrite the .hnsw from the retained writer"
    );

    // A real change still triggers exactly one more write cycle.
    store
        .upsert_document_by_internal_id(10, vec_doc(10))
        .await
        .unwrap();
    store.commit().await.unwrap();
    assert!(
        counting.create_count_matching(".hnsw") > writes_after_commit,
        "a commit with pending changes must write the index again"
    );
    assert_eq!(store.stats().unwrap().document_count, 11);
}

/// #864 review follow-up: `optimize()` must flush a dirty writer whose buffer
/// was **emptied by deletions** — `pending_docs() == 0` there, but dropping
/// it would silently discard the uncommitted delete-everything mutation and
/// leave the stale vector on disk.
#[tokio::test(flavor = "multi_thread")]
async fn optimize_flushes_writer_emptied_by_deletions() {
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn Storage> = counting.clone();
    let store = laurus::vector::VectorStore::new(storage, make_config(false, 0.9)).unwrap();

    store
        .upsert_document_by_internal_id(0, vec_doc(0))
        .await
        .unwrap();
    store.commit().await.unwrap();
    assert_eq!(store.stats().unwrap().document_count, 1);

    // Re-upsert the same internal id with a doc carrying NO embeddable
    // fields (an integer is neither a vector nor embeddable text/bytes):
    // the writer buffer-deletes the old vector and adds nothing, leaving a
    // dirty writer with pending_docs() == 0.
    let empty_doc = Document::builder()
        .add_field("note", DataValue::Int64(42))
        .build();
    store
        .upsert_document_by_internal_id(0, empty_doc)
        .await
        .unwrap();

    store.optimize().await.unwrap();

    assert_eq!(
        store.stats().unwrap().document_count,
        0,
        "optimize must flush the emptied writer's uncommitted deletion \
         instead of dropping it and resurrecting the stale vector"
    );
}

/// #864 review follow-up: a commit that fails mid-ladder must DROP the
/// retained writer (the pre-retention behavior on every path) — the
/// writer/disk agreement is unknown after a partial failure, so the next
/// upsert must reload ground truth from storage.
#[tokio::test(flavor = "multi_thread")]
async fn failed_commit_drops_retained_writer() {
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn Storage> = counting.clone();
    let store = laurus::vector::VectorStore::new(storage, make_config(false, 0.5)).unwrap();

    for id in 0..5u64 {
        store
            .upsert_document_by_internal_id(id, vec_doc(id))
            .await
            .unwrap();
    }
    store.commit().await.unwrap();

    // Make the writer dirty, then fail its flush inside the next commit.
    store
        .upsert_document_by_internal_id(5, vec_doc(5))
        .await
        .unwrap();
    counting.set_fail_create_matching(Some(".hnsw"));
    store
        .commit()
        .await
        .expect_err("the injected create_output failure must fail the commit");
    counting.set_fail_create_matching(None);

    // The failed writer must be gone: the next upsert reloads the intact
    // on-disk index (observable as a fresh .hnsw open) instead of reusing a
    // writer in an unknown state.
    let opens_before = counting.open_count(HNSW_FILE);
    store
        .upsert_document_by_internal_id(6, vec_doc(6))
        .await
        .unwrap();
    assert!(
        counting.open_count(HNSW_FILE) > opens_before,
        "after a failed commit the cache must be empty, forcing a reload"
    );

    // And the store recovers fully: doc 5 was lost with the failed writer
    // (it is WAL-replayable at engine level), docs 0-4 + 6 are intact.
    store.commit().await.unwrap();
    let ids = hit_ids(&store, 10);
    assert!(
        ids.contains(&6),
        "post-recovery upsert must be live: {ids:?}"
    );
    assert_eq!(
        store.stats().unwrap().document_count,
        6,
        "the intact on-disk docs (0-4) plus the post-recovery doc (6)"
    );
}
