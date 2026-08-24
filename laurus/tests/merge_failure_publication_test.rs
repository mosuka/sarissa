//! Reproduction for #1032: a failed merge publishes a partially merged
//! segment through the merge writer's implicit `Drop`-commit.
//!
//! `MergeEngine::perform_merge` replays the source documents through an
//! internally constructed `InvertedIndexWriter` and writes the merged
//! segment via `flush_buffered_to_segment`. If that fails (or the replay
//! loop errors), the function returns early and the writer is dropped with
//! its buffers full — and `Drop` runs `close()` → `commit()`, whose
//! `flush_segment()` + `publish_pending_segments()` write a `segment_*`
//! holding the partially merged documents and flip it to `committed: true`
//! in the same call. The source segments still exist, so every document in
//! the orphaned segment is now live TWICE.

use std::io::Read;
use std::sync::Arc;

use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore, TermQuery};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{Storage, StorageInput, StorageOutput};
use laurus::{Document, LaurusError, Result as LaurusResult};

/// Storage decorator failing the next `create_output` whose name matches
/// the armed `(prefix, suffix)` — aimed at a chosen file of the merged
/// segment, so the merge dies inside `flush_buffered_to_segment` with the
/// writer's buffers still full.
#[derive(Debug)]
struct FailingStorage {
    inner: Arc<dyn Storage>,
    fail_create_matching: parking_lot::Mutex<Option<(String, String)>>,
}

impl FailingStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            fail_create_matching: parking_lot::Mutex::new(None),
        }
    }

    /// Arm a one-shot failure for the next `create_output` whose name has
    /// this prefix (suffix unconstrained).
    fn fail_next_create_with_prefix(&self, prefix: &str) {
        self.fail_next_create_matching(prefix, "");
    }

    /// Arm a one-shot failure for the next `create_output` whose name has
    /// both this prefix and this suffix.
    fn fail_next_create_matching(&self, prefix: &str, suffix: &str) {
        *self.fail_create_matching.lock() = Some((prefix.to_string(), suffix.to_string()));
    }
}

impl Storage for FailingStorage {
    fn create_output(&self, name: &str) -> LaurusResult<Box<dyn StorageOutput>> {
        let armed = {
            let mut guard = self.fail_create_matching.lock();
            if guard
                .as_ref()
                .is_some_and(|(p, x)| name.starts_with(p) && name.ends_with(x))
            {
                *guard = None;
                true
            } else {
                false
            }
        };
        if armed {
            return Err(LaurusError::storage(format!(
                "injected failure creating {name}"
            )));
        }
        self.inner.create_output(name)
    }

    fn create_output_append(&self, name: &str) -> LaurusResult<Box<dyn StorageOutput>> {
        self.inner.create_output_append(name)
    }

    fn open_input(&self, name: &str) -> LaurusResult<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }

    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }

    fn delete_file(&self, name: &str) -> LaurusResult<()> {
        self.inner.delete_file(name)
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> LaurusResult<()> {
        self.inner.rename_file(old_name, new_name)
    }

    fn list_files(&self) -> LaurusResult<Vec<String>> {
        self.inner.list_files()
    }

    fn file_size(&self, name: &str) -> LaurusResult<u64> {
        self.inner.file_size(name)
    }

    fn sync(&self) -> LaurusResult<()> {
        self.inner.sync()
    }

    fn metadata(&self, name: &str) -> LaurusResult<laurus::storage::FileMetadata> {
        self.inner.metadata(name)
    }

    fn create_temp_output(&self, prefix: &str) -> LaurusResult<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }

    fn close(&mut self) -> LaurusResult<()> {
        Ok(())
    }
}

fn doc(body: &str) -> Document {
    Document::builder().add_text("body", body).build()
}

/// Auto-merging config: the second commit's `maybe_merge` merges the two
/// single-doc segments.
fn merging_config() -> LexicalIndexConfig {
    LexicalIndexConfig::builder()
        .max_segments(1)
        .merge_factor(2)
        .build()
}

/// A failed merge must not leave a published segment holding the partially
/// merged documents while the source segments still exist.
#[test]
fn failed_merge_must_not_duplicate_documents() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let store = LexicalStore::new(storage.clone(), merging_config()).unwrap();
    store.upsert_document(1, doc("apple")).unwrap();
    store.commit().unwrap();
    store.upsert_document(2, doc("banana")).unwrap();

    // The second commit's maybe_merge merges the two segments; kill the
    // merged segment's first file write.
    failing.fail_next_create_with_prefix("merged_");
    let result = store.commit();
    assert!(result.is_err(), "the injected merge failure must surface");
    drop(store);

    // Reopen over the clean storage: discovery sees whatever the failed
    // merge left behind.
    let reopened = LexicalStore::new(inner.clone(), merging_config()).unwrap();
    let hits = reopened
        .search(LexicalSearchRequest::new(Box::new(TermQuery::new(
            "body", "apple",
        ))))
        .unwrap();
    assert_eq!(
        hits.hits.len(),
        1,
        "a failed merge must not leave a second live copy of a document"
    );
}

/// Structural variant of the same defect: after a failed merge, no NEW
/// committed segment may exist beyond the sources.
#[test]
fn failed_merge_must_not_publish_a_segment() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let store = LexicalStore::new(storage.clone(), merging_config()).unwrap();
    store.upsert_document(1, doc("apple")).unwrap();
    store.commit().unwrap();

    let metas_before: Vec<String> = list_metas(&inner);

    store.upsert_document(2, doc("banana")).unwrap();
    failing.fail_next_create_with_prefix("merged_");
    assert!(store.commit().is_err());
    drop(store);

    // The new source segment from commit 2 is expected; a further
    // `segment_*` beyond it is the Drop-published orphan.
    let metas_after = list_metas(&inner);
    let new_metas: Vec<&String> = metas_after
        .iter()
        .filter(|m| !metas_before.contains(m))
        .collect();
    assert_eq!(
        new_metas.len(),
        1,
        "a failed merge must add only commit 2's own segment, found: {new_metas:?}"
    );
}

/// The segment `.meta` must be written LAST, so a failure in any data file
/// cannot leave a discoverable segment with files missing.
///
/// Kills the merged segment's `.bkd` write — under the old order the
/// `.meta` (with `committed: true`) had already been written by then, so
/// the half-segment was discoverable and its numeric-range queries would
/// silently return zero hits.
#[test]
fn segment_meta_is_written_after_every_data_file() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let store = LexicalStore::new(storage.clone(), merging_config()).unwrap();
    // Numeric field so the segment has a `.bkd` to fail.
    let numbered = |body: &str, n: i64| {
        Document::builder()
            .add_text("body", body)
            .add_integer("rank", n)
            .build()
    };
    store.upsert_document(1, numbered("apple", 1)).unwrap();
    store.commit().unwrap();
    store.upsert_document(2, numbered("banana", 2)).unwrap();

    failing.fail_next_create_matching("merged_", ".bkd");
    assert!(
        store.commit().is_err(),
        "the injected .bkd failure must surface"
    );
    drop(store);

    let leaked: Vec<String> = inner
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.starts_with("merged_") && f.ends_with(".meta"))
        .collect();
    assert!(
        leaked.is_empty(),
        "a failed data-file write must not leave a discoverable merged segment: {leaked:?}"
    );
}

/// List the committed `segment_*` / `merged_*` `.meta` names.
fn list_metas(storage: &Arc<dyn Storage>) -> Vec<String> {
    let mut metas: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".meta") && f != "index.meta")
        .filter(|f| {
            // Only count committed (published) segments.
            let mut input = storage.open_input(f).unwrap();
            let mut bytes = Vec::new();
            input.read_to_end(&mut bytes).unwrap();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            v["committed"].as_bool().unwrap_or(true)
        })
        .collect();
    metas.sort();
    metas
}
