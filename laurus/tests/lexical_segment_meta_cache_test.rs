//! Integration tests for the `InvertedIndexWriter` segment-range cache and
//! cached `DeletionManager` (Issue #864, closing #559 / #571).
//!
//! Before the fix, every upsert ran `find_segments_for_doc`, which listed the
//! storage and JSON-parsed **every** `*.meta` file, and every overwrite of a
//! committed document constructed a fresh `DeletionManager` (reloading every
//! `.delmap` bitmap). The cache mirrors the `.meta` ranges in memory: it is
//! seeded by the constructor's existing recovery scan, extended on flush, and
//! rebuilt by `invalidate_segment_cache` after `LexicalStore::optimize`'s
//! force-merge — the only path that rewrites segments behind a live writer.
//!
//! The primary gate is deterministic: a `CountingStorage` decorator counts
//! `list_files` calls, so the tests assert exact I/O counts instead of timing.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use laurus::storage::{FileMetadata, LoadingMode, Storage, StorageInput, StorageOutput};
use laurus::{Document, Result};

use laurus::lexical::{
    InvertedIndexWriter, InvertedIndexWriterConfig, LexicalIndexConfig, LexicalSearchRequest,
    LexicalStore, TermQuery,
};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

/// Decorator over [`MemoryStorage`] counting `list_files` invocations —
/// the deterministic signal for "did this operation rescan segment metadata?"
/// — plus an optional fault injector failing `list_files`, to exercise the
/// cache-rebuild error path.
#[derive(Debug)]
struct CountingStorage {
    inner: MemoryStorage,
    list_files_calls: AtomicUsize,
    fail_list_files: std::sync::atomic::AtomicBool,
}

impl CountingStorage {
    fn new() -> Self {
        Self {
            inner: MemoryStorage::new(MemoryStorageConfig::default()),
            list_files_calls: AtomicUsize::new(0),
            fail_list_files: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn list_files_count(&self) -> usize {
        self.list_files_calls.load(Ordering::SeqCst)
    }

    fn set_fail_list_files(&self, fail: bool) {
        self.fail_list_files.store(fail, Ordering::SeqCst);
    }
}

impl Storage for CountingStorage {
    fn loading_mode(&self) -> LoadingMode {
        self.inner.loading_mode()
    }
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }
    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
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
        if self.fail_list_files.load(Ordering::SeqCst) {
            return Err(laurus::LaurusError::storage("injected list_files failure"));
        }
        self.list_files_calls.fetch_add(1, Ordering::SeqCst);
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

fn doc(title: &str) -> Document {
    Document::builder().add_text("title", title).build()
}

/// Write one committed segment holding `ids` onto `storage`.
fn seed_committed_segment(storage: &Arc<dyn Storage>, ids: &[u64]) {
    let mut writer =
        InvertedIndexWriter::new(storage.clone(), InvertedIndexWriterConfig::default()).unwrap();
    for &id in ids {
        writer
            .upsert_document(id, doc(&format!("seed{id}")))
            .unwrap();
    }
    writer.commit().unwrap();
}

/// #559: fresh-id upserts must not rescan segment metadata — the constructor's
/// single recovery scan is the only `list_files` a pure-ingest writer pays.
#[test]
fn fresh_id_upserts_never_rescan_meta_files() {
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn Storage> = counting.clone();
    seed_committed_segment(&storage, &[1, 2, 3]);

    let before_construction = counting.list_files_count();
    let mut writer =
        InvertedIndexWriter::new(storage.clone(), InvertedIndexWriterConfig::default()).unwrap();
    assert_eq!(
        counting.list_files_count(),
        before_construction + 1,
        "the constructor performs exactly one recovery scan"
    );

    let after_construction = counting.list_files_count();
    for id in 10..60u64 {
        writer.upsert_document(id, doc(&format!("t{id}"))).unwrap();
    }
    assert_eq!(
        counting.list_files_count(),
        after_construction,
        "50 fresh-id upserts must not list the storage at all \
         (pre-#864 behavior: one full .meta list+parse per upsert)"
    );
}

/// #559 + #571: overwriting committed docs resolves the segment from the
/// cache (no rescan) and reuses one `DeletionManager` across calls — the
/// first overwrite pays the manager's single bitmap-loading scan, subsequent
/// overwrites pay zero. The deletion itself must still land (`.delmap`
/// exists).
#[test]
fn overwrite_committed_docs_reuses_cache_and_manager() {
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn Storage> = counting.clone();
    seed_committed_segment(&storage, &[1, 2, 3]);

    let mut writer =
        InvertedIndexWriter::new(storage.clone(), InvertedIndexWriterConfig::default()).unwrap();
    let after_construction = counting.list_files_count();

    // First overwrite: the lazily created DeletionManager loads existing
    // bitmaps once (one list_files); the segment lookup itself is cached.
    writer.upsert_document(1, doc("one-v2")).unwrap();
    assert_eq!(
        counting.list_files_count(),
        after_construction + 1,
        "the first overwrite pays exactly the DeletionManager's one-time \
         bitmap-loading scan"
    );

    // Subsequent overwrites: fully served from memory.
    writer.upsert_document(2, doc("two-v2")).unwrap();
    writer.upsert_document(3, doc("three-v2")).unwrap();
    assert_eq!(
        counting.list_files_count(),
        after_construction + 1,
        "subsequent overwrites must reuse the cached manager and ranges \
         (pre-#864: fresh manager + full .delmap reload per overwrite)"
    );

    // The deletions actually landed: the seeded segment now has a bitmap.
    let delmaps: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".delmap"))
        .collect();
    assert_eq!(
        delmaps.len(),
        1,
        "the overwrites must have marked deletions in the seeded segment: {delmaps:?}"
    );
}

/// #864: a segment flushed mid-life by this writer (auto-flush at
/// `max_buffered_docs`) extends the cache in place, so a later overwrite of
/// one of its docs marks the deletion without any rescan.
#[test]
fn mid_life_flush_extends_cache() {
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn Storage> = counting.clone();

    let config = InvertedIndexWriterConfig {
        max_buffered_docs: 2, // force an auto-flush after two upserts
        ..Default::default()
    };
    let mut writer = InvertedIndexWriter::new(storage.clone(), config).unwrap();

    writer.upsert_document(1, doc("alpha")).unwrap();
    writer.upsert_document(2, doc("bravo")).unwrap(); // buffer full -> flush_segment

    // Everything below must be answered from the in-place extended cache.
    // (write_segment_files itself lists once to enumerate the files it wrote;
    // snapshot after the flush.)
    let after_flush = counting.list_files_count();

    // Overwrite a doc that lives in the segment flushed above. The segment
    // lookup must hit the extended cache; only the lazy DeletionManager
    // construction may scan (once).
    writer.upsert_document(1, doc("alpha-v2")).unwrap();
    assert_eq!(
        counting.list_files_count(),
        after_flush + 1,
        "the overwrite must resolve the mid-life flushed segment from the \
         cache, paying only the one-time DeletionManager construction"
    );

    let delmaps: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".delmap"))
        .collect();
    assert_eq!(
        delmaps.len(),
        1,
        "the overwrite must have marked a deletion in the flushed segment: {delmaps:?}"
    );
}

/// #864: `LexicalStore::optimize` force-merges every segment behind a live
/// writer; the invalidation hook must rebuild the writer's cached ranges so a
/// later overwrite marks its deletion in the **merged** segment — with a
/// stale cache it would target a deleted ghost segment and the old version
/// would resurface in search.
#[test]
fn optimize_rebuilds_live_writer_cache() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    let hits = |field: &str, term: &str| -> usize {
        let query = Box::new(TermQuery::new(field, term));
        store
            .search(LexicalSearchRequest::new(query))
            .unwrap()
            .hits
            .len()
    };

    // Two commits -> two segments.
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    store.upsert_document(2, doc("bravo")).unwrap();
    store.commit().unwrap();

    // A live writer with a buffered fresh doc, created BEFORE the merge.
    store.upsert_document(3, doc("charlie")).unwrap();

    // Force-merge both committed segments into one behind the live writer.
    store.optimize().unwrap();

    // Overwrite doc 1 (now living in the merged segment) through the SAME
    // live writer, then commit everything. (Term is hyphen-free on purpose:
    // the tokenizer would split "alpha-v2" into "alpha" + "v2", making the
    // replacement itself match `title:alpha`.)
    store.upsert_document(1, doc("alphav2")).unwrap();
    store.commit().unwrap();

    assert_eq!(
        hits("title", "alpha"),
        0,
        "the pre-merge version must be dead — a stale segment cache would \
         have marked the deletion in a ghost segment and left it alive"
    );
    assert_eq!(hits("title", "alphav2"), 1, "the overwrite must be live");
    assert_eq!(
        hits("title", "bravo"),
        1,
        "untouched doc survives the merge"
    );
    assert_eq!(
        hits("title", "charlie"),
        1,
        "buffered doc survives the merge"
    );

    // The deletion bitmap must belong to the merged segment, not a ghost.
    let delmaps: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".delmap"))
        .collect();
    assert!(
        delmaps.iter().all(|f| f.starts_with("merged_")),
        "deletions must land in the merged segment only: {delmaps:?}"
    );
}

/// #864 review follow-up: `invalidate_segment_cache` must be atomic — when
/// the rebuild scan fails, the OLD (still-valid) cache must survive intact.
/// A clear-then-fail implementation would leave the writer with an empty
/// cache and `max_committed_doc_id == 0`, silently skipping every subsequent
/// overwrite's deletion via the fast path.
#[test]
fn invalidate_failure_preserves_old_cache() {
    let counting = Arc::new(CountingStorage::new());
    let storage: Arc<dyn Storage> = counting.clone();
    seed_committed_segment(&storage, &[1, 2, 3]);

    let mut writer =
        InvertedIndexWriter::new(storage.clone(), InvertedIndexWriterConfig::default()).unwrap();

    // Rebuild fails: the error must propagate...
    counting.set_fail_list_files(true);
    writer
        .invalidate_segment_cache()
        .expect_err("the injected list_files failure must propagate");
    counting.set_fail_list_files(false);

    // ...but the previous cache must still be in force: the segments were
    // not actually replaced (no merge ran), so an overwrite must still
    // resolve the seeded segment and mark its deletion.
    writer.upsert_document(1, doc("one-v2")).unwrap();
    let delmaps: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".delmap"))
        .collect();
    assert_eq!(
        delmaps.len(),
        1,
        "the overwrite must still mark the deletion from the preserved \
         cache — an emptied cache would have skipped it silently: {delmaps:?}"
    );
}
