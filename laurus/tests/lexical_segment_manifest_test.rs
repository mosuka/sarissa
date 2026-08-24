//! #1021 PR 1 — the lexical segment manifest is written alongside the
//! `.meta` files, and the two records agree after every mutation class.
//!
//! In this PR segment DISCOVERY still runs on the `.meta` scan, so a
//! manifest content bug would be behaviorally invisible — and PR 2 would
//! then trust that manifest and enforce it with the orphan sweep. This
//! equivalence property suite is what makes such a bug fail loudly here:
//! after every mutation class, `segments.json` must equal the committed
//! `.meta` scan, `has_deletions` and `generation` included.

use std::io::Read;
use std::sync::Arc;

use laurus::lexical::index::inverted::segment::SegmentInfo;
use laurus::lexical::{LexicalIndexConfig, LexicalStore};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{Storage, StorageInput, StorageOutput};
use laurus::{Document, LaurusError, Result as LaurusResult};

/// Storage decorator failing the next `rename_file` whose destination
/// matches the armed `(prefix, suffix)` — the atomic-write publish step of
/// either the manifest (`segments.json`) or an advisory `.meta` flip.
#[derive(Debug)]
struct FailingStorage {
    inner: Arc<dyn Storage>,
    fail_rename_to: parking_lot::Mutex<Option<(String, String)>>,
}

impl FailingStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            fail_rename_to: parking_lot::Mutex::new(None),
        }
    }

    fn fail_next_rename_matching(&self, prefix: &str, suffix: &str) {
        *self.fail_rename_to.lock() = Some((prefix.to_string(), suffix.to_string()));
    }
}

impl Storage for FailingStorage {
    fn create_output(&self, name: &str) -> LaurusResult<Box<dyn StorageOutput>> {
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
        let armed = {
            let mut guard = self.fail_rename_to.lock();
            if guard
                .as_ref()
                .is_some_and(|(p, x)| new_name.starts_with(p) && new_name.ends_with(x))
            {
                *guard = None;
                true
            } else {
                false
            }
        };
        if armed {
            return Err(LaurusError::storage(format!(
                "injected failure renaming {old_name} to {new_name}"
            )));
        }
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

/// Read a file's bytes.
fn read_bytes(storage: &Arc<dyn Storage>, name: &str) -> Vec<u8> {
    let mut input = storage.open_input(name).unwrap();
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes).unwrap();
    bytes
}

/// Parse the checksummed manifest: `varint(len) || json || crc`.
fn read_manifest(storage: &Arc<dyn Storage>) -> Vec<SegmentInfo> {
    let bytes = read_bytes(storage, "segments.json");
    let payload = if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
        value
    } else {
        let mut len: u64 = 0;
        let mut shift = 0;
        let mut cursor = 0usize;
        loop {
            let byte = bytes[cursor];
            cursor += 1;
            len |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        serde_json::from_slice(&bytes[cursor..cursor + len as usize]).unwrap()
    };
    let segments = payload["segments"].as_array().unwrap().clone();
    segments
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect()
}

/// Scan the committed `.meta` files, exactly as discovery does today.
fn scan_committed_metas(storage: &Arc<dyn Storage>) -> Vec<SegmentInfo> {
    let mut segments: Vec<SegmentInfo> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| (f.starts_with("segment_") || f.starts_with("merged_")) && f.ends_with(".meta"))
        .map(|f| serde_json::from_slice::<SegmentInfo>(&read_bytes(storage, &f)).unwrap())
        .filter(|s| s.committed)
        .collect();
    segments.sort_by(|a, b| a.segment_id.cmp(&b.segment_id));
    segments
}

/// The equivalence property: the manifest and the committed-`.meta` scan
/// describe the same segments, field for field.
#[track_caller]
fn assert_equiv(storage: &Arc<dyn Storage>) {
    let mut manifest = read_manifest(storage);
    manifest.sort_by(|a, b| a.segment_id.cmp(&b.segment_id));
    let metas = scan_committed_metas(storage);
    assert_eq!(
        manifest, metas,
        "segments.json must equal the committed .meta scan"
    );
}

fn doc(body: &str) -> Document {
    Document::builder().add_text("body", body).build()
}

fn memory_storage() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
}

/// Plain commits: every commit adds its segment to the manifest.
#[test]
fn manifest_matches_meta_after_plain_commits() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    assert_eq!(
        read_manifest(&storage).len(),
        0,
        "create() must write an empty manifest from birth"
    );

    for round in 1..=2u64 {
        store.upsert_document(round, doc("alpha")).unwrap();
        store.commit().unwrap();
        assert_equiv(&storage);
    }
    assert_eq!(read_manifest(&storage).len(), 2);
}

/// A deleting commit: the `has_deletions` flip reaches the manifest.
#[test]
fn manifest_matches_meta_after_a_deleting_commit() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    // Overwrite the committed doc: marks a deletion in its segment.
    store.upsert_document(1, doc("alpha-v2")).unwrap();
    store.commit().unwrap();

    assert_equiv(&storage);
    assert!(
        read_manifest(&storage).iter().any(|s| s.has_deletions),
        "the deleting commit must flip has_deletions in the manifest"
    );
}

/// The BLOCKER-1 canary: a segment that is deleted-from and published in
/// the SAME commit must enter the manifest with `has_deletions: true`.
///
/// The pending `SegmentInfo` is captured at flush time, before the
/// deletion exists; without the `pending_publish` update in
/// `flush_deletions`, the manifest would say `false` while the `.meta`
/// (read back from disk at flip time) says `true` — and once discovery
/// trusts the manifest, the deleted version would resurrect.
#[test]
fn delete_and_publish_in_one_commit_keeps_has_deletions() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    // Past max_buffered_docs (10_000) so a segment auto-flushes,
    // uncommitted. Doc 1 lands in it.
    for i in 1..=10_050u64 {
        store.upsert_document(i, doc("alpha")).unwrap();
    }
    // Overwrite doc 1: the deletion targets the FLUSHED, unpublished
    // segment.
    store.upsert_document(1, doc("alpha-v2")).unwrap();
    store.commit().unwrap();

    assert_equiv(&storage);
    assert!(
        read_manifest(&storage)
            .iter()
            .any(|s| s.has_deletions && s.segment_id.starts_with("segment_")),
        "the flushed-then-deleted-from segment must carry has_deletions in the manifest"
    );
}

/// Auto-merge: the merge transition (drop sources, add merged with its
/// final generation) reaches the manifest atomically.
#[test]
fn manifest_matches_meta_after_auto_merge() {
    let storage = memory_storage();
    let config = LexicalIndexConfig::builder()
        .max_segments(1)
        .merge_factor(2)
        .build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    for round in 1..=3u64 {
        store.upsert_document(round, doc("alpha")).unwrap();
        store.commit().unwrap();
        assert_equiv(&storage);
    }
}

/// Optimize (force-merge-all): same property on the other merge path.
#[test]
fn manifest_matches_meta_after_optimize() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    for round in 1..=3u64 {
        store.upsert_document(round, doc("alpha")).unwrap();
        store.commit().unwrap();
    }
    store.optimize().unwrap();
    assert_equiv(&storage);
}

/// save-then-swap: a failed manifest save leaves the previous manifest
/// intact, and the retained writer's retry republishes without
/// double-adding.
#[test]
fn failed_manifest_save_retries_cleanly() {
    let inner = memory_storage();
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    let before = read_bytes(&inner, "segments.json");

    store.upsert_document(2, doc("beta")).unwrap();
    failing.fail_next_rename_matching("segments.json", "");
    assert!(
        store.commit().is_err(),
        "the injected manifest-save failure must surface"
    );
    assert_eq!(
        read_bytes(&inner, "segments.json"),
        before,
        "a failed save must leave the previous manifest byte-identical"
    );

    // The retained writer retries (#875) and republishes.
    store.commit().unwrap();
    assert_equiv(&inner);
    let manifest = read_manifest(&inner);
    let mut ids: Vec<&str> = manifest.iter().map(|s| s.segment_id.as_str()).collect();
    ids.sort_unstable();
    let unique = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), unique, "a retried publish must not double-add");
    assert_eq!(manifest.len(), 2);
}

/// The `mem::take` partial-publish regression: an advisory `.meta` flip
/// failing mid-publish must not drop the unflipped segments — the retry
/// publishes every pending segment.
#[test]
fn partial_meta_flip_failure_republishes_everything() {
    let inner = memory_storage();
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    // Two pending segments in one commit: one auto-flushed, one flushed by
    // the commit itself.
    for i in 1..=10_050u64 {
        store.upsert_document(i, doc("alpha")).unwrap();
    }
    // Fail the first advisory `.meta` publish flip (a rename onto a
    // `segment_*.meta`); the manifest save (`segments.json`) has already
    // succeeded by then.
    failing.fail_next_rename_matching("segment_", ".meta");
    assert!(
        store.commit().is_err(),
        "the injected .meta flip failure must surface"
    );

    // Retry: everything is republished — manifest and .meta agree and
    // every segment is committed.
    store.commit().unwrap();
    assert_equiv(&inner);
    assert_eq!(
        read_manifest(&inner).len(),
        2,
        "both pending segments must be published after the retry"
    );
}

/// Legacy migration: an index without a manifest opens through the `.meta`
/// scan, and the first mutation persists the migrated manifest.
#[test]
fn legacy_index_without_manifest_migrates_on_first_mutation() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    drop(store);

    // Simulate a pre-manifest index.
    storage.delete_file("segments.json").unwrap();

    let reopened = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    reopened.upsert_document(2, doc("beta")).unwrap();
    reopened.commit().unwrap();

    assert_equiv(&storage);
    assert_eq!(
        read_manifest(&storage).len(),
        2,
        "the migrated manifest must contain the legacy segment AND the new one"
    );
}
