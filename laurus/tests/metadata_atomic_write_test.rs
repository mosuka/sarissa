//! Durability of the `metadata.json` control files (#1023).
//!
//! Three subsystems wrote theirs with a plain `create_output`, which
//! truncates in place. The payload is a few hundred bytes and the writer is
//! buffered, so nothing reaches the OS until `close` — meaning the torn state
//! is a **zero-byte file** for the whole window, not half-written JSON.
//!
//! The lexical one is the worst of the three: `InvertedIndex::open` requires
//! it, so losing it takes the entire engine down — vector and document data
//! included — with nothing to recreate it.
//!
//! These tests pin two properties per subsystem: a crash mid-write leaves the
//! previous file intact and readable, and a corrupted file is refused rather
//! than read as though it were valid.

use std::io::{Read, Write};
use std::sync::Arc;

use laurus::Document;
use laurus::lexical::{LexicalIndexConfig, LexicalStore};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{Storage, StorageInput, StorageOutput};
use laurus::{LaurusError, Result as LaurusResult};

/// Storage decorator that fails the next rename onto a named file.
///
/// The crash-injection harnesses already in this repository intercept
/// creating operations but pass `rename_file` straight through — and the
/// rename is precisely the step an atomic write turns on, so none of them can
/// exercise it.
#[derive(Debug)]
struct FailingStorage {
    inner: Arc<dyn Storage>,
    /// Destination name whose next `rename_file` should fail.
    fail_rename_to: parking_lot::Mutex<Option<String>>,
}

impl FailingStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            fail_rename_to: parking_lot::Mutex::new(None),
        }
    }

    /// Arm a one-shot failure for the next `rename_file` **onto** `name`.
    ///
    /// Targeted by destination rather than blanket: segment `.meta` files are
    /// also published by rename, so a blanket failure would abort the commit
    /// before it ever reached the metadata write — and the test would pass
    /// without exercising anything.
    fn fail_next_rename_to(&self, name: &str) {
        *self.fail_rename_to.lock() = Some(name.to_string());
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
            if guard.as_deref() == Some(new_name) {
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

/// A document with one text field.
fn doc(body: &str) -> Document {
    Document::builder().add_text("body", body).build()
}

/// Build a committed lexical store over `storage`.
fn seeded_lexical_store(storage: Arc<dyn Storage>) -> LexicalStore {
    let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    store
}

/// A crash while writing lexical `metadata.json` must leave the previous one
/// intact, so the index still opens.
///
/// Without an atomic write the file is truncated at `create_output` and the
/// contents never arrive, leaving zero bytes — and `InvertedIndex::open`
/// hard-errors on that, taking the whole engine with it.
#[test]
fn lexical_metadata_survives_a_failed_write() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner));
    let storage: Arc<dyn Storage> = failing.clone();

    let store = seeded_lexical_store(storage.clone());
    let before = read_bytes(&storage, "metadata.json");
    assert!(!before.is_empty(), "the seed commit must have written it");

    // Fail the rename that publishes the next write. The previous file must
    // be untouched.
    failing.fail_next_rename_to("metadata.json");
    store.upsert_document(2, doc("beta")).unwrap();
    let _ = store.commit();

    let after = read_bytes(&storage, "metadata.json");
    assert_eq!(
        after, before,
        "a failed publish must leave the previous metadata byte-identical"
    );

    // And the index still opens over it.
    drop(store);
    LexicalStore::new(storage, LexicalIndexConfig::default())
        .expect("the index must still open after a failed metadata write");
}

/// A corrupted lexical `metadata.json` must be refused, not read as valid.
#[test]
fn lexical_metadata_corruption_is_refused() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = seeded_lexical_store(storage.clone());
    drop(store);

    // Flip a byte inside the payload, leaving the framing intact.
    let mut bytes = read_bytes(&storage, "metadata.json");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    let mut output = storage.create_output("metadata.json").unwrap();
    output.write_all(&bytes).unwrap();
    output.close().unwrap();

    let err = LexicalStore::new(storage, LexicalIndexConfig::default())
        .expect_err("a corrupted metadata.json must not be accepted");
    assert!(
        err.to_string().contains("checksum"),
        "corruption must be reported as such, got: {err}"
    );
}

/// An index written before the checksummed framing existed must keep opening.
///
/// This is the compatibility guarantee: no existing index needs rewriting.
#[test]
fn lexical_legacy_metadata_still_opens() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = seeded_lexical_store(storage.clone());
    drop(store);

    // Rewrite it in the pre-framing form: bare JSON in place. The current
    // file may already be raw (before this change) or framed (after), so
    // derive the payload either way rather than assuming one.
    let json = raw_json_payload(&read_bytes(&storage, "metadata.json"));
    let mut output = storage.create_output("metadata.json").unwrap();
    output.write_all(&json).unwrap();
    output.close().unwrap();

    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default())
        .expect("a legacy raw-JSON metadata.json must still open");

    // And the next commit upgrades it to the framed form.
    store.upsert_document(2, doc("beta")).unwrap();
    store.commit().unwrap();
    let upgraded = read_bytes(&storage, "metadata.json");
    assert_ne!(upgraded, json, "the next commit must rewrite it framed");
    LexicalStore::new(storage, LexicalIndexConfig::default())
        .expect("the upgraded file must open too");
}

/// Return the bare-JSON payload of `content`, whether it is already raw or
/// wrapped in the `varint(len) || json || crc` framing.
fn raw_json_payload(content: &[u8]) -> Vec<u8> {
    if serde_json::from_slice::<serde_json::Value>(content).is_ok() {
        return content.to_vec();
    }

    let mut len: u64 = 0;
    let mut shift = 0;
    let mut cursor = 0usize;
    loop {
        let byte = content[cursor];
        cursor += 1;
        len |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    content[cursor..cursor + len as usize].to_vec()
}

/// The index-side writer must be atomic too.
///
/// Two code paths write this file: the writer's, on every commit, and
/// `InvertedIndex::write_metadata`, reached through `create` and through
/// `optimize`. Covering only the first would leave the second free to
/// truncate the file in place — so this drives `optimize()` specifically.
#[test]
fn index_side_metadata_write_is_atomic_too() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner));
    let storage: Arc<dyn Storage> = failing.clone();

    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    // Two committed segments so the force-merge has work to do.
    for i in 0..2u64 {
        store.upsert_document(i + 1, doc("alpha")).unwrap();
        store.commit().unwrap();
    }
    let before = read_bytes(&storage, "metadata.json");

    // `optimize` ends in `InvertedIndex::update_metadata`, i.e. the index-side
    // write. Fail its publish and the previous file must survive.
    failing.fail_next_rename_to("metadata.json");
    let _ = store.optimize();

    let after = read_bytes(&storage, "metadata.json");
    assert_eq!(
        after, before,
        "a failed index-side publish must leave the previous metadata intact"
    );

    drop(store);
    LexicalStore::new(storage, LexicalIndexConfig::default())
        .expect("the index must still open after a failed index-side write");
}

/// A successful commit must not leave the staging file behind.
#[test]
fn no_temp_file_survives_a_commit() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = seeded_lexical_store(storage.clone());
    store.upsert_document(2, doc("beta")).unwrap();
    store.commit().unwrap();
    drop(store);

    let leftovers: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "staging files must not survive: {leftovers:?}"
    );
}
