//! Unified document log combining WAL, doc_id generation, and document storage.
//!
//! [`DocumentLog`] provides a single component that:
//!
//! - Generates monotonically increasing document IDs
//! - Writes all operations to a durable append-only log (WAL)
//! - Stores documents in segmented files for retrieval
//! - Supports recovery by replaying the log
//!
//! ## Architecture
//!
//! ```text
//! DocumentLog
//! ├── WAL (append-only log file)
//! │   └── All fields stored for recovery
//! └── Document Store (segmented files)
//!     └── Only stored fields kept for retrieval
//! ```
//!
//! ## File format
//!
//! The WAL log file stores records in a simple binary format:
//! `[u32: length][json: LogRecord]` repeated for each entry.
//! Each entry is followed by `flush_and_sync()` for durability.
//!
//! ## Recovery
//!
//! [`DocumentLog::read_all`] replays the file and stops at the first record it
//! cannot read in full — a short length prefix, a short body, or a body that
//! fails to deserialize. Such a torn trailing record (e.g. from a crash
//! mid-append) is dropped along with everything after it, and the valid prefix
//! is recovered so the engine can still open. Recovery never skips a bad record
//! to continue, which keeps the recovered sequence gap-free and consistent.

use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::data::Document;
use crate::error::Result;
use crate::storage::Storage;
use crate::store::document::UnifiedDocumentStore;

/// Sequence number for log entries.
pub type SeqNumber = u64;

/// A single operation in the document log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogEntry {
    /// Insert or update a document.
    Upsert {
        doc_id: u64,
        external_id: String,
        document: Document,
    },
    /// Delete a document.
    Delete {
        doc_id: u64,
        /// External ID of the deleted document.
        /// Uses `#[serde(default)]` for backward compatibility with old WAL
        /// entries that lack this field.
        #[serde(default)]
        external_id: String,
    },
}

/// A log record combining a sequence number with an entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRecord {
    pub seq: SeqNumber,
    pub entry: LogEntry,
}

/// Unified document log providing WAL, doc_id generation, and document storage.
///
/// This component replaces the separate `WalManager` and `UnifiedDocumentStore`,
/// combining:
/// - **WAL**: durable append-only log for crash recovery
/// - **doc_id generation**: monotonically increasing document IDs
/// - **Document storage**: segmented files for stored-field retrieval
///
/// # Thread safety
///
/// WAL writes are serialized through an internal [`Mutex`].
/// Document store access uses [`parking_lot::RwLock`] for concurrent reads.
/// The `next_doc_id` and `next_seq` counters use [`AtomicU64`] for lock-free reads.
#[derive(Debug)]
pub struct DocumentLog {
    wal_storage: Arc<dyn Storage>,
    wal_path: String,
    next_doc_id: AtomicU64,
    wal_writer: Mutex<Option<Box<dyn crate::storage::StorageOutput>>>,
    next_seq: AtomicU64,
    doc_store: RwLock<UnifiedDocumentStore>,
}

impl DocumentLog {
    /// Create a new document log with WAL and document storage.
    ///
    /// The internal `next_doc_id` counter starts at **1** and is **not**
    /// automatically recovered from any existing WAL file or document store.
    /// Callers must invoke [`read_all`](Self::read_all) after construction to
    /// replay the WAL and synchronize both the `next_doc_id` and `next_seq`
    /// counters with the persisted state. Failing to do so may cause
    /// duplicate document IDs.
    ///
    /// # Arguments
    ///
    /// * `wal_storage` - Storage backend for the WAL file.
    /// * `wal_path` - Path (relative to the storage root) of the WAL file.
    /// * `doc_store_storage` - Storage backend for the document store.
    ///
    /// # Errors
    ///
    /// Returns an error if the document store cannot be opened.
    pub fn new(
        wal_storage: Arc<dyn Storage>,
        wal_path: &str,
        doc_store_storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        let doc_store = UnifiedDocumentStore::open(doc_store_storage)?;
        Ok(Self {
            wal_storage,
            wal_path: wal_path.to_string(),
            next_doc_id: AtomicU64::new(1),
            wal_writer: Mutex::new(None),
            next_seq: AtomicU64::new(1),
            doc_store: RwLock::new(doc_store),
        })
    }

    /// Open or create the WAL file for appending.
    fn ensure_writer(&self) -> Result<()> {
        let mut writer_guard = self.wal_writer.lock();
        if writer_guard.is_none() {
            let writer = self.wal_storage.create_output_append(&self.wal_path)?;
            *writer_guard = Some(writer);
        }
        Ok(())
    }

    // ── WAL operations ──────────────────────────────────────────────

    /// Append an upsert entry to the log.
    ///
    /// Atomically assigns a new doc_id and sequence number, then writes
    /// the entry to the log file with fsync.
    ///
    /// Returns `(doc_id, seq_number)`.
    pub fn append(&self, external_id: &str, doc: Document) -> Result<(u64, SeqNumber)> {
        self.ensure_writer()?;

        let mut writer_guard = self.wal_writer.lock();

        let doc_id = self.next_doc_id.fetch_add(1, Ordering::SeqCst);
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);

        let record = LogRecord {
            seq,
            entry: LogEntry::Upsert {
                doc_id,
                external_id: external_id.to_string(),
                document: doc,
            },
        };

        Self::write_record(&mut writer_guard, &record)?;

        Ok((doc_id, seq))
    }

    /// Append a delete entry to the log.
    ///
    /// Returns the assigned sequence number.
    pub fn append_delete(&self, doc_id: u64, external_id: &str) -> Result<SeqNumber> {
        self.ensure_writer()?;

        let mut writer_guard = self.wal_writer.lock();

        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);

        let record = LogRecord {
            seq,
            entry: LogEntry::Delete {
                doc_id,
                external_id: external_id.to_string(),
            },
        };

        Self::write_record(&mut writer_guard, &record)?;

        Ok(seq)
    }

    /// Write a single record to the WAL file.
    fn write_record(
        writer_guard: &mut Option<Box<dyn crate::storage::StorageOutput>>,
        record: &LogRecord,
    ) -> Result<()> {
        let bytes = serde_json::to_vec(record)?;
        let len: u32 = bytes.len().try_into().map_err(|_| {
            crate::error::LaurusError::InvalidOperation(format!(
                "WAL record size {} exceeds u32::MAX",
                bytes.len()
            ))
        })?;

        if let Some(writer) = writer_guard.as_mut() {
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&bytes)?;
            writer.flush_and_sync()?;
        }

        Ok(())
    }

    /// Read all records from the WAL.
    ///
    /// Also updates internal counters (`next_seq`, `next_doc_id`) to be
    /// greater than the maximum values found in the log, and syncs
    /// `next_doc_id` with the committed document store segments.
    pub fn read_all(&self) -> Result<Vec<LogRecord>> {
        if !self.wal_storage.file_exists(&self.wal_path) {
            // Even with an empty WAL, sync next_doc_id with doc_store.
            let store_next = self.doc_store.read().next_doc_id();
            self.set_next_doc_id(store_next);
            return Ok(Vec::new());
        }

        let mut reader = self.wal_storage.open_input(&self.wal_path)?;
        let mut records = Vec::new();
        let size = reader.size()?;
        let mut position = 0;
        let mut max_seq: u64 = 0;
        let mut max_doc_id: u64 = 0;

        while position < size {
            let mut len_bytes = [0u8; 4];
            if position + 4 > size {
                break;
            }
            reader.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as u64;
            position += 4;

            if position + len > size {
                break;
            }

            let mut buffer = vec![0u8; len as usize];
            reader.read_exact(&mut buffer)?;
            position += len;

            // A trailing record with an intact length prefix but a corrupt or
            // partially written body indicates a torn write (e.g. a crash
            // mid-append). Stop at the last valid record and recover the durable
            // prefix instead of aborting recovery — this mirrors the short-read
            // breaks above and lets the engine open. Recovery stops at the FIRST
            // bad record (never skip-and-continue), so a later intact-looking
            // record can never resurrect an op whose predecessors were dropped.
            // (Issue #542, Phase 0.)
            let record: LogRecord = match serde_json::from_slice(&buffer) {
                Ok(record) => record,
                Err(e) => {
                    ::log::warn!(
                        "WAL recovery: dropping corrupt trailing record at byte offset {} \
                         ({len} body bytes): {e}; recovered {} valid record(s) before it",
                        position - len,
                        records.len()
                    );
                    break;
                }
            };
            if record.seq > max_seq {
                max_seq = record.seq;
            }
            if let LogEntry::Upsert { doc_id, .. } = &record.entry
                && *doc_id > max_doc_id
            {
                max_doc_id = *doc_id;
            }
            records.push(record);
        }

        // Update counters to continue from the highest values found.
        let current_next_seq = self.next_seq.load(Ordering::SeqCst);
        if max_seq >= current_next_seq {
            self.next_seq.store(max_seq + 1, Ordering::SeqCst);
        }
        let current_next_doc = self.next_doc_id.load(Ordering::SeqCst);
        if max_doc_id >= current_next_doc {
            self.next_doc_id.store(max_doc_id + 1, Ordering::SeqCst);
        }

        // Sync next_doc_id with committed doc_store segments.
        let store_next = self.doc_store.read().next_doc_id();
        self.set_next_doc_id(store_next);

        Ok(records)
    }

    /// Truncate (clear) the WAL.
    ///
    /// Called after a successful commit to discard processed entries.
    pub fn truncate(&self) -> Result<()> {
        {
            let mut writer_guard = self.wal_writer.lock();
            *writer_guard = None;
        }

        let mut writer = self.wal_storage.create_output(&self.wal_path)?;
        writer.flush_and_sync()?;
        writer.close()?;

        // Sync storage to ensure the truncated WAL file is visible.
        // Critical on Windows where file metadata may be cached.
        self.wal_storage.sync()?;

        Ok(())
    }

    /// Get the last used sequence number.
    pub fn last_seq(&self) -> SeqNumber {
        self.next_seq.load(Ordering::SeqCst).saturating_sub(1)
    }

    /// Get the current next_doc_id value.
    pub fn next_doc_id(&self) -> u64 {
        self.next_doc_id.load(Ordering::SeqCst)
    }

    /// Set the next_doc_id if the given value is higher than the current one.
    pub fn set_next_doc_id(&self, id: u64) {
        let current = self.next_doc_id.load(Ordering::SeqCst);
        if id > current {
            self.next_doc_id.store(id, Ordering::SeqCst);
        }
    }

    // ── Document store operations ───────────────────────────────────

    /// Store a document with a specific doc_id.
    ///
    /// This stores the document in the segmented document store for later
    /// retrieval. The document should already have non-stored fields
    /// filtered out.
    pub fn store_document(&self, doc_id: u64, doc: Document) {
        self.doc_store.write().put_document_with_id(doc_id, doc);
    }

    /// Get a document by its internal doc_id.
    pub fn get_document(&self, doc_id: u64) -> Result<Option<Document>> {
        self.doc_store.read().get_document(doc_id)
    }

    /// Retrieve multiple documents by their internal IDs in a single batch.
    ///
    /// More efficient than individual [`get_document()`](Self::get_document) calls
    /// because each segment file is opened and scanned only once.
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - Slice of internal document IDs to retrieve.
    ///
    /// # Returns
    ///
    /// A map of doc_id to [`Document`] for all found documents.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on storage I/O or deserialization failure.
    pub fn get_documents_batch(
        &self,
        doc_ids: &[u64],
    ) -> Result<std::collections::HashMap<u64, Document>> {
        self.doc_store.read().get_documents_batch(doc_ids)
    }

    /// Find internal doc_id by external ID.
    pub fn find_by_external_id(&self, external_id: &str) -> Result<Option<u64>> {
        self.doc_store.read().find_by_external_id(external_id)
    }

    /// Find all internal doc_ids by external ID.
    pub fn find_all_by_external_id(&self, external_id: &str) -> Result<Vec<u64>> {
        self.doc_store.read().find_all_by_external_id(external_id)
    }

    /// Commit the document store (flush pending docs to segments).
    pub fn commit_documents(&self) -> Result<()> {
        self.doc_store.write().commit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataValue, Document};
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    fn make_storage() -> Arc<dyn Storage> {
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
    }

    fn make_log() -> DocumentLog {
        let wal_storage = make_storage();
        let doc_storage = make_storage();
        DocumentLog::new(wal_storage, "test.log", doc_storage).unwrap()
    }

    #[test]
    fn test_append_and_read() {
        let log = make_log();

        let doc = Document::builder()
            .add_field("body", DataValue::Text("hello".to_string()))
            .build();

        // Append upsert.
        let (doc_id, seq1) = log.append("ext_1", doc.clone()).unwrap();
        assert_eq!(doc_id, 1);
        assert_eq!(seq1, 1);

        // Append delete.
        let seq2 = log.append_delete(doc_id, "ext_1").unwrap();
        assert_eq!(seq2, 2);

        // Read all.
        let records = log.read_all().unwrap();
        assert_eq!(records.len(), 2);

        assert_eq!(records[0].seq, 1);
        match &records[0].entry {
            LogEntry::Upsert {
                doc_id,
                external_id,
                ..
            } => {
                assert_eq!(*doc_id, 1);
                assert_eq!(external_id, "ext_1");
            }
            _ => panic!("Expected Upsert"),
        }

        assert_eq!(records[1].seq, 2);
        match &records[1].entry {
            LogEntry::Delete {
                doc_id,
                external_id,
            } => {
                assert_eq!(*doc_id, 1);
                assert_eq!(external_id, "ext_1");
            }
            _ => panic!("Expected Delete"),
        }
    }

    #[test]
    fn test_truncate() {
        let log = make_log();

        let doc = Document::builder()
            .add_field("body", DataValue::Text("hello".to_string()))
            .build();

        log.append("ext_1", doc).unwrap();
        log.truncate().unwrap();

        let records = log.read_all().unwrap();
        assert!(records.is_empty());

        // Sequence and doc_id should continue monotonically.
        let doc2 = Document::builder()
            .add_field("body", DataValue::Text("world".to_string()))
            .build();
        let (doc_id, seq) = log.append("ext_2", doc2).unwrap();
        assert_eq!(doc_id, 2);
        assert_eq!(seq, 2);
    }

    #[test]
    fn test_doc_id_recovery() {
        let wal_storage = make_storage();
        let doc_storage = make_storage();

        // Write some entries.
        {
            let log =
                DocumentLog::new(wal_storage.clone(), "test.log", doc_storage.clone()).unwrap();
            let doc = Document::builder()
                .add_field("body", DataValue::Text("hello".to_string()))
                .build();
            log.append("ext_1", doc.clone()).unwrap();
            log.append("ext_2", doc).unwrap();
        }

        // Reopen and verify counters are restored.
        {
            let log =
                DocumentLog::new(wal_storage.clone(), "test.log", doc_storage.clone()).unwrap();
            let records = log.read_all().unwrap();
            assert_eq!(records.len(), 2);
            assert_eq!(log.next_doc_id(), 3); // max doc_id was 2

            let doc = Document::builder()
                .add_field("body", DataValue::Text("world".to_string()))
                .build();
            let (doc_id, seq) = log.append("ext_3", doc).unwrap();
            assert_eq!(doc_id, 3);
            assert_eq!(seq, 3);
        }
    }

    /// Write one framed record manually, returning its `[u32 len][body]` bytes.
    fn frame(body: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + body.len());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        out
    }

    /// A trailing record with an intact length prefix but a corrupt body (e.g. a
    /// crash mid-append) must NOT abort recovery: `read_all` recovers the valid
    /// prefix, drops the torn record, and resyncs counters from the prefix only
    /// (Issue #542, Phase 0).
    #[test]
    fn read_all_recovers_prefix_when_trailing_record_body_is_corrupt() {
        use std::io::Write as _;

        let wal_storage = make_storage();
        let doc_storage = make_storage();

        // One valid record followed by a frame whose length prefix is intact but
        // whose body is not valid JSON.
        let valid = LogRecord {
            seq: 1,
            entry: LogEntry::Upsert {
                doc_id: 1,
                external_id: "ext_1".to_string(),
                document: Document::builder()
                    .add_field("body", DataValue::Text("hello".to_string()))
                    .build(),
            },
        };
        {
            let mut out = wal_storage.create_output("test.log").unwrap();
            out.write_all(&frame(&serde_json::to_vec(&valid).unwrap()))
                .unwrap();
            out.write_all(&frame(b"this is not valid json")).unwrap();
            out.flush_and_sync().unwrap();
            out.close().unwrap();
        }

        let log = DocumentLog::new(wal_storage, "test.log", doc_storage).unwrap();

        // Recovery must succeed (not error) and yield only the valid prefix.
        let records = log.read_all().unwrap();
        assert_eq!(records.len(), 1, "only the valid prefix is recovered");
        assert_eq!(records[0].seq, 1);

        // Counters are resynced from the recovered prefix only: the next append
        // continues monotonically without a gap.
        assert_eq!(log.last_seq(), 1);
        assert_eq!(log.next_doc_id(), 2);
        let doc = Document::builder()
            .add_field("body", DataValue::Text("next".to_string()))
            .build();
        let (doc_id, seq) = log.append("ext_2", doc).unwrap();
        assert_eq!(doc_id, 2);
        assert_eq!(seq, 2);
    }

    /// An empty/garbage-only WAL (no valid leading record) recovers nothing
    /// rather than erroring.
    #[test]
    fn read_all_recovers_nothing_when_first_record_is_corrupt() {
        use std::io::Write as _;

        let wal_storage = make_storage();
        let doc_storage = make_storage();
        {
            let mut out = wal_storage.create_output("test.log").unwrap();
            out.write_all(&frame(b"garbage")).unwrap();
            out.flush_and_sync().unwrap();
            out.close().unwrap();
        }

        let log = DocumentLog::new(wal_storage, "test.log", doc_storage).unwrap();
        let records = log.read_all().unwrap();
        assert!(records.is_empty(), "no valid prefix to recover");
    }

    #[test]
    fn test_set_next_doc_id() {
        let log = make_log();

        // Sync with a higher doc_id from document store.
        log.set_next_doc_id(100);
        assert_eq!(log.next_doc_id(), 100);

        // Setting a lower value should be ignored.
        log.set_next_doc_id(50);
        assert_eq!(log.next_doc_id(), 100);

        // Append should use the higher value.
        let doc = Document::builder()
            .add_field("body", DataValue::Text("hello".to_string()))
            .build();
        let (doc_id, _) = log.append("ext_1", doc).unwrap();
        assert_eq!(doc_id, 100);
    }

    #[test]
    fn test_store_and_get_document() {
        let log = make_log();

        let doc = Document::builder()
            .add_field("body", DataValue::Text("hello world".to_string()))
            .build();

        // Store document.
        log.store_document(1, doc.clone());

        // Retrieve from pending.
        let retrieved = log.get_document(1).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.unwrap().fields.get("body"),
            doc.fields.get("body")
        );

        // After commit, retrieve from segment.
        log.commit_documents().unwrap();
        let retrieved = log.get_document(1).unwrap();
        assert!(retrieved.is_some());
    }
}
