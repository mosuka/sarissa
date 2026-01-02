//! Write-Ahead Log (WAL) for vector operations.
//!
//! This module provides durability for vector operations by logging them to disk
//! before applying them to the in-memory index. This ensures that data can be
//! recovered in the event of a crash.

use crate::error::Result;
use crate::storage::Storage;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use crate::vector::core::document::DocumentVector;
use std::sync::atomic::{AtomicU64, Ordering};

pub type SeqNumber = u64;

/// A single operation in the Write-Ahead Log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    /// Insert or update a document.
    Upsert {
        doc_id: u64,
        document: DocumentVector,
    },
    /// Delete a document.
    Delete { doc_id: u64 },
}

/// A wrapper for WAL entry with sequence number.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    pub seq: SeqNumber,
    pub entry: WalEntry,
}

/// Manages the Write-Ahead Log.
#[derive(Debug)]
pub struct WalManager {
    storage: Arc<dyn Storage>,
    path: String,
    // We keep a mutex on the writer to ensure atomic appends
    writer: Mutex<Option<Box<dyn crate::storage::StorageOutput>>>,
    next_seq: AtomicU64,
}

impl WalManager {
    /// Create a new WAL manager.
    pub fn new(storage: Arc<dyn Storage>, path: &str) -> Result<Self> {
        let manager = Self {
            storage,
            path: path.to_string(),
            writer: Mutex::new(None),
            next_seq: AtomicU64::new(1),
        };
        Ok(manager)
    }

    /// Open or create the WAL file for appending.
    fn ensure_writer(&self) -> Result<()> {
        let mut writer_guard = self.writer.lock().unwrap();
        if writer_guard.is_none() {
            let writer = self.storage.create_output_append(&self.path)?;
            *writer_guard = Some(writer);
        }
        Ok(())
    }

    /// Set the next sequence number (e.g. after loading snapshot).
    pub fn set_next_seq(&self, seq: SeqNumber) {
        self.next_seq.store(seq, Ordering::SeqCst);
    }

    /// Get the last used sequence number.
    pub fn last_seq(&self) -> SeqNumber {
        self.next_seq.load(Ordering::SeqCst).saturating_sub(1)
    }

    /// Append an entry to the WAL.
    /// Returns the assigned sequence number.
    pub fn append(&self, entry: &WalEntry) -> Result<SeqNumber> {
        self.ensure_writer()?;

        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let record = WalRecord {
            seq,
            entry: entry.clone(),
        };

        // Serialize entry to bytes (using JSON for simplicity for now, could be binary later)
        // Format: [Length: u32][JSON Bytes]
        let bytes = serde_json::to_vec(&record)?;
        let len = bytes.len() as u32;

        let mut writer_guard = self.writer.lock().unwrap();
        if let Some(writer) = writer_guard.as_mut() {
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&bytes)?;
            writer.flush_and_sync()?;
        }

        Ok(seq)
    }

    /// Read all entries from the WAL.
    /// Also updates the internal next_seq to max_seq + 1.
    pub fn read_all(&self) -> Result<Vec<WalRecord>> {
        // If file doesn't exist, return empty
        if !self.storage.file_exists(&self.path) {
            return Ok(Vec::new());
        }

        let mut reader = self.storage.open_input(&self.path)?;
        let mut records = Vec::new();
        let size = reader.size()?;
        let mut position = 0;
        let mut max_seq = 0;

        // Simple loop to read [Length][Data]
        while position < size {
            let mut len_bytes = [0u8; 4];
            // Check if we can read 4 bytes
            if position + 4 > size {
                break; // Incomplete entry or end of file
            }
            reader.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as u64;
            position += 4;

            if position + len > size {
                break; // Incomplete entry
            }

            let mut buffer = vec![0u8; len as usize];
            reader.read_exact(&mut buffer)?;
            position += len;

            let record: WalRecord = serde_json::from_slice(&buffer)?;
            if record.seq > max_seq {
                max_seq = record.seq;
            }
            records.push(record);
        }

        // Update next_seq if we read records with higher sequence
        let current_next = self.next_seq.load(Ordering::SeqCst);
        if max_seq >= current_next {
            self.next_seq.store(max_seq + 1, Ordering::SeqCst);
        }

        Ok(records)
    }

    /// Truncate (clear) the WAL.
    /// This is typically called after a successful flush/checkpoint.
    pub fn truncate(&self) -> Result<()> {
        // Close existing writer if open
        {
            let mut writer_guard = self.writer.lock().unwrap();
            *writer_guard = None;
        }

        // Overwrite with empty file
        let mut writer = self.storage.create_output(&self.path)?;
        writer.flush_and_sync()?;
        // Writer drops/closes here

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::core::document::{DocumentVector, StoredVector};

    #[test]
    fn test_wal_append_read_truncate() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let wal = WalManager::new(storage.clone(), "test.wal").unwrap();

        let mut doc = DocumentVector::new();
        doc.set_field(
            "body",
            StoredVector::new(Arc::<[f32]>::from([1.0, 2.0, 3.0])),
        );

        // 1. Append
        let seq1 = wal
            .append(&WalEntry::Upsert {
                doc_id: 1,
                document: doc.clone(),
            })
            .unwrap();
        assert_eq!(seq1, 1);

        let seq2 = wal.append(&WalEntry::Delete { doc_id: 2 }).unwrap();
        assert_eq!(seq2, 2);

        // 2. Read back
        let records = wal.read_all().unwrap();
        assert_eq!(records.len(), 2);

        assert_eq!(records[0].seq, 1);
        match &records[0].entry {
            WalEntry::Upsert { doc_id, document } => {
                assert_eq!(*doc_id, 1);
                assert_eq!(document.fields.len(), doc.fields.len());
            }
            _ => panic!("Expected Upsert"),
        }

        assert_eq!(records[1].seq, 2);
        match &records[1].entry {
            WalEntry::Delete { doc_id } => {
                assert_eq!(*doc_id, 2);
            }
            _ => panic!("Expected Delete"),
        }

        // 3. Truncate
        wal.truncate().unwrap();
        let records_after = wal.read_all().unwrap();
        assert!(records_after.is_empty());

        // Next seq should continue or be preserved? WalManager read_all updates it.
        // Assuming we want strict monotonicity, we might want to manually set it or trust read_all.
        // If truncated, read_all returns valid next_seq?
        // Ah, read_all only updates if it sees higher records.
        // So after truncate, next_seq remains 3.
        let seq3 = wal.append(&WalEntry::Delete { doc_id: 3 }).unwrap();
        assert_eq!(seq3, 3);
    }
}
