//! Write-Ahead Log (WAL) for vector operations.
//!
//! This module provides durability for vector operations by logging them to disk
//! before applying them to the in-memory index. This ensures that data can be
//! recovered in the event of a crash.

use crate::error::Result;
use crate::storage::Storage;
use crate::vector::core::vector::Vector;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

/// A single operation in the Write-Ahead Log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    /// Insert or update a vector.
    Insert { doc_id: u64, vector: Vector },
    /// Delete a document.
    Delete { doc_id: u64 },
}

/// Manages the Write-Ahead Log.
#[derive(Debug)]
pub struct WalManager {
    storage: Arc<dyn Storage>,
    path: String,
    // We keep a mutex on the writer to ensure atomic appends
    writer: Mutex<Option<Box<dyn crate::storage::StorageOutput>>>,
}

impl WalManager {
    /// Create a new WAL manager.
    pub fn new(storage: Arc<dyn Storage>, path: &str) -> Result<Self> {
        let manager = Self {
            storage,
            path: path.to_string(),
            writer: Mutex::new(None),
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

    /// Append an entry to the WAL.
    pub fn append(&self, entry: &WalEntry) -> Result<()> {
        self.ensure_writer()?;

        // Serialize entry to bytes (using JSON for simplicity for now, could be binary later)
        // Format: [Length: u32][JSON Bytes]
        let bytes = serde_json::to_vec(entry)?;
        let len = bytes.len() as u32;

        let mut writer_guard = self.writer.lock().unwrap();
        if let Some(writer) = writer_guard.as_mut() {
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&bytes)?;
            writer.flush_and_sync()?;
        }

        Ok(())
    }

    /// Read all entries from the WAL.
    pub fn read_all(&self) -> Result<Vec<WalEntry>> {
        // If file doesn't exist, return empty
        if !self.storage.file_exists(&self.path) {
            return Ok(Vec::new());
        }

        let mut reader = self.storage.open_input(&self.path)?;
        let mut entries = Vec::new();
        let size = reader.size()?;
        let mut position = 0;

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

            let entry: WalEntry = serde_json::from_slice(&buffer)?;
            entries.push(entry);
        }

        Ok(entries)
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
    use std::collections::HashMap;

    #[test]
    fn test_wal_append_read_truncate() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let wal = WalManager::new(storage.clone(), "test.wal").unwrap();

        let vector = Vector {
            data: vec![1.0, 2.0, 3.0],
            metadata: HashMap::new(),
        };

        // 1. Append
        wal.append(&WalEntry::Insert {
            doc_id: 1,
            vector: vector.clone(),
        })
        .unwrap();

        wal.append(&WalEntry::Delete { doc_id: 2 }).unwrap();

        // 2. Read back
        let entries = wal.read_all().unwrap();
        assert_eq!(entries.len(), 2);

        match &entries[0] {
            WalEntry::Insert { doc_id, vector: v } => {
                assert_eq!(*doc_id, 1);
                assert_eq!(v.data, vector.data);
            }
            _ => panic!("Expected Insert"),
        }

        match &entries[1] {
            WalEntry::Delete { doc_id } => {
                assert_eq!(*doc_id, 2);
            }
            _ => panic!("Expected Delete"),
        }

        // 3. Truncate
        wal.truncate().unwrap();
        let entries_after = wal.read_all().unwrap();
        assert!(entries_after.is_empty());
    }
}
