//! In-memory storage implementation for testing and caching.
//!
//! This module provides a complete [`Storage`] implementation backed entirely
//! by in-process memory (`HashMap<String, Box<[u8]>>`). It is designed for:
//!
//! - **Unit and integration testing** -- fast, deterministic, no filesystem
//!   side effects.
//! - **Temporary indexes** -- building an ephemeral index that does not need
//!   to survive process restarts.
//! - **Benchmarking** -- removing disk I/O from the critical path so that
//!   algorithmic performance can be measured in isolation.
//!
//! ## Thread Safety
//!
//! The file map is protected by a [`parking_lot::RwLock`] so that concurrent
//! readers do not block each other -- important because reads vastly outnumber
//! writes during search. [`MemoryStorage`] can be shared across threads via
//! `Arc<MemoryStorage>`. Individual I/O handles ([`MemoryInput`] /
//! [`MemoryOutput`]) are **not** `Sync` but are safe to send between threads.
//!
//! ## When to Use `MemoryStorage` vs `FileStorage`
//!
//! | Criterion | `MemoryStorage` | `FileStorage` |
//! |---|---|---|
//! | Persistence | None (lost on drop) | Durable on disk |
//! | Performance | Very fast (no syscalls) | Disk-bound |
//! | Capacity | Limited by process memory | Limited by disk |
//! | Use case | Tests, benchmarks, transient data | Production indexes |

use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use parking_lot::RwLock;

use crate::error::{LaurusError, Result};
use crate::storage::{
    LockManager, Storage, StorageError, StorageInput, StorageLock, StorageOutput,
};

/// Configuration specific to memory-based storage.
///
/// Allows callers to tune the initial capacity of the internal file map,
/// reducing re-allocations when the approximate number of files is known
/// ahead of time.
#[derive(Debug, Clone)]
pub struct MemoryStorageConfig {
    /// Initial capacity hint for the file map.
    pub initial_capacity: usize,
}

impl Default for MemoryStorageConfig {
    fn default() -> Self {
        MemoryStorageConfig {
            initial_capacity: 16,
        }
    }
}

/// An in-memory storage implementation.
///
/// All files are held in a shared `HashMap<String, Box<[u8]>>` guarded by a
/// [`parking_lot::RwLock`]. `Box<[u8]>` is used instead of `Vec<u8>` for
/// finalized files to avoid wasting memory on unused capacity. The `RwLock`
/// allows multiple concurrent readers without blocking, which is important
/// because reads vastly outnumber writes during search operations.
///
/// This is useful for testing and for creating temporary indexes in memory.
#[derive(Debug)]
pub struct MemoryStorage {
    /// The files stored in memory with optimized memory layout.
    files: Arc<RwLock<HashMap<String, Box<[u8]>>>>,
    /// Lock manager for coordinating access.
    lock_manager: Arc<MemoryLockManager>,
    /// Storage configuration.
    #[allow(dead_code)]
    config: MemoryStorageConfig,
    /// Whether the storage is closed.
    closed: bool,
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new(MemoryStorageConfig::default())
    }
}

impl MemoryStorage {
    /// Create a new memory storage with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration controlling initial capacity and other
    ///   tuning parameters.
    ///
    /// # Returns
    ///
    /// A new, empty `MemoryStorage`.
    pub fn new(config: MemoryStorageConfig) -> Self {
        let initial_capacity = config.initial_capacity;
        MemoryStorage {
            files: Arc::new(RwLock::new(HashMap::with_capacity(initial_capacity))),
            lock_manager: Arc::new(MemoryLockManager::new()),
            config,
            closed: false,
        }
    }

    /// Check if the storage is closed, returning an error if so.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::StorageClosed`] when the storage has been
    /// closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed {
            Err(StorageError::StorageClosed.into())
        } else {
            Ok(())
        }
    }

    /// Get the number of files currently stored.
    ///
    /// # Returns
    ///
    /// The file count.
    #[inline]
    pub fn file_count(&self) -> usize {
        self.files.read().len()
    }

    /// Get the total size of all files in bytes.
    ///
    /// # Returns
    ///
    /// The sum of all file sizes.
    pub fn total_size(&self) -> u64 {
        let files = self.files.read();
        files.values().map(|data| data.len() as u64).sum()
    }

    /// Remove all files from storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage is closed.
    pub fn clear(&self) -> Result<()> {
        self.check_closed()?;
        let mut files = self.files.write();
        files.clear();
        Ok(())
    }
}

impl Storage for MemoryStorage {
    #[inline]
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.check_closed()?;

        // Use read lock: concurrent readers do not block each other.
        let files = self.files.read();
        let data = files
            .get(name)
            .ok_or_else(|| StorageError::FileNotFound(name.to_string()))?;

        Ok(Box::new(MemoryInput::new(data.clone())))
    }

    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.check_closed()?;

        Ok(Box::new(MemoryOutput::new(
            name.to_string(),
            Arc::clone(&self.files),
        )))
    }

    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.check_closed()?;

        // For memory storage, append is same as create (data persists in memory)
        Ok(Box::new(MemoryOutput::new_append(
            name.to_string(),
            Arc::clone(&self.files),
        )))
    }

    fn file_exists(&self, name: &str) -> bool {
        if self.closed {
            return false;
        }

        let files = self.files.read();
        files.contains_key(name)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.check_closed()?;

        let mut files = self.files.write();
        files.remove(name);
        Ok(())
    }

    fn list_files(&self) -> Result<Vec<String>> {
        self.check_closed()?;

        let files = self.files.read();
        let mut file_names: Vec<String> = files.keys().cloned().collect();
        file_names.sort();
        Ok(file_names)
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        self.check_closed()?;

        let files = self.files.read();
        let data = files
            .get(name)
            .ok_or_else(|| StorageError::FileNotFound(name.to_string()))?;

        Ok(data.len() as u64)
    }

    /// Returns metadata for the named in-memory file.
    ///
    /// **Note:** Because in-memory storage does not track real modification or
    /// creation timestamps, both `modified` and `created` are set to the
    /// current wall-clock time at the moment this method is called. They do
    /// **not** reflect when the file was actually written or first created.
    fn metadata(&self, name: &str) -> Result<crate::storage::FileMetadata> {
        self.check_closed()?;

        let files = self.files.read();
        if let Some(data) = files.get(name) {
            let now = crate::util::time::now_secs();

            Ok(crate::storage::FileMetadata {
                size: data.len() as u64,
                modified: now,
                created: now,
                readonly: false,
            })
        } else {
            Err(LaurusError::storage(format!("File not found: {name}")))
        }
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.check_closed()?;

        let mut files = self.files.write();
        let data = files
            .remove(old_name)
            .ok_or_else(|| StorageError::FileNotFound(old_name.to_string()))?;

        files.insert(new_name.to_string(), data);
        Ok(())
    }

    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        self.check_closed()?;

        let mut counter = 0;
        let mut temp_name;

        loop {
            temp_name = format!("{prefix}_{counter}.tmp");
            if !self.file_exists(&temp_name) {
                break;
            }
            counter += 1;

            if counter > 10000 {
                return Err(
                    StorageError::IoError("Could not create temporary file".to_string()).into(),
                );
            }
        }

        let output = self.create_output(&temp_name)?;
        Ok((temp_name, output))
    }

    fn sync(&self) -> Result<()> {
        self.check_closed()?;
        // For memory storage, sync is a no-op
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.closed = true;
        self.lock_manager.release_all()?;
        Ok(())
    }
}

/// A read handle backed by an in-memory byte buffer.
///
/// `MemoryInput` wraps a [`Cursor<Vec<u8>>`] and implements [`StorageInput`]
/// so that callers can seek and read just as they would with a file-backed
/// input.
#[derive(Debug)]
pub struct MemoryInput {
    /// Cursor providing Read + Seek over the byte buffer.
    cursor: Cursor<Vec<u8>>,
    /// Cached size of the data (avoids re-computing from the cursor).
    size: u64,
}

impl MemoryInput {
    fn new(data: Box<[u8]>) -> Self {
        let data_vec = data.into_vec();
        let size = data_vec.len() as u64;
        let cursor = Cursor::new(data_vec);
        MemoryInput { cursor, size }
    }
}

impl Read for MemoryInput {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl Seek for MemoryInput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.cursor.seek(pos)
    }
}

impl StorageInput for MemoryInput {
    fn size(&self) -> Result<u64> {
        Ok(self.size)
    }

    fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
        Ok(Box::new(MemoryInput::new(
            self.cursor.get_ref().clone().into_boxed_slice(),
        )))
    }

    fn close(&mut self) -> Result<()> {
        // Nothing to close for memory input
        Ok(())
    }

    /// Borrow the in-memory buffer from the current read position to
    /// the end. Issue #504 zero-copy path: callers (the lexical
    /// posting decoder) can take a slice into the buffer instead of
    /// re-allocating + `copy_from_slice` through `Read`.
    fn as_slice(&self) -> Option<&[u8]> {
        let pos = self.cursor.position() as usize;
        let data = self.cursor.get_ref();
        Some(&data[pos.min(data.len())..])
    }
}

/// A write handle backed by an in-memory byte buffer.
///
/// `MemoryOutput` accumulates written bytes in a `Vec<u8>`. When the handle
/// is closed (or dropped), the buffer is stored in the shared file map so
/// that subsequent [`MemoryStorage::open_input`] calls can read the data.
#[derive(Debug)]
pub struct MemoryOutput {
    /// The logical file name.
    name: String,
    /// Write buffer accumulating output bytes.
    buffer: Vec<u8>,
    /// Shared reference to the storage file map for committing on close.
    files: Arc<RwLock<HashMap<String, Box<[u8]>>>>,
    /// Current write position within the buffer.
    position: u64,
    /// Whether this output handle has been closed.
    closed: bool,
}

impl MemoryOutput {
    fn new(name: String, files: Arc<RwLock<HashMap<String, Box<[u8]>>>>) -> Self {
        MemoryOutput {
            name,
            buffer: Vec::new(),
            files,
            position: 0,
            closed: false,
        }
    }

    fn new_append(name: String, files: Arc<RwLock<HashMap<String, Box<[u8]>>>>) -> Self {
        // For append mode, load existing data into buffer
        let existing_data = {
            let files_guard = files.read();
            files_guard
                .get(&name)
                .map(|data| data.to_vec())
                .unwrap_or_default()
        };

        let position = existing_data.len() as u64;

        MemoryOutput {
            name,
            buffer: existing_data,
            files,
            position,
            closed: false,
        }
    }
}

impl Write for MemoryOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.closed {
            return Err(std::io::Error::other("Output is closed"));
        }

        let end_pos = (self.position as usize)
            .checked_add(buf.len())
            .ok_or_else(|| std::io::Error::other("File too large"))?;

        if end_pos > self.buffer.len() {
            // Resize buffer if needed, filling gaps with zeros
            self.buffer.resize(end_pos, 0);
        }

        // Write data at current position
        self.buffer[self.position as usize..end_pos].copy_from_slice(buf);

        self.position += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // For memory output, flushing is a no-op
        Ok(())
    }
}

impl Seek for MemoryOutput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        if self.closed {
            return Err(std::io::Error::other("Output is closed"));
        }

        let new_pos = match pos {
            SeekFrom::Start(offset) => offset,
            SeekFrom::End(offset) => {
                if offset < 0 {
                    let abs_offset = (-offset) as u64;
                    if abs_offset > self.buffer.len() as u64 {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Invalid seek position",
                        ));
                    }
                    self.buffer.len() as u64 - abs_offset
                } else {
                    self.buffer.len() as u64 + offset as u64
                }
            }
            SeekFrom::Current(offset) => {
                if offset < 0 {
                    let abs_offset = (-offset) as u64;
                    if abs_offset > self.position {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Invalid seek position",
                        ));
                    }
                    self.position - abs_offset
                } else {
                    self.position + offset as u64
                }
            }
        };

        self.position = new_pos;
        Ok(new_pos)
    }
}

impl StorageOutput for MemoryOutput {
    fn flush_and_sync(&mut self) -> Result<()> {
        // Update the shared storage with current buffer content.
        // This mimics filesystem behavior where flushed data is visible to readers
        // even if the writer is still open.
        let mut files = self.files.write();
        files.insert(self.name.clone(), self.buffer.clone().into_boxed_slice());
        Ok(())
    }

    fn position(&self) -> Result<u64> {
        Ok(self.position)
    }

    fn close(&mut self) -> Result<()> {
        if !self.closed {
            // Store the buffer in the files map, converting Vec<u8> to Box<[u8]>.
            let mut files = self.files.write();
            files.insert(self.name.clone(), self.buffer.clone().into_boxed_slice());
            self.closed = true;
        }
        Ok(())
    }
}

impl Drop for MemoryOutput {
    fn drop(&mut self) {
        // Ensure the file is stored when the output is dropped
        let _ = self.close();
    }
}

/// A memory-based lock manager for coordinating concurrent access.
///
/// Locks are tracked in a `HashMap` keyed by lock name. This provides the
/// same semantics as file-based locks but without touching the filesystem.
#[derive(Debug)]
pub struct MemoryLockManager {
    /// Map of active locks, keyed by lock name.
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<MemoryLock>>>>>,
}

impl MemoryLockManager {
    fn new() -> Self {
        MemoryLockManager {
            locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl LockManager for MemoryLockManager {
    fn acquire_lock(&self, name: &str) -> Result<Box<dyn StorageLock>> {
        let mut locks = self.locks.lock().unwrap();

        if locks.contains_key(name) {
            return Err(StorageError::LockFailed(name.to_string()).into());
        }

        let lock = Arc::new(Mutex::new(MemoryLock::new(name.to_string())));
        locks.insert(name.to_string(), lock.clone());

        Ok(Box::new(MemoryLockWrapper { lock }))
    }

    fn try_acquire_lock(&self, name: &str) -> Result<Option<Box<dyn StorageLock>>> {
        match self.acquire_lock(name) {
            Ok(lock) => Ok(Some(lock)),
            Err(e) => {
                if let LaurusError::Storage(ref msg) = e
                    && msg.contains("Failed to acquire lock")
                {
                    return Ok(None);
                }
                Err(e)
            }
        }
    }

    fn lock_exists(&self, name: &str) -> bool {
        let locks = self.locks.lock().unwrap();
        locks.contains_key(name)
    }

    fn release_all(&self) -> Result<()> {
        let mut locks = self.locks.lock().unwrap();
        locks.clear();
        Ok(())
    }
}

/// A memory-based lock implementation.
#[derive(Debug)]
struct MemoryLock {
    #[allow(dead_code)]
    name: String,
    released: bool,
}

impl MemoryLock {
    fn new(name: String) -> Self {
        MemoryLock {
            name,
            released: false,
        }
    }
}

/// A wrapper around [`MemoryLock`] that implements [`StorageLock`].
#[derive(Debug)]
struct MemoryLockWrapper {
    lock: Arc<Mutex<MemoryLock>>,
}

impl StorageLock for MemoryLockWrapper {
    /// Returns a fixed identifier `"memory_lock"` rather than the actual lock name.
    ///
    /// Because the real name is stored behind a `Mutex`, returning a borrowed
    /// reference to it is not possible with the current `&str` return type.
    fn name(&self) -> &str {
        "memory_lock"
    }

    fn release(&mut self) -> Result<()> {
        let mut lock = self.lock.lock().unwrap();
        lock.released = true;
        Ok(())
    }

    fn is_valid(&self) -> bool {
        let lock = self.lock.lock().unwrap();
        !lock.released
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_memory_storage_creation() {
        let storage = MemoryStorage::default();
        assert_eq!(storage.file_count(), 0);
        assert_eq!(storage.total_size(), 0);
    }

    #[test]
    fn test_create_and_read_file() {
        let storage = MemoryStorage::default();

        // Create a file
        let mut output = storage.create_output("test.txt").unwrap();
        output.write_all(b"Hello, Memory!").unwrap();
        output.close().unwrap();

        // Read the file
        let mut input = storage.open_input("test.txt").unwrap();
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer).unwrap();

        assert_eq!(buffer, b"Hello, Memory!");
        assert_eq!(input.size().unwrap(), 14);
        assert_eq!(storage.file_count(), 1);
        assert_eq!(storage.total_size(), 14);
    }

    #[test]
    fn as_slice_returns_remaining_bytes() {
        // Issue #504: MemoryInput exposes a zero-copy slice into the
        // in-memory buffer for the lexical posting decoder.
        let storage = MemoryStorage::default();
        let mut output = storage.create_output("data.bin").unwrap();
        output.write_all(b"abcdefghij").unwrap();
        output.close().unwrap();

        let mut input = storage.open_input("data.bin").unwrap();
        // Initial slice covers the whole file.
        assert_eq!(input.as_slice(), Some(&b"abcdefghij"[..]));

        // After advancing the read cursor by 3 bytes the slice tail
        // matches.
        let mut head = [0u8; 3];
        input.read_exact(&mut head).unwrap();
        assert_eq!(&head, b"abc");
        assert_eq!(input.as_slice(), Some(&b"defghij"[..]));

        // Seek to end → empty slice.
        input.seek(SeekFrom::End(0)).unwrap();
        assert_eq!(input.as_slice(), Some(&[][..]));
    }

    #[test]
    fn test_file_operations() {
        let storage = MemoryStorage::default();

        // File doesn't exist initially
        assert!(!storage.file_exists("nonexistent.txt"));

        // Create a file
        let mut output = storage.create_output("test.txt").unwrap();
        output.write_all(b"Test content").unwrap();
        output.close().unwrap();

        // File exists now
        assert!(storage.file_exists("test.txt"));

        // Check file size
        assert_eq!(storage.file_size("test.txt").unwrap(), 12);

        // List files
        let files = storage.list_files().unwrap();
        assert_eq!(files, vec!["test.txt"]);

        // Rename file
        storage.rename_file("test.txt", "renamed.txt").unwrap();
        assert!(!storage.file_exists("test.txt"));
        assert!(storage.file_exists("renamed.txt"));

        // Delete file
        storage.delete_file("renamed.txt").unwrap();
        assert!(!storage.file_exists("renamed.txt"));
        assert_eq!(storage.file_count(), 0);
    }

    #[test]
    fn test_multiple_files() {
        let storage = MemoryStorage::default();

        // Create multiple files
        for i in 0..5 {
            let mut output = storage.create_output(&format!("file_{i}.txt")).unwrap();
            output.write_all(format!("Content {i}").as_bytes()).unwrap();
            output.close().unwrap();
        }

        assert_eq!(storage.file_count(), 5);

        let files = storage.list_files().unwrap();
        assert_eq!(files.len(), 5);

        // Check that files are sorted
        for (i, file) in files.iter().enumerate().take(5) {
            assert_eq!(file, &format!("file_{i}.txt"));
        }
    }

    #[test]
    fn test_temp_file_creation() {
        let storage = MemoryStorage::default();

        let (temp_name, mut output) = storage.create_temp_output("test").unwrap();

        assert!(temp_name.starts_with("test_"));
        assert!(temp_name.ends_with(".tmp"));

        output.write_all(b"Temporary content").unwrap();
        output.close().unwrap();

        assert!(storage.file_exists(&temp_name));
        assert_eq!(storage.file_size(&temp_name).unwrap(), 17);
    }

    #[test]
    fn test_input_clone() {
        let storage = MemoryStorage::default();

        // Create a file
        let mut output = storage.create_output("test.txt").unwrap();
        output.write_all(b"Hello, Clone!").unwrap();
        output.close().unwrap();

        // Open input and clone it
        let mut input1 = storage.open_input("test.txt").unwrap();
        let mut input2 = input1.clone_input().unwrap();

        // Read from both inputs
        let mut buffer1 = Vec::new();
        let mut buffer2 = Vec::new();

        input1.read_to_end(&mut buffer1).unwrap();
        input2.read_to_end(&mut buffer2).unwrap();

        assert_eq!(buffer1, b"Hello, Clone!");
        assert_eq!(buffer2, b"Hello, Clone!");
        assert_eq!(buffer1, buffer2);
    }

    #[test]
    fn test_seek_operations() {
        let storage = MemoryStorage::default();

        // Create a file
        let mut output = storage.create_output("test.txt").unwrap();
        output.write_all(b"0123456789").unwrap();
        output.close().unwrap();

        // Test seeking in input
        let mut input = storage.open_input("test.txt").unwrap();

        // Seek to position 5
        input.seek(SeekFrom::Start(5)).unwrap();
        let mut buffer = [0u8; 3];
        input.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"567");

        // Seek from end
        input.seek(SeekFrom::End(-2)).unwrap();
        let mut buffer = [0u8; 2];
        input.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"89");
    }

    #[test]
    fn test_file_not_found() {
        let storage = MemoryStorage::default();

        let result = storage.open_input("nonexistent.txt");
        assert!(result.is_err());

        let result = storage.file_size("nonexistent.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_storage_close() {
        let mut storage = MemoryStorage::default();

        storage.close().unwrap();
        assert!(storage.closed);

        // Operations should fail after close
        let result = storage.create_output("test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_storage() {
        let storage = MemoryStorage::default();

        // Create some files
        for i in 0..3 {
            let mut output = storage.create_output(&format!("file_{i}.txt")).unwrap();
            output.write_all(b"content").unwrap();
            output.close().unwrap();
        }

        assert_eq!(storage.file_count(), 3);

        // Clear storage
        storage.clear().unwrap();

        assert_eq!(storage.file_count(), 0);
        assert_eq!(storage.total_size(), 0);
    }
}
