//! In-memory storage implementation for testing and caching.

use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use crate::error::{Result, SageError};
use crate::storage::traits::{
    LockManager, Storage, StorageConfig, StorageError, StorageInput, StorageLock, StorageOutput,
};

/// An in-memory storage implementation.
///
/// This is useful for testing and for creating temporary indexes in memory.
/// Uses Box<[u8]> for memory efficiency when files are finalized.
#[derive(Debug)]
pub struct MemoryStorage {
    /// The files stored in memory with optimized memory layout.
    files: Arc<Mutex<HashMap<String, Box<[u8]>>>>,
    /// Lock manager for coordinating access.
    lock_manager: Arc<MemoryLockManager>,
    /// Storage configuration.
    #[allow(dead_code)]
    config: StorageConfig,
    /// Whether the storage is closed.
    closed: bool,
}

impl MemoryStorage {
    /// Create a new memory storage.
    pub fn new(config: StorageConfig) -> Self {
        MemoryStorage {
            files: Arc::new(Mutex::new(HashMap::new())),
            lock_manager: Arc::new(MemoryLockManager::new()),
            config,
            closed: false,
        }
    }

    /// Create a new memory storage with default configuration.
    pub fn new_default() -> Self {
        Self::new(StorageConfig::default())
    }

    /// Check if the storage is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed {
            Err(StorageError::StorageClosed.into())
        } else {
            Ok(())
        }
    }

    /// Get the number of files stored.
    pub fn file_count(&self) -> usize {
        self.files.lock().unwrap().len()
    }

    /// Get the total size of all files.
    pub fn total_size(&self) -> u64 {
        let files = self.files.lock().unwrap();
        files.values().map(|data| data.len() as u64).sum()
    }

    /// Clear all files from storage.
    pub fn clear(&self) -> Result<()> {
        self.check_closed()?;
        let mut files = self.files.lock().unwrap();
        files.clear();
        Ok(())
    }
}

impl Storage for MemoryStorage {
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.check_closed()?;

        let files = self.files.lock().unwrap();
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

        let files = self.files.lock().unwrap();
        files.contains_key(name)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.check_closed()?;

        let mut files = self.files.lock().unwrap();
        files.remove(name);
        Ok(())
    }

    fn list_files(&self) -> Result<Vec<String>> {
        self.check_closed()?;

        let files = self.files.lock().unwrap();
        let mut file_names: Vec<String> = files.keys().cloned().collect();
        file_names.sort();
        Ok(file_names)
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        self.check_closed()?;

        let files = self.files.lock().unwrap();
        let data = files
            .get(name)
            .ok_or_else(|| StorageError::FileNotFound(name.to_string()))?;

        Ok(data.len() as u64)
    }

    fn metadata(&self, name: &str) -> Result<crate::storage::traits::FileMetadata> {
        self.check_closed()?;

        let files = self.files.lock().unwrap();
        if let Some(data) = files.get(name) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            Ok(crate::storage::traits::FileMetadata {
                size: data.len() as u64,
                modified: now,
                created: now,
                readonly: false,
            })
        } else {
            Err(SageError::storage(format!("File not found: {name}")))
        }
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.check_closed()?;

        let mut files = self.files.lock().unwrap();
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

/// A memory-based input implementation.
#[derive(Debug)]
pub struct MemoryInput {
    cursor: Cursor<Vec<u8>>,
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
}

/// A memory-based output implementation.
#[derive(Debug)]
pub struct MemoryOutput {
    name: String,
    buffer: Vec<u8>,
    files: Arc<Mutex<HashMap<String, Box<[u8]>>>>,
    position: u64,
    closed: bool,
}

impl MemoryOutput {
    fn new(name: String, files: Arc<Mutex<HashMap<String, Box<[u8]>>>>) -> Self {
        MemoryOutput {
            name,
            buffer: Vec::new(),
            files,
            position: 0,
            closed: false,
        }
    }

    fn new_append(name: String, files: Arc<Mutex<HashMap<String, Box<[u8]>>>>) -> Self {
        // For append mode, load existing data into buffer
        let existing_data = {
            let files_guard = files.lock().unwrap();
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

        self.buffer.extend_from_slice(buf);
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
        // For memory output, sync is a no-op
        Ok(())
    }

    fn position(&self) -> Result<u64> {
        Ok(self.position)
    }

    fn close(&mut self) -> Result<()> {
        if !self.closed {
            // Store the buffer in the files map, converting Vec<u8> to Box<[u8]>
            let mut files = self.files.lock().unwrap();
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

/// A memory-based lock manager.
#[derive(Debug)]
pub struct MemoryLockManager {
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
                if let SageError::Storage(ref msg) = e
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

/// A wrapper around MemoryLock that implements StorageLock.
#[derive(Debug)]
struct MemoryLockWrapper {
    lock: Arc<Mutex<MemoryLock>>,
}

impl StorageLock for MemoryLockWrapper {
    fn name(&self) -> &str {
        // We can't return a reference to the name inside the mutex
        // For now, we'll return a static string
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
        let storage = MemoryStorage::new_default();
        assert_eq!(storage.file_count(), 0);
        assert_eq!(storage.total_size(), 0);
    }

    #[test]
    fn test_create_and_read_file() {
        let storage = MemoryStorage::new_default();

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
    fn test_file_operations() {
        let storage = MemoryStorage::new_default();

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
        let storage = MemoryStorage::new_default();

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
        let storage = MemoryStorage::new_default();

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
        let storage = MemoryStorage::new_default();

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
        let storage = MemoryStorage::new_default();

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
        let storage = MemoryStorage::new_default();

        let result = storage.open_input("nonexistent.txt");
        assert!(result.is_err());

        let result = storage.file_size("nonexistent.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_storage_close() {
        let mut storage = MemoryStorage::new_default();

        storage.close().unwrap();
        assert!(storage.closed);

        // Operations should fail after close
        let result = storage.create_output("test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_storage() {
        let storage = MemoryStorage::new_default();

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
