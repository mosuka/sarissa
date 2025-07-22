//! File-based storage implementation.

use crate::error::{SarissaError, Result};
use crate::storage::traits::{
    LockManager, Storage, StorageConfig, StorageError, StorageInput, StorageLock, StorageOutput,
};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// A file-based storage implementation.
#[derive(Debug)]
pub struct FileStorage {
    /// The root directory for storage.
    directory: PathBuf,
    /// Storage configuration.
    config: StorageConfig,
    /// Lock manager for coordinating access.
    lock_manager: Arc<FileLockManager>,
    /// Whether the storage is closed.
    closed: bool,
}

impl FileStorage {
    /// Create a new file storage in the given directory.
    pub fn new<P: AsRef<Path>>(directory: P, config: StorageConfig) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !directory.exists() {
            std::fs::create_dir_all(&directory)
                .map_err(|e| SarissaError::storage(format!("Failed to create directory: {e}")))?;
        }

        // Verify it's a directory
        if !directory.is_dir() {
            return Err(SarissaError::storage(format!(
                "Path is not a directory: {}",
                directory.display()
            )));
        }

        let lock_manager = Arc::new(FileLockManager::new(directory.clone()));

        Ok(FileStorage {
            directory,
            config,
            lock_manager,
            closed: false,
        })
    }

    /// Get the full path for a file name.
    fn file_path(&self, name: &str) -> PathBuf {
        self.directory.join(name)
    }

    /// Check if the storage is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed {
            Err(StorageError::StorageClosed.into())
        } else {
            Ok(())
        }
    }
}

impl Storage for FileStorage {
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.check_closed()?;

        let path = self.file_path(name);
        let file = File::open(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::FileNotFound(name.to_string())
            } else {
                StorageError::IoError(e.to_string())
            }
        })?;

        Ok(Box::new(FileInput::new(file, self.config.buffer_size)?))
    }

    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.check_closed()?;

        let path = self.file_path(name);
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| StorageError::IoError(e.to_string()))?;

        Ok(Box::new(FileOutput::new(
            file,
            self.config.buffer_size,
            self.config.sync_writes,
        )?))
    }

    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.check_closed()?;

        let path = self.file_path(name);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| StorageError::IoError(e.to_string()))?;

        Ok(Box::new(FileOutput::new(
            file,
            self.config.buffer_size,
            self.config.sync_writes,
        )?))
    }

    fn file_exists(&self, name: &str) -> bool {
        if self.closed {
            return false;
        }

        self.file_path(name).exists()
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.check_closed()?;

        let path = self.file_path(name);
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| StorageError::IoError(format!("Failed to delete file: {e}")))?;
        }

        Ok(())
    }

    fn list_files(&self) -> Result<Vec<String>> {
        self.check_closed()?;

        let mut files = Vec::new();

        for entry in
            std::fs::read_dir(&self.directory).map_err(|e| StorageError::IoError(e.to_string()))?
        {
            let entry = entry.map_err(|e| StorageError::IoError(e.to_string()))?;
            let path = entry.path();

            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    files.push(name.to_string());
                }
            }
        }

        files.sort();
        Ok(files)
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        self.check_closed()?;

        let path = self.file_path(name);
        let metadata = path.metadata().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::FileNotFound(name.to_string())
            } else {
                StorageError::IoError(e.to_string())
            }
        })?;

        Ok(metadata.len())
    }

    fn metadata(&self, name: &str) -> Result<crate::storage::FileMetadata> {
        self.check_closed()?;

        let path = self.file_path(name);
        let metadata = path.metadata().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::FileNotFound(name.to_string())
            } else {
                StorageError::IoError(e.to_string())
            }
        })?;

        let modified = metadata
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let created = metadata
            .created()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(crate::storage::FileMetadata {
            size: metadata.len(),
            modified,
            created,
            readonly: metadata.permissions().readonly(),
        })
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.check_closed()?;

        let old_path = self.file_path(old_name);
        let new_path = self.file_path(new_name);

        std::fs::rename(&old_path, &new_path)
            .map_err(|e| StorageError::IoError(format!("Failed to rename file: {e}")))?;

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
        // For file storage, we don't need to do anything special for sync
        // Individual files are synced when they are closed
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.closed = true;
        self.lock_manager.release_all()?;
        Ok(())
    }
}

/// A file input implementation.
#[derive(Debug)]
pub struct FileInput {
    reader: BufReader<File>,
    size: u64,
}

impl FileInput {
    fn new(file: File, buffer_size: usize) -> Result<Self> {
        let metadata = file
            .metadata()
            .map_err(|e| SarissaError::storage(format!("Failed to get file metadata: {e}")))?;

        let size = metadata.len();
        let reader = BufReader::with_capacity(buffer_size, file);

        Ok(FileInput { reader, size })
    }
}

impl Read for FileInput {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.reader.read(buf)
    }
}

impl Seek for FileInput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.reader.seek(pos)
    }
}

impl StorageInput for FileInput {
    fn size(&self) -> Result<u64> {
        Ok(self.size)
    }

    fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
        // For file inputs, we can't easily clone the underlying file
        // This would require reopening the file, which we'll implement later
        Err(SarissaError::storage("Clone not supported for file inputs"))
    }

    fn close(&mut self) -> Result<()> {
        // BufReader doesn't have an explicit close method
        // The file will be closed when the BufReader is dropped
        Ok(())
    }
}

/// A file output implementation.
#[derive(Debug)]
pub struct FileOutput {
    writer: BufWriter<File>,
    sync_writes: bool,
    position: u64,
}

impl FileOutput {
    fn new(file: File, buffer_size: usize, sync_writes: bool) -> Result<Self> {
        let writer = BufWriter::with_capacity(buffer_size, file);

        Ok(FileOutput {
            writer,
            sync_writes,
            position: 0,
        })
    }
}

impl Write for FileOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let bytes_written = self.writer.write(buf)?;
        self.position += bytes_written as u64;

        if self.sync_writes {
            self.writer.flush()?;
        }

        Ok(bytes_written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

impl Seek for FileOutput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = self.writer.seek(pos)?;
        self.position = new_pos;
        Ok(new_pos)
    }
}

impl StorageOutput for FileOutput {
    fn flush_and_sync(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|e| SarissaError::storage(format!("Failed to flush: {e}")))?;

        self.writer
            .get_ref()
            .sync_all()
            .map_err(|e| SarissaError::storage(format!("Failed to sync: {e}")))?;

        Ok(())
    }

    fn position(&self) -> Result<u64> {
        Ok(self.position)
    }

    fn close(&mut self) -> Result<()> {
        self.flush_and_sync()?;
        Ok(())
    }
}

/// A file-based lock manager.
#[derive(Debug)]
pub struct FileLockManager {
    directory: PathBuf,
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<FileLock>>>>>,
}

impl FileLockManager {
    fn new(directory: PathBuf) -> Self {
        FileLockManager {
            directory,
            locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn lock_path(&self, name: &str) -> PathBuf {
        self.directory.join(format!("{name}.lock"))
    }
}

impl LockManager for FileLockManager {
    fn acquire_lock(&self, name: &str) -> Result<Box<dyn StorageLock>> {
        let lock_path = self.lock_path(name);

        // Try to create the lock file
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    StorageError::LockFailed(name.to_string())
                } else {
                    StorageError::IoError(e.to_string())
                }
            })?;

        let lock = Arc::new(Mutex::new(FileLock::new(name.to_string(), lock_path, file)));

        // Store the lock
        {
            let mut locks = self.locks.lock().unwrap();
            locks.insert(name.to_string(), lock.clone());
        }

        Ok(Box::new(FileLockWrapper { lock }))
    }

    fn try_acquire_lock(&self, name: &str) -> Result<Option<Box<dyn StorageLock>>> {
        match self.acquire_lock(name) {
            Ok(lock) => Ok(Some(lock)),
            Err(e) => {
                if let SarissaError::Storage(ref msg) = e {
                    if msg.contains("Failed to acquire lock") {
                        return Ok(None);
                    }
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

        for (_, lock) in locks.drain() {
            let mut file_lock = lock.lock().unwrap();
            file_lock.release()?;
        }

        Ok(())
    }
}

/// A file-based lock implementation.
#[derive(Debug)]
struct FileLock {
    #[allow(dead_code)]
    name: String,
    path: PathBuf,
    _file: File,
    released: bool,
}

impl FileLock {
    fn new(name: String, path: PathBuf, file: File) -> Self {
        FileLock {
            name,
            path,
            _file: file,
            released: false,
        }
    }

    fn release(&mut self) -> Result<()> {
        if !self.released {
            std::fs::remove_file(&self.path)
                .map_err(|e| SarissaError::storage(format!("Failed to release lock: {e}")))?;
            self.released = true;
        }
        Ok(())
    }
}

/// A wrapper around FileLock that implements StorageLock.
#[derive(Debug)]
struct FileLockWrapper {
    lock: Arc<Mutex<FileLock>>,
}

impl StorageLock for FileLockWrapper {
    fn name(&self) -> &str {
        // We can't return a reference to the name inside the mutex
        // For now, we'll return a static string
        "file_lock"
    }

    fn release(&mut self) -> Result<()> {
        let mut lock = self.lock.lock().unwrap();
        lock.release()
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
    use tempfile::TempDir;

    fn create_test_storage() -> (TempDir, FileStorage) {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig::default();
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();
        (temp_dir, storage)
    }

    #[test]
    fn test_file_storage_creation() {
        let (_temp_dir, storage) = create_test_storage();
        assert!(!storage.closed);
    }

    #[test]
    fn test_create_and_read_file() {
        let (_temp_dir, storage) = create_test_storage();

        // Create a file
        let mut output = storage.create_output("test.txt").unwrap();
        output.write_all(b"Hello, World!").unwrap();
        output.close().unwrap();

        // Read the file
        let mut input = storage.open_input("test.txt").unwrap();
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer).unwrap();

        assert_eq!(buffer, b"Hello, World!");
        assert_eq!(input.size().unwrap(), 13);
    }

    #[test]
    fn test_file_operations() {
        let (_temp_dir, storage) = create_test_storage();

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
    }

    #[test]
    fn test_temp_file_creation() {
        let (_temp_dir, storage) = create_test_storage();

        let (temp_name, mut output) = storage.create_temp_output("test").unwrap();

        assert!(temp_name.starts_with("test_"));
        assert!(temp_name.ends_with(".tmp"));

        output.write_all(b"Temporary content").unwrap();
        output.close().unwrap();

        assert!(storage.file_exists(&temp_name));
        assert_eq!(storage.file_size(&temp_name).unwrap(), 17);
    }

    #[test]
    fn test_file_not_found() {
        let (_temp_dir, storage) = create_test_storage();

        let result = storage.open_input("nonexistent.txt");
        assert!(result.is_err());

        let result = storage.file_size("nonexistent.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_storage_close() {
        let (_temp_dir, mut storage) = create_test_storage();

        storage.close().unwrap();
        assert!(storage.closed);

        // Operations should fail after close
        let result = storage.create_output("test.txt");
        assert!(result.is_err());
    }
}
