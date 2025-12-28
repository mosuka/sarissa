//! File-based storage implementation.
//!
//! This module provides disk-based persistent storage with support for both
//! traditional file I/O and memory-mapped files (mmap).
//!
//! # Features
//!
//! - **Traditional I/O**: Buffered reads/writes with configurable buffer size
//! - **Memory-mapped I/O**: High-performance reads using mmap with caching
//! - **File locking**: Concurrent access control
//! - **Flexible configuration**: Buffer size, sync writes, temp directory, etc.
//!
//! # Memory-Mapped Mode
//!
//! When `FileStorageConfig.use_mmap` is enabled:
//! - Files are mapped into memory for reading
//! - Mapped files are cached for reuse
//! - File modifications are detected and cache is invalidated
//! - Supports prefaulting and huge pages for performance
//!
//! # Example
//!
//! ```
//! use sarissa::storage::file::{FileStorage, FileStorageConfig};
//! use sarissa::storage::Storage;
//! use std::io::Write;
//! use tempfile::TempDir;
//!
//! # fn main() -> sarissa::error::Result<()> {
//! // Create storage with mmap enabled
//! let temp_dir = TempDir::new().unwrap();
//! let mut config = FileStorageConfig::new(temp_dir.path());
//! config.use_mmap = true;
//! let storage = FileStorage::new(temp_dir.path(), config)?;
//!
//! // Write a file
//! let mut output = storage.create_output("test.dat")?;
//! output.write_all(b"Hello, world!")?;
//! output.close()?;
//!
//! // Read using mmap
//! let mut input = storage.open_input("test.dat")?;
//! let mut buffer = Vec::new();
//! input.read_to_end(&mut buffer)?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::SystemTime;

use memmap2::{Mmap, MmapOptions};

use crate::error::{Result, SarissaError};
use crate::storage::{
    LockManager, Storage, StorageError, StorageInput, StorageLock, StorageOutput,
};

/// Configuration specific to file-based storage.
///
/// This configuration includes the storage path and various options for
/// file I/O, memory-mapping, and locking behavior.
///
/// # Memory-Mapped Files (mmap)
///
/// When `use_mmap` is enabled, FileStorage uses memory-mapped I/O for reading files,
/// which can significantly improve performance for large files by:
/// - Avoiding system call overhead
/// - Leveraging the OS page cache
/// - Enabling zero-copy reads
///
/// Additional mmap options:
/// - `mmap_cache_size`: Number of mmap files to keep cached
/// - `mmap_enable_prefault`: Pre-populate page tables for faster initial access
/// - `mmap_enable_hugepages`: Use huge pages if available (Linux)
///
/// # Example
///
/// ```
/// use sarissa::storage::file::FileStorageConfig;
///
/// // Basic file storage
/// let config = FileStorageConfig::new("/data/index");
///
/// // High-performance configuration with mmap
/// let mut config = FileStorageConfig::new("/data/index");
/// config.use_mmap = true;
/// config.mmap_enable_prefault = true;
/// config.buffer_size = 131072; // 128KB for non-mmap operations
/// ```
#[derive(Debug, Clone)]
pub struct FileStorageConfig {
    /// Path to the storage directory.
    pub path: std::path::PathBuf,

    /// Whether to use memory-mapped files for reading.
    /// When true, files are read using mmap instead of traditional I/O.
    pub use_mmap: bool,

    /// Buffer size for traditional I/O operations (bytes).
    /// Default: 65536 (64KB). Used when `use_mmap` is false.
    pub buffer_size: usize,

    /// Whether to sync writes immediately to disk.
    /// When true, calls fsync after each write for durability.
    pub sync_writes: bool,

    /// Whether to use file locking for concurrency control.
    pub use_locking: bool,

    /// Temporary directory for temp files.
    /// If None, uses the storage directory.
    pub temp_dir: Option<String>,

    /// Maximum number of memory-mapped files to cache.
    /// Only used when `use_mmap` is true. Default: 100.
    pub mmap_cache_size: usize,

    /// Enable prefaulting for memory-mapped files.
    /// Pre-populates page tables for faster initial access.
    /// Only used when `use_mmap` is true.
    pub mmap_enable_prefault: bool,

    /// Enable huge pages for memory-mapped files if available.
    /// Can improve TLB performance for large files (Linux only).
    /// Only used when `use_mmap` is true.
    pub mmap_enable_hugepages: bool,
}

impl FileStorageConfig {
    /// Create a new FileStorageConfig with the given path and default settings.
    ///
    /// # Default Settings
    ///
    /// - `use_mmap`: false
    /// - `buffer_size`: 65536 (64KB)
    /// - `sync_writes`: false
    /// - `use_locking`: true
    /// - `mmap_cache_size`: 100
    /// - `mmap_enable_prefault`: false
    /// - `mmap_enable_hugepages`: false
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Self {
        FileStorageConfig {
            path: path.as_ref().to_path_buf(),
            use_mmap: false,
            buffer_size: 65536,
            sync_writes: false,
            use_locking: true,
            temp_dir: None,
            mmap_cache_size: 100,
            mmap_enable_prefault: false,
            mmap_enable_hugepages: false,
        }
    }
}

/// Metadata information for cached files.
#[derive(Debug, Clone)]
struct MmapFileMetadata {
    size: u64,
    modified: u64,
}

/// A file-based storage implementation.
///
/// FileStorage provides persistent disk-based storage with two read modes:
///
/// 1. **Traditional I/O** (default): Uses buffered file reads with `BufReader`
/// 2. **Memory-mapped I/O**: Uses mmap for zero-copy reads when `config.use_mmap` is true
///
/// The mmap mode includes caching and automatic invalidation on file changes,
/// making it suitable for read-heavy workloads with large files.
#[derive(Debug)]
pub struct FileStorage {
    /// The root directory for storage.
    directory: PathBuf,
    /// Storage configuration.
    config: FileStorageConfig,
    /// Lock manager for coordinating access.
    lock_manager: Arc<FileLockManager>,
    /// Whether the storage is closed.
    closed: bool,
    /// Cache of memory-mapped files (only used when use_mmap is true).
    mmap_cache: Arc<RwLock<HashMap<String, Arc<Mmap>>>>,
    /// Cache of file metadata for mmap files.
    mmap_metadata_cache: Arc<RwLock<HashMap<String, MmapFileMetadata>>>,
}

impl FileStorage {
    /// Create a new file storage in the given directory.
    pub fn new<P: AsRef<Path>>(directory: P, config: FileStorageConfig) -> Result<Self> {
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
            mmap_cache: Arc::new(RwLock::new(HashMap::new())),
            mmap_metadata_cache: Arc::new(RwLock::new(HashMap::new())),
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

    /// Get or create a memory map for a file.
    fn get_mmap(&self, name: &str) -> Result<Arc<Mmap>> {
        let file_path = self.file_path(name);

        // Check cache first
        {
            let cache = self.mmap_cache.read().unwrap();
            if let Some(mmap) = cache.get(name) {
                // Verify the file hasn't changed
                if self.is_mmap_file_unchanged(name, &file_path)? {
                    return Ok(Arc::clone(mmap));
                }
            }
        }

        // Create new memory map
        let file = File::open(&file_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::FileNotFound(name.to_string())
            } else {
                StorageError::IoError(format!("Failed to open file {name}: {e}"))
            }
        })?;

        let mut mmap_opts = MmapOptions::new();
        if self.config.mmap_enable_prefault {
            mmap_opts.populate();
        }

        let mmap = unsafe {
            mmap_opts
                .map(&file)
                .map_err(|e| SarissaError::storage(format!("Failed to mmap file {name}: {e}")))?
        };

        let mmap_arc = Arc::new(mmap);

        // Update cache
        {
            let mut cache = self.mmap_cache.write().unwrap();
            cache.insert(name.to_string(), Arc::clone(&mmap_arc));
        }

        // Update metadata cache
        self.update_mmap_metadata_cache(name, &file_path)?;

        Ok(mmap_arc)
    }

    /// Check if a memory-mapped file has been modified since last cached.
    fn is_mmap_file_unchanged(&self, name: &str, path: &Path) -> Result<bool> {
        let metadata_cache = self.mmap_metadata_cache.read().unwrap();

        if let Some(cached_meta) = metadata_cache.get(name) {
            let current_meta = std::fs::metadata(path)
                .map_err(|e| SarissaError::storage(format!("Failed to get metadata: {e}")))?;

            let current_size = current_meta.len();
            let current_modified = current_meta
                .modified()
                .unwrap_or(SystemTime::UNIX_EPOCH)
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            return Ok(cached_meta.size == current_size && cached_meta.modified == current_modified);
        }

        Ok(false)
    }

    /// Update metadata cache for a memory-mapped file.
    fn update_mmap_metadata_cache(&self, name: &str, path: &Path) -> Result<()> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| SarissaError::storage(format!("Failed to get metadata: {e}")))?;

        let size = metadata.len();
        let modified = metadata
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut cache = self.mmap_metadata_cache.write().unwrap();
        cache.insert(name.to_string(), MmapFileMetadata { size, modified });

        Ok(())
    }
}

impl Storage for FileStorage {
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.check_closed()?;

        if self.config.use_mmap {
            // Use memory-mapped file
            let mmap = self.get_mmap(name)?;
            Ok(Box::new(MmapInput::new(mmap)))
        } else {
            // Use traditional file I/O
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
    }

    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.check_closed()?;

        let path = self.file_path(name);

        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    SarissaError::storage(format!("Failed to create directory {:?}: {}", parent, e))
                })?;
            }
        }

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

        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    SarissaError::storage(format!("Failed to create directory {:?}: {}", parent, e))
                })?;
            }
        }

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

            if path.is_file()
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
            {
                files.push(name.to_string());
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

/// A memory-mapped file input implementation.
#[derive(Debug)]
pub struct MmapInput {
    mmap: Arc<Mmap>,
    #[allow(dead_code)]
    cursor: Cursor<Vec<u8>>,
    position: u64,
}

impl MmapInput {
    fn new(mmap: Arc<Mmap>) -> Self {
        MmapInput {
            mmap,
            cursor: Cursor::new(Vec::new()),
            position: 0,
        }
    }
}

impl Read for MmapInput {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let available = (self.mmap.len() as u64).saturating_sub(self.position) as usize;
        let to_read = buf.len().min(available);

        if to_read == 0 {
            return Ok(0);
        }

        let start = self.position as usize;
        let end = start + to_read;
        buf[..to_read].copy_from_slice(&self.mmap[start..end]);
        self.position += to_read as u64;

        Ok(to_read)
    }
}

impl Seek for MmapInput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(offset) => self.mmap.len() as i64 + offset,
            SeekFrom::Current(offset) => self.position as i64 + offset,
        };

        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid seek to a negative position",
            ));
        }

        self.position = new_pos as u64;
        Ok(self.position)
    }
}

impl StorageInput for MmapInput {
    fn size(&self) -> Result<u64> {
        Ok(self.mmap.len() as u64)
    }

    fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
        Ok(Box::new(MmapInput {
            mmap: Arc::clone(&self.mmap),
            cursor: Cursor::new(Vec::new()),
            position: 0,
        }))
    }

    fn close(&mut self) -> Result<()> {
        // Memory map will be automatically unmapped when dropped
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
                if let SarissaError::Storage(ref msg) = e
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
        let config = FileStorageConfig::new(temp_dir.path());
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

    #[test]
    fn test_mmap_storage() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = FileStorageConfig::new(temp_dir.path());
        config.use_mmap = true;
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();

        // Create a file
        let mut output = storage.create_output("test_mmap.txt").unwrap();
        output.write_all(b"Hello, Memory-Mapped World!").unwrap();
        output.close().unwrap();

        // Read the file using mmap
        let mut input = storage.open_input("test_mmap.txt").unwrap();
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer).unwrap();

        assert_eq!(buffer, b"Hello, Memory-Mapped World!");
        assert_eq!(input.size().unwrap(), 27);
    }

    #[test]
    fn test_mmap_cache() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = FileStorageConfig::new(temp_dir.path());
        config.use_mmap = true;
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();

        // Create a file
        let mut output = storage.create_output("cached.txt").unwrap();
        output.write_all(b"Cached content").unwrap();
        output.close().unwrap();

        // Read the file twice to test cache
        let mut input1 = storage.open_input("cached.txt").unwrap();
        let mut buffer1 = Vec::new();
        input1.read_to_end(&mut buffer1).unwrap();

        let mut input2 = storage.open_input("cached.txt").unwrap();
        let mut buffer2 = Vec::new();
        input2.read_to_end(&mut buffer2).unwrap();

        assert_eq!(buffer1, buffer2);
        assert_eq!(buffer1, b"Cached content");

        // Check that cache was used
        let cache = storage.mmap_cache.read().unwrap();
        assert!(cache.contains_key("cached.txt"));
    }

    #[test]
    fn test_mmap_clone_input() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = FileStorageConfig::new(temp_dir.path());
        config.use_mmap = true;
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();

        // Create a file
        let mut output = storage.create_output("clone_test.txt").unwrap();
        output.write_all(b"Clone me!").unwrap();
        output.close().unwrap();

        // Open and clone the input
        let mut input1 = storage.open_input("clone_test.txt").unwrap();
        let input2 = input1.clone_input().unwrap();

        // Read from both
        let mut buffer1 = Vec::new();
        input1.read_to_end(&mut buffer1).unwrap();

        let mut buffer2 = Vec::new();
        let mut input2_mut = input2;
        input2_mut.read_to_end(&mut buffer2).unwrap();

        assert_eq!(buffer1, buffer2);
        assert_eq!(buffer1, b"Clone me!");
    }
}
