//! Storage abstraction layer for Platypus.
//!
//! This module exposes a pluggable storage facade shared by the lexical, vector,
//! and hybrid engines. File and memory backends can be swapped without touching
//! higher-level code, making it easy to move from prototyping to production.
//!
//! # Architecture
//!
//! - **Storage trait**: Unified interface for all storage backends
//! - **StorageConfig enum**: Type-safe configuration for supported backends
//! - **StorageFactory**: Helper for constructing concrete storage instances
//!
//! # Storage Types
//!
//! ## FileStorage
//! - Disk-based persistent storage
//! - Supports memory-mapped files (mmap) for high-performance reads
//! - Configurable buffering, syncing, and locking
//!
//! ## MemoryStorage
//! - In-memory storage for testing and temporary data
//! - Fast but non-persistent
//!
//! # Example
//!
//! ```
//! use platypus::storage::{StorageFactory, StorageConfig};
//! use platypus::storage::file::FileStorageConfig;
//! use platypus::storage::memory::MemoryStorageConfig;
//!
//! # fn main() -> platypus::error::Result<()> {
//! // Create file storage with mmap enabled
//! let mut file_config = FileStorageConfig::new("/tmp/test_index");
//! file_config.use_mmap = true;
//! let storage = StorageFactory::create(StorageConfig::File(file_config))?;
//!
//! // Create memory storage
//! let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
//! # Ok(())
//! # }
//! ```

use std::io::{Read, Seek, Write};
use std::sync::Arc;

use crate::error::{PlatypusError, Result};

pub mod column;
pub mod file;
pub mod memory;
pub mod structured;

/// File metadata information.
#[derive(Debug, Clone)]
pub struct FileMetadata {
    /// File size in bytes.
    pub size: u64,

    /// Last modified time (seconds since epoch).
    pub modified: u64,

    /// Creation time (seconds since epoch).
    pub created: u64,

    /// Whether the file is read-only.
    pub readonly: bool,
}

/// A trait for storage backends that can store and retrieve data.
///
/// This provides a pluggable interface for different storage implementations
/// like file system, memory, or remote storage.
pub trait Storage: Send + Sync + std::fmt::Debug {
    /// Open a file for reading.
    ///
    /// Opens an existing file and returns a `StorageInput` for reading its contents.
    /// The file must exist, or this will return an error.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file to open (relative to the storage root)
    ///
    /// # Returns
    ///
    /// * `Ok(Box<dyn StorageInput>)` - A reader for accessing the file contents
    /// * `Err(PlatypusError)` - If the file doesn't exist or cannot be opened
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::{Read, Write};
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// // First create a file
    /// let mut output = storage.create_output("index.bin")?;
    /// output.write_all(b"test data")?;
    /// output.close()?;
    ///
    /// // Now open it for reading
    /// let mut input = storage.open_input("index.bin")?;
    /// let mut buffer = Vec::new();
    /// input.read_to_end(&mut buffer)?;
    /// assert_eq!(buffer, b"test data");
    /// # Ok(())
    /// # }
    /// ```
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>>;

    /// Create a file for writing.
    ///
    /// Creates a new file or truncates an existing file for writing.
    /// If the file already exists, its contents will be overwritten.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file to create (relative to the storage root)
    ///
    /// # Returns
    ///
    /// * `Ok(Box<dyn StorageOutput>)` - A writer for writing data to the file
    /// * `Err(PlatypusError)` - If the file cannot be created
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::Write;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// let mut output = storage.create_output("index.bin")?;
    /// output.write_all(b"Hello, World!")?;
    /// output.close()?;
    ///
    /// assert!(storage.file_exists("index.bin"));
    /// # Ok(())
    /// # }
    /// ```
    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>>;

    /// Create a file for appending.
    ///
    /// Opens an existing file for appending, or creates a new file if it doesn't exist.
    /// New data will be written at the end of the existing file contents.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file to open/create (relative to the storage root)
    ///
    /// # Returns
    ///
    /// * `Ok(Box<dyn StorageOutput>)` - A writer positioned at the end of the file
    /// * `Err(PlatypusError)` - If the file cannot be opened or created
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::{Read, Write};
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// // Create file with initial content
    /// let mut output = storage.create_output("log.txt")?;
    /// output.write_all(b"First entry\n")?;
    /// output.close()?;
    ///
    /// // Append to the file
    /// let mut output = storage.create_output_append("log.txt")?;
    /// output.write_all(b"Second entry\n")?;
    /// output.close()?;
    ///
    /// // Verify both entries are present
    /// let mut input = storage.open_input("log.txt")?;
    /// let mut buffer = String::new();
    /// input.read_to_string(&mut buffer)?;
    /// assert!(buffer.contains("First entry"));
    /// assert!(buffer.contains("Second entry"));
    /// # Ok(())
    /// # }
    /// ```
    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>>;

    /// Check if a file exists.
    ///
    /// Tests whether a file with the given name exists in the storage.
    /// This is a fast operation that doesn't open the file.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file to check (relative to the storage root)
    ///
    /// # Returns
    ///
    /// * `true` - If the file exists
    /// * `false` - If the file doesn't exist
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::Write;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// assert!(!storage.file_exists("index.bin"));
    ///
    /// let mut output = storage.create_output("index.bin")?;
    /// output.write_all(b"data")?;
    /// output.close()?;
    ///
    /// assert!(storage.file_exists("index.bin"));
    /// # Ok(())
    /// # }
    /// ```
    fn file_exists(&self, name: &str) -> bool;

    /// Delete a file.
    ///
    /// Removes a file from the storage. If the file doesn't exist, this may
    /// return an error or succeed silently, depending on the implementation.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file to delete (relative to the storage root)
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the file was successfully deleted
    /// * `Err(PlatypusError)` - If the file cannot be deleted (e.g., permission denied)
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::Write;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// // Create a file
    /// let mut output = storage.create_output("temp.bin")?;
    /// output.write_all(b"temporary data")?;
    /// output.close()?;
    ///
    /// assert!(storage.file_exists("temp.bin"));
    ///
    /// // Delete the file
    /// storage.delete_file("temp.bin")?;
    ///
    /// assert!(!storage.file_exists("temp.bin"));
    /// # Ok(())
    /// # }
    /// ```
    fn delete_file(&self, name: &str) -> Result<()>;

    /// List all files in the storage.
    ///
    /// Returns a vector of all file names in the storage.
    /// This is useful for discovering what files exist, performing cleanup,
    /// or implementing backup/restore functionality.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - A list of all file names in the storage
    /// * `Err(PlatypusError)` - If the storage cannot be read
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::Write;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// // Create some files
    /// let mut output = storage.create_output("file1.bin")?;
    /// output.write_all(b"data1")?;
    /// output.close()?;
    ///
    /// let mut output = storage.create_output("file2.bin")?;
    /// output.write_all(b"data2")?;
    /// output.close()?;
    ///
    /// // List all files
    /// let files = storage.list_files()?;
    /// assert_eq!(files.len(), 2);
    /// assert!(files.contains(&"file1.bin".to_string()));
    /// assert!(files.contains(&"file2.bin".to_string()));
    /// # Ok(())
    /// # }
    /// ```
    fn list_files(&self) -> Result<Vec<String>>;

    /// Get the size of a file in bytes.
    ///
    /// Returns the size of the file without opening it for reading.
    /// This is faster than opening the file and seeking to the end.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file (relative to the storage root)
    ///
    /// # Returns
    ///
    /// * `Ok(u64)` - The size of the file in bytes
    /// * `Err(PlatypusError)` - If the file doesn't exist or cannot be accessed
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::Write;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// // Create a file with known size
    /// let mut output = storage.create_output("index.bin")?;
    /// output.write_all(b"12345")?;
    /// output.close()?;
    ///
    /// // Get file size
    /// let size = storage.file_size("index.bin")?;
    /// assert_eq!(size, 5);
    /// # Ok(())
    /// # }
    /// ```
    fn file_size(&self, name: &str) -> Result<u64>;

    /// Get file metadata.
    ///
    /// Returns detailed metadata about a file, including size, modification time,
    /// creation time, and read-only status. This is useful for versioning,
    /// cache invalidation, and file management.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file (relative to the storage root)
    ///
    /// # Returns
    ///
    /// * `Ok(FileMetadata)` - Metadata structure with file information
    /// * `Err(PlatypusError)` - If the file doesn't exist or cannot be accessed
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::Write;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// // Create a file
    /// let mut output = storage.create_output("index.bin")?;
    /// output.write_all(b"test")?;
    /// output.close()?;
    ///
    /// // Get metadata
    /// let meta = storage.metadata("index.bin")?;
    /// assert_eq!(meta.size, 4);
    /// assert!(meta.modified > 0);
    /// # Ok(())
    /// # }
    /// ```
    fn metadata(&self, name: &str) -> Result<FileMetadata>;

    /// Rename a file.
    ///
    /// Atomically renames a file from `old_name` to `new_name`.
    /// This is commonly used for atomic file replacement: write to a temporary file,
    /// then rename it to the final name to ensure readers never see partial data.
    ///
    /// # Arguments
    ///
    /// * `old_name` - The current name of the file
    /// * `new_name` - The new name for the file
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the file was successfully renamed
    /// * `Err(PlatypusError)` - If the source file doesn't exist, destination exists,
    ///   or the operation fails
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::Write;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// // Create a temp file
    /// let mut output = storage.create_output("index.tmp")?;
    /// output.write_all(b"new data")?;
    /// output.close()?;
    ///
    /// // Rename to final name (atomic replacement pattern)
    /// storage.rename_file("index.tmp", "index.bin")?;
    ///
    /// assert!(storage.file_exists("index.bin"));
    /// assert!(!storage.file_exists("index.tmp"));
    /// # Ok(())
    /// # }
    /// ```
    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()>;

    /// Create a temporary file.
    ///
    /// Creates a file with a unique name based on the given prefix.
    /// The actual filename will include additional characters to ensure uniqueness.
    /// Temporary files should be cleaned up by the caller when no longer needed.
    ///
    /// # Arguments
    ///
    /// * `prefix` - A prefix for the temporary file name (e.g., "temp_")
    ///
    /// # Returns
    ///
    /// * `Ok((String, Box<dyn StorageOutput>))` - A tuple of (filename, writer)
    ///   where filename is the unique generated name
    /// * `Err(PlatypusError)` - If the temporary file cannot be created
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::Write;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// // Create a temp file with prefix
    /// let (temp_name, mut temp_output) = storage.create_temp_output("merge_")?;
    /// temp_output.write_all(b"temporary data")?;
    /// temp_output.close()?;
    ///
    /// // Verify file was created
    /// assert!(storage.file_exists(&temp_name));
    /// assert!(temp_name.starts_with("merge_"));
    ///
    /// // Clean up
    /// storage.delete_file(&temp_name)?;
    /// # Ok(())
    /// # }
    /// ```
    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)>;

    /// Sync all pending writes to storage.
    ///
    /// Ensures that all buffered writes are flushed and persisted to the underlying
    /// storage medium (disk, network, etc.). This is important for durability:
    /// after sync() returns, data should survive system crashes.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If all pending writes were successfully synced
    /// * `Err(PlatypusError)` - If the sync operation fails
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::Write;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// // Write some data
    /// let mut output = storage.create_output("important.dat")?;
    /// output.write_all(b"critical data")?;
    /// output.close()?;
    ///
    /// // Ensure data is persisted
    /// storage.sync()?;
    /// # Ok(())
    /// # }
    /// ```
    fn sync(&self) -> Result<()>;

    /// Close the storage and release resources.
    ///
    /// Closes all open files, flushes pending writes, and releases any resources
    /// held by the storage. After calling close(), the storage should not be used.
    /// This method takes `&mut self` to ensure exclusive access during cleanup.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the storage was successfully closed
    /// * `Err(PlatypusError)` - If errors occur during cleanup (though resources
    ///   should still be released as much as possible)
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use platypus::storage::Storage;
    /// use std::io::Write;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let mut storage = MemoryStorage::new(MemoryStorageConfig::default());
    ///
    /// // Use storage
    /// let mut output = storage.create_output("data.bin")?;
    /// output.write_all(b"some data")?;
    /// output.close()?;
    ///
    /// // Clean shutdown
    /// storage.close()?;
    /// # Ok(())
    /// # }
    /// ```
    fn close(&mut self) -> Result<()>;
}

/// A trait for reading data from storage.
pub trait StorageInput: Read + Seek + Send + std::fmt::Debug {
    /// Get the size of the input stream.
    fn size(&self) -> Result<u64>;

    /// Clone this input stream.
    fn clone_input(&self) -> Result<Box<dyn StorageInput>>;

    /// Close the input stream.
    fn close(&mut self) -> Result<()>;
}

/// A trait for writing data to storage.
pub trait StorageOutput: Write + Seek + Send + std::fmt::Debug {
    /// Flush and sync the output to storage.
    fn flush_and_sync(&mut self) -> Result<()>;

    /// Get the current position in the output stream.
    fn position(&self) -> Result<u64>;

    /// Close the output stream.
    fn close(&mut self) -> Result<()>;
}

// Implement StorageOutput for Box<dyn StorageOutput> to allow trait objects
impl StorageOutput for Box<dyn StorageOutput> {
    fn flush_and_sync(&mut self) -> Result<()> {
        self.as_mut().flush_and_sync()
    }

    fn position(&self) -> Result<u64> {
        self.as_ref().position()
    }

    fn close(&mut self) -> Result<()> {
        self.as_mut().close()
    }
}

// Implement StorageInput for Box<dyn StorageInput> to allow trait objects
impl StorageInput for Box<dyn StorageInput> {
    fn size(&self) -> Result<u64> {
        self.as_ref().size()
    }

    fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
        self.as_ref().clone_input()
    }

    fn close(&mut self) -> Result<()> {
        self.as_mut().close()
    }
}

/// A lock manager for coordinating access to storage.
pub trait LockManager: Send + Sync + std::fmt::Debug {
    /// Acquire a lock with the given name.
    fn acquire_lock(&self, name: &str) -> Result<Box<dyn StorageLock>>;

    /// Try to acquire a lock with the given name, returning None if not available.
    fn try_acquire_lock(&self, name: &str) -> Result<Option<Box<dyn StorageLock>>>;

    /// Check if a lock with the given name exists.
    fn lock_exists(&self, name: &str) -> bool;

    /// Release all locks (for cleanup).
    fn release_all(&self) -> Result<()>;
}

/// A lock on a resource in storage.
pub trait StorageLock: Send + std::fmt::Debug {
    /// Get the name of the lock.
    fn name(&self) -> &str;

    /// Release the lock.
    fn release(&mut self) -> Result<()>;

    /// Check if the lock is still valid.
    fn is_valid(&self) -> bool;
}

/// Configuration for storage backends.
///
/// This enum provides type-safe configuration for different storage implementations.
/// Each variant contains the configuration specific to that storage type, including
/// the path for file-based storage.
///
/// # Design Pattern
///
/// This follows an enum-based configuration pattern where:
/// - Each storage type has its own dedicated config struct
/// - The path is part of `FileStorageConfig` (not a separate parameter)
/// - Pattern matching ensures exhaustive handling of all storage types
///
/// # Example
///
/// ```
/// use platypus::storage::StorageConfig;
/// use platypus::storage::file::FileStorageConfig;
/// use platypus::storage::memory::MemoryStorageConfig;
///
/// // File storage with custom settings
/// let mut file_config = FileStorageConfig::new("/data/index");
/// file_config.use_mmap = true;
/// file_config.buffer_size = 131072; // 128KB
/// let config = StorageConfig::File(file_config);
///
/// // Memory storage with default settings
/// let config = StorageConfig::Memory(MemoryStorageConfig::default());
/// ```
#[derive(Debug, Clone)]
pub enum StorageConfig {
    /// File-based storage configuration (includes path)
    File(file::FileStorageConfig),

    /// Memory-based storage configuration
    Memory(memory::MemoryStorageConfig),
}

impl Default for StorageConfig {
    fn default() -> Self {
        StorageConfig::Memory(memory::MemoryStorageConfig::default())
    }
}

/// A factory for creating storage instances.
///
/// This factory creates appropriate storage implementations based on the
/// provided configuration. It follows the same pattern as LexicalIndexFactory.
pub struct StorageFactory;

impl StorageFactory {
    /// Create a new storage instance with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Storage configuration (enum containing type-specific settings including path for file storage)
    ///
    /// # Returns
    ///
    /// A boxed storage implementation based on the configured storage type.
    ///
    /// # Example
    ///
    /// ```
    /// use platypus::storage::{StorageFactory, StorageConfig};
    /// use platypus::storage::file::FileStorageConfig;
    /// use platypus::storage::memory::MemoryStorageConfig;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// // Create memory storage
    /// let config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// let storage = StorageFactory::create(config)?;
    ///
    /// // Create file storage
    /// let file_config = FileStorageConfig::new("/tmp/index");
    /// let config = StorageConfig::File(file_config);
    /// let storage = StorageFactory::create(config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create(config: StorageConfig) -> Result<Arc<dyn Storage>> {
        match config {
            StorageConfig::Memory(mem_config) => {
                let storage = memory::MemoryStorage::new(mem_config);
                Ok(Arc::new(storage))
            }
            StorageConfig::File(file_config) => {
                let path = file_config.path.clone();
                let storage = file::FileStorage::new(&path, file_config)?;
                Ok(Arc::new(storage))
            }
        }
    }

    /// Open an existing storage instance.
    ///
    /// This is similar to `create` but intended for opening existing storage.
    /// The behavior is currently the same, but may differ in the future.
    pub fn open(config: StorageConfig) -> Result<Arc<dyn Storage>> {
        Self::create(config)
    }
}

/// Error types specific to storage operations.
#[derive(Debug, Clone)]
pub enum StorageError {
    /// File not found.
    FileNotFound(String),

    /// File already exists.
    FileExists(String),

    /// Permission denied.
    PermissionDenied(String),

    /// I/O error.
    IoError(String),

    /// Lock acquisition failed.
    LockFailed(String),

    /// Storage is closed.
    StorageClosed,

    /// Invalid operation.
    InvalidOperation(String),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::FileNotFound(name) => write!(f, "File not found: {name}"),
            StorageError::FileExists(name) => write!(f, "File already exists: {name}"),
            StorageError::PermissionDenied(name) => write!(f, "Permission denied: {name}"),
            StorageError::IoError(msg) => write!(f, "I/O error: {msg}"),
            StorageError::LockFailed(name) => write!(f, "Failed to acquire lock: {name}"),
            StorageError::StorageClosed => write!(f, "Storage is closed"),
            StorageError::InvalidOperation(msg) => write!(f, "Invalid operation: {msg}"),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<StorageError> for PlatypusError {
    fn from(err: StorageError) -> Self {
        PlatypusError::storage(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::file::FileStorageConfig;
    use crate::storage::memory::MemoryStorageConfig;

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();

        // Default is Memory
        match config {
            StorageConfig::Memory(mem_config) => {
                assert_eq!(mem_config.initial_capacity, 16);
            }
            _ => panic!("Expected Memory config"),
        }
    }

    #[test]
    fn test_file_storage_config() {
        let config = FileStorageConfig::new("/tmp/test");

        assert_eq!(config.path, std::path::PathBuf::from("/tmp/test"));
        assert!(!config.use_mmap);
        assert_eq!(config.buffer_size, 65536);
        assert!(!config.sync_writes);
        assert!(config.use_locking);
        assert!(config.temp_dir.is_none());
    }

    #[test]
    fn test_storage_error_display() {
        let err = StorageError::FileNotFound("test.txt".to_string());
        assert_eq!(err.to_string(), "File not found: test.txt");

        let err = StorageError::FileExists("test.txt".to_string());
        assert_eq!(err.to_string(), "File already exists: test.txt");

        let err = StorageError::PermissionDenied("test.txt".to_string());
        assert_eq!(err.to_string(), "Permission denied: test.txt");

        let err = StorageError::IoError("connection failed".to_string());
        assert_eq!(err.to_string(), "I/O error: connection failed");

        let err = StorageError::LockFailed("write.lock".to_string());
        assert_eq!(err.to_string(), "Failed to acquire lock: write.lock");

        let err = StorageError::StorageClosed;
        assert_eq!(err.to_string(), "Storage is closed");

        let err = StorageError::InvalidOperation("cannot write to read-only storage".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid operation: cannot write to read-only storage"
        );
    }

    #[test]
    fn test_file_metadata() {
        let metadata = FileMetadata {
            size: 1024,
            modified: 1234567890,
            created: 1234567890,
            readonly: false,
        };

        assert_eq!(metadata.size, 1024);
        assert_eq!(metadata.modified, 1234567890);
        assert_eq!(metadata.created, 1234567890);
        assert!(!metadata.readonly);
    }

    #[test]
    fn test_storage_factory_memory() {
        let config = StorageConfig::Memory(MemoryStorageConfig::default());
        let storage = StorageFactory::create(config).unwrap();

        // Test that we can use the storage
        assert!(!storage.file_exists("test.txt"));
    }

    #[test]
    fn test_storage_factory_file() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let file_config = FileStorageConfig::new(temp_dir.path());
        let config = StorageConfig::File(file_config);
        let storage = StorageFactory::create(config).unwrap();

        // Test that we can use the storage
        assert!(!storage.file_exists("test.txt"));
    }

    #[test]
    fn test_storage_factory_with_mmap() {
        use std::io::Write;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let mut file_config = FileStorageConfig::new(temp_dir.path());
        file_config.use_mmap = true;

        let config = StorageConfig::File(file_config);
        let storage = StorageFactory::create(config).unwrap();

        // Create and write a file
        let mut output = storage.create_output("test.txt").unwrap();
        output.write_all(b"Hello, Factory!").unwrap();
        output.close().unwrap();

        // Read the file back (should use mmap)
        let mut input = storage.open_input("test.txt").unwrap();
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer).unwrap();

        assert_eq!(buffer, b"Hello, Factory!");
    }
}
