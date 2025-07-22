//! Storage abstraction trait and common types.

use crate::error::{SarissaError, Result};
use std::io::{Read, Seek, Write};

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
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>>;

    /// Create a file for writing.
    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>>;

    /// Create a file for appending.
    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>>;

    /// Check if a file exists.
    fn file_exists(&self, name: &str) -> bool;

    /// Delete a file.
    fn delete_file(&self, name: &str) -> Result<()>;

    /// List all files in the storage.
    fn list_files(&self) -> Result<Vec<String>>;

    /// Get the size of a file in bytes.
    fn file_size(&self, name: &str) -> Result<u64>;

    /// Get file metadata.
    fn metadata(&self, name: &str) -> Result<FileMetadata>;

    /// Rename a file.
    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()>;

    /// Create a temporary file.
    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)>;

    /// Sync all pending writes to storage.
    fn sync(&self) -> Result<()>;

    /// Close the storage and release resources.
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
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Whether to use memory-mapped files (if supported).
    pub use_mmap: bool,

    /// Buffer size for I/O operations.
    pub buffer_size: usize,

    /// Whether to sync writes immediately.
    pub sync_writes: bool,

    /// Whether to use file locking.
    pub use_locking: bool,

    /// Temporary directory for temp files.
    pub temp_dir: Option<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        StorageConfig {
            use_mmap: false,
            buffer_size: 65536, // 64KB buffer for better I/O performance
            sync_writes: false,
            use_locking: true,
            temp_dir: None,
        }
    }
}

/// A factory for creating storage instances.
pub trait StorageFactory: Send + Sync + std::fmt::Debug {
    /// Create a new storage instance.
    fn create_storage(&self, config: &StorageConfig) -> Result<Box<dyn Storage>>;

    /// Get the name of this storage type.
    fn storage_type(&self) -> &str;
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

impl From<StorageError> for SarissaError {
    fn from(err: StorageError) -> Self {
        SarissaError::storage(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();

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
}
