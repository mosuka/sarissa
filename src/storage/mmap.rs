//! Memory-mapped storage backend for high-performance file access.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use memmap2::{Mmap, MmapOptions};

use crate::error::{Result, SarissaError};
use crate::storage::{Storage, StorageConfig, StorageInput, StorageOutput};

/// Memory-mapped storage backend that uses mmap for efficient file access.
#[derive(Debug)]
pub struct MmapStorage {
    /// Base directory for storage.
    base_path: PathBuf,
    /// Configuration options.
    #[allow(dead_code)]
    config: StorageConfig,
    /// Cache of memory-mapped files.
    mmap_cache: Arc<RwLock<HashMap<String, Arc<Mmap>>>>,
    /// Cache of file metadata.
    metadata_cache: Arc<RwLock<HashMap<String, FileMetadata>>>,
    /// Lock manager for file operations.
    #[allow(dead_code)]
    lock_manager: Arc<Mutex<HashMap<String, bool>>>,
}

/// Metadata information for cached files.
#[derive(Debug, Clone)]
struct FileMetadata {
    size: u64,
    modified: u64,
}

impl MmapStorage {
    /// Create a new memory-mapped storage backend.
    pub fn new<P: AsRef<Path>>(path: P, config: StorageConfig) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !base_path.exists() {
            std::fs::create_dir_all(&base_path)
                .map_err(|e| SarissaError::storage(format!("Failed to create directory: {e}")))?;
        }

        Ok(MmapStorage {
            base_path,
            config,
            mmap_cache: Arc::new(RwLock::new(HashMap::new())),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            lock_manager: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get the full path for a file.
    fn get_file_path(&self, name: &str) -> PathBuf {
        self.base_path.join(name)
    }

    /// Get or create a memory map for a file.
    fn get_mmap(&self, name: &str) -> Result<Arc<Mmap>> {
        let file_path = self.get_file_path(name);

        // Check cache first
        {
            let cache = self.mmap_cache.read().unwrap();
            if let Some(mmap) = cache.get(name) {
                // Verify the file hasn't changed
                if self.is_file_unchanged(name, &file_path)? {
                    return Ok(Arc::clone(mmap));
                }
            }
        }

        // Create new memory map
        let file = File::open(&file_path)
            .map_err(|e| SarissaError::storage(format!("Failed to open file {name}: {e}")))?;

        let mmap = unsafe {
            MmapOptions::new()
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
        self.update_metadata_cache(name, &file_path)?;

        Ok(mmap_arc)
    }

    /// Check if a file has been modified since last cached.
    fn is_file_unchanged(&self, name: &str, path: &Path) -> Result<bool> {
        let metadata_cache = self.metadata_cache.read().unwrap();

        if let Some(cached_meta) = metadata_cache.get(name) {
            let current_meta = std::fs::metadata(path)
                .map_err(|e| SarissaError::storage(format!("Failed to get metadata: {e}")))?;

            let current_size = current_meta.len();
            let current_modified = current_meta
                .modified()
                .unwrap_or(std::time::UNIX_EPOCH)
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            return Ok(cached_meta.size == current_size && cached_meta.modified == current_modified);
        }

        Ok(false)
    }

    /// Update metadata cache for a file.
    fn update_metadata_cache(&self, name: &str, path: &Path) -> Result<()> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| SarissaError::storage(format!("Failed to get metadata: {e}")))?;

        let size = metadata.len();
        let modified = metadata
            .modified()
            .unwrap_or(std::time::UNIX_EPOCH)
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut cache = self.metadata_cache.write().unwrap();
        cache.insert(name.to_string(), FileMetadata { size, modified });

        Ok(())
    }

    /// Clear cache entries for a file.
    fn invalidate_cache(&self, name: &str) {
        {
            let mut mmap_cache = self.mmap_cache.write().unwrap();
            mmap_cache.remove(name);
        }
        {
            let mut metadata_cache = self.metadata_cache.write().unwrap();
            metadata_cache.remove(name);
        }
    }
}

impl Storage for MmapStorage {
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        let mmap = self.get_mmap(name)?;
        Ok(Box::new(MmapInput::new(mmap)))
    }

    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        let file_path = self.get_file_path(name);

        // Invalidate cache since we're creating a new file
        self.invalidate_cache(name);

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)
            .map_err(|e| SarissaError::storage(format!("Failed to create file {name}: {e}")))?;

        Ok(Box::new(MmapOutput::new(
            file,
            name.to_string(),
            Arc::clone(&self.mmap_cache),
        )))
    }

    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        let file_path = self.get_file_path(name);

        // Invalidate cache since we're modifying the file
        self.invalidate_cache(name);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .map_err(|e| {
                SarissaError::storage(format!("Failed to create append file {name}: {e}"))
            })?;

        Ok(Box::new(MmapOutput::new(
            file,
            name.to_string(),
            Arc::clone(&self.mmap_cache),
        )))
    }

    fn file_exists(&self, name: &str) -> bool {
        self.get_file_path(name).exists()
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        let file_path = self.get_file_path(name);

        // Invalidate cache first
        self.invalidate_cache(name);

        if file_path.exists() {
            std::fs::remove_file(&file_path)
                .map_err(|e| SarissaError::storage(format!("Failed to delete file {name}: {e}")))?;
        }

        Ok(())
    }

    fn list_files(&self) -> Result<Vec<String>> {
        let mut files = Vec::new();

        let entries = std::fs::read_dir(&self.base_path)
            .map_err(|e| SarissaError::storage(format!("Failed to read directory: {e}")))?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                SarissaError::storage(format!("Failed to read directory entry: {e}"))
            })?;

            if entry
                .file_type()
                .map_err(|e| SarissaError::storage(format!("Failed to get file type: {e}")))?
                .is_file()
            {
                if let Some(name) = entry.file_name().to_str() {
                    files.push(name.to_string());
                }
            }
        }

        Ok(files)
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        let file_path = self.get_file_path(name);
        let metadata = std::fs::metadata(&file_path)
            .map_err(|e| SarissaError::storage(format!("Failed to get file size: {e}")))?;
        Ok(metadata.len())
    }

    fn metadata(&self, name: &str) -> Result<crate::storage::FileMetadata> {
        let file_path = self.get_file_path(name);
        let metadata = std::fs::metadata(&file_path)
            .map_err(|e| SarissaError::storage(format!("Failed to get file metadata: {e}")))?;

        let modified = metadata
            .modified()
            .unwrap_or(std::time::UNIX_EPOCH)
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let created = metadata
            .created()
            .unwrap_or(std::time::UNIX_EPOCH)
            .duration_since(std::time::UNIX_EPOCH)
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
        let old_path = self.get_file_path(old_name);
        let new_path = self.get_file_path(new_name);

        // Invalidate cache for both files
        self.invalidate_cache(old_name);
        self.invalidate_cache(new_name);

        std::fs::rename(&old_path, &new_path)
            .map_err(|e| SarissaError::storage(format!("Failed to rename file: {e}")))?;

        Ok(())
    }

    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        let temp_name = format!("{prefix}_{}", uuid::Uuid::new_v4());
        let output = self.create_output(&temp_name)?;
        Ok((temp_name, output))
    }

    fn sync(&self) -> Result<()> {
        // Memory-mapped files are automatically synced by the OS
        // but we can force a sync if needed
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        // Clear all caches
        {
            let mut mmap_cache = self.mmap_cache.write().unwrap();
            mmap_cache.clear();
        }
        {
            let mut metadata_cache = self.metadata_cache.write().unwrap();
            metadata_cache.clear();
        }

        Ok(())
    }
}

/// Memory-mapped input stream.
#[derive(Debug)]
pub struct MmapInput {
    mmap: Arc<Mmap>,
    cursor: Cursor<&'static [u8]>,
}

impl MmapInput {
    fn new(mmap: Arc<Mmap>) -> Self {
        // SAFETY: We ensure the mmap lives as long as this input
        let slice: &'static [u8] = unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };

        let cursor = Cursor::new(slice);

        MmapInput { mmap, cursor }
    }
}

impl Read for MmapInput {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl Seek for MmapInput {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }
}

impl StorageInput for MmapInput {
    fn size(&self) -> Result<u64> {
        Ok(self.mmap.len() as u64)
    }

    fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
        Ok(Box::new(MmapInput::new(Arc::clone(&self.mmap))))
    }

    fn close(&mut self) -> Result<()> {
        // Nothing to do for mmap input
        Ok(())
    }
}

/// Memory-mapped output stream (buffered writes).
#[derive(Debug)]
pub struct MmapOutput {
    file: File,
    buffer: Vec<u8>,
    file_name: String,
    mmap_cache: Arc<RwLock<HashMap<String, Arc<Mmap>>>>,
    position: u64,
}

impl MmapOutput {
    fn new(
        file: File,
        file_name: String,
        mmap_cache: Arc<RwLock<HashMap<String, Arc<Mmap>>>>,
    ) -> Self {
        MmapOutput {
            file,
            buffer: Vec::new(),
            file_name,
            mmap_cache,
            position: 0,
        }
    }
}

impl Write for MmapOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        self.position += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            self.file.write_all(&self.buffer)?;
            self.file.flush()?;
            self.buffer.clear();
        }
        Ok(())
    }
}

impl Seek for MmapOutput {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        // Flush buffer before seeking
        self.flush()?;
        let new_pos = self.file.seek(pos)?;
        self.position = new_pos;
        Ok(new_pos)
    }
}

impl StorageOutput for MmapOutput {
    fn flush_and_sync(&mut self) -> Result<()> {
        self.flush()
            .map_err(|e| SarissaError::storage(format!("Failed to flush: {e}")))?;
        self.file
            .sync_all()
            .map_err(|e| SarissaError::storage(format!("Failed to sync: {e}")))?;
        Ok(())
    }

    fn position(&self) -> Result<u64> {
        Ok(self.position)
    }

    fn close(&mut self) -> Result<()> {
        self.flush_and_sync()?;

        // Invalidate cache since file has been modified
        let mut cache = self.mmap_cache.write().unwrap();
        cache.remove(&self.file_name);

        Ok(())
    }
}

impl Drop for MmapOutput {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Configuration options specific to memory-mapped storage.
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Base storage configuration.
    pub base: StorageConfig,
    /// Maximum number of files to keep in mmap cache.
    pub max_cached_files: usize,
    /// Enable prefaulting (pre-populate page tables).
    pub enable_prefault: bool,
    /// Enable huge pages if available.
    pub enable_hugepages: bool,
    /// Cache size in bytes (0 = unlimited).
    pub cache_size_bytes: usize,
}

impl Default for MmapConfig {
    fn default() -> Self {
        MmapConfig {
            base: StorageConfig::default(),
            max_cached_files: 100,
            enable_prefault: false,
            enable_hugepages: false,
            cache_size_bytes: 0, // Unlimited
        }
    }
}

/// Advanced memory-mapped storage with LRU cache and additional optimizations.
#[derive(Debug)]
pub struct AdvancedMmapStorage {
    base_storage: MmapStorage,
    #[allow(dead_code)]
    config: MmapConfig,
    // TODO: Add LRU cache implementation
}

impl AdvancedMmapStorage {
    /// Create a new advanced memory-mapped storage.
    pub fn new<P: AsRef<Path>>(path: P, config: MmapConfig) -> Result<Self> {
        let base_storage = MmapStorage::new(path, config.base.clone())?;

        Ok(AdvancedMmapStorage {
            base_storage,
            config,
        })
    }

    /// Pre-warm the cache by loading frequently accessed files.
    pub fn prewarm_cache(&self, file_names: &[&str]) -> Result<()> {
        for &name in file_names {
            if self.base_storage.file_exists(name) {
                let _ = self.base_storage.get_mmap(name)?;
            }
        }
        Ok(())
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        let mmap_cache = self.base_storage.mmap_cache.read().unwrap();
        let metadata_cache = self.base_storage.metadata_cache.read().unwrap();

        CacheStats {
            mmap_entries: mmap_cache.len(),
            metadata_entries: metadata_cache.len(),
            total_mapped_bytes: mmap_cache.values().map(|m| m.len()).sum(),
        }
    }
}

/// Statistics about the memory map cache.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub mmap_entries: usize,
    pub metadata_entries: usize,
    pub total_mapped_bytes: usize,
}

impl Storage for AdvancedMmapStorage {
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.base_storage.open_input(name)
    }

    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.base_storage.create_output(name)
    }

    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.base_storage.create_output_append(name)
    }

    fn file_exists(&self, name: &str) -> bool {
        self.base_storage.file_exists(name)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.base_storage.delete_file(name)
    }

    fn list_files(&self) -> Result<Vec<String>> {
        self.base_storage.list_files()
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        self.base_storage.file_size(name)
    }

    fn metadata(&self, name: &str) -> Result<crate::storage::FileMetadata> {
        self.base_storage.metadata(name)
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.base_storage.rename_file(old_name, new_name)
    }

    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        self.base_storage.create_temp_output(prefix)
    }

    fn sync(&self) -> Result<()> {
        self.base_storage.sync()
    }

    fn close(&mut self) -> Result<()> {
        self.base_storage.close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_mmap_storage_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig::default();
        let storage = MmapStorage::new(temp_dir.path(), config).unwrap();

        // Test file creation and writing
        {
            let mut output = storage.create_output("test.txt").unwrap();
            output.write_all(b"Hello, World!").unwrap();
            output.flush_and_sync().unwrap();
        }

        // Test file existence and reading
        assert!(storage.file_exists("test.txt"));

        let mut input = storage.open_input("test.txt").unwrap();
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer).unwrap();
        assert_eq!(buffer, b"Hello, World!");

        // Test file size
        let size = storage.file_size("test.txt").unwrap();
        assert_eq!(size, 13);

        // Test file listing
        let files = storage.list_files().unwrap();
        assert!(files.contains(&"test.txt".to_string()));
    }

    #[test]
    fn test_mmap_input_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig::default();
        let storage = MmapStorage::new(temp_dir.path(), config).unwrap();

        // Create test file
        {
            let mut output = storage.create_output("test.txt").unwrap();
            output.write_all(b"0123456789").unwrap();
            output.flush_and_sync().unwrap();
        }

        // Test seeking and reading
        let mut input = storage.open_input("test.txt").unwrap();

        // Read from beginning
        let mut buffer = [0u8; 5];
        input.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"01234");

        // Seek to position 5
        input.seek(SeekFrom::Start(5)).unwrap();
        input.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"56789");

        // Test input cloning
        let mut cloned_input = input.clone_input().unwrap();
        cloned_input.seek(SeekFrom::Start(0)).unwrap();
        let mut clone_buffer = [0u8; 3];
        cloned_input.read_exact(&mut clone_buffer).unwrap();
        assert_eq!(&clone_buffer, b"012");
    }

    #[test]
    fn test_mmap_cache_functionality() {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig::default();
        let storage = MmapStorage::new(temp_dir.path(), config).unwrap();

        // Create test file
        {
            let mut output = storage.create_output("cached.txt").unwrap();
            output.write_all(b"Cached content").unwrap();
            output.flush_and_sync().unwrap();
        }

        // First access should create cache entry
        let _input1 = storage.open_input("cached.txt").unwrap();

        // Second access should use cached mmap
        let _input2 = storage.open_input("cached.txt").unwrap();

        // Cache should contain the file
        let cache = storage.mmap_cache.read().unwrap();
        assert!(cache.contains_key("cached.txt"));
    }

    #[test]
    fn test_advanced_mmap_storage() {
        let temp_dir = TempDir::new().unwrap();
        let config = MmapConfig::default();
        let storage = AdvancedMmapStorage::new(temp_dir.path(), config).unwrap();

        // Create test files
        for i in 0..5 {
            let filename = format!("file{i}.txt");
            let mut output = storage.create_output(&filename).unwrap();
            output.write_all(format!("Content {i}").as_bytes()).unwrap();
            output.flush_and_sync().unwrap();
        }

        // Test prewarm cache
        let filenames = ["file0.txt", "file1.txt", "file2.txt"];
        storage.prewarm_cache(&filenames).unwrap();

        // Check cache stats
        let stats = storage.cache_stats();
        assert!(stats.mmap_entries >= 3);
        assert!(stats.total_mapped_bytes > 0);
    }

    #[test]
    fn test_mmap_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig::default();
        let storage = MmapStorage::new(temp_dir.path(), config).unwrap();

        // Create file
        {
            let mut output = storage.create_output("rename_test.txt").unwrap();
            output.write_all(b"rename content").unwrap();
            output.flush_and_sync().unwrap();
        }

        // Test rename
        storage
            .rename_file("rename_test.txt", "renamed.txt")
            .unwrap();
        assert!(!storage.file_exists("rename_test.txt"));
        assert!(storage.file_exists("renamed.txt"));

        // Test delete
        storage.delete_file("renamed.txt").unwrap();
        assert!(!storage.file_exists("renamed.txt"));

        // Test temporary file creation
        let (temp_name, mut temp_output) = storage.create_temp_output("temp_prefix").unwrap();
        temp_output.write_all(b"temporary data").unwrap();
        temp_output.flush_and_sync().unwrap();

        assert!(storage.file_exists(&temp_name));
        assert!(temp_name.starts_with("temp_prefix"));
    }
}
